"""Align a LeRobot DP no-depth dataset with RL-100 chunk transitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .tasks.icra_task1 import ICRATask1ReleaseTerminal


def _as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    return int(value)


class LeRobotDPNoDepthTransitionDataset(Dataset):
    """Return the tensors consumed by RL-100 Critic, Dynamics, and BPPO.

    For center frame ``t`` and the validated LeTools Diffusion configuration:

    * observation: ``[t-1, t]``
    * action horizon: ``[t-1, ..., t+14]``
    * executed actor chunk: ``[t, ..., t+7]``
    * next observation: ``[t+7, t+8]``
    * next action horizon: ``[t+7, ..., t+22]``
    """

    OBS_OFFSETS = (-1, 0, 7, 8)
    ACTION_OFFSETS = tuple(range(-1, 23))
    CURRENT_ACTION_SLICE = slice(0, 16)
    NEXT_ACTION_SLICE = slice(8, 24)
    RL_OFFSETS = tuple(range(-1, 15))

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        dataset_repo_id: str,
        terminal_annotation_path: str | Path,
        excluded_episode_indices: list[int] | tuple[int, ...],
        gamma: float = 0.99,
        video_backend: str = "pyav",
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        info_path = self.dataset_root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        self.num_episodes = int(info["total_episodes"])
        self.fps = int(info["fps"])
        self.gamma = float(gamma)
        self.image_keys = tuple(
            key
            for key, feature in info["features"].items()
            if feature.get("dtype") in {"video", "image"}
        )
        if not self.image_keys:
            raise ValueError("The LeRobot dataset contains no RGB observations")

        observation_deltas = [offset / self.fps for offset in self.OBS_OFFSETS]
        action_deltas = [offset / self.fps for offset in self.ACTION_OFFSETS]
        self.base = LeRobotDataset(
            repo_id=dataset_repo_id,
            root=self.dataset_root,
            delta_timestamps={
                "observation.state": observation_deltas,
                "action": action_deltas,
                **{key: observation_deltas for key in self.image_keys},
            },
            video_backend=video_backend,
            return_uint8=False,
        )
        self.episode_lengths = self._read_episode_lengths()
        self.task = ICRATask1ReleaseTerminal(
            terminal_annotation_path,
            self.episode_lengths,
            excluded_episode_indices,
        )
        self.eligible_episode_indices = self.task.eligible_episode_indices
        self._normalizer = None

    def _read_episode_lengths(self) -> dict[int, int]:
        lengths: dict[int, int] = {}
        for episode_value, frame_value in zip(
            self.base.hf_dataset["episode_index"],
            self.base.hf_dataset["frame_index"],
            strict=True,
        ):
            episode = _as_int(episode_value)
            frame = _as_int(frame_value)
            lengths[episode] = max(lengths.get(episode, 0), frame + 1)
        if set(lengths) != set(range(self.num_episodes)):
            raise ValueError("Episode indices do not match meta/info.json")
        return lengths

    def set_normalizer(self, normalizer) -> None:
        self._normalizer = normalizer

    def get_normalizer(self):
        if self._normalizer is None:
            raise RuntimeError("Policy checkpoint normalizer has not been attached")
        return self._normalizer

    def indices_for_episodes(self, episode_indices: set[int]) -> list[int]:
        episodes = self.base.hf_dataset["episode_index"]
        frames = self.base.hf_dataset["frame_index"]
        return [
            index
            for index, (episode_value, frame_value) in enumerate(
                zip(episodes, frames, strict=True)
            )
            if (episode := _as_int(episode_value)) in episode_indices
            and _as_int(frame_value) <= self.task.terminal_frame(episode)
        ]

    def __len__(self) -> int:
        return len(self.base)

    def _transition_targets(
        self, episode_index: int, frame_index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        terminal_frame = self.task.terminal_frame(episode_index)
        terminal_reward = self.task.terminal_rewards[episode_index]
        rewards: list[float] = []
        not_done: list[float] = []
        returns: list[float] = []
        for offset in self.RL_OFFSETS:
            absolute_frame = frame_index + offset
            in_episode = 0 <= absolute_frame <= terminal_frame
            rewards.append(terminal_reward if absolute_frame == terminal_frame else 0.0)
            not_done.append(1.0 if absolute_frame < terminal_frame else 0.0)
            returns.append(
                terminal_reward * self.gamma ** (terminal_frame - absolute_frame)
                if in_episode
                else 0.0
            )
        reward = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        not_done_tensor = torch.tensor(not_done, dtype=torch.float32).unsqueeze(-1)
        return_value = torch.tensor(returns, dtype=torch.float32).unsqueeze(-1)
        done = 1.0 - not_done_tensor
        return reward, not_done_tensor, return_value, done

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        episode_index = _as_int(item["episode_index"])
        frame_index = _as_int(item["frame_index"])
        if episode_index not in self.task.terminal_frames:
            raise IndexError(f"Episode {episode_index} is excluded from training")

        state = item["observation.state"].to(torch.float32)
        action = item["action"].to(torch.float32)
        obs = {"agent_pos": state[:2]}
        next_obs = {"agent_pos": state[2:4]}
        for key in self.image_keys:
            images = item[key]
            if images.dtype == torch.uint8:
                images = images.to(torch.float32) / 255.0
            else:
                images = images.to(torch.float32)
            obs[key] = images[:2]
            next_obs[key] = images[2:4]

        reward, not_done, return_value, done = self._transition_targets(
            episode_index, frame_index
        )
        return {
            "obs": obs,
            "next_obs": next_obs,
            "action": action[self.CURRENT_ACTION_SLICE],
            "next_action": action[self.NEXT_ACTION_SLICE],
            "reward": reward,
            "not_done": not_done,
            "done": done,
            "return": return_value,
            "episode_index": torch.tensor(episode_index, dtype=torch.long),
            "frame_index": torch.tensor(frame_index, dtype=torch.long),
            "next_obs_padding_mask": torch.tensor(
                [
                    frame_index + offset >= self.episode_lengths[episode_index]
                    for offset in (7, 8)
                ],
                dtype=torch.bool,
            ),
        }
