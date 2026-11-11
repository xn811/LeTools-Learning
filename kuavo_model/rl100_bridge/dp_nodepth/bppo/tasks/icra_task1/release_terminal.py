"""ICRA Task 1 sparse reward at the later stable gripper release."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


LABEL_SEMANTICS = "icra_task1_all_success_later_stable_gripper_release"
FAILURE_LABEL_SEMANTICS = "icra_task1_failure_fixed_horizon_terminal"
CSV_FIELDS = (
    "episode_index",
    "episode_length",
    "left_command_open_frame",
    "left_stable_open_frame",
    "right_command_open_frame",
    "right_stable_open_frame",
    "terminal_frame",
    "terminal_reward",
    "label_semantics",
)


@dataclass(frozen=True)
class FrameRow:
    frame: int
    action: list[float]
    state: list[float]


def _locate_release(
    rows: list[FrameRow], column: int, side: str, stable_open_frames: int
) -> tuple[int, int]:
    actions = np.asarray([float(row.action[column]) for row in rows])
    states = np.asarray([float(row.state[column]) for row in rows])
    action_midpoint = float((actions.min() + actions.max()) / 2.0)
    falling_edges = np.flatnonzero(
        (actions[:-1] > action_midpoint) & (actions[1:] <= action_midpoint)
    ) + 1
    if len(falling_edges) != 1:
        raise ValueError(f"{side}: expected one release, found {len(falling_edges)}")

    command_frame = int(falling_edges[0])
    open_threshold = float(states.min() + 0.10 * (states.max() - states.min()))
    stable_frame = None
    for frame in range(command_frame, len(states) - stable_open_frames + 1):
        if np.all(states[frame : frame + stable_open_frames] <= open_threshold):
            stable_frame = frame
            break
    if stable_frame is None:
        raise ValueError(f"{side}: no stable open state after release")
    return command_frame, stable_frame


def build_release_terminal_records(
    dataset_root: str | Path,
    *,
    state_dim: int,
    action_dim: int,
    left_gripper_index: int,
    right_gripper_index: int,
    stable_open_frames: int,
    excluded_episodes: set[int] | None = None,
) -> tuple[list[dict], dict]:
    root = Path(dataset_root).expanduser().resolve()
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No LeRobot parquet files found below {root / 'data'}")

    by_episode: dict[int, list[FrameRow]] = defaultdict(list)
    for path in parquet_files:
        table = pq.read_table(
            path,
            columns=["episode_index", "frame_index", "action", "observation.state"],
        )
        for episode, frame, action, state in zip(
            table["episode_index"].to_pylist(),
            table["frame_index"].to_pylist(),
            table["action"].to_pylist(),
            table["observation.state"].to_pylist(),
            strict=True,
        ):
            by_episode[int(episode)].append(FrameRow(int(frame), action, state))

    expected = int(info["total_episodes"])
    if set(by_episode) != set(range(expected)):
        raise ValueError("LeRobot episode indices are incomplete or non-contiguous")
    if sum(map(len, by_episode.values())) != int(info["total_frames"]):
        raise ValueError("Parquet frame count differs from meta/info.json")

    excluded = set(excluded_episodes or ())
    if not excluded <= set(range(expected)):
        raise ValueError("Excluded episode index is outside the dataset")

    records: list[dict] = []
    errors: list[str] = []
    for episode in range(expected):
        if episode in excluded:
            continue
        rows = sorted(by_episode[episode], key=lambda row: row.frame)
        if [row.frame for row in rows] != list(range(len(rows))):
            errors.append(f"episode {episode}: non-contiguous frame indices")
            continue
        if any(
            len(row.action) != action_dim or len(row.state) != state_dim
            for row in rows
        ):
            errors.append(
                f"episode {episode}: expected action_dim={action_dim}, "
                f"state_dim={state_dim}"
            )
            continue
        try:
            left_command, left_stable = _locate_release(
                rows, left_gripper_index, f"episode {episode} left", stable_open_frames
            )
            right_command, right_stable = _locate_release(
                rows, right_gripper_index, f"episode {episode} right", stable_open_frames
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue

        terminal = max(left_stable, right_stable)
        if not 0 <= terminal < len(rows) - 1:
            errors.append(f"episode {episode}: invalid terminal frame {terminal}")
            continue
        records.append(
            {
                "episode_index": episode,
                "episode_length": len(rows),
                "left_command_open_frame": left_command,
                "left_stable_open_frame": left_stable,
                "right_command_open_frame": right_command,
                "right_stable_open_frame": right_stable,
                "terminal_frame": terminal,
                "terminal_reward": 1.0,
                "label_semantics": LABEL_SEMANTICS,
            }
        )

    if errors:
        raise ValueError(
            f"Release detection failed for {len(errors)}/{expected} episodes; "
            "no CSV was written.\n" + "\n".join(errors[:50])
        )
    return records, info


class ICRATask1ReleaseTerminal:
    """Load and validate ICRA Task 1 terminal frames for offline training."""

    def __init__(
        self,
        annotation_path: str | Path,
        episode_lengths: dict[int, int],
        excluded_episode_indices: list[int] | tuple[int, ...],
    ) -> None:
        self.annotation_path = Path(annotation_path).expanduser().resolve()
        self.episode_lengths = episode_lengths
        self.excluded_episode_indices = frozenset(
            int(value) for value in excluded_episode_indices
        )
        self.terminal_rewards: dict[int, float] = {}
        self.terminal_frames = self._load()

    @property
    def eligible_episode_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self.terminal_frames))

    def _load(self) -> dict[int, int]:
        if not self.annotation_path.is_file():
            raise FileNotFoundError(
                f"ICRA Task 1 terminal CSV not found: {self.annotation_path}"
            )
        exclusion_path = Path(str(self.annotation_path) + ".exclusions.json")
        if not exclusion_path.is_file():
            raise FileNotFoundError(f"Exclusion audit not found: {exclusion_path}")
        audit = json.loads(exclusion_path.read_text(encoding="utf-8"))
        if int(audit.get("dataset_total_episodes", -1)) != len(self.episode_lengths):
            raise ValueError("Exclusion audit dataset size mismatch")
        audited = frozenset(int(v) for v in audit.get("excluded_episode_indices", ()))
        if audited != self.excluded_episode_indices:
            raise ValueError("Configured exclusions differ from the exclusion audit")
        if audited and not str(audit.get("reason", "")).strip():
            raise ValueError("Exclusion audit reason is empty")

        terminals: dict[int, int] = {}
        with self.annotation_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            missing = set(CSV_FIELDS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Terminal CSV missing fields: {sorted(missing)}")
            for row in reader:
                episode = int(row["episode_index"])
                length = int(row["episode_length"])
                terminal = int(row["terminal_frame"])
                if episode in terminals or episode not in self.episode_lengths:
                    raise ValueError(f"Invalid/duplicate episode {episode}")
                if length != self.episode_lengths[episode]:
                    raise ValueError(f"Episode {episode}: length mismatch")
                reward = float(row["terminal_reward"])
                if reward == 1.0 and row["label_semantics"] == LABEL_SEMANTICS:
                    left = int(row["left_stable_open_frame"])
                    right = int(row["right_stable_open_frame"])
                    if terminal != max(left, right) or not 0 <= terminal < length - 1:
                        raise ValueError(f"Episode {episode}: invalid success terminal frame")
                elif reward == 0.0 and row["label_semantics"] == FAILURE_LABEL_SEMANTICS:
                    if terminal != length - 1:
                        raise ValueError(f"Episode {episode}: failure terminal must be final frame")
                    if any(row[key].strip() for key in (
                        "left_command_open_frame", "left_stable_open_frame",
                        "right_command_open_frame", "right_stable_open_frame",
                    )):
                        raise ValueError(f"Episode {episode}: failure release fields must be empty")
                else:
                    raise ValueError(f"Episode {episode}: invalid reward or label semantics")
                terminals[episode] = terminal
                self.terminal_rewards[episode] = reward

        expected = set(self.episode_lengths) - set(self.excluded_episode_indices)
        if set(terminals) != expected:
            raise ValueError("Terminal CSV episode set does not match the dataset")
        if int(audit.get("labeled_episodes", -1)) != len(terminals):
            raise ValueError("Exclusion audit labeled count mismatch")
        return terminals

    def terminal_frame(self, episode_index: int) -> int:
        return self.terminal_frames[episode_index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--left-gripper-index", type=int, default=7)
    parser.add_argument("--right-gripper-index", type=int, default=15)
    parser.add_argument("--stable-open-frames", type=int, default=3)
    parser.add_argument("--exclude-episode", type=int, action="append", default=[])
    parser.add_argument("--exclusion-reason", default="")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    exclusion_path = Path(str(output) + ".exclusions.json")
    if output.exists() or exclusion_path.exists():
        raise FileExistsError("Terminal annotation output already exists")
    excluded = set(args.exclude_episode)
    if excluded and not args.exclusion_reason.strip():
        raise ValueError("--exclusion-reason is required with --exclude-episode")
    records, info = build_release_terminal_records(
        args.dataset_root,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        left_gripper_index=args.left_gripper_index,
        right_gripper_index=args.right_gripper_index,
        stable_open_frames=args.stable_open_frames,
        excluded_episodes=excluded,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    exclusion_path.write_text(
        json.dumps(
            {
                "dataset_total_episodes": int(info["total_episodes"]),
                "labeled_episodes": len(records),
                "excluded_episode_indices": sorted(excluded),
                "reason": args.exclusion_reason.strip(),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    tails = [row["episode_length"] - 1 - row["terminal_frame"] for row in records]
    print(f"episodes={len(records)} excluded={sorted(excluded)}")
    print(f"terminal_tail_frequency={dict(sorted(Counter(tails).items()))}")
    print(f"RELEASE_TERMINAL_ANNOTATIONS_READY: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
