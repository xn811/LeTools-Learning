"""Pure-offline LeTools Diffusion -> IQL/Dynamics/BPPO training workspace."""

from __future__ import annotations

import csv
import json
import logging
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from .bppo import BehaviorProximalPolicyOptimization
from .checkpoint_export import export_letools_checkpoint
from .critic import IQL_Q_V_no
from .ensemble_dynamics import EnsembleDynamics_batch
from .policy_adapter import LeToolsDPPolicyAdapter
from .transition_dataset import LeRobotDPNoDepthTransitionDataset


LOG = logging.getLogger(__name__)


def _move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _infinite_batches(loader: DataLoader) -> Iterator[dict]:
    while True:
        yield from loader


def _append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    """Append one immediately flushed metrics row, creating a header as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _slice_observation_steps(observation: dict, step_slice: slice) -> dict:
    return {
        key: value[:, step_slice, ...].reshape(-1, *value.shape[2:])
        for key, value in observation.items()
    }


class OfflineBPPODPNoDepthWorkspace:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(str(cfg.runtime.device))
        self._seed_everything(int(cfg.training.seed))

        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        advantage_mode = str(cfg.bppo.advantage_mode)
        rollout_mode = str(cfg.dynamics.rollout_mode)
        terminal_mode = str(cfg.dynamics.terminal_mode)
        run_tag_value = OmegaConf.select(cfg, "training.run_tag")
        run_tag = f"{str(run_tag_value)}_" if run_tag_value else ""
        self.run_name = (
            f"dp_offline_bppo_{advantage_mode}_{rollout_mode}_{terminal_mode}_"
            f"{run_tag}rl100_dp_nodepth_"
            f"{int(cfg.data.expected_episode_count)}episodes_"
            f"{timestamp}_{int(cfg.bppo.steps):06d}steps"
        )
        self.run_dir = Path(cfg.paths.output_root).expanduser().resolve() / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "rl100_modules").mkdir()
        self.metrics_dir = self.run_dir / "training_metrics"
        self.metrics_dir.mkdir()
        OmegaConf.save(cfg, self.run_dir / "config_resolved.yaml", resolve=True)

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _make_dataset_and_loaders(self, policy: LeToolsDPPolicyAdapter):
        cfg = self.cfg
        dataset = LeRobotDPNoDepthTransitionDataset(
            dataset_root=cfg.paths.dataset_root,
            dataset_repo_id=cfg.data.dataset_repo_id,
            terminal_annotation_path=cfg.paths.terminal_annotations,
            gamma=cfg.training.gamma,
            excluded_episode_indices=list(cfg.task.excluded_episode_indices),
            video_backend=cfg.data.video_backend,
        )
        if (str(cfg.dynamics.terminal_mode) == "reward_threshold"
                and any(reward == 0.0 for reward in dataset.task.terminal_rewards.values())):
            raise ValueError(
                "Zero-reward failure terminals cannot be inferred by reward_threshold "
                "dynamics; a separate terminal model or explicit horizon-aware rollout "
                "is required before mixed-outcome BPPO training."
            )
        # Keep CUDA modules out of DataLoader worker copies. The dataset API
        # still exposes the checkpoint-derived normalizer, but as a tiny CPU copy.
        dataset.set_normalizer(deepcopy(policy.normalizer).cpu())

        episode_ids = list(dataset.eligible_episode_indices)
        split_rng = random.Random(int(cfg.training.seed))
        split_rng.shuffle(episode_ids)
        validation_count = max(
            1, int(round(len(episode_ids) * float(cfg.data.validation_fraction)))
        )
        failures = [episode for episode in episode_ids
                    if dataset.task.terminal_rewards[episode] == 0.0]
        if failures and str(cfg.dynamics.terminal_mode) == "learned":
            # Keep at least one zero-reward failure in the training partition.
            # With only one failure it cannot also appear in validation.
            validation_ids = episode_ids[:validation_count]
            training_ids = episode_ids[validation_count:]
            if not any(episode in failures for episode in training_ids):
                failed = next(episode for episode in validation_ids if episode in failures)
                replacement = next((episode for episode in training_ids
                                    if episode not in failures), None)
                if replacement is None:
                    raise ValueError("Cannot put a failure episode in the training split")
                episode_ids[episode_ids.index(failed)] = replacement
                episode_ids[episode_ids.index(replacement, validation_count)] = failed
            if len(failures) == 1:
                LOG.warning("Only one failure episode: failure generalization cannot be validated")
        validation_episodes = set(episode_ids[:validation_count])
        train_episodes = set(episode_ids[validation_count:])
        train_indices = dataset.indices_for_episodes(train_episodes)
        validation_indices = dataset.indices_for_episodes(validation_episodes)
        if not train_indices or not validation_indices:
            raise RuntimeError("Train/validation episode split produced an empty subset.")

        loader_kwargs = {
            "batch_size": int(cfg.training.batch_size),
            "num_workers": int(cfg.training.num_workers),
            "pin_memory": False,
            "persistent_workers": int(cfg.training.num_workers) > 0,
        }
        train_loader = DataLoader(
            Subset(dataset, train_indices), shuffle=True, drop_last=True, **loader_kwargs
        )
        validation_loader = DataLoader(
            Subset(dataset, validation_indices), shuffle=False, drop_last=False, **loader_kwargs
        )
        LOG.info(
            "Dataset ready: %d frames, %d train episodes/%d samples, %d validation episodes/%d samples",
            len(dataset),
            len(train_episodes),
            len(train_indices),
            len(validation_episodes),
            len(validation_indices),
        )
        return dataset, train_loader, validation_loader

    def _evaluate_iql(self, iql, validation_loader) -> dict[str, float | int]:
        """Evaluate chunk-IQL losses without changing parameters or target state."""
        if not (iql.is_share_encoder and iql.fix_encoder and iql.chunk_as_single_action):
            raise RuntimeError(
                "Critic validation requires the bridge's fixed shared encoder "
                "and chunk_as_single_action configuration."
            )

        q_losses: list[float] = []
        value_losses: list[float] = []
        sample_counts: list[int] = []
        max_batches = self.cfg.training.max_batches_per_epoch
        iql.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(validation_loader):
                batch = _move_to_device(batch, self.device)
                normalized_obs = iql.normalizer.normalize(batch["obs"])
                normalized_next_obs = iql.normalizer.normalize(batch["next_obs"])
                normalized_actions = iql.normalizer["action"].normalize(batch["action"])
                batch_size = int(normalized_actions.shape[0])

                current_obs = _slice_observation_steps(
                    normalized_obs, slice(None, iql.n_obs_steps)
                )
                next_obs = _slice_observation_steps(
                    normalized_next_obs, slice(-iql.n_obs_steps, None)
                )
                current_features = iql.obs_encoder(current_obs).reshape(batch_size, -1)
                next_features = iql.obs_encoder(next_obs).reshape(batch_size, -1)

                action_start = iql.n_obs_steps - 1
                action_end = action_start + iql.n_action_steps
                action_chunk = normalized_actions[:, action_start:action_end]
                reward_chunk = batch["reward"][:, action_start:action_end].reshape(
                    batch_size, iql.n_action_steps, -1
                )
                gamma_weights = torch.pow(
                    iql._gamma,
                    torch.arange(
                        iql.n_action_steps,
                        device=reward_chunk.device,
                        dtype=reward_chunk.dtype,
                    ),
                )
                discounted_reward = (
                    reward_chunk.squeeze(-1) * gamma_weights
                ).sum(dim=-1, keepdim=True)
                not_done = iql._as_column(
                    batch["not_done"][:, action_end - 1], "not_done"
                )

                target_q_for_value = iql.target_minQ(current_features, action_chunk)
                value = iql._value(current_features)
                value_loss = iql.expectile_loss(target_q_for_value - value).mean()

                next_value = iql._value(next_features)
                td_target = discounted_reward + not_done * (
                    iql._gamma ** iql.n_action_steps
                ) * next_value
                current_q1, current_q2 = iql._Q(current_features, action_chunk)
                q_loss = (
                    (current_q1 - td_target).pow(2)
                    + (current_q2 - td_target).pow(2)
                ).mean()

                q_losses.append(float(q_loss.cpu()))
                value_losses.append(float(value_loss.cpu()))
                sample_counts.append(batch_size)
                if max_batches is not None and batch_index + 1 >= int(max_batches):
                    break

        if not sample_counts:
            raise RuntimeError("Critic validation loader produced no batches.")
        weights = np.asarray(sample_counts, dtype=np.float64)
        return {
            "q_loss": float(np.average(q_losses, weights=weights)),
            "value_loss": float(np.average(value_losses, weights=weights)),
            "batches": len(sample_counts),
            "samples": int(weights.sum()),
        }

    def _train_iql(self, policy, train_loader, validation_loader):
        cfg = self.cfg
        critic_encoder = deepcopy(policy.obs_encoder).to(self.device)
        critic_encoder.eval()
        iql = IQL_Q_V_no(
            device=self.device,
            state_dim=policy.global_cond_dim,
            feature_dim=policy.obs_feature_dim,
            action_dim=policy.action_dim,
            q_hidden_dim=int(cfg.critic.hidden_dim),
            q_depth=int(cfg.critic.depth),
            Q_lr=float(cfg.critic.q_lr),
            target_update_freq=int(cfg.critic.target_update_freq),
            tau=float(cfg.critic.tau),
            gamma=float(cfg.training.gamma),
            v_hidden_dim=int(cfg.critic.hidden_dim),
            v_depth=int(cfg.critic.depth),
            v_lr=float(cfg.critic.v_lr),
            omega=float(cfg.critic.expectile),
            is_double_q=True,
            dp3_normalizer=policy.normalizer,
            obs_encoder=critic_encoder,
            n_obs_steps=policy.n_obs_steps,
            is_share_encoder=True,
            use_pc_color=True,
            use_action_embed=False,
            fix_encoder=True,
            action_norm=True,
            n_action_steps=policy.n_action_steps,
            chunk_as_single_action=True,
        )
        max_batches = cfg.training.max_batches_per_epoch
        log_every = int(
            OmegaConf.select(cfg, "training.log_every_batches", default=1)
        )
        if log_every < 1:
            raise ValueError("training.log_every_batches must be at least 1.")
        step_fields = [
            "global_step", "epoch", "batch", "batch_size", "q_loss", "value_loss"
        ]
        epoch_fields = [
            "epoch", "train_batches", "train_q_loss", "train_value_loss",
            "validation_batches", "validation_samples", "validation_q_loss",
            "validation_value_loss",
        ]
        global_step = 0
        for epoch in range(int(cfg.critic.epochs)):
            iql.train()
            q_losses: list[float] = []
            v_losses: list[float] = []
            for batch_index, batch in enumerate(train_loader):
                batch = _move_to_device(batch, self.device)
                q_loss, v_loss = iql.update(batch=batch)
                q_loss = float(q_loss)
                v_loss = float(v_loss)
                q_losses.append(q_loss)
                v_losses.append(v_loss)
                global_step += 1
                row = {
                    "global_step": global_step,
                    "epoch": epoch + 1,
                    "batch": batch_index + 1,
                    "batch_size": int(batch["action"].shape[0]),
                    "q_loss": q_loss,
                    "value_loss": v_loss,
                }
                _append_csv(
                    self.metrics_dir / "critic_train_steps.csv", step_fields, row
                )
                if global_step % log_every == 0:
                    LOG.info(
                        "Critic train step %d: epoch=%d/%d batch=%d "
                        "q_loss=%.8f value_loss=%.8f",
                        global_step,
                        epoch + 1,
                        int(cfg.critic.epochs),
                        batch_index + 1,
                        q_loss,
                        v_loss,
                    )
                if max_batches is not None and batch_index + 1 >= int(max_batches):
                    break
            validation = self._evaluate_iql(iql, validation_loader)
            train_q_loss = float(np.mean(q_losses))
            train_value_loss = float(np.mean(v_losses))
            LOG.info(
                "Critic epoch %d/%d: train_q_loss=%.8f train_value_loss=%.8f "
                "validation_q_loss=%.8f validation_value_loss=%.8f "
                "train_batches=%d validation_batches=%d",
                epoch + 1,
                int(cfg.critic.epochs),
                train_q_loss,
                train_value_loss,
                validation["q_loss"],
                validation["value_loss"],
                len(q_losses),
                validation["batches"],
            )
            _append_csv(
                self.metrics_dir / "critic_epochs.csv",
                epoch_fields,
                {
                    "epoch": epoch + 1,
                    "train_batches": len(q_losses),
                    "train_q_loss": train_q_loss,
                    "train_value_loss": train_value_loss,
                    "validation_batches": validation["batches"],
                    "validation_samples": validation["samples"],
                    "validation_q_loss": validation["q_loss"],
                    "validation_value_loss": validation["value_loss"],
                },
            )

        critic_summary = {
            "configured_epochs": int(cfg.critic.epochs),
            "completed_epochs": int(cfg.critic.epochs),
            "optimizer_steps": global_step,
            "stop_reason": "configured_epoch_limit_reached",
            "metrics": {
                "train_steps_csv": str(self.metrics_dir / "critic_train_steps.csv"),
                "epochs_csv": str(self.metrics_dir / "critic_epochs.csv"),
            },
        }
        (self.metrics_dir / "critic_training_summary_rl100.json").write_text(
            json.dumps(critic_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        LOG.info(
            "Critic training stopped: reason=%s completed_epochs=%d optimizer_steps=%d",
            critic_summary["stop_reason"],
            critic_summary["completed_epochs"],
            global_step,
        )

        artifact_dir = self.run_dir / "rl100_modules" / "critic_value"
        artifact_dir.mkdir()
        iql.save(
            str(artifact_dir / "Q_bc_rl100.pt"),
            str(artifact_dir / "value_rl100.pt"),
            None,
        )
        iql.eval()
        return iql

    def _make_dynamics_cfg(self, policy) -> DictConfig:
        cfg = self.cfg
        return OmegaConf.create(
            {
                "n_obs_steps": policy.n_obs_steps,
                "n_action_steps": policy.n_action_steps,
                "chunk_as_single_action": True,
                "action_norm": True,
                "use_action_embed": False,
                "use_conv_action_embed": False,
                "predict_r": True,
                "dynamics_type": "mlp",
                "policy": {"use_pc_color": True},
                "dynamics": {
                    "fix_encoder": True,
                    "predict_delta": True,
                    "prediction_mode": "full",
                    "action_embed_layer_norm": False,
                    "action_scale_norm": False,
                    "reward_loss_weight": float(cfg.dynamics.reward_loss_weight),
                    "positive_reward_weight": float(cfg.dynamics.positive_reward_weight),
                    "rollout_mode": str(cfg.dynamics.rollout_mode),
                    "terminal_mode": str(cfg.dynamics.terminal_mode),
                    "terminal_reward_threshold": OmegaConf.select(
                        cfg, "dynamics.terminal_reward_threshold"
                    ),
                    "terminal_classifier_threshold": float(OmegaConf.select(
                        cfg, "dynamics.terminal_classifier_threshold", default=0.5
                    )),
                },
            }
        )

    def _train_dynamics(self, policy, train_loader, validation_loader):
        try:
            from rl_100.unidpg.transition_model.models.dynamics_model import EnsembleDynamicsModel
            from rl_100.unidpg.transition_model.utils.termination_fns import terminaltion_fn_door
        except ImportError as exc:
            raise RuntimeError(
                "RL-100 Dynamics dependencies are missing from the active LeTools environment."
            ) from exc

        cfg = self.cfg
        dynamics_cfg = self._make_dynamics_cfg(policy)
        state_dim = policy.global_cond_dim
        chunk_action_dim = policy.action_dim * policy.n_action_steps
        hidden_dims = [int(value) for value in cfg.dynamics.hidden_dims]
        weight_decays = [float(value) for value in cfg.dynamics.weight_decays]
        if len(weight_decays) != len(hidden_dims) + 1:
            raise ValueError("dynamics.weight_decays must have len(hidden_dims)+1 entries.")
        model = EnsembleDynamicsModel(
            obs_dim=state_dim,
            action_dim=chunk_action_dim,
            hidden_dims=hidden_dims,
            num_ensemble=int(cfg.dynamics.ensemble_size),
            num_elites=int(cfg.dynamics.num_elites),
            weight_decays=weight_decays,
            with_reward=True,
            device=str(self.device),
            cfg=dynamics_cfg,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.dynamics.lr))
        encoder = policy.get_dynamics_encoder().to(self.device)
        encoder.eval()
        dynamics = EnsembleDynamics_batch(
            model=model,
            optim=optimizer,
            terminal_fn=terminaltion_fn_door,
            env=None,
            normalizer=policy.normalizer,
            encoder=encoder,
            cfg=dynamics_cfg,
            action_dim=chunk_action_dim,
            gamma=float(cfg.training.gamma),
            device=str(self.device),
            chunk_as_single_action=True,
            n_action_steps=policy.n_action_steps,
            prediction_mode="full",
        )
        terminal_optimizer = None
        terminal_coverage_loader = None
        if dynamics.terminal_mode == "learned":
            dynamics.terminal_classifier = torch.nn.Sequential(
                torch.nn.Linear(state_dim + chunk_action_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 1),
            ).to(self.device)
            terminal_optimizer = torch.optim.Adam(
                dynamics.terminal_classifier.parameters(), lr=float(cfg.dynamics.lr)
            )
            # The dynamics batches may be capped for smoke tests. Guarantee the
            # classifier sees each training episode's first and terminal center.
            subset = train_loader.dataset
            dataset = subset.dataset
            episodes = dataset.base.hf_dataset["episode_index"]
            frames = dataset.base.hf_dataset["frame_index"]
            coverage_indices = [index for index in subset.indices
                                if int(frames[index]) in (
                                    0, dataset.task.terminal_frame(int(episodes[index]))
                                )]
            if not coverage_indices:
                raise ValueError("No terminal classifier coverage samples")
            terminal_coverage_loader = DataLoader(
                Subset(dataset, coverage_indices), batch_size=int(cfg.training.batch_size),
                shuffle=False, num_workers=0, drop_last=False,
            )
            LOG.warning(
                "Learned terminal prediction is experimental; time-limit failures "
                "cannot be inferred reliably from state alone without elapsed time."
            )

        max_batches = cfg.training.max_batches_per_epoch
        best_losses = np.full(int(cfg.dynamics.ensemble_size), np.inf, dtype=np.float64)
        max_epochs = int(cfg.dynamics.epochs)
        patience = int(OmegaConf.select(cfg, "dynamics.patience", default=max_epochs))
        relative_improvement_threshold = float(
            OmegaConf.select(
                cfg, "dynamics.relative_improvement_threshold", default=0.01
            )
        )
        log_every = int(
            OmegaConf.select(cfg, "training.log_every_batches", default=1)
        )
        if patience < 1:
            raise ValueError("dynamics.patience must be at least 1.")
        if relative_improvement_threshold < 0:
            raise ValueError(
                "dynamics.relative_improvement_threshold must be non-negative."
            )
        step_fields = ["global_step", "epoch", "batch", "batch_size", "train_loss"]
        epoch_fields = [
            "epoch", "train_batches", "validation_batches", "validation_samples",
            "train_loss", "holdout_loss",
            "best_holdout_loss", "improved_members", "epochs_without_improvement",
            "reward_mse", "positive_reward_mse", "zero_reward_mse",
            "positive_reward_prediction_mean", "zero_reward_prediction_mean",
            "positive_target_fraction",
        ]
        epoch_fields.extend(
            f"member_{index}_holdout_loss"
            for index in range(int(cfg.dynamics.ensemble_size))
        )
        epoch_fields.extend(
            f"member_{index}_best_loss"
            for index in range(int(cfg.dynamics.ensemble_size))
        )
        best_holdout = np.inf
        best_epoch = 0
        epochs_without_improvement = 0
        global_step = 0
        completed_epochs = 0
        stop_reason = "configured_epoch_limit_reached"
        terminal_train_positive = 0
        terminal_train_total = 0
        terminal_validation_positive = 0
        terminal_validation_total = 0
        terminal_failure_centers_seen = 0
        for epoch in range(max_epochs):
            model.train()
            train_losses: list[float] = []
            for batch_index, batch in enumerate(train_loader):
                batch = _move_to_device(batch, self.device)
                with torch.no_grad():
                    current = dynamics.obs2latent(batch["obs"]).reshape(batch["action"].shape[0], -1)
                    following = dynamics.next_obs2latent(batch["next_obs"]).reshape(
                        batch["action"].shape[0], -1
                    )
                loss = dynamics.learn(batch, current, following)
                dynamics.optimize(loss)
                if terminal_optimizer is not None:
                    inputs, _ = dynamics.format_samples_for_training(batch, current, following)
                    last = policy.n_obs_steps - 1 + policy.n_action_steps - 1
                    target = 1.0 - batch["not_done"][:, last].reshape(-1, 1)
                    logits = dynamics.terminal_classifier(inputs.detach())
                    terminal_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
                    terminal_optimizer.zero_grad()
                    terminal_loss.backward()
                    terminal_optimizer.step()
                    terminal_train_positive += int(target.sum().item())
                    terminal_train_total += int(target.numel())
                    for episode, frame in zip(batch["episode_index"].tolist(),
                                              batch["frame_index"].tolist(), strict=True):
                        if (dataset.task.terminal_rewards[int(episode)] == 0.0
                                and int(frame) == dataset.task.terminal_frame(int(episode))):
                            terminal_failure_centers_seen += 1
                train_loss = float(loss.detach().cpu().item())
                train_losses.append(train_loss)
                global_step += 1
                _append_csv(
                    self.metrics_dir / "dynamics_train_steps.csv",
                    step_fields,
                    {
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "batch": batch_index + 1,
                        "batch_size": int(batch["action"].shape[0]),
                        "train_loss": train_loss,
                    },
                )
                if global_step % log_every == 0:
                    LOG.info(
                        "Dynamics train step %d: epoch=%d/%d batch=%d loss=%.8f",
                        global_step,
                        epoch + 1,
                        max_epochs,
                        batch_index + 1,
                        train_loss,
                    )
                if max_batches is not None and batch_index + 1 >= int(max_batches):
                    break

            if terminal_coverage_loader is not None:
                for batch in terminal_coverage_loader:
                    batch = _move_to_device(batch, self.device)
                    with torch.no_grad():
                        current = dynamics.obs2latent(batch["obs"]).reshape(batch["action"].shape[0], -1)
                        following = dynamics.next_obs2latent(batch["next_obs"]).reshape(batch["action"].shape[0], -1)
                        inputs, _ = dynamics.format_samples_for_training(batch, current, following)
                        last = policy.n_obs_steps - 1 + policy.n_action_steps - 1
                        target = 1.0 - batch["not_done"][:, last].reshape(-1, 1)
                    logits = dynamics.terminal_classifier(inputs.detach())
                    terminal_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
                    terminal_optimizer.zero_grad()
                    terminal_loss.backward()
                    terminal_optimizer.step()
                    terminal_train_positive += int(target.sum().item())
                    terminal_train_total += int(target.numel())

                    for episode, frame in zip(batch["episode_index"].tolist(),
                                              batch["frame_index"].tolist(), strict=True):
                        if (dataset.task.terminal_rewards[int(episode)] == 0.0
                                and int(frame) == dataset.task.terminal_frame(int(episode))):
                            terminal_failure_centers_seen += 1

            validation_losses: list[np.ndarray] = []
            validation_batch_sizes: list[int] = []
            reward_metric_totals = {
                "reward_sq_sum": 0.0,
                "reward_count": 0,
                "positive_reward_sq_sum": 0.0,
                "positive_reward_count": 0,
                "zero_reward_sq_sum": 0.0,
                "zero_reward_count": 0,
                "positive_target_count": 0,
                "target_count": 0,
                "positive_reward_prediction_sum": 0.0,
                "zero_reward_prediction_sum": 0.0,
            }
            model.eval()
            with torch.no_grad():
                for batch_index, batch in enumerate(validation_loader):
                    batch = _move_to_device(batch, self.device)
                    current = dynamics.obs2latent(batch["obs"]).reshape(batch["action"].shape[0], -1)
                    following = dynamics.next_obs2latent(batch["next_obs"]).reshape(
                        batch["action"].shape[0], -1
                    )
                    inputs, targets = dynamics.format_samples_for_training(batch, current, following)
                    if terminal_optimizer is not None:
                        last = policy.n_obs_steps - 1 + policy.n_action_steps - 1
                        target = 1.0 - batch["not_done"][:, last].reshape(-1, 1)
                        terminal_validation_positive += int(target.sum().item())
                        terminal_validation_total += int(target.numel())
                    val_loss, reward_metrics = dynamics.validate_with_reward_metrics(
                        inputs, targets
                    )
                    validation_losses.append(np.asarray(val_loss))
                    validation_batch_sizes.append(int(batch["action"].shape[0]))
                    for key, value in reward_metrics.items():
                        reward_metric_totals[key] += value
                    if max_batches is not None and batch_index + 1 >= int(max_batches):
                        break
            mean_validation = np.average(
                np.stack(validation_losses),
                axis=0,
                weights=np.asarray(validation_batch_sizes, dtype=np.float64),
            )
            relative_improvement = np.full_like(best_losses, np.inf)
            finite_best = np.isfinite(best_losses)
            relative_improvement[finite_best] = (
                best_losses[finite_best] - mean_validation[finite_best]
            ) / best_losses[finite_best]
            improved = np.flatnonzero(
                relative_improvement > relative_improvement_threshold
            )
            if len(improved):
                model.update_save(improved.tolist())
                best_losses[improved] = mean_validation[improved]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            holdout_loss = float(
                np.sort(mean_validation)[: int(cfg.dynamics.num_elites)].mean()
            )
            saved_elite_holdout = float(
                np.sort(best_losses)[: int(cfg.dynamics.num_elites)].mean()
            )
            if saved_elite_holdout < best_holdout:
                best_holdout = saved_elite_holdout
                best_epoch = epoch + 1
            metrics = {
                "reward_mse": reward_metric_totals["reward_sq_sum"]
                / max(1, reward_metric_totals["reward_count"]),
                "positive_reward_mse": reward_metric_totals["positive_reward_sq_sum"]
                / max(1, reward_metric_totals["positive_reward_count"]),
                "zero_reward_mse": reward_metric_totals["zero_reward_sq_sum"]
                / max(1, reward_metric_totals["zero_reward_count"]),
                "positive_reward_prediction_mean": reward_metric_totals[
                    "positive_reward_prediction_sum"
                ] / max(1, reward_metric_totals["positive_reward_count"]),
                "zero_reward_prediction_mean": reward_metric_totals[
                    "zero_reward_prediction_sum"
                ] / max(1, reward_metric_totals["zero_reward_count"]),
                "positive_target_fraction": reward_metric_totals[
                    "positive_target_count"
                ] / max(1, reward_metric_totals["target_count"]),
            }
            LOG.info(
                "Dynamics epoch %d/%d: train_loss=%.6f holdout_loss=%.6f "
                "reward_mse=%.6f positive_reward_mse=%.6f zero_reward_mse=%.6f "
                "positive_reward_prediction_mean=%.6f "
                "zero_reward_prediction_mean=%.6f positive_target_fraction=%.4f "
                "best_holdout_loss=%.6f best_epoch=%d no_improvement=%d/%d",
                epoch + 1,
                max_epochs,
                float(np.mean(train_losses)),
                holdout_loss,
                metrics["reward_mse"],
                metrics["positive_reward_mse"],
                metrics["zero_reward_mse"],
                metrics["positive_reward_prediction_mean"],
                metrics["zero_reward_prediction_mean"],
                metrics["positive_target_fraction"],
                best_holdout,
                best_epoch,
                epochs_without_improvement,
                patience,
            )
            _append_csv(
                self.metrics_dir / "dynamics_epochs.csv",
                epoch_fields,
                {
                    "epoch": epoch + 1,
                    "train_batches": len(train_losses),
                    "validation_batches": len(validation_losses),
                    "validation_samples": sum(validation_batch_sizes),
                    "train_loss": float(np.mean(train_losses)),
                    "holdout_loss": holdout_loss,
                    "best_holdout_loss": best_holdout,
                    "improved_members": len(improved),
                    "epochs_without_improvement": epochs_without_improvement,
                    **metrics,
                    **{
                        f"member_{index}_holdout_loss": float(value)
                        for index, value in enumerate(mean_validation)
                    },
                    **{
                        f"member_{index}_best_loss": float(value)
                        for index, value in enumerate(best_losses)
                    },
                },
            )
            completed_epochs = epoch + 1
            if epochs_without_improvement >= patience:
                stop_reason = "early_stopping_patience_exhausted"
                LOG.info(
                    "Dynamics early stopping triggered at epoch %d: "
                    "best_epoch=%d best_holdout_loss=%.8f patience=%d "
                    "relative_improvement_threshold=%.4f",
                    completed_epochs,
                    best_epoch,
                    best_holdout,
                    patience,
                    relative_improvement_threshold,
                )
                break

        elites = np.argsort(best_losses)[: int(cfg.dynamics.num_elites)].tolist()
        model.set_elites(elites)
        model.load_save()
        model.eval()
        if terminal_optimizer is not None:
            dynamics.terminal_classifier.eval()
            LOG.info("Learned terminal samples: train_positive=%d/%d validation_positive=%d/%d",
                     terminal_train_positive, terminal_train_total,
                     terminal_validation_positive, terminal_validation_total)
        artifact_dir = self.run_dir / "rl100_modules" / "dynamics"
        artifact_dir.mkdir()
        dynamics.save(str(artifact_dir))
        dynamics_summary = {
            "configured_max_epochs": max_epochs,
            "completed_epochs": completed_epochs,
            "optimizer_steps": global_step,
            "patience": patience,
            "relative_improvement_threshold": relative_improvement_threshold,
            "stop_reason": stop_reason,
            "best_epoch": best_epoch,
            "best_holdout_loss": float(best_holdout),
            "elite_indices": elites,
            "best_member_losses": [float(value) for value in best_losses],
            "terminal_classifier": ({
                "train_positive_samples": terminal_train_positive,
                "train_samples": terminal_train_total,
                "validation_positive_samples": terminal_validation_positive,
                "validation_samples": terminal_validation_total,
                "failure_terminal_centers_seen": terminal_failure_centers_seen,
                "threshold": dynamics.terminal_classifier_threshold,
            } if terminal_optimizer is not None else None),
            "metrics": {
                "train_steps_csv": str(self.metrics_dir / "dynamics_train_steps.csv"),
                "epochs_csv": str(self.metrics_dir / "dynamics_epochs.csv"),
            },
        }
        (self.metrics_dir / "dynamics_training_summary_rl100.json").write_text(
            json.dumps(dynamics_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        LOG.info(
            "Dynamics training stopped: reason=%s completed_epochs=%d/%d "
            "best_epoch=%d best_holdout_loss=%.8f optimizer_steps=%d",
            stop_reason,
            completed_epochs,
            max_epochs,
            best_epoch,
            best_holdout,
            global_step,
        )
        return dynamics

    def _make_bppo_cfg(self, policy) -> DictConfig:
        cfg = self.cfg
        return OmegaConf.create(
            {
                "horizon": policy.horizon,
                "n_obs_steps": policy.n_obs_steps,
                "n_action_steps": policy.n_action_steps,
                "no_pre_action": False,
                "chunk_as_single_action": True,
                "ft_all_actions": False,
                "bppo_chunk_level_ratio": True,
                "offline_chunk_ratio_mode": str(cfg.bppo.chunk_ratio_mode),
                "offline_chunk_adv_mode": str(cfg.bppo.advantage_mode),
                "chunk_adv_clip": cfg.bppo.advantage_clip,
                "distill2mean": False,
                "ppo": {
                    "enable_ratio_logging": bool(cfg.bppo.enable_ratio_logging),
                    "ratio_log_every_updates": int(cfg.bppo.ratio_log_every_updates),
                    "ratio_plot_on_final_flush": False,
                },
                "training": {"ema_decay": 0.0, "max_grad_norm": 0.5},
            }
        )

    @staticmethod
    def _terminal_classification_metrics(predictions, positives, threshold):
        predicted_positive = predictions >= threshold
        tp = int(np.logical_and(predicted_positive, positives).sum())
        fp = int(np.logical_and(predicted_positive, ~positives).sum())
        fn = int(np.logical_and(~predicted_positive, positives).sum())
        tn = int(np.logical_and(~predicted_positive, ~positives).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        false_positive_rate = fp / max(1, fp + tn)
        return {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "false_positive_rate": float(false_positive_rate),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def _calibrate_terminal_threshold(self, dynamics, validation_loader) -> None:
        """Fit the reward-terminal threshold on the fixed episode holdout."""
        cfg = self.cfg
        enabled = bool(
            OmegaConf.select(
                cfg,
                "dynamics.auto_calibrate_terminal_threshold",
                default=False,
            )
        )
        if not enabled:
            return
        if str(cfg.dynamics.terminal_mode) != "reward_threshold":
            raise ValueError(
                "Automatic terminal calibration requires "
                "dynamics.terminal_mode=reward_threshold."
            )

        prediction_parts = []
        target_parts = []
        dynamics.model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(validation_loader):
                batch = _move_to_device(batch, self.device)
                batch_size = batch["action"].shape[0]
                current = dynamics.obs2latent(batch["obs"]).reshape(batch_size, -1)
                following = dynamics.next_obs2latent(batch["next_obs"]).reshape(
                    batch_size, -1
                )
                inputs, targets = dynamics.format_samples_for_training(
                    batch, current, following
                )
                mean, _ = dynamics.model(inputs)
                elite_indices = dynamics.model.elites.detach().to(
                    device=self.device, dtype=torch.long
                )
                elite_reward_mean = mean.index_select(0, elite_indices)[..., -1].mean(0)
                prediction_parts.append(elite_reward_mean.reshape(-1).cpu())
                target_parts.append(targets[..., -1].reshape(-1).detach().cpu())
                max_batches = cfg.training.max_batches_per_epoch
                if max_batches is not None and batch_index + 1 >= int(max_batches):
                    break

        predictions = torch.cat(prediction_parts).numpy()
        targets = torch.cat(target_parts).numpy()
        if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
            raise RuntimeError("Non-finite terminal-calibration values.")
        positives = np.abs(targets) > 1e-8
        if not positives.any() or positives.all():
            raise RuntimeError(
                "Terminal calibration requires both positive and zero-reward "
                "validation samples."
            )

        threshold_min = float(
            OmegaConf.select(
                cfg,
                "dynamics.terminal_threshold_search_min",
                default=0.0,
            )
        )
        threshold_max = float(
            OmegaConf.select(
                cfg,
                "dynamics.terminal_threshold_search_max",
                default=1.0,
            )
        )
        threshold_count = int(
            OmegaConf.select(
                cfg,
                "dynamics.terminal_threshold_search_count",
                default=201,
            )
        )
        candidates = [
            self._terminal_classification_metrics(predictions, positives, threshold)
            for threshold in np.linspace(threshold_min, threshold_max, threshold_count)
        ]
        best = max(
            candidates,
            key=lambda item: (
                item["f1"],
                -item["false_positive_rate"],
                item["threshold"],
            ),
        )
        minimum_f1 = float(
            OmegaConf.select(
                cfg,
                "dynamics.minimum_terminal_validation_f1",
                default=0.0,
            )
        )
        calibration = {
            **best,
            "samples": int(targets.size),
            "positives": int(positives.sum()),
            "zeros": int((~positives).sum()),
            "minimum_required_f1": minimum_f1,
        }
        calibration_path = (
            self.run_dir
            / "rl100_modules"
            / "dynamics"
            / "terminal_calibration_rl100.json"
        )
        calibration_path.write_text(
            json.dumps(calibration, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        LOG.info(
            "Dynamics terminal calibration: threshold=%.3f precision=%.6f "
            "recall=%.6f f1=%.6f false_positive_rate=%.6f",
            best["threshold"],
            best["precision"],
            best["recall"],
            best["f1"],
            best["false_positive_rate"],
        )
        if best["f1"] < minimum_f1:
            raise RuntimeError(
                "Dynamics terminal validation F1 "
                f"{best['f1']:.6f} is below the configured minimum "
                f"{minimum_f1:.6f}; BPPO was not started."
            )

        threshold = float(best["threshold"])
        cfg.dynamics.terminal_reward_threshold = threshold
        dynamics.terminal_reward_threshold = threshold
        dynamics.cfg.dynamics.terminal_reward_threshold = threshold
        OmegaConf.save(
            cfg,
            self.run_dir / "config_resolved.yaml",
            resolve=True,
        )

    def _export_actor(self, bppo, step: int) -> Path:
        checkpoint_dir = self.run_dir / "checkpoints" / f"bppo_{step:06d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        destination = checkpoint_dir / "pretrained_model"
        return export_letools_checkpoint(
            bppo._policy,
            self.cfg.paths.bc_checkpoint,
            destination,
            bppo_step=step,
            run_name=self.run_name,
        )

    def _train_bppo(self, policy, iql, dynamics, train_loader):
        cfg = self.cfg
        bppo_cfg = self._make_bppo_cfg(policy)
        bppo = BehaviorProximalPolicyOptimization(
            policy=policy,
            device=self.device,
            policy_lr=float(cfg.bppo.lr),
            clip_ratio=float(cfg.bppo.clip_ratio),
            entropy_weight=0.0,
            decay=1.0,
            omega=0.5,
            batch_size=int(cfg.training.batch_size),
            is_iql=True,
            temperature=None,
            ratio_strategy="last",
            top_k=1,
            num_inference_steps=policy.num_inference_steps,
            fix_encoder=True,
            cfg=bppo_cfg,
        )
        # RL-100's helper reads this attribute directly although its original
        # constructor does not initialize it.
        bppo.n_obs_steps = policy.n_obs_steps
        bppo.n_action_steps = policy.n_action_steps
        ratio_dir = self.run_dir / "ratio_logs"
        bppo.set_ratio_log_dir(str(ratio_dir))
        del policy
        torch.cuda.empty_cache()

        # scalar_iql is the model-free baseline. The vdelta modes inject the
        # trained dynamics model into the actor advantage calculation.
        dynamics_for_actor = dynamics if str(cfg.bppo.advantage_mode) != "scalar_iql" else None
        batches = _infinite_batches(train_loader)
        save_every = int(cfg.bppo.save_every)
        guard_cfg = OmegaConf.select(cfg, "bppo.stability_guard")
        guard_enabled = bool(guard_cfg and guard_cfg.enabled)
        if guard_enabled and (
            not bool(cfg.bppo.enable_ratio_logging)
            or int(cfg.bppo.ratio_log_every_updates) != 1
        ):
            raise ValueError(
                "BPPO stability_guard requires enable_ratio_logging=true and "
                "ratio_log_every_updates=1."
            )
        violation_streak = 0
        bppo_fields = [
            "step", "actor_loss", "ratio_q95_max", "ratio_mean_min",
            "ratio_mean_max", "ratio_guard_violation", "violation_streak",
        ]
        for step in range(1, int(cfg.bppo.steps) + 1):
            batch = _move_to_device(next(batches), self.device)
            loss = bppo.update_distribution(
                batch=batch,
                value=iql._value,
                Q=iql.minQ,
                iql=iql,
                is_clip_decay=False,
                is_lr_decay=False,
                dynamics=dynamics_for_actor,
                use_gae=False,
                fix_encoder=True,
                final_reward=True,
                gamma=float(cfg.training.gamma),
                lamda=0.95,
            )
            if not np.isfinite(loss):
                raise FloatingPointError(f"Non-finite BPPO actor loss at step {step}.")
            LOG.info("BPPO step %d/%d: actor_loss=%.6f", step, int(cfg.bppo.steps), loss)

            summary = bppo.latest_ratio_summary() if bool(cfg.bppo.enable_ratio_logging) else None
            violated = False
            if guard_enabled and step >= int(guard_cfg.warmup_steps):
                if summary is None:
                    raise RuntimeError(
                        "BPPO stability guard received no ratio records for "
                        f"step {step}."
                    )
                violated = (
                    summary["ratio_q95_max"] > float(guard_cfg.ratio_q95_max)
                    or summary["ratio_mean_min"] < float(guard_cfg.ratio_mean_min)
                    or summary["ratio_mean_max"] > float(guard_cfg.ratio_mean_max)
                )
                violation_streak = violation_streak + 1 if violated else 0
                if violated:
                    LOG.warning(
                        "BPPO stability warning at step %d: q95_max=%.6f "
                        "mean_range=[%.6f, %.6f] consecutive=%d/%d",
                        step,
                        summary["ratio_q95_max"],
                        summary["ratio_mean_min"],
                        summary["ratio_mean_max"],
                        violation_streak,
                        int(guard_cfg.consecutive_violations),
                    )
                if violation_streak >= int(guard_cfg.consecutive_violations):
                    bppo.flush_ratio_logs(force=True)
                    exported = self._export_actor(bppo, step)
                    stop_record = {
                        "stopped_at_bppo_step": step,
                        "reason": "consecutive_ratio_guard_violations",
                        "consecutive_violations": violation_streak,
                        "limits": {
                            "ratio_q95_max": float(guard_cfg.ratio_q95_max),
                            "ratio_mean_min": float(guard_cfg.ratio_mean_min),
                            "ratio_mean_max": float(guard_cfg.ratio_mean_max),
                        },
                        "latest": summary,
                        "diagnostic_checkpoint": str(exported),
                    }
                    (self.run_dir / "stability_guard_stop_rl100.json").write_text(
                        json.dumps(stop_record, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    raise RuntimeError(
                        "BPPO stability guard stopped the run at step "
                        f"{step}. Diagnostic checkpoint: {exported}"
                    )
            _append_csv(
                self.metrics_dir / "bppo_steps.csv",
                bppo_fields,
                {
                    "step": step,
                    "actor_loss": float(loss),
                    "ratio_q95_max": "" if summary is None else summary["ratio_q95_max"],
                    "ratio_mean_min": "" if summary is None else summary["ratio_mean_min"],
                    "ratio_mean_max": "" if summary is None else summary["ratio_mean_max"],
                    "ratio_guard_violation": int(violated),
                    "violation_streak": violation_streak,
                },
            )
            if step % save_every == 0 or step == int(cfg.bppo.steps):
                exported = self._export_actor(bppo, step)
                LOG.info("Exported LeTools checkpoint: %s", exported)
        bppo.flush_ratio_logs(force=True)
        bppo_summary = {
            "configured_steps": int(cfg.bppo.steps),
            "completed_steps": int(cfg.bppo.steps),
            "stop_reason": "configured_step_limit_reached",
            "metrics": {
                "steps_csv": str(self.metrics_dir / "bppo_steps.csv"),
                "ratio_directory": str(ratio_dir),
            },
        }
        (self.metrics_dir / "bppo_training_summary_rl100.json").write_text(
            json.dumps(bppo_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        LOG.info(
            "BPPO training stopped: reason=%s completed_steps=%d",
            bppo_summary["stop_reason"],
            bppo_summary["completed_steps"],
        )
        return bppo

    def run(self) -> Path:
        LOG.info("Run directory: %s", self.run_dir)
        LOG.info("Loading LeTools BC checkpoint: %s", self.cfg.paths.bc_checkpoint)
        policy = LeToolsDPPolicyAdapter(
            self.cfg.paths.bc_checkpoint,
            device=str(self.device),
            num_inference_steps=int(self.cfg.bppo.num_inference_steps),
        )
        _dataset, train_loader, validation_loader = self._make_dataset_and_loaders(policy)

        sample = next(iter(train_loader))
        LOG.info(
            "Adapter smoke check: obs=%s action=%s reward=%s next_obs=%s",
            tuple(sample["obs"]["agent_pos"].shape),
            tuple(sample["action"].shape),
            tuple(sample["reward"].shape),
            tuple(sample["next_obs"]["agent_pos"].shape),
        )
        del sample

        iql = self._train_iql(policy, train_loader, validation_loader)
        dynamics = self._train_dynamics(policy, train_loader, validation_loader)
        self._calibrate_terminal_threshold(dynamics, validation_loader)
        self._train_bppo(policy, iql, dynamics, train_loader)
        final_step = int(self.cfg.bppo.steps)
        final_checkpoint = (
            self.run_dir / "checkpoints" / f"bppo_{final_step:06d}" / "pretrained_model"
        )
        LOG.info("Offline pipeline complete. Final checkpoint: %s", final_checkpoint)
        return final_checkpoint
