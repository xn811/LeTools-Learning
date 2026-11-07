import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, List, Tuple, Dict, Optional
from .base_dynamics import BaseDynamics

from rl_100.common.pytorch_util import dict_apply

from collections import defaultdict
class EnsembleDynamics_batch(BaseDynamics):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        # scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        env,
        normalizer,
        encoder,
        cfg,
        action_dim: int,
        gamma: float = 0.99,
        lamda: float = 0.95,
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric",
        device: str = "cpu",
        chunk_as_single_action: bool = False,
        n_action_steps: int = 1,
        prediction_mode: str = "last",  # "last" or "full"
    ) -> None:
        super().__init__(model, optim)
        # self.scaler = scaler
        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode
        self.env = env
        self.normalizer = normalizer
        self.cfg = cfg
        self.n_obs_steps = cfg.n_obs_steps
        self.use_pc_color = cfg.policy.use_pc_color
        self.predict_delta = cfg.dynamics.predict_delta
        self.use_conv_action_embed = getattr(cfg, 'use_conv_action_embed', False)
        self.encoder = encoder
        self.chunk_as_single_action = chunk_as_single_action
        self.n_action_steps = n_action_steps
        self.prediction_mode = prediction_mode  # "last": only predict last obs; "full": predict whole obs window
        # Initialize holdout losses for all model types (both diffusion and MLP ensemble now have num_ensemble)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        self.holdout_losses = [1e10 for i in range(model.num_ensemble)]
        self.lamda = lamda
        self.gamma = gamma
        self.cnt = 0
        self.action_dim = action_dim
        self.predict_r = cfg.predict_r
        self.dynamics_type = cfg.dynamics_type
        self.device = device
        self.reward_loss_weight = float(
            getattr(cfg.dynamics, "reward_loss_weight", 0.0)
        )
        self.positive_reward_weight = float(
            getattr(cfg.dynamics, "positive_reward_weight", 1.0)
        )
        self.rollout_mode = str(
            getattr(cfg.dynamics, "rollout_mode", "stochastic")
        )
        self.terminal_mode = str(
            getattr(cfg.dynamics, "terminal_mode", "external")
        )
        terminal_threshold = getattr(
            cfg.dynamics, "terminal_reward_threshold", None
        )
        self.terminal_reward_threshold = (
            None if terminal_threshold is None else float(terminal_threshold)
        )
        if self.reward_loss_weight < 0:
            raise ValueError("dynamics.reward_loss_weight must be non-negative.")
        if self.positive_reward_weight < 1:
            raise ValueError("dynamics.positive_reward_weight must be at least 1.")
        if self.rollout_mode not in {"stochastic", "elite_mean"}:
            raise ValueError(
                "dynamics.rollout_mode must be 'stochastic' or 'elite_mean'."
            )
        if self.terminal_mode not in {"external", "reward_threshold"}:
            raise ValueError(
                "dynamics.terminal_mode must be 'external' or 'reward_threshold'."
            )
        if (
            self.terminal_mode == "reward_threshold"
            and self.terminal_reward_threshold is None
        ):
            raise ValueError(
                "dynamics.terminal_reward_threshold is required when "
                "terminal_mode='reward_threshold'."
            )
        if (
            self.terminal_reward_threshold is not None
            and not np.isfinite(self.terminal_reward_threshold)
        ):
            raise ValueError("dynamics.terminal_reward_threshold must be finite.")
    def set_logger(self, logger):
        self.logger = logger
        self.logger.log("Training dynamics:")

    @staticmethod
    def _q_requires_raw_obs(Q: Callable[[torch.tensor, torch.tensor], torch.tensor]) -> bool:
        q_owner = getattr(Q, '__self__', None)
        if q_owner is not None and getattr(q_owner, 'eval_with_raw_obs', False):
            return True
        if q_owner is not None and hasattr(q_owner, 'is_share_encoder'):
            return not bool(q_owner.is_share_encoder)

        q_module = getattr(q_owner, '_Q', None) if q_owner is not None else Q
        return getattr(q_module, '_obs_encoder', None) is not None

    @staticmethod
    def _as_column_tensor(value, ref: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=ref.device, dtype=ref.dtype)
        tensor = tensor.reshape(ref.shape[0], -1)
        if tensor.shape[1] != 1:
            tensor = tensor[:, :1]
        return tensor

    @staticmethod
    def _as_chunk_reward_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3:
            tensor = tensor[..., 0]
        return tensor

    def _discounted_chunk_rewards(
        self,
        reward_chunk: torch.Tensor,
        not_done_chunk: Optional[torch.Tensor] = None,
        done_chunk: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reward_chunk = self._as_chunk_reward_tensor(reward_chunk)
        gamma_weights = torch.pow(
            torch.tensor(self.gamma, device=reward_chunk.device, dtype=reward_chunk.dtype),
            torch.arange(reward_chunk.shape[1], device=reward_chunk.device, dtype=reward_chunk.dtype),
        )
        if not_done_chunk is None and done_chunk is not None:
            done_chunk = self._as_chunk_reward_tensor(done_chunk).to(
                device=reward_chunk.device,
                dtype=reward_chunk.dtype,
            )
            not_done_chunk = 1.0 - done_chunk
        if not_done_chunk is not None:
            not_done_chunk = self._as_chunk_reward_tensor(not_done_chunk).to(
                device=reward_chunk.device,
                dtype=reward_chunk.dtype,
            )
            prior_not_done = torch.ones_like(reward_chunk)
            if reward_chunk.shape[1] > 1:
                prior_not_done[:, 1:] = torch.cumprod(not_done_chunk[:, :-1], dim=1)
            reward_chunk = reward_chunk * prior_not_done
        return torch.sum(reward_chunk * gamma_weights, dim=-1, keepdim=True)

    def _return_and_gae(
        self,
        rewards,
        terminals,
        q_values=None,
        final_bootstrap: Optional[torch.Tensor] = None,
        gamma: Optional[float] = None,
    ):
        if gamma is None:
            gamma = self.gamma
        if q_values is not None and len(q_values) > 0:
            if len(q_values) != len(rewards):
                raise ValueError(
                    f"Expected one q_value per reward, got "
                    f"{len(q_values)} q_values and {len(rewards)} rewards."
                )
            ref = self._as_column_tensor(q_values[0], q_values[0])
            q_values = [self._as_column_tensor(q_value, ref) for q_value in q_values]
        elif final_bootstrap is not None:
            ref = self._as_column_tensor(final_bootstrap, final_bootstrap)
        else:
            raise ValueError("Need q_values or final_bootstrap to infer rollout shape.")
        if final_bootstrap is not None:
            final_bootstrap = self._as_column_tensor(final_bootstrap, ref)

        returns = torch.zeros_like(ref)
        discount = torch.ones_like(ref)
        alive = torch.ones_like(ref)
        reward_tensors, alive_tensors, nonterminals = [], [], []

        for reward, terminal in zip(rewards, terminals):
            r = self._as_column_tensor(reward, ref)
            terminal_t = self._as_column_tensor(terminal, ref).clamp(0.0, 1.0)
            alive_before = alive
            nonterminal = alive_before * (1.0 - terminal_t)

            masked_reward = alive_before * r
            returns = returns + discount * masked_reward
            reward_tensors.append(masked_reward)
            alive_tensors.append(alive_before)
            nonterminals.append(nonterminal)

            discount = discount * gamma
            alive = nonterminal

        if final_bootstrap is not None:
            returns = returns + discount * alive * final_bootstrap

        gae_advantages = None
        if q_values is not None and len(q_values) > 0:
            final_q = final_bootstrap if final_bootstrap is not None else torch.zeros_like(ref)
            deltas = []
            for i, q_value in enumerate(q_values):
                next_q = q_values[i + 1] if i < len(q_values) - 1 else final_q
                delta = (
                    reward_tensors[i]
                    + gamma * nonterminals[i] * next_q
                    - alive_tensors[i] * q_value
                )
                deltas.append(delta)

            gae = torch.zeros_like(ref)
            gae_advantages = []
            for i in reversed(range(len(deltas))):
                gae = deltas[i] + gamma * self.lamda * nonterminals[i] * gae
                gae_advantages.insert(0, gae)
            gae_advantages = torch.stack(gae_advantages).squeeze(-1)

        return returns, gae_advantages

    @ torch.no_grad()
    def step(
        self,
        nobs_features: torch.tensor,
        action: torch.tensor,
        policy_features: torch.tensor = None,  # 完整观测窗口 [B, n_obs_steps, feature_dim]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Imagine single forward step.

        Args:
            nobs_features: single obs feature [B, feature_dim]
            action: [B, action_dim] or [B, n_action_steps, action_dim]
            policy_features: whole obss window features [B, n_obs_steps, feature_dim]

        Returns:
            next_obs:
                - "last" mode: [B, feature_dim] - single obs prediction
                - "full" mode: [B, n_obs_steps, feature_dim] - whole obss window prediction
        """
        batch_size = nobs_features.shape[0]
        if self.use_conv_action_embed:
            model = self.model.module if hasattr(self.model, 'module') else self.model
            action_t = torch.as_tensor(action, dtype=torch.float32).to(nobs_features.device if isinstance(nobs_features, torch.Tensor) else self.device)
            Da = model._single_action_dim
            a_chunk = action_t.reshape(batch_size, -1, Da)
            z = model._conv_action_encoder(a_chunk)
            action = model._conv_action_layer_norm(z.reshape(batch_size, -1))
        elif self.cfg.use_action_embed:
            action = action.reshape(batch_size, -1)
            # Handle DDP wrapper - unified for both diffusion and MLP ensemble
            model = self.model.module if hasattr(self.model, 'module') else self.model
            action = model._action_encoder(action)
        else:
            # Raw chunk action path. Always convert to tensor first so downstream
            # .cpu().numpy() and the optional LayerNorm both work whether the
            # caller passed np.ndarray or torch.Tensor.
            model = self.model.module if hasattr(self.model, 'module') else self.model
            target_device = nobs_features.device if isinstance(nobs_features, torch.Tensor) else self.device
            action = torch.as_tensor(action, dtype=torch.float32).to(target_device).reshape(batch_size, -1)
            if getattr(model, 'use_action_scale_norm', False):
                action = model._action_scale_layer_norm(action)
        action = action.reshape(batch_size, -1)

        if self.prediction_mode == "full" and policy_features is not None:
            input_features = policy_features.reshape(batch_size, -1)  # [B, n_obs_steps * feature_dim]
        else:
            input_features = nobs_features  # [B, feature_dim]

        input_features_np, action_np = input_features.cpu().data.numpy(), action.cpu().data.numpy()
        obs_act = np.concatenate([input_features_np, action_np], axis=-1)

        # Unified handling for both diffusion and MLP ensemble models
        # Both now return (mean, logvar) format thanks to EnsembleDiffusionDynamicsModel wrapper
        mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        is_diffusion_dynamics = bool(getattr(model, "is_diffusion_dynamics", False))

        if self.predict_delta:
            if self.cfg.predict_r:
                mean[..., :-1] += input_features_np
            else:
                mean += input_features_np

        std = np.sqrt(np.exp(logvar))
        # Handle DDP wrapper for method calls.
        model = self.model.module if hasattr(self.model, 'module') else self.model
        if self.rollout_mode == "elite_mean":
            elite_indices = model.elites.detach().cpu().numpy().astype(np.int64)
            samples = mean[elite_indices].mean(axis=0).astype(np.float32)
        else:
            if is_diffusion_dynamics:
                ensemble_samples = mean.astype(np.float32)
            else:
                ensemble_samples = (
                    mean + np.random.normal(size=mean.shape) * std
                ).astype(np.float32)
            _, batch_size, _ = ensemble_samples.shape
            model_idxs = model.random_elite_idxs(batch_size)
            samples = ensemble_samples[model_idxs, np.arange(batch_size)]

        if self.cfg.predict_r:
            next_obs = samples[..., :-1]
            reward = samples[..., -1:]
        else:
            next_obs = samples
            reward = np.zeros((next_obs.shape[0], 1), dtype=np.float32)

        if self.terminal_mode == "reward_threshold":
            terminal = (reward >= self.terminal_reward_threshold).astype(np.bool_)
        else:
            terminal = self.terminal_fn(
                input_features_np, action_np, next_obs, self.env
            )
        info = {}
        info["raw_reward"] = reward

        if self._penalty_coef:
            if self._uncertainty_mode == "aleatoric":
                penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
            elif self._uncertainty_mode == "pairwise-diff":
                next_obses_mean = mean[..., :-1]
                next_obs_mean = np.mean(next_obses_mean, axis=0)
                diff = next_obses_mean - next_obs_mean
                penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
            elif self._uncertainty_mode == "ensemble_std":
                next_obses_mean = mean[..., :-1]
                penalty = np.sqrt(next_obses_mean.var(0).mean(1))
            else:
                raise ValueError
            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            assert penalty.shape == reward.shape
            reward = reward - self._penalty_coef * penalty
            info["penalty"] = penalty

        if self.prediction_mode == "full":
            feature_dim = next_obs.shape[-1] // self.n_obs_steps
            next_obs = next_obs.reshape(batch_size, self.n_obs_steps, feature_dim)  # [B, n_obs_steps, feature_dim]

        return next_obs, reward, terminal, info
    @ torch.no_grad()
    def multi_step(
        self,
        single_nob_features: torch.tensor,
        nobs_features: torch.tensor,
        nactions: torch.tensor,
        reward_strategy: str = "sum",
        discount: float = 1.0,
        Return: float = 0.0,
        Qs: List[torch.tensor] = None,
        Q: Callable[[torch.tensor, torch.tensor], torch.tensor] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Multi-step rollout (single action per step).
        Supports both "last" and "full" prediction modes.
        """
        n_step_actions = nactions.shape[1]
        all_obs_features, rewards, terminals, infos = [], [], [], []
        rewards = 0
        batch_size = nactions.shape[0]
        for i in range(n_step_actions):
            if Q != None:
                Qs.append(Q(nobs_features.reshape(batch_size, -1), nactions[:, i, :self.action_dim]))
            next_obs, reward, terminal, info = self.step(single_nob_features, nactions[:, i, :self.action_dim], nobs_features)
            if reward_strategy == "sum":
                rewards += reward
            Return += discount * reward
            discount *= self.gamma
            # Handle DDP wrapper for device access
            model = self.model.module if hasattr(self.model, 'module') else self.model
            device = model._device if hasattr(model, '_device') else next(model.parameters()).device

            if self.prediction_mode == "full":
                nobs_features = torch.from_numpy(next_obs).to(device)
                single_nob_features = nobs_features[:, -1, :]
            else:
                single_nob_features = torch.from_numpy(next_obs).to(device)
                nobs_features = torch.cat((nobs_features[:, 1:, :], single_nob_features.unsqueeze(1)), dim=1)

        return next_obs, rewards, terminal, info, Return, Qs, discount

    @ torch.no_grad()
    def multi_step_evaluation(
        self,
        nobs_features: torch.tensor,
        nactions: torch.tensor,
        Q: Callable[[torch.tensor, torch.tensor], torch.tensor],
        state_dict: Dict = None,
        use_gae: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        n_step_actions = nactions.shape[1]
        all_obs_features, rewards, terminals, infos, Qs = [], [], [], [], []
        G, discount = 0, 1
        use_state_dict_q = self._q_requires_raw_obs(Q)
        if use_state_dict_q and n_step_actions > 1:
            raise ValueError(
                "multi_step_evaluation cannot update raw observation inputs "
                "across latent dynamics rollout when Q owns its encoder."
            )
        batch_size = nactions.shape[0]
        policy_features = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
        single_nob_features = policy_features[:, -1, :]

        for i in range(n_step_actions):

            if use_state_dict_q:
                q_input = state_dict
            else:
                q_input = policy_features.reshape(batch_size, -1)
            Qs.append(Q(q_input, nactions[:, i, :self.action_dim]))
            all_obs_features.append(policy_features)
            next_obs, reward, terminal, info = self.step(single_nob_features, nactions[:, i, :self.action_dim], policy_features)
            G += discount * reward
            discount *= self.gamma
            rewards.append(reward); terminals.append(terminal); infos.append(info)
            # Handle DDP wrapper for device access
            model = self.model.module if hasattr(self.model, 'module') else self.model
            device = model._device if hasattr(model, '_device') else next(model.parameters()).device

            if self.prediction_mode == "full":
                # "full" 模式: next_obs 是 [B, n_obs_steps, feature_dim]
                policy_features = torch.from_numpy(next_obs).to(device)
                single_nob_features = policy_features[:, -1, :]
            else:
                # "last" 模式: next_obs 是 [B, feature_dim]
                single_nob_features = torch.from_numpy(next_obs).to(device)
                policy_features = torch.cat((policy_features[:, 1:, :], single_nob_features.unsqueeze(1)), dim=1)

        if use_gae:
            G_tensor, gae_advantages = self._return_and_gae(
                rewards,
                terminals,
                q_values=Qs,
                final_bootstrap=None,
                gamma=self.gamma,
            )
            return all_obs_features, rewards, terminals, infos, G_tensor.squeeze(-1), gae_advantages
        else:
            bootstrap_q = Qs[-1].detach().cpu()
            G_tensor = torch.as_tensor(G, dtype=bootstrap_q.dtype) + discount * bootstrap_q
            return all_obs_features, rewards, terminals, infos, G_tensor.squeeze(), None
    @ torch.no_grad()
    def chunk_evaluation(
        self,
        nobs_features: torch.tensor,
        nactions: torch.tensor,
        Q: Callable[[torch.tensor, torch.tensor], torch.tensor],
        state_dict: Dict = None,
        use_gae: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        batch_size = nactions.shape[0]
        policy_features = nobs_features.reshape(batch_size, self.n_obs_steps, -1) # if n_obs_steps == 1, policy_features = nobs_features.unsqueeze(1) # B, n_obs, feature_dim
        use_state_dict_q = self._q_requires_raw_obs(Q)
        if use_state_dict_q:
            if state_dict is None:
                raise ValueError(
                    "chunk_evaluation requires raw state_dict when Q owns "
                    "its observation encoder."
                )
            q_input = state_dict
        else:
            # Q was trained with full flattened state features (n_obs_steps * feature_dim),
            # not just the last frame. Use the full flattened features to match training.
            q_input = policy_features.reshape(batch_size, -1)  # B, n_obs_steps * feature_dim
        q_owner = getattr(Q, '__self__', None)
        if q_owner is not None and hasattr(q_owner, 'get_advantage'):
            # IQL candidate reranking uses A(s, a)=Q(s, a)-V(s); for one
            # state's candidates this preserves the Q ordering.
            return q_owner.get_advantage(q_input, nactions)
        return Q(q_input, nactions)

    def obs2latent(self, nobs):
        nobs = self.normalizer.normalize(nobs)
        batch_size = nobs['agent_pos'].shape[0]
        if not self.use_pc_color:
            if 'point_cloud' in nobs:
                nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        this_nobs = dict_apply(nobs,
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))

        if self.cfg.dynamics.fix_encoder:
            self.encoder.eval()
            with torch.no_grad():
                nobs_features = self.encoder(this_nobs)
        else:
            nobs_features = self.encoder(this_nobs)

        nobs_features = nobs_features.reshape(batch_size, self.n_obs_steps, -1) # B, n_obs, feature_dim
        return nobs_features
    def next_obs2latent(self, nobs):
        nobs = self.normalizer.normalize(nobs)
        batch_size = nobs['agent_pos'].shape[0]
        if not self.use_pc_color:
            if 'point_cloud' in nobs:
                nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        this_nobs = dict_apply(nobs,
            lambda x: x[:,-self.n_obs_steps:,...].reshape(-1,*x.shape[2:])) # TODO for chunk as single action, next obs should be the last n_obs_steps of the next obs

        if self.cfg.dynamics.fix_encoder:
            self.encoder.eval()
            with torch.no_grad():
                nobs_features = self.encoder(this_nobs)
        else:
            nobs_features = self.encoder(this_nobs)

        nobs_features = nobs_features.reshape(batch_size, self.n_obs_steps, -1) # B, n_obs, feature_dim
        return nobs_features
    def format_samples_for_training(self, data: Dict, nobs_features: torch.tensor, next_nobs_features: torch.tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Format samples for training.

        Args:
            nobs_features:
                - "last" mode: [B, feature_dim] (single_nob_features)
                - "full" mode: [B, n_obs_steps * feature_dim] (flattened full features)
            next_nobs_features:
                - "last" mode: [B, feature_dim] (single_next_nob_features)
                - "full" mode: [B, n_obs_steps * feature_dim] (flattened full features)
        """
        batch_size = data['action'].shape[0]

        if self.predict_delta:
            targets = next_nobs_features - nobs_features  # shape depends on prediction_mode
        else:
            targets = next_nobs_features

        if self.cfg.action_norm:
            actions = self.normalizer['action'].normalize(data['action'])
        else:
            actions = data['action']

        if self.cfg.chunk_as_single_action:
            # 参考 critic.py，取 n_action_steps 个 action
            action = actions[:, self.n_obs_steps - 1 : self.n_obs_steps - 1 + self.n_action_steps]

            if self.predict_r:
                # 取 n_action_steps 个 reward 并计算 discounted sum -> [B, 1]
                reward_chunk = data["reward"][:, self.n_obs_steps - 1 : self.n_obs_steps - 1 + self.n_action_steps]
                not_done_chunk = None
                done_chunk = None
                if "not_done" in data:
                    not_done_chunk = data["not_done"][:, self.n_obs_steps - 1 : self.n_obs_steps - 1 + self.n_action_steps]
                elif "done" in data:
                    done_chunk = data["done"][:, self.n_obs_steps - 1 : self.n_obs_steps - 1 + self.n_action_steps]
                rewards = self._discounted_chunk_rewards(reward_chunk, not_done_chunk, done_chunk)
        else:
            action = actions[:, self.n_obs_steps - 1]
            if self.predict_r:
                rewards = data["reward"][:, self.n_obs_steps - 1]

        if self.use_conv_action_embed:
            model = self.model.module if hasattr(self.model, 'module') else self.model
            Da = model._single_action_dim
            a_chunk = action.reshape(batch_size, -1, Da)
            z = model._conv_action_encoder(a_chunk)
            action = model._conv_action_layer_norm(z.reshape(batch_size, -1))
        elif self.cfg.use_action_embed:
            action = action.reshape(batch_size, -1)
            # Handle DDP wrapper - unified for both diffusion and MLP ensemble
            model = self.model.module if hasattr(self.model, 'module') else self.model
            action = model._action_encoder(action)
        else:
            model = self.model.module if hasattr(self.model, 'module') else self.model
            if getattr(model, 'use_action_scale_norm', False):
                action = model._action_scale_layer_norm(action.reshape(batch_size, -1))

        batch_size = nobs_features.shape[0]
        inputs = torch.cat((nobs_features, action.reshape(batch_size, -1)), dim=-1)
        if self.predict_r:
            targets = torch.cat((targets, rewards.reshape(batch_size, -1)), dim=-1)
        return inputs, targets

    def post_well_learned(self,) -> None:
        # Unified elite selection for both diffusion and MLP ensemble
        indexes = self.select_elites(self.holdout_losses)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.set_elites(indexes)
        model.load_save()
        self.logger.log("elites:{} , holdout loss: {}".format(indexes, (np.sort(self.holdout_losses)[:model.num_elites]).mean()))
        self.save(self.logger.model_dir)
        self.model.eval()

    @ torch.no_grad()
    def validation(self, holdout_data: dict, nobs_features: torch.tensor, next_nobs_features: torch.tensor, train_loss: float, wandb, epoch, max_epochs_since_update=5, max_epochs=500) -> list:
        holdout_inputs, holdout_targets = self.format_samples_for_training(data=holdout_data, nobs_features=nobs_features, next_nobs_features=next_nobs_features,)
        # holdout_losses = [1e10 for i in range(self.model.num_ensemble)]
        new_holdout_losses = self.validate(holdout_inputs, holdout_targets)
        return self._update_holdout_and_log(new_holdout_losses, train_loss, wandb, epoch, max_epochs_since_update, max_epochs)

    def _update_holdout_and_log(self, new_holdout_losses: list, train_loss: float, wandb, epoch, max_epochs_since_update=5, max_epochs=500) -> bool:
        """
        Update holdout losses and log metrics. Separated from validation() for batched validation support.

        Args:
            new_holdout_losses: List of validation losses for each ensemble member
            train_loss: Training loss to log
            wandb: Wandb logger
            epoch: Current epoch

        Returns:
            True if training should stop (well-learned), False otherwise
        """
        # Handle DDP wrapper when accessing model attributes
        model = self.model.module if hasattr(self.model, 'module') else self.model
        holdout_loss = (np.sort(new_holdout_losses)[:model.num_elites]).mean()

        # add logger
        self.logger.logkv("loss/dynamics_train_loss", train_loss)
        self.logger.logkv("loss/dynamics_holdout_loss", holdout_loss)
        self.logger.set_timestep(epoch)
        self.logger.dumpkvs(exclude=["policy_training_progress"])

        # Log key-value metrics
        wandb.log({"loss/dynamics_train_loss": train_loss})
        wandb.log({"loss/dynamics_holdout_loss": holdout_loss})

        indexes = []
        for i, new_loss, old_loss in zip(range(len(self.holdout_losses)), new_holdout_losses, self.holdout_losses):
            improvement = (old_loss - new_loss) / old_loss
            if improvement > 0.01:
                indexes.append(i)
                self.holdout_losses[i] = new_loss

        if len(indexes) > 0:
            # Handle DDP wrapper for method calls
            model = self.model.module if hasattr(self.model, 'module') else self.model
            model.update_save(indexes)
            self.cnt = 0
        else:
            self.cnt += 1

        if (self.cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
            return True # well-learned
        else:
            return False # continuing training

    def learn(
        self,
        batch: dict,
        nobs_features: torch.tensor,
        next_nobs_features: torch.tensor,
        logvar_loss_coef: float = 0.01
    ) -> float:
        # For eval/train mode, we keep it on the DDP wrapper
        self.model.train()
        model = self.model.module if hasattr(self.model, 'module') else self.model

        if self.use_conv_action_embed and not hasattr(model, 'compute_loss'):
            # Conv AE path: use forward_train() so all params (encoder, decoder, backbone)
            # are used inside a single DDP-compatible forward pass.
            batch_size = nobs_features.shape[0]
            if self.predict_delta:
                targets = next_nobs_features - nobs_features
            else:
                targets = next_nobs_features

            if self.cfg.action_norm:
                actions = self.normalizer['action'].normalize(batch['action'])
            else:
                actions = batch['action']
            if self.cfg.chunk_as_single_action:
                action = actions[:, self.n_obs_steps - 1 : self.n_obs_steps - 1 + self.n_action_steps]
                if self.predict_r:
                    reward_chunk = batch["reward"][:, self.n_obs_steps - 1 : self.n_obs_steps - 1 + self.n_action_steps]
                    not_done_chunk = None
                    done_chunk = None
                    if "not_done" in batch:
                        not_done_chunk = batch["not_done"][:, self.n_obs_steps - 1 : self.n_obs_steps - 1 + self.n_action_steps]
                    elif "done" in batch:
                        done_chunk = batch["done"][:, self.n_obs_steps - 1 : self.n_obs_steps - 1 + self.n_action_steps]
                    rewards = self._discounted_chunk_rewards(reward_chunk, not_done_chunk, done_chunk)
            else:
                action = actions[:, self.n_obs_steps - 1]
                if self.predict_r:
                    rewards = batch["reward"][:, self.n_obs_steps - 1]

            if self.predict_r:
                targets = torch.cat((targets, rewards.reshape(batch_size, -1)), dim=-1)

            Da = model._single_action_dim
            a_chunk = action.reshape(batch_size, -1, Da)
            action_recon_beta = getattr(self.cfg, 'action_recon_beta', 0.5)
            loss = self.model(nobs_features, targets=targets, action_chunk=a_chunk,
                              logvar_loss_coef=logvar_loss_coef, action_recon_beta=action_recon_beta)
            return loss

        # Non-conv path: unchanged
        inputs_batch, targets_batch = self.format_samples_for_training(batch, nobs_features, next_nobs_features)

        # Check if model has compute_loss method (diffusion model)
        if hasattr(model, 'compute_loss'):
            loss = model.compute_loss(inputs_batch, targets_batch)
        else:
            # MLP ensemble: use variance-weighted loss
            mean, logvar = self.model(inputs_batch)
            inv_var = torch.exp(-logvar)
            # Average over batch and dim, sum over ensembles
            mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss = loss + model.get_decay_loss()
            loss = loss + logvar_loss_coef * model.max_logvar.sum() - logvar_loss_coef * model.min_logvar.sum()
            if self.predict_r and self.reward_loss_weight > 0:
                reward_target = targets_batch[..., -1]
                reward_prediction = mean[..., -1]
                reward_weights = torch.where(
                    reward_target.abs() > 1e-8,
                    torch.full_like(reward_target, self.positive_reward_weight),
                    torch.ones_like(reward_target),
                )
                reward_squared_error = (reward_prediction - reward_target) ** 2
                reward_mse = (
                    reward_squared_error * reward_weights.unsqueeze(0)
                ).sum() / (
                    reward_weights.sum() * reward_prediction.shape[0]
                ).clamp_min(1.0)
                loss = loss + self.reward_loss_weight * reward_mse

        return loss

    def optimize(self, loss: float) -> None:
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    @ torch.no_grad()
    def rollout(self,
        policy: nn.Module,
        Q: nn.Module,
        iql: nn.Module,
        batch: dict,
        rollout_length: int,
        is_iql: bool = True,
        use_gae: bool = False,
        first_action: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        if is_iql:
            q_eval = iql.minQ
        else:
            q_eval = Q
        num_transitions = 0
        rewards_arr = np.array([])
        total_q = np.array([])
        rollout_transitions = defaultdict(list)

        # nobs_features = self.obs2latent(init_obss)
        # rollout
        # observations = init_obss
        length = 0
        rollout_rewards, rollout_terminals, rollout_qs = [], [], []
        # get latent obs from encoder
        batch_size = batch['obs']['agent_pos'].shape[0]
        nobs_features = policy.obs2latent(batch['obs'], eval_policy=True)
        policy_features = nobs_features.reshape(batch_size, self.n_obs_steps, -1) # if n_obs_steps == 1, policy_features = nobs_features.unsqueeze(1) # B, n_obs, feature_dim
        single_nob_features = policy_features[:, -1, :] # B, feature_dim
        if self.cfg.chunk_as_single_action:
            for _ in range(int(rollout_length)):
                actions = policy.sample_action(policy_features, get_np = False, batch_size=batch_size)
                rollout_qs.append(q_eval(policy_features.reshape(batch_size, -1), actions))
                next_obs, reward, terminal, info = self.step(single_nob_features, actions, policy_features)
                rollout_rewards.append(reward)
                rollout_terminals.append(terminal)
                rewards_arr = np.append(rewards_arr, reward.flatten())
                # Handle DDP wrapper for device access
                model = self.model.module if hasattr(self.model, 'module') else self.model
                device = model._device if hasattr(model, '_device') else next(model.parameters()).device

                if self.prediction_mode == "full":
                    policy_features = torch.from_numpy(next_obs).to(device)  # [B, n_obs_steps, feature_dim]
                    single_nob_features = policy_features[:, -1, :]  # [B, feature_dim]
                else:
                    single_nob_features = torch.from_numpy(next_obs).to(device)
                    policy_features = torch.cat((policy_features[:, 1:, :], single_nob_features.unsqueeze(1)), dim=1)

            if use_gae:
                final_actions = policy.sample_action(policy_features, get_np=False, batch_size=batch_size)
                final_bootstrap = q_eval(policy_features.reshape(batch_size, -1), final_actions)
                _, gae_advantages = self._return_and_gae(
                    rollout_rewards,
                    rollout_terminals,
                    q_values=rollout_qs,
                    final_bootstrap=final_bootstrap,
                    gamma=self.gamma ** self.n_action_steps,
                )
            else:
                q_evaluation = torch.mean(torch.stack(rollout_qs))
        else:
            if rollout_length < self.cfg.n_action_steps or rollout_length % self.cfg.n_action_steps != 0:
                raise ValueError(
                    "non-chunk dynamics rollout expects rollout_length to be "
                    f"a positive multiple of n_action_steps, got "
                    f"rollout_length={rollout_length}, "
                    f"n_action_steps={self.cfg.n_action_steps}"
                )
            n_chunks = max(1, int(rollout_length / self.cfg.n_action_steps))
            for _ in range(n_chunks):

                actions = policy.sample_action(policy_features, get_np = False, batch_size=batch_size)
                for i in range(actions.shape[1]):
                    action_i = actions[:, i, :self.action_dim]
                    rollout_qs.append(q_eval(policy_features.reshape(batch_size, -1), action_i))
                    next_obs, reward, terminal, info = self.step(single_nob_features, action_i, policy_features)
                    rollout_rewards.append(reward)
                    rollout_terminals.append(terminal)
                    rewards_arr = np.append(rewards_arr, reward.flatten())
                    # Handle DDP wrapper for device access
                    model = self.model.module if hasattr(self.model, 'module') else self.model
                    device = model._device if hasattr(model, '_device') else next(model.parameters()).device

                    if self.prediction_mode == "full":
                        policy_features = torch.from_numpy(next_obs).to(device)
                        single_nob_features = policy_features[:, -1, :]
                    else:
                        single_nob_features = torch.from_numpy(next_obs).to(device)
                        policy_features = torch.cat((policy_features[:, 1:, :], single_nob_features.unsqueeze(1)), dim=1)

            if use_gae:
                final_actions = policy.sample_action(policy_features, get_np=False, batch_size=batch_size)
                final_bootstrap = q_eval(
                    policy_features.reshape(batch_size, -1),
                    final_actions[:, 0, :self.action_dim],
                )
                _, gae_advantages = self._return_and_gae(
                    rollout_rewards,
                    rollout_terminals,
                    q_values=rollout_qs,
                    final_bootstrap=final_bootstrap,
                    gamma=self.gamma,
                )
            else:
                q_evaluation = torch.mean(torch.stack(rollout_qs))

        if use_gae:
            if first_action:
                q_evaluation = torch.mean(gae_advantages[0])
            else:
                q_evaluation = torch.mean(gae_advantages)

        return q_evaluation, rewards_arr.mean()

    @ torch.no_grad()
    def validate(self, inputs: np.ndarray, targets: np.ndarray) -> List[float]:
        # For eval/train mode, we keep it on the DDP wrapper
        self.model.eval()
        # Handle DDP wrapper for device access
        model = self.model.module if hasattr(self.model, 'module') else self.model
        device = model._device if hasattr(model, '_device') else next(model.parameters()).device
        targets = torch.as_tensor(targets).to(device)
        mean, _ = self.model(inputs)
        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss

    @torch.no_grad()
    def validate_with_reward_metrics(self, inputs, targets):
        """Return ensemble validation loss plus non-diluted reward metrics."""
        self.model.eval()
        model = self.model.module if hasattr(self.model, 'module') else self.model
        device = model._device if hasattr(model, '_device') else next(model.parameters()).device
        targets = torch.as_tensor(targets, device=device, dtype=torch.float32)
        mean, _ = self.model(inputs)
        squared_error = (mean - targets) ** 2
        val_loss = list(squared_error.mean(dim=(1, 2)).cpu().numpy())

        reward_target = targets[..., -1]
        reward_prediction = mean[..., -1]
        reward_squared_error = (reward_prediction - reward_target) ** 2
        positive = reward_target.abs() > 1e-8
        positive_expanded = positive.unsqueeze(0).expand_as(reward_prediction)
        zero_expanded = ~positive_expanded
        metrics = {
            "reward_sq_sum": float(reward_squared_error.sum().cpu()),
            "reward_count": int(reward_squared_error.numel()),
            "positive_reward_sq_sum": float(
                reward_squared_error[positive_expanded].sum().cpu()
            ),
            "positive_reward_count": int(positive_expanded.sum().cpu()),
            "zero_reward_sq_sum": float(
                reward_squared_error[zero_expanded].sum().cpu()
            ),
            "zero_reward_count": int(zero_expanded.sum().cpu()),
            "positive_target_count": int(positive.sum().cpu()),
            "target_count": int(positive.numel()),
            "positive_reward_prediction_sum": float(
                reward_prediction[positive_expanded].sum().cpu()
            ),
            "zero_reward_prediction_sum": float(
                reward_prediction[zero_expanded].sum().cpu()
            ),
        }
        return val_loss, metrics

    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        # Handle DDP wrapper when accessing model attributes
        model = self.model.module if hasattr(self.model, 'module') else self.model
        elites = [pairs[i][1] for i in range(model.num_elites)]
        return elites

    def save(self, save_path: str) -> None:
        # Handle DDP-wrapped models by saving the underlying module's state_dict
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), os.path.join(save_path, "dynamics.pth"))
        print('dynamics model saved in {}'.format(str(save_path)))

    def load(self, load_path: str) -> None:
        # Handle DDP wrapper for loading state dict
        model = self.model.module if hasattr(self.model, 'module') else self.model
        # Also handle _device access
        device = model._device if hasattr(model, '_device') else next(model.parameters()).device
        model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=device))
        print('dynamics model loaded from {}'.format(str(load_path)))
