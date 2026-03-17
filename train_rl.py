#!/usr/bin/env python3
"""
PPO Training for Attention-Based Coupling Discovery
=====================================================
Trains the AttentionCouplingPolicy to learn the inter-organ coupling equation
between CircAdapt (heart) and Hallow (kidney) models using Proximal Policy
Optimization (Schulman et al. 2017).

Two-stage training:
  Stage 1: Synthetic pre-training with dense per-step reward
  Stage 2: Fine-tuning with terminal V7-only reward (reduced shaped reward)

Usage:
    python train_rl.py --stage 1 --n_iter 500 --save models/rl_stage1.pt
    python train_rl.py --stage 2 --n_iter 200 --load models/rl_stage1.pt --save models/rl_attention_policy.pt
"""

import argparse
import time
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from config import RL_CONFIG, CARDIAC_FEATURE_NAMES, RENAL_FEATURE_NAMES, META_FEATURE_NAMES, TEMPORAL_FEATURE_NAMES
from models.attention_coupling import AttentionCouplingPolicy
from rl_env import CardiorenalCouplingEnv


class RolloutBuffer:
    """Stores transitions from environment rollouts for PPO updates.

    Each entry corresponds to one (obs, action, reward, done) transition.
    After collection, compute_returns_and_advantages() must be called
    to fill in GAE advantages and discounted returns.
    """

    def __init__(self):
        self.observations = []      # list of (32,) np arrays
        self.actions = []           # list of (15,) np arrays
        self.log_probs = []         # list of floats
        self.rewards = []           # list of floats
        self.values = []            # list of floats
        self.dones = []             # list of bools
        self.advantages = None      # (N,) tensor after compute
        self.returns = None         # (N,) tensor after compute

    def add(self, obs, action, log_prob, reward, value, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95,
    ):
        """Compute GAE advantages and discounted returns."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        self.advantages = torch.from_numpy(advantages)
        self.returns = torch.from_numpy(returns)

    def get_tensors(self) -> Dict[str, torch.Tensor]:
        """Convert buffer to tensors for PPO update."""
        obs = np.array(self.observations, dtype=np.float32)
        acts = np.array(self.actions, dtype=np.float32)
        return {
            'observations': torch.from_numpy(obs),
            'actions': torch.from_numpy(acts),
            'log_probs': torch.tensor(self.log_probs, dtype=torch.float32),
            'advantages': self.advantages,
            'returns': self.returns,
        }

    def __len__(self):
        return len(self.rewards)


def split_obs_tensor(obs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Split (B, 32) observation into organ-specific tensors."""
    c_dim = len(CARDIAC_FEATURE_NAMES)   # 12
    r_dim = len(RENAL_FEATURE_NAMES)     # 10
    m_dim = len(META_FEATURE_NAMES)      # 5
    t_dim = len(TEMPORAL_FEATURE_NAMES)  # 5

    cardiac = obs[:, :c_dim]
    renal = obs[:, c_dim:c_dim + r_dim]
    meta = obs[:, c_dim + r_dim:c_dim + r_dim + m_dim]
    temporal = obs[:, c_dim + r_dim + m_dim:]
    return cardiac, renal, meta, temporal


class PPOTrainer:
    """Proximal Policy Optimization for the attention coupling policy.

    Parameters
    ----------
    policy : AttentionCouplingPolicy
        The neural network policy to train.
    env : CardiorenalCouplingEnv
        The Gymnasium environment.
    config : dict or None
        Override RL_CONFIG hyperparameters.
    device : str
        PyTorch device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        policy: AttentionCouplingPolicy,
        env: CardiorenalCouplingEnv,
        config: Optional[Dict] = None,
        device: str = 'cpu',
    ):
        self.policy = policy.to(device)
        self.env = env
        self.config = {**RL_CONFIG, **(config or {})}
        self.device = device

        self.optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=self.config['lr'],
        )

        # Training statistics
        self.total_timesteps = 0
        self.iteration = 0
        self.best_reward = -float('inf')

    def collect_rollout(self, n_steps: int) -> RolloutBuffer:
        """Collect transitions by running the policy in the environment.

        Parameters
        ----------
        n_steps : int
            Number of environment steps to collect.

        Returns
        -------
        buffer : RolloutBuffer
        """
        buffer = RolloutBuffer()
        obs, info = self.env.reset()
        episode_rewards = []
        current_episode_reward = 0.0

        self.policy.eval()
        for _ in range(n_steps):
            # Get action from policy
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            cardiac, renal, meta, temporal = split_obs_tensor(obs_tensor)

            with torch.no_grad():
                a_mean, a_std, r_mean, r_std, value, _ = self.policy(
                    cardiac, renal, meta, temporal,
                )

            # Sample action
            from torch.distributions import Normal
            alpha_dist = Normal(a_mean, a_std)
            res_dist = Normal(r_mean, r_std)
            alpha_sample = alpha_dist.sample()
            res_sample = res_dist.sample()

            log_prob = (alpha_dist.log_prob(alpha_sample).sum() +
                       res_dist.log_prob(res_sample).sum()).item()

            # Combine and rescale to [-1, 1] for env
            alpha_np = alpha_sample.squeeze(0).numpy()
            res_np = res_sample.squeeze(0).numpy()

            # Map from bounded [alpha_min, alpha_max] back to [-1, 1]
            a_min, a_max = self.config['alpha_min'], self.config['alpha_max']
            r_min, r_max = self.config['residual_min'], self.config['residual_max']
            alpha_raw = 2.0 * (alpha_np - a_min) / (a_max - a_min) - 1.0
            res_raw = 2.0 * (res_np - r_min) / (r_max - r_min) - 1.0
            action = np.concatenate([
                np.clip(alpha_raw, -1, 1),
                np.clip(res_raw, -1, 1),
            ])

            # Step environment
            next_obs, reward, terminated, truncated, step_info = self.env.step(action)
            done = terminated or truncated

            buffer.add(
                obs=obs,
                action=np.concatenate([alpha_np, res_np]),
                log_prob=log_prob,
                reward=reward,
                value=value.item(),
                done=done,
            )

            current_episode_reward += reward
            self.total_timesteps += 1

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                obs, info = self.env.reset()
            else:
                obs = next_obs

        # Compute last value for GAE
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        cardiac, renal, meta, temporal = split_obs_tensor(obs_tensor)
        with torch.no_grad():
            _, _, _, _, last_value, _ = self.policy(cardiac, renal, meta, temporal)
        last_value = last_value.item()

        buffer.compute_returns_and_advantages(
            last_value,
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
        )

        return buffer

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """PPO clipped objective update.

        Parameters
        ----------
        buffer : RolloutBuffer
            Collected transitions with computed advantages.

        Returns
        -------
        metrics : dict — training loss components
        """
        self.policy.train()
        data = buffer.get_tensors()

        obs = data['observations'].to(self.device)
        actions = data['actions'].to(self.device)
        old_log_probs = data['log_probs'].to(self.device)
        advantages = data['advantages'].to(self.device)
        returns = data['returns'].to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(buffer)
        batch_size = min(self.config['batch_size'], n)
        n_epochs = self.config['n_epochs_per_update']

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(n_epochs):
            indices = torch.randperm(n)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = indices[start:end]

                b_obs = obs[idx]
                b_actions = actions[idx]
                b_old_lp = old_log_probs[idx]
                b_adv = advantages[idx]
                b_ret = returns[idx]

                # Split observations
                cardiac, renal, meta, temporal = split_obs_tensor(b_obs)

                # Evaluate actions under current policy
                log_probs, entropy, values = self.policy.evaluate_actions(
                    cardiac, renal, meta, temporal, b_actions,
                )

                # PPO clipped surrogate
                ratio = torch.exp(log_probs - b_old_lp)
                clip_ratio = self.config['clip_ratio']
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values.squeeze(-1), b_ret)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = (policy_loss +
                       self.config['value_loss_coeff'] * value_loss +
                       self.config['entropy_coeff'] * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config['max_grad_norm'],
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
        }

    def train(
        self,
        n_iterations: int = 500,
        n_steps_per_rollout: int = 256,
        save_path: Optional[str] = None,
        log_interval: int = 10,
    ) -> List[Dict]:
        """Main training loop.

        Parameters
        ----------
        n_iterations : int
            Number of collect-update cycles.
        n_steps_per_rollout : int
            Steps to collect per rollout.
        save_path : str or None
            Path to save best checkpoint.
        log_interval : int
            Print progress every N iterations.

        Returns
        -------
        history : list of dicts with per-iteration metrics
        """
        history = []
        print(f"Starting PPO training: {n_iterations} iterations, "
              f"{n_steps_per_rollout} steps/rollout")
        start_time = time.time()

        for iteration in range(1, n_iterations + 1):
            self.iteration = iteration

            # Collect rollout
            buffer = self.collect_rollout(n_steps_per_rollout)
            mean_reward = np.mean(buffer.rewards)

            # PPO update
            metrics = self.update(buffer)
            metrics['mean_reward'] = mean_reward
            metrics['total_timesteps'] = self.total_timesteps
            history.append(metrics)

            # Save best
            if mean_reward > self.best_reward and save_path:
                self.best_reward = mean_reward
                self.save(save_path)

            # Log progress
            if iteration % log_interval == 0:
                elapsed = time.time() - start_time
                print(f"  Iter {iteration}/{n_iterations} | "
                      f"reward={mean_reward:.4f} | "
                      f"policy_loss={metrics['policy_loss']:.4f} | "
                      f"value_loss={metrics['value_loss']:.4f} | "
                      f"entropy={metrics['entropy']:.4f} | "
                      f"steps={self.total_timesteps} | "
                      f"time={elapsed:.0f}s")

        # Final save
        if save_path:
            self.save(save_path)
            print(f"Final model saved to {save_path}")

        return history

    def save(self, path: str):
        """Save policy checkpoint with full metadata."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.policy.get_config(),
            'iteration': self.iteration,
            'total_timesteps': self.total_timesteps,
            'best_reward': self.best_reward,
        }, path)

    def load(self, path: str):
        """Load policy checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.iteration = ckpt.get('iteration', 0)
        self.total_timesteps = ckpt.get('total_timesteps', 0)
        self.best_reward = ckpt.get('best_reward', -float('inf'))
        print(f"Loaded checkpoint from {path} (iter={self.iteration})")


def main():
    parser = argparse.ArgumentParser(description='Train RL coupling policy')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                       help='Training stage (1=synthetic, 2=fine-tune)')
    parser.add_argument('--n_iter', type=int, default=500,
                       help='Number of training iterations')
    parser.add_argument('--n_steps', type=int, default=256,
                       help='Steps per rollout')
    parser.add_argument('--n_months', type=int, default=None,
                       help='Fixed episode length (None=random)')
    parser.add_argument('--load', type=str, default=None,
                       help='Load checkpoint to resume from')
    parser.add_argument('--save', type=str, default=None,
                       help='Save path for best checkpoint')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    # Default save paths
    if args.save is None:
        if args.stage == 1:
            args.save = 'models/rl_stage1.pt'
        else:
            args.save = 'models/rl_attention_policy.pt'

    # Configure environment
    env_config = dict(RL_CONFIG)
    if args.stage == 2:
        # Reduce shaped reward in Stage 2
        env_config['physiology_penalty_coeff'] *= env_config['shaped_reward_scale_stage2']
        env_config['coupling_reg_coeff'] *= env_config['shaped_reward_scale_stage2']

    env = CardiorenalCouplingEnv(config=env_config, n_months=args.n_months)

    # Create policy
    policy = AttentionCouplingPolicy(
        cardiac_dim=RL_CONFIG['cardiac_dim'],
        renal_dim=RL_CONFIG['renal_dim'],
        meta_dim=RL_CONFIG['meta_dim'],
        temporal_dim=RL_CONFIG['temporal_dim'],
        embed_dim=RL_CONFIG['embed_dim'],
        n_heads=RL_CONFIG['n_heads'],
        n_cross_layers=RL_CONFIG['n_cross_layers'],
        dropout=RL_CONFIG['dropout'],
        alpha_min=RL_CONFIG['alpha_min'],
        alpha_max=RL_CONFIG['alpha_max'],
        residual_min=RL_CONFIG['residual_min'],
        residual_max=RL_CONFIG['residual_max'],
    )

    trainer = PPOTrainer(policy, env, config=env_config, device=args.device)

    # Load checkpoint if resuming
    if args.load:
        trainer.load(args.load)

    print(f"\n{'='*60}")
    print(f"  RL Coupling Discovery — Stage {args.stage}")
    print(f"  Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  Episodes: {args.n_iter} iterations × {args.n_steps} steps")
    print(f"  Save: {args.save}")
    print(f"{'='*60}\n")

    history = trainer.train(
        n_iterations=args.n_iter,
        n_steps_per_rollout=args.n_steps,
        save_path=args.save,
    )

    print(f"\nTraining complete. Best reward: {trainer.best_reward:.4f}")


if __name__ == '__main__':
    main()
