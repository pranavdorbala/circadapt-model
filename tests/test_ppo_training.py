"""Smoke tests for the PPO training loop."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import tempfile
import numpy as np
import torch
from config import RL_CONFIG


@pytest.fixture
def trainer():
    """Create a trainer with a short-episode env for fast testing."""
    from models.attention_coupling import AttentionCouplingPolicy
    from rl_env import CardiorenalCouplingEnv
    from train_rl import PPOTrainer

    policy = AttentionCouplingPolicy(
        cardiac_dim=RL_CONFIG['cardiac_dim'],
        renal_dim=RL_CONFIG['renal_dim'],
        meta_dim=RL_CONFIG['meta_dim'],
        temporal_dim=RL_CONFIG['temporal_dim'],
        embed_dim=32,  # Smaller for fast testing
        n_heads=2,
        n_cross_layers=1,
    )
    env = CardiorenalCouplingEnv(n_months=4)
    return PPOTrainer(policy, env, config={**RL_CONFIG, 'n_epochs_per_update': 2})


class TestRolloutBuffer:

    def test_buffer_add_and_len(self):
        from train_rl import RolloutBuffer
        buf = RolloutBuffer()
        for i in range(10):
            buf.add(
                obs=np.zeros(32), action=np.zeros(15),
                log_prob=-1.0, reward=0.1, value=0.5, done=(i == 9),
            )
        assert len(buf) == 10

    def test_gae_computation(self):
        """GAE should produce finite advantages and returns."""
        from train_rl import RolloutBuffer
        buf = RolloutBuffer()
        for i in range(8):
            buf.add(
                obs=np.random.randn(32).astype(np.float32),
                action=np.random.randn(15).astype(np.float32),
                log_prob=-0.5,
                reward=0.1 * (i + 1),
                value=0.5,
                done=(i == 7),
            )
        buf.compute_returns_and_advantages(last_value=0.0, gamma=0.99, gae_lambda=0.95)
        assert buf.advantages is not None
        assert buf.returns is not None
        assert torch.isfinite(buf.advantages).all()
        assert torch.isfinite(buf.returns).all()
        assert len(buf.advantages) == 8

    def test_get_tensors(self):
        from train_rl import RolloutBuffer
        buf = RolloutBuffer()
        for i in range(5):
            buf.add(
                obs=np.random.randn(32).astype(np.float32),
                action=np.random.randn(15).astype(np.float32),
                log_prob=-0.5, reward=0.1, value=0.5, done=False,
            )
        buf.compute_returns_and_advantages(0.0)
        tensors = buf.get_tensors()
        assert tensors['observations'].shape == (5, 32)
        assert tensors['actions'].shape == (5, 15)
        assert tensors['log_probs'].shape == (5,)


class TestPPOTrainer:

    def test_collect_rollout(self, trainer):
        """collect_rollout should return a buffer with correct sizes."""
        buffer = trainer.collect_rollout(n_steps=8)
        assert len(buffer) == 8
        assert buffer.advantages is not None

    def test_one_update_cycle(self, trainer):
        """A single update should not produce NaN losses."""
        buffer = trainer.collect_rollout(n_steps=16)
        metrics = trainer.update(buffer)
        assert np.isfinite(metrics['policy_loss'])
        assert np.isfinite(metrics['value_loss'])
        assert np.isfinite(metrics['entropy'])

    def test_checkpoint_save_load(self, trainer):
        """Save and load should preserve policy state."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            tmppath = f.name

        try:
            trainer.iteration = 42
            trainer.save(tmppath)

            # Create new trainer and load
            from models.attention_coupling import AttentionCouplingPolicy
            from rl_env import CardiorenalCouplingEnv
            from train_rl import PPOTrainer

            policy2 = AttentionCouplingPolicy(
                embed_dim=32, n_heads=2, n_cross_layers=1,
            )
            env2 = CardiorenalCouplingEnv(n_months=4)
            trainer2 = PPOTrainer(policy2, env2)
            trainer2.load(tmppath)

            assert trainer2.iteration == 42
        finally:
            os.unlink(tmppath)
