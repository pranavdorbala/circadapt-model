"""Integration tests for the CardiorenalCouplingEnv Gymnasium environment."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from config import RL_CONFIG


@pytest.fixture
def env():
    """Create a short-episode environment for fast testing."""
    from rl_env import CardiorenalCouplingEnv
    return CardiorenalCouplingEnv(n_months=4)


class TestEnvReset:

    def test_returns_valid_obs(self, env):
        """reset() should return (obs, info) with correct obs shape."""
        obs, info = env.reset(seed=42)
        assert obs.shape == (RL_CONFIG['obs_dim'],)
        assert obs.dtype == np.float32
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))

    def test_info_contains_metadata(self, env):
        obs, info = env.reset(seed=42)
        assert 'n_months' in info
        assert 'initial_MAP' in info
        assert 'initial_GFR' in info

    def test_reproducibility_with_seed(self, env):
        """Same seed should produce identical initial observation."""
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

    def test_multiple_resets(self, env):
        """Multiple resets should all return valid observations."""
        for seed in [1, 2, 3]:
            obs, info = env.reset(seed=seed)
            assert obs.shape == (RL_CONFIG['obs_dim'],)
            assert not np.any(np.isnan(obs))


class TestEnvStep:

    def test_step_with_neutral_action(self, env):
        """Neutral action (zeros → alpha=1.0, residuals=0) should not crash."""
        env.reset(seed=42)
        # action=0 maps to midpoint of each bound
        action = np.zeros(RL_CONFIG['action_dim'], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (RL_CONFIG['obs_dim'],)
        assert not np.any(np.isnan(obs))
        assert isinstance(reward, float)
        assert not np.isnan(reward)

    def test_step_with_random_action(self, env):
        """Random actions should produce valid outputs."""
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (RL_CONFIG['obs_dim'],)
        assert isinstance(reward, float)

    def test_episode_terminates(self, env):
        """Episode should terminate after n_months steps."""
        env.reset(seed=42)
        n_months = env._n_months
        terminated = False
        for i in range(n_months + 5):  # extra steps to be safe
            action = np.zeros(RL_CONFIG['action_dim'], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        assert terminated, f"Episode did not terminate after {n_months} steps"

    def test_reward_finite(self, env):
        """All rewards should be finite."""
        env.reset(seed=42)
        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward), f"Non-finite reward: {reward}"
            if terminated:
                break

    def test_obs_in_expected_ranges(self, env):
        """Key observation features should be in physiologically plausible ranges."""
        from config import CARDIAC_FEATURE_NAMES
        obs, info = env.reset(seed=42)

        # MAP is the first cardiac feature
        MAP_idx = CARDIAC_FEATURE_NAMES.index('MAP')
        MAP = obs[MAP_idx]
        assert 40 < MAP < 200, f"MAP={MAP} out of plausible range"

        # EF index
        EF_idx = CARDIAC_FEATURE_NAMES.index('EF')
        EF = obs[EF_idx]
        assert 10 < EF < 85, f"EF={EF} out of plausible range"


class TestEnvSpaces:

    def test_observation_space(self, env):
        assert env.observation_space.shape == (RL_CONFIG['obs_dim'],)

    def test_action_space(self, env):
        assert env.action_space.shape == (RL_CONFIG['action_dim'],)
        assert env.action_space.low.min() == -1.0
        assert env.action_space.high.max() == 1.0

    def test_action_sample_in_bounds(self, env):
        for _ in range(10):
            action = env.action_space.sample()
            assert np.all(action >= -1.0)
            assert np.all(action <= 1.0)
