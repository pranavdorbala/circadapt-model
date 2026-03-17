"""End-to-end integration tests for the RL coupling system."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import torch
from config import RL_CONFIG


class TestRLSimulationIntegration:

    def test_rl_simulation_runs_with_identity_coupling(self):
        """run_coupled_simulation_rl with alpha_fn=None should run without errors."""
        from cardiorenal_coupling import run_coupled_simulation_rl

        hist = run_coupled_simulation_rl(
            n_steps=3,
            dt_renal_hours=180.0,
            renal_substeps=2,
            cardiac_schedule=[1.0, 0.95, 0.9],
            kidney_schedule=[1.0, 0.9, 0.8],
            stiffness_schedule=[1.0, 1.1, 1.2],
            alpha_fn=None,  # Identity coupling
            verbose=False,
        )

        assert 'MAP' in hist
        assert 'GFR' in hist
        assert 'observations' in hist
        assert 'actions_alpha' in hist
        assert 'actions_residual' in hist
        assert len(hist['MAP']) == 3
        assert len(hist['observations']) == 3

    def test_rl_simulation_with_random_policy(self):
        """run_coupled_simulation_rl with a random policy should not crash."""
        from cardiorenal_coupling import run_coupled_simulation_rl
        from models.attention_coupling import AttentionCouplingPolicy

        policy = AttentionCouplingPolicy(embed_dim=32, n_heads=2, n_cross_layers=1)
        policy.eval()

        def alpha_fn(obs_dict, step):
            action, _, _ = policy.get_action(obs_dict, deterministic=True)
            return action[:5], action[5:]

        hist = run_coupled_simulation_rl(
            n_steps=3,
            alpha_fn=alpha_fn,
            baselines=RL_CONFIG['baselines'],
            verbose=False,
        )

        assert len(hist['MAP']) == 3
        # Check that alphas are not all ones (policy should produce varied output)
        alphas = np.array(hist['actions_alpha'])
        # At least check they are valid numbers
        assert np.all(np.isfinite(alphas))

    def test_observation_vector_consistency(self):
        """Observation dict and vector should be consistent."""
        from cardiorenal_coupling import run_coupled_simulation_rl, obs_dict_to_vector
        from config import CARDIAC_FEATURE_NAMES, RENAL_FEATURE_NAMES, META_FEATURE_NAMES, TEMPORAL_FEATURE_NAMES

        # Use shorter dt for numerical stability in default-param test
        hist = run_coupled_simulation_rl(
            n_steps=2, dt_renal_hours=6.0, renal_substeps=1, verbose=False,
        )

        # Use the first observation (always valid from initial step)
        obs_dict = hist['observations'][0]
        obs_vec = obs_dict_to_vector(obs_dict)

        all_names = CARDIAC_FEATURE_NAMES + RENAL_FEATURE_NAMES + META_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES
        assert obs_vec.shape == (len(all_names),)
        assert obs_vec.dtype == np.float32

        # Check that vector values match dict values (skip NaN)
        for i, name in enumerate(all_names):
            if np.isnan(obs_vec[i]) and np.isnan(obs_dict[name]):
                continue  # Both NaN is consistent
            assert obs_vec[i] == pytest.approx(obs_dict[name], rel=1e-5), \
                f"Mismatch for {name}: vec={obs_vec[i]}, dict={obs_dict[name]}"


class TestAttentionInterpretability:

    def test_cross_attention_weight_shapes(self):
        """Cross-attention weights should have interpretable shapes."""
        from models.attention_coupling import AttentionCouplingPolicy

        policy = AttentionCouplingPolicy(
            cardiac_dim=12, renal_dim=10, embed_dim=32, n_heads=2, n_cross_layers=2,
        )
        policy.eval()

        cardiac = torch.randn(1, 12)
        renal = torch.randn(1, 10)
        meta = torch.randn(1, 5)
        temporal = torch.randn(1, 5)

        with torch.no_grad():
            _, _, _, _, _, attn = policy(cardiac, renal, meta, temporal)

        # h_cross_r: heart attends to kidney
        # After appending context token: cardiac has 13 tokens, renal has 11
        h_cross_wts = attn['h_cross_r']
        assert len(h_cross_wts) == 2  # 2 layers
        # Each should be (1, n_cardiac+1, n_renal+1)
        assert h_cross_wts[0].shape[0] == 1
        assert h_cross_wts[0].shape[1] == 13  # 12 cardiac + 1 context
        assert h_cross_wts[0].shape[2] == 11  # 10 renal + 1 context

    def test_attention_maps_accessible_after_forward(self):
        """get_attention_maps should return stored weights."""
        from models.attention_coupling import AttentionCouplingPolicy

        policy = AttentionCouplingPolicy(embed_dim=32, n_heads=2, n_cross_layers=1)
        policy.eval()

        with torch.no_grad():
            policy(torch.randn(1, 12), torch.randn(1, 10),
                  torch.randn(1, 5), torch.randn(1, 5))

        maps = policy.get_attention_maps()
        assert 'h_cross_r' in maps
        assert len(maps['h_cross_r']) > 0


class TestEnvPolicyIntegration:

    def test_env_with_policy(self):
        """The env should work when stepping with policy-generated actions."""
        from rl_env import CardiorenalCouplingEnv
        from models.attention_coupling import AttentionCouplingPolicy

        env = CardiorenalCouplingEnv(n_months=3)
        policy = AttentionCouplingPolicy(embed_dim=32, n_heads=2, n_cross_layers=1)
        policy.eval()

        obs, info = env.reset(seed=42)
        total_reward = 0.0

        for _ in range(3):
            # Use policy to generate action
            obs_dict = {}
            from config import CARDIAC_FEATURE_NAMES, RENAL_FEATURE_NAMES, META_FEATURE_NAMES, TEMPORAL_FEATURE_NAMES
            all_names = CARDIAC_FEATURE_NAMES + RENAL_FEATURE_NAMES + META_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES
            for i, name in enumerate(all_names):
                obs_dict[name] = float(obs[i])

            action, log_prob, value = policy.get_action(obs_dict, deterministic=True)
            # Convert to [-1, 1] range for env
            a_min, a_max = RL_CONFIG['alpha_min'], RL_CONFIG['alpha_max']
            r_min, r_max = RL_CONFIG['residual_min'], RL_CONFIG['residual_max']
            alpha_raw = 2.0 * (action[:5] - a_min) / (a_max - a_min) - 1.0
            res_raw = 2.0 * (action[5:] - r_min) / (r_max - r_min) - 1.0
            env_action = np.concatenate([np.clip(alpha_raw, -1, 1),
                                         np.clip(res_raw, -1, 1)])

            obs, reward, terminated, truncated, step_info = env.step(env_action)
            total_reward += reward
            if terminated:
                break

        assert np.isfinite(total_reward)
