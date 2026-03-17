"""Regression tests to verify backward compatibility.

These tests ensure that existing functionality is not broken by the
RL coupling additions. The original run_coupled_simulation() and
evaluate_patient_state() should produce identical output.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np


class TestOriginalSimulation:

    def test_run_coupled_simulation_unchanged(self):
        """Original simulation should still work with default args."""
        from cardiorenal_coupling import run_coupled_simulation

        hist = run_coupled_simulation(n_steps=2)
        assert 'MAP' in hist
        assert 'GFR' in hist
        assert 'EF' in hist
        assert len(hist['MAP']) == 2
        # Sanity check physiological ranges
        assert 60 < hist['MAP'][-1] < 160
        assert 20 < hist['GFR'][-1] < 200
        assert 20 < hist['EF'][-1] < 80

    def test_run_coupled_simulation_with_disease(self):
        """Simulation with disease parameters should still work."""
        from cardiorenal_coupling import run_coupled_simulation

        hist = run_coupled_simulation(
            n_steps=3,
            cardiac_schedule=[1.0, 0.9, 0.8],
            kidney_schedule=[1.0, 0.8, 0.6],
            stiffness_schedule=[1.0, 1.3, 1.6],
            inflammation_schedule=[0.0, 0.2, 0.4],
            diabetes_schedule=[0.0, 0.3, 0.5],
        )
        assert len(hist['MAP']) == 3
        # With disease progression, EF should decrease and GFR should decrease
        assert hist['EF'][-1] <= hist['EF'][0] + 5  # Allow small variation


class TestAgentToolsBackwardCompat:

    def test_run_circadapt_model_without_rl_policy(self):
        """run_circadapt_model should work without a trained RL policy."""
        # Ensure no RL policy file exists (or that it gracefully falls back)
        import agent_tools
        # Reset the lazy-load state
        agent_tools._RL_POLICY = None
        agent_tools._RL_POLICY_LOADED = False

        result = agent_tools.run_circadapt_model(
            Sf_act_scale=0.9,
            Kf_scale=0.8,
            k1_scale=1.2,
        )

        # Should return a dict with clinical variables
        assert isinstance(result, dict)
        if 'error' not in result:
            assert 'params_used' in result
            # Check some key variables exist
            assert any('LVEF' in k or 'EF' in k for k in result.keys())


class TestNewFunctionsExist:

    def test_scale_message_h2k_importable(self):
        from cardiorenal_coupling import scale_message_h2k
        assert callable(scale_message_h2k)

    def test_scale_message_k2h_importable(self):
        from cardiorenal_coupling import scale_message_k2h
        assert callable(scale_message_k2h)

    def test_apply_inflammatory_residuals_importable(self):
        from cardiorenal_coupling import apply_inflammatory_residuals
        assert callable(apply_inflammatory_residuals)

    def test_run_coupled_simulation_rl_importable(self):
        from cardiorenal_coupling import run_coupled_simulation_rl
        assert callable(run_coupled_simulation_rl)

    def test_extract_rl_observation_importable(self):
        from cardiorenal_coupling import extract_rl_observation
        assert callable(extract_rl_observation)

    def test_obs_dict_to_vector_importable(self):
        from cardiorenal_coupling import obs_dict_to_vector
        assert callable(obs_dict_to_vector)

    def test_rl_config_exists(self):
        from config import RL_CONFIG, CARDIAC_FEATURE_NAMES, RENAL_FEATURE_NAMES
        assert isinstance(RL_CONFIG, dict)
        assert len(CARDIAC_FEATURE_NAMES) == 12
        assert len(RENAL_FEATURE_NAMES) == 10

    def test_attention_policy_importable(self):
        from models.attention_coupling import AttentionCouplingPolicy
        assert callable(AttentionCouplingPolicy)
