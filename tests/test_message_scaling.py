"""Unit tests for message scaling and inflammatory residual functions."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from cardiorenal_coupling import (
    HeartToKidneyMessage, KidneyToHeartMessage, InflammatoryState,
    scale_message_h2k, scale_message_k2h, apply_inflammatory_residuals,
)

BASELINES = {'MAP': 93.0, 'CO': 5.0, 'Pven': 3.0, 'V_blood': 5000.0, 'SVR_ratio': 1.0}


# ─── HeartToKidney scaling ───────────────────────────────────────────────

class TestScaleMessageH2K:

    def _make_h2k(self, MAP=100.0, CO=4.5, Pven=5.0):
        return HeartToKidneyMessage(MAP=MAP, CO=CO, Pven=Pven, SBP=130.0, DBP=80.0)

    def test_alpha_identity(self):
        """alpha=1.0 should pass the message through unchanged."""
        msg = self._make_h2k()
        scaled = scale_message_h2k(msg, np.array([1.0, 1.0, 1.0]), BASELINES)
        assert scaled.MAP == pytest.approx(msg.MAP)
        assert scaled.CO == pytest.approx(msg.CO)
        assert scaled.Pven == pytest.approx(msg.Pven)

    def test_alpha_zero_returns_baseline(self):
        """alpha=0.0 should return the baseline values."""
        msg = self._make_h2k(MAP=120.0, CO=3.0, Pven=8.0)
        scaled = scale_message_h2k(msg, np.array([0.0, 0.0, 0.0]), BASELINES)
        assert scaled.MAP == pytest.approx(BASELINES['MAP'])
        assert scaled.CO == pytest.approx(BASELINES['CO'])
        assert scaled.Pven == pytest.approx(BASELINES['Pven'])

    def test_alpha_amplifies(self):
        """alpha=2.0 should double the deviation from baseline."""
        msg = self._make_h2k(MAP=103.0)  # deviation = 10
        scaled = scale_message_h2k(msg, np.array([2.0, 1.0, 1.0]), BASELINES)
        expected_MAP = 93.0 + 2.0 * (103.0 - 93.0)  # 93 + 20 = 113
        assert scaled.MAP == pytest.approx(expected_MAP)

    def test_alpha_dampens(self):
        """alpha=0.5 should halve the deviation from baseline."""
        msg = self._make_h2k(MAP=113.0)  # deviation = 20
        scaled = scale_message_h2k(msg, np.array([0.5, 1.0, 1.0]), BASELINES)
        expected_MAP = 93.0 + 0.5 * (113.0 - 93.0)  # 93 + 10 = 103
        assert scaled.MAP == pytest.approx(expected_MAP)

    def test_sbp_dbp_unchanged(self):
        """SBP and DBP should pass through regardless of alpha."""
        msg = self._make_h2k()
        scaled = scale_message_h2k(msg, np.array([0.0, 0.0, 0.0]), BASELINES)
        assert scaled.SBP == msg.SBP
        assert scaled.DBP == msg.DBP

    def test_per_channel_independence(self):
        """Each channel's alpha should affect only its own message."""
        msg = self._make_h2k(MAP=110.0, CO=6.0, Pven=5.0)
        scaled = scale_message_h2k(msg, np.array([0.0, 1.0, 1.0]), BASELINES)
        assert scaled.MAP == pytest.approx(BASELINES['MAP'])
        assert scaled.CO == pytest.approx(msg.CO)
        assert scaled.Pven == pytest.approx(msg.Pven)


# ─── KidneyToHeart scaling ──────────────────────────────────────────────

class TestScaleMessageK2H:

    def _make_k2h(self, V_blood=5500.0, SVR_ratio=1.2):
        return KidneyToHeartMessage(V_blood=V_blood, SVR_ratio=SVR_ratio, GFR=90.0)

    def test_alpha_identity(self):
        msg = self._make_k2h()
        scaled = scale_message_k2h(msg, np.array([1.0, 1.0]), BASELINES)
        assert scaled.V_blood == pytest.approx(msg.V_blood)
        assert scaled.SVR_ratio == pytest.approx(msg.SVR_ratio)

    def test_alpha_zero_returns_baseline(self):
        msg = self._make_k2h(V_blood=6000.0, SVR_ratio=1.5)
        scaled = scale_message_k2h(msg, np.array([0.0, 0.0]), BASELINES)
        assert scaled.V_blood == pytest.approx(BASELINES['V_blood'])
        assert scaled.SVR_ratio == pytest.approx(BASELINES['SVR_ratio'])

    def test_gfr_unchanged(self):
        """GFR is informational and should pass through."""
        msg = self._make_k2h()
        scaled = scale_message_k2h(msg, np.array([0.0, 0.0]), BASELINES)
        assert scaled.GFR == msg.GFR


# ─── Inflammatory residuals ─────────────────────────────────────────────

class TestInflammatoryResiduals:

    def test_zero_residuals_identity(self):
        """Zero residuals should leave the state unchanged."""
        ist = InflammatoryState()
        ist.Sf_act_factor = 0.8
        ist.Kf_factor = 0.7
        result = apply_inflammatory_residuals(ist, np.zeros(10))
        assert result.Sf_act_factor == pytest.approx(ist.Sf_act_factor)
        assert result.Kf_factor == pytest.approx(ist.Kf_factor)

    def test_copy_semantics(self):
        """Input InflammatoryState should NOT be mutated."""
        ist = InflammatoryState()
        original_sf = ist.Sf_act_factor
        residuals = np.array([0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        result = apply_inflammatory_residuals(ist, residuals)
        assert ist.Sf_act_factor == pytest.approx(original_sf)
        assert result.Sf_act_factor != original_sf

    def test_positive_residual_increases(self):
        ist = InflammatoryState()
        ist.Sf_act_factor = 0.8
        residuals = np.zeros(10)
        residuals[0] = 0.1  # dSf_act_factor
        result = apply_inflammatory_residuals(ist, residuals)
        assert result.Sf_act_factor == pytest.approx(0.9)

    def test_clamping_sf_act_factor(self):
        """Sf_act_factor should be clamped at ≥ 0.3."""
        ist = InflammatoryState()
        ist.Sf_act_factor = 0.4
        residuals = np.zeros(10)
        residuals[0] = -0.3  # would be 0.1, but clamped to 0.3
        result = apply_inflammatory_residuals(ist, residuals)
        assert result.Sf_act_factor >= 0.3

    def test_clamping_kf_factor(self):
        """Kf_factor should be clamped to [0.05, 2.0]."""
        ist = InflammatoryState()
        ist.Kf_factor = 0.1
        # Large negative residual
        residuals = np.zeros(10)
        residuals[4] = -0.3
        result = apply_inflammatory_residuals(ist, residuals)
        assert result.Kf_factor >= 0.05

        # Large positive residual
        ist.Kf_factor = 1.9
        residuals[4] = 0.3
        result = apply_inflammatory_residuals(ist, residuals)
        assert result.Kf_factor <= 2.0

    def test_clamping_eta_pt_offset(self):
        """eta_PT_offset should be clamped to [-0.1, 0.2]."""
        ist = InflammatoryState()
        ist.eta_PT_offset = 0.0
        residuals = np.zeros(10)
        residuals[8] = -0.3
        result = apply_inflammatory_residuals(ist, residuals)
        assert result.eta_PT_offset >= -0.1

    def test_clamping_map_setpoint(self):
        """MAP_setpoint_offset should be clamped to [-10, 20]."""
        ist = InflammatoryState()
        ist.MAP_setpoint_offset = 15.0
        residuals = np.zeros(10)
        residuals[9] = 10.0
        result = apply_inflammatory_residuals(ist, residuals)
        assert result.MAP_setpoint_offset <= 20.0
