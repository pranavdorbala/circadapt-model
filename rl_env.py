"""
Gymnasium Environment for Cardiorenal Coupling Discovery
=========================================================
Wraps the coupled CircAdapt + Hallow simulator as a Gymnasium environment
for RL-based discovery of the inter-organ coupling equation.

Episode: One patient trajectory over 72-120 monthly steps.
Observation: 32-dim (12 cardiac + 10 renal + 5 meta + 5 temporal).
Action: 15-dim continuous (5 coupling alphas + 10 inflammatory residuals).
Reward: Shaped per-step physiological plausibility + terminal V7 MSE.
"""

import gymnasium
import numpy as np
from gymnasium import spaces
from typing import Dict, Optional, Tuple

from config import (
    RL_CONFIG, CORE_VARIABLES, SAMPLING_CONFIG,
    CARDIAC_FEATURE_NAMES, RENAL_FEATURE_NAMES,
    META_FEATURE_NAMES, TEMPORAL_FEATURE_NAMES,
)
from cardiorenal_coupling import (
    CircAdaptHeartModel, HallowRenalModel, InflammatoryState,
    update_inflammatory_state, update_renal_model,
    heart_to_kidney, kidney_to_heart,
    scale_message_h2k, scale_message_k2h,
    apply_inflammatory_residuals, extract_rl_observation,
    obs_dict_to_vector, ML_TO_M3,
)


class CardiorenalCouplingEnv(gymnasium.Env):
    """Gymnasium environment for learning the cardiorenal coupling equation.

    The RL agent observes the state of both organ models at each monthly
    time step and produces per-message coupling weights (alphas) and
    inflammatory modifier corrections (residuals). The goal is to discover
    the coupling equation that makes the joint system match real disease
    trajectories.

    Parameters
    ----------
    config : dict or None
        Override default RL_CONFIG settings.
    n_months : int or None
        Fixed episode length. If None, sampled from [min_months, max_months].
    """

    metadata = {'render_modes': []}

    def __init__(self, config: Optional[Dict] = None, n_months: Optional[int] = None):
        super().__init__()
        self.config = {**RL_CONFIG, **(config or {})}
        self._fixed_n_months = n_months

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.config['obs_dim'],),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.config['action_dim'],),
            dtype=np.float32,
        )

        # Internal state (initialized in reset)
        self._heart = None
        self._renal = None
        self._ist = None
        self._schedules = None
        self._n_months = None
        self._current_step = 0
        self._prev_obs = None
        self._baselines = self.config['baselines']

        # Running observation statistics for normalization
        self._obs_mean = np.zeros(self.config['obs_dim'], dtype=np.float64)
        self._obs_var = np.ones(self.config['obs_dim'], dtype=np.float64)
        self._obs_count = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Sample a new patient and initialize the simulation.

        Parameters
        ----------
        seed : int or None
            Random seed for reproducibility.
        options : dict or None
            Optional overrides: 'n_months', 'schedules', 'params'.

        Returns
        -------
        obs : np.ndarray (32,)
        info : dict
        """
        super().reset(seed=seed)
        rng = self.np_random

        options = options or {}

        # Determine episode length
        if 'n_months' in options:
            self._n_months = options['n_months']
        elif self._fixed_n_months is not None:
            self._n_months = self._fixed_n_months
        else:
            self._n_months = rng.integers(
                self.config['min_months'],
                self.config['max_months'] + 1,
            )

        # Sample or use provided disease parameters
        if 'schedules' in options:
            self._schedules = options['schedules']
        else:
            params = self._sample_patient_params(rng)
            self._schedules = self._generate_schedules(params, rng)

        # Initialize models
        self._heart = CircAdaptHeartModel()
        self._renal = HallowRenalModel()
        self._ist = InflammatoryState()
        self._current_step = 0
        self._prev_obs = None

        # Run initial step with neutral coupling to get first observation
        obs_dict = self._run_step_internal(
            alpha_vec=np.ones(5),
            residuals=np.zeros(10),
        )
        obs_vec = obs_dict_to_vector(obs_dict)
        self._prev_obs = obs_dict
        self._current_step = 1

        info = {
            'n_months': self._n_months,
            'initial_MAP': obs_dict['MAP'],
            'initial_GFR': obs_dict['GFR'],
            'initial_EF': obs_dict['EF'],
        }

        return obs_vec, info

    def step(
        self, action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one monthly coupling step with learned alphas/residuals.

        Parameters
        ----------
        action : np.ndarray (15,)
            Raw action in [-1, 1], rescaled internally to valid bounds.

        Returns
        -------
        obs : np.ndarray (32,)
        reward : float
        terminated : bool — True at end of episode
        truncated : bool — always False (no truncation)
        info : dict
        """
        # Rescale action from [-1, 1] to valid bounds
        alpha_vec, residuals = self._rescale_action(action)

        # Run one monthly coupling step (with NaN fallback)
        try:
            obs_dict = self._run_step_internal(alpha_vec, residuals)
            obs_vec = obs_dict_to_vector(obs_dict)
            # Check for NaN — CircAdapt solver can fail at extreme params
            if np.any(np.isnan(obs_vec)):
                obs_dict = self._prev_obs if self._prev_obs is not None else obs_dict
                obs_vec = obs_dict_to_vector(obs_dict)
                # Replace any remaining NaN with zeros
                obs_vec = np.nan_to_num(obs_vec, nan=0.0)
                solver_failed = True
            else:
                solver_failed = False
        except Exception:
            # On any solver crash, use previous observation
            obs_dict = self._prev_obs if self._prev_obs is not None else {}
            obs_vec = obs_dict_to_vector(obs_dict) if obs_dict else np.zeros(self.config['obs_dim'], dtype=np.float32)
            obs_vec = np.nan_to_num(obs_vec, nan=0.0)
            solver_failed = True

        # Update step counter
        self._current_step += 1
        terminated = self._current_step >= self._n_months

        # Compute reward (large penalty if solver failed)
        if solver_failed:
            reward = -1.0
        else:
            reward = self._compute_reward(obs_dict, alpha_vec, terminated)

        # Update observation statistics
        self._update_obs_stats(obs_vec)

        info = {
            'step': self._current_step,
            'MAP': obs_dict['MAP'],
            'GFR': obs_dict['GFR'],
            'EF': obs_dict['EF'],
            'alpha': alpha_vec.tolist(),
        }

        self._prev_obs = obs_dict
        return obs_vec, reward, terminated, False, info

    def _rescale_action(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rescale raw [-1, 1] action to coupling alphas and residuals."""
        raw_alpha = action[:5]
        raw_residual = action[5:]

        # Alpha: [-1,1] → [alpha_min, alpha_max]
        alpha_min = self.config['alpha_min']
        alpha_max = self.config['alpha_max']
        alpha_vec = alpha_min + 0.5 * (raw_alpha + 1.0) * (alpha_max - alpha_min)
        alpha_vec = np.clip(alpha_vec, alpha_min, alpha_max)

        # Residuals: [-1,1] → [residual_min, residual_max]
        res_min = self.config['residual_min']
        res_max = self.config['residual_max']
        residuals = res_min + 0.5 * (raw_residual + 1.0) * (res_max - res_min)
        residuals = np.clip(residuals, res_min, res_max)

        return alpha_vec, residuals

    def _run_step_internal(
        self, alpha_vec: np.ndarray, residuals: np.ndarray,
    ) -> Dict[str, float]:
        """Execute one monthly coupling step on the internal simulator state.

        Runs N inner coupling iterations (heart→kidney→feedback→heart) per
        monthly RL step. Each inner iteration uses dt_renal_substep hours
        for the renal Euler integrator. This tight coupling prevents the
        blood-volume instability that occurs with open-loop renal substeps
        at large timesteps.

        The RL policy's alpha scaling and inflammatory residuals are applied
        consistently across all inner iterations within a monthly step.
        """
        s = self._current_step
        schedules = self._schedules

        # Read schedule values for this step
        sf = self._get_schedule_val(schedules['Sf_scale'], s)
        kf = self._get_schedule_val(schedules['Kf_scale'], s)
        k1 = self._get_schedule_val(schedules['k1_scale'], s)
        infl = self._get_schedule_val(schedules['inflammation'], s)
        diab = self._get_schedule_val(schedules['diabetes'], s)

        # Step 0: Update inflammatory state and apply RL residuals
        self._ist = update_inflammatory_state(self._ist, infl, diab)
        ist_corrected = apply_inflammatory_residuals(self._ist, residuals)

        # Apply disease parameters (these stay fixed within the monthly step)
        self._heart.apply_inflammatory_modifiers(ist_corrected)
        effective_k1 = k1 * ist_corrected.passive_k1_factor
        self._heart.apply_stiffness(effective_k1)
        effective_sf = max(sf * ist_corrected.Sf_act_factor, 0.20)
        self._heart.apply_deterioration(effective_sf)
        self._renal.Kf_scale = kf
        effective_kf = kf * ist_corrected.Kf_factor
        raas = self._get_schedule_val(schedules['RAAS_gain'], s)
        tgf = self._get_schedule_val(schedules['TGF_gain'], s)
        self._renal.RAAS_gain = raas
        self._renal.TGF_gain = tgf

        # Inner coupling iterations: closed-loop heart ↔ kidney feedback
        # Each iteration runs the full Algorithm 1 cycle (heart steady-state →
        # H→K message → renal update → K→H message → heart feedback) so that
        # blood volume changes are immediately reflected in cardiac hemodynamics.
        # This matches the original run_coupled_simulation() at dt=6h.
        dt = self.config['dt_renal_substep']
        n_inner = self.config['renal_substeps']
        hemo = None
        for _ in range(n_inner):
            # Heart to steady state
            hemo = self._heart.run_to_steady_state()

            # Heart → Kidney message (scaled by RL alpha)
            h2k_raw = heart_to_kidney(hemo)
            h2k = scale_message_h2k(h2k_raw, alpha_vec[:3], self._baselines)

            # Renal update (single step at dt hours)
            self._renal = update_renal_model(
                self._renal, h2k.MAP, h2k.CO, h2k.Pven,
                dt, inflammatory_state=ist_corrected,
            )

            # Kidney → Heart message (scaled by RL alpha)
            k2h_raw = kidney_to_heart(self._renal, h2k.MAP, h2k.CO, h2k.Pven)
            k2h = scale_message_k2h(k2h_raw, alpha_vec[3:5], self._baselines)

            # Apply kidney feedback to heart for next iteration
            self._heart.apply_kidney_feedback(
                V_blood_m3=k2h.V_blood * ML_TO_M3,
                SVR_ratio=k2h.SVR_ratio,
            )

        # ── Volume homeostasis correction ──────────────────────────────
        # The Hallow model's water balance ODE (Eq. 11) lacks V_blood-
        # dependent water excretion feedback (ADH suppression, ANP release,
        # pressure-diuresis). Over long trajectories (96 monthly steps =
        # 384 coupling iterations), this causes V_blood to drift
        # monotonically upward, eventually crashing CircAdapt.
        #
        # This correction applies first-order relaxation toward a disease-
        # adjusted equilibrium V_blood, representing the integrated effect
        # of renal volume-regulatory mechanisms over the remaining ~29 days
        # of each month not explicitly simulated.
        #
        # V_target accounts for disease-induced volume overload:
        #   - HFpEF (high k1): impaired Frank-Starling → higher V_blood
        #   - Inflammation: fluid retention
        #   - CKD (low Kf): reduced excretory capacity → higher V_blood
        V_target = 5000.0 + 300.0 * max(effective_k1 - 1.0, 0.0) \
                          + 200.0 * infl \
                          + 150.0 * max(0.8 - effective_kf, 0.0)
        V_target = np.clip(V_target, 4500.0, 6500.0)

        # Exponential relaxation: τ = 48h (renal volume regulation timescale).
        # Over total_hours simulated, correct ~(1-exp(-t/τ)) of the difference.
        total_hours = n_inner * dt
        relax_factor = 1.0 - np.exp(-total_hours / 48.0)
        self._renal.V_blood += relax_factor * (V_target - self._renal.V_blood)
        self._renal.V_blood = np.clip(self._renal.V_blood, 3500.0, 7000.0)

        # Extract observation from final iteration's hemodynamics
        t_normalized = (s + 1) / max(self._n_months, 1)
        obs_dict = extract_rl_observation(
            hemo, self._renal, ist_corrected,
            effective_sf, effective_kf, effective_k1,
            infl, diab, t_normalized, self._prev_obs,
        )

        return obs_dict

    def _compute_reward(
        self, obs: Dict[str, float], alpha_vec: np.ndarray, done: bool,
    ) -> float:
        """Compute shaped + terminal reward.

        Per-step shaped reward:
          - Penalize physiological implausibility
          - Penalize coupling deviation from identity (regularization)

        Terminal reward:
          - Negative weighted MSE if V7 targets are available
        """
        reward = 0.0
        coeff_phys = self.config['physiology_penalty_coeff']
        coeff_reg = self.config['coupling_reg_coeff']

        # Physiological plausibility penalties
        MAP = obs['MAP']
        GFR = obs['GFR']
        EF = obs['EF']

        # Severe hypertension / hypotension
        if MAP > 160:
            reward -= coeff_phys * (MAP - 160)
        if MAP < 60:
            reward -= coeff_phys * (60 - MAP)
        # Implausible GFR
        if GFR > 200:
            reward -= coeff_phys * (GFR - 200)
        if GFR < 5:
            reward -= coeff_phys * (5 - GFR)
        # Implausible EF
        if EF > 80:
            reward -= coeff_phys * (EF - 80)
        if EF < 10:
            reward -= coeff_phys * (10 - EF)

        # Coupling regularization: penalize deviation from identity (alpha=1)
        alpha_deviation = np.sum((alpha_vec - 1.0) ** 2)
        reward -= coeff_reg * alpha_deviation

        # Small per-step survival bonus to encourage completing episodes
        reward += 0.001

        return reward

    @staticmethod
    def _truncated_normal(
        rng: np.random.Generator, center: float, sigma: float,
        low: float, high: float,
    ) -> float:
        """Sample a single value from a truncated normal distribution.

        Draws from N(center, sigma^2) clipped to [low, high]. Under this scheme
        ~95% of samples fall within [center-2*sigma, center+2*sigma], making
        extreme values near the bounds exponentially unlikely.
        """
        return float(np.clip(rng.normal(center, sigma), low, high))

    def _sample_patient_params(self, rng: np.random.Generator) -> Dict:
        """Sample a single patient's baseline disease parameters.

        Uses truncated normal distributions (from SAMPLING_CONFIG) centered on
        clinically realistic values. This replaces the old bimodal
        healthy/diseased uniform splits that over-represented extreme parameter
        combinations and caused frequent CircAdapt solver divergence.

        The truncated normal ensures:
          - Most patients cluster around mild-to-moderate disease
          - Extreme values (e.g., k1 > 2.0 AND Kf < 0.4) are jointly rare
          - The full pathological range is still reachable, just with low probability
        """
        SC = SAMPLING_CONFIG
        tn = self._truncated_normal
        params = {}

        # k1_scale: LV passive stiffness [1.0=healthy, >1.5=moderate HFpEF]
        c, s, lo, hi = SC['k1_scale']
        params['k1_scale'] = tn(rng, c, s, lo, hi)

        # Sf_scale: contractility [1.0=healthy, <0.8=impaired]
        c, s, lo, hi = SC['Sf_scale']
        params['Sf_scale'] = tn(rng, c, s, lo, hi)

        # Kf_scale: nephron mass [1.0=healthy, <0.7=CKD]
        c, s, lo, hi = SC['Kf_scale']
        params['Kf_scale'] = tn(rng, c, s, lo, hi)

        # diabetes: metabolic burden [0=none, 1=severe]
        c, s, lo, hi = SC['diabetes']
        params['diabetes'] = tn(rng, c, s, lo, hi)

        # inflammation: systemic inflammatory index
        spec = SC['inflammation']
        if spec[0] == 'exponential':
            _, scale, lo, hi = spec
            params['inflammation'] = float(np.clip(rng.exponential(scale), lo, hi))
        else:
            c, s, lo, hi = spec
            params['inflammation'] = tn(rng, c, s, lo, hi)

        # Coupling and feedback parameters (already normal-distributed)
        c, s, lo, hi = SC['RAAS_gain']
        params['RAAS_gain'] = tn(rng, c, s, lo, hi)

        c, s, lo, hi = SC['TGF_gain']
        params['TGF_gain'] = tn(rng, c, s, lo, hi)

        c, s, lo, hi = SC['na_intake']
        params['na_intake'] = tn(rng, c, s, lo, hi)

        return params

    def _generate_schedules(
        self, params: Dict, rng: np.random.Generator,
    ) -> Dict:
        """Generate per-month disease progression schedules for one patient.

        Mirrors generate_progression_schedule() from synthetic_cohort_monthly.py
        but for a single patient (not vectorized).

        Progression rates are drawn from truncated normal distributions
        (SAMPLING_CONFIG) rather than uniform, so that most patients progress
        at moderate rates and rapid progressors are rare.
        """
        SC = SAMPLING_CONFIG
        tn = self._truncated_normal
        n = self._n_months
        schedules = {}

        # --- k1_scale progression (cardiac stiffness) ---
        c, s, lo, hi = SC['k1_annual_rate']
        k1_annual = tn(rng, c, s, lo, hi) * (1 + 0.5 * params['diabetes'])
        k1_rate = k1_annual / 12
        k1_noise = rng.normal(0, 0.002, n)
        k1_cumul = np.cumsum(k1_rate + k1_noise)
        schedules['k1_scale'] = np.clip(params['k1_scale'] + k1_cumul, 1.0, 3.0)

        # --- Kf_scale progression (nephron loss, nonlinear acceleration) ---
        c, s, lo, hi = SC['Kf_annual_rate']
        kf_annual = tn(rng, c, s, lo, hi) * (1 + 0.4 * params['diabetes'])
        kf_rate = kf_annual / 12
        kf = np.zeros(n)
        kf[0] = params['Kf_scale']
        for t in range(1, n):
            accel = 1 + 0.5 * (1 - kf[t-1])
            kf[t] = np.clip(kf[t-1] - kf_rate * accel + rng.normal(0, 0.001), 0.15, 1.0)
        schedules['Kf_scale'] = kf

        # --- Sf_scale progression (contractility) ---
        c, s, lo, hi = SC['Sf_annual_rate']
        sf_annual = tn(rng, c, s, lo, hi) * (1 + 0.3 * params['diabetes'])
        sf_rate = sf_annual / 12
        sf_noise = rng.normal(0, 0.001, n)
        sf_cumul = np.cumsum(sf_rate + sf_noise)
        schedules['Sf_scale'] = np.clip(params['Sf_scale'] - sf_cumul, 0.4, 1.0)

        # --- Diabetes progression ---
        if params['diabetes'] > 0.1:
            c, s, lo, hi = SC['d_annual_rate']
            d_rate = tn(rng, c, s, lo, hi) / 12
        else:
            c, s, lo, hi = SC['d_annual_rate_low']
            d_rate = tn(rng, c, s, lo, hi) / 12
        d_noise = rng.normal(0, 0.002, n)
        d_cumul = np.cumsum(d_rate + d_noise)
        schedules['diabetes'] = np.clip(params['diabetes'] + d_cumul, 0.0, 1.0)

        # --- Inflammation progression ---
        # Drift rate uses truncated normal; stochastic flares remain uniform
        # because flare magnitude is inherently unpredictable.
        i_drift = np.clip(rng.normal(0.003, 0.001), 0.001, 0.005) / 12
        i_flares = (rng.random(n) < 0.02) * rng.uniform(0.05, 0.15, n)
        i_cumul = np.cumsum(i_drift + i_flares)
        schedules['inflammation'] = np.clip(params['inflammation'] + i_cumul, 0.0, 0.8)

        # --- Static parameters ---
        schedules['RAAS_gain'] = np.full(n, params['RAAS_gain'])
        schedules['TGF_gain'] = np.full(n, params['TGF_gain'])
        schedules['na_intake'] = np.full(n, params['na_intake'])

        return schedules

    @staticmethod
    def _get_schedule_val(schedule: np.ndarray, step: int) -> float:
        """Get schedule value at step, plateauing at last value."""
        idx = min(step, len(schedule) - 1)
        return float(schedule[idx])

    def _update_obs_stats(self, obs: np.ndarray):
        """Update running mean/variance for observation normalization."""
        self._obs_count += 1
        delta = obs - self._obs_mean
        self._obs_mean += delta / self._obs_count
        delta2 = obs - self._obs_mean
        self._obs_var += (delta * delta2 - self._obs_var) / self._obs_count

    def get_normalized_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics."""
        std = np.sqrt(self._obs_var + 1e-8)
        return ((obs - self._obs_mean) / std).astype(np.float32)
