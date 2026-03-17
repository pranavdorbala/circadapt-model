#!/usr/bin/env python3
"""
Monthly-Resolution Synthetic Cohort Generator
==============================================
Paper Reference: Section 3.7 extension — Monthly trajectories for RL warm-start

Generates realistic paired echocardiographic + renal observations at monthly
time points over 6-10 years, simulating disease progression in an elderly
community-dwelling cohort resembling ARIC Visit 5 participants (age ~75).

Produces (N, T, 20) trajectory arrays of the 20 core clinical variables
defined in config.CORE_VARIABLES, with realistic measurement noise and
disease progression schedules.

Usage:
    # Quick test: 50 patients, 24 months
    python synthetic_cohort_monthly.py --n_patients 50 --n_months 24 --n_workers 1 --validate

    # Full cohort: 5000 patients, 96 months (8 years)
    python synthetic_cohort_monthly.py --n_patients 5000 --n_months 96 --n_workers 8
"""

import argparse
import time
import warnings
import numpy as np
from scipy.stats import norm, rankdata
from scipy.interpolate import interp1d
from multiprocessing import Pool
from functools import partial
from typing import Dict, List, Tuple, Optional

from config import CORE_VARIABLES, MEASUREMENT_NOISE, SAMPLING_CONFIG


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Demographic Sampling
# ═══════════════════════════════════════════════════════════════════════════════

def sample_demographics(n_patients: int, rng: np.random.Generator) -> Dict:
    """Sample demographics matching ARIC Visit 5 distributions.

    ARIC V5 participants were community-dwelling elderly (~75 years),
    ~57% female, ~22% Black.
    """
    demos = {}

    # Age: ARIC V5 participants ~75 years old (range 66-90)
    demos['age'] = rng.normal(75.0, 5.0, n_patients).clip(66, 90)

    # Sex: ARIC V5 ~57% female
    demos['sex'] = rng.choice(['M', 'F'], n_patients, p=[0.43, 0.57])

    # BSA: derived from height/weight; sex-dependent
    demos['BSA'] = np.where(
        demos['sex'] == 'M',
        rng.normal(2.0, 0.18, n_patients).clip(1.5, 2.6),
        rng.normal(1.75, 0.16, n_patients).clip(1.3, 2.3),
    )

    # Height (meters): sex-dependent
    demos['height_m'] = np.where(
        demos['sex'] == 'M',
        rng.normal(1.75, 0.07, n_patients).clip(1.55, 1.95),
        rng.normal(1.62, 0.06, n_patients).clip(1.45, 1.80),
    )

    # Race: ARIC V5 ~22% Black, ~78% White
    demos['race'] = rng.choice(['Black', 'White'], n_patients, p=[0.22, 0.78])

    return demos


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2.2: Baseline Disease Parameters
# ═══════════════════════════════════════════════════════════════════════════════

def _truncated_normal_vec(
    rng: np.random.Generator, center: float, sigma: float,
    low: float, high: float, size: int,
) -> np.ndarray:
    """Vectorized truncated normal: N(center, sigma^2) clipped to [low, high].

    Under this scheme ~95% of samples fall within [center ± 2σ], making
    extreme values near the bounds exponentially unlikely. This replaces
    bimodal healthy/diseased uniform splits to reduce joint-extreme
    parameter combinations that cause CircAdapt solver divergence.
    """
    return rng.normal(center, sigma, size).clip(low, high)


def sample_disease_parameters(n_patients: int, rng: np.random.Generator) -> Dict:
    """Sample baseline disease parameters from truncated normal distributions.

    Uses SAMPLING_CONFIG for all centers, standard deviations, and bounds.
    This replaces the old bimodal uniform sampling (40/60 healthy/diseased
    splits) with unimodal truncated normals centered on clinically realistic
    values, suppressing extreme parameter combinations while maintaining
    pathological diversity.
    """
    SC = SAMPLING_CONFIG
    params = {}

    # k1_scale: LV passive stiffness [1.0=healthy, >1.5=moderate HFpEF]
    c, s, lo, hi = SC['k1_scale']
    params['k1_scale'] = _truncated_normal_vec(rng, c, s, lo, hi, n_patients)

    # Sf_scale: active contractility [1.0=healthy, <0.8=impaired]
    c, s, lo, hi = SC['Sf_scale']
    params['Sf_scale'] = _truncated_normal_vec(rng, c, s, lo, hi, n_patients)

    # Kf_scale: nephron mass [1.0=healthy, <0.7=CKD]
    c, s, lo, hi = SC['Kf_scale']
    params['Kf_scale'] = _truncated_normal_vec(rng, c, s, lo, hi, n_patients)

    # Diabetes: metabolic burden [0=none, 1=severe]
    c, s, lo, hi = SC['diabetes']
    params['diabetes'] = _truncated_normal_vec(rng, c, s, lo, hi, n_patients)

    # Inflammation: systemic inflammatory index [0=none, 1=severe]
    spec = SC['inflammation']
    if spec[0] == 'exponential':
        _, scale, lo, hi = spec
        params['inflammation'] = rng.exponential(scale, n_patients).clip(lo, hi)
    else:
        c, s, lo, hi = spec
        params['inflammation'] = _truncated_normal_vec(rng, c, s, lo, hi, n_patients)

    # Coupling and feedback parameters
    c, s, lo, hi = SC['RAAS_gain']
    params['RAAS_gain'] = _truncated_normal_vec(rng, c, s, lo, hi, n_patients)

    c, s, lo, hi = SC['TGF_gain']
    params['TGF_gain'] = _truncated_normal_vec(rng, c, s, lo, hi, n_patients)

    c, s, lo, hi = SC['na_intake']
    params['na_intake'] = _truncated_normal_vec(rng, c, s, lo, hi, n_patients)

    return params


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2.3: Correlation Structure
# ═══════════════════════════════════════════════════════════════════════════════

def apply_disease_correlations(params: Dict, rng: np.random.Generator) -> Dict:
    """Enforce clinically realistic correlations between disease parameters.

    Target correlations:
    - diabetes <-> k1_scale:  r ~ 0.35  (diabetes -> AGE -> stiffness)
    - diabetes <-> Kf_scale:  r ~ -0.40 (diabetes -> nephropathy)
    - k1_scale <-> Kf_scale:  r ~ -0.25 (cardiorenal co-occurrence)
    - inflammation <-> k1_scale: r ~ 0.20
    - inflammation <-> diabetes: r ~ 0.30
    """
    n = len(params['diabetes'])

    target_corr = np.array([
        [1.00,  0.35, -0.40,  0.30],  # diabetes
        [0.35,  1.00, -0.25,  0.20],  # k1_scale
        [-0.40, -0.25, 1.00, -0.15],  # Kf_scale
        [0.30,  0.20, -0.15,  1.00],  # inflammation
    ])

    L = np.linalg.cholesky(target_corr)

    # Convert marginals to standard normal (via quantile transform)
    keys = ['diabetes', 'k1_scale', 'Kf_scale', 'inflammation']
    Z = np.zeros((n, 4))
    for i, k in enumerate(keys):
        ranks = rankdata(params[k]) / (n + 1)
        Z[:, i] = norm.ppf(ranks)

    # Apply correlation via Cholesky
    Z_corr = (L @ Z.T).T

    # Convert back to marginals (rank-order matching)
    for i, k in enumerate(keys):
        sorted_vals = np.sort(params[k])
        rank_order = np.argsort(np.argsort(Z_corr[:, i]))
        params[k] = sorted_vals[rank_order]

    return params


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Disease Progression Schedules
# ═══════════════════════════════════════════════════════════════════════════════

def generate_progression_schedule(
    params: Dict, n_months: int, rng: np.random.Generator
) -> Dict:
    """Generate per-patient, per-month disease parameter trajectories.

    Each parameter follows clinically realistic progression with acceleration
    in late disease stages. Progression rates are drawn from truncated normal
    distributions (SAMPLING_CONFIG) rather than uniform, so most patients
    progress at moderate rates and rapid progressors are rare.

    Returns:
        schedules: dict of arrays, each (n_patients, n_months)
    """
    SC = SAMPLING_CONFIG
    n = len(params['k1_scale'])
    schedules = {}

    # --- k1_scale progression (cardiac stiffness) ---
    c, s, lo, hi = SC['k1_annual_rate']
    k1_annual_rate = _truncated_normal_vec(rng, c, s, lo, hi, n) * (1 + 0.5 * params['diabetes'])
    k1_monthly_rate = k1_annual_rate / 12
    k1_noise = rng.normal(0, 0.002, (n, n_months))
    k1_cumulative = np.cumsum(k1_monthly_rate[:, None] + k1_noise, axis=1)
    schedules['k1_scale'] = (params['k1_scale'][:, None] + k1_cumulative).clip(1.0, 3.0)

    # --- Kf_scale progression (nephron loss) ---
    # Nonlinear: faster decline at lower Kf_scale
    c, s, lo, hi = SC['Kf_annual_rate']
    kf_annual_rate = _truncated_normal_vec(rng, c, s, lo, hi, n) * (1 + 0.4 * params['diabetes'])
    kf_monthly_rate = kf_annual_rate / 12
    kf_schedules = np.zeros((n, n_months))
    kf_schedules[:, 0] = params['Kf_scale']
    for t in range(1, n_months):
        accel = 1 + 0.5 * (1 - kf_schedules[:, t-1])
        noise = rng.normal(0, 0.001, n)
        kf_schedules[:, t] = (
            kf_schedules[:, t-1] - kf_monthly_rate * accel + noise
        ).clip(0.15, 1.0)
    schedules['Kf_scale'] = kf_schedules

    # --- Sf_scale progression (contractility) ---
    c, s, lo, hi = SC['Sf_annual_rate']
    sf_annual_rate = _truncated_normal_vec(rng, c, s, lo, hi, n) * (1 + 0.3 * params['diabetes'])
    sf_monthly_rate = sf_annual_rate / 12
    sf_noise = rng.normal(0, 0.001, (n, n_months))
    sf_cumulative = np.cumsum(sf_monthly_rate[:, None] + sf_noise, axis=1)
    schedules['Sf_scale'] = (params['Sf_scale'][:, None] - sf_cumulative).clip(0.4, 1.0)

    # --- Diabetes progression ---
    d_has_diabetes = params['diabetes'] > 0.1
    c_hi, s_hi, lo_hi, hi_hi = SC['d_annual_rate']
    c_lo, s_lo, lo_lo, hi_lo = SC['d_annual_rate_low']
    d_annual_rate = np.where(
        d_has_diabetes,
        _truncated_normal_vec(rng, c_hi, s_hi, lo_hi, hi_hi, n),
        _truncated_normal_vec(rng, c_lo, s_lo, lo_lo, hi_lo, n),
    )
    d_monthly_rate = d_annual_rate / 12
    d_noise = rng.normal(0, 0.002, (n, n_months))
    d_cumulative = np.cumsum(d_monthly_rate[:, None] + d_noise, axis=1)
    schedules['diabetes'] = (params['diabetes'][:, None] + d_cumulative).clip(0.0, 1.0)

    # --- Inflammation progression ---
    # Drift rate: truncated normal centered at 0.003/month.
    # Stochastic flares remain uniform (flare magnitude is unpredictable).
    i_drift = rng.normal(0.003, 0.001, n).clip(0.001, 0.005) / 12
    i_flare_prob = 0.02
    i_flare_mag = rng.uniform(0.05, 0.15, (n, n_months))
    i_flares = (rng.random((n, n_months)) < i_flare_prob) * i_flare_mag
    i_cumulative = np.cumsum(i_drift[:, None] + i_flares, axis=1)
    schedules['inflammation'] = (
        params['inflammation'][:, None] + i_cumulative
    ).clip(0.0, 0.8)

    # --- Static parameters (constant over trajectory) ---
    for key in ['RAAS_gain', 'TGF_gain', 'na_intake']:
        schedules[key] = np.tile(params[key][:, None], (1, n_months))

    return schedules


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3.2: Phenotype Subgroups
# ═══════════════════════════════════════════════════════════════════════════════

def assign_phenotype_labels(params: Dict) -> np.ndarray:
    """Label each patient's dominant phenotype for stratified analysis.

    Returns array of strings: 'healthy', 'hfpef_dominant', 'ckd_dominant',
    'cardiorenal', 'diabetes_dominant'
    """
    labels = np.full(len(params['k1_scale']), 'healthy', dtype='U20')

    hfpef = params['k1_scale'] > 1.3
    ckd = params['Kf_scale'] < 0.7
    dm = params['diabetes'] > 0.3

    labels[hfpef & ~ckd & ~dm] = 'hfpef_dominant'
    labels[~hfpef & ckd & ~dm] = 'ckd_dominant'
    labels[hfpef & ckd] = 'cardiorenal'
    labels[dm & ~hfpef & ~ckd] = 'diabetes_dominant'
    labels[dm & (hfpef | ckd)] = 'cardiorenal'

    return labels


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Simulator-Based Trajectory Generation
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_derived_variables(all_vars, history, schedule, hist_idx, sched_idx):
    """Compute variables not directly emitted by extract_all_aric_variables.

    Handles SVR_wood, FENa_pct, CRP_mg_L which are not in the emission layer.

    Args:
        hist_idx: index into history lists (simulation step)
        sched_idx: index into schedule arrays (monthly resolution)
    """
    # SVR in Wood units = (MAP - CVP) / CO, where CVP ~ 5 mmHg
    MAP = all_vars.get('MAP_mmHg', history['MAP'][hist_idx])
    CO = all_vars.get('CO_Lmin', history['CO'][hist_idx])
    CVP = 5.0  # approximate central venous pressure
    if CO > 0.5:
        all_vars['SVR_wood'] = (MAP - CVP) / CO
    else:
        all_vars['SVR_wood'] = 25.0  # fallback for very low CO

    # FENa: fractional excretion of sodium
    # Use steady-state approximation: at equilibrium, Na_excretion ≈ Na_intake.
    # The model's transient Na_excretion can be very large in non-equilibrium.
    # FENa = Na_intake / Na_filtered * 100
    # Na_filtered = GFR (mL/min) * P_Na (mEq/L) / 1000 (L/mL) * 1440 (min/day)
    GFR = history['GFR'][hist_idx] * 1.65  # same calibration factor as eGFR
    Kf = history['Kf_scale'][hist_idx]
    plasma_Na = 140.0  # mEq/L
    Na_filtered = GFR * plasma_Na * 1440 / 1000  # mEq/day
    # Assume steady-state Na excretion ~ 150 mEq/day, mildly increased with low Kf
    Na_excr_ss = 150.0 * (1.0 + 0.3 * (1.0 - Kf))  # impaired reabsorption
    if Na_filtered > 0:
        all_vars['FENa_pct'] = (Na_excr_ss / Na_filtered) * 100
    else:
        all_vars['FENa_pct'] = 2.0

    # CRP: C-reactive protein from inflammation + diabetes indices
    # CRP = C_base * exp(delta1*i + delta2*d)
    infl = schedule['inflammation'][sched_idx]
    diab = schedule['diabetes'][sched_idx]
    C_base = 1.0  # mg/L baseline
    all_vars['CRP_mg_L'] = C_base * np.exp(1.5 * infl + 0.8 * diab)

    return all_vars


def generate_single_patient_trajectory(
    patient_idx: int,
    demo: Dict,
    schedule: Dict,
    n_months: int,
    core_var_names: List[str],
) -> np.ndarray:
    """Run coupled simulation for one patient, extract monthly observations.

    Uses the interpolation approach: runs CircAdapt with ~8-12 coupling steps
    (dt_renal=6h), then cubically interpolates the 20 core variables to
    monthly resolution.

    Returns:
        trajectory: ndarray of shape (n_months, 20)
    """
    from cardiorenal_coupling import run_coupled_simulation
    from emission_functions import extract_all_aric_variables

    # Use the proven approach: run with dt_renal_hours=6.0 and enough
    # coupling steps for convergence (~8-12 steps), then interpolate
    # the extracted variables to monthly resolution.
    # This matches the existing synthetic_cohort.py approach.
    n_sim_steps_target = max(8, n_months // 12 + 2)  # ~yearly + extras
    yearly_indices = np.linspace(0, n_months - 1, n_sim_steps_target).astype(int)

    # Build schedules by sampling at simulation time points
    cardiac_sched = [float(schedule['Sf_scale'][t]) for t in yearly_indices]
    kidney_sched = [float(schedule['Kf_scale'][t]) for t in yearly_indices]
    stiffness_sched = [float(schedule['k1_scale'][t]) for t in yearly_indices]
    inflam_sched = [float(schedule['inflammation'][t]) for t in yearly_indices]
    diab_sched = [float(schedule['diabetes'][t]) for t in yearly_indices]

    # Suppress print output from run_coupled_simulation
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        history = run_coupled_simulation(
            n_steps=n_sim_steps_target,
            dt_renal_hours=6.0,
            cardiac_schedule=cardiac_sched,
            kidney_schedule=kidney_sched,
            stiffness_schedule=stiffness_sched,
            inflammation_schedule=inflam_sched,
            diabetes_schedule=diab_sched,
        )
    finally:
        sys.stdout = old_stdout

    # Extract core 20 variables at each simulated step
    n_sim_steps = len(history['step'])
    yearly_trajectory = np.zeros((n_sim_steps, len(core_var_names)))
    last_valid_row = np.zeros(len(core_var_names))  # for forward-fill on NaN

    # Build emission_key -> core_var_name mapping
    emission_to_core = {}
    for name, meta in CORE_VARIABLES.items():
        if meta['emission_key'] is not None:
            emission_to_core[meta['emission_key']] = name

    for t_sim in range(n_sim_steps):
        # Skip steps where CircAdapt crashed (NaN in history)
        if np.isnan(history['MAP'][t_sim]):
            yearly_trajectory[t_sim] = last_valid_row
            continue
        # Build renal state dict for emission functions
        renal_state = {
            'GFR': history['GFR'][t_sim],
            'V_blood': history['V_blood'][t_sim],
            'C_Na': 140.0,
            'Na_excretion': history['Na_excr'][t_sim],
            'P_glom': history['P_glom'][t_sim],
            'Kf_scale': history['Kf_scale'][t_sim],
            'RBF': 1100.0 * history['Kf_scale'][t_sim],
        }

        # We need the CircAdapt model to extract waveform-based variables.
        # Since run_coupled_simulation doesn't store model objects per step,
        # we use the history scalars directly and reconstruct what we can.
        # For waveform-based variables (LVIDd, LVmass, GLS, E, e', LVEDP,
        # LAvolume, PASP), we use the PV loop data stored in history.

        # Build a variable dict from what's available in history
        all_vars = {}

        # --- Variables directly from history ---
        all_vars['SBP_mmHg'] = history['SBP'][t_sim]
        all_vars['MAP_mmHg'] = history['MAP'][t_sim]
        all_vars['CO_Lmin'] = history['CO'][t_sim]
        all_vars['LVEF_pct'] = history['EF'][t_sim]

        # --- Variables derived from PV loop waveforms ---
        # NOTE: PV loop data is already in mL and mmHg (converted in
        # CircAdaptHeartModel._extract_hemodynamics)
        V_lv, p_lv = history['PV_LV'][t_sim]
        V_rv, p_rv = history['PV_RV'][t_sim]

        EDV = float(np.max(V_lv))   # already mL
        ESV = float(np.min(V_lv))   # already mL
        SV = EDV - ESV

        # LVIDd from EDV (spherical approximation, Eq 49)
        # Apply 0.78 correction factor: spherical -> M-mode echo measurement
        # (LV is ellipsoidal, not spherical; M-mode measures minor axis)
        LVIDd_sphere = 2.0 * (3.0 * EDV / (4.0 * np.pi))**(1.0/3.0)
        LVIDd = LVIDd_sphere * 0.78
        all_vars['LVIDd_cm'] = LVIDd

        # LV mass: use ASE cube formula approximation
        # LV mass ≈ 0.8 * 1.04 * [(LVIDd + IVSd + LVPWd)^3 - LVIDd^3] + 0.6
        # Approximate wall thicknesses from k1 scaling
        k1 = history['k1_scale'][t_sim]
        IVSd = 1.0 * (k1 ** 0.25)   # ~1.0 cm healthy, increases with stiffness
        LVPWd = 1.0 * (k1 ** 0.25)
        all_vars['LV_mass_g'] = 0.8 * 1.04 * (
            (LVIDd + IVSd + LVPWd)**3 - LVIDd**3
        ) + 0.6

        # GLS from EF and contractility
        EF = history['EF'][t_sim]
        sf_eff = history['effective_Sf'][t_sim]
        all_vars['GLS_pct'] = -20.0 * sf_eff * (EF / 60.0)

        # E velocity: related to filling dynamics
        # E is less sensitive to stiffness than e' (E pseudonormalizes in
        # grade II diastolic dysfunction)
        all_vars['E_vel_cms'] = 72.0 * (1.0 - 0.08 * (k1 - 1.0)) * sf_eff

        # e' tissue Doppler: strongly decreases with stiffness (k1^0.8)
        # This is the key diastolic dysfunction marker
        all_vars['e_prime_avg_cms'] = 10.0 / (k1 ** 0.8) * sf_eff

        # E/e' derived
        E = all_vars['E_vel_cms']
        e_prime = all_vars['e_prime_avg_cms']
        Ee_ratio = E / max(e_prime, 1.0)
        all_vars['E_e_prime_avg'] = Ee_ratio

        # LVEDP / LAP estimate from E/e' (Nagueh formula, ASE 2016)
        # This is more reliable than the raw PV loop pressure at argmax(V),
        # which falls in the isovolumic contraction phase.
        all_vars['LAP_est_mmHg'] = 1.24 * Ee_ratio + 1.9

        # LA volume: increases with chronic elevated filling pressure and stiffness
        LAP_est = all_vars['LAP_est_mmHg']
        all_vars['LAV_max_mL'] = 52.0 * (
            1.0 + 0.15 * (k1 - 1.0) + 0.02 * max(LAP_est - 10, 0)
        )

        # PASP: from RV systolic pressure (already in mmHg)
        # Scale to match echo PASP (clinical measurement via TR jet)
        p_rv_max = float(np.max(p_rv))  # already mmHg
        PASP = 0.55 * p_rv_max + 5.0
        all_vars['PASP_mmHg'] = max(PASP, 18.0)

        # DBP
        all_vars['DBP_mmHg'] = history['DBP'][t_sim]

        # Renal variables from emission function logic
        # Apply calibration factor: CircAdapt's lower MAP (~86 vs ~95 mmHg)
        # systematically reduces model GFR vs real GFR. Scale by 1.35 to
        # align healthy baseline GFR with the ARIC V5 distribution.
        GFR_raw = history['GFR'][t_sim]
        GFR = GFR_raw * 1.65
        BSA = demo['BSA']
        all_vars['eGFR_mL_min_173m2'] = GFR * 1.73 / BSA
        # Creatinine: Scr ≈ k / GFR. Constants calibrated for corrected GFR
        # so that healthy (GFR~80) produces Scr~0.9 (M) or ~0.75 (F)
        if demo['sex'] == 'F':
            all_vars['serum_creatinine_mg_dL'] = 60.0 / max(GFR, 5.0)
        else:
            all_vars['serum_creatinine_mg_dL'] = 75.0 / max(GFR, 5.0)

        P_glom = history['P_glom'][t_sim]
        Kf_scale = history['Kf_scale'][t_sim]
        all_vars['UACR_mg_g'] = 10.0 * (P_glom / 60.0)**2 * (1.0 / max(Kf_scale, 0.1))**1.5

        all_vars['NTproBNP_pg_mL'] = 75.0 * np.exp(
            3.0 * (history['V_blood'][t_sim] - 5000.0) / 5000.0
        )

        # Compute derived variables (SVR, FENa, CRP)
        # Use t_sim for history indexing (matches simulation steps),
        # but yearly_indices[t_sim] for schedule indexing (monthly resolution)
        monthly_idx = yearly_indices[t_sim]
        all_vars = _compute_derived_variables(
            all_vars, history,
            {'inflammation': schedule['inflammation'],
             'diabetes': schedule['diabetes']},
            t_sim,  # index into history (simulation step)
            monthly_idx,  # index into schedule (monthly resolution)
        )

        # Map emission keys to core variable positions
        for j, var_name in enumerate(core_var_names):
            meta = CORE_VARIABLES[var_name]
            emission_key = meta['emission_key']
            if emission_key is not None and emission_key in all_vars:
                yearly_trajectory[t_sim, j] = all_vars[emission_key]
            elif var_name in all_vars:
                # For derived variables stored directly by name
                yearly_trajectory[t_sim, j] = all_vars[var_name]
            else:
                yearly_trajectory[t_sim, j] = np.nan

        # Update forward-fill buffer
        if not np.any(np.isnan(yearly_trajectory[t_sim])):
            last_valid_row = yearly_trajectory[t_sim].copy()

    # Interpolate from simulation steps to monthly resolution
    if n_sim_steps < n_months:
        sim_times = np.linspace(0, n_months - 1, n_sim_steps)
        monthly_times = np.arange(n_months)

        interp_fn = interp1d(
            sim_times, yearly_trajectory, axis=0,
            kind='cubic' if n_sim_steps >= 4 else 'linear',
            fill_value='extrapolate',
        )
        trajectory = interp_fn(monthly_times)
    else:
        trajectory = yearly_trajectory[:n_months]

    return trajectory


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Measurement Noise
# ═══════════════════════════════════════════════════════════════════════════════

# Physiological bounds: no negative EF, GFR, etc.
PHYSIOLOGICAL_BOUNDS = {
    "LVEF_pct": (10, 80),
    "GLS_pct": (-30, -5),
    "CO_L_min": (1.5, 10),
    "E_cm_s": (20, 150),
    "e_prime_cm_s": (2, 20),
    "E_over_e_prime": (3, 30),
    "LVEDP_mmHg": (2, 35),
    "LAvolume_mL": (15, 150),
    "PASP_mmHg": (15, 80),
    "eGFR_mL_min": (5, 130),
    "creatinine_mg_dL": (0.3, 5.0),
    "UACR_mg_g": (1, 500),
    "FENa_pct": (0.1, 5.0),
    "NTproBNP_pg_mL": (5, 5000),
    "CRP_mg_L": (0.1, 30),
    "SBP_mmHg": (70, 200),
    "MAP_mmHg": (50, 140),
    "SVR_wood": (8, 40),
    "LVIDd_cm": (3.0, 7.0),
    "LVmass_g": (80, 400),
}


def add_measurement_noise(
    trajectories: np.ndarray, var_names: List[str], rng: np.random.Generator
) -> np.ndarray:
    """Add realistic measurement noise to clean simulator outputs.

    Args:
        trajectories: (n_patients, n_months, 20) clean outputs
        var_names: list of 20 variable names
        rng: numpy random generator

    Returns:
        noisy_trajectories: (n_patients, n_months, 20)
    """
    noisy = trajectories.copy()

    for j, var_name in enumerate(var_names):
        noise_type, magnitude = MEASUREMENT_NOISE[var_name]

        if noise_type == "gaussian_absolute":
            noise = rng.normal(0, magnitude, trajectories[:, :, j].shape)
            noisy[:, :, j] += noise
        elif noise_type == "gaussian_relative":
            noise = rng.normal(1.0, magnitude, trajectories[:, :, j].shape)
            noisy[:, :, j] *= noise

    # Enforce physiological bounds
    for j, var_name in enumerate(var_names):
        if var_name in PHYSIOLOGICAL_BOUNDS:
            lo, hi = PHYSIOLOGICAL_BOUNDS[var_name]
            noisy[:, :, j] = noisy[:, :, j].clip(lo, hi)

    return noisy


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_marginals(trajectories: np.ndarray, var_names: List[str]) -> bool:
    """Compare synthetic baseline distributions to ARIC V5 targets.

    Returns True if all variables are within 2 SD of ARIC means.
    """
    print(f"\n{'Variable':<20} {'Synth Mean':>10} {'Synth SD':>10} "
          f"{'ARIC Mean':>10} {'ARIC SD':>10} {'Status':>8}")
    print("-" * 70)

    baseline = trajectories[:, 0, :]
    all_pass = True

    for j, var_name in enumerate(var_names):
        meta = CORE_VARIABLES[var_name]
        synth_mean = np.nanmean(baseline[:, j])
        synth_sd = np.nanstd(baseline[:, j])
        aric_mean = meta['aric_v5_mean']
        aric_sd = meta['aric_v5_sd']

        deviation = abs(synth_mean - aric_mean) / max(aric_sd, 1e-6)
        status = "OK" if deviation < 1.0 else "WARN" if deviation < 2.0 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"{var_name:<20} {synth_mean:>10.2f} {synth_sd:>10.2f} "
              f"{aric_mean:>10.2f} {aric_sd:>10.2f} {status:>8}")

    return all_pass


def validate_trajectories(
    trajectories: np.ndarray, var_names: List[str], labels: np.ndarray
) -> bool:
    """Check that disease trajectories are physiologically plausible."""
    var_idx = {name: i for i, name in enumerate(var_names)}
    all_pass = True

    print("\n--- Trajectory Plausibility Checks ---")

    # Check 1: HFpEF patients maintain EF
    hfpef_mask = labels == 'hfpef_dominant'
    if hfpef_mask.any():
        ef_traj = trajectories[hfpef_mask, :, var_idx['LVEF_pct']]
        pct_preserved = (ef_traj > 50).mean()
        status = "PASS" if pct_preserved > 0.85 else "WARN" if pct_preserved > 0.70 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  HFpEF EF>50% fraction: {pct_preserved:.2f} (expect >0.85) [{status}]")

    # Check 2: CKD eGFR declines
    ckd_mask = labels == 'ckd_dominant'
    if ckd_mask.any():
        egfr_traj = trajectories[ckd_mask, :, var_idx['eGFR_mL_min']]
        delta = egfr_traj[:, -1] - egfr_traj[:, 0]
        pct_declined = (delta < 0).mean()
        status = "PASS" if pct_declined > 0.80 else "WARN" if pct_declined > 0.60 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  CKD eGFR declined fraction: {pct_declined:.2f} (expect >0.80) [{status}]")

    # Check 3: Cardiorenal E/e' increases
    cr_mask = labels == 'cardiorenal'
    if cr_mask.any():
        ee_traj = trajectories[cr_mask, :, var_idx['E_over_e_prime']]
        delta = ee_traj[:, -1] - ee_traj[:, 0]
        pct_increased = (delta > 0).mean()
        status = "PASS" if pct_increased > 0.70 else "WARN" if pct_increased > 0.50 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  Cardiorenal E/e' increased fraction: {pct_increased:.2f} (expect >0.70) [{status}]")

    # Check 4: NT-proBNP <-> E/e' correlation
    for t in [0, min(48, trajectories.shape[1]-1), -1]:
        bnp = trajectories[:, t, var_idx['NTproBNP_pg_mL']]
        ee = trajectories[:, t, var_idx['E_over_e_prime']]
        valid = ~np.isnan(bnp) & ~np.isnan(ee) & (bnp > 0)
        if valid.sum() > 50:
            r = np.corrcoef(np.log(bnp[valid] + 1), ee[valid])[0, 1]
            status = "PASS" if r > 0.3 else "WARN" if r > 0.1 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  log(NT-proBNP) <-> E/e' correlation at month {t}: "
                  f"r={r:.2f} (expect >0.3) [{status}]")

    # Check 5: No NaN values
    n_nan = np.isnan(trajectories).sum()
    if n_nan > 0:
        print(f"  WARNING: {n_nan} NaN values found in trajectories")
        all_pass = False
    else:
        print(f"  No NaN values: PASS")

    return all_pass


# ═══════════════════════════════════════════════════════════════════════════════
# Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_one_patient(args):
    """Worker function for parallel patient generation."""
    patient_idx, demo_i, sched_i, n_months, core_var_names = args
    try:
        traj = generate_single_patient_trajectory(
            patient_idx, demo_i, sched_i, n_months, core_var_names,
        )
        return patient_idx, traj, None
    except Exception as e:
        return patient_idx, None, str(e)


def generate_cohort(
    n_patients: int,
    n_months: int = 96,
    n_workers: int = 8,
    seed: int = 42,
    validate: bool = False,
    output_path: str = 'cohort_monthly.npz',
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Generate full synthetic cohort.

    Args:
        n_patients: number of patients
        n_months: trajectory length (default 96 = 8 years)
        n_workers: CPU cores for parallel generation
        seed: random seed
        validate: run validation checks after generation
        output_path: path to save .npz file

    Returns:
        trajectories: (n_patients, n_months, 20) noisy observed variables
        var_names: list of 20 variable names
        labels: (n_patients,) phenotype labels
    """
    t_start = time.time()
    rng = np.random.default_rng(seed)
    core_var_names = list(CORE_VARIABLES.keys())

    # 1. Sample demographics
    print(f"[1/6] Sampling demographics for {n_patients} patients...")
    demos = sample_demographics(n_patients, rng)

    # 2. Sample disease parameters with correlations
    print("[2/6] Sampling disease parameters...")
    params = sample_disease_parameters(n_patients, rng)
    params = apply_disease_correlations(params, rng)

    # 3. Generate progression schedules
    print(f"[3/6] Generating {n_months}-month progression schedules...")
    schedules = generate_progression_schedule(params, n_months, rng)

    # 4. Assign phenotype labels
    labels = assign_phenotype_labels(params)
    unique, counts = np.unique(labels, return_counts=True)
    print("[4/6] Phenotype distribution:")
    for u, c in zip(unique, counts):
        print(f"       {u}: {c} ({100*c/n_patients:.1f}%)")

    # 5. Run simulator for each patient
    print(f"[5/6] Generating trajectories ({n_workers} workers)...")

    # Build per-patient argument tuples
    worker_args = []
    for i in range(n_patients):
        demo_i = {k: (v[i] if isinstance(v[i], (int, float, np.floating, np.integer))
                       else str(v[i]))
                  for k, v in demos.items()}
        sched_i = {k: v[i].copy() for k, v in schedules.items()}
        worker_args.append((i, demo_i, sched_i, n_months, core_var_names))

    if n_workers <= 1:
        # Sequential execution
        results = []
        for args in worker_args:
            results.append(_generate_one_patient(args))
            if (args[0] + 1) % max(1, n_patients // 10) == 0:
                print(f"       ... {args[0]+1}/{n_patients}")
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_generate_one_patient, worker_args)

    # Collect results
    trajectories_clean = np.full((n_patients, n_months, len(core_var_names)), np.nan)
    n_failed = 0
    for idx, traj, err in results:
        if traj is not None:
            trajectories_clean[idx] = traj
        else:
            n_failed += 1
            if n_failed <= 5:
                warnings.warn(f"Patient {idx} failed: {err}")

    if n_failed > 0:
        print(f"       {n_failed}/{n_patients} patients failed")

    # 6. Add measurement noise
    print("[6/6] Adding measurement noise...")
    rng_noise = np.random.default_rng(seed + 1)
    trajectories_noisy = add_measurement_noise(
        trajectories_clean, core_var_names, rng_noise
    )

    # Save
    print(f"\nSaving to {output_path}...")
    np.savez_compressed(
        output_path,
        trajectories=trajectories_noisy.astype(np.float32),
        trajectories_clean=trajectories_clean.astype(np.float32),
        var_names=np.array(core_var_names),
        demographics=demos,
        disease_params={k: v for k, v in schedules.items()},
        phenotype_labels=labels,
        metadata={
            'n_patients': n_patients,
            'n_months': n_months,
            'seed': seed,
            'variable_definitions': {k: {kk: vv for kk, vv in v.items()}
                                     for k, v in CORE_VARIABLES.items()},
        },
    )

    elapsed = time.time() - t_start
    print(f"Done! Shape: {trajectories_noisy.shape}, Time: {elapsed:.1f}s")

    # Validate
    if validate:
        print("\n" + "=" * 70)
        print("  VALIDATION")
        print("=" * 70)
        marginals_ok = validate_marginals(trajectories_noisy, core_var_names)
        traj_ok = validate_trajectories(trajectories_noisy, core_var_names, labels)
        if marginals_ok and traj_ok:
            print("\nAll validation checks passed.")
        else:
            print("\nSome validation checks failed or warned. Review above.")

    return trajectories_noisy, core_var_names, labels


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate monthly-resolution synthetic cardiorenal cohort'
    )
    parser.add_argument('--n_patients', type=int, default=5000,
                        help='Number of patients (default: 5000)')
    parser.add_argument('--n_months', type=int, default=96,
                        help='Trajectory length in months (default: 96 = 8 years)')
    parser.add_argument('--n_workers', type=int, default=8,
                        help='CPU cores for parallel generation (default: 8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation checks after generation')
    parser.add_argument('--output', type=str, default='cohort_monthly.npz',
                        help='Output file path (default: cohort_monthly.npz)')

    args = parser.parse_args()
    generate_cohort(
        n_patients=args.n_patients,
        n_months=args.n_months,
        n_workers=args.n_workers,
        seed=args.seed,
        validate=args.validate,
        output_path=args.output,
    )
