#!/usr/bin/env python3
"""
synthetic_cohort_monthly.py — Generate monthly-resolution synthetic cardiorenal
trajectories for RL warm-start and pipeline testing.

Usage:
    python synthetic_cohort_monthly.py --n_patients 5000 --n_months 96 \
                                       --n_workers 8 --seed 42 [--validate]
Output:
    cohort_monthly.npz with keys:
        trajectories:       float32, (N, T, 20)  — noisy observations
        trajectories_clean: float32, (N, T, 20)  — clean simulator output
        var_names:          str, (20,)
        demographics:       dict of arrays, each (N,)
        disease_params:     dict of arrays, each (N, T)
        phenotype_labels:   str, (N,)
        metadata:           dict
"""

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
import multiprocessing as mp
import argparse
import time
import warnings
import sys

from config import (
    CORE_20_VARIABLES, MEASUREMENT_NOISE, CYSTATIN_C_PARAMS,
    PASP_MISSING_PARAMS,
)

# Ordered variable names — defines column order in the (N, T, 20) array
VAR_NAMES = [
    "LVIDd_cm", "LVmass_g",
    "LVEF_pct", "GLS_pct", "CO_L_min",
    "E_cm_s", "e_prime_sept_cm_s", "E_over_e_prime_sept", "E_over_A_ratio",
    "LAvolume_mL", "PASP_mmHg",
    "SBP_mmHg", "MAP_mmHg", "SVR_wood",
    "eGFR_mL_min", "creatinine_mg_dL", "UACR_mg_g", "cystatin_C_mg_L",
    "NTproBNP_pg_mL", "CRP_mg_L",
]
assert len(VAR_NAMES) == 20

# Variable-name to column-index map for fast lookup
VAR_IDX = {name: i for i, name in enumerate(VAR_NAMES)}


# ============================================================================
# FUNCTION 1: sample_demographics
# ============================================================================

def sample_demographics(n_patients, rng):
    """Sample ARIC V5-like demographics for n_patients.

    Returns dict with keys: age, sex, BSA, height_m, race.
    Each value is ndarray of shape (n_patients,).
    """
    age = rng.normal(75.0, 5.0, n_patients).clip(66, 90)
    sex = rng.choice(['M', 'F'], size=n_patients, p=[0.43, 0.57])
    race = rng.choice(['Black', 'White'], size=n_patients, p=[0.22, 0.78])

    height_m = np.empty(n_patients)
    BSA = np.empty(n_patients)
    for i in range(n_patients):
        if sex[i] == 'M':
            height_m[i] = rng.normal(1.75, 0.07)
            BSA[i] = rng.normal(2.0, 0.18)
        else:
            height_m[i] = rng.normal(1.62, 0.06)
            BSA[i] = rng.normal(1.75, 0.16)
    height_m = height_m.clip(1.45, 1.95)
    BSA = BSA.clip(1.3, 2.6)

    return {
        'age': age,
        'sex': sex,
        'BSA': BSA,
        'height_m': height_m,
        'race': race,
    }


# ============================================================================
# FUNCTION 2: sample_disease_parameters
# ============================================================================

def sample_disease_parameters(n_patients, rng):
    """Sample baseline disease parameters for n_patients.

    Returns dict with keys matching simulator inputs.
    """
    # k1_scale: 40% healthy U(1.0,1.1), 60% U(1.0,2.5)
    k1_scale = np.empty(n_patients)
    healthy_mask = rng.random(n_patients) < 0.40
    k1_scale[healthy_mask] = rng.uniform(1.0, 1.1, healthy_mask.sum())
    k1_scale[~healthy_mask] = rng.uniform(1.0, 2.5, (~healthy_mask).sum())

    # Sf_scale: Normal(0.95, 0.08) clipped [0.5, 1.0]
    Sf_scale = rng.normal(0.95, 0.08, n_patients).clip(0.5, 1.0)

    # Kf_scale: 50% healthy U(0.85,1.0), 50% Beta(3,2)*0.7+0.3
    Kf_scale = np.empty(n_patients)
    kf_healthy = rng.random(n_patients) < 0.50
    Kf_scale[kf_healthy] = rng.uniform(0.85, 1.0, kf_healthy.sum())
    Kf_scale[~kf_healthy] = rng.beta(3, 2, (~kf_healthy).sum()) * 0.7 + 0.3

    # diabetes: 35% diabetic Beta(2,3), 65% non-diabetic U(0,0.05)
    diabetes = np.empty(n_patients)
    dm_mask = rng.random(n_patients) < 0.35
    diabetes[dm_mask] = rng.beta(2, 3, dm_mask.sum())
    diabetes[~dm_mask] = rng.uniform(0, 0.05, (~dm_mask).sum())

    # inflammation: Exponential(0.12) clipped [0, 0.8]
    inflammation = rng.exponential(0.12, n_patients).clip(0, 0.8)

    # RAAS_gain: Normal(1.5, 0.3) clipped [0.5, 3.0]
    RAAS_gain = rng.normal(1.5, 0.3, n_patients).clip(0.5, 3.0)

    # TGF_gain: Normal(2.0, 0.4) clipped [1.0, 4.0]
    TGF_gain = rng.normal(2.0, 0.4, n_patients).clip(1.0, 4.0)

    # na_intake: Normal(150, 30) clipped [80, 250]
    na_intake = rng.normal(150, 30, n_patients).clip(80, 250)

    return {
        'k1_scale': k1_scale,
        'Sf_scale': Sf_scale,
        'Kf_scale': Kf_scale,
        'diabetes': diabetes,
        'inflammation': inflammation,
        'RAAS_gain': RAAS_gain,
        'TGF_gain': TGF_gain,
        'na_intake': na_intake,
    }


# ============================================================================
# FUNCTION 3: apply_disease_correlations
# ============================================================================

def apply_disease_correlations(params, rng):
    """Iman-Conover rank-based correlation induction for [d, k1, Kf, i].

    Modifies params in-place to impose target correlations.
    """
    keys = ['diabetes', 'k1_scale', 'Kf_scale', 'inflammation']
    n = len(params[keys[0]])

    # Target correlation matrix for [d, k1, Kf, i]
    target_corr = np.array([
        [1.00,  0.35, -0.40,  0.30],   # diabetes
        [0.35,  1.00, -0.25,  0.20],   # k1_scale
        [-0.40, -0.25, 1.00, -0.15],   # Kf_scale
        [0.30,  0.20, -0.15,  1.00],   # inflammation
    ])

    # Cholesky decomposition
    L = np.linalg.cholesky(target_corr)

    # Convert each marginal to ranks -> quantiles -> standard normal
    data = np.column_stack([params[k] for k in keys])
    ranks = np.empty_like(data)
    for j in range(4):
        ranks[:, j] = stats.rankdata(data[:, j])

    # Ranks to quantiles to standard normal
    U = (ranks - 0.5) / n
    Z = stats.norm.ppf(U)

    # Apply correlation structure
    Z_corr = (L @ Z.T).T

    # Rank-match back to original marginals
    for j in range(4):
        target_order = np.argsort(np.argsort(Z_corr[:, j]))
        original_sorted = np.sort(data[:, j])
        params[keys[j]] = original_sorted[target_order]

    return params


# ============================================================================
# FUNCTION 4: generate_progression_schedule
# ============================================================================

def generate_progression_schedule(params, n_months, rng):
    """Generate per-month disease parameter trajectories.

    Returns dict of (n_patients, n_months) arrays.
    """
    n = len(params['k1_scale'])

    # --- k1_scale progression ---
    k1_annual_rate = rng.uniform(0.02, 0.10, n) * (1 + 0.5 * params['diabetes'])
    k1_monthly = k1_annual_rate / 12.0
    k1_sched = np.empty((n, n_months))
    k1_sched[:, 0] = params['k1_scale']
    for t in range(1, n_months):
        noise = rng.normal(0, 0.002, n)
        k1_sched[:, t] = k1_sched[:, t - 1] + k1_monthly + noise
    k1_sched = k1_sched.clip(1.0, 3.0)

    # --- Kf_scale progression (nonlinear: accelerates as Kf drops) ---
    Kf_annual_rate = rng.uniform(0.01, 0.05, n) * (1 + 0.4 * params['diabetes'])
    Kf_monthly = Kf_annual_rate / 12.0
    Kf_sched = np.empty((n, n_months))
    Kf_sched[:, 0] = params['Kf_scale']
    for t in range(1, n_months):
        accel = 1 + 0.5 * (1 - Kf_sched[:, t - 1])
        noise = rng.normal(0, 0.001, n)
        Kf_sched[:, t] = Kf_sched[:, t - 1] - Kf_monthly * accel + noise
    Kf_sched = Kf_sched.clip(0.15, 1.0)

    # --- Sf_scale progression ---
    Sf_annual_rate = rng.uniform(0.002, 0.015, n) * (1 + 0.3 * params['diabetes'])
    Sf_monthly = Sf_annual_rate / 12.0
    Sf_sched = np.empty((n, n_months))
    Sf_sched[:, 0] = params['Sf_scale']
    for t in range(1, n_months):
        noise = rng.normal(0, 0.001, n)
        Sf_sched[:, t] = Sf_sched[:, t - 1] - Sf_monthly + noise
    Sf_sched = Sf_sched.clip(0.4, 1.0)

    # --- diabetes progression ---
    dm_rate = np.where(
        params['diabetes'] > 0.1,
        rng.uniform(0.02, 0.06, n),
        np.full(n, 0.005),
    )
    dm_monthly = dm_rate / 12.0
    dm_sched = np.empty((n, n_months))
    dm_sched[:, 0] = params['diabetes']
    for t in range(1, n_months):
        noise = rng.normal(0, 0.002, n)
        dm_sched[:, t] = dm_sched[:, t - 1] + dm_monthly + noise
    dm_sched = dm_sched.clip(0, 1)

    # --- inflammation (drift + stochastic flares) ---
    infl_drift = rng.uniform(0.001, 0.005, n) / 12.0
    infl_sched = np.empty((n, n_months))
    infl_sched[:, 0] = params['inflammation']
    for t in range(1, n_months):
        flare = (rng.random(n) < 0.02) * rng.uniform(0.05, 0.15, n)
        infl_sched[:, t] = infl_sched[:, t - 1] + infl_drift + flare
    infl_sched = infl_sched.clip(0, 0.8)

    # --- Static parameters: constant over trajectory ---
    RAAS_sched = np.tile(params['RAAS_gain'][:, None], (1, n_months))
    TGF_sched = np.tile(params['TGF_gain'][:, None], (1, n_months))
    na_sched = np.tile(params['na_intake'][:, None], (1, n_months))

    return {
        'k1_scale': k1_sched,
        'Sf_scale': Sf_sched,
        'Kf_scale': Kf_sched,
        'diabetes': dm_sched,
        'inflammation': infl_sched,
        'RAAS_gain': RAAS_sched,
        'TGF_gain': TGF_sched,
        'na_intake': na_sched,
    }


# ============================================================================
# FUNCTION 5: assign_phenotype_labels
# ============================================================================

def assign_phenotype_labels(params):
    """Assign phenotype labels based on baseline disease parameters.

    Returns (n_patients,) string array.
    """
    n = len(params['k1_scale'])
    labels = np.full(n, 'healthy', dtype='U20')

    hfpef = params['k1_scale'] > 1.3
    ckd = params['Kf_scale'] < 0.7
    dm = params['diabetes'] > 0.3

    cardiorenal = (hfpef & ckd) | (dm & (hfpef | ckd))
    hfpef_dominant = hfpef & ~ckd & ~dm
    ckd_dominant = ckd & ~hfpef & ~dm
    diabetes_dominant = dm & ~hfpef & ~ckd

    # Assign in priority order (cardiorenal overrides)
    labels[hfpef_dominant] = 'hfpef_dominant'
    labels[ckd_dominant] = 'ckd_dominant'
    labels[diabetes_dominant] = 'diabetes_dominant'
    labels[cardiorenal] = 'cardiorenal'

    return labels


# ============================================================================
# FUNCTION 6: generate_single_patient_trajectory
# ============================================================================

def generate_single_patient_trajectory(patient_idx, demo, schedule, n_months,
                                        var_names, cystatin_params, use_circadapt):
    """Generate one patient's monthly trajectory of 20 clinical variables.

    Tries CircAdapt coupled simulation first. If unavailable or too slow,
    falls back to cubic-interpolated yearly simulation or parametric model.

    Returns: trajectory (n_months, 20) float32 array
    """
    if use_circadapt:
        try:
            return _run_circadapt_trajectory(
                n_months, schedule, demo, cystatin_params,
            )
        except Exception:
            pass

    return _parametric_trajectory(
        n_months, schedule, demo, cystatin_params,
    )


def _run_circadapt_trajectory(n_months, schedule, demo, cystatin_params):
    """Run CircAdapt at yearly resolution and cubic-interpolate to monthly."""
    from cardiorenal_coupling import run_coupled_simulation
    from emission_functions import extract_all_aric_variables

    k1_sched = schedule['k1_scale']
    Sf_sched = schedule['Sf_scale']
    Kf_sched = schedule['Kf_scale']
    dm_sched = schedule['diabetes']
    infl_sched = schedule['inflammation']

    age = demo['age']
    sex = demo['sex']
    BSA = demo['BSA']
    height_m = demo['height_m']
    is_black = demo['race'] == 'Black'

    n_years = max(n_months // 12, 2)
    yearly_indices = np.linspace(0, n_months - 1, n_years, dtype=int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist = run_coupled_simulation(
            n_steps=n_years,
            dt_renal_hours=6.0,  # Keep stable dt; yearly steps via schedule
            cardiac_schedule=[float(Sf_sched[i]) for i in yearly_indices],
            kidney_schedule=[float(Kf_sched[i]) for i in yearly_indices],
            stiffness_schedule=[float(k1_sched[i]) for i in yearly_indices],
            inflammation_schedule=[float(infl_sched[i]) for i in yearly_indices],
            diabetes_schedule=[float(dm_sched[i]) for i in yearly_indices],
        )

    # Check for NaN divergence
    if any(np.isnan(v) for v in hist.get('MAP', [0])):
        raise RuntimeError("CircAdapt diverged")

    # Extract 20 variables at each yearly step
    n_vars = len(VAR_NAMES)
    yearly_data = np.empty((n_years, n_vars), dtype=np.float32)

    for s in range(n_years):
        yearly_data[s, VAR_IDX['SBP_mmHg']] = hist['SBP'][s]
        yearly_data[s, VAR_IDX['MAP_mmHg']] = hist['MAP'][s]
        yearly_data[s, VAR_IDX['CO_L_min']] = hist['CO'][s]
        yearly_data[s, VAR_IDX['LVEF_pct']] = hist['EF'][s]
        yearly_data[s, VAR_IDX['eGFR_mL_min']] = hist['GFR'][s]

        # Derived variables from history
        CO = max(hist['CO'][s], 0.5)
        MAP = hist['MAP'][s]
        CVP = 5.0
        yearly_data[s, VAR_IDX['SVR_wood']] = (MAP - CVP) / CO

        # Renal variables
        GFR = max(hist['GFR'][s], 5.0)
        infl_t = infl_sched[yearly_indices[s]]
        yearly_data[s, VAR_IDX['creatinine_mg_dL']] = (
            90.0 / GFR if sex == 'M' else 60.0 / GFR
        )
        yearly_data[s, VAR_IDX['cystatin_C_mg_L']] = (
            cystatin_params['D_cys'] / GFR * (1 + cystatin_params['epsilon'] * infl_t)
        )
        yearly_data[s, VAR_IDX['UACR_mg_g']] = max(
            5.0 * (1 + 2.0 * (1 - Kf_sched[yearly_indices[s]])) *
            (1 + 0.5 * dm_sched[yearly_indices[s]]),
            0.5
        )

        # Approximate remaining variables from hemodynamics
        SV = hist['SV'][s]
        EF = hist['EF'][s]
        EDV = SV / max(EF / 100, 0.1)
        yearly_data[s, VAR_IDX['LVIDd_cm']] = 0.895 * EDV ** (1/3)

        k1_t = k1_sched[yearly_indices[s]]
        yearly_data[s, VAR_IDX['LVmass_g']] = 150 * (1 + 0.15 * (k1_t - 1))
        yearly_data[s, VAR_IDX['GLS_pct']] = -18 * (EF / 65) * Sf_sched[yearly_indices[s]]
        yearly_data[s, VAR_IDX['E_cm_s']] = 68 * (1 + 0.2 * (k1_t - 1))
        yearly_data[s, VAR_IDX['e_prime_sept_cm_s']] = 5.7 / k1_t
        yearly_data[s, VAR_IDX['E_over_e_prime_sept']] = (
            yearly_data[s, VAR_IDX['E_cm_s']] /
            max(yearly_data[s, VAR_IDX['e_prime_sept_cm_s']], 1.0)
        )
        yearly_data[s, VAR_IDX['E_over_A_ratio']] = 0.86 / (1 + 0.15 * (k1_t - 1))
        yearly_data[s, VAR_IDX['LAvolume_mL']] = 49 * (1 + 0.2 * (k1_t - 1))
        yearly_data[s, VAR_IDX['PASP_mmHg']] = 28 + 5 * (k1_t - 1) + 0.1 * (MAP - 93)

        # Biomarkers
        LVEDP_est = 10 * k1_t
        yearly_data[s, VAR_IDX['NTproBNP_pg_mL']] = (
            50 * np.exp(0.05 * LVEDP_est + 0.02 * CVP + 0.003 * 150 / BSA)
        )
        yearly_data[s, VAR_IDX['CRP_mg_L']] = (
            2.0 * np.exp(3.0 * infl_t + 1.5 * dm_sched[yearly_indices[s]])
        )

    # Cubic interpolation to monthly
    x_yearly = np.array(yearly_indices, dtype=float)
    x_monthly = np.arange(n_months, dtype=float)
    trajectory = np.empty((n_months, len(VAR_NAMES)), dtype=np.float32)
    for j in range(len(VAR_NAMES)):
        f = interp1d(x_yearly, yearly_data[:, j], kind='cubic',
                     fill_value='extrapolate')
        trajectory[:, j] = f(x_monthly)

    # Clip to physiological bounds
    for j, vn in enumerate(VAR_NAMES):
        lo, hi = CORE_20_VARIABLES[vn]['physiological_bounds']
        trajectory[:, j] = trajectory[:, j].clip(lo, hi)

    return trajectory


def _parametric_trajectory(n_months, schedule, demo, cystatin_params):
    """Parametric (non-CircAdapt) trajectory generation.

    Uses physiological relationships to compute clinical variables from
    disease parameter schedules without running the coupled simulator.
    """
    k1_sched = schedule['k1_scale']
    Sf_sched = schedule['Sf_scale']
    Kf_sched = schedule['Kf_scale']
    dm_sched = schedule['diabetes']
    infl_sched = schedule['inflammation']
    RAAS_val = schedule['RAAS_gain'][0]
    TGF_val = schedule['TGF_gain'][0]
    na_val = schedule['na_intake'][0]

    age = demo['age']
    sex = demo['sex']
    BSA = demo['BSA']

    n_vars = len(VAR_NAMES)
    traj = np.empty((n_months, n_vars), dtype=np.float32)

    for t in range(n_months):
        k1 = k1_sched[t]
        Sf = Sf_sched[t]
        Kf = Kf_sched[t]
        d = dm_sched[t]
        i = infl_sched[t]

        # Hemodynamics
        MAP_base = 93 + 5 * (RAAS_val - 1.5) + 3 * d + 2 * i + 0.05 * (na_val - 150)
        CO_base = 4.6 * Sf * (1 - 0.1 * (k1 - 1)) * (1 - 0.05 * i)
        CO = max(CO_base, 1.5)
        SBP = MAP_base * 1.4 * (1 + 0.02 * (k1 - 1))
        MAP = MAP_base
        CVP = 5.0 + 2 * (k1 - 1) + 1 * (1 - Kf) + 0.5 * d
        SVR = (MAP - CVP) / max(CO, 0.5)

        # LV volumes and EF
        EDV = 120 * (1 + 0.05 * (1 - Sf)) * (1 - 0.03 * (k1 - 1))
        ESV = EDV * (1 - Sf * 0.65 * (1 - 0.1 * (k1 - 1)))
        ESV = max(ESV, EDV * 0.15)
        EF = (EDV - ESV) / EDV * 100
        SV = EDV - ESV
        HR = CO * 1000 / max(SV, 20)

        # LV structure
        # Calibrated: 0.895 * EDV^(1/3) matches ARIC V5 mean LVIDd=4.42 at EDV=120
        LVIDd = 0.895 * EDV ** (1/3)
        LVmass = 150 * (1 + 0.15 * (k1 - 1) + 0.1 * d)

        # Systolic function
        GLS = -18 * (EF / 65) * Sf

        # Diastolic function
        E = 68 * (1 + 0.20 * (k1 - 1) + 0.15 * (1 - Kf))
        e_prime = 5.7 / (k1 * (1 + 0.1 * i))
        e_prime = max(e_prime, 1.5)
        E_e_prime = E / e_prime
        E_A = 0.86 / (1 + 0.15 * (k1 - 1))
        if k1 > 2.0:
            E_A = 0.86 + 0.5 * (k1 - 2.0)

        # Atrial / RV
        LAv = 49 * (1 + 0.20 * (k1 - 1) + 0.10 * d)
        PASP = 28 + 5 * (k1 - 1) + 3 * (1 - Kf) + 0.1 * (MAP - 93)

        # Renal
        GFR = 120 * Kf * (1 - 0.1 * d) * (1 - 0.05 * i) * (MAP / 93) ** 0.3
        GFR = max(GFR, 3.0)
        eGFR = GFR * 0.55
        eGFR = max(eGFR, 3.0)
        # Creatinine: inversely proportional to GFR, sex-adjusted
        # Calibrated so male GFR=90 -> Scr~1.0, female GFR=90 -> Scr~0.8
        cr_base = 90.0 if sex == 'M' else 72.0
        creatinine = cr_base / max(GFR, 5.0)
        UACR = max(5.0 * (1 + 2.0 * (1 - Kf)) * (1 + 0.5 * d), 0.5)
        # Cystatin C: inversely proportional to GFR with inflammation correction
        # Small sex correction (females ~5% lower production rate)
        sex_factor = 0.95 if sex == 'F' else 1.0
        cystatin = cystatin_params['D_cys'] / max(GFR, 5.0) * (
            1 + cystatin_params['epsilon'] * i
        ) * sex_factor

        # Biomarkers
        LVEDP_est = 10 * k1
        NTproBNP = 50 * np.exp(
            0.05 * LVEDP_est + 0.02 * CVP + 0.003 * LVmass / BSA
        )
        CRP = 2.0 * np.exp(3.0 * i + 1.5 * d)

        traj[t, VAR_IDX['LVIDd_cm']] = LVIDd
        traj[t, VAR_IDX['LVmass_g']] = LVmass
        traj[t, VAR_IDX['LVEF_pct']] = EF
        traj[t, VAR_IDX['GLS_pct']] = GLS
        traj[t, VAR_IDX['CO_L_min']] = CO
        traj[t, VAR_IDX['E_cm_s']] = E
        traj[t, VAR_IDX['e_prime_sept_cm_s']] = e_prime
        traj[t, VAR_IDX['E_over_e_prime_sept']] = E_e_prime
        traj[t, VAR_IDX['E_over_A_ratio']] = E_A
        traj[t, VAR_IDX['LAvolume_mL']] = LAv
        traj[t, VAR_IDX['PASP_mmHg']] = PASP
        traj[t, VAR_IDX['SBP_mmHg']] = SBP
        traj[t, VAR_IDX['MAP_mmHg']] = MAP
        traj[t, VAR_IDX['SVR_wood']] = SVR
        traj[t, VAR_IDX['eGFR_mL_min']] = eGFR
        traj[t, VAR_IDX['creatinine_mg_dL']] = creatinine
        traj[t, VAR_IDX['UACR_mg_g']] = UACR
        traj[t, VAR_IDX['cystatin_C_mg_L']] = cystatin
        traj[t, VAR_IDX['NTproBNP_pg_mL']] = NTproBNP
        traj[t, VAR_IDX['CRP_mg_L']] = CRP

    # Clip to physiological bounds
    for j, vn in enumerate(VAR_NAMES):
        lo, hi = CORE_20_VARIABLES[vn]['physiological_bounds']
        traj[:, j] = traj[:, j].clip(lo, hi)

    return traj


# ============================================================================
# FUNCTION 7: add_measurement_noise
# ============================================================================

def add_measurement_noise(trajectories, var_names, rng):
    """Add realistic measurement noise and PASP missingness.

    Parameters
    ----------
    trajectories : ndarray, shape (N, T, 20), clean values
    var_names : list of str
    rng : numpy random Generator

    Returns
    -------
    noisy : ndarray, shape (N, T, 20) — noisy copy with PASP NaNs
    """
    N, T, V = trajectories.shape
    noisy = trajectories.copy()

    for j, vn in enumerate(var_names):
        noise_type, magnitude = MEASUREMENT_NOISE[vn]
        if noise_type == 'gaussian_absolute':
            noisy[:, :, j] += rng.normal(0, magnitude, (N, T))
        elif noise_type == 'gaussian_relative':
            noisy[:, :, j] *= rng.normal(1, magnitude, (N, T))

        # Clip to physiological bounds
        lo, hi = CORE_20_VARIABLES[vn]['physiological_bounds']
        noisy[:, :, j] = noisy[:, :, j].clip(lo, hi)

    # PASP missingness model
    pasp_idx = var_names.index('PASP_mmHg')
    pasp_vals = noisy[:, :, pasp_idx]
    p_miss = PASP_MISSING_PARAMS['base_missing_prob'] + \
        PASP_MISSING_PARAMS['pasp_slope'] * (pasp_vals - 25)
    p_miss = p_miss.clip(0.10, 0.60)
    miss_mask = rng.random((N, T)) < p_miss
    noisy[:, :, pasp_idx] = np.where(miss_mask, np.nan, noisy[:, :, pasp_idx])

    return noisy


# ============================================================================
# FUNCTION 8: validate_cystatin_c
# ============================================================================

def validate_cystatin_c(params=None):
    """Run 7 validation checks for the cystatin C emission model.

    Returns True if all checks pass.
    """
    if params is None:
        params = CYSTATIN_C_PARAMS

    D_cys = params["D_cys"]
    eps = params["epsilon"]

    checks = [
        ("Healthy (GFR=100, i=0)",      100, 0.0, 0.60, 1.00),
        ("Mild CKD (GFR=60, i=0)",       60, 0.0, 1.10, 1.50),
        ("Mod CKD (GFR=30, i=0)",        30, 0.0, 2.00, 3.50),
        ("Inflamed (GFR=60, i=0.5)",      60, 0.5, 1.25, 1.80),
        ("Severe inflam (GFR=60, i=1)",   60, 1.0, 1.45, 2.10),
    ]

    print(f"\n{'Check':<35} {'GFR':>5} {'i':>5} {'CysC':>7} "
          f"{'Range':>12} {'Status':>8}")
    print("-" * 75)

    all_pass = True
    for label, gfr, inflam, lo, hi in checks:
        cys_c = D_cys / gfr * (1 + eps * inflam)
        status = "PASS" if lo <= cys_c <= hi else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"{label:<35} {gfr:>5} {inflam:>5.1f} {cys_c:>7.2f} "
              f"[{lo:.2f}-{hi:.2f}] {status:>8}")

    # Cross-check with CKD-EPI 2012 cystatin-only equation
    cys_c_test = D_cys / 60.0  # GFR=60, i=0
    age_test = 75
    ratio = cys_c_test / 0.8
    if ratio <= 1:
        egfr_cys = 133 * (ratio ** -0.499) * (0.996 ** age_test)
    else:
        egfr_cys = 133 * (ratio ** -1.328) * (0.996 ** age_test)
    pct_error = abs(egfr_cys - 60) / 60 * 100
    status = "PASS" if pct_error < 30 else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"\nCKD-EPI cross-check: CysC={cys_c_test:.2f} -> "
          f"eGFR_cys={egfr_cys:.1f} (true=60, err={pct_error:.1f}%) {status}")

    # Population check: ARIC V5 mean GFR=66, mean inflammation=0.12
    pop_cys = D_cys / 66.0 * (1 + eps * 0.12)
    status = "PASS" if 0.90 <= pop_cys <= 1.30 else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"Pop-level check: GFR=66, i=0.12 -> CysC={pop_cys:.2f} "
          f"(target: 1.10+/-0.20) {status}")

    return all_pass


# ============================================================================
# FUNCTION 9: validate_marginals
# ============================================================================

def validate_marginals(trajectories, var_names):
    """Compare month-0 distributions to ARIC V5 targets.

    Prints table with deviation in units of V5 SD.
    """
    print(f"\n{'Variable':<25} {'Synth Mean':>10} {'Synth SD':>10} "
          f"{'V5 Mean':>10} {'V5 SD':>10} {'Dev (SDs)':>10} {'Status':>8}")
    print("-" * 85)

    n_warn = 0
    n_fail = 0
    for j, vn in enumerate(var_names):
        meta = CORE_20_VARIABLES[vn]
        synth_vals = trajectories[:, 0, j]
        valid = synth_vals[~np.isnan(synth_vals)]
        if len(valid) == 0:
            continue
        synth_mean = np.mean(valid)
        synth_sd = np.std(valid)
        v5_mean = meta['v5_mean']
        v5_sd = meta['v5_sd']
        dev = abs(synth_mean - v5_mean) / v5_sd if v5_sd > 0 else 0

        if dev > 2.0:
            status = "FAIL"
            n_fail += 1
        elif dev > 1.0:
            status = "WARN"
            n_warn += 1
        else:
            status = "OK"

        print(f"{vn:<25} {synth_mean:>10.2f} {synth_sd:>10.2f} "
              f"{v5_mean:>10.2f} {v5_sd:>10.2f} {dev:>10.2f} {status:>8}")

    print(f"\nSummary: {n_fail} FAIL, {n_warn} WARN, "
          f"{len(var_names) - n_fail - n_warn} OK")


# ============================================================================
# FUNCTION 10: validate_trajectories
# ============================================================================

def validate_trajectories(trajectories, var_names, labels):
    """Plausibility checks on generated trajectories.

    Returns True if all critical checks pass.
    """
    all_pass = True

    print("\n--- Trajectory Plausibility Checks ---")

    ef_idx = var_names.index('LVEF_pct')
    egfr_idx = var_names.index('eGFR_mL_min')
    ee_idx = var_names.index('E_over_e_prime_sept')
    bnp_idx = var_names.index('NTproBNP_pg_mL')
    cys_idx = var_names.index('cystatin_C_mg_L')
    cr_idx = var_names.index('creatinine_mg_dL')

    # 1. HFpEF patients: LVEF stays >50% for >85% of trajectory
    hfpef_mask = labels == 'hfpef_dominant'
    if hfpef_mask.sum() > 0:
        ef_vals = trajectories[hfpef_mask, :, ef_idx]
        pct_above_50 = (ef_vals > 50).mean(axis=1)
        pass_rate = (pct_above_50 > 0.85).mean()
        status = "PASS" if pass_rate > 0.70 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"1. HFpEF LVEF>50% for >85% of traj: {pass_rate:.1%} pass [{status}]")
    else:
        print("1. HFpEF check: no hfpef_dominant patients [SKIP]")

    # 2. CKD patients: eGFR declines in >80% of patients
    ckd_mask = (labels == 'ckd_dominant') | (labels == 'cardiorenal')
    if ckd_mask.sum() > 0:
        egfr_start = trajectories[ckd_mask, 0, egfr_idx]
        egfr_end = trajectories[ckd_mask, -1, egfr_idx]
        decline_rate = (egfr_end < egfr_start).mean()
        status = "PASS" if decline_rate > 0.80 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"2. CKD eGFR decline: {decline_rate:.1%} decline [{status}]")
    else:
        print("2. CKD eGFR decline: no CKD patients [SKIP]")

    # 3. Cardiorenal: E/e' increases in >70% of patients
    cr_mask = labels == 'cardiorenal'
    if cr_mask.sum() > 0:
        ee_start = trajectories[cr_mask, 0, ee_idx]
        ee_end = trajectories[cr_mask, -1, ee_idx]
        increase_rate = (ee_end > ee_start).mean()
        status = "PASS" if increase_rate > 0.70 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"3. Cardiorenal E/e' increase: {increase_rate:.1%} increase [{status}]")
    else:
        print("3. Cardiorenal E/e' increase: no cardiorenal patients [SKIP]")

    # 4. log(NT-proBNP) <-> E/e' correlation > 0.3
    for t_check in [0, trajectories.shape[1] // 2, trajectories.shape[1] - 1]:
        bnp_vals = trajectories[:, t_check, bnp_idx]
        ee_vals = trajectories[:, t_check, ee_idx]
        valid = ~(np.isnan(bnp_vals) | np.isnan(ee_vals)) & (bnp_vals > 0)
        if valid.sum() > 10:
            corr = np.corrcoef(np.log(bnp_vals[valid]), ee_vals[valid])[0, 1]
            status = "PASS" if corr > 0.3 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"4. log(BNP) vs E/e' corr at t={t_check}: {corr:.3f} [{status}]")

    # 5. Cystatin C <-> creatinine correlation > 0.7
    for t_check in [0, trajectories.shape[1] - 1]:
        cys_vals = trajectories[:, t_check, cys_idx]
        cr_vals = trajectories[:, t_check, cr_idx]
        valid = ~(np.isnan(cys_vals) | np.isnan(cr_vals))
        if valid.sum() > 10:
            corr = np.corrcoef(cys_vals[valid], cr_vals[valid])[0, 1]
            status = "PASS" if corr > 0.7 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"5. CysC vs creatinine corr at t={t_check}: {corr:.3f} [{status}]")

    # 6. eGFR <-> creatinine correlation < -0.5
    for t_check in [0, trajectories.shape[1] - 1]:
        egfr_vals = trajectories[:, t_check, egfr_idx]
        cr_vals = trajectories[:, t_check, cr_idx]
        valid = ~(np.isnan(egfr_vals) | np.isnan(cr_vals))
        if valid.sum() > 10:
            corr = np.corrcoef(egfr_vals[valid], cr_vals[valid])[0, 1]
            status = "PASS" if corr < -0.5 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"6. eGFR vs creatinine corr at t={t_check}: {corr:.3f} [{status}]")

    # 7. No patients with eGFR<10 AND LVEF>70%
    bad = (trajectories[:, :, egfr_idx] < 10) & (trajectories[:, :, ef_idx] > 70)
    n_bad = bad.any(axis=1).sum()
    status = "PASS" if n_bad == 0 else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"7. eGFR<10 & LVEF>70% implausible combos: {n_bad} patients [{status}]")

    return all_pass


# ============================================================================
# FUNCTION 11: generate_cohort (main orchestrator)
# ============================================================================

def _worker_fn(args):
    """Worker function for multiprocessing patient generation."""
    idx, demo_dict, schedule_dict, n_months, var_names, cystatin_params, use_circadapt = args
    return generate_single_patient_trajectory(
        idx, demo_dict, schedule_dict, n_months, var_names,
        cystatin_params, use_circadapt,
    )


def generate_cohort(n_patients, n_months, n_workers, seed, validate,
                    output_path='cohort_monthly.npz'):
    """Main orchestrator for synthetic cohort generation.

    Steps:
      1. Sample demographics
      2. Sample disease parameters + correlations
      3. Generate progression schedules
      4. Assign phenotype labels
      5. Generate per-patient trajectories (parallel)
      6. Add measurement noise
      7. Validate (optional)
      8. Save to .npz
    """
    rng = np.random.default_rng(seed)
    t_start = time.time()

    # --- Step 1: Demographics ---
    print(f"[1/8] Sampling demographics for {n_patients} patients...")
    demographics = sample_demographics(n_patients, rng)

    # --- Step 2: Disease parameters + correlations ---
    print("[2/8] Sampling disease parameters...")
    params = sample_disease_parameters(n_patients, rng)
    params = apply_disease_correlations(params, rng)

    # --- Step 3: Progression schedules ---
    print(f"[3/8] Generating {n_months}-month progression schedules...")
    schedules = generate_progression_schedule(params, n_months, rng)

    # --- Step 4: Phenotype labels ---
    print("[4/8] Assigning phenotype labels...")
    labels = assign_phenotype_labels(params)
    unique, counts = np.unique(labels, return_counts=True)
    for lab, cnt in zip(unique, counts):
        print(f"  {lab}: {cnt} ({cnt/n_patients*100:.1f}%)")

    # --- Step 5: Check if CircAdapt is available and stable ---
    use_circadapt = False
    try:
        from cardiorenal_coupling import run_coupled_simulation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Test with small dt_renal_hours (6h) to avoid divergence
            test_hist = run_coupled_simulation(n_steps=2, dt_renal_hours=6.0)
        # Check for NaN — CircAdapt diverges with large renal time steps
        if any(np.isnan(v) for v in test_hist.get('MAP', [0])):
            raise RuntimeError("CircAdapt produces NaN — renal model divergence")
        use_circadapt = True
        print("[5/8] CircAdapt available — using coupled simulator with interpolation")
    except Exception as e:
        print(f"[5/8] CircAdapt unavailable ({e}) — using parametric fallback")

    # --- Step 6: Generate trajectories ---
    print(f"[6/8] Generating trajectories ({n_workers} workers)...")

    worker_args = []
    for i in range(n_patients):
        demo_i = {k: v[i] for k, v in demographics.items()}
        sched_i = {k: v[i] for k, v in schedules.items()}
        worker_args.append((
            i, demo_i, sched_i, n_months, VAR_NAMES,
            CYSTATIN_C_PARAMS, use_circadapt,
        ))

    if n_workers <= 1:
        results = []
        for i, args in enumerate(worker_args):
            results.append(_worker_fn(args))
            if (i + 1) % max(1, n_patients // 10) == 0:
                print(f"  {i+1}/{n_patients} patients done "
                      f"({(i+1)/n_patients*100:.0f}%)")
    else:
        with mp.Pool(n_workers) as pool:
            results = []
            for i, result in enumerate(pool.imap(
                _worker_fn, worker_args,
                chunksize=max(1, n_patients // (n_workers * 4))
            )):
                results.append(result)
                if (i + 1) % max(1, n_patients // 10) == 0:
                    print(f"  {i+1}/{n_patients} patients done "
                          f"({(i+1)/n_patients*100:.0f}%)")

    trajectories_clean = np.stack(results, axis=0)  # (N, T, 20)
    print(f"  Clean trajectories shape: {trajectories_clean.shape}")

    # --- Step 7: Add measurement noise ---
    print("[7/8] Adding measurement noise...")
    trajectories_noisy = add_measurement_noise(trajectories_clean, VAR_NAMES, rng)

    pasp_idx = VAR_NAMES.index('PASP_mmHg')
    non_pasp_nans = np.isnan(trajectories_noisy[:, :, :pasp_idx]).sum() + \
                    np.isnan(trajectories_noisy[:, :, pasp_idx+1:]).sum()
    pasp_nans = np.isnan(trajectories_noisy[:, :, pasp_idx]).sum()
    print(f"  PASP NaN count: {pasp_nans} "
          f"({pasp_nans / (n_patients * n_months) * 100:.1f}%)")
    if non_pasp_nans > 0:
        print(f"  WARNING: {non_pasp_nans} unexpected NaNs in non-PASP variables!")

    # --- Step 8: Validate and save ---
    if validate:
        print("[8/8] Running validation...")
        cysc_ok = validate_cystatin_c()
        validate_marginals(trajectories_noisy, VAR_NAMES)
        traj_ok = validate_trajectories(trajectories_clean, VAR_NAMES, labels)
        if cysc_ok and traj_ok:
            print("\nAll validations PASSED.")
        else:
            print("\nSome validations FAILED — review output above.")
    else:
        print("[8/8] Skipping validation (use --validate to enable)")

    # Save
    print(f"\nSaving to {output_path}...")
    np.savez_compressed(
        output_path,
        trajectories=trajectories_noisy.astype(np.float32),
        trajectories_clean=trajectories_clean.astype(np.float32),
        var_names=np.array(VAR_NAMES),
        demographics_age=demographics['age'],
        demographics_sex=demographics['sex'],
        demographics_BSA=demographics['BSA'],
        demographics_height_m=demographics['height_m'],
        demographics_race=demographics['race'],
        disease_k1_scale=schedules['k1_scale'].astype(np.float32),
        disease_Sf_scale=schedules['Sf_scale'].astype(np.float32),
        disease_Kf_scale=schedules['Kf_scale'].astype(np.float32),
        disease_diabetes=schedules['diabetes'].astype(np.float32),
        disease_inflammation=schedules['inflammation'].astype(np.float32),
        disease_RAAS_gain=schedules['RAAS_gain'].astype(np.float32),
        disease_TGF_gain=schedules['TGF_gain'].astype(np.float32),
        disease_na_intake=schedules['na_intake'].astype(np.float32),
        phenotype_labels=labels,
        metadata=np.array([{
            'n_patients': n_patients,
            'n_months': n_months,
            'seed': seed,
            'use_circadapt': use_circadapt,
            'generation_time_s': time.time() - t_start,
        }]),
    )

    elapsed = time.time() - t_start
    print(f"Done. {n_patients} patients x {n_months} months x 20 vars "
          f"in {elapsed:.1f}s")
    print(f"Output: {output_path} "
          f"({trajectories_noisy.nbytes / 1e6:.1f} MB uncompressed)")

    return trajectories_noisy, trajectories_clean


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate monthly synthetic cardiorenal cohort')
    parser.add_argument('--n_patients', type=int, default=5000)
    parser.add_argument('--n_months', type=int, default=96,
                        help='Number of months (default 96 = 8 years)')
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validate', action='store_true',
                        help='Run all validation checks after generation')
    parser.add_argument('--output', default='cohort_monthly.npz')
    args = parser.parse_args()

    generate_cohort(
        n_patients=args.n_patients,
        n_months=args.n_months,
        n_workers=args.n_workers,
        seed=args.seed,
        validate=args.validate,
        output_path=args.output,
    )
