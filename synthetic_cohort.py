#!/usr/bin/env python3
"""
Synthetic Cohort Generator
==========================
Paper Reference: Section 3.7 — Synthetic Cohort Generation

Two generation modes:

1. **Paired (V5→V7)**: Generates paired (Visit 5, Visit 7) clinical vectors
   using the full CircAdapt + Hallow coupled model. Output: cohort_data.npz
   with ~100 ARIC variables per visit. Used to train the residual MLP (Section 3.8).

2. **Monthly**: Generates monthly-resolution trajectories (N, T, 20) of 20
   core clinical variables. Output: cohort_monthly.npz. Used for RL warm-start
   and pipeline testing.

Usage:
    # Paired V5→V7 cohort for NN training
    python synthetic_cohort.py --mode paired --n_patients 10000 --n_workers 8

    # Monthly trajectories for RL / testing
    python synthetic_cohort.py --mode monthly --n_patients 5000 --n_months 96

    # Quick test (either mode)
    python synthetic_cohort.py --mode paired --n_patients 100 --n_workers 1
"""

import os
import sys
import argparse
import time
import warnings
import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
from scipy import stats
from scipy.interpolate import interp1d
from typing import Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim_logging import sim_logger, extract_key_outputs
from config import (
    TUNABLE_PARAMS, NUMERIC_VAR_NAMES, NON_NUMERIC_VARS, COHORT_DEFAULTS,
    CORE_20_VARIABLES, MEASUREMENT_NOISE, CYSTATIN_C_PARAMS, PASP_MISSING_PARAMS,
)

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# Constants: 20 core clinical variables (monthly trajectories)
# ═══════════════════════════════════════════════════════════════════════════════

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

VAR_IDX = {name: i for i, name in enumerate(VAR_NAMES)}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Stabilized Renal Model
# ═══════════════════════════════════════════════════════════════════════════════
# Hallow et al. (2017) renal model with heavier TGF damping (0.85/0.15)
# calibrated for CircAdapt's lower MAP (~86 mmHg).

def _update_renal_stable(r, MAP, CO, Pven, dt_hours=6.0):
    """One timestep of the Hallow renal model with improved TGF damping.

    Computes GFR, RBF, P_glom, Na/water excretion, and volume balance.
    Heavy damping (0.85 old + 0.15 new) prevents TGF oscillation at
    CircAdapt's lower MAP range.
    """
    Kf_eff = r['Kf'] * r['Kf_scale']
    r['P_renal_vein'] = max(Pven, 2.0)

    # RAAS response to MAP deviation from setpoint
    dMAP = MAP - r['MAP_setpoint']
    RAAS_factor = float(np.clip(1.0 - r['RAAS_gain'] * 0.005 * dMAP, 0.5, 2.0))
    R_EA = r['R_EA0'] * RAAS_factor
    eta_CD = r['eta_CD0'] * RAAS_factor

    # TGF iteration (20 steps with heavy damping)
    R_AA = r['R_AA0']
    GFR = r.get('GFR', 120.0)
    Na_filt = 0.0
    P_gc = 60.0
    RBF = 1100.0

    for _ in range(20):
        R_total = r['R_preAA'] + R_AA + R_EA
        RBF = max((MAP - r['P_renal_vein']) / R_total * 1000.0, 100.0)
        RPF = RBF * (1.0 - r['Hct'])
        P_gc = max(MAP - RBF / 1000.0 * (r['R_preAA'] + R_AA), 25.0)
        FF = float(np.clip(GFR / max(RPF, 1.0), 0.01, 0.45))
        pi_avg = r['pi_plasma'] * (1.0 + FF / (2.0 * (1.0 - FF)))
        NFP = max(P_gc - r['P_Bow'] - pi_avg, 0.0)
        SNGFR = Kf_eff * NFP
        GFR = max(2.0 * r['N_nephrons'] * SNGFR * 1e-6, 5.0)
        FF = float(np.clip(GFR / max(RPF, 1.0), 0.01, 0.45))
        Na_filt = GFR * r['C_Na'] * 1e-3
        MD_Na = Na_filt * (1.0 - r['eta_PT']) * (1.0 - r['eta_LoH'])
        if r['TGF_setpoint'] <= 0:
            r['TGF_setpoint'] = MD_Na
        TGF_err = (MD_Na - r['TGF_setpoint']) / max(r['TGF_setpoint'], 1e-6)
        R_AA_new = r['R_AA0'] * (1.0 + r['TGF_gain'] * TGF_err)
        R_AA_new = float(np.clip(R_AA_new, 0.5 * r['R_AA0'], 3.0 * r['R_AA0']))
        R_AA = 0.85 * R_AA + 0.15 * R_AA_new

    # Tubular Na handling
    Na_after_PT = Na_filt * (1.0 - r['eta_PT'])
    Na_after_LoH = Na_after_PT * (1.0 - r['eta_LoH'])
    Na_after_DT = Na_after_LoH * (1.0 - r['eta_DT'])
    Na_after_CD = Na_after_DT * (1.0 - eta_CD)

    if MAP > r['MAP_setpoint']:
        pn = 1.0 + 0.03 * (MAP - r['MAP_setpoint'])
    else:
        pn = max(0.3, 1.0 + 0.015 * (MAP - r['MAP_setpoint']))

    Na_excr_min = Na_after_CD * pn
    Na_excr_day = Na_excr_min * 1440.0

    # Water excretion
    water_excr_min = GFR * (1.0 - r['frac_water_reabs'])
    water_excr_day = water_excr_min * 1440.0 / 1000.0

    # Volume / Na balance (Euler integration)
    dt_min = dt_hours * 60.0
    Na_in_min = r['Na_intake'] / 1440.0
    r['Na_total'] = max(r['Na_total'] + (Na_in_min - Na_excr_min) * dt_min, 800.0)

    W_in_min = r['water_intake'] * 1000.0 / 1440.0
    dV = (W_in_min - water_excr_min) * dt_min
    r['V_blood'] = float(np.clip(r['V_blood'] + dV * 0.33, 3000.0, 8000.0))

    V_ECF = r['V_blood'] / 0.33
    r['C_Na'] = float(np.clip(r['Na_total'] / (V_ECF * 1e-3), 125.0, 155.0))

    r['GFR'] = round(float(GFR), 1)
    r['RBF'] = round(float(RBF), 1)
    r['P_glom'] = round(float(P_gc), 1)
    r['Na_excretion'] = round(float(Na_excr_day), 1)
    r['water_excretion'] = round(float(water_excr_day), 2)


def _create_renal_state_circadapt(na_intake=150.0, raas_gain=1.5, tgf_gain=2.0, kf_scale=1.0):
    """Create renal state dict calibrated for CircAdapt's MAP (~86 mmHg)."""
    return {
        'N_nephrons': 1e6, 'Kf': 8.0,
        'R_preAA': 9.5, 'R_AA0': 20.5, 'R_EA0': 43.0,
        'P_Bow': 18.0, 'P_renal_vein': 4.0,
        'pi_plasma': 25.0, 'Hct': 0.45,
        'eta_PT': 0.67, 'eta_LoH': 0.25, 'eta_DT': 0.05,
        'eta_CD0': 0.024, 'frac_water_reabs': 0.99,
        'Na_intake': na_intake, 'water_intake': 2.0,
        'TGF_gain': tgf_gain, 'TGF_setpoint': 0.0,
        'RAAS_gain': raas_gain, 'MAP_setpoint': 86.0,
        'V_blood': 5000.0, 'Na_total': 2100.0, 'C_Na': 140.0,
        'GFR': 120.0, 'RBF': 1100.0, 'P_glom': 60.0,
        'Na_excretion': 150.0, 'water_excretion': 1.5,
        'Kf_scale': kf_scale,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Core Model Evaluation (used by agent_tools, pipeline, agent_loop)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_patient_state(
    params: Dict,
    demographics: Dict,
    n_coupling_steps: int = 2,
    dt_renal_hours: float = 6.0,
) -> Optional[Dict]:
    """Run CircAdapt heart + Hallow renal coupled model, extract ~113 ARIC variables.

    This is the core forward model: disease parameters → clinical measurements.

    Parameters
    ----------
    params : dict
        Disease parameters (Sf_act_scale, Kf_scale, inflammation_scale,
        diabetes_scale, k1_scale, RAAS_gain, TGF_gain, na_intake).
    demographics : dict
        Patient demographics (age, sex, BSA, height_m).

    Returns
    -------
    dict or None
        ~113 ARIC variable names → float values, or None on failure.
    """
    from cardiorenal_coupling import (
        CircAdaptHeartModel, InflammatoryState,
        update_inflammatory_state, ML_TO_M3,
    )
    from emission_functions import extract_all_aric_variables

    try:
        heart = CircAdaptHeartModel()
        ist = InflammatoryState()

        renal = _create_renal_state_circadapt(
            na_intake=params.get('na_intake', 150.0),
            raas_gain=params.get('RAAS_gain', 1.5),
            tgf_gain=params.get('TGF_gain', 2.0),
            kf_scale=params.get('Kf_scale', 1.0),
        )

        ist = update_inflammatory_state(
            ist,
            inflammation_scale=params.get('inflammation_scale', 0.0),
            diabetes_scale=params.get('diabetes_scale', 0.0),
        )

        heart.apply_inflammatory_modifiers(ist)

        effective_k1 = params.get('k1_scale', 1.0) * ist.passive_k1_factor
        heart.apply_stiffness(effective_k1)

        effective_sf = max(
            params.get('Sf_act_scale', 1.0) * ist.Sf_act_factor, 0.20
        )
        heart.apply_deterioration(effective_sf)

        hemo = heart.run_to_steady_state()

        # Equilibrate renal model (5 passes, hold volume/Na fixed)
        for _ in range(5):
            renal['C_Na'] = 140.0
            renal['TGF_setpoint'] = 0.0
            _update_renal_stable(renal, hemo['MAP'], hemo['CO'], hemo['Pven'], dt_renal_hours)
            renal['V_blood'] = 5000.0
            renal['Na_total'] = 2100.0
            renal['C_Na'] = 140.0

        renal_state = {
            'GFR': renal['GFR'],
            'V_blood': 5000.0,
            'C_Na': 140.0,
            'Na_excretion': params.get('na_intake', 150.0),
            'P_glom': renal['P_glom'],
            'Kf_scale': renal['Kf_scale'],
            'RBF': renal['RBF'],
        }

        aric_vars = extract_all_aric_variables(
            heart.model, renal_state,
            BSA=demographics.get('BSA', 1.9),
            height_m=demographics.get('height_m', 1.70),
            age=demographics.get('age', 75.0),
            sex=demographics.get('sex', 'M'),
        )

        sim_logger.log_run(
            params=params,
            outputs=extract_key_outputs(hemo, renal_state),
            success=True,
            source='evaluate_patient_state',
            demographics=demographics,
        )

        return aric_vars

    except Exception as e:
        sim_logger.log_run(
            params=params,
            success=False,
            error=str(e),
            source='evaluate_patient_state',
            demographics=demographics,
        )
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: V5→V7 Paired Cohort Generation
# ═══════════════════════════════════════════════════════════════════════════════

# Correlation matrix for disease progression deltas (V5→V7, ~6 years).
# Order: delta_Sf_act, delta_Kf, delta_inflammation, delta_diabetes,
#         delta_RAAS, delta_na_intake
_DELTA_CORR = np.array([
    [ 1.0, 0.4, -0.2, -0.1,  0.0,  0.0],   # Sf_act
    [ 0.4, 1.0, -0.3, -0.2, -0.1,  0.0],   # Kf
    [-0.2,-0.3,  1.0,  0.3,  0.2,  0.0],   # inflammation
    [-0.1,-0.2,  0.3,  1.0,  0.1,  0.0],   # diabetes
    [ 0.0,-0.1,  0.2,  0.1,  1.0,  0.1],   # RAAS
    [ 0.0, 0.0,  0.0,  0.0,  0.1,  1.0],   # Na intake
])
_DELTA_CHOL = np.linalg.cholesky(_DELTA_CORR)


def sample_correlated_deltas(rng: np.random.Generator) -> Dict:
    """Sample correlated disease progression deltas for V5→V7 (~6 years)."""
    z = rng.standard_normal(6)
    corr_z = _DELTA_CHOL @ z

    return {
        'delta_Sf_act': -abs(corr_z[0]) * 0.08,
        'delta_Kf': -abs(corr_z[1]) * 0.10,
        'delta_inflammation': abs(corr_z[2]) * 0.08,
        'delta_diabetes': abs(corr_z[3]) * 0.05,
        'delta_RAAS': corr_z[4] * 0.15,
        'delta_na': corr_z[5] * 15.0,
    }


def generate_patient_params(rng: np.random.Generator) -> Dict:
    """Generate one patient's V5 and V7 parameter sets plus demographics."""
    age_v5 = rng.uniform(65, 85)
    sex = 'M' if rng.random() < 0.5 else 'F'
    BSA = float(np.clip(rng.normal(1.9, 0.15), 1.4, 2.5))
    height_m = float(np.clip(rng.normal(1.70, 0.08), 1.50, 1.95))

    demographics_v5 = {'age': age_v5, 'sex': sex, 'BSA': BSA, 'height_m': height_m}
    demographics_v7 = {'age': age_v5 + 6.0, 'sex': sex, 'BSA': BSA, 'height_m': height_m}

    # V5 baseline disease parameters
    Sf_act_v5 = float(np.clip(rng.normal(0.92, 0.12), 0.35, 1.0))
    Kf_v5 = float(np.clip(rng.beta(5, 2), 0.3, 1.0))
    inflammation_v5 = float(np.clip(rng.exponential(0.10), 0.0, 0.6))
    diabetes_v5 = float(rng.choice(
        [0.0, 0.0, 0.0, np.clip(rng.uniform(0.1, 0.7), 0.0, 1.0)]
    ))
    RAAS_v5 = float(np.clip(rng.normal(1.5, 0.3), 0.8, 2.5))
    TGF_v5 = float(np.clip(rng.normal(2.0, 0.3), 1.0, 3.5))
    na_v5 = float(np.clip(rng.normal(150, 40), 50, 300))
    k1_v5 = 1.0 if diabetes_v5 == 0.0 else float(np.clip(1.0 + diabetes_v5 * 0.5, 1.0, 2.0))

    v5_params = {
        'Sf_act_scale': Sf_act_v5, 'Kf_scale': Kf_v5,
        'inflammation_scale': inflammation_v5, 'diabetes_scale': diabetes_v5,
        'k1_scale': k1_v5, 'RAAS_gain': RAAS_v5, 'TGF_gain': TGF_v5,
        'na_intake': na_v5,
    }

    deltas = sample_correlated_deltas(rng)
    delta_k1 = abs(deltas['delta_diabetes']) * 0.3

    v7_params = {
        'Sf_act_scale': float(np.clip(Sf_act_v5 + deltas['delta_Sf_act'],
                                       *TUNABLE_PARAMS['Sf_act_scale']['range'])),
        'Kf_scale': float(np.clip(Kf_v5 + deltas['delta_Kf'],
                                   *TUNABLE_PARAMS['Kf_scale']['range'])),
        'inflammation_scale': float(np.clip(inflammation_v5 + deltas['delta_inflammation'],
                                            *TUNABLE_PARAMS['inflammation_scale']['range'])),
        'diabetes_scale': float(np.clip(diabetes_v5 + deltas['delta_diabetes'],
                                        *TUNABLE_PARAMS['diabetes_scale']['range'])),
        'k1_scale': float(np.clip(k1_v5 + delta_k1,
                                   *TUNABLE_PARAMS['k1_scale']['range'])),
        'RAAS_gain': float(np.clip(RAAS_v5 + deltas['delta_RAAS'],
                                    *TUNABLE_PARAMS['RAAS_gain']['range'])),
        'TGF_gain': TGF_v5,
        'na_intake': float(np.clip(na_v5 + deltas['delta_na'],
                                    *TUNABLE_PARAMS['na_intake']['range'])),
    }

    return {
        'v5_params': v5_params, 'v7_params': v7_params,
        'demographics_v5': demographics_v5, 'demographics_v7': demographics_v7,
    }


def _process_patient(args: Tuple) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
    """Worker: generate one patient's V5 and V7 ARIC variable vectors.

    Wrapped in try/except so that any unexpected Python-level error
    returns None instead of killing the multiprocessing pool.
    C-level crashes (e.g., CircAdapt segfault) are handled at the
    pool level in generate_paired_cohort().
    """
    try:
        patient_idx, seed = args
        rng = np.random.default_rng(seed)
        patient = generate_patient_params(rng)

        v5_vars = evaluate_patient_state(
            patient['v5_params'], patient['demographics_v5'],
            n_coupling_steps=COHORT_DEFAULTS['n_coupling_steps'],
            dt_renal_hours=COHORT_DEFAULTS['dt_renal_hours'],
        )
        if v5_vars is None:
            return None

        v7_vars = evaluate_patient_state(
            patient['v7_params'], patient['demographics_v7'],
            n_coupling_steps=COHORT_DEFAULTS['n_coupling_steps'],
            dt_renal_hours=COHORT_DEFAULTS['dt_renal_hours'],
        )
        if v7_vars is None:
            return None

        v5_vec = np.array([float(v5_vars.get(k, 0.0)) for k in NUMERIC_VAR_NAMES])
        v7_vec = np.array([float(v7_vars.get(k, 0.0)) for k in NUMERIC_VAR_NAMES])

        if np.any(~np.isfinite(v5_vec)) or np.any(~np.isfinite(v7_vec)):
            return None

        return (v5_vec, v7_vec, patient)

    except Exception:
        return None


def generate_paired_cohort(
    n_patients: int = 10000,
    seed: int = 42,
    n_workers: int = 8,
) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """Generate paired V5/V7 ARIC variable vectors for NN training.

    Returns
    -------
    v5_array : (N_valid, N_features)
    v7_array : (N_valid, N_features)
    var_names : list of str
    patient_metadata : list of dicts
    """
    print(f"Generating {n_patients} synthetic patients (seed={seed}, workers={n_workers})...")

    rng_master = np.random.default_rng(seed)
    seeds = rng_master.integers(0, 2**31, size=n_patients)
    args_list = [(i, int(seeds[i])) for i in range(n_patients)]

    t0 = time.time()

    if n_workers <= 1:
        results = []
        for i, args in enumerate(args_list):
            result = _process_patient(args)
            results.append(result)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_patients - i - 1) / rate
                print(f"  [{i+1}/{n_patients}] {rate:.1f} pts/s, ETA {eta:.0f}s")
    else:
        with Pool(n_workers) as pool:
            results = []
            for i, result in enumerate(pool.imap_unordered(_process_patient, args_list, chunksize=10)):
                results.append(result)
                if (i + 1) % 200 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (n_patients - i - 1) / rate
                    print(f"  [{i+1}/{n_patients}] {rate:.1f} pts/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0

    valid_results = [r for r in results if r is not None]
    n_valid = len(valid_results)
    n_failed = n_patients - n_valid
    print(f"Done in {elapsed:.1f}s. Valid: {n_valid}/{n_patients} ({n_failed} failed)")

    v5_list, v7_list, meta_list = [], [], []
    for v5_vec, v7_vec, patient in valid_results:
        v5_list.append(v5_vec)
        v7_list.append(v7_vec)
        meta_list.append(patient)

    v5_array = np.stack(v5_list)
    v7_array = np.stack(v7_list)

    return v5_array, v7_array, NUMERIC_VAR_NAMES, meta_list


def load_real_aric_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """Load real ARIC V5/V7 paired data from CSV. Columns: v5_<var>, v7_<var>."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    v5_cols = [f'v5_{k}' for k in NUMERIC_VAR_NAMES]
    v7_cols = [f'v7_{k}' for k in NUMERIC_VAR_NAMES]

    missing_v5 = [c for c in v5_cols if c not in df.columns]
    missing_v7 = [c for c in v7_cols if c not in df.columns]
    if missing_v5 or missing_v7:
        available = [c for c in v5_cols if c in df.columns]
        print(f"Warning: {len(missing_v5)} V5 cols and {len(missing_v7)} V7 cols missing. "
              f"Using {len(available)} available columns, filling rest with 0.")

    v5_array = np.zeros((len(df), len(NUMERIC_VAR_NAMES)))
    v7_array = np.zeros((len(df), len(NUMERIC_VAR_NAMES)))

    for i, var_name in enumerate(NUMERIC_VAR_NAMES):
        v5_col = f'v5_{var_name}'
        v7_col = f'v7_{var_name}'
        if v5_col in df.columns:
            v5_array[:, i] = df[v5_col].values
        if v7_col in df.columns:
            v7_array[:, i] = df[v7_col].values

    return v5_array, v7_array, NUMERIC_VAR_NAMES, []


# Backward-compatible alias: old code calls generate_cohort() for paired mode
generate_cohort = generate_paired_cohort


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Monthly Cohort Generation
# ═══════════════════════════════════════════════════════════════════════════════

def sample_demographics(n_patients, rng):
    """Sample ARIC V5-like demographics (vectorized).

    Returns dict with keys: age, sex, BSA, height_m, race.
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

    return {'age': age, 'sex': sex, 'BSA': BSA, 'height_m': height_m, 'race': race}


def sample_disease_parameters(n_patients, rng):
    """Sample baseline disease parameters (vectorized)."""
    k1_scale = np.empty(n_patients)
    healthy_mask = rng.random(n_patients) < 0.40
    k1_scale[healthy_mask] = rng.uniform(1.0, 1.1, healthy_mask.sum())
    k1_scale[~healthy_mask] = rng.uniform(1.0, 2.5, (~healthy_mask).sum())

    Sf_scale = rng.normal(0.95, 0.08, n_patients).clip(0.5, 1.0)

    Kf_scale = np.empty(n_patients)
    kf_healthy = rng.random(n_patients) < 0.50
    Kf_scale[kf_healthy] = rng.uniform(0.85, 1.0, kf_healthy.sum())
    Kf_scale[~kf_healthy] = rng.beta(3, 2, (~kf_healthy).sum()) * 0.7 + 0.3

    diabetes = np.empty(n_patients)
    dm_mask = rng.random(n_patients) < 0.35
    diabetes[dm_mask] = rng.beta(2, 3, dm_mask.sum())
    diabetes[~dm_mask] = rng.uniform(0, 0.05, (~dm_mask).sum())

    inflammation = rng.exponential(0.12, n_patients).clip(0, 0.8)
    RAAS_gain = rng.normal(1.5, 0.3, n_patients).clip(0.5, 3.0)
    TGF_gain = rng.normal(2.0, 0.4, n_patients).clip(1.0, 4.0)
    na_intake = rng.normal(150, 30, n_patients).clip(80, 250)

    return {
        'k1_scale': k1_scale, 'Sf_scale': Sf_scale, 'Kf_scale': Kf_scale,
        'diabetes': diabetes, 'inflammation': inflammation,
        'RAAS_gain': RAAS_gain, 'TGF_gain': TGF_gain, 'na_intake': na_intake,
    }


def apply_disease_correlations(params, rng):
    """Iman-Conover rank-based correlation induction for [d, k1, Kf, i]."""
    keys = ['diabetes', 'k1_scale', 'Kf_scale', 'inflammation']
    n = len(params[keys[0]])

    target_corr = np.array([
        [1.00,  0.35, -0.40,  0.30],
        [0.35,  1.00, -0.25,  0.20],
        [-0.40, -0.25, 1.00, -0.15],
        [0.30,  0.20, -0.15,  1.00],
    ])

    L = np.linalg.cholesky(target_corr)

    data = np.column_stack([params[k] for k in keys])
    ranks = np.empty_like(data)
    for j in range(4):
        ranks[:, j] = stats.rankdata(data[:, j])

    U = (ranks - 0.5) / n
    Z = stats.norm.ppf(U)
    Z_corr = (L @ Z.T).T

    for j in range(4):
        target_order = np.argsort(np.argsort(Z_corr[:, j]))
        original_sorted = np.sort(data[:, j])
        params[keys[j]] = original_sorted[target_order]

    return params


def generate_progression_schedule(params, n_months, rng):
    """Generate per-month disease parameter trajectories (vectorized).

    Returns dict of (n_patients, n_months) arrays.
    """
    n = len(params['k1_scale'])

    # k1_scale
    k1_annual_rate = rng.uniform(0.02, 0.10, n) * (1 + 0.5 * params['diabetes'])
    k1_monthly = k1_annual_rate / 12.0
    k1_sched = np.empty((n, n_months))
    k1_sched[:, 0] = params['k1_scale']
    for t in range(1, n_months):
        noise = rng.normal(0, 0.002, n)
        k1_sched[:, t] = k1_sched[:, t - 1] + k1_monthly + noise
    k1_sched = k1_sched.clip(1.0, 3.0)

    # Kf_scale (nonlinear: accelerates as Kf drops)
    Kf_annual_rate = rng.uniform(0.01, 0.05, n) * (1 + 0.4 * params['diabetes'])
    Kf_monthly = Kf_annual_rate / 12.0
    Kf_sched = np.empty((n, n_months))
    Kf_sched[:, 0] = params['Kf_scale']
    for t in range(1, n_months):
        accel = 1 + 0.5 * (1 - Kf_sched[:, t - 1])
        noise = rng.normal(0, 0.001, n)
        Kf_sched[:, t] = Kf_sched[:, t - 1] - Kf_monthly * accel + noise
    Kf_sched = Kf_sched.clip(0.15, 1.0)

    # Sf_scale
    Sf_annual_rate = rng.uniform(0.002, 0.015, n) * (1 + 0.3 * params['diabetes'])
    Sf_monthly = Sf_annual_rate / 12.0
    Sf_sched = np.empty((n, n_months))
    Sf_sched[:, 0] = params['Sf_scale']
    for t in range(1, n_months):
        noise = rng.normal(0, 0.001, n)
        Sf_sched[:, t] = Sf_sched[:, t - 1] - Sf_monthly + noise
    Sf_sched = Sf_sched.clip(0.4, 1.0)

    # diabetes
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

    # inflammation (drift + stochastic flares)
    infl_drift = rng.uniform(0.001, 0.005, n) / 12.0
    infl_sched = np.empty((n, n_months))
    infl_sched[:, 0] = params['inflammation']
    for t in range(1, n_months):
        flare = (rng.random(n) < 0.02) * rng.uniform(0.05, 0.15, n)
        infl_sched[:, t] = infl_sched[:, t - 1] + infl_drift + flare
    infl_sched = infl_sched.clip(0, 0.8)

    # Static parameters
    RAAS_sched = np.tile(params['RAAS_gain'][:, None], (1, n_months))
    TGF_sched = np.tile(params['TGF_gain'][:, None], (1, n_months))
    na_sched = np.tile(params['na_intake'][:, None], (1, n_months))

    return {
        'k1_scale': k1_sched, 'Sf_scale': Sf_sched, 'Kf_scale': Kf_sched,
        'diabetes': dm_sched, 'inflammation': infl_sched,
        'RAAS_gain': RAAS_sched, 'TGF_gain': TGF_sched, 'na_intake': na_sched,
    }


def assign_phenotype_labels(params):
    """Assign phenotype labels based on baseline disease parameters."""
    n = len(params['k1_scale'])
    labels = np.full(n, 'healthy', dtype='U20')

    hfpef = params['k1_scale'] > 1.3
    ckd = params['Kf_scale'] < 0.7
    dm = params['diabetes'] > 0.3

    cardiorenal = (hfpef & ckd) | (dm & (hfpef | ckd))
    hfpef_dominant = hfpef & ~ckd & ~dm
    ckd_dominant = ckd & ~hfpef & ~dm
    diabetes_dominant = dm & ~hfpef & ~ckd

    labels[hfpef_dominant] = 'hfpef_dominant'
    labels[ckd_dominant] = 'ckd_dominant'
    labels[diabetes_dominant] = 'diabetes_dominant'
    labels[cardiorenal] = 'cardiorenal'

    return labels


def generate_single_patient_trajectory(patient_idx, demo, schedule, n_months,
                                        var_names, cystatin_params, use_circadapt):
    """Generate one patient's monthly trajectory of 20 clinical variables.

    Tries CircAdapt first; falls back to parametric model.
    """
    if use_circadapt:
        try:
            return _run_circadapt_trajectory(n_months, schedule, demo, cystatin_params)
        except Exception:
            pass

    return _parametric_trajectory(n_months, schedule, demo, cystatin_params)


def _run_circadapt_trajectory(n_months, schedule, demo, cystatin_params):
    """Run CircAdapt at yearly resolution and cubic-interpolate to monthly."""
    from cardiorenal_coupling import run_coupled_simulation

    k1_sched = schedule['k1_scale']
    Sf_sched = schedule['Sf_scale']
    Kf_sched = schedule['Kf_scale']
    dm_sched = schedule['diabetes']
    infl_sched = schedule['inflammation']

    age = demo['age']
    sex = demo['sex']
    BSA = demo['BSA']

    n_years = max(n_months // 12, 2)
    yearly_indices = np.linspace(0, n_months - 1, n_years, dtype=int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist = run_coupled_simulation(
            n_steps=n_years,
            dt_renal_hours=6.0,
            cardiac_schedule=[float(Sf_sched[i]) for i in yearly_indices],
            kidney_schedule=[float(Kf_sched[i]) for i in yearly_indices],
            stiffness_schedule=[float(k1_sched[i]) for i in yearly_indices],
            inflammation_schedule=[float(infl_sched[i]) for i in yearly_indices],
            diabetes_schedule=[float(dm_sched[i]) for i in yearly_indices],
        )

    if any(np.isnan(v) for v in hist.get('MAP', [0])):
        raise RuntimeError("CircAdapt diverged")

    n_vars = len(VAR_NAMES)
    yearly_data = np.empty((n_years, n_vars), dtype=np.float32)

    for s in range(n_years):
        yearly_data[s, VAR_IDX['SBP_mmHg']] = hist['SBP'][s]
        yearly_data[s, VAR_IDX['MAP_mmHg']] = hist['MAP'][s]
        yearly_data[s, VAR_IDX['CO_L_min']] = hist['CO'][s]
        yearly_data[s, VAR_IDX['LVEF_pct']] = hist['EF'][s]
        yearly_data[s, VAR_IDX['eGFR_mL_min']] = hist['GFR'][s]

        CO = max(hist['CO'][s], 0.5)
        MAP = hist['MAP'][s]
        CVP = 5.0
        yearly_data[s, VAR_IDX['SVR_wood']] = (MAP - CVP) / CO

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
            (1 + 0.5 * dm_sched[yearly_indices[s]]), 0.5
        )

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

    for j, vn in enumerate(VAR_NAMES):
        lo, hi = CORE_20_VARIABLES[vn]['physiological_bounds']
        trajectory[:, j] = trajectory[:, j].clip(lo, hi)

    return trajectory


def _parametric_trajectory(n_months, schedule, demo, cystatin_params):
    """Parametric trajectory generation (no CircAdapt dependency)."""
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

        MAP_base = 93 + 5 * (RAAS_val - 1.5) + 3 * d + 2 * i + 0.05 * (na_val - 150)
        CO_base = 4.6 * Sf * (1 - 0.1 * (k1 - 1)) * (1 - 0.05 * i)
        CO = max(CO_base, 1.5)
        SBP = MAP_base * 1.4 * (1 + 0.02 * (k1 - 1))
        MAP = MAP_base
        CVP = 5.0 + 2 * (k1 - 1) + 1 * (1 - Kf) + 0.5 * d
        SVR = (MAP - CVP) / max(CO, 0.5)

        EDV = 120 * (1 + 0.05 * (1 - Sf)) * (1 - 0.03 * (k1 - 1))
        ESV = EDV * (1 - Sf * 0.65 * (1 - 0.1 * (k1 - 1)))
        ESV = max(ESV, EDV * 0.15)
        EF = (EDV - ESV) / EDV * 100
        SV = EDV - ESV

        LVIDd = 0.895 * EDV ** (1/3)
        LVmass = 150 * (1 + 0.15 * (k1 - 1) + 0.1 * d)
        GLS = -18 * (EF / 65) * Sf

        E = 68 * (1 + 0.20 * (k1 - 1) + 0.15 * (1 - Kf))
        e_prime = max(5.7 / (k1 * (1 + 0.1 * i)), 1.5)
        E_e_prime = E / e_prime
        E_A = 0.86 / (1 + 0.15 * (k1 - 1))
        if k1 > 2.0:
            E_A = 0.86 + 0.5 * (k1 - 2.0)

        LAv = 49 * (1 + 0.20 * (k1 - 1) + 0.10 * d)
        PASP = 28 + 5 * (k1 - 1) + 3 * (1 - Kf) + 0.1 * (MAP - 93)

        GFR = max(120 * Kf * (1 - 0.1 * d) * (1 - 0.05 * i) * (MAP / 93) ** 0.3, 3.0)
        eGFR = max(GFR * 0.55, 3.0)
        cr_base = 90.0 if sex == 'M' else 72.0
        creatinine = cr_base / max(GFR, 5.0)
        UACR = max(5.0 * (1 + 2.0 * (1 - Kf)) * (1 + 0.5 * d), 0.5)
        sex_factor = 0.95 if sex == 'F' else 1.0
        cystatin = cystatin_params['D_cys'] / max(GFR, 5.0) * (
            1 + cystatin_params['epsilon'] * i
        ) * sex_factor

        LVEDP_est = 10 * k1
        NTproBNP = 50 * np.exp(0.05 * LVEDP_est + 0.02 * CVP + 0.003 * LVmass / BSA)
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

    for j, vn in enumerate(VAR_NAMES):
        lo, hi = CORE_20_VARIABLES[vn]['physiological_bounds']
        traj[:, j] = traj[:, j].clip(lo, hi)

    return traj


def add_measurement_noise(trajectories, var_names, rng):
    """Add realistic measurement noise and PASP missingness.

    Parameters
    ----------
    trajectories : (N, T, 20) clean values
    var_names : list of str
    rng : numpy random Generator

    Returns
    -------
    noisy : (N, T, 20) with PASP NaNs
    """
    N, T, V = trajectories.shape
    noisy = trajectories.copy()

    for j, vn in enumerate(var_names):
        noise_type, magnitude = MEASUREMENT_NOISE[vn]
        if noise_type == 'gaussian_absolute':
            noisy[:, :, j] += rng.normal(0, magnitude, (N, T))
        elif noise_type == 'gaussian_relative':
            noisy[:, :, j] *= rng.normal(1, magnitude, (N, T))

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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_cystatin_c(params=None):
    """Run 7 validation checks for cystatin C emission model. Returns True if all pass."""
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

    # CKD-EPI cross-check
    cys_c_test = D_cys / 60.0
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

    # Population check
    pop_cys = D_cys / 66.0 * (1 + eps * 0.12)
    status = "PASS" if 0.90 <= pop_cys <= 1.30 else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"Pop-level check: GFR=66, i=0.12 -> CysC={pop_cys:.2f} "
          f"(target: 1.10+/-0.20) {status}")

    return all_pass


def validate_marginals(trajectories, var_names):
    """Compare month-0 distributions to ARIC V5 targets."""
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


def validate_trajectories(trajectories, var_names, labels):
    """Plausibility checks on generated trajectories. Returns True if all pass."""
    all_pass = True
    print("\n--- Trajectory Plausibility Checks ---")

    ef_idx = var_names.index('LVEF_pct')
    egfr_idx = var_names.index('eGFR_mL_min')
    ee_idx = var_names.index('E_over_e_prime_sept')
    bnp_idx = var_names.index('NTproBNP_pg_mL')
    cys_idx = var_names.index('cystatin_C_mg_L')
    cr_idx = var_names.index('creatinine_mg_dL')

    # 1. HFpEF: LVEF stays >50% for >85% of trajectory
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

    # 2. CKD: eGFR declines in >80%
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

    # 3. Cardiorenal: E/e' increases in >70%
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

    # 7. No eGFR<10 AND LVEF>70%
    bad = (trajectories[:, :, egfr_idx] < 10) & (trajectories[:, :, ef_idx] > 70)
    n_bad = bad.any(axis=1).sum()
    status = "PASS" if n_bad == 0 else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"7. eGFR<10 & LVEF>70% implausible combos: {n_bad} patients [{status}]")

    return all_pass


def _monthly_worker_fn(args):
    """Worker for multiprocessing monthly patient generation."""
    idx, demo_dict, schedule_dict, n_months, var_names, cystatin_params, use_circadapt = args
    return generate_single_patient_trajectory(
        idx, demo_dict, schedule_dict, n_months, var_names,
        cystatin_params, use_circadapt,
    )


def generate_monthly_cohort(n_patients, n_months, n_workers, seed, validate,
                             output_path='cohort_monthly.npz'):
    """Generate monthly-resolution synthetic cohort (N, T, 20).

    Steps: demographics → disease params → correlations → progression →
    phenotypes → trajectories → noise → validate → save.
    """
    rng = np.random.default_rng(seed)
    t_start = time.time()

    print(f"[1/8] Sampling demographics for {n_patients} patients...")
    demographics = sample_demographics(n_patients, rng)

    print("[2/8] Sampling disease parameters...")
    params = sample_disease_parameters(n_patients, rng)
    params = apply_disease_correlations(params, rng)

    print(f"[3/8] Generating {n_months}-month progression schedules...")
    schedules = generate_progression_schedule(params, n_months, rng)

    print("[4/8] Assigning phenotype labels...")
    labels = assign_phenotype_labels(params)
    unique, counts = np.unique(labels, return_counts=True)
    for lab, cnt in zip(unique, counts):
        print(f"  {lab}: {cnt} ({cnt/n_patients*100:.1f}%)")

    # Check CircAdapt availability
    use_circadapt = False
    try:
        from cardiorenal_coupling import run_coupled_simulation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_hist = run_coupled_simulation(n_steps=2, dt_renal_hours=6.0)
        if any(np.isnan(v) for v in test_hist.get('MAP', [0])):
            raise RuntimeError("CircAdapt produces NaN")
        use_circadapt = True
        print("[5/8] CircAdapt available — using coupled simulator with interpolation")
    except Exception as e:
        print(f"[5/8] CircAdapt unavailable ({e}) — using parametric fallback")

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
            results.append(_monthly_worker_fn(args))
            if (i + 1) % max(1, n_patients // 10) == 0:
                print(f"  {i+1}/{n_patients} patients done "
                      f"({(i+1)/n_patients*100:.0f}%)")
    else:
        with mp.Pool(n_workers) as pool:
            results = []
            for i, result in enumerate(pool.imap(
                _monthly_worker_fn, worker_args,
                chunksize=max(1, n_patients // (n_workers * 4))
            )):
                results.append(result)
                if (i + 1) % max(1, n_patients // 10) == 0:
                    print(f"  {i+1}/{n_patients} patients done "
                          f"({(i+1)/n_patients*100:.0f}%)")

    trajectories_clean = np.stack(results, axis=0)
    print(f"  Clean trajectories shape: {trajectories_clean.shape}")

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


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic cardiorenal cohort')
    parser.add_argument('--mode', choices=['paired', 'monthly'], default='paired',
                        help='paired = V5/V7 for NN training, monthly = trajectories for RL')
    parser.add_argument('--n_patients', type=int, default=None)
    parser.add_argument('--n_months', type=int, default=96,
                        help='Months for monthly mode (default 96 = 8 years)')
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validate', action='store_true',
                        help='Run validation checks (monthly mode)')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.mode == 'paired':
        n_patients = args.n_patients or COHORT_DEFAULTS['n_patients']
        output = args.output or 'cohort_data.npz'

        v5, v7, var_names, metadata = generate_paired_cohort(
            n_patients=n_patients, seed=args.seed, n_workers=args.n_workers,
        )

        outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), output)
        np.savez(outpath, v5=v5, v7=v7, var_names=np.array(var_names))
        print(f"Saved to {outpath}: v5 {v5.shape}, v7 {v7.shape}, {len(var_names)} variables")

        print(f"\n{'='*60}")
        print(f"  Cohort Summary ({v5.shape[0]} patients, {v5.shape[1]} variables)")
        print(f"{'='*60}")
        for i, name in enumerate(var_names[:10]):
            print(f"  {name:30s}  V5: {v5[:,i].mean():.2f} +/- {v5[:,i].std():.2f}  "
                  f"V7: {v7[:,i].mean():.2f} +/- {v7[:,i].std():.2f}")
        print(f"  ... and {len(var_names) - 10} more variables")

    else:  # monthly
        n_patients = args.n_patients or 5000
        output = args.output or 'cohort_monthly.npz'

        generate_monthly_cohort(
            n_patients=n_patients, n_months=args.n_months,
            n_workers=args.n_workers, seed=args.seed,
            validate=args.validate, output_path=output,
        )


if __name__ == '__main__':
    main()
