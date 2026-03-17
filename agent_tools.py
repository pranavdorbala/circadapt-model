#!/usr/bin/env python3
"""
Agent Tools: LLM-callable functions wrapping the CircAdapt + Hallow model.
===========================================================================
(Paper Section 3.9 — "The 4 Tools of the Agentic Inference Engine")

This module defines the four tools that the LLM agent uses to interact with
the mechanistic cardiorenal model during the optimization loop. Each tool is:
    1. A Python function that wraps model functionality
    2. An OpenAI function-calling JSON schema (for LiteLLM/OpenAI API)

The four tools (described in Section 3.9 of the paper):
    ┌───────────────────────────┬──────────────────────────────────────────────┐
    │ Tool                      │ Purpose                                      │
    ├───────────────────────────┼──────────────────────────────────────────────┤
    │ run_circadapt_model       │ Execute the full coupled CircAdapt + Hallow  │
    │                           │ model, returning ~113 ARIC clinical vars     │
    ├───────────────────────────┼──────────────────────────────────────────────┤
    │ compute_error             │ Compute weighted normalized error between    │
    │                           │ model output and V7 clinical target          │
    ├───────────────────────────┼──────────────────────────────────────────────┤
    │ get_sensitivity           │ Finite-difference Jacobian: how each ARIC   │
    │                           │ variable responds to perturbing a parameter  │
    ├───────────────────────────┼──────────────────────────────────────────────┤
    │ compare_to_clinical_norms │ Classify model output against clinical       │
    │                           │ thresholds (HF staging, CKD staging, etc.)  │
    └───────────────────────────┴──────────────────────────────────────────────┘

Tool design philosophy:
- Each tool returns a JSON-serializable dict, which is automatically
  stringified and appended to the LLM conversation context.
- Tools are stateless — all state is carried in the conversation history.
- Error handling: every tool returns a dict with an 'error' key on failure,
  so the LLM can reason about what went wrong and retry.
- Tools are intentionally simple wrappers — all complex logic lives in the
  model code (synthetic_cohort.py, emission_functions.py, etc.).
"""

# ─── Standard library imports ────────────────────────────────────────────────
import os
import sys
import json
import numpy as np
from typing import Dict, List, Optional

# Ensure the project root is on sys.path for sibling module imports.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Internal project imports ────────────────────────────────────────────────
# TUNABLE_PARAMS: the 8 disease-progression parameters with ranges and defaults
#     (paper Section 3.4). Used for parameter validation and clamping.
# ARIC_VARIABLES: metadata for all ~113 ARIC-compatible clinical variables,
#     including normal ranges and importance weights (paper Section 3.3).
# NUMERIC_VAR_NAMES: sorted list of the ~113 numeric variable names.
# CLINICAL_THRESHOLDS: ordered thresholds for disease staging (HF, CKD, etc.)
#     based on published clinical guidelines (paper Section 3.9).
# N_FEATURES: total count of numeric ARIC variables (~113).
from config import (
    TUNABLE_PARAMS, ARIC_VARIABLES, NUMERIC_VAR_NAMES,
    CLINICAL_THRESHOLDS, N_FEATURES,
)

# evaluate_patient_state: the core model evaluation function that runs the
# coupled CircAdapt + Hallow model for a given set of disease parameters
# and demographics, returning a dict of ~113 ARIC clinical variables.
# (paper Section 3.5 — "Synthetic Cohort Generation")
from synthetic_cohort import evaluate_patient_state


# ═══════════════════════════════════════════════════════════════════════════════
# RL Coupling Policy (lazy-loaded)
# ═══════════════════════════════════════════════════════════════════════════════
# If a trained RL attention policy exists at models/rl_attention_policy.pt,
# it is loaded once and used to enhance the coupled simulation with learned
# per-message coupling weights. The LLM agent is unaware of this — the tool
# signature is unchanged.

import torch

_RL_POLICY = None
_RL_POLICY_LOADED = False  # Distinguish "not yet tried" from "tried and missing"


def _load_rl_policy():
    """Lazy-load the trained RL coupling policy (if available)."""
    global _RL_POLICY, _RL_POLICY_LOADED
    if _RL_POLICY_LOADED:
        return _RL_POLICY

    _RL_POLICY_LOADED = True
    rl_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'models', 'rl_attention_policy.pt',
    )
    if os.path.exists(rl_path):
        try:
            from models.attention_coupling import AttentionCouplingPolicy
            ckpt = torch.load(rl_path, map_location='cpu', weights_only=False)
            policy = AttentionCouplingPolicy(**ckpt['config'])
            policy.load_state_dict(ckpt['model_state_dict'])
            policy.eval()
            _RL_POLICY = policy
            print(f"[agent_tools] Loaded RL coupling policy from {rl_path}")
        except Exception as e:
            print(f"[agent_tools] Warning: failed to load RL policy: {e}")
            _RL_POLICY = None
    return _RL_POLICY


def _run_with_rl_coupling(params: Dict, demographics: Dict) -> Optional[Dict]:
    """Run the coupled simulation using the RL-learned coupling equation.

    The frozen RL policy provides per-message alpha scaling and inflammatory
    residual corrections at each coupling step.
    """
    from cardiorenal_coupling import run_coupled_simulation_rl, obs_dict_to_vector
    from emission_functions import extract_all_aric_variables
    from config import RL_CONFIG

    policy = _load_rl_policy()
    if policy is None:
        return None  # Fall back to standard evaluation

    # Build parameter schedules (same as evaluate_patient_state)
    n_steps = 8
    cardiac_sched = [params['Sf_act_scale']] * n_steps
    kidney_sched = [params['Kf_scale']] * n_steps
    stiffness_sched = [params['k1_scale']] * n_steps
    inflam_sched = [params['inflammation_scale']] * n_steps
    diab_sched = [params['diabetes_scale']] * n_steps

    def alpha_fn(obs_dict, step):
        action, _, _ = policy.get_action(obs_dict, deterministic=True)
        return action[:5], action[5:]

    try:
        hist = run_coupled_simulation_rl(
            n_steps=n_steps,
            dt_renal_hours=180.0,
            renal_substeps=4,
            cardiac_schedule=cardiac_sched,
            kidney_schedule=kidney_sched,
            stiffness_schedule=stiffness_sched,
            inflammation_schedule=inflam_sched,
            diabetes_schedule=diab_sched,
            alpha_fn=alpha_fn,
            baselines=RL_CONFIG['baselines'],
            verbose=False,
        )

        # Extract ARIC variables from final state
        # Use the same emission pipeline as evaluate_patient_state
        from cardiorenal_coupling import CircAdaptHeartModel, HallowRenalModel
        # Reconstruct a minimal state for emission extraction from the last step
        # The history contains the raw hemodynamic values we need
        last_idx = -1
        result = {
            'SBP_mmHg': hist['SBP'][last_idx],
            'DBP_mmHg': hist['DBP'][last_idx],
            'MAP_mmHg': hist['MAP'][last_idx],
            'CO_Lmin': hist['CO'][last_idx],
            'SV_mL': hist['SV'][last_idx],
            'LVEF_pct': hist['EF'][last_idx],
        }

        # Fall back to standard evaluation to get full 113 variables
        # (the RL coupling modifies the simulation dynamics but we still
        # need the full emission pipeline for complete ARIC output)
        return None  # For now, fall back — full emission integration in Phase 2

    except Exception:
        return None  # Fall back to standard evaluation


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 1: Run CircAdapt Model
# ═══════════════════════════════════════════════════════════════════════════════
# (Paper Section 3.9 — Tool 1: "Model Evaluation")
#
# This is the most frequently called tool. It wraps the full coupled
# CircAdapt (cardiac) + Hallow (renal) model evaluation, which:
#   1. Sets up cardiac parameters (Sf_act, k1 passive stiffness, etc.)
#   2. Sets up renal parameters (Kf, RAAS_gain, TGF_gain, na_intake)
#   3. Applies inflammatory/diabetic scaling via InflammatoryState
#   4. Runs the bidirectional coupling loop (heart→kidney→heart)
#   5. Extracts ~113 ARIC-compatible clinical variables
#
# Runtime: ~0.5 seconds per call (dominated by the ODE solver for the
# cardiac cycle and the renal steady-state iteration).

def run_circadapt_model(
    Sf_act_scale: float = 1.0,
    Kf_scale: float = 1.0,
    inflammation_scale: float = 0.0,
    diabetes_scale: float = 0.0,
    k1_scale: float = 1.0,
    RAAS_gain: float = 1.5,
    TGF_gain: float = 2.0,
    na_intake: float = 150.0,
    age: float = 75.0,
    sex: str = 'M',
    BSA: float = 1.9,
    height_m: float = 1.75,
) -> Dict:
    """
    Run the coupled CircAdapt heart + Hallow renal model and return ~113
    ARIC-compatible clinical variables.

    (Paper Section 3.9, Tool 1)

    This function serves as the LLM's primary interface to the mechanistic
    model. The LLM calls this tool with candidate disease parameters, and
    the returned clinical variables are compared against the V7 target
    via compute_error.

    Parameters
    ----------
    Sf_act_scale : float
        Active fiber stress scale (0.2-1.0). Controls LV contractility.
        - 1.0 = normal contractile function
        - <1.0 = reduced contractility (HFrEF pathway)
        - Maps to Patch[Sf_act] on LV and septal wall in CircAdapt
        (Paper Section 3.2 — "Cardiac Model: Time-varying Elastance")
    Kf_scale : float
        Glomerular ultrafiltration coefficient scale (0.05-1.0).
        - 1.0 = normal nephron function (GFR ~120 mL/min)
        - <1.0 = nephron loss / CKD (podocyte injury, mesangial expansion)
        - Maps to Kf in Hallow et al. 2017 glomerular model
        (Paper Section 3.3 — "Renal Model: Hallow et al.")
    inflammation_scale : float
        Systemic inflammation index (0.0-1.0).
        - 0.0 = no inflammation
        - 1.0 = severe systemic inflammation
        - Drives multi-organ effects via InflammatoryState: reduced Sf_act,
          increased SVR (p0), increased arterial stiffness, reduced Kf,
          increased afferent arteriole resistance, RAAS gain modulation
        (Paper Section 3.4 — "Inflammatory and Metabolic Scaling")
    diabetes_scale : float
        Diabetes metabolic burden (0.0-1.0).
        - 0.0 = no diabetes
        - 1.0 = severe diabetes
        - Drives: increased passive myocardial stiffness (k1 → HFpEF),
          AGE-mediated arterial stiffness, biphasic Kf effect (early
          hyperfiltration, late nephron loss), increased efferent arteriole
          resistance, SGLT2-mediated proximal tubule reabsorption
        (Paper Section 3.4 — "Inflammatory and Metabolic Scaling")
    k1_scale : float
        Passive myocardial stiffness multiplier (1.0-3.0).
        - 1.0 = normal diastolic compliance
        - >1.0 = increased diastolic stiffness (HFpEF pathway)
        - Maps to Patch[k1] on LV and septal wall in CircAdapt
        (Paper Section 3.2 — "Exponential EDPVR")
    RAAS_gain : float
        Renin-angiotensin-aldosterone system sensitivity (0.5-3.0).
        - 1.5 = normal RAAS reactivity (calibrated default)
        - Higher = more reactive to MAP drops → more AngII → increased
          efferent arteriole tone and collecting duct Na reabsorption
        (Paper Section 3.3 — "RAAS Module in Hallow Model")
    TGF_gain : float
        Tubuloglomerular feedback gain (1.0-4.0).
        - 2.0 = normal TGF reactivity (calibrated default)
        - Senses macula densa Na delivery → adjusts afferent arteriole
          resistance to stabilize distal delivery
        (Paper Section 3.3 — "TGF and Macula Densa Sensing")
    na_intake : float
        Dietary sodium intake in mEq/day (50-300).
        - 150 = average American diet (~3.5g Na/day)
        - Affects the pressure-natriuresis equilibrium and blood volume
        (Paper Section 3.3 — "Pressure-Natriuresis and Volume Balance")
    age : float
        Patient age in years. Affects eGFR calculation (CKD-EPI equation)
        and some age-dependent model parameters.
    sex : str
        'M' or 'F'. Affects eGFR calculation and normal ranges.
    BSA : float
        Body surface area in m^2. Used for indexing (LVMi, LVEDVi, etc.).
    height_m : float
        Height in meters. Used for allometric indexing (LVMi/height^2.7).

    Returns
    -------
    dict
        ~113 ARIC variable name → value pairs, plus 'params_used' dict.
        On failure, returns {'error': '...', 'params': {...}}.
    """
    # ─── Parameter clamping ──────────────────────────────────────────────
    # Clip all disease parameters to their valid ranges defined in config.py.
    # This prevents the LLM from accidentally requesting out-of-range values
    # (e.g., Sf_act_scale=0.0 would crash the cardiac solver, Kf_scale=0
    # would cause division by zero in the glomerular model).
    params = {
        'Sf_act_scale': float(np.clip(Sf_act_scale, *TUNABLE_PARAMS['Sf_act_scale']['range'])),
        'Kf_scale': float(np.clip(Kf_scale, *TUNABLE_PARAMS['Kf_scale']['range'])),
        'inflammation_scale': float(np.clip(inflammation_scale, *TUNABLE_PARAMS['inflammation_scale']['range'])),
        'diabetes_scale': float(np.clip(diabetes_scale, *TUNABLE_PARAMS['diabetes_scale']['range'])),
        'k1_scale': float(np.clip(k1_scale, *TUNABLE_PARAMS['k1_scale']['range'])),
        'RAAS_gain': float(np.clip(RAAS_gain, *TUNABLE_PARAMS['RAAS_gain']['range'])),
        'TGF_gain': float(np.clip(TGF_gain, *TUNABLE_PARAMS['TGF_gain']['range'])),
        'na_intake': float(np.clip(na_intake, *TUNABLE_PARAMS['na_intake']['range'])),
    }
    # Demographics are passed through without clamping (they represent
    # measured patient characteristics, not tunable disease knobs).
    demographics = {
        'age': float(age),
        'sex': str(sex),
        'BSA': float(BSA),
        'height_m': float(height_m),
    }

    # ─── Try RL-enhanced simulation first ─────────────────────────────────
    # If a trained RL coupling policy is available, use it to enhance the
    # coupled simulation with learned per-message coupling weights. This
    # produces more physiologically accurate trajectories without changing
    # the tool interface — the LLM agent is completely unaware.
    rl_result = _run_with_rl_coupling(params, demographics)
    if rl_result is not None:
        output = {}
        for k, v in rl_result.items():
            if isinstance(v, (int, float)):
                output[k] = round(float(v), 3)
            else:
                output[k] = v
        output['params_used'] = params
        return output

    # ─── Run the full coupled model (standard path) ────────────────────
    # evaluate_patient_state() (from synthetic_cohort.py) orchestrates:
    # 1. CircAdapt cardiac simulation with Sf_act_scale, k1_scale, etc.
    # 2. Hallow renal model with Kf_scale, RAAS_gain, TGF_gain, na_intake
    # 3. InflammatoryState scaling (inflammation_scale, diabetes_scale)
    # 4. Bidirectional coupling (MAP/CO/CVP → kidney; V_blood/SVR → heart)
    # 5. ARIC variable extraction (emission_functions.py)
    # Returns None on solver failure (e.g., ODE divergence at extreme params).
    result = evaluate_patient_state(params, demographics)
    if result is None:
        # Return an error dict so the LLM can see what parameters caused
        # the failure and adjust its next guess accordingly.
        return {'error': 'Model evaluation failed', 'params': params}

    # ─── Format output for the LLM ──────────────────────────────────────
    # Round all numeric values to 3 decimal places for readability in the
    # LLM conversation context. This reduces token count and makes the
    # output easier for the LLM to parse.
    output = {}
    for k, v in result.items():
        if isinstance(v, (int, float)):
            output[k] = round(float(v), 3)
        else:
            output[k] = v
    # Include the clamped parameters in the output so the LLM (and our
    # tracking code in agent_loop.py) can see exactly what was used.
    output['params_used'] = params
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 2: Compute Error vs Target
# ═══════════════════════════════════════════════════════════════════════════════
# (Paper Section 3.9 — Tool 2: "Error Computation")
#
# This tool computes a weighted, normalized error metric that the LLM uses
# to assess how close the current model output is to the V7 clinical target.
#
# The error metric is designed to:
# 1. Be scale-invariant: each variable's error is normalized by its normal
#    range (e.g., LVEF normal range 55-70% means a 5% error is ~33% of range).
# 2. Be importance-weighted: clinically critical variables (LVEF, GFR, E/e',
#    NTproBNP) have weight 2.0, while less important ones have weight 0.3-0.5.
# 3. Report the worst offenders: the "worst 5 variables" list tells the LLM
#    where to focus its next parameter adjustment.
#
# The aggregate error formula:
#   aggregate = sqrt(sum(w_i * (|model_i - target_i| / range_i)^2) / sum(w_i))
# This is a weighted root-mean-square normalized error.

def compute_error(model_output: Dict, target: Dict) -> Dict:
    """
    Compute weighted per-variable error between model output and target.

    (Paper Section 3.9, Tool 2)

    This is the LLM's "loss function" — it tells the agent how far the
    current model output is from the desired V7 target. The agent uses
    this information to decide which parameters to adjust and by how much.

    The error metric is a weighted RMS normalized error:
        aggregate = sqrt( sum(w_i * err_i^2) / sum(w_i) )
    where err_i = |model_i - target_i| / normal_range_i
    and w_i is the clinical importance weight from ARIC_VARIABLES.

    Parameters
    ----------
    model_output : dict
        ARIC variable name → value from run_circadapt_model.
    target : dict
        ARIC variable name → target value (from NN prediction or real data).

    Returns
    -------
    dict with keys:
        'aggregate_error' : float
            Scalar summary error (0 = perfect, typically 0.01-0.5).
        'n_variables_compared' : int
            How many variables were compared (depends on overlap).
        'worst_5_variables' : dict
            The 5 variables with the highest weighted error, showing
            model value, target value, absolute error, and weight.
            This guides the LLM on where to focus next.
    """
    errors = {}
    # Running sums for the weighted RMS computation.
    weighted_sq_sum = 0.0
    weight_sum = 0.0
    # Track direction mismatches (currently unused but reserved for future
    # use in detecting when the model is moving in the wrong direction).
    direction_mismatches = []

    # Iterate over all numeric ARIC variables (sorted list from config.py).
    for var_name in NUMERIC_VAR_NAMES:
        # Skip variables not present in both model output and target.
        # This handles cases where the target is a partial specification
        # (e.g., only key variables from the NN prediction).
        if var_name not in model_output or var_name not in target:
            continue

        model_val = float(model_output[var_name])
        target_val = float(target[var_name])

        # Look up the normal range and weight for this variable.
        # normal = (low, high) tuple from ARIC_VARIABLES in config.py.
        # weight = clinical importance weight (0.3 to 2.0).
        normal = ARIC_VARIABLES.get(var_name, {}).get('normal', (0, 1))
        weight = ARIC_VARIABLES.get(var_name, {}).get('weight', 0.5)

        # Compute the normal range width. This is the denominator for
        # normalization. We use max(..., 1e-6) to avoid division by zero
        # for variables with zero-width normal ranges (e.g., diastolic_grade
        # has normal=(0,0)).
        normal_range = max(abs(normal[1] - normal[0]), 1e-6)

        # Normalized absolute error: how many "normal range widths" away
        # is the model from the target. For example, if LVEF target=45%,
        # model=55%, and normal range is 55-70% (width=15%), then
        # abs_err = |55-45|/15 = 0.67 (67% of the normal range).
        abs_err = abs(model_val - target_val) / normal_range

        # Store per-variable error details for the LLM to inspect.
        errors[var_name] = {
            'model': round(model_val, 3),
            'target': round(target_val, 3),
            'abs_error': round(abs_err, 4),
            'weight': weight,
        }

        # Accumulate for weighted RMS.
        weighted_sq_sum += abs_err ** 2 * weight
        weight_sum += weight

    # Compute aggregate error: weighted RMS normalized error.
    # This is a single scalar that the agent_loop.py uses for convergence
    # checking. A value of 0.05 means the average weighted variable is
    # off by 5% of its normal range — our default convergence threshold.
    aggregate = (weighted_sq_sum / max(weight_sum, 1e-6)) ** 0.5

    # Find the worst 5 variables by weighted error (abs_error * weight).
    # These are the variables the LLM should focus on improving. By
    # reporting these explicitly, we guide the LLM toward the most
    # impactful parameter adjustments.
    ranked = sorted(errors.items(), key=lambda x: x[1]['abs_error'] * x[1]['weight'], reverse=True)
    worst_5 = {k: v for k, v in ranked[:5]}

    return {
        'aggregate_error': round(float(aggregate), 4),
        'n_variables_compared': len(errors),
        'worst_5_variables': worst_5,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 3: Parameter Sensitivity (Jacobian)
# ═══════════════════════════════════════════════════════════════════════════════
# (Paper Section 3.9 — Tool 3: "Sensitivity Analysis via Finite-Difference
#  Jacobian")
#
# This tool computes the local sensitivity (partial derivative) of each ARIC
# variable with respect to a single disease parameter using the central
# finite-difference method:
#
#   d(var) / d(param) ≈ (var(param + delta) - var(param - delta)) / (2 * delta)
#
# This is the Jacobian column for one parameter. The LLM uses this to:
# 1. Identify which parameter has the most influence on a problematic variable
# 2. Estimate the parameter adjustment needed: delta_param ≈ error / sensitivity
# 3. Detect unexpected cross-coupling (e.g., changing Sf_act affects GFR
#    through cardiorenal coupling)
#
# Cost: 2 model evaluations per call (~1 second total).
# The LLM typically calls this 2-4 times per optimization run, focusing on
# the worst-matching variables from compute_error.

def get_sensitivity(
    base_params: Dict,
    param_name: str,
    delta: float = 0.05,
    age: float = 75.0,
    sex: str = 'M',
    BSA: float = 1.9,
    height_m: float = 1.75,
) -> Dict:
    """
    Finite-difference sensitivity: d(ARIC_var)/d(param) for all variables.

    (Paper Section 3.9, Tool 3)

    Computes the central finite-difference approximation to the partial
    derivative of each ARIC variable with respect to the specified parameter.
    This gives the LLM a local linear approximation of how the model responds
    to parameter changes, enabling Newton-like update steps.

    Method: Central difference
        sensitivity = (f(param + delta) - f(param - delta)) / (2 * delta)
    where f is the full CircAdapt + Hallow model evaluation.

    Parameters
    ----------
    base_params : dict
        Current disease parameter values (all 8 parameters).
    param_name : str
        Name of the parameter to perturb. Must be one of the 8 tunable
        parameters defined in TUNABLE_PARAMS.
    delta : float
        Perturbation size (default 0.05). This is in the parameter's natural
        units. For Sf_act_scale (range 0.2-1.0), delta=0.05 is a 5% change.
        For na_intake (range 50-300), delta=0.05 is tiny — the LLM should
        use a larger delta for parameters with wider ranges.
    age, sex, BSA, height_m : demographics
        Patient demographics, passed through to the model.

    Returns
    -------
    dict with keys:
        'parameter' : str
            The perturbed parameter name.
        'base_value' : float
            The current value of the parameter.
        'delta' : float
            The actual perturbation used (may differ from requested if at
            a boundary, e.g., Sf_act_scale=0.22 with delta=0.05 would clip
            the lower perturbation to 0.2).
        'top_10_sensitivities' : dict
            The 10 ARIC variables most sensitive to this parameter, with
            their d(var)/d(param) values. Positive = var increases when
            param increases.
        'all_sensitivities' : dict
            Complete sensitivity map for all ~113 variables.
    """
    # ─── Validate the parameter name ─────────────────────────────────────
    if param_name not in TUNABLE_PARAMS:
        return {'error': f'Unknown parameter: {param_name}',
                'valid_params': list(TUNABLE_PARAMS.keys())}

    demographics = {'age': age, 'sex': sex, 'BSA': BSA, 'height_m': height_m}
    prange = TUNABLE_PARAMS[param_name]['range']
    # Use the base_params value if available; otherwise fall back to default.
    base_val = base_params.get(param_name, TUNABLE_PARAMS[param_name]['default'])

    # ─── Compute perturbed values ────────────────────────────────────────
    # Clip to the parameter's valid range to avoid out-of-bounds evaluation.
    # If the parameter is at a boundary, the perturbation will be one-sided
    # (e.g., only upward if at the lower bound), which is less accurate but
    # still provides useful gradient information.
    val_lo = max(base_val - delta, prange[0])
    val_hi = min(base_val + delta, prange[1])
    # actual_delta accounts for boundary clipping. For central difference,
    # this is the full spread (hi - lo), and the derivative approximation
    # is (f(hi) - f(lo)) / actual_delta.
    actual_delta = val_hi - val_lo
    # If both perturbations clip to the same value (parameter is exactly at
    # a boundary with delta=0), we cannot compute a meaningful sensitivity.
    if actual_delta < 1e-8:
        return {'error': f'Cannot perturb {param_name} (at boundary)'}

    # ─── Evaluate model at perturbed parameter values ────────────────────
    # Create two copies of base_params with the parameter set to lo and hi.
    params_lo = dict(base_params)
    params_lo[param_name] = val_lo
    params_hi = dict(base_params)
    params_hi[param_name] = val_hi

    # Run the model twice: once at (param - delta) and once at (param + delta).
    # This is the main cost of sensitivity analysis (~1 second total).
    result_lo = evaluate_patient_state(params_lo, demographics)
    result_hi = evaluate_patient_state(params_hi, demographics)

    # If either evaluation fails, we cannot compute sensitivities.
    if result_lo is None or result_hi is None:
        return {'error': 'Model evaluation failed during sensitivity analysis'}

    # ─── Compute finite-difference sensitivities ─────────────────────────
    # For each ARIC variable, compute:
    #   d(var) / d(param) ≈ (var_hi - var_lo) / actual_delta
    # This is the central difference formula (or forward/backward if clipped).
    sensitivities = {}
    for var_name in NUMERIC_VAR_NAMES:
        if var_name in result_lo and var_name in result_hi:
            lo_val = float(result_lo[var_name])
            hi_val = float(result_hi[var_name])
            # The derivative: units are [ARIC variable units] / [parameter units].
            # For example, d(LVEF_pct) / d(Sf_act_scale) might be 40 %/1.0,
            # meaning a 0.1 increase in Sf_act_scale increases LVEF by ~4%.
            deriv = (hi_val - lo_val) / actual_delta
            sensitivities[var_name] = round(deriv, 4)

    # ─── Rank by absolute sensitivity ────────────────────────────────────
    # The top 10 most sensitive variables are reported prominently so the
    # LLM can quickly see which variables this parameter most affects.
    # This helps the LLM decide which parameter to adjust when a specific
    # variable is off-target.
    ranked = sorted(sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)
    top_10 = {k: v for k, v in ranked[:10]}

    return {
        'parameter': param_name,
        'base_value': round(base_val, 4),
        'delta': round(actual_delta, 4),
        'top_10_sensitivities': top_10,
        'all_sensitivities': sensitivities,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 4: Clinical Classification
# ═══════════════════════════════════════════════════════════════════════════════
# (Paper Section 3.9 — Tool 4: "Clinical Norm Comparison")
#
# This tool provides clinical context to the LLM by classifying the model
# output against standard clinical thresholds. It answers questions like:
# - Does this patient have HFrEF, HFmrEF, or HFpEF?
# - What CKD stage are they in?
# - Are filling pressures elevated?
# - Is there micro- or macroalbuminuria?
# - Is NT-proBNP in the heart failure range?
#
# The LLM uses this tool primarily at the start of optimization (to identify
# the phenotype) and at the end (to verify the optimized state makes clinical
# sense). It helps the LLM catch situations where the error is low but the
# clinical classification is wrong (e.g., the model says normal GFR but the
# target is CKD G3a).
#
# Thresholds are based on published clinical guidelines:
# - LVEF: ESC 2021 HF guidelines (HFrEF <40%, HFmrEF 40-49%, HFpEF ≥50%)
# - eGFR: KDIGO 2012 CKD staging (G1 ≥90, G2 60-89, G3a 45-59, etc.)
# - E/e': ASE/EACVI 2016 diastolic guidelines (normal <8, indeterminate 8-14)
# - UACR: KDIGO albuminuria staging (A1 <30, A2 30-299, A3 ≥300)
# - NT-proBNP: ESC 2021 HF diagnostic thresholds

def compare_to_clinical_norms(variables: Dict) -> Dict:
    """
    Classify each variable as normal/borderline/abnormal per clinical thresholds.

    (Paper Section 3.9, Tool 4)

    This tool provides the LLM with clinical context by:
    1. Applying threshold-based disease staging (HF type, CKD stage,
       filling pressure grade, albuminuria category, HF biomarker level)
    2. Checking all ~113 variables against their normal ranges and
       reporting which ones are out of range

    The LLM uses this tool to:
    - Identify the dominant disease phenotype at the start of optimization
    - Verify that the final optimized state is clinically coherent
    - Detect pathophysiological inconsistencies (e.g., low LVEF but
      normal NT-proBNP would suggest a model calibration issue)

    Parameters
    ----------
    variables : dict
        ARIC variable name → value mapping (from run_circadapt_model
        or the V7 target).

    Returns
    -------
    dict with keys:
        'disease_staging' : dict
            Variable name → {'value': float, 'classification': str}
            for each variable in CLINICAL_THRESHOLDS.
        'out_of_range_count' : int
            Total number of variables outside their normal range.
        'out_of_range' : dict
            Variable name → {'value': float, 'normal_range': tuple,
            'status': 'below_normal' | 'above_normal'} for all
            variables outside their normal range.
    """
    classifications = {}

    # ─── Disease staging via clinical thresholds ─────────────────────────
    # CLINICAL_THRESHOLDS (from config.py) is a dict where each key is a
    # variable name and the value is a list of (threshold, label) tuples,
    # sorted in DESCENDING order. We iterate through the thresholds and
    # assign the first label where the value >= threshold.
    #
    # Example for LVEF_pct:
    #   [(50.0, 'HFpEF'), (40.0, 'HFmrEF'), (0.0, 'HFrEF')]
    #   If LVEF=45 → first match is 40.0 → 'HFmrEF'
    #   If LVEF=55 → first match is 50.0 → 'HFpEF'
    #   If LVEF=30 → first match is 0.0 → 'HFrEF'
    for var_name, thresholds in CLINICAL_THRESHOLDS.items():
        if var_name not in variables:
            continue
        val = float(variables[var_name])
        label = 'unknown'
        # Walk through thresholds in descending order. The first threshold
        # the value exceeds determines the classification.
        for threshold_val, threshold_label in thresholds:
            if val >= threshold_val:
                label = threshold_label
                break
        classifications[var_name] = {
            'value': round(val, 2),
            'classification': label,
        }

    # ─── Normal-range check for all numeric variables ────────────────────
    # For every ARIC variable that has a defined normal range, check if the
    # current value falls outside that range. This gives the LLM a broad
    # overview of how many variables are abnormal and which direction they
    # deviate.
    out_of_range = {}
    for var_name in NUMERIC_VAR_NAMES:
        if var_name not in variables:
            continue
        # Look up the normal range from ARIC_VARIABLES metadata.
        normal = ARIC_VARIABLES.get(var_name, {}).get('normal')
        if normal is None:
            continue
        val = float(variables[var_name])
        # Check if the value is below or above the normal range.
        if val < normal[0]:
            out_of_range[var_name] = {
                'value': round(val, 2),
                'normal_range': normal,
                'status': 'below_normal',
            }
        elif val > normal[1]:
            out_of_range[var_name] = {
                'value': round(val, 2),
                'normal_range': normal,
                'status': 'above_normal',
            }

    return {
        'disease_staging': classifications,
        'out_of_range_count': len(out_of_range),
        'out_of_range': out_of_range,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LiteLLM Tool Schemas (OpenAI function-calling format)
# ═══════════════════════════════════════════════════════════════════════════════
# (Paper Section 3.9 — "Tool Schema Design")
#
# These JSON schemas define the tool interfaces that the LLM sees. They follow
# the OpenAI function-calling specification, which LiteLLM translates to the
# appropriate format for each backend (Gemini, Claude, etc.).
#
# Schema design principles:
# 1. Descriptions include physiological context (not just data types) so the
#    LLM can make informed choices about parameter values.
# 2. Default values are documented in descriptions (not enforced by schema)
#    so the LLM can omit parameters and get reasonable defaults.
# 3. "required" is minimal — most parameters are optional with defaults.
#    This lets the LLM call run_circadapt_model({}) to get a healthy baseline.
# 4. Numeric types use "number" (not "integer") since all disease parameters
#    are continuous.

TOOL_SCHEMAS = [
    # ─── Tool 1 Schema: run_circadapt_model ──────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "run_circadapt_model",
            "description": (
                "Run the coupled CircAdapt heart + Hallow renal model with given "
                "disease parameters. Returns 113 ARIC-compatible clinical variables "
                "including echocardiographic measurements, hemodynamics, and renal markers. "
                "Takes ~0.5 seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "Sf_act_scale": {
                        "type": "number",
                        "description": "Active fiber stress scale (0.2-1.0). Lower = weaker contraction (HFrEF). Default 1.0.",
                    },
                    "Kf_scale": {
                        "type": "number",
                        "description": "Glomerular ultrafiltration coefficient scale (0.05-1.0). Lower = nephron loss (CKD). Default 1.0.",
                    },
                    "inflammation_scale": {
                        "type": "number",
                        "description": "Systemic inflammation (0.0-1.0). Increases SVR, arterial stiffness, reduces Sf_act and Kf. Default 0.0.",
                    },
                    "diabetes_scale": {
                        "type": "number",
                        "description": "Diabetes burden (0.0-1.0). Increases diastolic stiffness (k1→HFpEF), arterial stiffness, affects Kf. Default 0.0.",
                    },
                    "k1_scale": {
                        "type": "number",
                        "description": "Passive myocardial stiffness scale (1.0-3.0). Higher = more diastolic dysfunction (HFpEF). Default 1.0.",
                    },
                    "RAAS_gain": {
                        "type": "number",
                        "description": "RAAS sensitivity (0.5-3.0). Higher = more reactive to MAP drops. Default 1.5.",
                    },
                    "TGF_gain": {
                        "type": "number",
                        "description": "Tubuloglomerular feedback gain (1.0-4.0). Default 2.0.",
                    },
                    "na_intake": {
                        "type": "number",
                        "description": "Dietary sodium intake in mEq/day (50-300). Default 150.",
                    },
                },
                # No parameters are required — the LLM can call with no args
                # to get a healthy baseline, or specify only the parameters it
                # wants to change.
                "required": [],
            },
        },
    },
    # ─── Tool 2 Schema: compute_error ────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "compute_error",
            "description": (
                "Compute weighted error between model output and a target clinical state. "
                "Returns aggregate error (0=perfect match), worst 5 variables, and per-variable errors."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_output": {
                        "type": "object",
                        "description": "Dict of ARIC variable names → values from run_circadapt_model.",
                    },
                    "target": {
                        "type": "object",
                        "description": "Dict of ARIC variable names → target values to match.",
                    },
                },
                # Both arguments are required because compute_error needs two
                # data sets to compare. The LLM must provide both.
                "required": ["model_output", "target"],
            },
        },
    },
    # ─── Tool 3 Schema: get_sensitivity ──────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_sensitivity",
            "description": (
                "Compute finite-difference sensitivity of all ARIC variables to a single "
                "disease parameter. Returns d(variable)/d(parameter) for the top 10 most "
                "sensitive variables. Takes ~1 second (two model runs)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "base_params": {
                        "type": "object",
                        "description": "Current disease parameter values.",
                    },
                    "param_name": {
                        "type": "string",
                        "description": "Name of parameter to perturb. One of: Sf_act_scale, Kf_scale, inflammation_scale, diabetes_scale, k1_scale, RAAS_gain, TGF_gain, na_intake.",
                    },
                    "delta": {
                        "type": "number",
                        "description": "Perturbation size. Default 0.05.",
                    },
                },
                # base_params and param_name are required; delta is optional.
                "required": ["base_params", "param_name"],
            },
        },
    },
    # ─── Tool 4 Schema: compare_to_clinical_norms ────────────────────────
    {
        "type": "function",
        "function": {
            "name": "compare_to_clinical_norms",
            "description": (
                "Classify model output against clinical thresholds. Returns disease staging "
                "(HFpEF/HFrEF, CKD stage, filling pressure grade, albuminuria) and a list "
                "of variables outside their normal range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "variables": {
                        "type": "object",
                        "description": "Dict of ARIC variable names → values.",
                    },
                },
                "required": ["variables"],
            },
        },
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# Tool Dispatch Registry
# ═══════════════════════════════════════════════════════════════════════════════
# Maps tool function names (as strings matching the schema "name" fields)
# to the actual Python callables. This registry is used by execute_tool()
# to dispatch LLM tool calls to the correct function.

TOOL_FUNCTIONS = {
    'run_circadapt_model': run_circadapt_model,
    'compute_error': compute_error,
    'get_sensitivity': get_sensitivity,
    'compare_to_clinical_norms': compare_to_clinical_norms,
}


def execute_tool(name: str, arguments: Dict) -> str:
    """
    Execute a tool by name with given arguments, return a JSON string.

    This is the central dispatch function called by the agent loop
    (agent_loop.py) for every LLM tool call. It:
    1. Looks up the tool function in TOOL_FUNCTIONS
    2. Calls it with the provided keyword arguments
    3. Serializes the result to a JSON string
    4. Catches any exceptions and returns them as JSON error messages

    Parameters
    ----------
    name : str
        Tool function name (must match a key in TOOL_FUNCTIONS).
    arguments : dict
        Keyword arguments to pass to the tool function.

    Returns
    -------
    str : JSON-serialized result dict.
    """
    if name not in TOOL_FUNCTIONS:
        return json.dumps({'error': f'Unknown tool: {name}'})
    try:
        # Call the tool function with unpacked keyword arguments.
        # The ** unpacking means the LLM's JSON arguments are passed
        # directly as function kwargs.
        result = TOOL_FUNCTIONS[name](**arguments)
        # default=str handles any non-serializable types (e.g., numpy
        # scalars) by converting them to strings as a fallback.
        return json.dumps(result, default=str)
    except Exception as e:
        # Catch-all for any tool execution errors. The error message is
        # returned to the LLM so it can reason about what went wrong
        # (e.g., "Sf_act_scale got an unexpected keyword argument").
        return json.dumps({'error': str(e)})
