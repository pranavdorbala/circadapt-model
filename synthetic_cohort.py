#!/usr/bin/env python3
"""
Synthetic Cohort Generator: ARIC Visit 5 -> Visit 7 Paired Data
================================================================
Paper Reference: Section 3.7 -- Synthetic Cohort Generation

This module generates a synthetic cohort of paired (V5, V7) patient vectors,
where each vector contains 100+ ARIC-compatible clinical variables computed
from the full CircAdapt cardiac model coupled with the Hallow et al. (2017)
renal physiology module. The resulting dataset is used to train the residual
neural network described in Section 3.8.

High-level pipeline:
    1. Sample patient demographics (age, sex, BSA, height) from
       distributions matching the ARIC Visit 5 population (elderly, ~50% female).
    2. Sample V5 "baseline" disease parameters (contractility, nephron
       function, inflammation, diabetes, RAAS sensitivity, TGF gain, diet).
    3. Generate correlated disease-progression deltas (V5 -> V7, ~6 years)
       using a Cholesky-decomposed correlation matrix that encodes known
       clinical co-progressions (e.g., worsening kidney <-> worsening heart).
    4. For each patient, run the coupled CircAdapt + Hallow model at V5 and
       V7 parameter sets to produce full hemodynamic + renal clinical vectors.
    5. Collect valid paired vectors into (N, D) arrays saved as .npz.

Usage:
    python synthetic_cohort.py --n_patients 10000 --n_workers 8
    python synthetic_cohort.py --n_patients 100 --n_workers 1  # quick test
"""

import os
import sys
import argparse
import time
import warnings
import copy
import numpy as np
from dataclasses import asdict
from multiprocessing import Pool
from typing import Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so we can import sibling modules
# (config.py, cardiorenal_coupling.py, emission_functions.py).
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# config.py defines:
#   TUNABLE_PARAMS  -- parameter names, valid ranges, and defaults
#   NUMERIC_VAR_NAMES -- the ordered list of ~100 numeric ARIC variable names
#   NON_NUMERIC_VARS  -- variables excluded from NN training (e.g., diastolic_label)
#   COHORT_DEFAULTS   -- default cohort generation settings (n_patients, seed, etc.)
from config import TUNABLE_PARAMS, NUMERIC_VAR_NAMES, NON_NUMERIC_VARS, COHORT_DEFAULTS

# Suppress warnings (mainly NumPy runtime warnings from edge-case hemodynamics
# and occasional CircAdapt convergence warnings).
warnings.filterwarnings('ignore')


# ===========================================================================
# Stabilized Renal Model (adapted from dashboard.py with better TGF damping)
# ===========================================================================
#
# Paper Reference: Section 3.7, "Renal Equilibration"
#
# The Hallow et al. (2017) renal model uses tubuloglomerular feedback (TGF)
# and RAAS to regulate glomerular filtration. When coupled with CircAdapt's
# cardiac output (which produces MAP values around 86-90 mmHg -- somewhat
# lower than the standalone renal model's 93 mmHg calibration point), the
# TGF loop can oscillate or diverge if damping is insufficient.
#
# This stabilized version uses a heavier damping coefficient of 0.85/0.15
# (i.e., 85% old value + 15% new value per iteration) versus the original
# 0.6/0.4. This trades convergence speed for stability, which is critical
# when running 10,000+ patients in batch -- even rare divergences corrupt
# the training dataset.
# ===========================================================================

def _update_renal_stable(r, MAP, CO, Pven, dt_hours=6.0):
    """
    Hallow renal model update with improved TGF loop damping.

    This function implements one timestep of the Hallow et al. (2017)
    renal physiology module. It takes the current cardiac hemodynamics
    (MAP, CO, Pven) as inputs from the heart model and computes:
      - Glomerular filtration rate (GFR)
      - Renal blood flow (RBF)
      - Glomerular capillary pressure (P_glom)
      - Sodium and water excretion
      - Blood volume and sodium balance updates

    The key modification vs. the original is the damping factor on line
    R_AA = 0.85*R_AA + 0.15*R_AA_new (increased from 0.6/0.4 in
    dashboard.py). This prevents oscillation of the afferent arteriolar
    resistance during TGF iteration, which can occur at CircAdapt's
    lower MAP range (~86 mmHg vs the standalone model's ~93 mmHg).

    Parameters
    ----------
    r : dict
        Renal state dictionary (modified IN-PLACE). Contains all renal
        model parameters and state variables (see _create_renal_state_circadapt).
    MAP : float
        Mean arterial pressure from the cardiac model (mmHg).
    CO : float
        Cardiac output from the cardiac model (L/min). Not directly used
        in renal calculations but available for future coupling extensions.
    Pven : float
        Central venous pressure (mmHg). Sets renal vein back-pressure.
    dt_hours : float
        Time step for volume/sodium balance integration (hours). Default
        6 hours provides smooth volume dynamics without overshooting.
    """

    # -----------------------------------------------------------------------
    # Effective Kf: the product of baseline Kf (calibrated at 8.0 nL/min/mmHg
    # per nephron to give GFR ~120 mL/min) and Kf_scale (1.0 = healthy,
    # <1.0 = nephron loss/podocyte injury in CKD).
    # -----------------------------------------------------------------------
    Kf_eff = r['Kf'] * r['Kf_scale']

    # Renal vein pressure is driven by central venous pressure (CVP).
    # Floor at 2.0 mmHg to prevent unrealistically low renal perfusion
    # pressure gradients that would cause division-by-zero in RBF calc.
    r['P_renal_vein'] = max(Pven, 2.0)

    # -----------------------------------------------------------------------
    # Step 1: RAAS (Renin-Angiotensin-Aldosterone System)
    # -----------------------------------------------------------------------
    # RAAS responds to the deviation of MAP from the kidney's setpoint.
    # When MAP drops below setpoint, RAAS_factor > 1 (vasoconstriction of
    # efferent arteriole, increased CD sodium reabsorption). When MAP is
    # above setpoint, RAAS_factor < 1 (vasodilation, natriuresis).
    #
    # The gain factor (RAAS_gain * 0.005) is empirically tuned so that a
    # 10 mmHg MAP drop produces ~7.5% increase in R_EA and eta_CD when
    # RAAS_gain = 1.5 (default). Clipped to [0.5, 2.0] to prevent extreme
    # renal hemodynamic changes.
    # -----------------------------------------------------------------------
    dMAP = MAP - r['MAP_setpoint']
    RAAS_factor = float(np.clip(1.0 - r['RAAS_gain'] * 0.005 * dMAP, 0.5, 2.0))

    # Apply RAAS to efferent arteriolar resistance and collecting duct
    # sodium reabsorption efficiency.
    R_EA = r['R_EA0'] * RAAS_factor         # Efferent arteriole resistance
    eta_CD = r['eta_CD0'] * RAAS_factor     # Collecting duct Na reabsorption fraction

    # -----------------------------------------------------------------------
    # Step 2: Tubuloglomerular Feedback (TGF) Iteration
    # -----------------------------------------------------------------------
    # TGF is a negative feedback loop: macula densa senses Na delivery ->
    # adjusts afferent arteriolar resistance (R_AA) -> changes GFR -> changes
    # Na delivery. This is solved iteratively because GFR depends on R_AA
    # and R_AA depends on GFR (via Na delivery).
    #
    # We iterate 20 times (more than the 10 in the original) to ensure
    # convergence even with heavy damping. Each iteration:
    #   a) Compute total renal resistance and renal blood flow
    #   b) Compute glomerular capillary pressure (Kirchhoff's law)
    #   c) Compute GFR via Starling equation (NFP * Kf)
    #   d) Compute macula densa Na delivery (Na after proximal tubule + loop)
    #   e) Compute TGF error signal and update R_AA with heavy damping
    # -----------------------------------------------------------------------
    R_AA = r['R_AA0']                     # Initialize afferent arteriolar resistance
    GFR = r.get('GFR', 120.0)            # Warm-start from previous GFR estimate
    Na_filt = 0.0                          # Filtered sodium (updated in loop)
    P_gc = 60.0                            # Glomerular capillary pressure (mmHg)
    RBF = 1100.0                           # Renal blood flow (mL/min)

    for _ in range(20):  # 20 TGF iterations for convergence with heavy damping
        # a) Renal blood flow: total pressure drop / total resistance.
        #    R_preAA = pre-afferent (arcuate + interlobular arteries),
        #    R_AA = afferent arteriole (TGF-regulated),
        #    R_EA = efferent arteriole (RAAS-regulated).
        #    Factor 1000 converts from mL to L (resistance in mmHg*min/L).
        R_total = r['R_preAA'] + R_AA + R_EA
        RBF = max((MAP - r['P_renal_vein']) / R_total * 1000.0, 100.0)

        # b) Renal plasma flow: RBF corrected for hematocrit.
        #    Only plasma is filtered through the glomerulus.
        RPF = RBF * (1.0 - r['Hct'])

        # c) Glomerular capillary pressure: MAP minus pressure drop across
        #    pre-afferent and afferent arteriolar resistances (Kirchhoff's law).
        #    Floor at 25 mmHg -- below this, filtration effectively ceases.
        P_gc = MAP - RBF / 1000.0 * (r['R_preAA'] + R_AA)
        P_gc = max(P_gc, 25.0)

        # d) Filtration fraction: GFR / RPF. Clamped to [0.01, 0.45] to
        #    prevent unrealistic values. Normal FF ~ 0.20.
        FF = float(np.clip(GFR / max(RPF, 1.0), 0.01, 0.45))

        # e) Average oncotic pressure along the glomerular capillary.
        #    As plasma is filtered, proteins concentrate, raising oncotic
        #    pressure. The Hallow approximation uses pi_avg = pi_plasma *
        #    (1 + FF / (2*(1-FF))), which integrates the linear rise
        #    along the capillary length. pi_plasma ~ 25 mmHg at baseline.
        pi_avg = r['pi_plasma'] * (1.0 + FF / (2.0 * (1.0 - FF)))

        # f) Net filtration pressure (NFP): the driving force for
        #    ultrafiltration. NFP = P_gc - P_Bowman - pi_oncotic.
        #    When NFP drops to zero, filtration equilibrium is reached.
        NFP = max(P_gc - r['P_Bow'] - pi_avg, 0.0)

        # g) Single-nephron GFR: Kf_eff * NFP (Starling equation).
        SNGFR = Kf_eff * NFP

        # h) Total GFR: 2 kidneys * N_nephrons * SNGFR.
        #    Factor 1e-6 converts from nL/min (per-nephron) to mL/min.
        #    Floor at 5 mL/min (severe renal failure, but not zero).
        GFR = max(2.0 * r['N_nephrons'] * SNGFR * 1e-6, 5.0)

        # i) Recalculate FF with updated GFR for Na filtration.
        FF = float(np.clip(GFR / max(RPF, 1.0), 0.01, 0.45))

        # j) Filtered sodium load (mmol/min): GFR * plasma [Na].
        #    GFR in mL/min, C_Na in mEq/L -> multiply by 1e-3 for mEq/mL.
        Na_filt = GFR * r['C_Na'] * 1e-3

        # k) Macula densa sodium delivery: Na remaining after proximal
        #    tubule (reabsorbs ~67%) and loop of Henle (reabsorbs ~25%
        #    of what PT delivers). This is the TGF sensor signal.
        MD_Na = Na_filt * (1.0 - r['eta_PT']) * (1.0 - r['eta_LoH'])

        # l) Initialize TGF setpoint on first call. The setpoint represents
        #    the "target" macula densa Na delivery that the TGF loop tries
        #    to maintain. Setting it to the first computed value means TGF
        #    error starts at zero and adapts from there.
        if r['TGF_setpoint'] <= 0:
            r['TGF_setpoint'] = MD_Na

        # m) TGF error signal: fractional deviation from setpoint.
        #    Positive TGF_err (excess Na delivery) -> raise R_AA to
        #    reduce GFR. Negative (deficit) -> lower R_AA to raise GFR.
        TGF_err = (MD_Na - r['TGF_setpoint']) / max(r['TGF_setpoint'], 1e-6)

        # n) Compute new R_AA based on TGF error. TGF_gain (default 2.0)
        #    controls how aggressively R_AA responds. Clamp to [0.5x, 3.0x]
        #    of baseline R_AA0 to prevent runaway vasoconstriction/dilation.
        R_AA_new = r['R_AA0'] * (1.0 + r['TGF_gain'] * TGF_err)
        R_AA_new = float(np.clip(R_AA_new, 0.5 * r['R_AA0'], 3.0 * r['R_AA0']))

        # o) CRITICAL DAMPING: Heavy exponential moving average (0.85/0.15)
        #    to prevent TGF oscillation. The original dashboard.py used
        #    0.6/0.4 which works at MAP ~93 but oscillates at CircAdapt's
        #    MAP ~86. The lower MAP means smaller pressure gradients and
        #    more sensitive GFR responses, requiring stronger damping.
        R_AA = 0.85 * R_AA + 0.15 * R_AA_new

    # -----------------------------------------------------------------------
    # Step 3: Tubular Sodium Handling
    # -----------------------------------------------------------------------
    # Sequential reabsorption through nephron segments:
    #   PT (proximal tubule):  ~67% of filtered Na (eta_PT = 0.67)
    #   LoH (loop of Henle):   ~25% of PT output  (eta_LoH = 0.25)
    #   DT (distal tubule):    ~5% of LoH output   (eta_DT = 0.05)
    #   CD (collecting duct):  ~2.4% of DT output  (eta_CD0 = 0.024, RAAS-modified)
    # -----------------------------------------------------------------------
    Na_after_PT = Na_filt * (1.0 - r['eta_PT'])     # ~33% passes PT
    Na_after_LoH = Na_after_PT * (1.0 - r['eta_LoH'])  # ~75% of that passes LoH
    Na_after_DT = Na_after_LoH * (1.0 - r['eta_DT'])   # ~95% of that passes DT
    Na_after_CD = Na_after_DT * (1.0 - eta_CD)          # RAAS-modified CD reabsorption

    # Pressure-natriuresis: at MAP above setpoint, the kidney excretes
    # proportionally more sodium (suppressing volume expansion). Below
    # setpoint, excretion drops (retaining volume). The asymmetric gains
    # (0.03 above vs 0.015 below) reflect the kidney's stronger natriuretic
    # response to hypertension vs. its more gradual retention in hypotension.
    # Floor at 0.3 prevents Na excretion from dropping to near-zero.
    if MAP > r['MAP_setpoint']:
        pn = 1.0 + 0.03 * (MAP - r['MAP_setpoint'])
    else:
        pn = max(0.3, 1.0 + 0.015 * (MAP - r['MAP_setpoint']))

    Na_excr_min = Na_after_CD * pn         # mEq/min of sodium excreted in urine
    Na_excr_day = Na_excr_min * 1440.0     # mEq/day (1440 min/day)

    # -----------------------------------------------------------------------
    # Step 4: Water Excretion
    # -----------------------------------------------------------------------
    # Fractional water reabsorption (frac_water_reabs = 0.99 means 99% of
    # filtered water is reabsorbed, producing ~1.2 L/day urine at GFR=120).
    water_excr_min = GFR * (1.0 - r['frac_water_reabs'])  # mL/min
    water_excr_day = water_excr_min * 1440.0 / 1000.0     # L/day

    # -----------------------------------------------------------------------
    # Step 5: Volume and Sodium Balance (Euler integration)
    # -----------------------------------------------------------------------
    # Integrate sodium and water balance over dt_hours. This is the slow
    # dynamics of the model -- over many timesteps, the body reaches a
    # steady state where intake = excretion.
    dt_min = dt_hours * 60.0                 # Convert hours -> minutes

    # Sodium balance: intake (spread evenly over 24h) minus excretion.
    Na_in_min = r['Na_intake'] / 1440.0      # mEq/min dietary Na intake
    r['Na_total'] = max(
        r['Na_total'] + (Na_in_min - Na_excr_min) * dt_min,
        800.0   # Floor: minimum body sodium content (mEq)
    )

    # Water balance: intake minus excretion, with 0.33 partition to blood.
    # The 0.33 factor reflects that only ~1/3 of total body water is in
    # the intravascular compartment (blood volume). The rest distributes
    # to interstitial and intracellular fluid.
    W_in_min = r['water_intake'] * 1000.0 / 1440.0  # mL/min water intake
    dV = (W_in_min - water_excr_min) * dt_min        # mL of water change
    r['V_blood'] = float(np.clip(
        r['V_blood'] + dV * 0.33,
        3000.0,   # Min blood volume: ~3L (severe dehydration limit)
        8000.0    # Max blood volume: ~8L (severe volume overload limit)
    ))

    # Compute extracellular fluid volume (ECF = blood / 0.33) and update
    # serum sodium concentration. Clamped to [125, 155] mEq/L to stay
    # within survivable hypo/hypernatremia bounds.
    V_ECF = r['V_blood'] / 0.33
    r['C_Na'] = float(np.clip(r['Na_total'] / (V_ECF * 1e-3), 125.0, 155.0))

    # -----------------------------------------------------------------------
    # Step 6: Store Computed Outputs
    # -----------------------------------------------------------------------
    # These are read by the emission functions to produce ARIC variables.
    r['GFR'] = round(float(GFR), 1)                    # mL/min
    r['RBF'] = round(float(RBF), 1)                    # mL/min
    r['P_glom'] = round(float(P_gc), 1)                # mmHg
    r['Na_excretion'] = round(float(Na_excr_day), 1)   # mEq/day
    r['water_excretion'] = round(float(water_excr_day), 2)  # L/day


def _create_renal_state_circadapt(na_intake=150.0, raas_gain=1.5, tgf_gain=2.0, kf_scale=1.0):
    """
    Create a renal state dictionary calibrated for CircAdapt's MAP range (~86 mmHg).

    Paper Reference: Section 3.7, "Renal Model Calibration"

    The resistance values (R_preAA=9.5, R_AA0=20.5, R_EA0=43.0) are
    specifically tuned for CircAdapt's lower MAP output (~86 mmHg), which
    differs from the standalone dashboard model (R_preAA=12, R_AA0=26).
    These lower resistances maintain physiological GFR (~120 mL/min) despite
    the lower driving pressure.

    Key calibration targets:
      - GFR = ~120 mL/min with Kf_scale=1.0 and MAP=86
      - RBF = ~1100 mL/min
      - P_glom = ~55-62 mmHg
      - Na excretion = Na intake at steady state

    Parameters
    ----------
    na_intake : float
        Dietary sodium intake in mEq/day. Default 150 = typical Western diet.
    raas_gain : float
        RAAS feedback sensitivity. Default 1.5 = moderate responsiveness.
    tgf_gain : float
        Tubuloglomerular feedback gain. Default 2.0 = normal TGF.
    kf_scale : float
        Glomerular ultrafiltration coefficient scale. 1.0 = healthy,
        <1.0 = CKD (nephron loss, podocyte damage, mesangial expansion).

    Returns
    -------
    dict
        Renal state dictionary with all parameters and initial state variables.
    """
    return {
        # ----- Structural / fixed parameters -----
        'N_nephrons': 1e6,        # Number of nephrons per kidney (normal ~1 million)
        'Kf': 8.0,               # Ultrafiltration coefficient (nL/min/mmHg per nephron),
                                  #   calibrated so 2*1e6*8.0*NFP*1e-6 ~ 120 mL/min

        # ----- Renal vascular resistances (mmHg*min/L) -----
        # These are LOWER than the standalone model (R_preAA=12, R_AA0=26)
        # to compensate for CircAdapt's lower MAP (~86 vs ~93 mmHg).
        'R_preAA': 9.5,          # Pre-afferent arteriolar resistance (arcuate arteries)
        'R_AA0': 20.5,           # Baseline afferent arteriolar resistance (TGF-regulated)
        'R_EA0': 43.0,           # Baseline efferent arteriolar resistance (RAAS-regulated)

        # ----- Pressures (mmHg) -----
        'P_Bow': 18.0,           # Bowman's capsule pressure
        'P_renal_vein': 4.0,     # Renal venous pressure (will be updated from CVP)

        # ----- Oncotic / blood composition -----
        'pi_plasma': 25.0,       # Baseline plasma oncotic pressure (mmHg)
        'Hct': 0.45,             # Hematocrit (fraction)

        # ----- Tubular reabsorption fractions -----
        # These determine what fraction of filtered Na is reabsorbed at each
        # nephron segment. Sequential: PT -> LoH -> DT -> CD.
        'eta_PT': 0.67,          # Proximal tubule: reabsorbs 67% of filtered Na
        'eta_LoH': 0.25,         # Loop of Henle: reabsorbs 25% of PT output
        'eta_DT': 0.05,          # Distal tubule: reabsorbs 5% of LoH output
        'eta_CD0': 0.024,        # Collecting duct baseline: 2.4% of DT output (RAAS-modified)
        'frac_water_reabs': 0.99,  # 99% of filtered water reabsorbed

        # ----- Intake -----
        'Na_intake': na_intake,  # Dietary Na (mEq/day)
        'water_intake': 2.0,     # Dietary water intake (L/day)

        # ----- Feedback gains -----
        'TGF_gain': tgf_gain,    # Tubuloglomerular feedback gain
        'TGF_setpoint': 0.0,     # Initialized to 0; set on first TGF iteration
        'RAAS_gain': raas_gain,  # RAAS sensitivity to MAP deviation
        'MAP_setpoint': 86.0,    # Kidney's MAP setpoint (calibrated for CircAdapt)

        # ----- State variables (initial conditions) -----
        'V_blood': 5000.0,       # Blood volume (mL), normal ~5L
        'Na_total': 2100.0,      # Total body sodium (mEq), normal ~2100
        'C_Na': 140.0,           # Serum sodium concentration (mEq/L), normal ~140

        # ----- Output variables (computed by _update_renal_stable) -----
        'GFR': 120.0,            # Glomerular filtration rate (mL/min)
        'RBF': 1100.0,           # Renal blood flow (mL/min)
        'P_glom': 60.0,          # Glomerular capillary pressure (mmHg)
        'Na_excretion': 150.0,   # Sodium excretion (mEq/day)
        'water_excretion': 1.5,  # Water excretion (L/day)

        # ----- Disease parameter -----
        'Kf_scale': kf_scale,    # 1.0 = healthy, <1 = CKD nephron loss
    }


# ===========================================================================
# Core Model Evaluation
# ===========================================================================
#
# Paper Reference: Section 3.7, "Coupled Model Evaluation"
#
# For each patient at a given parameter set, we:
#   1. Construct a CircAdapt heart model (provides full cardiac waveforms:
#      pressures, volumes, flows for all four chambers + great vessels).
#   2. Construct the Hallow renal model calibrated for CircAdapt's MAP.
#   3. Apply inflammatory and metabolic modifiers (from InflammatoryState)
#      which bidirectionally affect both cardiac and renal parameters.
#   4. Equilibrate the renal model at the cardiac hemodynamics to find
#      the coupled steady-state GFR, RBF, P_glom.
#   5. Extract 100+ ARIC-compatible clinical variables from the combined
#      cardiac waveforms + renal state using emission_functions.py.
# ===========================================================================

def evaluate_patient_state(
    params: Dict,
    demographics: Dict,
    n_coupling_steps: int = 2,
    dt_renal_hours: float = 6.0,
) -> Optional[Dict]:
    """
    Run CircAdapt heart + stabilized Hallow renal model coupled together,
    and extract ARIC-compatible clinical variables.

    This is the core "forward model" that maps disease parameters (latent
    variables) to observable clinical measurements (emission variables).
    It implements the coupled heart-kidney simulation described in
    Section 3.7 of the paper.

    Parameters
    ----------
    params : dict
        Disease parameters with keys matching TUNABLE_PARAMS:
          - Sf_act_scale : float -- active fiber stress (contractility), 1.0 = normal
          - Kf_scale : float -- glomerular filtration capacity, 1.0 = normal
          - inflammation_scale : float -- systemic inflammation, 0.0 = none
          - diabetes_scale : float -- metabolic/diabetic burden, 0.0 = none
          - k1_scale : float -- passive myocardial stiffness, 1.0 = normal
          - RAAS_gain : float -- RAAS feedback sensitivity
          - TGF_gain : float -- tubuloglomerular feedback gain
          - na_intake : float -- dietary sodium (mEq/day)
    demographics : dict
        Patient demographics:
          - age : float -- age in years
          - sex : str -- 'M' or 'F'
          - BSA : float -- body surface area (m^2)
          - height_m : float -- height in meters
    n_coupling_steps : int
        Not used (kept for API compatibility with earlier versions that
        iterated heart-kidney coupling). Renal model is equilibrated
        in 5 internal passes instead.
    dt_renal_hours : float
        Renal model time step (hours) for volume/sodium integration.

    Returns
    -------
    dict or None
        Dictionary of ~113 ARIC variable names -> float values.
        Returns None if CircAdapt fails to converge or any other error occurs
        (the patient is then excluded from the cohort).
    """
    # Lazy imports: these modules are heavy (CircAdapt loads .npy reference
    # data, emission_functions has many numpy operations). Importing inside
    # the function allows multiprocessing workers to import independently.
    from cardiorenal_coupling import (
        CircAdaptHeartModel, InflammatoryState,
        update_inflammatory_state, ML_TO_M3,
    )
    from emission_functions import extract_all_aric_variables

    try:
        # ===================================================================
        # Step 1: Create CircAdapt Heart Model
        # ===================================================================
        # CircAdapt is a lumped-parameter model of the cardiovascular system.
        # It simulates all four cardiac chambers, valves, great vessels, and
        # systemic/pulmonary circulations. The model solves for pressure-
        # volume loops, providing full waveform data (LV pressure, LV volume,
        # aortic pressure, etc.) needed to compute echocardiographic variables.
        heart = CircAdaptHeartModel()

        # InflammatoryState holds multiplicative/additive modifiers that
        # represent the effects of systemic inflammation and diabetes on
        # both cardiac and renal parameters. Initialized to no-disease state.
        ist = InflammatoryState()

        # ===================================================================
        # Step 2: Create Renal Model
        # ===================================================================
        # Initialize Hallow renal model with patient-specific parameters.
        # The renal model runs at CircAdapt's MAP (~86 mmHg) and needs
        # resistances calibrated for that lower pressure regime.
        renal = _create_renal_state_circadapt(
            na_intake=params.get('na_intake', 150.0),
            raas_gain=params.get('RAAS_gain', 1.5),
            tgf_gain=params.get('TGF_gain', 2.0),
            kf_scale=params.get('Kf_scale', 1.0),
        )

        # ===================================================================
        # Step 3: Update Inflammatory / Metabolic State
        # ===================================================================
        # The InflammatoryState computes cascading modifiers:
        #   - inflammation -> Sf_act_factor (reduces contractility),
        #     p0_factor (raises SVR), stiffness_factor (arterial stiffening),
        #     Kf_factor (reduces filtration), R_AA_factor, RAAS_gain_factor,
        #     eta_PT_offset, MAP_setpoint_offset
        #   - diabetes -> passive_k1_factor (diastolic stiffness, key for HFpEF),
        #     stiffness_factor (AGE cross-linking), Kf_factor (biphasic:
        #     early hyperfiltration then decline), R_EA_factor, eta_PT_offset
        #     (SGLT2 effect), MAP_setpoint_offset
        ist = update_inflammatory_state(
            ist,
            inflammation_scale=params.get('inflammation_scale', 0.0),
            diabetes_scale=params.get('diabetes_scale', 0.0),
        )

        # ===================================================================
        # Step 4: Apply Inflammatory Modifiers to Heart Model
        # ===================================================================
        # This adjusts CircAdapt parameters for SVR, arterial compliance,
        # and other systemic effects driven by inflammation/diabetes.
        heart.apply_inflammatory_modifiers(ist)

        # ===================================================================
        # Step 5: Apply Diastolic Stiffness
        # ===================================================================
        # Effective k1 = user-specified k1_scale * inflammatory k1 factor.
        # k1_scale > 1 represents primary HFpEF (e.g., from hypertensive
        # remodeling, amyloid, etc.). The inflammatory k1 factor adds
        # diabetes-driven stiffening (AGE cross-linking of titin/collagen).
        # The product captures both primary and secondary diastolic dysfunction.
        effective_k1 = params.get('k1_scale', 1.0) * ist.passive_k1_factor
        heart.apply_stiffness(effective_k1)

        # ===================================================================
        # Step 6: Apply Contractility Deterioration
        # ===================================================================
        # Effective Sf_act = user-specified scale * inflammatory Sf_act factor.
        # Floor at 0.20 to prevent the solver from encountering zero
        # contractility (which causes CircAdapt to fail to converge).
        effective_sf = max(
            params.get('Sf_act_scale', 1.0) * ist.Sf_act_factor, 0.20
        )
        heart.apply_deterioration(effective_sf)

        # ===================================================================
        # Step 7: Run CircAdapt to Hemodynamic Steady State
        # ===================================================================
        # CircAdapt iterates multiple cardiac cycles until pressures, volumes,
        # and flows converge (typically 20-50 beats). Returns a dict with:
        #   MAP, CO, Pven (central venous pressure), SBP, DBP, etc.
        hemo = heart.run_to_steady_state()

        # ===================================================================
        # Step 8: Equilibrate Renal Model at Cardiac Hemodynamics
        # ===================================================================
        # Run 5 passes of the renal model to find hemodynamic equilibrium
        # (GFR, P_glom, RBF) at the cardiac model's MAP/CO/CVP.
        #
        # IMPORTANT: We hold V_blood, Na_total, and C_Na fixed during
        # equilibration. This is because we are finding the HEMODYNAMIC
        # steady state (what GFR would be at these pressures), not the
        # VOLUME steady state (which would take simulated weeks). The
        # volume balance is imposed analytically: at true steady state,
        # Na_excretion = Na_intake and blood volume is stable.
        #
        # The TGF_setpoint is reset to 0 each pass so it re-calibrates
        # to the current hemodynamic conditions rather than carrying over
        # a stale setpoint from a previous iteration.
        for _ in range(5):
            renal['C_Na'] = 140.0           # Hold serum Na at normal
            renal['TGF_setpoint'] = 0.0     # Re-calibrate TGF each pass
            _update_renal_stable(renal, hemo['MAP'], hemo['CO'], hemo['Pven'], dt_renal_hours)
            renal['V_blood'] = 5000.0       # Reset blood volume (no drift)
            renal['Na_total'] = 2100.0      # Reset total body Na
            renal['C_Na'] = 140.0           # Reset serum Na

        # ===================================================================
        # Step 9: Build Renal State Dict for Emission Functions
        # ===================================================================
        # At true steady state, Na excretion equals Na intake (by definition
        # of steady state). We enforce this rather than using the model's
        # computed excretion, which may not have fully equilibrated.
        renal_state = {
            'GFR': renal['GFR'],
            'V_blood': 5000.0,              # Normal blood volume at steady state
            'C_Na': 140.0,                  # Normal serum Na at steady state
            'Na_excretion': params.get('na_intake', 150.0),  # Steady state: excr = intake
            'P_glom': renal['P_glom'],
            'Kf_scale': renal['Kf_scale'],
            'RBF': renal['RBF'],
        }

        # ===================================================================
        # Step 10: Extract ARIC Variables
        # ===================================================================
        # emission_functions.extract_all_aric_variables() takes the raw
        # CircAdapt model object (with full waveform data) and the renal
        # state dict, plus demographics, and computes ~113 clinical variables
        # matching ARIC echocardiographic, hemodynamic, and lab measurements.
        # These include LV volumes/EF, Doppler velocities, tissue Doppler,
        # filling pressures, strain, RV function, vascular coupling,
        # biomarkers (NT-proBNP, troponin, creatinine), and renal function.
        age = demographics.get('age', 75.0)
        sex = demographics.get('sex', 'M')
        BSA = demographics.get('BSA', 1.9)
        height_m = demographics.get('height_m', 1.70)

        aric_vars = extract_all_aric_variables(
            heart.model, renal_state,
            BSA=BSA, height_m=height_m, age=age, sex=sex,
        )

        return aric_vars

    except Exception as e:
        # Any failure (CircAdapt divergence, numerical issues, etc.) causes
        # this patient to be skipped. This is expected for ~1-5% of patients,
        # particularly those with extreme parameter combinations (very low
        # Sf_act + very low Kf + high inflammation). The caller handles
        # None returns by excluding the patient from the cohort.
        return None


# ===========================================================================
# Patient Sampling
# ===========================================================================
#
# Paper Reference: Section 3.7, "Patient Sampling Strategy"
#
# The synthetic cohort must cover the clinically relevant parameter space
# while maintaining realistic correlations between disease axes. The
# strategy uses:
#   1. Marginal distributions for each V5 baseline parameter chosen to
#      match the ARIC Visit 5 population demographics and disease prevalence.
#   2. A Cholesky-decomposed correlation matrix for disease PROGRESSION
#      deltas (V5 -> V7) that encodes known clinical co-progressions.
# ===========================================================================

# ---------------------------------------------------------------------------
# Correlation Matrix for Disease Progression Deltas
# ---------------------------------------------------------------------------
#
# Paper Reference: Section 3.7, "Correlated Disease Progression"
#
# This 6x6 matrix encodes the correlations between CHANGES in disease
# parameters over the ~6-year V5-to-V7 interval. The signs and magnitudes
# reflect established clinical knowledge about cardiorenal syndrome:
#
# Row/Col order: delta_Sf_act, delta_Kf, delta_inflammation,
#                delta_diabetes, delta_RAAS, delta_na_intake
#
#   [0,1] = 0.4  (Sf_act <-> Kf):
#       Heart failure and kidney disease co-progress. Patients whose
#       contractility worsens also tend to lose nephron function, and
#       vice versa. This is the hallmark of cardiorenal syndrome (CRS)
#       Types 1-4. Correlation is positive because BOTH deltas are negative
#       when disease worsens (decline in Sf_act AND decline in Kf).
#
#   [0,2] = -0.2 (Sf_act <-> inflammation):
#       Worsening contractility (negative delta_Sf_act) correlates with
#       increasing inflammation (positive delta_inflammation). The negative
#       sign reflects the opposite conventions: Sf_act decreases while
#       inflammation increases during disease progression.
#
#   [1,2] = -0.3 (Kf <-> inflammation):
#       Declining nephron function correlates with rising inflammation.
#       CKD drives a chronic inflammatory state (uremic toxins, reduced
#       clearance of inflammatory mediators), and inflammation accelerates
#       glomerulosclerosis. Stronger than [0,2] because the kidney-
#       inflammation link is more direct than heart-inflammation.
#
#   [2,3] = 0.3  (inflammation <-> diabetes):
#       Inflammation and diabetes co-progress. Metabolic syndrome involves
#       both insulin resistance and chronic low-grade inflammation (elevated
#       CRP, IL-6, TNF-alpha). Diabetes worsening drives more inflammation,
#       and inflammation worsens insulin resistance.
#
#   [1,3] = -0.2 (Kf <-> diabetes):
#       Diabetic nephropathy: worsening diabetes damages glomeruli,
#       reducing Kf. This is the leading cause of CKD worldwide.
#
#   [3,4] = 0.1  (diabetes <-> RAAS):
#       Mild correlation: worsening diabetes is associated with RAAS
#       activation (hyperglycemia stimulates intrarenal RAAS). Weak
#       because many patients are on ACE-inhibitors/ARBs.
#
#   [1,4] = -0.1 (Kf <-> RAAS):
#       CKD progression mildly correlates with RAAS activation. Nephron
#       loss triggers compensatory RAAS activation to maintain GFR.
#
#   [4,5] = 0.1  (RAAS <-> na_intake):
#       Very weak: RAAS changes and dietary Na changes are nearly
#       independent, with only a slight correlation from dietary counseling
#       that often accompanies RAAS-targeted therapy.
#
#   All other off-diagonal elements = 0.0: assumed independent.
#   Diagonal = 1.0: unit variance for standard normal draws.
# ---------------------------------------------------------------------------
_DELTA_CORR = np.array([
    [ 1.0, 0.4, -0.2, -0.1,  0.0,  0.0],  # Sf_act (contractility)
    [ 0.4, 1.0, -0.3, -0.2, -0.1,  0.0],  # Kf (nephron function)
    [-0.2,-0.3,  1.0,  0.3,  0.2,  0.0],  # inflammation
    [-0.1,-0.2,  0.3,  1.0,  0.1,  0.0],  # diabetes
    [ 0.0,-0.1,  0.2,  0.1,  1.0,  0.1],  # RAAS gain
    [ 0.0, 0.0,  0.0,  0.0,  0.1,  1.0],  # Na intake
])

# Cholesky decomposition: L such that L @ L^T = _DELTA_CORR.
# Multiplying L by a vector of independent standard normals produces
# a vector with the desired correlation structure. This is the standard
# technique for sampling from a multivariate normal with specified
# correlation matrix.
_DELTA_CHOL = np.linalg.cholesky(_DELTA_CORR)


def sample_correlated_deltas(rng: np.random.Generator) -> Dict:
    """
    Sample correlated disease progression deltas for the V5 -> V7 interval (~6 years).

    Paper Reference: Section 3.7, "Correlated Disease Progression"

    Uses Cholesky decomposition of the correlation matrix to generate
    correlated standard normal draws, then maps them to clinically
    meaningful delta magnitudes.

    The mapping from correlated normals to deltas uses:
      - abs() for deltas that should be unidirectional (disease only worsens):
        contractility declines, kidney function declines, inflammation rises,
        diabetes progresses
      - Signed values for bidirectional changes: RAAS can increase or decrease
        (e.g., if patient starts/stops ACEi), Na intake can change either way
      - Scale factors calibrated to produce realistic 6-year progression rates

    Parameters
    ----------
    rng : np.random.Generator
        Seeded random number generator for reproducibility.

    Returns
    -------
    dict
        Keys: delta_Sf_act, delta_Kf, delta_inflammation, delta_diabetes,
              delta_RAAS, delta_na
        Values: float deltas to add to V5 parameters to get V7 parameters.
    """
    # Draw 6 independent standard normal samples
    z = rng.standard_normal(6)

    # Apply Cholesky factor to induce correlations from _DELTA_CORR.
    # After this, corr_z[i] and corr_z[j] have correlation _DELTA_CORR[i,j].
    corr_z = _DELTA_CHOL @ z

    # Map correlated normals to disease progression deltas:

    # Contractility decline: abs() ensures it always decreases (negative delta).
    # Scale 0.08 means ~1 SD of decline over 6 years is 8% of Sf_act.
    # This is consistent with ~1-2% annual EF decline in HF populations.
    delta_Sf_act = -abs(corr_z[0]) * 0.08

    # Nephron function decline: abs() ensures Kf always decreases.
    # Scale 0.10 means ~1 SD of decline is 10% of Kf over 6 years,
    # consistent with ~2-4 mL/min/year eGFR decline in CKD stage 2-3.
    delta_Kf = -abs(corr_z[1]) * 0.10

    # Inflammation increase: abs() ensures it always rises.
    # Scale 0.08 means ~1 SD of increase is 0.08 on the 0-1 scale.
    # Chronic low-grade inflammation tends to worsen with aging.
    delta_inflammation = abs(corr_z[2]) * 0.08

    # Diabetes progression: abs() ensures it always worsens (HbA1c tends
    # to drift up). Scale 0.05 reflects slow metabolic deterioration.
    delta_diabetes = abs(corr_z[3]) * 0.05

    # RAAS change: can go either way (medication changes, disease progression).
    # Scale 0.15 allows moderate RAAS gain shifts.
    delta_RAAS = corr_z[4] * 0.15

    # Dietary Na change: bidirectional (patients may reduce Na on medical
    # advice, or increase due to dietary drift). Scale 15 mEq/day.
    delta_na = corr_z[5] * 15.0

    return {
        'delta_Sf_act': delta_Sf_act,
        'delta_Kf': delta_Kf,
        'delta_inflammation': delta_inflammation,
        'delta_diabetes': delta_diabetes,
        'delta_RAAS': delta_RAAS,
        'delta_na': delta_na,
    }


def generate_patient_params(rng: np.random.Generator) -> Dict:
    """
    Generate one synthetic patient's V5 and V7 parameter sets plus demographics.

    Paper Reference: Section 3.7, "Patient Sampling Strategy"

    The sampling distributions for each parameter are chosen to match the
    ARIC Visit 5 population characteristics:
      - Age 65-85 (uniform): ARIC V5 participants are elderly
      - Sex 50/50: ARIC is roughly balanced by design
      - BSA ~ N(1.9, 0.15): matches ARIC anthropometric distribution
      - Height ~ N(1.70, 0.08): matches US elderly height distribution

    Disease parameters use distributions that produce the right prevalence
    of HFpEF, CKD, diabetes, etc. in the synthetic population:
      - Sf_act ~ N(0.92, 0.12): most patients near-normal, tail into HFrEF
      - Kf ~ Beta(5, 2): right-skewed, most near-normal, tail into CKD
      - inflammation ~ Exp(0.10): most low, long tail into chronic inflammation
      - diabetes: 75% zero, 25% uniform(0.1, 0.7) -- matches ~25% prevalence
      - RAAS ~ N(1.5, 0.3): centered on default, moderate spread
      - TGF ~ N(2.0, 0.3): centered on default
      - Na intake ~ N(150, 40): typical Western diet with variation

    Parameters
    ----------
    rng : np.random.Generator
        Seeded random generator for reproducibility.

    Returns
    -------
    dict with keys:
        'v5_params' : dict of V5 disease parameters
        'v7_params' : dict of V7 disease parameters (V5 + correlated deltas)
        'demographics_v5' : dict with age, sex, BSA, height_m at Visit 5
        'demographics_v7' : dict with age, sex, BSA, height_m at Visit 7
    """

    # -----------------------------------------------------------------------
    # Demographics
    # -----------------------------------------------------------------------
    # Age: uniform over 65-85 to cover the ARIC V5 age range.
    # Uniform (not normal) because ARIC actively enrolled across age strata.
    age_v5 = rng.uniform(65, 85)

    # Sex: 50/50 coin flip. ARIC has roughly equal male/female enrollment.
    sex = 'M' if rng.random() < 0.5 else 'F'

    # BSA (body surface area): normal with mean 1.9 m^2, SD 0.15 m^2.
    # Clipped to [1.4, 2.5] to avoid unrealistic extremes.
    # Used for indexing cardiac volumes (LVEDVi, LAVi, etc.).
    BSA = float(np.clip(rng.normal(1.9, 0.15), 1.4, 2.5))

    # Height: normal with mean 1.70 m, SD 0.08 m (US elderly average).
    # Clipped to [1.50, 1.95]. Used for LV mass indexing to height^2.7.
    height_m = float(np.clip(rng.normal(1.70, 0.08), 1.50, 1.95))

    # V5 demographics (age at ARIC Visit 5)
    demographics_v5 = {'age': age_v5, 'sex': sex, 'BSA': BSA, 'height_m': height_m}

    # V7 demographics: same person 6 years later. BSA and height assumed
    # stable (elderly adults don't grow, and weight changes are captured
    # indirectly through volume/Na balance in the renal model).
    demographics_v7 = {'age': age_v5 + 6.0, 'sex': sex, 'BSA': BSA, 'height_m': height_m}

    # -----------------------------------------------------------------------
    # V5 Baseline Disease Parameters
    # -----------------------------------------------------------------------

    # Sf_act_scale (active fiber stress / contractility):
    # Normal ~ N(0.92, 0.12), clipped to [0.35, 1.0].
    # Mean 0.92 (slightly below 1.0) reflects the mild age-related decline
    # in systolic function in an elderly cohort. SD 0.12 places ~95% of
    # patients between 0.68 and 1.0, with a tail down to 0.35 (moderate HFrEF).
    Sf_act_v5 = float(np.clip(rng.normal(0.92, 0.12), 0.35, 1.0))

    # Kf_scale (glomerular filtration capacity):
    # Beta(5, 2) distribution, clipped to [0.3, 1.0].
    # Beta(5,2) has mean 0.71 and is left-skewed, producing many patients
    # with near-normal Kf (~0.8-1.0) but a substantial tail into moderate
    # CKD (0.3-0.5). This matches the high CKD prevalence in elderly ARIC.
    Kf_v5 = float(np.clip(rng.beta(5, 2), 0.3, 1.0))

    # inflammation_scale:
    # Exponential(rate=1/0.10) = Exp(mean=0.10), clipped to [0.0, 0.6].
    # Exponential is chosen because most elderly patients have low-grade
    # inflammation (CRP slightly elevated), with a long right tail capturing
    # the subset with significant chronic inflammatory conditions. The 0.6
    # ceiling prevents extreme inflammation that would crash the model.
    inflammation_v5 = float(np.clip(rng.exponential(0.10), 0.0, 0.6))

    # diabetes_scale:
    # Mixture model: 75% zero (no diabetes) + 25% Uniform(0.1, 0.7).
    # This produces ~25% diabetes prevalence, matching ARIC V5.
    # Diabetic patients have severity uniformly distributed from mild (0.1)
    # to moderately severe (0.7). The 4-element array with 3 zeros and 1
    # nonzero implements the 75/25 mixture via random choice.
    diabetes_v5 = float(rng.choice(
        [0.0, 0.0, 0.0, np.clip(rng.uniform(0.1, 0.7), 0.0, 1.0)]
    ))

    # RAAS_gain:
    # Normal ~ N(1.5, 0.3), clipped to [0.8, 2.5].
    # Centered on the default 1.5. Lower values represent patients on
    # ACE-inhibitors/ARBs (pharmacologically suppressed RAAS). Higher
    # values represent hyperactive RAAS (untreated HF, CKD with secondary
    # hyperaldosteronism).
    RAAS_v5 = float(np.clip(rng.normal(1.5, 0.3), 0.8, 2.5))

    # TGF_gain:
    # Normal ~ N(2.0, 0.3), clipped to [1.0, 3.5].
    # Centered on the default 2.0 (normal TGF responsiveness). Spread
    # captures inter-individual variation in macula densa sensitivity.
    TGF_v5 = float(np.clip(rng.normal(2.0, 0.3), 1.0, 3.5))

    # na_intake (dietary sodium):
    # Normal ~ N(150, 40) mEq/day, clipped to [50, 300].
    # 150 mEq/day is the average Western dietary sodium intake (~3.5g Na).
    # SD of 40 captures the wide range from low-sodium diets (~50-80 mEq)
    # to high-sodium diets (~200-300 mEq).
    na_v5 = float(np.clip(rng.normal(150, 40), 50, 300))

    # k1_scale (passive myocardial stiffness):
    # For non-diabetic patients: k1 = 1.0 (normal diastolic compliance).
    # For diabetic patients: k1 = 1.0 + diabetes_scale * 0.5.
    # This reflects the known association between diabetes and diastolic
    # dysfunction: diabetic cardiomyopathy involves AGE cross-linking of
    # collagen and titin, increasing passive myocardial stiffness. The
    # scaling factor 0.5 means a diabetes_scale of 0.7 produces k1 = 1.35
    # (moderate diastolic dysfunction). Clipped to [1.0, 2.0].
    k1_v5 = 1.0 if diabetes_v5 == 0.0 else float(np.clip(1.0 + diabetes_v5 * 0.5, 1.0, 2.0))

    # Assemble V5 parameter dict
    v5_params = {
        'Sf_act_scale': Sf_act_v5,
        'Kf_scale': Kf_v5,
        'inflammation_scale': inflammation_v5,
        'diabetes_scale': diabetes_v5,
        'k1_scale': k1_v5,
        'RAAS_gain': RAAS_v5,
        'TGF_gain': TGF_v5,
        'na_intake': na_v5,
    }

    # -----------------------------------------------------------------------
    # V5 -> V7 Disease Progression (Correlated Deltas)
    # -----------------------------------------------------------------------
    # Sample correlated deltas using the Cholesky-decomposed correlation
    # matrix. These represent ~6 years of disease progression where
    # cardiac, renal, inflammatory, and metabolic trajectories are
    # statistically coupled as observed in longitudinal CRS studies.
    deltas = sample_correlated_deltas(rng)

    # k1 progression: tied to diabetes progression because AGE-mediated
    # myocardial stiffening is driven by cumulative hyperglycemic exposure.
    # Factor 0.3 means every 0.1 increase in diabetes_scale over 6 years
    # adds 0.03 to k1_scale. Note: abs() because diabetes only worsens
    # (delta_diabetes is always positive due to abs() in sample_correlated_deltas).
    delta_k1 = abs(deltas['delta_diabetes']) * 0.3

    # -----------------------------------------------------------------------
    # V7 Parameters: V5 + Correlated Deltas, Clipped to Valid Ranges
    # -----------------------------------------------------------------------
    # Each V7 parameter is the V5 baseline plus the correlated delta,
    # clipped to the valid range defined in config.TUNABLE_PARAMS.
    # This ensures all V7 parameters remain physiologically plausible.
    v7_params = {
        # Contractility: V5 + (negative) delta, clipped to [0.2, 1.0]
        'Sf_act_scale': float(np.clip(Sf_act_v5 + deltas['delta_Sf_act'],
                                       *TUNABLE_PARAMS['Sf_act_scale']['range'])),

        # Nephron function: V5 + (negative) delta, clipped to [0.05, 1.0]
        'Kf_scale': float(np.clip(Kf_v5 + deltas['delta_Kf'],
                                   *TUNABLE_PARAMS['Kf_scale']['range'])),

        # Inflammation: V5 + (positive) delta, clipped to [0.0, 1.0]
        'inflammation_scale': float(np.clip(inflammation_v5 + deltas['delta_inflammation'],
                                            *TUNABLE_PARAMS['inflammation_scale']['range'])),

        # Diabetes: V5 + (positive) delta, clipped to [0.0, 1.0]
        'diabetes_scale': float(np.clip(diabetes_v5 + deltas['delta_diabetes'],
                                        *TUNABLE_PARAMS['diabetes_scale']['range'])),

        # Diastolic stiffness: V5 + diabetes-driven delta, clipped to [1.0, 3.0]
        'k1_scale': float(np.clip(k1_v5 + delta_k1,
                                   *TUNABLE_PARAMS['k1_scale']['range'])),

        # RAAS gain: V5 + (bidirectional) delta, clipped to [0.5, 3.0]
        'RAAS_gain': float(np.clip(RAAS_v5 + deltas['delta_RAAS'],
                                    *TUNABLE_PARAMS['RAAS_gain']['range'])),

        # TGF gain: UNCHANGED from V5 to V7. TGF gain is considered a
        # structural property of the juxtaglomerular apparatus that does
        # not change appreciably over 6 years (unlike RAAS, which can
        # change with medications and disease state).
        'TGF_gain': TGF_v5,

        # Dietary sodium: V5 + (bidirectional) delta, clipped to [50, 300]
        'na_intake': float(np.clip(na_v5 + deltas['delta_na'],
                                    *TUNABLE_PARAMS['na_intake']['range'])),
    }

    return {
        'v5_params': v5_params,
        'v7_params': v7_params,
        'demographics_v5': demographics_v5,
        'demographics_v7': demographics_v7,
    }


# ===========================================================================
# Worker Function for Multiprocessing
# ===========================================================================
#
# Paper Reference: Section 3.7, "Parallelized Cohort Generation"
#
# Each patient is processed independently: generate parameters, run V5
# model, run V7 model, convert to numeric vectors. This is embarrassingly
# parallel and distributed across n_workers processes using Python's
# multiprocessing.Pool.
# ===========================================================================

def _process_patient(args: Tuple) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
    """
    Process one patient: generate V5 and V7 ARIC variable vectors.

    This is the worker function called by multiprocessing.Pool.imap_unordered().
    Each call is independent and receives a unique seed for reproducibility.

    Parameters
    ----------
    args : tuple of (patient_idx: int, seed: int)
        patient_idx is used only for progress tracking.
        seed initializes this patient's random generator.

    Returns
    -------
    tuple of (v5_vec, v7_vec, patient_metadata) or None
        v5_vec : ndarray of shape (N_features,) -- V5 ARIC variables
        v7_vec : ndarray of shape (N_features,) -- V7 ARIC variables
        patient_metadata : dict with params and demographics
        Returns None if either V5 or V7 model evaluation fails.
    """
    patient_idx, seed = args

    # Each patient gets its own RNG seeded deterministically from the
    # master seed. This ensures reproducibility regardless of n_workers
    # or processing order.
    rng = np.random.default_rng(seed)

    # Generate V5 + V7 parameter sets and demographics
    patient = generate_patient_params(rng)

    # -----------------------------------------------------------------------
    # Evaluate V5 state: run coupled model at V5 parameters
    # -----------------------------------------------------------------------
    v5_vars = evaluate_patient_state(
        patient['v5_params'], patient['demographics_v5'],
        n_coupling_steps=COHORT_DEFAULTS['n_coupling_steps'],
        dt_renal_hours=COHORT_DEFAULTS['dt_renal_hours'],
    )
    if v5_vars is None:
        # CircAdapt failed to converge at V5 parameters; skip this patient.
        return None

    # -----------------------------------------------------------------------
    # Evaluate V7 state: run coupled model at V7 parameters
    # -----------------------------------------------------------------------
    v7_vars = evaluate_patient_state(
        patient['v7_params'], patient['demographics_v7'],
        n_coupling_steps=COHORT_DEFAULTS['n_coupling_steps'],
        dt_renal_hours=COHORT_DEFAULTS['dt_renal_hours'],
    )
    if v7_vars is None:
        # CircAdapt failed to converge at V7 parameters; skip this patient.
        return None

    # -----------------------------------------------------------------------
    # Convert variable dicts to ordered numeric vectors
    # -----------------------------------------------------------------------
    # NUMERIC_VAR_NAMES is a sorted list of ~100 variable names (from config.py).
    # Both V5 and V7 must use the same ordering for the NN to work correctly.
    # Missing variables default to 0.0 (should not happen with proper emission functions).
    v5_vec = np.array([float(v5_vars.get(k, 0.0)) for k in NUMERIC_VAR_NAMES])
    v7_vec = np.array([float(v7_vars.get(k, 0.0)) for k in NUMERIC_VAR_NAMES])

    # -----------------------------------------------------------------------
    # Sanity check: reject patients with NaN or Inf in any variable.
    # -----------------------------------------------------------------------
    # This catches rare numerical issues (e.g., division by near-zero in
    # emission functions, log of negative number in biomarker calculations).
    if np.any(~np.isfinite(v5_vec)) or np.any(~np.isfinite(v7_vec)):
        return None

    return (v5_vec, v7_vec, patient)


# ===========================================================================
# Cohort Generation
# ===========================================================================
#
# Paper Reference: Section 3.7, "Cohort Assembly"
#
# Orchestrates parallel generation of N synthetic patients, collects valid
# results, and assembles them into (N_valid, D) arrays. The default is
# N=10,000 patients with 8 workers.
# ===========================================================================

def generate_cohort(
    n_patients: int = 10000,
    seed: int = 42,
    n_workers: int = 8,
) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    Generate a full synthetic cohort of V5/V7 paired ARIC variable vectors.

    This is the main entry point for cohort generation. It:
      1. Creates a master RNG with the given seed
      2. Derives per-patient seeds for reproducibility
      3. Dispatches patient processing to a worker pool
      4. Collects and filters results (removing failed patients)
      5. Stacks valid results into NumPy arrays

    Parameters
    ----------
    n_patients : int
        Number of synthetic patients to attempt generating. Some will fail
        (~1-5% depending on parameter extremes), so the output will have
        slightly fewer patients.
    seed : int
        Master random seed for full reproducibility.
    n_workers : int
        Number of parallel worker processes. Use 1 for debugging (sequential
        execution with progress updates every 100 patients).

    Returns
    -------
    v5_array : ndarray of shape (N_valid, N_features)
        Visit 5 ARIC variables for each valid patient.
    v7_array : ndarray of shape (N_valid, N_features)
        Visit 7 ARIC variables for each valid patient.
    var_names : list of str
        Ordered column names (matches NUMERIC_VAR_NAMES from config.py).
    patient_metadata : list of dicts
        Demographics and disease parameters for each valid patient.
    """
    print(f"Generating {n_patients} synthetic patients (seed={seed}, workers={n_workers})...")

    # Master RNG: all per-patient seeds are derived from this single seed,
    # ensuring the entire cohort is reproducible.
    rng_master = np.random.default_rng(seed)

    # Generate unique seeds for each patient. Using integers in [0, 2^31)
    # ensures compatibility with np.random.default_rng.
    seeds = rng_master.integers(0, 2**31, size=n_patients)

    # Build argument list: (patient_index, seed) tuples
    args_list = [(i, int(seeds[i])) for i in range(n_patients)]

    t0 = time.time()

    if n_workers <= 1:
        # ---------------------------------------------------------------
        # Sequential mode: useful for debugging (errors surface immediately).
        # Progress updates every 100 patients with rate and ETA.
        # ---------------------------------------------------------------
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
        # ---------------------------------------------------------------
        # Parallel mode: multiprocessing Pool with imap_unordered.
        # imap_unordered is used (vs imap) because patient order doesn't
        # matter -- we shuffle the dataset during training anyway.
        # chunksize=10 batches work to reduce IPC overhead.
        # Progress updates every 200 patients.
        # ---------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Filter out failed patients (None results)
    # -----------------------------------------------------------------------
    valid_results = [r for r in results if r is not None]
    n_valid = len(valid_results)
    n_failed = n_patients - n_valid
    print(f"Done in {elapsed:.1f}s. Valid: {n_valid}/{n_patients} ({n_failed} failed)")

    # -----------------------------------------------------------------------
    # Assemble arrays: stack individual vectors into (N_valid, D) matrices
    # -----------------------------------------------------------------------
    v5_list, v7_list, meta_list = [], [], []
    for v5_vec, v7_vec, patient in valid_results:
        v5_list.append(v5_vec)
        v7_list.append(v7_vec)
        meta_list.append(patient)

    v5_array = np.stack(v5_list)   # Shape: (N_valid, N_features)
    v7_array = np.stack(v7_list)   # Shape: (N_valid, N_features)

    return v5_array, v7_array, NUMERIC_VAR_NAMES, meta_list


def load_real_aric_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    Load real ARIC V5/V7 paired data from a CSV file.

    This function provides an alternative to synthetic data generation,
    loading actual ARIC cohort data for model validation or fine-tuning.

    Expected CSV format: columns matching NUMERIC_VAR_NAMES prefixed with
    'v5_' and 'v7_' (e.g., 'v5_LVEF_pct', 'v7_LVEF_pct'). Missing columns
    are filled with zeros and a warning is printed.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing paired ARIC data.

    Returns
    -------
    Same format as generate_cohort():
        v5_array, v7_array, var_names, patient_metadata (empty list for real data)
    """
    import pandas as pd
    df = pd.read_csv(csv_path)

    # Expected column names with v5_/v7_ prefixes
    v5_cols = [f'v5_{k}' for k in NUMERIC_VAR_NAMES]
    v7_cols = [f'v7_{k}' for k in NUMERIC_VAR_NAMES]

    # Check which columns are present in the CSV
    missing_v5 = [c for c in v5_cols if c not in df.columns]
    missing_v7 = [c for c in v7_cols if c not in df.columns]
    if missing_v5 or missing_v7:
        available = [c for c in v5_cols if c in df.columns]
        print(f"Warning: {len(missing_v5)} V5 cols and {len(missing_v7)} V7 cols missing. "
              f"Using {len(available)} available columns, filling rest with 0.")

    # Initialize arrays with zeros (default for missing columns)
    v5_array = np.zeros((len(df), len(NUMERIC_VAR_NAMES)))
    v7_array = np.zeros((len(df), len(NUMERIC_VAR_NAMES)))

    # Fill in columns that exist in the CSV
    for i, var_name in enumerate(NUMERIC_VAR_NAMES):
        v5_col = f'v5_{var_name}'
        v7_col = f'v7_{var_name}'
        if v5_col in df.columns:
            v5_array[:, i] = df[v5_col].values
        if v7_col in df.columns:
            v7_array[:, i] = df[v7_col].values

    # Return with empty metadata (not available for real data)
    return v5_array, v7_array, NUMERIC_VAR_NAMES, []


# ===========================================================================
# CLI (Command-Line Interface)
# ===========================================================================
#
# Entry point for running cohort generation from the command line.
# Generates the cohort, saves to .npz, and prints summary statistics.
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic ARIC V5/V7 cohort')
    parser.add_argument('--n_patients', type=int, default=COHORT_DEFAULTS['n_patients'],
                        help='Number of patients to generate (default: 10000)')
    parser.add_argument('--seed', type=int, default=COHORT_DEFAULTS['seed'],
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--n_workers', type=int, default=COHORT_DEFAULTS['n_workers'],
                        help='Number of parallel workers (default: 8, use 1 for debugging)')
    parser.add_argument('--output', type=str, default='cohort_data.npz',
                        help='Output file path (.npz)')
    args = parser.parse_args()

    # Generate the cohort
    v5, v7, var_names, metadata = generate_cohort(
        n_patients=args.n_patients,
        seed=args.seed,
        n_workers=args.n_workers,
    )

    # Save to .npz (NumPy compressed archive)
    # Contains three arrays: v5 (N, D), v7 (N, D), var_names (D,)
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    np.savez(
        outpath,
        v5=v5, v7=v7,
        var_names=np.array(var_names),
    )
    print(f"Saved to {outpath}: v5 {v5.shape}, v7 {v7.shape}, {len(var_names)} variables")

    # Print summary statistics for the first 10 variables to verify
    # reasonable distributions (spot-check for calibration errors).
    print(f"\n{'='*60}")
    print(f"  Cohort Summary ({v5.shape[0]} patients, {v5.shape[1]} variables)")
    print(f"{'='*60}")
    for i, name in enumerate(var_names[:10]):
        print(f"  {name:30s}  V5: {v5[:,i].mean():.2f} +/- {v5[:,i].std():.2f}  "
              f"V7: {v7[:,i].mean():.2f} +/- {v7[:,i].std():.2f}")
    print(f"  ... and {len(var_names) - 10} more variables")


if __name__ == '__main__':
    main()
