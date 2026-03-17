#!/usr/bin/env python3
"""
Configuration for the Cardiorenal V5→V7 Prediction + Agentic Framework
======================================================================
(Paper Sections 3.2-3.4, 3.7, 3.9)

This module centralizes all configuration constants, metadata, and hyperparameters
used across the cardiorenal modeling pipeline. It serves as the single source of
truth for:

1. TUNABLE_PARAMS: The 8 disease-progression parameters that the LLM agent
   optimizes (Section 3.4 — "Disease Progression Parameter Space").
2. ARIC_VARIABLES: Metadata for the ~113 ARIC-compatible clinical variables
   output by the model, including normal ranges and importance weights
   (Section 3.3 — "Emission Functions: Model → Clinical Variables").
3. CLINICAL_THRESHOLDS: Guideline-based thresholds for disease staging
   (Section 3.9 — "Clinical Norm Comparison Tool").
4. LLM_CONFIG: Hyperparameters for the agentic inference engine
   (Section 3.9 — "Agent Configuration").
5. COHORT_DEFAULTS: Settings for synthetic cohort generation
   (Section 3.5 — "Synthetic Cohort Generation").
6. NN_DEFAULTS: Neural network architecture and training hyperparameters
   (Section 3.7 — "Neural Network Training").

Design principle: Every magic number in the codebase should trace back to
either (a) a clinical guideline citation, (b) a calibration experiment, or
(c) a hyperparameter search. This file documents those sources.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Tunable Disease Progression Parameters
# ═══════════════════════════════════════════════════════════════════════════════
# (Paper Section 3.4 — "Disease Progression Parameter Space")
#
# These 8 parameters define the disease state of a virtual patient. They map
# directly to mechanistic model components (CircAdapt cardiac patches,
# Hallow renal hemodynamics, InflammatoryState scaling). The LLM agent adjusts
# these parameters to reproduce a V7 clinical target from a V5 baseline.
#
# Parameter selection rationale (Section 3.4):
# - Sf_act_scale and k1_scale control the two primary HF phenotypes
#   (HFrEF = systolic dysfunction, HFpEF = diastolic dysfunction).
# - Kf_scale controls nephron function (the primary CKD mechanism).
# - inflammation_scale and diabetes_scale are systemic modifiers that
#   affect multiple organs simultaneously (multi-organ scaling).
# - RAAS_gain and TGF_gain control renal autoregulatory feedback loops.
# - na_intake is a modifiable lifestyle factor affecting volume balance.
#
# Each parameter has:
# - 'range': (min, max) — hard bounds enforced by np.clip in the tools.
#   Bounds are set to avoid model solver divergence (e.g., Sf_act_scale=0
#   would produce zero contractile force, crashing the ODE solver).
# - 'default': the value for a healthy reference patient. All defaults
#   together produce a "normal" clinical profile (LVEF ~60%, GFR ~120, etc.).
# - 'desc': human-readable description used in the LLM system prompt.

TUNABLE_PARAMS = {
    # ── Sf_act_scale ─────────────────────────────────────────────────────
    # Active fiber stress scale. This is the primary systolic function knob.
    # Maps to CircAdapt Patch[Sf_act] on the LV free wall and septal wall.
    # (Paper Section 3.2 — "Cardiac Model: Active Stress Generation")
    #
    # Range justification:
    # - Lower bound 0.2: below this, the LV cannot generate enough pressure
    #   to open the aortic valve, causing solver divergence (LVEF → 0%).
    # - Upper bound 1.0: normal contractility (no enhancement modeled).
    # - Default 1.0: healthy contractile function.
    #
    # Clinical mapping:
    # - 1.0 = normal (LVEF ~60%)
    # - 0.7-0.9 = mildly reduced contractility (LVEF ~45-55%)
    # - 0.5-0.7 = moderately reduced (LVEF ~30-45%, HFmrEF/mild HFrEF)
    # - 0.2-0.5 = severely reduced (LVEF <30%, severe HFrEF)
    'Sf_act_scale': {
        'range': (0.2, 1.0), 'default': 1.0,
        'desc': 'Active fiber stress scale (HFrEF: <1 reduces contractility). '
                'Maps to Patch[Sf_act] on LV/SV in CircAdapt.',
    },

    # ── Kf_scale ─────────────────────────────────────────────────────────
    # Glomerular ultrafiltration coefficient scale. This is the primary
    # renal function knob, representing nephron mass / podocyte health.
    # Maps to Kf in the Hallow et al. 2017 glomerular hemodynamics module.
    # (Paper Section 3.3 — "Renal Model: Glomerular Filtration")
    #
    # Range justification:
    # - Lower bound 0.05: ~5% of nephrons remaining (ESKD). Below this,
    #   the renal model cannot maintain Na balance, causing divergence.
    # - Upper bound 1.0: full nephron complement.
    # - Default 1.0: healthy kidneys (GFR ~120 mL/min with Kf=8.0 nL/s/mmHg).
    #
    # Clinical mapping:
    # - 1.0 = normal (eGFR >90, CKD G1)
    # - 0.6-0.9 = mild CKD (eGFR 60-89, CKD G2)
    # - 0.3-0.6 = moderate CKD (eGFR 30-59, CKD G3)
    # - 0.1-0.3 = severe CKD (eGFR 15-29, CKD G4)
    # - 0.05-0.1 = ESKD (eGFR <15, CKD G5)
    'Kf_scale': {
        'range': (0.05, 1.0), 'default': 1.0,
        'desc': 'Glomerular ultrafiltration coefficient (CKD: <1 = nephron loss, '
                'podocyte injury, mesangial expansion).',
    },

    # ── inflammation_scale ───────────────────────────────────────────────
    # Systemic inflammation index. This is a composite parameter that
    # drives multi-organ effects via the InflammatoryState class.
    # (Paper Section 3.4 — "Inflammatory and Metabolic Scaling")
    #
    # The InflammatoryState maps inflammation_scale to 8 organ-specific
    # scaling factors using sigmoid/linear transfer functions:
    # 1. Sf_act_factor: reduces contractility (TNF-alpha → myocardial depression)
    # 2. p0_factor: increases SVR (IL-6 → endothelial dysfunction)
    # 3. stiffness_factor: increases arterial stiffness (CRP → vascular remodeling)
    # 4. Kf_factor: reduces glomerular permeability (immune complex deposition)
    # 5. R_AA_factor: increases afferent arteriole resistance
    # 6. RAAS_gain_factor: modulates RAAS sensitivity
    # 7. eta_PT_offset: reduces proximal tubule reabsorption
    # 8. MAP_setpoint_offset: shifts the pressure-natriuresis setpoint
    #
    # Range: 0.0 (no inflammation) to 1.0 (severe systemic inflammation).
    # Default 0.0: healthy baseline with no inflammatory burden.
    'inflammation_scale': {
        'range': (0.0, 1.0), 'default': 0.0,
        'desc': 'Systemic inflammation index (0=none, 1=severe). Drives via '
                'InflammatoryState: Sf_act_factor, p0_factor (SVR), stiffness_factor '
                '(arterial), Kf_factor, R_AA_factor, RAAS_gain_factor, eta_PT_offset, '
                'MAP_setpoint_offset.',
    },

    # ── diabetes_scale ───────────────────────────────────────────────────
    # Diabetes metabolic burden. Like inflammation_scale, this drives
    # multi-organ effects via InflammatoryState, but through diabetes-
    # specific pathways (AGE, SGLT2, hyperglycemia).
    # (Paper Section 3.4 — "Inflammatory and Metabolic Scaling")
    #
    # Diabetes-specific effects:
    # 1. passive_k1_factor: increases myocardial passive stiffness
    #    (AGE cross-linking of collagen → HFpEF pathway). This is the
    #    primary mechanism linking diabetes to HFpEF.
    # 2. stiffness_factor: increases arterial stiffness (AGE on elastin)
    # 3. Kf_factor: biphasic — early diabetes causes hyperfiltration
    #    (increased Kf), advanced diabetes causes nephropathy (decreased Kf)
    # 4. R_EA_factor: increases efferent arteriole resistance (intraglomerular
    #    hypertension → proteinuria)
    # 5. eta_PT_offset: increases proximal tubule reabsorption (SGLT2-mediated
    #    sodium-glucose cotransport → reduced macula densa Na delivery → TGF
    #    → afferent vasodilation → hyperfiltration)
    # 6. MAP_setpoint_offset: shifts the pressure-natriuresis curve rightward
    #
    # Range: 0.0 (no diabetes) to 1.0 (severe / longstanding diabetes).
    # Default 0.0: no metabolic burden.
    'diabetes_scale': {
        'range': (0.0, 1.0), 'default': 0.0,
        'desc': 'Diabetes metabolic burden (0=none, 1=severe). Drives via '
                'InflammatoryState: passive_k1_factor (diastolic stiffness → HFpEF), '
                'stiffness_factor (AGE arterial), Kf_factor (biphasic), R_EA_factor, '
                'eta_PT_offset (SGLT2), MAP_setpoint_offset.',
    },

    # ── k1_scale ─────────────────────────────────────────────────────────
    # Passive myocardial stiffness scale. This directly multiplies the
    # exponential EDPVR stiffness parameter (k1) on the LV free wall
    # and septal wall in CircAdapt.
    # (Paper Section 3.2 — "Cardiac Model: Exponential EDPVR")
    #
    # The EDPVR (End-Diastolic Pressure-Volume Relationship) is modeled as:
    #   P_ed = A_ed * (exp(k1 * k1_scale * strain) - 1)
    # where A_ed is the amplitude and k1 is the stiffness exponent.
    # Increasing k1_scale makes the ventricle stiffer in diastole,
    # leading to higher filling pressures (LVEDP, E/e') at the same volume.
    #
    # Range justification:
    # - Lower bound 1.0: normal stiffness (no softening modeled).
    # - Upper bound 3.0: 3x normal stiffness (severe HFpEF). Beyond this,
    #   the LV cannot fill adequately and CO drops precipitously.
    # - Default 1.0: normal diastolic compliance.
    #
    # This parameter is INDEPENDENT of diabetes_scale's passive_k1_factor.
    # The total stiffness multiplier is k1_scale * passive_k1_factor,
    # so both diabetes and direct fibrosis can increase stiffness.
    'k1_scale': {
        'range': (1.0, 3.0), 'default': 1.0,
        'desc': 'Passive myocardial stiffness scale (HFpEF: >1 increases diastolic '
                'dysfunction). Maps to Patch[k1] on LV/SV in CircAdapt.',
    },

    # ── RAAS_gain ────────────────────────────────────────────────────────
    # Renin-angiotensin-aldosterone system sensitivity. Controls how
    # aggressively the RAAS responds to changes in MAP (mean arterial
    # pressure) in the Hallow renal model.
    # (Paper Section 3.3 — "Renal Model: RAAS Module")
    #
    # In the model, RAAS activation:
    # 1. Increases efferent arteriole resistance (R_EA) via AngII →
    #    raises glomerular pressure → maintains GFR when perfusion drops
    #    but worsens proteinuria.
    # 2. Increases collecting duct Na reabsorption via aldosterone →
    #    fluid retention → volume expansion → higher preload.
    # 3. Increases systemic vascular resistance → afterload.
    #
    # Range justification:
    # - Lower bound 0.5: suppressed RAAS (simulates ACE inhibitor / ARB therapy).
    # - Upper bound 3.0: hyperactive RAAS (severe heart failure with
    #   neurohormonal activation).
    # - Default 1.5: calibrated to reproduce normal RAAS behavior in the
    #   standalone Hallow model (see CLAUDE.md calibration notes).
    'RAAS_gain': {
        'range': (0.5, 3.0), 'default': 1.5,
        'desc': 'RAAS sensitivity on HallowRenalModel. Higher = more reactive '
                'to MAP drops (renin → AngII → R_EA + aldosterone → CD reabsorption).',
    },

    # ── TGF_gain ─────────────────────────────────────────────────────────
    # Tubuloglomerular feedback gain. Controls the sensitivity of the
    # macula densa sensing mechanism in the Hallow renal model.
    # (Paper Section 3.3 — "Renal Model: TGF and Macula Densa Sensing")
    #
    # TGF mechanism: the macula densa senses Na/Cl delivery at the distal
    # tubule. High delivery → adenosine release → afferent arteriole
    # constriction → reduced GFR (negative feedback to stabilize delivery).
    #
    # Range justification:
    # - Lower bound 1.0: minimal TGF (impaired autoregulation).
    # - Upper bound 4.0: aggressive TGF (exaggerated autoregulatory response).
    # - Default 2.0: calibrated to reproduce normal autoregulatory behavior
    #   (GFR stays stable across MAP 80-160 mmHg).
    'TGF_gain': {
        'range': (1.0, 4.0), 'default': 2.0,
        'desc': 'Tubuloglomerular feedback gain. Senses macula densa Na delivery '
                '→ adjusts afferent arteriole resistance.',
    },

    # ── na_intake ────────────────────────────────────────────────────────
    # Dietary sodium intake in mEq/day. Affects the pressure-natriuresis
    # equilibrium and steady-state blood volume in the Hallow renal model.
    # (Paper Section 3.3 — "Renal Model: Pressure-Natriuresis and Volume
    #  Balance")
    #
    # In steady state, Na excretion must equal Na intake. Higher intake
    # requires higher MAP (via pressure-natriuresis) to excrete the extra
    # sodium, leading to volume expansion and hypertension.
    #
    # Range justification:
    # - Lower bound 50 mEq/day (~1.2g Na/day): very low sodium diet
    #   (therapeutic restriction for HF/HTN).
    # - Upper bound 300 mEq/day (~7g Na/day): very high sodium diet
    #   (extreme processed food consumption).
    # - Default 150 mEq/day (~3.5g Na/day): average American diet
    #   (CDC: mean US adult intake ~3.4g/day).
    'na_intake': {
        'range': (50.0, 300.0), 'default': 150.0,
        'desc': 'Dietary sodium intake (mEq/day). Affects volume balance and '
                'pressure-natriuresis.',
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# ARIC Variable Metadata
# ═══════════════════════════════════════════════════════════════════════════════
# (Paper Section 3.3 — "Emission Functions: Model → Clinical Variables")
#
# This dictionary defines the metadata for every numeric ARIC-compatible
# clinical variable output by the model (~113 variables). The keys match
# the output of extract_all_aric_variables() in emission_functions.py.
#
# Each variable has:
# - 'cat': category (for grouping in visualizations and reports).
# - 'units': physical units (for axis labels and unit checking).
# - 'normal': (low, high) normal range (for normalization in compute_error
#   and for the out-of-range check in compare_to_clinical_norms).
# - 'weight': importance weight for the NN loss function AND for the
#   agent's compute_error metric.
#
# WEIGHT SELECTION RATIONALE (Paper Section 3.7 — "Loss Function Design"):
# ─────────────────────────────────────────────────────────────────────────
# Weights range from 0.3 (least important) to 2.0 (most important).
# The weighting strategy is based on three criteria:
#
# 1. CLINICAL IMPORTANCE (highest priority):
#    Variables that clinicians use for diagnosis and treatment decisions
#    get weight 2.0. These are the "headline" variables:
#    - LVEF (2.0): primary HF classification (HFrEF vs HFpEF)
#    - GFR/eGFR (2.0): primary CKD staging
#    - E/e' average (2.0): diastolic function / filling pressure
#    - NT-proBNP (2.0): HF biomarker, prognostic
#    - GLS (2.0): subclinical systolic dysfunction, prognostic
#
# 2. MECHANISTIC SENSITIVITY (medium priority):
#    Variables that are sensitive to the disease parameters and help
#    the agent distinguish phenotypes get weight 1.0-1.5:
#    - SBP, DBP, MAP (1.0): hemodynamic state, sensitive to inflammation/RAAS
#    - CO (1.0): cardiac output, sensitive to Sf_act and volume status
#    - E/e' sep and lat (1.5-2.0): diastolic markers, sensitive to k1/diabetes
#    - PASP, mPAP (1.0): pulmonary pressures, sensitive to filling pressures
#    - serum creatinine (1.0): renal marker, sensitive to Kf
#    - UACR (1.0): proteinuria marker, sensitive to glomerular pressure
#    - hsTnT (1.0): myocardial injury marker
#
# 3. REDUNDANCY (lowest priority):
#    Variables that are mathematically derived from other variables (and
#    thus carry redundant information) get weight 0.3-0.5:
#    - LV_mass_cube_g (0.3): alternative LV mass formula (redundant with LV_mass_g)
#    - Indexed variables (0.3-0.5): LVMi, LAVi etc. are just mass/BSA
#    - Alternative Doppler measures (0.3): a_prime values (less diagnostic than e')
#    - LA phasic strains (0.3): derived from volumes
#    - RA volumes (0.3): less clinically actionable than LA
#
# NORMAL RANGE SOURCES:
# ─────────────────────
# Normal ranges are sourced from published guidelines:
# - Echocardiographic: ASE/EACVI 2015 chamber quantification guidelines
#   (Lang et al., JASE 2015; 28:1-39)
# - Diastolic function: ASE/EACVI 2016 diastolic function guidelines
#   (Nagueh et al., JASE 2016; 29:277-314)
# - Blood pressure: ACC/AHA 2017 hypertension guidelines
# - Renal: KDIGO 2012 CKD guidelines
# - Biomarkers: ESC 2021 heart failure guidelines
# - Ranges are for adults aged 60-80 (ARIC V5 age range)

ARIC_VARIABLES = {
    # ══════════════════════════════════════════════════════════════════════
    # LV Structure
    # ══════════════════════════════════════════════════════════════════════
    # These variables describe LV cavity dimensions and wall thickness,
    # measured in M-mode or 2D echocardiography. They reflect chronic
    # remodeling (hypertrophy, dilation) rather than acute hemodynamic state.
    # Weight 0.5: important for remodeling tracking but not primary
    # diagnostic variables.

    # LV internal dimension at end-diastole (M-mode or 2D).
    # Normal: 4.2-5.9 cm (ASE/EACVI 2015, Table 1).
    # Elevated in eccentric hypertrophy (volume overload, dilated CMP).
    'LVIDd_cm':           {'cat': 'LV_structure',  'units': 'cm',     'normal': (4.2, 5.9), 'weight': 0.5},

    # LV internal dimension at end-systole.
    # Normal: 2.5-4.0 cm (ASE/EACVI 2015).
    # Elevated in systolic dysfunction (HFrEF).
    'LVIDs_cm':           {'cat': 'LV_structure',  'units': 'cm',     'normal': (2.5, 4.0), 'weight': 0.5},

    # Interventricular septum thickness at end-diastole.
    # Normal: 0.6-1.1 cm (ASE/EACVI 2015).
    # Elevated in concentric hypertrophy (pressure overload, HCM).
    'IVSd_cm':            {'cat': 'LV_structure',  'units': 'cm',     'normal': (0.6, 1.1), 'weight': 0.5},

    # LV posterior wall thickness at end-diastole.
    # Normal: 0.6-1.1 cm (ASE/EACVI 2015).
    # Elevated in concentric hypertrophy.
    'LVPWd_cm':           {'cat': 'LV_structure',  'units': 'cm',     'normal': (0.6, 1.1), 'weight': 0.5},

    # LV mass calculated from ASE-recommended formula:
    # LV mass = 0.8 * 1.04 * [(LVIDd + IVSd + LVPWd)^3 - LVIDd^3] + 0.6
    # Normal: 66-150 g (sex-dependent; ARIC uses sex-specific thresholds).
    # Weight 0.5: important for LVH detection but redundant with wall thickness.
    'LV_mass_g':          {'cat': 'LV_structure',  'units': 'g',      'normal': (66, 150),  'weight': 0.5},

    # LV mass from cube method (alternative formula, less commonly used).
    # Weight 0.3: redundant with LV_mass_g.
    'LV_mass_cube_g':     {'cat': 'LV_structure',  'units': 'g',      'normal': (50, 120),  'weight': 0.3},

    # Relative wall thickness = 2 * LVPWd / LVIDd.
    # Normal: 0.22-0.42 (ASE/EACVI 2015).
    # >0.42 = concentric geometry; <0.22 = eccentric geometry.
    # Used with LV mass to classify remodeling pattern.
    'RWT':                {'cat': 'LV_structure',  'units': '',       'normal': (0.22, 0.42), 'weight': 0.5},

    # ══════════════════════════════════════════════════════════════════════
    # LV Systolic Function
    # ══════════════════════════════════════════════════════════════════════
    # These are the primary systolic function variables. LVEF is the single
    # most important variable in the entire system (weight 2.0) because it
    # determines HF phenotype classification.

    # LV end-diastolic volume (Simpson's biplane method in echo).
    # Normal: 80-150 mL (sex-dependent).
    # Elevated in volume overload, dilated CMP.
    # Weight 1.0: important for preload assessment and LV remodeling.
    'LVEDV_mL':           {'cat': 'LV_systolic',   'units': 'mL',     'normal': (80, 150),  'weight': 1.0},

    # LV end-systolic volume.
    # Normal: 25-60 mL (sex-dependent).
    # Elevated in systolic dysfunction.
    # Weight 1.0: directly determines EF (with EDV).
    'LVESV_mL':           {'cat': 'LV_systolic',   'units': 'mL',     'normal': (25, 60),   'weight': 1.0},

    # Stroke volume = EDV - ESV.
    # Normal: 50-100 mL.
    # Weight 1.0: determines cardiac output (SV * HR).
    'SV_mL':              {'cat': 'LV_systolic',   'units': 'mL',     'normal': (50, 100),  'weight': 1.0},

    # LV ejection fraction = (EDV - ESV) / EDV * 100.
    # Normal: 55-70% (ASE/EACVI 2015).
    # THIS IS THE MOST IMPORTANT VARIABLE (weight 2.0):
    # - <40% = HFrEF (ESC 2021 definition)
    # - 40-49% = HFmrEF
    # - ≥50% = HFpEF (if symptomatic with structural/diastolic abnormality)
    'LVEF_pct':           {'cat': 'LV_systolic',   'units': '%',      'normal': (55, 70),   'weight': 2.0},

    # Cardiac output = SV * HR / 1000.
    # Normal: 4.0-7.0 L/min.
    # Weight 1.0: integrative measure of cardiac pump function.
    # Reduced CO causes renal hypoperfusion → cardiorenal syndrome.
    'CO_Lmin':            {'cat': 'LV_systolic',   'units': 'L/min',  'normal': (4.0, 7.0), 'weight': 1.0},

    # Heart rate.
    # Normal: 60-100 bpm.
    # Weight 0.5: in our model HR is derived from cardiac cycle length,
    # not independently tunable. It provides context but is not a primary
    # optimization target.
    'HR_bpm':             {'cat': 'LV_systolic',   'units': 'bpm',    'normal': (60, 100),  'weight': 0.5},

    # Fractional shortening = (LVIDd - LVIDs) / LVIDd * 100.
    # Normal: 25-45%.
    # Weight 0.5: redundant with LVEF (M-mode vs. volumetric measure).
    'FS_pct':             {'cat': 'LV_systolic',   'units': '%',      'normal': (25, 45),   'weight': 0.5},

    # Global longitudinal strain (negative = shortening).
    # Normal: -22% to -16% (ASE/EACVI 2015; vendor-dependent).
    # Weight 2.0: GLS is more sensitive than LVEF for detecting subclinical
    # systolic dysfunction and has strong prognostic value. It is one of the
    # "headline" variables the agent must match accurately.
    'GLS_pct':            {'cat': 'LV_systolic',   'units': '%',      'normal': (-22, -16), 'weight': 2.0},

    # ══════════════════════════════════════════════════════════════════════
    # Mitral Inflow Doppler
    # ══════════════════════════════════════════════════════════════════════
    # These variables are measured by pulsed-wave Doppler at the mitral
    # valve tips. They reflect diastolic function and filling pressures.
    # (ASE/EACVI 2016 diastolic function guidelines)

    # Peak E-wave velocity (early diastolic filling).
    # Normal: 50-100 cm/s.
    # Elevated E velocity suggests elevated LA pressure (volume overload).
    # Weight 1.0: key diastolic function variable.
    'E_vel_cms':          {'cat': 'doppler',       'units': 'cm/s',   'normal': (50, 100),  'weight': 1.0},

    # Peak A-wave velocity (atrial contraction filling).
    # Normal: 40-80 cm/s.
    # Weight 0.5: less diagnostic than E/e' for filling pressures.
    'A_vel_cms':          {'cat': 'doppler',       'units': 'cm/s',   'normal': (40, 80),   'weight': 0.5},

    # E/A ratio = E_vel / A_vel.
    # Normal: 0.8-2.0.
    # Used in diastolic grading: <0.8 = impaired relaxation (Grade I),
    # 0.8-2.0 = normal or pseudonormal, >2.0 = restrictive (Grade III).
    # Weight 1.0: key diastolic function variable.
    'EA_ratio':           {'cat': 'doppler',       'units': '',       'normal': (0.8, 2.0), 'weight': 1.0},

    # Deceleration time of E wave.
    # Normal: 150-220 ms.
    # Short DT (<150 ms) = restrictive filling (high LA pressure).
    # Long DT (>220 ms) = impaired relaxation.
    # Weight 0.5: supportive diastolic variable.
    'DT_ms':              {'cat': 'doppler',       'units': 'ms',     'normal': (150, 220), 'weight': 0.5},

    # Isovolumic relaxation time.
    # Normal: 60-100 ms.
    # Prolonged in impaired relaxation; shortened with elevated LA pressure.
    # Weight 0.5: supportive diastolic variable.
    'IVRT_ms':            {'cat': 'doppler',       'units': 'ms',     'normal': (60, 100),  'weight': 0.5},

    # ══════════════════════════════════════════════════════════════════════
    # Tissue Doppler
    # ══════════════════════════════════════════════════════════════════════
    # Tissue Doppler imaging (TDI) measures myocardial velocities at the
    # mitral annulus. These are preload-independent markers of diastolic
    # (e') and systolic (s') function.
    # (ASE/EACVI 2016 diastolic function guidelines)

    # Septal e' velocity. Normal: 7-15 cm/s. Reduced in diastolic dysfunction.
    # Weight 1.0: key variable for E/e' calculation.
    'e_prime_sep_cms':    {'cat': 'tissue_doppler','units': 'cm/s',   'normal': (7, 15),    'weight': 1.0},

    # Lateral e' velocity. Normal: 10-20 cm/s. Reduced in diastolic dysfunction.
    # Weight 1.0: key variable for E/e' calculation.
    'e_prime_lat_cms':    {'cat': 'tissue_doppler','units': 'cm/s',   'normal': (10, 20),   'weight': 1.0},

    # Average e' = (septal + lateral) / 2. Normal: 8-17 cm/s.
    # Weight 1.0: used in E/e' average calculation.
    'e_prime_avg_cms':    {'cat': 'tissue_doppler','units': 'cm/s',   'normal': (8, 17),    'weight': 1.0},

    # Septal s' velocity. Normal: 6-10 cm/s. Reflects longitudinal systolic function.
    # Weight 0.5: supportive systolic marker (less diagnostic than GLS or LVEF).
    's_prime_sep_cms':    {'cat': 'tissue_doppler','units': 'cm/s',   'normal': (6, 10),    'weight': 0.5},

    # Lateral s' velocity. Normal: 8-14 cm/s.
    # Weight 0.5: supportive systolic marker.
    's_prime_lat_cms':    {'cat': 'tissue_doppler','units': 'cm/s',   'normal': (8, 14),    'weight': 0.5},

    # Septal a' velocity (late diastolic annular motion). Normal: 6-12 cm/s.
    # Weight 0.3: less diagnostic than e'; mainly relevant for diastolic grading.
    'a_prime_sep_cms':    {'cat': 'tissue_doppler','units': 'cm/s',   'normal': (6, 12),    'weight': 0.3},

    # Lateral a' velocity. Normal: 8-14 cm/s.
    # Weight 0.3: less diagnostic.
    'a_prime_lat_cms':    {'cat': 'tissue_doppler','units': 'cm/s',   'normal': (8, 14),    'weight': 0.3},

    # ══════════════════════════════════════════════════════════════════════
    # Filling Pressures
    # ══════════════════════════════════════════════════════════════════════
    # E/e' is the primary non-invasive marker of LV filling pressure.
    # ASE/EACVI 2016: E/e' avg >14 = elevated filling pressure.
    # These are among the most important variables (weight 1.5-2.0).

    # E/e' septal. Normal: 4-13.
    # Weight 2.0: primary diastolic dysfunction marker.
    'E_e_prime_sep':      {'cat': 'filling',       'units': '',       'normal': (4, 13),    'weight': 2.0},

    # E/e' lateral. Normal: 4-13.
    # Weight 1.5: important but slightly less reliable than septal E/e'.
    'E_e_prime_lat':      {'cat': 'filling',       'units': '',       'normal': (4, 13),    'weight': 1.5},

    # E/e' average = E / ((e'sep + e'lat) / 2). Normal: 4-13.
    # Weight 2.0: THE primary filling pressure estimate.
    # This is one of the "headline" variables the agent must match.
    'E_e_prime_avg':      {'cat': 'filling',       'units': '',       'normal': (4, 13),    'weight': 2.0},

    # Estimated left atrial pressure from E/e'.
    # Normal: 5-12 mmHg.
    # Weight 1.5: clinically interpretable estimate of LAP.
    'LAP_est_mmHg':       {'cat': 'filling',       'units': 'mmHg',   'normal': (5, 12),    'weight': 1.5},

    # ══════════════════════════════════════════════════════════════════════
    # Left Atrium
    # ══════════════════════════════════════════════════════════════════════
    # LA size reflects chronic filling pressure elevation (LA remodeling).
    # LA volume > 34 mL/m² is an HFpEF diagnostic criterion.

    # LA maximal volume (at end-systole). Normal: 22-58 mL.
    # Weight 0.5: important for LA remodeling but not primary diagnostic.
    'LAV_max_mL':         {'cat': 'LA',            'units': 'mL',     'normal': (22, 58),   'weight': 0.5},

    # LA minimal volume (at end-diastole). Normal: 8-25 mL.
    # Weight 0.3: less commonly reported than LAV max.
    'LAV_min_mL':         {'cat': 'LA',            'units': 'mL',     'normal': (8, 25),    'weight': 0.3},

    # LA pre-A volume (just before atrial contraction). Normal: 18-50 mL.
    # Weight 0.3: used for phasic LA function calculation.
    'LAV_preA_mL':        {'cat': 'LA',            'units': 'mL',     'normal': (18, 50),   'weight': 0.3},

    # LA anterior-posterior diameter. Normal: 3.0-4.5 cm.
    # Weight 0.5: traditional LA size measure (being replaced by volume).
    'LA_diameter_cm':     {'cat': 'LA',            'units': 'cm',     'normal': (3.0, 4.5), 'weight': 0.5},

    # LA total emptying fraction. Normal: 50-75%.
    # Reduced in HFpEF with atrial myopathy.
    'LA_total_EF_pct':    {'cat': 'LA',            'units': '%',      'normal': (50, 75),   'weight': 0.5},

    # LA passive emptying fraction (reservoir → conduit). Normal: 5-30%.
    # Weight 0.3: phasic function measure, less commonly used clinically.
    'LA_passive_EF_pct':  {'cat': 'LA',            'units': '%',      'normal': (5, 30),    'weight': 0.3},

    # LA active emptying fraction (pump function). Normal: 30-60%.
    # Weight 0.3: phasic function measure.
    'LA_active_EF_pct':   {'cat': 'LA',            'units': '%',      'normal': (30, 60),   'weight': 0.3},

    # LA reservoir strain (longitudinal atrial strain). Normal: 20-50%.
    # Weight 0.5: emerging prognostic marker in HFpEF.
    'LARS_pct':           {'cat': 'LA',            'units': '%',      'normal': (20, 50),   'weight': 0.5},

    # Component LA strains. These are decomposed from the overall LA strain
    # curve into reservoir, conduit, and pump phases.
    'LA_reservoir_strain_pct': {'cat': 'LA',       'units': '%',      'normal': (15, 40),   'weight': 0.5},
    # Conduit strain: passive emptying during early LV filling.
    # Weight 0.3: less commonly used.
    'LA_conduit_strain_pct':   {'cat': 'LA',       'units': '%',      'normal': (5, 20),    'weight': 0.3},
    # Pump strain: active contraction phase.
    # Weight 0.3: less commonly used.
    'LA_pump_strain_pct':      {'cat': 'LA',       'units': '%',      'normal': (10, 30),   'weight': 0.3},

    # ══════════════════════════════════════════════════════════════════════
    # Right Ventricle
    # ══════════════════════════════════════════════════════════════════════
    # RV function is clinically important in HF (RV failure = poor prognosis)
    # but is less precisely modeled in CircAdapt (simpler RV geometry).
    # Weights are moderate (0.3-0.5) reflecting both clinical importance
    # and model accuracy limitations.

    # RV end-diastolic volume. Normal: 80-150 mL.
    'RVEDV_mL':           {'cat': 'RV',            'units': 'mL',     'normal': (80, 150),  'weight': 0.5},

    # RV end-systolic volume. Normal: 25-60 mL.
    'RVESV_mL':           {'cat': 'RV',            'units': 'mL',     'normal': (25, 60),   'weight': 0.5},

    # RV ejection fraction. Normal: 45-70%.
    # Weight 0.5: important prognostic marker in HF.
    'RVEF_pct':           {'cat': 'RV',            'units': '%',      'normal': (45, 70),   'weight': 0.5},

    # RV stroke volume. Normal: 50-100 mL.
    # Weight 0.3: redundant with RVEDV/RVESV.
    'RVSV_mL':            {'cat': 'RV',            'units': 'mL',     'normal': (50, 100),  'weight': 0.3},

    # Tricuspid annular plane systolic excursion. Normal: 16-30 mm.
    # Weight 0.5: standard RV systolic function measure.
    'TAPSE_mm':           {'cat': 'RV',            'units': 'mm',     'normal': (16, 30),   'weight': 0.5},

    # RV fractional area change. Normal: 35-60%.
    # Weight 0.3: alternative RV function measure.
    'RV_FAC_pct':         {'cat': 'RV',            'units': '%',      'normal': (35, 60),   'weight': 0.3},

    # RV s' velocity (tissue Doppler at tricuspid annulus). Normal: 8-15 cm/s.
    # Weight 0.5: simple RV systolic function measure.
    'RV_s_prime_cms':     {'cat': 'RV',            'units': 'cm/s',   'normal': (8, 15),    'weight': 0.5},

    # RV free wall longitudinal strain. Normal: -30% to -18%.
    # Weight 0.5: emerging RV function measure with prognostic value.
    'RV_free_wall_strain_pct': {'cat': 'RV',       'units': '%',      'normal': (-30, -18), 'weight': 0.5},

    # RV basal diameter. Normal: 3.5-5.5 cm.
    # Weight 0.3: structural measure, less important for function.
    'RV_basal_diam_cm':   {'cat': 'RV',            'units': 'cm',     'normal': (3.5, 5.5), 'weight': 0.3},

    # ══════════════════════════════════════════════════════════════════════
    # Aortic Doppler
    # ══════════════════════════════════════════════════════════════════════
    # These variables describe flow across the aortic valve and LVOT.
    # Important for detecting aortic stenosis and calculating CO.

    # LVOT diameter. Normal: 1.8-2.6 cm.
    # Weight 0.3: structural, rarely changes acutely.
    'LVOT_diam_cm':       {'cat': 'aortic',        'units': 'cm',     'normal': (1.8, 2.6), 'weight': 0.3},

    # LVOT velocity-time integral. Normal: 18-28 cm.
    # Used to calculate SV = LVOT_VTI * LVOT_CSA.
    # Weight 0.5: important for CO estimation but redundant with SV.
    'LVOT_VTI_cm':        {'cat': 'aortic',        'units': 'cm',     'normal': (18, 28),   'weight': 0.5},

    # Aortic valve peak velocity. Normal: 50-150 cm/s.
    # Elevated in aortic stenosis.
    # Weight 0.5: important for valve disease detection.
    'AV_Vmax_cms':        {'cat': 'aortic',        'units': 'cm/s',   'normal': (50, 150),  'weight': 0.5},

    # Aortic valve peak and mean gradients. Normal: 0-20 mmHg and 0-10 mmHg.
    # Weight 0.3: our model does not currently simulate aortic stenosis,
    # so these are always in normal range (included for ARIC compatibility).
    'AV_peak_grad_mmHg':  {'cat': 'aortic',        'units': 'mmHg',   'normal': (0, 20),    'weight': 0.3},
    'AV_mean_grad_mmHg':  {'cat': 'aortic',        'units': 'mmHg',   'normal': (0, 10),    'weight': 0.3},

    # Aortic valve area. Normal: 2.5-5.5 cm².
    # Weight 0.3: not a primary target (no valve disease model).
    'AVA_cm2':            {'cat': 'aortic',        'units': 'cm2',    'normal': (2.5, 5.5), 'weight': 0.3},

    # ══════════════════════════════════════════════════════════════════════
    # Pulmonary Pressures
    # ══════════════════════════════════════════════════════════════════════
    # Pulmonary pressures reflect the backward transmission of elevated
    # LV filling pressures (post-capillary pulmonary hypertension in HF).
    # PASP >35 mmHg is a major criterion for HFpEF diagnosis.

    # Pulmonary artery systolic pressure (from RV systolic pressure model).
    # Normal: 15-30 mmHg.
    # Weight 1.0: important for HFpEF and RV failure assessment.
    'PASP_mmHg':          {'cat': 'pulmonary',     'units': 'mmHg',   'normal': (15, 30),   'weight': 1.0},

    # PASP estimated from TR Vmax via modified Bernoulli equation.
    # PASP = 4*Vmax^2 + RAP. Normal: 15-35 mmHg.
    # Weight 0.5: redundant with PASP_mmHg (just a different estimation method).
    'PASP_bernoulli_mmHg':{'cat': 'pulmonary',     'units': 'mmHg',   'normal': (15, 35),   'weight': 0.5},

    # Pulmonary artery diastolic pressure. Normal: 4-12 mmHg.
    # Weight 0.5: less commonly used than PASP clinically.
    'PADP_mmHg':          {'cat': 'pulmonary',     'units': 'mmHg',   'normal': (4, 12),    'weight': 0.5},

    # Mean pulmonary artery pressure. Normal: 8-20 mmHg.
    # mPAP >20 mmHg = pulmonary hypertension (2022 ESC/ERS guidelines).
    # Weight 1.0: key hemodynamic variable for PH diagnosis.
    'mPAP_mmHg':          {'cat': 'pulmonary',     'units': 'mmHg',   'normal': (8, 20),    'weight': 1.0},

    # Tricuspid regurgitation peak velocity. Normal: 1.5-2.8 m/s.
    # Used to estimate PASP via Bernoulli equation.
    # Weight 0.5: intermediate measure.
    'TR_Vmax_ms':         {'cat': 'pulmonary',     'units': 'm/s',    'normal': (1.5, 2.8), 'weight': 0.5},

    # Estimated right atrial pressure. Normal: 0-5 mmHg.
    # Estimated from IVC diameter and collapsibility.
    # Weight 0.5: affects PASP estimation accuracy.
    'RAP_est_mmHg':       {'cat': 'pulmonary',     'units': 'mmHg',   'normal': (0, 5),     'weight': 0.5},

    # ══════════════════════════════════════════════════════════════════════
    # Blood Pressure
    # ══════════════════════════════════════════════════════════════════════
    # Blood pressure is both an input (demographics) and output (model
    # prediction) of the cardiorenal model. Systemic BP reflects the balance
    # between CO, SVR, arterial compliance, and renal volume regulation.

    # Systolic blood pressure. Normal: 90-140 mmHg (ACC/AHA 2017).
    # Weight 1.0: key hemodynamic variable, sensitive to SVR and CO.
    'SBP_mmHg':           {'cat': 'BP',            'units': 'mmHg',   'normal': (90, 140),  'weight': 1.0},

    # Central (aortic) SBP. Normal: 85-130 mmHg.
    # Lower than brachial due to pulse pressure amplification.
    # Weight 0.5: less commonly measured clinically.
    'SBP_central_mmHg':   {'cat': 'BP',            'units': 'mmHg',   'normal': (85, 130),  'weight': 0.5},

    # Diastolic blood pressure. Normal: 60-90 mmHg.
    # Weight 1.0: key hemodynamic variable, sensitive to SVR and compliance.
    'DBP_mmHg':           {'cat': 'BP',            'units': 'mmHg',   'normal': (60, 90),   'weight': 1.0},

    # Central DBP. Normal: 55-85 mmHg.
    # Weight 0.5: less commonly measured.
    'DBP_central_mmHg':   {'cat': 'BP',            'units': 'mmHg',   'normal': (55, 85),   'weight': 0.5},

    # Mean arterial pressure = DBP + 1/3 * (SBP - DBP).
    # Normal: 70-105 mmHg.
    # Weight 1.0: the pressure that drives organ perfusion. Critical for
    # the renal model (MAP → pressure-natriuresis → volume balance).
    'MAP_mmHg':           {'cat': 'BP',            'units': 'mmHg',   'normal': (70, 105),  'weight': 1.0},

    # Pulse pressure = SBP - DBP. Normal: 25-60 mmHg.
    # Elevated in arterial stiffness (isolated systolic HTN in elderly).
    # Weight 0.5: marker of arterial stiffness.
    'pulse_pressure_mmHg':{'cat': 'BP',            'units': 'mmHg',   'normal': (25, 60),   'weight': 0.5},

    # ══════════════════════════════════════════════════════════════════════
    # Right Atrium
    # ══════════════════════════════════════════════════════════════════════
    # RA volumes reflect chronic right-sided filling pressure elevation.
    # Weight 0.3: less clinically actionable than LA measures.

    'RAV_max_mL':         {'cat': 'RA',            'units': 'mL',     'normal': (15, 40),   'weight': 0.3},
    'RAV_min_mL':         {'cat': 'RA',            'units': 'mL',     'normal': (5, 20),    'weight': 0.3},
    'RA_diameter_cm':     {'cat': 'RA',            'units': 'cm',     'normal': (3.0, 4.5), 'weight': 0.3},

    # ══════════════════════════════════════════════════════════════════════
    # Timing / Myocardial Performance Index
    # ══════════════════════════════════════════════════════════════════════
    # Cardiac cycle timing intervals. MPI (Tei index) combines systolic
    # and diastolic function into a single measure.
    # Weight 0.3: these are derived measures with moderate clinical utility.

    # Isovolumic contraction time. Normal: 30-80 ms.
    'IVCT_ms':            {'cat': 'timing',        'units': 'ms',     'normal': (30, 80),   'weight': 0.3},

    # Ejection time. Normal: 200-350 ms.
    'ET_ms':              {'cat': 'timing',        'units': 'ms',     'normal': (200, 350), 'weight': 0.3},

    # LV isovolumic relaxation time (from timing data, not Doppler).
    # Normal: 60-100 ms.
    'IVRT_lv_ms':         {'cat': 'timing',        'units': 'ms',     'normal': (60, 100),  'weight': 0.3},

    # Myocardial performance index (Tei index) = (IVCT + IVRT) / ET.
    # Normal: 0.3-0.5. Elevated = combined systolic + diastolic dysfunction.
    'MPI_LV':             {'cat': 'timing',        'units': '',       'normal': (0.3, 0.5), 'weight': 0.3},

    # ══════════════════════════════════════════════════════════════════════
    # Myocardial Work
    # ══════════════════════════════════════════════════════════════════════
    # Non-invasive myocardial work indices derived from strain and BP.
    # (Russell et al., Eur Heart J 2012).
    # GWI integrates the LV pressure-strain loop area.

    # Global work index. Normal: 1500-2500 mmHg%.
    # Weight 0.5: emerging prognostic marker.
    'GWI_mmHgpct':        {'cat': 'myocardial_work','units': 'mmHg%', 'normal': (1500, 2500), 'weight': 0.5},

    # Global constructive work. Normal: 300-700 mmHg%.
    # Weight 0.3: component of GWI, less commonly reported standalone.
    'GCW_mmHgpct':        {'cat': 'myocardial_work','units': 'mmHg%', 'normal': (300, 700),   'weight': 0.3},

    # Global wasted work. Normal: 0-100 mmHg%.
    # Weight 0.3: component of GWI.
    'GWW_mmHgpct':        {'cat': 'myocardial_work','units': 'mmHg%', 'normal': (0, 100),     'weight': 0.3},

    # Global work efficiency = GCW / (GCW + GWW) * 100. Normal: 85-98%.
    # Weight 0.5: integrative efficiency measure.
    'GWE_pct':            {'cat': 'myocardial_work','units': '%',      'normal': (85, 98),     'weight': 0.5},

    # ══════════════════════════════════════════════════════════════════════
    # Diastolic Grade
    # ══════════════════════════════════════════════════════════════════════
    # Diastolic function grade (0=normal, 1=Grade I impaired relaxation,
    # 2=Grade II pseudonormal, 3=Grade III restrictive).
    # (ASE/EACVI 2016 algorithm-based grading)

    # Numeric diastolic grade (0-3).
    # Weight 1.0: clinically important classification.
    # Normal range (0,0) means only grade 0 is "normal."
    'diastolic_grade':    {'cat': 'diastolic',     'units': '',       'normal': (0, 0),     'weight': 1.0},

    # Number of abnormal criteria met in the 2016 ASE algorithm.
    # Normal: 0-1 (0 or 1 abnormal criteria = indeterminate or normal).
    # Weight 0.5: supportive variable for diastolic grading.
    'n_abnormal_criteria':{'cat': 'diastolic',     'units': '',       'normal': (0, 1),     'weight': 0.5},

    # ══════════════════════════════════════════════════════════════════════
    # Vascular Mechanics
    # ══════════════════════════════════════════════════════════════════════
    # These variables describe arterial load and ventricular-arterial
    # coupling. They reflect arterial stiffness, SVR, and LV mechanics.

    # Arterial elastance = ESP / SV. Normal: 1.0-2.5 mmHg/mL.
    # Elevated in HTN, arterial stiffness, vasoconstriction.
    # Weight 0.5: important for ventricular-arterial coupling analysis.
    'Ea_mmHg_mL':         {'cat': 'vascular',      'units': 'mmHg/mL','normal': (1.0, 2.5), 'weight': 0.5},

    # End-systolic elastance = ESP / ESV. Normal: 1.5-4.0 mmHg/mL.
    # Represents LV contractility (load-independent).
    # Weight 0.5: important for contractility assessment.
    'Ees_mmHg_mL':        {'cat': 'vascular',      'units': 'mmHg/mL','normal': (1.5, 4.0), 'weight': 0.5},

    # Ventricular-arterial coupling ratio = Ea / Ees. Normal: 0.3-0.8.
    # Optimal efficiency at ~0.5-0.7. >1.0 = afterload mismatch.
    # Weight 0.5: integrative coupling measure.
    'VA_coupling':        {'cat': 'vascular',      'units': '',       'normal': (0.3, 0.8), 'weight': 0.5},

    # Total arterial compliance = SV / pulse_pressure. Normal: 1.0-2.5 mL/mmHg.
    # Reduced in arterial stiffening (aging, HTN, diabetes).
    # Weight 0.5: marker of arterial stiffness.
    'C_total_mL_mmHg':    {'cat': 'vascular',      'units': 'mL/mmHg','normal': (1.0, 2.5), 'weight': 0.5},

    # Pulse wave velocity surrogate. Normal: 4-8 m/s.
    # Higher = stiffer arteries. Derived from compliance and geometry.
    # Weight 0.5: non-invasive arterial stiffness marker.
    'PWV_surrogate_ms':   {'cat': 'vascular',      'units': 'm/s',    'normal': (4, 8),     'weight': 0.5},

    # ══════════════════════════════════════════════════════════════════════
    # Body-Size-Indexed Variables
    # ══════════════════════════════════════════════════════════════════════
    # These are the standard indexed versions of cardiac measurements,
    # normalized by BSA or height^2.7. Used in clinical guidelines for
    # sex/size-independent thresholds.
    # Weight 0.3-0.5: mathematically derived from absolute measures
    # (somewhat redundant), but important for guideline-based classification.

    # LV mass index (g/m²). Normal: 43-95 (sex-dependent; ASE/EACVI 2015).
    # Weight 0.5: primary LVH criterion.
    'LVMi_g_m2':          {'cat': 'indexed',       'units': 'g/m2',   'normal': (43, 95),   'weight': 0.5},

    # LV mass indexed to height^2.7. Normal: 18-44 g/m^2.7.
    # Less affected by obesity than BSA indexing.
    # Weight 0.5: alternative LVH criterion.
    'LVMi_height_g_m27':  {'cat': 'indexed',       'units': 'g/m2.7', 'normal': (18, 44),   'weight': 0.5},

    # LV end-diastolic volume index. Normal: 40-80 mL/m².
    # Weight 0.5: size-adjusted preload measure.
    'LVEDVi_mL_m2':      {'cat': 'indexed',       'units': 'mL/m2',  'normal': (40, 80),   'weight': 0.5},

    # LV end-systolic volume index. Normal: 15-35 mL/m².
    'LVESVi_mL_m2':      {'cat': 'indexed',       'units': 'mL/m2',  'normal': (15, 35),   'weight': 0.5},

    # Stroke volume index. Normal: 25-50 mL/m².
    'SVi_mL_m2':          {'cat': 'indexed',       'units': 'mL/m2',  'normal': (25, 50),   'weight': 0.5},

    # Cardiac index. Normal: 2.5-4.0 L/min/m².
    # Weight 0.5: size-adjusted CO measure.
    'CI_L_min_m2':        {'cat': 'indexed',       'units': 'L/min/m2','normal': (2.5, 4.0), 'weight': 0.5},

    # LA volume index. Normal: 16-34 mL/m².
    # >34 mL/m² = LA dilation (HFpEF diagnostic criterion).
    # Weight 0.5: important for HFpEF diagnosis.
    'LAVi_mL_m2':         {'cat': 'indexed',       'units': 'mL/m2',  'normal': (16, 34),   'weight': 0.5},

    # RA volume index. Normal: 8-22 mL/m².
    # Weight 0.3: less commonly used.
    'RAVi_mL_m2':         {'cat': 'indexed',       'units': 'mL/m2',  'normal': (8, 22),    'weight': 0.3},

    # RV end-diastolic volume index. Normal: 40-80 mL/m².
    # Weight 0.3: less commonly used.
    'RVEDVi_mL_m2':      {'cat': 'indexed',       'units': 'mL/m2',  'normal': (40, 80),   'weight': 0.3},

    # ══════════════════════════════════════════════════════════════════════
    # Renal / Laboratory Variables
    # ══════════════════════════════════════════════════════════════════════
    # (Paper Section 3.3 — "Renal Model Outputs")
    #
    # These variables come from the Hallow renal model and associated
    # biomarker calculations. eGFR and GFR are the primary renal function
    # variables (weight 2.0), mirroring LVEF's importance for the heart.

    # Estimated GFR from CKD-EPI equation. Normal: 90-120 mL/min/1.73m².
    # THIS IS A "HEADLINE" VARIABLE (weight 2.0):
    # - ≥90 = normal (CKD G1)
    # - 60-89 = mild decrease (CKD G2)
    # - 45-59 = mild-moderate (CKD G3a)
    # - 30-44 = moderate-severe (CKD G3b)
    # - 15-29 = severe (CKD G4)
    # - <15 = kidney failure (CKD G5)
    # Source: KDIGO 2012 Clinical Practice Guideline for CKD.
    'eGFR_mL_min_173m2':  {'cat': 'renal',         'units': 'mL/min/1.73m2', 'normal': (90, 120), 'weight': 2.0},

    # Absolute GFR from the Hallow model (not body-size adjusted).
    # Normal: 90-130 mL/min.
    # Weight 2.0: the primary renal output of the mechanistic model.
    'GFR_mL_min':         {'cat': 'renal',         'units': 'mL/min', 'normal': (90, 130),  'weight': 2.0},

    # Renal blood flow. Normal: 900-1300 mL/min (~20% of CO).
    # Weight 0.5: important for renal perfusion assessment but not
    # a primary clinical measurement.
    'RBF_mL_min':         {'cat': 'renal',         'units': 'mL/min', 'normal': (900, 1300),'weight': 0.5},

    # Serum creatinine. Normal: 0.6-1.2 mg/dL.
    # Weight 1.0: the most commonly measured renal marker in clinical practice.
    # Inversely related to GFR (but nonlinearly — GFR can drop 50% before
    # creatinine rises above normal).
    'serum_creatinine_mg_dL': {'cat': 'renal',     'units': 'mg/dL',  'normal': (0.6, 1.2), 'weight': 1.0},

    # Cystatin C. Normal: 0.5-1.0 mg/L.
    # Weight 0.5: alternative GFR marker, less affected by muscle mass.
    'cystatin_C_mg_L':    {'cat': 'renal',         'units': 'mg/L',   'normal': (0.5, 1.0), 'weight': 0.5},

    # Blood urea nitrogen. Normal: 7-20 mg/dL.
    # Weight 0.5: affected by both GFR and non-renal factors (diet, catabolism).
    'BUN_mg_dL':          {'cat': 'renal',         'units': 'mg/dL',  'normal': (7, 20),    'weight': 0.5},

    # Urine albumin-to-creatinine ratio. Normal: 0-30 mg/g.
    # Weight 1.0: primary marker of glomerular barrier integrity.
    # Elevated in diabetic nephropathy, glomerular hypertension.
    # A1 (<30) = normal, A2 (30-299) = microalbuminuria,
    # A3 (≥300) = macroalbuminuria (KDIGO 2012).
    'UACR_mg_g':          {'cat': 'renal',         'units': 'mg/g',   'normal': (0, 30),    'weight': 1.0},

    # Serum sodium. Normal: 135-145 mEq/L.
    # Weight 0.5: important for volume status assessment.
    # Hyponatremia (<135) is a poor prognostic sign in HF.
    'serum_Na_mEq_L':     {'cat': 'renal',         'units': 'mEq/L',  'normal': (135, 145), 'weight': 0.5},

    # Serum potassium. Normal: 3.5-5.0 mEq/L.
    # Weight 0.5: critical for arrhythmia risk (affected by RAAS, diuretics).
    'serum_K_mEq_L':      {'cat': 'renal',         'units': 'mEq/L',  'normal': (3.5, 5.0), 'weight': 0.5},

    # Total blood volume. Normal: 4500-5500 mL (for 70-80 kg adult).
    # Weight 0.5: integrative volume status measure from the Hallow model.
    'blood_volume_mL':    {'cat': 'renal',         'units': 'mL',     'normal': (4500, 5500),'weight': 0.5},

    # Plasma volume. Normal: 2500-3200 mL.
    # Weight 0.3: component of blood volume, less commonly used clinically.
    'plasma_volume_mL':   {'cat': 'renal',         'units': 'mL',     'normal': (2500, 3200),'weight': 0.3},

    # N-terminal pro-B-type natriuretic peptide. Normal: 0-125 pg/mL.
    # THIS IS A "HEADLINE" VARIABLE (weight 2.0):
    # - <125 pg/mL = HF unlikely (ESC 2021 threshold for chronic HF)
    # - 125-450 pg/mL = HF possible (grey zone)
    # - >450 pg/mL = HF likely (diagnostic threshold)
    # NT-proBNP is released by cardiomyocytes in response to wall stress
    # (elevated filling pressures). It integrates systolic and diastolic
    # dysfunction severity into a single biomarker.
    'NTproBNP_pg_mL':     {'cat': 'renal',         'units': 'pg/mL',  'normal': (0, 125),   'weight': 2.0},

    # High-sensitivity troponin T. Normal: 0-14 ng/L.
    # Weight 1.0: marker of myocardial injury (necrosis, apoptosis).
    # Chronically elevated in HF due to ongoing myocyte loss.
    'hsTnT_ng_L':         {'cat': 'renal',         'units': 'ng/L',   'normal': (0, 14),    'weight': 1.0},

    # Glomerular capillary pressure. Normal: 45-65 mmHg.
    # Weight 0.5: internal model variable, not directly measured clinically
    # but important for understanding intraglomerular hemodynamics.
    'P_glom_mmHg':        {'cat': 'renal',         'units': 'mmHg',   'normal': (45, 65),   'weight': 0.5},

    # Renal resistive index = (peak systolic - end diastolic) / peak systolic
    # velocity in renal arteries. Normal: 0.55-0.70.
    # Weight 0.5: marker of intrarenal vascular resistance.
    # Elevated in CKD, diabetes, and hypertension.
    'renal_resistive_index': {'cat': 'renal',      'units': '',       'normal': (0.55, 0.70),'weight': 0.5},

    # Kf_scale echoed from input parameters (included in output for reference).
    # Normal: 0.8-1.0 (values below 0.8 indicate nephron loss).
    # Weight 0.5: parameter tracking, not a clinical measurement.
    'Kf_scale':           {'cat': 'renal',         'units': '',       'normal': (0.8, 1.0), 'weight': 0.5},

    # Sodium excretion rate. Normal: 100-250 mEq/day.
    # In steady state, excretion equals intake (pressure-natriuresis equilibrium).
    # Weight 0.5: reflects the renal model's volume regulation function.
    'Na_excretion_mEq_day': {'cat': 'renal',       'units': 'mEq/day','normal': (100, 250), 'weight': 0.5},
}

# ═══════════════════════════════════════════════════════════════════════════════
# Non-numeric variables excluded from NN training
# ═══════════════════════════════════════════════════════════════════════════════
# Some ARIC variables are categorical (e.g., diastolic_label = "Normal",
# "Grade I", "Grade II", "Grade III"). These cannot be used as NN inputs/outputs
# and are excluded from NUMERIC_VAR_NAMES.
NON_NUMERIC_VARS = {'diastolic_label'}

# ═══════════════════════════════════════════════════════════════════════════════
# Ordered list of numeric variable names for NN input/output
# ═══════════════════════════════════════════════════════════════════════════════
# This sorted list defines the canonical ordering of variables for:
# 1. NN input/output vectors (must be consistent between training and inference)
# 2. compute_error iteration order
# 3. Sensitivity analysis variable iteration
#
# Sorting ensures deterministic ordering regardless of dict insertion order
# (important for reproducibility across Python versions).
NUMERIC_VAR_NAMES = sorted(k for k in ARIC_VARIABLES if k not in NON_NUMERIC_VARS)
# N_FEATURES is used to set the NN input/output dimension.
N_FEATURES = len(NUMERIC_VAR_NAMES)


# ═══════════════════════════════════════════════════════════════════════════════
# Clinical Thresholds
# ═══════════════════════════════════════════════════════════════════════════════
# (Paper Section 3.9 — "Clinical Norm Comparison Tool")
#
# These thresholds are used by compare_to_clinical_norms() in agent_tools.py
# to classify model output into clinical categories. The LLM agent uses these
# classifications to identify the dominant disease phenotype and verify that
# its optimized parameters produce a clinically coherent state.
#
# Each entry is a list of (threshold, label) tuples, sorted in DESCENDING
# order. The classification algorithm walks through the list and assigns the
# first label where value >= threshold.
#
# Threshold sources:
# - LVEF: ESC 2021 HF guidelines (McDonagh et al., Eur Heart J 2021)
#   HFrEF <40%, HFmrEF 40-49%, HFpEF ≥50%
# - eGFR: KDIGO 2012 Clinical Practice Guideline for CKD
#   G1 ≥90, G2 60-89, G3a 45-59, G3b 30-44, G4 15-29, G5 <15
# - E/e' average: ASE/EACVI 2016 Diastolic Function Guidelines
#   Normal <8, Indeterminate 8-14, Elevated ≥14
# - UACR: KDIGO 2012 Albuminuria Classification
#   A1 <30, A2 30-299, A3 ≥300 mg/g
# - NT-proBNP: ESC 2021 HF guidelines
#   Normal <125, Grey zone 125-450, HF likely >450 pg/mL

CLINICAL_THRESHOLDS = {
    # ── LVEF-based HF classification ─────────────────────────────────────
    # Descending order: check ≥50 first (HFpEF), then ≥40 (HFmrEF), then ≥0 (HFrEF).
    'LVEF_pct': [
        (50.0, 'HFpEF'),       # ≥50%: preserved EF (may still be HFpEF if symptomatic)
        (40.0, 'HFmrEF'),      # 40-49%: mildly reduced EF
        (0.0,  'HFrEF'),       # <40%: reduced EF
    ],

    # ── CKD staging by eGFR ──────────────────────────────────────────────
    # KDIGO 2012 stages. G1 (≥90) is subdivided into normal/high-normal
    # only if albuminuria is present, which we handle separately.
    'eGFR_mL_min_173m2': [
        (90.0, 'normal'),      # G1: ≥90 mL/min/1.73m²
        (60.0, 'CKD_G2'),     # G2: 60-89
        (45.0, 'CKD_G3a'),    # G3a: 45-59 (mild-moderate decrease)
        (30.0, 'CKD_G3b'),    # G3b: 30-44 (moderate-severe decrease)
        (15.0, 'CKD_G4'),     # G4: 15-29 (severe decrease)
        (0.0,  'CKD_G5'),     # G5: <15 (kidney failure)
    ],

    # ── Filling pressure estimation by E/e' average ──────────────────────
    # ASE/EACVI 2016 algorithm. E/e' is the primary non-invasive marker
    # of LV filling pressure.
    'E_e_prime_avg': [
        (14.0, 'elevated_filling_pressure'),  # ≥14: elevated LAP
        (8.0,  'indeterminate'),              # 8-13: indeterminate
        (0.0,  'normal'),                     # <8: normal filling pressure
    ],

    # ── Albuminuria staging ──────────────────────────────────────────────
    # KDIGO 2012 categories. UACR reflects glomerular barrier integrity.
    'UACR_mg_g': [
        (300.0, 'macroalbuminuria'),   # A3: ≥300 mg/g (severe)
        (30.0,  'microalbuminuria'),   # A2: 30-299 mg/g (moderate)
        (0.0,   'normal'),             # A1: <30 mg/g (normal)
    ],

    # ── NT-proBNP-based HF probability ───────────────────────────────────
    # ESC 2021 diagnostic thresholds for chronic HF.
    'NTproBNP_pg_mL': [
        (450.0, 'HF_likely'),     # >450 pg/mL: HF diagnosis likely
        (125.0, 'HF_possible'),   # 125-450: grey zone, further workup needed
        (0.0,   'normal'),        # <125: HF unlikely
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Configuration
# ═══════════════════════════════════════════════════════════════════════════════
# (Paper Section 3.9 — "Agent Hyperparameters")
#
# These settings control the LLM agent's behavior during optimization.

LLM_CONFIG = {
    # Default LiteLLM model string. GPT-4o is the recommended model because:
    # 1. Strong function-calling support (accurate tool use)
    # 2. Good reasoning about physiological relationships
    # 3. Fast inference (~1-2s per turn)
    # Alternatives: "gemini/gemini-2.5-pro", "anthropic/claude-sonnet-4-20250514",
    # "ollama/llama3.1" (local, no API costs but slower convergence).
    'model': 'gpt-4o',

    # Maximum number of LLM conversation turns. Each turn may include
    # multiple tool calls. 15 turns is typically sufficient for convergence:
    # - Turns 1-2: phenotype analysis + initial parameter guess
    # - Turns 3-5: initial model runs + error computation
    # - Turns 6-10: sensitivity analysis + parameter refinement
    # - Turns 11-15: fine-tuning + convergence
    # If the agent hasn't converged by turn 10, stagnation detection may
    # trigger the Nelder-Mead fallback.
    'max_iterations': 15,

    # Convergence threshold: weighted normalized error below which we
    # declare the optimization successful. 0.05 means the average weighted
    # variable is within 5% of its normal range of the target.
    # This is aggressive but achievable for most phenotypes.
    # Lowering this (e.g., 0.02) increases accuracy but requires more
    # iterations and may trigger more Nelder-Mead fallbacks.
    'convergence_threshold': 0.05,    # 5% normalized error

    # LLM sampling temperature. 0.3 provides a balance between:
    # - Determinism (important for reproducible optimization)
    # - Exploration (some randomness helps escape local optima)
    # We found 0.0 caused repetitive stuck behavior, while >0.5 caused
    # erratic parameter jumps.
    'temperature': 0.3,

    # Maximum tokens per LLM response. 4096 allows the LLM to provide
    # detailed reasoning alongside tool calls. Reducing this saves API
    # cost but may truncate the LLM's thought process.
    'max_tokens': 4096,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data Generation Defaults
# ═══════════════════════════════════════════════════════════════════════════════
# (Paper Section 3.5 — "Synthetic Cohort Generation")
#
# These settings control the synthetic cohort generation process in
# synthetic_cohort.py. The cohort is used to train the V5→V7 NN.

COHORT_DEFAULTS = {
    # Number of synthetic patients to generate. 10,000 provides a good
    # balance between dataset size and generation time (~5000 seconds
    # at 0.5s/patient with 8 workers).
    'n_patients': 10000,

    # Random seed for reproducibility. Ensures the same cohort is generated
    # on every run (important for NN training reproducibility).
    'seed': 42,

    # Number of parallel workers for cohort generation. Each worker runs
    # an independent model evaluation. 8 = typical number of CPU cores.
    'n_workers': 8,

    # Number of bidirectional coupling steps between the cardiac and renal
    # models. 3 steps is sufficient for hemodynamic equilibration:
    # Step 1: heart → kidney (MAP, CO, CVP)
    # Step 2: kidney → heart (V_blood, SVR) → heart re-evaluated
    # Step 3: heart → kidney → verify convergence
    'n_coupling_steps': 3,

    # Renal model time step in hours. The Hallow model simulates over
    # a 6-hour period to reach steady state for the given hemodynamic inputs.
    # This is long enough for pressure-natriuresis equilibration but short
    # enough to keep computation fast.
    'dt_renal_hours': 6.0,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Neural Network Defaults
# ═══════════════════════════════════════════════════════════════════════════════
# (Paper Section 3.7 — "Neural Network Architecture and Training")
#
# These settings define the residual MLP architecture and training
# hyperparameters for the V5→V7 prediction network.

NN_DEFAULTS = {
    # Hidden layer dimension. 256 neurons per layer provides sufficient
    # capacity to learn the V5→V7 mapping across ~113 variables.
    # The architecture is: Input(113) → [Linear(256) + ReLU + Dropout] × n_blocks → Output(113)
    # with residual connections across each block.
    'hidden_dim': 256,

    # Number of residual blocks. 3 blocks = 6 linear layers total.
    # Deeper networks (5+ blocks) showed marginal improvement but increased
    # overfitting risk on our 10K-patient dataset.
    'n_blocks': 3,

    # Dropout rate for regularization. 0.1 = 10% of neurons dropped during
    # training. Applied after each ReLU activation. Helps prevent overfitting
    # on the synthetic cohort (which has limited diversity compared to real data).
    'dropout': 0.1,

    # Learning rate for AdamW optimizer. 1e-3 is a standard starting point.
    # The learning rate scheduler (ReduceLROnPlateau) will reduce this by
    # 10x if validation loss plateaus.
    'lr': 1e-3,

    # Weight decay (L2 regularization) for AdamW. 1e-4 provides mild
    # regularization to prevent large weight magnitudes.
    'weight_decay': 1e-4,

    # Maximum number of training epochs. Training typically converges
    # in 80-120 epochs; 200 is a generous upper bound with early stopping.
    'epochs': 200,

    # Mini-batch size. 256 provides good GPU utilization and stable
    # gradient estimates for our ~10K dataset (39 batches per epoch).
    'batch_size': 256,

    # Early stopping patience: stop training if validation loss has not
    # improved for 20 consecutive epochs. This prevents overfitting and
    # saves training time.
    'patience': 20,

    # Train/validation/test split fractions. 70/15/15 is standard.
    # The test set is held out for final evaluation and is never seen
    # during training or hyperparameter tuning.
    'train_frac': 0.70,
    'val_frac': 0.15,
    'test_frac': 0.15,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Core 20 Variables for Monthly Trajectory Generation
# ═══════════════════════════════════════════════════════════════════════════════
# (Section 3.7 extension — Monthly-resolution synthetic cohort)
#
# These 20 variables are the subset of 113 ARIC emission variables selected for
# monthly trajectory generation. Selection criteria:
#   (a) Directly and accurately computed from CircAdapt/Hallow internal state
#   (b) Clinically important for HFpEF/CKD diagnosis and monitoring
#   (c) Span all organ systems needed for the coupling equation
#
# Each entry includes:
#   - emission_key: the key name returned by extract_all_aric_variables()
#     (None if the variable must be computed separately from history/schedules)
#   - weight: importance weight for RL reward computation
#   - aric_v5_mean/sd: target distribution from ARIC Visit 5 (age ~75)
#   - abnormal_threshold: clinical cutoff for disease classification
#   - direction_with_disease: expected trajectory direction

CORE_VARIABLES = {
    # === LV STRUCTURE (2) ===
    "LVIDd_cm": {
        "emission_key": "LVIDd_cm",
        "emission_eq": 49,
        "source": "CircAdapt EDV",
        "weight": 0.5,
        "unit": "cm",
        "healthy_mean": 4.7, "healthy_sd": 0.4,
        "aric_v5_mean": 4.8, "aric_v5_sd": 0.5,
        "abnormal_threshold": 5.8,
        "direction_with_disease": "increase",
    },
    "LVmass_g": {
        "emission_key": "LV_mass_g",
        "emission_eq": 53,
        "source": "CircAdapt wall volumes",
        "weight": 0.5,
        "unit": "g",
        "healthy_mean": 170, "healthy_sd": 40,
        "aric_v5_mean": 180, "aric_v5_sd": 50,
        "abnormal_threshold": 224,
        "direction_with_disease": "increase",
    },

    # === LV SYSTOLIC FUNCTION (3) ===
    "LVEF_pct": {
        "emission_key": "LVEF_pct",
        "emission_eq": 58,
        "source": "CircAdapt EDV, ESV",
        "weight": 1.0,
        "unit": "%",
        "healthy_mean": 62, "healthy_sd": 5,
        "aric_v5_mean": 61, "aric_v5_sd": 6,
        "abnormal_threshold": 52,
        "direction_with_disease": "preserved_then_decrease",
    },
    "GLS_pct": {
        "emission_key": "GLS_pct",
        "emission_eq": 61,
        "source": "CircAdapt sarcomere mechanics",
        "weight": 0.8,
        "unit": "%",
        "healthy_mean": -20.0, "healthy_sd": 2.5,
        "aric_v5_mean": -18.5, "aric_v5_sd": 3.0,
        "abnormal_threshold": -16.0,
        "direction_with_disease": "increase",
    },
    "CO_L_min": {
        "emission_key": "CO_Lmin",
        "emission_eq": 60,
        "source": "CircAdapt SV, HR",
        "weight": 0.8,
        "unit": "L/min",
        "healthy_mean": 5.0, "healthy_sd": 1.0,
        "aric_v5_mean": 4.8, "aric_v5_sd": 1.1,
        "abnormal_threshold": 3.5,
        "direction_with_disease": "decrease",
    },

    # === DIASTOLIC FUNCTION (4) ===
    "E_cm_s": {
        "emission_key": "E_vel_cms",
        "emission_eq": 62,
        "source": "CircAdapt mitral valve flow waveform",
        "weight": 0.8,
        "unit": "cm/s",
        "healthy_mean": 72, "healthy_sd": 16,
        "aric_v5_mean": 68, "aric_v5_sd": 18,
        "abnormal_threshold": None,
        "direction_with_disease": "pseudonormalize",
    },
    "e_prime_cm_s": {
        "emission_key": "e_prime_avg_cms",
        "emission_eq": 66,
        "source": "CircAdapt sarcomere lengthening rate",
        "weight": 0.9,
        "unit": "cm/s",
        "healthy_mean": 10.0, "healthy_sd": 2.5,
        "aric_v5_mean": 7.5, "aric_v5_sd": 2.0,
        "abnormal_threshold": 7.0,
        "direction_with_disease": "decrease",
    },
    "E_over_e_prime": {
        "emission_key": "E_e_prime_avg",
        "emission_eq": 68,
        "source": "derived from E and e'",
        "weight": 1.0,
        "unit": "ratio",
        "healthy_mean": 8.0, "healthy_sd": 2.5,
        "aric_v5_mean": 10.0, "aric_v5_sd": 4.0,
        "abnormal_threshold": 14.0,
        "direction_with_disease": "increase",
    },
    "LVEDP_mmHg": {
        "emission_key": "LAP_est_mmHg",
        "emission_eq": 69,
        "source": "CircAdapt LV pressure (via LAP estimate)",
        "weight": 1.0,
        "unit": "mmHg",
        "healthy_mean": 10, "healthy_sd": 3,
        "aric_v5_mean": 12, "aric_v5_sd": 4,
        "abnormal_threshold": 16,
        "direction_with_disease": "increase",
    },

    # === ATRIAL / RV (2) ===
    "LAvolume_mL": {
        "emission_key": "LAV_max_mL",
        "emission_eq": 70,
        "source": "CircAdapt LA cavity",
        "weight": 0.7,
        "unit": "mL",
        "healthy_mean": 52, "healthy_sd": 15,
        "aric_v5_mean": 58, "aric_v5_sd": 18,
        "abnormal_threshold": 68,
        "direction_with_disease": "increase",
    },
    "PASP_mmHg": {
        "emission_key": "PASP_mmHg",
        "emission_eq": 74,
        "source": "CircAdapt RV pressure, CVP",
        "weight": 0.8,
        "unit": "mmHg",
        "healthy_mean": 25, "healthy_sd": 5,
        "aric_v5_mean": 28, "aric_v5_sd": 8,
        "abnormal_threshold": 35,
        "direction_with_disease": "increase",
    },

    # === HEMODYNAMICS (3) ===
    "SBP_mmHg": {
        "emission_key": "SBP_mmHg",
        "emission_eq": 75,
        "source": "CircAdapt aortic pressure",
        "weight": 0.8,
        "unit": "mmHg",
        "healthy_mean": 120, "healthy_sd": 12,
        "aric_v5_mean": 130, "aric_v5_sd": 18,
        "abnormal_threshold": 140,
        "direction_with_disease": "increase_then_decrease",
    },
    "MAP_mmHg": {
        "emission_key": "MAP_mmHg",
        "emission_eq": 77,
        "source": "CircAdapt aortic pressure",
        "weight": 0.8,
        "unit": "mmHg",
        "healthy_mean": 93, "healthy_sd": 8,
        "aric_v5_mean": 95, "aric_v5_sd": 10,
        "abnormal_threshold": None,
        "direction_with_disease": "variable",
    },
    "SVR_wood": {
        "emission_key": None,
        "emission_eq": 79,
        "source": "derived from CircAdapt hemodynamics: (MAP - CVP) / CO",
        "weight": 0.7,
        "unit": "Wood units",
        "healthy_mean": 18, "healthy_sd": 4,
        "aric_v5_mean": 20, "aric_v5_sd": 5,
        "abnormal_threshold": 25,
        "direction_with_disease": "increase",
    },

    # === RENAL (4) ===
    "eGFR_mL_min": {
        "emission_key": "eGFR_mL_min_173m2",
        "emission_eq": 85,
        "source": "Hallow GFR -> creatinine -> CKD-EPI",
        "weight": 1.0,
        "unit": "mL/min/1.73m2",
        "healthy_mean": 90, "healthy_sd": 15,
        "aric_v5_mean": 72, "aric_v5_sd": 20,
        "abnormal_threshold": 60,
        "direction_with_disease": "decrease",
    },
    "creatinine_mg_dL": {
        "emission_key": "serum_creatinine_mg_dL",
        "emission_eq": 86,
        "source": "Hallow GFR, demographics",
        "weight": 0.9,
        "unit": "mg/dL",
        "healthy_mean": 0.9, "healthy_sd": 0.2,
        "aric_v5_mean": 1.05, "aric_v5_sd": 0.3,
        "abnormal_threshold": 1.3,
        "direction_with_disease": "increase",
    },
    "UACR_mg_g": {
        "emission_key": "UACR_mg_g",
        "emission_eq": 88,
        "source": "Hallow glomerular pressure",
        "weight": 0.8,
        "unit": "mg/g",
        "healthy_mean": 10, "healthy_sd": 8,
        "aric_v5_mean": 25, "aric_v5_sd": 40,
        "abnormal_threshold": 30,
        "direction_with_disease": "increase",
        "distribution": "lognormal",
    },
    "FENa_pct": {
        "emission_key": None,
        "emission_eq": 89,
        "source": "Hallow tubular sodium handling",
        "weight": 0.7,
        "unit": "%",
        "healthy_mean": 0.8, "healthy_sd": 0.3,
        "aric_v5_mean": 1.0, "aric_v5_sd": 0.5,
        "abnormal_threshold": 2.0,
        "direction_with_disease": "increase",
    },

    # === BIOMARKERS (2) ===
    "NTproBNP_pg_mL": {
        "emission_key": "NTproBNP_pg_mL",
        "emission_eq": 90,
        "source": "CircAdapt hemodynamics + demographics",
        "weight": 1.0,
        "unit": "pg/mL",
        "healthy_mean": 50, "healthy_sd": 30,
        "aric_v5_mean": 120, "aric_v5_sd": 150,
        "abnormal_threshold": 125,
        "direction_with_disease": "increase",
        "distribution": "lognormal",
    },
    "CRP_mg_L": {
        "emission_key": None,
        "emission_eq": 92,
        "source": "Inflammation + diabetes indices",
        "weight": 0.7,
        "unit": "mg/L",
        "healthy_mean": 1.5, "healthy_sd": 1.0,
        "aric_v5_mean": 3.0, "aric_v5_sd": 4.0,
        "abnormal_threshold": 3.0,
        "direction_with_disease": "increase",
        "distribution": "lognormal",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# Measurement Noise Parameters
# ═══════════════════════════════════════════════════════════════════════════════
# Noise magnitudes from ARIC echo reproducibility studies and ASE guidelines.
# Used by synthetic_cohort_monthly.py to add realistic measurement variability.

MEASUREMENT_NOISE = {
    # Echocardiographic measures: ~5-10% CV for volumes, ~3-5% for linear dims
    "LVIDd_cm":       ("gaussian_relative", 0.04),
    "LVmass_g":       ("gaussian_relative", 0.08),
    "LVEF_pct":       ("gaussian_absolute", 3.0),
    "GLS_pct":        ("gaussian_absolute", 1.5),
    "CO_L_min":       ("gaussian_relative", 0.10),
    "E_cm_s":         ("gaussian_relative", 0.08),
    "e_prime_cm_s":   ("gaussian_relative", 0.10),
    "E_over_e_prime": ("gaussian_relative", 0.12),
    "LVEDP_mmHg":     ("gaussian_absolute", 2.0),
    "LAvolume_mL":    ("gaussian_relative", 0.10),
    "PASP_mmHg":      ("gaussian_absolute", 5.0),
    # Hemodynamics: blood pressure cuff variability
    "SBP_mmHg":       ("gaussian_absolute", 6.0),
    "MAP_mmHg":       ("gaussian_absolute", 4.0),
    "SVR_wood":       ("gaussian_relative", 0.08),
    # Renal: lab assay variability
    "eGFR_mL_min":    ("gaussian_relative", 0.05),
    "creatinine_mg_dL": ("gaussian_relative", 0.05),
    "UACR_mg_g":      ("gaussian_relative", 0.15),
    "FENa_pct":       ("gaussian_relative", 0.10),
    # Biomarkers: assay CV
    "NTproBNP_pg_mL": ("gaussian_relative", 0.08),
    "CRP_mg_L":       ("gaussian_relative", 0.10),
}


# ═══════════════════════════════════════════════════════════════════════════════
# RL Coupling Discovery Configuration
# ═══════════════════════════════════════════════════════════════════════════════
# Configuration for the attention-based RL agent that learns the inter-organ
# coupling equation between CircAdapt (heart) and Hallow (kidney) models.

# Feature names for RL observation extraction from simulator state.
# These define the 32-dim observation vector: 12 cardiac + 10 renal + 5 meta + 5 temporal.

CARDIAC_FEATURE_NAMES = [
    'MAP', 'SBP', 'DBP', 'CO', 'SV', 'EF', 'EDV', 'ESV',
    'Pven', 'HR', 'V_blood_total', 'LVEDP',
]

RENAL_FEATURE_NAMES = [
    'GFR', 'RBF', 'P_glom', 'Na_excretion', 'V_blood',
    'C_Na', 'Na_total', 'Kf_scale', 'water_excretion', 'Kf_effective',
]

META_FEATURE_NAMES = [
    'effective_Sf', 'effective_Kf', 'effective_k1',
    'inflammation_scale', 'diabetes_scale',
]

TEMPORAL_FEATURE_NAMES = [
    't_normalized', 'delta_MAP', 'delta_GFR', 'delta_EF', 'delta_Vblood',
]

RL_CONFIG = {
    # --- Observation space ---
    'cardiac_dim': len(CARDIAC_FEATURE_NAMES),   # 12
    'renal_dim': len(RENAL_FEATURE_NAMES),       # 10
    'meta_dim': len(META_FEATURE_NAMES),          # 5
    'temporal_dim': len(TEMPORAL_FEATURE_NAMES),  # 5
    'obs_dim': 32,  # 12 + 10 + 5 + 5

    # --- Action space ---
    'n_coupling_channels': 5,       # MAP, CO, Pven, V_blood, SVR_ratio
    'n_inflammatory_residuals': 10, # one per InflammatoryState modifier
    'action_dim': 15,               # 5 + 10

    # --- Message channel names ---
    'h2k_channels': ['MAP', 'CO', 'Pven'],
    'k2h_channels': ['V_blood', 'SVR_ratio'],

    # --- Coupling alpha bounds ---
    'alpha_min': 0.5,
    'alpha_max': 1.5,

    # --- Inflammatory residual bounds ---
    'residual_min': -0.3,
    'residual_max': 0.3,

    # --- Healthy message baselines (for alpha scaling reference) ---
    'baselines': {
        'MAP': 93.0,        # Mean arterial pressure [mmHg]
        'CO': 5.0,          # Cardiac output [L/min]
        'Pven': 3.0,        # Central venous pressure [mmHg]
        'V_blood': 5000.0,  # Blood volume [mL]
        'SVR_ratio': 1.0,   # Systemic vascular resistance ratio [dimensionless]
    },

    # --- Episode configuration ---
    'min_months': 72,       # 6 years
    'max_months': 120,      # 10 years
    'default_months': 96,   # 8 years
    'renal_substeps': 2,    # Inner coupling iterations per monthly RL step
    'dt_renal_substep': 6.0,  # Hours per renal sub-step (matches original simulation)

    # --- Attention architecture ---
    'embed_dim': 64,
    'n_heads': 4,
    'n_cross_layers': 2,
    'dropout': 0.1,

    # --- PPO hyperparameters ---
    'lr': 3e-4,
    'clip_ratio': 0.2,
    'entropy_coeff': 0.01,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'value_loss_coeff': 0.5,
    'max_grad_norm': 0.5,
    'n_epochs_per_update': 10,
    'batch_size': 64,
    'n_steps_per_rollout': 2048,

    # --- Reward shaping ---
    'terminal_reward_scale': 10.0,
    'coupling_reg_coeff': 0.005,    # Penalize |alpha - 1.0|^2
    'physiology_penalty_coeff': 0.01,
    'shaped_reward_scale_stage2': 0.1,  # Reduce shaped reward in Stage 2
}


# ═══════════════════════════════════════════════════════════════════════════════
# Patient Sampling Configuration — Truncated Normal Distributions
# ═══════════════════════════════════════════════════════════════════════════════
#
# Replaces bimodal (healthy/diseased) uniform splits with unimodal truncated
# normal distributions centered on clinically realistic values. This suppresses
# extreme parameter combinations that cause CircAdapt solver divergence while
# maintaining sufficient pathological diversity for RL training.
#
# Design principle: for each parameter, choose center and σ so that:
#   - The mode sits at a clinically representative value
#   - ~95% of samples fall within [center - 2σ, center + 2σ]
#   - Extreme values (near hard bounds) occur with <2.5% probability
#   - Joint extremes (multiple parameters at bounds simultaneously) are
#     exponentially unlikely under independent normals, unlike the ~4% rate
#     under the old joint-uniform scheme
#
# To tune: increase σ for more pathological diversity (more solver failures),
#          decrease σ to concentrate around mild disease (safer but less varied).

SAMPLING_CONFIG = {
    # --- Baseline disease parameters ---
    # Each entry: (center, sigma, low_bound, high_bound)
    #
    # k1_scale: LV passive stiffness multiplier. 1.0 = healthy, >1.5 = moderate HFpEF.
    # Center at 1.35 (mild disease) rather than midpoint 1.75 because the ARIC V5
    # cohort is community-dwelling and predominantly Grade I diastolic dysfunction.
    # σ=0.35 → P(k1 > 2.0) ≈ 3.2%, P(k1 > 2.3) ≈ 0.3%.
    'k1_scale':      (1.35, 0.35, 1.0, 2.5),

    # Sf_scale: active contractility. 1.0 = healthy, <0.8 = systolic impairment.
    # Already was N(0.95, 0.08). Unchanged — already well-behaved.
    'Sf_scale':      (0.95, 0.08, 0.5, 1.0),

    # Kf_scale: glomerular filtration coefficient (nephron mass). 1.0 = healthy.
    # Center at 0.80 (mild CKD Stage 2). σ=0.15 → P(Kf < 0.5) ≈ 2.3%.
    'Kf_scale':      (0.80, 0.15, 0.30, 1.0),

    # diabetes: metabolic burden index [0, 1]. Center at 0.12 (pre-diabetic).
    # σ=0.15 → P(d > 0.5) ≈ 0.6%. Uses folded normal (|N(0, σ)|) to enforce
    # non-negativity naturally rather than clipping a centered normal.
    'diabetes':      (0.12, 0.15, 0.0, 0.8),

    # inflammation: systemic inflammatory index [0, 1].
    # Uses exponential(λ=0.12) — already right-skewed with P(i > 0.4) ≈ 3.6%.
    # Not changed to normal because the exponential tail matches clinical CRP data.
    # Entry format: ('exponential', scale, low, high) to flag special handling.
    'inflammation':  ('exponential', 0.12, 0.0, 0.8),

    # RAAS_gain, TGF_gain, na_intake: already normal. Listed for completeness.
    'RAAS_gain':     (1.5, 0.3, 0.5, 3.0),
    'TGF_gain':      (2.0, 0.4, 1.0, 4.0),
    'na_intake':     (150.0, 30.0, 80.0, 250.0),

    # --- Annual progression rates ---
    # Each entry: (center, sigma, low_bound, high_bound)
    # Center = midpoint of old uniform range, σ = range/4 for 95% coverage.
    #
    # k1 annual rate: old U[0.02, 0.10]. Center 0.05 (conservative), σ=0.02.
    # Before diabetes acceleration factor.
    'k1_annual_rate':  (0.05, 0.02, 0.01, 0.12),

    # Kf annual rate: old U[0.01, 0.05]. Center 0.025, σ=0.01.
    # Before diabetes acceleration and nonlinear Kf feedback.
    'Kf_annual_rate':  (0.025, 0.01, 0.005, 0.06),

    # Sf annual rate: old U[0.002, 0.015]. Center 0.007, σ=0.003.
    'Sf_annual_rate':  (0.007, 0.003, 0.001, 0.020),

    # Diabetes annual rate (for patients with baseline d > 0.1):
    # old U[0.02, 0.06]. Center 0.035, σ=0.01.
    'd_annual_rate':   (0.035, 0.01, 0.01, 0.08),

    # Diabetes annual rate for non-diabetic patients (slow acquisition):
    'd_annual_rate_low': (0.005, 0.002, 0.001, 0.015),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Core 20 Clinical Variables for Monthly Synthetic Cohort
# ═══════════════════════════════════════════════════════════════════════════════
# (Paper Section 3.6 — Emission Functions, Monthly Resolution)
#
# Each variable maps to a specific ARIC codebook field.
# V5 = Visit 5 (2011-2013, N≈6118), V7 = Visit 7 (2018-2019, N≈3065)
# Stats are (mean, sd) from the ARIC Echo Codebooks unless noted.

CORE_20_VARIABLES = {
    # ===== LV STRUCTURE (2) =====
    "LVIDd_cm": {
        "description": "LV end-diastolic internal diameter",
        "unit": "cm",
        "aric_field_v5": "ECH.ECH4 (LVEDD)",
        "aric_field_v7": "ECH.ECH4 (LVEDD)",
        "v5_mean": 4.415, "v5_sd": 0.525, "v5_n": 6069, "v5_missing_pct": 0.8,
        "v7_mean": 4.323, "v7_sd": 0.520, "v7_n": 2998, "v7_missing_pct": 2.2,
        "v5_range": (2.67, 8.68), "v7_range": (2.30, 7.06),
        "distribution": "normal",
        "clinical_weight": 0.5,
        "physiological_bounds": (2.5, 8.0),
        "emission_eq": "Eq.49: 2*(3*EDV/(4*pi))^(1/3)",
        "emission_source": "CircAdapt EDV",
    },
    "LVmass_g": {
        "description": "LV mass",
        "unit": "g",
        "aric_field_v5": "ECH.ECH11 (LVM)",
        "aric_field_v7": "ECH.ECH11 (LVM)",
        "v5_mean": 150.043, "v5_sd": 46.674, "v5_n": 6067, "v5_missing_pct": 0.8,
        "v7_mean": 152.969, "v7_sd": 46.576, "v7_n": 2986, "v7_missing_pct": 2.6,
        "v5_range": (56.31, 484.33), "v7_range": (37.94, 445.03),
        "distribution": "normal",
        "clinical_weight": 0.5,
        "physiological_bounds": (50, 500),
        "emission_eq": "Eq.53: ASE cube formula from LVIDd, IVSd, LVPWd",
        "emission_source": "CircAdapt wall volumes + cavity dimensions",
    },
    # ===== LV SYSTOLIC FUNCTION (3) =====
    "LVEF_pct": {
        "description": "LV ejection fraction",
        "unit": "%",
        "aric_field_v5": "ECH.ECH10 (LVEF)",
        "aric_field_v7": "ECH.ECH10 (LVEF)",
        "v5_mean": 65.06, "v5_sd": 6.86, "v5_n": 5919, "v5_missing_pct": 3.3,
        "v7_mean": 63.16, "v7_sd": 7.37, "v7_n": 2893, "v7_missing_pct": 5.6,
        "v5_range": (22.8, 86.3), "v7_range": (14.0, 81.3),
        "distribution": "normal",
        "clinical_weight": 1.0,
        "physiological_bounds": (10, 85),
        "emission_eq": "Eq.58: (EDV-ESV)/EDV * 100",
        "emission_source": "CircAdapt EDV, ESV",
    },
    "GLS_pct": {
        "description": "Average peak longitudinal strain (negative = normal)",
        "unit": "%",
        "aric_field_v5": "ECH.ECH47 (AVEPLS)",
        "aric_field_v7": "ECH.ECH47 (AVEPLS)",
        "v5_mean": -17.895, "v5_sd": 2.597, "v5_n": 5698, "v5_missing_pct": 6.9,
        "v7_mean": -17.52, "v7_sd": 2.76, "v7_n": 2918, "v7_missing_pct": 4.8,
        "v5_range": (-25.79, -3.78), "v7_range": (-24.1, -4.1),
        "distribution": "normal",
        "clinical_weight": 0.8,
        "physiological_bounds": (-28, -3),
        "emission_eq": "Eq.61: (Ls_min - Ls_max)/Ls_max * 100",
        "emission_source": "CircAdapt sarcomere mechanics",
    },
    "CO_L_min": {
        "description": "Cardiac output (echo-derived)",
        "unit": "L/min",
        "aric_field_v5": "DERIVED from ECH.ECH31 (LVOTVTI) + LVOT diameter + HR",
        "aric_field_v7": "DERIVED from ECH.ECH31 (LVOTVTI) + LVOT diameter + HR",
        "v5_mean": 4.6, "v5_sd": 1.1, "v5_n": None, "v5_missing_pct": 0.2,
        "v7_mean": 4.5, "v7_sd": 1.2, "v7_n": None, "v7_missing_pct": 0.6,
        "v5_range": (2.0, 9.0), "v7_range": (2.0, 9.0),
        "distribution": "normal",
        "clinical_weight": 0.8,
        "physiological_bounds": (1.5, 10.0),
        "emission_eq": "Eq.60: SV * HR / 1000",
        "emission_source": "CircAdapt SV, HR",
    },
    # ===== DIASTOLIC FUNCTION (4) =====
    "E_cm_s": {
        "description": "Peak E wave (early diastolic mitral inflow) velocity",
        "unit": "cm/s",
        "aric_field_v5": "ECH.ECH20 (EWAVE)",
        "aric_field_v7": "ECH.ECH20 (EWAVE)",
        "v5_mean": 67.6, "v5_sd": 19.2, "v5_n": 6098, "v5_missing_pct": 0.3,
        "v7_mean": 75.7, "v7_sd": 21.7, "v7_n": 3041, "v7_missing_pct": 0.8,
        "v5_range": (23, 189), "v7_range": (25, 215),
        "distribution": "normal",
        "clinical_weight": 0.8,
        "physiological_bounds": (20, 220),
        "emission_eq": "Eq.62: peak mitral flow velocity in early diastole",
        "emission_source": "CircAdapt mitral valve flow waveform",
    },
    "e_prime_sept_cm_s": {
        "description": "Septal early diastolic tissue velocity (e' septal)",
        "unit": "cm/s",
        "aric_field_v5": "ECH.ECH26 (EASEPT)",
        "aric_field_v7": "ECH.ECH26 (EASEPT)",
        "v5_mean": 5.67, "v5_sd": 1.49, "v5_n": 6100, "v5_missing_pct": 0.3,
        "v7_mean": 5.27, "v7_sd": 1.44, "v7_n": 3018, "v7_missing_pct": 1.5,
        "v5_range": (1.9, 19.6), "v7_range": (2.0, 14.5),
        "distribution": "normal",
        "clinical_weight": 0.9,
        "physiological_bounds": (1.5, 20),
        "emission_eq": "Eq.66: k_e' * dLs/dt at early diastole",
        "emission_source": "CircAdapt sarcomere lengthening rate",
    },
    "E_over_e_prime_sept": {
        "description": "E/e' septal ratio (filling pressure surrogate)",
        "unit": "ratio",
        "aric_field_v5": "ECO.ECO7 (EEPRIMESEPT)",
        "aric_field_v7": "ECH.ECH64 (EEPRIMESEPT)",
        "v5_mean": 12.57, "v5_sd": 4.73, "v5_n": 6090, "v5_missing_pct": 0.5,
        "v7_mean": 15.17, "v7_sd": 5.55, "v7_n": 3007, "v7_missing_pct": 1.9,
        "v5_range": (3.4, 74.2), "v7_range": (3.7, 48.4),
        "distribution": "lognormal",
        "clinical_weight": 1.0,
        "physiological_bounds": (3, 50),
        "emission_eq": "Eq.68: E / e'",
        "emission_source": "Derived from E and e' (both from CircAdapt)",
        "abnormal_threshold": 15,
    },
    "E_over_A_ratio": {
        "description": "E/A ratio (diastolic filling pattern)",
        "unit": "ratio",
        "aric_field_v5": "ECH.ECH28 (EARATIO)",
        "aric_field_v7": "ECH.ECH28 (EARATIO)",
        "v5_mean": 0.86, "v5_sd": 0.29, "v5_n": 5829, "v5_missing_pct": 4.7,
        "v7_mean": 0.87, "v7_sd": 0.31, "v7_n": 2829, "v7_missing_pct": 7.7,
        "v5_range": (0.3, 3.6), "v7_range": (0.3, 4.2),
        "distribution": "lognormal",
        "clinical_weight": 0.9,
        "physiological_bounds": (0.2, 5.0),
        "emission_eq": "Derived: E_cm_s / A_cm_s",
        "emission_source": "CircAdapt mitral valve flow waveform (E and A peaks)",
    },
    # ===== ATRIAL / RV (2) =====
    "LAvolume_mL": {
        "description": "Left atrial volume",
        "unit": "mL",
        "aric_field_v5": "ECH.ECH16 (LAV)",
        "aric_field_v7": "ECH.ECH16 (LAV)",
        "v5_mean": 49.147, "v5_sd": 19.412, "v5_n": 6052, "v5_missing_pct": 1.1,
        "v7_mean": 51.988, "v7_sd": 19.029, "v7_n": 2961, "v7_missing_pct": 3.4,
        "v5_range": (11.86, 353.11), "v7_range": (10.73, 219.88),
        "distribution": "lognormal",
        "clinical_weight": 0.7,
        "physiological_bounds": (10, 250),
        "emission_eq": "Eq.70: 4/3 * pi * r_LA^3",
        "emission_source": "CircAdapt LA cavity volume",
    },
    "PASP_mmHg": {
        "description": "Pulmonary artery systolic pressure (estimated)",
        "unit": "mmHg",
        "aric_field_v5": "ECH.ECH41 (RVRAG) + estimated RAP",
        "aric_field_v7": "ECH.ECH41 (RVRAG) + estimated RAP",
        "v5_mean": 28.3, "v5_sd": 6.1, "v5_n": 3573, "v5_missing_pct": 41.6,
        "v7_mean": 32.3, "v7_sd": 8.2, "v7_n": 2290, "v7_missing_pct": 25.3,
        "v5_range": (14.3, 80.8), "v7_range": (13.5, 82.6),
        "distribution": "normal",
        "clinical_weight": 0.8,
        "physiological_bounds": (12, 85),
        "emission_eq": "Eq.74: 4*v_TR^2 + CVP",
        "emission_source": "CircAdapt RV pressure + CVP",
    },
    # ===== HEMODYNAMICS (3) =====
    "SBP_mmHg": {
        "description": "Systolic blood pressure",
        "unit": "mmHg",
        "aric_field_v5": "Visit exam form (not echo codebook)",
        "aric_field_v7": "Visit exam form (not echo codebook)",
        "v5_mean": 130, "v5_sd": 18, "v5_n": None, "v5_missing_pct": 0.1,
        "v7_mean": 130, "v7_sd": 20, "v7_n": None, "v7_missing_pct": 0.1,
        "v5_range": (80, 210), "v7_range": (75, 220),
        "distribution": "normal",
        "clinical_weight": 0.8,
        "physiological_bounds": (70, 230),
        "emission_eq": "Eq.75: max(P_aorta(t))",
        "emission_source": "CircAdapt aortic pressure waveform",
    },
    "MAP_mmHg": {
        "description": "Mean arterial pressure (derived from SBP/DBP)",
        "unit": "mmHg",
        "aric_field_v5": "Derived: DBP + (SBP-DBP)/3",
        "aric_field_v7": "Derived: DBP + (SBP-DBP)/3",
        "v5_mean": 93, "v5_sd": 10, "v5_n": None, "v5_missing_pct": 0.1,
        "v7_mean": 93, "v7_sd": 11, "v7_n": None, "v7_missing_pct": 0.1,
        "v5_range": (55, 145), "v7_range": (50, 150),
        "distribution": "normal",
        "clinical_weight": 0.8,
        "physiological_bounds": (50, 150),
        "emission_eq": "Eq.77: integral of P_aorta / T_beat",
        "emission_source": "CircAdapt aortic pressure waveform",
    },
    "SVR_wood": {
        "description": "Systemic vascular resistance (Wood units)",
        "unit": "Wood units (mmHg*min/L)",
        "aric_field_v5": "Derived: (MAP - CVP) / CO",
        "aric_field_v7": "Derived: (MAP - CVP) / CO",
        "v5_mean": 20, "v5_sd": 5, "v5_n": None, "v5_missing_pct": 1.0,
        "v7_mean": 21, "v7_sd": 6, "v7_n": None, "v7_missing_pct": 1.0,
        "v5_range": (8, 40), "v7_range": (8, 45),
        "distribution": "normal",
        "clinical_weight": 0.7,
        "physiological_bounds": (6, 50),
        "emission_eq": "Eq.79: (MAP - CVP) / CO",
        "emission_source": "Derived from CircAdapt hemodynamics",
    },
    # ===== RENAL (4) =====
    "eGFR_mL_min": {
        "description": "Estimated GFR (CKD-EPI 2021 from creatinine)",
        "unit": "mL/min/1.73m2",
        "aric_field_v5": "Lab: CKD-EPI from serum creatinine",
        "aric_field_v7": "Lab: CKD-EPI from serum creatinine",
        "v5_mean": 66, "v5_sd": 18, "v5_n": 5170, "v5_missing_pct": 1.0,
        "v7_mean": 62, "v7_sd": 20, "v7_n": None, "v7_missing_pct": 1.0,
        "v5_range": (5, 130), "v7_range": (5, 125),
        "distribution": "normal",
        "clinical_weight": 1.0,
        "physiological_bounds": (3, 140),
        "emission_eq": "Eq.85: CKD-EPI 2021 from modeled creatinine",
        "emission_source": "Hallow GFR -> creatinine -> CKD-EPI",
    },
    "creatinine_mg_dL": {
        "description": "Serum creatinine",
        "unit": "mg/dL",
        "aric_field_v5": "Lab: serum creatinine (IDMS-traceable)",
        "aric_field_v7": "Lab: serum creatinine",
        "v5_mean": 1.05, "v5_sd": 0.35, "v5_n": None, "v5_missing_pct": 0.5,
        "v7_mean": 1.15, "v7_sd": 0.45, "v7_n": None, "v7_missing_pct": 0.5,
        "v5_range": (0.3, 5.0), "v7_range": (0.3, 6.0),
        "distribution": "lognormal",
        "clinical_weight": 0.9,
        "physiological_bounds": (0.2, 8.0),
        "emission_eq": "Eq.86: B_Cr / GFR * (1 + 0.02*age/10)",
        "emission_source": "Hallow GFR + demographics",
    },
    "UACR_mg_g": {
        "description": "Urine albumin-to-creatinine ratio",
        "unit": "mg/g",
        "aric_field_v5": "Lab: urine albumin / urine creatinine",
        "aric_field_v7": "Lab: urine albumin / urine creatinine",
        "v5_mean": 18, "v5_sd": 35, "v5_n": 5170, "v5_missing_pct": 3.0,
        "v5_median": 11, "v5_iqr": (6, 22),
        "v7_mean": 25, "v7_sd": 50, "v7_n": None, "v7_missing_pct": 3.0,
        "v5_range": (1, 3000), "v7_range": (1, 5000),
        "distribution": "lognormal",
        "clinical_weight": 0.8,
        "physiological_bounds": (0.5, 5000),
        "emission_eq": "Eq.88: Na_alb * (1 + beta_alb * dP_gc) / V_urine",
        "emission_source": "Hallow glomerular pressure + filtration",
    },
    "cystatin_C_mg_L": {
        "description": "Serum cystatin C",
        "unit": "mg/L",
        "aric_field_v5": "Lab: cystatin C (Latex immunoassay, ERM-DA471 traceable)",
        "aric_field_v7": "Lab: cystatin C",
        "v5_mean": 1.10, "v5_sd": 0.30, "v5_n": None, "v5_missing_pct": 2.0,
        "v7_mean": 1.20, "v7_sd": 0.35, "v7_n": None, "v7_missing_pct": 2.0,
        "v5_range": (0.4, 4.0), "v7_range": (0.4, 5.0),
        "distribution": "lognormal",
        "clinical_weight": 0.8,
        "physiological_bounds": (0.3, 6.0),
        "emission_eq": "Eq.93: D_cys / GFR * (1 + epsilon * i)",
        "emission_source": "Hallow GFR + inflammation index",
    },
    # ===== BIOMARKERS (2) =====
    "NTproBNP_pg_mL": {
        "description": "N-terminal pro-B-type natriuretic peptide",
        "unit": "pg/mL",
        "aric_field_v5": "Lab: electrochemiluminescent immunoassay",
        "aric_field_v7": "Lab: same assay",
        "v5_mean": 130, "v5_sd": 180, "v5_n": None, "v5_missing_pct": 2.0,
        "v5_median": 80,
        "v7_mean": 180, "v7_sd": 250, "v7_n": None, "v7_missing_pct": 2.0,
        "v5_range": (5, 10000), "v7_range": (5, 15000),
        "distribution": "lognormal",
        "clinical_weight": 1.0,
        "physiological_bounds": (1, 30000),
        "emission_eq": "Eq.90: B_BNP * exp(b1*LVEDP + b2*CVP + b3*LVmass/BSA)",
        "emission_source": "CircAdapt hemodynamics + demographics",
    },
    "CRP_mg_L": {
        "description": "High-sensitivity C-reactive protein",
        "unit": "mg/L",
        "aric_field_v5": "Lab: immunonephelometric assay (hs-CRP)",
        "aric_field_v7": "Lab: same assay",
        "v5_mean": 3.5, "v5_sd": 5.0, "v5_n": None, "v5_missing_pct": 1.0,
        "v5_median": 2.0,
        "v7_mean": 4.0, "v7_sd": 6.0, "v7_n": None, "v7_missing_pct": 1.0,
        "v5_range": (0.1, 50), "v7_range": (0.1, 60),
        "distribution": "lognormal",
        "clinical_weight": 0.7,
        "physiological_bounds": (0.05, 100),
        "emission_eq": "Eq.92: C_base * exp(delta1*i + delta2*d)",
        "emission_source": "Inflammation index + diabetes index",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Cystatin C Emission Parameters
# ═══════════════════════════════════════════════════════════════════════════════
# Paper Eq. 93: CysC = D_cys / GFR * (1 + epsilon * i)
# Calibrated so healthy (GFR=100, i=0) -> CysC ~ 0.80 mg/L
CYSTATIN_C_PARAMS = {
    "D_cys": 80.0,          # mg*min/(L*mL) — production rate constant
    "epsilon": 0.30,         # inflammation sensitivity coefficient
    "healthy_target": 0.80,  # mg/L at GFR=100, i=0
}


# ═══════════════════════════════════════════════════════════════════════════════
# Measurement Noise (from echo reproducibility literature)
# ═══════════════════════════════════════════════════════════════════════════════
# "gaussian_absolute": add N(0, magnitude)
# "gaussian_relative": multiply by N(1, magnitude)
MEASUREMENT_NOISE = {
    "LVIDd_cm":            ("gaussian_relative", 0.04),
    "LVmass_g":            ("gaussian_relative", 0.08),
    "LVEF_pct":            ("gaussian_absolute", 3.5),
    "GLS_pct":             ("gaussian_absolute", 1.5),
    "CO_L_min":            ("gaussian_relative", 0.10),
    "E_cm_s":              ("gaussian_relative", 0.08),
    "e_prime_sept_cm_s":   ("gaussian_relative", 0.10),
    "E_over_e_prime_sept": ("gaussian_relative", 0.12),
    "E_over_A_ratio":      ("gaussian_relative", 0.10),
    "LAvolume_mL":         ("gaussian_relative", 0.10),
    "PASP_mmHg":           ("gaussian_absolute", 5.0),
    "SBP_mmHg":            ("gaussian_absolute", 6.0),
    "MAP_mmHg":            ("gaussian_absolute", 4.0),
    "SVR_wood":            ("gaussian_relative", 0.10),
    "eGFR_mL_min":         ("gaussian_relative", 0.05),
    "creatinine_mg_dL":    ("gaussian_relative", 0.05),
    "UACR_mg_g":           ("gaussian_relative", 0.20),
    "cystatin_C_mg_L":     ("gaussian_relative", 0.05),
    "NTproBNP_pg_mL":      ("gaussian_relative", 0.08),
    "CRP_mg_L":            ("gaussian_relative", 0.10),
}


# ═══════════════════════════════════════════════════════════════════════════════
# PASP Missingness Model
# ═══════════════════════════════════════════════════════════════════════════════
# 41.6% missing at V5 due to absent/insufficient TR jet.
# Probability of PASP being missing depends on disease severity.
PASP_MISSING_PARAMS = {
    "base_missing_prob": 0.42,
    "pasp_slope": -0.015,        # per mmHg above 25: less likely missing
    "v7_missing_prob": 0.25,
}
