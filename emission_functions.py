#!/usr/bin/env python3
"""
Emission Functions: CircAdapt + Hallow → ARIC Echocardiographic & Renal Variables
==================================================================================
Paper Section 3.6: Clinical Emission Functions

This module implements the clinical emission layer (Section 3.6 of Dorbala 2025),
which maps internal CircAdapt waveforms and Hallow renal outputs to 113 ARIC-
compatible clinical variables. This is the bridge between the latent disease state
(simulator parameters) and observable clinical measurements, analogous to the
observation model in a hidden Markov model or state-space formulation.

Generates synthetic echocardiographic, Doppler, tissue-Doppler, speckle-tracking,
and renal-health variables matching the measurement protocol of ARIC visits 5
(2011-2013) and 7 (2018-2019).

Every emit_* function documents:
  - The ARIC variable name and units
  - The physiological derivation from CircAdapt signals and/or Hallow renal outputs
  - The published reference equations or normal ranges

Variable categories (8 groups, 113 total):
  - LV structure (7): LVIDd, LVIDs, IVSd, LVPWd, LV mass, RWT
  - LV systolic function (6): LVEDV, LVESV, LVEF, SV, CO, GLS
  - Diastolic function (13): E, A, E/A, DT, IVRT, e', a', s', E/e'
  - Atrial/RV (20): LA volume/strain, RV volumes, TAPSE, PASP
  - Hemodynamics (6): SBP, DBP, MAP, heart rate, pulse pressure
  - Vascular (5): arterial compliance, PWV, VA coupling
  - Renal/lab (16): GFR, eGFR, creatinine, UACR, NT-proBNP, troponin
  - Indexed (9): body-size normalized versions

Sources:
  CircAdapt components used:
    Cavity   — V (volume), p (pressure) for cLv, cRv, La, Ra, SyArt, SyVen, PuArt, PuVen
    Patch    — Sf_act, Sf (fiber stress), Ef (fiber strain), l_s (sarcomere length),
               Am_ref, V_wall, k1
    Valve    — q (flow) for MV, AV, TV, PV
    ArtVen   — q (flow) for systemic and pulmonary beds
    General  — t_cycle
    Solver   — t (time vector)
    Wall     — Am (midwall area)
    TriSeg   — signals for septal interaction

  Hallow renal model outputs (from cardiorenal_coupling.py):
    GFR, RBF, P_glom, Na_excretion, water_excretion, V_blood, C_Na

Usage:
    from circadapt import VanOsta2024
    from emission_functions import extract_all_aric_variables

    model = VanOsta2024()
    model.run(stable=True)  # solve to hemodynamic steady state
    model.run(1)            # store 1 beat for waveform extraction

    renal_state = {...}     # from Hallow model update

    variables = extract_all_aric_variables(model, renal_state)
    # Returns dict with 113 ARIC-compatible clinical variables

References:
    - Dorbala 2025, Section 3.6 (Clinical Emission Functions)
    - CircAdapt: van Osta et al., EHJ-DH, 2024
    - Hallow et al., CPT:PSP, 6(1):48-57, 2017
    - ASE guidelines for echocardiographic measurements
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# ── Unit Conversion Constants ───────────────────────────────────────────────
# CircAdapt uses SI units internally (Pa, m³, m³/s).
# Clinical measurements use mmHg, mL, mL/s.
PA_TO_MMHG = 7.5e-3    # 1 Pa = 7.5e-3 mmHg (pressure conversion)
M3_TO_ML = 1e6          # 1 m³ = 1e6 mL (volume conversion)
ML_TO_M3 = 1e-6         # 1 mL = 1e-6 m³ (inverse volume conversion)
M3S_TO_MLS = 1e6        # 1 m³/s = 1e6 mL/s (flow conversion)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: extract raw waveforms from a CircAdapt VanOsta2024 model
# ═══════════════════════════════════════════════════════════════════════════

def _get_waveforms(model) -> Dict[str, np.ndarray]:
    """
    Extract all raw time-series waveforms from a CircAdapt VanOsta2024 model.

    This is the core data extraction function. CircAdapt stores one full cardiac
    cycle of pressure, volume, and flow waveforms after model.run(1). We pull
    these waveforms and convert from SI units to clinical units.

    The extracted waveforms are used by all 17 emit_* functions to compute
    ARIC-compatible clinical variables.

    Parameters
    ----------
    model : circadapt.VanOsta2024
        Must have run at least 1 stored beat (model.run(1) after stable).

    Returns
    -------
    dict : Waveform arrays in clinical units (mmHg, mL, mL/s, ms).
        Keys: t, t_ms, dt, t_cycle, p_lv, p_rv, p_la, p_ra, p_ao, p_sv,
              p_pa, p_pv, V_lv, V_rv, V_la, V_ra, q_mv, q_av, q_tv, q_pv,
              q_sys, q_pul, Ef_all, Sf_all, ls_all
    """
    # Time vector from CircAdapt's ODE solver (one stored cardiac cycle)
    t = model['Solver']['t']  # seconds — typically ~0.8s for 75 bpm
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.002  # time step ~2ms
    t_ms = t * 1e3  # convert to milliseconds for clinical timing measurements

    # ── Pressures [mmHg] ────────────────────────────────────────────────
    # CircAdapt Cavity pressures: 4 cardiac chambers + 4 vascular compartments
    # Used for: BP, filling pressures, pulmonary pressures, PV loops
    p_lv    = model['Cavity']['p'][:, 'cLv']  * PA_TO_MMHG   # left ventricle
    p_rv    = model['Cavity']['p'][:, 'cRv']  * PA_TO_MMHG   # right ventricle
    p_la    = model['Cavity']['p'][:, 'La']   * PA_TO_MMHG   # left atrium
    p_ra    = model['Cavity']['p'][:, 'Ra']   * PA_TO_MMHG   # right atrium
    p_ao    = model['Cavity']['p'][:, 'SyArt'] * PA_TO_MMHG  # systemic arterial (aorta)
    p_sv    = model['Cavity']['p'][:, 'SyVen'] * PA_TO_MMHG  # systemic venous (CVP proxy)
    p_pa    = model['Cavity']['p'][:, 'PuArt'] * PA_TO_MMHG  # pulmonary arterial
    p_pv    = model['Cavity']['p'][:, 'PuVen'] * PA_TO_MMHG  # pulmonary venous

    # ── Volumes [mL] ───────────────────────────────────────────────────
    # CircAdapt Cavity volumes: used for EDV, ESV, EF, stroke volume, PV loops
    V_lv    = model['Cavity']['V'][:, 'cLv']  * M3_TO_ML   # LV volume waveform
    V_rv    = model['Cavity']['V'][:, 'cRv']  * M3_TO_ML   # RV volume waveform
    V_la    = model['Cavity']['V'][:, 'La']   * M3_TO_ML   # LA volume (for LAVi)
    V_ra    = model['Cavity']['V'][:, 'Ra']   * M3_TO_ML   # RA volume

    # ── Valve flows [mL/s] ─────────────────────────────────────────────
    # CircAdapt models 6 valves in fixed order; we extract the 4 cardiac valves
    # Used for: Doppler velocities, cardiac output, regurgitation assessment
    q_all = model['Valve']['q'] * M3S_TO_MLS  # shape (ntime, 6)
    q_tv  = q_all[:, 1]   # tricuspid valve flow (RA→RV) — for TV Doppler
    q_pv  = q_all[:, 2]   # pulmonic valve flow (RV→PA) — for PV Doppler
    q_mv  = q_all[:, 4]   # mitral valve flow (LA→LV) — for E/A, DT, IVRT
    q_av  = q_all[:, 5]   # aortic valve flow (LV→Aorta) — for LVOT VTI, AV gradients

    # ── ArtVen flows [mL/s] ────────────────────────────────────────────
    # Arteriovenous bed flows — used for cardiac output calculation
    q_sys = model['ArtVen']['q'][:, 0] * M3S_TO_MLS  # systemic bed flow
    q_pul = model['ArtVen']['q'][:, 1] * M3S_TO_MLS  # pulmonary bed flow

    # ── Patch signals (sarcomere-level mechanics) ──────────────────────
    # Patch objects represent myocardial wall segments: pLa0, pRa0, pLv0, pSv0, pRv0
    # These are essential for strain, tissue Doppler, and myocardial work calculations
    Ef_all = model['Patch']['Ef']     # natural (Green) fiber strain — for GLS
    Sf_all = model['Patch']['Sf']     # total fiber stress [Pa] — for myocardial work
    ls_all = model['Patch']['l_s']    # sarcomere length [µm] — for tissue velocities

    # Cardiac cycle duration (determines heart rate: HR = 60/t_cycle)
    t_cycle = float(model['General']['t_cycle'])

    return dict(
        t=t, t_ms=t_ms, dt=dt, t_cycle=t_cycle,
        p_lv=p_lv, p_rv=p_rv, p_la=p_la, p_ra=p_ra,
        p_ao=p_ao, p_sv=p_sv, p_pa=p_pa, p_pv=p_pv,
        V_lv=V_lv, V_rv=V_rv, V_la=V_la, V_ra=V_ra,
        q_mv=q_mv, q_av=q_av, q_tv=q_tv, q_pv=q_pv,
        q_sys=q_sys, q_pul=q_pul,
        Ef_all=Ef_all, Sf_all=Sf_all, ls_all=ls_all,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: LV Structure  [Paper Section 3.6, Category: LV structure (7 variables)]
# ═══════════════════════════════════════════════════════════════════════════
# ARIC variables: LVIDd, LVIDs, IVSd, LVPWd, LV mass, LVMi, RWT

def emit_LV_structure(model, w: Dict) -> Dict:
    """
    LV linear dimensions and mass from CircAdapt wall/patch geometry.

    CircAdapt source:
      Wall['Am']         — midwall area → cavity radius via sphere model
      Patch['Am_ref']    — reference area
      Patch['V_wall']    — wall volume → wall thickness

    Derivations:
      For a thick-walled sphere: V_cav = (4/3)π r_endo³
        r_endo = (3·V_cav / 4π)^(1/3)
        r_epi  = ((3·(V_cav + V_wall)) / 4π)^(1/3)
        wall_thickness = r_epi − r_endo

      LV mass = V_wall × ρ_myocardium  (ρ ≈ 1.05 g/mL)

    ARIC protocol: parasternal long-axis, ASE convention.
    """
    V_lv = w['V_lv']
    V_wall_lv_m3 = float(model['Patch']['V_wall']['pLv0'])
    V_wall_sv_m3 = float(model['Patch']['V_wall']['pSv0'])
    V_wall_lv_mL = V_wall_lv_m3 * M3_TO_ML
    V_wall_sv_mL = V_wall_sv_m3 * M3_TO_ML

    rho_myo = 1.05  # g/mL

    # End-diastolic and end-systolic volumes
    EDV = float(np.max(V_lv))
    ESV = float(np.min(V_lv))

    # Sphere model for internal diameter
    r_endo_ed = (3.0 * EDV / (4.0 * np.pi)) ** (1.0/3.0)  # cm (mL → cm³)
    r_endo_es = (3.0 * ESV / (4.0 * np.pi)) ** (1.0/3.0)
    LVIDd = 2.0 * r_endo_ed  # cm
    LVIDs = 2.0 * r_endo_es

    # LV free wall thickness (from V_wall)
    r_epi_ed = (3.0 * (EDV + V_wall_lv_mL) / (4.0 * np.pi)) ** (1.0/3.0)
    LVPWd = r_epi_ed - r_endo_ed  # cm — posterior wall thickness at diastole

    # IVS thickness (from septal wall volume)
    # Approximation: septum is shared between LV and RV cavities
    V_rv_ed = float(np.max(w['V_rv']))
    r_rv_endo = (3.0 * V_rv_ed / (4.0 * np.pi)) ** (1.0/3.0)
    r_rv_epi = (3.0 * (V_rv_ed + V_wall_sv_mL) / (4.0 * np.pi)) ** (1.0/3.0)
    IVSd = r_rv_epi - r_rv_endo  # cm — approximate IVS thickness

    # LV mass (ASE cube formula):
    #   LV_mass = 0.8 × 1.04 × [(LVIDd + IVSd + LVPWd)³ − LVIDd³] + 0.6
    LV_mass_cube = 0.8 * 1.04 * ((LVIDd + IVSd + LVPWd)**3 - LVIDd**3) + 0.6

    # Direct from wall volume (more accurate in CircAdapt)
    total_wall_mL = V_wall_lv_mL + V_wall_sv_mL
    LV_mass_direct = total_wall_mL * rho_myo  # grams

    # Relative wall thickness
    RWT = 2.0 * LVPWd / LVIDd if LVIDd > 0 else 0.42

    return {
        'LVIDd_cm':         LVIDd,              # LV internal diameter, diastole
        'LVIDs_cm':         LVIDs,              # LV internal diameter, systole
        'IVSd_cm':          IVSd,               # interventricular septum, diastole
        'LVPWd_cm':         LVPWd,              # LV posterior wall, diastole
        'LV_mass_g':        LV_mass_direct,     # LV mass (direct, grams)
        'LV_mass_cube_g':   LV_mass_cube,       # LV mass (ASE cube formula)
        'RWT':              RWT,                 # relative wall thickness
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: LV Volumes and Systolic Function  [Paper Section 3.6, Category: LV systolic (6 variables)]
# ═══════════════════════════════════════════════════════════════════════════
# ARIC: LVEDV, LVESV, LVEF, SV, CO, GLS, fractional shortening

def emit_LV_systolic(model, w: Dict) -> Dict:
    """
    LV volumes, ejection fraction, and global longitudinal strain.

    CircAdapt source:
      Cavity['V'][:, 'cLv']    — LV volume waveform
      Patch['Ef'][:, 'pLv0']   — LV natural fiber strain (≈ GLS)
      General['t_cycle']       — for HR → CO calculation

    ARIC variables:
      LVEF (biplane Simpson), LVEDV, LVESV, CO, SV, fractional shortening
      GLS by 2D speckle tracking (average of 3 apical views)
    """
    V_lv = w['V_lv']
    t_cycle = w['t_cycle']

    EDV = float(np.max(V_lv))
    ESV = float(np.min(V_lv))
    SV = EDV - ESV
    EF = SV / EDV * 100.0 if EDV > 0 else 0.0
    HR = 60.0 / t_cycle
    CO = SV * HR / 1000.0  # L/min

    # Fractional shortening (from diameters, sphere approx)
    r_ed = (3.0 * EDV / (4.0*np.pi))**(1.0/3.0)
    r_es = (3.0 * ESV / (4.0*np.pi))**(1.0/3.0)
    FS = (r_ed - r_es) / r_ed * 100.0 if r_ed > 0 else 0.0

    # Global Longitudinal Strain (GLS)
    # CircAdapt Patch['Ef'] is the natural fiber strain.
    # In the one-fiber model, Ef = (1/2) ln(Am / Am_ref).
    # GLS ≈ (L_es - L_ed) / L_ed × 100 ≈ exp(Ef_es - Ef_ed) - 1
    # We use the peak-to-peak excursion of Ef on the LV patch.
    try:
        Ef_lv = w['Ef_all'][:, 2]  # pLv0 is typically index 2
        Ef_ed = Ef_lv[np.argmax(w['V_lv'])]
        Ef_es = Ef_lv[np.argmin(w['V_lv'])]
        # Convert natural strain to engineering strain (%)
        GLS = (np.exp(Ef_es - Ef_ed) - 1.0) * 100.0
        # GLS is negative (shortening)
    except (IndexError, KeyError):
        GLS = -20.0  # fallback normal

    return {
        'LVEDV_mL':          EDV,
        'LVESV_mL':          ESV,
        'SV_mL':             SV,
        'LVEF_pct':          EF,
        'CO_Lmin':           CO,
        'HR_bpm':            HR,
        'FS_pct':            FS,                 # fractional shortening
        'GLS_pct':           GLS,                # global longitudinal strain (negative)
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Doppler — Mitral Inflow (Diastolic Function)  [Paper Section 3.6, Diastolic function]
# ═══════════════════════════════════════════════════════════════════════════
# ARIC: E wave, A wave, E/A ratio, DT, IVRT

def emit_mitral_inflow_doppler(model, w: Dict) -> Dict:
    """
    Pulsed-wave Doppler of transmitral flow.

    CircAdapt source:
      Valve['q'][:, 'LaLv'] — mitral valve flow waveform

    Derivation:
      E velocity = peak flow during early filling / mitral annulus area
      A velocity = peak flow during atrial contraction
      The mitral annulus area is estimated from Valve['A_open']['LaLv'].

    ARIC protocol: PW Doppler at mitral leaflet tips, sweep 100 cm/s.
    """
    q_mv = w['q_mv']  # mL/s
    t = w['t']
    t_cycle = w['t_cycle']

    # Mitral annulus area (from valve A_open parameter)
    try:
        A_mv_m2 = float(model['Valve']['A_open']['LaLv'])
        A_mv_cm2 = A_mv_m2 * 1e4  # m² → cm²
    except (KeyError, IndexError):
        A_mv_cm2 = 4.0  # default ~4 cm²

    # Velocity = flow / area
    v_mv = q_mv / A_mv_cm2  # cm/s

    # Identify E and A peaks:
    # E wave = first (larger) peak of positive mitral flow
    # A wave = second peak (atrial systole, last ~20% of cycle)
    positive_flow = np.where(v_mv > 0)[0]
    if len(positive_flow) == 0:
        return dict(E_vel_cms=70.0, A_vel_cms=50.0, EA_ratio=1.4,
                    DT_ms=200.0, IVRT_ms=80.0)

    # Split cycle: first 60% = early filling (E), last 30% = A wave
    n = len(v_mv)
    cutoff = int(0.55 * n)

    v_early = v_mv[:cutoff]
    v_late = v_mv[cutoff:]

    E_vel = float(np.max(v_early)) if np.any(v_early > 0) else 70.0
    A_vel = float(np.max(v_late)) if np.any(v_late > 0) else 50.0
    EA_ratio = E_vel / A_vel if A_vel > 0 else 1.0

    # Deceleration time: time from E peak to half-E or baseline
    E_peak_idx = np.argmax(v_early)
    E_half = E_vel * 0.5
    dt_s = w['dt']
    DT_samples = 0
    for i in range(E_peak_idx, len(v_early)):
        if v_early[i] < E_half:
            DT_samples = i - E_peak_idx
            break
    DT_ms = DT_samples * dt_s * 1e3 if DT_samples > 0 else 200.0

    # IVRT: time from aortic valve closure to mitral valve opening
    # AV closure ≈ time of min LV volume; MV opening ≈ first positive MV flow after
    t_es_idx = int(np.argmin(w['V_lv']))
    mv_open_idx = t_es_idx
    for i in range(t_es_idx, n):
        if q_mv[i] > 0.5:
            mv_open_idx = i
            break
    IVRT_ms = (mv_open_idx - t_es_idx) * dt_s * 1e3

    return {
        'E_vel_cms':         E_vel,              # mitral E wave velocity (cm/s)
        'A_vel_cms':         A_vel,              # mitral A wave velocity (cm/s)
        'EA_ratio':          EA_ratio,           # E/A ratio
        'DT_ms':             DT_ms,              # E wave deceleration time (ms)
        'IVRT_ms':           IVRT_ms,            # isovolumic relaxation time (ms)
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Tissue Doppler Imaging (TDI)  [Paper Section 3.6, Diastolic function]
# ═══════════════════════════════════════════════════════════════════════════
# ARIC: e' (septal, lateral), s', E/e', a'

def emit_tissue_doppler(model, w: Dict) -> Dict:
    """
    Tissue Doppler velocities of the mitral annulus.

    CircAdapt source:
      Patch['Ef'][:, 'pLv0']  — LV fiber strain waveform
      Patch['Ef'][:, 'pSv0']  — septal fiber strain

    Derivation:
      TDI annular velocity ≈ dL/dt where L is the long-axis length.
      From CircAdapt: dEf/dt × L_ref gives velocity.
      e' = peak early-diastolic annular velocity
      s' = peak systolic annular velocity
      a' = peak late-diastolic (atrial) annular velocity

    The conversion from fiber strain rate to annular velocity uses
    an assumed LV long-axis length (~8-9 cm).

    ARIC protocol: PW TDI at septal and lateral annulus, sweep 100 cm/s,
    sample volume 5mm, filter 100 Hz.
    """
    try:
        Ef_lv = w['Ef_all'][:, 2]  # pLv0
        Ef_sv = w['Ef_all'][:, 3]  # pSv0 (septal)
    except (IndexError, KeyError):
        Ef_lv = np.zeros(len(w['t']))
        Ef_sv = np.zeros(len(w['t']))

    dt = w['dt']
    t_cycle = w['t_cycle']
    n = len(Ef_lv)

    # Strain rate = dEf/dt
    dEf_dt_lv = np.gradient(Ef_lv, dt)
    dEf_dt_sv = np.gradient(Ef_sv, dt)

    # Annular velocity = strain rate × long-axis length
    L_lv = 8.5   # cm (approximate LV long-axis)
    L_sv = 7.0   # cm (septal contribution)

    v_lat = dEf_dt_lv * L_lv   # lateral annular velocity (cm/s)
    v_sep = dEf_dt_sv * L_sv   # septal annular velocity (cm/s)

    # Systolic (first ~35% of cycle): find peak positive (toward apex)
    sys_end = int(0.40 * n)
    dia_start = sys_end

    s_prime_lat = float(np.max(np.abs(v_lat[:sys_end])))
    s_prime_sep = float(np.max(np.abs(v_sep[:sys_end])))

    # Early diastolic e' (first major peak after AVC)
    cutoff = int(0.55 * n)
    e_prime_lat = float(np.max(np.abs(v_lat[dia_start:cutoff])))
    e_prime_sep = float(np.max(np.abs(v_sep[dia_start:cutoff])))

    # Late diastolic a' (atrial contraction, last ~25% of cycle)
    a_start = int(0.75 * n)
    a_prime_lat = float(np.max(np.abs(v_lat[a_start:])))
    a_prime_sep = float(np.max(np.abs(v_sep[a_start:])))

    # Average e'
    e_prime_avg = (e_prime_lat + e_prime_sep) / 2.0

    return {
        'e_prime_sep_cms':   e_prime_sep,        # septal e' (cm/s)
        'e_prime_lat_cms':   e_prime_lat,        # lateral e' (cm/s)
        'e_prime_avg_cms':   e_prime_avg,        # average e'
        's_prime_sep_cms':   s_prime_sep,        # septal s'
        's_prime_lat_cms':   s_prime_lat,        # lateral s'
        'a_prime_sep_cms':   a_prime_sep,        # septal a'
        'a_prime_lat_cms':   a_prime_lat,        # lateral a'
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: E/e' and Filling Pressure Estimation  [Paper Section 3.6, key HFpEF marker]
# ═══════════════════════════════════════════════════════════════════════════
# ARIC: E/e' (septal, lateral, average), LAP estimate

def emit_filling_pressures(mitral: Dict, tdi: Dict) -> Dict:
    """
    E/e' ratio — non-invasive estimate of LV filling pressure.

    Derivation:
      E/e' = mitral_E / tissue_e'
      Septal E/e' > 15 → elevated LAP
      Lateral E/e' > 13 → elevated LAP
      Average E/e' > 14 → elevated LAP

    LAP estimate (Nagueh): LAP ≈ 1.24 × (E/e') + 1.9  [mmHg]
    PCWP estimate: PCWP ≈ 1.55 + 1.47 × (E/e'_avg)

    ARIC protocol: E/e' calculated from PW Doppler E and PW TDI e'.
    """
    E = mitral['E_vel_cms']
    e_sep = tdi['e_prime_sep_cms']
    e_lat = tdi['e_prime_lat_cms']
    e_avg = tdi['e_prime_avg_cms']

    Ee_sep = E / e_sep if e_sep > 0 else 15.0
    Ee_lat = E / e_lat if e_lat > 0 else 12.0
    Ee_avg = E / e_avg if e_avg > 0 else 13.0

    # Nagueh LAP estimate
    LAP_est = 1.24 * Ee_avg + 1.9

    return {
        'E_e_prime_sep':     Ee_sep,             # E/e' (septal)
        'E_e_prime_lat':     Ee_lat,             # E/e' (lateral)
        'E_e_prime_avg':     Ee_avg,             # E/e' (average)
        'LAP_est_mmHg':      LAP_est,            # estimated LA pressure
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: LA Size and Function  [Paper Section 3.6, Atrial/RV category]
# ═══════════════════════════════════════════════════════════════════════════
# ARIC: LAVi (biplane), LA diameter, LA reservoir strain (LARS)

def emit_LA(model, w: Dict, BSA: float = 1.9) -> Dict:
    """
    Left atrial volumes and strain.

    CircAdapt source:
      Cavity['V'][:, 'La']    — LA volume waveform
      Patch['Ef'][:, 'pLa0']  — LA fiber strain (≈ reservoir strain)

    Derivation:
      LA max volume = max during ventricular systole (just before MV opens)
      LA min volume = min just after atrial contraction
      LA pre-A volume = volume at onset of P-wave / atrial contraction

    LA reservoir strain (LARS):
      LARS = peak positive strain during ventricular systole
      ≈ max(Ef_LA) × 100  [%]

    ARIC: biplane method of disks (Simpson), indexed to BSA.
    """
    V_la = w['V_la']
    n = len(V_la)

    LAV_max = float(np.max(V_la))
    LAV_min = float(np.min(V_la))

    # Pre-A volume: at ~70% of cycle (before atrial contraction)
    pre_A_idx = int(0.70 * n)
    LAV_preA = float(V_la[pre_A_idx])

    LAVi = LAV_max / BSA  # mL/m²

    # LA emptying fractions
    LA_total_EF = (LAV_max - LAV_min) / LAV_max * 100.0 if LAV_max > 0 else 0
    LA_passive_EF = (LAV_max - LAV_preA) / LAV_max * 100.0 if LAV_max > 0 else 0
    LA_active_EF = (LAV_preA - LAV_min) / LAV_preA * 100.0 if LAV_preA > 0 else 0

    # LA reservoir strain from Patch Ef
    try:
        Ef_la = w['Ef_all'][:, 0]  # pLa0 is typically index 0
        LARS = float(np.max(Ef_la) - np.min(Ef_la)) * 100.0
        # Reservoir, conduit, pump components
        LA_reservoir_strain = float(np.max(Ef_la)) * 100.0
        Ef_preA = Ef_la[pre_A_idx]
        LA_conduit_strain = float(np.max(Ef_la) - Ef_preA) * 100.0
        LA_pump_strain = float(Ef_preA - np.min(Ef_la)) * 100.0
    except (IndexError, KeyError):
        LARS = 35.0
        LA_reservoir_strain = 35.0
        LA_conduit_strain = 18.0
        LA_pump_strain = 17.0

    # LA diameter (sphere model)
    LA_diameter = 2.0 * (3.0 * LAV_max / (4.0 * np.pi)) ** (1.0/3.0)

    return {
        'LAV_max_mL':        LAV_max,
        'LAV_min_mL':        LAV_min,
        'LAV_preA_mL':       LAV_preA,
        'LAVi_mL_m2':        LAVi,               # indexed to BSA
        'LA_diameter_cm':    LA_diameter,
        'LA_total_EF_pct':   LA_total_EF,
        'LA_passive_EF_pct': LA_passive_EF,
        'LA_active_EF_pct':  LA_active_EF,
        'LARS_pct':          LARS,                # LA reservoir strain
        'LA_reservoir_strain_pct': LA_reservoir_strain,
        'LA_conduit_strain_pct':   LA_conduit_strain,
        'LA_pump_strain_pct':      LA_pump_strain,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: RV Structure and Function  [Paper Section 3.6, Atrial/RV category]
# ═══════════════════════════════════════════════════════════════════════════
# ARIC: RV diameter, TAPSE, RV s', RVFAC, RV free wall strain

def emit_RV(model, w: Dict) -> Dict:
    """
    Right ventricular size and function.

    CircAdapt source:
      Cavity['V'][:, 'cRv']   — RV volume
      Cavity['p'][:, 'cRv']   — RV pressure
      Patch['Ef'][:, 'pRv0']  — RV fiber strain

    TAPSE derivation:
      TAPSE = tricuspid annular plane systolic excursion
      ≈ ΔL_long_axis from ED to ES
      We estimate from RV volume change: TAPSE ≈ k × (RVEDV - RVESV)^(1/3)
      where k is calibrated to ~22mm for normal RV.

    RV FAC (fractional area change):
      Approximate from volume: FAC ≈ 1 - (ESV/EDV)^(2/3)  [sphere]

    ARIC: RV focused apical 4-chamber view.
    """
    V_rv = w['V_rv']
    p_rv = w['p_rv']

    RVEDV = float(np.max(V_rv))
    RVESV = float(np.min(V_rv))
    RVSV = RVEDV - RVESV
    RVEF = RVSV / RVEDV * 100.0 if RVEDV > 0 else 0

    # RV diameter (sphere approx at diastole)
    RV_basal_diam = 2.0 * (3.0 * RVEDV / (4.0 * np.pi)) ** (1.0/3.0)

    # TAPSE — approximate from longitudinal contribution to SV
    # Normal ~22mm; scales roughly with cube root of SV
    SV_ref = 60.0  # mL
    TAPSE_ref = 22.0  # mm
    TAPSE = TAPSE_ref * (RVSV / SV_ref) ** (1.0/3.0) if RVSV > 0 else 10.0

    # FAC from volume (sphere: Area ∝ V^{2/3})
    if RVEDV > 0:
        FAC = (1.0 - (RVESV / RVEDV) ** (2.0/3.0)) * 100.0
    else:
        FAC = 0.0

    # RV free wall longitudinal strain (from Patch Ef)
    try:
        Ef_rv = w['Ef_all'][:, 4]  # pRv0 typically index 4
        Ef_rv_ed = Ef_rv[np.argmax(V_rv)]
        Ef_rv_es = Ef_rv[np.argmin(V_rv)]
        RV_free_wall_strain = (np.exp(Ef_rv_es - Ef_rv_ed) - 1.0) * 100.0
    except (IndexError, KeyError):
        RV_free_wall_strain = -25.0

    # RV s' from strain rate × long-axis
    try:
        dEf_rv = np.gradient(w['Ef_all'][:, 4], w['dt'])
        RV_s_prime = float(np.max(np.abs(dEf_rv[:int(0.40*len(dEf_rv))]))) * 7.0
    except (IndexError, KeyError):
        RV_s_prime = 14.0

    return {
        'RVEDV_mL':          RVEDV,
        'RVESV_mL':          RVESV,
        'RVEF_pct':          RVEF,
        'RVSV_mL':           RVSV,
        'RV_basal_diam_cm':  RV_basal_diam,
        'TAPSE_mm':          TAPSE,
        'RV_FAC_pct':        FAC,
        'RV_s_prime_cms':    RV_s_prime,          # TDI s' at tricuspid annulus
        'RV_free_wall_strain_pct': RV_free_wall_strain,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: Aortic Doppler and Hemodynamics  [Paper Section 3.6, Hemodynamics]
# ═══════════════════════════════════════════════════════════════════════════
# ARIC: LVOT VTI, LVOT diameter, aortic valve Vmax, mean gradient, AVA

def emit_aortic_doppler(model, w: Dict) -> Dict:
    """
    Aortic valve and LVOT flow Doppler.

    CircAdapt source:
      Valve['q'][:, 'LvSyArt']   — aortic flow waveform
      Valve['A_open']['LvSyArt'] — aortic valve orifice area
      Cavity['p'][:, 'cLv'] and ['SyArt'] — pressure gradient

    LVOT VTI: integral of flow velocity during ejection
    AV Vmax: peak transaortic velocity
    Mean gradient: mean(p_lv - p_ao) during ejection
    AVA: SV / VTI_av (continuity equation)

    ARIC: PW Doppler at LVOT, CW Doppler through AV.
    """
    q_av = w['q_av']  # mL/s
    p_lv = w['p_lv']
    p_ao = w['p_ao']
    dt = w['dt']

    # Aortic valve area
    try:
        A_av_m2 = float(model['Valve']['A_open']['LvSyArt'])
        A_av_cm2 = A_av_m2 * 1e4
    except (KeyError, IndexError):
        A_av_cm2 = 4.0

    # LVOT area (slightly larger than AV)
    LVOT_diam = 2.0 * np.sqrt(A_av_cm2 / np.pi)  # cm
    A_lvot = np.pi * (LVOT_diam / 2.0) ** 2

    # Velocity through LVOT
    v_lvot = q_av / A_lvot  # cm/s

    # Ejection phase: positive aortic flow
    ejecting = q_av > 0.5
    if np.any(ejecting):
        v_lvot_ej = v_lvot[ejecting]
        AV_Vmax = float(np.max(v_lvot_ej))
        LVOT_VTI = float(np.sum(v_lvot_ej) * dt)  # cm

        # Pressure gradient during ejection
        dp = (p_lv - p_ao)
        dp_ej = dp[ejecting]
        AV_mean_grad = float(np.mean(dp_ej[dp_ej > 0])) if np.any(dp_ej > 0) else 0
        AV_peak_grad = float(np.max(dp_ej)) if len(dp_ej) > 0 else 0
    else:
        AV_Vmax = 100.0
        LVOT_VTI = 20.0
        AV_mean_grad = 5.0
        AV_peak_grad = 10.0

    # AVA by continuity: AVA = (A_lvot × VTI_lvot) / VTI_av
    # For simplicity with CircAdapt (no separate LVOT/AV area distinction):
    SV = float(np.max(w['V_lv']) - np.min(w['V_lv']))
    AVA_cont = SV / LVOT_VTI if LVOT_VTI > 0 else 3.0

    return {
        'LVOT_diam_cm':      LVOT_diam,
        'LVOT_VTI_cm':       LVOT_VTI,
        'AV_Vmax_cms':       AV_Vmax,
        'AV_peak_grad_mmHg': AV_peak_grad,
        'AV_mean_grad_mmHg': AV_mean_grad,
        'AVA_cm2':           AVA_cont,           # aortic valve area (continuity)
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: Pulmonary Pressures  [Paper Section 3.6, Hemodynamics]
# ═══════════════════════════════════════════════════════════════════════════
# ARIC: PASP (from TR jet), TR Vmax, mean PAP estimate

def emit_pulmonary_pressures(model, w: Dict) -> Dict:
    """
    Pulmonary artery pressures from TR jet and PA flow.

    CircAdapt source:
      Cavity['p'][:, 'PuArt'] — pulmonary arterial pressure
      Cavity['p'][:, 'Ra']    — RA pressure (≈ CVP for simplified Bernoulli)
      Valve['q'][:, 'RaRv']   — tricuspid flow (TR jet ∝ pressure gradient)

    PASP derivation:
      PASP = 4 × (TR_Vmax)² + RAP
      In CircAdapt: PASP ≈ max(p_PA)
      TR_Vmax estimated from max(p_RV - p_RA) during systole

    ARIC: CW Doppler through TR jet, estimated RAP.
    """
    p_pa = w['p_pa']
    p_rv = w['p_rv']
    p_ra = w['p_ra']

    # Direct from CircAdapt PA pressure
    PASP = float(np.max(p_pa))
    PADP = float(np.min(p_pa))
    mPAP = (PASP + 2.0 * PADP) / 3.0

    # TR velocity estimate
    dp_rv_ra = p_rv - p_ra
    dp_rv_ra_sys = dp_rv_ra[dp_rv_ra > 0]
    if len(dp_rv_ra_sys) > 0:
        peak_dp = float(np.max(dp_rv_ra_sys))
        # Bernoulli: ΔP = 4V² → V = sqrt(ΔP/4)
        TR_Vmax = np.sqrt(peak_dp / 4.0) * 100.0  # m/s → cm/s
        TR_Vmax_ms = TR_Vmax / 100.0  # m/s for clinical reporting
    else:
        TR_Vmax_ms = 2.5

    RAP_est = float(np.mean(p_ra))

    # PASP from simplified Bernoulli (should match direct)
    PASP_bernoulli = 4.0 * TR_Vmax_ms**2 + RAP_est

    return {
        'PASP_mmHg':         PASP,
        'PADP_mmHg':         PADP,
        'mPAP_mmHg':         mPAP,
        'TR_Vmax_ms':        TR_Vmax_ms,          # TR peak velocity (m/s)
        'RAP_est_mmHg':      RAP_est,
        'PASP_bernoulli_mmHg': PASP_bernoulli,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10: Blood Pressure  [Paper Section 3.6, Hemodynamics — H→K message source]
# ═══════════════════════════════════════════════════════════════════════════
# ARIC: SBP, DBP, pulse pressure, MAP, heart rate

def emit_blood_pressure(w: Dict) -> Dict:
    """
    Aortic/brachial blood pressure.

    CircAdapt source:
      Cavity['p'][:, 'SyArt'] — systemic arterial pressure

    Note: CircAdapt models central aortic pressure. Brachial SBP is
    typically ~5-10 mmHg higher due to pulse amplification; we apply
    a small correction.
    """
    p_ao = w['p_ao']

    SBP_central = float(np.max(p_ao))
    DBP_central = float(np.min(p_ao))

    # Brachial approximation (pulse amplification)
    SBP_brachial = SBP_central + 8.0  # small amplification offset
    DBP_brachial = DBP_central - 1.0

    MAP = (SBP_brachial + 2.0 * DBP_brachial) / 3.0
    PP = SBP_brachial - DBP_brachial
    HR = 60.0 / w['t_cycle']

    return {
        'SBP_mmHg':          SBP_brachial,
        'DBP_mmHg':          DBP_brachial,
        'MAP_mmHg':          MAP,
        'pulse_pressure_mmHg': PP,
        'HR_bpm':            HR,
        'SBP_central_mmHg':  SBP_central,
        'DBP_central_mmHg':  DBP_central,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11: RA Size
# ═══════════════════════════════════════════════════════════════════════════

def emit_RA(w: Dict, BSA: float = 1.9) -> Dict:
    """
    Right atrial volume.

    CircAdapt source:
      Cavity['V'][:, 'Ra'] — RA volume

    ARIC: RA area from apical 4-chamber, RA volume by area-length.
    """
    V_ra = w['V_ra']
    RAV_max = float(np.max(V_ra))
    RAV_min = float(np.min(V_ra))
    RAVi = RAV_max / BSA

    RA_diameter = 2.0 * (3.0 * RAV_max / (4.0*np.pi))**(1.0/3.0)

    return {
        'RAV_max_mL':        RAV_max,
        'RAV_min_mL':        RAV_min,
        'RAVi_mL_m2':        RAVi,
        'RA_diameter_cm':    RA_diameter,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 12: Myocardial Performance Index (Tei Index)
# ═══════════════════════════════════════════════════════════════════════════

def emit_MPI(w: Dict) -> Dict:
    """
    LV and RV Myocardial Performance Index (Tei index).

    MPI = (IVCT + IVRT) / ET

    CircAdapt source: timing from valve flow waveforms.
      IVCT = time from mitral valve closure to aortic valve opening
      ET   = aortic ejection time
      IVRT = time from aortic valve closure to mitral valve opening
    """
    q_mv = w['q_mv']
    q_av = w['q_av']
    dt = w['dt']
    n = len(q_mv)

    # Find valve events from flow sign changes
    mv_closed = q_mv < 0.1
    av_open = q_av > 0.5

    # Ejection time
    et_indices = np.where(av_open)[0]
    ET_ms = len(et_indices) * dt * 1e3 if len(et_indices) > 0 else 300.0

    # IVCT: MV closes → AV opens
    if np.any(mv_closed) and len(et_indices) > 0:
        mv_close_idx = np.where(mv_closed)[0][0]
        av_open_idx = et_indices[0]
        IVCT_ms = max((av_open_idx - mv_close_idx) * dt * 1e3, 0)
    else:
        IVCT_ms = 50.0

    # IVRT: from section 3
    t_es = int(np.argmin(w['V_lv']))
    mv_open_idx = t_es
    for i in range(t_es, n):
        if q_mv[i] > 0.5:
            mv_open_idx = i
            break
    IVRT_ms = (mv_open_idx - t_es) * dt * 1e3

    MPI_lv = (IVCT_ms + IVRT_ms) / ET_ms if ET_ms > 0 else 0.4

    return {
        'IVCT_ms':           IVCT_ms,
        'ET_ms':             ET_ms,                # ejection time
        'IVRT_lv_ms':        IVRT_ms,
        'MPI_LV':            MPI_lv,               # Tei index (LV)
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 13: LV Myocardial Work (Pressure-Strain Loop)
# ═══════════════════════════════════════════════════════════════════════════

def emit_myocardial_work(w: Dict) -> Dict:
    """
    LV myocardial work indices from pressure-strain loops.

    Derivation:
      GWI = ∮ p_LV · dε  (area of pressure-strain loop) [mmHg·%]
      GCW = positive work segments (shortening in systole + lengthening in IVR)
      GWW = wasted work (lengthening in systole)
      GWE = GCW / (GCW + GWW) × 100

    CircAdapt source:
      Cavity['p'][:, 'cLv'] and Patch['Ef'][:, 'pLv0']
    """
    p_lv = w['p_lv']
    try:
        Ef_lv = w['Ef_all'][:, 2]
        strain_pct = Ef_lv * 100.0  # convert to %

        # Pressure-strain loop area (GWI)
        dstrain = np.gradient(strain_pct)
        work_increments = p_lv * dstrain
        GWI = float(np.abs(np.sum(work_increments)))

        # Constructive vs wasted (simplified)
        # Constructive: shortening during systole (positive work)
        n = len(p_lv)
        sys_end = int(0.40 * n)

        systolic_work = work_increments[:sys_end]
        GCW = float(np.sum(np.abs(systolic_work[systolic_work < 0])))
        GWW = float(np.sum(np.abs(systolic_work[systolic_work > 0])))
        GWE = GCW / (GCW + GWW) * 100.0 if (GCW + GWW) > 0 else 95.0
    except (IndexError, KeyError):
        GWI = 2000.0
        GCW = 1900.0
        GWW = 100.0
        GWE = 95.0

    return {
        'GWI_mmHgpct':       GWI,                # global work index
        'GCW_mmHgpct':       GCW,                # global constructive work
        'GWW_mmHgpct':       GWW,                # global wasted work
        'GWE_pct':           GWE,                # global work efficiency
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 14: Diastolic Function Grade
# ═══════════════════════════════════════════════════════════════════════════

def emit_diastolic_grade(mitral: Dict, tdi: Dict, filling: Dict,
                          la: Dict, pulm: Dict) -> Dict:
    """
    Diastolic function grading per ASE/EACVI 2016/2025 algorithm.

    Algorithm:
      Normal:   e'_sep ≥ 7, e'_lat ≥ 10, E/e'_avg < 14, LAVi < 34, TR_v < 2.8
      Grade I:  e' abnormal, E/A < 0.8, DT > 200
      Grade II: (pseudonormal) E/A 0.8-2, E/e' > 14, LAVi > 34
      Grade III: E/A > 2, DT < 160, E/e' > 14
    """
    e_sep = tdi['e_prime_sep_cms']
    e_lat = tdi['e_prime_lat_cms']
    Ee = filling['E_e_prime_avg']
    LAVi = la['LAVi_mL_m2']
    TR_v = pulm['TR_Vmax_ms']
    EA = mitral['EA_ratio']
    DT = mitral['DT_ms']

    # Count abnormal criteria
    abnormal_count = 0
    if e_sep < 7.0:
        abnormal_count += 1
    if e_lat < 10.0:
        abnormal_count += 1
    if Ee > 14.0:
        abnormal_count += 1
    if LAVi > 34.0:
        abnormal_count += 1
    if TR_v > 2.8:
        abnormal_count += 1

    if abnormal_count <= 1:
        grade = 0  # Normal
        grade_label = 'Normal'
    elif EA < 0.8 and DT > 200:
        grade = 1  # Grade I (impaired relaxation)
        grade_label = 'Grade I (impaired relaxation)'
    elif EA > 2.0 and DT < 160:
        grade = 3  # Grade III (restrictive)
        grade_label = 'Grade III (restrictive)'
    else:
        grade = 2  # Grade II (pseudonormal)
        grade_label = 'Grade II (pseudonormal)'

    return {
        'diastolic_grade':   grade,
        'diastolic_label':   grade_label,
        'n_abnormal_criteria': abnormal_count,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 15: Vascular (Arterial Stiffness Surrogates)  [Paper Section 3.6, Vascular category]
# ═══════════════════════════════════════════════════════════════════════════

def emit_vascular(model, w: Dict) -> Dict:
    """
    Arterial stiffness surrogates from CircAdapt Tube0D parameters.

    CircAdapt source:
      Tube0D['k']         — vessel stiffness exponent
      Tube0D['p0']        — reference pressure
      Tube0D['A0']        — reference cross-section
      Cavity['p']['SyArt'] — aortic pressure waveform

    Arterial elastance (Ea):
      Ea = ESP / SV  where ESP ≈ 0.9 × SBP  (end-systolic pressure)

    Total arterial compliance:
      C_total ≈ SV / pulse_pressure

    ARIC visit 5/7: carotid-femoral pulse wave velocity (cfPWV) measured.
    We emit a surrogate from Tube0D stiffness.
    """
    p_ao = w['p_ao']
    V_lv = w['V_lv']

    SBP = float(np.max(p_ao))
    DBP = float(np.min(p_ao))
    PP = SBP - DBP
    ESP = 0.9 * SBP  # end-systolic pressure approximation
    SV = float(np.max(V_lv) - np.min(V_lv))

    Ea = ESP / SV if SV > 0 else 2.0  # arterial elastance [mmHg/mL]
    C_total = SV / PP if PP > 0 else 1.5  # total arterial compliance [mL/mmHg]

    # Ventricular-arterial coupling
    try:
        Sf_act = float(model['Patch']['Sf_act']['pLv0'])
        # Ees proxy (from active stress and volume)
        EDV = float(np.max(V_lv))
        ESV = float(np.min(V_lv))
        ESP_Pa = ESP / PA_TO_MMHG
        Ees = ESP / (ESV - 10.0) if (ESV - 10.0) > 0 else 2.5  # simplified
        VA_coupling = Ea / Ees if Ees > 0 else 1.0
    except (KeyError, IndexError):
        Ees = 2.5
        VA_coupling = 0.8

    # PWV surrogate from Tube0D stiffness
    try:
        k_ao = float(model['Tube0D']['k']['SyArt'])
        p0_ao = float(model['Tube0D']['p0']['SyArt'])
        rho_blood = 1050.0
        # c ∝ sqrt(k × p / ρ)  (from Tube0D equations)
        PWV_surrogate = np.sqrt(k_ao * (SBP / PA_TO_MMHG) / rho_blood)
    except (KeyError, IndexError):
        PWV_surrogate = 8.0

    return {
        'Ea_mmHg_mL':        Ea,                 # arterial elastance
        'Ees_mmHg_mL':       Ees,                # ventricular elastance (proxy)
        'VA_coupling':       VA_coupling,         # Ea/Ees
        'C_total_mL_mmHg':   C_total,            # total arterial compliance
        'PWV_surrogate_ms':  PWV_surrogate,       # pulse wave velocity surrogate
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 16: Renal Health Variables  [Paper Section 3.6, Renal/lab — from Hallow model]
# ═══════════════════════════════════════════════════════════════════════════
# ARIC: eGFR (CKD-EPI creatinine, cystatin C), UACR, serum creatinine,
#       cystatin C, BUN, serum Na, K, blood volume

def emit_renal(renal_state: Dict, age: float = 75.0, sex: str = 'M',
               is_black: bool = False) -> Dict:
    """
    Renal health variables from the Hallow model.

    Hallow model outputs → clinical lab values:

    GFR → eGFR (direct, mL/min/1.73m²)

    GFR → serum creatinine:
      Using CKD-EPI equation inverted:
        GFR = 141 × min(Scr/κ, 1)^α × max(Scr/κ, 1)^{-1.209} × 0.993^age
        Numerically invert for Scr given GFR, age, sex.

    GFR → cystatin C:
      CKD-EPI cystatin C: GFR = 133 × min(Scys/0.8, 1)^{-0.499} × max(...)^{-1.328} × 0.996^age
      Invert for cystatin C.

    Na excretion → UACR:
      Albuminuria correlates with glomerular damage (Kf reduction).
      UACR ≈ f(P_glom, Kf_scale) — higher glomerular pressure and
      lower Kf_scale → higher proteinuria.

    Blood volume → plasma volume, hematocrit effects.

    BUN ≈ (urea production − GFR × clearance) / V_distribution
      Simplified: BUN ∝ 1/GFR for steady state

    ARIC visits 5 & 7: serum creatinine, cystatin C, eGFR (CKD-EPI),
    UACR, BUN, serum sodium, potassium.
    """
    GFR = renal_state.get('GFR', 90.0)
    V_blood = renal_state.get('V_blood', 5000.0)
    C_Na = renal_state.get('C_Na', 140.0)
    Na_excretion = renal_state.get('Na_excretion', 150.0)
    P_glom = renal_state.get('P_glom', 60.0)
    Kf_scale = renal_state.get('Kf_scale', 1.0)
    RBF = renal_state.get('RBF', 1100.0)

    # ── eGFR (direct from model, indexed to 1.73 m²) ──────────────────
    BSA = 1.9  # average for ARIC elderly
    eGFR = GFR * 1.73 / BSA

    # ── Serum creatinine (inverted CKD-EPI) ────────────────────────────
    # Simplified steady-state: Scr ≈ k / GFR  (proportional)
    # Normal: GFR=120 → Scr~0.9 mg/dL
    if sex == 'F':
        Scr = 72.0 / max(GFR, 5.0)  # calibrated for female
    else:
        Scr = 90.0 / max(GFR, 5.0)  # calibrated for male

    # ── Cystatin C (inverted from GFR) ─────────────────────────────────
    # Normal GFR=120 → CysC ~ 0.8 mg/L
    CysC = 96.0 / max(GFR, 5.0)

    # ── BUN (inversely proportional to GFR at steady state) ────────────
    # Normal: GFR=120 → BUN~15 mg/dL
    BUN = 1800.0 / max(GFR, 5.0)

    # ── UACR (function of glomerular damage) ──────────────────────────
    # Higher glomerular pressure + lower Kf → more albumin leak
    # Normal: UACR < 30 mg/g ; microalbuminuria 30-300; macroalbuminuria >300
    # Model: UACR = baseline × (P_glom/60)^2 × (1/Kf_scale)^1.5
    UACR_baseline = 10.0  # mg/g
    UACR = UACR_baseline * (P_glom / 60.0)**2 * (1.0 / max(Kf_scale, 0.1))**1.5

    # ── Serum sodium and potassium ────────────────────────────────────
    serum_Na = C_Na  # mEq/L (directly from Hallow model)
    # K inversely related to GFR (reduced excretion in CKD)
    serum_K = 4.0 + 1.5 * (1.0 - min(GFR / 120.0, 1.0))

    # ── Blood/plasma volumes ──────────────────────────────────────────
    Hct = 0.42  # assumed
    plasma_volume = V_blood * (1.0 - Hct)

    # ── NT-proBNP surrogate (from volume overload / filling pressure) ──
    # Rises with wall stress = f(EDV, pressure)
    # Simplified: NT-proBNP ∝ exp(k × (V_blood - 5000) / 5000)
    V_excess = (V_blood - 5000.0) / 5000.0
    NTproBNP = 75.0 * np.exp(3.0 * V_excess)  # pg/mL baseline ~75

    # ── hs-Troponin T surrogate (myocardial stress) ───────────────────
    # Rises with chronic wall stress and reduced Kf (cardiorenal)
    hsTnT = 10.0 * (1.0 / max(Kf_scale, 0.1))**0.5  # ng/L

    # ── Renal resistive index (from RBF and pressures) ────────────────
    # RI = (Vsystolic - Vdiastolic) / Vsystolic
    # Approximate from MAP/RBF relationship
    RI = 1.0 - (0.6 * RBF / max(RBF, 1.0))  # simplified
    RI = max(0.55, min(0.85, 0.65 + 0.15 * (1.0 - Kf_scale)))

    return {
        # Filtration
        'eGFR_mL_min_173m2': eGFR,
        'GFR_mL_min':        GFR,                # direct model GFR
        'RBF_mL_min':        RBF,

        # Serum markers
        'serum_creatinine_mg_dL': Scr,
        'cystatin_C_mg_L':  CysC,
        'BUN_mg_dL':        BUN,

        # Urine
        'UACR_mg_g':        UACR,               # urine albumin-to-creatinine ratio

        # Electrolytes
        'serum_Na_mEq_L':   serum_Na,
        'serum_K_mEq_L':    serum_K,

        # Volume
        'blood_volume_mL':  V_blood,
        'plasma_volume_mL': plasma_volume,

        # Cardiac biomarkers
        'NTproBNP_pg_mL':   NTproBNP,
        'hsTnT_ng_L':       hsTnT,

        # Hemodynamic
        'P_glom_mmHg':      P_glom,
        'renal_resistive_index': RI,

        # Input parameters
        'Kf_scale':         Kf_scale,
        'Na_excretion_mEq_day': Na_excretion,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 17: Indexing to Body Size  [Paper Section 3.6, Indexed category]
# ═══════════════════════════════════════════════════════════════════════════

def emit_indexed(structure: Dict, systolic: Dict,
                 la: Dict, rv: Dict, ra: Dict,
                 BSA: float = 1.9, height_m: float = 1.70) -> Dict:
    """
    Body-size-indexed versions of ARIC echo variables.

    ARIC: LVMi (g/m²), LVEDVi, LVESVi (mL/m²), LAVi, etc.
    All indexed to BSA per ASE guidelines.
    Height-indexed alternatives also provided.
    """
    return {
        'LVMi_g_m2':        structure['LV_mass_g'] / BSA,
        'LVEDVi_mL_m2':     systolic['LVEDV_mL'] / BSA,
        'LVESVi_mL_m2':     systolic['LVESV_mL'] / BSA,
        'SVi_mL_m2':        systolic['SV_mL'] / BSA,
        'CI_L_min_m2':       systolic['CO_Lmin'] / BSA,
        'LAVi_mL_m2':        la['LAVi_mL_m2'],
        'RAVi_mL_m2':        ra['RAVi_mL_m2'],
        'RVEDVi_mL_m2':      rv['RVEDV_mL'] / BSA,
        'LVMi_height_g_m27': structure['LV_mass_g'] / (height_m**2.7),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION: Extract All ARIC Variables
# ═══════════════════════════════════════════════════════════════════════════

def extract_all_aric_variables(
    model,
    renal_state: Optional[Dict] = None,
    BSA: float = 1.9,
    height_m: float = 1.70,
    age: float = 75.0,
    sex: str = 'M',
    is_black: bool = False,
) -> Dict:
    """
    Extract all ARIC visit 5 / visit 7 echocardiographic, Doppler,
    and renal variables from a CircAdapt VanOsta2024 model and a
    Hallow renal model state.

    Parameters
    ----------
    model : circadapt.VanOsta2024
        Must have run at least 1 stored beat (model.run(1) after stable).
    renal_state : dict or None
        Dictionary with keys from HallowRenalModel attributes.
        If None, only cardiac variables are emitted.
    BSA : float
        Body surface area [m²] for indexing.
    height_m : float
        Height [m] for allometric indexing.
    age, sex, is_black : demographics for renal equations.

    Returns
    -------
    dict : flat dictionary with all ~80+ ARIC-matched variables.
    """
    w = _get_waveforms(model)

    # Emit each section
    structure = emit_LV_structure(model, w)
    systolic  = emit_LV_systolic(model, w)
    mitral    = emit_mitral_inflow_doppler(model, w)
    tdi       = emit_tissue_doppler(model, w)
    filling   = emit_filling_pressures(mitral, tdi)
    la        = emit_LA(model, w, BSA)
    rv        = emit_RV(model, w)
    aortic    = emit_aortic_doppler(model, w)
    pulm      = emit_pulmonary_pressures(model, w)
    bp        = emit_blood_pressure(w)
    ra        = emit_RA(w, BSA)
    mpi       = emit_MPI(w)
    mwork     = emit_myocardial_work(w)
    diast_grd = emit_diastolic_grade(mitral, tdi, filling, la, pulm)
    vasc      = emit_vascular(model, w)
    indexed   = emit_indexed(structure, systolic, la, rv, ra, BSA, height_m)

    # Combine all cardiac
    all_vars = {}
    for d in [structure, systolic, mitral, tdi, filling, la, rv,
              aortic, pulm, bp, ra, mpi, mwork, diast_grd, vasc, indexed]:
        all_vars.update(d)

    # Renal
    if renal_state is not None:
        renal_vars = emit_renal(renal_state, age, sex, is_black)
        all_vars.update(renal_vars)

    return all_vars


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: pretty-print all variables
# ═══════════════════════════════════════════════════════════════════════════

def print_aric_variables(variables: Dict):
    """Print all extracted variables grouped by category."""
    categories = {
        'LV Structure': ['LVIDd', 'LVIDs', 'IVSd', 'LVPWd', 'LV_mass', 'RWT'],
        'LV Systolic Function': ['LVEDV', 'LVESV', 'SV_mL', 'LVEF', 'CO', 'HR', 'FS', 'GLS'],
        'Mitral Inflow Doppler': ['E_vel', 'A_vel', 'EA_ratio', 'DT_ms', 'IVRT_ms'],
        'Tissue Doppler': ['e_prime', 's_prime', 'a_prime'],
        'Filling Pressures': ['E_e_prime', 'LAP_est'],
        'LA': ['LAV', 'LA_diameter', 'LA_total_EF', 'LARS', 'LA_reservoir', 'LA_conduit', 'LA_pump'],
        'RV': ['RVEDV', 'RVESV', 'RVEF', 'TAPSE', 'RV_FAC', 'RV_s_prime', 'RV_free_wall'],
        'Aortic Doppler': ['LVOT', 'AV_Vmax', 'AV_peak', 'AV_mean', 'AVA'],
        'Pulmonary Pressures': ['PASP', 'PADP', 'mPAP', 'TR_Vmax', 'RAP'],
        'Blood Pressure': ['SBP', 'DBP', 'MAP', 'pulse_pressure'],
        'RA': ['RAV'],
        'Timing / MPI': ['IVCT', 'ET_ms', 'IVRT_lv', 'MPI'],
        'Myocardial Work': ['GWI', 'GCW', 'GWW', 'GWE'],
        'Diastolic Grade': ['diastolic'],
        'Vascular': ['Ea_', 'Ees_', 'VA_coupling', 'C_total', 'PWV'],
        'Indexed': ['LVMi', 'LVEDVi', 'LVESVi', 'SVi', 'CI_', 'LAVi', 'RAVi', 'RVEDVi'],
        'Renal / Lab': ['eGFR', 'GFR_mL', 'RBF', 'serum_creatinine', 'cystatin',
                        'BUN', 'UACR', 'serum_Na', 'serum_K', 'blood_volume',
                        'plasma_volume', 'NTproBNP', 'hsTnT', 'P_glom',
                        'renal_resistive', 'Kf_scale', 'Na_excretion'],
    }

    for cat_name, prefixes in categories.items():
        matching = {k: v for k, v in variables.items()
                    if any(k.startswith(p) for p in prefixes)}
        if matching:
            print(f"\n{'─'*50}")
            print(f"  {cat_name}")
            print(f"{'─'*50}")
            for k, v in matching.items():
                if isinstance(v, float):
                    print(f"    {k:40s} = {v:10.2f}")
                else:
                    print(f"    {k:40s} = {v}")
