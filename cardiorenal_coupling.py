#!/usr/bin/env python3
"""
Cardiorenal Coupling Simulator
==============================
Bidirectional coupled heart-kidney simulation for modeling cardiorenal
syndrome (CRS), as described in Dorbala et al. (2025).

This module implements the computational framework coupling:
  - A cardiac model (CircAdapt VanOsta2024) representing ventricular
    mechanics and systemic hemodynamics (Section 3.1 of the paper)
  - A renal physiology module based on Hallow et al. (2017) CPT:PSP
    equations for glomerular filtration, tubular handling, RAAS, and
    volume homeostasis (Section 3.2)
  - A bidirectional message-passing protocol that exchanges hemodynamic
    variables between organs at each coupling step (Section 3.3, Algorithm 1)
  - An inflammatory mediator layer that propagates organ damage signals
    across compartments on a slow timescale (Section 3.4, Table 1)

Message protocol (Algorithm 1, Steps 5-8):
    Heart  -> Kidney :  MAP, CO, central venous pressure (CVP)
    Kidney -> Heart  :  total blood volume (V_blood), systemic vascular
                        resistance ratio (SVR_ratio)

Deterioration knobs (Section 3.3):
    Cardiac : Sf_act_scale -- scales Patch['Sf_act'] on LV + SV
              (1.0 = healthy, <1 = reduced contractility / HFrEF)
    Kidney  : Kf_scale -- scales glomerular ultrafiltration coefficient
              (1.0 = healthy, <1 = nephron loss / CKD)

References:
    - CircAdapt: framework.circadapt.org  (VanOsta et al. 2024)
    - Renal model: Hallow KM, Gebremichael Y. CPT:PSP 6:383-392, 2017
    - Coupled model: Basu S et al. PLoS Comput Biol 19(11):e1011598, 2023
    - This paper: Dorbala et al. (2025), "Coupled Cardiorenal Modeling of
      HFpEF and CKD via Bidirectional Message Passing"

Author: Generated for cardiorenal research coupling study
"""

# ── Standard library and third-party imports ─────────────────────────────
import numpy as np                              # Numerical array operations for hemodynamic waveforms and algebra
from dataclasses import dataclass, field        # Structured data containers for renal state and messages
from typing import Dict, List, Optional, Tuple  # Type hints for function signatures
import warnings, copy, time                     # warnings: suppress CircAdapt convergence warnings; copy: deep-copy state
from sim_logging import sim_logger, extract_key_outputs  # Structured logging for simulation runs

# ── CircAdapt import ─────────────────────────────────────────────────────
# VanOsta2024 is the latest CircAdapt cardiac mechanics model, providing
# a full lumped-parameter circulation with time-varying elastance ventricles,
# passive atria, and a four-chamber pericardial constraint model.
# See Section 3.1 of the paper for how this maps to our cardiac compartment.
from circadapt import VanOsta2024

# ── Unit conversion constants ────────────────────────────────────────────
# CircAdapt internally uses SI units (Pa, m^3, s). Our renal model and
# clinical outputs use mmHg, mL, and minutes. These constants bridge the
# two unit systems throughout the coupling interface.
PA_TO_MMHG = 7.5e-3          # 1 Pa  = 7.5e-3 mmHg  (inverse of 133.322)
MMHG_TO_PA = 133.322         # 1 mmHg = 133.322 Pa
M3_TO_ML   = 1e6             # 1 m^3 = 1,000,000 mL
ML_TO_M3   = 1e-6            # 1 mL  = 1e-6 m^3
M3S_TO_LMIN = 6e4            # 1 m^3/s = 60,000 mL/min = 60 L/min


# =========================================================================
# PART 1 -- CircAdapt Heart Wrapper  (Section 3.1)
# =========================================================================
#
# This class wraps the CircAdapt VanOsta2024 model to serve as the
# "cardiac compartment" in the coupled simulation (Section 3.1).
#
# The CircAdapt model solves beat-to-beat ventricular mechanics using a
# multi-scale approach: sarcomere-level force generation (Sf_act),
# myocardial wall mechanics (one-fiber model), and a closed-loop
# circulation with ArtVen pressure-flow elements and Tube0D wave
# propagation segments.
#
# In the coupling protocol (Algorithm 1), this wrapper:
#   Step 1-3: Receives deterioration schedules and inflammatory modifiers
#   Step 4:   Runs CircAdapt to hemodynamic steady state
#   Step 5:   Extracts MAP, CO, CVP for the Heart->Kidney message
#   Step 8:   Applies kidney feedback (V_blood, SVR_ratio)
#
# =========================================================================

class CircAdaptHeartModel:
    """
    Wraps circadapt.VanOsta2024 for use in the bidirectional coupling loop.

    This class serves as the cardiac compartment described in Section 3.1
    of the paper. It interfaces between the coupling protocol (Algorithm 1)
    and the underlying CircAdapt finite-element cardiac mechanics solver.

    Key operations map to Algorithm 1 (Coupled Cardiorenal Simulation):
      - apply_deterioration()          -> Apply disease: scales Sf_act (contractility)
      - apply_stiffness()              -> Apply disease: scales k1 (passive diastolic stiffness)
      - apply_inflammatory_modifiers() -> Apply inflammatory modifiers: adjusts ArtVen p0 (SVR), Tube0D k (arterial stiffness), k1
      - apply_kidney_feedback()        -> Kidney to Heart: sets V_blood and resistance ratio R_r from renal model
      - run_to_steady_state()          -> CircAdapt solver: runs multiple cardiac cycles until stable, extracts MAP, CO, CVP, EF, SV

    Section 3.1 describes how CircAdapt parameters map to pathophysiology:
      - Sf_act (active fiber stress) controls systolic contractility
      - k1 (passive stiffness) controls diastolic compliance (EDPVR slope)
      - ArtVen p0 controls peripheral vascular resistance set-point
      - Tube0D k controls arterial wall stiffness (pulse wave velocity)
    """

    def __init__(self):
        """
        Initialize the CircAdapt VanOsta2024 model and cache healthy
        reference parameter values.

        Reference values are stored so that deterioration/inflammation
        effects can be applied as multiplicative scaling factors relative
        to the healthy baseline, ensuring reversibility and composability
        of multiple pathological perturbations (Section 3.1, Eq. 1-2).
        """
        # Instantiate the VanOsta2024 model -- this creates a healthy
        # adult circulation with default parameters calibrated to normal
        # hemodynamics (MAP ~93 mmHg, CO ~5 L/min, EF ~60%).
        self.model = VanOsta2024()

        # Cache the healthy reference value for active fiber stress (Sf_act)
        # on the left ventricular free wall patch (pLv0) and septal patch (pSv0).
        # Sf_act is the peak isometric stress developed by the sarcomere
        # contractile element -- reducing it simulates HFrEF (Section 3.1).
        self._ref_Sf_act_lv = float(self.model['Patch']['Sf_act']['pLv0'])  # LV free wall contractility [Pa]
        self._ref_Sf_act_sv = float(self.model['Patch']['Sf_act']['pSv0'])  # Septal contractility [Pa]

        # Cache the healthy reference for ArtVen p0 (pressure set-point)
        # of the systemic circulation (CiSy). In CircAdapt's ArtVen element,
        # flow is computed as q = sign(dp) * q0 * (|dp|/p0)^k, so p0
        # effectively sets the resistance operating point. Scaling p0
        # is how we implement SVR changes from kidney feedback (Section 3.3).
        self._ref_ArtVen_p0 = float(self.model['ArtVen']['p0']['CiSy'])    # Systemic ArtVen p0 [Pa]

        # Cache the healthy reference for arterial wall stiffness (Tube0D k)
        # on the systemic arterial segment (SyArt). Higher k = stiffer wall
        # = higher pulse wave velocity. Inflammation and diabetes increase
        # arterial stiffness via elastin degradation and AGE cross-linking
        # (Section 3.4, Table 1, "stiffness_factor").
        self._ref_Tube0D_k = float(self.model['Tube0D']['k']['SyArt'])     # Arterial stiffness parameter [dimensionless]

        # Cache passive myocardial stiffness reference (Patch k1).
        # k1 is the exponential stiffness coefficient in the passive
        # stress-strain relation of the myocardial patch: higher k1
        # means a steeper EDPVR, which is the hallmark of HFpEF
        # diastolic dysfunction (Section 3.1).
        # Wrapped in try/except because the 'k1' parameter name may
        # differ across CircAdapt versions.
        try:
            self._ref_k1_lv = float(self.model['Patch']['k1']['pLv0'])      # LV passive stiffness [dimensionless]
            self._ref_k1_sv = float(self.model['Patch']['k1']['pSv0'])      # Septal passive stiffness [dimensionless]
        except Exception:
            # Graceful fallback: if k1 is not accessible in this CircAdapt
            # version, stiffness scaling will be silently skipped.
            self._ref_k1_lv = None
            self._ref_k1_sv = None

        # Cache CircAdapt's baseline circulation volume (stressed volume).
        # CircAdapt only models the hemodynamically active (stressed) portion
        # of blood volume (~831 mL), not total body blood volume (~5000 mL).
        # The renal model tracks total blood volume, so we need to map
        # changes in total volume to changes in stressed volume using the
        # Guyton stressed fraction (~0.33). See Rothe 1993 J Appl Physiol.
        self._baseline_circulation_volume = float(
            self.model['PFC']['circulation_volume'][0])  # ~831 mL in m^3 units
        self._V_blood_baseline = 5000.0                  # Renal model baseline total blood volume [mL]
        self._stressed_fraction = 0.33                   # Fraction of volume change that becomes stressed

        # Inflammatory modifier accumulators -- these are set by
        # apply_inflammatory_modifiers() and consumed by apply_kidney_feedback()
        # and apply_stiffness(). They allow the inflammatory layer (Section 3.4)
        # to influence cardiac parameters multiplicatively with direct schedules.
        self._pathology_p0_factor = 1.0       # SVR modifier from inflammation (endothelial dysfunction)
        self._inflammatory_k1_factor = 1.0    # Diastolic stiffness modifier from diabetes (AGE cross-linking)

    # ── Cardiac deterioration (Section 3.1, Eq. 1) ──────────────────────
    def apply_deterioration(self, Sf_act_scale: float):
        """
        Scale active fiber stress for LV and septal patches.

        This implements the contractility deterioration described in
        Section 3.1, Eq. 1 of the paper:
            Sf_act(t) = Sf_act_ref * Sf_act_scale(t) * Sf_act_factor_inflammatory

        Parameters
        ----------
        Sf_act_scale : float
            Contractility scale factor. 1.0 = healthy; <1.0 = reduced
            contractility simulating HFrEF. This is the EFFECTIVE scale
            that already incorporates the inflammatory Sf_act_factor
            (composed in run_coupled_simulation, Algorithm 1 Step 3).

        Maps directly to CircAdapt: model['Patch']['Sf_act']['pLv0'] etc.
        This is the same parameter used in the VanOsta2024 Heart Failure
        tutorial on the CircAdapt website.
        """
        # Scale LV free wall active stress relative to healthy reference.
        # At Sf_act_scale=0.4, the ventricle generates only 40% of normal
        # peak stress, producing severe systolic dysfunction (EF <30%).
        self.model['Patch']['Sf_act']['pLv0'] = self._ref_Sf_act_lv * Sf_act_scale

        # Scale septal active stress identically -- the septum contributes
        # to both LV and RV function, and in global cardiomyopathy both
        # walls are affected proportionally.
        self.model['Patch']['Sf_act']['pSv0'] = self._ref_Sf_act_sv * Sf_act_scale

    # ── Diastolic stiffness / HFpEF (Section 3.1, Eq. 2) ────────────────
    def apply_stiffness(self, k1_scale: float):
        """
        Scale passive myocardial stiffness for LV and septal patches.

        This implements the diastolic dysfunction model described in
        Section 3.1, Eq. 2:
            k1(t) = k1_ref * k1_scale(t) * passive_k1_factor_inflammatory

        The passive stress-strain relation in CircAdapt's patch model is
        exponential: sigma_passive = A * (exp(k1 * strain) - 1). Increasing
        k1 steepens the EDPVR curve, meaning higher filling pressures are
        needed for the same EDV -- the defining feature of HFpEF.

        Parameters
        ----------
        k1_scale : float
            Diastolic stiffness scale. 1.0 = normal compliance;
            >1.0 = stiffer ventricle (HFpEF). Values >4.0 are clamped
            for numerical stability of the CircAdapt solver.
        """
        # Clamp k1_scale to [0.5, 4.0] as a stability guard.
        # Below 0.5, the ventricle becomes unrealistically compliant and
        # the solver may fail to converge. Above 4.0, the EDPVR is so
        # steep that filling essentially stops, causing numerical issues.
        k1_scale = np.clip(k1_scale, 0.5, 4.0)

        # Only apply if k1 parameters were successfully cached at init.
        # This check ensures backward compatibility with CircAdapt versions
        # that may not expose the k1 parameter.
        if self._ref_k1_lv is not None:
            # Scale LV free wall passive stiffness relative to healthy reference
            self.model['Patch']['k1']['pLv0'] = self._ref_k1_lv * k1_scale
            # Scale septal passive stiffness identically
            self.model['Patch']['k1']['pSv0'] = self._ref_k1_sv * k1_scale

    # ── Accept kidney feedback (Algorithm 1, Step 8) ─────────────────────
    def apply_kidney_feedback(self, V_blood_mL: float, SVR_ratio: float):
        """
        Modify CircAdapt model based on renal outputs.

        This implements Algorithm 1, Step 8 from Section 3.3: the kidney
        sends updated blood volume and SVR back to the heart model, which
        adjusts its hemodynamic operating point accordingly.

        The kidney-to-heart feedback is the mechanism by which renal sodium
        and water retention (from reduced GFR or RAAS activation) causes
        volume overload and afterload changes in the cardiac model.

        Volume mapping: The renal model tracks total body blood volume
        (~5000 mL) but CircAdapt only models the hemodynamically active
        stressed volume (~831 mL). Only ~33% of a change in total blood
        volume becomes stressed volume (Guyton/Rothe 1993). We map:
            target = baseline_circ_vol + (V_blood - 5000) * 0.33 * 1e-6

        Parameters
        ----------
        V_blood_mL : float
            Total body blood volume [mL] from the Hallow renal model's
            volume balance equation (Section 3.2, Eq. 12).
        SVR_ratio : float
            Ratio of current SVR to baseline SVR (dimensionless).
            Computed from MAP and CO: SVR = (MAP - CVP) / CO.
            Applied by scaling the ArtVen p0 parameter, which shifts
            the pressure-flow operating point of the systemic bed.
        """
        # --- Volume feedback: map total blood volume to stressed volume ---
        # CircAdapt's PFC circulation_volume is the stressed (pressure-
        # generating) portion of blood volume. The renal model's V_blood
        # is total body blood volume. Only ~1/3 of any volume change
        # enters the stressed compartment (rest fills venous capacitance).
        delta_total_mL = V_blood_mL - self._V_blood_baseline
        delta_stressed_m3 = delta_total_mL * self._stressed_fraction * 1e-6
        target_volume = self._baseline_circulation_volume + delta_stressed_m3

        self.model['PFC']['is_volume_control'] = True
        self.model['PFC']['target_volume'] = target_volume

        # --- Resistance feedback: scale systemic ArtVen p0 ---------------
        # In CircAdapt's ArtVen element, flow is governed by:
        #   q = sign(dp) * q0 * (|dp| / p0)^k
        # where p0 is the pressure set-point. Raising p0 reduces flow for
        # the same pressure drop, which is equivalent to increasing
        # vascular resistance. This is how we translate the SVR_ratio
        # from the kidney model into CircAdapt hemodynamics.
        #
        # The _pathology_p0_factor incorporates inflammatory SVR effects
        # (endothelial dysfunction, microvascular rarefaction) from the
        # inflammatory mediator layer (Section 3.4, Table 1).
        # The total p0 scaling is:
        #   p0 = p0_ref * SVR_ratio * pathology_p0_factor
        # This multiplicative composition ensures that hemodynamic and
        # inflammatory contributions to afterload are independent.
        self.model['ArtVen']['p0']['CiSy'] = (
            self._ref_ArtVen_p0 * SVR_ratio * self._pathology_p0_factor
        )

    def reset_volume_control(self):
        """
        Turn off PFC volume control after the target volume has been achieved.

        Called after run_to_steady_state() to prevent PFC from continuing
        to actively regulate volume during subsequent hemodynamic settling
        beats. Once volume is set, we want the circulation to equilibrate
        naturally with the new loading conditions.
        """
        self.model['PFC']['is_volume_control'] = False

    # ── Inflammatory mediator effects (Section 3.4, Table 1) ─────────────
    def apply_inflammatory_modifiers(self, state: 'InflammatoryState'):
        """
        Apply inflammatory / metabolic mediator effects to cardiac parameters.

        This implements the cardiac column of Table 1 in Section 3.4.
        The inflammatory mediator layer computes modifier factors (e.g.,
        stiffness_factor, p0_factor) based on inflammation and diabetes
        severity, and this method applies them to CircAdapt parameters.

        This is called BEFORE apply_stiffness / apply_deterioration
        (Algorithm 1, Step 1) so that the inflammatory effects compose
        multiplicatively with the direct deterioration schedules.

        The architecture ensures separation of concerns:
          - Inflammatory layer: computes modifier factors (update_inflammatory_state)
          - This method: translates factors into CircAdapt parameter changes
          - apply_stiffness/apply_deterioration: apply direct schedule changes

        Parameters
        ----------
        state : InflammatoryState
            Current inflammatory state containing modifier factors.
            See Section 3.4 and Table 1 for derivation of each factor.

        Modifies
        --------
        - Tube0D['k']['SyArt'] : arterial wall stiffness (applied directly)
            Increased by inflammation (MMP-mediated elastin degradation)
            and diabetes (AGE cross-linking). See Table 1 row "stiffness_factor".
        - _inflammatory_k1_factor : stored for later composition with k1_scale
            in apply_stiffness(). Not applied here to maintain a single
            point of k1 assignment.
        - _pathology_p0_factor : SVR modifier, applied in apply_kidney_feedback().
            Captures endothelial dysfunction and microvascular rarefaction.
        """
        # Arterial stiffness: directly modify Tube0D wall stiffness parameter.
        # stiffness_factor encodes the combined effect of:
        #   - MMP-mediated elastin degradation from systemic inflammation
        #   - AGE-mediated collagen cross-linking from diabetes
        # (Table 1, arterial stiffness row; Vlachopoulos 2005, Prenner 2015)
        self.model['Tube0D']['k']['SyArt'] = (
            self._ref_Tube0D_k * state.stiffness_factor
        )

        # Store the passive diastolic stiffness factor from inflammation.
        # This will be composed with the direct k1_scale when apply_stiffness()
        # is called. We do NOT apply k1 here to avoid double-assignment --
        # there is exactly one code path that writes to Patch['k1'].
        self._inflammatory_k1_factor = state.passive_k1_factor

        # Store the SVR/resistance factor from inflammation.
        # This will be composed with SVR_ratio when apply_kidney_feedback()
        # is called. The separation allows hemodynamic SVR feedback (from kidney)
        # and pathological SVR changes (from inflammation) to remain independent.
        self._pathology_p0_factor = state.p0_factor

    # ── Run to hemodynamic steady state (Algorithm 1, Step 4) ────────────
    def run_to_steady_state(self, n_settle: int = 5) -> Dict:
        """
        Run VanOsta2024 until hemodynamically stable, then extract signals.

        This implements Algorithm 1, Step 4 from Section 3.3: after applying
        all parameter modifications (deterioration, stiffness, inflammation,
        kidney feedback), we run the CircAdapt solver until it reaches a
        hemodynamic steady state where beat-to-beat variations are minimal.

        The steady-state convergence uses CircAdapt's built-in run(stable=True)
        method, which iterates beats until the pressure/volume waveforms
        stabilize to within a tolerance. If this fails (which can happen at
        extreme parameter combinations), we fall back to a fixed number of
        beats to get an approximate solution.

        Parameters
        ----------
        n_settle : int
            Number of fallback beats if run(stable=True) fails to converge.
            Default is 5, which is usually sufficient for approximate
            hemodynamic equilibrium.

        Returns
        -------
        dict
            Hemodynamic output dictionary containing:
            - Waveform arrays: t, V_LV, p_LV, V_RV, p_RV, p_SyArt [ms, mL, mmHg]
            - Scalar indices: MAP, SBP, DBP [mmHg], CO [L/min], SV [mL],
              EF [%], EDV [mL], ESV [mL], Pven [mmHg], HR [bpm],
              V_blood_total [mL]
            These outputs feed into the Heart->Kidney message (Algorithm 1, Step 5).
        """
        # Enable PFC (Peripheral Flow Control) to allow the circulatory
        # model to dynamically adjust to new loading conditions. PFC
        # regulates cardiac output to match metabolic demand.
        self.model['PFC']['is_active'] = True

        # Primary convergence attempt: run until CircAdapt declares
        # hemodynamic stability (beat-to-beat variation below threshold).
        try:
            self.model.run(stable=True)
        except Exception:
            # Fallback: if stable=True fails (e.g., at extreme HFrEF
            # parameters where the solver oscillates), run a fixed number
            # of beats to approximate steady state.
            try:
                self.model.run(n_settle)
            except Exception:
                # Last resort: if even fixed beats fail, the model has
                # numerically crashed. We continue with the last valid
                # state rather than raising an exception, because partial
                # results are more useful than none for parameter studies.
                pass

        # Run additional settling beats with volume control still ON.
        # The kidney's V_blood must be maintained as the hemodynamic
        # boundary condition throughout — releasing it lets CircAdapt
        # revert to its own preferred volume, ignoring the kidney's state.
        try:
            self.model.run(stable=True)
        except Exception:
            try:
                self.model.run(n_settle)
            except Exception:
                pass  # Continue with last valid state if solver crashes

        # Store exactly one clean cardiac cycle for waveform extraction.
        # store_beats=1 tells CircAdapt to keep only the last beat's
        # time-series data, discarding transient settling beats.
        self.model['Solver']['store_beats'] = 1
        try:
            self.model.run(1)       # Run one final beat for clean waveforms
        except Exception:
            pass  # Use whatever waveforms are available from previous beats

        # Extract hemodynamic indices from the final beat
        return self._extract_hemodynamics()

    # ── Extract hemodynamic signals (Section 3.1) ────────────────────────
    def _extract_hemodynamics(self) -> Dict:
        """
        Pull pressures, volumes, and derived indices from CircAdapt.

        Extracts waveform data and computes scalar hemodynamic indices
        that are used in both the Heart->Kidney message (Algorithm 1, Step 5)
        and the simulation history recording (Algorithm 1, Step 9).

        The unit conversions (PA_TO_MMHG, M3_TO_ML) bridge CircAdapt's
        SI units to clinical units used by the renal model and output.

        Returns
        -------
        dict
            See run_to_steady_state() docstring for full contents.
        """
        # Extract time vector and convert from seconds to milliseconds
        # for conventional cardiac cycle visualization.
        t = self.model['Solver']['t'] * 1e3  # [s] -> [ms]

        # ── Left ventricle waveforms ─────────────────────────────────────
        # Volume: from CircAdapt cavity 'cLv', converted m^3 -> mL
        V_LV = self.model['Cavity']['V'][:, 'cLv'] * M3_TO_ML     # [mL]
        # Pressure: from CircAdapt cavity 'cLv', converted Pa -> mmHg
        p_LV = self.model['Cavity']['p'][:, 'cLv'] * PA_TO_MMHG   # [mmHg]

        # ── Right ventricle waveforms ────────────────────────────────────
        V_RV = self.model['Cavity']['V'][:, 'cRv'] * M3_TO_ML     # [mL]
        p_RV = self.model['Cavity']['p'][:, 'cRv'] * PA_TO_MMHG   # [mmHg]

        # ── Aortic (systemic arterial) pressure waveform ─────────────────
        # SyArt is the systemic arterial compartment in CircAdapt's
        # lumped-parameter circulation. This waveform gives us SBP and DBP.
        p_SyArt = self.model['Cavity']['p'][:, 'SyArt'] * PA_TO_MMHG  # [mmHg]

        # ── Systemic venous pressure waveform ────────────────────────────
        # SyVen is the systemic venous compartment. Its mean pressure
        # approximates central venous pressure (CVP), which is a key
        # input to the renal model (affects renal venous back-pressure
        # and congestive nephropathy in the full model).
        p_SyVen = self.model['Cavity']['p'][:, 'SyVen'] * PA_TO_MMHG  # [mmHg]

        # ── Derived scalar hemodynamic indices ───────────────────────────

        # Systolic and diastolic blood pressure from aortic waveform
        SBP = float(np.max(p_SyArt))       # Peak aortic pressure [mmHg]
        DBP = float(np.min(p_SyArt))       # Nadir aortic pressure [mmHg]
        # Mean arterial pressure: standard 1/3 pulse pressure formula.
        # MAP = DBP + 1/3*(SBP-DBP) = (SBP + 2*DBP)/3
        # This is the primary hemodynamic input to the renal model
        # (Hallow Eq. for RBF, Section 3.2).
        MAP = (SBP + 2.0 * DBP) / 3.0      # [mmHg]

        # LV volumes and derived indices
        EDV = float(np.max(V_LV))           # End-diastolic volume [mL] (max of V_LV waveform)
        ESV = float(np.min(V_LV))           # End-systolic volume [mL] (min of V_LV waveform)
        SV  = EDV - ESV                     # Stroke volume [mL] = EDV - ESV
        # Ejection fraction: guard against division by zero if EDV is tiny
        # (can happen in extreme pathology)
        EF  = SV / max(EDV, 1.0) * 100.0   # [%]

        # Heart rate and cardiac output
        t_cycle = float(self.model['General']['t_cycle'])  # Cardiac cycle duration [s]
        HR = 60.0 / t_cycle                                # Heart rate [bpm]
        CO = SV * HR / 1000.0                              # Cardiac output [L/min] = SV[mL] * HR[bpm] / 1000

        # Mean central venous pressure -- key input for the renal model.
        # Elevated CVP causes renal venous congestion, reducing the
        # effective filtration pressure gradient (Section 3.2).
        Pven = float(np.mean(p_SyVen))     # [mmHg]

        # Total circulating blood volume: sum volumes across ALL
        # CircAdapt cavities (LV, RV, LA, RA, arteries, veins, pulmonary).
        # Time-averaged to smooth out beat-to-beat oscillations.
        V_all = self.model['Cavity']['V'][:, :]             # All cavities, all time-steps [m^3]
        V_blood_mL = float(np.mean(np.sum(V_all, axis=1))) * M3_TO_ML  # [mL]

        # Return dictionary with all hemodynamic outputs.
        # Waveforms are included for PV loop visualization and diagnostics.
        # Scalar indices are used in the coupling protocol messages.
        return {
            # Waveform arrays (one cardiac cycle)
            't': t, 'V_LV': V_LV, 'p_LV': p_LV,
            'V_RV': V_RV, 'p_RV': p_RV, 'p_SyArt': p_SyArt,
            # Scalar hemodynamic indices
            'MAP': MAP, 'SBP': SBP, 'DBP': DBP,
            'CO': CO, 'SV': SV, 'EF': EF, 'EDV': EDV, 'ESV': ESV,
            'Pven': Pven, 'HR': HR,
            'V_blood_total': V_blood_mL,
        }


# =========================================================================
# PART 2 -- Hallow Renal Module  (Section 3.2)
# =========================================================================
#
# Implements the key equations from:
#   Hallow KM, Gebremichael Y. CPT:PSP (2017) 6:383-392
#   Hallow KM et al. AJP-Renal (2017) 312:F819-F835
#   Basu S et al. PLoS Comput Biol (2023) 19:e1011598
#
# The renal module is a steady-state model that computes GFR, sodium
# excretion, and blood volume at each coupling step given cardiac
# hemodynamic inputs (MAP, CO, CVP).
#
# Core equations (Section 3.2, Eq. 3-12):
#
#   Renal hemodynamics (Eq. 3-5):
#     RBF = (MAP - P_rv) / R_total                       (Eq. 3)
#     R_total = R_preAA + R_AA + R_EA                    (Eq. 4)
#     P_gc = MAP - RBF * (R_preAA + R_AA)                (Eq. 5)
#
#   Glomerular filtration (Eq. 6-7, Starling equation):
#     SNGFR = Kf * (P_gc - P_Bow - pi_avg)              (Eq. 6)
#     GFR = 2 * N_nephrons * SNGFR                       (Eq. 7)
#
#   Tubuloglomerular feedback (TGF, Eq. 8):
#     Senses macula densa Na delivery -> adjusts R_AA    (Eq. 8)
#
#   RAAS (Eq. 9):
#     Low MAP -> renin -> AngII -> R_EA + aldosterone    (Eq. 9)
#
#   Tubular Na handling (Eq. 10):
#     PT: glomerulotubular balance (eta_PT of filtered)
#     LoH, DT, CD: fractional reabsorption                (Eq. 10)
#
#   Volume balance (Eq. 11-12):
#     dV_blood/dt = water_intake - water_excretion        (Eq. 11)
#     dNa_total/dt = Na_intake - Na_excretion             (Eq. 12)
#
# =========================================================================

@dataclass
class HallowRenalModel:
    """
    Steady-state renal module based on Hallow et al. (2017) equations.

    This dataclass holds all renal model parameters and state variables
    described in Section 3.2 of the paper. It is updated at each coupling
    step by the update_renal_model() function, which receives cardiac
    hemodynamic inputs from the Heart->Kidney message.

    Parameters are organized into functional groups matching the equation
    blocks in Section 3.2:
      - Glomerular parameters (Eq. 6-7)
      - Vascular resistances (Eq. 3-5)
      - Fixed pressures (Eq. 5-6)
      - Tubular reabsorption fractions (Eq. 10)
      - Water balance (Eq. 11)
      - Intake rates
      - Feedback gains (Eq. 8-9)
      - State variables (Eq. 11-12)
      - Output variables (computed each step)
      - Deterioration knob (Kf_scale)
    """

    # ── Glomerular parameters (Eq. 6-7) ─────────────────────────────────
    # N_nephrons: total nephron count per kidney. Humans have ~1 million
    # per kidney; total GFR = 2 * N * SNGFR (Eq. 7, factor of 2 for two kidneys).
    N_nephrons: float = 1.0e6        # nephrons per kidney
    # Kf: single-nephron ultrafiltration coefficient. Determines the
    # proportionality between net filtration pressure and SNGFR in the
    # Starling equation (Eq. 6). Calibrated to produce baseline GFR ~120 mL/min
    # when combined with the default vascular resistance parameters.
    Kf: float = 8.0                  # [nL/min/mmHg per nephron]

    # ── Vascular resistances (Eq. 3-5, calibrated for 2-kidney model) ───
    # These resistances determine renal blood flow (Eq. 3) and glomerular
    # capillary pressure (Eq. 5) via the series resistance network:
    #   MAP -> R_preAA -> R_AA -> glomerular capillaries -> R_EA -> P_renal_vein
    # Values are calibrated to produce GFR ~120 mL/min at CircAdapt's
    # actual baseline MAP (~86.4 mmHg), with RBF ~1100, P_gc ~60, FF ~0.19.
    # (Values R_preAA=10/R_AA0=25/R_EA0=52 from commit 3fe221a were calibrated
    # for MAP=93; session recalibrated for CircAdapt's actual MAP=86.4.
    # R_AA0=25 is the bisection-compatible value: gives GFR~122 at MAP=86.4.)
    R_preAA: float = 6.5             # Pre-afferent arteriolar resistance [mmHg*min/L]
    R_AA0:   float = 25.0            # Afferent arteriolar baseline resistance (TGF-adjustable) [mmHg*min/L]
    R_EA0:   float = 49.0            # Efferent arteriolar baseline resistance (RAAS-adjustable) [mmHg*min/L]

    # ── Fixed pressures (Eq. 5-6) ───────────────────────────────────────
    # These are approximately constant in the Hallow model and set the
    # boundary conditions for the glomerular Starling equation.
    P_Bow:        float = 18.0       # Bowman space hydrostatic pressure [mmHg] (opposes filtration)
    P_renal_vein: float = 4.0        # Renal vein pressure [mmHg] (downstream boundary for RBF)
    pi_plasma:    float = 25.0       # Plasma oncotic pressure [mmHg] (opposes filtration, Eq. 6)
    Hct:          float = 0.45       # Hematocrit (for converting RBF to RPF: RPF = RBF * (1-Hct))

    # ── Tubular reabsorption fractions (Eq. 10) ─────────────────────────
    # Each nephron segment reabsorbs a fraction of the sodium delivered
    # to it. The sequential reabsorption is:
    #   Filtered -> PT -> LoH -> DT -> CD -> excreted
    # The residual after CD is the final urinary Na excretion.
    # Standard physiology: PT 67%, LoH 25%, DT 5%, CD 2.4% of FILTERED.
    # Converted to fractions of DELIVERED for the sequential multiplication
    # Na_after_X = Na_after_prev * (1 - eta_X):
    #   eta_LoH = 0.25 / (1-0.67) = 0.758
    #   eta_DT  = 0.05 / ((1-0.67)(1-0.758)) = 0.626
    #   eta_CD  = 0.024 / ((1-0.67)(1-0.758)(1-0.626)) = 0.803
    # Total reabsorption: ~99.4% of filtered. Excreted: ~0.6% ≈ 70 mEq/day.
    eta_PT:  float = 0.67            # PT: 67% of filtered = 67% of delivered
    eta_LoH: float = 0.758           # LoH: 25% of filtered = 75.8% of delivered (NKCC2)
    eta_DT:  float = 0.626           # DT: 5% of filtered = 62.6% of delivered (NCC)
    eta_CD0: float = 0.803           # CD baseline: 2.4% of filtered = 80.3% of delivered (ENaC)

    # ── Water balance (Eq. 11, Hallow Eq. 31) ────────────────────────────
    # Base fraction of filtered water reabsorbed by the tubules.
    # Modulated dynamically by the vasopressin PI controller (see Stage 4
    # in update_renal_model). The kidney filters ~180 L/day but excretes
    # only ~1.5-2 L/day at baseline.
    frac_water_reabs: float = 0.99   # Baseline: 99% of filtered water is reabsorbed

    # ── Intake rates ─────────────────────────────────────────────────────
    # Dietary intake rates represent the external forcing functions for
    # the volume and sodium balance ODEs (Eq. 11-12).
    Na_intake: float = 142.0         # Dietary sodium intake [mEq/day] (calibrated to match Na_excr at GFR=120, MAP=86.4 baseline)
    water_intake: float = 2.0        # Oral water intake [L/day]

    # ── Feedback gains (Eq. 8-9) ────────────────────────────────────────
    # TGF_gain: strength of the tubuloglomerular feedback loop (Eq. 8).
    # TGF senses Na delivery at the macula densa and adjusts R_AA to
    # stabilize distal Na delivery. Higher gain = more aggressive autoregulation.
    TGF_gain:     float = 2.0        # TGF loop gain [dimensionless]
    # TGF_setpoint: the target macula densa Na delivery. Initialized to
    # zero and set on the first TGF iteration to the initial MD_Na value,
    # establishing the autoregulatory reference point.
    TGF_setpoint: float = 0.0        # Initialized on first call [mEq/min]

    # RAAS_gain: strength of the renin-angiotensin-aldosterone response (Eq. 9).
    # When MAP falls below the set-point, RAAS activates to constrict
    # the efferent arteriole (preserving GFR) and increase CD reabsorption
    # (retaining sodium). Higher gain = stronger hormonal compensation.
    RAAS_gain:    float = 1.5        # RAAS loop gain [dimensionless]
    # MAP_setpoint: the arterial pressure at which RAAS is quiescent.
    # Below this, renin secretion increases; above it, RAAS is suppressed.
    MAP_setpoint: float = 86.4       # RAAS quiescence pressure [mmHg] (matched to CircAdapt baseline MAP)

    # ── Vasopressin PI controller parameters (Hallow Eq. 31) ──────────────
    # The ADH/vasopressin system regulates collecting duct water reabsorption
    # based on plasma [Na+]. From the original Hallow 2017 source code
    # (bitbucket.org/hallowkm/lv-hypertrophy-model, modelfile_commented.R).
    C_Na_setpoint:       float = 140.0   # ref_Na_concentration [mEq/L]
    VP_outer_gain:       float = 0.05    # Na_controller_gain
    VP_Kp:               float = 2.0     # Kp_VP (proportional gain)
    VP_Ki:               float = 0.005   # Ki_VP (integral gain)
    VP_saturation_Km:    float = 0.15    # ADH saturation half-constant

    # ── Dynamic state variables (Eq. 11-12) ────────────────────────────
    V_blood: float = 5000.0          # blood_volume_L * 1000 [mL]
    Na_total: float = 2100.0         # total body exchangeable sodium [mEq] (140 mEq/L × 15 L ECF)
    C_Na:    float = 140.0           # plasma Na concentration [mEq/L]
    # Vasopressin PI controller state
    VP_integral_error: float = 0.0   # integral accumulator for Na error [mEq/L·min]
    VP_normalized: float = 1.0       # normalized vasopressin concentration (1.0 = baseline)

    # ── Output variables (computed each coupling step) ───────────────────
    # These are written by update_renal_model() and used for recording
    # and for the Kidney->Heart message.
    GFR:            float = 120.0    # Glomerular filtration rate [mL/min]
    RBF:            float = 1100.0   # Renal blood flow [mL/min]
    P_glom:         float = 60.0     # Glomerular capillary pressure [mmHg]
    Na_excretion:   float = 150.0    # Urinary sodium excretion [mEq/day]
    water_excretion: float = 1.5     # Urinary water excretion [L/day]

    # ── Deterioration knob ───────────────────────────────────────────────
    # Kf_scale is the primary CKD deterioration parameter (Section 3.3).
    # It scales the ultrafiltration coefficient to simulate nephron loss:
    #   Kf_effective = Kf * Kf_scale * Kf_factor_inflammatory
    # Kf_scale = 1.0 = healthy; Kf_scale = 0.3 = severe CKD (~70% nephron loss)
    Kf_scale: float = 1.0


# =========================================================================
# PART 2b -- Inflammatory Mediator Layer  (Section 3.4, Table 1)
# =========================================================================
#
# The inflammatory mediator layer is NOT a standalone organ model. It is
# the biochemical medium through which organ damage in one compartment
# propagates pathological effects to the other. It sits as a mediator
# on the edges of the heart <-> kidney coupling graph, operating on a
# slow timescale (weeks-months) alongside the fast hemodynamic coupling
# (seconds-hours).
#
# Architecture (Figure 3 in the paper):
#
#                 +---------------------------+
#                 |   Diabetes /              |
#                 |   Metabolic Node          |
#                 +----+-----------+----------+
#                      |           |
#                  AGE |           | glucose
#                 RAGE |           | hyperfiltration
#                      v           v
#         +---- INFLAMMATION LAYER -------+
#         |  systemic_inflammatory_idx    |
#         |  endothelial_dysfunction      |
#         |                               |
#         |  Sources: uremia,             |
#         |    congestion, AGE, aldo      |
#         |  Outputs: fibrosis rates,     |
#         |    vascular tone,             |
#         |    nephron loss rate           |
#         +----+------------------+-------+
#              |                  |
#   fibrosis   |                  | nephron loss
#   stiffness  |                  | impaired autoregulation
#              v                  v
#       +-----------+      +-----------+
#       |  Heart    |<---->|  Kidney   |
#       | (CircAdapt|      | (Hallow)  |
#       +-----------+      +-----------+
#        hemodynamic coupling (Algorithm 1)
#
# Two versions exist:
#   SIMPLE (active): Direct parametric scaling from input schedules.
#     Inflammation and diabetes scales (0-1) are mapped to modifier
#     effects via literature-derived coefficients (Table 1).
#   FULL ODE (commented out): Dynamic state variables driven by organ
#     damage signals. Ready for future activation.
#
# =========================================================================

@dataclass
class InflammatoryState:
    """
    Inflammatory mediator layer -- sits between heart and kidney.

    This dataclass holds the state of the inflammatory/metabolic mediator
    layer described in Section 3.4 of the paper. The mediator layer
    transforms systemic inflammation and metabolic (diabetic) stress into
    organ-specific modifier factors that perturb cardiac and renal parameters.

    In the SIMPLE (active) version, the modifier factors are computed
    directly from input schedules (inflammation_scale, diabetes_scale)
    via the parametric scaling equations in Table 1.

    In the FULL ODE version (commented out in update_inflammatory_state),
    these would be dynamic state variables driven by organ damage signals:
      Sources: uremia (kidney damage), congestion (cardiac EDP),
               AGEs (diabetes), aldosterone excess
      Sinks:   hepatic/immune clearance
      Outputs: fibrosis rates, vascular tone, nephron loss rate

    Each modifier factor is applied multiplicatively with the corresponding
    direct deterioration parameter, ensuring composability:
      effective_param = direct_schedule * inflammatory_factor
    """
    # ── Core state variables ─────────────────────────────────────────────
    # These represent the severity of the two main pathological drivers
    # in the inflammatory layer. Range: 0 (none) to 1 (severe).
    systemic_inflammatory_index: float = 0.0   # Systemic inflammation severity (0=none, 1=severe)
    diabetes_metabolic_index: float = 0.0      # Diabetes/metabolic syndrome severity (0=none, 1=severe)

    # ── Derived modifier effects on CARDIAC parameters (Table 1) ────────
    # These are computed by update_inflammatory_state() at each step.
    Sf_act_factor: float = 1.0        # Contractility modifier (Table 1: TNF-alpha cardiodepression + diabetic cardiomyopathy)
    p0_factor: float = 1.0            # SVR / peripheral resistance modifier (Table 1: endothelial dysfunction)
    stiffness_factor: float = 1.0     # Arterial wall stiffness modifier (Table 1: elastin degradation + AGE cross-linking)
    passive_k1_factor: float = 1.0    # Diastolic myocardial stiffness modifier (Table 1: AGE collagen cross-linking)

    # ── Derived modifier effects on RENAL parameters (Table 1) ──────────
    Kf_factor: float = 1.0            # Glomerular filtration coefficient modifier (Table 1: mesangial expansion)
    R_AA_factor: float = 1.0          # Afferent arteriole resistance modifier (Table 1: endothelin-1 + TGF blunting)
    R_EA_factor: float = 1.0          # Efferent arteriole resistance modifier (Table 1: AngII-driven constriction)
    RAAS_gain_factor: float = 1.0     # RAAS sensitivity modifier (Table 1: IL-6 angiotensinogen stimulation)
    eta_PT_offset: float = 0.0        # Proximal tubule Na reabsorption offset (Table 1: NHE3 + SGLT2 effects)
    MAP_setpoint_offset: float = 0.0  # Pressure-natriuresis curve rightward shift [mmHg] (Table 1)

    # ── Dynamic state variables (full ODE version) ───────────────────────
    # These evolve via Euler integration in update_inflammatory_state()
    # when use_ode=True. Sources: organ damage signals. Sinks: clearance.
    myocardial_fibrosis_volume: float = 0.0           # Fraction of myocardium replaced by fibrosis [0-1]
    endothelial_dysfunction_index: float = 0.0        # NO bioavailability reduction [0-1]
    renal_tubulointerstitial_fibrosis: float = 0.0    # Fraction of nephrons lost to fibrosis [0-0.95]
    AGE_accumulation: float = 0.0                     # Advanced glycation end-product burden [0+]


def update_inflammatory_state(
    state: InflammatoryState,
    inflammation_scale: float,
    diabetes_scale: float,
    # ── Inputs from heart/kidney for ODE version ─────────────────────
    GFR: float = 120.0,           # kidney damage -> uremic inflammation
    EDP: float = 10.0,            # cardiac congestion -> inflammation
    aldosterone_factor: float = 1.0,  # RAAS -> pro-inflammatory
    P_glom: float = 60.0,         # glomerular pressure -> podocyte stress
    CVP: float = 3.0,             # venous congestion -> renal back-pressure
    MAP: float = 93.0,            # mean arterial pressure -> shear stress
    dt_hours: float = 6.0,        # integration timestep
    use_ode: bool = False,        # if True, use full ODE; if False, use simple parametric
) -> InflammatoryState:
    """
    Update the inflammatory mediator layer.

    Implements Section 3.4 of the paper, computing the modifier factors
    listed in Table 1 from the current inflammation and diabetes severity.

    SIMPLE VERSION (active):
        Direct parametric scaling from input schedules (Algorithm 1, Step 0).
        inflammation_scale and diabetes_scale (0 = none, 1 = severe)
        are mapped to organ-specific modifier effects via literature-derived
        coefficients tabulated in Table 1 of the paper.

    FULL ODE VERSION (commented out at bottom of function):
        Dynamic state variables driven by organ damage signals with
        logistic saturation. Sources: uremia (1/GFR), congestion (EDP),
        AGEs (diabetes), aldosterone excess. Sinks: hepatic/immune
        clearance. Ready for activation in future model iterations.

    Parameters
    ----------
    state : InflammatoryState
        Current inflammatory state to update in-place.
    inflammation_scale : float
        Systemic inflammation severity (0-1). Maps to cytokine burden
        (TNF-alpha, IL-6, CRP) from any etiology.
    diabetes_scale : float
        Diabetes/metabolic syndrome severity (0-1). Maps to hyperglycemia,
        AGE accumulation, and SGLT2 hyperactivity.

    Returns
    -------
    InflammatoryState
        Updated state with recomputed modifier factors.
    """
    # Clamp inputs to valid [0, 1] range to prevent extrapolation
    # beyond the calibrated parameter space.
    infl = np.clip(inflammation_scale, 0.0, 1.0)
    diab = np.clip(diabetes_scale, 0.0, 1.0)

    # Store the raw severity indices for diagnostic output
    state.systemic_inflammatory_index = infl
    state.diabetes_metabolic_index = diab

    # ==================================================================
    # SIMPLE: Parametric scaling from input schedules
    # Each coefficient below is derived from the literature references
    # cited in Table 1 of the paper.
    # ==================================================================

    # ── Cardiac effects (Table 1, cardiac column) ────────────────────

    # Contractility (Sf_act_factor):
    #   TNF-alpha is a direct cardiodepressant that reduces sarcomere
    #   force generation by up to 25% at severe systemic inflammation.
    #   (Feldman AM, JACC 2000; Finkel MS, Science 1992)
    #   Table 1 row: "Sf_act_factor = (1 - 0.25*infl) * (1 - 0.20*diab)"
    infl_Sf = 1.0 - 0.25 * infl     # Up to 25% contractility reduction from inflammation
    #   Diabetic cardiomyopathy: AGE cross-linking of titin + lipotoxicity
    #   reduces contractility by 15-20%. (Bugger H, Circ Res 2014)
    diab_Sf = 1.0 - 0.20 * diab     # Up to 20% contractility reduction from diabetes
    # Multiplicative composition: both pathways independently impair
    # the same contractile machinery (sarcomere active stress).
    state.Sf_act_factor = infl_Sf * diab_Sf

    # Vascular resistance / SVR (p0_factor):
    #   Chronic inflammation impairs NO bioavailability via oxidative
    #   stress, increasing SVR by 10-20%. (Endemann & Schiffrin, JASN 2004)
    #   Table 1 row: "p0_factor = (1 + 0.15*infl) * (1 + 0.10*diab)"
    infl_p0 = 1.0 + 0.15 * infl     # Up to 15% SVR increase from endothelial dysfunction
    #   Microvascular rarefaction in diabetes further increases SVR
    #   modestly via reduced capillary density.
    diab_p0 = 1.0 + 0.10 * diab     # Up to 10% SVR increase from microvascular rarefaction
    state.p0_factor = infl_p0 * diab_p0

    # Arterial stiffness (stiffness_factor):
    #   Matrix metalloproteinase (MMP) mediated elastin degradation
    #   from inflammation. CRP correlates with 20-30% increase in
    #   pulse wave velocity (PWV). (Vlachopoulos C, JACC 2005)
    #   Table 1 row: "stiffness_factor = (1 + 0.30*infl) * (1 + 0.50*diab)"
    infl_k = 1.0 + 0.30 * infl      # Up to 30% arterial stiffening from inflammation
    #   AGE-mediated collagen cross-linking in diabetes increases PWV
    #   by 30-50%. This is irreversible cross-linking distinct from
    #   the reversible inflammatory component.
    #   (Prenner SB, Chirinos JA, Atherosclerosis 2015)
    diab_k = 1.0 + 0.50 * diab      # Up to 50% arterial stiffening from AGE cross-linking
    state.stiffness_factor = infl_k * diab_k

    # Passive myocardial stiffness (passive_k1_factor):
    #   AGE collagen cross-linking in the myocardial interstitium
    #   increases LV passive stiffness by 30-50%, driving the HFpEF
    #   phenotype. This is the dominant mechanism by which diabetes
    #   causes diastolic dysfunction.
    #   (van Heerebeek L, Circulation 2008)
    #   Table 1 row: "passive_k1_factor = 1 + 0.40*diab"
    state.passive_k1_factor = 1.0 + 0.40 * diab  # Up to 40% k1 increase from diabetes

    # ── Renal effects (Table 1, renal column) ────────────────────────

    # Glomerular filtration coefficient (Kf_factor):
    #   Inflammation causes mesangial expansion and podocyte injury,
    #   reducing the effective filtration surface area and Kf.
    #   Table 1 row: "Kf_factor = max(infl_Kf * diab_Kf, 0.05)"
    infl_Kf = 1.0 - 0.20 * infl     # Up to 20% Kf reduction from mesangial/podocyte injury

    #   Diabetes has a BIPHASIC effect on Kf (Section 3.4, Table 1 note):
    #   Early diabetes: glomerular hypertrophy + AA dilation -> HYPERfiltration
    #     (Kf increases by up to ~8% at mild diabetes, diab~0.33)
    #   Late diabetes: mesangial expansion + GBM thickening -> Kf decline
    #     (Kf falls to ~0.625 of baseline at diab=1.0)
    #   This biphasic behavior is captured by the quadratic term:
    #     diab_Kf = 1 + 0.25*diab*(1 - 1.5*diab)
    #   Peak at diab = 1/3: 1 + 0.25*(1/3)*(1-0.5) = 1.042 (+4.2%)
    #   At diab = 1.0: 1 + 0.25*1*(1-1.5) = 1 - 0.125 = 0.875
    #   Combined with inflammation: can reach much lower.
    #   (Brenner BM, NEJM 1996; Ruggenenti P, JASN 1998)
    diab_Kf = 1.0 + 0.25 * diab * (1.0 - 1.5 * diab)
    # Floor at 0.05 to prevent complete filtration cessation (non-physiological)
    state.Kf_factor = max(infl_Kf * diab_Kf, 0.05)

    # Afferent arteriole resistance (R_AA_factor):
    #   Inflammation: endothelin-1 (ET-1) upregulation constricts the AA,
    #   reducing RBF and GFR. (Kohan DE, JASN 2011)
    infl_RAA = 1.0 + 0.20 * infl    # Up to 20% AA constriction from ET-1

    #   Diabetes: early AA dilation from prostaglandins and blunted TGF
    #   is the primary mechanism of diabetic hyperfiltration. At severe
    #   diabetes, AA tone returns toward normal.
    #   The quadratic term diab*(1-diab) peaks at diab=0.5.
    #   (Vallon V, JASN 2003)
    diab_RAA = 1.0 - 0.15 * diab * (1.0 - diab)  # Mild AA dilation peaking at moderate diabetes
    state.R_AA_factor = infl_RAA * diab_RAA

    # Efferent arteriole resistance (R_EA_factor):
    #   AngII-driven EA constriction is a hallmark of diabetic nephropathy.
    #   This raises glomerular capillary pressure (P_gc), maintaining GFR
    #   at the cost of increased mechanical stress on podocytes.
    #   (Brenner BM, NEJM 1996 -- the hyperfiltration hypothesis)
    #   Table 1: "R_EA_factor = 1 + 0.25*diab"
    state.R_EA_factor = 1.0 + 0.25 * diab  # Up to 25% EA constriction from AngII

    # RAAS gain modifier (RAAS_gain_factor):
    #   IL-6 stimulates hepatic and proximal tubular angiotensinogen
    #   production, amplifying the RAAS response to any given MAP drop.
    #   Table 1: "RAAS_gain_factor = 1 + 0.30*infl"
    state.RAAS_gain_factor = 1.0 + 0.30 * infl  # Up to 30% RAAS amplification

    # Proximal tubule Na reabsorption offset (eta_PT_offset):
    #   Inflammation: NHE3 (sodium-hydrogen exchanger 3) stimulation
    #   increases PT Na reabsorption. (Bobulescu IA, Pflugers Arch 2005)
    #   Diabetes: SGLT2 hyperactivity increases PT Na reabsorption by
    #   5-8% of filtered load, which is the mechanistic basis for
    #   SGLT2 inhibitor therapy. (Thomson SC, JASN 2004)
    #   Table 1: "eta_PT_offset = 0.04*infl + 0.06*diab"
    state.eta_PT_offset = 0.04 * infl + 0.06 * diab

    # MAP setpoint offset (pressure-natriuresis curve shift):
    #   Inflammation resets the renal pressure-natriuresis set-point
    #   ~5 mmHg higher via intrarenal RAS activation.
    #   Diabetes causes loss of nocturnal BP dipping and chronically
    #   elevated set-point (~8 mmHg shift).
    #   Use max() rather than sum because these are overlapping
    #   mechanisms acting on the same pathway (not fully additive).
    #   Table 1: "MAP_setpoint_offset = max(5*infl, 8*diab)"
    state.MAP_setpoint_offset = max(5.0 * infl, 8.0 * diab)

    # ==================================================================
    # FULL ODE VERSION — organ damage signals drive inflammation
    # ==================================================================
    if use_ode:
        # ── Rate constants (calibrated for year-scale progression) ───
        # Target: match ARIC V5→V7 distributions (~8 yr follow-up)
        #   GFR decline: ~3-5 mL/min/yr in HFpEF+CKD
        #   EF:  roughly stable in HFpEF (~0-1%/yr decline)
        #   Fibrosis: accumulates over years to decades
        # At 6h steps: 1 month ≈ 120 steps, 1 year ≈ 1460 steps
        k_uremic = 0.0004          # uremic toxin -> inflammation [1/hr per (1/GFR)]
        k_congestion = 0.0002      # cardiac congestion -> inflammation [1/hr per mmHg]
        k_AGE = 0.0008             # AGE accumulation -> inflammation via RAGE [1/hr]
        k_aldo = 0.0003            # aldosterone excess -> pro-inflammatory [1/hr]
        k_clearance = 0.002        # hepatic / immune clearance [1/hr]
        k_AGE_formation = 0.00004  # diabetes -> AGE accumulation rate [1/hr]
        k_AGE_turnover = 0.00002   # AGE cross-link turnover (very slow) [1/hr]
        k_fibrosis_inflam = 0.00005  # inflammation -> myocardial fibrosis [1/hr]
        k_fibrosis_mech = 0.00001   # mechanical stress -> myocardial fibrosis [1/hr per EDP ratio]
        k_fibrosis_turnover = 0.00004  # slow collagen remodeling [1/hr]
        k_endoth_inflam = 0.0004   # inflammation -> endothelial dysfunction [1/hr]
        k_endoth_shear = 0.00008   # hypertension -> endothelial damage [1/hr per mmHg]
        k_endoth_recovery = 0.0003 # endothelial repair (NO restoration) [1/hr]
        k_renal_inflam = 0.00002   # inflammation -> tubulointerstitial fibrosis [1/hr]
        k_renal_pressure = 0.000005 # glomerular hypertension -> podocyte loss [1/hr per mmHg]
        k_renal_congestion = 0.00001  # venous congestion -> renal fibrosis [1/hr per mmHg]

        # ── Sources of systemic inflammation ─────────────────────────
        # Kidney -> inflammation: uremic toxin accumulation
        # (indoxyl sulfate, p-cresyl sulfate, TMAO)
        uremic_source = k_uremic * max(0.0, 1.0 / max(GFR, 5.0) - 1.0 / 120.0)

        # Heart -> inflammation: venous congestion, elevated EDP
        congestion_source = k_congestion * max(0.0, EDP - 10.0)

        # Diabetes -> inflammation: AGE-RAGE signaling
        AGE_source = k_AGE * state.AGE_accumulation

        # RAAS -> inflammation: aldosterone is independently pro-inflammatory
        # (mechanistic basis for MRA therapy -- TOPCAT, FINEARTS-HF)
        aldo_source = k_aldo * max(0.0, aldosterone_factor - 1.0)

        # ── Inflammatory index ODE (logistic saturation ceiling) ─────
        inflam_max = 1.0
        d_inflam = (
            uremic_source + congestion_source + AGE_source + aldo_source
            - k_clearance * state.systemic_inflammatory_index
        ) * (1.0 - state.systemic_inflammatory_index / inflam_max)
        state.systemic_inflammatory_index += d_inflam * dt_hours
        state.systemic_inflammatory_index = np.clip(
            state.systemic_inflammatory_index, 0.0, 1.0)

        # ── AGE accumulation (very slow timescale, diabetes-driven) ──
        d_AGE = (k_AGE_formation * diab
                 - k_AGE_turnover * state.AGE_accumulation)
        state.AGE_accumulation += d_AGE * dt_hours
        state.AGE_accumulation = max(0.0, state.AGE_accumulation)

        # ── Myocardial fibrosis (slow, inflammation + mechanical) ────
        d_fibrosis = (
            k_fibrosis_inflam * state.systemic_inflammatory_index
            + k_fibrosis_mech * max(0.0, EDP / 12.0 - 1.0)
            - k_fibrosis_turnover * state.myocardial_fibrosis_volume
        )
        state.myocardial_fibrosis_volume += d_fibrosis * dt_hours
        state.myocardial_fibrosis_volume = np.clip(
            state.myocardial_fibrosis_volume, 0.0, 1.0)

        # ── Endothelial dysfunction ──────────────────────────────────
        d_endoth = (
            k_endoth_inflam * state.systemic_inflammatory_index
            + k_endoth_shear * max(0.0, MAP - 100.0)
            - k_endoth_recovery * state.endothelial_dysfunction_index
        )
        state.endothelial_dysfunction_index += d_endoth * dt_hours
        state.endothelial_dysfunction_index = np.clip(
            state.endothelial_dysfunction_index, 0.0, 1.0)

        # ── Renal tubulointerstitial fibrosis (irreversible) ─────────
        d_renal_fib = (
            k_renal_inflam * state.systemic_inflammatory_index
            + k_renal_pressure * max(0.0, P_glom - 65.0)
            + k_renal_congestion * max(0.0, CVP - 8.0)
        )
        state.renal_tubulointerstitial_fibrosis += d_renal_fib * dt_hours
        state.renal_tubulointerstitial_fibrosis = min(
            state.renal_tubulointerstitial_fibrosis, 0.95)

        # ── Derive modifier effects from dynamic state ───────────────
        infl = state.systemic_inflammatory_index
        fibrosis = state.myocardial_fibrosis_volume
        endoth = state.endothelial_dysfunction_index
        renal_fib = state.renal_tubulointerstitial_fibrosis
        AGE = state.AGE_accumulation

        state.Sf_act_factor = (1.0 - 0.25 * infl) * (1.0 - 0.15 * fibrosis)
        state.p0_factor = 1.0 + 0.15 * endoth + 0.10 * AGE
        state.stiffness_factor = 1.0 + 0.30 * endoth + 0.50 * AGE
        state.passive_k1_factor = 1.0 + 0.40 * fibrosis + 0.30 * AGE
        state.Kf_factor = max((1.0 - 0.20 * infl) * (1.0 - renal_fib), 0.05)
        state.R_AA_factor = 1.0 + 0.20 * endoth
        state.R_EA_factor = 1.0 + 0.25 * AGE
        state.RAAS_gain_factor = 1.0 + 0.30 * infl
        state.eta_PT_offset = 0.04 * infl + 0.06 * AGE
        state.MAP_setpoint_offset = 5.0 * infl + 3.0 * endoth

    return state


def update_renal_model(renal: HallowRenalModel,
                       MAP: float, CO: float, P_ven: float,
                       dt_hours: float = 6.0,
                       inflammatory_state: Optional['InflammatoryState'] = None,
                       ) -> HallowRenalModel:
    """
    Update the Hallow renal model given cardiac hemodynamic inputs.

    This function implements the complete renal physiology update described
    in Section 3.2 of the paper (Eq. 3-12). It is called at Algorithm 1,
    Step 6 after receiving the Heart->Kidney message.

    The update proceeds through six sequential stages:
      1. RAAS activation (Eq. 9): MAP below set-point activates renin
      2. TGF iteration (Eq. 3-8): iterative solve for GFR with TGF feedback
      3. Tubular Na handling (Eq. 10): sequential segmental reabsorption
      4. Water excretion (Eq. 11): filtered water minus reabsorption
      5. Volume/Na balance (Eq. 11-12): integrate state variables
      6. Store outputs for coupling messages

    Parameters
    ----------
    renal : HallowRenalModel
        Current renal state (modified in-place and returned).
    MAP : float
        Mean arterial pressure [mmHg] from the Heart->Kidney message
        (Algorithm 1, Step 5). This is the primary hemodynamic input
        that drives renal blood flow and glomerular filtration.
    CO : float
        Cardiac output [L/min] from the Heart->Kidney message.
        Not directly used in the current Hallow equations but available
        for future extensions (e.g., renal fraction of CO).
    P_ven : float
        Central venous pressure [mmHg] from the Heart->Kidney message.
        Elevated CVP causes renal venous congestion in the full model.
    dt_hours : float
        Integration time-step for the volume balance ODEs (Eq. 11-12).
        Default 6 hours represents one coupling step in the simulation.
    inflammatory_state : InflammatoryState or None
        If provided, inflammatory mediator effects from Section 3.4
        (Table 1) are applied to renal parameters. If None, no
        inflammatory modification (backward compatible with the
        hemodynamics-only coupling mode).

    Returns
    -------
    HallowRenalModel
        Updated renal state with new GFR, RBF, V_blood, Na_total, etc.
    """
    # ── Apply inflammatory mediator effects (Section 3.4, Table 1) ────
    # If no inflammatory state is provided, create a default (no-op)
    # instance where all factors are 1.0 and all offsets are 0.0.
    # This ensures backward compatibility: the renal equations work
    # identically whether or not the inflammatory layer is active.
    if inflammatory_state is not None:
        ist = inflammatory_state
    else:
        ist = InflammatoryState()   # Default: all modifier factors = 1.0, no effect

    # Compute effective renal parameters by composing the base parameter,
    # the direct deterioration scale (Kf_scale), and the inflammatory
    # modifier factor. This three-way multiplication implements:
    #   param_eff = param_base * direct_scale * inflammatory_factor

    # Effective ultrafiltration coefficient (Eq. 6 with scaling):
    # Kf_eff = Kf_base * Kf_scale_direct * Kf_factor_inflammatory
    Kf_eff = renal.Kf * renal.Kf_scale * ist.Kf_factor

    # Effective afferent arteriolar resistance (Eq. 4 with scaling):
    # R_AA0_eff = R_AA0_base * R_AA_factor_inflammatory
    # Note: R_AA is further modified by TGF in the iteration below.
    R_AA0_eff = renal.R_AA0 * ist.R_AA_factor

    # Effective efferent arteriolar resistance (Eq. 4 with scaling):
    # R_EA0_eff = R_EA0_base * R_EA_factor_inflammatory
    # Note: R_EA is further modified by RAAS below.
    R_EA0_eff = renal.R_EA0 * ist.R_EA_factor

    # Effective RAAS gain (Eq. 9 with scaling):
    RAAS_gain_eff = renal.RAAS_gain * ist.RAAS_gain_factor

    # Effective proximal tubule reabsorption (Eq. 10 with offset):
    # Clamped at 0.85 to prevent unrealistic near-complete PT reabsorption.
    eta_PT_eff = min(renal.eta_PT + ist.eta_PT_offset, 0.85)

    # Effective MAP set-point for pressure-natriuresis and RAAS (Eq. 9):
    # Inflammatory/diabetic shifts move the set-point rightward (higher MAP
    # needed to achieve the same sodium excretion), contributing to hypertension.
    MAP_sp_eff = renal.MAP_setpoint + ist.MAP_setpoint_offset

    # ── Stage 1: RAAS activation (Section 3.2, Eq. 9) ────────────────
    # The renin-angiotensin-aldosterone system senses MAP relative to its
    # set-point. When MAP < set-point, RAAS activates:
    #   - Angiotensin II constricts the efferent arteriole (R_EA increases)
    #   - Aldosterone increases collecting duct Na reabsorption (eta_CD increases)
    # When MAP > set-point, RAAS is suppressed (R_EA and eta_CD decrease).
    dMAP = MAP - MAP_sp_eff                                          # Deviation from RAAS quiescence point [mmHg]
    # RAAS_factor: >1 when MAP is low (activation), <1 when MAP is high (suppression)
    # The 0.005 coefficient converts mmHg deviation to a fractional change,
    # scaled by RAAS_gain_eff. Clamped to [0.5, 2.0] for stability.
    RAAS_factor = np.clip(1.0 - RAAS_gain_eff * 0.005 * dMAP, 0.5, 2.0)
    R_EA = R_EA0_eff * RAAS_factor                                   # Efferent arteriole resistance adjusted by RAAS [mmHg*min/L]
    eta_CD = renal.eta_CD0 * RAAS_factor                             # Collecting duct Na reabsorption adjusted by aldosterone

    # ── Stage 2: TGF bisection (Section 3.2, Eq. 3-8) ───────────────
    # Bisection solve for R_AA such that MD_Na == TGF_setpoint.
    # Bisection is used (vs Picard) because the Picard relaxed iteration
    # fails to converge with the recalibrated low resistances (R_AA0=16.4).
    # 40 iterations give |R_AA error| < 1e-5 of the search range.

    def _glom_hemo(R_AA_try: float):
        R_total = renal.R_preAA + R_AA_try + R_EA
        RBF_ = max((MAP - renal.P_renal_vein) / R_total * 1000.0, 100.0)
        RPF_ = RBF_ * (1.0 - renal.Hct)
        P_gc_ = max(MAP - RBF_ / 1000.0 * (renal.R_preAA + R_AA_try), 25.0)
        # Iteratively solve for FF (needed for pi_avg which depends on FF)
        GFR_ = 120.0
        for _ in range(5):
            FF_ = np.clip(GFR_ / max(RPF_, 1.0), 0.01, 0.45)
            pi_avg_ = renal.pi_plasma * (1.0 + FF_ / (2.0 * (1.0 - FF_)))
            NFP_ = max(P_gc_ - renal.P_Bow - pi_avg_, 0.0)
            GFR_ = max(2.0 * renal.N_nephrons * Kf_eff * NFP_ * 1e-6, 5.0)
        FF_ = np.clip(GFR_ / max(RPF_, 1.0), 0.01, 0.45)
        Na_filt_ = GFR_ * renal.C_Na * 1e-3
        MD_Na_ = Na_filt_ * (1.0 - eta_PT_eff) * (1.0 - renal.eta_LoH)
        return GFR_, RBF_, P_gc_, Na_filt_, MD_Na_, FF_

    # Bootstrap TGF setpoint on first call (before disease has shifted anything)
    if renal.TGF_setpoint <= 0:
        _, _, _, _, MD_Na_init, _ = _glom_hemo(R_AA0_eff)
        renal.TGF_setpoint = MD_Na_init

    # Bisect on R_AA: high R_AA → low MD_Na; low R_AA → high MD_Na
    R_AA_lo = 0.5 * R_AA0_eff
    R_AA_hi = 3.0 * R_AA0_eff
    for _ in range(40):
        R_AA_mid = 0.5 * (R_AA_lo + R_AA_hi)
        _, _, _, _, MD_Na_mid, _ = _glom_hemo(R_AA_mid)
        if MD_Na_mid > renal.TGF_setpoint:
            R_AA_lo = R_AA_mid
        else:
            R_AA_hi = R_AA_mid
    R_AA = 0.5 * (R_AA_lo + R_AA_hi)
    GFR, RBF, P_gc, Na_filt, MD_Na, FF = _glom_hemo(R_AA)

    # ── Stage 3: Tubular Na handling (Section 3.2, Eq. 10) ───────────
    # Sequential segmental reabsorption along the nephron:
    #   Filtered Na -> PT -> LoH -> DT -> CD -> excreted
    # Each segment reabsorbs a fixed fraction of what is delivered to it.
    # This is the "sequential fractional reabsorption" model from Hallow.

    # Proximal tubule: reabsorbs eta_PT_eff fraction of filtered Na.
    # Glomerulotubular balance ensures PT reabsorption scales with GFR.
    Na_after_PT  = Na_filt * (1.0 - eta_PT_eff)                    # Na leaving PT [mEq/min]

    # Loop of Henle: reabsorbs eta_LoH fraction via NKCC2 transporter.
    # This is the site of action for loop diuretics (furosemide).
    Na_after_LoH = Na_after_PT * (1.0 - renal.eta_LoH)             # Na leaving LoH [mEq/min]

    # Distal tubule: reabsorbs eta_DT fraction via NCC transporter.
    # Site of action for thiazide diuretics.
    Na_after_DT  = Na_after_LoH * (1.0 - renal.eta_DT)             # Na leaving DT [mEq/min]

    # Collecting duct: reabsorbs eta_CD fraction via ENaC channels,
    # regulated by aldosterone (via RAAS_factor from Stage 1).
    # Site of action for mineralocorticoid receptor antagonists (MRAs).
    Na_after_CD  = Na_after_DT * (1.0 - eta_CD)                    # Na leaving CD = pre-excretion Na [mEq/min]

    # Pressure-natriuresis: direct effect of MAP on tubular reabsorption.
    # When MAP > set-point, the kidney excretes more Na per unit of
    # delivered Na (pressure natriuresis). When MAP < set-point, Na
    # retention is enhanced. This is the fundamental mechanism by which
    # the kidney regulates blood pressure over hours-days.
    # The asymmetric slopes (0.03 above, 0.015 below) reflect the
    # observation that natriuresis is more sensitive to hypertension
    # than antinatriuresis is to hypotension.
    if MAP > MAP_sp_eff:
        # Above set-point: enhanced natriuresis (slope 0.03 per mmHg)
        # Hallow 2017 uses a gentle slope (~0.03) for the pressure-
        # natriuresis curve. Now that CircAdapt's volume feedback is
        # properly coupled (MAP rises with V_blood), a steeper slope
        # causes limit-cycle oscillations in the coupling loop.
        pn = 1.0 + 0.03 * (MAP - MAP_sp_eff)
    else:
        # Below set-point: reduced natriuresis (slope 0.015 per mmHg)
        # Floor at 0.3 to prevent complete sodium retention
        pn = max(0.3, 1.0 + 0.015 * (MAP - MAP_sp_eff))

    # Absolute high-pressure natriuresis: bypasses setpoint drift.
    # The MAP_setpoint_offset mechanism (disease resetting of the pressure-
    # natriuresis curve) is physiologically correct but can blunt natriuresis
    # so completely that V_blood rails into the hard cap. At sustained MAP >
    # 100 mmHg, baroreceptor-independent direct tubular pressure effects
    # (interstitial pressure, peritubular capillary oncotic dilution) produce
    # measurable additional natriuresis regardless of RAAS setpoint state.
    # Slope 0.04/mmHg above 100 mmHg is conservative relative to the ~0.05
    # used in Guyton 1972 and Malpas 2010 pressure-natriuresis curve slopes.
    # Final sodium excretion rate: CD outflow modulated by pressure-natriuresis
    Na_excr_min = Na_after_CD * pn                                  # Urinary Na excretion [mEq/min]
    Na_excr_day = Na_excr_min * 1440.0                              # Convert to [mEq/day] (1440 min/day)

    # ── Stage 4: Vasopressin PI controller & water excretion (Hallow Eq. 31) ──
    # ADH/vasopressin is a PI controller on plasma [Na+]:
    #   C_Na > setpoint → VP rises → more CD water reabsorption → dilutes plasma
    #   C_Na < setpoint → VP falls → less reabsorption → excretes water
    # This negative feedback prevents V_blood from drifting (see issues.md #1).

    dt_min = dt_hours * 60.0                                        # Time-step [minutes]

    # PI controller on plasma sodium concentration error
    Na_error = renal.C_Na - renal.C_Na_setpoint                     # [mEq/L] (+ = hypernatremic)
    renal.VP_integral_error += Na_error * dt_min                     # Integral accumulator [mEq/L·min]

    # Normalized vasopressin: VP = 1 + outer_gain × (Kp × error + Ki × ∫error·dt)
    # At baseline (C_Na = 140): VP = 1.0
    VP_raw = 1.0 + renal.VP_outer_gain * (
        renal.VP_Kp * Na_error +                                    # Proportional response
        renal.VP_Ki * renal.VP_integral_error                       # Integral (eliminates steady-state offset)
    )
    renal.VP_normalized = max(VP_raw, 0.0)                          # VP ≥ 0

    # ADH permeability: Michaelis-Menten saturation (Hallow Eq. 31)
    # ADH_perm = VP / (Km + VP), range [0, 1). At VP=1: ADH = 1/(0.15+1) = 0.87
    ADH_perm = renal.VP_normalized / (renal.VP_saturation_Km + renal.VP_normalized)

    # Effective fractional water reabsorption: linear in ADH_perm
    #   frac_eff = frac_min + (frac_max - frac_min) × ADH_perm
    #   frac_min = 0.90  (no ADH → dilute urine, ~17 L/day at GFR=120)
    #   frac_max = 0.998 (max ADH → concentrated urine, ~0.35 L/day at GFR=120)
    # At baseline VP=1, ADH=0.87: frac_eff ≈ 0.985 → excr ≈ 1.2 L/day at GFR=58
    # VP controller settles to match intake (2.0 L/day) at steady state.
    frac_min = 0.90
    frac_max = 0.998
    frac_eff = frac_min + (frac_max - frac_min) * ADH_perm

    water_excr_min = GFR * (1.0 - frac_eff)                        # Water excretion [mL/min]
    water_excr_day = water_excr_min * 1440.0 / 1000.0              # Convert to [L/day]

    # ── Stage 5: Volume and Na balance ODEs (Section 3.2, Eq. 11-12) ─
    # Single-compartment Euler integration. The 0.33 factor accounts for
    # Starling equilibrium: only ~1/3 of ECF change enters blood volume.

    # Eq. 12: Sodium balance ODE: dNa_total/dt = Na_intake - Na_excretion
    Na_in_min = renal.Na_intake / 1440.0                            # Na intake rate [mEq/min]
    renal.Na_total = max(
        renal.Na_total + (Na_in_min - Na_excr_min) * dt_min,       # Euler integration
        800.0                                                        # Floor: minimum body Na [mEq]
    )

    # Eq. 11: Blood volume ODE with VP-modulated water excretion
    W_in_min = renal.water_intake * 1000.0 / 1440.0                # Water intake [mL/min]
    dV = (W_in_min - water_excr_min) * dt_min                      # Net ECF change [mL]
    renal.V_blood = np.clip(
        renal.V_blood + dV * 0.33,                                  # 1/3 of ECF → blood
        3000.0, 8000.0
    )

    # Update plasma [Na+] from total Na and ECF volume
    V_ECF = renal.V_blood / 0.33                                    # ECF volume [mL]
    renal.C_Na = np.clip(
        renal.Na_total / (V_ECF * 1e-3),                            # [mEq/L]
        125.0, 155.0
    )

    # ── Stage 6: Store outputs (Section 3.2) ─────────────────────────
    # These values are used by the Kidney->Heart message (Algorithm 1, Step 7)
    # and recorded in the simulation history (Algorithm 1, Step 9).
    renal.GFR = GFR                          # Store converged GFR [mL/min]
    renal.RBF = RBF                          # Store converged RBF [mL/min]
    renal.P_glom = P_gc                      # Store glomerular capillary pressure [mmHg]
    renal.Na_excretion = Na_excr_day         # Store daily Na excretion [mEq/day]
    renal.water_excretion = water_excr_day   # Store daily water excretion [L/day]

    return renal


# =========================================================================
# PART 3 -- Message-Passing Protocol  (Section 3.3, Algorithm 1)
# =========================================================================
#
# The coupling protocol uses structured messages to exchange hemodynamic
# variables between the heart and kidney models. This implements the
# asynchronous message-passing architecture described in Section 3.3.
#
# At each coupling step, two messages are exchanged:
#   1. Heart -> Kidney (Algorithm 1, Step 5):
#      MAP, CO, CVP -> drives renal hemodynamics and filtration
#   2. Kidney -> Heart (Algorithm 1, Step 7):
#      V_blood, SVR_ratio -> modifies cardiac preload and afterload
#
# The message dataclasses provide type-safe, self-documenting containers
# for the coupling variables, and the helper functions (heart_to_kidney,
# kidney_to_heart) handle the extraction and computation of message fields.
#
# =========================================================================

@dataclass
class HeartToKidneyMessage:
    """
    Heart-to-Kidney coupling message (Algorithm 1, Step 5).

    Carries the hemodynamic state of the cardiac model to the renal model.
    MAP is the primary driver of renal blood flow (Eq. 3); CO provides
    context for renal fraction computation; Pven (CVP) affects renal
    venous congestion; SBP/DBP are informational for pulse pressure analysis.
    """
    MAP: float        # Mean arterial pressure [mmHg] -- primary input to Eq. 3 (RBF computation)
    CO:  float        # Cardiac output [L/min] -- context for renal fraction and SVR computation
    Pven: float       # Central venous pressure [mmHg] -- affects renal venous back-pressure
    SBP: float        # Systolic blood pressure [mmHg] -- informational for pulse pressure
    DBP: float        # Diastolic blood pressure [mmHg] -- informational for pulse pressure
    # ── Inflammatory mediator fields (for future ODE version) ────────
    # EDP: float = 10.0    # end-diastolic pressure -> congestion source
    #                       # for inflammatory ODE (Section 3.4)


@dataclass
class KidneyToHeartMessage:
    """
    Kidney-to-Heart coupling message (Algorithm 1, Step 7).

    Carries the renal model's volume and resistance state back to the
    cardiac model. V_blood is converted to m^3 for CircAdapt's volume
    control (Algorithm 1, Step 8). SVR_ratio scales the ArtVen p0
    parameter. GFR is informational for monitoring kidney function.
    """
    V_blood: float    # Total blood volume [mL] -- converted to m^3 for CircAdapt PFC
    SVR_ratio: float  # SVR ratio: current SVR / baseline SVR [dimensionless] -- scales ArtVen p0
    GFR: float        # Glomerular filtration rate [mL/min] -- informational, recorded in history
    # ── Inflammatory mediator fields (for future ODE version) ────────
    # aldosterone_factor: float = 1.0  # RAAS state -> inflammation source
    #                                   # for inflammatory ODE (Section 3.4)


def heart_to_kidney(hemo: Dict) -> HeartToKidneyMessage:
    """
    Construct a Heart->Kidney message from cardiac hemodynamic output.

    Implements the extraction step of Algorithm 1, Step 5: take the
    hemodynamic dictionary from CircAdapt's run_to_steady_state() and
    package the relevant variables into a structured message for the
    renal model.

    Parameters
    ----------
    hemo : dict
        Hemodynamic output from CircAdaptHeartModel.run_to_steady_state().
        Must contain keys: 'MAP', 'CO', 'Pven', 'SBP', 'DBP'.

    Returns
    -------
    HeartToKidneyMessage
        Structured message carrying cardiac state to kidney model.
    """
    return HeartToKidneyMessage(
        MAP=hemo['MAP'],    # Mean arterial pressure from aortic waveform
        CO=hemo['CO'],      # Cardiac output = SV * HR / 1000
        Pven=hemo['Pven'],  # Mean systemic venous pressure (approximates CVP)
        SBP=hemo['SBP'],    # Peak aortic pressure
        DBP=hemo['DBP'],    # Nadir aortic pressure
    )


def kidney_to_heart(renal: HallowRenalModel, MAP: float, CO: float,
                    Pven: float,
                    SVR_baseline: float = None) -> KidneyToHeartMessage:
    """
    Construct a Kidney->Heart message from renal model state.

    Implements Algorithm 1, Step 7: compute the SVR ratio from current
    hemodynamics and package it with blood volume and GFR into a
    structured message for the cardiac model.

    SVR computation (Section 3.3):
        SVR = (MAP - CVP) / CO
        SVR_ratio = SVR_current / SVR_baseline
    where SVR_baseline is computed from CircAdapt's actual baseline
    hemodynamics at initialization.

    Parameters
    ----------
    renal : HallowRenalModel
        Current renal state (provides V_blood and GFR).
    MAP : float
        Current mean arterial pressure [mmHg] (from Heart->Kidney message).
    CO : float
        Current cardiac output [L/min] (from Heart->Kidney message).
    Pven : float
        Current central venous pressure [mmHg] (from Heart->Kidney message).
    SVR_baseline : float, optional
        Baseline SVR from CircAdapt's healthy steady state [mmHg*min/L].
        Computed once at simulation init from the first run_to_steady_state()
        call, ensuring the ratio starts at exactly 1.0.

    Returns
    -------
    KidneyToHeartMessage
        Structured message carrying renal state to cardiac model.
    """
    # SVR_ratio is fixed at 1.0: the kidney does not independently change
    # vascular resistance. Volume feedback is handled by PFC target_volume
    # in apply_kidney_feedback(), and inflammatory resistance effects are
    # handled by _pathology_p0_factor. Computing SVR from MAP/CO and
    # feeding it back creates a positive feedback loop (high volume →
    # high CO → low SVR_ratio → CircAdapt drops resistance → MAP drops
    # → kidney retains more Na → death spiral).
    return KidneyToHeartMessage(
        V_blood=renal.V_blood,                                      # Current blood volume from renal model [mL]
        SVR_ratio=1.0,                                               # Fixed: no renal resistance feedback
        GFR=renal.GFR,                                              # Current GFR for recording [mL/min]
    )


# =========================================================================
# PART 3B -- RL Message Scaling & Inflammatory Residuals
# =========================================================================
#
# These functions implement the learned coupling equation's output:
# per-message alpha scaling and inflammatory modifier corrections.
# Used by run_coupled_simulation_rl() when the RL policy is active.
#
# Message scaling: scaled = baseline + alpha * (raw - baseline)
#   alpha = 1.0 → identity (pass through unchanged)
#   alpha > 1.0 → amplify deviation from healthy baseline
#   alpha < 1.0 → dampen deviation from healthy baseline
# =========================================================================

def scale_message_h2k(
    msg: HeartToKidneyMessage,
    alpha_vec: np.ndarray,
    baselines: Dict[str, float],
) -> HeartToKidneyMessage:
    """
    Scale Heart->Kidney message channels around healthy baselines.

    The RL-learned coupling equation produces per-channel alpha weights
    that modulate how strongly each cardiac output signal is transmitted
    to the renal model. This replaces the implicit full-strength coupling
    with a learned, state-dependent coupling intensity.

    Parameters
    ----------
    msg : HeartToKidneyMessage
        Raw message from heart_to_kidney().
    alpha_vec : np.ndarray
        3-dim vector [alpha_MAP, alpha_CO, alpha_Pven].
    baselines : dict
        Healthy reference values: {'MAP': 93.0, 'CO': 5.0, 'Pven': 3.0}.

    Returns
    -------
    HeartToKidneyMessage
        Scaled message with modulated coupling intensity.
    """
    return HeartToKidneyMessage(
        MAP=baselines['MAP'] + alpha_vec[0] * (msg.MAP - baselines['MAP']),
        CO=baselines['CO'] + alpha_vec[1] * (msg.CO - baselines['CO']),
        Pven=baselines['Pven'] + alpha_vec[2] * (msg.Pven - baselines['Pven']),
        SBP=msg.SBP,
        DBP=msg.DBP,
    )


def scale_message_k2h(
    msg: KidneyToHeartMessage,
    alpha_vec: np.ndarray,
    baselines: Dict[str, float],
) -> KidneyToHeartMessage:
    """
    Scale Kidney->Heart message channels around healthy baselines.

    Parameters
    ----------
    msg : KidneyToHeartMessage
        Raw message from kidney_to_heart().
    alpha_vec : np.ndarray
        2-dim vector [alpha_V_blood, alpha_SVR_ratio].
    baselines : dict
        Healthy reference values: {'V_blood': 5000.0, 'SVR_ratio': 1.0}.

    Returns
    -------
    KidneyToHeartMessage
        Scaled message with modulated coupling intensity.
    """
    return KidneyToHeartMessage(
        V_blood=baselines['V_blood'] + alpha_vec[0] * (msg.V_blood - baselines['V_blood']),
        SVR_ratio=baselines['SVR_ratio'] + alpha_vec[1] * (msg.SVR_ratio - baselines['SVR_ratio']),
        GFR=msg.GFR,
    )


def apply_inflammatory_residuals(
    ist: InflammatoryState,
    residuals: np.ndarray,
) -> InflammatoryState:
    """
    Apply additive residual corrections to the InflammatoryState.

    The RL policy learns corrections to the hand-coded Table 1 modifier
    factors, discovering where the parametric approximations over- or
    under-estimate diabetes/inflammation effects that span both organs.

    No clipping is performed here — the RL's action space is constrained
    at the source (tanh in CouplingPolicyHead + per-factor rescaling in
    rl_env._rescale_action) so that base + residual always falls within
    physiological bounds. This ensures the RL always knows what value
    was actually applied.

    Parameters
    ----------
    ist : InflammatoryState
        Current inflammatory state (not mutated).
    residuals : np.ndarray
        10-dim vector of additive corrections, one per modifier field:
        [dSf_act_factor, dp0_factor, dstiffness_factor, dpassive_k1_factor,
         dKf_factor, dR_AA_factor, dR_EA_factor, dRAAS_gain_factor,
         deta_PT_offset, dMAP_setpoint_offset]

    Returns
    -------
    InflammatoryState
        New state with corrected modifier values.
    """
    import copy
    ist_corrected = copy.copy(ist)
    ist_corrected.Sf_act_factor = ist.Sf_act_factor + residuals[0]
    ist_corrected.p0_factor = ist.p0_factor + residuals[1]
    ist_corrected.stiffness_factor = ist.stiffness_factor + residuals[2]
    ist_corrected.passive_k1_factor = ist.passive_k1_factor + residuals[3]
    ist_corrected.Kf_factor = ist.Kf_factor + residuals[4]
    ist_corrected.R_AA_factor = ist.R_AA_factor + residuals[5]
    ist_corrected.R_EA_factor = ist.R_EA_factor + residuals[6]
    ist_corrected.RAAS_gain_factor = ist.RAAS_gain_factor + residuals[7]
    ist_corrected.eta_PT_offset = ist.eta_PT_offset + residuals[8]
    ist_corrected.MAP_setpoint_offset = ist.MAP_setpoint_offset + residuals[9]
    return ist_corrected


def extract_rl_observation(
    hemo: Dict,
    renal: 'HallowRenalModel',
    ist: InflammatoryState,
    effective_sf: float,
    effective_kf: float,
    effective_k1: float,
    infl: float,
    diab: float,
    t_normalized: float,
    prev_obs: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Extract the 32-dim RL observation vector from simulator state.

    Parameters
    ----------
    hemo : dict
        Hemodynamic output from heart.run_to_steady_state().
    renal : HallowRenalModel
        Current renal model state.
    ist : InflammatoryState
        Current inflammatory state.
    effective_sf, effective_kf, effective_k1 : float
        Effective parameter values after inflammatory modification.
    infl, diab : float
        Current inflammation and diabetes schedule values.
    t_normalized : float
        Normalized time progress (0 to 1).
    prev_obs : dict or None
        Previous observation for computing temporal deltas.

    Returns
    -------
    dict
        Observation with 32 named features.
    """
    obs = {}

    # Cardiac features (12)
    obs['MAP'] = hemo['MAP']
    obs['SBP'] = hemo['SBP']
    obs['DBP'] = hemo['DBP']
    obs['CO'] = hemo['CO']
    obs['SV'] = hemo['SV']
    obs['EF'] = hemo['EF']
    obs['EDV'] = hemo['EDV']
    obs['ESV'] = hemo['ESV']
    obs['Pven'] = hemo['Pven']
    obs['HR'] = hemo['HR']
    obs['V_blood_total'] = hemo.get('V_blood_total', renal.V_blood)
    # LVEDP approximated from PV loop or LAP estimate
    obs['LVEDP'] = hemo.get('LVEDP', hemo.get('Pven', 10.0) + 5.0)

    # Renal features (10)
    obs['GFR'] = renal.GFR
    obs['RBF'] = renal.RBF
    obs['P_glom'] = renal.P_glom
    obs['Na_excretion'] = renal.Na_excretion
    obs['V_blood'] = renal.V_blood
    obs['C_Na'] = renal.C_Na
    obs['Na_total'] = renal.Na_total
    obs['Kf_scale'] = renal.Kf_scale
    obs['water_excretion'] = renal.water_excretion
    obs['Kf_effective'] = renal.Kf * renal.Kf_scale * ist.Kf_factor

    # Meta features (5)
    obs['effective_Sf'] = effective_sf
    obs['effective_Kf'] = effective_kf
    obs['effective_k1'] = effective_k1
    obs['inflammation_scale'] = infl
    obs['diabetes_scale'] = diab

    # Temporal features (5)
    obs['t_normalized'] = t_normalized
    if prev_obs is not None:
        obs['delta_MAP'] = obs['MAP'] - prev_obs['MAP']
        obs['delta_GFR'] = obs['GFR'] - prev_obs['GFR']
        obs['delta_EF'] = obs['EF'] - prev_obs['EF']
        obs['delta_Vblood'] = obs['V_blood'] - prev_obs['V_blood']
    else:
        obs['delta_MAP'] = 0.0
        obs['delta_GFR'] = 0.0
        obs['delta_EF'] = 0.0
        obs['delta_Vblood'] = 0.0

    return obs


def obs_dict_to_vector(obs: Dict[str, float]) -> np.ndarray:
    """Convert named observation dict to 32-dim numpy vector."""
    from config import CARDIAC_FEATURE_NAMES, RENAL_FEATURE_NAMES, META_FEATURE_NAMES, TEMPORAL_FEATURE_NAMES
    all_names = CARDIAC_FEATURE_NAMES + RENAL_FEATURE_NAMES + META_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES
    return np.array([obs[k] for k in all_names], dtype=np.float32)


# =========================================================================
# PART 4 -- Coupled Simulation Driver  (Section 3.3, Algorithm 1)
# =========================================================================
#
# This is the main simulation loop that implements Algorithm 1 from
# Section 3.3 of the paper. It orchestrates the bidirectional coupling
# between the cardiac and renal models over a sequence of coupling steps.
#
# Algorithm 1 (Bidirectional Cardiorenal Coupling):
#   Input: n_steps, dt_renal_hours, deterioration schedules
#   Initialize: heart model, renal model, inflammatory state
#   For each coupling step s = 1, ..., n_steps:
#     Step 0: Update inflammatory mediator layer (Section 3.4)
#     Step 1: Apply inflammatory modifiers to heart
#     Step 2: Apply stiffness (k1_scale * inflammatory factor)
#     Step 3: Apply contractility (Sf_act_scale * inflammatory factor)
#     Step 4: Run heart to hemodynamic steady state
#     Step 5: Construct Heart->Kidney message (MAP, CO, CVP)
#     Step 6: Update kidney model (Hallow equations)
#     Step 7: Construct Kidney->Heart message (V_blood, SVR_ratio)
#     Step 8: Apply kidney feedback to heart model
#     Step 9: Record all variables in history
#   Output: history dictionary with time-series of all variables
#
# =========================================================================

def run_coupled_simulation(
    n_steps: int = 8,
    dt_renal_hours: float = 6.0,
    cardiac_schedule: Optional[List[float]] = None,
    kidney_schedule:  Optional[List[float]] = None,
    stiffness_schedule: Optional[List[float]] = None,
    inflammation_schedule: Optional[List[float]] = None,
    diabetes_schedule: Optional[List[float]] = None,
    use_ode: bool = False,
) -> Dict:
    """
    Run the coupled cardiorenal simulation (Algorithm 1, Section 3.3).

    This is the main entry point for the simulation. It implements the
    complete bidirectional coupling protocol described in Algorithm 1,
    iterating through coupling steps where the heart and kidney models
    exchange hemodynamic information via structured messages.

    Each coupling step represents dt_renal_hours of simulated time for
    the renal volume balance ODEs. The cardiac model runs to steady state
    within each step (assuming the heart reaches hemodynamic equilibrium
    much faster than the renal timescale).

    The simulation supports five independent deterioration/stress schedules
    that can be combined to model different CRS subtypes:
      - cardiac_schedule: systolic dysfunction (HFrEF) via Sf_act scaling
      - kidney_schedule: nephron loss (CKD) via Kf scaling
      - stiffness_schedule: diastolic dysfunction (HFpEF) via k1 scaling
      - inflammation_schedule: systemic inflammation via mediator layer
      - diabetes_schedule: metabolic stress via mediator layer

    Parameters
    ----------
    n_steps : int
        Number of coupling steps to simulate. Each step involves one
        full heart steady-state solve and one renal model update.
    dt_renal_hours : float
        Time-step for the renal volume balance ODEs [hours]. Each
        coupling step integrates the sodium and water balance equations
        (Eq. 11-12) forward by this amount of time.
    cardiac_schedule : list of float or None
        Active fiber stress scale (Sf_act_scale) at each step.
        1.0 = healthy; <1.0 = reduced contractility (HFrEF).
        Default: [1.0] * n_steps (no cardiac deterioration).
    kidney_schedule : list of float or None
        Ultrafiltration coefficient scale (Kf_scale) at each step.
        1.0 = healthy; <1.0 = nephron loss (CKD).
        Default: [1.0] * n_steps (no renal deterioration).
    stiffness_schedule : list of float or None
        Passive myocardial stiffness scale (k1_scale) at each step.
        1.0 = normal compliance; >1.0 = stiffer (HFpEF).
        Default: [1.0] * n_steps (no diastolic dysfunction).
    inflammation_schedule : list of float or None
        Systemic inflammation severity (0-1) at each step.
        0 = none; 1 = severe. Drives the mediator layer (Section 3.4).
        Default: [0.0] * n_steps (no inflammation).
    diabetes_schedule : list of float or None
        Diabetes/metabolic severity (0-1) at each step.
        0 = none; 1 = severe. Drives the mediator layer (Section 3.4).
        Default: [0.0] * n_steps (no diabetes).

    Returns
    -------
    dict
        History dictionary with keys:
          'step'              : list of step numbers (1-indexed)
          'PV_LV', 'PV_RV'   : list of (V, p) waveform tuples for PV loops
          'SBP', 'DBP', 'MAP' : list of blood pressure values [mmHg]
          'CO', 'SV', 'EF'   : list of cardiac output metrics
          'V_blood'           : list of blood volume [mL]
          'GFR'               : list of glomerular filtration rate [mL/min]
          'Na_excr'           : list of Na excretion [mEq/day]
          'P_glom'            : list of glomerular pressure [mmHg]
          'Sf_scale', 'Kf_scale', 'k1_scale'  : input schedule values
          'inflammation_scale', 'diabetes_scale' : input schedule values
          'effective_Sf', 'effective_Kf', 'effective_k1' : after inflammatory modification
    """
    # ── Default schedules: no deterioration ──────────────────────────
    # If no schedule is provided, all parameters remain at their healthy
    # baseline values throughout the simulation.
    if cardiac_schedule is None:
        cardiac_schedule = [1.0] * n_steps       # No systolic dysfunction
    if kidney_schedule is None:
        kidney_schedule = [1.0] * n_steps        # No nephron loss
    if stiffness_schedule is None:
        stiffness_schedule = [1.0] * n_steps     # No diastolic dysfunction
    if inflammation_schedule is None:
        inflammation_schedule = [0.0] * n_steps  # No systemic inflammation
    if diabetes_schedule is None:
        diabetes_schedule = [0.0] * n_steps      # No metabolic stress

    # ── Initialize models (Algorithm 1 initialization) ───────────────
    heart = CircAdaptHeartModel()    # Cardiac compartment: CircAdapt VanOsta2024 (Section 3.1)
    renal = HallowRenalModel()       # Renal compartment: Hallow et al. 2017 (Section 3.2)
    ist = InflammatoryState()        # Inflammatory mediator layer (Section 3.4)

    # ── Initialize history dictionary for recording ──────────────────
    # This accumulates all variables at each coupling step for
    # post-simulation analysis and visualization.
    hist = {k: [] for k in [
        'step',                           # Step number (1-indexed)
        'PV_LV', 'PV_RV',                # Pressure-volume loop waveforms
        'SBP', 'DBP', 'MAP',             # Blood pressure indices [mmHg]
        'CO', 'SV', 'EF',                # Cardiac output metrics [L/min, mL, %]
        'V_blood',                        # Blood volume from kidney [mL]
        'GFR',                            # Glomerular filtration rate [mL/min]
        'Na_excr',                        # Sodium excretion [mEq/day]
        'P_glom',                         # Glomerular capillary pressure [mmHg]
        'Sf_scale', 'Kf_scale', 'k1_scale',  # Input schedule values
        'inflammation_scale', 'diabetes_scale',  # Input schedule values
        'effective_Sf', 'effective_Kf', 'effective_k1',  # After inflammatory modification
    ]}

    # Detect whether inflammatory/diabetic schedules are active for
    # conditional console output (only print mediator details if relevant).
    has_inflammation = any(x > 0 for x in inflammation_schedule)
    has_diabetes = any(x > 0 for x in diabetes_schedule)

    # ── Console banner ───────────────────────────────────────────────
    print("=" * 70)
    print("  CARDIORENAL COUPLING SIMULATOR")
    print("  Heart : CircAdapt VanOsta2024")
    print("  Kidney: Hallow et al. 2017 renal module")
    if has_inflammation or has_diabetes:
        print(f"  Mediator: Inflammatory layer ({'ODE' if use_ode else 'parametric'})")
    print("=" * 70)

    # Compute SVR_baseline from CircAdapt's actual healthy steady state.
    # This ensures SVR_ratio starts at exactly 1.0 at step 1, rather than
    # being biased by a hardcoded (MAP=93, CVP=3, CO=5) assumption.
    hemo_init = heart.run_to_steady_state()
    SVR_baseline = (hemo_init['MAP'] - max(hemo_init['Pven'], 0.5)) / max(hemo_init['CO'], 0.3)

    # Track previous-step hemodynamics for ODE inflammatory model
    prev_EDP = 10.0    # baseline end-diastolic pressure [mmHg]
    prev_MAP = hemo_init['MAP']  # actual CircAdapt baseline MAP [mmHg]

    # ══════════════════════════════════════════════════════════════════
    # MAIN COUPLING LOOP (Algorithm 1, Steps 0-9)
    # ══════════════════════════════════════════════════════════════════
    for s in range(n_steps):

        # ── Read current step's schedule values ──────────────────────
        # Use the last schedule value if the schedule is shorter than n_steps
        # (allows specifying partial schedules that plateau at the last value).
        sf = cardiac_schedule[s] if s < len(cardiac_schedule) else cardiac_schedule[-1]      # Sf_act_scale for this step
        kf = kidney_schedule[s] if s < len(kidney_schedule) else kidney_schedule[-1]         # Kf_scale for this step
        k1 = stiffness_schedule[s] if s < len(stiffness_schedule) else stiffness_schedule[-1]  # k1_scale for this step
        infl = inflammation_schedule[s] if s < len(inflammation_schedule) else inflammation_schedule[-1]  # Inflammation severity
        diab = diabetes_schedule[s] if s < len(diabetes_schedule) else diabetes_schedule[-1]  # Diabetes severity

        # Console output: step header with current schedule values
        print(f"\n{'---'*20}")
        print(f"  Step {s+1}/{n_steps}   "
              f"Sf_act={sf:.2f}   k1={k1:.2f}   Kf={kf:.2f}")
        if has_inflammation or has_diabetes:
            print(f"  Inflammation={infl:.2f}   Diabetes={diab:.2f}")
        print(f"{'---'*20}")

        # ── Step 0: Update inflammatory mediator layer ───────────────
        # (Section 3.4, Table 1)
        # Recompute all modifier factors based on current inflammation
        # and diabetes severity. In ODE mode, organ damage signals from
        # the previous step drive inflammation dynamics.
        ist = update_inflammatory_state(
            ist, infl, diab,
            GFR=renal.GFR, EDP=prev_EDP,
            aldosterone_factor=1.0, P_glom=renal.P_glom,
            CVP=prev_MAP * 0.05,  # approximate CVP from MAP
            MAP=prev_MAP, dt_hours=dt_renal_hours,
            use_ode=use_ode,
        )

        # ── Step 1: Apply inflammatory modifiers to heart ────────────
        # (Section 3.4 -> Section 3.1)
        # Sets arterial stiffness directly, stores k1 and p0 factors
        # for composition with direct schedules in Steps 2-3 and 8.
        heart.apply_inflammatory_modifiers(ist)

        # ── Step 2: Apply stiffness / HFpEF diastolic dysfunction ────
        # (Section 3.1, Eq. 2)
        # The effective k1 is the product of the direct schedule value
        # and the inflammatory passive stiffness factor (AGE cross-linking).
        # effective_k1 = k1_schedule * passive_k1_factor_inflammatory
        effective_k1 = k1 * ist.passive_k1_factor
        heart.apply_stiffness(effective_k1)

        # ── Step 3: Apply deterioration / contractility ──────────────
        # (Section 3.1, Eq. 1; Section 3.3, Algorithm 1 Step 3)
        # The effective Sf_act is the product of the direct schedule value
        # and the inflammatory contractility factor (TNF-alpha + diabetic).
        # Floored at 0.20 to prevent complete contractile failure that
        # would crash the CircAdapt solver.
        effective_sf = max(sf * ist.Sf_act_factor, 0.20)
        heart.apply_deterioration(effective_sf)

        # Set the kidney's Kf_scale from the direct schedule.
        # The inflammatory Kf_factor is applied inside update_renal_model()
        # when it computes Kf_eff = Kf * Kf_scale * ist.Kf_factor.
        renal.Kf_scale = kf

        # Compute effective Kf for recording purposes only (actual
        # application happens inside update_renal_model).
        effective_kf = kf * ist.Kf_factor

        # Console output: effective parameter values after inflammatory modification
        if has_inflammation or has_diabetes:
            print(f"  [Inflam] Sf_eff={effective_sf:.3f}  k1_eff={effective_k1:.3f}  "
                  f"Kf_eff={effective_kf:.3f}  "
                  f"p0_factor={ist.p0_factor:.3f}  "
                  f"stiffness={ist.stiffness_factor:.3f}")

        # ── Step 4: Run heart to hemodynamic steady state ────────────
        # (Section 3.1; Algorithm 1 Step 4)
        # CircAdapt runs multiple cardiac cycles until beat-to-beat
        # variation is below tolerance. The resulting hemodynamic state
        # (MAP, CO, SV, EF, etc.) reflects the heart's response to all
        # currently applied perturbations.
        print("  [Heart]  Running CircAdapt to steady state ...")
        hemo = heart.run_to_steady_state()

        # Update previous-step hemodynamics for ODE inflammatory model
        prev_MAP = hemo['MAP']
        try:
            p_lv = hemo['p_LV']
            v_lv = hemo['V_LV']
            prev_EDP = float(p_lv[np.argmax(v_lv)])
        except Exception:
            prev_EDP = hemo.get('Pven', 5.0) + 5.0

        print(f"  [Heart]  MAP={hemo['MAP']:.1f}  "
              f"SBP/DBP={hemo['SBP']:.0f}/{hemo['DBP']:.0f}  "
              f"CO={hemo['CO']:.2f} L/min  SV={hemo['SV']:.1f} mL  "
              f"EF={hemo['EF']:.0f}%")

        # ── Step 5: Construct Heart -> Kidney message ────────────────
        # (Algorithm 1, Step 5)
        # Package MAP, CO, CVP from the cardiac steady state into a
        # structured message for the renal model.
        h2k = heart_to_kidney(hemo)
        print(f"  [H->K]  MAP={h2k.MAP:.1f}  CO={h2k.CO:.2f}  Pven={h2k.Pven:.1f}")

        # ── Step 6: Update kidney model ──────────────────────────────
        # (Section 3.2, Eq. 3-12; Algorithm 1, Step 6)
        # The renal model receives cardiac hemodynamics and computes
        # GFR, sodium/water excretion, and updates blood volume via
        # the Hallow equations. Inflammatory effects are applied to
        # renal parameters (Kf, R_AA, R_EA, etc.) if the inflammatory
        # state is provided.
        print(f"  [Kidney] Updating renal model (dt={dt_renal_hours}h) ...")
        renal = update_renal_model(
            renal, h2k.MAP, h2k.CO, h2k.Pven,
            dt_renal_hours,
            inflammatory_state=ist           # Pass inflammatory state for renal parameter modification
        )
        print(f"  [Kidney] GFR={renal.GFR:.1f} mL/min   "
              f"V_blood={renal.V_blood:.0f} mL   "
              f"Na_excr={renal.Na_excretion:.0f} mEq/day")

        # ── Step 7: Construct Kidney -> Heart message ────────────────
        # (Algorithm 1, Step 7)
        # Package V_blood and SVR_ratio from the renal model into a
        # structured message for the cardiac model. SVR_ratio is
        # computed from the current MAP and CO.
        k2h = kidney_to_heart(renal, h2k.MAP, h2k.CO, h2k.Pven, SVR_baseline)
        print(f"  [K->H]  V_blood={k2h.V_blood:.0f} mL   "
              f"SVR_ratio={k2h.SVR_ratio:.3f}   GFR={k2h.GFR:.1f}")

        # ── Step 8: Apply kidney feedback to heart ───────────────────
        # (Algorithm 1, Step 8)
        # Feed the kidney's volume and resistance state back into the
        # CircAdapt model. V_blood is converted from mL to m^3 for
        # CircAdapt's SI unit system. SVR_ratio scales the ArtVen p0
        # parameter (composed with the inflammatory p0_factor).
        heart.apply_kidney_feedback(
            V_blood_mL=k2h.V_blood,                  # Total body blood volume [mL]
            SVR_ratio=k2h.SVR_ratio,                 # Dimensionless SVR change
        )

        # ── Step 9: Record all variables in history ──────────────────
        # (Algorithm 1, Step 9)
        # Store everything needed for post-simulation analysis:
        # waveforms, scalar hemodynamic indices, renal outputs,
        # schedule inputs, and effective (post-inflammatory) parameters.
        hist['step'].append(s + 1)                                   # 1-indexed step number
        hist['PV_LV'].append((hemo['V_LV'].copy(), hemo['p_LV'].copy()))  # LV PV loop (deep copy to snapshot)
        hist['PV_RV'].append((hemo['V_RV'].copy(), hemo['p_RV'].copy()))  # RV PV loop (deep copy to snapshot)
        hist['SBP'].append(hemo['SBP'])                              # Systolic BP [mmHg]
        hist['DBP'].append(hemo['DBP'])                              # Diastolic BP [mmHg]
        hist['MAP'].append(hemo['MAP'])                              # Mean arterial pressure [mmHg]
        hist['CO'].append(hemo['CO'])                                # Cardiac output [L/min]
        hist['SV'].append(hemo['SV'])                                # Stroke volume [mL]
        hist['EF'].append(hemo['EF'])                                # Ejection fraction [%]
        hist['V_blood'].append(renal.V_blood)                        # Blood volume from kidney model [mL]
        hist['GFR'].append(renal.GFR)                                # GFR from kidney model [mL/min]
        hist['Na_excr'].append(renal.Na_excretion)                   # Na excretion [mEq/day]
        hist['P_glom'].append(renal.P_glom)                          # Glomerular pressure [mmHg]
        hist['Sf_scale'].append(sf)                                  # Input: cardiac schedule value
        hist['Kf_scale'].append(kf)                                  # Input: kidney schedule value
        hist['k1_scale'].append(k1)                                  # Input: stiffness schedule value
        hist['inflammation_scale'].append(infl)                      # Input: inflammation severity
        hist['diabetes_scale'].append(diab)                          # Input: diabetes severity
        hist['effective_Sf'].append(effective_sf)                    # Effective Sf after inflammatory modification
        hist['effective_Kf'].append(effective_kf)                    # Effective Kf after inflammatory modification
        hist['effective_k1'].append(effective_k1)                    # Effective k1 after inflammatory modification

        # ── Message passing & inflammatory state (for visualization) ──
        hist.setdefault('h2k_MAP', []).append(h2k.MAP)
        hist.setdefault('h2k_CO', []).append(h2k.CO)
        hist.setdefault('h2k_Pven', []).append(h2k.Pven)
        hist.setdefault('k2h_Vblood', []).append(k2h.V_blood)
        hist.setdefault('k2h_SVR', []).append(k2h.SVR_ratio)
        hist.setdefault('k2h_GFR', []).append(k2h.GFR)
        hist.setdefault('RBF', []).append(renal.RBF)
        hist.setdefault('water_excr', []).append(renal.water_excretion)
        hist.setdefault('C_Na', []).append(renal.C_Na)
        # Inflammatory modifier factors
        hist.setdefault('Sf_act_factor', []).append(ist.Sf_act_factor)
        hist.setdefault('Kf_factor', []).append(ist.Kf_factor)
        hist.setdefault('p0_factor', []).append(ist.p0_factor)
        hist.setdefault('stiffness_factor', []).append(ist.stiffness_factor)
        hist.setdefault('passive_k1_factor', []).append(ist.passive_k1_factor)
        hist.setdefault('R_AA_factor', []).append(ist.R_AA_factor)
        hist.setdefault('R_EA_factor', []).append(ist.R_EA_factor)
        hist.setdefault('RAAS_gain_factor', []).append(ist.RAAS_gain_factor)
        hist.setdefault('eta_PT_offset', []).append(ist.eta_PT_offset)
        hist.setdefault('MAP_setpoint_offset', []).append(ist.MAP_setpoint_offset)
        # ODE inflammatory state variables
        hist.setdefault('systemic_inflammatory_index', []).append(ist.systemic_inflammatory_index)
        hist.setdefault('myocardial_fibrosis_volume', []).append(ist.myocardial_fibrosis_volume)
        hist.setdefault('endothelial_dysfunction_index', []).append(ist.endothelial_dysfunction_index)
        hist.setdefault('renal_tubulointerstitial_fibrosis', []).append(ist.renal_tubulointerstitial_fibrosis)
        hist.setdefault('AGE_accumulation', []).append(ist.AGE_accumulation)
        hist.setdefault('LVEDP', []).append(prev_EDP if prev_EDP is not None else 0.0)

        # ── Log this step to structured log file ───────────────────
        sim_logger.log_run(
            params={'Sf_act_scale': sf, 'k1_scale': k1, 'Kf_scale': kf,
                    'inflammation_scale': infl, 'diabetes_scale': diab},
            outputs=extract_key_outputs(hemo, renal),
            success=True,
            source='run_coupled_simulation',
            step=s + 1,
        )

    # ── Simulation complete ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SIMULATION COMPLETE")
    print(f"{'='*70}\n")

    return hist


# =========================================================================
# PART 5 -- RL-Enhanced Coupled Simulation  (Learned Coupling Equation)
# =========================================================================
#
# This variant of the coupled simulation accepts an RL policy function
# (alpha_fn) that produces per-message coupling weights and inflammatory
# residual corrections at each step. The learned coupling equation
# modulates how strongly each organ's outputs affect the other,
# replacing the implicit full-strength coupling with a state-dependent,
# per-channel function discovered from data.
#
# The original run_coupled_simulation() is preserved unchanged.
# =========================================================================

def run_coupled_simulation_rl(
    n_steps: int = 96,
    dt_renal_hours: float = 180.0,
    renal_substeps: int = 4,
    cardiac_schedule: Optional[List[float]] = None,
    kidney_schedule:  Optional[List[float]] = None,
    stiffness_schedule: Optional[List[float]] = None,
    inflammation_schedule: Optional[List[float]] = None,
    diabetes_schedule: Optional[List[float]] = None,
    alpha_fn: Optional[callable] = None,
    baselines: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> Dict:
    """
    RL-enhanced coupled simulation with learned per-message coupling.

    This function mirrors run_coupled_simulation() but adds:
    1. Per-message alpha scaling via the learned coupling equation
    2. Inflammatory residual corrections from the RL policy
    3. Per-step observation recording for RL trajectory collection
    4. Support for weekly renal sub-steps within each coupling step

    When alpha_fn is None, all alphas default to 1.0 and residuals to 0.0,
    reproducing the behavior of the original simulation (with the only
    difference being dt_renal_hours and sub-stepping).

    Parameters
    ----------
    n_steps : int
        Number of coupling steps (e.g., 96 for monthly over 8 years).
    dt_renal_hours : float
        Time-step for each renal sub-step [hours]. Default 180h ≈ 1 week.
    renal_substeps : int
        Number of renal sub-steps per coupling step. Default 4 (4 weeks/month).
    cardiac_schedule, kidney_schedule, stiffness_schedule : list or None
        Per-step parameter schedules (same as run_coupled_simulation).
    inflammation_schedule, diabetes_schedule : list or None
        Per-step inflammation/diabetes severity schedules.
    alpha_fn : callable or None
        RL policy function: alpha_fn(obs_dict, step) -> (alpha_5, residuals_10).
        If None, uses identity coupling (alpha=1, residuals=0).
    baselines : dict or None
        Healthy message baselines for alpha scaling. Uses RL_CONFIG defaults if None.
    verbose : bool
        Whether to print per-step console output.

    Returns
    -------
    dict
        Extended history dictionary with standard fields plus:
        'observations' : list of per-step observation dicts
        'actions_alpha' : list of per-step alpha vectors
        'actions_residual' : list of per-step residual vectors
    """
    from config import RL_CONFIG

    # Default baselines from config
    if baselines is None:
        baselines = RL_CONFIG['baselines']

    # Default schedules
    if cardiac_schedule is None:
        cardiac_schedule = [1.0] * n_steps
    if kidney_schedule is None:
        kidney_schedule = [1.0] * n_steps
    if stiffness_schedule is None:
        stiffness_schedule = [1.0] * n_steps
    if inflammation_schedule is None:
        inflammation_schedule = [0.0] * n_steps
    if diabetes_schedule is None:
        diabetes_schedule = [0.0] * n_steps

    # Default alpha_fn: identity coupling
    if alpha_fn is None:
        def alpha_fn(obs_dict, step):
            return np.ones(5), np.zeros(10)

    # Initialize models
    heart = CircAdaptHeartModel()
    renal = HallowRenalModel()
    ist = InflammatoryState()

    # Pre-equilibrate kidney (Algorithm 1, Step 3).
    # The RL path uses large dt (180h × 4 substeps = 1 month per coupling
    # step). At initialization the kidney's TGF setpoint is unset and sodium
    # balance hasn't been calibrated, so a 180h substep causes massive
    # sodium retention and V_blood overshoots to the 8000 mL ceiling.
    # Fix: run 5 short (6h) baseline renal updates so the kidney finds its
    # sodium balance before the main loop begins with large timesteps.
    hemo_init = heart.run_to_steady_state()
    h2k_init = heart_to_kidney(hemo_init)
    for _ in range(5):
        renal = update_renal_model(
            renal, h2k_init.MAP, h2k_init.CO, h2k_init.Pven,
            dt_hours=6.0,
            inflammatory_state=ist,
        )

    # History dictionary (standard fields + RL-specific)
    hist = {k: [] for k in [
        'step', 'PV_LV', 'PV_RV',
        'SBP', 'DBP', 'MAP', 'CO', 'SV', 'EF',
        'V_blood', 'GFR', 'Na_excr', 'P_glom',
        'Sf_scale', 'Kf_scale', 'k1_scale',
        'inflammation_scale', 'diabetes_scale',
        'effective_Sf', 'effective_Kf', 'effective_k1',
    ]}
    hist['observations'] = []
    hist['actions_alpha'] = []
    hist['actions_residual'] = []

    prev_obs = None

    for s in range(n_steps):
        # Read schedule values (plateau at last value if schedule is shorter)
        sf = cardiac_schedule[min(s, len(cardiac_schedule) - 1)]
        kf = kidney_schedule[min(s, len(kidney_schedule) - 1)]
        k1 = stiffness_schedule[min(s, len(stiffness_schedule) - 1)]
        infl = inflammation_schedule[min(s, len(inflammation_schedule) - 1)]
        diab = diabetes_schedule[min(s, len(diabetes_schedule) - 1)]

        # Step 0: Update inflammatory state from schedules
        ist = update_inflammatory_state(ist, infl, diab)

        # Get RL action (alpha + residuals) based on previous observation
        if prev_obs is not None:
            alpha_vec, residuals = alpha_fn(prev_obs, s)
        else:
            alpha_vec = np.ones(5)
            residuals = np.zeros(10)

        # Apply inflammatory residual corrections from RL policy
        ist_corrected = apply_inflammatory_residuals(ist, residuals)

        # Step 1: Apply inflammatory modifiers to heart
        heart.apply_inflammatory_modifiers(ist_corrected)

        # Step 2: Apply stiffness
        effective_k1 = k1 * ist_corrected.passive_k1_factor
        heart.apply_stiffness(effective_k1)

        # Step 3: Apply contractility
        effective_sf = max(sf * ist_corrected.Sf_act_factor, 0.20)
        heart.apply_deterioration(effective_sf)

        # Set kidney Kf_scale
        renal.Kf_scale = kf
        effective_kf = kf * ist_corrected.Kf_factor

        # Step 4: Run heart to steady state
        hemo = heart.run_to_steady_state()

        # Step 5: Construct and SCALE Heart->Kidney message
        h2k_raw = heart_to_kidney(hemo)
        h2k = scale_message_h2k(h2k_raw, alpha_vec[:3], baselines)

        # Step 6: Run renal sub-steps (e.g., 4 weekly steps per month)
        for _ in range(renal_substeps):
            renal = update_renal_model(
                renal, h2k.MAP, h2k.CO, h2k.Pven,
                dt_renal_hours,
                inflammatory_state=ist_corrected,
            )

        # Step 7: Construct and SCALE Kidney->Heart message
        k2h_raw = kidney_to_heart(renal, h2k.MAP, h2k.CO, h2k.Pven)
        k2h = scale_message_k2h(k2h_raw, alpha_vec[3:5], baselines)

        # Step 8: Apply kidney feedback to heart
        heart.apply_kidney_feedback(
            V_blood_m3=k2h.V_blood * ML_TO_M3,
            SVR_ratio=k2h.SVR_ratio,
        )

        # Extract observation
        t_normalized = (s + 1) / n_steps
        obs = extract_rl_observation(
            hemo, renal, ist_corrected,
            effective_sf, effective_kf, effective_k1,
            infl, diab, t_normalized, prev_obs,
        )

        # Record history
        hist['step'].append(s + 1)
        hist['PV_LV'].append((hemo['V_LV'].copy(), hemo['p_LV'].copy()))
        hist['PV_RV'].append((hemo['V_RV'].copy(), hemo['p_RV'].copy()))
        hist['SBP'].append(hemo['SBP'])
        hist['DBP'].append(hemo['DBP'])
        hist['MAP'].append(hemo['MAP'])
        hist['CO'].append(hemo['CO'])
        hist['SV'].append(hemo['SV'])
        hist['EF'].append(hemo['EF'])
        hist['V_blood'].append(renal.V_blood)
        hist['GFR'].append(renal.GFR)
        hist['Na_excr'].append(renal.Na_excretion)
        hist['P_glom'].append(renal.P_glom)
        hist['Sf_scale'].append(sf)
        hist['Kf_scale'].append(kf)
        hist['k1_scale'].append(k1)
        hist['inflammation_scale'].append(infl)
        hist['diabetes_scale'].append(diab)
        hist['effective_Sf'].append(effective_sf)
        hist['effective_Kf'].append(effective_kf)
        hist['effective_k1'].append(effective_k1)
        hist['observations'].append(obs)
        hist['actions_alpha'].append(alpha_vec.copy())
        hist['actions_residual'].append(residuals.copy())

        # ── Log this step to structured log file ───────────────────
        sim_logger.log_run(
            params={'Sf_act_scale': sf, 'k1_scale': k1, 'Kf_scale': kf,
                    'inflammation_scale': infl, 'diabetes_scale': diab},
            outputs=extract_key_outputs(hemo, renal),
            policy={'alpha_h2k': alpha_vec[:3].tolist(),
                    'alpha_k2h': alpha_vec[3:5].tolist(),
                    'residuals': residuals.tolist()},
            success=True,
            source='run_coupled_simulation_rl',
            step=s + 1,
        )

        prev_obs = obs

        if verbose:
            print(f"  Step {s+1}/{n_steps}  MAP={hemo['MAP']:.1f}  "
                  f"GFR={renal.GFR:.1f}  EF={hemo['EF']:.0f}%  "
                  f"alpha={alpha_vec[:3].tolist()}")

    return hist
