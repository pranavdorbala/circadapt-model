"""
Simulation Logging
==================
Centralized logging for the cardiorenal simulator. Provides a structured
simulation log writer (JSON lines) for recording parameter sets, 8 key
outputs, and success/failure labels at each coupling step.
"""

import json
import time
import os
import warnings
from typing import Dict, Optional, Any


# ─── Key output variables to log ─────────────────────────────────────────────
# We log 8 outputs chosen to cover all major physiological subsystems
# with minimal redundancy. Each variable is directly available in the
# coupling loop without calling emission functions.
#
# NOT included:
#   E_over_e_prime_sept (E/e') — the primary diastolic dysfunction marker
#     and the key diagnostic for HFpEF. Not logged because computing it
#     requires running the full emission pipeline (emit_mitral_inflow_doppler
#     + emit_tissue_doppler), which processes raw CircAdapt pressure/volume
#     waveforms to extract peak velocities at specific points in the cardiac
#     cycle. This is too slow to run at every coupling step. E/e' cannot be
#     reconstructed from the 8 variables below — it depends on waveform shape.
#
#   NTproBNP_pg_mL — cardiac wall stress biomarker. Same issue: computed by
#     the emission pipeline from waveform-derived wall stress, not from
#     scalar hemodynamic outputs.

KEY_OUTPUTS = {
    # Cardiac systolic function — primary diagnostic for HFrEF.
    # If EF drops, the heart is failing as a pump.
    'EF': 'LV ejection fraction [%] — primary systolic function marker',

    # Hemodynamic perfusion pressure — the main input to the kidney.
    # MAP is the variable that couples heart → kidney in Algorithm 1.
    'MAP': 'Mean arterial pressure [mmHg] — coupling variable H→K',

    # Cardiac output — total forward flow to organs.
    # Low CO causes renal hypoperfusion (cardiorenal syndrome type 1).
    'CO': 'Cardiac output [L/min] — organ perfusion driver',

    # Blood volume — the main feedback from kidney → heart.
    # Volume overload raises preload and drives congestion.
    'V_blood': 'Blood volume [mL] — coupling variable K→H',

    # Kidney filtration — primary renal outcome.
    # GFR is the single number that defines CKD stage.
    'GFR': 'Glomerular filtration rate [mL/min] — primary renal function marker',

    # Sodium excretion — renal volume regulation effectiveness.
    # Mismatch between intake and excretion drives volume overload.
    'Na_excr': 'Sodium excretion [mEq/day] — renal volume regulation',

    # Glomerular pressure — early marker of kidney damage.
    # Elevated P_glom causes hyperfiltration injury before GFR drops.
    'P_glom': 'Glomerular capillary pressure [mmHg] — hyperfiltration injury marker',

    # Stroke volume — preload sensitivity indicator.
    # Distinguishes volume-responsive vs volume-unresponsive states.
    'SV': 'Stroke volume [mL] — preload sensitivity indicator',
}


def extract_key_outputs(hemo, renal):
    """Build the 8-key outputs dict from hemodynamics + renal state.

    Parameters
    ----------
    hemo : dict  — hemodynamics dict with keys EF, MAP, CO, SV
    renal : HallowRenalModel or dict — renal state (attribute or key access)
    """
    def _get(obj, key):
        return obj[key] if isinstance(obj, dict) else getattr(obj, key)
    return {
        'EF': hemo['EF'], 'MAP': hemo['MAP'], 'CO': hemo['CO'], 'SV': hemo['SV'],
        'GFR': _get(renal, 'GFR'), 'V_blood': _get(renal, 'V_blood'),
        'Na_excr': _get(renal, 'Na_excretion'), 'P_glom': _get(renal, 'P_glom'),
    }


# ─── Structured simulation log (JSON lines) ──────────────────────────────────

class SimulationLogger:
    """
    Appends one JSON line per simulation run to logs/simulation_runs.jsonl.

    Each entry records:
      - timestamp (ISO 8601)
      - params (the 8 disease parameters used)
      - demographics (age, sex, BSA, height)
      - outputs (8 key variables — see KEY_OUTPUTS above)
      - success (bool)
      - error (string, if failed)
      - duration_s (wall-clock seconds)
      - source (which function produced this entry)
    """

    def __init__(self, log_dir: str = 'logs', filename: str = 'simulation_runs.jsonl'):
        self.log_dir = os.path.join(os.path.dirname(__file__), log_dir)
        self.log_path = os.path.join(self.log_dir, filename)
        self._ensured = False
        self._warned_write_failure = False

    def _ensure_dir(self):
        if not self._ensured:
            os.makedirs(self.log_dir, exist_ok=True)
            self._ensured = True

    def log_run(
        self,
        params: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None,
        duration_s: Optional[float] = None,
        source: str = 'unknown',
        demographics: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None,
        policy: Optional[Dict[str, Any]] = None,
    ):
        """Append a single simulation run record as one JSON line.

        Parameters
        ----------
        params : dict       — the 8 disease parameters used
        outputs : dict      — simulation outputs (filtered to KEY_OUTPUTS)
        success : bool      — whether the step completed without error
        error : str         — error message if success=False
        duration_s : float  — wall-clock time for this step
        source : str        — which function produced this entry
        demographics : dict — patient demographics (age, sex, BSA, height)
        step : int          — coupling step number (1-indexed)
        policy : dict       — RL policy actions, logged only for RL runs.
                              Expected keys: alpha_h2k (list[3]), alpha_k2h (list[2]),
                              residuals (list[10]).
        """
        self._ensure_dir()

        # Extract only the 8 key outputs
        key_outputs = None
        if outputs is not None:
            key_outputs = {}
            for k in KEY_OUTPUTS:
                if k in outputs:
                    val = outputs[k]
                    key_outputs[k] = round(float(val), 2) if val is not None else None

        entry = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'source': source,
            'success': success,
            'params': {k: round(float(v), 4) if isinstance(v, (int, float)) else v
                       for k, v in params.items()},
        }
        if step is not None:
            entry['step'] = step
        if demographics is not None:
            entry['demographics'] = demographics
        if key_outputs is not None:
            entry['outputs'] = key_outputs
        if policy is not None:
            entry['policy'] = policy
        if error is not None:
            entry['error'] = str(error)
        if duration_s is not None:
            entry['duration_s'] = round(duration_s, 3)

        # NOTE: Multiple processes may write to this file concurrently
        # (e.g., multiprocessing.Pool in synthetic_cohort.py). Each JSON line
        # is well under PIPE_BUF (4096 bytes), so POSIX O_APPEND writes are
        # atomic and lines will not interleave.
        try:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except OSError as e:
            if not self._warned_write_failure:
                warnings.warn(f"SimulationLogger: failed to write to {self.log_path}: {e}")
                self._warned_write_failure = True


# Module-level default instance
sim_logger = SimulationLogger()
