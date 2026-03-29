# Simulator Change Log

All changes to `cardiorenal_coupling.py`, in chronological order by commit then by session.

---

## Commit `1a04768` — fixed errors mentioned in issues (2026-03-24)

### Tubular reabsorption fractions corrected
The original values were raw segmental fractions. Converted to "fraction of delivered" for sequential multiplication:
- `eta_LoH: 0.25 → 0.758` (25% of filtered = 75.8% of delivered to LoH)
- `eta_DT: 0.05 → 0.626` (5% of filtered = 62.6% of delivered to DT)
- `eta_CD0: 0.024 → 0.803` (2.4% of filtered = 80.3% of delivered to CD)
- `eta_PT: 0.67` (unchanged, still fraction of filtered)

### Na_total initial value changed
- `Na_total: 700.0 → 2100.0 mEq` (correctly represents total body exchangeable Na = 140 mEq/L × 15 L ECF)

### Interstitial compartment added then removed in same era
Commit `1a04768` added `V_interstitial`, `IF_sodium`, `IF_C_Na`, `Q_water`, `Q_Na` (two-compartment blood-interstitial exchange). Commit `7cfee82` removed the interstitial compartment as overcomplicating the volume balance and reverted to single-compartment.

### Vasopressin PI controller added
New fields on `HallowRenalModel`:
- `C_Na_setpoint: 140.0 mEq/L`
- `VP_outer_gain: 0.05`
- `VP_Kp: 2.0`
- `VP_Ki: 0.005`
- `VP_saturation_Km: 0.15`
- `VP_integral_error: 0.0`
- `VP_normalized: 1.0`

Water excretion switched from fixed `frac_water_reabs=0.99` to ADH-modulated:
- `frac_min=0.90`, `frac_max=0.998`, modulated by `ADH_perm = VP / (Km + VP)`

### Pressure-natriuresis implemented
- Above setpoint: `pn = 1 + 0.03 × (MAP - MAP_sp_eff)`
- Below setpoint: `pn = max(0.3, 1 + 0.015 × (MAP - MAP_sp_eff))`

### Volume ODE changed
- Old: `dV_blood = (W_in - water_excr + Q_water×dC_Na) × dt` (two-compartment)
- New: `dV_blood = (W_in - water_excr) × dt × 0.33` (single-compartment, 1/3 of ECF change enters blood)
- `C_Na` now computed from `Na_total / (V_ECF × 1e-3)` where `V_ECF = V_blood / 0.33`

### InflammatoryState residual clamping removed
Agent loop residuals: removed per-factor clamps (e.g. `max(..., 0.3)`) from `apply_inflammatory_residuals`. Values now passed through unclamped.

---

## Commit `7cfee82` — added more issues, fixed excess water issue (2026-03-24)

### Interstitial compartment removed
Reverted two-compartment blood-interstitial model. Removed `V_interstitial`, `IF_sodium`, `IF_C_Na`, `Q_water`, `Q_Na` fields. Single-compartment volume balance retained.

### C_Na denominator fixed
- Old: `C_Na = Na_total / (V_blood × 1e-3)` (used blood volume as denominator)
- New: `C_Na = Na_total / (V_ECF × 1e-3)` where `V_ECF = V_blood / 0.33` (correctly uses ECF volume)

---

## Commit `3fe241a` — fixed the simulator, produces reliable results (2026-03-25)

### Vascular resistance recalibration (round 1)
Tuned to produce GFR=120 at MAP=93 mmHg:
- `R_preAA: 12.0 → 10.0`
- `R_AA0: 26.0 → 25.0`
- `R_EA0: 43.0 → 52.0`

### Na_intake recalibrated
- `Na_intake: 150.0 → 137.0 mEq/day`

### MAP_setpoint corrected
- `MAP_setpoint: 93.0 → 86.4 mmHg` (matched to CircAdapt's actual baseline MAP)

### SVR_baseline hardcoded
- `SVR_baseline = (93 - 3) / 5 = 18 mmHg·min/L` (hardcoded reference, later found to be wrong)

### TGF: switched from Picard to relaxed fixed-point
- Old: simple proportional update `R_AA = R_AA0 × (1 + TGF_gain × err)`
- New: 30-iteration relaxed iteration, `R_AA = 0.8×R_AA_old + 0.2×R_AA_new`, clamped to [0.5×, 3.0×] baseline

### History tracking expanded
Added to history dict: `k2h_Vblood`, `water_excr`, `Kf_factor`, `passive_k1_factor`, `R_AA_factor`, `R_EA_factor`, `RAAS_gain_factor`, `eta_PT_offset`, `MAP_setpoint_offset`

### research_figures.py added
First version of 8-year simulation figure generation script.

---

## Commit `4b1e157` — full ODE implemented (2026-03-25)

### Inflammatory ODE state variables activated
Previously commented out. Now live:
- `myocardial_fibrosis_volume: 0.0`
- `endothelial_dysfunction_index: 0.0`
- `renal_tubulointerstitial_fibrosis: 0.0`
- `AGE_accumulation: 0.0`

### ODE inputs wired from organ outputs
`update_inflammatory_state` now receives: `GFR`, `EDP`, `aldosterone_factor`, `P_glom`, `CVP`, `MAP`, `dt_hours` — all fed from actual simulation outputs each step.

### ODE rate constants (first live values)
- `k_uremic = 0.01`
- `k_congestion = 0.005`
- `k_AGE = 0.02`
- `k_aldo = 0.008`
- `k_clearance = 0.05`
- `k_AGE_formation = 0.001`
- `k_AGE_turnover = 0.0005`
- `k_fibrosis_inflam = 0.003`
- `k_fibrosis_mech = 0.002`
- `k_fibrosis_turnover = 0.001`
- `k_endoth_inflam = 0.01`
- `k_endoth_shear = 0.002`
- `k_endoth_recovery = 0.008`
- `k_renal_inflam = 0.004`
- `k_renal_pressure = 0.002`
- `k_renal_congestion = 0.003`

### Inflammatory modifier equations (first live values)
- `Sf_act_factor = (1 - 0.25×infl) × (1 - 0.15×fibrosis)`
- `p0_factor = 1 + 0.15×endoth + 0.10×AGE`
- `stiffness_factor = 1 + 0.30×endoth + 0.50×AGE`
- `passive_k1_factor = 1 + 0.40×fibrosis + 0.30×AGE`
- `Kf_factor = max((1 - 0.20×infl) × (1 - renal_fib), 0.05)`
- `R_AA_factor = 1 + 0.20×endoth`
- `R_EA_factor = 1 + 0.25×AGE`
- `RAAS_gain_factor = 1 + 0.30×infl`
- `eta_PT_offset = 0.04×infl + 0.06×AGE`
- `MAP_setpoint_offset = 5.0×infl + 3.0×endoth`

---

## Session changes (2026-03-29, uncommitted)

### 1. Volume mapping fix
**Problem:** `apply_kidney_feedback` passed `V_blood` in m³ directly to CircAdapt PFC `target_volume`, treating total blood volume as stressed volume — ~6× too large.

**Fix:** Map via Guyton stressed fraction (0.33):
```python
target_volume = baseline_circ_vol + (V_blood_mL - 5000) × 0.33 × 1e-6
```
Also: kept PFC volume control ON during settling beats (previously released, letting CircAdapt revert to own volume).

### 2. SVR feedback loop removed
**Problem:** SVR_ratio computed from MAP/CO created a positive feedback loop (volume overload → high CO → low SVR → CircAdapt drops resistance → MAP drops → Na retention).

**Fix:** `SVR_ratio` fixed at 1.0. Volume handled entirely by PFC. Inflammatory resistance via `p0_factor`.

### 3. SVR_baseline computed from CircAdapt actual state
**Problem:** Hardcoded `SVR_baseline = (93-3)/5 = 18` while CircAdapt baseline MAP = 86.4 mmHg, biasing SVR_ratio to ~0.95 from step 1.

**Fix:** Compute from actual first `run_to_steady_state()` call.

### 4. Vascular resistance recalibration (round 2)
Recalibrated for CircAdapt's actual MAP=86.4 mmHg (vs commit 3fe221a which targeted MAP=93):
- `R_preAA: 10.0 → 6.5` (reduces upstream pressure drop, raises P_gc at lower MAP)
- `R_AA0: 25.0 → 16.4` *(temporary — paired with dynamic TGF; reverted in round 3)*
- `R_EA0: 52.0 → 49.0`

### 5. Na_intake recalibrated
- `Na_intake: 137 → 142 mEq/day` to balance Na excretion at new baseline.

### 6. ODE rate constants rescaled (×10–100 reduction)
Previous values accumulated fibrosis in days. New values:
- `k_uremic: 0.01 → 0.0004`
- `k_congestion: 0.005 → 0.0002`
- `k_AGE: 0.02 → 0.0008`
- `k_aldo: 0.008 → 0.0003`
- `k_clearance: 0.05 → 0.002`
- `k_AGE_formation: 0.001 → 0.00004`
- `k_AGE_turnover: 0.0005 → 0.00002`
- `k_fibrosis_inflam: 0.003 → 0.00005`
- `k_fibrosis_mech: 0.002 → 0.00001`
- `k_fibrosis_turnover: 0.001 → 0.00004`
- `k_endoth_inflam: 0.01 → 0.0004`
- `k_endoth_shear: 0.002 → 0.00008`
- `k_endoth_recovery: 0.008 → 0.0003`
- `k_renal_inflam: 0.004 → 0.00002`
- `k_renal_pressure: 0.002 → 0.000005`
- `k_renal_congestion: 0.003 → 0.00001`

### 7. Absolute pressure natriuresis term added *(trajectory-fitting — REVERTED)*
**Problem:** MAP_setpoint_offset (RAAS resetting) shifts natriuresis setpoint up to 8 mmHg rightward, blunting natriuresis so V_blood rails into 8000 mL cap at MAP=104 mmHg.

**Fix (added then reverted):**
```python
pn += 0.04 × max(0.0, MAP - 100.0)
```
**Why reverted:** This was added to prevent V_blood from hitting the cap — a trajectory-fitting patch, not a mechanistic need. Reverted so the simulator produces whatever the physics dictates.

### 8. TGF: replaced Picard with dynamic R_AA state *(trajectory-fitting — REVERTED)*
**Problem:** 40-iteration bisection solved TGF to exact equilibrium every 6h. GFR was fully compensated every step — chronic hyperfiltration never truly "progressed."

**Fix (added then reverted):** Dynamic R_AA state + TGF setpoint adaptation (tau=4320h, 85% floor). Intended to let GFR drift downward over years as the setpoint erodes.

**Why reverted:** This is trajectory-fitting. Whether GFR hyperfiltrates for 8 years or progresses is determined by input parameters (Kf_scale, inflammation), not by TGF gain tuning. The simulator should report what the physics produces; RL/cohort generation chooses the inputs.

### 9. TGF: restored bisection; recalibrated R_AA0 *(genuine fix)*
After reverting dynamic TGF, the Picard relaxed iteration (from commit 3fe221a) failed to converge with R_AA0=16.4 → GFR=208. Restored the 40-iteration bisection, and recalibrated R_AA0 to match the bisection's exact equilibrium:
- `R_AA0: 16.4 → 25.0` (bisection gives GFR=122.7 at MAP=86.4; R_AA0=16.4 was paired with dynamic TGF)

**Verification:** Healthy baseline: GFR=122.7, MAP=86.4, EF=60%, V_blood stable ~5000 mL.

### 10. Additional ODE state variables added to history
- `systemic_inflammatory_index`, `myocardial_fibrosis_volume`, `endothelial_dysfunction_index`, `renal_tubulointerstitial_fibrosis`, `AGE_accumulation`, `LVEDP`

---

## Current State (after session reverts)

All trajectory-fitting changes from changes #7–8 have been reverted. Retained genuine fixes:
1. Volume mapping via 0.33 Guyton fraction (#1)
2. SVR_ratio = 1.0 (#2)
3. SVR_baseline from actual CircAdapt state (#3)
4. ODE rate constant rescaling (#6)
5. Bisection TGF with R_AA0 recalibrated for bisection (#9)

**Healthy baseline:** GFR=122.7, MAP=86.4, EF=60%, V_blood ~5000 mL (stable)
**Disease baseline (Kf=0.75, infl=0.2):** GFR=92.6, MAP=87.2, EF=58%

The simulator now produces whatever the physics dictates. Disease progression rate and GFR trajectory are determined by the input schedules passed to `run_coupled_simulation`. Aggressive inputs (Kf=0.50, infl=0.40) will produce faster CKD progression for cohort patients who reach CKD3–4 in 8 years.
