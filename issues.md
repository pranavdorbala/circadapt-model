# Known Issues

Tracked bugs, design issues, and fixes in the cardiorenal simulator.

---

## Open Issues

### 1. V_blood Still Drifts Upward (~50 mL/step)

**Status:** PARTIALLY FIXED — drift halved but not eliminated

**What happens:** V_blood rises ~50 mL per 6h step (was ~94 mL/step before VP controller + reabsorption fix). Over 8 steps: 5000 → 5064 → 5098 → 5133 → 5175 → 5226 → 5284 → 5346 → 5412.

**Root cause:** Na_excr (49 mEq/day) < Na_intake (150 mEq/day). Sodium is being retained, which raises C_Na, which makes the VP controller retain more water to maintain osmolality. The pressure natriuresis slopes (0.03/0.015) aren't strong enough to close the Na gap at the current GFR (~58 mL/min).

**Impact:**
- Non-RL path (4-8 steps): drift is +175-350 mL. Physiologically acceptable.
- RL path (120 six-hour steps per month): drift would be ~6000 mL → hits 8000 ceiling.

**Possible fixes:**
1. Tune pressure natriuresis slopes so Na_excr converges to Na_intake at steady state
2. Add myogenic autoregulation to R_AA (increases GFR → increases filtered Na → increases excretion)
3. Full two-compartment volume model with substeps (attempted, was unstable at 6h Euler steps)

---

### 2. RL Path V_blood = 8000 at Step 1

**Status:** OPEN — improved by Issue #1 fixes but likely still hits ceiling on RL timescale

**What happens:** The RL path uses `dt_renal_hours=180.0` with `renal_substeps=4` (720h total per coupling step). Even with the reduced drift rate (~50 mL/6h), over 30 days this accumulates ~6000 mL.

**Blocked by:** Requires fully solving Issue #1 (Na_excr must converge to Na_intake).

---

### 3. Simplified Renal Model vs Full Hallow 2017

**Status:** OPEN — architectural limitation from coarse coupling timestep

**Why we can't use the full Hallow model directly:** The coupling architecture calls `update_renal_model()` once per 6h (non-RL) or 180h (RL) step, using single-step Euler integration. Hallow's original model uses a proper ODE solver with adaptive timesteps (seconds-minutes). Our large dt makes fast dynamics unstable. The fundamental constraint is CircAdapt: it's an expensive black-box solver that runs to steady state, so we can't call it every few seconds inside a renal ODE.

**Differences between our model and Hallow 2017:**

| Component | Hallow 2017 | Ours | Status |
|---|---|---|---|
| Glomerular filtration (Starling eq) | Kf × (P_gc - P_Bow - pi_plasma) | Same | Matches |
| RBF / vascular resistances | Series R_preAA, R_AA, R_EA | Same | Matches |
| TGF (R_AA adjustment) | Adjusts R_AA from macula densa Na delivery | Same | Matches |
| RAAS (R_EA + CD adjustment) | Adjusts R_EA and eta_CD from MAP vs setpoint | Same | Matches |
| Tubular Na reabsorption | Sequential fractions of delivered Na | Same (fixed — was using wrong values) | Matches |
| Pressure natriuresis | MAP-dependent Na excretion modulation | Same (but slopes may differ) | ~Matches |
| Vasopressin / ADH | PI controller on C_Na → CD water permeability | Added (PI controller on C_Na → frac_water_reabs) | Matches |
| Volume model | Two-compartment: blood + interstitial with osmotic/diffusive transfer | Single-compartment: V_blood with 0.33 ECF factor | Simplified |
| Na balance | Two-compartment (plasma + interstitial Na) | Single pool (Na_total = total body Na) | Simplified |
| Myogenic autoregulation | R_AA responds to renal perfusion pressure | Missing | Missing |
| Aldosterone dynamics | Explicit concentration with synthesis/clearance ODE | Instantaneous RAAS_factor | Simplified |
| RAAS cascade | Renin → angiotensinogen → Ang I → Ang II → aldosterone | Collapsed to single RAAS_factor | Simplified |

**Missing components — implementation difficulty:**

**Easy (no architecture change, few lines of code):**
- Myogenic autoregulation — just add a term to R_AA that responds to renal perfusion pressure, alongside the existing TGF term.
- Tune pressure natriuresis slopes — adjust the 0.03/0.015 gains so Na_excr converges to Na_intake. Just constants.

**Medium (substep within `update_renal_model()`):**
- Two-compartment volume — we already wrote it, just needs substeps (split the 6h dt into smaller Euler steps within `update_renal_model`). The code got complicated but it's doable.
- Aldosterone kinetics — track aldosterone concentration with a synthesis/clearance ODE, substep it within the renal update.

**Hard (needs `scipy.integrate.solve_ivp` inside renal update):**
- Full RAAS cascade (renin → Ang I → Ang II → aldosterone) — multiple coupled ODEs with different time constants. Would need `scipy.integrate.solve_ivp` inside the renal update, packaging the full renal state as an ODE vector.

All of these happen *within* `update_renal_model()` — none require changing the CircAdapt coupling architecture. The 6h coupling timestep stays the same; we just do more work inside each call.

---

## Resolved Issues

### 4. Tubular Reabsorption Fractions Were Fractions-of-Filtered, Not Fractions-of-Delivered

**Status:** FIXED

**What was wrong:** `eta_LoH=0.25`, `eta_DT=0.05`, `eta_CD0=0.024` are standard physiology values for % of **filtered** Na reabsorbed by each segment. But the code applies them sequentially as `Na_after_X = Na_after_prev * (1 - eta)`, which treats them as fractions of **delivered**. This resulted in only 77% total reabsorption instead of 99.4%. Na_excr = 2400 mEq/day (16x the 150 mEq/day intake).

**Fix:** Converted fractions-of-filtered to fractions-of-delivered:
- `eta_LoH`: 0.25 → 0.758 (= 0.25 / 0.33)
- `eta_DT`: 0.05 → 0.626 (= 0.05 / 0.0799)
- `eta_CD0`: 0.024 → 0.803 (= 0.024 / 0.0299)

Na_excr dropped from 2400 to ~49 mEq/day. Underlying physiology numbers unchanged (67/25/5/2.4% of filtered).

**File changed:** `cardiorenal_coupling.py` (HallowRenalModel dataclass)

---

### 5. Water Balance Had No Feedback — V_blood Drifted ~94 mL/step

**Status:** FIXED (reduced to ~50 mL/step, see Issue #1 for remaining drift)

**What was wrong:** `frac_water_reabs = 0.99` was a fixed constant. Water excretion = GFR × 0.01 ≈ 0.6 mL/min = 0.86 L/day, but water intake = 2.0 L/day. Net gain always positive.

**Fix:** Added vasopressin PI controller (Hallow Eq. 31) that modulates `frac_water_reabs` based on plasma [Na+]:
1. PI controller: `VP = 1 + 0.05 × (2.0 × (C_Na - 140) + 0.005 × ∫(C_Na-140)dt)`
2. ADH permeability: `ADH = VP / (0.15 + VP)` (Michaelis-Menten saturation)
3. Effective reabsorption: `frac_eff = 0.90 + 0.098 × ADH` (linear in ADH_perm)
4. Low C_Na → low VP → low frac → more water excreted (negative feedback on volume)

Drift reduced from ~94 to ~50 mL/step. Remaining drift is from sodium retention (Issue #1).

**Files changed:** `cardiorenal_coupling.py` (HallowRenalModel dataclass + update_renal_model Stages 4-5)

---

### 6. Silent RL Residual Clamping — RL Didn't Know What Value Was Applied

**Status:** FIXED

**What was wrong:** `apply_inflammatory_residuals()` silently clamped corrected factor values with `max()` floors and `np.clip()`. When clamping activated, the RL thought it applied `base + residual` but the simulator used the clamped value. The RL learned wrong gradients.

**Fix:** Moved bounding into the RL's action space:
1. Per-factor residual bounds defined in `config.py` (`residual_min`/`residual_max` arrays)
2. Policy network (`CouplingPolicyHead`) uses tanh to map to per-factor ranges
3. Env (`rl_env._rescale_action`) rescales [-1,1] to per-factor ranges
4. `apply_inflammatory_residuals` now does pure addition — no clipping

**Files changed:** `config.py`, `rl_env.py`, `models/attention_coupling.py`, `train_rl.py`, `tests/test_end_to_end.py`, `tests/test_message_scaling.py`, `cardiorenal_coupling.py`

**Details:** See code_workings.md Section 6.

---

### 7. Wrong Step Numbers in CircAdapt Wrapper Docstring

**Status:** FIXED

**What was wrong:** `CircAdaptWrapper` docstring (~line 94) had arbitrary "Step 1, Step 2, Step 3" labels that didn't match Algorithm 1. Confusing when trying to map code to paper.

**Fix:** Changed to descriptive names: "Apply disease: scales Sf_act (contractility)", "Kidney to Heart: sets blood volume and SVR", "CircAdapt solver: runs to steady state", etc.

**File changed:** `cardiorenal_coupling.py`
