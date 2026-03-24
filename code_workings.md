# Code Workings

How the code in this repository implements the paper's equations and algorithms. Each section explains what the paper says, what the code does, and where the computation lives.

---

## 1. Disease Parameters

The coupled simulator is controlled by 8 disease parameters. Each parameter is applied through two steps: (1) an inflammatory modifier computed from `inflammation_scale` and `diabetes_scale`, and (2) the final value set in the underlying model (CircAdapt or Hallow). These two steps are split across different locations in the code.

### 1.1 Sf_act — Active Contractility

**What it is:** Peak isometric force the cardiac muscle fiber generates during contraction. Lower = weaker heart = systolic heart failure.

**Paper:** Section 3.1 — "active fiber stress governed by a Hill-type cross-bridge cycling model." No numbered equation for the scaling; the Hill model is internal to CircAdapt and we only scale the amplitude.

**Full scaling equation:**
```
Sf_act(t) = Sf_act_ref × Sf_act_scale(t) × Sf_act_factor_inflammatory
```

| Term | What it is | Where in code |
|---|---|---|
| `Sf_act_ref` | Healthy baseline, cached at init | `CircAdaptWrapper.__init__` (~line 127) |
| `Sf_act_scale` | Direct disease knob (0.4 = severe HF) | Input parameter from config.py |
| `Sf_act_factor` | `(1 - 0.25×infl) × (1 - 0.20×diab)` | `compute_inflammatory_state` (~line 860) |

**How the code applies it (split across two places):**

1. Simulation loop (`run_coupled_simulation`, ~line 1955):
   ```python
   effective_sf = max(sf * ist.Sf_act_factor, 0.20)
   heart.apply_deterioration(effective_sf)
   ```
   Multiplies the disease knob by the inflammatory factor. The `max(_, 0.20)` floor prevents zero contractility (solver crash).

2. CircAdapt wrapper (`apply_deterioration`, ~line 191):
   ```python
   self.model['Patch']['Sf_act']['pLv0'] = self._ref_Sf_act_lv * Sf_act_scale
   ```
   By this point, `Sf_act_scale` already includes the inflammatory factor. Multiplies by cached healthy reference to set the final CircAdapt value.

**Example:** `Sf_act_scale=0.7`, `inflammation=0.4`, `diabetes=0.3`
```
Sf_act_factor = (1 - 0.25×0.4) × (1 - 0.20×0.3) = 0.90 × 0.94 = 0.846
effective_sf = 0.7 × 0.846 = 0.592
→ Heart at ~59% contractility
```

### 1.2 k1 — Passive Myocardial Stiffness

**What it is:** Coefficient in the exponential passive stress-strain curve. Higher k1 = stiffer heart wall = higher filling pressures = HFpEF.

**Paper equation (Eq. 6):**
```
S_f,pas = k1 × (exp(k2 × (λ − 1)) − 1)
```
CircAdapt implements Eq. 6 internally. We only scale the k1 coefficient — but because it sits inside an exponential, small changes produce large hemodynamic effects.

**Scaling equation:**
```
k1(t) = k1_ref × k1_scale(t) × passive_k1_factor_inflammatory
```

| Term | What it is | Where in code |
|---|---|---|
| `k1_ref` | Healthy baseline, cached at init | `CircAdaptWrapper.__init__` (~line 140) |
| `k1_scale` | Direct disease knob (>1.0 = HFpEF), clamped [0.5, 4.0] | Input parameter from config.py |
| `passive_k1_factor` | `1 + 0.40×diab` (only diabetes, not inflammation) | `compute_inflammatory_state` (~line 899) |

**How the code applies it (split across two places):**

1. Simulation loop (`run_coupled_simulation`, ~line 1946):
   ```python
   effective_k1 = k1 * ist.passive_k1_factor
   heart.apply_stiffness(effective_k1)
   ```

2. CircAdapt wrapper (`apply_stiffness`, ~line 230):
   ```python
   k1_scale = np.clip(k1_scale, 0.5, 4.0)
   self.model['Patch']['k1']['pLv0'] = self._ref_k1_lv * k1_scale
   ```
   Clamp prevents solver instability (below 0.5 = unrealistically compliant, above 4.0 = filling stops).

**Why only diabetes affects k1:** Diabetes causes AGE (Advanced Glycation End-products) collagen cross-linking — sugar molecules physically stiffen heart tissue. General inflammation doesn't have the same direct structural effect on passive stiffness.

**Example:** `k1_scale=1.5`, `diabetes=0.5`
```
passive_k1_factor = 1 + 0.40×0.5 = 1.20
effective_k1 = 1.5 × 1.20 = 1.80
→ Heart wall 80% stiffer than normal → elevated filling pressures (diastolic dysfunction)
```

---

## 2. Vascular Resistance (p0) — Multiplicative Composition

> **KEY FINDING FOR PAPER:** The multiplicative composition of vascular resistance modifiers is not an arbitrary implementation choice — it is the standard convention in cardiovascular physiology modeling, grounded in Poiseuille's law and used by Guyton (1972), HumMod, and Hallow et al. (2017). This should be explicitly stated and justified in the paper.

### What p0 is

In CircAdapt, there is no direct "SVR knob." Vascular resistance is controlled indirectly through `p0`, the pressure set-point in the ArtVen element. CircAdapt's pressure-flow relationship (Appendix B.3.2, Eq. 22):

```
q = sign(Δp) · q0 · (|Δp| / p0)^k
```

Raising p0 means more pressure is needed to drive the same flow — equivalent to increasing vascular resistance.

### What modifies p0

Two independent mechanisms modify p0 simultaneously:

| Source | What it represents | Paper reference |
|---|---|---|
| `SVR_ratio` | Kidney → Heart message: RAAS activation + nephron loss increase resistance | Eq. 27: `SVR_t = RAAS_t · (1 + 0.4(1 - K_f^scale))`, dampened by α (Eq. 5) |
| `_pathology_p0_factor` | Diabetes/inflammation: endothelial dysfunction + microvascular rarefaction | Eq. 43: `p0_eff = p0^(0) · (1 + φ4 · d)`, φ4 = 0.10; plus inflammation via `p0_factor = (1 + 0.15·infl) · (1 + 0.10·diab)` |

### How they combine in code

```python
# cardiorenal_coupling.py, apply_kidney_feedback(), ~line 283
self.model['ArtVen']['p0']['CiSy'] = (
    self._ref_ArtVen_p0 * SVR_ratio * self._pathology_p0_factor
)
```

The paper describes each factor separately (Eq. 27, Eq. 43) but does not explicitly write out the combined product. The multiplicative composition is an implementation choice.

### Why multiplicative is correct — literature justification

**Poiseuille's Law** provides the physical basis: `R = 8ηL / (πr⁴)`. Different mechanisms affect different physical quantities:
- **RAAS** contracts smooth muscle → changes arteriolar radius `r`
- **Inflammation** increases endothelial dysfunction / viscosity → changes `η`
- **Diabetes** reduces capillary density (microvascular rarefaction) → changes effective `L`

Since these act on independent physical parameters, their effects on resistance compose multiplicatively.

**Guyton / HumMod models** (the gold standard for cardiovascular simulation) use this exact product-of-multipliers approach:
```
TPR = R_base × M_autoregulation × M_angiotensin × M_sympathetic × M_viscosity × ...
```
Each modifier is a dimensionless multiplier (1.0 = baseline). This has been the convention since Guyton 1972 and is used in all major descendants (HumMod, CellML reimplementations).

**Hallow et al. 2017** (the kidney model in this framework) also uses multiplicative composition for neurohormonal modulation of resistance segments — RAAS modulates arteriole diameters through logistic saturation functions, and resistance follows Poiseuille's law (R ~ 1/d⁴).

**Key references:**
- Guyton AC, Coleman TG, Granger HJ. "Circulation: overall regulation." *Annual Review of Physiology*, 1972.
- Hester RL et al. "HumMod: an integrative model of integrative human physiology." *Frontiers in Physiology*, 2011.
- Hallow KM et al. "Cardiorenal modeling of blood pressure and kidney function." *CPT: Pharmacometrics & Systems Pharmacology*, 2017.

### Why this matters

Additive composition (`p0_ref × (1 + (SVR_ratio - 1) + (p0_factor - 1))`) would underestimate the combined effect in severe disease. At small perturbations the difference is negligible, but at large perturbations (severe CKD + diabetes), multiplicative compounding produces qualitatively different long-horizon trajectories — exactly the regime where accurate coupling matters most for forecasting.

### Note on p0 as SVR proxy

Scaling p0 linearly does not produce a perfectly linear SVR change because of the exponent `k` in Eq. 22 (`SVR ∝ p0^k`, not `p0`). If `k ≠ 1`, the relationship is nonlinear. However, after p0 is set, CircAdapt runs to steady state and re-equilibrates all pressures and flows. The final SVR is determined by the full hemodynamic equilibrium, not just p0 alone. The p0 scaling is a directional nudge; CircAdapt's solver finds the correct equilibrium.

---

## 3. Run to Steady State — Three-Phase Hemodynamic Convergence

After all parameters are set (disease knobs, inflammatory modifiers, kidney feedback), CircAdapt needs to find the new hemodynamic equilibrium. This is done in three phases inside `run_to_steady_state()` (`cardiorenal_coupling.py`, ~line 356).

### Why three phases?

When the kidney model updates blood volume (e.g., from 5.0L to 5.2L due to sodium retention), CircAdapt can't just instantly accept that — the extra 200mL needs to be distributed across all vascular compartments (arteries, veins, heart chambers), and the heart needs to adapt to the new loading conditions over multiple beats.

### Phase 1 — PFC ON: Volume redistribution

```python
self.model['PFC']['is_active'] = True
self.model.run(stable=True)
```

PFC (Peripheral Flow Control) is CircAdapt's built-in mechanism for distributing blood volume across the circulation. With PFC enabled, CircAdapt actively moves fluid between compartments until the target volume (set by `apply_kidney_feedback`) is achieved. The solver runs beat after beat until pressures and volumes converge.

**Analogy:** Pouring water into a system of connected vessels — PFC ensures the water reaches every compartment in the right proportions.

### Phase 2 — PFC OFF: Free equilibration

```python
self.reset_volume_control()   # PFC off
self.model.run(stable=True)
```

Now PFC is turned off and the circulation runs again to stability. Why? With PFC on, it was *actively forcing* volume distribution — like holding a ball underwater. Turning it off lets the circulation settle naturally under its own pressure-flow dynamics. This gives the true hemodynamic equilibrium: the pressures, flows, and volumes that would actually exist with the new blood volume and disease parameters.

**Why this matters:** If we read measurements while PFC is still active, we'd get artificially clamped values, not the natural equilibrium. The heart needs to settle without external forcing.

### Phase 3 — Clean beat extraction

```python
self.model['Solver']['store_beats'] = 1
self.model.run(1)
```

Run one final cardiac cycle and store only that beat's waveforms. All the transient settling beats from phases 1 and 2 are discarded. This clean beat is what we extract measurements from (MAP, CO, EF, SV, full PV-loop waveforms for the emission functions).

### Error handling

Each phase is wrapped in try/except because extreme disease parameters (e.g., `Sf_act_scale=0.2` with severe volume overload) can crash the CircAdapt solver. The fallback strategy is:
1. Try `run(stable=True)` — let CircAdapt decide when it's converged
2. If that fails, try `run(n_settle)` — run a fixed number of beats (default 5)
3. If that also fails, continue with the last valid state — partial results are better than crashing the entire simulation

This defensive approach is essential for synthetic cohort generation, where thousands of random parameter combinations are tested and some will inevitably push the solver to its limits.

---

## 4. Inflammatory Mediator Layer (Our Contribution)

CircAdapt (Van Osta et al. 2024) and Hallow (Hallow et al. 2017) are existing, validated simulators built by other groups. CircAdapt simulates a heart. Hallow simulates a kidney. Neither knows anything about inflammation, diabetes, or the other organ.

The inflammatory mediator layer, the bidirectional coupling protocol (Algorithm 1), and the translation of `inflammation_scale`/`diabetes_scale` into parameter modifications for both simulators are **our additions** — the novel contribution that connects two independent organ models into a coupled cardiorenal system with shared metabolic disease drivers.

In code, this lives in `compute_inflammatory_state()` (~line 830) and `apply_inflammatory_modifiers()` (~line 299) in `cardiorenal_coupling.py`.

### How it works

The inflammatory layer takes two inputs — `inflammation_scale` ∈ [0,1] and `diabetes_scale` ∈ [0,1] — and computes modifier factors for parameters in both organs. For example:

```python
infl_Sf = 1.0 - 0.25 * infl     # Up to 25% contractility reduction from inflammation
diab_Sf = 1.0 - 0.20 * diab     # Up to 20% contractility reduction from diabetes
state.Sf_act_factor = infl_Sf * diab_Sf  # Multiplicative: independent mechanisms
```

This factor then gets multiplied with `Sf_act_scale` in the simulation loop — so if a patient has `Sf_act_scale=0.7`, `inflammation=0.4`, `diabetes=0.3`, the effective contractility is `0.7 × 0.90 × 0.94 = 0.59` (heart at 59% strength).

The coefficients (0.25, 0.20, 0.40, etc.) are approximate magnitudes from published clinical studies (Table 5 in the paper). The RL agent learns residual corrections (Δφ_j) on top of them from ARIC data.

### Parameters affected by the inflammatory layer

| Modifier | Formula | What it does | Affected by | Source |
|---|---|---|---|---|
| `Sf_act_factor` | `(1 - 0.25·infl) × (1 - 0.20·diab)` | Reduces heart contractility | Inflammation + Diabetes | Feldman 2000, Bugger 2014 |
| `passive_k1_factor` | `1 + 0.40·diab` | Increases heart wall stiffness | Diabetes only | van Heerebeek 2008 |
| `stiffness_factor` | `(1 + 0.30·infl) × (1 + 0.50·diab)` | Increases arterial stiffness (Tube0D k) | Inflammation + Diabetes | Vlachopoulos 2005, Prenner 2015 |
| `p0_factor` | `(1 + 0.15·infl) × (1 + 0.10·diab)` | Increases vascular resistance (SVR) | Inflammation + Diabetes | Endemann 2004 |
| `Kf_factor` | `(1 - 0.15·infl) × biphasic_diabetes` | Reduces kidney filtration capacity | Inflammation + Diabetes | Brenner 1996 |
| `EA_constriction_factor` | `1 + 0.25·diab` | Constricts efferent arteriole | Diabetes only | Brenner 1996 |
| `eta_PT_offset` | `0.04·infl + 0.06·diab` | Increases proximal tubule sodium reabsorption | Inflammation + Diabetes | Thomson 2004 |
| `MAP_setpoint_offset` | `max(5·infl, 8·diab)` | Shifts kidney's MAP set-point upward | Inflammation or Diabetes (whichever is worse) | — |

### Parameters NOT affected by the inflammatory layer

| Parameter | Why not |
|---|---|
| `TGF_gain` | Patient-level autoregulation strength — varies between people, not a disease mechanism |
| `na_intake` | Dietary sodium — purely behavioral (how much salt they eat) |
| `RAAS_gain` | Patient-level hormonal sensitivity — set directly, not modified by inflammation |

### How modifiers are applied (separation of concerns)

Not all modifiers are applied in the same place. The architecture ensures single write paths:

| Modifier | Applied where | Why |
|---|---|---|
| `stiffness_factor` | Directly in `apply_inflammatory_modifiers()` | Only source — no other factor modifies arterial stiffness |
| `passive_k1_factor` | Stored, applied later in `apply_stiffness()` | Must compose with `k1_scale` (direct disease knob) |
| `p0_factor` | Stored, applied later in `apply_kidney_feedback()` | Must compose with `SVR_ratio` (kidney feedback) |
| `Sf_act_factor` | Applied in simulation loop before `apply_deterioration()` | Must compose with `Sf_act_scale` (direct disease knob) |
| Renal modifiers | Applied inside `update_renal_model()` | Must compose with kidney-side disease parameters |

### Simple vs Full ODE Inflammatory Model

The codebase contains two versions of the inflammatory layer. The **simple version** (currently active) and a **full ODE version** (commented out, ~line 969–1085).

**Simple version (active):** Stateless parametric scaling. Set `inflammation_scale=0.5`, instantly get `Sf_act_factor = 0.875`. No memory of how long the patient has been inflamed — same scale always produces the same modifiers.

**Full ODE version (commented out):** Models inflammation as a dynamic state variable evolving via `dI/dt` with explicit sources and sinks:

| Sources (create inflammation) | Mechanism |
|---|---|
| Uremic toxins | Kidney failure → toxin buildup → inflammation |
| Cardiac congestion | Elevated CVP → gut edema → bacterial translocation → inflammation |
| AGE-RAGE signaling | Diabetes → advanced glycation end-products → chronic inflammation |
| Aldosterone | RAAS activation → mineralocorticoid-driven inflammation |

| Sinks (clear inflammation) | Mechanism |
|---|---|
| Immune clearance | Body's natural resolution, modeled with a half-life |

The ODE version also tracks **dynamic state variables** that accumulate over time:
- Fibrosis (irreversible tissue scarring)
- Endothelial dysfunction
- AGE accumulation
- Renal tubulointerstitial fibrosis

**Why this matters:** In the ODE version, a patient inflamed for 5 years develops irreversible fibrosis, while a patient inflamed for 1 month does not — the simple version cannot distinguish these cases. The ODE version would also make the inflammatory layer **bidirectional**: kidney damage causes inflammation which worsens heart damage which worsens kidney damage, creating the cardiorenal syndrome vicious cycle as an emergent property rather than a prescribed input.

**Why it's commented out:** The simple version is sufficient for the current paper. The ODE version is ready for future activation but adds parameters that would need calibration against longitudinal clinical data.
