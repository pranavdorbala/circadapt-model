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
