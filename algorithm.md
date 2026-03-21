# Algorithm 1: Coupled Cardiorenal Simulation

## Step 1 — Inputs

The algorithm takes three inputs:

- **Disease schedules** {θ_H} and {θ_K}: The 8 disease parameters that define how sick the patient is.

  | Parameter | Organ | What it controls |
  |---|---|---|
  | `Sf_act_scale` | Heart | Contractility — how hard the heart squeezes |
  | `k1_scale` | Heart | Passive wall stiffness — how stiff the myocardium is |
  | `Kf_scale` | Kidney | Glomerular filtration capacity |
  | `RAAS_gain` | Kidney | Hormonal blood pressure regulation strength |
  | `TGF_gain` | Kidney | Tubuloglomerular feedback sensitivity |
  | `na_intake` | Kidney | Daily sodium intake |
  | `inflammation_scale` | Both | Systemic inflammatory burden (0 = none, 1 = severe) |
  | `diabetes_scale` | Both | Diabetes burden (0 = none, 1 = severe) |

  A healthy patient has all scales at 1.0 and inflammation/diabetes at 0.0. A sick patient might have `Sf_act_scale=0.7` (weak heart), `Kf_scale=0.6` (damaged kidneys), `inflammation_scale=0.5`.

- **Coupling α**: Controls how aggressively the two organs influence each other. At α=1, the full signal passes between heart and kidney. At α=0.5, only half the change passes through. This prevents oscillation — without dampening, the heart and kidney could ping-pong (heart raises MAP → kidney retains fluid → heart raises MAP more → diverges). In the RL version, α becomes 5 learned values that change at each time step.

- **Time step Δt**: The duration of each coupling step — one month. This tells the kidney model how long to simulate renal physiology (sodium accumulation, volume changes, GFR shifts) before passing results back to the heart.

## Step 2 — Model Initialization

Two models are initialized to a healthy baseline state:

- **CircAdapt heart model (H)**: Loads the VanOsta2024 cardiac simulator with a reference pressure waveform (`P_ref_VanOsta2024.npy`). This represents a healthy heart with normal chamber sizes, normal contractility, and normal pressures. No disease has been applied yet.

- **Hallow renal state (r_0)**: Creates a renal state with healthy baseline values — GFR = 100 mL/min, blood volume = 5000 mL, normal glomerular pressure, normal sodium excretion. This is a healthy kidney before any disease.

Both models start healthy. Disease is applied during the simulation loop (Step 4 onward).

## Step 3 — Pre-equilibration

Before the main simulation loop begins, the kidney's tubuloglomerular feedback (TGF) setpoint is stabilized by running 5 renal updates at baseline conditions.

**Why this is needed**: TGF is a feedback mechanism where the kidney adjusts its own filtration rate based on sodium delivery to the distal tubule. At initialization, the TGF setpoint may not match the kidney's actual operating point, causing artificial transients in the first few steps. Running 5 baseline updates lets the TGF settle to a self-consistent starting point so the simulation doesn't begin with spurious kidney oscillations.

After pre-equilibration, baseline values are stored:
- Blood volume V_b = 5000 mL
- Resistance ratio R_r = 1.0 (no scaling — heart sees normal vascular resistance)

These baselines are referenced later in the dampening equations (e.g., MAP_hat = MAP_baseline + α(MAP_current - MAP_baseline)) to compute how much each variable has changed relative to the healthy starting point.
