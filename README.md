# Simulation-Based Digital Twins for Cardiorenal Disease Progression via Agentic Message Passing

**Pranav Dorbala**
Department of Electrical and Computer Engineering
University of Illinois Urbana-Champaign
dorbala2@illinois.edu

> Preprint: Under Review, Machine Learning for Healthcare (MLHC) 2025

---

## Abstract

Heart failure with preserved ejection fraction (HFpEF) presents a fundamental modeling challenge: patients maintain normal systolic function while experiencing progressive multi-organ deterioration driven by tightly coupled feedback loops among the heart, kidneys, and vasculature. This repository implements a simulation-based digital twin framework that addresses three core requirements:

1. **Mechanistic cardiorenal simulator** coupling the CircAdapt cardiovascular model with a Hallow et al. (2017) renal physiology module via structured bidirectional message passing (MAP, CO, CVP, blood volume, SVR).
2. **Synthetic data generation pipeline** mapping coupled simulator states to 113 ARIC-compatible clinical variables, producing paired Visit 5/Visit 7 trajectories for training a residual neural network.
3. **Agentic LLM-based inference engine** that uses tool-calling over the mechanistic simulator to recover individualized disease parameters from clinical observations, producing interpretable parameter policies and mechanistic explanations.

---

## Repository Structure

```
.
├── cardiorenal_coupling.py    # Core coupled heart-kidney simulation engine
├── emission_functions.py      # 113 ARIC clinical variable extraction layer
├── synthetic_cohort.py        # Synthetic V5→V7 paired cohort generator
├── train_nn.py                # Residual neural network (V5→V7 prediction)
├── agent_loop.py              # Agentic LLM inference engine
├── agent_tools.py             # LLM-callable tools (run model, compute error, etc.)
├── pipeline.py                # End-to-end pipeline integrating NN + agent
├── config.py                  # Shared configuration, ARIC variable metadata
├── P_ref_VanOsta2024.npy      # CircAdapt reference hemodynamic data
├── models/
│   └── v5_to_v7_best.pt       # Pre-trained residual NN weights
├── manuscript/
│   ├── paper.pdf              # Full manuscript
│   ├── paper.tex              # LaTeX source
│   ├── paper_refs.bib         # References
│   ├── plot.pdf               # Figure
│   ├── jmlr.cls               # JMLR document class
│   ├── jmlrutils.sty          # JMLR utilities
│   └── mlhc2024.sty           # MLHC style file
└── README.md
```

### File Descriptions

| File | Paper Section | Description |
|------|--------------|-------------|
| `cardiorenal_coupling.py` | 3.1-3.4 | CircAdapt heart model wrapper, Hallow renal model, inflammatory mediator layer, bidirectional message passing (Algorithm 1) |
| `emission_functions.py` | 3.6 | Maps CircAdapt waveforms + Hallow outputs to 113 ARIC echocardiographic and renal variables |
| `synthetic_cohort.py` | 3.7 | Generates synthetic patient cohorts with correlated disease progression over 6 years |
| `train_nn.py` | 3.8 | Residual MLP: X_7 = W_skip * X_5 + g_phi(X_5), with composite loss (Eq. 14) |
| `agent_loop.py` | 3.9 | LLM-based agentic optimizer with tool-calling loop and Nelder-Mead fallback |
| `agent_tools.py` | 3.9 | Four LLM tools: `run_circadapt_model`, `compute_error`, `get_sensitivity`, `compare_to_clinical_norms` |
| `pipeline.py` | 3.11 | Full pipeline: V5 input → NN prediction → Agent optimization → result |
| `config.py` | All | Tunable parameter definitions, ARIC variable metadata with clinical weights, thresholds |

---

## Installation

### Prerequisites

- Python 3.9+
- CircAdapt Python package (`pip install circadapt`)
- PyTorch 1.13+ (for neural network)
- LiteLLM (for agentic inference, optional)

### Setup

```bash
# Clone the repository
git clone https://github.com/pranavdorbala/circadapt-model.git
cd circadapt-model

# Install dependencies
pip install circadapt numpy scipy torch litellm

# Verify CircAdapt installation
python -c "from circadapt import VanOsta2024; m = VanOsta2024(); m.run(stable=True); print('CircAdapt OK')"
```

---

## Reproducing Results

The framework has five sequential stages. Each stage depends on the output of the previous one.

### Stage 1: Coupled Cardiorenal Simulation (Section 3.1-3.4)

The core simulation couples CircAdapt (heart) with Hallow et al. 2017 (kidney) via bidirectional message passing. To run a single coupled simulation:

```python
from cardiorenal_coupling import (
    CircAdaptHeartModel, HallowRenalModel, InflammatoryState,
    update_inflammatory_state, update_renal_model,
    heart_to_kidney, kidney_to_heart, run_coupled_simulation
)

# Run 8-step coupled simulation with progressive HFrEF
history = run_coupled_simulation(
    n_steps=8,
    dt_renal_hours=6.0,
    cardiac_schedule=[1.0, 0.92, 0.82, 0.72, 0.62, 0.52, 0.45, 0.40],  # Sf_act declining
    kidney_schedule=[1.0] * 8,  # Kidney healthy
    inflammation_schedule=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
    diabetes_schedule=[0.0] * 8,
)

# Access results
print(f"Final MAP: {history['MAP'][-1]:.1f} mmHg")
print(f"Final GFR: {history['GFR'][-1]:.1f} mL/min")
print(f"Final EF:  {history['EF'][-1]:.0f}%")
```

**Key classes:**
- `CircAdaptHeartModel`: Wraps VanOsta2024 with methods for deterioration, stiffness, inflammatory modifiers, and kidney feedback
- `HallowRenalModel`: Dataclass holding renal state (GFR, blood volume, Na balance, RAAS, TGF)
- `InflammatoryState`: Mediator layer implementing Table 1 (diabetes + inflammation effects on both organs)

### Stage 2: Clinical Emission Layer (Section 3.6)

Extract 113 ARIC-compatible clinical variables from a CircAdapt simulation:

```python
from circadapt import VanOsta2024
from emission_functions import extract_all_aric_variables

# Run CircAdapt to steady state
model = VanOsta2024()
model.run(stable=True)
model.run(1)  # Store 1 beat for waveform extraction

# Renal state from Hallow model
renal_state = {
    'GFR': 110.0, 'V_blood': 5000.0, 'C_Na': 140.0,
    'Na_excretion': 150.0, 'P_glom': 55.0, 'Kf_scale': 1.0, 'RBF': 1100.0,
}

# Extract all variables
variables = extract_all_aric_variables(
    model, renal_state,
    BSA=1.9, height_m=1.75, age=72, sex='M'
)
print(f"Extracted {len(variables)} variables")
print(f"LVEF: {variables['LVEF_pct']:.1f}%, GFR: {variables['GFR_mL_min']:.1f} mL/min")
```

**Variable categories:** LV structure (7), LV systolic (6), Doppler/diastolic (13), LA (11), RV (9), aortic (6), pulmonary (6), blood pressure (6), timing/MPI (4), myocardial work (4), vascular (5), renal/lab (16), indexed (9), diastolic grade (2).

### Stage 3: Synthetic Cohort Generation (Section 3.7)

Generate paired V5/V7 ARIC variable vectors for training:

```bash
# Generate 10,000 synthetic patients (parallelized, ~2 hours on 8 cores)
python synthetic_cohort.py --n_patients 10000 --n_workers 8 --seed 42

# Quick test with 100 patients
python synthetic_cohort.py --n_patients 100 --n_workers 1
```

**Output:** `cohort_data.npz` containing:
- `v5`: ndarray of shape (N, 113) — Visit 5 clinical variables
- `v7`: ndarray of shape (N, 113) — Visit 7 clinical variables
- `var_names`: ordered variable names

**Sampling strategy:**
- Demographics: age ~ U(65, 85), BSA ~ N(1.9, 0.15), height ~ N(1.70, 0.08)
- Disease params: Sf_act ~ N(0.92, 0.12), Kf ~ Beta(5,2), inflammation ~ Exp(0.10)
- Progression: correlated deltas via Cholesky decomposition of 6x6 correlation matrix

### Stage 4: Neural Network Training (Section 3.8)

Train the residual MLP for V5→V7 prediction:

```bash
# Train (requires cohort_data.npz from Stage 3)
python train_nn.py --data cohort_data.npz --epochs 200 --hidden_dim 256 --n_blocks 3

# Evaluate a pre-trained model
python train_nn.py --evaluate models/v5_to_v7_best.pt --data cohort_data.npz
```

**Architecture (Eq. 14):** `X_7 = W_skip * X_5 + g_phi(X_5)` where:
- `W_skip`: learnable linear skip connection (identity-like)
- `g_phi`: 3 residual blocks (Linear→BatchNorm→ReLU→Dropout→Linear→BatchNorm + skip), hidden dim 256
- **Loss**: weighted MSE / training_std + 0.1 * direction-of-change penalty
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4) with cosine annealing
- **Early stopping**: patience=20 on validation loss

### Stage 5: Agentic Inference Engine (Section 3.9)

Run the LLM-based agent to recover disease parameters from clinical observations:

```bash
# Requires an LLM API key (e.g., OPENAI_API_KEY for GPT-4o)
export OPENAI_API_KEY=your_key_here

# Full pipeline: V5 → NN prediction → Agent optimization
python pipeline.py --params '{"Sf_act_scale": 0.8, "Kf_scale": 0.7}'

# Direct agent test
python agent_loop.py --model gpt-4o --max_iter 15
```

**Agent tools:**
1. `run_circadapt_model(theta)`: Runs coupled model, returns 113 ARIC variables (~0.5s)
2. `compute_error(X_model, X_target)`: Weighted per-variable error with worst-5 analysis
3. `get_sensitivity(theta, p, delta)`: Finite-difference Jacobian dX/dp (~1s, two model runs)
4. `compare_to_clinical_norms(X)`: Disease staging (HF class, CKD stage, filling pressure grade)

**Convergence:** Agent iterates until aggregate error < 5% (normalized). Falls back to Nelder-Mead simplex if stagnated for 3 consecutive iterations after 10 total.

---

## Key Equations

| Eq. | Description | Implementation |
|-----|-------------|----------------|
| 1-2 | Coupled dynamical system | `run_coupled_simulation()` |
| 3-4 | Message vectors (H→K: MAP,CO,CVP; K→H: V_blood,SVR) | `heart_to_kidney()`, `kidney_to_heart()` |
| 5 | Coupling intensity scaling | `apply_kidney_feedback()` |
| 6 | Passive fiber stress (EDPVR): S_f,pas = k1(e^{k2(lambda-1)} - 1) | `apply_stiffness()` |
| 7 | Pressure-flow: q = sign(dp) * q0 * (\|dp\|/p0)^k | CircAdapt ArtVen `p0` parameter |
| 8 | RAAS: clip(1 - g_R * 0.005 * dMAP, 0.5, 2.0) | `update_renal_model()` |
| 9 | TGF: R_AA^(k+1) = R_AA^0 * (1 + g_TGF * (MD_Na - MD_Na*) / MD_Na*) | `update_renal_model()` |
| 10-12 | Renal hemodynamics (RBF, P_gc, SNGFR) | `update_renal_model()` |
| 13 | Diabetic Kf: Kf * (1 + 0.25d(1 - 1.5d)) | `update_inflammatory_state()` |
| 14 | Residual NN: X_7 = W_skip * X_5 + g_phi(X_5) | `V5toV7Net` |
| 15-17 | Cardiac counterfactual decomposition | `run_coupled_simulation()` |
| 18-19 | Renal counterfactual decomposition | `run_coupled_simulation()` |
| 21 | Rate amplification metric | `run_coupled_simulation()` |

---

## Disease Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `Sf_act_scale` | 0.2-1.0 | 1.0 | Active fiber stress (HFrEF: <1) |
| `Kf_scale` | 0.05-1.0 | 1.0 | Glomerular ultrafiltration (CKD: <1) |
| `k1_scale` | 1.0-3.0 | 1.0 | Passive myocardial stiffness (HFpEF: >1) |
| `inflammation_scale` | 0.0-1.0 | 0.0 | Systemic inflammation index |
| `diabetes_scale` | 0.0-1.0 | 0.0 | Diabetes metabolic burden |
| `RAAS_gain` | 0.5-3.0 | 1.5 | Renin-angiotensin sensitivity |
| `TGF_gain` | 1.0-4.0 | 2.0 | Tubuloglomerular feedback gain |
| `na_intake` | 50-300 | 150 | Dietary sodium (mEq/day) |

---

## References

- **CircAdapt:** van Osta et al., "CircAdapt VanOsta2024: a computational model of the human cardiovascular system for simulation of heart failure phenotypes," *European Heart Journal - Digital Health*, 2024. Framework: https://framework.circadapt.org
- **Renal model:** Hallow et al., "A model-based approach to investigating the pathophysiological mechanisms of hypertension and response to antihypertensive therapies," *CPT:PSP*, 6(1):48-57, 2017.
- **Cardiorenal coupling:** Basu et al., "A coupled computational model of cardiorenal hemodynamics and volume regulation," *PLoS Computational Biology*, 19(11):e1011598, 2023.
- **TriSeg:** Lumens et al., "Three-wall segment (TriSeg) model describing mechanics and hemodynamics of ventricular interaction," *Annals of Biomedical Engineering*, 37(11):2234-2255, 2009.

---

## Citation

```bibtex
@article{dorbala2025cardiorenal,
  title={Simulation-Based Digital Twins for Cardiorenal Disease Progression via Agentic Message Passing},
  author={Dorbala, Pranav},
  journal={Preprint, Machine Learning for Healthcare},
  year={2025}
}
```

---

## License

This project is for academic research purposes. The CircAdapt framework is available at https://framework.circadapt.org under its own license terms.
