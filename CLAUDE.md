# Cardiorenal Digital Twin Framework

## Project Overview
Simulation-based digital twin for cardiorenal disease progression in HFpEF. Couples CircAdapt VanOsta2024 cardiovascular model with Hallow et al. 2017 renal physiology via bidirectional message passing. Includes synthetic cohort generation, residual neural network for V5→V7 prediction, and agentic LLM-based inference engine.

Paper: "Simulation-Based Digital Twins for Cardiorenal Disease Progression via Agentic Message Passing" (Dorbala, MLHC 2025)

## Key Files
- `cardiorenal_coupling.py` — Core coupled simulation: CircAdapt heart + Hallow kidney + inflammatory mediator layer (Sections 3.1-3.4)
- `emission_functions.py` — 113 ARIC clinical variable extraction from CircAdapt waveforms + Hallow outputs (Section 3.6)
- `synthetic_cohort.py` — Synthetic V5/V7 paired cohort generation with correlated disease progression (Section 3.7)
- `train_nn.py` — Residual MLP: X7 = W_skip·X5 + g_phi(X5) with composite loss (Section 3.8, Eq. 14)
- `agent_loop.py` — Agentic LLM optimizer with tool-calling and Nelder-Mead fallback (Section 3.9)
- `agent_tools.py` — Four LLM-callable tools wrapping the coupled model (Section 3.9)
- `pipeline.py` — End-to-end: V5 → NN prediction → Agent optimization → result (Section 3.11)
- `config.py` — Shared configuration: parameter ranges, ARIC variable metadata, clinical thresholds

## Architecture
- **Heart**: CircAdapt VanOsta2024 — sarcomere mechanics, TriSeg ventricular interaction, closed-loop circulation
- **Kidney**: Hallow et al. 2017 — glomerular hemodynamics, TGF (Eq. 9), RAAS (Eq. 8), tubular Na handling, volume balance
- **Coupling**: Bidirectional messages per Algorithm 1. H→K: MAP, CO, CVP. K→H: V_blood, SVR
- **Inflammatory layer**: Diabetes (d∈[0,1]) and inflammation (i∈[0,1]) modify both organs (Table 1)
- **Emission**: 113 ARIC variables across 8 categories from CircAdapt waveforms + Hallow outputs
- **NN**: Residual MLP, 3 blocks, hidden dim 256, dropout 0.1, weighted MSE + direction penalty
- **Agent**: LLM tool-calling loop (GPT-4o/Gemini/Claude via LiteLLM) with 4 tools

## Dependencies
```bash
pip install circadapt numpy scipy torch litellm
```

## Quick Start
```bash
# Run coupled simulation
python -c "from cardiorenal_coupling import run_coupled_simulation; run_coupled_simulation(n_steps=4)"

# Generate synthetic cohort (100 patients, quick test)
python synthetic_cohort.py --n_patients 100 --n_workers 1

# Train neural network
python train_nn.py --data cohort_data.npz --epochs 50

# Run full pipeline (requires LLM API key)
python pipeline.py --params '{"Sf_act_scale": 0.8, "Kf_scale": 0.7}'
```
