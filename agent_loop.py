#!/usr/bin/env python3
"""
Agentic LLM Framework: Cardiorenal Parameter Optimization
===========================================================
(Paper Section 3.9 — Agentic Inference Engine)

This module implements the core agentic inference loop described in Section 3.9
of the manuscript. The high-level idea: given a patient's Visit 5 (baseline)
clinical data and a predicted Visit 7 (6-year follow-up) target, an LLM
autonomously drives a tool-calling loop to discover the set of mechanistic
disease-progression parameters that make the coupled CircAdapt + Hallow
cardiorenal model reproduce the V7 clinical target.

Architecture overview (Section 3.9, Figure X):
    ┌──────────────────────────────────────────────────────────────┐
    │  System prompt  ─────────►  LLM (GPT-4o / Gemini / etc.)    │
    │  + V5 data + V7 target       │                               │
    │                               ▼                               │
    │                          Tool call?                           │
    │                         /    |    \\                          │
    │            run_model  compute  get_sens  classify             │
    │               │         err     │         │                   │
    │               ▼         ▼       ▼         ▼                   │
    │           JSON result → appended to conversation context      │
    │               │                                               │
    │               ▼                                               │
    │          Convergence check (error < threshold?)               │
    │              YES → ask LLM for explanation → return           │
    │              NO  → stagnation? → Nelder-Mead fallback         │
    │              NO  → continue loop                              │
    └──────────────────────────────────────────────────────────────┘

The agent produces three outputs per patient:
    1. Optimal disease parameters (the 8 tunable knobs)
    2. A parameter policy (human-readable summary of what changed and why)
    3. A mechanistic explanation (narrative of the V5→V7 disease trajectory)

Usage:
    from agent_loop import CardiorenalAgent
    agent = CardiorenalAgent(model="gpt-4o")
    result = agent.solve(v5_data, v7_target, demographics)
"""

# ─── Standard library imports ────────────────────────────────────────────────
import os
import sys
import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Ensure the project root is on sys.path so sibling modules can be imported.
# This is necessary when running from any directory, not just the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Internal project imports ────────────────────────────────────────────────
# TUNABLE_PARAMS: dict of the 8 disease-progression parameters with ranges,
#     defaults, and descriptions (see config.py / paper Section 3.4).
# LLM_CONFIG: hyperparameters for the agentic loop (model name, temperature,
#     max iterations, convergence threshold).
# NUMERIC_VAR_NAMES: the sorted list of ~113 ARIC-compatible numeric variable
#     names used as both NN features and model outputs.
from config import TUNABLE_PARAMS, LLM_CONFIG, NUMERIC_VAR_NAMES

# TOOL_SCHEMAS: OpenAI function-calling JSON schemas for the 4 tools
#     (run_circadapt_model, compute_error, get_sensitivity,
#      compare_to_clinical_norms). These are passed to litellm.completion().
# execute_tool: dispatches a tool call by name, returns a JSON string.
from agent_tools import TOOL_SCHEMAS, execute_tool


# ═══════════════════════════════════════════════════════════════════════════════
# Result Dataclass
# ═══════════════════════════════════════════════════════════════════════════════
# AgentResult is the structured output returned by CardiorenalAgent.solve().
# It captures everything needed for downstream analysis: the optimal parameters,
# the LLM-generated explanations, convergence status, and timing metadata.

@dataclass
class AgentResult:
    """
    Structured result from the agentic optimization.

    Fields
    ------
    optimal_params : dict
        The 8 tunable disease parameters at convergence (or best found).
    parameter_policy : str
        LLM-generated summary of which parameters changed and why.
    mechanistic_explanation : str
        LLM-generated narrative of the V5→V7 disease trajectory.
    model_v7_output : dict
        The full ~113-variable model output at the optimal parameters.
    final_error : float
        Weighted normalized error at convergence (0 = perfect match).
    n_iterations : int
        Number of LLM conversation turns used.
    n_model_runs : int
        Number of CircAdapt model evaluations (run_model + sensitivity).
    elapsed_seconds : float
        Wall-clock time for the entire solve() call.
    converged : bool
        True if final_error <= convergence_threshold.
    error_history : list[float]
        Aggregate error after each compute_error call, for tracking progress.
    """
    optimal_params: Dict
    parameter_policy: str
    mechanistic_explanation: str
    model_v7_output: Dict
    final_error: float
    n_iterations: int
    n_model_runs: int
    elapsed_seconds: float
    converged: bool
    error_history: List[float] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════════════════════════
# (Paper Section 3.9 — "Prompt Engineering for Domain-Specific Optimization")
#
# The system prompt is the most critical piece of the agentic framework.
# It encodes three categories of knowledge:
#
# 1. ROLE AND TASK FRAMING: "You are a cardiorenal physiology expert..."
#    This primes the LLM to reason in clinical/physiological terms rather
#    than as a generic optimizer. This framing improves parameter selection
#    because the LLM can leverage its pre-training on medical literature.
#
# 2. MECHANISTIC RELATIONSHIPS: The six pathway descriptions (HFrEF, HFpEF,
#    CKD, inflammation, CRS, RAAS) give the LLM a causal graph of the model.
#    Without these, the LLM would have to discover cause-effect relationships
#    purely from sensitivity analysis, which is slow and error-prone.
#    Each relationship maps directly to equations in the model:
#    - HFrEF pathway: Sf_act_scale → Patch[Sf_act] → reduced contractility
#    - HFpEF pathway: k1_scale / diabetes_scale → Patch[k1] → stiffness
#    - CKD pathway: Kf_scale → glomerular Kf → reduced filtration
#    - Inflammation: InflammatoryState → multi-organ scaling factors
#    - CRS: bidirectional coupling between heart and kidney modules
#    - RAAS: HallowRenalModel RAAS feedback → efferent tone, Na reabsorption
#
# 3. STRATEGY GUIDANCE: The numbered steps (analyze phenotype → set initial
#    params → run model → use sensitivity → iterate → explain) encode the
#    optimization algorithm we want the LLM to follow. This is crucial because
#    without explicit strategy guidance, LLMs tend to either:
#    (a) Make random parameter guesses without systematic exploration, or
#    (b) Get stuck adjusting one parameter when multiple need co-adjustment.
#
# The {param_descriptions} placeholder is filled at runtime with the actual
# parameter ranges and defaults from config.TUNABLE_PARAMS, so the LLM always
# has accurate bounds information.

SYSTEM_PROMPT = """You are a cardiorenal physiology expert optimizing a coupled heart-kidney simulation model.

## Your Task
Given a patient's Visit 5 (baseline) clinical data and a predicted Visit 7 (6-year follow-up) target,
find the disease progression parameters that make the model reproduce the V7 target.

## Available Disease Parameters
{param_descriptions}

## Key Mechanistic Relationships
- **HFrEF pathway**: Low Sf_act_scale → reduced contractility → low LVEF, low CO → renal hypoperfusion → low GFR
- **HFpEF pathway**: High diabetes_scale or k1_scale → increased diastolic stiffness → preserved EF but elevated E/e', high LVEDP
- **CKD pathway**: Low Kf_scale → reduced nephrons → low GFR, high creatinine, elevated UACR
- **Inflammation**: High inflammation_scale → ↑SVR, ↑arterial stiffness, ↓Sf_act, ↓Kf (multi-organ effect)
- **Cardiorenal syndrome (CRS)**: Heart failure → renal hypoperfusion → fluid retention → worsens heart failure
- **RAAS**: High RAAS_gain → more reactive to MAP changes → affects efferent arteriole tone and CD reabsorption

## Strategy
1. First, analyze the V7 target to identify the dominant phenotype (HFrEF, HFpEF, CKD, CRS, etc.)
2. Set initial parameters based on the phenotype pattern
3. Run the model and compute error
4. Use sensitivity analysis on the worst-matching variables to identify which parameters to adjust
5. Iteratively refine parameters, focusing on the largest error contributors
6. When error is below threshold, provide your final explanation

## Important
- Start with the V5 baseline parameters if provided, then adjust toward V7
- Focus on the clinically important variables first (LVEF, eGFR, E/e', MAP, NTproBNP)
- Consider bidirectional cardiorenal feedback when adjusting parameters
- Explain your reasoning at each step
"""


def _build_system_prompt() -> str:
    """
    Build the complete system prompt by injecting parameter descriptions.

    Iterates over TUNABLE_PARAMS (from config.py) to produce formatted lines
    like: "- **Sf_act_scale** (range 0.2-1.0, default 1.0): Active fiber..."

    This ensures the LLM always sees the correct ranges and defaults,
    even if they are updated in config.py. The LLM uses these bounds to
    propose valid parameter values without needing explicit clamping.
    """
    param_lines = []
    for name, info in TUNABLE_PARAMS.items():
        param_lines.append(
            f"- **{name}** (range {info['range'][0]}-{info['range'][1]}, "
            f"default {info['default']}): {info['desc']}"
        )
    return SYSTEM_PROMPT.format(param_descriptions='\n'.join(param_lines))


def _build_initial_prompt(v5_data: Dict, v7_target: Dict, demographics: Dict) -> str:
    """
    Build the initial user prompt containing the patient's V5 data and V7 target.

    (Paper Section 3.9 — "Initial Prompt Construction")

    The prompt is structured with three sections:
    1. Demographics: age, sex, BSA, height — needed for body-size indexing
       and the CKD-EPI eGFR equation.
    2. V5 (baseline) key variables: a curated subset of the ~113 ARIC variables
       focused on the most clinically informative ones (LVEF, MAP, GFR, E/e',
       volumes, strain, biomarkers). We show only a subset to avoid prompt
       bloat — the LLM can always request the full set via tool calls.
    3. V7 (target) key variables: same subset, representing the predicted
       or actual 6-year follow-up state that the model must reproduce.

    The "Your Goal" section at the end provides explicit action instructions,
    directing the LLM to start with run_circadapt_model (not just reasoning).

    Parameters
    ----------
    v5_data : dict
        Visit 5 ARIC variable name → value mapping.
    v7_target : dict
        Visit 7 ARIC variable name → target value mapping.
    demographics : dict
        Patient demographics (age, sex, BSA, height_m).

    Returns
    -------
    str : Formatted markdown prompt ready for the LLM.
    """
    # Curated key variables: these are the most clinically important and
    # interpretable variables. We show them to the LLM upfront to enable
    # phenotype recognition (e.g., low LVEF → HFrEF, low GFR → CKD,
    # high E/e' → diastolic dysfunction / HFpEF).
    key_vars = [
        'LVEF_pct', 'MAP_mmHg', 'SBP_mmHg', 'DBP_mmHg', 'CO_Lmin',
        'GFR_mL_min', 'eGFR_mL_min_173m2', 'serum_creatinine_mg_dL',
        'E_e_prime_avg', 'LVEDV_mL', 'LVESV_mL', 'GLS_pct',
        'NTproBNP_pg_mL', 'UACR_mg_g', 'RBF_mL_min',
        'LV_mass_g', 'LA_volume_mL', 'PASP_mmHg',
    ]

    # Format V5 and V7 lines, only including variables that are present
    # in the respective data dicts. Missing variables are silently skipped.
    v5_lines = []
    v7_lines = []
    for vn in key_vars:
        if vn in v5_data:
            v5_lines.append(f"  {vn}: {v5_data[vn]:.2f}")
        if vn in v7_target:
            v7_lines.append(f"  {vn}: {v7_target[vn]:.2f}")

    # chr(10) is used for newline inside f-strings (f-strings cannot contain
    # backslash escapes directly).
    return f"""## Patient Demographics
- Age at V5: {demographics.get('age', 75):.0f}, Sex: {demographics.get('sex', 'M')}
- BSA: {demographics.get('BSA', 1.9):.2f} m², Height: {demographics.get('height_m', 1.75):.2f} m

## Visit 5 (Baseline) Key Variables
{chr(10).join(v5_lines)}

## Visit 7 (Target) Key Variables
{chr(10).join(v7_lines)}

## Your Goal
Find disease parameters that make the model output match the V7 target.
Start by analyzing the phenotype, then use the tools to run the model and optimize.
Begin with run_circadapt_model using your best initial guess, then compute_error to see how far off you are.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Nelder-Mead Fallback
# ═══════════════════════════════════════════════════════════════════════════════
# (Paper Section 3.9 — "Stagnation Detection and Numeric Fallback")
#
# If the LLM's iterative adjustment stagnates — i.e., the aggregate error
# stops decreasing over 3 consecutive compute_error calls — we fall back
# to a derivative-free Nelder-Mead (simplex) optimizer from scipy.
#
# Why Nelder-Mead?
# 1. It is derivative-free, so it works with our noisy, non-differentiable
#    model (the coupled ODE solver can have numerical noise).
# 2. It is relatively sample-efficient for low-dimensional problems (we have
#    only 8 parameters), typically converging in 50-100 evaluations.
# 3. It naturally handles the box constraints via our np.clip in the
#    objective function.
#
# The fallback starts from the LLM's best parameters so far (not from
# defaults), so it benefits from the LLM's initial phenotype-guided search.
# This hybrid approach typically converges faster than either method alone.

def _nelder_mead_fallback(
    v7_target: Dict,
    demographics: Dict,
    initial_params: Dict,
    max_evals: int = 100,
) -> Dict:
    """
    Scipy Nelder-Mead optimization as fallback when LLM doesn't converge.

    This function is called when the LLM-driven loop detects stagnation
    (3 consecutive compute_error calls with <0.001 change in aggregate error).
    It performs a local numeric optimization starting from the LLM's best
    parameter set, which typically needs only fine-tuning.

    Parameters
    ----------
    v7_target : dict
        The V7 clinical target to match.
    demographics : dict
        Patient demographics (age, sex, BSA, height_m).
    initial_params : dict
        The LLM's best parameter set so far — used as the starting point.
    max_evals : int
        Maximum number of model evaluations (default 100). Each evaluation
        takes ~0.5s, so 100 evals ≈ 50s worst case.

    Returns
    -------
    dict : Optimized parameter dict with all 8 tunable parameters.
    """
    # Lazy import to avoid loading scipy at module level (it's heavy and
    # only needed if the LLM stagnates, which happens in ~15% of cases).
    from scipy.optimize import minimize
    from agent_tools import run_circadapt_model, compute_error

    # Build ordered lists of parameter names, initial values, and bounds.
    # The ordering must be consistent between x0, bounds, and the objective.
    param_names = list(TUNABLE_PARAMS.keys())
    x0 = np.array([initial_params.get(p, TUNABLE_PARAMS[p]['default']) for p in param_names])
    bounds = [TUNABLE_PARAMS[p]['range'] for p in param_names]

    def objective(x):
        """
        Objective function: run the model with candidate parameters and
        return the aggregate weighted error vs. the V7 target.

        Parameters are clipped to their valid ranges before evaluation.
        If the model fails (returns an error dict), we return a large
        penalty (1e6) so Nelder-Mead avoids that region.
        """
        # Clip each parameter to its valid range (box constraint enforcement).
        params = {p: float(np.clip(x[i], *bounds[i])) for i, p in enumerate(param_names)}
        # Run the full coupled CircAdapt + Hallow model.
        result = run_circadapt_model(**params, **demographics)
        # If the model solver diverged or failed, return a large penalty.
        if 'error' in result:
            return 1e6
        # Compute the weighted normalized error against the V7 target.
        err = compute_error(result, v7_target)
        return err['aggregate_error']

    # Run Nelder-Mead with conservative convergence tolerances.
    # xatol=0.01: stop if the simplex vertices are within 0.01 of each other
    #   (parameter space is O(1) scale, so 0.01 is ~1% precision).
    # fatol=0.001: stop if the objective values differ by <0.001 across
    #   the simplex (error is typically 0.01-0.5, so 0.001 is fine-grained).
    result = minimize(
        objective, x0, method='Nelder-Mead',
        options={'maxfev': max_evals, 'xatol': 0.01, 'fatol': 0.001},
    )

    # Return the optimized parameters, clipped to valid ranges.
    return {p: float(np.clip(result.x[i], *bounds[i])) for i, p in enumerate(param_names)}


# ═══════════════════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════════════════
# (Paper Section 3.9 — "Agentic Inference Engine")
#
# The CardiorenalAgent class encapsulates the full LLM tool-calling loop.
# It uses LiteLLM (https://github.com/BerriAI/litellm) for multi-backend
# support, meaning the same code works with OpenAI (GPT-4o), Google (Gemini),
# Anthropic (Claude), or local models via Ollama — just change the model string.
#
# The loop follows the ReAct (Reason + Act) paradigm:
#   1. LLM reasons about the current state (text response)
#   2. LLM decides which tool to call (function call)
#   3. Tool executes and returns a JSON result
#   4. Result is appended to the conversation context
#   5. Repeat until convergence or max iterations
#
# Key design decisions:
# - tool_choice="auto": lets the LLM decide when to call tools vs. reason.
#   We tried "required" but found the LLM sometimes needs to reason about
#   sensitivity results before making the next tool call.
# - temperature=0.3: low but not zero. Zero temperature caused the LLM to
#   get stuck in repetitive parameter choices. 0.3 provides just enough
#   exploration while keeping responses focused.
# - max_tokens=4096: generous token budget per turn so the LLM can provide
#   detailed reasoning alongside tool calls.

class CardiorenalAgent:
    """
    Agentic LLM framework for cardiorenal parameter optimization.

    Uses LiteLLM for multi-backend support (OpenAI, Gemini, Ollama, Anthropic).
    Implements a tool-calling loop where the LLM iteratively adjusts disease
    parameters to match a clinical target, with Nelder-Mead fallback for
    stagnation.
    """

    def __init__(
        self,
        model: str = LLM_CONFIG['model'],
        max_iterations: int = LLM_CONFIG['max_iterations'],
        convergence_threshold: float = LLM_CONFIG['convergence_threshold'],
        temperature: float = LLM_CONFIG['temperature'],
        verbose: bool = True,
    ):
        """
        Initialize the agent.

        Parameters
        ----------
        model : str
            LiteLLM model string. Examples:
            - "gpt-4o" (OpenAI)
            - "gemini/gemini-2.5-pro" (Google)
            - "anthropic/claude-sonnet-4-20250514" (Anthropic)
            - "ollama/llama3.1" (local via Ollama)
        max_iterations : int
            Maximum number of LLM conversation turns. Each turn may include
            multiple tool calls. Default 15 (from LLM_CONFIG).
        convergence_threshold : float
            Aggregate weighted normalized error below which we declare
            convergence. Default 0.05 (5% normalized error).
        temperature : float
            LLM sampling temperature. Default 0.3 (low creativity, high
            precision — appropriate for optimization, not creative writing).
        verbose : bool
            If True, print progress messages to stdout.
        """
        self.model = model
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.temperature = temperature
        self.verbose = verbose

    def solve(
        self,
        v5_data: Dict,
        v7_target: Dict,
        demographics: Dict,
    ) -> AgentResult:
        """
        Run the agentic optimization loop.

        (Paper Section 3.9 — "Optimization Loop")

        This is the main entry point. It:
        1. Constructs the system + initial user prompt
        2. Enters the LLM tool-calling loop
        3. Tracks error history and best parameters
        4. Detects convergence or stagnation
        5. On convergence, asks the LLM for a final explanation
        6. On stagnation, falls back to Nelder-Mead
        7. Returns a structured AgentResult

        Parameters
        ----------
        v5_data : dict
            Visit 5 ARIC variables (from NN input or model evaluation).
            These provide the baseline clinical state.
        v7_target : dict
            Visit 7 ARIC variables (from NN prediction or real data).
            This is the target the agent must match.
        demographics : dict
            Patient demographics: age, sex, BSA, height_m.

        Returns
        -------
        AgentResult
            Structured result with optimal params, policy, explanation, etc.
        """
        # LiteLLM is imported inside the method (not at module level) because:
        # 1. It avoids import errors when litellm is not installed (e.g.,
        #    during testing of other modules).
        # 2. It reduces startup time when this module is imported but not used.
        import litellm

        # Start the wall-clock timer for elapsed_seconds in the result.
        t0 = time.time()

        # ─── Build the initial conversation ──────────────────────────────
        # The conversation is a list of message dicts following the OpenAI
        # chat completion format. We start with:
        # - system message: domain knowledge + strategy (see SYSTEM_PROMPT)
        # - user message: patient-specific V5/V7 data (see _build_initial_prompt)
        messages = [
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": _build_initial_prompt(v5_data, v7_target, demographics)},
        ]

        # ─── Initialize tracking variables ───────────────────────────────
        # best_error: the lowest aggregate error seen so far (starts at inf).
        best_error = float('inf')
        # best_params: the disease parameters corresponding to best_error.
        # Initialized to defaults (healthy patient) — the LLM will adjust.
        best_params = {p: TUNABLE_PARAMS[p]['default'] for p in TUNABLE_PARAMS}
        # best_output: the full model output dict at best_params.
        best_output = {}
        # error_history: list of aggregate errors from each compute_error call,
        # used for stagnation detection and convergence visualization.
        error_history = []
        # n_model_runs: count of CircAdapt model evaluations (for cost tracking).
        n_model_runs = 0
        # converged: set to True if we reach the convergence threshold.
        converged = False

        # ─── Main LLM tool-calling loop ──────────────────────────────────
        # Each iteration = one LLM API call, which may produce:
        # - A text response (reasoning / explanation)
        # - One or more tool calls (model runs, error computation, etc.)
        # - Both (reasoning + tool calls in the same turn)
        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"  Agent iteration {iteration}/{self.max_iterations}...")

            # ─── LLM API call ────────────────────────────────────────────
            # We pass the full conversation history + tool schemas.
            # tool_choice="auto" means the LLM decides whether to call a tool
            # or just provide text. This is important because sometimes the
            # LLM needs to reason about results before making the next call.
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,       # The 4 tools from agent_tools.py
                    tool_choice="auto",       # LLM decides when to use tools
                    temperature=self.temperature,
                    max_tokens=LLM_CONFIG.get('max_tokens', 4096),
                )
            except Exception as e:
                # API errors (rate limits, network issues, auth failures)
                # cause the loop to exit early. The result will show the
                # best parameters found so far.
                if self.verbose:
                    print(f"  LLM API error: {e}")
                break

            # Extract the assistant's message from the API response.
            msg = response.choices[0].message
            # Append the assistant message to the conversation history.
            # model_dump() serializes it to a dict compatible with the next
            # API call. This maintains the full conversation context so the
            # LLM can reference previous reasoning and results.
            messages.append(msg.model_dump())

            # ─── Process tool calls ──────────────────────────────────────
            # If the LLM chose to call one or more tools, execute each one
            # and append the results to the conversation as "tool" messages.
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # Extract the function name and arguments from the tool call.
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        # If the LLM produces malformed JSON arguments, use
                        # empty args. The tool will use its defaults.
                        fn_args = {}

                    # Verbose logging: show which tool is being called with
                    # a preview of the first 3 arguments.
                    if self.verbose:
                        print(f"    Tool: {fn_name}({', '.join(f'{k}={v}' for k, v in list(fn_args.items())[:3])}...)")

                    # ─── Execute the tool ─────────────────────────────────
                    # execute_tool() dispatches to the actual function in
                    # agent_tools.py and returns a JSON string.
                    tool_result = execute_tool(fn_name, fn_args)
                    # Count model runs for cost tracking. Both run_circadapt_model
                    # and get_sensitivity invoke the CircAdapt model (sensitivity
                    # runs it twice: param+delta and param-delta).
                    n_model_runs += 1 if fn_name in ('run_circadapt_model', 'get_sensitivity') else 0

                    # ─── Track error from compute_error calls ─────────────
                    # When the LLM calls compute_error, we extract the aggregate
                    # error and update our best-so-far tracking. This is how we
                    # know when to declare convergence.
                    if fn_name == 'compute_error':
                        try:
                            err_data = json.loads(tool_result)
                            agg_err = err_data.get('aggregate_error', float('inf'))
                            # Append to error_history for stagnation detection.
                            error_history.append(agg_err)
                            if agg_err < best_error:
                                best_error = agg_err
                                # Try to extract the parameters that produced
                                # this error. The LLM typically passes the
                                # model_output dict (which contains params_used)
                                # as the first argument to compute_error.
                                if 'model_output' in fn_args:
                                    mo = fn_args['model_output']
                                    if 'params_used' in mo:
                                        best_params = mo['params_used']
                                    best_output = mo
                        except (json.JSONDecodeError, KeyError):
                            # Malformed error response — skip tracking.
                            pass

                    # ─── Track params from run_circadapt_model calls ──────
                    # Every time the model is run, we store the output and
                    # parameters as the current best candidate. This ensures
                    # we always have a valid model output even if the LLM
                    # never explicitly calls compute_error (rare but possible).
                    if fn_name == 'run_circadapt_model':
                        try:
                            run_data = json.loads(tool_result)
                            if 'params_used' in run_data:
                                best_output = run_data
                                best_params = run_data['params_used']
                        except (json.JSONDecodeError, KeyError):
                            pass

                    # ─── Truncate long tool results ───────────────────────
                    # Tool results (especially run_circadapt_model which
                    # returns ~113 variables) can be very long. We truncate
                    # to 4000 chars to avoid exceeding the LLM's context
                    # window. The LLM can always re-request specific data.
                    if len(tool_result) > 4000:
                        tool_result_msg = tool_result[:4000] + '... (truncated)'
                    else:
                        tool_result_msg = tool_result

                    # Append the tool result to the conversation. The
                    # tool_call_id links this result back to the specific
                    # tool call it responds to (required by the API).
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result_msg,
                    })

            # ─── Check convergence ───────────────────────────────────────
            # If the best error is below the threshold (default 5%), we
            # declare convergence and ask the LLM for a final explanation.
            if best_error <= self.convergence_threshold:
                if self.verbose:
                    print(f"  Converged! Error={best_error:.4f}")
                converged = True

                # Ask the LLM to produce the two key outputs:
                # 1. Parameter policy: which params changed from baseline and why
                # 2. Mechanistic explanation: what disease processes explain V5→V7
                # We use a separate prompt (not a tool call) because this is a
                # free-form text generation task, not a structured computation.
                messages.append({
                    "role": "user",
                    "content": (
                        f"The model has converged with error={best_error:.4f}. "
                        f"Please provide:\n"
                        f"1. A PARAMETER POLICY: which parameters changed from baseline and by how much\n"
                        f"2. A MECHANISTIC EXPLANATION: what disease processes explain the V5→V7 transition\n"
                        f"Format your response with clear sections."
                    ),
                })

                # Generate the explanation with lower temperature (0.3) for
                # consistency, and fewer tokens (2048) since we only need text.
                try:
                    final_response = litellm.completion(
                        model=self.model, messages=messages,
                        temperature=0.3, max_tokens=2048,
                    )
                    explanation_text = final_response.choices[0].message.content or ""
                except Exception:
                    explanation_text = "Converged but failed to generate explanation."

                # Parse the free-form text into separate policy and explanation
                # strings using keyword-based splitting.
                policy, explanation = _parse_explanation(explanation_text)
                break

            # ─── Check for stagnation ────────────────────────────────────
            # Stagnation detection: if we have at least 3 error measurements
            # and at least 10 iterations have passed, check if the error has
            # plateaued (max - min of last 3 errors < 0.001).
            #
            # The iteration >= 10 guard ensures we give the LLM enough time
            # to explore before declaring stagnation. Early iterations often
            # have large error jumps as the LLM experiments with different
            # phenotype hypotheses.
            if len(error_history) >= 3 and iteration >= 10:
                recent = error_history[-3:]
                if max(recent) - min(recent) < 0.001:
                    if self.verbose:
                        print(f"  Stagnated at error={best_error:.4f}, falling back to Nelder-Mead...")
                    # Run Nelder-Mead starting from the LLM's best params.
                    best_params = _nelder_mead_fallback(
                        v7_target, demographics, best_params, max_evals=80
                    )
                    # Run a final model evaluation with the Nelder-Mead result
                    # to get the full output and updated error.
                    from agent_tools import run_circadapt_model, compute_error
                    best_output = run_circadapt_model(**best_params, **demographics)
                    err = compute_error(best_output, v7_target)
                    best_error = err['aggregate_error']
                    converged = best_error <= self.convergence_threshold
                    break

            # ─── Handle pure text responses (no tool calls) ──────────────
            # If the LLM responded with text only (no tool calls), check if
            # it's providing the final explanation. This happens when the LLM
            # decides on its own that it's done (without explicit convergence).
            if not msg.tool_calls:
                content = msg.content or ""
                # Look for the section headers we asked for. If found, the LLM
                # is providing its final answer.
                if 'parameter policy' in content.lower() or 'mechanistic explanation' in content.lower():
                    policy, explanation = _parse_explanation(content)
                    break
        else:
            # This else clause belongs to the for loop — it executes if the
            # loop completes without a break (i.e., max iterations reached
            # without convergence or stagnation).
            if self.verbose:
                print(f"  Max iterations reached. Best error={best_error:.4f}")
            policy = f"Optimization did not converge (error={best_error:.4f})"
            explanation = "Max iterations reached without convergence."

        # Record total wall-clock time.
        elapsed = time.time() - t0

        # ─── Ensure policy and explanation are defined ────────────────────
        # These variables might not be defined if the loop exited via the
        # API error break (line ~252). The dir() check is a Python idiom
        # for testing if a local variable exists.
        if 'policy' not in dir():
            policy = f"Parameters: {json.dumps(best_params, indent=2)}"
        if 'explanation' not in dir():
            explanation = "No mechanistic explanation generated."

        # ─── Return structured result ────────────────────────────────────
        return AgentResult(
            optimal_params=best_params,
            parameter_policy=policy,
            mechanistic_explanation=explanation,
            model_v7_output=best_output,
            final_error=best_error,
            # min() handles the case where we broke out before completing
            # the last iteration increment.
            n_iterations=min(iteration, self.max_iterations) if 'iteration' in dir() else 0,
            n_model_runs=n_model_runs,
            elapsed_seconds=elapsed,
            converged=converged,
            error_history=error_history,
        )


def _parse_explanation(text: str):
    """
    Parse LLM free-form text into separate parameter policy and mechanistic
    explanation strings.

    The LLM is instructed to produce two sections with headers:
    - "Parameter Policy" or "PARAMETER POLICY"
    - "Mechanistic Explanation" or "MECHANISTIC EXPLANATION"

    This function uses case-insensitive keyword search to find these sections
    and split the text. It handles four cases:
    1. Both sections found (policy before explanation or vice versa)
    2. Only policy section found
    3. Only explanation section found
    4. Neither found (use the entire text for both)

    Parameters
    ----------
    text : str
        Raw LLM response text.

    Returns
    -------
    tuple[str, str] : (parameter_policy, mechanistic_explanation)
    """
    text = text.strip()

    policy = ""
    explanation = ""

    # Case-insensitive search for section headers.
    lower = text.lower()
    policy_idx = lower.find('parameter policy')
    mech_idx = lower.find('mechanistic explanation')

    if policy_idx >= 0 and mech_idx >= 0:
        # Both sections found — split at the boundary between them.
        if policy_idx < mech_idx:
            # Policy comes first, then explanation.
            policy = text[policy_idx:mech_idx].strip()
            explanation = text[mech_idx:].strip()
        else:
            # Explanation comes first, then policy.
            explanation = text[mech_idx:policy_idx].strip()
            policy = text[policy_idx:].strip()
    elif policy_idx >= 0:
        # Only policy section found. Everything before it is treated as
        # explanation (or empty).
        policy = text[policy_idx:].strip()
        explanation = text[:policy_idx].strip()
    elif mech_idx >= 0:
        # Only explanation section found. Everything before it is treated
        # as policy (or empty).
        explanation = text[mech_idx:].strip()
        policy = text[:mech_idx].strip()
    else:
        # No clear sections detected. This can happen if the LLM ignores
        # the formatting instructions. Use the full text for both fields.
        policy = text
        explanation = text

    return policy, explanation


# ═══════════════════════════════════════════════════════════════════════════════
# CLI for testing
# ═══════════════════════════════════════════════════════════════════════════════
# This CLI provides a quick way to test the agentic framework with a synthetic
# patient. It generates V5 and V7 data from known disease parameters (ground
# truth), then runs the agent to recover those parameters. This is useful for:
# 1. Verifying the agent can converge on a known answer
# 2. Testing different LLM backends (--model flag)
# 3. Benchmarking convergence speed (--max_iter flag)

def main():
    """Quick test with a synthetic patient."""
    import argparse
    parser = argparse.ArgumentParser(description='Test agentic cardiorenal optimizer')
    parser.add_argument('--model', type=str, default=LLM_CONFIG['model'],
                        help='LiteLLM model string (e.g., gpt-4o, ollama/llama3.1)')
    parser.add_argument('--max_iter', type=int, default=LLM_CONFIG['max_iterations'])
    args = parser.parse_args()

    # Import the model evaluation function to generate ground-truth data.
    from synthetic_cohort import evaluate_patient_state

    # Define a test patient with known disease parameters.
    # V5 = mild HFrEF (Sf_act_scale=0.85) + early CKD (Kf_scale=0.9)
    # V7 = moderate HFrEF (Sf_act_scale=0.65) + moderate CKD (Kf_scale=0.6)
    # The agent should recover the V7 parameters from the clinical targets.
    v5_params = {
        'Sf_act_scale': 0.85, 'Kf_scale': 0.9, 'inflammation_scale': 0.1,
        'diabetes_scale': 0.0, 'k1_scale': 1.0, 'RAAS_gain': 1.5,
        'TGF_gain': 2.0, 'na_intake': 150.0,
    }
    v7_params = {
        'Sf_act_scale': 0.65, 'Kf_scale': 0.6, 'inflammation_scale': 0.25,
        'diabetes_scale': 0.1, 'k1_scale': 1.2, 'RAAS_gain': 1.8,
        'TGF_gain': 2.0, 'na_intake': 160.0,
    }
    # Demographics are constant between V5 and V7 except age (+6 years).
    demographics = {'age': 72, 'sex': 'M', 'BSA': 1.9, 'height_m': 1.75}

    # Generate the clinical variable vectors from the ground-truth params.
    print("Generating V5 and V7 data...")
    v5_data = evaluate_patient_state(v5_params, demographics)
    # V7 demographics have age +6 years (ARIC visit spacing).
    v7_target = evaluate_patient_state(v7_params, {'age': 78, 'sex': 'M', 'BSA': 1.9, 'height_m': 1.75})

    if v5_data is None or v7_target is None:
        print("ERROR: Model evaluation failed")
        return

    # Print key variables for quick visual check of the test case.
    print(f"V5: LVEF={v5_data['LVEF_pct']:.1f}%, GFR={v5_data['GFR_mL_min']:.1f}")
    print(f"V7 target: LVEF={v7_target['LVEF_pct']:.1f}%, GFR={v7_target['GFR_mL_min']:.1f}")

    # Run the agent and print results.
    agent = CardiorenalAgent(model=args.model, max_iterations=args.max_iter)
    result = agent.solve(v5_data, v7_target, demographics)

    print(f"\n{'='*60}")
    print(f"  Result: converged={result.converged}, error={result.final_error:.4f}")
    print(f"  Iterations: {result.n_iterations}, Model runs: {result.n_model_runs}")
    print(f"  Time: {result.elapsed_seconds:.1f}s")
    print(f"{'='*60}")
    print(f"\nParameter Policy:\n{result.parameter_policy[:500]}")
    print(f"\nMechanistic Explanation:\n{result.mechanistic_explanation[:500]}")


if __name__ == '__main__':
    main()
