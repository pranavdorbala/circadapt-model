#!/usr/bin/env python3
"""
End-to-End Pipeline: V5 → NN Prediction → Agent Optimization → Result
=======================================================================
(Paper Section 3.11 — "End-to-End Pipeline")

This module implements the two-stage inference pipeline that connects the
neural network predictor (Stage 1) with the agentic LLM optimizer (Stage 2)
to produce a fully mechanistic explanation of a patient's V5→V7 disease
trajectory.

Pipeline architecture (Section 3.11, Figure Y):

    ┌─────────────────────────────────────────────────────────────────────┐
    │                         Stage 1: NN Prediction                      │
    │                                                                     │
    │  V5 clinical data ──► Trained NN (V5→V7 model) ──► V7 prediction   │
    │  (~113 variables)      (~1 ms inference)          (~113 variables)  │
    │                                                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │                      Stage 2: Agent Optimization                    │
    │                                                                     │
    │  V5 data + V7 target ──► LLM Agent Loop ──► Optimal parameters     │
    │  + demographics          (4 tools, ~15     + parameter policy       │
    │                           iterations)      + mechanistic explanation│
    │                          (~30-60 seconds)                           │
    └─────────────────────────────────────────────────────────────────────┘

Why two stages?
1. The NN is fast (~1ms) but opaque: it predicts WHAT will happen at V7
   but not WHY. It cannot tell us which disease processes drive the change.
2. The agent is slow (~30-60s) but interpretable: it discovers which
   mechanistic parameters (contractility loss, nephron loss, inflammation,
   etc.) reproduce the predicted V7 state, providing causal explanation.
3. Together, they provide both speed (for screening large cohorts) and
   interpretability (for individual patient analysis).

The pipeline also supports:
- Bypassing the NN (providing a V7 target directly)
- Batch processing with thread-level parallelism
- Multiple input modes (JSON, CSV, model params, demo mode)

Usage:
    # Single patient (JSON)
    python pipeline.py --v5 '{"LVEF_pct": 58, "GFR_mL_min": 90, ...}'

    # Single patient from model params
    python pipeline.py --params '{"Sf_act_scale": 0.85, "Kf_scale": 0.9}'

    # Batch from CSV
    python pipeline.py --v5_csv patients.csv --model gpt-4o --workers 4
"""

# ─── Standard library imports ────────────────────────────────────────────────
import os
import sys
import json
import argparse
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

# Ensure the project root is on sys.path for sibling module imports.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Internal project imports ────────────────────────────────────────────────
# TUNABLE_PARAMS: the 8 disease parameters with ranges/defaults (paper Sec 3.4)
# NUMERIC_VAR_NAMES: sorted list of ~113 numeric ARIC variable names
# LLM_CONFIG: agent hyperparameters (model name, max iterations, threshold)
# NN_DEFAULTS: neural network architecture and training hyperparameters
from config import TUNABLE_PARAMS, NUMERIC_VAR_NAMES, LLM_CONFIG, NN_DEFAULTS


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
# (Paper Section 3.11 — "End-to-End Pipeline Class")
#
# The CardiorenalPipeline class orchestrates both stages and provides a
# unified interface for single-patient and batch processing.

class CardiorenalPipeline:
    """
    End-to-end pipeline: V5 input → NN prediction → Agent optimization → result.

    (Paper Section 3.11)

    This class encapsulates the two-stage inference process:
        Stage 1 (NN): V5 clinical data → predicted V7 clinical state (~1ms)
        Stage 2 (Agent): V5 + V7 target → disease parameters + explanation (~30-60s)

    The pipeline is designed to be flexible:
    - If a trained NN model exists, it predicts V7 from V5 automatically.
    - If no NN is available, the user must provide an explicit V7 target.
    - Batch processing uses thread-level parallelism (not multiprocessing)
      because the bottleneck is LLM API calls (I/O bound, not CPU bound).
    """

    def __init__(
        self,
        nn_model_path: str = 'models/v5_to_v7_best.pt',
        llm_model: str = LLM_CONFIG['model'],
        max_iterations: int = LLM_CONFIG['max_iterations'],
        verbose: bool = True,
    ):
        """
        Initialize the pipeline by loading the NN model and creating the agent.

        Parameters
        ----------
        nn_model_path : str
            Path to the trained NN model checkpoint (.pt file). Can be
            relative (to the project root) or absolute. The checkpoint
            contains the model weights, architecture config, and the
            ordered list of variable names (nn_var_names).
            (Paper Section 3.7 — "Neural Network Training")
        llm_model : str
            LiteLLM model string for the agent. Examples:
            - "gpt-4o" (OpenAI, best performance)
            - "gemini/gemini-2.5-pro" (Google, good alternative)
            - "ollama/llama3.1" (local, no API costs)
        max_iterations : int
            Maximum LLM conversation turns for the agent (default 15).
        verbose : bool
            If True, print progress messages.
        """
        # Resolve the NN model path. If relative, prepend the project root
        # directory so the pipeline works from any working directory.
        base = os.path.dirname(os.path.abspath(__file__))
        nn_path = os.path.join(base, nn_model_path) if not os.path.isabs(nn_model_path) else nn_model_path

        # ─── Load the trained NN model (Stage 1) ────────────────────────
        # The NN is a residual MLP trained on synthetic cohort data
        # (paper Section 3.7). It maps V5 clinical variables → V7 clinical
        # variables. The checkpoint stores:
        # - 'model_state_dict': PyTorch weights
        # - 'var_names': ordered list of variable names (must match model I/O)
        # - 'hidden_dim', 'n_blocks': architecture config for reconstruction
        if os.path.exists(nn_path):
            # Lazy import train_nn to avoid importing PyTorch at module level
            # (allows the pipeline to work in environments without PyTorch
            # when only the agent stage is needed).
            from train_nn import load_trained_model, predict
            self.nn_model, self.nn_ckpt = load_trained_model(nn_path)
            # The variable names from training determine the input/output
            # ordering. These must match the NUMERIC_VAR_NAMES in config.py.
            self.nn_var_names = self.nn_ckpt['var_names']
            # Store the predict function for use in predict_v7().
            self.predict_fn = predict
            self.has_nn = True
            if verbose:
                print(f"Loaded NN from {nn_path} ({len(self.nn_var_names)} features)")
        else:
            # No NN model found — the pipeline can still work if the user
            # provides an explicit V7 target (bypassing Stage 1).
            self.has_nn = False
            if verbose:
                print(f"NN model not found at {nn_path}. "
                      f"Pipeline will work without NN (requires explicit V7 target).")

        # ─── Create the LLM agent (Stage 2) ─────────────────────────────
        # The agent is always initialized, even if the NN is missing,
        # because the user can provide an explicit V7 target.
        from agent_loop import CardiorenalAgent
        self.agent = CardiorenalAgent(
            model=llm_model,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        self.verbose = verbose

    def predict_v7(self, v5_data: Dict) -> Dict:
        """
        Use the trained NN to predict V7 clinical state from V5 data.

        (Paper Section 3.11 — Stage 1)

        This is the fast inference stage (~1ms). The NN takes a V5 clinical
        vector and outputs a predicted V7 clinical vector. Both vectors have
        the same ~113 ARIC variables.

        Parameters
        ----------
        v5_data : dict
            V5 ARIC variable name → value mapping. Missing variables are
            filled with 0.0 (which may reduce prediction quality).

        Returns
        -------
        dict : Predicted V7 ARIC variable name → value mapping.

        Raises
        ------
        RuntimeError : If no NN model is loaded.
        """
        if not self.has_nn:
            raise RuntimeError("No NN model loaded. Train one first with train_nn.py.")

        # Convert the V5 dict to a numpy array in the correct variable order.
        # Variables not present in v5_data default to 0.0.
        v5_vec = np.array(
            [float(v5_data.get(k, 0.0)) for k in self.nn_var_names],
            dtype=np.float32,
        )
        # Run NN inference (forward pass through the residual MLP).
        v7_vec = self.predict_fn(self.nn_model, v5_vec)

        # Convert back to a named dict.
        return {k: float(v7_vec[i]) for i, k in enumerate(self.nn_var_names)}

    def predict_and_explain(
        self,
        v5_data: Dict,
        demographics: Dict,
        v7_target: Optional[Dict] = None,
    ) -> Dict:
        """
        Full pipeline: V5 → NN prediction → Agent optimization → result.

        (Paper Section 3.11 — "Full Pipeline Execution")

        This is the main entry point for single-patient analysis. It runs
        both stages sequentially and returns a comprehensive result dict.

        Parameters
        ----------
        v5_data : dict
            V5 ARIC variables (the baseline clinical state).
        demographics : dict
            Patient demographics: age, sex, BSA, height_m.
        v7_target : dict, optional
            If provided, skip the NN prediction (Stage 1) and use this as
            the V7 target directly. This is useful when:
            - You have actual V7 data (validation mode)
            - The NN is not available
            - You want to test specific V7 targets

        Returns
        -------
        dict with keys:
            'v5_input' : dict
                The input V5 data (echoed for traceability).
            'v7_nn_prediction' : dict or None
                The NN's V7 prediction (None if v7_target was provided).
            'v7_target_used' : dict
                The V7 target actually used by the agent (either NN
                prediction or the user-provided target).
            'optimal_params' : dict
                The 8 optimal disease parameters found by the agent.
            'parameter_policy' : str
                LLM-generated summary of parameter changes.
            'mechanistic_explanation' : str
                LLM-generated disease trajectory narrative.
            'model_v7_output' : dict
                The model's ~113 output variables at optimal parameters.
            'prediction_error' : float
                Final aggregate error (0 = perfect match).
            'agent_converged' : bool
                Whether the agent reached the convergence threshold.
            'error_history' : list[float]
                Error at each compute_error call during optimization.
            'timing' : dict
                Timing breakdown: nn_seconds, agent_seconds, total_seconds,
                n_model_runs, n_iterations.
        """
        # Start the overall timer.
        t0 = time.time()

        # ─── Stage 1: NN Prediction ──────────────────────────────────────
        # If no explicit V7 target is provided, use the NN to predict V7
        # from V5. If a target is provided, skip the NN entirely.
        if v7_target is None:
            # Run the NN forward pass (~1ms).
            v7_nn = self.predict_v7(v5_data)
            # Use the NN prediction as the target for Stage 2.
            v7_target_used = v7_nn
        else:
            # User provided an explicit V7 target — no NN needed.
            v7_nn = None
            v7_target_used = v7_target

        # Record NN inference time (should be ~0.001 seconds).
        t_nn = time.time() - t0

        # ─── Stage 2: Agent Optimization ─────────────────────────────────
        # Run the full LLM agentic loop to find disease parameters that
        # reproduce the V7 target. This is the expensive stage (~30-60s),
        # dominated by LLM API calls and model evaluations.
        t_agent_start = time.time()
        agent_result = self.agent.solve(v5_data, v7_target_used, demographics)
        t_agent = time.time() - t_agent_start

        # ─── Assemble and return the comprehensive result dict ───────────
        return {
            'v5_input': v5_data,
            'v7_nn_prediction': v7_nn,
            'v7_target_used': v7_target_used,
            'optimal_params': agent_result.optimal_params,
            'parameter_policy': agent_result.parameter_policy,
            'mechanistic_explanation': agent_result.mechanistic_explanation,
            'model_v7_output': agent_result.model_v7_output,
            'prediction_error': agent_result.final_error,
            'agent_converged': agent_result.converged,
            'error_history': agent_result.error_history,
            'timing': {
                'nn_seconds': round(t_nn, 3),
                'agent_seconds': round(t_agent, 1),
                'total_seconds': round(time.time() - t0, 1),
                'n_model_runs': agent_result.n_model_runs,
                'n_iterations': agent_result.n_iterations,
            },
        }

    def batch_predict(
        self,
        v5_batch: List[Dict],
        demographics_batch: List[Dict],
        v7_targets: Optional[List[Dict]] = None,
        n_workers: int = 4,
    ) -> List[Dict]:
        """
        Batch processing: run the pipeline on multiple patients.

        (Paper Section 3.11 — "Batch Processing and Parallelism")

        For batch processing, the NN stage is inherently fast (~1ms per
        patient) so it does not need parallelism. The agent stage (~30-60s
        per patient) is the bottleneck and is parallelized using threads.

        Why threads instead of processes?
        - The agent's bottleneck is LLM API calls (I/O-bound, not CPU-bound).
        - Thread-level parallelism avoids the overhead of spawning processes
          and serializing data across process boundaries.
        - Python's GIL is released during I/O operations (network calls to
          the LLM API), so threads provide real parallelism here.
        - CircAdapt model evaluations are also relatively fast (~0.5s) and
          the GIL is released during numpy operations.

        Parameters
        ----------
        v5_batch : list[dict]
            List of V5 ARIC variable dicts, one per patient.
        demographics_batch : list[dict]
            List of demographics dicts, one per patient.
        v7_targets : list[dict] or None
            Optional list of V7 targets. If None, the NN predicts V7 for
            each patient.
        n_workers : int
            Number of threads for parallel agent execution (default 4).
            Setting this too high may hit LLM API rate limits.

        Returns
        -------
        list[dict] : List of pipeline result dicts, one per patient.
        """
        n = len(v5_batch)
        # If no V7 targets provided, use None for each patient (NN will predict).
        if v7_targets is None:
            v7_targets = [None] * n

        def process_one(i):
            """Process a single patient through the full pipeline."""
            return self.predict_and_explain(
                v5_batch[i], demographics_batch[i], v7_targets[i]
            )

        if n_workers <= 1:
            # Sequential processing (useful for debugging — stack traces
            # are clearer without threading).
            results = [process_one(i) for i in range(n)]
        else:
            # Parallel processing with ThreadPoolExecutor.
            # pool.map() preserves ordering: results[i] corresponds to
            # patient i, regardless of thread execution order.
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(process_one, range(n)))

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI (Command-Line Interface)
# ═══════════════════════════════════════════════════════════════════════════════
# (Paper Section 3.11 — "CLI Modes")
#
# The CLI supports four input modes, each targeting a different use case:
#
# 1. --params mode: Provide disease parameters as JSON. The model generates
#    V5 data from these params, then the pipeline runs normally. Useful for
#    testing with known ground-truth parameters.
#
# 2. --v5 mode: Provide V5 clinical data as JSON directly. Useful when you
#    have real patient data that you want to analyze.
#
# 3. --v5_csv mode: Provide a CSV file with V5 data for multiple patients.
#    Each row is a patient; columns are ARIC variable names. Batch processing
#    with optional thread parallelism (--workers flag).
#
# 4. Demo mode (no args): Generates a synthetic patient with known V5 and V7
#    parameters, then runs the pipeline to verify end-to-end functionality.
#    This is the "smoke test" for the entire system.

def main():
    """CLI entry point: parse arguments and dispatch to the appropriate mode."""
    parser = argparse.ArgumentParser(description='Cardiorenal V5→V7 Pipeline')

    # ─── CLI argument definitions ────────────────────────────────────────
    parser.add_argument('--v5', type=str, default=None,
                        help='V5 data as JSON string')
    parser.add_argument('--params', type=str, default=None,
                        help='Disease params as JSON (will run model to get V5 data)')
    parser.add_argument('--v7_target', type=str, default=None,
                        help='Optional V7 target as JSON (skips NN prediction)')
    parser.add_argument('--v5_csv', type=str, default=None,
                        help='CSV file with V5 data for batch processing')
    parser.add_argument('--model', type=str, default=LLM_CONFIG['model'],
                        help='LiteLLM model string')
    parser.add_argument('--nn_path', type=str, default='models/v5_to_v7_best.pt')
    parser.add_argument('--max_iter', type=int, default=LLM_CONFIG['max_iterations'])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    args = parser.parse_args()

    # ─── Mode 1: Single patient from disease parameters ──────────────────
    # Use case: Testing with known ground-truth parameters.
    # The model is run with the given params to generate V5 data, then the
    # pipeline discovers what those params should be (round-trip test).
    if args.params:
        from synthetic_cohort import evaluate_patient_state
        params = json.loads(args.params)
        # Default demographics for param-based mode. In a real application,
        # these would be provided alongside the params.
        demographics = {'age': 72, 'sex': 'M', 'BSA': 1.9, 'height_m': 1.75}
        print("Running CircAdapt model with given params to generate V5 data...")
        v5_data = evaluate_patient_state(params, demographics)
        if v5_data is None:
            print("ERROR: Model evaluation failed")
            return

        # Initialize the pipeline (loads NN + creates agent).
        pipeline = CardiorenalPipeline(
            nn_model_path=args.nn_path, llm_model=args.model,
            max_iterations=args.max_iter,
        )

        # If a V7 target is provided, parse it; otherwise let the NN predict.
        v7_target = json.loads(args.v7_target) if args.v7_target else None
        result = pipeline.predict_and_explain(v5_data, demographics, v7_target)
        _print_result(result)

        # Optionally save the full result to a JSON file.
        if args.output:
            _save_result(result, args.output)
        return

    # ─── Mode 2: Single patient from V5 JSON ────────────────────────────
    # Use case: Analyzing a real patient whose V5 clinical data is available.
    if args.v5:
        v5_data = json.loads(args.v5)
        # Default demographics (would ideally be extracted from the JSON).
        demographics = {'age': 72, 'sex': 'M', 'BSA': 1.9, 'height_m': 1.75}

        pipeline = CardiorenalPipeline(
            nn_model_path=args.nn_path, llm_model=args.model,
            max_iterations=args.max_iter,
        )

        v7_target = json.loads(args.v7_target) if args.v7_target else None
        result = pipeline.predict_and_explain(v5_data, demographics, v7_target)
        _print_result(result)

        if args.output:
            _save_result(result, args.output)
        return

    # ─── Mode 3: Batch processing from CSV ───────────────────────────────
    # Use case: Analyzing a cohort of patients from a CSV file.
    # Each row in the CSV is a patient, with columns matching ARIC variable
    # names plus optional demographics columns (age, sex, BSA, height_m).
    if args.v5_csv:
        import pandas as pd
        df = pd.read_csv(args.v5_csv)
        print(f"Loaded {len(df)} patients from {args.v5_csv}")

        # Convert each row to a dict of V5 clinical variables.
        v5_batch = [dict(row) for _, row in df.iterrows()]
        # Extract demographics from the CSV if available; otherwise use defaults.
        demographics_batch = [
            {'age': row.get('age', 72), 'sex': row.get('sex', 'M'),
             'BSA': row.get('BSA', 1.9), 'height_m': row.get('height_m', 1.75)}
            for _, row in df.iterrows()
        ]

        pipeline = CardiorenalPipeline(
            nn_model_path=args.nn_path, llm_model=args.model,
            max_iterations=args.max_iter,
        )

        # Run batch processing with optional thread parallelism.
        # The --workers flag controls the number of concurrent agent threads.
        results = pipeline.batch_predict(
            v5_batch, demographics_batch, n_workers=args.workers
        )

        # Print brief summaries for each patient.
        for i, r in enumerate(results):
            print(f"\nPatient {i+1}:")
            _print_result(r, brief=True)

        if args.output:
            _save_result(results, args.output)
        return

    # ─── Mode 4: Demo mode (no input provided) ──────────────────────────
    # Use case: Smoke test / demonstration of the full pipeline.
    # Generates a synthetic patient with known V5 and V7 parameters,
    # then runs the pipeline to verify end-to-end functionality.
    print("No input provided. Running demo with a synthetic patient...")
    from synthetic_cohort import evaluate_patient_state

    # Define V5 parameters: mild HFrEF (Sf_act=0.8) + early CKD (Kf=0.85)
    # with mild inflammation and diabetes.
    v5_params = {
        'Sf_act_scale': 0.8, 'Kf_scale': 0.85, 'inflammation_scale': 0.15,
        'diabetes_scale': 0.1, 'k1_scale': 1.0, 'RAAS_gain': 1.5,
        'TGF_gain': 2.0, 'na_intake': 150.0,
    }
    # Define V7 parameters: moderate HFrEF (Sf_act=0.6) + moderate CKD (Kf=0.55)
    # with increased inflammation and diastolic stiffness. This represents
    # 6 years of disease progression.
    v7_params = {
        'Sf_act_scale': 0.6, 'Kf_scale': 0.55, 'inflammation_scale': 0.3,
        'diabetes_scale': 0.2, 'k1_scale': 1.3, 'RAAS_gain': 1.8,
        'TGF_gain': 2.0, 'na_intake': 170.0,
    }
    demographics = {'age': 72, 'sex': 'M', 'BSA': 1.9, 'height_m': 1.75}

    # Generate V5 and V7 clinical data from the ground-truth parameters.
    print("Generating V5 and V7 ground truth from model...")
    v5_data = evaluate_patient_state(v5_params, demographics)
    # V7 uses age+6 years (ARIC V5→V7 interval is approximately 6 years).
    v7_target = evaluate_patient_state(v7_params, {'age': 78, 'sex': 'M', 'BSA': 1.9, 'height_m': 1.75})

    if v5_data is None or v7_target is None:
        print("ERROR: Model evaluation failed")
        return

    # Print key variables for quick comparison.
    print(f"V5: LVEF={v5_data['LVEF_pct']:.1f}%, GFR={v5_data['GFR_mL_min']:.1f}")
    print(f"V7: LVEF={v7_target['LVEF_pct']:.1f}%, GFR={v7_target['GFR_mL_min']:.1f}")

    # Run the full pipeline. Since we have explicit V7 data (ground truth),
    # the NN stage is bypassed and the agent works directly on the known target.
    pipeline = CardiorenalPipeline(
        nn_model_path=args.nn_path, llm_model=args.model,
        max_iterations=args.max_iter,
    )

    result = pipeline.predict_and_explain(v5_data, demographics, v7_target)
    _print_result(result)

    if args.output:
        _save_result(result, args.output)


def _print_result(result: Dict, brief: bool = False):
    """
    Print a pipeline result to stdout.

    Parameters
    ----------
    result : dict
        Pipeline result from predict_and_explain().
    brief : bool
        If True, print only the summary line (for batch mode).
        If False, print full details including parameters and explanations.
    """
    print(f"\n{'='*60}")
    t = result.get('timing', {})
    # Summary line: always printed.
    print(f"  Converged: {result.get('agent_converged', '?')}  "
          f"Error: {result.get('prediction_error', '?'):.4f}  "
          f"Time: {t.get('total_seconds', '?')}s  "
          f"Model runs: {t.get('n_model_runs', '?')}")

    if not brief:
        # Detailed output: optimal parameters with defaults for comparison.
        print(f"\nOptimal Parameters:")
        for k, v in result.get('optimal_params', {}).items():
            default = TUNABLE_PARAMS.get(k, {}).get('default', '?')
            print(f"  {k:25s} = {v:.4f}  (default: {default})")

        # Truncated policy and explanation (first 400 chars).
        print(f"\nParameter Policy:")
        print(f"  {result.get('parameter_policy', 'N/A')[:400]}")

        print(f"\nMechanistic Explanation:")
        print(f"  {result.get('mechanistic_explanation', 'N/A')[:400]}")
    print(f"{'='*60}")


def _save_result(result, path: str):
    """
    Save a pipeline result (or list of results) to a JSON file.

    Handles numpy types (float32/64, int32/64, ndarray) which are not
    JSON-serializable by default. The custom `convert` function coerces
    them to native Python types.

    Parameters
    ----------
    result : dict or list[dict]
        Pipeline result(s) to save.
    path : str
        Output file path (.json).
    """
    def convert(obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        # Fallback: convert to string (handles any other non-serializable types).
        return str(obj)

    with open(path, 'w') as f:
        json.dump(result, f, indent=2, default=convert)
    print(f"Result saved to {path}")


if __name__ == '__main__':
    main()
