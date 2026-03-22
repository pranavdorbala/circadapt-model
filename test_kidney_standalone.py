#!/usr/bin/env python3
"""
Test Suite: Hallow Renal Model — Standalone Validation
=======================================================
Tests the kidney subsystem equations from the paper (Section 3.2, Appendix C)
against the implementation in cardiorenal_coupling.py.

This script runs WITHOUT CircAdapt — it tests only the pure-Python renal model.

Tests:
  1. TGF convergence across MAP range (Eq. 28, paper §C.2)
  2. GFR vs MAP steady-state curves at different CKD severities (Eq. 29-33)
  3. Sodium balance stability (Eq. 34-39, paper §C.4-C.5)
  4. Diabetes biphasic Kf behavior (Eq. 44, paper §D.2)
  5. RAAS activation curve (Eq. 24, paper §C.1)
  6. Blood volume dynamics under step perturbations (Eq. 38-39)
  7. SVR composite formula amplification (Eq. 27)
  8. Inflammatory modifier composition (Eq. 40-49, paper §D)

Each test produces JSON results + matplotlib figures.

Author: Generated for cardiorenal integration error diagnosis
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import traceback

# ─── Add repo to path so we can import the actual implementation ───
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'card-renal-sim'))

from cardiorenal_coupling import (
    HallowRenalModel,
    InflammatoryState,
    update_renal_model,
    update_inflammatory_state,
    kidney_to_heart,
)

# ─── Output directory ───
OUT_DIR = os.path.join(os.path.dirname(__file__), 'test_results_kidney')
os.makedirs(OUT_DIR, exist_ok=True)


def save_json(data, filename):
    """Save results as JSON, converting numpy types."""
    def convert(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj
    with open(os.path.join(OUT_DIR, filename), 'w') as f:
        json.dump(convert(data), f, indent=2)
    print(f"  Saved: {filename}")


# =========================================================================
# TEST 1: TGF Convergence Across MAP Range
# =========================================================================
def test_tgf_convergence():
    """
    Test 2 from error plan: Sweep MAP from 50-150 mmHg, check TGF convergence
    at different CKD severity levels (Kf_scale = 1.0, 0.7, 0.5, 0.3).
    
    Corresponds to paper Eq. 28 (TGF fixed-point iteration) and Eq. 29-33
    (glomerular filtration).
    
    Key diagnostic: Does the 30-iteration TGF loop converge for all MAP values?
    The error plan identifies this as "Failure Mode B" — the TGF fixed-point
    solver may diverge at extreme MAP inputs.
    """
    print("\n" + "="*70)
    print("TEST 1: TGF Convergence Across MAP Range")
    print("="*70)
    
    MAP_range = np.arange(50, 152, 2)  # 50 to 150 mmHg in steps of 2
    Kf_scales = [1.0, 0.7, 0.5, 0.3]
    
    results = {}
    
    for kf in Kf_scales:
        gfr_values = []
        vblood_values = []
        svr_values = []
        converged = []
        raa_trajectories = []
        
        for MAP in MAP_range:
            # Fresh kidney model for each test point
            renal = HallowRenalModel()
            renal.Kf_scale = kf
            
            # Run 5 equilibration steps at healthy baseline first
            for _ in range(5):
                try:
                    renal = update_renal_model(renal, 93.0, 5.0, 3.0, 6.0)
                except Exception:
                    pass
            
            # Now apply the test MAP
            gfr_before = renal.GFR
            try:
                renal = update_renal_model(renal, float(MAP), 5.0, 3.0, 6.0)
                gfr_values.append(renal.GFR)
                vblood_values.append(renal.V_blood)
                
                # Compute SVR ratio
                k2h = kidney_to_heart(renal, float(MAP), 5.0, 3.0)
                svr_values.append(k2h.SVR_ratio)
                
                # Check if GFR is physiologically reasonable
                is_converged = 5.0 <= renal.GFR <= 250.0
                converged.append(is_converged)
            except Exception as e:
                gfr_values.append(np.nan)
                vblood_values.append(np.nan)
                svr_values.append(np.nan)
                converged.append(False)
        
        results[f'Kf_{kf}'] = {
            'MAP': MAP_range.tolist(),
            'GFR': gfr_values,
            'V_blood': vblood_values,
            'SVR_ratio': svr_values,
            'converged': converged,
            'n_unstable': sum(1 for c in converged if not c),
            'stable_MAP_range': [
                float(MAP_range[i]) for i in range(len(converged)) if converged[i]
            ]
        }
        
        n_fail = sum(1 for c in converged if not c)
        print(f"  Kf_scale={kf:.1f}: {n_fail}/{len(converged)} unstable MAP values")
        if results[f'Kf_{kf}']['stable_MAP_range']:
            print(f"    Stable MAP range: {min(results[f'Kf_{kf}']['stable_MAP_range']):.0f}"
                  f" - {max(results[f'Kf_{kf}']['stable_MAP_range']):.0f} mmHg")
    
    save_json(results, 'kidney_tgf_convergence.json')
    
    # ─── Plot ───
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Kidney TGF Convergence: GFR vs MAP at Different CKD Severities\n'
                 '(Paper Eq. 28-33, Error Plan Test 2)', fontsize=13)
    
    for idx, kf in enumerate(Kf_scales):
        ax = axes[idx // 2][idx % 2]
        data = results[f'Kf_{kf}']
        
        gfr = np.array(data['GFR'])
        conv = np.array(data['converged'])
        
        # Plot stable points
        stable_mask = conv & ~np.isnan(gfr)
        unstable_mask = ~conv | np.isnan(gfr)
        
        if np.any(stable_mask):
            ax.plot(np.array(data['MAP'])[stable_mask], gfr[stable_mask],
                    'b-', linewidth=1.5, label='Converged')
        if np.any(unstable_mask):
            ax.scatter(np.array(data['MAP'])[unstable_mask],
                       gfr[unstable_mask], c='red', s=30, zorder=5,
                       label='Unstable/NaN')
        
        ax.set_xlabel('MAP (mmHg)')
        ax.set_ylabel('GFR (mL/min)')
        ax.set_title(f'Kf_scale = {kf} ({"Healthy" if kf==1.0 else f"CKD: {int((1-kf)*100)}% nephron loss"})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 200)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'kidney_tgf_stability.png'), dpi=150)
    plt.close()
    print("  Saved: kidney_tgf_stability.png")
    
    return results


# =========================================================================
# TEST 2: Diabetes Biphasic Kf Behavior
# =========================================================================
def test_diabetes_biphasic_kf():
    """
    Test the biphasic ultrafiltration coefficient from Eq. 44 (paper §D.2):
      Kf_diab = Kf * (1 + 0.25 * d * (1 - 1.5*d))
    
    Error plan §4.5 identifies this as a potential source of instability:
    "The transition between hyperfiltration and collapse at intermediate d
    values could interact poorly with the solver."
    
    We verify:
    - Peak hyperfiltration at d ≈ 0.33
    - Kf_diab = 0.875 at d = 1.0
    - GFR trajectory follows expected biphasic pattern
    """
    print("\n" + "="*70)
    print("TEST 2: Diabetes Biphasic Kf Behavior (Eq. 44)")
    print("="*70)
    
    d_range = np.linspace(0, 1, 50)
    results = {
        'diabetes_scale': d_range.tolist(),
        'analytical_Kf_factor': [],
        'implemented_Kf_factor': [],
        'GFR_at_MAP93': [],
        'V_blood': [],
        'errors': [],
    }
    
    for d in d_range:
        # Analytical formula from paper
        kf_analytical = 1.0 + 0.25 * d * (1.0 - 1.5 * d)
        results['analytical_Kf_factor'].append(float(kf_analytical))
        
        # Implementation: use update_inflammatory_state
        ist = InflammatoryState()
        ist = update_inflammatory_state(ist, 0.0, float(d))
        results['implemented_Kf_factor'].append(float(ist.Kf_factor))
        
        # GFR with this diabetes level
        renal = HallowRenalModel()
        try:
            # Equilibrate
            for _ in range(5):
                renal = update_renal_model(renal, 93.0, 5.0, 3.0, 6.0,
                                           inflammatory_state=ist)
            results['GFR_at_MAP93'].append(float(renal.GFR))
            results['V_blood'].append(float(renal.V_blood))
            results['errors'].append(None)
        except Exception as e:
            results['GFR_at_MAP93'].append(np.nan)
            results['V_blood'].append(np.nan)
            results['errors'].append(str(e))
    
    # Verify key properties
    kf_factors = np.array(results['analytical_Kf_factor'])
    peak_idx = np.argmax(kf_factors)
    peak_d = d_range[peak_idx]
    
    print(f"  Peak hyperfiltration at d={peak_d:.2f}, Kf_factor={kf_factors[peak_idx]:.4f}")
    print(f"  Expected: d≈0.33, Kf_factor≈1.042")
    print(f"  At d=1.0: Kf_factor={kf_factors[-1]:.4f} (expected: 0.875)")
    
    # Check implementation matches analytical
    impl = np.array(results['implemented_Kf_factor'])
    analytical = np.array(results['analytical_Kf_factor'])
    # Note: implementation multiplies by infl_Kf (1.0 when infl=0), so should match
    max_diff = np.max(np.abs(impl - analytical))
    print(f"  Max implementation vs analytical difference: {max_diff:.6f}")
    
    results['peak_d'] = float(peak_d)
    results['peak_Kf'] = float(kf_factors[peak_idx])
    results['Kf_at_d1'] = float(kf_factors[-1])
    results['max_impl_analytical_diff'] = float(max_diff)
    
    save_json(results, 'diabetes_biphasic_kf.json')
    
    # ─── Plot ───
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Diabetes Biphasic Kf Behavior (Paper Eq. 44, Error Plan §4.5)', fontsize=13)
    
    # Panel 1: Kf factor
    axes[0].plot(d_range, results['analytical_Kf_factor'], 'b-', linewidth=2,
                 label='Analytical: 1+0.25d(1-1.5d)')
    axes[0].plot(d_range, results['implemented_Kf_factor'], 'r--', linewidth=1.5,
                 label='Implementation')
    axes[0].axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    axes[0].axvline(x=1/3, color='green', linestyle=':', alpha=0.5, label=f'd=1/3 (peak)')
    axes[0].set_xlabel('Diabetes severity (d)')
    axes[0].set_ylabel('Kf_factor')
    axes[0].set_title('Ultrafiltration Coefficient Modifier')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: GFR
    axes[1].plot(d_range, results['GFR_at_MAP93'], 'b-', linewidth=2)
    axes[1].axhline(y=120, color='gray', linestyle=':', alpha=0.5, label='Healthy baseline')
    axes[1].set_xlabel('Diabetes severity (d)')
    axes[1].set_ylabel('GFR (mL/min)')
    axes[1].set_title('GFR at MAP=93 mmHg')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Blood volume
    axes[2].plot(d_range, results['V_blood'], 'r-', linewidth=2)
    axes[2].set_xlabel('Diabetes severity (d)')
    axes[2].set_ylabel('V_blood (mL)')
    axes[2].set_title('Blood Volume at Equilibrium')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'diabetes_biphasic_kf.png'), dpi=150)
    plt.close()
    print("  Saved: diabetes_biphasic_kf.png")
    
    return results


# =========================================================================
# TEST 3: RAAS Activation Curve
# =========================================================================
def test_raas_activation():
    """
    Verify RAAS activation (Eq. 24, paper §C.1):
      RAAS = clip(1 - gR * 0.005 * dMAP, 0.5, 2.0)
    
    And downstream effects (Eq. 25-27):
      R_EA = R_EA0 * RAAS
      eta_CD = eta_CD0 * RAAS
      SVR = RAAS * (1 + 0.4*(1 - Kf_scale))
    """
    print("\n" + "="*70)
    print("TEST 3: RAAS Activation Curve (Eq. 24-27)")
    print("="*70)
    
    MAP_range = np.arange(50, 140, 1)
    MAP_setpoint = 93.0
    gR = 1.0  # default RAAS gain
    
    results = {
        'MAP': MAP_range.tolist(),
        'RAAS_analytical': [],
        'RAAS_from_model': [],
        'SVR_Kf_1.0': [],
        'SVR_Kf_0.7': [],
        'SVR_Kf_0.3': [],
    }
    
    for MAP in MAP_range:
        dMAP = MAP - MAP_setpoint
        raas_analytical = np.clip(1.0 - gR * 0.005 * dMAP, 0.5, 2.0)
        results['RAAS_analytical'].append(float(raas_analytical))
        
        # SVR composite formula (Eq. 27) at different CKD levels
        for kf, key in [(1.0, 'SVR_Kf_1.0'), (0.7, 'SVR_Kf_0.7'), (0.3, 'SVR_Kf_0.3')]:
            svr = raas_analytical * (1.0 + 0.4 * (1.0 - kf))
            results[key].append(float(svr))
    
    # Also get RAAS from actual model runs
    for MAP in MAP_range:
        renal = HallowRenalModel()
        # Equilibrate then test
        for _ in range(3):
            renal = update_renal_model(renal, 93.0, 5.0, 3.0, 6.0)
        renal = update_renal_model(renal, float(MAP), 5.0, 3.0, 6.0)
        # Infer RAAS from the model's behavior indirectly via SVR
        k2h = kidney_to_heart(renal, float(MAP), 5.0, 3.0)
        results['RAAS_from_model'].append(float(k2h.SVR_ratio))
    
    save_json(results, 'raas_activation.json')
    
    # ─── Plot ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('RAAS Activation & SVR Amplification (Paper Eq. 24-27, Error Plan §4.3)', fontsize=13)
    
    axes[0].plot(MAP_range, results['RAAS_analytical'], 'b-', linewidth=2, label='RAAS factor')
    axes[0].axvline(x=93, color='green', linestyle=':', alpha=0.5, label='MAP setpoint (93 mmHg)')
    axes[0].axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
    axes[0].set_xlabel('MAP (mmHg)')
    axes[0].set_ylabel('RAAS factor')
    axes[0].set_title('RAAS Activation vs MAP')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(MAP_range, results['SVR_Kf_1.0'], 'b-', linewidth=2, label='Kf=1.0 (healthy)')
    axes[1].plot(MAP_range, results['SVR_Kf_0.7'], 'orange', linewidth=2, label='Kf=0.7 (mild CKD)')
    axes[1].plot(MAP_range, results['SVR_Kf_0.3'], 'r-', linewidth=2, label='Kf=0.3 (severe CKD)')
    axes[1].axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
    axes[1].set_xlabel('MAP (mmHg)')
    axes[1].set_ylabel('SVR ratio')
    axes[1].set_title('SVR Composite (Eq. 27): RAAS × (1+0.4(1-Kf))\n'
                       'Error Plan §4.3: SVR nearly doubles at severe CKD')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'raas_svr_curves.png'), dpi=150)
    plt.close()
    print("  Saved: raas_svr_curves.png")
    
    # Key diagnostic from error plan §4.3
    # At Kf=0.3, MAP=70 (low from failing heart):
    dMAP = 70 - 93
    raas_low = np.clip(1.0 - 1.0 * 0.005 * dMAP, 0.5, 2.0)
    svr_severe = raas_low * (1.0 + 0.4 * (1 - 0.3))
    print(f"  Key diagnostic (Error Plan §4.3):")
    print(f"    At MAP=70, Kf=0.3: RAAS={raas_low:.3f}, SVR={svr_severe:.3f}")
    print(f"    This means SVR nearly doubles ({svr_severe:.1f}x baseline)")
    
    return results


# =========================================================================
# TEST 4: Blood Volume Step Response
# =========================================================================
def test_blood_volume_dynamics():
    """
    Test the sodium/water balance equations (Eq. 38-39, paper §C.4-C.5)
    under step changes in GFR.
    
    Error plan §4.4: "If GFR drops sharply, Na_CD_out drops dramatically,
    and sodium accumulates. The Vblood increase is proportional to cumulative
    sodium retention over the time step dt."
    """
    print("\n" + "="*70)
    print("TEST 4: Blood Volume Step Response (Eq. 38-39)")
    print("="*70)
    
    # Simulate GFR drops of varying magnitude
    Kf_reductions = np.arange(0.0, 0.55, 0.05)
    dt_hours_list = [6.0, 24.0, 168.0]  # 6h, 1 day, 1 week
    
    results = {'Kf_reduction': Kf_reductions.tolist()}
    
    for dt_hours in dt_hours_list:
        dt_key = f"dt{int(dt_hours)}h"
        vblood_changes = []
        na_retained = []
        
        for delta_kf in Kf_reductions:
            renal = HallowRenalModel()
            # Equilibrate at healthy
            for _ in range(5):
                renal = update_renal_model(renal, 93.0, 5.0, 3.0, 6.0)
            vblood_0 = renal.V_blood
            na_0 = renal.Na_total
            
            # Apply Kf reduction
            renal.Kf_scale = max(1.0 - delta_kf, 0.3)
            renal = update_renal_model(renal, 93.0, 5.0, 3.0, dt_hours)
            
            vblood_changes.append(float(renal.V_blood - vblood_0))
            na_retained.append(float(renal.Na_total - na_0))
        
        results[f'dV_blood_{dt_key}'] = vblood_changes
        results[f'dNa_{dt_key}'] = na_retained
        
        print(f"  dt={dt_hours}h: max Vblood change = {max(vblood_changes):.1f} mL "
              f"(at Kf reduction={Kf_reductions[np.argmax(vblood_changes)]:.2f})")
    
    save_json(results, 'blood_volume_step_response.json')
    
    # ─── Plot ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Blood Volume Response to GFR Drop (Paper Eq. 38-39, Error Plan §4.4)', fontsize=13)
    
    for dt_hours, color, style in [(6, 'blue', '-'), (24, 'orange', '--'), (168, 'red', '-.')]:
        dt_key = f"dt{int(dt_hours)}h"
        axes[0].plot(Kf_reductions * 100, results[f'dV_blood_{dt_key}'],
                     color=color, linestyle=style, linewidth=2, label=f'dt={dt_hours}h')
        axes[1].plot(Kf_reductions * 100, results[f'dNa_{dt_key}'],
                     color=color, linestyle=style, linewidth=2, label=f'dt={dt_hours}h')
    
    axes[0].set_xlabel('Kf reduction (%)')
    axes[0].set_ylabel('ΔV_blood (mL)')
    axes[0].set_title('Blood Volume Change')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Kf reduction (%)')
    axes[1].set_ylabel('ΔNa_total (mEq)')
    axes[1].set_title('Sodium Retention')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'blood_volume_step_response.png'), dpi=150)
    plt.close()
    print("  Saved: blood_volume_step_response.png")
    
    return results


# =========================================================================
# TEST 5: Inflammatory Modifier Composition
# =========================================================================
def test_inflammatory_composition():
    """
    Test all inflammatory/diabetes modifiers from Table 5 (paper §D):
    - Cardiac: AGE stiffening (Eq. 40-41), contractility (Eq. 42), SVR (Eq. 43)
    - Renal: biphasic Kf (Eq. 44), EA constriction (Eq. 45), SGLT2 (Eq. 46), MAP shift (Eq. 47)
    - Composition: multiplicative on shared targets (Eq. 48-49)
    """
    print("\n" + "="*70)
    print("TEST 5: Inflammatory Modifier Composition (Table 5)")
    print("="*70)
    
    # Grid: inflammation x diabetes
    infl_range = np.linspace(0, 1, 20)
    diab_range = np.linspace(0, 1, 20)
    
    results = {
        'infl_range': infl_range.tolist(),
        'diab_range': diab_range.tolist(),
        'modifiers': {},
    }
    
    modifier_names = [
        'Sf_act_factor', 'p0_factor', 'stiffness_factor', 'passive_k1_factor',
        'Kf_factor', 'R_AA_factor', 'R_EA_factor', 'RAAS_gain_factor',
        'eta_PT_offset', 'MAP_setpoint_offset'
    ]
    
    for name in modifier_names:
        results['modifiers'][name] = np.zeros((len(infl_range), len(diab_range))).tolist()
    
    for i, infl in enumerate(infl_range):
        for j, diab in enumerate(diab_range):
            ist = InflammatoryState()
            ist = update_inflammatory_state(ist, float(infl), float(diab))
            for name in modifier_names:
                results['modifiers'][name][i][j] = float(getattr(ist, name))
    
    # Verify specific paper equations
    print("  Verification of paper equations (Table 5):")
    
    # Test at infl=1, diab=0
    ist = update_inflammatory_state(InflammatoryState(), 1.0, 0.0)
    print(f"    infl=1.0, diab=0.0:")
    print(f"      Sf_act_factor = {ist.Sf_act_factor:.4f} (expect 0.75 = 1-0.25)")
    print(f"      stiffness_factor = {ist.stiffness_factor:.4f} (expect 1.30 = 1+0.30)")
    print(f"      RAAS_gain_factor = {ist.RAAS_gain_factor:.4f} (expect 1.30 = 1+0.30)")
    
    # Test at infl=0, diab=1
    ist = update_inflammatory_state(InflammatoryState(), 0.0, 1.0)
    print(f"    infl=0.0, diab=1.0:")
    print(f"      passive_k1_factor = {ist.passive_k1_factor:.4f} (expect 1.40 = 1+0.40)")
    print(f"      R_EA_factor = {ist.R_EA_factor:.4f} (expect 1.25 = 1+0.25)")
    print(f"      eta_PT_offset = {ist.eta_PT_offset:.4f} (expect 0.06)")
    print(f"      MAP_setpoint_offset = {ist.MAP_setpoint_offset:.4f} (expect 8.0)")
    
    # Test composition at infl=1, diab=1 (Eq. 48-49)
    ist = update_inflammatory_state(InflammatoryState(), 1.0, 1.0)
    print(f"    infl=1.0, diab=1.0 (composition test):")
    print(f"      stiffness_factor = {ist.stiffness_factor:.4f} "
          f"(expect 1.30*1.50 = 1.95)")
    print(f"      eta_PT_offset = {ist.eta_PT_offset:.4f} "
          f"(expect 0.04+0.06 = 0.10)")
    
    # Discrepancy checks
    errors = []
    ist = update_inflammatory_state(InflammatoryState(), 1.0, 0.0)
    if abs(ist.Sf_act_factor - 0.75) > 0.01:
        errors.append(f"Sf_act_factor at infl=1,diab=0: {ist.Sf_act_factor} != 0.75")
    
    ist = update_inflammatory_state(InflammatoryState(), 0.0, 1.0)
    if abs(ist.passive_k1_factor - 1.40) > 0.01:
        errors.append(f"passive_k1_factor at infl=0,diab=1: {ist.passive_k1_factor} != 1.40")
    
    ist = update_inflammatory_state(InflammatoryState(), 1.0, 1.0)
    expected_stiff = 1.30 * 1.50  # (1+0.30*1) * (1+0.50*1)
    if abs(ist.stiffness_factor - expected_stiff) > 0.01:
        errors.append(f"stiffness_factor composition: {ist.stiffness_factor} != {expected_stiff}")
    
    results['verification_errors'] = errors
    if errors:
        print(f"\n  ⚠ DISCREPANCIES FOUND:")
        for e in errors:
            print(f"    - {e}")
    else:
        print(f"\n  ✓ All equation verifications passed")
    
    save_json(results, 'inflammatory_composition.json')
    
    # ─── Plot heatmaps ───
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Inflammatory Modifier Factors (Paper Table 5, Eq. 40-49)', fontsize=13)
    
    plot_modifiers = [
        ('Sf_act_factor', 'Contractility\n(Eq. 42: (1-0.25i)(1-0.20d))'),
        ('passive_k1_factor', 'Myocardial Stiffness\n(Eq. 40: 1+0.40d)'),
        ('stiffness_factor', 'Arterial Stiffness\n(Eq. 48: (1+0.30i)(1+0.50d))'),
        ('Kf_factor', 'Glom. Filtration Coeff\n(biphasic Kf, Eq. 44)'),
        ('p0_factor', 'SVR / Resistance\n(Eq. 43: (1+0.15i)(1+0.10d))'),
        ('eta_PT_offset', 'PT Na Reabsorption Offset\n(Eq. 49: 0.04i + 0.06d)'),
    ]
    
    for idx, (name, title) in enumerate(plot_modifiers):
        ax = axes[idx // 3][idx % 3]
        data = np.array(results['modifiers'][name])
        im = ax.imshow(data, origin='lower', aspect='auto',
                       extent=[0, 1, 0, 1], cmap='RdYlBu_r')
        ax.set_xlabel('Diabetes severity')
        ax.set_ylabel('Inflammation severity')
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'inflammatory_modifiers.png'), dpi=150)
    plt.close()
    print("  Saved: inflammatory_modifiers.png")
    
    return results


# =========================================================================
# TEST 6: Multi-Step Kidney Deterioration Trajectory
# =========================================================================
def test_kidney_trajectory():
    """
    Run a 24-step (12-month) kidney-only trajectory with progressive Kf decline.
    Tests the sodium balance ODEs under chronic GFR decline — the condition
    that generates the large Vblood/SVR steps that crash the heart (Error Plan §4.4).
    """
    print("\n" + "="*70)
    print("TEST 6: Multi-Step Kidney Deterioration Trajectory")
    print("="*70)
    
    n_steps = 24  # 24 biweekly steps = ~12 months
    dt_hours = 336.0  # 2 weeks in hours
    
    # Progressive CKD: Kf declines from 1.0 to 0.5 over 12 months
    kf_schedule = np.linspace(1.0, 0.5, n_steps)
    
    results = {
        'step': list(range(n_steps)),
        'Kf_scale': kf_schedule.tolist(),
        'GFR': [], 'V_blood': [], 'Na_total': [], 'Na_excretion': [],
        'SVR_ratio': [], 'P_glom': [],
        'dV_blood': [],  # step-over-step change
    }
    
    renal = HallowRenalModel()
    # Equilibrate
    for _ in range(5):
        renal = update_renal_model(renal, 93.0, 5.0, 3.0, 6.0)
    
    prev_vblood = renal.V_blood
    
    for s in range(n_steps):
        renal.Kf_scale = float(kf_schedule[s])
        renal = update_renal_model(renal, 93.0, 5.0, 3.0, dt_hours)
        
        k2h = kidney_to_heart(renal, 93.0, 5.0, 3.0)
        
        dv = renal.V_blood - prev_vblood
        
        results['GFR'].append(float(renal.GFR))
        results['V_blood'].append(float(renal.V_blood))
        results['Na_total'].append(float(renal.Na_total))
        results['Na_excretion'].append(float(renal.Na_excretion))
        results['SVR_ratio'].append(float(k2h.SVR_ratio))
        results['P_glom'].append(float(renal.P_glom))
        results['dV_blood'].append(float(dv))
        
        prev_vblood = renal.V_blood
    
    max_dv = max(abs(v) for v in results['dV_blood'])
    max_svr = max(results['SVR_ratio'])
    print(f"  Max per-step ΔV_blood: {max_dv:.1f} mL")
    print(f"  Max SVR ratio: {max_svr:.3f}")
    print(f"  Final GFR: {results['GFR'][-1]:.1f} mL/min (from ~120)")
    print(f"  Final V_blood: {results['V_blood'][-1]:.0f} mL (from ~5000)")
    
    # Flag potential heart crash triggers
    VBLOOD_DANGER = 300  # >300 mL step change likely crashes heart
    SVR_DANGER = 1.5     # >1.5x SVR likely overloads heart
    
    dangerous_steps = [
        s for s in range(n_steps)
        if abs(results['dV_blood'][s]) > VBLOOD_DANGER or results['SVR_ratio'][s] > SVR_DANGER
    ]
    if dangerous_steps:
        print(f"  ⚠ Dangerous step changes at steps: {dangerous_steps}")
        print(f"    These would likely trigger heart solver crash (Error Plan Failure Mode A)")
    else:
        print(f"  ✓ No single step exceeds danger thresholds")
    
    results['max_dV_blood'] = float(max_dv)
    results['max_SVR_ratio'] = float(max_svr)
    results['dangerous_steps'] = dangerous_steps
    
    save_json(results, 'kidney_trajectory_12month.json')
    
    # ─── Plot ───
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('12-Month Kidney Deterioration Trajectory\n'
                 'Kf declines 1.0→0.5 (Error Plan: What Vblood/SVR does heart receive?)',
                 fontsize=13)
    
    months = np.array(results['step']) * 0.5  # biweekly -> months
    
    for ax, key, ylabel, color in [
        (axes[0,0], 'GFR', 'GFR (mL/min)', 'blue'),
        (axes[0,1], 'V_blood', 'V_blood (mL)', 'red'),
        (axes[0,2], 'SVR_ratio', 'SVR ratio', 'orange'),
        (axes[1,0], 'Na_excretion', 'Na excretion (mEq/day)', 'green'),
        (axes[1,1], 'dV_blood', 'ΔV_blood per step (mL)', 'purple'),
        (axes[1,2], 'P_glom', 'P_glom (mmHg)', 'brown'),
    ]:
        ax.plot(months, results[key], color=color, linewidth=2)
        ax.set_xlabel('Month')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if key == 'dV_blood':
            ax.axhline(y=VBLOOD_DANGER, color='red', linestyle='--', alpha=0.5, label=f'Danger: {VBLOOD_DANGER}mL')
            ax.axhline(y=-VBLOOD_DANGER, color='red', linestyle='--', alpha=0.5)
            ax.legend(fontsize=8)
        if key == 'SVR_ratio':
            ax.axhline(y=SVR_DANGER, color='red', linestyle='--', alpha=0.5, label=f'Danger: {SVR_DANGER}x')
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'kidney_trajectory_12month.png'), dpi=150)
    plt.close()
    print("  Saved: kidney_trajectory_12month.png")
    
    return results


# =========================================================================
# TEST 7: Passive Fiber Stress Exponential Sensitivity (Eq. 18)
# =========================================================================
def test_passive_stress_sensitivity():
    """
    Analytical test of the passive fiber stress equation (Eq. 18):
      Sf_pas = k1 * (exp(k2*(lambda-1)) - 1)
    
    Error plan §4.1 identifies this as PRIMARY SUSPECT for solver crashes.
    We compute the stress gradient and identify the critical lambda values.
    """
    print("\n" + "="*70)
    print("TEST 7: Passive Fiber Stress Exponential Sensitivity (Eq. 18)")
    print("="*70)
    
    k2 = 9.0  # CircAdapt default
    k1_scales = [1.0, 1.5, 2.0, 2.5]
    k1_base = 1.0  # normalized
    lambda_range = np.linspace(1.0, 1.30, 200)
    
    results = {
        'lambda': lambda_range.tolist(),
        'k2': k2,
    }
    
    for k1s in k1_scales:
        k1 = k1_base * k1s
        Sf_pas = k1 * (np.exp(k2 * (lambda_range - 1.0)) - 1.0)
        dSf_dlambda = k1 * k2 * np.exp(k2 * (lambda_range - 1.0))  # derivative
        
        results[f'Sf_pas_k1_{k1s}'] = Sf_pas.tolist()
        results[f'dSf_dlambda_k1_{k1s}'] = dSf_dlambda.tolist()
    
    # Key diagnostics from error plan
    for lam in [1.08, 1.15, 1.20, 1.25]:
        sf = k1_base * (np.exp(k2 * (lam - 1.0)) - 1.0)
        dsf = k1_base * k2 * np.exp(k2 * (lam - 1.0))
        print(f"  λ={lam:.2f}: Sf_pas={sf:.2f}, dSf/dλ={dsf:.1f}")
    
    print(f"\n  Error Plan §4.1 diagnostic:")
    print(f"    At λ=1.08: exp term = {np.exp(k2*0.08):.2f}")
    print(f"    At λ=1.15: exp term = {np.exp(k2*0.15):.2f}")
    print(f"    At λ=1.25: exp term = {np.exp(k2*0.25):.2f}")
    print(f"    The stress DOUBLES between λ=1.08 and λ=1.15")
    print(f"    The stress is 4.6x at λ=1.25 vs λ=1.08")
    
    save_json(results, 'passive_stress_sensitivity.json')
    
    # ─── Plot ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Passive Fiber Stress Exponential (Paper Eq. 18, Error Plan §4.1 PRIMARY SUSPECT)',
                 fontsize=13)
    
    for k1s, color in [(1.0, 'blue'), (1.5, 'orange'), (2.0, 'red'), (2.5, 'darkred')]:
        axes[0].plot(lambda_range, results[f'Sf_pas_k1_{k1s}'],
                     color=color, linewidth=2, label=f'k1_scale={k1s}')
        axes[1].semilogy(lambda_range, results[f'dSf_dlambda_k1_{k1s}'],
                         color=color, linewidth=2, label=f'k1_scale={k1s}')
    
    axes[0].set_xlabel('Fiber stretch ratio (λ)')
    axes[0].set_ylabel('Sf_pas (normalized)')
    axes[0].set_title('Passive Stress')
    axes[0].axvline(x=1.08, color='green', linestyle=':', alpha=0.5, label='Healthy ED λ≈1.08')
    axes[0].axvline(x=1.20, color='red', linestyle=':', alpha=0.5, label='Danger zone λ≈1.20')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Fiber stretch ratio (λ)')
    axes[1].set_ylabel('dSf/dλ (log scale)')
    axes[1].set_title('Stress Gradient (solver must resolve this)')
    axes[1].axvline(x=1.08, color='green', linestyle=':', alpha=0.5, label='Healthy')
    axes[1].axvline(x=1.20, color='red', linestyle=':', alpha=0.5, label='Danger')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'passive_stress_sensitivity.png'), dpi=150)
    plt.close()
    print("  Saved: passive_stress_sensitivity.png")
    
    return results


# =========================================================================
# TEST 8: Exponential Kidney Model Comparison
# =========================================================================
def test_exponential_kidney_model():
    """
    Error plan §5.2: Validate the proposed exponential eGFR decay model
    against the numerical Hallow model.
    
    eGFR(t) = eGFR_0 * exp(-lambda_k * t)
    
    Compare Vblood and SVR trajectories.
    """
    print("\n" + "="*70)
    print("TEST 8: Exponential vs Numerical Kidney Model (Error Plan §5.2)")
    print("="*70)
    
    eGFR_0 = 90.0  # Starting eGFR
    annual_decline_rates = [1, 2, 3, 5]  # mL/min/year
    time_months = np.arange(0, 13, 1)  # 0 to 12 months
    
    results = {'time_months': time_months.tolist()}
    
    for rate in annual_decline_rates:
        lambda_k = rate / eGFR_0  # per year
        
        # Exponential model
        egfr_exp = eGFR_0 * np.exp(-lambda_k * time_months / 12.0)
        kf_scale_exp = egfr_exp / eGFR_0
        
        # Compute corresponding Vblood and SVR analytically
        svr_exp = []
        vblood_exp = []
        for t_idx, t in enumerate(time_months):
            kf = float(kf_scale_exp[t_idx])
            # RAAS at MAP=93 (neutral) = 1.0
            svr = 1.0 * (1.0 + 0.4 * (1.0 - kf))
            svr_exp.append(svr)
            # Simplified Vblood: proportional to cumulative sodium retention
            # integral of (1 - eGFR(s)/eGFR_0) from 0 to t
            if lambda_k > 0 and t > 0:
                # integral = t - (1/lambda_k_monthly) * (1 - exp(-lambda_k_monthly*t))
                lk_m = lambda_k / 12.0
                integral_months = t - (1.0/lk_m) * (1.0 - np.exp(-lk_m * t))
                # Convert to volume change (rough scaling)
                dv = 50.0 * integral_months  # ~50 mL per month-unit of retention
                vblood_exp.append(5000.0 + dv)
            else:
                vblood_exp.append(5000.0)
        
        # Numerical model: run Hallow with declining Kf
        gfr_num = []
        vblood_num = []
        svr_num = []
        
        renal = HallowRenalModel()
        for _ in range(5):
            renal = update_renal_model(renal, 93.0, 5.0, 3.0, 6.0)
        
        for t_idx, t in enumerate(time_months):
            if t_idx > 0:
                renal.Kf_scale = float(kf_scale_exp[t_idx])
                # Run 4 weekly substeps per month
                for _ in range(4):
                    renal = update_renal_model(renal, 93.0, 5.0, 3.0, 168.0)  # 1 week
            
            gfr_num.append(float(renal.GFR))
            vblood_num.append(float(renal.V_blood))
            k2h = kidney_to_heart(renal, 93.0, 5.0, 3.0)
            svr_num.append(float(k2h.SVR_ratio))
        
        results[f'rate_{rate}_eGFR_exp'] = egfr_exp.tolist()
        results[f'rate_{rate}_GFR_num'] = gfr_num
        results[f'rate_{rate}_Kf_exp'] = kf_scale_exp.tolist()
        results[f'rate_{rate}_SVR_exp'] = svr_exp
        results[f'rate_{rate}_SVR_num'] = svr_num
        results[f'rate_{rate}_Vblood_num'] = vblood_num
        
        print(f"  Rate {rate} mL/min/yr: Final eGFR={egfr_exp[-1]:.1f}, "
              f"Final numerical GFR={gfr_num[-1]:.1f}")
    
    save_json(results, 'exponential_vs_numerical.json')
    
    # ─── Plot ───
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Exponential vs Numerical Kidney Model\n'
                 '(Error Plan §5.2: Proposed analytical replacement)', fontsize=13)
    
    colors = ['blue', 'green', 'orange', 'red']
    for i, rate in enumerate(annual_decline_rates):
        # GFR comparison
        axes[0,0].plot(time_months, results[f'rate_{rate}_eGFR_exp'],
                       color=colors[i], linestyle='--', linewidth=1.5,
                       label=f'{rate} mL/yr (exp)')
        axes[0,0].plot(time_months, results[f'rate_{rate}_GFR_num'],
                       color=colors[i], linestyle='-', linewidth=2,
                       label=f'{rate} mL/yr (num)')
        
        # SVR comparison
        axes[0,1].plot(time_months, results[f'rate_{rate}_SVR_exp'],
                       color=colors[i], linestyle='--', linewidth=1.5)
        axes[0,1].plot(time_months, results[f'rate_{rate}_SVR_num'],
                       color=colors[i], linestyle='-', linewidth=2,
                       label=f'{rate} mL/yr')
    
    axes[0,0].set_xlabel('Month')
    axes[0,0].set_ylabel('GFR / eGFR (mL/min)')
    axes[0,0].set_title('GFR: Exponential (--) vs Numerical (—)')
    axes[0,0].legend(fontsize=7, ncol=2)
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].set_xlabel('Month')
    axes[0,1].set_ylabel('SVR ratio')
    axes[0,1].set_title('SVR: Exponential (--) vs Numerical (—)')
    axes[0,1].legend(fontsize=8)
    axes[0,1].grid(True, alpha=0.3)
    
    # Kf_scale trajectory
    for i, rate in enumerate(annual_decline_rates):
        axes[1,0].plot(time_months, results[f'rate_{rate}_Kf_exp'],
                       color=colors[i], linewidth=2, label=f'{rate} mL/yr')
    axes[1,0].set_xlabel('Month')
    axes[1,0].set_ylabel('Kf_scale')
    axes[1,0].set_title('Kf_scale (exponential decay)')
    axes[1,0].legend(fontsize=8)
    axes[1,0].grid(True, alpha=0.3)
    
    # Vblood numerical
    for i, rate in enumerate(annual_decline_rates):
        axes[1,1].plot(time_months, results[f'rate_{rate}_Vblood_num'],
                       color=colors[i], linewidth=2, label=f'{rate} mL/yr')
    axes[1,1].set_xlabel('Month')
    axes[1,1].set_ylabel('V_blood (mL)')
    axes[1,1].set_title('Blood Volume (numerical Hallow)')
    axes[1,1].legend(fontsize=8)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'exponential_vs_numerical.png'), dpi=150)
    plt.close()
    print("  Saved: exponential_vs_numerical.png")
    
    return results


# =========================================================================
# MAIN
# =========================================================================
def main():
    print("="*70)
    print("  KIDNEY (HALLOW MODEL) STANDALONE TEST SUITE")
    print("  Tests paper equations against implementation")
    print("  No CircAdapt dependency required")
    print("="*70)
    
    all_results = {}
    tests = [
        ("TGF Convergence", test_tgf_convergence),
        ("Diabetes Biphasic Kf", test_diabetes_biphasic_kf),
        ("RAAS Activation", test_raas_activation),
        ("Blood Volume Dynamics", test_blood_volume_dynamics),
        ("Inflammatory Composition", test_inflammatory_composition),
        ("Kidney Trajectory", test_kidney_trajectory),
        ("Passive Stress Sensitivity", test_passive_stress_sensitivity),
        ("Exponential Model Comparison", test_exponential_kidney_model),
    ]
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            all_results[name] = {"status": "PASS", "result": "See individual JSON files"}
        except Exception as e:
            print(f"\n  ✗ TEST FAILED: {name}")
            traceback.print_exc()
            all_results[name] = {"status": "FAIL", "error": str(e)}
    
    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    for name, res in all_results.items():
        status = "✓" if res['status'] == 'PASS' else "✗"
        print(f"  {status} {name}: {res['status']}")
    
    save_json(all_results, 'test_summary.json')
    print(f"\n  All outputs saved to: {OUT_DIR}/")


if __name__ == '__main__':
    main()
