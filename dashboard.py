"""
Cardiorenal Disease Progression Interactive Dashboard

Interactive visualization of bidirectional heart-kidney message passing
with Weibull deterioration model. Shows cycle-by-cycle organ deterioration,
message exchanges, crash boundaries, and exponential vs numerical comparison.

Usage:
    python dashboard.py

Runs on http://localhost:8050
"""

import dash
from dash import dcc, html, callback_context, no_update, ctx
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import os

# ── Try live simulation mode ────────────────────────────────────────────
try:
    from cardiorenal_coupling import (
        CircAdaptHeartModel, HallowRenalModel, InflammatoryState,
        update_inflammatory_state, update_renal_model,
        heart_to_kidney, kidney_to_heart, ML_TO_M3,
    )
    LIVE_MODE = True
except ImportError:
    LIVE_MODE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEART_DIR = os.path.join(BASE_DIR, "test_results_heart")
KIDNEY_DIR = os.path.join(BASE_DIR, "test_results_kidney")

# ── Weibull deterioration model ─────────────────────────────────────────

def weibull_damage(t_months, lambda_eff, k):
    """Cumulative damage fraction D(t) in [0, 1]."""
    t = np.asarray(t_months, dtype=float)
    return 1.0 - np.exp(-((t / max(lambda_eff, 0.1)) ** k))


def effective_lambda(lambda_base, alpha, Z):
    """Shared stress Z reduces effective lambda (faster deterioration)."""
    return max(1.0, lambda_base * (1.0 - alpha * Z))


def build_schedules(n_cycles, months_per_cycle, k1_init, Kf_init,
                    lambda_c, k_c, lambda_r, k_r, alpha_c, alpha_r, Z):
    """Build deterioration schedules from Weibull parameters."""
    months = np.arange(n_cycles) * months_per_cycle
    lam_c = effective_lambda(lambda_c, alpha_c, Z)
    lam_r = effective_lambda(lambda_r, alpha_r, Z)
    D_c = weibull_damage(months, lam_c, k_c)
    D_r = weibull_damage(months, lam_r, k_r)
    k1_sched = (k1_init * (1.0 + D_c * 2.0)).tolist()
    Kf_sched = np.maximum(0.05, Kf_init * (1.0 - D_r * 0.9)).tolist()
    Sf_sched = np.maximum(0.3, 1.0 - D_c * 0.4).tolist()
    return k1_sched, Kf_sched, Sf_sched, D_c.tolist(), D_r.tolist()


# ── Demo data generation (when CircAdapt not available) ─────────────────

def generate_demo_data(n_cycles, months_per_cycle, k1_init, Kf_init,
                       lambda_c, k_c, lambda_r, k_r, alpha_c, alpha_r, Z):
    """Generate synthetic simulation data using simplified physiology."""
    k1_sched, Kf_sched, Sf_sched, D_c, D_r = build_schedules(
        n_cycles, months_per_cycle, k1_init, Kf_init,
        lambda_c, k_c, lambda_r, k_r, alpha_c, alpha_r, Z)

    hist = {k: [] for k in [
        'step', 'SBP', 'DBP', 'MAP', 'CO', 'SV', 'EF',
        'V_blood', 'GFR', 'Na_excr', 'P_glom', 'Pven', 'SVR_ratio', 'LVEDP',
        'Kf_scale', 'k1_scale', 'Sf_scale',
        'effective_Kf', 'effective_k1', 'effective_Sf',
        'D_cardiac', 'D_renal',
        'h2k_MAP', 'h2k_CO', 'h2k_Pven',
        'k2h_Vblood', 'k2h_SVR', 'k2h_GFR',
    ]}

    # Baseline healthy values
    MAP, CO, Pven = 93.0, 5.0, 5.0
    V_blood, GFR = 5000.0, 120.0
    EF, SV = 60.0, 70.0
    SVR_baseline = 18.0

    for i in range(n_cycles):
        kf = Kf_sched[i]
        k1 = k1_sched[i]
        sf = Sf_sched[i]

        # Simplified heart response
        EF = max(25, 60 * sf - (k1 - 1.0) * 3)
        MAP = 93 + (V_blood - 5000) * 0.008 + (k1 - 1.0) * 5
        SBP = MAP * 1.35
        DBP = MAP * 0.75
        CO = max(2.0, 5.0 * sf * (1 - (k1 - 1.0) * 0.05))
        Pven = 5.0 + (k1 - 1.0) * 4 + (V_blood - 5000) * 0.003
        LVEDP = 8.0 + (k1 - 1.0) * 6 + (V_blood - 5000) * 0.004
        SV = CO * 1000 / 72

        # Heart deposits message
        h2k_MAP, h2k_CO, h2k_Pven = MAP, CO, Pven

        # Kidney adjusts
        GFR = max(5, 120 * kf * (MAP / 93) * 0.9)
        P_glom = 60 * kf * (MAP / 93)
        Na_excr = max(10, 150 * (GFR / 120))
        V_blood = 5000 + (1.0 - kf) * 800 + (Pven - 5) * 20
        SVR_ratio = max(0.5, (MAP - Pven) / max(CO, 0.3) / SVR_baseline)

        # Kidney deposits message
        k2h_V, k2h_SVR, k2h_GFR = V_blood, SVR_ratio, GFR

        hist['step'].append(i + 1)
        hist['MAP'].append(round(MAP, 1))
        hist['SBP'].append(round(SBP, 1))
        hist['DBP'].append(round(DBP, 1))
        hist['CO'].append(round(CO, 2))
        hist['SV'].append(round(SV, 1))
        hist['EF'].append(round(EF, 1))
        hist['Pven'].append(round(Pven, 1))
        hist['LVEDP'].append(round(LVEDP, 1))
        hist['V_blood'].append(round(V_blood, 0))
        hist['GFR'].append(round(GFR, 1))
        hist['Na_excr'].append(round(Na_excr, 0))
        hist['P_glom'].append(round(P_glom, 1))
        hist['SVR_ratio'].append(round(SVR_ratio, 3))
        hist['Kf_scale'].append(round(kf, 3))
        hist['k1_scale'].append(round(k1, 3))
        hist['Sf_scale'].append(round(sf, 3))
        hist['effective_Kf'].append(round(kf, 3))
        hist['effective_k1'].append(round(k1, 3))
        hist['effective_Sf'].append(round(sf, 3))
        hist['D_cardiac'].append(round(D_c[i], 4))
        hist['D_renal'].append(round(D_r[i], 4))
        hist['h2k_MAP'].append(round(h2k_MAP, 1))
        hist['h2k_CO'].append(round(h2k_CO, 2))
        hist['h2k_Pven'].append(round(h2k_Pven, 1))
        hist['k2h_Vblood'].append(round(k2h_V, 0))
        hist['k2h_SVR'].append(round(k2h_SVR, 3))
        hist['k2h_GFR'].append(round(k2h_GFR, 1))

    return hist


def run_live_simulation(n_cycles, months_per_cycle, k1_init, Kf_init,
                        lambda_c, k_c, lambda_r, k_r, alpha_c, alpha_r, Z):
    """Run the simulation. Uses demo data generator (fast, stable).
    CircAdapt live mode available but requires small time steps for stability."""
    # Always use demo data for the dashboard visualization.
    # The demo generator produces physiologically reasonable trajectories
    # using the Weibull model + simplified hemodynamic relationships.
    # For full CircAdapt fidelity, use run_coupled_simulation() directly.
    return generate_demo_data(n_cycles, months_per_cycle, k1_init, Kf_init,
                              lambda_c, k_c, lambda_r, k_r, alpha_c, alpha_r, Z)

    # Live CircAdapt mode (kept for future use with stabilized time stepping)
    if not LIVE_MODE:
        return generate_demo_data(n_cycles, months_per_cycle, k1_init, Kf_init,
                                  lambda_c, k_c, lambda_r, k_r, alpha_c, alpha_r, Z)

    k1_sched, Kf_sched, Sf_sched, D_c, D_r = build_schedules(
        n_cycles, months_per_cycle, k1_init, Kf_init,
        lambda_c, k_c, lambda_r, k_r, alpha_c, alpha_r, Z)

    dt_renal_step = 6.0  # hours per sub-step (Hallow default)
    n_renal_substeps = max(1, int(months_per_cycle * 30 * 24 / dt_renal_step))

    hist = {k: [] for k in [
        'step', 'SBP', 'DBP', 'MAP', 'CO', 'SV', 'EF',
        'V_blood', 'GFR', 'Na_excr', 'P_glom', 'Pven', 'SVR_ratio', 'LVEDP',
        'Kf_scale', 'k1_scale', 'Sf_scale',
        'effective_Kf', 'effective_k1', 'effective_Sf',
        'D_cardiac', 'D_renal',
        'h2k_MAP', 'h2k_CO', 'h2k_Pven',
        'k2h_Vblood', 'k2h_SVR', 'k2h_GFR',
    ]}

    heart = CircAdaptHeartModel()
    renal = HallowRenalModel()
    ist = InflammatoryState()

    for s in range(n_cycles):
        sf, kf, k1 = Sf_sched[s], Kf_sched[s], k1_sched[s]

        ist = update_inflammatory_state(ist, 0.0, 0.0)
        heart.apply_inflammatory_modifiers(ist)
        eff_k1 = k1 * ist.passive_k1_factor
        heart.apply_stiffness(eff_k1)
        eff_sf = max(sf * ist.Sf_act_factor, 0.20)
        heart.apply_deterioration(eff_sf)
        renal.Kf_scale = kf
        eff_kf = kf * ist.Kf_factor

        try:
            hemo = heart.run_to_steady_state()
        except Exception as e:
            print(f"  [DASH] Heart solver failed at cycle {s+1}: {e}")
            break

        h2k = heart_to_kidney(hemo)
        # Sub-step the renal model (Hallow needs small dt for Euler stability)
        for _ in range(n_renal_substeps):
            renal = update_renal_model(renal, h2k.MAP, h2k.CO, h2k.Pven, dt_renal_step)
        k2h = kidney_to_heart(renal, h2k.MAP, h2k.CO, h2k.Pven)
        heart.apply_kidney_feedback(k2h.V_blood * ML_TO_M3, k2h.SVR_ratio)

        # Extract LVEDP from PV loop
        try:
            p_lv = hemo['p_LV']
            v_lv = hemo['V_LV']
            edv_idx = np.argmax(v_lv)
            lvedp = float(p_lv[edv_idx])
        except Exception:
            lvedp = hemo.get('Pven', 8.0) + 3.0

        hist['step'].append(s + 1)
        hist['MAP'].append(round(hemo['MAP'], 1))
        hist['SBP'].append(round(hemo['SBP'], 1))
        hist['DBP'].append(round(hemo['DBP'], 1))
        hist['CO'].append(round(hemo['CO'], 2))
        hist['SV'].append(round(hemo['SV'], 1))
        hist['EF'].append(round(hemo['EF'], 1))
        hist['Pven'].append(round(hemo.get('Pven', 5.0), 1))
        hist['LVEDP'].append(round(lvedp, 1))
        hist['V_blood'].append(round(renal.V_blood, 0))
        hist['GFR'].append(round(renal.GFR, 1))
        hist['Na_excr'].append(round(renal.Na_excretion, 0))
        hist['P_glom'].append(round(renal.P_glom, 1))
        hist['SVR_ratio'].append(round(k2h.SVR_ratio, 3))
        hist['Kf_scale'].append(round(kf, 3))
        hist['k1_scale'].append(round(k1, 3))
        hist['Sf_scale'].append(round(sf, 3))
        hist['effective_Kf'].append(round(eff_kf, 3))
        hist['effective_k1'].append(round(eff_k1, 3))
        hist['effective_Sf'].append(round(eff_sf, 3))
        hist['D_cardiac'].append(round(D_c[s], 4))
        hist['D_renal'].append(round(D_r[s], 4))
        hist['h2k_MAP'].append(round(h2k.MAP, 1))
        hist['h2k_CO'].append(round(h2k.CO, 2))
        hist['h2k_Pven'].append(round(h2k.Pven, 1))
        hist['k2h_Vblood'].append(round(k2h.V_blood, 0))
        hist['k2h_SVR'].append(round(k2h.SVR_ratio, 3))
        hist['k2h_GFR'].append(round(k2h.GFR, 1))

    print(f"  [DASH] Live simulation completed {len(hist['step'])}/{n_cycles} cycles")
    return hist


# ── Load cached JSON data ───────────────────────────────────────────────

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


CRASH_DATA = load_json(os.path.join(HEART_DIR, "coupled_crash_boundary.json"))
STIFFNESS_DATA = load_json(os.path.join(HEART_DIR, "stiffness_sweep.json"))
EXP_NUM_DATA = load_json(os.path.join(KIDNEY_DIR, "exponential_vs_numerical.json"))

# ── Dash App ────────────────────────────────────────────────────────────

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Cardiorenal Coupling Simulator"

# ── Color palette ───────────────────────────────────────────────────────
C_HEART = "#ff6b6b"
C_KIDNEY = "#4ecdc4"
C_MSG = "#ffd93d"
C_BG = "#1a1a2e"
C_CARD = "#16213e"
C_TEXT = "#e0e0e0"
C_ACCENT = "#0f3460"

DARK_TEMPLATE = "plotly_dark"

# ── Slider helper ───────────────────────────────────────────────────────

def make_slider(id_, label, min_, max_, value, step, marks=None):
    return html.Div([
        html.Label(label, style={"fontSize": "12px", "color": C_TEXT, "marginBottom": "2px"}),
        dcc.Slider(id=id_, min=min_, max=max_, value=value, step=step,
                   marks=marks or {min_: str(min_), max_: str(max_)},
                   tooltip={"placement": "bottom", "always_visible": False}),
    ], style={"marginBottom": "10px"})


# ── Layout ──────────────────────────────────────────────────────────────

sidebar = html.Div([
    html.H3("Controls", style={"color": C_MSG, "marginBottom": "15px"}),

    # Mode indicator
    html.Div(
        f"WEIBULL + SIMPLIFIED MODEL",
        style={"color": "#2ecc71",
               "fontWeight": "bold", "fontSize": "11px", "marginBottom": "15px",
               "padding": "4px 8px", "borderRadius": "4px",
               "backgroundColor": C_ACCENT, "textAlign": "center"}
    ),

    html.Hr(style={"borderColor": C_ACCENT}),
    html.H4("Simulation", style={"color": C_TEXT, "fontSize": "13px"}),
    make_slider("n-cycles", "Number of Cycles", 1, 12, 6, 1),
    make_slider("months-per-cycle", "Months per Cycle", 1, 6, 2, 1),
    make_slider("k1-init", "Initial Stiffness (k1)", 1.0, 3.0, 1.2, 0.1),
    make_slider("Kf-init", "Initial Filtration (Kf)", 0.1, 1.0, 0.9, 0.05),

    html.Hr(style={"borderColor": C_ACCENT}),
    html.H4("Cardiac Weibull", style={"color": C_HEART, "fontSize": "13px"}),
    make_slider("k-cardiac", "Shape k (hazard accel.)", 1.0, 3.0, 1.8, 0.1),
    make_slider("lambda-cardiac", "Scale \u03bb (months)", 5, 50, 20, 1),

    html.Hr(style={"borderColor": C_ACCENT}),
    html.H4("Renal Weibull", style={"color": C_KIDNEY, "fontSize": "13px"}),
    make_slider("k-renal", "Shape k (hazard accel.)", 1.0, 4.0, 2.5, 0.1),
    make_slider("lambda-renal", "Scale \u03bb (months)", 5, 50, 15, 1),

    html.Hr(style={"borderColor": C_ACCENT}),
    html.H4("Shared Stress", style={"color": C_MSG, "fontSize": "13px"}),
    make_slider("Z-stress", "Z (systemic load)", 0.0, 2.0, 0.5, 0.1),
    make_slider("alpha-c", "\u03b1_cardiac (Z loading)", 0.0, 1.0, 0.3, 0.05),
    make_slider("alpha-r", "\u03b1_renal (Z loading)", 0.0, 1.0, 0.5, 0.05),

    html.Hr(style={"borderColor": C_ACCENT}),

    # Effective lambda readout
    html.Div(id="lambda-readout", style={"fontSize": "11px", "color": C_TEXT,
                                          "marginBottom": "10px", "fontFamily": "monospace"}),

    # Buttons
    html.Div([
        html.Button("Run", id="run-btn", n_clicks=0,
                     style={"backgroundColor": "#2ecc71", "color": "white",
                            "border": "none", "padding": "8px 16px", "marginRight": "5px",
                            "borderRadius": "4px", "cursor": "pointer", "fontWeight": "bold"}),
        html.Button("Reset", id="reset-btn", n_clicks=0,
                     style={"backgroundColor": "#e74c3c", "color": "white",
                            "border": "none", "padding": "8px 16px",
                            "borderRadius": "4px", "cursor": "pointer"}),
    ], style={"marginBottom": "10px"}),

    html.Div([
        html.Button("\u25c0", id="step-back", n_clicks=0,
                     style={"padding": "6px 12px", "marginRight": "5px",
                            "backgroundColor": C_ACCENT, "color": C_TEXT,
                            "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
        html.Button("\u25b6 Play", id="play-btn", n_clicks=0,
                     style={"padding": "6px 12px", "marginRight": "5px",
                            "backgroundColor": C_ACCENT, "color": C_TEXT,
                            "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
        html.Button("\u25b6", id="step-fwd", n_clicks=0,
                     style={"padding": "6px 12px",
                            "backgroundColor": C_ACCENT, "color": C_TEXT,
                            "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
    ], style={"marginBottom": "10px"}),

    # Cycle indicator
    html.Div(id="cycle-indicator",
             style={"fontSize": "14px", "fontWeight": "bold",
                    "color": C_MSG, "textAlign": "center"}),

], style={"width": "260px", "padding": "15px", "backgroundColor": C_CARD,
          "overflowY": "auto", "height": "100vh", "position": "fixed",
          "left": "0", "top": "0"})

main_content = html.Div([
    dcc.Tabs(id="tabs", value="tab-1", children=[
        dcc.Tab(label="Message Passing", value="tab-1",
                style={"backgroundColor": C_BG, "color": C_TEXT},
                selected_style={"backgroundColor": C_ACCENT, "color": C_MSG}),
        dcc.Tab(label="Deterioration Curves", value="tab-2",
                style={"backgroundColor": C_BG, "color": C_TEXT},
                selected_style={"backgroundColor": C_ACCENT, "color": C_MSG}),
        dcc.Tab(label="Crash Boundary", value="tab-3",
                style={"backgroundColor": C_BG, "color": C_TEXT},
                selected_style={"backgroundColor": C_ACCENT, "color": C_MSG}),
        dcc.Tab(label="Exp vs Numerical", value="tab-4",
                style={"backgroundColor": C_BG, "color": C_TEXT},
                selected_style={"backgroundColor": C_ACCENT, "color": C_MSG}),
    ]),
    html.Div(id="tab-content", style={"padding": "15px"}),
], style={"marginLeft": "290px", "backgroundColor": C_BG,
          "minHeight": "100vh", "color": C_TEXT})

app.layout = html.Div([
    sidebar,
    main_content,
    dcc.Store(id="sim-data", data=None),
    dcc.Store(id="cycle-idx", data=0),
    dcc.Interval(id="play-interval", interval=1500, disabled=True),
], style={"backgroundColor": C_BG, "fontFamily": "Segoe UI, sans-serif"})


# ═══════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════

# ── Lambda readout ──────────────────────────────────────────────────────

@app.callback(
    Output("lambda-readout", "children"),
    [Input("lambda-cardiac", "value"), Input("lambda-renal", "value"),
     Input("alpha-c", "value"), Input("alpha-r", "value"),
     Input("Z-stress", "value")])
def update_lambda_readout(lc, lr, ac, ar, z):
    lc_eff = effective_lambda(lc, ac, z)
    lr_eff = effective_lambda(lr, ar, z)
    return [
        html.Div(f"\u03bb_cardiac_eff = {lc_eff:.1f} months"),
        html.Div(f"\u03bb_renal_eff  = {lr_eff:.1f} months"),
    ]


# ── Run simulation ─────────────────────────────────────────────────────

@app.callback(
    [Output("sim-data", "data"), Output("cycle-idx", "data", allow_duplicate=True)],
    Input("run-btn", "n_clicks"),
    [State("n-cycles", "value"), State("months-per-cycle", "value"),
     State("k1-init", "value"), State("Kf-init", "value"),
     State("k-cardiac", "value"), State("lambda-cardiac", "value"),
     State("k-renal", "value"), State("lambda-renal", "value"),
     State("alpha-c", "value"), State("alpha-r", "value"),
     State("Z-stress", "value")],
    prevent_initial_call=True)
def run_simulation(n_clicks, n_cyc, mpc, k1i, kfi, kc, lc, kr, lr, ac, ar, z):
    if not n_clicks:
        return no_update, no_update
    hist = run_live_simulation(n_cyc, mpc, k1i, kfi, lc, kc, lr, kr, ac, ar, z)
    return hist, 0


# ── Reset ───────────────────────────────────────────────────────────────

@app.callback(
    [Output("sim-data", "data", allow_duplicate=True),
     Output("cycle-idx", "data", allow_duplicate=True)],
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True)
def reset_sim(n):
    return None, 0


# ── Step controls ───────────────────────────────────────────────────────

@app.callback(
    Output("cycle-idx", "data", allow_duplicate=True),
    [Input("step-fwd", "n_clicks"), Input("step-back", "n_clicks"),
     Input("play-interval", "n_intervals")],
    [State("cycle-idx", "data"), State("sim-data", "data")],
    prevent_initial_call=True)
def step_cycle(fwd, back, interval, idx, data):
    if not data or not data.get('step'):
        return no_update
    n = len(data['step'])
    triggered = ctx.triggered_id
    if triggered == "step-fwd" or triggered == "play-interval":
        new_idx = min((idx or 0) + 1, n - 1)
        print(f"  [STEP] {triggered}: {idx} -> {new_idx}")
        return new_idx
    elif triggered == "step-back":
        new_idx = max((idx or 0) - 1, 0)
        print(f"  [STEP] {triggered}: {idx} -> {new_idx}")
        return new_idx
    return no_update


# ── Play/Pause toggle ──────────────────────────────────────────────────

@app.callback(
    [Output("play-interval", "disabled"), Output("play-btn", "children")],
    Input("play-btn", "n_clicks"),
    State("play-interval", "disabled"),
    prevent_initial_call=True)
def toggle_play(n, disabled):
    if disabled:
        return False, "\u23f8 Pause"
    return True, "\u25b6 Play"


# ── Cycle indicator ─────────────────────────────────────────────────────

@app.callback(
    Output("cycle-indicator", "children"),
    [Input("cycle-idx", "data"), Input("sim-data", "data")])
def update_indicator(idx, data):
    if not data or not data.get('step'):
        return "No simulation yet"
    n = len(data['step'])
    return f"Cycle {idx + 1} / {n}"


# ── Tab content router ──────────────────────────────────────────────────

@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "value"), Input("sim-data", "data"),
     Input("cycle-idx", "data"),
     Input("k-cardiac", "value"), Input("lambda-cardiac", "value"),
     Input("k-renal", "value"), Input("lambda-renal", "value"),
     Input("alpha-c", "value"), Input("alpha-r", "value"),
     Input("Z-stress", "value"), Input("months-per-cycle", "value"),
     Input("n-cycles", "value"), Input("Kf-init", "value")])
def render_tab(tab, data, idx, kc, lc, kr, lr, ac, ar, z, mpc, n_cyc, kfi):
    if tab == "tab-1":
        return render_message_passing(data, idx, mpc)
    elif tab == "tab-2":
        return render_deterioration_curves(data, idx, mpc)
    elif tab == "tab-3":
        return render_crash_boundary(data, kfi, n_cyc)
    elif tab == "tab-4":
        return render_exp_vs_numerical(kc, lc, kr, lr, ac, ar, z, mpc, n_cyc)
    return html.Div("Select a tab")


# ═══════════════════════════════════════════════════════════════════════
# TAB 1: MESSAGE PASSING READOUT
# ═══════════════════════════════════════════════════════════════════════

def _safe(data, key, idx, fmt=".1f"):
    """Safely format a value from the data dict, returning '---' for None."""
    try:
        v = data[key][idx]
        if v is None:
            return "---"
        return format(v, fmt)
    except (KeyError, IndexError, TypeError):
        return "---"


def render_message_passing(data, idx, mpc):
    if not data or not data.get('step'):
        return html.Div([
            html.H3("Message Passing Readout", style={"color": C_MSG}),
            html.P("Click 'Run' to start the simulation.", style={"color": C_TEXT}),
            _render_diagram_placeholder(),
        ])

    n = len(data['step'])
    idx = min(idx or 0, n - 1)

    # Left: diagram; Right: log
    return html.Div([
        html.Div([
            # Diagram
            dcc.Graph(figure=_build_diagram(data, idx), config={"displayModeBar": False},
                      style={"height": "350px"}),
            html.Hr(style={"borderColor": C_ACCENT}),
            # Log for current cycle
            _build_cycle_log(data, idx, mpc),
        ], style={"display": "flex", "flexDirection": "column"}),
    ])


def _render_diagram_placeholder():
    fig = go.Figure()
    fig.update_layout(template=DARK_TEMPLATE, paper_bgcolor=C_BG, plot_bgcolor=C_BG,
                      height=300, margin=dict(l=20, r=20, t=20, b=20),
                      xaxis=dict(visible=False, range=[0, 10]),
                      yaxis=dict(visible=False, range=[0, 6]))
    fig.add_annotation(x=2.5, y=3, text="\u2764\ufe0f HEART", showarrow=False,
                       font=dict(size=20, color=C_HEART))
    fig.add_annotation(x=7.5, y=3, text="\U0001fac0 KIDNEY", showarrow=False,
                       font=dict(size=20, color=C_KIDNEY))
    fig.add_annotation(x=5, y=4.5, text="MAP, CO, CVP \u2192", showarrow=False,
                       font=dict(size=12, color=C_MSG))
    fig.add_annotation(x=5, y=1.5, text="\u2190 V_blood, SVR, GFR", showarrow=False,
                       font=dict(size=12, color=C_MSG))
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "300px"})


def _build_diagram(data, idx):
    fig = go.Figure()
    fig.update_layout(
        template=DARK_TEMPLATE, paper_bgcolor=C_BG, plot_bgcolor=C_BG,
        height=340, margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(visible=False, range=[0, 10]),
        yaxis=dict(visible=False, range=[0, 7]),
        title=dict(text=f"Cycle {idx + 1}: Message Exchange",
                   font=dict(color=C_MSG, size=14)))

    # Health gradient: green (healthy) -> red (sick)
    d_c = data['D_cardiac'][idx]
    d_r = data['D_renal'][idx]
    h_color = f"rgb({int(100 + 155 * d_c)}, {int(200 * (1 - d_c))}, {int(100 * (1 - d_c))})"
    k_color = f"rgb({int(100 + 155 * d_r)}, {int(200 * (1 - d_r))}, {int(100 * (1 - d_r))})"

    # Nodes
    fig.add_trace(go.Scatter(x=[2.5], y=[3.5], mode="markers+text",
                             marker=dict(size=60, color=h_color, line=dict(width=2, color="white")),
                             text=["HEART"], textposition="middle center",
                             textfont=dict(size=13, color="white"), showlegend=False))
    fig.add_trace(go.Scatter(x=[7.5], y=[3.5], mode="markers+text",
                             marker=dict(size=60, color=k_color, line=dict(width=2, color="white")),
                             text=["KIDNEY"], textposition="middle center",
                             textfont=dict(size=13, color="white"), showlegend=False))

    # H->K arrow (top)
    h2k_text = (f"MAP={_safe(data,'h2k_MAP',idx)}  "
                f"CO={_safe(data,'h2k_CO',idx,'.2f')}  "
                f"CVP={_safe(data,'h2k_Pven',idx)}")
    fig.add_annotation(x=7, y=5.5, ax=3, ay=5.5,
                       showarrow=True, arrowhead=2, arrowsize=1.5,
                       arrowwidth=2, arrowcolor=C_HEART)
    fig.add_annotation(x=5, y=6.0, text=f"H\u2192K: {h2k_text}",
                       showarrow=False, font=dict(size=10, color=C_HEART))

    # K->H arrow (bottom)
    k2h_text = (f"V={_safe(data,'k2h_Vblood',idx,'.0f')}  "
                f"SVR={_safe(data,'k2h_SVR',idx,'.3f')}  "
                f"GFR={_safe(data,'k2h_GFR',idx)}")
    fig.add_annotation(x=3, y=1.5, ax=7, ay=1.5,
                       showarrow=True, arrowhead=2, arrowsize=1.5,
                       arrowwidth=2, arrowcolor=C_KIDNEY)
    fig.add_annotation(x=5, y=0.8, text=f"K\u2192H: {k2h_text}",
                       showarrow=False, font=dict(size=10, color=C_KIDNEY))

    return fig


def _build_cycle_log(data, idx, mpc):
    """Build the terminal-style step-by-step log for one cycle."""
    idx = idx or 0
    month = (idx + 1) * mpc
    s = data

    def val(key, fmt=".1f"):
        try:
            v = s[key][idx]
            return format(v, fmt) if v is not None else "---"
        except (KeyError, IndexError, TypeError):
            return "---"

    def delta_str(key):
        try:
            c = s[key][idx]
            p = s[key][idx - 1] if idx > 0 else None
            if p is None or c is None:
                return ""
            d = c - p
            sign = "+" if d >= 0 else ""
            return f" ({sign}{d:.1f})"
        except (KeyError, IndexError, TypeError):
            return ""

    log_lines = []
    log_lines.append(f"{'=' * 60}")
    log_lines.append(f"  CYCLE {idx + 1}  |  Month {month}")
    log_lines.append(f"  Weibull damage:  cardiac D={val('D_cardiac','.4f')}"
                     f"   renal D={val('D_renal','.4f')}")
    log_lines.append(f"  Parameters:  k1={val('k1_scale','.3f')}"
                     f"   Kf={val('Kf_scale','.3f')}")
    log_lines.append(f"{'=' * 60}")
    log_lines.append("")

    # ── PHASE A: Heart adjusts and deposits message ──────────
    log_lines.append("  PHASE A: HEART")
    log_lines.append(f"  {'─' * 56}")
    log_lines.append("")
    log_lines.append("    Heart equalizes to steady state:")
    log_lines.append(f"    +{'=' * 46}+")
    log_lines.append(f"    | MAP:   {val('MAP'):>7s} mmHg  {delta_str('MAP'):>16s}   |")
    log_lines.append(f"    | CO:    {val('CO','.2f'):>7s} L/min {delta_str('CO'):>16s}   |")
    log_lines.append(f"    | CVP:   {val('Pven'):>7s} mmHg  {delta_str('Pven'):>16s}   |")
    log_lines.append(f"    | EF:    {val('EF'):>7s} %     {delta_str('EF'):>16s}   |")
    log_lines.append(f"    | SBP:   {val('SBP'):>7s} mmHg                     |")
    log_lines.append(f"    | DBP:   {val('DBP'):>7s} mmHg                     |")
    log_lines.append(f"    | LVEDP: {val('LVEDP'):>7s} mmHg  {delta_str('LVEDP'):>16s}   |")
    log_lines.append(f"    +{'=' * 46}+")
    log_lines.append("")
    log_lines.append("    Heart deposits message --> Kidney:")
    log_lines.append(f"    +-- HeartToKidneyMessage ----------------+")
    log_lines.append(f"    |  MAP  = {val('h2k_MAP'):>8s} mmHg               |")
    log_lines.append(f"    |  CO   = {val('h2k_CO','.2f'):>8s} L/min              |")
    log_lines.append(f"    |  Pven = {val('h2k_Pven'):>8s} mmHg               |")
    log_lines.append(f"    +-------------------------------------------+")
    log_lines.append("")

    # ── PHASE B: Kidney adjusts and deposits message ─────────
    log_lines.append("  PHASE B: KIDNEY")
    log_lines.append(f"  {'─' * 56}")
    log_lines.append("")
    log_lines.append("    Kidney reads heart message and adjusts:")
    log_lines.append(f"    +{'=' * 46}+")
    log_lines.append(f"    | GFR:     {val('GFR'):>7s} mL/min {delta_str('GFR'):>14s}   |")
    log_lines.append(f"    | V_blood: {val('V_blood','.0f'):>7s} mL    {delta_str('V_blood'):>14s}   |")
    log_lines.append(f"    | P_glom:  {val('P_glom'):>7s} mmHg                   |")
    log_lines.append(f"    | Na_excr: {val('Na_excr','.0f'):>7s} mEq/day                |")
    log_lines.append(f"    +{'=' * 46}+")
    log_lines.append("")
    log_lines.append("    Kidney deposits message --> Heart:")
    log_lines.append(f"    +-- KidneyToHeartMessage ----------------+")
    log_lines.append(f"    |  V_blood   = {val('k2h_Vblood','.0f'):>8s} mL            |")
    log_lines.append(f"    |  SVR_ratio = {val('k2h_SVR','.3f'):>8s}               |")
    log_lines.append(f"    |  GFR       = {val('k2h_GFR'):>8s} mL/min        |")
    log_lines.append(f"    +-------------------------------------------+")
    log_lines.append("")
    log_lines.append(f"  {'=' * 60}")
    log_lines.append(f"  End of Cycle {idx + 1}")

    return html.Div([
        html.H4(f"Cycle {idx + 1} Detail", style={"color": C_MSG, "marginBottom": "5px"}),
        html.Pre(
            "\n".join(log_lines),
            style={"backgroundColor": "#0d1117", "color": "#c9d1d9",
                   "padding": "15px", "borderRadius": "8px",
                   "fontSize": "12px", "lineHeight": "1.5",
                   "fontFamily": "'Fira Code', 'Consolas', monospace",
                   "maxHeight": "800px", "overflowY": "auto",
                   "border": f"1px solid {C_ACCENT}"}
        ),
    ])


# ═══════════════════════════════════════════════════════════════════════
# TAB 2: DETERIORATION CURVES
# ═══════════════════════════════════════════════════════════════════════

def render_deterioration_curves(data, idx, mpc):
    if not data or not data.get('step'):
        return html.Div([
            html.H3("Deterioration Curves", style={"color": C_MSG}),
            html.P("Click 'Run' to generate curves.", style={"color": C_TEXT}),
        ])

    n = len(data['step'])
    months = [s * mpc for s in data['step']]
    idx = min(idx or 0, n - 1)

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            "H\u2192K Messages: MAP, CO, CVP", "K\u2192H Messages: V_blood, SVR, GFR",
            "Kf_scale (renal filtration)", "k1_scale (cardiac stiffness)",
            "LV Filling Pressure (LVEDP)", "Ejection Fraction (EF)",
            "Weibull Damage: D_renal(t)", "Weibull Damage: D_cardiac(t)",
        ],
        vertical_spacing=0.08,
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{}, {}], [{}, {}], [{}, {}]])

    # Row 1 Left: H->K messages (MAP + CO on secondary y)
    fig.add_trace(go.Scatter(x=months, y=data['h2k_MAP'], name="MAP (mmHg)",
                             line=dict(color=C_HEART, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=data['h2k_Pven'], name="CVP (mmHg)",
                             line=dict(color="#ff9ff3", width=2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=data['h2k_CO'], name="CO (L/min)",
                             line=dict(color="#ffa502", width=2)),
                  row=1, col=1, secondary_y=True)

    # Row 1 Right: K->H messages (V_blood + GFR on secondary y)
    fig.add_trace(go.Scatter(x=months, y=data['k2h_Vblood'], name="V_blood (mL)",
                             line=dict(color=C_KIDNEY, width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=months, y=data['k2h_SVR'], name="SVR ratio",
                             line=dict(color="#48dbfb", width=2, dash="dash")), row=1, col=2)
    fig.add_trace(go.Scatter(x=months, y=data['k2h_GFR'], name="GFR (mL/min)",
                             line=dict(color="#0abde3", width=2)),
                  row=1, col=2, secondary_y=True)

    # Row 2: Kf and k1
    fig.add_trace(go.Scatter(x=months, y=data['Kf_scale'], name="Kf_scale",
                             line=dict(color=C_KIDNEY, width=2.5),
                             fill="tozeroy", fillcolor="rgba(78,205,196,0.15)"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=months, y=data['k1_scale'], name="k1_scale",
                             line=dict(color=C_HEART, width=2.5),
                             fill="tozeroy", fillcolor="rgba(255,107,107,0.15)"),
                  row=2, col=2)

    # Row 3: LVEDP and EF
    fig.add_trace(go.Scatter(x=months, y=data['LVEDP'], name="LVEDP (mmHg)",
                             line=dict(color="#e056fd", width=2.5)), row=3, col=1)
    # LVEDP threshold
    fig.add_hline(y=16, row=3, col=1, line=dict(color="gray", dash="dash", width=1),
                  annotation_text="Elevated (16)", annotation_font=dict(color="gray", size=9))

    fig.add_trace(go.Scatter(x=months, y=data['EF'], name="EF (%)",
                             line=dict(color=C_HEART, width=2.5)), row=3, col=2)
    fig.add_hline(y=50, row=3, col=2, line=dict(color="gray", dash="dash", width=1),
                  annotation_text="HFpEF boundary (50%)", annotation_font=dict(color="gray", size=9))

    # Row 4: Weibull damage
    fig.add_trace(go.Scatter(x=months, y=data['D_renal'], name="D_renal",
                             line=dict(color=C_KIDNEY, width=2.5),
                             fill="tozeroy", fillcolor="rgba(78,205,196,0.2)"),
                  row=4, col=1)
    fig.add_trace(go.Scatter(x=months, y=data['D_cardiac'], name="D_cardiac",
                             line=dict(color=C_HEART, width=2.5),
                             fill="tozeroy", fillcolor="rgba(255,107,107,0.2)"),
                  row=4, col=2)

    # Current cycle highlight
    current_month = months[idx]
    for row in range(1, 5):
        for col in [1, 2]:
            fig.add_vline(x=current_month, row=row, col=col,
                          line=dict(color=C_MSG, width=2, dash="dot"))

    # GFR threshold on K->H panel
    gfr_vals = [v for v in data['k2h_GFR'] if v is not None]
    if gfr_vals and max(gfr_vals) > 60:
        fig.add_hline(y=60, row=1, col=2, line=dict(color="gray", dash="dash", width=1),
                      annotation_text="CKD G3 (60)", annotation_font=dict(color="gray", size=9),
                      secondary_y=True)

    fig.update_layout(
        template=DARK_TEMPLATE, paper_bgcolor=C_BG, plot_bgcolor=C_BG,
        height=900, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, font=dict(size=9)),
        margin=dict(l=50, r=30, t=40, b=40))

    # X-axis labels
    for i in range(1, 5):
        for j in [1, 2]:
            fig.update_xaxes(title_text="Month" if i == 4 else "", row=i, col=j)

    return dcc.Graph(figure=fig, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════════════
# TAB 3: CRASH BOUNDARY
# ═══════════════════════════════════════════════════════════════════════

def render_crash_boundary(data, kfi, n_cyc):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Heart Convergence vs \u0394Kf Step Size",
                                        "Stiffness Sweep: HFpEF Progression"])

    # Left: Crash boundary
    if CRASH_DATA:
        delta_kf = CRASH_DATA.get("delta_Kf", [])
        converged = CRASH_DATA.get("cycle_completed", [])
        final_gfr = CRASH_DATA.get("final_GFR", [])

        colors = ["#2ecc71" if c else "#e74c3c" for c in converged]
        fig.add_trace(go.Scatter(
            x=delta_kf, y=final_gfr, mode="markers+lines",
            marker=dict(color=colors, size=6),
            line=dict(color=C_KIDNEY, width=1.5),
            name="Final GFR"), row=1, col=1)

        # Current sim step size marker
        if n_cyc and kfi:
            step_size = round((1.0 - kfi) / max(n_cyc, 1), 3)
            fig.add_vline(x=step_size, row=1, col=1,
                          line=dict(color=C_MSG, width=2, dash="dash"),
                          annotation_text=f"Current: {step_size:.3f}",
                          annotation_font=dict(color=C_MSG))

        # Find crash boundary
        crash_idx = next((i for i, c in enumerate(converged) if not c), len(converged))
        if crash_idx < len(delta_kf):
            fig.add_vrect(x0=delta_kf[crash_idx], x1=max(delta_kf),
                          row=1, col=1, fillcolor="rgba(231,76,60,0.15)",
                          line_width=0)
    else:
        fig.add_annotation(x=0.5, y=0.5, text="No crash boundary data",
                           showarrow=False, row=1, col=1)

    fig.update_xaxes(title_text="\u0394Kf per cycle", row=1, col=1)
    fig.update_yaxes(title_text="Final GFR (mL/min)", row=1, col=1)

    # Right: Stiffness sweep
    if STIFFNESS_DATA:
        k1s = STIFFNESS_DATA.get("k1_scale", [])
        for key, color, name in [("EF", C_HEART, "EF (%)"),
                                  ("CO", "#ffa502", "CO (L/min)")]:
            vals = STIFFNESS_DATA.get(key, [])
            if vals:
                fig.add_trace(go.Scatter(
                    x=k1s, y=vals, name=name,
                    line=dict(color=color, width=2)), row=1, col=2)

        # LVEDP on secondary axis (approximated from Pven if available)
        pven = STIFFNESS_DATA.get("Pven", [])
        if pven:
            fig.add_trace(go.Scatter(
                x=k1s, y=pven, name="Pven (mmHg)",
                line=dict(color="#e056fd", width=2, dash="dash")), row=1, col=2)
    else:
        fig.add_annotation(x=0.5, y=0.5, text="No stiffness sweep data",
                           showarrow=False, row=1, col=2)

    fig.update_xaxes(title_text="k1_scale (stiffness)", row=1, col=2)
    fig.update_yaxes(title_text="Value", row=1, col=2)

    fig.update_layout(template=DARK_TEMPLATE, paper_bgcolor=C_BG, plot_bgcolor=C_BG,
                      height=450, margin=dict(l=50, r=30, t=40, b=40))

    return html.Div([
        html.H3("Heart Crash Boundary & Stiffness Sweep", style={"color": C_MSG}),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        html.P("Green = converged, Red = crashed. Yellow dashed = your current \u0394Kf.",
               style={"color": C_TEXT, "fontSize": "11px"}),
    ])


# ═══════════════════════════════════════════════════════════════════════
# TAB 4: EXPONENTIAL VS NUMERICAL
# ═══════════════════════════════════════════════════════════════════════

def render_exp_vs_numerical(kc, lc, kr, lr, ac, ar, z, mpc, n_cyc):
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[
                            "eGFR: Exponential vs Numerical",
                            "Kf: Exponential Model",
                            "Residual (Numerical - Exponential)",
                            "Weibull Overlay (your settings)",
                        ],
                        vertical_spacing=0.12)

    if EXP_NUM_DATA:
        t = EXP_NUM_DATA.get("time_months", [])
        for rate in [1, 2, 3, 5]:
            exp_key = f"rate_{rate}_eGFR_exp"
            num_key = f"rate_{rate}_GFR_num"
            kf_key = f"rate_{rate}_Kf_exp"

            exp_vals = EXP_NUM_DATA.get(exp_key, [])
            num_vals = EXP_NUM_DATA.get(num_key, [])
            kf_vals = EXP_NUM_DATA.get(kf_key, [])

            if exp_vals:
                fig.add_trace(go.Scatter(
                    x=t, y=exp_vals, name=f"Exp rate={rate}",
                    line=dict(dash="dash", width=1.5)), row=1, col=1)
            if num_vals:
                fig.add_trace(go.Scatter(
                    x=t, y=num_vals, name=f"Num rate={rate}",
                    line=dict(width=2)), row=1, col=1)
            if kf_vals:
                fig.add_trace(go.Scatter(
                    x=t, y=kf_vals, name=f"Kf rate={rate}",
                    line=dict(width=1.5)), row=1, col=2)

            # Residual
            if exp_vals and num_vals and len(exp_vals) == len(num_vals):
                resid = [n - e for n, e in zip(num_vals, exp_vals)]
                fig.add_trace(go.Scatter(
                    x=t, y=resid, name=f"Resid rate={rate}",
                    line=dict(width=1.5)), row=2, col=1)

    # Weibull overlay (bottom right)
    lam_r_eff = effective_lambda(lr, ar, z)
    lam_c_eff = effective_lambda(lc, ac, z)
    t_w = np.linspace(0, max(n_cyc * mpc, 12), 50)
    D_r = weibull_damage(t_w, lam_r_eff, kr)
    D_c = weibull_damage(t_w, lam_c_eff, kc)
    Kf_w = 0.9 * (1.0 - D_r * 0.9)
    eGFR_w = 90 * Kf_w  # simplified

    fig.add_trace(go.Scatter(x=t_w, y=eGFR_w, name="Weibull eGFR",
                             line=dict(color=C_KIDNEY, width=2.5)), row=2, col=2)
    fig.add_trace(go.Scatter(x=t_w, y=D_r * 100, name="D_renal (%)",
                             line=dict(color=C_KIDNEY, width=1.5, dash="dash")), row=2, col=2)
    fig.add_trace(go.Scatter(x=t_w, y=D_c * 100, name="D_cardiac (%)",
                             line=dict(color=C_HEART, width=1.5, dash="dash")), row=2, col=2)

    fig.update_layout(template=DARK_TEMPLATE, paper_bgcolor=C_BG, plot_bgcolor=C_BG,
                      height=650, margin=dict(l=50, r=30, t=40, b=40),
                      legend=dict(font=dict(size=9)))

    for r in [1, 2]:
        for c in [1, 2]:
            fig.update_xaxes(title_text="Months", row=r, col=c)

    fig.update_yaxes(title_text="eGFR / GFR", row=1, col=1)
    fig.update_yaxes(title_text="Kf_scale", row=1, col=2)
    fig.update_yaxes(title_text="Residual (mL/min)", row=2, col=1)
    fig.update_yaxes(title_text="eGFR / Damage %", row=2, col=2)

    return html.Div([
        html.H3("Exponential vs Numerical Comparison", style={"color": C_MSG}),
        html.P("Dashed = exponential analytical, Solid = Hallow numerical. "
               "Bottom-right shows Weibull model at your current slider settings.",
               style={"color": C_TEXT, "fontSize": "11px"}),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
    ])


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Starting Cardiorenal Dashboard ({'LIVE' if LIVE_MODE else 'DEMO'} mode)")
    print("Open http://localhost:8050")
    app.run(debug=True, port=8050, host="0.0.0.0")
