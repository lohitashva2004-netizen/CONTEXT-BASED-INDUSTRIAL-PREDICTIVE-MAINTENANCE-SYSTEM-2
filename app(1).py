"""
============================================================
  Predictive Maintenance Dashboard — Induction Motor
  Stack : Python Dash + Plotly + paho-mqtt + Random Forest
  Theme : Dark Industrial
  Deploy: Render (gunicorn app:server)
============================================================

  Files needed in same directory:
    - app.py               (this file)
    - machine_model.pkl    (your trained Random Forest model)
    - requirements.txt

  Install:
    pip install dash dash-bootstrap-components plotly paho-mqtt gunicorn joblib scikit-learn

  Run locally:
    python app.py

  Render start command:
    gunicorn app:server --bind 0.0.0.0:$PORT
============================================================
"""

import threading
import json
import os
import numpy as np
from collections import deque
from datetime import datetime

import joblib
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import paho.mqtt.client as mqtt

# ─────────────────────────────────────────────────────────────
# LOAD RANDOM FOREST MODEL
# Features order: vibration, temperature, current
# Labels: Normal / Warning / Critical
# ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "machine_model.pkl")
rf_model   = joblib.load(MODEL_PATH)
print("✅ Random Forest model loaded. Classes:", rf_model.classes_)

# How many future readings to predict ahead
PREDICT_WINDOW = 10

# ─────────────────────────────────────────────────────────────
# MQTT CONFIG
# ─────────────────────────────────────────────────────────────
MQTT_BROKER    = "922e3176f2ba4e7ba2c3b4b2c3bcea3e.s1.eu.hivemq.cloud"
MQTT_PORT      = 8883
MQTT_USER      = "esp32user"
MQTT_PASS      = "7E&2p7chx&DddKf"
MQTT_TOPIC_ALL = "motor/all"

# ─────────────────────────────────────────────────────────────
# SHARED DATA STORE
# ─────────────────────────────────────────────────────────────
MAX_POINTS = 60

sensor_data = {
    "time":        deque(maxlen=MAX_POINTS),
    "vibration":   deque(maxlen=MAX_POINTS),
    "temperature": deque(maxlen=MAX_POINTS),
    "current":     deque(maxlen=MAX_POINTS),
    "status":      deque(maxlen=MAX_POINTS),
}

latest = {
    "vibration":   0,
    "temperature": 0.0,
    "current":     0.0,
    "status":      "Normal",
}

# ML prediction results
ml_result = {
    "current_prediction":  "Normal",
    "current_confidence":  0.0,
    "future_prediction":   "Normal",
    "future_confidence":   0.0,
    "fault_probability":   0.0,
    "class_probabilities": {"Normal": 0.0, "Warning": 0.0, "Critical": 0.0},
    "trend":               "Stable →",
}

# Fault history log
fault_log = deque(maxlen=50)

# MQTT connection flag
mqtt_connected = {"value": False}


# ─────────────────────────────────────────────────────────────
# RANDOM FOREST PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────
def run_prediction(vibration, temperature, current):
    # ── Current reading prediction ──────────────────────────
    X_now = pd.DataFrame(
        [[vibration, temperature, current]],
        columns=["vibration", "temperature", "current"]
    )
    pred_now  = rf_model.predict(X_now)[0]
    proba_now = rf_model.predict_proba(X_now)[0]
    classes   = list(rf_model.classes_)  # ['Critical', 'Normal', 'Warning']

    proba_dict = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, proba_now)}
    confidence = round(float(max(proba_now)) * 100, 1)
    fault_prob = round(proba_dict.get("Warning", 0) + proba_dict.get("Critical", 0), 1)

    # ── Trend analysis ───────────────────────────────────────
    temps = list(sensor_data["temperature"])
    currs = list(sensor_data["current"])
    vibs  = list(sensor_data["vibration"])

    trend = "Stable →"
    if len(temps) >= 10:
        recent  = np.mean(temps[-5:])  + np.mean(currs[-5:])  * 10 + np.mean(vibs[-5:])  * 2
        earlier = np.mean(temps[-10:-5]) + np.mean(currs[-10:-5]) * 10 + np.mean(vibs[-10:-5]) * 2
        if   recent > earlier * 1.05: trend = "Rising ↑"
        elif recent < earlier * 0.95: trend = "Falling ↓"
        else:                          trend = "Stable →"

    # ── Future prediction via linear extrapolation ───────────
    def extrapolate(values, steps):
        arr = np.array(values)
        if len(arr) < 3:
            return float(arr[-1]) if len(arr) else 0.0
        x      = np.arange(len(arr))
        coeffs = np.polyfit(x[-10:], arr[-10:], 1)
        return max(0.0, float(np.polyval(coeffs, len(arr) - 1 + steps)))

    n = len(sensor_data["temperature"])
    future_temp = extrapolate(list(sensor_data["temperature"]), PREDICT_WINDOW) if n >= 3 else temperature
    future_curr = extrapolate(list(sensor_data["current"]),     PREDICT_WINDOW) if n >= 3 else current
    future_vib  = extrapolate(list(sensor_data["vibration"]),   PREDICT_WINDOW) if n >= 3 else vibration

    X_future    = pd.DataFrame(
        [[future_vib, future_temp, future_curr]],
        columns=["vibration", "temperature", "current"]
    )
    pred_future  = rf_model.predict(X_future)[0]
    proba_future = rf_model.predict_proba(X_future)[0]
    conf_future  = round(float(max(proba_future)) * 100, 1)

    # ── Write results ────────────────────────────────────────
    ml_result["current_prediction"]  = pred_now
    ml_result["current_confidence"]  = confidence
    ml_result["future_prediction"]   = pred_future
    ml_result["future_confidence"]   = conf_future
    ml_result["fault_probability"]   = fault_prob
    ml_result["class_probabilities"] = proba_dict
    ml_result["trend"]               = trend


# ─────────────────────────────────────────────────────────────
# MQTT LISTENER
# ─────────────────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        mqtt_connected["value"] = True
        client.subscribe(MQTT_TOPIC_ALL)
        print("✅ MQTT Connected — subscribed to", MQTT_TOPIC_ALL)
    else:
        mqtt_connected["value"] = False
        print(f"❌ MQTT failed. RC={rc}")

def on_disconnect(client, userdata, rc):
    mqtt_connected["value"] = False

def on_message(client, userdata, msg):
    try:
        payload     = json.loads(msg.payload.decode())
        prev_status = latest["status"]

        latest["vibration"]   = payload.get("vibration",   0)
        latest["temperature"] = round(payload.get("temperature", 0.0), 1)
        latest["current"]     = round(payload.get("current",     0.0), 2)
        latest["status"]      = payload.get("status", "Normal")

        now = datetime.now().strftime("%H:%M:%S")
        sensor_data["time"].append(now)
        sensor_data["vibration"].append(latest["vibration"])
        sensor_data["temperature"].append(latest["temperature"])
        sensor_data["current"].append(latest["current"])
        sensor_data["status"].append(latest["status"])

        # Run RF prediction on every new reading
        run_prediction(latest["vibration"], latest["temperature"], latest["current"])

        # Log fault entry on status change
        if latest["status"] in ("Warning", "Critical") and latest["status"] != prev_status:
            fault_log.appendleft({
                "time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status":     latest["status"],
                "rf_pred":    ml_result["current_prediction"],
                "temp":       latest["temperature"],
                "curr":       latest["current"],
                "vib":        latest["vibration"],
                "fault_prob": ml_result["fault_probability"],
            })

    except Exception as e:
        print(f"MQTT message error: {e}")

def start_mqtt():
    client = mqtt.Client(client_id="Dashboard_RF_001", transport="websockets")
    client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.tls_set()
    client.ws_set_options(path="/mqtt")
    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_message    = on_message
    client.connect(MQTT_BROKER, 8884, keepalive=60)
    client.loop_forever()

mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
mqtt_thread.start()


# ─────────────────────────────────────────────────────────────
# DASH APP INIT
# ─────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap",
    ],
    title="Motor PdM Dashboard",
)
server = app.server  # for gunicorn

# ─────────────────────────────────────────────────────────────
# STYLE CONSTANTS
# ─────────────────────────────────────────────────────────────
COLORS = {
    "bg":       "#0a0c10",
    "panel":    "#0f1318",
    "border":   "#1e2530",
    "accent":   "#00e5ff",
    "green":    "#00e676",
    "yellow":   "#ffea00",
    "red":      "#ff1744",
    "text":     "#c8d6e5",
    "subtext":  "#546e7a",
    "gridline": "#1a2030",
}
FONT_MONO = "'Share Tech Mono', monospace"
FONT_BODY = "'Exo 2', sans-serif"

PANEL_STYLE = {
    "background":   COLORS["panel"],
    "border":       f"1px solid {COLORS['border']}",
    "borderRadius": "4px",
    "padding":      "18px",
}
SECTION_LABEL = {
    "fontFamily":    FONT_MONO,
    "fontSize":      "10px",
    "letterSpacing": "3px",
    "color":         COLORS["subtext"],
    "marginBottom":  "12px",
}


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def status_color(s):
    return {"Normal": COLORS["green"], "Warning": COLORS["yellow"],
            "Critical": COLORS["red"]}.get(s, COLORS["subtext"])

def td_style(color):
    return {"fontFamily": FONT_MONO, "fontSize": "11px",
            "color": color, "padding": "7px 12px"}

def make_gauge(value, title, unit, max_val, warn, crit):
    color = COLORS["green"]
    if value >= crit:   color = COLORS["red"]
    elif value >= warn: color = COLORS["yellow"]
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value,
        number= {"suffix": unit, "font": {"color": color, "family": FONT_MONO, "size": 26}},
        title = {"text": title,  "font": {"color": COLORS["text"], "family": FONT_BODY, "size": 12}},
        gauge = {
            "axis":       {"range": [0, max_val], "tickcolor": COLORS["subtext"],
                           "tickfont": {"color": COLORS["subtext"], "size": 9}},
            "bar":        {"color": color, "thickness": 0.25},
            "bgcolor":    COLORS["bg"],
            "borderwidth": 1, "bordercolor": COLORS["border"],
            "steps": [
                {"range": [0,    warn],    "color": "#0d1520"},
                {"range": [warn, crit],    "color": "#1a1500"},
                {"range": [crit, max_val], "color": "#1a0508"},
            ],
            "threshold": {"line": {"color": COLORS["red"], "width": 2},
                          "thickness": 0.75, "value": crit},
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=40, b=10, l=20, r=20), height=200)
    return fig

def make_timeseries(times, values, unit, color, warn, crit):
    times, values = list(times), list(values)
    fig = go.Figure()
    fig.add_hrect(y0=warn, y1=crit,        fillcolor=COLORS["yellow"], opacity=0.06, line_width=0)
    fig.add_hrect(y0=crit, y1=crit * 1.5, fillcolor=COLORS["red"],    opacity=0.06, line_width=0)
    fig.add_trace(go.Scatter(x=times, y=values, mode="lines",
                             line=dict(color=color, width=2),
                             fill="tozeroy", fillcolor=color + "12"))
    if times:
        fig.add_hline(y=warn, line_dash="dot", line_color=COLORS["yellow"], line_width=1, opacity=0.5)
        fig.add_hline(y=crit, line_dash="dot", line_color=COLORS["red"],    line_width=1, opacity=0.5)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=30, l=50, r=20), height=160, showlegend=False,
        xaxis=dict(showgrid=True, gridcolor=COLORS["gridline"],
                   tickfont=dict(color=COLORS["subtext"], size=9, family=FONT_MONO),
                   tickangle=-30, nticks=8),
        yaxis=dict(showgrid=True, gridcolor=COLORS["gridline"],
                   tickfont=dict(color=COLORS["subtext"], size=9, family=FONT_MONO),
                   title=dict(text=unit, font=dict(color=COLORS["subtext"], size=10))),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={"backgroundColor": COLORS["bg"], "minHeight": "100vh",
           "fontFamily": FONT_BODY, "color": COLORS["text"]},
    children=[

        dcc.Interval(id="interval", interval=2000, n_intervals=0),

        # ── HEADER ───────────────────────────────────────────
        html.Div(style={
            "background": "linear-gradient(90deg, #0d1520 0%, #0a0c10 60%)",
            "borderBottom": f"1px solid {COLORS['border']}",
            "padding": "14px 28px", "display": "flex",
            "alignItems": "center", "justifyContent": "space-between",
        }, children=[
            html.Div([
                html.Span("⚙ ", style={"color": COLORS["accent"], "fontSize": "20px"}),
                html.Span("PREDICTIVE MAINTENANCE SYSTEM", style={
                    "fontFamily": FONT_MONO, "fontSize": "15px",
                    "letterSpacing": "3px", "color": COLORS["accent"],
                }),
                html.Span(" — INDUCTION MOTOR", style={
                    "fontFamily": FONT_MONO, "fontSize": "11px",
                    "color": COLORS["subtext"], "letterSpacing": "2px",
                }),
            ]),
            html.Div([
                html.Span(id="mqtt-dot", style={
                    "display": "inline-block", "width": "9px", "height": "9px",
                    "borderRadius": "50%", "background": COLORS["red"], "marginRight": "7px",
                }),
                html.Span(id="mqtt-text", style={
                    "fontFamily": FONT_MONO, "fontSize": "11px", "color": COLORS["subtext"],
                }),
                html.Span(id="clock", style={
                    "fontFamily": FONT_MONO, "fontSize": "12px",
                    "color": COLORS["subtext"], "marginLeft": "24px",
                }),
            ]),
        ]),

        html.Div(style={"padding": "20px 24px"}, children=[

            # ── ROW 1: STATUS + ALERTS ────────────────────────
            dbc.Row(style={"marginBottom": "16px"}, children=[
                dbc.Col(width=4, children=[
                    html.Div(style={**PANEL_STYLE, "textAlign": "center",
                                    "position": "relative", "overflow": "hidden"}, children=[
                        html.Div("MACHINE STATUS", style=SECTION_LABEL),
                        html.Div(id="status-badge"),
                        html.Div(id="status-subtext", style={
                            "fontFamily": FONT_BODY, "fontSize": "11px",
                            "color": COLORS["subtext"], "marginTop": "6px",
                        }),
                        html.Div(id="status-glow", style={
                            "position": "absolute", "bottom": "-30px", "left": "50%",
                            "transform": "translateX(-50%)", "width": "80px",
                            "height": "60px", "borderRadius": "50%",
                            "filter": "blur(20px)", "opacity": "0.25",
                        }),
                    ]),
                ]),
                dbc.Col(width=8, children=[
                    html.Div(style={**PANEL_STYLE, "height": "100%"}, children=[
                        html.Div("ACTIVE ALERTS", style=SECTION_LABEL),
                        html.Div(id="alerts-panel"),
                    ]),
                ]),
            ]),

            # ── ROW 2: GAUGES ─────────────────────────────────
            dbc.Row(style={"marginBottom": "16px"}, children=[
                dbc.Col(width=4, children=[html.Div(style=PANEL_STYLE, children=[dcc.Graph(id="gauge-temp",    config={"displayModeBar": False})])]),
                dbc.Col(width=4, children=[html.Div(style=PANEL_STYLE, children=[dcc.Graph(id="gauge-current", config={"displayModeBar": False})])]),
                dbc.Col(width=4, children=[html.Div(style=PANEL_STYLE, children=[dcc.Graph(id="gauge-vib",     config={"displayModeBar": False})])]),
            ]),

            # ── ROW 3: RANDOM FOREST PANEL ────────────────────
            dbc.Row(style={"marginBottom": "16px"}, children=[
                dbc.Col(width=12, children=[
                    html.Div(style=PANEL_STYLE, children=[
                        html.Div("RANDOM FOREST — PREDICTIVE ANALYSIS", style=SECTION_LABEL),
                        dbc.Row([

                            # Current RF prediction
                            dbc.Col(width=3, children=[
                                html.Div(style={
                                    "background": COLORS["bg"], "border": f"1px solid {COLORS['border']}",
                                    "borderRadius": "4px", "padding": "14px", "textAlign": "center",
                                }, children=[
                                    html.Div("CURRENT STATE", style={**SECTION_LABEL, "marginBottom": "6px"}),
                                    html.Div(id="rf-current-pred"),
                                    html.Div(id="rf-current-conf", style={
                                        "fontFamily": FONT_MONO, "fontSize": "11px",
                                        "color": COLORS["subtext"], "marginTop": "4px",
                                    }),
                                ]),
                            ]),

                            # Future prediction
                            dbc.Col(width=3, children=[
                                html.Div(style={
                                    "background": COLORS["bg"], "border": f"1px solid {COLORS['border']}",
                                    "borderRadius": "4px", "padding": "14px", "textAlign": "center",
                                }, children=[
                                    html.Div(f"IN ~{PREDICT_WINDOW} READINGS", style={**SECTION_LABEL, "marginBottom": "6px"}),
                                    html.Div(id="rf-future-pred"),
                                    html.Div(id="rf-future-conf", style={
                                        "fontFamily": FONT_MONO, "fontSize": "11px",
                                        "color": COLORS["subtext"], "marginTop": "4px",
                                    }),
                                ]),
                            ]),

                            # Fault probability
                            dbc.Col(width=3, children=[
                                html.Div(style={
                                    "background": COLORS["bg"], "border": f"1px solid {COLORS['border']}",
                                    "borderRadius": "4px", "padding": "14px", "textAlign": "center",
                                }, children=[
                                    html.Div("FAULT PROBABILITY", style={**SECTION_LABEL, "marginBottom": "6px"}),
                                    html.Div(id="rf-fault-prob"),
                                    html.Div("Warning + Critical chance", style={
                                        "fontFamily": FONT_MONO, "fontSize": "10px",
                                        "color": COLORS["subtext"], "marginTop": "4px",
                                    }),
                                ]),
                            ]),

                            # Sensor trend
                            dbc.Col(width=3, children=[
                                html.Div(style={
                                    "background": COLORS["bg"], "border": f"1px solid {COLORS['border']}",
                                    "borderRadius": "4px", "padding": "14px", "textAlign": "center",
                                }, children=[
                                    html.Div("SENSOR TREND", style={**SECTION_LABEL, "marginBottom": "6px"}),
                                    html.Div(id="rf-trend"),
                                    html.Div("last 10 readings", style={
                                        "fontFamily": FONT_MONO, "fontSize": "10px",
                                        "color": COLORS["subtext"], "marginTop": "4px",
                                    }),
                                ]),
                            ]),

                        ]),

                        # Probability bars
                        html.Div(style={"marginTop": "16px"}, children=[
                            html.Div("CLASS PROBABILITIES", style={**SECTION_LABEL, "marginBottom": "8px"}),
                            html.Div(id="rf-proba-bars"),
                        ]),
                    ]),
                ]),
            ]),

            # ── ROW 4: TIME SERIES ────────────────────────────
            dbc.Row(style={"marginBottom": "16px"}, children=[
                dbc.Col(width=12, children=[
                    html.Div(style=PANEL_STYLE, children=[
                        html.Div("SENSOR HISTORY", style=SECTION_LABEL),
                        dbc.Row([
                            dbc.Col(width=4, children=[
                                html.Div("TEMPERATURE (°C)", style={**SECTION_LABEL, "marginBottom": "4px"}),
                                dcc.Graph(id="graph-temp", config={"displayModeBar": False}),
                            ]),
                            dbc.Col(width=4, children=[
                                html.Div("CURRENT (A)", style={**SECTION_LABEL, "marginBottom": "4px"}),
                                dcc.Graph(id="graph-current", config={"displayModeBar": False}),
                            ]),
                            dbc.Col(width=4, children=[
                                html.Div("VIBRATION (pulses/2s)", style={**SECTION_LABEL, "marginBottom": "4px"}),
                                dcc.Graph(id="graph-vib", config={"displayModeBar": False}),
                            ]),
                        ]),
                    ]),
                ]),
            ]),

            # ── ROW 5: FAULT HISTORY TABLE ────────────────────
            dbc.Row(children=[
                dbc.Col(width=12, children=[
                    html.Div(style=PANEL_STYLE, children=[
                        html.Div("FAULT HISTORY LOG", style=SECTION_LABEL),
                        html.Div(id="fault-table"),
                    ]),
                ]),
            ]),

        ]),

        # ── FOOTER ───────────────────────────────────────────
        html.Div(style={
            "borderTop": f"1px solid {COLORS['border']}",
            "padding": "10px 28px", "display": "flex",
            "justifyContent": "space-between", "alignItems": "center",
        }, children=[
            html.Span("SRM Institute of Science and Technology — Lohitashva V.S",
                      style={"fontFamily": FONT_MONO, "fontSize": "10px", "color": COLORS["subtext"]}),
            html.Span("Context-Based Predictive Maintenance | IoT + Random Forest ML",
                      style={"fontFamily": FONT_MONO, "fontSize": "10px", "color": COLORS["subtext"]}),
        ]),
    ],
)


# ─────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────

@app.callback(Output("clock", "children"), Input("interval", "n_intervals"))
def update_clock(n):
    return datetime.now().strftime("%Y-%m-%d  %H:%M:%S")


@app.callback(
    Output("mqtt-dot",  "style"),
    Output("mqtt-text", "children"),
    Input("interval",   "n_intervals"),
)
def update_mqtt_status(n):
    connected = mqtt_connected["value"]
    color = COLORS["green"] if connected else COLORS["red"]
    return (
        {"display": "inline-block", "width": "9px", "height": "9px",
         "borderRadius": "50%", "marginRight": "7px",
         "background": color, "boxShadow": f"0 0 6px {color}"},
        "MQTT LIVE" if connected else "MQTT DISCONNECTED",
    )


@app.callback(
    Output("status-badge",   "children"),
    Output("status-subtext", "children"),
    Output("status-glow",    "style"),
    Input("interval", "n_intervals"),
)
def update_status(n):
    s     = latest["status"]
    color = status_color(s)
    msgs  = {
        "Normal":   "All parameters within safe limits",
        "Warning":  "One or more parameters approaching threshold",
        "Critical": "IMMEDIATE MAINTENANCE REQUIRED",
    }
    badge = html.Span(s.upper(), style={
        "fontFamily": FONT_MONO, "fontSize": "34px", "fontWeight": "700",
        "letterSpacing": "4px", "color": color,
        "textShadow": f"0 0 20px {color}88",
    })
    glow = {
        "position": "absolute", "bottom": "-30px", "left": "50%",
        "transform": "translateX(-50%)", "width": "80px", "height": "60px",
        "borderRadius": "50%", "filter": "blur(20px)",
        "opacity": "0.3", "background": color,
    }
    return badge, msgs.get(s, ""), glow


@app.callback(Output("alerts-panel", "children"), Input("interval", "n_intervals"))
def update_alerts(n):
    alerts = []
    t, c, v = latest["temperature"], latest["current"], latest["vibration"]

    if t >= 70:   alerts.append(("CRITICAL", f"🌡 Thermal Fault — {t}°C exceeds critical limit (70°C)",    COLORS["red"]))
    elif t >= 55: alerts.append(("WARNING",  f"🌡 High Temperature — {t}°C approaching critical threshold", COLORS["yellow"]))
    if c >= 4.5:  alerts.append(("CRITICAL", f"⚡ Overload Fault — {c}A exceeds critical limit (4.5A)",     COLORS["red"]))
    elif c >= 3:  alerts.append(("WARNING",  f"⚡ High Current — {c}A approaching overload threshold",       COLORS["yellow"]))
    if v >= 7:    alerts.append(("CRITICAL", f"📳 Bearing Fault — {v} pulses exceeds critical limit",       COLORS["red"]))
    elif v >= 3:  alerts.append(("WARNING",  f"📳 Abnormal Vibration — {v} pulses, check motor alignment",  COLORS["yellow"]))

    # RF forecast alert — warns BEFORE threshold breach
    fp = ml_result["future_prediction"]
    if fp in ("Warning", "Critical") and latest["status"] == "Normal":
        fc     = ml_result["future_confidence"]
        fcolor = COLORS["red"] if fp == "Critical" else COLORS["yellow"]
        alerts.append(("ML FORECAST",
                        f"🤖 RF Model predicts {fp.upper()} state in next {PREDICT_WINDOW} readings ({fc}% confidence)",
                        fcolor))

    if not alerts:
        return html.Div("✅  No active alerts — motor operating normally.",
                        style={"fontFamily": FONT_MONO, "fontSize": "12px",
                               "color": COLORS["green"], "padding": "8px 0"})

    return html.Div([
        html.Div(style={
            "display": "flex", "alignItems": "center", "gap": "10px",
            "padding": "8px 12px", "marginBottom": "6px",
            "borderLeft": f"3px solid {color}",
            "background": f"{color}0d", "borderRadius": "2px",
        }, children=[
            html.Span(level, style={
                "fontFamily": FONT_MONO, "fontSize": "9px", "letterSpacing": "2px",
                "color": color, "border": f"1px solid {color}",
                "padding": "2px 6px", "borderRadius": "2px", "whiteSpace": "nowrap",
            }),
            html.Span(msg, style={"fontFamily": FONT_BODY, "fontSize": "12px", "color": COLORS["text"]}),
        ])
        for level, msg, color in alerts
    ])


@app.callback(
    Output("gauge-temp",    "figure"),
    Output("gauge-current", "figure"),
    Output("gauge-vib",     "figure"),
    Input("interval", "n_intervals"),
)
def update_gauges(n):
    return (
        make_gauge(latest["temperature"], "TEMPERATURE", "°C",  100, 55,  70),
        make_gauge(latest["current"],     "CURRENT",     " A",    6,  3.0, 4.5),
        make_gauge(latest["vibration"],   "VIBRATION",  " pls",  15,  3,   7),
    )


@app.callback(
    Output("graph-temp",    "figure"),
    Output("graph-current", "figure"),
    Output("graph-vib",     "figure"),
    Input("interval", "n_intervals"),
)
def update_graphs(n):
    t = sensor_data["time"]
    return (
        make_timeseries(t, sensor_data["temperature"], "°C",  COLORS["red"],    55,  70),
        make_timeseries(t, sensor_data["current"],     "A",   COLORS["accent"],  3.0, 4.5),
        make_timeseries(t, sensor_data["vibration"],   "pls", COLORS["green"],   3,   7),
    )


@app.callback(
    Output("rf-current-pred", "children"),
    Output("rf-current-conf", "children"),
    Output("rf-future-pred",  "children"),
    Output("rf-future-conf",  "children"),
    Output("rf-fault-prob",   "children"),
    Output("rf-fault-prob",   "style"),
    Output("rf-trend",        "children"),
    Output("rf-trend",        "style"),
    Output("rf-proba-bars",   "children"),
    Input("interval", "n_intervals"),
)
def update_rf_panel(n):
    cp    = ml_result["current_prediction"]
    cc    = ml_result["current_confidence"]
    fp    = ml_result["future_prediction"]
    fc    = ml_result["future_confidence"]
    fprob = ml_result["fault_probability"]
    trend = ml_result["trend"]
    proba = ml_result["class_probabilities"]

    cp_el = html.Span(cp.upper(), style={
        "fontFamily": FONT_MONO, "fontSize": "22px", "fontWeight": "700",
        "color": status_color(cp), "textShadow": f"0 0 12px {status_color(cp)}66",
    })
    fp_el = html.Span(fp.upper(), style={
        "fontFamily": FONT_MONO, "fontSize": "22px", "fontWeight": "700",
        "color": status_color(fp), "textShadow": f"0 0 12px {status_color(fp)}66",
    })

    fprob_color = COLORS["red"] if fprob > 60 else (COLORS["yellow"] if fprob > 30 else COLORS["green"])
    fprob_style = {"fontFamily": FONT_MONO, "fontSize": "28px",
                   "fontWeight": "700", "color": fprob_color}

    trend_color = COLORS["red"] if "↑" in trend else (COLORS["green"] if "↓" in trend else COLORS["accent"])
    trend_style = {"fontFamily": FONT_MONO, "fontSize": "20px",
                   "fontWeight": "700", "color": trend_color}

    # Probability bars
    bar_order  = ["Normal", "Warning", "Critical"]
    bar_colors = [COLORS["green"], COLORS["yellow"], COLORS["red"]]
    bars = []
    for cls, bar_color in zip(bar_order, bar_colors):
        pct = proba.get(cls, 0)
        bars.append(html.Div(style={"marginBottom": "8px"}, children=[
            html.Div(style={"display": "flex", "justifyContent": "space-between",
                            "marginBottom": "3px"}, children=[
                html.Span(cls.upper(), style={"fontFamily": FONT_MONO, "fontSize": "10px",
                                              "color": bar_color, "letterSpacing": "1px"}),
                html.Span(f"{pct}%",  style={"fontFamily": FONT_MONO, "fontSize": "10px",
                                              "color": COLORS["subtext"]}),
            ]),
            html.Div(style={"background": COLORS["border"], "borderRadius": "2px",
                             "height": "6px", "width": "100%"}, children=[
                html.Div(style={
                    "background": bar_color, "borderRadius": "2px",
                    "height": "6px", "width": f"{pct}%",
                    "boxShadow": f"0 0 6px {bar_color}88",
                    "transition": "width 0.4s ease",
                }),
            ]),
        ]))

    return (
        cp_el, f"Confidence: {cc}%",
        fp_el, f"Confidence: {fc}%",
        f"{fprob}%", fprob_style,
        trend, trend_style,
        bars,
    )


@app.callback(Output("fault-table", "children"), Input("interval", "n_intervals"))
def update_fault_table(n):
    if not fault_log:
        return html.Div("No faults recorded in this session.",
                        style={"fontFamily": FONT_MONO, "fontSize": "11px",
                               "color": COLORS["subtext"], "padding": "6px 0"})
    cols = ["TIMESTAMP", "STATUS", "RF PREDICTION", "TEMP (°C)", "CURRENT (A)", "VIBRATION", "FAULT PROB %"]
    header = html.Tr([
        html.Th(c, style={"fontFamily": FONT_MONO, "fontSize": "9px", "letterSpacing": "2px",
                          "color": COLORS["subtext"], "padding": "6px 12px",
                          "borderBottom": f"1px solid {COLORS['border']}",
                          "background": COLORS["bg"]})
        for c in cols
    ])
    rows = []
    for e in list(fault_log)[:20]:
        sc = status_color(e["status"])
        rc = status_color(e["rf_pred"])
        rows.append(html.Tr([
            html.Td(e["time"],             style=td_style(COLORS["subtext"])),
            html.Td(e["status"].upper(),   style=td_style(sc)),
            html.Td(e["rf_pred"].upper(),  style=td_style(rc)),
            html.Td(str(e["temp"]),         style=td_style(COLORS["text"])),
            html.Td(str(e["curr"]),         style=td_style(COLORS["text"])),
            html.Td(str(e["vib"]),          style=td_style(COLORS["text"])),
            html.Td(f"{e['fault_prob']}%",  style=td_style(COLORS["yellow"])),
        ], style={"borderBottom": f"1px solid {COLORS['border']}"}))

    return html.Table([html.Thead(header), html.Tbody(rows)],
                      style={"width": "100%", "borderCollapse": "collapse"})


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
