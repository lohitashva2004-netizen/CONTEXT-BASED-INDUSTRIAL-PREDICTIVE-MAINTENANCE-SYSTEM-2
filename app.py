"""
============================================================
  Predictive Maintenance Dashboard — Induction Motor
  Stack : Python Dash + Plotly + paho-mqtt
  Theme : Dark Industrial
  Deploy: Render (gunicorn app:server)
============================================================

  Install:
    pip install dash dash-bootstrap-components plotly paho-mqtt gunicorn

  Run locally:
    python app.py

  Render start command:
    gunicorn app:server --bind 0.0.0.0:$PORT
============================================================
"""

import threading
import json
from collections import deque
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import paho.mqtt.client as mqtt

# ─────────────────────────────────────────────────────────────
# MQTT CONFIG  — must match your ESP32 exactly
# ─────────────────────────────────────────────────────────────
MQTT_BROKER    = "broker.hivemq.com"
MQTT_PORT      = 1883
MQTT_TOPIC_ALL = "motor/all"

# ─────────────────────────────────────────────────────────────
# SHARED DATA STORE  (thread-safe ring buffer)
# ─────────────────────────────────────────────────────────────
MAX_POINTS = 60   # ~2 minutes of history at 2s interval

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

# Fault history log (timestamp, fault type, values snapshot)
fault_log = deque(maxlen=50)

# Connection status
mqtt_connected = {"value": False}

# ─────────────────────────────────────────────────────────────
# MQTT LISTENER  (runs in background thread)
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
        payload = json.loads(msg.payload.decode())

        prev_status = latest["status"]

        latest["vibration"]   = payload.get("vibration",   0)
        latest["temperature"] = round(payload.get("temperature", 0.0), 1)
        latest["current"]     = round(payload.get("current",     0.0), 2)
        latest["status"]      = payload.get("status",      "Normal")

        now = datetime.now().strftime("%H:%M:%S")
        sensor_data["time"].append(now)
        sensor_data["vibration"].append(latest["vibration"])
        sensor_data["temperature"].append(latest["temperature"])
        sensor_data["current"].append(latest["current"])
        sensor_data["status"].append(latest["status"])

        # Log a fault entry whenever status changes to Warning or Critical
        if latest["status"] in ("Warning", "Critical") and latest["status"] != prev_status:
            fault_log.appendleft({
                "time":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": latest["status"],
                "temp":   latest["temperature"],
                "curr":   latest["current"],
                "vib":    latest["vibration"],
            })

    except Exception as e:
        print(f"MQTT message parse error: {e}")

def start_mqtt():
    client = mqtt.Client(client_id="Dashboard_Main_001")
    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_message    = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_forever()

mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
mqtt_thread.start()

# ─────────────────────────────────────────────────────────────
# DASH APP
# ─────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap",
    ],
    title="Motor PdM Dashboard",
)
server = app.server   # for gunicorn

# ─────────────────────────────────────────────────────────────
# STYLE CONSTANTS
# ─────────────────────────────────────────────────────────────
COLORS = {
    "bg":        "#0a0c10",
    "panel":     "#0f1318",
    "border":    "#1e2530",
    "accent":    "#00e5ff",
    "green":     "#00e676",
    "yellow":    "#ffea00",
    "red":       "#ff1744",
    "text":      "#c8d6e5",
    "subtext":   "#546e7a",
    "gridline":  "#1a2030",
}

FONT_MONO  = "'Share Tech Mono', monospace"
FONT_BODY  = "'Exo 2', sans-serif"

PANEL_STYLE = {
    "background":   COLORS["panel"],
    "border":       f"1px solid {COLORS['border']}",
    "borderRadius": "4px",
    "padding":      "18px",
}

# ─────────────────────────────────────────────────────────────
# HELPER: BUILD GAUGE FIGURE
# ─────────────────────────────────────────────────────────────
def make_gauge(value, title, unit, max_val, warn, crit):
    color = COLORS["green"]
    if value >= crit:
        color = COLORS["red"]
    elif value >= warn:
        color = COLORS["yellow"]

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value,
        number= {"suffix": unit, "font": {"color": color, "family": FONT_MONO, "size": 28}},
        title = {"text": title, "font": {"color": COLORS["text"], "family": FONT_BODY, "size": 13}},
        gauge = {
            "axis": {
                "range": [0, max_val],
                "tickcolor": COLORS["subtext"],
                "tickfont":  {"color": COLORS["subtext"], "size": 10},
            },
            "bar":             {"color": color, "thickness": 0.25},
            "bgcolor":         COLORS["bg"],
            "borderwidth":     1,
            "bordercolor":     COLORS["border"],
            "steps": [
                {"range": [0,    warn],    "color": "#0d1520"},
                {"range": [warn, crit],    "color": "#1a1500"},
                {"range": [crit, max_val], "color": "#1a0508"},
            ],
            "threshold": {
                "line":  {"color": COLORS["red"], "width": 2},
                "thickness": 0.75,
                "value": crit,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        margin        = dict(t=40, b=10, l=20, r=20),
        height        = 200,
        font          = {"color": COLORS["text"]},
    )
    return fig

# ─────────────────────────────────────────────────────────────
# HELPER: BUILD TIME-SERIES FIGURE
# ─────────────────────────────────────────────────────────────
def make_timeseries(times, values, label, unit, color, warn, crit):
    times  = list(times)
    values = list(values)
    fig = go.Figure()

    # Warning band
    fig.add_hrect(y0=warn, y1=crit, fillcolor=COLORS["yellow"],
                  opacity=0.06, line_width=0)
    # Critical band
    fig.add_hrect(y0=crit, y1=crit * 1.5, fillcolor=COLORS["red"],
                  opacity=0.06, line_width=0)

    # Main line
    fig.add_trace(go.Scatter(
        x=times, y=values,
        mode="lines",
        name=label,
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=color.replace(")", ",0.06)").replace("rgb", "rgba") if color.startswith("rgb") else color + "12",
    ))

    # Warn / crit reference lines
    if times:
        fig.add_hline(y=warn, line_dash="dot", line_color=COLORS["yellow"],
                      line_width=1, opacity=0.5)
        fig.add_hline(y=crit, line_dash="dot", line_color=COLORS["red"],
                      line_width=1, opacity=0.5)

    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        margin        = dict(t=10, b=30, l=50, r=20),
        height        = 160,
        showlegend    = False,
        xaxis = dict(
            showgrid=True, gridcolor=COLORS["gridline"],
            tickfont=dict(color=COLORS["subtext"], size=9, family=FONT_MONO),
            tickangle=-30, nticks=8,
        ),
        yaxis = dict(
            showgrid=True, gridcolor=COLORS["gridline"],
            tickfont=dict(color=COLORS["subtext"], size=9, family=FONT_MONO),
            title=dict(text=unit, font=dict(color=COLORS["subtext"], size=10)),
        ),
    )
    return fig

# ─────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={"backgroundColor": COLORS["bg"], "minHeight": "100vh",
           "fontFamily": FONT_BODY, "color": COLORS["text"], "padding": "0"},
    children=[

        # ── Auto-refresh interval ──────────────────────────
        dcc.Interval(id="interval", interval=2000, n_intervals=0),

        # ── TOP HEADER BAR ─────────────────────────────────
        html.Div(
            style={
                "background":   "linear-gradient(90deg, #0d1520 0%, #0a0c10 60%)",
                "borderBottom": f"1px solid {COLORS['border']}",
                "padding":      "14px 28px",
                "display":      "flex",
                "alignItems":   "center",
                "justifyContent": "space-between",
            },
            children=[
                html.Div([
                    html.Span("⚙", style={"fontSize": "22px", "marginRight": "10px",
                                          "color": COLORS["accent"]}),
                    html.Span("PREDICTIVE MAINTENANCE SYSTEM",
                              style={"fontFamily": FONT_MONO, "fontSize": "16px",
                                     "letterSpacing": "3px", "color": COLORS["accent"]}),
                    html.Span(" — INDUCTION MOTOR",
                              style={"fontFamily": FONT_MONO, "fontSize": "12px",
                                     "color": COLORS["subtext"], "letterSpacing": "2px"}),
                ]),
                html.Div([
                    html.Span(id="mqtt-status-dot",
                              style={"display": "inline-block", "width": "9px",
                                     "height": "9px", "borderRadius": "50%",
                                     "background": COLORS["red"], "marginRight": "7px"}),
                    html.Span(id="mqtt-status-text",
                              style={"fontFamily": FONT_MONO, "fontSize": "11px",
                                     "color": COLORS["subtext"]}),
                    html.Span(id="clock",
                              style={"fontFamily": FONT_MONO, "fontSize": "12px",
                                     "color": COLORS["subtext"], "marginLeft": "24px"}),
                ]),
            ],
        ),

        # ── MAIN CONTENT ───────────────────────────────────
        html.Div(style={"padding": "20px 24px"}, children=[

            # ── ROW 1: STATUS BADGE + ALERT PANEL ──────────
            dbc.Row(style={"marginBottom": "16px"}, children=[

                # Status Badge
                dbc.Col(width=4, children=[
                    html.Div(style={**PANEL_STYLE, "textAlign": "center",
                                    "position": "relative", "overflow": "hidden"}, children=[
                        html.Div("MACHINE STATUS", style={
                            "fontFamily": FONT_MONO, "fontSize": "10px",
                            "letterSpacing": "3px", "color": COLORS["subtext"],
                            "marginBottom": "10px",
                        }),
                        html.Div(id="status-badge", style={
                            "fontFamily": FONT_MONO, "fontSize": "36px",
                            "fontWeight": "700", "letterSpacing": "4px",
                            "transition": "color 0.4s",
                        }),
                        html.Div(id="status-subtext", style={
                            "fontFamily": FONT_BODY, "fontSize": "11px",
                            "color": COLORS["subtext"], "marginTop": "6px",
                        }),
                        html.Div(id="status-glow", style={
                            "position": "absolute", "bottom": "-30px", "left": "50%",
                            "transform": "translateX(-50%)",
                            "width": "80px", "height": "60px",
                            "borderRadius": "50%", "filter": "blur(20px)",
                            "opacity": "0.25", "transition": "background 0.4s",
                        }),
                    ]),
                ]),

                # Alerts Panel
                dbc.Col(width=8, children=[
                    html.Div(style={**PANEL_STYLE, "height": "100%"}, children=[
                        html.Div("ACTIVE ALERTS", style={
                            "fontFamily": FONT_MONO, "fontSize": "10px",
                            "letterSpacing": "3px", "color": COLORS["subtext"],
                            "marginBottom": "10px",
                        }),
                        html.Div(id="alerts-panel"),
                    ]),
                ]),
            ]),

            # ── ROW 2: GAUGES ───────────────────────────────
            dbc.Row(style={"marginBottom": "16px"}, children=[
                dbc.Col(width=4, children=[
                    html.Div(style=PANEL_STYLE, children=[
                        dcc.Graph(id="gauge-temp",    config={"displayModeBar": False}),
                    ]),
                ]),
                dbc.Col(width=4, children=[
                    html.Div(style=PANEL_STYLE, children=[
                        dcc.Graph(id="gauge-current", config={"displayModeBar": False}),
                    ]),
                ]),
                dbc.Col(width=4, children=[
                    html.Div(style=PANEL_STYLE, children=[
                        dcc.Graph(id="gauge-vib",     config={"displayModeBar": False}),
                    ]),
                ]),
            ]),

            # ── ROW 3: TIME SERIES GRAPHS ───────────────────
            dbc.Row(style={"marginBottom": "16px"}, children=[
                dbc.Col(width=12, children=[
                    html.Div(style=PANEL_STYLE, children=[
                        html.Div("SENSOR HISTORY", style={
                            "fontFamily": FONT_MONO, "fontSize": "10px",
                            "letterSpacing": "3px", "color": COLORS["subtext"],
                            "marginBottom": "12px",
                        }),
                        dbc.Row([
                            dbc.Col(width=4, children=[
                                html.Div("TEMPERATURE (°C)", style={
                                    "fontFamily": FONT_MONO, "fontSize": "9px",
                                    "color": COLORS["subtext"], "marginBottom": "4px",
                                }),
                                dcc.Graph(id="graph-temp", config={"displayModeBar": False}),
                            ]),
                            dbc.Col(width=4, children=[
                                html.Div("CURRENT (A)", style={
                                    "fontFamily": FONT_MONO, "fontSize": "9px",
                                    "color": COLORS["subtext"], "marginBottom": "4px",
                                }),
                                dcc.Graph(id="graph-current", config={"displayModeBar": False}),
                            ]),
                            dbc.Col(width=4, children=[
                                html.Div("VIBRATION (pulses/2s)", style={
                                    "fontFamily": FONT_MONO, "fontSize": "9px",
                                    "color": COLORS["subtext"], "marginBottom": "4px",
                                }),
                                dcc.Graph(id="graph-vib", config={"displayModeBar": False}),
                            ]),
                        ]),
                    ]),
                ]),
            ]),

            # ── ROW 4: FAULT HISTORY TABLE ──────────────────
            dbc.Row(children=[
                dbc.Col(width=12, children=[
                    html.Div(style=PANEL_STYLE, children=[
                        html.Div("FAULT HISTORY LOG", style={
                            "fontFamily": FONT_MONO, "fontSize": "10px",
                            "letterSpacing": "3px", "color": COLORS["subtext"],
                            "marginBottom": "12px",
                        }),
                        html.Div(id="fault-table"),
                    ]),
                ]),
            ]),

        ]),  # end main content

        # ── FOOTER ─────────────────────────────────────────
        html.Div(
            style={
                "borderTop":  f"1px solid {COLORS['border']}",
                "padding":    "10px 28px",
                "display":    "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
            },
            children=[
                html.Span("SRM Institute of Science and Technology — Lohitashva V.S",
                          style={"fontFamily": FONT_MONO, "fontSize": "10px",
                                 "color": COLORS["subtext"]}),
                html.Span("Context-Based Predictive Maintenance | IoT + ML",
                          style={"fontFamily": FONT_MONO, "fontSize": "10px",
                                 "color": COLORS["subtext"]}),
            ],
        ),
    ],
)

# ─────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────

# ── Clock ────────────────────────────────────────────────────
@app.callback(Output("clock", "children"), Input("interval", "n_intervals"))
def update_clock(n):
    return datetime.now().strftime("%Y-%m-%d  %H:%M:%S")


# ── MQTT Status Indicator ────────────────────────────────────
@app.callback(
    Output("mqtt-status-dot",  "style"),
    Output("mqtt-status-text", "children"),
    Input("interval", "n_intervals"),
)
def update_mqtt_status(n):
    connected = mqtt_connected["value"]
    dot_style = {
        "display": "inline-block", "width": "9px", "height": "9px",
        "borderRadius": "50%", "marginRight": "7px",
        "background": COLORS["green"] if connected else COLORS["red"],
        "boxShadow":  (f"0 0 6px {COLORS['green']}" if connected
                       else f"0 0 6px {COLORS['red']}"),
    }
    text = "MQTT LIVE" if connected else "MQTT DISCONNECTED"
    return dot_style, text


# ── Status Badge ─────────────────────────────────────────────
@app.callback(
    Output("status-badge",   "children"),
    Output("status-badge",   "style"),
    Output("status-subtext", "children"),
    Output("status-glow",    "style"),
    Input("interval", "n_intervals"),
)
def update_status_badge(n):
    s = latest["status"]

    color_map = {
        "Normal":   COLORS["green"],
        "Warning":  COLORS["yellow"],
        "Critical": COLORS["red"],
    }
    msg_map = {
        "Normal":   "All parameters within safe limits",
        "Warning":  "One or more parameters approaching threshold",
        "Critical": "IMMEDIATE MAINTENANCE REQUIRED",
    }

    color = color_map.get(s, COLORS["subtext"])
    glow_style = {
        "position": "absolute", "bottom": "-30px", "left": "50%",
        "transform": "translateX(-50%)",
        "width": "80px", "height": "60px",
        "borderRadius": "50%", "filter": "blur(20px)",
        "opacity": "0.3", "transition": "background 0.4s",
        "background": color,
    }
    badge_style = {
        "fontFamily": FONT_MONO, "fontSize": "36px",
        "fontWeight": "700", "letterSpacing": "4px",
        "transition": "color 0.4s", "color": color,
        "textShadow": f"0 0 20px {color}88",
    }
    return s.upper(), badge_style, msg_map.get(s, ""), glow_style


# ── Alerts Panel ─────────────────────────────────────────────
@app.callback(Output("alerts-panel", "children"), Input("interval", "n_intervals"))
def update_alerts(n):
    s    = latest["status"]
    temp = latest["temperature"]
    curr = latest["current"]
    vib  = latest["vibration"]

    alerts = []

    if temp >= 70:
        alerts.append(("CRITICAL", f"🌡 Thermal Fault — Temperature {temp}°C exceeds critical limit (70°C). Motor overheating.", COLORS["red"]))
    elif temp >= 55:
        alerts.append(("WARNING",  f"🌡 High Temperature — {temp}°C approaching critical. Check motor cooling.", COLORS["yellow"]))

    if curr >= 4.5:
        alerts.append(("CRITICAL", f"⚡ Overload Fault — Current {curr}A exceeds critical limit (4.5A). Possible motor jam.", COLORS["red"]))
    elif curr >= 3.0:
        alerts.append(("WARNING",  f"⚡ High Current — {curr}A approaching overload. Check mechanical load.", COLORS["yellow"]))

    if vib >= 7:
        alerts.append(("CRITICAL", f"📳 Bearing Fault — Vibration {vib} pulses exceeds critical limit. Possible bearing failure.", COLORS["red"]))
    elif vib >= 3:
        alerts.append(("WARNING",  f"📳 Abnormal Vibration — {vib} pulses. Check motor alignment and bearings.", COLORS["yellow"]))

    if not alerts:
        return html.Div(
            "✅  No active alerts — motor operating normally.",
            style={"fontFamily": FONT_MONO, "fontSize": "12px",
                   "color": COLORS["green"], "padding": "8px 0"},
        )

    return html.Div([
        html.Div(
            style={
                "display":       "flex",
                "alignItems":    "center",
                "gap":           "10px",
                "padding":       "8px 12px",
                "marginBottom":  "6px",
                "borderLeft":    f"3px solid {color}",
                "background":    f"{color}0d",
                "borderRadius":  "2px",
            },
            children=[
                html.Span(level, style={
                    "fontFamily": FONT_MONO, "fontSize": "9px",
                    "letterSpacing": "2px", "color": color,
                    "border": f"1px solid {color}", "padding": "2px 6px",
                    "borderRadius": "2px", "whiteSpace": "nowrap",
                }),
                html.Span(msg, style={
                    "fontFamily": FONT_BODY, "fontSize": "12px",
                    "color": COLORS["text"],
                }),
            ],
        )
        for level, msg, color in alerts
    ])


# ── Gauges ───────────────────────────────────────────────────
@app.callback(
    Output("gauge-temp",    "figure"),
    Output("gauge-current", "figure"),
    Output("gauge-vib",     "figure"),
    Input("interval", "n_intervals"),
)
def update_gauges(n):
    fig_temp = make_gauge(latest["temperature"], "TEMPERATURE",  "°C", 100, 55,  70)
    fig_curr = make_gauge(latest["current"],     "CURRENT",      " A",  6,  3.0, 4.5)
    fig_vib  = make_gauge(latest["vibration"],   "VIBRATION", " pls", 15,  3,   7)
    return fig_temp, fig_curr, fig_vib


# ── Time Series Graphs ───────────────────────────────────────
@app.callback(
    Output("graph-temp",    "figure"),
    Output("graph-current", "figure"),
    Output("graph-vib",     "figure"),
    Input("interval", "n_intervals"),
)
def update_graphs(n):
    times = sensor_data["time"]
    fig_t = make_timeseries(times, sensor_data["temperature"], "Temp",      "°C", COLORS["red"],    55,  70)
    fig_c = make_timeseries(times, sensor_data["current"],     "Current",   "A",  COLORS["accent"],  3.0, 4.5)
    fig_v = make_timeseries(times, sensor_data["vibration"],   "Vibration", "pls",COLORS["green"],   3,   7)
    return fig_t, fig_c, fig_v


# ── Fault History Table ──────────────────────────────────────
@app.callback(Output("fault-table", "children"), Input("interval", "n_intervals"))
def update_fault_table(n):
    if not fault_log:
        return html.Div(
            "No faults recorded in this session.",
            style={"fontFamily": FONT_MONO, "fontSize": "11px",
                   "color": COLORS["subtext"], "padding": "6px 0"},
        )

    header = html.Tr([
        html.Th(col, style={
            "fontFamily": FONT_MONO, "fontSize": "9px", "letterSpacing": "2px",
            "color": COLORS["subtext"], "padding": "6px 12px",
            "borderBottom": f"1px solid {COLORS['border']}",
            "background": COLORS["bg"],
        })
        for col in ["TIMESTAMP", "STATUS", "TEMP (°C)", "CURRENT (A)", "VIBRATION"]
    ])

    rows = []
    for entry in list(fault_log)[:20]:
        color = COLORS["red"] if entry["status"] == "Critical" else COLORS["yellow"]
        rows.append(html.Tr([
            html.Td(entry["time"],            style=td_style(COLORS["subtext"])),
            html.Td(entry["status"].upper(),  style=td_style(color)),
            html.Td(str(entry["temp"]),        style=td_style(COLORS["text"])),
            html.Td(str(entry["curr"]),        style=td_style(COLORS["text"])),
            html.Td(str(entry["vib"]),         style=td_style(COLORS["text"])),
        ], style={"borderBottom": f"1px solid {COLORS['border']}"}))

    return html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse"},
    )


def td_style(color):
    return {
        "fontFamily": FONT_MONO, "fontSize": "11px",
        "color": color, "padding": "7px 12px",
    }


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
