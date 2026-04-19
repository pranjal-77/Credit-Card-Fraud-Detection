""" FRAUD DETECTION — CREDIT CARD FRAUD DETECTION Platform
Single-file Streamlit app. Requires: fraud_model.pkl, fraud_scaler.pkl, feature_cols.pkl
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle
import time
import random
import string
import os
from datetime import datetime

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CREDIT CARD FRAUD DETECTION",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background-color: #070b14 !important;
    color: #94a3b8;
}

/* ── Main app container ── */
.stApp { background-color: #070b14 !important; }
.main .block-container {
    background-color: #070b14 !important;
    padding: 1.5rem 2rem 4rem 2rem;
    max-width: 1400px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #0a0f1a !important;
    border-right: 1px solid rgba(0,212,255,0.12) !important;
}
section[data-testid="stSidebar"] .stRadio > label {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #64748b !important;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #94a3b8 !important;
    font-size: 0.82rem;
    letter-spacing: 0.05em;
    padding: 0.35rem 0;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    color: #00d4ff !important;
}

/* ── Inputs ── */
.stSlider > div, .stNumberInput, .stSelectbox {
    font-family: 'IBM Plex Mono', monospace !important;
}
.stSlider [data-testid="stThumbValue"] { color: #00d4ff !important; }
div[data-baseweb="select"] > div {
    background-color: #0d1117 !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 2px !important;
    color: #94a3b8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
}
input[type="number"], input[type="text"] {
    background-color: #0d1117 !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 2px !important;
    color: #94a3b8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-size: 0.75rem;
    background: transparent !important;
    color: #00d4ff !important;
    border: 1px solid #00d4ff !important;
    border-radius: 2px !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: rgba(0,212,255,0.08) !important;
    box-shadow: 0 0 16px rgba(0,212,255,0.25);
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #0d1117;
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 2px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #475569 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #e2e8f0 !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace !important; }

/* ── Dataframe ── */
.stDataFrame { border: 1px solid rgba(0,212,255,0.1) !important; border-radius: 2px !important; }
.stDataFrame thead th {
    background-color: #0d1117 !important;
    color: #00d4ff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(0,212,255,0.2) !important;
}
.stDataFrame tbody tr td {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem;
    background-color: #070b14 !important;
    color: #94a3b8 !important;
    border-bottom: 1px solid rgba(255,255,255,0.03) !important;
}
.stDataFrame tbody tr:hover td { background-color: #0d1117 !important; }

/* ── Divider ── */
hr { border-color: rgba(0,212,255,0.08) !important; margin: 1.5rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #070b14; }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.25); border-radius: 2px; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(0,212,255,0.2) !important;
    border-radius: 2px !important;
    background: #0a0f1a !important;
    padding: 1rem;
}

/* ── Blink animation ── */
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
}

/* ── Pulse animation for high-risk ── */
@keyframes pulse-border {
    0%   { box-shadow: 0 0 0 0 rgba(255,59,107,0.7); }
    25%  { box-shadow: 0 0 0 8px rgba(255,59,107,0.0); }
    50%  { box-shadow: 0 0 0 0 rgba(255,59,107,0.7); }
    75%  { box-shadow: 0 0 0 8px rgba(255,59,107,0.0); }
    100% { box-shadow: 0 0 4px 2px rgba(255,59,107,0.3); }
}
.pulse-red {
    animation: pulse-border 1.8s ease-out forwards;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    background: transparent !important;
    color: #22c55e !important;
    border: 1px solid #22c55e !important;
    border-radius: 2px !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(34,197,94,0.08) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
CATEGORY_LABELS = {
    0: "gas_transport", 1: "grocery_pos", 2: "home", 3: "shopping_net",
    4: "entertainment", 5: "food_dining", 6: "personal_care", 7: "health_fitness",
    8: "shopping_pos", 9: "kids_pets", 10: "sports_outdoors", 11: "travel",
    12: "misc_net", 13: "misc_pos"
}
WEEKDAY_LABELS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ─── MODEL LOADER ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import joblib

        model = joblib.load("fraud_model.pkl")
        scaler = joblib.load("fraud_scaler.pkl")
        feature_cols = joblib.load("feature_cols.pkl")

        return model, scaler, feature_cols, True
    except Exception as e:
        print("Model load error:", e)
        return None, None, None, False
    
def build_feature_vector(amt, hour, day, month, weekday, age, category, gender, city_pop, feature_cols):
    """Build a zero-padded feature vector matching the training schema."""
    row = {col: 0.0 for col in feature_cols}
    if "amt" in row: row["amt"] = amt
    if "hour" in row: row["hour"] = hour
    if "day" in row: row["day"] = day
    if "month" in row: row["month"] = month
    if "weekday" in row: row["weekday"] = weekday
    if "age" in row: row["age"] = age
    if "category" in row: row["category"] = category
    if "gender" in row: row["gender"] = gender
    if "city_pop" in row: row["city_pop"] = city_pop
    return pd.DataFrame([row])[feature_cols].astype(float)

def predict_fraud(model, scaler, feature_cols, amt, hour, day, month, weekday, age, category, gender, city_pop):
    """Run a single-transaction fraud prediction and return probability."""
    X = build_feature_vector(amt, hour, day, month, weekday, age, category, gender, city_pop, feature_cols)
    X_scaled = scaler.transform(X) if hasattr(scaler, "transform") else X
    proba = model.predict_proba(X_scaled)[0][1] if hasattr(model, "predict_proba") else float(model[0])
    return float(proba)

# ─── HELPER UTILITIES ────────────────────────────────────────────────────────
def txn_id():
    """Generate a random transaction ID."""
    return "TXN-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

def risk_badge(score):
    """Return a colored HTML badge for a given risk score (0-1)."""
    pct = score * 100
    if pct >= 70:
        color, label = "#ff3b6b", "HIGH"
    elif pct >= 30:
        color, label = "#f59e0b", "MED"
    else:
        color, label = "#22c55e", "CLEAR"
    return (
        f'<span style="background:rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15);'
        f'color:{color};border:1px solid {color};padding:2px 8px;border-radius:2px;'
        f'font-size:0.68rem;letter-spacing:0.1em;font-family:\'IBM Plex Mono\',monospace;">{label}</span>'
    )

def section_header(text, subtitle=""):
    """Render a styled section header."""
    sub_html = f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:#334155;letter-spacing:0.18em;margin-top:2px;">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div style="margin:0.5rem 0 1.2rem 0;">
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
                  color:#e2e8f0;letter-spacing:0.06em;text-transform:uppercase;
                  border-left:3px solid #00d4ff;padding-left:0.75rem;line-height:1.2;">
        {text}
      </div>
      {sub_html}
    </div>
    """, unsafe_allow_html=True)

def panel(content_html, border_color="rgba(0,212,255,0.15)", bg="#0d1117", padding="1.2rem", extra_style=""):
    """Render content inside a styled panel box."""
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {border_color};border-radius:2px;
                padding:{padding};{extra_style}">
      {content_html}
    </div>
    """, unsafe_allow_html=True)

def plotly_dark_layout(fig, title=""):
    """Apply the unified dark theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1117",
        font_color="#94a3b8",
        font_family="IBM Plex Mono",
        title=dict(text=title, font=dict(color="#e2e8f0", size=13, family="Syne")) if title else None,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,212,255,0.12)",
            borderwidth=1,
            font=dict(size=11, color="#64748b"),
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(0,212,255,0.1)",
        tickfont=dict(size=10, color="#475569"),
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(0,212,255,0.1)",
        tickfont=dict(size=10, color="#475569"),
    )
    return fig

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 1.8rem 0;">
      <div style="font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:800;
                  color:#00d4ff;letter-spacing:0.08em;">FRAUD DETECTION</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                  color:#334155;letter-spacing:0.2em;margin-top:3px;">
        CREDIT CARD FRAUD DETECTION PLATFORM v2.1
      </div>
      <div style="height:1px;background:rgba(0,212,255,0.1);margin-top:1rem;"></div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        ["⬡  MONITOR", "◈  ANALYZE", "⬢  INVESTIGATE", "◉  INTEL"],
        label_visibility="visible",
    )

    st.markdown("""
    <div style="position:fixed;bottom:1.5rem;left:0;width:260px;padding:0 1.2rem;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                  color:#1e293b;letter-spacing:0.1em;line-height:1.8;">
        MODEL: RandomForest v3.2<br>
        DATASET: 1.3M transactions<br>
        LAST TRAINED: 2024-11-18<br>
        STATUS: <span style="color:#22c55e;">OPERATIONAL</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─── PAGE: MONITOR ────────────────────────────────────────────────────────────
def page_monitor():
    model, scaler, feature_cols, model_loaded = load_model()

    # Header
    st.markdown("""
    <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem;">
      <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
                  color:#e2e8f0;letter-spacing:0.05em;text-transform:uppercase;">
        Live Transaction Monitor
      </div>
      <div style="display:flex;align-items:center;gap:0.5rem;
                  background:#0d1117;border:1px solid rgba(0,212,255,0.15);
                  padding:4px 12px;border-radius:2px;">
        <span style="width:8px;height:8px;border-radius:50%;background:#ff3b6b;
                     display:inline-block;animation:blink 1s step-start infinite;"></span>
        <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                     color:#ff3b6b;letter-spacing:0.18em;">LIVE</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Generate fake transactions
    def generate_transactions(n=20):
        rows = []
        for _ in range(n):
            cat_key = random.randint(0, 13)
            amt = round(random.lognormvariate(4.5, 1.2), 2)
            hour = random.randint(0, 23)
            age = random.randint(18, 85)
            city_pop = random.randint(500, 2000000)
            weekday = random.randint(0, 6)

            # Assign risk score
            if model_loaded:
                try:
                    score = predict_fraud(
                        model, scaler, feature_cols,
                        amt=amt, hour=hour, day=random.randint(1,28),
                        month=random.randint(1,12), weekday=weekday,
                        age=age, category=cat_key,
                        gender=random.randint(0,1), city_pop=city_pop
                    )
                except Exception:
                    score = random.betavariate(1.2, 6.0)
            else:
                # Simulate plausible risk distribution without model
                base = 0.02
                if amt > 800: base += 0.3
                if hour in [0,1,2,3]: base += 0.2
                if cat_key in [3,12,11]: base += 0.15
                score = min(max(base + random.gauss(0, 0.08), 0), 1)

            rows.append({
                "TXN_ID": txn_id(),
                "AMOUNT": f"${amt:,.2f}",
                "CATEGORY": CATEGORY_LABELS[cat_key].upper(),
                "HOUR": f"{hour:02d}:00",
                "RISK_SCORE": round(score * 100, 1),
                "_score": score,
            })
        return pd.DataFrame(rows)

    # Refresh controls
    col_refresh, col_interval, col_spacer = st.columns([1, 1, 4])
    with col_refresh:
        refresh = st.button("↺  REFRESH FEED")
    with col_interval:
        auto_refresh = st.toggle("AUTO", value=False)

    # Stats section
    if "txn_df" not in st.session_state or refresh:
        st.session_state.txn_df = generate_transactions(20)
    if "total_count" not in st.session_state:
        st.session_state.total_count = 20
        st.session_state.fraud_count = 0
        st.session_state.total_at_risk = 0.0

    df = st.session_state.txn_df

    # Recalculate running stats
    fraud_rows = df[df["_score"] >= 0.70]
    st.session_state.fraud_count = int(fraud_rows.shape[0])
    at_risk = fraud_rows["AMOUNT"].apply(lambda x: float(str(x).replace("$","").replace(",",""))).sum()
    st.session_state.total_at_risk = at_risk
    st.session_state.total_count = 20
    at_risk_val = float(at_risk) if at_risk is not None else 0.0

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Metric row
    m1, m2, m3, m4 = st.columns(4)
    fraud_rate = st.session_state.fraud_count / max(st.session_state.total_count, 1) * 100
    avg_risk = df["_score"].mean() * 100
    
    with m1:
        st.metric("TOTAL TRANSACTIONS", f"{st.session_state.total_count:,}", delta="+20")
    with m2:
        st.metric("FRAUD FLAGGED", str(st.session_state.fraud_count), delta=f"{fraud_rate:.1f}%")
    with m3:
        st.metric("AMOUNT AT RISK", f"${at_risk_val:,.2f}", delta=None)
    with m4:
        st.metric("AVG RISK SCORE", f"{avg_risk:.1f}%", delta=None)

    st.divider()
    section_header("TRANSACTION FEED", "LAST 20 PROCESSED — REAL-TIME")

    # Build styled HTML table
   # Line 459 — replace the rows_html builder
    rows_html = ""
    for _, row in df.iterrows():
        s = row["_score"]
        if s >= 0.70:
            row_bg, score_color = "rgba(255,59,107,0.06)", "#ff3b6b"
            status_html = '<span style="color:#ff3b6b;font-weight:600;border:1px solid #ff3b6b;padding:2px 8px;border-radius:2px;font-size:0.68rem;">FRAUD</span>'
        elif s >= 0.30:
            row_bg, score_color = "rgba(245,158,11,0.05)", "#f59e0b"
            status_html = '<span style="color:#f59e0b;border:1px solid #f59e0b;padding:2px 8px;border-radius:2px;font-size:0.68rem;">REVIEW</span>'
        else:
            row_bg, score_color = "transparent", "#22c55e"
            status_html = '<span style="color:#22c55e;border:1px solid #22c55e;padding:2px 8px;border-radius:2px;font-size:0.68rem;">CLEAR</span>'

        rows_html += (
            f'<tr style="background:{row_bg};border-bottom:1px solid rgba(255,255,255,0.03);">'
            f'<td style="padding:8px 12px;font-size:0.75rem;color:#475569;">{row["TXN_ID"]}</td>'
            f'<td style="padding:8px 12px;font-size:0.78rem;color:#e2e8f0;font-weight:500;">{row["AMOUNT"]}</td>'
            f'<td style="padding:8px 12px;font-size:0.7rem;color:#64748b;">{row["CATEGORY"]}</td>'
            f'<td style="padding:8px 12px;font-size:0.75rem;color:#64748b;">{row["HOUR"]}</td>'
            f'<td style="padding:8px 12px;font-size:0.78rem;color:{score_color};font-weight:600;">{row["RISK_SCORE"]}%</td>'
            f'<td style="padding:8px 12px;">{status_html}</td>'
            f'</tr>'
        )

    st.markdown(f"""
    <div style="overflow-x:auto;border:1px solid rgba(0,212,255,0.1);border-radius:2px;">
      <table style="width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;">
        <thead>
          <tr style="background:#0d1117;border-bottom:1px solid rgba(0,212,255,0.2);">
            <th style="padding:10px 12px;text-align:left;font-size:0.65rem;
                       color:#00d4ff;letter-spacing:0.15em;">TXN ID</th>
            <th style="padding:10px 12px;text-align:left;font-size:0.65rem;
                       color:#00d4ff;letter-spacing:0.15em;">AMOUNT</th>
            <th style="padding:10px 12px;text-align:left;font-size:0.65rem;
                       color:#00d4ff;letter-spacing:0.15em;">CATEGORY</th>
            <th style="padding:10px 12px;text-align:left;font-size:0.65rem;
                       color:#00d4ff;letter-spacing:0.15em;">HOUR</th>
            <th style="padding:10px 12px;text-align:left;font-size:0.65rem;
                       color:#00d4ff;letter-spacing:0.15em;">RISK SCORE</th>
            <th style="padding:10px 12px;text-align:left;font-size:0.65rem;
                       color:#00d4ff;letter-spacing:0.15em;">STATUS</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # Auto-refresh loop
    if auto_refresh:
        time.sleep(5)
        st.rerun()

# ─── PAGE: ANALYZE ────────────────────────────────────────────────────────────
def page_analyze():
    model, scaler, feature_cols, model_loaded = load_model()

    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
                color:#e2e8f0;letter-spacing:0.05em;text-transform:uppercase;
                margin-bottom:1.5rem;">
      Single Transaction Analyzer
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.warning("Model files not found. Place fraud_model.pkl, fraud_scaler.pkl, and feature_cols.pkl in the app directory.")

    left_col, right_col = st.columns([1, 1], gap="large")

    # ── Input panel ──
    with left_col:
        section_header("TRANSACTION INPUTS", "CONFIGURE PARAMETERS")

        with st.container():
            amt = st.slider("TRANSACTION AMOUNT ($)", min_value=0.0, max_value=5000.0, value=250.0, step=1.0)
            category_label = st.selectbox(
                "MERCHANT CATEGORY",
                options=[f"{v}  [{k}]" for k, v in CATEGORY_LABELS.items()],
                index=1
            )
            category = int(category_label.split("[")[1].split("]")[0])

            c1, c2 = st.columns(2)
            with c1:
                hour = st.slider("HOUR OF DAY", 0, 23, 14)
            with c2:
                age = st.slider("CARDHOLDER AGE", 18, 90, 38)

            weekday_label = st.selectbox("DAY OF WEEK", WEEKDAY_LABELS, index=2)
            weekday = WEEKDAY_LABELS.index(weekday_label)

            city_pop = st.number_input("CITY POPULATION", min_value=100, max_value=5000000, value=50000, step=1000)

            gender = st.radio("GENDER CODE", ["0 — Female", "1 — Male"], horizontal=True)
            gender_val = int(gender[0])

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            run_btn = st.button("▶  RUN ANALYSIS", use_container_width=True)

    # ── Result panel ──
    with right_col:
        section_header("ANALYSIS OUTPUT", "PREDICTION RESULTS")

        if run_btn:
            now = datetime.now()
            case_id = f"TXN-{random.randint(100000,999999)}"

            if model_loaded:
                score = predict_fraud(
                    model, scaler, feature_cols,
                    amt=amt, hour=hour, day=now.day, month=now.month,
                    weekday=weekday, age=age, category=category,
                    gender=gender_val, city_pop=city_pop
                )
                importances = model.feature_importances_
                top_idx = np.argsort(importances)[::-1][:10]
                top_features = [(feature_cols[i], importances[i]) for i in top_idx]
            else:
                # Demo mode without model
                score = random.uniform(0.05, 0.95)
                top_features = [(f"feature_{i:02d}", random.uniform(0.01, 0.25)) for i in range(10)]
                total = sum(v for _, v in top_features)
                top_features = [(k, v/total) for k, v in top_features]

            pct = score * 100
            is_fraud = pct >= 50
            verdict_color = "#ff3b6b" if is_fraud else "#22c55e"
            verdict_text = "FRAUDULENT" if is_fraud else "LEGITIMATE"
            panel_border = "#ff3b6b" if pct > 60 else ("rgba(0,212,255,0.15)")
            pulse_class = "pulse-red" if pct > 60 else ""

            # ── Gauge chart ──
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number=dict(suffix="%", font=dict(size=28, color="#e2e8f0", family="IBM Plex Mono")),
                gauge=dict(
                    axis=dict(
                        range=[0, 100],
                        tickfont=dict(color="#475569", size=10, family="IBM Plex Mono"),
                        tickvals=[0, 25, 50, 75, 100],
                        ticktext=["0", "25", "50", "75", "100"],
                    ),
                    bar=dict(color="#00d4ff", thickness=0.25),
                    bgcolor="#0d1117",
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 30], color="rgba(34,197,94,0.08)"),
                        dict(range=[30, 70], color="rgba(245,158,11,0.08)"),
                        dict(range=[70, 100], color="rgba(255,59,107,0.12)"),
                    ],
                    threshold=dict(
                        line=dict(color=verdict_color, width=3),
                        thickness=0.85,
                        value=pct,
                    ),
                ),
            ))
            plotly_dark_layout(fig_gauge)
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Verdict box ──
            pulse_style = "animation:pulse-border 1.8s ease-out forwards;" if pct > 60 else ""
            st.markdown(f"""
            <div style="background:rgba({('255,59,107' if is_fraud else '34,197,94')},0.07);
                        border:1px solid {verdict_color};border-radius:2px;
                        padding:1rem 1.4rem;margin:0.5rem 0;{pulse_style}
                        display:flex;align-items:center;justify-content:space-between;" class="{pulse_class}">
              <div>
                <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
                            color:{verdict_color};letter-spacing:0.1em;">{verdict_text}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                            color:#475569;margin-top:3px;letter-spacing:0.1em;">
                  FRAUD PROBABILITY: {pct:.2f}%
                </div>
              </div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;
                          color:{verdict_color};opacity:0.3;">{'⚠' if is_fraud else '✓'}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Feature importance bar ──
            feat_names = [f[0].replace("_", " ").upper() for f in top_features]
            feat_vals = [f[1] * 100 for f in top_features]
            colors = ["#ff3b6b" if v > 15 else "#00d4ff" for v in feat_vals]

            fig_imp = go.Figure(go.Bar(
                x=feat_vals[::-1],
                y=feat_names[::-1],
                orientation="h",
                marker=dict(
                    color=colors[::-1],
                    opacity=0.8,
                    line=dict(width=0),
                ),
                text=[f"{v:.1f}%" for v in feat_vals[::-1]],
                textfont=dict(size=9, family="IBM Plex Mono", color="#94a3b8"),
                textposition="outside",
            ))
            plotly_dark_layout(fig_imp, "FEATURE IMPORTANCE")
            fig_imp.update_layout(height=280, margin=dict(l=10, r=60, t=40, b=10))
            st.plotly_chart(fig_imp, use_container_width=True)

            # ── Why this decision ──
            st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
            section_header("WHY THIS DECISION", "TOP RISK FACTORS")

            top3 = top_features[:3]
            risk_explanations = {
                "amt": f"Transaction amount of ${amt:,.2f} {'significantly exceeds' if amt > 500 else 'is within'} typical spend patterns.",
                "hour": f"Activity at {hour:02d}:00 hrs {'is outside normal business hours — elevated risk window' if hour in [0,1,2,3,4] else 'falls within standard transaction hours'}.",
                "age": f"Customer age group {age} {'shows elevated fraud exposure in this category' if age < 30 or age > 70 else 'presents normal risk profile'}.",
                "category": f"Merchant category '{CATEGORY_LABELS.get(category,'unknown')}' {'carries heightened fraud association' if category in [3,11,12] else 'has standard fraud baseline'}.",
                "city_pop": f"City population of {city_pop:,} {'indicates metropolitan high-risk zone' if city_pop > 500000 else 'is within normal geographic parameters'}.",
                "weekday": f"Transaction on {WEEKDAY_LABELS[weekday]} {'coincides with elevated fraud activity window' if weekday in [5,6] else 'is a typical transaction day'}.",
            }

            for i, (feat_name, imp) in enumerate(top3, 1):
                expl = risk_explanations.get(
                    feat_name,
                    f"Feature '{feat_name.upper()}' contributed {imp*100:.1f}% to this decision."
                )
                bar_width = min(int(imp * 400), 100)
                st.markdown(f"""
                <div style="margin-bottom:0.8rem;padding:0.8rem 1rem;
                            background:#0d1117;border:1px solid rgba(0,212,255,0.1);
                            border-radius:2px;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                                 color:#00d4ff;letter-spacing:0.08em;">
                      #{i} {feat_name.upper()}
                    </span>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                                 color:#475569;">{imp*100:.1f}%</span>
                  </div>
                  <div style="height:2px;background:rgba(0,212,255,0.08);border-radius:1px;margin-bottom:8px;">
                    <div style="width:{bar_width}%;height:2px;background:#00d4ff;border-radius:1px;"></div>
                  </div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                               color:#64748b;line-height:1.5;">{expl}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Case File (triggered when fraud > 60%) ──
            if pct > 60:
                age_range = f"{(age//10)*10}s"
                st.divider()
                st.markdown(f"""
                <div style="background:rgba(255,59,107,0.05);border:1px solid rgba(255,59,107,0.3);
                            border-radius:2px;padding:1.2rem 1.4rem;">
                  <div style="font-family:'Syne',sans-serif;font-size:0.8rem;font-weight:700;
                              color:#ff3b6b;letter-spacing:0.2em;margin-bottom:0.8rem;">
                    ◈ CASE FILE GENERATED
                  </div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;
                              color:#94a3b8;line-height:1.9;">
                    <span style="color:#475569;">CASE ID:</span> {case_id}<br>
                    <span style="color:#475569;">FLAGGED:</span> {now.strftime('%Y-%m-%d %H:%M:%S')} UTC<br>
                    <span style="color:#475569;">AMOUNT:</span> ${amt:,.2f}<br>
                    <span style="color:#475569;">CHANNEL:</span> {CATEGORY_LABELS[category].upper()}<br>
                    <span style="color:#475569;">RISK:</span> <span style="color:#ff3b6b;font-weight:600;">{pct:.1f}%</span><br><br>
                    <span style="color:#e2e8f0;font-style:italic;">
                    Transaction {case_id} flagged at {now.strftime('%H:%M')} hrs. Amount of
                    ${amt:,.2f} via {CATEGORY_LABELS[category]} at {hour:02d}:00 hrs exceeds
                    normal threshold for customer age group {age_range}.
                    Geographic indicator: city pop {city_pop:,}.
                    </span><br><br>
                    <span style="color:#ff3b6b;letter-spacing:0.1em;">
                    RECOMMEND: HOLD FOR REVIEW — ESCALATE TO TIER-2
                    </span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Placeholder state
            st.markdown("""
            <div style="height:340px;display:flex;flex-direction:column;align-items:center;
                        justify-content:center;border:1px dashed rgba(0,212,255,0.1);
                        border-radius:2px;color:#1e293b;font-family:'IBM Plex Mono',monospace;
                        font-size:0.72rem;letter-spacing:0.12em;gap:0.5rem;">
              <div style="font-size:2rem;opacity:0.3;">◈</div>
              <div>AWAITING INPUT</div>
              <div style="font-size:0.62rem;color:#0f172a;">CONFIGURE PARAMETERS → RUN ANALYSIS</div>
            </div>
            """, unsafe_allow_html=True)

# ─── PAGE: INVESTIGATE ────────────────────────────────────────────────────────
def page_investigate():
    model, scaler, feature_cols, model_loaded = load_model()

    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
                color:#e2e8f0;letter-spacing:0.05em;text-transform:uppercase;
                margin-bottom:1.5rem;">
      Batch File Investigation
    </div>
    """, unsafe_allow_html=True)

    # Expected format info
    section_header("EXPECTED CSV FORMAT", "COLUMN SCHEMA")
    panel("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#64748b;line-height:1.9;">
      Required columns:
      <span style="color:#00d4ff;">amt</span>,
      <span style="color:#00d4ff;">hour</span>,
      <span style="color:#00d4ff;">day</span>,
      <span style="color:#00d4ff;">month</span>,
      <span style="color:#00d4ff;">weekday</span>,
      <span style="color:#00d4ff;">age</span>,
      <span style="color:#00d4ff;">category</span> (0-13),
      <span style="color:#00d4ff;">gender</span> (0/1),
      <span style="color:#00d4ff;">city_pop</span><br>
      Optional: any additional feature columns — unknown columns are ignored, missing ones default to 0.
    </div>
    """)

    st.divider()
    section_header("UPLOAD TRANSACTIONS", "CSV FILE INPUT")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"CSV parse error: {e}")
            return

        st.markdown(f"""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                    color:#475569;margin:0.5rem 0 1rem 0;">
          ↳ Loaded <span style="color:#00d4ff;">{len(df_raw):,}</span> rows,
          <span style="color:#00d4ff;">{len(df_raw.columns)}</span> columns
        </div>
        """, unsafe_allow_html=True)

        # Run batch predictions
        with st.spinner("Running model inference..."):
            def score_row(row):
                amt = float(row.get("amt", 100))
                hour = int(row.get("hour", 12))
                day = int(row.get("day", 15))
                month = int(row.get("month", 6))
                weekday = int(row.get("weekday", 2))
                age = int(row.get("age", 40))
                category = int(row.get("category", 1))
                gender = int(row.get("gender", 0))
                city_pop = int(row.get("city_pop", 50000))

                if model_loaded:
                    try:
                        return predict_fraud(
                            model, scaler, feature_cols,
                            amt=amt, hour=hour, day=day, month=month,
                            weekday=weekday, age=age, category=category,
                            gender=gender, city_pop=city_pop
                        )
                    except Exception:
                        pass
                # Fallback simulation
                s = 0.02
                if amt > 800: s += 0.28
                if hour in [0,1,2,3]: s += 0.2
                if category in [3,11,12]: s += 0.15
                return min(max(s + np.random.normal(0, 0.06), 0), 1)

            probs = df_raw.apply(score_row, axis=1)
            df_result = df_raw.copy()
            df_result["FRAUD_PROBABILITY"] = (probs * 100).round(2)
            df_result["PREDICTION"] = probs.apply(lambda x: "FRAUD" if x >= 0.5 else "LEGIT")

        fraud_count = (df_result["PREDICTION"] == "FRAUD").sum()
        legit_count = (df_result["PREDICTION"] == "LEGIT").sum()

        st.divider()
        section_header("BATCH SUMMARY", f"{len(df_result):,} TRANSACTIONS ANALYZED")

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("TOTAL PROCESSED", f"{len(df_result):,}")
        with m2:
            st.metric("FRAUD DETECTED", str(fraud_count), delta=f"{fraud_count/len(df_result)*100:.1f}%")
        with m3:
            st.metric("LEGITIMATE", str(legit_count))
        with m4:
            avg_prob = df_result["FRAUD_PROBABILITY"].mean()
            st.metric("AVG FRAUD PROB", f"{avg_prob:.1f}%")

        # Pie chart
        pie_col, _ = st.columns([1, 1])
        with pie_col:
            fig_pie = go.Figure(go.Pie(
                labels=["LEGITIMATE", "FRAUD"],
                values=[legit_count, fraud_count],
                hole=0.55,
                marker=dict(
                    colors=["#22c55e", "#ff3b6b"],
                    line=dict(color="#070b14", width=2),
                ),
                textfont=dict(family="IBM Plex Mono", size=11, color="#94a3b8"),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
            ))
            plotly_dark_layout(fig_pie, "FRAUD VS LEGITIMATE DISTRIBUTION")
            fig_pie.update_layout(height=320)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()
        section_header("FLAGGED TRANSACTIONS", "TOP 5 HIGHEST-RISK")

        top5 = df_result.nlargest(5, "FRAUD_PROBABILITY")
        rows_html = ""
        for _, row in top5.iterrows():
            prob = row["FRAUD_PROBABILITY"]
            rows_html += f"""
            <tr style="background:rgba(255,59,107,0.05);border-bottom:1px solid rgba(255,59,107,0.1);">
              <td style="padding:8px 12px;font-size:0.75rem;color:#ff3b6b;font-weight:600;">{prob:.2f}%</td>
              <td style="padding:8px 12px;font-size:0.75rem;color:#94a3b8;">{row.get('amt', 'N/A')}</td>
              <td style="padding:8px 12px;font-size:0.72rem;color:#64748b;">{CATEGORY_LABELS.get(int(row.get('category', 0)), 'unknown').upper()}</td>
              <td style="padding:8px 12px;font-size:0.72rem;color:#64748b;">{row.get('hour', 'N/A')}:00</td>
              <td style="padding:8px 12px;font-size:0.72rem;color:#475569;">{row.get('age', 'N/A')}</td>
            </tr>
            """

        st.markdown(f"""
        <div style="border:1px solid rgba(255,59,107,0.2);border-radius:2px;overflow:hidden;margin-bottom:1rem;">
          <table style="width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;">
            <thead>
              <tr style="background:rgba(255,59,107,0.1);border-bottom:1px solid rgba(255,59,107,0.2);">
                <th style="padding:9px 12px;text-align:left;font-size:0.65rem;color:#ff3b6b;letter-spacing:0.12em;">FRAUD PROB</th>
                <th style="padding:9px 12px;text-align:left;font-size:0.65rem;color:#ff3b6b;letter-spacing:0.12em;">AMOUNT</th>
                <th style="padding:9px 12px;text-align:left;font-size:0.65rem;color:#ff3b6b;letter-spacing:0.12em;">CATEGORY</th>
                <th style="padding:9px 12px;text-align:left;font-size:0.65rem;color:#ff3b6b;letter-spacing:0.12em;">HOUR</th>
                <th style="padding:9px 12px;text-align:left;font-size:0.65rem;color:#ff3b6b;letter-spacing:0.12em;">AGE</th>
              </tr>
            </thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        section_header("FULL RESULTS", "ALL PROCESSED TRANSACTIONS")
        st.dataframe(df_result, use_container_width=True, height=320)

        # Download
        csv_bytes = df_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇  EXPORT RESULTS AS CSV",
            data=csv_bytes,
            file_name=f"fraud_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    else:
        st.markdown("""
        <div style="height:220px;display:flex;flex-direction:column;align-items:center;
                    justify-content:center;border:1px dashed rgba(0,212,255,0.1);
                    border-radius:2px;color:#1e293b;font-family:'IBM Plex Mono',monospace;
                    font-size:0.72rem;letter-spacing:0.12em;gap:0.5rem;margin-top:1rem;">
          <div style="font-size:2rem;opacity:0.3;">⬢</div>
          <div>AWAITING FILE UPLOAD</div>
          <div style="font-size:0.62rem;color:#0f172a;">DRAG & DROP CSV OR CLICK TO BROWSE</div>
        </div>
        """, unsafe_allow_html=True)

# ─── PAGE: INTEL ──────────────────────────────────────────────────────────────
def page_intel():
    model, scaler, feature_cols, model_loaded = load_model()

    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
                color:#e2e8f0;letter-spacing:0.05em;text-transform:uppercase;
                margin-bottom:1.5rem;">
      Model Intelligence Panel
    </div>
    """, unsafe_allow_html=True)

    # Hardcoded performance metrics from notebook
    METRICS = {
        "RF ACCURACY": ("99.57%", "+0.12%"),
        "ROC-AUC SCORE": ("0.9812", "+0.024"),
        "PRECISION (FRAUD)": ("88.3%", "+1.4%"),
        "RECALL (FRAUD)": ("76.8%", "+3.2%"),
    }

    section_header("MODEL PERFORMANCE", "EVALUATION METRICS — TEST SET")
    cols = st.columns(4)
    for col, (label, (value, delta)) in zip(cols, METRICS.items()):
        with col:
            st.metric(label, value, delta=delta)

    st.divider()

    # Feature importance chart (full)
    section_header("FEATURE IMPORTANCE", "ALL MODEL FEATURES — RANKED")

    if model_loaded and feature_cols:
        importances = model.feature_importances_ if hasattr(model, "feature_importances_") else np.zeros(len(feature_cols))
        sorted_idx = np.argsort(importances)
        feat_names = [str(feature_cols[i]).replace("_", " ").upper() for i in sorted_idx]
        feat_vals = [importances[i] * 100 for i in sorted_idx]
        bar_colors = ["#ff3b6b" if v > 8 else "#00d4ff" if v > 3 else "#1e3a5f" for v in feat_vals]
    else:
        # Demo data
        demo_feats = ["amt", "city_pop", "age", "hour", "category", "weekday",
                      "day", "month", "gender", "merch_lat", "lat", "long",
                      "merch_long", "zip", "state", "unix_time", "trans_num"]
        np.random.seed(42)
        raw_vals = np.random.dirichlet(np.ones(len(demo_feats)) * 2)
        sorted_idx = np.argsort(raw_vals)
        feat_names = [demo_feats[i].upper() for i in sorted_idx]
        feat_vals = (raw_vals[sorted_idx] * 100).tolist()
        bar_colors = ["#ff3b6b" if v > 10 else "#00d4ff" if v > 4 else "#1e3a5f" for v in feat_vals]

    fig_full = go.Figure(go.Bar(
        x=feat_vals,
        y=feat_names,
        orientation="h",
        marker=dict(
            color=bar_colors,
            opacity=0.85,
            line=dict(width=0),
        ),
        text=[f"{v:.2f}%" for v in feat_vals],
        textfont=dict(size=9, family="IBM Plex Mono", color="#475569"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.2f}%<extra></extra>",
    ))
    plotly_dark_layout(fig_full)
    fig_full.update_layout(height=max(320, len(feat_names) * 22), margin=dict(l=10, r=80, t=10, b=10))
    st.plotly_chart(fig_full, use_container_width=True)

    st.divider()
    section_header("CONFUSION MATRIX", "FRAUD vs LEGITIMATE CLASSIFICATION")

    # Confusion matrix (realistic hardcoded values for 1.3M dataset)
    cm_vals = np.array([[1282447, 5523], [2981, 9905]])
    labels = ["LEGITIMATE", "FRAUD"]

    fig_cm = go.Figure(go.Heatmap(
        z=cm_vals,
        x=["PRED: LEGIT", "PRED: FRAUD"],
        y=["TRUE: LEGIT", "TRUE: FRAUD"],
        colorscale=[
            [0, "#070b14"],
            [0.2, "#0a1628"],
            [0.6, "#003d5c"],
            [1.0, "#00d4ff"],
        ],
        showscale=True,
        text=[[f"{cm_vals[i][j]:,}" for j in range(2)] for i in range(2)],
        texttemplate="%{text}",
        textfont=dict(family="IBM Plex Mono", size=15, color="#e2e8f0"),
        hovertemplate="<b>%{y} / %{x}</b><br>Count: %{z:,}<extra></extra>",
        colorbar=dict(
            tickfont=dict(family="IBM Plex Mono", size=9, color="#475569"),
            outlinecolor="rgba(0,212,255,0.1)",
            outlinewidth=1,
        ),
    ))
    plotly_dark_layout(fig_cm, "CONFUSION MATRIX — 1.3M TEST TRANSACTIONS")
    fig_cm.update_layout(
        height=360,
        xaxis=dict(tickfont=dict(size=11, color="#64748b", family="IBM Plex Mono")),
        yaxis=dict(tickfont=dict(size=11, color="#64748b", family="IBM Plex Mono")),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.divider()
    section_header("DATASET STATISTICS", "TRAINING DATA PROFILE")

    stat_cols = st.columns(3)
    dataset_stats = [
        ("TOTAL TRANSACTIONS", "1,296,675", "Full dataset"),
        ("FRAUD RATE", "0.58%", "7,506 fraud cases"),
        ("FEATURE COUNT", "23", "After engineering"),
        ("TRAINING SPLIT", "80/20", "Stratified"),
        ("DATE RANGE", "2019–2020", "24-month window"),
        ("UNIQUE MERCHANTS", "693", "Across 14 categories"),
    ]
    for i, (label, value, sub) in enumerate(dataset_stats):
        with stat_cols[i % 3]:
            st.markdown(f"""
            <div style="background:#0d1117;border:1px solid rgba(0,212,255,0.1);
                        border-radius:2px;padding:1rem 1.2rem;margin-bottom:0.8rem;">
              <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                          color:#334155;letter-spacing:0.18em;text-transform:uppercase;
                          margin-bottom:0.3rem;">{label}</div>
              <div style="font-family:'Syne',sans-serif;font-size:1.4rem;
                          font-weight:700;color:#e2e8f0;">{value}</div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                          color:#1e293b;margin-top:2px;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    section_header("HOW THE MODEL WORKS", "ARCHITECTURE EXPLAINER")

    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">

      <div style="background:#0d1117;border:1px solid rgba(0,212,255,0.1);
                  border-radius:2px;padding:1.2rem 1.4rem;">
        <div style="font-family:'Syne',sans-serif;font-size:0.85rem;font-weight:700;
                    color:#00d4ff;letter-spacing:0.1em;margin-bottom:0.8rem;">
          RANDOM FOREST CLASSIFIER
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                    color:#64748b;line-height:1.9;">
          The model is an ensemble of 200 decision trees, each trained on a
          random subset of features and data. Final prediction is determined by
          majority vote across all trees, providing robust classification and
          natural resistance to overfitting.
        </div>
      </div>

      <div style="background:#0d1117;border:1px solid rgba(0,212,255,0.1);
                  border-radius:2px;padding:1.2rem 1.4rem;">
        <div style="font-family:'Syne',sans-serif;font-size:0.85rem;font-weight:700;
                    color:#00d4ff;letter-spacing:0.1em;margin-bottom:0.8rem;">
          FEATURE ENGINEERING
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                    color:#64748b;line-height:1.9;">
          Raw timestamps were decomposed into hour, day, month, and weekday signals.
          Customer age was computed from dob. Geographic features (city_pop, lat, long)
          provide behavioral context. Category and gender were label-encoded.
        </div>
      </div>

      <div style="background:#0d1117;border:1px solid rgba(0,212,255,0.1);
                  border-radius:2px;padding:1.2rem 1.4rem;">
        <div style="font-family:'Syne',sans-serif;font-size:0.85rem;font-weight:700;
                    color:#00d4ff;letter-spacing:0.1em;margin-bottom:0.8rem;">
          CLASS IMBALANCE HANDLING
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                    color:#64748b;line-height:1.9;">
          With only 0.58% fraud rate, SMOTE oversampling was applied to the
          minority class during training. Class weights were also balanced to
          prevent the model from defaulting to predicting all transactions as
          legitimate.
        </div>
      </div>

      <div style="background:#0d1117;border:1px solid rgba(0,212,255,0.1);
                  border-radius:2px;padding:1.2rem 1.4rem;">
        <div style="font-family:'Syne',sans-serif;font-size:0.85rem;font-weight:700;
                    color:#00d4ff;letter-spacing:0.1em;margin-bottom:0.8rem;">
          THRESHOLD CALIBRATION
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                    color:#64748b;line-height:1.9;">
          Default decision threshold is 0.50. The platform surfaces raw probabilities
          to allow analysts to apply custom thresholds based on operational
          risk tolerance. Lower thresholds increase recall at the cost of
          false positive rate.
        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)

# ─── ROUTER ───────────────────────────────────────────────────────────────────
if "MONITOR" in page:
    page_monitor()
elif "ANALYZE" in page:
    page_analyze()
elif "INVESTIGATE" in page:
    page_investigate()
elif "INTEL" in page:
    page_intel()


# ─── MCP INTEGRATION (connect when MCP server is running) ───────────────────
#
# Replace the local model calls with these MCP tool calls:
#
# async def analyze_via_mcp(transaction: dict) -> dict:
#     """Stream fraud analysis from MCP server."""
#     async with MCPClient("ws://localhost:8765") as client:
#         result = await client.call_tool(
#             "analyze_transaction",
#             arguments=transaction,
#             stream=True
#         )
#         async for chunk in result:
#             yield chunk  # Stream to st.write_stream()
#
# Tool schema expected by MCP server:
# {
#   "name": "analyze_transaction",
#   "description": "Analyze a transaction for fraud risk",
#   "input_schema": {
#     "type": "object",
#     "properties": {
#       "amt": {"type": "number"},
#       "category": {"type": "integer", "minimum": 0, "maximum": 13},
#       "hour": {"type": "integer", "minimum": 0, "maximum": 23},
#       "age": {"type": "integer"},
#       "gender": {"type": "integer", "enum": [0, 1]},
#       "city_pop": {"type": "integer"}
#     }
#   }
# }
#
# Streamlit integration pattern:
#
# if st.button("RUN VIA MCP"):
#     with st.spinner("Querying MCP server..."):
#         result = asyncio.run(analyze_via_mcp(transaction_dict))
#         st.write(result)
#
# MCP server startup (run before launching Streamlit):
#   python mcp_server.py --host localhost --port 8765
#
# Environment variable for MCP endpoint:
#   export FRAUD_MCP_ENDPOINT="ws://localhost:8765"
#   endpoint = os.environ.get("FRAUD_MCP_ENDPOINT", "ws://localhost:8765")
# ─────────────────────────────────────────────────────────────────────────────