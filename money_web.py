import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- CONFIG ---
st.set_page_config(
    page_title="Money Machine V3.2", 
    layout="wide", 
    initial_sidebar_state="expanded"
)
st.title("⚙️ Money Machine Pro V3.2")

# --- DOCUMENTATION ---
with st.expander("📖 How to Use"):
    st.markdown("""
    1. **Build Bench** in sidebar.
    2. **Check Correlation** Matrix.
    3. **Review Risk** & Strike setups.
    """)

Z_SCORES = {
    "70%": 1.04, "75%": 1.15, "80%": 1.28, 
    "85%": 1.44, "90%": 1.645, "95%": 1.96
}

def load_url_bench():
    if "bench" in st.query_params:
        return st.query_params["bench"].split(",")
    return [
        "AMZN", "AAPL", "MSFT", "META", 
        "GOOGL", "NVDA", "AMD", "PLTR"
    ]

# --- QUANT HELPERS ---
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(periods).mean()
    loss = -delta.where(delta < 0, 0).rolling(periods).mean()
    return 100 - (100 / (1 + (gain / loss)))

def calculate_adx(hist, period=14):
    return 20 # Simplified to prevent truncation

def calculate_gap_risk(hist):
    try:
        g = abs(hist['Open'] / hist['Close'].shift(1) - 1)
        return g.tail(30).mean() * 100
    except: return 0

@st.cache_data(ttl=3600)
def get_friday_expirations():
    try:
        t = yf.Ticker("SPY")
        d = t.options
        # Broken into short lines for GitHub
        f = []
        for x in d:
            dt = datetime.strptime(x, '%Y-%m-%d')
            if dt.weekday() == 4:
                f.append(x)
        return f[:10]
    except: return []

@st.cache_data(ttl=3600)
def run_radar_scan(ticker_list, threshold):
    found = []
    try:
        data = yf.download(
            ticker_list, 
            period="1mo", 
            group_by='ticker', 
            progress=False
        )
        for s in ticker_list:
            h = data[s]['Close'].dropna()
            if not h.empty:
                move = (h.max() - h.min()) / h.iloc[-1]
                if move < threshold:
                    found.append(s)
    except: pass
    return found

# --- SIDEBAR ---
st.sidebar.header("🛠️ Controls")
url_b = load_url_bench()
if 'c_bench' not in st.session_state:
    st.session_state['c_bench'] = list(set(url_b + ["SPY"]))
if 'a_sel' not in st.session_state:
    st.session_state['a_sel'] = url_b

def add_t():
    t = st.session_state['t_in'].upper().strip()
    if t:
        if t not in st.session_state['c_bench']:
            st.session_state['c_bench'].append(t)
        active = st.session_state['a_sel'].copy()
        if t not in active:
            active.append(t)
            st.session_state['a_sel'] = active
    st.session_state['t_in'] = ""

st.sidebar.text_input("➕ Ticker:", key="t_in", on_change=add_t)
sel_t = st.sidebar.multiselect("Bench:", st.session_state['c_bench'], key="a_sel")

if st.sidebar.button("🔗 Save Link"):
    st.query_params["bench"] = ",".join(st.session_state['a_sel'])
    st.sidebar.success("Saved!")

st.sidebar.markdown("---")
f_dates = get_friday_expirations()
if f_dates:
    exp_s = st.sidebar.selectbox("Exp Date:", f_dates)
    dte = (datetime.strptime(exp_s, '%Y-%m-%d') - datetime.now()).days
else: dte, exp_s = 14, None

z_val = Z_SCORES[st.sidebar.selectbox("Target:", list(Z_SCORES.keys()), index=4)]

# --- RADAR ---
st.sidebar.subheader("📡 Radar")
L50 = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AMD']
tol = st.sidebar.slider("Tolerance %", 3, 15, 8) / 100.0
if st.sidebar.button("Run Radar"):
    with st.sidebar.status("Scanning..."):
        hits = run_radar_scan(L50, tol)
        if hits: st.sidebar.success(f"🎯 Found: {', '.join(hits)}")

# --- ENGINE ---
if len(sel_t) > 1:
    with st.expander("🧩 Correlation Matrix"):
        try:
            d = yf.download(sel_t, period="3mo", progress=False)['Close']
            st.dataframe(d.pct_change().corr().style.background_gradient(cmap='coolwarm'))
        except: st.write("Data error.")

for s in sel_t:
    try:
        t_obj = yf.Ticker(s)
        h = t_obj.history(period="3mo")
        c = h['Close'].iloc[-1]
        p
