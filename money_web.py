import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- CONFIG & THEME ---
st.set_page_config(page_title="Money Machine Pro V3.2.1", layout="wide", initial_sidebar_state="expanded")
st.title("⚙️ Money Machine Pro V3.2.1")

# --- DETAILED INSTRUCTIONS ---
with st.expander("📖 How to Use This Engine & Risk Legend", expanded=False):
    st.markdown("""
    ### 🧠 The Strategic Workflow
    1. **Build Your Bench:** Add tickers manually in the sidebar or use the **Range-Bound Radar** to find stocks currently consolidating (chopping sideways).
    2. **Check Correlation:** Look at the **30-Day Matrix**. If two stocks are highly correlated (0.85+), they move together. Trading Iron Condors on both simultaneously effectively doubles your risk on a single move.
    3. **Review Risk Grades:** Expand a ticker to see the automated grade. 
    
    ### 🚦 Risk Legend & Veto System
    * 🟢 **LOW RISK (Neutral Chop):** Stock is holding above 3-month support and its 20-day Moving Average. High probability for Iron Condors.
    * 🟡 **MED RISK (Stalling):** Stock is above support but struggling to stay above its 20-day MA. Proceed with caution.
    * 🟡 **TRENDING (ADX > 25):** The stock has strong momentum. Iron Condors are risky here; consider directional credit spreads instead.
    * 🟠 **GAP RISK (> 1.5%):** The stock historically "gaps" (jumps) significantly overnight. This can bypass your stop-losses.
    * 🔴 **HIGH RISK (Falling Knife):** RSI is oversold or the stock broke a major support floor. Do not attempt to catch it.
    * ⛔ **EARNINGS VETO:** An earnings report occurs before your expiration. The engine suggests skipping to avoid the volatility spike.
    """)

# --- PROBABILITY Z-SCORES ---
Z_SCORES = {"70%": 1.04, "75%": 1.15, "80%": 1.28, "85%": 1.44, "90%": 1.645, "95%": 1.96}

# --- URL MEMORY ---
def load_url_bench():
    if "bench" in st.query_params:
        return st.query_params["bench"].split(",")
    return ["AMZN", "AAPL", "MSFT", "META", "GOOGL", "NVDA", "AMD", "PLTR", "TSLA", "NFLX"]

# --- QUANT HELPERS ---
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_adx(hist, period=14):
    try:
        high, low, close = hist['High'], hist['Low'], hist['Close']
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_dm = high.diff(); minus_dm = low.diff()
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
        plus_di = 100 * (pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        return dx.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    except: return 20

def calculate_gap_risk(hist):
    try: return abs((hist['Open'] - hist['Close'].shift(1)) / hist['Close'].shift(1)).tail(30).mean() * 100
    except: return 0

@st.cache_data(ttl=3600)
def get_friday_expirations():
    try:
        dates = yf.Ticker("SPY").options
        return [d for d in dates if datetime.strptime(d, '%Y-%m-%d').weekday() == 4][:10]
    except: return []

@st.cache_data(ttl=3600)
def run_radar_scan(ticker_list, threshold):
    found = []
    try:
        data = yf.download(ticker_list, period="1mo", group_by='ticker', progress=False)
        for sym in ticker_list:
            try:
                h = data[sym]['Close'].dropna()
                if not h.empty and (h.max() - h.min()) / h.iloc[-1] < threshold:
                    found.append(sym)
            except: continue
    except: pass
    return found

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🛠️ Dashboard Controls")
url_bench = load_url_bench()
if 'custom_bench' not in st.session_state: st.session_state['custom_bench'] = list(set(url_bench + ["SPY", "QQQ"]))
if 'active_selections' not in st.session_state: st.session_state['active_selections'] = url_bench

def add_custom_ticker():
    t = st.session_state['ticker_input'].upper().strip()
    if t:
        if t not in st.session_state['custom_bench']: st.session_state['custom_bench'].append(t)
        active = st.session_state['active_selections'].copy()
        if t not in active:
            active.append(t); st.session_state['active_selections'] = active
    st.session_state['ticker_input'] = ""

st.sidebar.text_input("➕ Add Custom Ticker:", key="ticker_input", on_change=add_custom_ticker)
selected_tickers = st.sidebar.multiselect("Active Bench:", options=st.session_state['custom_bench'], key="active_selections")

st.sidebar.caption("Note: To save your bench loadout, use this custom link.")
if st.sidebar.button("🔗 Generate Custom Link"):
    st.query_params["bench"] = ",".join(st.session_state['active_selections'])
    st.sidebar.success("URL updated! Bookmark it now.")

st.sidebar.markdown("---")
fridays = get_friday_expirations()
if fridays:
    exp_str = st.sidebar.selectbox("Expiration Date:", options=fridays)
    dte = (datetime.strptime(exp_str, '%Y-%m-%d') - datetime.now()).days
else: dte = 14; exp_str = None

prob_target = st.sidebar.selectbox("Success Target:", options=list(Z_SCORES.keys()), index=4)
z_val = Z_SCORES[prob_target]

# --- RADAR ---
st.sidebar.markdown("---")
st.sidebar.subheader("📡 Range-Bound Radar")
LIQUID_50 = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AMD', 'PLTR', 'NFLX', 'BA', 'DIS', 'BABA', 'UBER', 'COIN', 'HOOD', 'INTC', 'MU', 'AVGO', 'TSM', 'JPM', 'BAC', 'C', 'V', 'MA', 'PYPL', 'SQ', 'WMT', 'TGT', 'COST', 'HD', 'SBUX', 'NKE', 'MCD', 'XOM', 'CVX', 'CAT', 'GE', 'JNJ', 'PFE', 'UNH', 'LLY', 'CMCSA', 'VZ', 'T', 'QCOM', 'CRM', 'SNOW', 'SHOP', 'SPOT']
SP_100 = ['AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'AMD', 'AMGN', 'AMT', 'AMZN', 'AXP', 'BA', 'BAC', 'BK', 'BKNG', 'BLK', 'BMY', 'C', 'CAT', 'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS
