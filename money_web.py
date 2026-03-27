import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# 1. Page Config
st.set_page_config(page_title="Money Machine Pro", layout="wide", initial_sidebar_state="expanded")

# 2. Refined CSS (Desktop Pro Look)
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; }
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: bold; }
    .status-badge {
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85rem;
        float: right;
    }
    /* Syncing container style to match desktop app cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Logic Engine ---
def get_ticker_data(symbol):
    try:
        t = yf.Ticker(symbol.upper().strip())
        hist = t.history(period="3mo")
        if hist.empty: return None
        cp = hist['Close'].iloc[-1]
        ma20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        floor = hist['Close'].min()
        vol = np.std(hist['Close'].pct_change().dropna()) * np.sqrt(14)
        move = cp * (vol * 1.645)
        risk = "LOW" if cp > floor and cp > ma20 else "MED" if cp > floor else "HIGH"
        color = "#3fb950" if risk == "LOW" else "#d29922" if risk == "MED" else "#f85149"
        return {"symbol": symbol.upper(), "price": cp, "risk": risk, "color": color, 
                "put": round(cp - move), "call": round(cp + move), "ma20": round(ma20, 2), "floor": round(floor, 2)}
    except: return None

# --- Reusable UI Card Function ---
def display_ticker_card(data):
    with st.container(border=True):
        h1, h2 = st.columns([2, 1])
        h1.markdown(f"### {data['symbol']}")
        h2.markdown(f"<span class='status-badge' style='background:{data['color']}; color:white;'>{data['risk']}</span>", unsafe_allow_html=True)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Price", f"${data['price']:.2f}")
        m2.metric("90% Put", f"${data['put']}")
        m3.metric("90% Call", f"${data['call']}")
        st.caption(f"20-Day MA: ${data['ma20']} | 3-Mo Floor: ${data['floor']}")

# --- SIDEBAR ---
with st.sidebar:
    st.title("MM Pro")
    st.write("---")
    search_query = st.text_input("Manual Search", placeholder="Enter Ticker...").upper()
    if st.button("🔄 Refresh Bench"):
        st.rerun()

# --- MAIN DASHBOARD ---
st.title("💰 Market Velocity Dashboard")

# 1. Manual Search Result (Now styled as a Card)
if search_query:
    st.subheader(f"Manual Scan: {search_query}")
    search_data = get_ticker_data(search_query)
    if search_data:
        display_ticker_card(search_data)
    else:
        st.error(f"Could not find ticker: {search_query}")
    st.divider()

# 2. Master Bench (Grid Layout)
st.header("Master Bench Status")
bench = ["AMZN", "AAPL", "MSFT", "META", "GOOGL", "NVDA", "AMD", "PLTR", "TSLA", "NFLX"]
grid_cols = st.columns(2)

for i, ticker in enumerate(bench):
    data = get_ticker_data(ticker)
    if data:
        with grid_cols[i % 2]:
            display_ticker_card(data)