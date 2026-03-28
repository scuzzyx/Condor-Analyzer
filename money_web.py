import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- CONFIG ---
st.set_page_config(page_title="Money Machine Pro V3.2.7", layout="wide", initial_sidebar_state="expanded")
st.title("⚙️ Money Machine Pro V3.2.7 (High-Flow Data)")

# --- 📖 INSTRUCTIONS ---
with st.expander("📖 How to Use This Engine & Risk Legend", expanded=False):
    st.markdown("""
    ### 🧠 The Workflow
    1. **Build Your Bench:** Add tickers in the sidebar. Use the **Radar** to scan for sideways movers.
    2. **Check Correlation:** Avoid trading high-correlation pairs (0.85+) simultaneously.
    3. **Review Setups:** Check risk grades, **IV**, and **Earnings Dates**.
    """)

Z_SCORES = {"70%": 1.04, "75%": 1.15, "80%": 1.28, "85%": 1.44, "90%": 1.645, "95%": 1.96}

def load_url_bench():
    if "bench" in st.query_params:
        return st.query_params["bench"].split(",")
    return ["AMZN", "AAPL", "MSFT", "META", "GOOGL", "NVDA", "AMD", "PLTR", "TSLA", "NFLX"]

# --- QUANT HELPERS ---
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    return 100 - (100 / (1 + (gain / loss)))

@st.cache_data(ttl=3600)  
def get_friday_expirations():
    try:
        spy = yf.Ticker("SPY")
        dates = spy.options
        return [d for d in dates if datetime.strptime(d, '%Y-%m-%d').weekday() == 4][:10]
    except: return []

# --- SIDEBAR ---
st.sidebar.header("🛠️ Dashboard Controls")
url_bench = load_url_bench()
if 'c_bench' not in st.session_state: st.session_state['c_bench'] = list(set(url_bench + ["SPY"]))
if 'a_sel' not in st.session_state: st.session_state['a_sel'] = url_bench

def add_ticker():
    t = st.session_state['t_in'].upper().strip()
    if t:
        if t not in st.session_state['c_bench']: st.session_state['c_bench'].append(t)
        active = st.session_state['a_sel'].copy()
        if t not in active: active.append(t); st.session_state['a_sel'] = active
    st.session_state['t_in'] = ""

st.sidebar.text_input("➕ Add Ticker:", key="t_in", on_change=add_ticker)
selected_tickers = st.sidebar.multiselect("Active Bench:", options=st.session_state['c_bench'], key="a_sel")

if st.sidebar.button("🔗 Generate Custom Link"):
    st.query_params["bench"] = ",".join(st.session_state['a_sel'])
    st.sidebar.success("URL updated!")

st.sidebar.markdown("---")
fridays = get_friday_expirations()
if fridays:
    exp_str = st.sidebar.selectbox("Expiration Date:", options=fridays)
    dte = (datetime.strptime(exp_str, '%Y-%m-%d') - datetime.now()).days
else: dte, exp_str = 14, None

z_val = Z_SCORES[st.sidebar.selectbox("Success Target:", options=list(Z_SCORES.keys()), index=4)]

# --- RADAR ---
st.sidebar.subheader("📡 Radar")
L50 = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AMD', 'PLTR', 'NFLX']
if st.sidebar.button("Run Radar"):
    with st.sidebar.status("Scanning..."):
        hits = []
        for s in L50:
            try:
                px = yf.Ticker(s).history(period="1mo")['Close']
                if (px.max() - px.min()) / px.iloc[-1] < 0.08: hits.append(s)
            except: continue
        if hits: st.sidebar.success(f"🎯 Targets: {', '.join(hits)}")

# --- MATRIX ---
st.markdown("---")
if len(selected_tickers) > 1:
    with st.expander("🧩 Correlation Matrix", expanded=False):
        try:
            dfs = []
            for s in selected_tickers:
                px = yf.Ticker(s).history(period="3mo")['Close']
                px.name = s
                dfs.append(px)
            st.dataframe(pd.concat(dfs, axis=1).pct_change().tail(30).corr().style.background_gradient(cmap='coolwarm'))
        except: st.write("Data currently unavailable.")

# --- MAIN ENGINE ---
for sym in selected_tickers:
    try:
        t_obj = yf.Ticker(sym)
        h = t_obj.history(period="3mo")
        if h.empty: continue
            
        c = h['Close'].iloc[-1]; p = h['Close'].iloc[-2]
        rsi = calculate_rsi(h['Close']).iloc[-1]
        vol = np.std(h['Close'].pct_change().dropna()) * np.sqrt(dte if dte > 0 else 1)
        move = c * (vol * z_val)
        ps, cs = round(c - move), round(c + move)
        pt, ct = round(ps * 1.05, 2), round(cs * 0.95, 2)

        iv, ed, veto = "N/A", "Not scheduled", False
        try:
            if exp_str:
                chain = t_obj.option_chain(exp_str)
                idx = (chain.calls['strike'] - c).abs().idxmin()
                iv = f"{chain.calls.loc[idx, 'impliedVolatility'] * 100:.1f}%"
        except: pass
        try:
            cal = t_obj.calendar
            if not cal.empty:
                e_date = cal.iloc[0,0] if isinstance(cal, pd.DataFrame) else cal.get('Earnings Date')[0]
                ed = e_date.strftime('%Y-%m-%d')
                if datetime.now() < e_date < (datetime.now() + timedelta(days=dte)): veto = True
        except: pass

        if veto: r, clr = "⛔ DO NOT TRADE (EARNINGS VETO)", "red"
        elif rsi < 35 or c <= h['Close'].min(): r, clr = "🔴 HIGH RISK", "red"
        elif c > h['Close'].rolling(20).mean().iloc[-1]: r, clr = "🟢 LOW RISK", "green"
        else: r, clr = "🟡 MED RISK", "orange"

        with st.expander(f"{sym} | ${c:.2f} | {r}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Change", f"${c:.2f}", f"{c-p:.2f}")
            c2.metric("Put Strategy", f"Strike: ${ps}", f"Trip: ${pt}", delta_color="off")
            c3.metric("Call Strategy", f"Strike: ${cs}", f"Trip: ${ct}", delta_color="off")
            c4.metric("Market Data", f"IV: {iv}", f"Earnings: {ed}", delta_color="off")
            fig = go.Figure(data=[go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'])])
            fig.add_hline(y=cs, line_color="red"); fig.add_hline(y=ps, line_color="green")
            fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
    except: st.error(f"Error {sym}")
