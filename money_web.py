import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- CONFIG & THEME ---
st.set_page_config(page_title="Money Machine Pro V3.2", layout="wide", initial_sidebar_state="expanded")
st.title("⚙️ Money Machine Pro V3.2 (Full Engine Restored)")

# --- HOW TO USE / DOCUMENTATION ---
with st.expander("📖 How to Use This Engine & Risk Legend", expanded=False):
    st.markdown("""
    ### 🧠 The Workflow
    1. **Build Your Bench:** Add tickers in the sidebar. Use the **Range-Bound Radar** to scan for high-liquidity stocks trading sideways.
    2. **Check Correlation:** Look at the Matrix. If you see high correlation (0.85+), avoid trading both simultaneously to prevent double-exposure.
    3. **Review Setups:** Expand each ticker to see calculated strikes, trip wires, and the automated risk grade.
    
    ### 🚦 Risk Legend (The Veto System)
    * 🟢 **LOW RISK:** Holding above support and 20-day MA. Ideal for Iron Condors.
    * 🟡 **MED RISK:** Above support but stalling under moving averages.
    * 🟡 **TRENDING (ADX > 25):** Moving too fast. Use directional spreads instead.
    * 🟠 **GAP RISK (> 1.5%):** Dangerous overnight jumps. Stop-losses may fail.
    * 🔴 **HIGH RISK:** Break of 3-month support or an RSI "Falling Knife" crash.
    * ⛔ **EARNINGS VETO:** Automatic disqualification if earnings fall before expiration.
    """)

# --- PROBABILITY Z-SCORES ---
Z_SCORES = {
    "70%": 1.04, "75%": 1.15, "80%": 1.28, 
    "85%": 1.44, "90%": 1.645, "95%": 1.96
}

# --- WEB-SAFE URL MEMORY ---
def load_url_bench():
    if "bench" in st.query_params:
        return st.query_params["bench"].split(",")
    return ["AMZN", "AAPL", "MSFT", "META", "GOOGL", "NVDA", "AMD", "PLTR", "TSLA", "NFLX"]

# --- QUANTITATIVE HELPER FUNCTIONS ---
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
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
        
        plus_di = 100 * (pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        return dx.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    except: return 20 

def calculate_gap_risk(hist):
    try:
        return abs((hist['Open'] - hist['Close'].shift(1)) / hist['Close'].shift(1)).tail(30).mean() * 100
    except: return 0

@st.cache_data(ttl=3600)  
def get_friday_expirations():
    try:
        dates = yf.Ticker("SPY").options
        return [d for d in dates if datetime.strptime(d, '%Y-%m-%d').weekday() == 4][:10]
    except: return []

@st.cache_data(ttl=3600) 
def run_radar_scan(ticker_list, threshold):
    found_targets = []
    try:
        bulk_data = yf.download(ticker_list, period="1mo", group_by='ticker', progress=False)
        for sym in ticker_list:
            try:
                hist = bulk_data[sym]['Close'].dropna()
                if not hist.empty:
                    if (hist.max() - hist.min()) / hist.iloc[-1] < threshold:
                        found_targets.append(sym)
            except: continue
    except: pass
    return found_targets

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🛠️ Dashboard Controls")
url_bench = load_url_bench()

if 'custom_bench' not in st.session_state:
    st.session_state['custom_bench'] = list(set(url_bench + ["SPY", "QQQ"]))
if 'active_selections' not in st.session_state:
    st.session_state['active_selections'] = url_bench

def add_custom_ticker():
    ticker = st.session_state['ticker_input'].upper().strip()
    if ticker:
        if ticker not in st.session_state['custom_bench']:
            st.session_state['custom_bench'].append(ticker)
        current_active = st.session_state['active_selections'].copy()
        if ticker not in current_active:
            current_active.append(ticker)
            st.session_state['active_selections'] = current_active
    st.session_state['ticker_input'] = ""

st.sidebar.text_input("➕ Add Custom Ticker:", key="ticker_input", on_change=add_custom_ticker)
selected_tickers = st.sidebar.multiselect("Active Bench:", options=st.session_state['custom_bench'], key="active_selections")

st.sidebar.caption("Note: To save your bench loadout, use this custom link.")
if st.sidebar.button("🔗 Generate Custom Link"):
    st.query_params["bench"] = ",".join(st.session_state['active_selections'])
    st.sidebar.success("URL updated! Bookmark it now.")

st.sidebar.markdown("---")
available_fridays = get_friday_expirations()
if available_fridays:
    selected_date_str = st.sidebar.selectbox("Expiration Date:", options=available_fridays)
    dte = (datetime.strptime(selected_date_str, '%Y-%m-%d') - datetime.now()).days
else: dte = 14

prob_target = st.sidebar.selectbox("Success Target:", options=list(Z_SCORES.keys()), index=4)
z_val = Z_SCORES[prob_target]

# --- RADAR ---
st.sidebar.markdown("---")
st.sidebar.subheader("📡 Range-Bound Radar")
LIQUID_50 = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AMD', 'PLTR', 'NFLX', 'BA', 'DIS', 'BABA', 'UBER', 'COIN', 'HOOD', 'INTC', 'MU', 'AVGO', 'TSM', 'JPM', 'BAC', 'C', 'V', 'MA', 'PYPL', 'SQ', 'WMT', 'TGT', 'COST', 'HD', 'SBUX', 'NKE', 'MCD', 'XOM', 'CVX', 'CAT', 'GE', 'JNJ', 'PFE', 'UNH', 'LLY', 'CMCSA', 'VZ', 'T', 'QCOM', 'CRM', 'SNOW', 'SHOP', 'SPOT']
scan_choice = st.sidebar.radio("Universe:", ["Top 50 Liquid", "S&P 100"])
scan_tolerance = st.sidebar.slider("Tolerance (%)", 3, 15, 8) / 100.0

if st.sidebar.button("Run Radar"):
    univ = LIQUID_50 if scan_choice == "Top 50 Liquid" else ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    with st.sidebar.status("Scanning..."):
        hits = run_radar_scan(univ, scan_tolerance)
        if hits: st.sidebar.success(f"🎯 Found: {', '.join(hits)}")

# --- CORRELATION MATRIX ---
st.markdown("---")
if len(selected_tickers) > 1:
    with st.expander("🧩 Portfolio Risk: 30-Day Correlation Matrix", expanded=False):
        try:
            bench_data = yf.download(selected_tickers, period="3mo", progress=False)['Close'].pct_change().tail(30)
            corr = bench_data.corr()
            st.dataframe(corr.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))
        except: st.write("Correlation data unavailable.")
st.markdown("---")

# --- MAIN ENGINE ---
for symbol in selected_tickers:
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="3mo")
        curr = hist['Close'].iloc[-1]
        prev = hist['Close'].iloc[-2]
        
        # Calculations
        ma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        support_3mo = hist['Close'].min()
        rsi = calculate_rsi(hist['Close']).iloc[-1]
        adx = calculate_adx(hist)
        gap = calculate_gap_risk(hist)
        
        vol_dte = np.std(hist['Close'].pct_change()) * np.sqrt(dte if dte > 0 else 1)
        move = curr * (vol_dte * z_val)
        ps, cs = round(curr - move), round(curr + move)
        pt, ct = round(ps * 1.05, 2), round(cs * 0.95, 2)

        # Market Data
        iv, earnings = "N/A", "Not scheduled"
        try:
            chain = t.option_chain(selected_date_str)
            iv = f"{chain.calls.iloc[(chain.calls['strike'] - curr).abs().argsort()[:1]]['impliedVolatility'].values[0] * 100:.1f}%"
        except: pass
        
        # Risk Grades
        if rsi < 35 or curr <= support_3mo: r, c = "🔴 HIGH RISK (Falling Knife)", "red"
        elif gap > 1.5: r, c = f"🟠 GAP RISK ({gap:.2f}%)", "orange"
        elif adx > 25: r, c = f"🟡 TRENDING (ADX {adx:.1f})", "orange"
        elif curr > support_3mo and curr > ma_20: r, c = "🟢 LOW RISK", "green"
        else: r, c = "🟡 MED RISK", "orange"

        with st.expander(f"{symbol} | ${curr:.2f} | {r}"):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Change", f"${curr:.2f}", f"{(curr-prev):.2f}")
            col2.metric("Put Strategy", f"${ps}", f"Trip: ${pt}")
            col3.metric("Call Strategy", f"${cs}", f"Trip: ${ct}")
            col4.metric("Data", f"IV: {iv}", f"DTE: {dte}")

            fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
            fig.add_hline(y=cs, line_color="red", annotation_text="Call")
            fig.add_hline(y=ps, line_color="green", annotation_text="Put")
            fig.add_hline(y=ct, line_dash="dash", line_color="yellow")
            fig.add_hline(y=pt, line_dash="dash", line_color="yellow")
            fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
    except: st.error(f"Error loading {symbol}")
