import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- CONFIG & THEME ---
st.set_page_config(page_title="Money Machine Pro V3.2.1", layout="wide", initial_sidebar_state="expanded")
st.title("⚙️ Money Machine Pro V3.2.1")

# --- HOW TO USE / DOCUMENTATION ---
with st.expander("📖 How to Use This Engine & Risk Legend", expanded=False):
    st.markdown("""
    ### 🧠 The Workflow
    1. **Build Your Bench:** Add tickers in the sidebar. Use the **Range-Bound Radar** to scan for sideways movers.
    2. **Check Correlation:** Avoid trading high-correlation pairs (0.85+) simultaneously.
    3. **Review Setups:** Check risk grades, **IV**, and **Earnings Dates** before entry.
    
    ### 🚦 Risk Legend
    * 🟢 **LOW RISK:** Ideal neutral chop. Above support and 20-day MA.
    * 🟡 **MED RISK:** Stalling or struggling under MA.
    * 🟡 **TRENDING (ADX > 25):** Moving too fast for Iron Condors.
    * 🟠 **GAP RISK (> 1.5%):** Dangerous overnight jumps.
    * 🔴 **HIGH RISK:** Support break or RSI "Falling Knife" crash.
    * ⛔ **EARNINGS VETO:** Trade expires after the next earnings report.
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
                if (h.max() - h.min()) / h.iloc[-1] < threshold: found.append(sym)
            except: continue
    except: pass
    return found

# --- SIDEBAR ---
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
scan_choice = st.sidebar.radio("Universe:", ["Top 50 Liquid", "S&P 100"])
tol = st.sidebar.slider("Tolerance (%)", 3, 15, 8) / 100.0

if st.sidebar.button("Run Radar"):
    univ = LIQUID_50 if scan_choice == "Top 50 Liquid" else ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    with st.sidebar.status("Scanning..."):
        hits = run_radar_scan(univ, tol)
        if hits: st.sidebar.success(f"🎯 Found: {', '.join(hits)}")

# --- CORRELATION MATRIX ---
st.markdown("---")
if len(selected_tickers) > 1:
    with st.expander("🧩 Portfolio Risk: 30-Day Correlation Matrix", expanded=False):
        try:
            c_data = yf.download(selected_tickers, period="3mo", progress=False)['Close'].pct_change().tail(30).corr()
            st.dataframe(c_data.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))
        except: st.write("Data error.")
st.markdown("---")

# --- MAIN ENGINE ---
for sym in selected_tickers:
    try:
        t = yf.Ticker(sym)
        h = t.history(period="3mo")
        curr = h['Close'].iloc[-1]; prev = h['Close'].iloc[-2]
        
        ma20 = h['Close'].rolling(20).mean().iloc[-1]; sup = h['Close'].min()
        rsi = calculate_rsi(h['Close']).iloc[-1]; adx = calculate_adx(h); gap = calculate_gap_risk(h)
        vol = np.std(h['Close'].pct_change()) * np.sqrt(dte if dte > 0 else 1)
        move = curr * (vol * z_val)
        ps, cs = round(curr - move), round(curr + move)
        pt, ct = round(ps * 1.05, 2), round(cs * 0.95, 2)

        iv_val, earnings_date, veto = "N/A", "Not scheduled", False
        try:
            if exp_str:
                chain = t.option_chain(exp_str)
                calls = chain.calls
                iv_raw = calls.iloc[(calls['strike'] - curr).abs().argsort()[:1]]['impliedVolatility'].values[0]
                iv_val = f"{iv_raw * 100:.1f}%"
        except: pass
        
        try:
            cal = t.calendar
            if cal is not None and not cal.empty:
                e_date = cal.iloc[0,0] if isinstance(cal, pd.DataFrame) else cal.get('Earnings Date', [None])[0]
                if e_date:
                    earnings_date = e_date.strftime('%Y-%m-%d')
                    if datetime.now() < e_date < (datetime.now() + timedelta(days=dte)): veto = True
        except: pass

        if veto: r, c = "⛔ DO NOT TRADE (EARNINGS VETO)", "red"
        elif rsi < 35 or curr <= sup: r, c = "🔴 HIGH RISK (Falling Knife)", "red"
        elif gap > 1.5: r, c = f"🟠 GAP RISK ({gap:.2f}%)", "orange"
        elif adx > 25: r, c = f"🟡 TRENDING (ADX {adx:.1f} - Use Directional Spreads)", "orange"
        elif curr > sup and curr > ma20: r, c = "🟢 LOW RISK (Neutral Chop)", "green"
        else: r, c = "🟡 MED RISK (Stalling)", "orange"

        with st.expander(f"{sym} | Price: ${curr:.2f} | Risk: {r}"):
            l, m, n, o = st.columns(4)
            l.metric("Today's Change", f"${curr:.2f}", f"{(curr-prev):.2f} ({( (curr-prev)/prev * 100):.2f}%)")
            m.metric("Put Strategy", f"Strike: ${ps}", f"Trip Wire: ${pt}", delta_color="off")
            n.metric("Call Strategy", f"Strike: ${cs}", f"Trip Wire: ${ct}", delta_color="off")
            o.metric("Market Data", f"IV: {iv_val}", f"Earnings: {earnings_date}", delta_color="off")

            fig = go.Figure(data=[go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'], name="Price")])
            fig.add_hline(y=cs, line_color="red", annotation_text="Call Strike")
            fig.add_hline(y=ps, line)
