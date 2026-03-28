import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- CONFIG & THEME ---
st.set_page_config(page_title="Condor-Tool", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Condor-Tool | Volatility & Probability Screener")

# --- PROBABILITY Z-SCORES ---
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

def calculate_adx(hist, period=14):
    try:
        high, low, close = hist['High'], hist['Low'], hist['Close']
        plus_dm = np.where((high.diff() > low.diff()) & (high.diff() > 0), high.diff(), 0.0)
        minus_dm = np.where((low.diff() > high.diff()) & (low.diff() > 0), low.diff(), 0.0)
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        return dx.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    except: return 20 

def calculate_gap_risk(hist):
    try:
        return abs((hist['Open'] - hist['Close'].shift(1)) / hist['Close'].shift(1)).tail(30).mean() * 100
    except: return 0

def calculate_volume_nodes(hist, current_price, bins=30):
    try:
        price_bins = np.linspace(hist['Low'].min(), hist['High'].max(), bins)
        inds = np.digitize(hist['Close'], price_bins)
        vol_profile = np.zeros(bins)
        for i in range(len(hist)):
            if 0 <= inds[i]-1 < bins: vol_profile[inds[i]-1] += hist['Volume'].iloc[i]
        poc = price_bins[np.argmax(vol_profile)]
        peaks = [price_bins[i] for i in range(1, bins-1) if vol_profile[i] > vol_profile[i-1] and vol_profile[i] > vol_profile[i+1] and vol_profile[i] > np.mean(vol_profile) * 0.5]
        upper = sorted([p for p in peaks if p > current_price])
        lower = sorted([p for p in peaks if p < current_price])
        r1 = f"${upper[0]:.2f}" if len(upper) > 0 else "Sky (None)"
        r2 = f"${upper[1]:.2f}" if len(upper) > 1 else "N/A"
        s1 = f"${lower[-1]:.2f}" if len(lower) > 0 else "Freefall (None)"
        s2 = f"${lower[-2]:.2f}" if len(lower) > 1 else "N/A"
        return f"${poc:.2f}", s1, s2, r1, r2
    except: return "N/A", "N/A", "N/A", "N/A", "N/A"

@st.cache_data(ttl=3600)  
def get_friday_expirations():
    try:
        return [d for d in yf.Ticker("SPY").options if datetime.strptime(d, '%Y-%m-%d').weekday() == 4][:10]
    except: return [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(14, 60) if (datetime.now() + timedelta(days=i)).weekday() == 4]

@st.cache_data(ttl=3600) 
def run_radar_scan(ticker_list, threshold):
    found = []
    try:
        bulk_data = yf.download(ticker_list, period="1mo", group_by='ticker', progress=False)
        for sym in ticker_list:
            try:
                hist = bulk_data[sym]['Close'].dropna() if len(ticker_list) > 1 else bulk_data['Close'].dropna()
                if len(hist) > 10:
                    h_max = hist.max().iloc[0] if isinstance(hist.max(), pd.Series) else hist.max()
                    h_min = hist.min().iloc[0] if isinstance(hist.min(), pd.Series) else hist.min()
                    curr = hist.iloc[-1].iloc[0] if isinstance(hist.iloc[-1], pd.Series) else hist.iloc[-1]
                    if (h_max - h_min) / curr < threshold: found.append(sym)
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
        curr = st.session_state['active_selections'].copy()
        if t not in curr: curr.append(t); st.session_state['active_selections'] = curr
    st.session_state['ticker_input'] = ""

st.sidebar.text_input("➕ Add Custom Ticker:", key="ticker_input", on_change=add_custom_ticker)
selected_tickers = st.sidebar.multiselect("Active Bench:", options=st.session_state['custom_bench'], key="active_selections")

if st.sidebar.button("🔗 Generate Custom Link"):
    st.query_params["bench"] = ",".join(st.session_state['active_selections'])
    st.sidebar.success("URL updated! Bookmark to save.")

st.sidebar.markdown("---")
fridays = get_friday_expirations()
if fridays:
    exp_str = st.sidebar.selectbox("Expiration Date (Fridays Only):", options=fridays)
    selected_date = datetime.strptime(exp_str, '%Y-%m-%d')
    dte = (selected_date - datetime.now()).days
else: st.sidebar.error("Error loading dates."); dte = 14; exp_str = None

z_score = Z_SCORES[st.sidebar.selectbox("Success Target:", options=list(Z_SCORES.keys()), index=4)]

# --- RADAR ---
st.sidebar.markdown("---")
st.sidebar.subheader("📡 Range-Bound Radar")
st.sidebar.caption("Scan restricted to Top 50 highest options liquidity stocks.")
LIQUID_50 = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AMD', 'PLTR', 'NFLX', 'BA', 'DIS', 'BABA', 'UBER', 'COIN', 'HOOD', 'INTC', 'MU', 'AVGO', 'TSM', 'JPM', 'BAC', 'C', 'V', 'MA', 'PYPL', 'SQ', 'WMT', 'TGT', 'COST', 'HD', 'SBUX', 'NKE', 'MCD', 'XOM', 'CVX', 'CAT', 'GE', 'JNJ', 'PFE', 'UNH', 'LLY', 'CMCSA', 'VZ', 'T', 'QCOM', 'CRM', 'SNOW', 'SHOP', 'SPOT']
scan_tol = st.sidebar.slider("Tolerance (%)", 3, 15, 8) / 100.0

if st.sidebar.button("Run Radar Scan Now"):
    with st.sidebar.status(f"Scanning at {int(scan_tol*100)}% tolerance..."):
        targs = run_radar_scan(LIQUID_50, scan_tol)
        if targs: st.sidebar.success(f"🎯 Targets: {', '.join(targs)}")
        else: st.sidebar.warning("No setups found.")

# --- MATRIX ---
st.markdown("---")
if len(selected_tickers) > 1:
    with st.expander("🧩 Portfolio Risk: 30-Day Correlation Matrix", expanded=False):
        try:
            ret = yf.download(selected_tickers, period="3mo", progress=False)['Close'].pct_change().tail(30)
            corr = ret.corr()
            high = [f"{corr.columns[i]} & {corr.columns[j]} ({corr.iloc[i,j]:.2f})" for i in range(len(corr.columns)) for j in range(i+1, len(corr.columns)) if corr.iloc[i,j] > 0.8]
            if high: st.warning(f"⚠️ HIGH CONCENTRATION RISK: {', '.join(high)}")
            else: st.success("🟢 No severe correlations found in active bench.")
            st.dataframe(corr.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))
        except: st.write("Not enough data to calculate matrix.")
st.markdown("---")

# --- MAIN ENGINE ---
for sym in selected_tickers:
    try:
        t = yf.Ticker(sym)
        hist = t.history(period="3mo")
        if len(hist) < 20: continue
            
        c = hist['Close'].iloc[-1]; p = hist['Close'].iloc[-2]
        chg = c - p; pct = (chg / p) * 100
        
        ma20, sup = hist['Close'].rolling(20).mean().iloc[-1], hist['Close'].min()
        rsi, adx, gap = calculate_rsi(hist['Close']).iloc[-1], calculate_adx(hist), calculate_gap_risk(hist)
        poc, s1, s2, r1, r2 = calculate_volume_nodes(hist, c)
        
        vol = np.std(hist['Close'].pct_change().dropna()) * np.sqrt(dte if dte > 0 else 1)
        move = c * (vol * z_score)
        ps, cs = round(c - move), round(c + move)
        pt, ct = round(ps * 1.05, 2), round(cs * 0.95, 2)
        
        iv = "N/A"
        try:
            if exp_str:
                calls = t.option_chain(exp_str).calls
                iv = f"{calls.iloc[(calls['strike'] - c).abs().argsort()[:1]]['impliedVolatility'].values[0] * 100:.1f}%"
        except: pass

        ed_str, veto = "Not scheduled", False
        try:
            cal = t.calendar
            e_date = cal['Earnings Date'][0] if isinstance(cal, dict) and 'Earnings Date' in cal and cal['Earnings Date'] else (cal.loc['Earnings Date'][0] if isinstance(cal, pd.DataFrame) and not cal.empty and 'Earnings Date' in cal.index else None)
            if e_date:
                e_date = pd.to_datetime(e_date)
                ed_str = e_date.strftime('%Y-%m-%d')
                if datetime.now() < e_date < selected_date: veto = True
        except: pass
            
        if veto: risk, clr = "⛔ DO NOT TRADE (EARNINGS VETO)", "red"
        elif rsi < 35 or c <= sup: risk, clr = "🔴 HIGH RISK (Structural Break / Falling Knife)", "red"
        elif gap > 1.5: risk, clr = f"🟠 GAP RISK ({gap:.2f}%)", "orange"
        elif adx > 25: risk, clr = f"🟡 TRENDING (ADX {adx:.1f})", "orange"
        elif c > sup and c > ma20: risk, clr = "🟢 LOW RISK (Neutral Chop)", "green"
        else: risk, clr = "🟡 MED RISK (Stalling)", "orange"

        with st.expander(f"{sym}  |  Price: ${c:.2f}  |  Risk: {risk}", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Today's Change", f"${c:.2
