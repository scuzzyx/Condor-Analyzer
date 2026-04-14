# --- START OF PART 1 ---
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy.stats import norm
import urllib.request
import urllib.parse
import json
import time
import requests

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# --- CONFIG & THEME ---
st.set_page_config(page_title="Aegis Option Scanner", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h2 style='font-size: 2.2rem; margin-bottom: 0rem;'>🛡️ Aegis Option Scanner | Delta-Based Underwriting</h2>", unsafe_allow_html=True)

# --- CACHE ENGINES ---
@st.cache_data(ttl=900, show_spinner=False) 
def get_cached_history(symbol, period="1y"):
    try: return yf.Ticker(symbol).history(period=period)
    except: return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def get_cached_options(symbol, target_date):
    """Bypasses IP blocks by routing Yahoo API requests through free public CORS scraper networks."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    base_url = f"https://query2.finance.yahoo.com/v7/finance/options/{symbol}"
    
    # 3 Free Public Relay APIs to mask the Streamlit Cloud IP address
    proxy_services = [
        lambda url: f"https://api.allorigins.win/raw?url={urllib.parse.quote(url)}",
        lambda url: f"https://corsproxy.io/?{urllib.parse.quote(url)}",
        lambda url: f"https://api.codetabs.com/v1/proxy/?quest={url}"
    ]

    for proxy_builder in proxy_services:
        try:
            res = requests.get(proxy_builder(base_url), headers=headers, timeout=6)
            if res.status_code == 200:
                data = res.json().get('optionChain', {}).get('result', [])
                if not data: continue

                valid_timestamps = data[0].get('expirationDates', [])
                if not valid_timestamps: continue

                target_ts = int(datetime.strptime(target_date, '%Y-%m-%d').timestamp())
                closest_ts = min(valid_timestamps, key=lambda x: abs(x - target_ts))
                snap_date = datetime.fromtimestamp(closest_ts).strftime('%Y-%m-%d')

                chain_url = f"{base_url}?date={closest_ts}"
                chain_res = requests.get(proxy_builder(chain_url), headers=headers, timeout=6)

                if chain_res.status_code == 200:
                    opts = chain_res.json()['optionChain']['result'][0]['options'][0]
                    calls = pd.DataFrame(opts.get('calls', []))
                    puts = pd.DataFrame(opts.get('puts', []))

                    if not calls.empty and 'impliedVolatility' not in calls.columns: calls['impliedVolatility'] = 0.3
                    if not puts.empty and 'impliedVolatility' not in puts.columns: puts['impliedVolatility'] = 0.3

                    return calls, puts, snap_date
        except: continue

    # Native yfinance Fallback
    try:
        t = yf.Ticker(symbol)
        valid_dates = t.options
        if valid_dates:
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            valid_dts = [datetime.strptime(d, '%Y-%m-%d') for d in valid_dates]
            snap_date = min(valid_dts, key=lambda d: abs(d - target_dt)).strftime('%Y-%m-%d')
            chain = t.option_chain(snap_date)
            return chain.calls, chain.puts, snap_date
    except: pass

    return None, None, None

@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_info(symbol):
    try: return yf.Ticker(symbol).info
    except: return {}

@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_calendar(symbol):
    try: return yf.Ticker(symbol).calendar
    except: return None

# --- BLACK-SCHOLES DELTA ENGINE ---
def calculate_delta(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0: return 0.5
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

def find_delta_strikes(chain_df, S, dte, target_delta, option_type):
    try:
        T = max(dte, 1) / 365.0 
        r = 0.04
        df = chain_df[chain_df['strike'] >= S].copy() if option_type == 'call' else chain_df[chain_df['strike'] <= S].copy()
        if df.empty: return None
        df['impliedVolatility'] = df['impliedVolatility'].replace(0, np.nan).fillna(0.3) 
        df['delta'] = df.apply(lambda x: calculate_delta(S, x['strike'], T, r, x['impliedVolatility'], option_type), axis=1)
        return df.loc[(df['delta'].abs() - target_delta).abs().idxmin(), 'strike']
    except: return None
# --- END OF PART 1 ---
# --- START OF PART 2 ---
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    return 100 - (100 / (1 + (gain / loss)))

def calculate_adx(hist, period=14):
    try:
        high, low, close = hist['High'], hist['Low'], hist['Close']
        plus_dm, minus_dm = high.diff(), low.diff()
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
        tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
        return ((abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100).ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    except: return 20 

def calculate_ivr(hist_1y, current_iv):
    try:
        if pd.isna(current_iv): return "N/A"
        curr_iv_val = float(current_iv) if not isinstance(current_iv, str) else float(current_iv.replace('%', ''))/100
        hv = hist_1y['Close'].pct_change().dropna().rolling(20).std() * np.sqrt(252)
        h_min, h_max = float(hv.min()), float(hv.max())
        if h_max == h_min: return 50.0
        return max(0, min(100, ((curr_iv_val - h_min) / (h_max - h_min)) * 100))
    except: return "N/A"

def calculate_gap_risk(hist):
    try: return (abs((hist['Open'] - hist['Close'].shift(1)) / hist['Close'].shift(1))).tail(30).mean() * 100
    except: return 0

def calculate_volume_nodes(hist, current_price, bins=30):
    try:
        min_p, max_p = float(hist['Low'].min()), float(hist['High'].max())
        if min_p == max_p or pd.isna(min_p): return f"${current_price:.2f}", "N/A", "N/A", "N/A", "N/A"
        price_bins = np.linspace(min_p, max_p, bins)
        inds = np.digitize(hist['Close'].fillna(current_price).values, price_bins)
        vol_profile = np.zeros(bins)
        volumes = hist['Volume'].fillna(0).values
        for i in range(len(hist)):
            if 0 <= inds[i]-1 < bins: vol_profile[inds[i]-1] += volumes[i]
        poc = price_bins[np.argmax(vol_profile)]
        peaks = [price_bins[i] for i in range(1, bins-1) if vol_profile[i] > vol_profile[i-1] and vol_profile[i] > vol_profile[i+1] and vol_profile[i] > np.mean(vol_profile)*0.5]
        upper, lower = sorted([p for p in peaks if p > current_price]), sorted([p for p in peaks if p < current_price])
        r1 = f"${upper[0]:.2f}" if upper else "Sky (None)"
        r2 = f"${upper[1]:.2f}" if len(upper) > 1 else "⚠️ No Wall"
        s1 = f"${lower[-1]:.2f}" if lower else "Freefall (None)"
        s2 = f"${lower[-2]:.2f}" if len(lower) > 1 else "⚠️ No Wall"
        return f"${poc:.2f}", s1, s2, r1, r2
    except: return "N/A", "N/A", "N/A", "N/A", "N/A"

@st.cache_data(ttl=86400)  
def get_pure_fridays(weeks=26):
    today = datetime.now()
    days_until_friday = (4 - today.weekday()) % 7
    next_friday = today + timedelta(days=days_until_friday)
    return [(next_friday + timedelta(weeks=i)).strftime('%Y-%m-%d') for i in range(weeks)]

@st.cache_data(ttl=3600)
def run_premium_hunter(ticker_list):
    targets = []
    try:
        bulk_data = yf.download(ticker_list, period="1y", progress=False)['Close']
        for sym in ticker_list:
            try:
                hist = bulk_data[sym].dropna()
                if len(hist) < 50: continue
                returns = hist.pct_change().dropna()
                hv_series = returns.rolling(20).std() * np.sqrt(252)
                curr_hv = hv_series.iloc[-1]
                hv_min, hv_max = hv_series.min(), hv_series.max()
                if hv_max > hv_min:
                    hv_rank = ((curr_hv - hv_min) / (hv_max - hv_min)) * 100
                    if hv_rank > 60:
                        targets.append((sym, hv_rank))
            except: continue
        targets.sort(key=lambda x: x[1], reverse=True)
        return [f"{t[0]} (Rank: {t[1]:.0f})" for t in targets[:6]]
    except: return []

@st.cache_data(ttl=900)
def fetch_macro_data():
    vix_val, vix_pct, fg_val, fg_rating = "N/A", "N/A", "N/A", "N/A"
    try:
        vix_hist = yf.Ticker("^VIX").history(period="5d")
        vix_val = float(vix_hist['Close'].iloc[-1])
        vix_pct = float(((vix_val - vix_hist['Close'].iloc[-2]) / vix_hist['Close'].iloc[-2]) * 100)
    except: pass
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            fg_val = round(data['fear_and_greed']['score'])
            fg_rating = data['fear_and_greed']['rating'].title()
    except: pass
    return vix_val, vix_pct, fg_val, fg_rating

def custom_metric_box(label, value, sub_value, val_color="#FAFAFA", sub_color="#a6a6a6"):
    return f'<div style="line-height: 1.4; margin-bottom: 14px;"><span style="font-size: 0.85rem; color: #a6a6a6; font-family: sans-serif;">{label}</span><br><span style="font-size: 1.8rem; font-weight: 600; color: {val_color}; font-family: sans-serif;">{value}</span><br><span style="font-size: 0.9rem; font-weight: 500; color: {sub_color}; font-family: sans-serif;">{sub_value}</span></div>'
# --- END OF PART 2 ---
# --- START OF PART 3 ---
st.sidebar.header("🛠️ Dashboard Controls")

if st.sidebar.button("🧹 Clear System Cache", type="primary"):
    st.cache_data.clear()
    st.sidebar.success("Cache Purged! Refreshing Data...")
    time.sleep(1)
    st.rerun()

gemini_api_key = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else st.sidebar.text_input("🔑 Gemini API Key", type="password")

vix_v, vix_p, fg_v, fg_r = fetch_macro_data()
st.sidebar.markdown("### 🌍 Macro Sentiment")
mac1, mac2 = st.sidebar.columns(2)
with mac1: st.markdown(custom_metric_box("VIX Index", f"{vix_v:.2f}" if isinstance(vix_v, float) else "N/A", f"{vix_p:+.2f}%" if isinstance(vix_p, float) else "", sub_color="#ff4b4b" if (isinstance(vix_p, float) and vix_p > 0) else "#09ab3b"), unsafe_allow_html=True)
with mac2: st.markdown(custom_metric_box("Fear & Greed", str(fg_v), str(fg_r), val_color="#ffcc00"), unsafe_allow_html=True)
st.sidebar.markdown("---")

def load_url_bench():
    if "bench" in st.query_params: return st.query_params["bench"].split(",")
    return ["AMZN", "AAPL", "MSFT", "META", "GOOGL", "NVDA", "AMD", "PLTR", "TSLA", "NFLX"]

url_bench = load_url_bench()
if 'custom_bench' not in st.session_state: st.session_state['custom_bench'] = list(set(url_bench + ["SPY", "QQQ"]))
if 'active_selections' not in st.session_state: st.session_state['active_selections'] = url_bench

def add_custom_ticker():
    ticker = st.session_state['ticker_input'].upper().strip()
    if ticker:
        if ticker not in st.session_state['custom_bench']: st.session_state['custom_bench'].append(ticker)
        if ticker not in st.session_state['active_selections']: st.session_state['active_selections'].append(ticker)
    st.session_state['ticker_input'] = ""

st.sidebar.text_input("➕ Add Custom Ticker:", key="ticker_input", on_change=add_custom_ticker)
selected_tickers = st.sidebar.multiselect("Active Bench:", options=st.session_state['custom_bench'], key="active_selections")

if st.sidebar.button("🔗 Generate Custom Link"):
    st.query_params["bench"] = ",".join(st.session_state['active_selections'])
    st.sidebar.success("URL updated!")

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Expiration & Risk")
use_custom_date = st.sidebar.checkbox("Use Custom / LEAPS Date")

if use_custom_date:
    custom_exp_date = st.sidebar.date_input("Select Expiration Date:")
    selected_date_str = custom_exp_date.strftime('%Y-%m-%d')
else:
    selected_date_str = st.sidebar.selectbox("Standard Friday Expirations (6 Mo):", options=get_pure_fridays(weeks=26))

dte = max(1, (datetime.strptime(selected_date_str, '%Y-%m-%d') - datetime.now()).days)

target_delta = st.sidebar.select_slider("Target Strike Delta:", options=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40], value=0.15)

st.sidebar.markdown("---")
st.sidebar.subheader("🔥 Premium Hunter Scanner")
LIQUID_50 = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AMD', 'PLTR', 'NFLX', 'BA', 'DIS', 'BABA', 'UBER', 'COIN', 'HOOD', 'INTC', 'MU', 'AVGO', 'TSM', 'JPM', 'BAC', 'C', 'V', 'MA', 'PYPL', 'SQ', 'WMT', 'TGT', 'COST', 'HD', 'SBUX', 'NKE', 'MCD', 'XOM', 'CVX', 'CAT', 'GE', 'JNJ', 'PFE', 'UNH', 'LLY', 'CMCSA', 'VZ', 'T', 'QCOM', 'CRM', 'SNOW', 'SHOP', 'SPOT']

if st.sidebar.button("Scan for High Premium"):
    with st.spinner("Analyzing Liquid 50..."):
        targets = run_premium_hunter(LIQUID_50)
        if targets: 
            st.sidebar.success("🎯 High Volatility Targets:")
            for t in targets: st.sidebar.write(f"- **{t}**")
        else: 
            st.sidebar.warning("Volatility is dead. No elevated IV environments found.")

st.markdown("---")
with st.expander("📖 Terminal Indicator Glossary (Quick Reference)", expanded=False):
    st.subheader("🚦 Title Risk & Veto Signals")
    st.write("- **IV Rank (IVR):** Relates current IV to the 52-week high/low. >50 is Tastytrade territory.")
    st.write("- **⚠️ [EARNINGS SOON]:** Earnings report occurs before expiration. Trade with caution.")
    st.write("- **⚠️ [EX-DIVIDEND DANGER]:** Ex-Div date occurs before expiration. High risk of early call assignment.")
    st.write("- **🔴 *FALLING KNIFE*:** Price below 8-EMA. Consider Call Spreads only.")
    st.write("- **🟠 *GAP RISK*:** Historical tendency to jump >1.5% overnight.")
    st.write("- **🟡 *TRENDING*:** ADX (>25). Stock is moving fast; pick a directional spread. Avoid Condors.")
    st.write("- **🟢 *FLOOR CONFIRMED*:** 8-EMA Reclaimed. Consider Put Spreads only.")
    st.write("- **🟢 *NEUTRAL CHOP*:** Ideal sideways environment for Iron Condors.")
    
if len(selected_tickers) > 1:
    with st.expander("🧩 Portfolio Risk: 30-Day Correlation Matrix", expanded=False):
        try:
            bench_data = yf.download(selected_tickers, period="3mo", progress=False)['Close']
            st.dataframe(bench_data.pct_change().tail(30).corr().style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))
        except: st.write("Not enough data.")

tab_scanner, tab_deepdive, tab_ai = st.tabs(["🛡️ Option Scanner", "🔬 Technical Deep Dive", "🧠 AI Quant Co-Pilot"])
# --- END OF PART 3 ---
# --- START OF PART 4 ---
with tab_scanner:
    if selected_tickers:
        st.markdown("##### 📡 Reading Memory Cache...")
        progress_bar = st.progress(0)
        
    for idx, symbol in enumerate(selected_tickers):
        try:
            hist_1y = get_cached_history(symbol)
            if len(hist_1y) < 20: continue
            
            hist = hist_1y.tail(63) 
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            change_dlr, change_pct = current_price - prev_close, ((current_price - prev_close) / prev_close) * 100
            change_color = "#ff4b4b" if change_dlr < 0 else "#09ab3b"
            
            ma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            ema_8 = hist['Close'].ewm(span=8, adjust=False).mean().iloc[-1]
            rsi_5 = calculate_rsi(hist['Close'], periods=5).iloc[-1]
            rsi_5_prev = calculate_rsi(hist['Close'], periods=5).iloc[-2]
            rsi_9 = calculate_rsi(hist['Close'], periods=9).iloc[-1]
            rsi_14 = calculate_rsi(hist['Close'], periods=14).iloc[-1]
            adx_14 = calculate_adx(hist)
            gap_risk = calculate_gap_risk(hist)
            poc, sup1, sup2, res1, res2 = calculate_volume_nodes(hist, current_price)
            
            atm_iv_display, ivr, max_pain, pc_ratio = "N/A", "N/A", "N/A", "N/A"
            put_strike, call_strike, put_trip, call_trip = None, None, "N/A", "N/A"
            
            calls, puts, active_date = get_cached_options(symbol, selected_date_str)
            target_date = active_date if active_date else selected_date_str
            
            if calls is not None and puts is not None:
                if not calls.empty:
                    closest_idx = (calls['strike'] - current_price).abs().idxmin()
                    atm_iv_raw = calls.loc[closest_idx, 'impliedVolatility']
                    if pd.notna(atm_iv_raw) and atm_iv_raw > 0:
                        atm_iv_display = f"{atm_iv_raw * 100:.1f}%"
                        ivr = f"{calculate_ivr(hist_1y, atm_iv_raw):.1f}"
                    
                    call_strike = find_delta_strikes(calls, current_price, dte, target_delta, 'call')
                    put_strike = find_delta_strikes(puts, current_price, dte, target_delta, 'put')
                    if call_strike: call_trip = f"${call_strike * 0.95:.2f}"
                    if put_strike: put_trip = f"${put_strike * 1.05:.2f}"
                
                if not calls.empty and not puts.empty:
                    tot_put_oi, tot_call_oi = puts['openInterest'].sum(), calls['openInterest'].sum()
                    if tot_call_oi > 0: pc_ratio = f"{tot_put_oi / tot_call_oi:.2f}"
                    all_strikes = sorted(list(set(calls['strike'].tolist() + puts['strike'].tolist())))
                    mp_val, mp_strike = float('inf'), "N/A"
                    for s in all_strikes:
                        c_loss = calls[calls['strike'] < s].apply(lambda x: (s - x['strike']) * x['openInterest'], axis=1).sum()
                        p_loss = puts[puts['strike'] > s].apply(lambda x: (x['strike'] - s) * x['openInterest'], axis=1).sum()
                        if (c_loss + p_loss) < mp_val: mp_val, mp_strike = c_loss + p_loss, s
                    if mp_strike != "N/A": max_pain = f"${mp_strike:.2f}"

            info = get_cached_info(symbol)
            ex_div_date, ex_div_veto = "None scheduled", False
            ex_ts = info.get('exDividendDate')
            if ex_ts:
                ex_dt = datetime.fromtimestamp(ex_ts)
                ex_div_date = ex_dt.strftime('%Y-%m-%d')
                if datetime.now() < ex_dt < datetime.strptime(target_date, '%Y-%m-%d'): ex_div_veto = True

            calendar = get_cached_calendar(symbol)
            earnings_date, earnings_veto = "Not scheduled", False
            try:
                if calendar is not None:
                    e_date = pd.to_datetime(calendar.get('Earnings Date')[0]) if isinstance(calendar, dict) else pd.to_datetime(calendar.loc['Earnings Date'].iloc[0])
                    if pd.notnull(e_date):
                        earnings_date = e_date.strftime('%Y-%m-%d')
                        if datetime.now() < e_date < datetime.strptime(target_date, '%Y-%m-%d'): earnings_veto = True
            except: pass

            if current_price < ema_8 and rsi_14 < 45: base_risk = "🔴 ***FALLING KNIFE***: Call Spreads Only"
            elif current_price > ema_8 and rsi_5 > rsi_5_prev and rsi_14 < 50: base_risk = "🟢 ***FLOOR CONFIRMED***: Put Spreads Only"
            elif gap_risk > 1.5: base_risk = f"🟠 ***GAP RISK***: High Overnight Vol ({gap_risk:.2f}%)"
            elif adx_14 > 25: base_risk = f"🟡 ***TRENDING***: ADX {adx_14:.1f} (Pick Directional)"
            elif current_price > ma_20: base_risk = "🟢 ***NEUTRAL CHOP***: Condor Territory"
            else: base_risk = "🟡 ***MED RISK***: Price Stalling"

            risk = base_risk + (" [EARNINGS SOON]" if earnings_veto else "") + (" ⚠️[EX-DIVIDEND DANGER]" if ex_div_veto else "")
            ivr_color = "#09ab3b" if (isinstance(ivr, str) and ivr != "N/A" and float(ivr) > 50) else "#a6a6a6"

            with st.expander(f"{symbol} | Price: ${current_price:.2f} | Target Chain: {target_date} | Risk: {risk}", expanded=False):
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1: st.markdown(custom_metric_box("Today's Change", f"${current_price:.2f}", f"{change_dlr:+.2f} ({change_pct:+.2f}%)", sub_color=change_color), unsafe_allow_html=True)
                with c2: st.markdown(custom_metric_box(f"{int(target_delta*100)}Δ Put", f"${put_strike}" if put_strike else "N/A", f"Trip: {put_trip}", sub_color="#ffcc00"), unsafe_allow_html=True)
                with c3: st.markdown(custom_metric_box(f"{int(target_delta*100)}Δ Call", f"${call_strike}" if call_strike else "N/A", f"Trip: {call_trip}", sub_color="#ffcc00"), unsafe_allow_html=True)
                with c4: st.markdown(custom_metric_box("Volatility Rank", f"IVR: {ivr}", f"ATM IV: {atm_iv_display}", val_color=ivr_color), unsafe_allow_html=True)
                with c5: st.markdown(custom_metric_box("Earnings Date", f"{earnings_date}", "Upcoming Catalyst", sub_color="#ffcc00" if earnings_veto else "#a6a6a6"), unsafe_allow_html=True)
                
                st.markdown("---")
                st.caption("🛡️ Risk Underwriting Data")
                u1, u2, u3 = st.columns(3)
                with u1: st.markdown(custom_metric_box("Max Pain", f"{max_pain}", "Gravity point for Expiration", sub_color="#a6a6a6"), unsafe_allow_html=True)
                with u2:
                    pc_color, pc_sub = "#a6a6a6", "Neutral Flow"
                    try:
                        pcr = float(pc_ratio)
                        if pcr > 1.2: pc_color, pc_sub = "#ff4b4b", "Heavy Bearish Flow"
                        elif pcr < 0.8: pc_color, pc_sub = "#09ab3b", "Heavy Bullish Flow"
                    except: pass
                    st.markdown(custom_metric_box("P/C OI Ratio", f"{pc_ratio}", pc_sub, sub_color=pc_color), unsafe_allow_html=True)
                with u3:
                    div_color = "#ffcc00" if ex_div_veto else "#a6a6a6"
                    st.markdown(custom_metric_box("Ex-Dividend", f"{ex_div_date}", "Early Assignment Risk" if ex_div_veto else "Upcoming Ex-Div Date", sub_color=div_color), unsafe_allow_html=True)

                st.markdown("---")
                v1, v2, v3, v4 = st.columns(4)
                def get_s(v): return "Oversold" if v <= 30 else "Overbought" if v >= 70 else "Neutral"
                with v1: st.caption("🧲 POC & Trend"); st.write(f"**POC:** {poc}"); st.write(f"**ADX:** {adx_14:.1f}")
                with v2: st.caption("📈 RSI Stack"); st.write(f"5D: {rsi_5:.1f} ({get_s(rsi_5)})"); st.write(f"9D: {rsi_9:.1f} ({get_s(rsi_9)})"); st.write(f"14D: {rsi_14:.1f} ({get_s(rsi_14)})")
                with v3: st.caption("🔴 Support Walls"); st.write(f"**Wall 1:** {sup1}"); st.write(f"**Wall 2:** {sup2}")
                with v4: st.caption("🟢 Resistance Walls"); st.write(f"**Wall 1:** {res1}"); st.write(f"**Wall 2:** {res2}")

                fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price")])
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].ewm(span=8, adjust=False).mean(), line=dict(color='#ff9900', width=1.5, dash='dot'), name="8-EMA"))
                if call_strike: fig.add_hline(y=call_strike, line_width=2, line_color="green", annotation_text=f"{target_delta}Δ Call")
                if put_strike: fig.add_hline(y=put_strike, line_width=2, line_color="red", annotation_text=f"{target_delta}Δ Put")
                fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e: st.error(f"Error loading {symbol}: {str(e)}")
        if selected_tickers: progress_bar.progress((idx + 1) / len(selected_tickers))
            
    if selected_tickers: progress_bar.empty()
# --- END OF PART 4 ---
# --- START OF PART 5 ---
with tab_deepdive:
    st.markdown("### 🔬 Automated Quantitative Analyst")
    st.write("Enter a single ticker below. The system will process the underlying mathematics, liquidity, and tail risks to translate the chart structure into plain English.")
    
    deep_ticker = st.text_input("Enter Ticker for Deep Dive (e.g., TSLA, SPY):", key="dd_ticker").upper().strip()
    
    if deep_ticker:
        try:
            hist_dd = get_cached_history(deep_ticker)
            if len(hist_dd) < 50:
                st.warning("Not enough trading history to generate a robust analysis.")
            else:
                hist_6mo = hist_dd.tail(126)
                dd_price = hist_6mo['Close'].iloc[-1]
                sma_20_dd = hist_6mo['Close'].rolling(window=20).mean().iloc[-1]
                sma_50_dd = hist_6mo['Close'].rolling(window=50).mean().iloc[-1]
                rsi_14_dd = calculate_rsi(hist_6mo['Close'], periods=14).iloc[-1]
                adx_14_dd = calculate_adx(hist_6mo)
                poc_dd, sup1_dd, sup2_dd, res1_dd, res2_dd = calculate_volume_nodes(hist_6mo, dd_price)

                info_dd = get_cached_info(deep_ticker)
                short_pct = info_dd.get('shortPercentOfFloat', 0)
                inst_pct = info_dd.get('heldPercentInstitutions', 0)
                target_price = info_dd.get('targetMeanPrice')
                
                tr = pd.concat([hist_6mo['High']-hist_6mo['Low'], abs(hist_6mo['High']-hist_6mo['Close'].shift(1)), abs(hist_6mo['Low']-hist_6mo['Close'].shift(1))], axis=1).max(axis=1)
                atr_14 = tr.rolling(14).mean().iloc[-1]
                
                returns_1y = hist_dd['Close'].pct_change().dropna()
                hv_20 = returns_1y.rolling(window=20).std() * np.sqrt(252)
                current_hv = hv_20.iloc[-1]
                
                skew_status, skew_text = "⚖️ **Balanced Skew:**", "Put and Call implied volatilities are relatively balanced."
                liq_status, liq_text = "⚖️ **Adequate Liquidity:**", "Volume is average. Be cautious of bid-ask spreads."
                atm_iv_dd = current_hv 
                oi_fig = None
                
                try:
                    target_dd = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
                    calls, puts, _ = get_cached_options(deep_ticker, target_dd)
                    
                    if calls is not None and puts is not None:
                        total_vol = calls['volume'].fillna(0).sum() + puts['volume'].fillna(0).sum()
                        avg_vol = total_vol / (len(calls) + len(puts))
                        if avg_vol > 300: liq_status, liq_text = "🌊 **A+ Liquidity:**", "Highly liquid options chain. Minimal slippage expected."
                        elif avg_vol < 50: liq_status, liq_text = "🧊 **Poor Liquidity:**", "Low volume. Expect massive slippage."
                        
                        otm_call = calls[calls['strike'] >= dd_price * 1.1]
                        otm_put = puts[puts['strike'] <= dd_price * 0.9]
                        
                        if not calls.empty: atm_iv_dd = calls.iloc[(calls['strike'] - dd_price).abs().argsort()[:1]]['impliedVolatility'].values[0]
                            
                        if not otm_call.empty and not otm_put.empty:
                            skew_diff = otm_put.iloc[-1]['impliedVolatility'] - otm_call.iloc[0]['impliedVolatility']
                            if skew_diff > 0.12: skew_status, skew_text = "🚨 **Severe Downside Skew:**", "Market pricing OTM Puts higher than Calls. Crash protection is expensive."
                            elif skew_diff < -0.12: skew_status, skew_text = "🚀 **Upside Call Skew:**", "OTM Calls pricing higher than Puts. Market anticipating upside."
                                
                        calls_oi = calls[(calls['strike'] >= dd_price * 0.85) & (calls['strike'] <= dd_price * 1.15)]
                        puts_oi = puts[(puts['strike'] >= dd_price * 0.85) & (puts['strike'] <= dd_price * 1.15)]
                        
                        oi_fig = go.Figure()
                        oi_fig.add_trace(go.Bar(x=calls_oi['strike'], y=calls_oi['openInterest'], name='Call OI', marker_color='#09ab3b', opacity=0.7))
                        oi_fig.add_trace(go.Bar(x=puts_oi['strike'], y=puts_oi['openInterest'], name='Put OI', marker_color='#ff4b4b', opacity=0.7))
                        oi_fig.update_layout(title="Open Interest Profile", template="plotly_dark", height=300, margin=dict(l=0, r=0, t=30, b=0), barmode='group')
                        oi_fig.add_vline(x=dd_price, line_width=2, line_dash="dash", line_color="white", annotation_text="Price")
                except: pass

                ivr_val = calculate_ivr(hist_dd, atm_iv_dd)
                ivr_str = f"{ivr_val:.1f}" if isinstance(ivr_val, (int, float)) else "N/A"

                if dd_price > sma_20_dd and dd_price > sma_50_dd: trend_status, trend_text = "🟢 **Bullish Uptrend:**", f"Trading at \${dd_price:.2f}, above 20MA and 50MA."
                elif dd_price < sma_20_dd and dd_price < sma_50_dd: trend_status, trend_text = "🔴 **Bearish Downtrend:**", f"Trading at \${dd_price:.2f}, below 20MA and 50MA."
                else: trend_status, trend_text = "🟡 **Mixed / Consolidation:**", f"Caught in battleground at \${dd_price:.2f}."

                if rsi_14_dd > 70: mom_status, mom_text = "🔥 **Overbought:**", f"RSI running hot ({rsi_14_dd:.1f})."
                elif rsi_14_dd < 30: mom_status, mom_text = "🧊 **Oversold:**", f"RSI indicates punishment ({rsi_14_dd:.1f})."
                else: mom_status, mom_text = "⚖️ **Neutral Momentum:**", f"RSI balanced ({rsi_14_dd:.1f})."
                if adx_14_dd > 25: mom_text += f" ADX is high ({adx_14_dd:.1f}), confirming trend."

                struct_text = f"POC located at **{poc_dd}**. "
                if sup1_dd != "Freefall (None)": struct_text += f"Support floor around **{sup1_dd}**. "
                if res1_dd != "Sky (None)": struct_text += f"Resistance ceiling around **{res1_dd}**."
                struct_text = struct_text.replace("$", r"\$")

                if isinstance(ivr_val, (int, float)) and ivr_val > 50: vol_status, vol_text = "🔥 **TASTYTRADE SELL ZONE:**", f"IVR is **{ivr_str}**. Premiums inflated. Sell Iron Condors."
                elif isinstance(ivr_val, (int, float)) and ivr_val < 20: vol_status, vol_text = "🧊 **LOW VOLATILITY:**", f"IVR is dead (**{ivr_str}**). Avoid selling premium."
                else: vol_status, vol_text = "⚖️ **Neutral Volatility:**", f"IVR is **{ivr_str}**. Premiums at fair value."

                sqz_status, sqz_text = ("🚨 **High Squeeze Risk:**", f"**{short_pct*100:.1f}%** float sold short.") if short_pct > 0.10 else ("✅ **Low Squeeze Risk:**", f"Minimal short interest ({short_pct*100:.1f}%).")
                
                implied_14d_move = dd_price * (atm_iv_dd * np.sqrt(14 / 365))
                if (atr_14 * np.sqrt(14)) > (implied_14d_move * 1.15): fat_status, fat_text = "⚠️ **Fat Tail Risk:**", "Historical swings outpace options pricing. Widen strikes."
                else: fat_status, fat_text = "✅ **Expected Distribution:**", "Swings within options bounds."

                st.markdown("---")
                st.subheader(f"Underwriting Translation for {deep_ticker}")
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("#### 📈 1. Technical Framework")
                    st.markdown(f"{trend_status} {trend_text}"); st.markdown(f"{mom_status} {mom_text}"); st.markdown(f"🏛️ **Price Magnets:** {struct_text}")
                    st.markdown("#### ⚖️ 2. Premium Context")
                    st.markdown(f"{vol_status} {vol_text}"); st.markdown(f"{liq_status} {liq_text}"); st.markdown(f"{skew_status} {skew_text}")
                    st.markdown("#### 🛡️ 3. Tail Risk")
                    st.markdown(f"{sqz_status} {sqz_text}"); st.markdown(f"{fat_status} {fat_text}")
                with col2:
                    fig_dd = go.Figure(data=[go.Candlestick(x=hist_6mo.index, open=hist_6mo['Open'], high=hist_6mo['High'], low=hist_6mo['Low'], close=hist_6mo['Close'], name="Price")])
                    fig_dd.update_layout(template="plotly_dark", height=300, margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig_dd, use_container_width=True)
                    if oi_fig: st.plotly_chart(oi_fig, use_container_width=True)
        except Exception as e: st.error(f"Error: {str(e)}")
# --- END OF PART 5 ---
# --- START OF PART 6 ---
with tab_ai:
    st.markdown("### 🧠 AI Quant Co-Pilot")
    st.write("Compare tickers, ask for a trade thesis, or summarize data using Google Gemini.")
    
    if not GENAI_AVAILABLE:
        st.error("⚠️ `google-generativeai` library not found. Please add it to your requirements.")
    else:
        ai_tickers = st.multiselect(
            "1. Select Tickers to include in AI Context:", 
            options=st.session_state['custom_bench'],
            default=[selected_tickers[0]] if selected_tickers else []
        )
        
        user_prompt = st.text_area(
            "2. Enter your question for the AI:", 
            placeholder="e.g., Compare the IV Rank, Point of Control, and Short Interest of ASTS vs RKLB. Which is better for an Iron Condor?"
        )
        
        if st.button("Ask Gemini 🤖"):
            if not gemini_api_key:
                st.warning("⚠️ Please enter your Gemini API Key in the sidebar controls (or add it to Streamlit Secrets).")
            elif not ai_tickers:
                st.warning("⚠️ Please select at least one ticker from the dropdown.")
            elif not user_prompt:
                st.warning("⚠️ Please enter a prompt.")
            else:
                with st.spinner("Fetching live quantitative data and consulting Gemini..."):
                    try:
                        genai.configure(api_key=gemini_api_key)
                        
                        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                        
                        if 'models/gemini-1.5-flash' in available_models: target_model = 'gemini-1.5-flash'
                        elif 'models/gemini-1.5-flash-latest' in available_models: target_model = 'gemini-1.5-flash-latest'
                        elif 'models/gemini-1.0-pro' in available_models: target_model = 'gemini-1.0-pro'
                        elif 'models/gemini-pro' in available_models: target_model = 'gemini-pro'
                        elif len(available_models) > 0: target_model = available_models[0].replace('models/', '')
                        else:
                            st.error("No valid text models found for this API key.")
                            st.stop()
                            
                        model = genai.GenerativeModel(target_model)
                        
                        context_str = "CURRENT QUANTITATIVE MARKET DATA:\n"
                        for sym in ai_tickers:
                            try:
                                hist_ai = get_cached_history(sym)
                                if len(hist_ai) < 50: continue
                                price = hist_ai['Close'].iloc[-1]
                                
                                hist_6mo = hist_ai.tail(126)
                                rsi = calculate_rsi(hist_6mo['Close']).iloc[-1]
                                adx = calculate_adx(hist_6mo)
                                poc, s1, s2, r1, r2 = calculate_volume_nodes(hist_6mo, price)
                                
                                atm_iv = 0.3
                                try:
                                    calls, puts, _ = get_cached_options(sym, (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'))
                                    if calls is not None and not calls.empty:
                                        atm_iv = calls.iloc[(calls['strike'] - price).abs().argsort()[:1]]['impliedVolatility'].values[0]
                                except: pass
                                
                                ivr = calculate_ivr(hist_ai, atm_iv)
                                ivr_str = f"{ivr:.1f}" if isinstance(ivr, (int, float)) else "N/A"
                                
                                info_ai = get_cached_info(sym)
                                short_pct = info_ai.get('shortPercentOfFloat', 0) * 100 if info_ai else 0
                                
                                context_str += f"\n--- {sym} ---\n"
                                context_str += f"Price: ${price:.2f}\n"
                                context_str += f"IV Rank (IVR): {ivr_str}\n"
                                context_str += f"RSI (14): {rsi:.1f}\n"
                                context_str += f"ADX Trend Strength: {adx:.1f}\n"
                                context_str += f"Point of Control (POC): {poc}\n"
                                context_str += f"Support Floors: {s1}, {s2}\n"
                                context_str += f"Resistance Ceilings: {r1}, {r2}\n"
                                context_str += f"Short Interest: {short_pct:.1f}%\n"
                            except Exception as data_err:
                                context_str += f"\n--- {sym} ---\nCould not fetch data: {str(data_err)}\n"
                                
                        final_prompt = (
                            "System: You are an expert quantitative options trader and volatility analyst, familiar with Tastytrade mechanics. "
                            "Use the provided quantitative data to answer the user's query intelligently. Keep your response structured, actionable, and focused on the data.\n\n"
                            f"{context_str}\n\n"
                            f"User Query: {user_prompt}"
                        )
                        
                        response = model.generate_content(final_prompt)
                        st.markdown(f"### 🤖 AI Response *(Powered by {target_model})*")
                        st.info(response.text)
                        
                    except Exception as e:
                        st.error(f"AI Generation Error. Check your API key. Error details: {str(e)}")
# --- END OF PART 6 ---
