# --- START OF PART 1 ---
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy.stats import norm
import urllib.request
import json
import time

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# --- CONFIG & THEME ---
st.set_page_config(page_title="Aegis Option Scanner", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h2 style='font-size: 2.2rem; margin-bottom: 0rem;'>🛡️ Aegis Option Scanner | Delta-Based Underwriting</h2>", unsafe_allow_html=True)

# --- BLACK-SCHOLES DELTA ENGINE ---
def calculate_delta(S, K, T, r, sigma, option_type='call'):
    """Calculates the Delta of an option using Black-Scholes."""
    if T <= 0 or sigma <= 0: return 0.5
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def find_delta_strikes(chain, current_price, dte, target_delta, option_type='call'):
    """Iterates through a chain to find the strike closest to target delta."""
    try:
        T = dte / 365.0
        r = 0.04 # Risk-free rate proxy (4%)
        
        # Filter for OTM strikes to speed up
        if option_type == 'call':
            df = chain[chain['strike'] >= current_price].copy()
        else:
            df = chain[chain['strike'] <= current_price].copy()
            
        if df.empty: return None
        
        def compute_row_delta(row):
            return calculate_delta(current_price, row['strike'], T, r, row['impliedVolatility'], option_type)
        
        df['delta'] = df.apply(compute_row_delta, axis=1)
        
        # Find strike with delta closest to absolute target
        df['delta_diff'] = (df['delta'].abs() - target_delta).abs()
        best_strike = df.loc[df['delta_diff'].idxmin(), 'strike']
        return best_strike
    except:
        return None

# --- INDICATORS ---
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_adx(hist, period=14):
    try:
        high, low, close = hist['High'], hist['Low'], hist['Close']
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return adx.iloc[-1]
    except:
        return 20 

def calculate_ivr(hist_1y, current_iv):
    try:
        if current_iv == "N/A" or current_iv is None or pd.isna(current_iv): 
            return "N/A"
        curr_iv_val = float(current_iv) if not isinstance(current_iv, str) else float(current_iv.replace('%', ''))/100
        returns = hist_1y['Close'].pct_change().dropna()
        hv_series = returns.rolling(20).std() * np.sqrt(252)
        hv_min, hv_max = float(hv_series.min()), float(hv_series.max())
        if hv_max == hv_min: return 50.0
        ivr = ((curr_iv_val - hv_min) / (hv_max - hv_min)) * 100
        return max(0, min(100, ivr))
    except: return "N/A"

def calculate_gap_risk(hist):
    try:
        gaps = abs((hist['Open'] - hist['Close'].shift(1)) / hist['Close'].shift(1))
        return gaps.tail(30).mean() * 100
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
        peaks = []
        mean_vol = np.mean(vol_profile)
        for i in range(1, bins-1):
            if vol_profile[i] > vol_profile[i-1] and vol_profile[i] > vol_profile[i+1] and vol_profile[i] > mean_vol * 0.5:
                peaks.append(price_bins[i])
        upper = sorted([p for p in peaks if p > current_price])
        lower = sorted([p for p in peaks if p < current_price])
        r1 = f"${upper[0]:.2f}" if len(upper) > 0 else "Sky (None)"
        r2 = f"${upper[1]:.2f}" if len(upper) > 1 else "⚠️ No Wall"
        s1 = f"${lower[-1]:.2f}" if len(lower) > 0 else "Freefall (None)"
        s2 = f"${lower[-2]:.2f}" if len(lower) > 1 else "⚠️ No Wall"
        return f"${poc:.2f}", s1, s2, r1, r2
    except: return "N/A", "N/A", "N/A", "N/A", "N/A"
# --- END OF PART 1 ---

# --- START OF PART 2 ---
@st.cache_data(ttl=86400)  
def get_pure_fridays(weeks=26):
    today = datetime.now()
    days_until_friday = (4 - today.weekday()) % 7
    next_friday = today + timedelta(days=days_until_friday)
    fridays = []
    for i in range(weeks):
        fridays.append((next_friday + timedelta(weeks=i)).strftime('%Y-%m-%d'))
    return fridays

@st.cache_data(ttl=3600) 
def run_radar_scan(ticker_list, threshold):
    found_targets = []
    try:
        bulk_data = yf.download(ticker_list, period="1mo", group_by='ticker', progress=False)
        for sym in ticker_list:
            try:
                hist = bulk_data[sym]['Close'].dropna() if len(ticker_list)>1 else bulk_data['Close'].dropna()
                if len(hist) > 10:
                    h, l = hist.max(), hist.min()
                    if (h - l) / hist.iloc[-1] < threshold: found_targets.append(sym)
            except: continue
    except: pass
    return found_targets

@st.cache_data(ttl=900)
def fetch_macro_data():
    vix_val, vix_pct = "N/A", "N/A"
    try:
        vix_hist = yf.Ticker("^VIX").history(period="5d")
        if len(vix_hist) >= 2:
            vix_val = float(vix_hist['Close'].iloc[-1])
            vix_pct = float(((vix_val - vix_hist['Close'].iloc[-2]) / vix_hist['Close'].iloc[-2]) * 100)
    except: pass
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            fg_val = round(data['fear_and_greed']['score'])
            fg_rating = data['fear_and_greed']['rating'].title()
    except: fg_val, fg_rating = "N/A", "N/A"
    return vix_val, vix_pct, fg_val, fg_rating

def custom_metric_box(label, value, sub_value, val_color="#FAFAFA", sub_color="#a6a6a6"):
    return f"""
    <div style="line-height: 1.4; margin-bottom: 14px;">
        <span style="font-size: 0.85rem; color: #a6a6a6;">{label}</span><br>
        <span style="font-size: 1.8rem; font-weight: 600; color: {val_color};">{value}</span><br>
        <span style="font-size: 0.9rem; font-weight: 500; color: {sub_color};">{sub_value}</span>
    </div>
    """

st.sidebar.header("🛠️ Dashboard Controls")
gemini_api_key = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else st.sidebar.text_input("🔑 Gemini API Key", type="password")

vix_v, vix_p, fg_v, fg_r = fetch_macro_data()
st.sidebar.markdown("### 🌍 Macro Sentiment")
mac1, mac2 = st.sidebar.columns(2)
with mac1:
    v_pct_str = f"{vix_p:+.2f}%" if isinstance(vix_p, float) else ""
    st.markdown(custom_metric_box("VIX Index", f"{vix_v:.2f}" if isinstance(vix_v, float) else "N/A", v_pct_str, sub_color="#ff4b4b" if (isinstance(vix_p, float) and vix_p > 0) else "#09ab3b"), unsafe_allow_html=True)
with mac2:
    st.markdown(custom_metric_box("Fear & Greed", str(fg_v), str(fg_r), val_color="#ffcc00"), unsafe_allow_html=True)
st.sidebar.markdown("---")
# --- END OF PART 2 ---

# --- START OF PART 3 ---
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

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Expiration & Risk")
use_custom_date = st.sidebar.checkbox("Use LEAPS / Custom Date")
if use_custom_date:
    selected_date_str = st.sidebar.date_input("Select Date:").strftime('%Y-%m-%d')
else:
    selected_date_str = st.sidebar.selectbox("Standard Friday (6 Mo):", options=get_pure_fridays(weeks=26))

dte = (datetime.strptime(selected_date_str, '%Y-%m-%d') - datetime.now()).days
if dte < 1: dte = 1

# --- THE DELTA SELECTOR ---
target_delta = st.sidebar.select_slider(
    "Target Strike Delta:",
    options=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    value=0.15,
    help="Higher Delta = Closer to price (More premium, Higher risk). Lower Delta = Farther away (Safe, but low premium)."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Radar Scan")
if st.sidebar.button("Run Range Radar"):
    targets = run_radar_scan(['AAPL','MSFT','NVDA','AMZN','META','GOOGL','TSLA','AMD','NFLX','BA','DIS','COIN','HOOD','MU','AVGO','JPM','BAC','V','MA','WMT','COST','SBUX','NKE'], 0.08)
    if targets: st.sidebar.success(f"🎯 Found: {', '.join(targets)}")
    else: st.sidebar.warning("No tight ranges.")

with st.expander("📖 Terminal Glossary", expanded=False):
    st.write("**Strike Delta:** A proxy for the chance of the option finishing in-the-money. A 0.15 Delta Put has a roughly 85% chance of expiring worthless (winning).")
    st.write("**IV Rank:** High IVR (>50) means options are expensive and great for selling.")
    st.write("**POC:** The price level where the most trading occurred. Acts as a magnet.")

if len(selected_tickers) > 1:
    with st.expander("🧩 Correlation Matrix", expanded=False):
        try:
            c_data = yf.download(selected_tickers, period="3mo", progress=False)['Close'].pct_change().tail(30)
            st.dataframe(c_data.corr().style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))
        except: st.write("Data pending...")

tab_scanner, tab_deepdive, tab_ai = st.tabs(["🛡️ Option Scanner", "🔬 Technical Deep Dive", "🧠 AI Quant Co-Pilot"])
# --- END OF PART 3 ---

# --- START OF PART 4 ---
with tab_scanner:
    if selected_tickers:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
    for idx, symbol in enumerate(selected_tickers):
        status_text.text(f"Scanning {symbol} Option Chain...")
        try:
            t = yf.Ticker(symbol)
            hist_1y = t.history(period="1y") 
            if len(hist_1y) < 20: continue
            hist = hist_1y.tail(63) 
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            change_dlr, change_pct = current_price - prev_close, ((current_price - prev_close)/prev_close)*100
            
            ema_8 = hist['Close'].ewm(span=8, adjust=False).mean().iloc[-1]
            rsi_14 = calculate_rsi(hist['Close']).iloc[-1]
            adx_14 = calculate_adx(hist)
            poc, sup1, sup2, res1, res2 = calculate_volume_nodes(hist, current_price)
            
            atm_iv_display, ivr, max_pain, pc_ratio = "N/A", "N/A", "N/A", "N/A"
            put_strike, call_strike = None, None
            target_date = selected_date_str
            
            try:
                valid_dates = t.options
                if valid_dates:
                    if target_date not in valid_dates:
                        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
                        valid_dts = [datetime.strptime(d, '%Y-%m-%d') for d in valid_dates]
                        target_date = min(valid_dts, key=lambda d: abs(d - target_dt)).strftime('%Y-%m-%d')
                    
                    chain = t.option_chain(target_date)
                    calls, puts = chain.calls, chain.puts
                    
                    if not calls.empty:
                        atm_idx = (calls['strike'] - current_price).abs().idxmin()
                        atm_iv = calls.loc[atm_idx, 'impliedVolatility']
                        atm_iv_display = f"{atm_iv * 100:.1f}%"
                        ivr = f"{calculate_ivr(hist_1y, atm_iv):.1f}"
                        
                        # DYNAMIC DELTA STRIKE FINDER
                        call_strike = find_delta_strikes(calls, current_price, dte, target_delta, 'call')
                        put_strike = find_delta_strikes(puts, current_price, dte, target_delta, 'put')

                    if not calls.empty and not puts.empty:
                        pc_ratio = f"{puts['openInterest'].sum() / calls['openInterest'].sum():.2f}"
                        all_s = sorted(list(set(calls['strike'].tolist() + puts['strike'].tolist())))
                        mp_val, mp_strike = float('inf'), "N/A"
                        for s in all_s:
                            loss = calls[calls['strike'] < s].apply(lambda x: (s - x['strike']) * x['openInterest'], axis=1).sum() + \
                                   puts[puts['strike'] > s].apply(lambda x: (x['strike'] - s) * x['openInterest'], axis=1).sum()
                            if loss < mp_val: mp_val, mp_strike = loss, s
                        max_pain = f"${mp_strike:.2f}"
            except: pass

            ex_div = "None"
            try:
                ex_ts = t.info.get('exDividendDate')
                if ex_ts: ex_div = datetime.fromtimestamp(ex_ts).strftime('%Y-%m-%d')
            except: pass

            earnings = "N/A"
            try:
                cal = t.calendar
                e_date = pd.to_datetime(cal.get('Earnings Date')[0]) if isinstance(cal, dict) else pd.to_datetime(cal.loc['Earnings Date'].iloc[0])
                earnings = e_date.strftime('%Y-%m-%d')
            except: pass

            risk = "🟢 NEUTRAL CHOP"
            if current_price < ema_8 and rsi_14 < 45: risk = "🔴 FALLING KNIFE"
            elif adx_14 > 25: risk = "🟡 TRENDING"
            
            with st.expander(f"{symbol} | Price: ${current_price:.2f} | {target_delta} Delta Strikes | {risk}", expanded=False):
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1: st.markdown(custom_metric_box("Change", f"${current_price:.2f}", f"{change_pct:+.2f}%", sub_color="#09ab3b" if change_pct>0 else "#ff4b4b"), unsafe_allow_html=True)
                with c2: st.markdown(custom_metric_box(f"{int(target_delta*100)} Delta Put", f"${put_strike}" if put_strike else "N/A", "Sell Below Support", sub_color="#ffcc00"), unsafe_allow_html=True)
                with c3: st.markdown(custom_metric_box(f"{int(target_delta*100)} Delta Call", f"${call_strike}" if call_strike else "N/A", "Sell Above Resis", sub_color="#ffcc00"), unsafe_allow_html=True)
                with c4: st.markdown(custom_metric_box("Volatility", f"IVR: {ivr}", f"IV: {atm_iv_display}"), unsafe_allow_html=True)
                with c5: st.markdown(custom_metric_box("Catalysts", f"ERN: {earnings}", f"DIV: {ex_div}"), unsafe_allow_html=True)
                
                st.markdown("---")
                v1, v2, v3, v4 = st.columns(4)
                with v1: st.write(f"**POC:** {poc}"); st.write(f"**ADX:** {adx_14:.1f}")
                with v2: st.write(f"**Max Pain:** {max_pain}"); st.write(f"**P/C Ratio:** {pc_ratio}")
                with v3: st.write(f"**Sup Walls:** {sup1}, {sup2}")
                with v4: st.write(f"**Res Walls:** {res1}, {res2}")

                fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].ewm(span=8).mean(), line=dict(color='#ff9900', dash='dot'), name="8-EMA"))
                if call_strike: fig.add_hline(y=call_strike, line_color="green", annotation_text=f"{target_delta} Delta Call")
                if put_strike: fig.add_hline(y=put_strike, line_color="red", annotation_text=f"{target_delta} Delta Put")
                fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e: st.error(f"Error {symbol}: {str(e)}")
        if selected_tickers: progress_bar.progress((idx+1)/len(selected_tickers)); time.sleep(0.3)
    if selected_tickers: status_text.empty(); progress_bar.empty()
# --- END OF PART 4 ---

# --- START OF PART 5 ---
with tab_deepdive:
    st.markdown("### 🔬 Deep Analysis")
    deep_t = st.text_input("Enter Ticker for Deep Dive:", key="dd_t").upper().strip()
    if deep_t:
        try:
            t_dd = yf.Ticker(deep_t)
            hist_dd = t_dd.history(period="1y")
            if len(hist_dd) < 50: st.warning("History too short.")
            else:
                hist_6 = hist_dd.tail(126); p = hist_6['Close'].iloc[-1]
                rsi = calculate_rsi(hist_6['Close']).iloc[-1]
                poc_dd, s1_dd, s2_dd, r1_dd, r2_dd = calculate_volume_nodes(hist_6, p)
                short_p = t_dd.info.get('shortPercentOfFloat', 0) * 100
                
                st.subheader(f"Quant Health: {deep_t}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Trend:** {'🟢 Bullish' if p > hist_6['Close'].rolling(50).mean().iloc[-1] else '🔴 Bearish'}")
                    st.markdown(f"**Momentum (RSI):** {rsi:.1f} ({'Overbought' if rsi>70 else 'Oversold' if rsi<30 else 'Neutral'})")
                    st.markdown(f"**Short Interest:** {short_p:.1f}%")
                    st.markdown(f"**Volume POC:** {poc_dd}")
                with col2:
                    fig_dd = go.Figure(data=[go.Candlestick(x=hist_6.index, open=hist_6['Open'], high=hist_6.High, low=hist_6.Low, close=hist_6.Close)])
                    fig_dd.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig_dd, use_container_width=True)
        except Exception as e: st.error(f"Analysis failed: {e}")
# --- END OF PART 5 ---
# --- START OF PART 6 ---
with tab_ai:
    st.markdown("### 🧠 AI Quant Co-Pilot")
    ai_t = st.multiselect("Select Tickers for AI Context:", options=st.session_state['custom_bench'], default=[selected_tickers[0]] if selected_tickers else [])
    user_p = st.text_area("Your Question:", placeholder="e.g. Compare ASTS and RKLB for a 15 Delta Iron Condor.")
    
    if st.button("Consult Gemini"):
        if not gemini_api_key: st.warning("Enter API Key in sidebar.")
        elif not ai_t or not user_p: st.warning("Select tickers and enter prompt.")
        else:
            with st.spinner("Processing market data..."):
                try:
                    genai.configure(api_key=gemini_api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    ctx = "CONTEXT:\n"
                    for s in ai_t:
                        t_ai = yf.Ticker(s); h = t_ai.history(period="6mo")
                        if h.empty: continue
                        iv = 0.4; chain = t_ai.option_chain(t_ai.options[0])
                        if not chain.calls.empty: iv = chain.calls['impliedVolatility'].mean()
                        ctx += f"{s}: Price ${h['Close'].iloc[-1]:.2f}, IV {iv*100:.1f}%, RSI {calculate_rsi(h['Close']).iloc[-1]:.1f}, Short {t_ai.info.get('shortPercentOfFloat',0)*100:.1f}%\n"
                    
                    full_p = f"System: Quantitative volatility expert. Data:\n{ctx}\nUser: {user_p}"
                    response = model.generate_content(full_p)
                    st.info(response.text)
                except Exception as e: st.error(f"AI Error: {e}")
# --- END OF PART 6 ---
