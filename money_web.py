import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import urllib.request
import json

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# --- CONFIG & THEME ---
st.set_page_config(page_title="Aegis Option Scanner", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h2 style='font-size: 2.2rem; margin-bottom: 0rem;'>🛡️ Aegis Option Scanner | Volatility & Directional Edge</h2>", unsafe_allow_html=True)

# --- PROBABILITY Z-SCORES ---
Z_SCORES = {
    "70%": 1.04, "75%": 1.15, "80%": 1.28, 
    "85%": 1.44, "90%": 1.645, "95%": 1.96
}

def load_url_bench():
    if "bench" in st.query_params:
        return st.query_params["bench"].split(",")
    return ["AMZN", "AAPL", "MSFT", "META", "GOOGL", "NVDA", "AMD", "PLTR", "TSLA", "NFLX"]

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
        if current_iv == "N/A" or current_iv is None: 
            return "N/A"
        curr_iv_val = float(current_iv.replace('%', '')) / 100 if isinstance(current_iv, str) else current_iv
        returns = hist_1y['Close'].pct_change().dropna()
        hv_series = returns.rolling(20).std() * np.sqrt(252)
        hv_min, hv_max = hv_series.min(), hv_series.max()
        ivr = ((curr_iv_val - hv_min) / (hv_max - hv_min)) * 100
        return max(0, min(100, ivr))
    except:
        return "N/A"

def calculate_gap_risk(hist):
    try:
        gaps = abs((hist['Open'] - hist['Close'].shift(1)) / hist['Close'].shift(1))
        return gaps.tail(30).mean() * 100
    except:
        return 0

def calculate_volume_nodes(hist, current_price, bins=30):
    try:
        min_p, max_p = hist['Low'].min(), hist['High'].max()
        price_bins = np.linspace(min_p, max_p, bins)
        inds = np.digitize(hist['Close'], price_bins)
        vol_profile = np.zeros(bins)
        for i in range(len(hist)):
            if 0 <= inds[i]-1 < bins:
                vol_profile[inds[i]-1] += hist['Volume'].iloc[i]
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
    except:
        return "N/A", "N/A", "N/A", "N/A", "N/A"

@st.cache_data(ttl=3600)  
def get_friday_expirations():
    try:
        spy = yf.Ticker("SPY")
        dates = spy.options
        fridays = [d for d in dates if datetime.strptime(d, '%Y-%m-%d').weekday() == 4]
        return fridays[:10]
    except:
        return [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(14, 60) if (datetime.now() + timedelta(days=i)).weekday() == 4]

@st.cache_data(ttl=3600) 
def run_radar_scan(ticker_list, threshold):
    found_targets = []
    try:
        bulk_data = yf.download(ticker_list, period="1mo", group_by='ticker', progress=False)
        for sym in ticker_list:
            try:
                if len(ticker_list) > 1:
                    hist = bulk_data[sym]['Close'].dropna()
                else:
                    hist = bulk_data['Close'].dropna()
                if len(hist) > 10:
                    h, l = hist.max(), hist.min()
                    cur = hist.iloc[-1]
                    if (h - l) / cur < threshold: found_targets.append(sym)
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
    
    fg_val, fg_rating = "N/A", "N/A"
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Referer': 'https://edition.cnn.com/'
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            fg_val = round(data['fear_and_greed']['score'])
            fg_rating = data['fear_and_greed']['rating'].title()
    except: pass
    
    return vix_val, vix_pct, fg_val, fg_rating

def custom_metric_box(label, value, sub_value, val_color="#FAFAFA", sub_color="#a6a6a6"):
    return f"""
    <div style="line-height: 1.4; margin-bottom: 14px;">
        <span style="font-size: 0.85rem; color: #a6a6a6; font-family: sans-serif;">{label}</span><br>
        <span style="font-size: 1.8rem; font-weight: 600; color: {val_color}; font-family: sans-serif;">{value}</span><br>
        <span style="font-size: 0.9rem; font-weight: 500; color: {sub_color}; font-family: sans-serif;">{sub_value}</span>
    </div>
    """

# --- SIDEBAR ---
st.sidebar.header("🛠️ Dashboard Controls")

# AI KEY INJECTION
gemini_api_key = st.sidebar.text_input("🔑 Gemini API Key (For AI Co-Pilot)", type="password", help="Get a free key at aistudio.google.com")

vix_v, vix_p, fg_v, fg_r = fetch_macro_data()
st.sidebar.markdown("### 🌍 Macro Sentiment")
mac1, mac2 = st.sidebar.columns(2)
with mac1:
    vix_color = "#ff4b4b" if (isinstance(vix_p, float) and vix_p > 0) else "#09ab3b"
    v_val_str = f"{vix_v:.2f}" if isinstance(vix_v, float) else "N/A"
    v_pct_str = f"{vix_p:+.2f}%" if isinstance(vix_p, float) else ""
    st.markdown(custom_metric_box("VIX Index", v_val_str, v_pct_str, sub_color=vix_color), unsafe_allow_html=True)
with mac2:
    fg_color = "#09ab3b" if (isinstance(fg_v, int) and fg_v >= 55) else ("#ff4b4b" if (isinstance(fg_v, int) and fg_v <= 45) else "#ffcc00")
    st.markdown(custom_metric_box("Fear & Greed", str(fg_v), str(fg_r), val_color=fg_color), unsafe_allow_html=True)

st.sidebar.markdown("---")

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
        if ticker not in st.session_state['active_selections']:
            current_active = st.session_state['active_selections']
            st.session_state['active_selections'] = current_active + [ticker]
    st.session_state['ticker_input'] = ""

st.sidebar.text_input("➕ Add Custom Ticker:", key="ticker_input", on_change=add_custom_ticker)
selected_tickers = st.sidebar.multiselect("Active Bench:", options=st.session_state['custom_bench'], key="active_selections")

if st.sidebar.button("🔗 Generate Custom Link"):
    st.query_params["bench"] = ",".join(st.session_state['active_selections'])
    st.sidebar.success("URL updated!")

st.sidebar.markdown("---")
available_fridays = get_friday_expirations()
if available_fridays:
    selected_date_str = st.sidebar.selectbox("Expiration:", options=available_fridays)
    selected_date = datetime.strptime(selected_date_str, '%Y-%m-%d')
    dte = (selected_date - datetime.now()).days
else:
    dte, selected_date_str = 14, None

prob_target = st.sidebar.selectbox("Probability Target:", options=list(Z_SCORES.keys()), index=4)
z_score = Z_SCORES[prob_target]

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Range-Bound Radar")
LIQUID_50 = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AMD', 'PLTR', 'NFLX', 'BA', 'DIS', 'BABA', 'UBER', 'COIN', 'HOOD', 'INTC', 'MU', 'AVGO', 'TSM', 'JPM', 'BAC', 'C', 'V', 'MA', 'PYPL', 'SQ', 'WMT', 'TGT', 'COST', 'HD', 'SBUX', 'NKE', 'MCD', 'XOM', 'CVX', 'CAT', 'GE', 'JNJ', 'PFE', 'UNH', 'LLY', 'CMCSA', 'VZ', 'T', 'QCOM', 'CRM', 'SNOW', 'SHOP', 'SPOT']
scan_tol = st.sidebar.slider("Tolerance (%)", 3, 15, 8) / 100.0
if st.sidebar.button("Run Radar Scan Now"):
    targets = run_radar_scan(LIQUID_50, scan_tol)
    if targets: st.sidebar.success(f"🎯 Found: {', '.join(targets)}")
    else: st.sidebar.warning("No targets.")

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
    
    g1, g2, g3 = st.columns(3)
    with g1:
        st.subheader("🛡️ Trend")
        st.write("**8-Day EMA:** Algorithmic Trend line.")
        st.write("**RSI Stack:** Momentum indicator.")
    with g2:
        st.subheader("🎯 Structure")
        st.write("**POC:** Point of Control. Price magnet.")
        st.write("**Walls:** Support & Resistance floors/ceilings.")
    with g3:
        st.subheader("⚖️ Risk")
        st.write("**Max Pain:** Options seller's magnet strike.")
        st.write("**P/C OI Ratio:** Put/Call Open Interest sentiment.")

if len(selected_tickers) > 1:
    with st.expander("🧩 Portfolio Risk: 30-Day Correlation Matrix", expanded=False):
        try:
            bench_data = yf.download(selected_tickers, period="3mo", progress=False)['Close']
            returns = bench_data.pct_change().tail(30)
            corr_matrix = returns.corr()
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))
        except:
            st.write("Not enough data.")

st.markdown("---")
tab_scanner, tab_deepdive, tab_ai = st.tabs(["🛡️ Option Scanner", "🔬 Technical Deep Dive", "🧠 AI Quant Co-Pilot"])

with tab_scanner:
    for symbol in selected_tickers:
        try:
            t = yf.Ticker(symbol)
            hist_1y = t.history(period="1y") 
            if len(hist_1y) < 20: continue
            
            hist = hist_1y.tail(63) 
                
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            change_dlr, change_pct = current_price - prev_close, ((current_price - prev_close) / prev_close) * 100
            change_color = "#ff4b4b" if change_dlr < 0 else "#09ab3b"
            
            ma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            ema_8 = hist['Close'].ewm(span=8, adjust=False).mean().iloc[-1]
            
            rsi_series_5 = calculate_rsi(hist['Close'], periods=5)
            rsi_5, rsi_5_prev = rsi_series_5.iloc[-1], rsi_series_5.iloc[-2]
            rsi_9 = calculate_rsi(hist['Close'], periods=9).iloc[-1]
            rsi_14 = calculate_rsi(hist['Close'], periods=14).iloc[-1]
            
            adx_14, gap_risk = calculate_adx(hist), calculate_gap_risk(hist)
            poc, sup1, sup2, res1, res2 = calculate_volume_nodes(hist, current_price)
            
            volatility_dte = np.std(hist['Close'].pct_change().dropna()) * np.sqrt(dte if dte > 0 else 1)
            expected_move = current_price * (volatility_dte * z_score)
            put_strike, call_strike = round(current_price - expected_move), round(current_price + expected_move)
            put_trip, call_trip = round(put_strike * 1.05, 2), round(call_strike * 0.95, 2)
            
            atm_iv_raw = 0
            atm_iv_display, ivr, max_pain, pc_ratio = "N/A", "N/A", "N/A", "N/A"
            
            try:
                valid_dates = t.options
                if valid_dates:
                    target_date = selected_date_str if selected_date_str in valid_dates else valid_dates[0]
                    chain = t.option_chain(target_date)
                    calls, puts = chain.calls, chain.puts
                    
                    if not calls.empty:
                        closest_idx = (calls['strike'] - current_price).abs().idxmin()
                        atm_iv_raw = calls.loc[closest_idx, 'impliedVolatility']
                        atm_iv_display = f"{atm_iv_raw * 100:.1f}%"
                        ivr_val = calculate_ivr(hist_1y, atm_iv_raw)
                        ivr = f"{ivr_val:.1f}" if isinstance(ivr_val, (int, float)) else "N/A"
                    
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
            except: pass

            ex_div_date, ex_div_veto = "None scheduled", False
            try:
                ex_ts = t.info.get('exDividendDate')
                if ex_ts:
                    ex_dt = datetime.fromtimestamp(ex_ts)
                    ex_div_date = ex_dt.strftime('%Y-%m-%d')
                    if datetime.now() < ex_dt < selected_date: ex_div_veto = True
            except: pass

            earnings_date, earnings_veto = "Not scheduled", False
            try:
                cal = t.calendar
                e_date = pd.to_datetime(cal.get('Earnings Date')[0]) if isinstance(cal, dict) else pd.to_datetime(cal.loc['Earnings Date'].iloc[0])
                if e_date and pd.notnull(e_date):
                    earnings_date = e_date.strftime('%Y-%m-%d')
                    if datetime.now() < e_date < selected_date: earnings_veto = True
            except: pass

            if current_price < ema_8 and rsi_14 < 45: base_risk = "🔴 ***FALLING KNIFE***: Call Spreads Only"
            elif current_price > ema_8 and rsi_5 > rsi_5_prev and rsi_14 < 50: base_risk = "🟢 ***FLOOR CONFIRMED***: Put Spreads Only"
            elif gap_risk > 1.5: base_risk = f"🟠 ***GAP RISK***: High Overnight Vol ({gap_risk:.2f}%)"
            elif adx_14 > 25: base_risk = f"🟡 ***TRENDING***: ADX {adx_14:.1f} (Pick Directional)"
            elif current_price > ma_20: base_risk = "🟢 ***NEUTRAL CHOP***: Condor Territory"
            else: base_risk = "🟡 ***MED RISK***: Price Stalling"

            risk = base_risk + (" [EARNINGS SOON]" if earnings_veto else "") + (" ⚠️[EX-DIVIDEND DANGER]" if ex_div_veto else "")
            ivr_color = "#09ab3b" if (isinstance(ivr, str) and ivr != "N/A" and float(ivr) > 50) else "#a6a6a6"

            with st.expander(f"{symbol} | Price: ${current_price:.2f} | IVR: {ivr} | Risk: {risk}", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.markdown(custom_metric_box("Today's Change", f"${current_price:.2f}", f"{change_dlr:+.2f} ({change_pct:+.2f}%)", sub_color=change_color), unsafe_allow_html=True)
                with c2: st.markdown(custom_metric_box("Put Strategy", f"${put_strike}", f"Trip Wire: ${put_trip}", sub_color="#ffcc00"), unsafe_allow_html=True)
                with c3: st.markdown(custom_metric_box("Call Strategy", f"${call_strike}", f"Trip Wire: ${call_trip}", sub_color="#ffcc00"), unsafe_allow_html=True)
                with c4: st.markdown(custom_metric_box("Volatility Rank", f"IVR: {ivr}", f"ATM IV: {atm_iv_display}", val_color=ivr_color), unsafe_allow_html=True)
                
                st.markdown("---")
                st.caption("🛡️ Risk Underwriting Data")
                u1, u2, u3 = st.columns(3)
                with u1: st.markdown(custom_metric_box("Max Pain", f"{max_pain}", "Gravity point for Friday", sub_color="#a6a6a6"), unsafe_allow_html=True)
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
                    div_sub = "EARLY ASSIGNMENT RISK" if ex_div_veto else "Upcoming Ex-Div Date"
                    st.markdown(custom_metric_box("Ex-Dividend", f"{ex_div_date}", div_sub, sub_color=div_color), unsafe_allow_html=True)

                st.markdown("---")
                v1, v2, v3, v4 = st.columns(4)
                def get_s(v): return "Oversold" if v <= 30 else "Overbought" if v >= 70 else "Neutral"
                with v1:
                    st.caption("🧲 POC & Trend")
                    st.write(f"**Price:** {poc}")
                    st.write(f"**ADX:** {adx_14:.1f}")
                with v2:
                    st.caption("📈 RSI Stack")
                    st.write(f"5D: {rsi_5:.1f} ({get_s(rsi_5)})")
                    st.write(f"9D: {rsi_9:.1f} ({get_s(rsi_9)})")
                    st.write(f"14D: {rsi_14:.1f} ({get_s(rsi_14)})")
                with v3:
                    st.caption("🔴 Support Walls (Red)")
                    st.write(f"**Support Wall 1:** {sup1}")
                    st.write(f"**Support Wall 2:** {sup2}")
                with v4:
                    st.caption("🟢 Resistance Walls (Green)")
                    st.write(f"**Resistance Wall 1:** {res1}")
                    st.write(f"**Resistance Wall 2:** {res2}")

                fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price")])
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].ewm(span=8, adjust=False).mean(), line=dict(color='#ff9900', width=1.5, dash='dot'), name="8-EMA"))
                fig.add_hline(y=call_strike, line_width=2, line_color="green", annotation_text="Call Strike")
                fig.add_hline(y=put_strike, line_width=2, line_color="red", annotation_text="Put Strike")
                fig.add_hline(y=call_trip, line_width=1, line_dash="dash", line_color="yellow", annotation_text="Call Alert")
                fig.add_hline(y=put_trip, line_width=1, line_dash="dash", line_color="yellow", annotation_text="Put Alert")
                fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading {symbol}: {str(e)}")

with tab_deepdive:
    st.markdown("### 🔬 Automated Quantitative Analyst")
    st.write("Enter a single ticker below. The system will process the underlying mathematics, liquidity, and tail risks to translate the chart structure into plain English.")
    
    deep_ticker = st.text_input("Enter Ticker for Deep Dive (e.g., TSLA, SPY):", key="dd_ticker").upper().strip()
    
    if deep_ticker:
        try:
            t_dd = yf.Ticker(deep_ticker)
            hist_dd = t_dd.history(period="1y") 
            
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

                info_dd = t_dd.info
                short_pct = info_dd.get('shortPercentOfFloat', 0)
                inst_pct = info_dd.get('heldPercentInstitutions', 0)
                target_price = info_dd.get('targetMeanPrice')
                
                tr1 = hist_6mo['High'] - hist_6mo['Low']
                tr2 = abs(hist_6mo['High'] - hist_6mo['Close'].shift(1))
                tr3 = abs(hist_6mo['Low'] - hist_6mo['Close'].shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr_14 = tr.rolling(14).mean().iloc[-1]
                
                returns_1y = hist_dd['Close'].pct_change().dropna()
                hv_20 = returns_1y.rolling(window=20).std() * np.sqrt(252)
                current_hv = hv_20.iloc[-1]
                
                skew_status, skew_text = "⚖️ **Balanced Skew:**", "Put and Call implied volatilities are relatively balanced."
                liq_status, liq_text = "⚖️ **Adequate Liquidity:**", "Volume is average. Be cautious of bid-ask spreads."
                atm_iv_dd = current_hv 
                oi_fig = None
                
                try:
                    dd_dates = t_dd.options
                    if dd_dates:
                        dd_chain = t_dd.option_chain(dd_dates[0])
                        total_vol = dd_chain.calls['volume'].fillna(0).sum() + dd_chain.puts['volume'].fillna(0).sum()
                        avg_vol = total_vol / (len(dd_chain.calls) + len(dd_chain.puts))
                        if avg_vol > 300: liq_status, liq_text = "🌊 **A+ Liquidity:**", "Highly liquid options chain. Minimal slippage expected."
                        elif avg_vol < 50: liq_status, liq_text = "🧊 **Poor Liquidity:**", "Low volume. Expect massive slippage."
                        
                        calls, puts = dd_chain.calls, dd_chain.puts
                        otm_call = calls[calls['strike'] >= dd_price * 1.1]
                        otm_put = puts[puts['strike'] <= dd_price * 0.9]
                        
                        if not calls.empty:
                            atm_iv_dd = calls.iloc[(calls['strike'] - dd_price).abs().argsort()[:1]]['impliedVolatility'].values[0]
                            
                        if not otm_call.empty and not otm_put.empty:
                            skew_diff = otm_put.iloc[-1]['impliedVolatility'] - otm_call.iloc[0]['impliedVolatility']
                            if skew_diff > 0.12: skew_status, skew_text = "🚨 **Severe Downside Skew:**", "Market pricing OTM Puts higher than Calls. Crash protection is expensive."
                            elif skew_diff < -0.12: skew_status, skew_text = "🚀 **Upside Call Skew:**", "OTM Calls pricing higher than Puts. Market anticipating upside."
                                
                        calls_oi = calls[['strike', 'openInterest']].copy()
                        puts_oi = puts[['strike', 'openInterest']].copy()
                        lb, ub = dd_price * 0.85, dd_price * 1.15
                        calls_oi = calls_oi[(calls_oi['strike'] >= lb) & (calls_oi['strike'] <= ub)]
                        puts_oi = puts_oi[(puts_oi['strike'] >= lb) & (puts_oi['strike'] <= ub)]
                        
                        oi_fig = go.Figure()
                        oi_fig.add_trace(go.Bar(x=calls_oi['strike'], y=calls_oi['openInterest'], name='Call OI', marker_color='#09ab3b', opacity=0.7))
                        oi_fig.add_trace(go.Bar(x=puts_oi['strike'], y=puts_oi['openInterest'], name='Put OI', marker_color='#ff4b4b', opacity=0.7))
                        oi_fig.update_layout(title="Open Interest Profile (Nearest Expiry)", template="plotly_dark", height=300, margin=dict(l=0, r=0, t=30, b=0), barmode='group')
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

        with tab_ai:
    st.markdown("### 🧠 AI Quant Co-Pilot")
    st.write("Compare tickers, ask for a trade thesis, or summarize data using Google Gemini.")
    
    if not GENAI_AVAILABLE:
        st.error("⚠️ `google-generativeai` library not found. Please run `pip install google-generativeai` in your terminal.")
    else:
        ai_tickers = st.multiselect(
            "1. Select Tickers to include in AI Context (It will fetch their live data behind the scenes):", 
            options=st.session_state['custom_bench'],
            default=[selected_tickers[0]] if selected_tickers else []
        )
        
        user_prompt = st.text_area(
            "2. Enter your question for the AI:", 
            placeholder="e.g., Compare the IV Rank, Point of Control, and Short Interest of ASTS vs RKLB. Which is better for an Iron Condor?"
        )
        
        if st.button("Ask Gemini 🤖"):
            if not gemini_api_key:
                st.warning("⚠️ Please enter your Gemini API Key in the sidebar controls.")
            elif not ai_tickers:
                st.warning("⚠️ Please select at least one ticker from the dropdown.")
            elif not user_prompt:
                st.warning("⚠️ Please enter a prompt.")
            else:
                with st.spinner("Fetching live quantitative data and consulting Gemini..."):
                    try:
                        genai.configure(api_key=gemini_api_key)
                        # We use gemini-1.5-flash as it is fast and highly capable for text/data parsing
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        context_str = "CURRENT QUANTITATIVE MARKET DATA:\n"
                        for sym in ai_tickers:
                            try:
                                t_ai = yf.Ticker(sym)
                                hist_ai = t_ai.history(period="1y")
                                if len(hist_ai) < 50: continue
                                price = hist_ai['Close'].iloc[-1]
                                
                                hist_6mo = hist_ai.tail(126)
                                rsi = calculate_rsi(hist_6mo['Close']).iloc[-1]
                                adx = calculate_adx(hist_6mo)
                                poc, s1, s2, r1, r2 = calculate_volume_nodes(hist_6mo, price)
                                
                                atm_iv = 0.3
                                try:
                                    calls = t_ai.option_chain(t_ai.options[0]).calls
                                    atm_iv = calls.iloc[(calls['strike'] - price).abs().argsort()[:1]]['impliedVolatility'].values[0]
                                except: pass
                                
                                ivr = calculate_ivr(hist_ai, atm_iv)
                                ivr_str = f"{ivr:.1f}" if isinstance(ivr, (int, float)) else "N/A"
                                
                                short_pct = t_ai.info.get('shortPercentOfFloat', 0) * 100
                                
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
                        st.markdown("### 🤖 AI Response")
                        st.info(response.text)
                        
                    except Exception as e:
                        st.error(f"AI Generation Error. Check your API key. Error details: {str(e)}")
