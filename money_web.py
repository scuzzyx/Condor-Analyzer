import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

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
    """Calculates IV Rank using 1-year Historical Volatility as the range proxy."""
    try:
        if current_iv == "N/A" or current_iv is None: 
            return "N/A"
        
        curr_iv_val = float(current_iv.replace('%', '')) / 100 if isinstance(current_iv, str) else current_iv
        
        returns = hist_1y['Close'].pct_change().dropna()
        hv_series = returns.rolling(20).std() * np.sqrt(252)
        hv_min = hv_series.min()
        hv_max = hv_series.max()
        
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
            if (vol_profile[i] > vol_profile[i-1] and 
                vol_profile[i] > vol_profile[i+1] and 
                vol_profile[i] > mean_vol * 0.5):
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
            except: 
                continue
    except: 
        pass
    return found_targets

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
            st.session_state['active_selections'] = st.session_state['active_selections'] + [ticker]
    st.session_state['ticker_input'] = ""

st.sidebar.text_input("➕ Add Custom Ticker:", key="ticker_input", on_change=add_custom_ticker)
selected_tickers = st.sidebar.multiselect("Active Bench:", options=st.session_state['custom_bench'], key="active_selections")

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

LIQUID_50 = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AMD', 'PLTR', 'NFLX', 
    'BA', 'DIS', 'BABA', 'UBER', 'COIN', 'HOOD', 'INTC', 'MU', 'AVGO', 'TSM', 
    'JPM', 'BAC', 'C', 'V', 'MA', 'PYPL', 'SQ', 'WMT', 'TGT', 'COST', 
    'HD', 'SBUX', 'NKE', 'MCD', 'XOM', 'CVX', 'CAT', 'GE', 'JNJ', 'PFE', 
    'UNH', 'LLY', 'CMCSA', 'VZ', 'T', 'QCOM', 'CRM', 'SNOW', 'SHOP', 'SPOT'
]

scan_tol = st.sidebar.slider("Tolerance (%)", 3, 15, 8) / 100.0

if st.sidebar.button("Run Radar Scan Now"):
    targets = run_radar_scan(LIQUID_50, scan_tol)
    if targets: 
        st.sidebar.success(f"🎯 Found: {', '.join(targets)}")
    else: 
        st.sidebar.warning("No targets.")

# --- GLOSSARY ---
st.markdown("---")
with st.expander("📖 Terminal Indicator Glossary (Quick Reference)", expanded=False):
    st.subheader("🚦 Title Risk & Veto Signals")
    st.write("- **IV Rank (IVR):** Relates current IV to the 52-week high/low. >50 is Tastytrade territory.")
    st.write("- **⚠️ [EARNINGS SOON]:** Earnings report occurs before expiration.")
    st.write("- **🔴 *FALLING KNIFE*:** Price below 8-EMA.")
    st.write("- **🟢 *NEUTRAL CHOP*:** Ideal environment for Iron Condors.")
    
    g1, g2, g3 = st.columns(3)
    with g1:
        st.subheader("🛡️ Trend")
        st.write("**8-Day EMA:** Algorithmic Trend line.")
        st.write("**RSI Stack:** Momentum indicator.")
    with g2:
        st.subheader("🎯 Structure")
        st.write("**POC:** Point of Control.")
        st.write("**Walls:** Support & Resistance floors/ceiling.")
    with g3:
        st.subheader("⚖️ Risk")
        st.write("**Max Pain:** Options seller's magnet strike.")
        st.write("**P/C OI Ratio:** Put/Call Open Interest sentiment.")

st.markdown("---")
tab_scanner, tab_deepdive = st.tabs(["🛡️ Option Scanner", "🔬 Technical Deep Dive"])

with tab_scanner:
    for symbol in selected_tickers:
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period="1y") 
            if len(hist) < 20: 
                continue
                
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            change_dlr = current_price - prev_close
            change_pct = ((current_price - prev_close) / prev_close) * 100
            change_color = "#ff4b4b" if change_dlr < 0 else "#09ab3b"
            
            ma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            ema_8 = hist['Close'].ewm(span=8, adjust=False).mean().iloc[-1]
            rsi_14 = calculate_rsi(hist['Close'], periods=14).iloc[-1]
            adx_14 = calculate_adx(hist)
            gap_risk = calculate_gap_risk(hist)
            poc, sup1, sup2, res1, res2 = calculate_volume_nodes(hist, current_price)
            
            volatility_dte = np.std(hist['Close'].pct_change().dropna()) * np.sqrt(dte if dte > 0 else 1)
            expected_move = current_price * (volatility_dte * z_score)
            put_strike = round(current_price - expected_move)
            call_strike = round(current_price + expected_move)
            put_trip = round(put_strike * 1.05, 2)
            call_trip = round(call_strike * 0.95, 2)
            
            atm_iv_raw = 0
            ivr = "N/A"
            max_pain = "N/A"
            pc_ratio = "N/A"
            atm_iv_display = "N/A"
            
            try:
                valid_dates = t.options
                if valid_dates:
                    target_date = selected_date_str if selected_date_str in valid_dates else valid_dates[0]
                    chain = t.option_chain(target_date)
                    calls = chain.calls
                    puts = chain.puts
                    
                    if not calls.empty:
                        closest_idx = (calls['strike'] - current_price).abs().idxmin()
                        atm_iv_raw = calls.loc[closest_idx, 'impliedVolatility']
                        atm_iv_display = f"{atm_iv_raw * 100:.1f}%"
                        ivr_val = calculate_ivr(hist, atm_iv_raw)
                        ivr = f"{ivr_val:.1f}" if isinstance(ivr_val, (int, float)) else "N/A"
                        
                    if not calls.empty and not puts.empty:
                        tot_put_oi = puts['openInterest'].sum()
                        tot_call_oi = calls['openInterest'].sum()
                        if tot_call_oi > 0: 
                            pc_ratio = f"{tot_put_oi / tot_call_oi:.2f}"
                            
                        all_strikes = sorted(list(set(calls['strike'].tolist() + puts['strike'].tolist())))
                        mp_val = float('inf')
                        mp_strike = "N/A"
                        
                        for s in all_strikes:
                            c_loss = calls[calls['strike'] < s].apply(lambda x: (s - x['strike']) * x['openInterest'], axis=1).sum()
                            p_loss = puts[puts['strike'] > s].apply(lambda x: (x['strike'] - s) * x['openInterest'], axis=1).sum()
                            if (c_loss + p_loss) < mp_val: 
                                mp_val = c_loss + p_loss
                                mp_strike = s
                                
                        if mp_strike != "N/A": 
                            max_pain = f"${mp_strike:.2f}"
            except: 
                pass

            earnings_veto = False
            try:
                cal = t.calendar
                if isinstance(cal, dict):
                    e_date = pd.to_datetime(cal.get('Earnings Date')[0])
                else:
                    e_date = pd.to_datetime(cal.loc['Earnings Date'].iloc[0])
                    
                if e_date and pd.notnull(e_date):
                    if datetime.now() < e_date < selected_date: 
                        earnings_veto = True
            except: 
                pass
                
            if current_price < ema_8 and rsi_14 < 45: 
                base_risk = "🔴 ***FALLING KNIFE***"
            elif current_price > ema_8 and rsi_14 < 50: 
                base_risk = "🟢 ***FLOOR CONFIRMED***"
            elif adx_14 > 25: 
                base_risk = f"🟡 ***TRENDING***: ADX {adx_14:.1f}"
            elif current_price > ma_20: 
                base_risk = "🟢 ***NEUTRAL CHOP***"
            else: 
                base_risk = "🟡 ***MED RISK***"

            risk = base_risk + (" [EARNINGS SOON]" if earnings_veto else "")
            ivr_color = "#09ab3b" if (isinstance(ivr, str) and ivr != "N/A" and float(ivr) > 50) else "#a6a6a6"

            with st.expander(f"{symbol} | Price: ${current_price:.2f} | IVR: {ivr} | Risk: {risk}", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                with c1: 
                    st.markdown(custom_metric_box("Today's Change", f"${current_price:.2f}", f"{change_pct:+.2f}%", sub_color=change_color), unsafe_allow_html=True)
                with c2: 
                    st.markdown(custom_metric_box("Put Strike", f"${put_strike}", f"Trip: ${put_trip}"), unsafe_allow_html=True)
                with c3: 
                    st.markdown(custom_metric_box("Call Strike", f"${call_strike}", f"Trip: ${call_trip}"), unsafe_allow_html=True)
                with c4: 
                    st.markdown(custom_metric_box("Volatility Rank", f"IVR: {ivr}", f"ATM IV: {atm_iv_display}", val_color=ivr_color), unsafe_allow_html=True)
                
                fig = go.Figure(data=[go.Candlestick(x=hist.index[-60:], open=hist['Open'][-60:], high=hist['High'][-60:], low=hist['Low'][-60:], close=hist['Close'][-60:], name="Price")])
                fig.add_trace(go.Scatter(x=hist.index[-60:], y=hist['Close'].ewm(span=8, adjust=False).mean()[-60:], line=dict(color='#ff9900', width=1.5, dash='dot'), name="8-EMA"))
                fig.add_hline(y=call_strike, line_width=2, line_color="green")
                fig.add_hline(y=put_strike, line_width=2, line_color="red")
                fig.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=20, b=0), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e: 
            st.error(f"Error processing {symbol}: {str(e)}")

with tab_deepdive:
    st.markdown("### 🔬 Automated Quantitative Analyst")
    deep_ticker = st.text_input("Enter Ticker for Deep Dive:", key="dd_ticker").upper().strip()
    
    if deep_ticker:
        try:
            t_dd = yf.Ticker(deep_ticker)
            hist_dd = t_dd.history(period="1y")
            
            if len(hist_dd) < 100: 
                st.warning("Insufficient trading history to process.")
            else:
                dd_price = hist_dd['Close'].iloc[-1]
                sma_20_dd = hist_dd['Close'].rolling(20).mean().iloc[-1]
                poc_dd, sup1_dd, _, res1_dd, _ = calculate_volume_nodes(hist_dd, dd_price)
                
                dd_iv = 0.30 
                try:
                    chain_dd = t_dd.option_chain(t_dd.options[0])
                    calls_dd = chain_dd.calls
                    closest_idx_dd = (calls_dd['strike'] - dd_price).abs().idxmin()
                    dd_iv = calls_dd.loc[closest_idx_dd, 'impliedVolatility']
                except: 
                    pass
                
                ivr_val = calculate_ivr(hist_dd, dd_iv)
                ivr_text = f"{ivr_val:.1f}" if isinstance(ivr_val, (int, float)) else "N/A"
                
                st.markdown("---")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Underwriting Translation")
                    st.write(f"**Current Price:** ${dd_price:.2f}")
                    st.write(f"**IV Rank:** {ivr_text}")
                    
                    if isinstance(ivr_val, (int, float)) and ivr_val > 50:
                        st.success("🎯 **TASTYTRADE SIGNAL:** High IV Rank. Premium is inflated. Ideal environment for Selling Iron Condors.")
                    elif isinstance(ivr_val, (int, float)) and ivr_val < 20:
                        st.info("🧊 **LOW VOLATILITY:** IV Rank is suppressed. Premium is cheap. Avoid Condors; look for net-debit directional trades.")
                    else:
                        st.warning("⚖️ **NEUTRAL VOL:** IV is in the middle of its yearly range.")

                    st.markdown(f"**Structure:** POC is located at {poc_dd.replace('$', r'\$')}. First floor support is at {sup1_dd.replace('$', r'\$')}.")
                    
                with col2:
                    fig_dd = go.Figure(data=[go.Candlestick(x=hist_dd.index[-90:], open=hist_dd['Open'][-90:], high=hist_dd['High'][-90:], low=hist_dd['Low'][-90:], close=hist_dd['Close'][-90:])])
                    fig_dd.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig_dd, use_container_width=True)
                    
        except Exception as e: 
            st.error(f"Deep Dive Processing Error: {str(e)}")
