import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- CONFIG & THEME ---
st.set_page_config(page_title="Condor-Tool", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Condor-Tool | Volatility & Probability Screener")

# --- HIDE STREAMLIT BRANDING, GITHUB BUTTONS, AND PROFILE BADGES ---
hide_streamlit_style = """
    <style>
    /* HIDE TOP HEADER & GITHUB/FORK BUTTONS */
    [data-testid="stHeader"] {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    
    /* HIDE THE STANDARD STREAMLIT FOOTER */
    footer {display: none !important;}
    
    /* HIDE THE "HOSTED WITH STREAMLIT" PROFILE BADGE (WILDCARD HUNT) */
    div[class^="viewerBadge"] {display: none !important;}
    div[class*="viewerBadge"] {display: none !important;}
    a[href*="streamlit.io/cloud"] {display: none !important;}
    
    /* PULL THE DASHBOARD UP TO REMOVE THE BLANK GAP */
    .block-container {padding-top: 1rem !important;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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
        r2 = f"${upper[1]:.2f}" if len(upper) > 1 else "⚠️ No 90d Wall"
        s1 = f"${lower[-1]:.2f}" if len(lower) > 0 else "Freefall (None)"
        s2 = f"${lower[-2]:.2f}" if len(lower) > 1 else "⚠️ No 90d Wall"
        
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
                    high_1m = hist.max().iloc[0] if isinstance(hist.max(), pd.Series) else hist.max()
                    low_1m = hist.min().iloc[0] if isinstance(hist.min(), pd.Series) else hist.min()
                    current = hist.iloc[-1].iloc[0] if isinstance(hist.iloc[-1], pd.Series) else hist.iloc[-1]
                    
                    if (high_1m - low_1m) / current < threshold:
                        found_targets.append(sym)
            except:
                continue
    except Exception as e:
        pass
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

st.sidebar.text_input("➕ Add Custom Ticker (e.g. CAVA):", key="ticker_input", on_change=add_custom_ticker)

selected_tickers = st.sidebar.multiselect(
    "Active Bench:", 
    options=st.session_state['custom_bench'], 
    key="active_selections"
)

if st.sidebar.button("🔗 Generate Custom Link"):
    bench_string = ",".join(st.session_state['active_selections'])
    st.query_params["bench"] = bench_string
    st.sidebar.success("URL updated! Bookmark this page to save your bench.")

st.sidebar.markdown("---")

available_fridays = get_friday_expirations()
if available_fridays:
    selected_date_str = st.sidebar.selectbox("Expiration Date (Fridays Only):", options=available_fridays)
    selected_date = datetime.strptime(selected_date_str, '%Y-%m-%d')
    dte = (selected_date - datetime.now()).days
else:
    st.sidebar.error("Error loading dates.")
    dte = 14
    selected_date_str = None

prob_target = st.sidebar.selectbox("Probability of Success Target:", options=list(Z_SCORES.keys()), index=4)
z_score = Z_SCORES[prob_target]

# --- RANGE-BOUND RADAR (MANUAL SCAN) ---
st.sidebar.markdown("---")
st.sidebar.subheader("📡 Range-Bound Radar")
st.sidebar.caption("Scan restricted to the Top 50 highest options liquidity stocks.")

LIQUID_50 = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AMD', 'PLTR', 'NFLX',
    'BA', 'DIS', 'BABA', 'UBER', 'COIN', 'HOOD', 'INTC', 'MU', 'AVGO', 'TSM',
    'JPM', 'BAC', 'C', 'V', 'MA', 'PYPL', 'SQ', 'WMT', 'TGT', 'COST',
    'HD', 'SBUX', 'NKE', 'MCD', 'XOM', 'CVX', 'CAT', 'GE', 'JNJ', 'PFE',
    'UNH', 'LLY', 'CMCSA', 'VZ', 'T', 'QCOM', 'CRM', 'SNOW', 'SHOP', 'SPOT'
]

scan_tolerance = st.sidebar.slider("Consolidation Tolerance (%)", min_value=3, max_value=15, value=8) / 100.0

if st.sidebar.button("Run Radar Scan Now"):
    with st.sidebar.status(f"Scanning Top 50 Liquid at {int(scan_tolerance*100)}% tolerance..."):
        targets = run_radar_scan(LIQUID_50, scan_tolerance)
        if targets:
            st.sidebar.success(f"🎯 Targets Found: {', '.join(targets)}")
        else:
            st.sidebar.warning("No setups found. Try increasing the Tolerance (%) slider.")

# --- PORTFOLIO CORRELATION ENGINE ---
st.markdown("---")
if len(selected_tickers) > 1:
    with st.expander("🧩 Portfolio Risk: 30-Day Correlation Matrix", expanded=False):
        try:
            bench_data = yf.download(selected_tickers, period="3mo", progress=False)['Close']
            returns = bench_data.pct_change().tail(30)
            corr_matrix = returns.corr()
            
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.8:
                        high_corr_pairs.append(f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]} ({corr_matrix.iloc[i, j]:.2f})")
                        
            if high_corr_pairs:
                st.warning(f"⚠️ HIGH CONCENTRATION RISK: {', '.join(high_corr_pairs)}. Avoid deploying capital on both simultaneously.")
            else:
                st.success("🟢 No severe correlations found in active bench. Good diversification.")
                
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))
        except Exception as e:
            st.write("Not enough data to calculate correlation matrix.")
st.markdown("---")

# --- MAIN ENGINE ---
for symbol in selected_tickers:
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="3mo")
        
        if len(hist) < 20:
            st.warning(f"Not enough data for {symbol}")
            continue
            
        # Daily Performance
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        change_dlr = current_price - prev_close
        change_pct = (change_dlr / prev_close) * 100
        
        # Quantitative Indicators
        ma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        ema_8 = hist['Close'].ewm(span=8, adjust=False).mean().iloc[-1]
        support_3mo = hist['Close'].min()
        
        # --- MULTI-TIMEFRAME RSI MATH ---
        rsi_series_5 = calculate_rsi(hist['Close'], periods=5)
        rsi_5 = rsi_series_5.iloc[-1]
        rsi_5_prev = rsi_series_5.iloc[-2]
        
        rsi_9 = calculate_rsi(hist['Close'], periods=9).iloc[-1]
        rsi_14 = calculate_rsi(hist['Close'], periods=14).iloc[-1]
        
        adx_14 = calculate_adx(hist)
        gap_risk = calculate_gap_risk(hist)
        poc, sup1, sup2, res1, res2 = calculate_volume_nodes(hist, current_price)
        
        # Options Math
        daily_returns = hist['Close'].pct_change().dropna()
        volatility_dte = np.std(daily_returns) * np.sqrt(dte if dte > 0 else 1)
        expected_move = current_price * (volatility_dte * z_score)
        
        put_strike = round(current_price - expected_move)
        call_strike = round(current_price + expected_move)
        
        # 5% Trip Wires
        put_trip = round(put_strike * 1.05, 2)
        call_trip = round(call_strike * 0.95, 2)
        
        # Live IV
        atm_iv = "N/A"
        try:
            if selected_date_str:
                chain = t.option_chain(selected_date_str)
                calls = chain.calls
                atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
                atm_iv = f"{atm_call['impliedVolatility'].values[0] * 100:.1f}%"
        except:
            pass

        # Earnings Check
        earnings_date = "Not yet scheduled"
        earnings_veto = False
        try:
            cal = t.calendar
            e_date = None
            if isinstance(cal, dict) and 'Earnings Date' in cal and cal['Earnings Date']:
                e_date = cal['Earnings Date'][0]
            elif isinstance(cal, pd.DataFrame) and not cal.empty and 'Earnings Date' in cal.index:
                e_date = cal.loc['Earnings Date'][0]
            if e_date:
                e_date = pd.to_datetime(e_date) 
                earnings_date = e_date.strftime('%Y-%m-%d')
                if datetime.now() < e_date < selected_date:
                    earnings_veto = True
        except:
            pass
            
        # --- BINARY RISK SYSTEM ---
        if earnings_veto:
            risk = "⛔ VETO: Earnings Before Expiration"
        elif current_price < ema_8 and rsi_14 < 45:
            risk = "🔴 FALLING KNIFE: Price below 8-EMA (Wait for reclaim)"
        elif current_price > ema_8 and rsi_5 > rsi_5_prev and rsi_14 < 50:
            risk = "🟢 FLOOR CONFIRMED: 8-EMA Reclaimed & Sellers Exhausted"
        elif gap_risk > 1.5:
            risk = f"🟠 GAP RISK: High Overnight Volatility ({gap_risk:.2f}%)"
        elif adx_14 > 25:
            risk = f"🟡 TRENDING: ADX {adx_14:.1f} (Avoid Iron Condors)"
        elif current_price > ma_20:
            risk = "🟢 NEUTRAL CHOP: Favorable for Premium Selling"
        else:
            risk = "🟡 MED RISK: Price Action Stalling"

        # --- UI DISPLAY ---
        with st.expander(f"{symbol}  |  Price: ${current_price:.2f}  |  Risk: {risk}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Today's Change", f"${current_price:.2f}", f"{change_dlr:.2f} ({change_pct:.2f}%)")
            # --- SWAPPED COLOR INDICATORS IN METRICS ---
            col2.metric("Put Strategy", f"Strike: ${put_strike}", f"Trip Wire: ${put_trip}", delta_color="inverse")
            col3.metric("Call Strategy", f"Strike: ${call_strike}", f"Trip Wire: ${call_trip}", delta_color="normal")
            col4.metric("Market Data", f"IV: {atm_iv}", f"Earnings: {earnings_date}", delta_color="off")
            
            st.markdown("---")
            
            def get_rsi_state(val):
                if pd.isna(val): return "Neutral"
                elif val >= 70: return "Overbought"
                elif val <= 30: return "Oversold"
                else: return "Neutral"
                
            v1, v2, v3, v4 = st.columns(4)
            
            with v1:
                st.caption("🧲 Point of Control (POC)")
                st.write(f"**Price:** {poc}")
                st.write("*(Highest Vol Magnet)*")
            
            with v2:
                st.caption("📈 RSI Momentum Stack")
                st.write(f"**5-Day:** {rsi_5:.1f} - {get_rsi_state(rsi_5)}")
                st.write(f"**9-Day:** {rsi_9:.1f} - {get_rsi_state(rsi_9)}")
                st.write(f"**14-Day:** {rsi_14:.1f} - {get_rsi_state(rsi_14)}")
                
            with v3:
                # --- COLOR SWAP: PUT RED ---
                st.caption("🔴 Put Defense (Floors)")
                st.write(f"**Wall 1:** {sup1}")
                st.write(f"**Wall 2:** {sup2}")
                
            with v4:
                # --- COLOR SWAP: CALL GREEN ---
                st.caption("🟢 Call Defense (Ceilings)")
                st.write(f"**Wall 1:** {res1}")
                st.write(f"**Wall 2:** {res2}")

            fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price")])
            
            ema_8_series = hist['Close'].ewm(span=8, adjust=False).mean()
            fig.add_trace(go.Scatter(x=hist.index, y=ema_8_series, line=dict(color='#ff9900', width=1.5, dash='dot'), name="8-Day EMA (Trend)"))
            
            # --- SWAPPED PLOTLY LINE COLORS ---
            fig.add_hline(y=call_strike, line_width=2, line_color="green", annotation_text="Call Strike")
            fig.add_hline(y=put_strike, line_width=2, line_color="red", annotation_text="Put Strike")
            fig.add_hline(y=call_trip, line_width=1, line_dash="dash", line_color="yellow", annotation_text="Call Alert")
            fig.add_hline(y=put_trip, line_width=1, line_dash="dash", line_color="yellow", annotation_text="Put Alert")
            
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading {symbol}. Options data may be missing.")
