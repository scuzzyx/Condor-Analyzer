import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- CONFIG & THEME ---
st.set_page_config(page_title="Money Machine Pro V3.2.5", layout="wide", initial_sidebar_state="expanded")
st.title("⚙️ Money Machine Pro V3.2.5 (Full Restoration)")

# --- 📖 NEW: INSTRUCTIONS & RISK LEGEND ---
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
Z_SCORES = {
    "70%": 1.04, "75%": 1.15, "80%": 1.28, 
    "85%": 1.44, "90%": 1.645, "95%": 1.96
}

# --- WEB-SAFE URL MEMORY HELPER ---
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
        
        # FIX: Force instant refresh of active selections
        current_active = st.session_state['active_selections'].copy()
        if ticker not in current_active:
            current_active.append(ticker)
            st.session_state['active_selections'] = current_active
            
    st.session_state['ticker_input'] = ""

st.sidebar.text_input("➕ Add Custom Ticker:", key="ticker_input", on_change=add_custom_ticker)

selected_tickers = st.sidebar.multiselect(
    "Active Bench:", 
    options=st.session_state['custom_bench'], 
    key="active_selections"
)

# 💾 SAVE NOTE ADDED HERE
st.sidebar.caption("Note: To save your bench loadout, use this custom link.")
if st.sidebar.button("🔗 Generate Custom Link"):
    bench_string = ",".join(st.session_state['active_selections'])
    st.query_params["bench"] = bench_string
    st.sidebar.success("URL updated! Bookmark it now.")

st.sidebar.markdown("---")

available_fridays = get_friday_expirations()
if available_fridays:
    selected_date_str = st.sidebar.selectbox("Expiration Date:", options=available_fridays)
    selected_date = datetime.strptime(selected_date_str, '%Y-%m-%d')
    dte = (selected_date - datetime.now()).days
else:
    st.sidebar.error("Error loading dates.")
    dte = 14

prob_target = st.sidebar.selectbox("Success Target:", options=list(Z_SCORES.keys()), index=4)
z_score = Z_SCORES[prob_target]

# --- RANGE-BOUND RADAR (MANUAL SCAN) ---
st.sidebar.markdown("---")
st.sidebar.subheader("📡 Range-Bound Radar")

LIQUID_50 = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AMD', 'PLTR', 'NFLX',
    'BA', 'DIS', 'BABA', 'UBER', 'COIN', 'HOOD', 'INTC', 'MU', 'AVGO', 'TSM',
    'JPM', 'BAC', 'C', 'V', 'MA', 'PYPL', 'SQ', 'WMT', 'TGT', 'COST',
    'HD', 'SBUX', 'NKE', 'MCD', 'XOM', 'CVX', 'CAT', 'GE', 'JNJ', 'PFE',
    'UNH', 'LLY', 'CMCSA', 'VZ', 'T', 'QCOM', 'CRM', 'SNOW', 'SHOP', 'SPOT'
]

SP_100 = ['AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'AMD', 'AMGN', 'AMT', 'AMZN', 'AXP', 'BA', 'BAC', 'BK', 'BKNG', 'BLK', 'BMY', 'C', 'CAT', 'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS', 'CVX', 'DHR', 'DIS', 'DOW', 'DUK', 'EMR', 'EXC', 'F', 'FDX', 'GD', 'GE', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KHC', 'KO', 'LIN', 'LLY', 'LMT', 'LOW', 'MA', 'MCD', 'MDLZ', 'MDT', 'MET', 'META', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'OXY', 'PEP', 'PFE', 'PG', 'PM', 'PYPL', 'QCOM', 'RTX', 'SBUX', 'SCHW', 'SO', 'SPG', 'T', 'TGT', 'TMO', 'TSLA', 'TXN', 'UNH', 'UNP', 'UPS', 'USB', 'V', 'VZ', 'WBA', 'WFC', 'WMT', 'XOM']

UNIVERSES = {
    "Top 50 Liquid Options (Fast & Safe)": LIQUID_50,
    "S&P 100 (Broader Scan)": SP_100
}

scan_choice = st.sidebar.radio("Select Scan Universe:", list(UNIVERSES.keys()))
scan_tolerance = st.sidebar.slider("Consolidation Tolerance (%)", min_value=3, max_value=15, value=8) / 100.0

if st.sidebar.button("Run Radar Scan Now"):
    with st.sidebar.status(f"Scanning {scan_choice}..."):
        targets = run_radar_scan(UNIVERSES[scan_choice], scan_tolerance)
        if targets:
            st.sidebar.success(f"🎯 Targets Found: {', '.join(targets)}")
        else:
            st.sidebar.warning("No setups found.")

# --- PORTFOLIO CORRELATION ENGINE ---
st.markdown("---")
if len(selected_tickers) > 1:
    with st.expander("🧩 Portfolio Risk: 30-Day Correlation Matrix", expanded=False):
        try:
            bench_data = yf.download(selected_tickers, period="3mo", progress=False)['Close']
            returns = bench_data.pct_change().tail(30)
            corr_matrix = returns.corr()
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))
        except Exception as e:
            st.write("Correlation data currently unavailable.")
st.markdown("---")

# --- MAIN ENGINE ---
for symbol in selected_tickers:
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="3mo")
        
        if len(hist) < 20:
            st.warning(f"Not enough data for {symbol}")
            continue
            
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        change_dlr = current_price - prev_close
        change_pct = (change_dlr / prev_close) * 100
        
        ma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        support_3mo = hist['Close'].min()
        rsi_14 = calculate_rsi(hist['Close']).iloc[-1]
        adx_14 = calculate_adx(hist)
        gap_risk = calculate_gap_risk(hist)
        
        volatility_dte = np.std(hist['Close'].pct_change().dropna()) * np.sqrt(dte if dte > 0 else 1)
        expected_move = current_price * (volatility_dte * z_score)
        
        put_strike = round(current_price - expected_move)
        call_strike = round(current_price + expected_move)
        put_trip = round(put_strike * 1.05, 2)
        call_trip = round(call_strike * 0.95, 2)
        
        # IV PATH FIXED
        atm_iv = "N/A"
        try:
            chain = t.option_chain(selected_date_str)
            idx = (chain.calls['strike'] - current_price).abs().idxmin()
            atm_iv = f"{chain.calls.loc[idx, 'impliedVolatility'] * 100:.1f}%"
        except: pass

        # EARNINGS PATH FIXED
        earnings_date = "Not yet scheduled"
        earnings_veto = False
        try:
            cal = t.calendar
            if not cal.empty:
                e_date = cal.iloc[0,0] if isinstance(cal, pd.DataFrame) else cal.get('Earnings Date')[0]
                earnings_date = e_date.strftime('%Y-%m-%d')
                if datetime.now() < e_date < selected_date:
                    earnings_veto = True
        except: pass
            
        if earnings_veto:
            risk, color = "⛔ DO NOT TRADE (EARNINGS VETO)", "red"
        elif rsi_14 < 35 or current_price <= support_3mo:
            risk, color = "🔴 HIGH RISK (Falling Knife)", "red"
        elif gap_risk > 1.5:
            risk, color = f"🟠 GAP RISK ({gap_risk:.2f}%)", "orange"
        elif adx_14 > 25:
            risk, color = f"🟡 TRENDING (ADX {adx_14:.1f})", "orange"
        elif current_price > support_3mo and current_price > ma_20:
            risk, color = "🟢 LOW RISK (Neutral Chop)", "green"
        else:
            risk, color = "🟡 MED RISK (Stalling)", "orange"

        with st.expander(f"{symbol}  |  Price: ${current_price:.2f}  |  Risk: {risk}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Change", f"${current_price:.2f}", f"{change_dlr:.2f} ({change_pct:.2f}%)")
            col2.metric("Put Strategy", f"Strike: ${put_strike}", f"Trip Wire: ${put_trip}", delta_color="off")
            col3.metric("Call Strategy", f"Strike: ${call_strike}", f"Trip Wire: ${call_trip}", delta_color="off")
            col4.metric("Market Data", f"IV: {atm_iv}", f"Earnings: {earnings_date}", delta_color="off")

            fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price")])
            fig.add_hline(y=call_strike, line_width=2, line_color="red", annotation_text="Call Strike")
            fig.add_hline(y=put_strike, line_width=2, line_color="green", annotation_text="Put Strike")
            fig.add_hline(y=call_trip, line_width=1, line_dash="dash", line_color="yellow", annotation_text="Call Alert")
            fig.add_hline(y=put_trip, line_width=1, line_dash="dash", line_color="yellow", annotation_text="Put Alert")
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading {symbol}")
