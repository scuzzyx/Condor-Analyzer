import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- CONFIG & THEME ---
st.set_page_config(page_title="Money Machine Pro V3", layout="wide", initial_sidebar_state="expanded")
st.title("⚙️ Money Machine Pro V3 (Risk Engine Active)")

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
    """Calculates the Average Directional Index (ADX) to measure trend strength."""
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
        return 20 # Fallback to neutral if math fails

def calculate_gap_risk(hist):
    """Calculates the average absolute overnight gap percentage over the last 30 days."""
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

@st.cache_data(ttl=86400) 
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        df = pd.read_html(url)[0]
        return df['Symbol'].str.replace('.', '-').tolist()
    except:
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ', 'JPM'] 

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

# --- THE ONE CHANGE: INSTANT AUTO-ADD CALLBACK ---
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
# --------------------------------------------------

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

prob_target = st.sidebar.selectbox("Probability of Success Target:", options=list(Z_SCORES.keys()), index=4)
z_score = Z_SCORES[prob_target]

# --- RANGE-BOUND RADAR (MANUAL SCAN) ---
st.sidebar.markdown("---")
st.sidebar.subheader("📡 Range-Bound Radar")

UNIVERSES = {
    "Nasdaq Proxy (Fast)": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AVGO', 'PEP', 'COST', 'CSCO', 'TMUS', 'ADBE', 'TXN', 'NFLX', 'QCOM', 'AMD', 'INTC', 'AMAT', 'ISRG'],
    "S&P 500 (Deep Scan)": get_sp500_tickers() 
}

scan_choice = st.sidebar.radio("Select Scan Universe:", list(UNIVERSES.keys()))
scan_tolerance = st.sidebar.slider("Consolidation Tolerance (%)", min_value=3, max_value=15, value=8) / 100.0

if st.sidebar.button("Run Radar Scan Now"):
    with st.sidebar.status(f"Scanning {scan_choice} at {int(scan_tolerance*100)}% tolerance..."):
        targets = run_radar_scan(UNIVERSES[scan_choice], scan_tolerance)
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
        
        # New Quantitative Indicators
        ma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        support_3mo = hist['Close'].min()
        rsi_14 = calculate_rsi(hist['Close']).iloc[-1]
        adx_14 = calculate_adx(hist)
        gap_risk = calculate_gap_risk(hist)
        
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
            if not cal.empty and 'Earnings Date' in cal.index:
                e_date = cal.loc['Earnings Date'][0].to_pydatetime()
                earnings_date = e_date.strftime('%Y-%m-%d')
                if datetime.now() < e_date < selected_date:
                    earnings_veto = True
        except:
            pass
            
        # TIER 1 RISK OVERRIDES
        if earnings_veto:
            risk, color = "⛔ DO NOT TRADE (EARNINGS VETO)", "red"
        elif rsi_14 < 35 or current_price <= support_3mo:
            risk, color = "🔴 HIGH RISK (Structural Break / Falling Knife)", "red"
        elif gap_risk > 1.5:
            risk, color = f"🟠 GAP RISK (Avg Overnight Gap: {gap_risk:.2f}%)", "orange"
        elif adx_14 > 25:
            risk, color = f"🟡 TRENDING (ADX {adx_14:.1f} - Use Directional Spreads)", "orange"
        elif current_price > support_3mo and current_price > ma_20:
            risk, color = "🟢 LOW RISK (Neutral Chop)", "green"
        else:
            risk, color = "🟡 MED RISK (Stalling)", "orange"

        # --- UI DISPLAY ---
        with st.expander(f"{symbol}  |  Price: ${current_price:.2f}  |  Risk: {risk}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Today's Change", f"${current_price:.2f}", f"{change_dlr:.2f} ({change_pct:.2f}%)")
            col2.metric("Put Strategy", f"Strike: ${put_strike}", f"Trip Wire: ${put_trip}", delta_color="off")
            col3.metric("Call Strategy", f"Strike: ${call_strike}", f"Trip Wire: ${call_trip}", delta_color="off")
            col4.metric("Market Data", f"IV: {atm_iv}", f"Earnings: {earnings_date}", delta_color="off")

            # Plotly Chart
            fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price")])
            
            fig.add_hline(y=call_strike, line_width=2, line_color="red", annotation_text="Call Strike")
            fig.add_hline(y=put_strike, line_width=2, line_color="green", annotation_text="Put Strike")
            fig.add_hline(y=call_trip, line_width=1, line_dash="dash", line_color="yellow", annotation_text="Call Alert")
            fig.add_hline(y=put_trip, line_width=1, line_dash="dash", line_color="yellow", annotation_text="Put Alert")
            
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading {symbol}. It may be missing options data.")
