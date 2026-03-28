import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- CONFIG ---
st.set_page_config(
    page_title="Money Machine Pro V3.2.3", 
    layout="wide", 
    initial_sidebar_state="expanded"
)
st.title("⚙️ Money Machine Pro V3.2.3")

# --- DOCUMENTATION ---
with st.expander("📖 How to Use This Engine & Risk Legend", expanded=False):
    st.markdown("""
    ### 🧠 The Workflow
    1. **Build Your Bench:** Add tickers in the sidebar.
    2. **Check Correlation:** Avoid trading high-correlation pairs (0.85+).
    3. **Review Setups:** Check risk grades, IV, and Earnings.
    """)

# --- PROBABILITY Z-SCORES ---
Z_SCORES = {
    "70%": 1.04, "75%": 1.15, "80%": 1.28, 
    "85%": 1.44, "90%": 1.645, "95%": 1.96
}

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

@st.cache_data(ttl=3600)
def get_friday_expirations():
    try:
        spy = yf.Ticker("SPY")
        dates = spy.options
        return [d for d in dates if datetime.strptime(d, '%Y-%m-%d').weekday() == 4][:10]
    except:
        return []

# --- SIDEBAR ---
st.sidebar.header("🛠️ Dashboard Controls")
url_bench = load_url_bench()

if 'custom_bench' not in st.session_state:
    st.session_state['custom_bench'] = list(set(url_bench + ["SPY", "QQQ"]))
if 'active_selections' not in st.session_state:
    st.session_state['active_selections'] = url_bench

def add_custom_ticker():
    t = st.session_state['ticker_input'].upper().strip()
    if t:
        if t not in st.session_state['custom_bench']:
            st.session_state['custom_bench'].append(t)
        active = st.session_state['active_selections'].copy()
        if t not in active:
            active.append(t)
            st.session_state['active_selections'] = active
    st.session_state['ticker_input'] = ""

st.sidebar.text_input("➕ Add Custom Ticker:", key="ticker_input", on_change=add_custom_ticker)
selected_tickers = st.sidebar.multiselect("Active Bench:", options=st.session_state['custom_bench'], key="active_selections")

if st.sidebar.button("🔗 Generate Custom Link"):
    st.query_params["bench"] = ",".join(st.session_state['active_selections'])
    st.sidebar.success("URL updated! Bookmark now.")

st.sidebar.markdown("---")
fridays = get_friday_expirations()
if fridays:
    exp_str = st.sidebar.selectbox("Expiration Date:", options=fridays)
    dte = (datetime.strptime(exp_str, '%Y-%m-%d') - datetime.now()).days
else:
    dte, exp_str = 14, None

z_val = Z_SCORES[st.sidebar.selectbox("Success Target:", options=list(Z_SCORES.keys()), index=4)]

# --- RADAR ---
st.sidebar.markdown("---")
st.sidebar.subheader("📡 Range-Bound Radar")
if st.sidebar.button("Run Radar Scan Now"):
    radar_list = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AMD', 'PLTR', 'NFLX']
    with st.sidebar.status("Scanning..."):
        hits = []
        for s in radar_list:
            try:
                p_data = yf.download(s, period="1mo", progress=False)['Close']
                if (p_data.max() - p_data.min()) / p_data.iloc[-1] < 0.08:
                    hits.append(s)
            except: continue
        if hits: st.sidebar.success(f"🎯 Found: {', '.join(hits)}")

# --- CORRELATION MATRIX (ROBUST LOOP) ---
st.markdown("---")
if len(selected_tickers) > 1:
    with st.expander("🧩 Portfolio Risk: 30-Day Correlation Matrix", expanded=False):
        try:
            series_list = []
            for s in selected_tickers:
                try:
                    px = yf.download(s, period="3mo", progress=False)['Close']
                    if not px.empty:
                        # Clean to handle single ticker Series vs Dataframe
                        s_clean = px.pct_change().tail(30)
                        s_clean.name = s
                        series_list.append(s_clean)
                except: continue
            
            if series_list:
                corr_df = pd.concat(series_list, axis=1).corr()
                st.dataframe(corr_df.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))
            else:
                st.write("Could not retrieve ticker data for matrix.")
        except:
            st.write("Error processing correlation matrix.")

# --- MAIN ENGINE ---
for sym in selected_tickers:
    try:
        t_obj = yf.Ticker(sym)
        # Fetching basic history
        h = t_obj.history(period="3mo")
        if h.empty:
            st.error(f"No price data found for {sym}")
            continue
            
        curr = h['Close'].iloc[-1]
        prev = h['Close'].iloc[-2]
        
        rsi_v = calculate_rsi(h['Close']).iloc[-1]
        ma20 = h['Close'].rolling(20).mean().iloc[-1]
        sup = h['Close'].min()
        
        # Volatility Calc
        vol = np.std(h['Close'].pct_change()) * np.sqrt(dte if dte > 0 else 1)
        move = curr * (vol * z_val)
        ps, cs = round(curr - move), round(curr + move)
        pt, ct = round(ps * 1.05, 2), round(cs * 0.95, 2)

        # Market Data Fetching (Individualized)
        iv_val = "N/A"
        ed_date = "Not scheduled"
        veto = False
        
        try:
            if exp_str:
                chain = t_obj.option_chain(exp_str)
                calls = chain.calls
                # Target nearest strike for IV
                target_idx = (calls['strike'] - curr).abs().idxmin()
                iv_raw = calls.loc[target_idx, 'impliedVolatility']
                iv_val = f"{iv_raw * 100:.1f}%"
        except: pass
        
        try:
            cal = t_obj.calendar
            if cal is not None and not cal.empty:
                # Handle DataFrame vs Dictionary return
                e_date = cal.iloc[0,0] if isinstance(cal, pd.DataFrame) else cal.get('Earnings Date')[0]
                ed_date = e_date.strftime('%Y-%m-%d')
                if datetime.now() < e_date < (datetime.now() + timedelta(days=dte)):
                    veto = True
        except: pass

        # Risk Decision
        if veto: r, clr = "⛔ DO NOT TRADE (EARNINGS VETO)", "red"
        elif rsi_v < 35 or curr <= sup: r, clr = "🔴 HIGH RISK (Falling Knife)", "red"
        elif curr > ma20 and curr > sup: r, clr = "🟢 LOW RISK (Neutral)", "green"
        else: r, clr = "🟡 MED RISK (Stalling)", "orange"

        with st.expander(f"{sym} | Price: ${curr:.2f} | Risk: {r}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Today's Change", f"${curr:.2f}", f"{(curr-prev):.2f}")
            c2.metric("Put Strategy", f"Strike: ${ps}", f"Trip: ${pt}", delta_color="off")
            c3.metric("Call Strategy", f"Strike: ${cs}", f"Trip: ${ct}", delta_color="off")
            c4.metric("Market Data", f"IV: {iv_val}", f"Earnings: {ed_date}", delta_color="off")

            fig = go.Figure(data=[go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'])])
            fig.add_hline(y=cs, line_color="red", annotation_text="Call")
            fig.add_hline(y=ps, line_color="green", annotation_text="Put")
            fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Unexpected error for {sym}: {e}")
