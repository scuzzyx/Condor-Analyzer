import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- CONFIG ---
st.set_page_config(
    page_title="Money Machine Pro V3.2.2", 
    layout="wide", 
    initial_sidebar_state="expanded"
)
st.title("⚙️ Money Machine Pro V3.2.2")

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
    return [
        "AMZN", "AAPL", "MSFT", "META", 
        "GOOGL", "NVDA", "AMD", "PLTR", 
        "TSLA", "NFLX"
    ]

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
        dates = yf.Ticker("SPY").options
        return [
            d for d in dates 
            if datetime.strptime(d, '%Y-%m-%d').weekday() == 4
        ][:10]
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

st.sidebar.text_input(
    "➕ Add Custom Ticker:", 
    key="ticker_input", 
    on_change=add_custom_ticker
)

selected_tickers = st.sidebar.multiselect(
    "Active Bench:", 
    options=st.session_state['custom_bench'], 
    key="active_selections"
)

st.sidebar.caption("Note: Use custom link to save.")
if st.sidebar.button("🔗 Generate Custom Link"):
    st.query_params["bench"] = ",".join(st.session_state['active_selections'])
    st.sidebar.success("URL updated!")

st.sidebar.markdown("---")
fridays = get_friday_expirations()
if fridays:
    exp_str = st.sidebar.selectbox("Expiration Date:", options=fridays)
    dte = (datetime.strptime(exp_str, '%Y-%m-%d') - datetime.now()).days
else:
    dte = 14
    exp_str = None

z_val = Z_SCORES[st.sidebar.selectbox("Success Target:", options=list(Z_SCORES.keys()), index=4)]

# --- RADAR ---
st.sidebar.markdown("---")
st.sidebar.subheader("📡 Range-Bound Radar")
LIQUID_LIST = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AMD', 'PLTR', 'NFLX']

if st.sidebar.button("Run Radar Scan Now"):
    with st.sidebar.status("Scanning..."):
        hits = []
        for s in LIQUID_LIST:
            try:
                px = yf.download(s, period="1mo", progress=False)['Close']
                if (px.max() - px.min()) / px.iloc[-1] < 0.08:
                    hits.append(s)
            except: continue
        if hits: st.sidebar.success(f"🎯 Found: {', '.join(hits)}")

# --- CORRELATION MATRIX FIX ---
st.markdown("---")
if len(selected_tickers) > 1:
    with st.expander("🧩 Correlation Matrix", expanded=False):
        try:
            # Individual fetch to prevent MultiIndex errors
            df_list = []
            for s in selected_tickers:
                d = yf.download(s, period="3mo", progress=False)['Close']
                d.name = s
                df_list.append(d)
            full_df = pd.concat(df_list, axis=1).pct_change().tail(30)
            st.dataframe(full_df.corr().style.background_gradient(cmap='coolwarm').format("{:.2f}"))
        except:
            st.write("Data currently unavailable.")

# --- MAIN ENGINE ---
for sym in selected_tickers:
    try:
        t_obj = yf.Ticker(sym)
        h = t_obj.history(period="3mo")
        curr = h['Close'].iloc[-1]
        prev = h['Close'].iloc[-2]
        
        rsi_v = calculate_rsi(h['Close']).iloc[-1]
        vol = np.std(h['Close'].pct_change()) * np.sqrt(dte if dte > 0 else 1)
        move = curr * (vol * z_val)
        ps, cs = round(curr - move), round(curr + move)
        pt, ct = round(ps * 1.05, 2), round(cs * 0.95, 2)

        iv_val = "N/A"
        ed_date = "Not scheduled"
        veto = False
        
        # Robust IV Fetch
        try:
            if exp_str:
                opt = t_obj.option_chain(exp_str)
                # Filter for nearest strike
                c_opt = opt.calls
                idx = (c_opt['strike'] - curr).abs().idxmin()
                iv_raw = c_opt.loc[idx, 'impliedVolatility']
                iv_val = f"{iv_raw * 100:.1f}%"
        except: pass
        
        # Robust Earnings Fetch
        try:
            cal = t_obj.calendar
            if cal is not None and not cal.empty:
                # Handle both dict and DF return types
                if isinstance(cal, pd.DataFrame):
                    e_date = cal.iloc[0,0]
                else:
                    e_date = cal.get('Earnings Date')[0]
                ed_date = e_date.strftime('%Y-%m-%d')
                if datetime.now() < e_date < (datetime.now() + timedelta(days=dte)):
                    veto = True
        except: pass

        # Risk Logic
        if veto: r, clr = "⛔ DO NOT TRADE (EARNINGS VETO)", "red"
        elif rsi_v < 35 or curr <= h['Close'].min(): r, clr = "🔴 HIGH RISK", "red"
        elif curr > h['Close'].rolling(20).mean().iloc[-1]: r, clr = "🟢 LOW RISK", "green"
        else: r, clr = "🟡 MED RISK", "orange"

        with st.expander(f"{sym} | Price: ${curr:.2f} | Risk: {r}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Change", f"${curr:.2f}", f"{(curr-prev):.2f}")
            c2.metric("Put Strategy", f"Strike: ${ps}", f"Trip: ${pt}", delta_color="off")
            c3.metric("Call Strategy", f"Strike: ${cs}", f"Trip: ${ct}", delta_color="off")
            c4.metric("Data", f"IV: {iv_val}", f"Earnings: {ed_date}", delta_color="off")

            fig = go.Figure(data=[go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'])])
            fig.add_hline(y=cs, line_color="red", annotation_text="Call")
            fig.add_hline(y=ps, line_color="green", annotation_text="Put")
            fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
    except: st.error(f"Error loading {sym}")
