import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="Money Machine Pro V3.2.1", layout="wide")
st.title("⚙️ Money Machine Pro V3.2.1")

with st.expander("📖 How to Use & Risk Legend"):
    st.markdown("""### 🧠 Workflow: 1. Build Bench 2. Check Correlation 3. Review Setups. 
    🚦 **Risk:** 🟢 Low 🟡 Med 🟡 Trending 🟠 Gap 🔴 High ⛔ Earnings Veto.""")

Z_SCORES = {"70%": 1.04, "75%": 1.15, "80%": 1.28, "85%": 1.44, "90%": 1.645, "95%": 1.96}

def load_url_bench():
    if "bench" in st.query_params: return st.query_params["bench"].split(",")
    return ["AMZN", "AAPL", "MSFT", "META", "GOOGL", "NVDA", "AMD", "PLTR", "TSLA", "NFLX"]

def calc_rsi(d, p=14):
    diff = d.diff(); g = diff.where(diff > 0, 0).rolling(p).mean(); l = -diff.where(diff < 0, 0).rolling(p).mean()
    return 100 - (100 / (1 + (g/l)))

def calc_adx(h, p=14):
    try:
        hi, lo, cl = h['High'], h['Low'], h['Close']
        tr = pd.concat([hi-lo, abs(hi-cl.shift(1)), abs(lo-cl.shift(1))], axis=1).max(axis=1).ewm(alpha=1/p).mean()
        pdn = (hi.diff().where((hi.diff() > lo.diff()) & (hi.diff() > 0), 0)).ewm(alpha=1/p).mean()
        mdn = (lo.diff().where((lo.diff() > hi.diff()) & (lo.diff() > 0), 0)).ewm(alpha=1/p).mean()
        dx = (abs((100*pdn/tr) - (100*mdn/tr)) / abs((100*pdn/tr) + (100*mdn/tr))) * 100
        return dx.ewm(alpha=1/p).mean().iloc[-1]
    except: return 20

def get_exp():
    try: return [d for d in yf.Ticker("SPY").options if datetime.strptime(d, '%Y-%m-%d').weekday() == 4][:10]
    except: return []

# --- SIDEBAR ---
st.sidebar.header("🛠️ Controls")
url_b = load_url_bench()
if 'custom_bench' not in st.session_state: st.session_state['custom_bench'] = list(set(url_b + ["SPY", "QQQ"]))
if 'active_selections' not in st.session_state: st.session_state['active_selections'] = url_b

def add_t():
    t = st.session_state['t_in'].upper().strip()
    if t:
        if t not in st.session_state['custom_bench']: st.session_state['custom_bench'].append(t)
        cur = st.session_state['active_selections'].copy()
        if t not in cur: cur.append(t); st.session_state['active_selections'] = cur
    st.session_state['t_in'] = ""

st.sidebar.text_input("➕ Add Ticker:", key="t_in", on_change=add_t)
sel_t = st.sidebar.multiselect("Active Bench:", options=st.session_state['custom_bench'], key="active_selections")
st.sidebar.caption("Note: Use custom link to save loadout.")
if st.sidebar.button("🔗 Generate Custom Link"):
    st.query_params["bench"] = ",".join(st.session_state['active_selections'])
    st.sidebar.success("URL updated! Bookmark it.")

st.sidebar.markdown("---")
f_dates = get_exp()
if f_dates:
    exp_s = st.sidebar.selectbox("Expiration:", options=f_dates)
    dte = (datetime.strptime(exp_s, '%Y-%m-%d') - datetime.now()).days
else: dte, exp_s = 14, None
z_v = Z_SCORES[st.sidebar.selectbox("Target:", options=list(Z_SCORES.keys()), index=4)]

# --- RADAR ---
L50 = ['AAPL','MSFT','NVDA','AMZN','META','GOOGL','TSLA','AMD','PLTR','NFLX','BA','DIS','BABA','UBER','COIN','HOOD','INTC','MU','AVGO','TSM','JPM','BAC','C','V','MA','PYPL','SQ','WMT','TGT','COST','HD','SBUX','NKE','MCD','XOM','CVX','CAT','GE','JNJ','PFE','UNH','LLY','CMCSA','VZ','T','QCOM','CRM','SNOW','SHOP','SPOT']
if st.sidebar.button("Run Radar"):
    d = yf.download(L50, period="1mo", group_by='ticker', progress=False)
    hits = [s for s in L50 if (d[s]['Close'].max()-d[s]['Close'].min())/d[s]['Close'].iloc[-1] < 0.08]
    if hits: st.sidebar.success(f"🎯 Found: {', '.join(hits)}")

# --- ENGINE ---
if len(sel_t) > 1:
    with st.expander("🧩 Correlation Matrix"):
        try: st.dataframe(yf.download(sel_t, period="3mo", progress=False)['Close'].pct_change().tail(30).corr().style.background_gradient(cmap='coolwarm').format("{:.2f}"))
        except: st.write("Data error.")

for s in sel_t:
    try:
        t_obj = yf.Ticker(s); h = t_obj.history(period="3mo"); c = h['Close'].iloc[-1]; p = h['Close'].iloc[-2]
        rsi_v = calc_rsi(h['Close']).iloc[-1]; adx_v = calc_adx(h)
        vol = np.std(h['Close'].pct_change()) * np.sqrt(dte if dte > 0 else 1)
        m = c * (vol * z_v); ps, cs = round(c - m), round(c + m); pt, ct = round(ps*1.05, 2), round(cs*0.95, 2)
        iv, ed, veto = "N/A", "Not scheduled", False
        try:
            if exp_s:
                clls = t_obj.option_chain(exp_s).calls
                iv = f"{clls.iloc[(clls['strike']-c).abs().argsort()[:1]]['impliedVolatility'].values[0]*100:.1f}%"
        except: pass
        try:
            cal = t_obj.calendar
            if cal is not None and not cal.empty:
                edate = cal.iloc[0,0] if isinstance(cal, pd.DataFrame) else cal.get('Earnings Date', [None])[0]
                ed = edate.strftime('%Y-%m-%d'); veto = datetime.now() < edate < (datetime.now() + timedelta(days=dte))
        except: pass
        if veto: r, clr = "⛔ DO NOT TRADE (EARNINGS VETO)", "red"
        elif rsi_v < 35 or c <= h['Close'].min(): r, clr = "🔴 HIGH RISK (Falling Knife)", "red"
        elif adx_v > 25: r, clr = f"🟡 TRENDING (ADX {adx_v:.1f})", "orange"
        elif c > h['Close'].min() and c > h['Close'].rolling(20).mean().iloc[-1]: r, clr = "🟢 LOW RISK (Neutral)", "green"
        else: r, clr = "🟡 MED RISK", "orange"

        with st.expander(f"{s} | ${c:.2f} | {r}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Change", f"${c:.2f}", f"{(c-p):.2f}")
            c2.metric("Put Strategy", f"Strike: ${ps}", f"Trip Wire: ${pt}", delta_color="off")
            c3.metric("Call Strategy", f"Strike: ${cs}", f"Trip Wire: ${ct}", delta_color="off")
            c4.metric("Data", f"IV: {iv}", f"Earnings: {ed}", delta_color="off")
            fig = go.Figure(data=[go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'])])
            fig.add_hline(y=cs, line_color="red"); fig.add_hline(y=ps, line_color="green")
            fig.add_hline(y=ct, line_dash="dash", line_color="yellow"); fig.add_hline(y=pt, line_dash="dash", line_color="yellow")
            fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
    except Exception as ex: st.error(f"Error {s}: {ex}")