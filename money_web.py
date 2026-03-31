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

# --- CACHED DATA FETCHERS TO PREVENT RATE LIMITS ---
@st.cache_data(ttl=900) 
def get_cached_ticker_info(symbol):
    try:
        t = yf.Ticker(symbol)
        # Returns a tuple of (info_dict, news_list, calendar_data)
        return t.info, t.news, t.calendar
    except:
        return {}, [], {}

@st.cache_data(ttl=600) 
def get_cached_options_data(symbol, date_str):
    try:
        t = yf.Ticker(symbol)
        chain = t.option_chain(date_str)
        return chain.calls, chain.puts
    except:
        return pd.DataFrame(), pd.DataFrame()

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
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        return dx.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
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
        s1 = f"${lower[-1]:.2f}" if len(lower) > 0 else "Freefall (None)"
        return f"${poc:.2f}", s1, r1
    except:
        return "N/A", "N/A", "N/A"

@st.cache_data(ttl=3600)  
def get_friday_expirations():
    try:
        spy = yf.Ticker("SPY")
        dates = spy.options
        fridays = [d for d in dates if datetime.strptime(d, '%Y-%m-%d').weekday() == 4]
        return fridays[:10]
    except:
        return [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(14, 60) if (datetime.now() + timedelta(days=i)).weekday() == 4]

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

# --- INDICATOR REFERENCE GLOSSARY ---
st.markdown("---")
with st.expander("📖 Terminal Indicator Glossary (Quick Reference)", expanded=False):
    st.subheader("🚦 Title Risk & Veto Signals")
    st.write("- **⚠️ [EARNINGS SOON]:** Earnings report occurs before expiration.")
    st.write("- **⚠️ [EX-DIVIDEND DANGER]:** Ex-Div date occurs before expiration. Assignment risk.")
    st.write("- **🔴 *FALLING KNIFE*:** Price below 8-EMA with bearish RSI momentum.")
    st.write("- **🟢 *FLOOR CONFIRMED*:** 8-EMA Reclaimed after oversold conditions.")
    
    g1, g2, g3 = st.columns(3)
    with g1:
        st.subheader("🛡️ Trend & Momentum")
        st.write("**8-Day EMA:** Algorithmic Trend line (Orange dotted).")
        st.write("**ADX:** >25 = Strong Trend. <25 = Chop.")
    with g2:
        st.subheader("🎯 Structure & Math")
        st.write("**POC:** Point of Control. High volume magnet.")
        st.write("**Support/Resistance:** Structural walls where flow reverses.")
    with g3:
        st.subheader("⚖️ Risk Underwriting")
        st.write("**Max Pain:** Strike where sellers lose the least. Price magnet.")
        st.write("**P/C OI Ratio:** >1.2 Bearish flow, <0.8 Bullish flow.")

st.markdown("---")

# --- MAIN ENGINE ---
for symbol in selected_tickers:
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="3mo")
        if len(hist) < 20: continue
            
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        change_dlr, change_pct = current_price - prev_close, ((current_price - prev_close) / prev_close) * 100
        change_color = "#ff4b4b" if change_dlr < 0 else "#09ab3b"
        
        # --- PULL CACHED INFO & NEWS ---
        ticker_info, ticker_news, ticker_calendar = get_cached_ticker_info(symbol)
        
        volatility_dte = np.std(hist['Close'].pct_change().dropna()) * np.sqrt(dte if dte > 0 else 1)
        expected_move = current_price * (volatility_dte * z_score)
        put_strike, call_strike = round(current_price - expected_move), round(current_price + expected_move)
        put_trip, call_trip = round(put_strike * 1.05, 2), round(call_strike * 0.95, 2)
        
        # --- CACHED OPTIONS DATA ---
        atm_iv, max_pain, pc_ratio = "N/A", "N/A", "N/A"
        try:
            calls, puts = get_cached_options_data(symbol, selected_date_str)
            if not calls.empty:
                closest_idx = (calls['strike'] - current_price).abs().idxmin()
                atm_iv = f"{calls.loc[closest_idx, 'impliedVolatility'] * 100:.1f}%"
            if not calls.empty and not puts.empty:
                tot_put_oi, tot_call_oi = puts['openInterest'].sum(), calls['openInterest'].sum()
                if tot_call_oi > 0: pc_ratio = f"{tot_put_oi / tot_call_oi:.2f}"
                all_strikes = sorted(list(set(calls['strike'].tolist() + puts['strike'].tolist())))
                mp_val, mp_strike = float('inf'), "N/A"
                for s in all_strikes:
                    c_l = calls[calls['strike'] < s].apply(lambda x: (s-x['strike'])*x['openInterest'], axis=1).sum()
                    p_l = puts[puts['strike'] > s].apply(lambda x: (x['strike']-s)*x['openInterest'], axis=1).sum()
                    if (c_l + p_l) < mp_val: mp_val, mp_strike = (c_l + p_l), s
                if mp_strike != "N/A": max_pain = f"${mp_strike:.2f}"
        except: pass

        # --- DIVIDEND & EARNINGS ---
        ex_div_date, ex_div_veto = "None scheduled", False
        try:
            ex_ts = ticker_info.get('exDividendDate')
            if ex_ts:
                ex_dt = datetime.fromtimestamp(ex_ts)
                ex_div_date = ex_dt.strftime('%Y-%m-%d')
                if datetime.now() < ex_dt < selected_date: ex_div_veto = True
        except: pass

        earnings_date, earnings_veto = "Not scheduled", False
        try:
            e_date = None
            if isinstance(ticker_calendar, dict) and 'Earnings Date' in ticker_calendar:
                e_date = pd.to_datetime(ticker_calendar['Earnings Date'][0])
            elif isinstance(ticker_calendar, pd.DataFrame) and 'Earnings Date' in ticker_calendar.index:
                e_date = pd.to_datetime(ticker_calendar.loc['Earnings Date'].iloc[0])
            if e_date and pd.notnull(e_date):
                earnings_date = e_date.strftime('%Y-%m-%d')
                if datetime.now() < e_date < selected_date: earnings_veto = True
        except: pass
            
        ema_8 = hist['Close'].ewm(span=8, adjust=False).mean().iloc[-1]
        rsi_14 = calculate_rsi(hist['Close'], periods=14).iloc[-1]
        poc, sup1, res1 = calculate_volume_nodes(hist, current_price)
        
        risk = "🟢 NEUTRAL CHOP"
        if current_price < ema_8 and rsi_14 < 45: risk = "🔴 FALLING KNIFE"
        if earnings_veto: risk += " [EARNINGS SOON]"
        if ex_div_veto: risk += " ⚠️[EX-DIVIDEND DANGER]"

        with st.expander(f"{symbol} | Price: ${current_price:.2f} | Risk: {risk}", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            def custom_metric_box(l, v, s, vc="#FAFAFA", sc="#a6a6a6"):
                return f"<div style='line-height:1.4;margin-bottom:14px;'><span style='font-size:0.85rem;color:#a6a6a6;'>{l}</span><br><span style='font-size:1.8rem;font-weight:600;color:{vc};'>{v}</span><br><span style='font-size:0.9rem;color:{sc};'>{s}</span></div>"
            with c1: st.markdown(custom_metric_box("Change", f"${current_price:.2f}", f"{change_dlr:+.2f} ({change_pct:+.2f}%)", sc=change_color), unsafe_allow_html=True)
            with c2: st.markdown(custom_metric_box("Put Strategy", f"${put_strike}", f"Trip: ${put_trip}", sc="#ffcc00"), unsafe_allow_html=True)
            with c3: st.markdown(custom_metric_box("Call Strategy", f"${call_strike}", f"Trip: ${call_trip}", sc="#ffcc00"), unsafe_allow_html=True)
            with c4: st.markdown(custom_metric_box("Market Data", f"{atm_iv} IV", f"Earnings: {earnings_date}"), unsafe_allow_html=True)
            
            st.markdown("---")
            st.caption("🛡️ Risk Underwriting Data")
            u1, u2, u3 = st.columns(3)
            with u1: st.markdown(custom_metric_box("Max Pain", f"{max_pain}", "Friday Magnet"), unsafe_allow_html=True)
            with u2:
                pc_c = "#ff4b4b" if pc_ratio != "N/A" and float(pc_ratio) > 1.2 else "#09ab3b" if pc_ratio != "N/A" and float(pc_ratio) < 0.8 else "#a6a6a6"
                st.markdown(custom_metric_box("P/C OI Ratio", f"{pc_ratio}", "Institutional Flow", sc=pc_c), unsafe_allow_html=True)
            with u3: st.markdown(custom_metric_box("Ex-Dividend", f"{ex_div_date}", "Assignment Risk", sc="#ffcc00" if ex_div_veto else "#a6a6a6"), unsafe_allow_html=True)

            st.markdown("---")
            v1, v2, v3, v4 = st.columns(4)
            with v1: st.write(f"**POC:** {poc}")
            with v2: st.write(f"**RSI 14D:** {rsi_14:.1f}")
            with v3: st.write(f"**Support Wall:** {sup1}")
            with v4: st.write(f"**Resistance Wall:** {res1}")

            fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price")])
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].ewm(span=8, adjust=False).mean(), line=dict(color='#ff9900', dash='dot'), name="8-EMA"))
            fig.add_hline(y=call_strike, line_width=2, line_color="green", annotation_text="Call Strike")
            fig.add_hline(y=put_strike, line_width=2, line_color="red", annotation_text="Put Strike")
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            if ticker_news:
                st.markdown("---")
                st.caption("📰 Recent Headlines")
                for item in ticker_news:
                    if isinstance(item, dict):
                        # Nested get for the new Yahoo structure
                        content = item.get('content', {})
                        title = content.get('title', item.get('title', 'Headline Unavailable'))
                        link = content.get('clickThroughUrl', {}).get('url', item.get('link', '#'))
                        st.markdown(f"- **[{title}]({link})**")
    except Exception as e:
        st.error(f"Error loading {symbol}: {str(e)}")
