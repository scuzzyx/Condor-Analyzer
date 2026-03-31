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
st.sidebar.caption("Scan restricted to the Top 50 highest options liquidity stocks.")
LIQUID_50 = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AMD', 'PLTR', 'NFLX', 'BA', 'DIS', 'BABA', 'UBER', 'COIN', 'HOOD', 'INTC', 'MU', 'AVGO', 'TSM', 'JPM', 'BAC', 'C', 'V', 'MA', 'PYPL', 'SQ', 'WMT', 'TGT', 'COST', 'HD', 'SBUX', 'NKE', 'MCD', 'XOM', 'CVX', 'CAT', 'GE', 'JNJ', 'PFE', 'UNH', 'LLY', 'CMCSA', 'VZ', 'T', 'QCOM', 'CRM', 'SNOW', 'SHOP', 'SPOT']
scan_tol = st.sidebar.slider("Tolerance (%)", 3, 15, 8) / 100.0
if st.sidebar.button("Run Radar Scan Now"):
    targets = run_radar_scan(LIQUID_50, scan_tol)
    if targets: st.sidebar.success(f"🎯 Found: {', '.join(targets)}")
    else: st.sidebar.warning("No targets.")

# --- INDICATOR REFERENCE GLOSSARY ---
st.markdown("---")
with st.expander("📖 Terminal Indicator Glossary (Quick Reference)", expanded=False):
    st.subheader("🚦 Title Risk & Veto Signals")
    st.write("- **⚠️ [EARNINGS SOON]:** Earnings report occurs before expiration. Trade with caution.")
    st.write("- **⚠️ [EX-DIVIDEND DANGER]:** Ex-Div date occurs before expiration. High risk of early call assignment.")
    st.write("- **🔴 *FALLING KNIFE* (Bearish Momentum):** Price below 8-EMA. Consider Call Spreads only.")
    st.write("- **🟠 *GAP RISK* (Overnight Vol):** Historical tendency to jump >1.5% overnight.")
    st.write("- **🟡 *TRENDING* (High ADX):** ADX (>25). Stock is moving fast; pick a directional spread. Avoid Condors.")
    st.write("- **🟢 *FLOOR CONFIRMED* (Bullish Reversal):** 8-EMA Reclaimed. Consider Put Spreads only.")
    st.write("- **🟢 *NEUTRAL CHOP* (Condor Territory):** Ideal sideways environment for Iron Condors.")
    
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("🛡️ Trend & Momentum")
        st.write("**8-Day EMA:** The 'Algorithmic Trend' line. Orange dotted line on chart.")
        st.write("**RSI Stack:** Overbought (>70), Oversold (<30), Neutral (31-69).")
        st.write("**ADX:** Above 25 = Strong Trend. Below 25 = Drifting/Chop.")
    with g2:
        st.subheader("🎯 Structure & Math")
        st.write("**POC:** Highest volume price point in 90 days. Price magnet.")
        st.write("**🔴 Support Walls (Floors):** Structural support where buyers step in.")
        st.write("**🟢 Resistance Walls (Ceilings):** Structural resistance where sellers emerge.")
        st.write("**Z-Score:** Probability math used to set the strike safety margin.")

# --- PORTFOLIO CORRELATION ---
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
        
        ma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        ema_8 = hist['Close'].ewm(span=8, adjust=False).mean().iloc[-1]
        support_3mo = hist['Close'].min()
        
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
        
        # --- ROBUST IV, MAX PAIN & P/C RATIO ---
        atm_iv = "N/A"
        max_pain = "N/A"
        pc_ratio = "N/A"
        try:
            valid_dates = t.options
            if valid_dates:
                target_date = selected_date_str if selected_date_str in valid_dates else valid_dates[0]
                chain = t.option_chain(target_date)
                calls = chain.calls
                puts = chain.puts
                
                if not calls.empty:
                    closest_idx = (calls['strike'] - current_price).abs().idxmin()
                    atm_iv = f"{calls.loc[closest_idx, 'impliedVolatility'] * 100:.1f}%"
                
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
                    if mp_strike != "N/A": max_pain = f"${mp_strike:.2f}"
        except: pass

        # --- EX-DIVIDEND EXTRACTOR ---
        ex_div_date = "None scheduled"
        ex_div_veto = False
        try:
            info = t.info
            ex_ts = info.get('exDividendDate')
            if ex_ts:
                ex_dt = datetime.fromtimestamp(ex_ts)
                ex_div_date = ex_dt.strftime('%Y-%m-%d')
                if datetime.now() < ex_dt < selected_date:
                    ex_div_veto = True
        except: pass

        # --- EARNINGS EXTRACTION ---
        earnings_date = "Not scheduled"
        earnings_veto = False
        try:
            cal = t.calendar
            e_date = None
            if isinstance(cal, dict) and 'Earnings Date' in cal:
                e_date = pd.to_datetime(cal['Earnings Date'][0])
            elif isinstance(cal, pd.DataFrame) and 'Earnings Date' in cal.index:
                e_date = pd.to_datetime(cal.loc['Earnings Date'].iloc[0])
            
            if e_date and pd.notnull(e_date):
                earnings_date = e_date.strftime('%Y-%m-%d')
                if datetime.now() < e_date < selected_date: 
                    earnings_veto = True
        except: pass
            
        if current_price < ema_8 and rsi_14 < 45: base_risk = "🔴 ***FALLING KNIFE***: Consider Call Spreads Only"
        elif current_price > ema_8 and rsi_5 > rsi_5_prev and rsi_14 < 50: base_risk = "🟢 ***FLOOR CONFIRMED***: Consider Put Spreads Only"
        elif gap_risk > 1.5: base_risk = f"🟠 ***GAP RISK***: High Overnight Vol ({gap_risk:.2f}%)"
        elif adx_14 > 25: base_risk = f"🟡 ***TRENDING***: ADX {adx_14:.1f} (Pick a Directional Spread)"
        elif current_price > ma_20: base_risk = "🟢 ***NEUTRAL CHOP***: Iron Condor Territory"
        else: base_risk = "🟡 ***MED RISK***: Price Stalling"

        # Apply Title Warnings
        risk = base_risk
        if earnings_veto: risk += " [EARNINGS SOON]"
        if ex_div_veto: risk += " ⚠️[EX-DIVIDEND DANGER]"

        with st.expander(f"{symbol} | Price: ${current_price:.2f} | Risk: {risk}", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            
            def custom_metric_box(label, value, sub_value, val_color="#FAFAFA", sub_color="#a6a6a6"):
                return f"""
                <div style="line-height: 1.4; margin-bottom: 14px;">
                    <span style="font-size: 0.85rem; color: #a6a6a6; font-family: sans-serif;">{label}</span><br>
                    <span style="font-size: 1.8rem; font-weight: 600; color: {val_color}; font-family: sans-serif;">{value}</span><br>
                    <span style="font-size: 0.9rem; font-weight: 500; color: {sub_color}; font-family: sans-serif;">{sub_value}</span>
                </div>
                """

            with c1: st.markdown(custom_metric_box("Today's Change", f"${current_price:.2f}", f"{change_dlr:+.2f} ({change_pct:+.2f}%)", sub_color=change_color), unsafe_allow_html=True)
            with c2: st.markdown(custom_metric_box("Put Strategy", f"${put_strike}", f"Trip Wire: ${put_trip}", sub_color="#ffcc00"), unsafe_allow_html=True)
            with c3: st.markdown(custom_metric_box("Call Strategy", f"${call_strike}", f"Trip Wire: ${call_trip}", sub_color="#ffcc00"), unsafe_allow_html=True)
            with c4: st.markdown(custom_metric_box("Market Data", f"{atm_iv} IV", f"Earnings: {earnings_date}"), unsafe_allow_html=True)
            
            # --- NEW RISK UNDERWRITING ROW ---
            st.markdown("---")
            st.caption("🛡️ Risk Underwriting Data")
            u1, u2, u3 = st.columns(3)
            with u1:
                st.markdown(custom_metric_box("Max Pain", f"{max_pain}", "Gravity point for Friday", sub_color="#a6a6a6"), unsafe_allow_html=True)
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

            # --- DEEP-PARSING NEWS EXTRACTION UI ---
            ticker_news = []
            try:
                news_data = t.news
                if isinstance(news_data, list): ticker_news = news_data[:3]
            except: pass

            if ticker_news:
                st.markdown("---")
                st.caption("📰 Recent Headlines")
                for item in ticker_news:
                    if not isinstance(item, dict): continue
                    title, link, publisher, pub_time = "Headline Unavailable", "#", "Finance News", ""
                    if 'content' in item and isinstance(item['content'], dict):
                        content = item['content']
                        title = content.get('title', title)
                        if 'clickThroughUrl' in content and isinstance(content['clickThroughUrl'], dict): link = content['clickThroughUrl'].get('url', link)
                        elif 'canonicalUrl' in content and isinstance(content['canonicalUrl'], dict): link = content['canonicalUrl'].get('url', link)
                        if 'provider' in content and isinstance(content['provider'], dict): publisher = content['provider'].get('displayName', publisher)
                        if 'pubDate' in content:
                            try: pub_time = pd.to_datetime(content['pubDate']).strftime('%b %d, %H:%M')
                            except: pass
                    else:
                        title = item.get('title', title)
                        link = item.get('link', item.get('url', link))
                        publisher = item.get('publisher', publisher)
                        if 'providerPublishTime' in item:
                            try: pub_time = datetime.fromtimestamp(item['providerPublishTime']).strftime('%b %d, %H:%M')
                            except: pass
                    time_str = f" - {pub_time}" if pub_time else ""
                    st.markdown(f"- **[{title}]({link})** *({publisher}{time_str})*")

    except Exception as e:
        st.error(f"Error loading {symbol}: {str(e)}")
