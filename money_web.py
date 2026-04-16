# --- START OF PART 1: SETUP & THE VAULT ---
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy.stats import norm
import time
import requests
import yfinance as yf # Added Yahoo Finance

# --- CONFIG & THEME ---
st.set_page_config(page_title="Aegis 2.0 | Command Center", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h2 style='font-size: 2.2rem; margin-bottom: 0rem;'>🛡️ Aegis 2.0 | API Command Center</h2>", unsafe_allow_html=True)

# --- INITIALIZE MEMORY VAULT ---
if "options_vault" not in st.session_state:
    st.session_state.options_vault = {}
if "last_pull_time" not in st.session_state:
    st.session_state.last_pull_time = 0.0
if "ui_refresh_time" not in st.session_state:
    st.session_state.ui_refresh_time = None

def custom_metric_box(label, value, sub_value, val_color="#FAFAFA", sub_color="#a6a6a6"):
    return f'<div style="line-height: 1.4; margin-bottom: 14px;"><span style="font-size: 0.85rem; color: #a6a6a6; font-family: sans-serif;">{label}</span><br><span style="font-size: 1.8rem; font-weight: 600; color: {val_color}; font-family: sans-serif;">{value}</span><br><span style="font-size: 0.9rem; font-weight: 500; color: {sub_color}; font-family: sans-serif;">{sub_value}</span></div>'
# --- END OF PART 1 ---
# --- START OF PART 2: API ENGINES ---
def get_alpaca_price(ticker):
    """Fetches instant live prices using Alpaca."""
    url = f"https://data.alpaca.markets/v2/stocks/{ticker}/trades/latest?feed=iex"
    headers = {"APCA-API-KEY-ID": st.secrets["ALPACA_KEY_ID"], "APCA-API-SECRET-KEY": st.secrets["ALPACA_SECRET_KEY"]}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()["trade"]["p"] 
        return 0.0
    except:
        return 0.0

def get_alpaca_history(ticker):
    """Fetches 150 days of daily candles for RSI, ADX, and Volume Walls."""
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d')
    url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars?timeframe=1Day&start={start}&end={end}&limit=150&feed=iex"
    headers = {"APCA-API-KEY-ID": st.secrets["ALPACA_KEY_ID"], "APCA-API-SECRET-KEY": st.secrets["ALPACA_SECRET_KEY"]}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if 'bars' in data and data['bars']:
                df = pd.DataFrame(data['bars'])
                df['t'] = pd.to_datetime(df['t'])
                df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume', 't': 'Date'})
                df.set_index('Date', inplace=True)
                return df
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def pull_master_payload(ticker, current_price):
    """Pulls the Option Chain using yfinance to bypass API limits and get free Open Interest."""
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        
        if not expirations:
            return None
            
        calls_list, puts_list = [], []
        
        # Grab the first 5 expiration dates to load into the vault quickly
        target_expirations = list(expirations[:5])
        
        for exp_date in target_expirations:
            chain = stock.option_chain(exp_date)
            
            # Process Calls
            for index, row in chain.calls.iterrows():
                calls_list.append((exp_date, {
                    'strike': row['strike'],
                    'openInterest': row['openInterest'] if pd.notna(row['openInterest']) else 0,
                    'volume': row['volume'] if pd.notna(row['volume']) else 0,
                    'impliedVolatility': row['impliedVolatility']
                }))
                
            # Process Puts
            for index, row in chain.puts.iterrows():
                puts_list.append((exp_date, {
                    'strike': row['strike'],
                    'openInterest': row['openInterest'] if pd.notna(row['openInterest']) else 0,
                    'volume': row['volume'] if pd.notna(row['volume']) else 0,
                    'impliedVolatility': row['impliedVolatility']
                }))
                
        return {
            "calls": calls_list, 
            "puts": puts_list, 
            "expirations": target_expirations
        }
    except Exception as e:
        st.error(f"yfinance Error: {str(e)}")
        return None
# --- END OF PART 2 ---
# --- START OF PART 3: AEGIS MATH ENGINE ---
def calculate_delta(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0: return 0.5
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

def find_delta_strikes(chain_df, S, dte, target_delta, option_type):
    try:
        T = max(dte, 1) / 365.0 
        r = 0.04
        df = chain_df[chain_df['strike'] >= S].copy() if option_type == 'call' else chain_df[chain_df['strike'] <= S].copy()
        if df.empty: return None
        df['impliedVolatility'] = df['impliedVolatility'].replace(0, np.nan).fillna(0.3) 
        df['delta'] = df.apply(lambda x: calculate_delta(S, x['strike'], T, r, x['impliedVolatility'], option_type), axis=1)
        return df.loc[(df['delta'].abs() - target_delta).abs().idxmin(), 'strike']
    except: return None

def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    return 100 - (100 / (1 + (gain / loss)))

def calculate_adx(hist, period=14):
    try:
        high, low, close = hist['High'], hist['Low'], hist['Close']
        plus_dm, minus_dm = high.diff(), low.diff()
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
        tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
        return ((abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100).ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    except: return 20 

def calculate_volume_nodes(hist, current_price, bins=30):
    try:
        min_p, max_p = float(hist['Low'].min()), float(hist['High'].max())
        if min_p == max_p or pd.isna(min_p): return f"${current_price:.2f}", "N/A", "N/A", "N/A", "N/A"
        price_bins = np.linspace(min_p, max_p, bins)
        inds = np.digitize(hist['Close'].fillna(current_price).values, price_bins)
        vol_profile = np.zeros(bins)
        volumes = hist['Volume'].fillna(0).values
        for i in range(len(hist)):
            if 0 <= inds[i]-1 < bins: vol_profile[inds[i]-1] += volumes[i]
        poc = price_bins[np.argmax(vol_profile)]
        peaks = [price_bins[i] for i in range(1, bins-1) if vol_profile[i] > vol_profile[i-1] and vol_profile[i] > vol_profile[i+1] and vol_profile[i] > np.mean(vol_profile)*0.5]
        upper, lower = sorted([p for p in peaks if p > current_price]), sorted([p for p in peaks if p < current_price])
        r1 = f"${upper[0]:.2f}" if upper else "Sky (None)"
        r2 = f"${upper[1]:.2f}" if len(upper) > 1 else "⚠️ No Wall"
        s1 = f"${lower[-1]:.2f}" if lower else "Freefall (None)"
        s2 = f"${lower[-2]:.2f}" if len(lower) > 1 else "⚠️ No Wall"
        return f"${poc:.2f}", s1, s2, r1, r2
    except: return "N/A", "N/A", "N/A", "N/A", "N/A"
# --- END OF PART 3 ---
# --- START OF PART 4: DASHBOARD UI & COOLDOWN ---
st.sidebar.header("The Bench (Max 5)")
bench = []
for i in range(1, 6):
    default = ["AAPL", "MSFT", "GOOGL", "META", "AMZN"][i-1]
    ticker = st.sidebar.text_input(f"Ticker {i}", default).upper()
    if ticker: bench.append(ticker)

st.markdown("### 📡 Live Market Pricing (Alpaca)")
cols = st.columns(len(bench))
live_prices = {}
for i, ticker in enumerate(bench):
    with cols[i]:
        live_price = get_alpaca_price(ticker)
        live_prices[ticker] = live_price
        st.metric(label=f"{ticker} Live", value=f"${live_price}")

st.divider()

current_time = time.time()
time_since_last_scan = current_time - st.session_state.last_pull_time
cooldown_period = 61.0

col1, col2 = st.columns([1, 3])

with col1:
    if time_since_last_scan < cooldown_period:
        seconds_left = int(cooldown_period - time_since_last_scan)
        st.warning(f"⏳ API Cooling Down: Please wait **{seconds_left}s**")
        st.button("🚀 PULL MASTER PAYLOAD", disabled=True, use_container_width=True)
    else:
        if st.button("🚀 PULL MASTER PAYLOAD (5 Calls)", use_container_width=True):
            st.session_state.options_vault.clear()
            st.session_state.last_pull_time = time.time()
            st.session_state.ui_refresh_time = datetime.now().strftime("%I:%M:%S %p")
            
            for ticker in bench:
                with st.spinner(f"Downloading Master Chain for {ticker}..."):
                    current_price = live_prices.get(ticker, 0.0)
                    st.session_state.options_vault[ticker] = {
                        "options": pull_master_payload(ticker, current_price),
                        "history": get_alpaca_history(ticker)
                    }
            time.sleep(1)
            st.rerun()

with col2:
    if st.session_state.ui_refresh_time:
        st.success(f"✅ Vault Loaded securely at: **{st.session_state.ui_refresh_time}**. You are offline and safe to explore.")
    else:
        st.warning("⚠️ Vault is empty. Click the button to pull market data.")
# --- END OF PART 4 ---
# --- START OF PART 5: THE SAFE ZONE UI ---
if st.session_state.options_vault:
    tabs = st.tabs(bench)
    
    for i, ticker in enumerate(bench):
        with tabs[i]:
            vault_data = st.session_state.options_vault.get(ticker)
            if not vault_data or not vault_data["options"]:
                st.error(f"Failed to pull data for {ticker}. Check ticker symbol or API limit.")
                continue
                
            history_df = vault_data["history"]
            current_price = live_prices.get(ticker, 0.0)
            
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                selected_exp = st.selectbox(f"Select Expiration ({ticker})", vault_data["options"]["expirations"], key=f"exp_{ticker}")
            with filter_col2:
                target_delta = st.select_slider(f"Target Strike Delta ({ticker})", options=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40], value=0.15, key=f"del_{ticker}")

            calls_raw = [row for date, row in vault_data["options"]["calls"] if date == selected_exp]
            puts_raw = [row for date, row in vault_data["options"]["puts"] if date == selected_exp]
            calls = pd.DataFrame(calls_raw)
            puts = pd.DataFrame(puts_raw)
            
            dte = max(1, (datetime.strptime(selected_exp, '%Y-%m-%d') - datetime.now()).days)
            
            max_pain, pc_ratio, call_strike, put_strike = "N/A", "N/A", "N/A", "N/A"
            poc, s1, s2, r1, r2 = calculate_volume_nodes(history_df, current_price) if not history_df.empty else ("N/A", "N/A", "N/A", "N/A", "N/A")
            
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
                
                c_strike = find_delta_strikes(calls, current_price, dte, target_delta, 'call')
                p_strike = find_delta_strikes(puts, current_price, dte, target_delta, 'put')
                if c_strike: call_strike = f"${c_strike:.2f}"
                if p_strike: put_strike = f"${p_strike:.2f}"

            st.markdown("---")
            st.caption("🛡️ Risk Underwriting Data")
            u1, u2, u3, u4 = st.columns(4)
            with u1: st.markdown(custom_metric_box("Max Pain", f"{max_pain}", "Gravity point", sub_color="#a6a6a6"), unsafe_allow_html=True)
            with u2: st.markdown(custom_metric_box("P/C OI Ratio", f"{pc_ratio}", "Flow", sub_color="#a6a6a6"), unsafe_allow_html=True)
            with u3: st.markdown(custom_metric_box(f"{int(target_delta*100)}Δ Call", f"{call_strike}", "Resistance", sub_color="#ffcc00"), unsafe_allow_html=True)
            with u4: st.markdown(custom_metric_box(f"{int(target_delta*100)}Δ Put", f"{put_strike}", "Support", sub_color="#ffcc00"), unsafe_allow_html=True)

            st.markdown("---")
            v1, v2, v3 = st.columns(3)
            with v1: st.caption("🧲 Point of Control"); st.write(f"**POC:** {poc}")
            with v2: st.caption("🔴 Support Walls"); st.write(f"**Wall 1:** {s1}"); st.write(f"**Wall 2:** {s2}")
            with v3: st.caption("🟢 Resistance Walls"); st.write(f"**Wall 1:** {r1}"); st.write(f"**Wall 2:** {r2}")

            if not history_df.empty:
                fig = go.Figure(data=[go.Candlestick(x=history_df.index, open=history_df['Open'], high=history_df['High'], low=history_df['Low'], close=history_df['Close'], name="Price")])
                if c_strike: fig.add_hline(y=float(c_strike.replace('$','')), line_width=2, line_color="green", annotation_text=f"{target_delta}Δ Call")
                if p_strike: fig.add_hline(y=float(p_strike.replace('$','')), line_width=2, line_color="red", annotation_text=f"{target_delta}Δ Put")
                fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
# --- END OF PART 5 ---
