# --- START OF PART 1: IMPORTS & INIT ---
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy.stats import norm
import requests

# --- CONFIG & THEME ---
st.set_page_config(page_title="Aegis Terminal", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h2 style='font-size: 2.2rem; margin-bottom: 0rem;'>🛡️ Aegis Terminal | Delta & Intraday Scanner</h2>", unsafe_allow_html=True)
# --- END OF PART 1 ---

# --- START OF PART 2: BLACK-SCHOLES ENGINE ---
def calculate_delta(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0: return 0.5
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

def find_delta_strikes(chain_df, S, dte, target_delta, option_type):
    try:
        T = max(dte, 1) / 365.0 
        r = 0.04
        df = chain_df[chain_df['strike'] >= S].copy() if option_type == 'call' else chain_df[chain_df['strike'] <= S].copy()
        if df.empty: return None, None
        df['impliedVolatility'] = df['impliedVolatility'].replace(0, np.nan).fillna(0.3) 
        df['delta'] = df.apply(lambda x: calculate_delta(S, x['strike'], T, r, x['impliedVolatility'], option_type), axis=1)
        best_match = df.loc[(df['delta'].abs() - target_delta).abs().idxmin()]
        return float(best_match['strike']), float(best_match['delta'])
    except: return None, None

def calculate_expected_move(price, iv, dte):
    if pd.isna(iv) or iv <= 0: return 0.0
    return price * iv * np.sqrt(max(1, dte) / 365.0)

def calculate_pop_metrics(delta_val):
    if pd.isna(delta_val): return "N/A", "N/A"
    pop = (1 - abs(delta_val)) * 100
    p50 = min(99.0, pop + ((100 - pop) * 0.4))
    return f"{pop:.1f}%", f"{p50:.1f}%"
# --- END OF PART 2 ---

# --- START OF PART 3: TECHNICAL INDICATORS & UI HELPERS ---
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

def calculate_ivr(hist_1y, current_iv):
    try:
        if pd.isna(current_iv): return "N/A"
        curr_iv_val = float(current_iv) if not isinstance(current_iv, str) else float(current_iv.replace('%', ''))/100
        hv = hist_1y['Close'].pct_change().dropna().rolling(20).std() * np.sqrt(252)
        h_min, h_max = float(hv.min()), float(hv.max())
        if h_max == h_min: return 50.0
        return max(0, min(100, ((curr_iv_val - h_min) / (h_max - h_min)) * 100))
    except: return "N/A"

def calculate_gap_risk(hist):
    try: return (abs((hist['Open'] - hist['Close'].shift(1)) / hist['Close'].shift(1))).tail(30).mean() * 100
    except: return 0

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

def calculate_vwap(df):
    """Calculates intraday VWAP."""
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['Typical_Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df

def get_pure_fridays(weeks=26):
    today = datetime.now()
    days_until_friday = (4 - today.weekday()) % 7
    next_friday = today + timedelta(days=days_until_friday)
    return [(next_friday + timedelta(weeks=i)).strftime('%Y-%m-%d') for i in range(weeks)]

def custom_metric_box(label, value, sub_value, val_color="#FAFAFA", sub_color="#a6a6a6"):
    return f'<div style="line-height: 1.4; margin-bottom: 14px;"><span style="font-size: 0.85rem; color: #a6a6a6; font-family: sans-serif;">{label}</span><br><span style="font-size: 1.8rem; font-weight: 600; color: {val_color}; font-family: sans-serif;">{value}</span><br><span style="font-size: 0.9rem; font-weight: 500; color: {sub_color}; font-family: sans-serif;">{sub_value}</span></div>'

def intraday_metric(label, value, sub_value="", val_color="#FAFAFA", sub_color="#a6a6a6"):
    return f'<div style="line-height: 1.4; margin-bottom: 14px; padding: 10px; background-color: #1e1e1e; border-radius: 8px;"><span style="font-size: 0.85rem; color: #a6a6a6;">{label}</span><br><span style="font-size: 1.8rem; font-weight: 600; color: {val_color};">{value}</span><br><span style="font-size: 0.9rem; font-weight: 500; color: {sub_color};">{sub_value}</span></div>'
# --- END OF PART 3 ---

# --- START OF PART 4: SCRAPERS & MACRO ---
@st.cache_data(ttl=3600, show_spinner=False)
def run_premium_hunter(ticker_list):
    targets = []
    try:
        bulk_data = yf.download(ticker_list, period="1y", progress=False)['Close']
        for sym in ticker_list:
            try:
                hist = bulk_data[sym].dropna()
                if len(hist) < 50: continue
                returns = hist.pct_change().dropna()
                hv_series = returns.rolling(20).std() * np.sqrt(252)
                curr_hv = hv_series.iloc[-1]
                hv_min, hv_max = hv_series.min(), hv_series.max()
                if hv_max > hv_min:
                    hv_rank = ((curr_hv - hv_min) / (hv_max - hv_min)) * 100
                    if hv_rank > 60:
                        targets.append((sym, hv_rank))
            except: continue
        targets.sort(key=lambda x: x[1], reverse=True)
        return [f"{t[0]} (Rank: {t[1]:.0f})" for t in targets[:6]]
    except: return []

@st.cache_data(ttl=300, show_spinner=False)
def run_short_hunter(ticker_list):
    targets = []
    try:
        # Bulk download 1-minute data to save time and API calls
        bulk_intraday = yf.download(ticker_list, period="1d", interval="1m", progress=False)
        
        for sym in ticker_list:
            try:
                # 1. Verify Price Action Breakdown (Fast Check)
                if 'Close' not in bulk_intraday or sym not in bulk_intraday['Close']:
                    continue
                    
                df_sym = pd.DataFrame({
                    'Open': bulk_intraday['Open'][sym],
                    'High': bulk_intraday['High'][sym],
                    'Low': bulk_intraday['Low'][sym],
                    'Close': bulk_intraday['Close'][sym],
                    'Volume': bulk_intraday['Volume'][sym]
                }).dropna()
                
                if df_sym.empty: continue
                
                df_sym = calculate_vwap(df_sym)
                df_sym['EMA_8'] = df_sym['Close'].ewm(span=8, adjust=False).mean()
                
                curr_price = df_sym['Close'].iloc[-1]
                curr_vwap = df_sym['VWAP'].iloc[-1]
                curr_ema8 = df_sym['EMA_8'].iloc[-1]
                
                score = 0
                score += 1 if curr_price > curr_vwap else -1
                score += 1 if curr_price > curr_ema8 else -1
                
                # If it's not already at -2 from price action, skip the slow options check
                if score > -2:
                    continue
                    
                # 2. Verify Options Flow Confirmation (Slow Check)
                tkr = yf.Ticker(sym)
                pcr_score = 0
                exps = tkr.options
                if exps:
                    chain = tkr.option_chain(exps[0])
                    c_vol = chain.calls['volume'].sum()
                    p_vol = chain.puts['volume'].sum()
                    if c_vol > 0:
                        pcr = p_vol / c_vol
                        if pcr > 1.15: pcr_score = -1
                        elif pcr < 0.85: pcr_score = 1
                        
                score += pcr_score
                
                # 3. Add to targets if it hits perfect -3
                if score == -3:
                    targets.append(sym)
            except:
                continue
                
        return targets
    except:
        return []

@st.cache_data(ttl=900, show_spinner=False)
def fetch_macro_data():
    vix_val, vix_pct, fg_val, fg_rating = "N/A", "N/A", "N/A", "N/A"
    try:
        vix_hist = yf.Ticker("^VIX").history(period="1mo")
        if not vix_hist.empty and len(vix_hist) >= 2:
            vix_val = float(vix_hist['Close'].iloc[-1])
            vix_pct = float(((vix_val - vix_hist['Close'].iloc[-2]) / vix_hist['Close'].iloc[-2]) * 100)
    except: pass
    
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Referer': 'https://www.cnn.com/',
            'Origin': 'https://www.cnn.com/',
            'Sec-Fetch-Site': 'cross-site',
            'Sec-Fetch-Mode': 'cors'
        }
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            fg_val = round(data['fear_and_greed']['score'])
            fg_rating = data['fear_and_greed']['rating'].title()
    except: pass
    
    return vix_val, vix_pct, fg_val, fg_rating
# --- END OF PART 4 ---

# --- START OF PART 5: TRADIER PAYLOAD ---
@st.cache_data(ttl=300, show_spinner=False)
def fetch_vault_payload(symbol, target_date):
    """Pulls options data and history using Tradier (Cached to prevent rate limits)."""
    try:
        headers = {
            "Authorization": f"Bearer {st.secrets['TRADIER_API_KEY'].strip()}",
            "Accept": "application/json"
        }
        
        # 1. Fetch History 
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        hist_url = f"https://api.tradier.com/v1/markets/history?symbol={symbol}&start={start_date.strftime('%Y-%m-%d')}&end={end_date.strftime('%Y-%m-%d')}"
        hist_resp = requests.get(hist_url, headers=headers)
                    
        hist_df = pd.DataFrame()
        if hist_resp.status_code == 200:
            hist_data = hist_resp.json()
            if 'history' in hist_data and hist_data['history'] and 'day' in hist_data['history']:
                day_data = hist_data['history']['day']
                if isinstance(day_data, dict): day_data = [day_data]
                hist_df = pd.DataFrame(day_data)
                if not hist_df.empty:
                    hist_df['Date'] = pd.to_datetime(hist_df['date'])
                    hist_df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                    hist_df.set_index('Date', inplace=True)
                    
        # Fallback to yfinance
        if hist_df.empty:
            hist_df = yf.Ticker(symbol).history(period="1y")

        # 2. Fetch Options Expirations
        exp_url = f"https://api.tradier.com/v1/markets/options/expirations?symbol={symbol}"
        exp_resp = requests.get(exp_url, headers=headers)
        
        snap_date = target_date
        if exp_resp.status_code == 200:
            exp_data = exp_resp.json()
            if 'expirations' in exp_data and exp_data['expirations'] and 'date' in exp_data['expirations']:
                dates = exp_data['expirations']['date']
                if isinstance(dates, str): dates = [dates]
                target_dt = datetime.strptime(target_date, '%Y-%m-%d')
                valid_dts = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
                if valid_dts:
                    snap_date = min(valid_dts, key=lambda d: abs(d - target_dt)).strftime('%Y-%m-%d')

        # 3. Fetch Option Chain 
        chain_url = f"https://api.tradier.com/v1/markets/options/chains?symbol={symbol}&expiration={snap_date}&greeks=true"
        chain_resp = requests.get(chain_url, headers=headers)
             
        calls_list, puts_list = [], []
        if chain_resp.status_code == 200:
            chain_data = chain_resp.json()
            if 'options' in chain_data and chain_data['options'] and 'option' in chain_data['options']:
                options_data = chain_data['options']['option']
                if isinstance(options_data, dict): options_data = [options_data]
                
                for opt in options_data:
                    greeks = opt.get('greeks')
                    iv = 0.3
                    if isinstance(greeks, dict):
                        iv = greeks.get
