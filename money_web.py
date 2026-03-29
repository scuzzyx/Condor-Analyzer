import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import find_peaks

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Condor-Tool Terminal", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. HIDE STREAMLIT BRANDING & GITHUB BUTTONS ---
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

# --- 3. MATH & INDICATOR FUNCTIONS ---
def calculate_rsi(df, periods=14):
    """Calculates the Relative Strength Index (RSI)."""
    close_delta = df['Close'].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    # Calculate the exponential moving averages
    ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    
    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    return rsi

def get_volume_profile(df, bins=50):
    """Calculates Volume Profile, Point of Control (POC), and Volume Walls."""
    hist, bin_edges = np.histogram(df['Close'], bins=bins, weights=df['Volume'])
    prices = (bin_edges[:-1] + bin_edges[1:]) / 2
    profile = pd.DataFrame({'Price': prices, 'Volume': hist})
    
    # Find POC (Price with highest volume)
    poc_idx = profile['Volume'].idxmax()
    poc_price = profile.loc[poc_idx, 'Price']
    
    # Find Walls (Significant peaks in volume distribution)
    peaks, _ = find_peaks(profile['Volume'], distance=3, prominence=profile['Volume'].max()*0.1)
    wall_prices = profile.loc[peaks, 'Price'].values
    
    return profile, poc_price, wall_prices

# --- 4. MAIN APP UI & SIDEBAR ---
st.title("🦅 Condor-Tool Terminal V3")

with st.sidebar:
    st.header("Terminal Parameters")
    ticker_symbol = st.text_input("Ticker Symbol", value="META").upper()
    days_to_exp = st.number_input("Days to Expiration", min_value=1, value=20)
    z_score = st.slider("Z-Score Defense Level", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

# --- 5. DATA FETCHING & LOGIC PIPELINE ---
if ticker_symbol:
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="3mo") # 90-day lookback for structure
    
    if not df.empty:
        current_price = df['Close'].iloc[-1]
        
        # --- INDICATOR MATH ---
        df['RSI'] = calculate_rsi(df)
        current_rsi = df['RSI'].iloc[-1]
        
        if current_rsi >= 70:
            rsi_state = "🔥 Overbought"
        elif current_rsi <= 30:
            rsi_state = "🧊 Oversold"
        else:
            rsi_state = "⚪ Neutral"
            
        profile, poc_price, walls = get_volume_profile(df)
        
        # Sort walls relative to current price to find floors and ceilings
        put_walls = sorted([w for w in walls if w < current_price], reverse=True)
        call_walls = sorted([w for w in walls if w > current_price])
        
        wall_1_put_text = f"${put_walls[0]:.2f}" if len(put_walls) > 0 else "Freefall (None)"
        wall_2_put_text = f"${put_walls[1]:.2f}" if len(put_walls) > 1 else "⚠️ No 90d Wall"
        
        wall_1_call_text = f"${call_walls[0]:.2f}" if len(call_walls) > 0 else "Blue Sky (None)"
        wall_2_call_text = f"${call_walls[1]:.2f}" if len(call_walls) > 1 else "⚠️ No 90d Wall"

        # --- EXPECTED MOVE & STRIKE MATH ---
        # Calculating Historical Volatility to approximate Options Expected Move
        daily_returns = df['Close'].pct_change().dropna()
        hist_vol = daily_returns.std() * np.sqrt(252)
        expected_move_pct = hist_vol * np.sqrt(days_to_exp / 365) * z_score
        expected_move_dollar = current_price * expected_move_pct
        
        put_strike_rec = current_price - expected_move_dollar
        call_strike_rec = current_price + expected_move_dollar
        
        # --- RISK SYSTEM EVALUATION ---
        sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
        
        if current_price < sma_20 and len(put_walls) == 0:
            risk_level = "🔴 HIGH RISK (Structural Break / Falling Knife)"
        elif len(put_walls) == 0:
            risk_level = "🔴 HIGH RISK (Air Pocket Below)"
        elif len(put_walls) > 0 and put_strike_rec < put_walls[0]:
            risk_level = "🟢 LOW RISK (Defensible Structure)"
        else:
            risk_level = "🟡 MODERATE RISK"

        # --- 6. DASHBOARD DISPLAY ---
        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Current Price", f"${current_price:.2f}")
        col_b.metric(f"Expected Move (+/- {z_score} Z)", f"${expected_move_dollar:.2f}")
        col_c.metric("System Risk Level", risk_level)
        
        st.markdown("---")
        
        # --- STRUCTURAL ROW (4 COLUMNS) ---
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(label="🧲 Point of Control (POC)", value=f"${poc_price:.2f}", delta="↑ Highest Vol Magnet", delta_color="off")

        with col2:
            st.metric(label="📈 14-Day RSI", value=f"{current_rsi:.1f}", delta=rsi_state, delta_color="off")

        with col3:
            st.metric(label="🟢 Put Defense (Floors)", value=f"Wall 1: {wall_1_put_text}", delta=f"↑ Wall 2: {wall_2_put_text}", delta_color="off")

        with col4:
            st.metric(label="🔴 Call Defense (Ceilings)", value=f"Wall 1: {wall_1_call_text}", delta=f"↑ Wall 2: {wall_2_call_text}", delta_color="off")
            
        st.markdown("---")
        
        # --- FINAL MATH OUTPUT ---
        st.subheader("Engine Strike Recommendations")
        st.write(f"**Safe Put Strike:** Below ${put_strike_rec:.2f}")
        st.write(f"**Safe Call Strike:** Above ${call_strike_rec:.2f}")
