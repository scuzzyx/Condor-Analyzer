# --- START OF PART 10: INTRADAY SNIPER (NEW) ---
with tab_intraday:
    if target_ticker:
        with st.spinner(f"Acquiring Intraday & Options Data for {target_ticker}..."):
            tkr = yf.Ticker(target_ticker)
            df_intraday = tkr.history(period="1d", interval="1m")
            
            if df_intraday.empty:
                st.error(f"Could not fetch 1-minute intraday data for {target_ticker}. The market might be closed or the ticker is invalid.")
            else:
                # 1. Price Action & Trend Calculation
                df_intraday = calculate_vwap(df_intraday)
                df_intraday['EMA_8'] = df_intraday['Close'].ewm(span=8, adjust=False).mean()
                
                curr_price_intra = df_intraday['Close'].iloc[-1]
                open_price = df_intraday['Open'].iloc[0]
                current_vwap = df_intraday['VWAP'].iloc[-1]
                curr_ema8 = df_intraday['EMA_8'].iloc[-1]
                
                price_color = "#09ab3b" if curr_price_intra >= open_price else "#ff4b4b"
                vwap_rel = ((curr_price_intra - current_vwap) / current_vwap) * 100
                
                # 2. Options Volume Flow Calculation
                pcr_vol, implied_move, exp_date_intra = "N/A", 0.0, "N/A"
                pcr_score = 0
                try:
                    exps = tkr.options
                    if exps:
                        exp_date_intra = exps[0] 
                        chain_intra = tkr.option_chain(exp_date_intra)
                        calls_intra, puts_intra = chain_intra.calls, chain_intra.puts
                        
                        total_call_vol = calls_intra['volume'].sum()
                        total_put_vol = puts_intra['volume'].sum()
                        if total_call_vol > 0: 
                            pcr_vol = total_put_vol / total_call_vol
                            # Grade the flow: < 0.85 is heavily bullish, > 1.15 is heavily bearish
                            if pcr_vol < 0.85: pcr_score = 1
                            elif pcr_vol > 1.15: pcr_score = -1
                        
                        atm_call_intra = calls_intra.iloc[(calls_intra['strike'] - curr_price_intra).abs().argsort()[:1]]
                        if not atm_call_intra.empty and 'impliedVolatility' in atm_call_intra:
                            iv_intra = atm_call_intra['impliedVolatility'].values[0]
                            implied_move = curr_price_intra * iv_intra * np.sqrt(1/365.0)
                except: pass

                # 3. The Conviction Engine (Scoring -3 to +3)
                score = 0
                score += 1 if curr_price_intra > current_vwap else -1  # Macro intraday trend
                score += 1 if curr_price_intra > curr_ema8 else -1     # Micro momentum
                score += pcr_score                                     # Smart money flow

                if score == 3: signal, sig_col = "🟢 HIGH CONVICTION LONG", "#09ab3b"
                elif score >= 1: signal, sig_col = "🟡 LEANING LONG (Scalp)", "#ffcc00"
                elif score <= -1 and score > -3: signal, sig_col = "🟡 LEANING SHORT (Scalp)", "#ffcc00"
                elif score == -3: signal, sig_col = "🔴 HIGH CONVICTION SHORT", "#ff4b4b"
                else: signal, sig_col = "⚪ NEUTRAL / CHOP ZONE", "#a6a6a6"

                # 4. UI Render
                st.markdown(f"### {target_ticker} | Intraday Profile")
                
                # Dynamic Conviction Banner
                st.markdown(f"<div style='text-align: center; padding: 15px; background-color: #1e1e1e; border: 2px solid {sig_col}; border-radius: 8px; margin-bottom: 20px;'><h3 style='color: {sig_col}; margin: 0;'>{signal}</h3><span style='color: #a6a6a6;'>Algo Score: {score}/3 | PCR: {pcr_vol if isinstance(pcr_vol, str) else round(pcr_vol, 2)} | VWAP Rel: {vwap_rel:+.2f}%</span></div>", unsafe_allow_html=True)

                m1, m2, m3, m4 = st.columns(4)
                with m1: st.markdown(intraday_metric("Last Price", f"${curr_price_intra:.2f}", f"Open: ${open_price:.2f}", val_color=price_color), unsafe_allow_html=True)
                with m2: st.markdown(intraday_metric("Intraday VWAP", f"${current_vwap:.2f}", f"8-EMA: ${curr_ema8:.2f}", val_color="#ffcc00"), unsafe_allow_html=True)
                with m3:
                    pcr_val = f"{pcr_vol:.2f}" if isinstance(pcr_vol, float) else pcr_vol
                    pcr_col = "#ff4b4b" if isinstance(pcr_vol, float) and pcr_vol > 1.0 else "#09ab3b"
                    st.markdown(intraday_metric(f"Nearest Vol PCR ({exp_date_intra})", pcr_val, "Volume Put/Call Ratio", val_color=pcr_col), unsafe_allow_html=True)
                with m4:
                    move_str = f"±${implied_move:.2f}" if implied_move > 0 else "N/A"
                    st.markdown(intraday_metric("1-Day Expected Move", move_str, "Based on ATM IV", val_color="#3498db"), unsafe_allow_html=True)

                fig_intra = go.Figure()
                fig_intra.add_trace(go.Candlestick(x=df_intraday.index, open=df_intraday['Open'], high=df_intraday['High'], low=df_intraday['Low'], close=df_intraday['Close'], name="Price"))
                fig_intra.add_trace(go.Scatter(x=df_intraday.index, y=df_intraday['VWAP'], line=dict(color='#ffcc00', width=2), name="VWAP"))
                fig_intra.add_trace(go.Scatter(x=df_intraday.index, y=df_intraday['EMA_8'], line=dict(color='#00d4ff', width=1.5, dash='dot'), name="8-EMA"))
                
                if implied_move > 0:
                    fig_intra.add_hline(y=open_price + implied_move, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="+1 EM")
                    fig_intra.add_hline(y=open_price - implied_move, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="-1 EM")

                fig_intra.update_layout(template="plotly_dark", height=550, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False, showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                st.plotly_chart(fig_intra, use_container_width=True)
# --- END OF PART 10 ---
