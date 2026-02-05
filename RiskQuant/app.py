import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from bs_model import black_scholes, calculate_greeks
from scipy.stats import norm
import tensorflow as tf
from tensorflow import keras
import os
import feedparser
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# --- Load Pre-trained AI Model (Lazy Load) ---
@st.cache_resource
def load_ai_model():
    # Use absolute path relative to this script
    model_path = os.path.join(os.path.dirname(__file__), "option_pricing_model.keras")
    if os.path.exists(model_path):
        return keras.models.load_model(model_path)
    return None

ai_model = load_ai_model()

# Download VADER
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# --- Sentiment Helper ---
@st.cache_data(ttl=600)
def get_live_sentiment(ticker_symbol):
    try:
        clean_query = ticker_symbol.split('.')[0] 
        url = f"https://news.google.com/rss/search?q={clean_query}+stock+market&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        news = feed.entries[:10]
        
        if not news: return 0.0
        
        sia = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(item.title)['compound'] for item in news]
        return sum(scores) / len(scores) if scores else 0.0
    except:
        return 0.0

# --- Page Config ---
st.set_page_config(page_title="RiskQuant Dashboard", layout="wide", page_icon="📉")

# --- Custom Theme (Light/Clean for better visibility) ---
st.markdown("""
<style>
    /* Global Text Visibility Fix */
    .stApp {
        background-color: #f8f9fa;
        color: #1a202c;
    }
    
    /* Input Fields */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #ffffff;
        color: #2d3748;
        border: 1px solid #e2e8f0;
    }
    
    /* Metrics - Force BLACK for Light Theme */
    div[data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }
    label[data-testid="stMetricLabel"] {
        color: #000000 !important;
    }
    div[data-testid="stMetric"] label {
        color: #000000 !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2c5282 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #edf2f7;
        border-radius: 8px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 5px;
        color: #4a5568;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #2b6cb0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Cards */
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    /* Buttons (Run Simulation & Download) */
    .stButton > button, .stDownloadButton > button {
        color: white !important;
        background-color: #2b6cb0 !important;
        border: none !important;
    }
    
    /* Checkbox - Force visibility */
    .stCheckbox label span {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* Force checkbox square itself to look normal */
    div[data-testid="stCheckbox"] > label > div[role="checkbox"] {
        background-color: #ffffff !important;
        border: 1px solid #a0aec0 !important;
    }
    div[data-testid="stCheckbox"] > label > div[role="checkbox"][aria-checked="true"] {
        background-color: #2b6cb0 !important;
        border-color: #2b6cb0 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📉 RiskQuant: Derivatives & Risk Analytics")

# --- SIDEBAR NAV ---
st.sidebar.header("Navigation")
nav = st.sidebar.radio("Select Module", [
    "Options Pricing (Black-Scholes)", 
    "AI Options Pricer (Neural Network)", 
    "VaR / CVaR Analysis", 
    "Monte Carlo Stress Test", 
    "Portfolio Comparison"
])

# ==========================================
# TAB 1: BLACK-SCHOLES PRICING
# ==========================================
if nav == "Options Pricing (Black-Scholes)":
    st.header("Option Pricing & Greeks")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        ticker = st.text_input("Underlying Ticker", "SPY")
        
        # Fetch Real-time Spot
        spot_price = 500.0
        try:
            data = yf.Ticker(ticker).history(period='1y') # Fetch 1y for technicals
            if not data.empty:
                spot_price = data['Close'].iloc[-1]
                st.success(f"Current Spot Price: ${spot_price:.2f}")
                
                # --- TECHNICAL INDICATORS (RSI & MA) ---
                # Moving Averages
                data['MA20'] = data['Close'].rolling(20).mean()
                data['MA50'] = data['Close'].rolling(50).mean()
                
                # RSI Calculation
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                data['RSI'] = 100 - (100 / (1 + rs))
                
                with st.expander("Technical Charts (RSI & MA)"):
                    # Price & MA
                    fig_tech = go.Figure()
                    fig_tech.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='OHLC'))
                    fig_tech.add_trace(go.Scatter(x=data.index, y=data['MA20'], line=dict(color='orange', width=1), name='MA 20'))
                    fig_tech.add_trace(go.Scatter(x=data.index, y=data['MA50'], line=dict(color='cyan', width=1), name='MA 50'))
                    fig_tech.update_layout(title=f"{ticker} Price & Moving Averages", height=400, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig_tech, use_container_width=True)
                    
                    # RSI
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    fig_rsi.update_layout(title="Relative Strength Index (RSI)", height=250, yaxis_range=[0, 100])
                    st.plotly_chart(fig_rsi, use_container_width=True)
            else:
                st.warning("No live data found. Using default.")
        except:
            st.warning("Data fetch failed. Using default.")

        S = st.number_input("Spot Price (S)", value=spot_price)
        K = st.number_input("Strike Price (K)", value=spot_price)
        T_days = st.slider("Days to Expiry", 1, 365, 30)
        T = T_days / 365.0
        r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.5) / 100
        sigma = st.slider("Implied Volatility (%)", 1.0, 200.0, 20.0) / 100
        opt_type = st.radio("Option Type", ["Call", "Put"])
        
    with col2:
        # Calculate Price
        price = black_scholes(S, K, T, r, sigma, opt_type.lower())
        greeks = calculate_greeks(S, K, T, r, sigma, opt_type.lower())
        
        st.subheader("Theoretical Price")
        st.markdown(f"<h1 style='color: #2b6cb0'>${price:.4f}</h1>", unsafe_allow_html=True)
        
        st.subheader("The Greeks")
        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Delta (Δ)", f"{greeks['Delta']:.4f}")
        g2.metric("Gamma (Γ)", f"{greeks['Gamma']:.4f}")
        g3.metric("Theta (Θ)", f"{greeks['Theta']:.4f}")
        g4.metric("Vega (ν)", f"{greeks['Vega']:.4f}")
        g5.metric("Rho (ρ)", f"{greeks['Rho']:.4f}")
        
        # Heatmap Visualization
        st.subheader("Sensitivity Analysis (Spot vs Volatility)")
        
        # Generate Meshgrid
        spot_range = np.linspace(S * 0.8, S * 1.2, 20)
        vol_range = np.linspace(sigma * 0.5, sigma * 1.5, 20)
        X, Y = np.meshgrid(spot_range, vol_range)
        Z = np.array([black_scholes(s, K, T, r, v, opt_type.lower()) for s, v in zip(np.ravel(X), np.ravel(Y))])
        Z = Z.reshape(X.shape)
        
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
        fig.update_layout(title='Option Price Surface', scene=dict(
            xaxis_title='Spot Price',
            yaxis_title='Volatility',
            zaxis_title='Option Price'
        ), height=500)
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB: AI OPTIONS PRICER
# ==========================================
elif nav == "AI Options Pricer (Neural Network)":
    st.header("🧠 AI-Powered Real-Time Pricing")
    st.markdown("This module fetches **Live Option Chains** and uses AI + Sentiment to find mispriced opportunities.")
    
    col_ai1, col_ai2 = st.columns([1, 2])
    
    with col_ai1:
        st.subheader("Select Asset")
        ticker = st.text_input("Ticker", "SPY", key="ai_ticker")
        expiry = st.date_input("Target Expiry Date (Approx)", pd.to_datetime("today") + pd.Timedelta(days=30))
        
        # Fetch Real Data
        if st.button("Fetch Market Data"):
            with st.spinner("Fetching Live Options & Sentiment..."):
                try:
                    tk = yf.Ticker(ticker)
                    current_price = tk.history(period="1d")['Close'].iloc[-1]
                    
                    # Get Option Chain
                    expirations = tk.options
                    # Find closest expiry
                    target_date = expiry.strftime("%Y-%m-%d")
                    # Simple logic: pick first expiry after target
                    chosen_exp = expirations[0] 
                    for exp in expirations:
                        if exp >= target_date:
                            chosen_exp = exp
                            break
                            
                    opt_chain = tk.option_chain(chosen_exp)
                    calls = opt_chain.calls
                    
                    # Get Sentiment
                    sentiment = get_live_sentiment(ticker)
                    
                    st.session_state['ai_data'] = {
                        'spot': current_price,
                        'calls': calls,
                        'expiry': chosen_exp,
                        'sentiment': sentiment
                    }
                    st.success(f"Loaded Data for {chosen_exp}")
                    
                except Exception as e:
                    st.error(f"Error fetching data: {e}")
    
    with col_ai2:
        if 'ai_data' in st.session_state:
            data = st.session_state['ai_data']
            spot = data['spot']
            sentiment = data['sentiment']
            
            st.metric("Live Spot Price", f"${spot:.2f}")
            st.metric("Live Sentiment Score", f"{sentiment:.2f}", 
                      "Bullish" if sentiment > 0.05 else "Bearish" if sentiment < -0.05 else "Neutral")
            
            # Filter Calls near the money
            calls = data['calls']
            near_money = calls[(calls['strike'] > spot * 0.9) & (calls['strike'] < spot * 1.1)].copy()
            
            if not near_money.empty:
                st.subheader(f"AI Valuation Analysis ({data['expiry']})")
                
                # Predict AI Prices for these strikes
                # T approx
                days_to_exp = (pd.to_datetime(data['expiry']) - pd.to_datetime("today")).days
                T = days_to_exp / 365.0
                r = 0.045 # Risk free approx
                
                ai_prices = []
                bs_prices = []
                
                for idx, row in near_money.iterrows():
                    K = row['strike']
                    sigma = row['impliedVolatility']
                    
                    # AI Input: [S, K, T, r, sigma, sentiment]
                    input_vec = np.array([[spot, K, T, r, sigma, sentiment]])
                    ai_p = ai_model.predict(input_vec)[0][0] if ai_model else 0
                    bs_p = black_scholes(spot, K, T, r, sigma, "call")
                    
                    ai_prices.append(ai_p)
                    bs_prices.append(bs_p)
                
                near_money['AI Fair Value'] = ai_prices
                near_money['Black-Scholes'] = bs_prices
                near_money['Difference'] = near_money['AI Fair Value'] - near_money['lastPrice']
                
                # Show Table
                display_cols = ['strike', 'lastPrice', 'AI Fair Value', 'Black-Scholes', 'Difference', 'impliedVolatility']
                st.dataframe(near_money[display_cols].style.format("{:.2f}"))
                
                st.info("💡 **Positive Difference** means AI thinks the option is **Undervalued** (Good Buy). **Negative** means **Overvalued**.")
            else:
                st.warning("No Near-the-Money options found.")
        else:
            st.info("Enter Ticker and Click 'Fetch Market Data'")

# ==========================================
# TAB 2: VaR / CVaR Analysis
# ==========================================
elif nav == "VaR / CVaR Analysis":
    st.header("Value at Risk (VaR) & Expected Shortfall (CVaR)")
    
    col_var1, col_var2 = st.columns([1, 2])
    
    with col_var1:
        st.subheader("Portfolio Config")
        portfolio_ticker = st.text_input("Portfolio Ticker/Index", "SPY", key="var_ticker")
        history_years = st.slider("History (Years)", 1, 10, 2)
        confidence_level = st.slider("Confidence Level (%)", 90.0, 99.9, 95.0) / 100
        investment = st.number_input("Portfolio Value ($)", value=100000)
        
    with col_var2:
        # Fetch Historical Data
        start_date = pd.to_datetime("today") - pd.DateOffset(years=history_years)
        try:
            df = yf.download(portfolio_ticker, start=start_date, progress=False)
            
            # yfinance MultiIndex Fix
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            if df.empty:
                st.error("No data found for this ticker.")
            else:
                # Calculate Daily Returns
                df['Returns'] = df['Close'].pct_change().dropna()
                returns = df['Returns'].dropna()
                
                if len(returns) == 0:
                     st.error("Not enough data to calculate returns.")
                else:
                    # Calculate VaR and CVaR (Historical Method)
                    var_percent = np.percentile(returns, (1 - confidence_level) * 100)
                    var_value = investment * var_percent
                    
                    # CVaR
                    cvar_percent = returns[returns <= var_percent].mean()
                    cvar_value = investment * cvar_percent
                    
                    # Display Metrics
                    m1, m2 = st.columns(2)
                    m1.metric(f"VaR ({confidence_level:.1%})", f"${abs(var_value):,.2f}", f"{var_percent:.2%}", delta_color="inverse")
                    m2.metric(f"CVaR (Expected Shortfall)", f"${abs(cvar_value):,.2f}", f"{cvar_percent:.2%}", delta_color="inverse")
                    
                    st.caption(f"This means with {confidence_level:.1%} confidence, you will not lose more than **${abs(var_value):,.2f}** in a day. "
                               f"If you do exceed that loss, the expected average loss is **${abs(cvar_value):,.2f}**.")
                    
                    # Plot Histogram
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(x=returns, nbinsx=50, name='Returns', marker_color='#2b6cb0'))
                    fig_hist.add_vline(x=var_percent, line_width=3, line_dash="dash", line_color="red", annotation_text="VaR")
                    fig_hist.update_layout(title=f'Historical Returns Distribution ({history_years} Years)', 
                                           xaxis_title='Daily Return', yaxis_title='Frequency', height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # --- DATA TABLE & EXPORT ---
                    st.subheader("Historical Data")
                    with st.expander("View & Download Data"):
                        st.dataframe(df.sort_index(ascending=False), use_container_width=True)
                        
                        csv = df.to_csv().encode('utf-8')
                        st.download_button(
                            label="📥 Download Data as CSV",
                            data=csv,
                            file_name=f"{portfolio_ticker}_historical_data.csv",
                            mime='text/csv',
                        )
                
        except Exception as e:
            st.error(f"Error calculating Risk Metrics: {e}")

# ==========================================
# TAB 3: MONTE CARLO STRESS TEST
# ==========================================
elif nav == "Monte Carlo Stress Test":
    st.header("Monte Carlo Stress Testing")
    
    col_mc1, col_mc2 = st.columns([1, 2])
    
    with col_mc1:
        st.subheader("Simulation Config")
        mc_ticker = st.text_input("Asset Ticker", "SPY", key="mc_ticker")
        simulations = st.slider("Number of Simulations", 100, 5000, 1000)
        time_horizon = st.slider("Time Horizon (Days)", 30, 365, 252)
        
        # Advanced Params
        st.markdown("---")
        st.caption("Advanced Parameters (Optional)")
        override_vol = st.checkbox("Override Volatility?")
        manual_vol = st.slider("Manual Volatility (%)", 1.0, 100.0, 20.0) / 100
        
    with col_mc2:
        if st.button("Run Simulation 🎲"):
            with st.spinner("Simulating Future Paths..."):
                try:
                    # Get Data
                    data = yf.download(mc_ticker, period="1y", progress=False)
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                        
                    last_price = data['Close'].iloc[-1]
                    log_returns = np.log(1 + data['Close'].pct_change())
                    
                    u = log_returns.mean()
                    var = log_returns.var()
                    drift = u - (0.5 * var)
                    
                    st_dev = log_returns.std()
                    if override_vol:
                        st_dev = manual_vol / np.sqrt(252)
                    
                    # Generate Paths
                    daily_returns = np.exp(drift + st_dev * norm.ppf(np.random.rand(time_horizon, simulations)))
                    price_paths = np.zeros_like(daily_returns)
                    price_paths[0] = last_price
                    
                    for t in range(1, time_horizon):
                        price_paths[t] = price_paths[t-1] * daily_returns[t]
                    
                    # Visualization
                    fig_mc = go.Figure()
                    for i in range(min(simulations, 100)):
                        fig_mc.add_trace(go.Scatter(y=price_paths[:, i], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
                        
                    mean_path = np.mean(price_paths, axis=1)
                    fig_mc.add_trace(go.Scatter(y=mean_path, mode='lines', name='Mean Path', line=dict(color='black', width=3)))
                    
                    fig_mc.update_layout(title=f"Monte Carlo Simulation ({simulations} Paths)", 
                                         xaxis_title="Days", yaxis_title="Price ($)", height=500)
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    # Stats
                    final_prices = price_paths[-1]
                    expected_price = np.mean(final_prices)
                    worst_case = np.percentile(final_prices, 5)
                    best_case = np.percentile(final_prices, 95)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Expected Price", f"${expected_price:.2f}")
                    c2.metric("95% Worst Case", f"${worst_case:.2f}", delta_color="inverse")
                    c3.metric("95% Best Case", f"${best_case:.2f}")
                    
                    # --- EXPORT SIMULATION ---
                    st.subheader("Export Results")
                    path_df = pd.DataFrame(price_paths)
                    csv_mc = path_df.to_csv().encode('utf-8')
                    st.download_button(
                        label="📥 Download Simulation Paths",
                        data=csv_mc,
                        file_name=f"{mc_ticker}_monte_carlo_paths.csv",
                        mime='text/csv',
                    )
                    
                except Exception as e:
                    st.error(f"Simulation Failed: {e}")

# ==========================================
# TAB 5: PORTFOLIO COMPARISON
# ==========================================
elif nav == "Portfolio Comparison":
    st.header("Portfolio Risk Comparison")
    
    col_p1, col_p2 = st.columns([1, 3])
    
    with col_p1:
        st.subheader("Config")
        # Allow custom input + default list
        default_tickers = ["SPY", "QQQ", "IWM", "GLD", "TLT", "TSLA", "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "AMD", "NFLX"]
        tickers = st.multiselect("Select Assets (Type to search or add)", default_tickers, default=["SPY", "QQQ"])
        
        # Custom Ticker Input
        new_ticker = st.text_input("Add Custom Ticker (e.g., BRK-B)", "").upper()
        if new_ticker and new_ticker not in tickers:
             tickers.append(new_ticker)
             
        comparison_period = st.slider("Lookback Period (Years)", 1, 5, 1)
        
    with col_p2:
        if len(tickers) > 0:
            try:
                # Fetch Data for All Tickers
                start_date = pd.to_datetime("today") - pd.DateOffset(years=comparison_period)
                data = yf.download(tickers, start=start_date, progress=False)['Close']
                
                # Calculate Returns & Volatility
                returns = data.pct_change().dropna()
                
                # Annualized Metrics
                metrics = pd.DataFrame()
                metrics['Return'] = returns.mean() * 252
                metrics['Volatility'] = returns.std() * np.sqrt(252)
                metrics['Sharpe Ratio'] = metrics['Return'] / metrics['Volatility'] # Simple Sharpe (Rf=0)
                metrics['Max Drawdown'] = (data / data.cummax() - 1).min()
                
                # Display Metrics Table
                st.subheader("Risk-Return Metrics (Annualized)")
                st.dataframe(metrics.style.format("{:.2%}"), use_container_width=True)
                
                # Charts
                st.subheader("Performance Comparison")
                
                # Normalized Performance Chart
                normalized_prices = data / data.iloc[0] * 100
                fig_perf = go.Figure()
                for ticker in tickers:
                    fig_perf.add_trace(go.Scatter(x=normalized_prices.index, y=normalized_prices[ticker], name=ticker))
                
                fig_perf.update_layout(title="Normalized Performance (Base=100)", xaxis_title="Date", yaxis_title="Value")
                st.plotly_chart(fig_perf, use_container_width=True)
                
                # Risk-Return Scatter Plot
                fig_risk = go.Figure()
                fig_risk.add_trace(go.Scatter(
                    x=metrics['Volatility'], 
                    y=metrics['Return'], 
                    mode='markers+text',
                    text=metrics.index,
                    textposition="top center",
                    marker=dict(size=12, color=metrics['Sharpe Ratio'], colorscale='Viridis', showscale=True)
                ))
                fig_risk.update_layout(title="Risk vs Return (Efficient Frontier View)", xaxis_title="Volatility (Risk)", yaxis_title="Annual Return")
                st.plotly_chart(fig_risk, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error fetching data: {e}")
        else:
            st.info("Please select at least one asset to compare.")
