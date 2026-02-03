import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Page Config
st.set_page_config(page_title="PreciousPulse", layout="wide", page_icon="💎")

# Download VADER lexicon
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# --- Custom Premium Dark CSS ---
st.markdown("""
<style>
    /* Gradient Background */
    .stApp {
        background: #000000;
        background: linear-gradient(to bottom, #0f2027, #203a43, #2c5364);
        color: white;
    }
    
    /* Force Sidebar Vertical Spacing */
    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] {
        margin-bottom: 20px !important;
        padding-bottom: 10px !important;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Sidebar Text Fix - Universal */
    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3, 
    section[data-testid="stSidebar"] .stMarkdown p, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stRadio div[role='radiogroup'] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] h2 {
        color: white !important;
    }

    /* Header Elements (Deploy, Stop, Menu, Running Man) */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    header[data-testid="stHeader"] * {
        color: white !important;
    }
    header[data-testid="stHeader"] svg {
        fill: white !important;
        stroke: white !important;
    }
    
    /* Running Animation Status */
    div[data-testid="stStatusWidget"] {
        color: white !important;
    }
    div[data-testid="stStatusWidget"] svg {
        fill: white !important;
    }

    /* Sidebar Toggle Arrow (Open & Closed) - Nuclear */
    button[kind="header"] svg,
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stSidebarExpandedControl"] svg {
        fill: #ffffff !important;
        stroke: #ffffff !important;
        color: #ffffff !important;
    }
    
    /* Just in case it's an icon font */
    [data-testid="stSidebarCollapsedControl"] i,
    [data-testid="stSidebarExpandedControl"] i {
        color: #ffffff !important;
    }
    
    /* Force background transparent to see white arrow */
    [data-testid="stSidebarCollapsedControl"] {
        background: rgba(0,0,0,0.2) !important;
        border-radius: 50%;
    }
    div[data-testid="stSidebarNav"] svg {
        fill: white !important;
    }
    
    /* Hide Streamlit Top Header/Navigation Bar */
    header[data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
        backdrop-filter: blur(0px);
    }    
    /* Fix Selectbox Input Background/Text */
    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    div[data-baseweb="select"] span {
        color: white !important;
    }
    
    /* Fix Radio Button Text (Global) */
    .stRadio label {
        color: white !important;
        font-weight: 600;
        font-size: 16px;
    }
    .stRadio div[role='radiogroup'] > label {
        color: white !important;
    }
    div[data-testid="stMarkdownContainer"] p {
        color: white !important;
    }

    /* Title Styling */
    h1 {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: 1px;
    }

    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        color: #00e5ff;
        font-weight: 700;
    }
    
    /* Force White Label Text - Super Aggressive */
    div[data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }
    div[data-testid="stMetricLabel"] * {
        color: #ffffff !important;
    }
    label[data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }
    div[data-testid="stMetric"] label {
        color: #ffffff !important;
    }

    /* Buttons & Download Button - Super Aggressive */
    .stButton > button, 
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(90deg, #1D976C 0%, #93F9B9 100%) !important;
        color: #000000 !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 20px !important;
        transition: transform 0.2s;
    }
    .stButton > button:hover, 
    div[data-testid="stDownloadButton"] > button:hover {
        transform: scale(1.05);
        color: #000000 !important;
    }
    
    /* Ensure child elements (like p tags inside button) inherit color */
    .stButton > button *, 
    div[data-testid="stDownloadButton"] > button * {
        color: #000000 !important;
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        color: white;
        font-size: 14px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        border: none;
        color: white;
    }
    
    /* Reduce Gap between Radio and Content */
    div[class*="stRadio"] {
        margin-bottom: -25px !important;
        padding-bottom: 0px !important;
    }
    hr {
        margin-top: 5px !important;
        margin-bottom: 5px !important;
        border-color: rgba(255,255,255,0.1);
    }
    div[data-testid="stVerticalBlock"] > div.stSubheader {
        margin-top: -20px !important;
        padding-top: 0px !important;
    }
    
    /* Expander Spacing Fix */
    div[data-testid="stExpander"] {
        margin-bottom: 20px !important;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 8px;
    }
    .streamlit-expanderHeader {
        background-color: rgba(0,0,0,0.2);
    }
    /* Nuclear fix for gap reduction */
    .stRadio + div {
        margin-top: -30px !important;
    }
    /* Move main content up */
    .block-container {
        padding-top: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("💎 PreciousPulse")
st.markdown("### AI-Powered Bullion Tracker & Predictor")

# --- Sidebar Configuration (Moved up so variables are available) ---
st.sidebar.header("⚙️ Market Config")
metal_choice = st.sidebar.radio("Select Asset", ["Gold 🟡", "Silver ⚪", "Copper 🟠"])

# Add LARGE spacing to force separation
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

period = st.sidebar.selectbox("History Period", ["1mo", "3mo", "6mo", "1y", "5y"], index=3)

# Unit Label Logic
if "Gold" in metal_choice:
    unit = "₹/10g"
elif "Silver" in metal_choice:
    unit = "₹/1kg"
elif "Copper" in metal_choice:
    unit = "₹/1kg"
else:
    unit = "₹"

# Disclaimer in a clean expander
with st.expander("ℹ️ How are these prices calculated?"):
    st.caption(f"Prices in the metrics section are estimated by converting the Global Spot Rate (USD) to **{unit}** using real-time FX rates. Actual local market prices (MCX/Retail) may vary slightly due to import duties and taxes.")

# Clean up old duplicate
# st.info(...) removed

# Mapping
tickers = {
    "Gold 🟡": "GC=F", 
    "Silver ⚪": "SI=F",
    "Copper 🟠": "HG=F"
}
ticker_symbol = tickers[metal_choice]

# --- Helper Functions ---

def convert_to_indian_standards(price_usd, metal, exchange_rate):
    # Conversion Factors
    # 1 Troy Ounce = 31.1034768 grams
    # 1 Pound = 0.453592 kg
    
    price_inr = price_usd * exchange_rate
    
    if "Gold" in metal:
        # Convert Price/Oz to Price/10g
        # (Price / 31.1035) * 10
        return (price_inr / 31.1034768) * 10
    elif "Silver" in metal:
        # Convert Price/Oz to Price/1kg
        # (Price / 31.1035) * 1000
        return (price_inr / 31.1034768) * 1000
    elif "Copper" in metal:
        # Convert Price/Lbs to Price/1kg
        # Price / 0.453592
        return price_inr / 0.453592
    
    return price_inr

@st.cache_data(ttl=300)
def get_price_data(ticker, period):
    data = yf.download(ticker, period=period)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

@st.cache_data(ttl=300)
def get_exchange_rate():
    try:
        # Fetch USD to INR rate
        ticker = "INR=X"
        data = yf.download(ticker, period="1d")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        rate = data['Close'].iloc[-1]
        
        # Sanity check: If rate is anomalously high or low (yfinance sometimes returns crazy data)
        # e.g., if it returns 4000, it's wrong. Range should be 80-90.
        if rate < 50 or rate > 100:
             return 86.5 # Safe fallback
             
        return rate
    except:
        return 86.5 # Fallback

@st.cache_data(ttl=600)
def get_news_sentiment(query):
    # Fetch News
    clean_query = query.split(" ")[0] # Remove emoji
    url = f"https://news.google.com/rss/search?q={clean_query}+price+market&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    entries = feed.entries[:10]
    
    # Analyze Sentiment
    sia = SentimentIntensityAnalyzer()
    sentiments = []
    for item in entries:
        text = item.title + " " + item.description
        score = sia.polarity_scores(text)
        sentiments.append(score['compound'])
    
    avg_score = sum(sentiments) / len(sentiments) if sentiments else 0
    return entries, avg_score

# --- Main Logic ---

data = get_price_data(ticker_symbol, period)
exchange_rate = get_exchange_rate() # Get INR rate
news_items, sentiment_score = get_news_sentiment(metal_choice)

# Display Exchange Rate in Sidebar
st.sidebar.markdown("---")
st.sidebar.metric("USD/INR Rate", f"₹{exchange_rate:.2f}")

# Convert DataFrame to INR (Approximate for display)
# Note: Using current exchange rate for historical data is an approximation for visualization consistency
data_inr = data.copy()
# We won't use this whole dataframe for charts anymore (charts are USD), 
# but we need it for metrics and calculations.
# Applying weight conversion to the *latest* values primarily.

# --- Top Metrics Row ---
col1, col2, col3, col4 = st.columns(4)

# Latest USD Prices
current_price_usd = data['Close'].iloc[-1]
prev_price_usd = data['Close'].iloc[-2]
day_high_usd = data['High'].iloc[-1]
day_low_usd = data['Low'].iloc[-1]

# Convert to Indian Standard Weights
current_price_inr = convert_to_indian_standards(current_price_usd, metal_choice, exchange_rate)
prev_price_inr = convert_to_indian_standards(prev_price_usd, metal_choice, exchange_rate)
day_high_inr = convert_to_indian_standards(day_high_usd, metal_choice, exchange_rate)
day_low_inr = convert_to_indian_standards(day_low_usd, metal_choice, exchange_rate)

price_diff_inr = current_price_inr - prev_price_inr
pct_diff = (price_diff_inr / prev_price_inr) * 100

sentiment_label = "Neutral ⚖️"
sentiment_color = "off"
if sentiment_score > 0.05:
    sentiment_label = "Bullish 🚀"
    sentiment_color = "normal"
elif sentiment_score < -0.05:
    sentiment_label = "Bearish 📉"
    sentiment_color = "inverse"

with col1:
    st.metric(f"{metal_choice} ({unit})", f"₹{current_price_inr:,.2f}", f"{pct_diff:.2f}%")
with col2:
    st.metric("Market Sentiment", sentiment_label, f"{sentiment_score:.2f}", delta_color="normal")
with col3:
    st.metric(f"Day High ({unit})", f"₹{day_high_inr:,.2f}")
with col4:
    st.metric(f"Day Low ({unit})", f"₹{day_low_inr:,.2f}")

# --- Tabs for Charts & AI ---
# Use Session State to persist active tab
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "📈 Market Charts"

# Custom Tab Selection using Radio (Styled as Horizontal Tabs if desired, or just use st.radio)
# For now, to solve the reset issue, we can use st.radio instead of st.tabs which rerenders content cleanly
# OR we rely on st.tabs but move the button logic outside? No, that's hard.
# Let's switch to st.radio for navigation to guarantee persistence.

tab_options = ["📈 Market Charts", "🤖 AI Prediction", "📰 News Feed"]
# Use a key that doesn't conflict
selected_tab = st.radio("", tab_options, horizontal=True, label_visibility="collapsed", key="nav_radio")

st.markdown("---")

if selected_tab == "📈 Market Charts":
    # Custom CSS to reduce top spacing
    st.markdown("""
        <style>
            div[data-testid="stVerticalBlock"] > div:nth-child(5) {
                margin-top: -20px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.subheader(f"Price Action: {metal_choice} (USD)")
    st.caption(f"Charts are displayed in **USD** (Global Spot Price) for technical accuracy. Metrics above are in **{unit}**.")
    
    # 1. Candlestick Chart
    st.markdown("#### 🕯️ Candlestick Chart")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                    open=data['Open'], high=data['High'],
                    low=data['Low'], close=data['Close'], name='OHLC'))
    
    # Add Moving Averages
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], line=dict(color='orange', width=1), name='MA 20'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], line=dict(color='cyan', width=1), name='MA 50'))

    fig.layout.update(template="plotly_dark", xaxis_rangeslider_visible=False, 
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                      height=500, margin=dict(l=20, r=20, t=20, b=20),
                      legend=dict(font=dict(color="white")))
    st.plotly_chart(fig, use_container_width=True)

    # 2. Time Series Line Chart
    st.markdown("#### 📉 Time Series Line Chart")
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='#00c6ff', width=2)))
    fig_line.layout.update(template="plotly_dark", xaxis_rangeslider_visible=True,
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           height=400, margin=dict(l=20, r=20, t=20, b=20),
                           legend=dict(font=dict(color="white")))
    st.plotly_chart(fig_line, use_container_width=True)

    # 3. Raw Data Table & Export
    st.markdown("#### 📄 Raw Historical Data (USD)")
    
    # Show dataframe
    st.dataframe(data.sort_index(ascending=False), height=300, use_container_width=True)
    
    # CSV Download Button
    csv = data.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f"{metal_choice.split()[0]}_price_data.csv",
        mime='text/csv',
    )

elif selected_tab == "🤖 AI Prediction":
    st.subheader("🔮 Live AI Price Prediction")
    st.markdown("Click the button below to train a **Lightweight LSTM Neural Network** on the live data and predict the next potential closing price.")
    
    # Check if we have a stored prediction for this session/metal
    pred_key = f"pred_{metal_choice}_{period}"
    
    if st.button("🚀 Run Live Prediction"):
        with st.spinner("Initializing TensorFlow & Training Model..."):
            try:
                import tensorflow as tf
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Dense, LSTM, Dropout
                
                # Preprocessing (Use USD Data for Model Training)
                dataset = data['Close'].values.reshape(-1, 1)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(dataset)
                
                # Create small sequences (Lightweight)
                seq_len = 30 
                if len(dataset) < seq_len + 10:
                    st.error("Not enough data. Please select a longer history period.")
                    st.stop()

                x_train, y_train = [], []
                for i in range(seq_len, len(scaled_data)):
                    x_train.append(scaled_data[i-seq_len:i, 0])
                    y_train.append(scaled_data[i, 0])
                
                x_train, y_train = np.array(x_train), np.array(y_train)
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                
                # Build Lightweight Model
                model = Sequential()
                model.add(LSTM(units=32, return_sequences=False, input_shape=(x_train.shape[1], 1)))
                model.add(Dense(units=1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                
                # Fast Training
                model.fit(x_train, y_train, batch_size=16, epochs=5, verbose=0)
                
                # Predict Next Day
                last_sequence = scaled_data[-seq_len:]
                last_sequence = last_sequence.reshape(1, seq_len, 1)
                predicted_price_scaled = model.predict(last_sequence)
                predicted_price_usd = scaler.inverse_transform(predicted_price_scaled)[0][0]
                
                # Convert Prediction to Indian Standard
                predicted_price_inr = convert_to_indian_standards(predicted_price_usd, metal_choice, exchange_rate)
                
                # Store result in session state
                delta_inr = predicted_price_inr - current_price_inr
                st.session_state[pred_key] = {
                    "price": predicted_price_inr,
                    "delta": delta_inr
                }
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # Display Result if available
    if pred_key in st.session_state:
        res = st.session_state[pred_key]
        st.success("Prediction Complete!")
        
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.metric("Current Price", f"₹{current_price_inr:,.2f}")
        
        with col_p2:
            delta_val = res['delta']
            if delta_val < 0:
                delta_str = f"-₹{abs(delta_val):,.2f}"
            else:
                delta_str = f"+₹{delta_val:,.2f}"
                
            st.metric("AI Predicted Price (Next Close)", f"₹{res['price']:,.2f}", delta_str, 
                      delta_color="normal")

        # Sentiment Impact Logic
        st.markdown("---")
        st.markdown("#### 🧠 Sentiment Integration")
        
        delta_inr = res['delta']
        if sentiment_score > 0.1 and delta_inr < 0:
            st.warning(f"**Conflict Detected:** AI Technicals suggest a drop, but News Sentiment is **Bullish**. The price might not drop as much as predicted.")
        elif sentiment_score < -0.1 and delta_inr > 0:
            st.warning(f"**Conflict Detected:** AI Technicals suggest a rise, but News Sentiment is **Bearish**. Proceed with caution.")
        elif sentiment_score > 0.1 and delta_inr > 0:
            st.success("**Strong Signal:** Both AI Technicals and News Sentiment are **Bullish**! 🚀")
        elif sentiment_score < -0.1 and delta_inr < 0:
            st.error("**Strong Signal:** Both AI Technicals and News Sentiment are **Bearish**! 📉")
        else:
            st.info("Sentiment is Neutral. Rely primarily on Technical Levels.")

elif selected_tab == "📰 News Feed":
    st.subheader("Global Headlines")
    for item in news_items:
        with st.expander(item.title):
            st.write(f"**Published:** {item.published}")
            st.write(item.link)
