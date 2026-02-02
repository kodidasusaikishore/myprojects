import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

# Set page config immediately
st.set_page_config(page_title="Stock Price Prediction", layout="wide", page_icon="📈")

# Custom CSS for Premium Look
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: #0f0c29;  /* fallback for old browsers */
        background: -webkit-linear-gradient(to right, #24243e, #302b63, #0f0c29);  /* Chrome 10-25, Safari 5.1-6 */
        background: linear-gradient(to right, #24243e, #302b63, #0f0c29); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Title Gradient */
    h1 {
        background: linear-gradient(120deg, #00f260 0%, #0575E6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-shadow: 0 0 30px rgba(5, 117, 230, 0.3);
    }

    /* Subheader & Text */
    h2, h3 {
        color: #e0e0e0;
        font-weight: 600;
    }
    p {
        color: #b0b0b0;
    }

    /* Dataframes / Tables */
    div[data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Inputs */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.05);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
    }
    .stDateInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.05);
        color: white;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #000;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 201, 255, 0.3);
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 201, 255, 0.5);
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #00f260;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock Price Prediction using LSTM")
st.markdown(f"Predicting **{ticker}** stock prices using Long Short-Term Memory (LSTM) neural networks.")

# Sidebar
st.sidebar.header("⚙️ Configuration")
ticker = st.sidebar.text_input("Stock Ticker", "RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
epochs = st.sidebar.slider("Training Epochs", min_value=1, max_value=50, value=10)

# Fetch Data
@st.cache_data(ttl=300) # Cache for 5 mins to prevent rate limiting
def load_data(ticker, start, end):
    try:
        # Ensure dates are strings for yfinance stability
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        
        data = yf.download(ticker, start=start_str, end=end_str, progress=False)
        
        # Flatten MultiIndex columns if present (common in new yfinance)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        return data
    except Exception as e:
        return pd.DataFrame()

# Manual Refresh Button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

data_load_state = st.text('Loading data...')
data = load_data(ticker, start_date, end_date)
data_load_state.text('Loading data... done!')

if data.empty:
    st.error(f"No data found for **{ticker}** from {start_date} to {end_date}.")
    st.warning("Possible reasons:\n1. Invalid Ticker Symbol.\n2. Yahoo Finance API Rate Limit (Try again in 1 min).\n3. Weekend/Holiday gaps (Try adjusting dates).")
    st.stop()

# Display Raw Data
st.subheader('Raw Data')
st.write(data.tail())

# Plot Raw Data
st.subheader("📊 Interactive Financial Charts")

# Calculate Moving Averages
data['MA50'] = data['Close'].rolling(50).mean()
data['MA200'] = data['Close'].rolling(200).mean()

tab1, tab2, tab3 = st.tabs(["Line Chart", "Candlestick Chart", "Technical Indicators"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Stock Close", line=dict(color='#00F260')))
    fig.layout.update(
        title_text='Time Series Data with Rangeslider', 
        xaxis_rangeslider_visible=True,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig_candle = go.Figure()
    fig_candle.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='OHLC'
    ))
    fig_candle.layout.update(
        title_text='Candlestick Chart',
        xaxis_rangeslider_visible=True,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_candle, use_container_width=True)

with tab3:
    fig_tech = go.Figure()
    fig_tech.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close", line=dict(color='#00F260', width=1)))
    fig_tech.add_trace(go.Scatter(x=data.index, y=data['MA50'], name="50 Day MA", line=dict(color='#FF4B4B', width=1.5)))
    fig_tech.add_trace(go.Scatter(x=data.index, y=data['MA200'], name="200 Day MA", line=dict(color='#00C9FF', width=1.5)))
    
    fig_tech.layout.update(
        title_text='Moving Averages (MA50 & MA200)',
        xaxis_rangeslider_visible=True,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    st.plotly_chart(fig_tech, use_container_width=True)

# Data Preprocessing
st.subheader('Model Training & Prediction')

if st.button('Train Model'):
    # Lazy import TensorFlow and Scikit-learn to prevent startup lag/blank screen
    with st.spinner('Initializing AI engine (TensorFlow) and processing data...'):
        try:
            import tensorflow as tf
            from sklearn.preprocessing import MinMaxScaler
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, LSTM, Dropout
        except ImportError as e:
            st.error(f"Failed to import libraries: {e}")
            st.stop()

        # Use 'Close' price for prediction
        dataset = data['Close'].values.reshape(-1, 1)

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # Create sequences
        def create_sequences(data, seq_length=60):
            x_train, y_train = [], []
            for i in range(seq_length, len(data)):
                x_train.append(data[i-seq_length:i, 0])
                y_train.append(data[i, 0])
            return np.array(x_train), np.array(y_train)

        sequence_length = 60
        training_data_len = int(np.ceil(len(dataset) * .80))

        if len(dataset) <= sequence_length:
             st.error("Not enough data points to train model. Choose a longer date range.")
             st.stop()

        train_data = scaled_data[0:training_data_len, :]
        x_train, y_train = create_sequences(train_data, sequence_length)

        # Reshape for LSTM [samples, time steps, features]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build LSTM Model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(x_train, y_train, batch_size=32, epochs=epochs)
        st.success('Training complete!')

        # Create test data
        test_data = scaled_data[training_data_len - sequence_length:, :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(sequence_length, len(test_data)):
            x_test.append(test_data[i-sequence_length:i, 0])
        
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Get predictions
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Calculate RMSE
        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

        # Plot Predictions
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=train.index, y=train['Close'], name='Training Data', line=dict(color='#888888')))
        fig2.add_trace(go.Scatter(x=valid.index, y=valid['Close'], name='Actual Price', line=dict(color='#00C9FF')))
        fig2.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], name='Predicted Price', line=dict(color='#FF4B4B')))
        
        fig2.layout.update(
            title_text='Model Predictions vs Actual', 
            xaxis_rangeslider_visible=True,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Predicted Prices vs Actual Prices")
        st.write(valid[['Close', 'Predictions']].tail(10))
