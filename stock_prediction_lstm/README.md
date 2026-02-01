# Stock Price Prediction with LSTM (Reliance Industries)

## Project Goal
Predict the future stock price of Reliance Industries (RELIANCE.NS) using Long Short-Term Memory (LSTM) networks.

## Tech Stack
- **Data Source:** `yfinance` (Yahoo Finance)
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `tensorflow` (Keras), `scikit-learn`
- **Visualization:** `matplotlib`, `plotly`
- **UI (Optional):** `streamlit` for an interactive dashboard

## Steps
1. **Data Collection:** Fetch historical data for 'RELIANCE.NS'.
2. **Preprocessing:** 
   - Normalize data (MinMaxScaler).
   - Create sequences (e.g., use last 60 days to predict next day).
3. **Model Building:**
   - LSTM layers.
   - Dense output layer.
4. **Training:** Train on historical data.
5. **Prediction & Visualization:** Compare predicted vs actual prices.
