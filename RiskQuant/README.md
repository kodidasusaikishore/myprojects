# RiskQuant 📉

**Advanced Derivatives & Risk Analytics Dashboard**

RiskQuant is an institutional-grade financial dashboard built with Streamlit, Python, and TensorFlow. It allows users to price options, analyze risk metrics, and stress-test portfolios using advanced quantitative methods.

## 🚀 Key Features

### 1. 📊 Options Pricing (Black-Scholes)
*   Real-time pricing of Call/Put options using the **Black-Scholes-Merton** model.
*   Live calculation of **The Greeks** ($\Delta, \Gamma, \Theta, \nu, \rho$).
*   Interactive **3D Volatility Surface** visualization.
*   Integrated **Technical Analysis** charts (Candlesticks, RSI, Moving Averages).

### 2. 🧠 AI Options Pricer (Neural Network)
*   Uses a pre-trained **TensorFlow/Keras** Neural Network to predict option premiums.
*   Incorporates **Market Sentiment** (The "Alpha Factor") to adjust pricing beyond standard math models.
*   Compares "AI Price" vs "Black-Scholes Price" to find market inefficiencies.

### 3. ⚠️ VaR / CVaR Analysis
*   Calculates **Value at Risk (VaR)** and **Conditional VaR (Expected Shortfall)**.
*   Uses historical simulation (1-10 years) to estimate tail risk.
*   Visualizes the return distribution with risk cutoff zones.

### 4. 🎲 Monte Carlo Stress Test
*   Simulates **1,000+ future price paths** using Geometric Brownian Motion (GBM).
*   Calculates **Expected Price**, **95% Worst Case**, and **95% Best Case** scenarios.
*   Allows manual volatility overrides for stress testing.

### 5. ⚖️ Portfolio Comparison
*   Compare multiple assets (e.g., SPY vs QQQ vs NVDA) side-by-side.
*   Visualizes the **Risk-Return Tradeoff** (Efficient Frontier view).
*   Metrics: Annualized Return, Volatility, Sharpe Ratio, Max Drawdown.

## 🛠️ Tech Stack
*   **Frontend:** Streamlit
*   **Data:** Yahoo Finance (`yfinance`)
*   **Math:** `scipy`, `numpy`
*   **AI/ML:** `TensorFlow`, `Keras`
*   **Visualization:** `Plotly`

## 📦 Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```
