# PreciousPulse 💎

## Project Goal
Live tracking and AI-powered prediction of precious metal prices (Gold, Silver, Copper) using a hybrid of technical analysis (LSTM Neural Networks) and fundamental analysis (News Sentiment).

## Tech Stack
- **Framework:** Streamlit
- **Market Data:** `yfinance`
- **News Source:** Google News RSS
- **Sentiment Analysis:** `NLTK` (VADER)
- **AI Model:** `TensorFlow/Keras` (LSTM)
- **Visualization:** `Plotly`

## Features
- **Live Dashboard:** Real-time prices for Gold (`GC=F`), Silver (`SI=F`), and Copper (`HG=F`).
- **Sentiment Analysis:** Scans top news headlines to gauge market mood (Bullish/Bearish).
- **AI Prediction:** Uses a lightweight LSTM neural network to forecast the next closing price.
