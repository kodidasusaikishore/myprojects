import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# --- 1. Generate Synthetic Option Data (Standard Black-Scholes) ---
# We train the NN to "learn" Black-Scholes first (simulating learning market dynamics)
def generate_option_data(n_samples=5000):
    np.random.seed(42)
    
    # Random Inputs
    S = np.random.uniform(50, 200, n_samples)   # Spot Price
    K = np.random.uniform(50, 200, n_samples)   # Strike Price
    T = np.random.uniform(0.1, 2.0, n_samples)  # Time (Years)
    r = np.random.uniform(0.01, 0.05, n_samples)# Risk-free Rate
    sigma = np.random.uniform(0.1, 0.5, n_samples) # Volatility
    sentiment = np.random.uniform(-1, 1, n_samples) # Sentiment (-1 to 1)

    # Calculate BS Price (Target)
    # NOTE: We add a "Sentiment Bias" to simulate market deviation from BS
    # If Sentiment > 0 (Bullish), Calls get expensive, Puts cheaper (Skew)
    
    # Simplified BS for Call
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bs_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    # Market Price = BS Price * (1 + 0.05 * Sentiment)
    # This teaches the AI that high sentiment pumps the premium slightly
    market_price = bs_price * (1 + 0.05 * sentiment)
    
    X = np.stack([S, K, T, r, sigma, sentiment], axis=1)
    y = market_price
    
    return X, y

# --- 2. Build & Train Model ---
def build_and_train_model():
    print("Generating Synthetic Market Data...")
    X, y = generate_option_data()
    
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(6,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1) # Output: Option Price
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    print("Training Neural Network...")
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    
    model.save("option_pricing_model.keras")
    print("Model Saved: option_pricing_model.keras")

if __name__ == "__main__":
    build_and_train_model()
