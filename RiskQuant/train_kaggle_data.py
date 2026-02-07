import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import glob

# --- CONFIG ---
DATA_DIR = 'data' # Looks for CSVs here
MODEL_PATH = 'option_pricing_model.keras'

# Define Feature Columns Mapping based on your Kaggle Dataset
# We need to map these to: S, K, T, r, sigma, sentiment
COL_MAP = {
    'S': ' [UNDERLYING_LAST]',       # Spot Price
    'K': ' [STRIKE]',                # Strike Price
    'T': ' [DTE]',                   # Days to Expiry (Need to convert to Years)
    'r': 0.045,                      # Risk-free rate (Approx 4.5% or load from external)
    'sigma_call': ' [C_IV]',         # Implied Volatility (Call)
    'sigma_put': ' [P_IV]',          # Implied Volatility (Put)
    'price_call': ' [C_LAST]',       # Target: Call Price (Last Traded) - or use (Bid+Ask)/2
    'price_put': ' [P_LAST]'         # Target: Put Price
}

def load_and_process_data():
    print("📂 Scanning for CSV files in 'data/'...")
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if not files:
        print("❌ No CSV files found! Please place 'spy_options.csv' in RiskQuant/data/")
        return None, None

    all_data = []
    
    for file in files:
        print(f"   Loading {file}...")
        try:
            df = pd.read_csv(file)
            
            # --- PREPROCESSING ---
            
            # 1. Clean Column Names (Strip whitespace)
            df.columns = df.columns.str.strip()
            
            # --- TYPE CONVERSION FIX ---
            # Force numeric types, coercing errors to NaN
            cols_to_numeric = ['[C_VOLUME]', '[C_IV]', '[UNDERLYING_LAST]', '[STRIKE]', '[DTE]', '[C_LAST]']
            for col in cols_to_numeric:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows with NaN in critical columns
            df.dropna(subset=cols_to_numeric, inplace=True)

            # 2. Filter Valid Rows (Volume > 0 to ensure liquidity/real price)
            # Using [C_VOLUME] or [P_VOLUME] > 0
            
            # --- PREPARE CALL DATA ---
            # Features: S, K, T, r, sigma, sentiment
            # We'll treat sentiment as 0.0 (Neutral) for historical data training
            # T = DTE / 365.0
            
            # Filter for Calls with valid data
            calls = df[ (df['[C_VOLUME]'] > 0) & (df['[C_IV]'] > 0) ].copy()
            
            # Construct Inputs (X)
            X_calls = pd.DataFrame()
            X_calls['S'] = calls['[UNDERLYING_LAST]']
            X_calls['K'] = calls['[STRIKE]']
            X_calls['T'] = calls['[DTE]'] / 365.0
            X_calls['r'] = COL_MAP['r'] # Constant rate
            X_calls['sigma'] = calls['[C_IV]']
            X_calls['sentiment'] = 0.0 # Placeholder
            
            # Construct Target (y)
            # Use Mid-point if Bid/Ask available, else Last
            # calls['price'] = (calls[' [C_BID]'] + calls[' [C_ASK]']) / 2
            X_calls['target_price'] = calls['[C_LAST]']
            
            all_data.append(X_calls)
            
            # (Optional: You could also train on Puts, but RiskQuant is primarily Call-focused currently?
            # Let's stick to Calls for consistency with the current BS model)
            
        except Exception as e:
            print(f"⚠️ Error reading {file}: {e}")
            continue

    if not all_data:
        return None, None
        
    final_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ Loaded {len(final_df)} valid option records.")
    
    # Split X, y
    X = final_df[['S', 'K', 'T', 'r', 'sigma', 'sentiment']].values
    y = final_df['target_price'].values
    
    return X, y

def train_kaggle_model():
    X, y = load_and_process_data()
    
    if X is None:
        print("❌ Training Aborted.")
        return

    # Normalization (Crucial for Neural Networks)
    # Ideally save the scaler too, but for now we'll rely on the model learning the scale
    # or assume inputs are somewhat reasonable. 
    # For robust production, use sklearn StandardScaler.
    
    print("🧠 Building Neural Network...")
    
    # Same architecture as before to maintain compatibility
    model = keras.Sequential([
        layers.Input(shape=(6,)), # S, K, T, r, sigma, sentiment
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2), # Prevent overfitting on specific historical patterns
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1) # Output: Option Price
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("🚀 Training on Real Market Data...")
    # Use validation split to monitor performance
    history = model.fit(
        X, y, 
        epochs=50, 
        batch_size=2048, # Larger batch size for large CSVs
        validation_split=0.2, 
        verbose=1
    )
    
    # Save
    model.save(MODEL_PATH)
    print(f"💾 Real-Data Model saved to {MODEL_PATH}")
    
    # Quick Test
    test_loss = model.evaluate(X[:100], y[:100], verbose=0)
    print(f"🔍 Test MAE (Mean Abs Error): ${test_loss[1]:.2f}")

if __name__ == "__main__":
    train_kaggle_model()
