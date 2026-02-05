import numpy as np
from scipy.stats import norm

# --- Black-Scholes Formula ---
def black_scholes(S, K, T, r, sigma, type="call"):
    """
    S: Spot Price
    K: Strike Price
    T: Time to Maturity (in years)
    r: Risk-free rate
    sigma: Volatility (IV)
    type: 'call' or 'put'
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# --- Greeks Calculation ---
def calculate_greeks(S, K, T, r, sigma, type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
        
    # Gamma (Same for Call & Put)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Vega (Same for Call & Put) - Sensitivity to Volatility
    vega = S * norm.pdf(d1) * np.sqrt(T) * 0.01 # Scaled for 1% change
    
    # Theta (Time Decay)
    if type == "call":
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
                 
    # Rho (Interest Rate)
    if type == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01
        
    return {
        "Delta": delta,
        "Gamma": gamma,
        "Theta": theta,
        "Vega": vega,
        "Rho": rho
    }
