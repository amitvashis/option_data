import numpy as np
from scipy.stats import norm

def implied_volatility(market_price, S, K, T, r, option_type="call",
                       tol=1e-6, max_iter=100):

    sigma = 0.3  # initial guess
    
    for _ in range(max_iter):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == "call":
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        diff = price - market_price
        
        if abs(diff) < tol:
            return sigma
        
        sigma -= diff / vega
    
    raise ValueError("IV did not converge")
vi=implied_volatility(727.8, 22161, 22000, 31/365, 0.04, option_type="call")
print("Implied Volatility:", vi)