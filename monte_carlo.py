import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

class MonteCarloModel:
    def __init__(self):
        """Initialize the Monte Carlo Option Pricing Model"""
        pass
    
    def price_option(self, S, K, T, r, sigma, num_simulations=10000, option_type='call', paths=False, seed=None):
        """
        Standard Monte Carlo Option Pricing Model
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free interest rate (annual)
        sigma: Volatility of the underlying asset (annual)
        num_simulations: Number of price path simulations
        option_type: 'call' or 'put'
        paths: Whether to return generated price paths
        seed: Random seed for reproducibility
        
        Returns:
        option_price: Estimated price of the option
        confidence_interval: 95% confidence interval for the estimate
        paths: (optional) Array of simulated price paths
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random normal returns
        Z = np.random.standard_normal(num_simulations)
        
        # Calculate final stock prices
        ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        # Calculate payoffs at expiration
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:  # put
            payoffs = np.maximum(K - ST, 0)
            
        # Discount payoffs to present value
        discount_factor = np.exp(-r * T)
        discounted_payoffs = payoffs * discount_factor
        
        # Calculate option price as the average of discounted payoffs
        option_price = np.mean(discounted_payoffs)
        
        # Calculate 95% confidence interval
        std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(num_simulations)
        confidence_interval = (option_price - 1.96 * std_error, option_price + 1.96 * std_error)
        
        if paths:
            # Generate paths for visualization (using fewer paths for efficiency)
            num_paths_to_show = min(100, num_simulations)
            num_steps = 100  # Number of steps per path
            dt = T / num_steps
            
            # Initialize array for paths
            price_paths = np.zeros((num_paths_to_show, num_steps + 1))
            price_paths[:, 0] = S
            
            # Generate paths
            for i in range(num_steps):
                Z = np.random.standard_normal(num_paths_to_show)
                price_paths[:, i+1] = price_paths[:, i] * np.exp(
                    (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            
            return option_price, confidence_interval, price_paths
        
        return option_price, confidence_interval
    
    def price_binary_option(self, S, K, T, r, sigma, num_simulations=10000, option_type='call'):
        """
        Monte Carlo pricing for binary (digital) options
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to maturity in years