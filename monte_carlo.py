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
        S, K, T, r, sigma: Standard option parameters
        num_simulations: Number of simulations
        option_type: 'call' or 'put'
        
        Returns:
        option_price: Price of the binary option (pays $1 if in-the-money)
        """
        Z = np.random.standard_normal(num_simulations)
        ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        if option_type.lower() == 'call':
            payoffs = np.where(ST > K, 1.0, 0.0)
        else:
            payoffs = np.where(ST < K, 1.0, 0.0)
            
        return np.mean(payoffs) * np.exp(-r * T)

    def plot_simulated_paths(self, price_paths, T, K, option_type='call'):
        """Plot the simulated price paths"""
        num_steps = price_paths.shape[1] - 1
        time_axis = np.linspace(0, T, num_steps + 1)
        
        plt.figure(figsize=(12, 6))
        for i in range(price_paths.shape[0]):
            plt.plot(time_axis, price_paths[i, :], lw=1, alpha=0.6)
            
        plt.axhline(y=K, color='r', linestyle='--', label=f'Strike Price (K=${K})')
        plt.title(f'Monte Carlo Simulation: {price_paths.shape[0]} Price Paths')
        plt.xlabel('Time to Maturity (Years)')
        plt.ylabel('Stock Price ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        return plt.gcf()

if __name__ == "__main__":
    mc_model = MonteCarloModel()
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    
    price, ci, paths = mc_model.price_option(S, K, T, r, sigma, num_simulations=10000, paths=True, seed=42)
    
    print(f"Monte Carlo Price: ${price:.4f}")
    print(f"95% Confidence Interval: (${ci[0]:.4f}, ${ci[1]:.4f})")
    
    fig = mc_model.plot_simulated_paths(paths, T, K)
    fig.savefig('monte_carlo_paths.png')
    print("Paths plot saved as 'monte_carlo_paths.png'")
