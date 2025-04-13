import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class BlackScholesModel:
    def __init__(self):
        """Initialize the Black-Scholes Option Pricing Model"""
        pass
    
    def price_option(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate option price and Greeks using Black-Scholes formula
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free interest rate (annual)
        sigma: Volatility of the underlying asset (annual)
        option_type: 'call' or 'put'
        
        Returns:
        option_price: Price of the option
        greeks: Dictionary containing the Greeks
        """
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate option price based on type
        if option_type.lower() == 'call':
            option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            # Calculate Greeks for call options
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = -(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
            vega = S * np.sqrt(T) * norm.pdf(d1)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
            
        elif option_type.lower() == 'put':
            option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            # Calculate Greeks for put options
            delta = norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = -(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
            vega = S * np.sqrt(T) * norm.pdf(d1)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
            
        else:
            raise ValueError("Option type must be 'call' or 'put'")
        
        greeks = {
            'delta': delta,  # Price sensitivity to changes in underlying price
            'gamma': gamma,  # Delta sensitivity to changes in underlying price
            'theta': theta / 365,  # Price sensitivity to time decay (daily)
            'vega': vega / 100,    # Price sensitivity to 1% change in volatility
            'rho': rho / 100       # Price sensitivity to 1% change in interest rate
        }
        
        return option_price, greeks
    
    def implied_volatility(self, market_price, S, K, T, r, option_type='call', max_iterations=100, precision=0.00001):
        """
        Calculate implied volatility using the bisection method
        
        Parameters:
        market_price: Observed price of the option in the market
        S, K, T, r: Standard BS parameters
        option_type: 'call' or 'put'
        max_iterations: Maximum number of iterations for convergence
        precision: Desired precision level
        
        Returns:
        Implied volatility value
        """
        # Set upper and lower bounds for volatility
        sigma_low = 0.001  # 0.1%
        sigma_high = 5.0   # 500%
        
        # Calculate price at bounds
        price_low, _ = self.price_option(S, K, T, r, sigma_low, option_type)
        price_high, _ = self.price_option(S, K, T, r, sigma_high, option_type)
        
        # Check if market price is within bounds
        if market_price <= price_low:
            return sigma_low
        if market_price >= price_high:
            return sigma_high
            
        # Bisection search
        for i in range(max_iterations):
            sigma_mid = (sigma_low + sigma_high) / 2
            price_mid, _ = self.price_option(S, K, T, r, sigma_mid, option_type)
            
            if abs(price_mid - market_price) < precision:
                return sigma_mid
            
            if price_mid < market_price:
                sigma_low = sigma_mid
            else:
                sigma_high = sigma_mid
                
        # Return best estimate after max iterations
        return (sigma_low + sigma_high) / 2
    
    def plot_option_prices(self, S_range, K, T, r, sigma, option_types=None):
        """
        Plot option prices for different underlying prices
        
        Parameters:
        S_range: Range of stock prices to evaluate
        K, T, r, sigma: Standard BS parameters
        option_types: List of option types to plot ('call', 'put', or both)
        """
        if option_types is None:
            option_types = ['call', 'put']
            
        plt.figure(figsize=(10, 6))
        
        for option_type in option_types:
            prices = []
            for S in S_range:
                price, _ = self.price_option(S, K, T, r, sigma, option_type)
                prices.append(price)
                
            plt.plot(S_range, prices, label=f"{option_type.capitalize()} Option")
            
        # Add the payoff at expiration
        if 'call' in option_types:
            plt.plot(S_range, [max(0, S - K) for S in S_range], 'k--', label='Call Payoff at Expiration')
        if 'put' in option_types:
            plt.plot(S_range, [max(0, K - S) for S in S_range], 'r--', label='Put Payoff at Expiration')
            
        plt.axvline(x=K, color='gray', linestyle='--', alpha=0.5)
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.title(f'Option Prices (Strike = ${K}, T = {T} yrs, σ = {sigma*100}%, r = {r*100}%)')
        plt.xlabel('Stock Price ($)')
        plt.ylabel('Option Price ($)')
        plt.legend()
        plt.tight_layout()
        
        return plt.gcf()  # Return the figure for saving or further modifications

    def plot_greeks(self, S_range, K, T, r, sigma, option_type='call', greeks_to_plot=None):
        """
        Plot option Greeks for different underlying prices
        
        Parameters:
        S_range: Range of stock prices to evaluate
        K, T, r, sigma: Standard BS parameters
        option_type: 'call' or 'put'
        greeks_to_plot: List of Greeks to plot ('delta', 'gamma', 'theta', 'vega', 'rho')
        """
        if greeks_to_plot is None:
            greeks_to_plot = ['delta', 'gamma', 'theta', 'vega', 'rho']
            
        plt.figure(figsize=(15, 10))
        
        for i, greek in enumerate(greeks_to_plot, 1):
            plt.subplot(len(greeks_to_plot)//2 + len(greeks_to_plot)%2, 2, i)
            
            values = []
            for S in S_range:
                _, greeks = self.price_option(S, K, T, r, sigma, option_type)
                values.append(greeks[greek.lower()])
                
            plt.plot(S_range, values)
            plt.axvline(x=K, color='gray', linestyle='--', alpha=0.5)
            plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            plt.grid(True, alpha=0.3)
            plt.title(f'{greek.capitalize()} - {option_type.capitalize()} Option')
            plt.xlabel('Stock Price ($)')
            
            # Add appropriate y-label based on Greek
            if greek.lower() == 'delta':
                plt.ylabel('Δ (Delta)')
            elif greek.lower() == 'gamma':
                plt.ylabel('Γ (Gamma)')
            elif greek.lower() == 'theta':
                plt.ylabel('Θ (Theta) - Daily')
            elif greek.lower() == 'vega':
                plt.ylabel('v (Vega) - per 1% vol change')
            elif greek.lower() == 'rho':
                plt.ylabel('ρ (Rho) - per 1% rate change')
                
        plt.tight_layout()
        
        return plt.gcf()  # Return the figure for saving or further modifications

    def plot_volatility_surface(self, S, K_range, T_range, r, sigma_func):
        """
        Plot implied volatility surface
        
        Parameters:
        S: Current stock price
        K_range: Range of strike prices
        T_range: Range of times to maturity
        r: Risk-free interest rate
        sigma_func: Function that takes K and T and returns implied volatility
        """
        # Create meshgrid
        K_mesh, T_mesh = np.meshgrid(K_range, T_range)
        
        # Calculate implied volatility for each point
        implied_vol = np.zeros_like(K_mesh)
        for i in range(len(T_range)):
            for j in range(len(K_range)):
                implied_vol[i, j] = sigma_func(K_mesh[i, j], T_mesh[i, j])
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surface = ax.plot_surface(K_mesh, T_mesh, implied_vol * 100, 
                                 cmap='viridis', edgecolor='none', alpha=0.8)
        
        # Add color bar
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, label='Implied Volatility (%)')
        
        # Set labels and title
        ax.set_xlabel('Strike Price ($)')
        ax.set_ylabel('Time to Maturity (years)')
        ax.set_zlabel('Implied Volatility (%)')
        ax.set_title('Implied Volatility Surface')
        
        return fig


if __name__ == "__main__":
    # Create a Black-Scholes model instance
    bs_model = BlackScholesModel()
    
    # Example parameters
    S = 100      # Current stock price
    K = 100      # Strike price
    T = 1.0      # Time to maturity (1 year)
    r = 0.05     # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)
    
    # Calculate option prices and Greeks
    call_price, call_greeks = bs_model.price_option(S, K, T, r, sigma, 'call')
    put_price, put_greeks = bs_model.price_option(S, K, T, r, sigma, 'put')
    
    # Display results
    print("\n=== Black-Scholes Option Pricing ===")
    print(f"Stock Price (S): ${S}")
    print(f"Strike Price (K): ${K}")
    print(f"Time to Maturity (T): {T} years")
    print(f"Risk-free Rate (r): {r*100}%")
    print(f"Volatility (σ): {sigma*100}%")
    
    print(f"\nCall Option Price: ${call_price:.4f}")
    print("Call Option Greeks:")
    for greek, value in call_greeks.items():
        print(f"  {greek.capitalize()}: {value:.6f}")
    
    print(f"\nPut Option Price: ${put_price:.4f}")
    print("Put Option Greeks:")
    for greek, value in put_greeks.items():
        print(f"  {greek.capitalize()}: {value:.6f}")
    
    # Calculate and display implied volatility
    print("\n=== Implied Volatility ===")
    # Simulate a market price slightly different from the model price
    market_price = call_price * 1.1  # 10% higher than model price
    implied_vol = bs_model.implied_volatility(market_price, S, K, T, r, 'call')
    print(f"Market Price: ${market_price:.4f}")
    print(f"Implied Volatility: {implied_vol*100:.2f}%")
    
    # Plot option prices for a range of stock prices
    print("\n=== Generating Option Price Plot ===")
    S_range = np.linspace(50, 150, 100)
    price_fig = bs_model.plot_option_prices(S_range, K, T, r, sigma)
    price_fig.savefig('black_scholes_option_prices.png')
    print("Option price plot saved as 'black_scholes_option_prices.png'")
    
    # Plot Greeks for a range of stock prices
    print("\n=== Generating Greeks Plots ===")
    greeks_fig = bs_model.plot_greeks(S_range, K, T, r, sigma, 'call')
    greeks_fig.savefig('black_scholes_greeks.png')
    print("Greeks plots saved as 'black_scholes_greeks.png'")
    
    # Create a simple smile pattern for demonstration
    def sample_vol_func(K, T):
        return sigma * (1 + 0.5 * ((K/S - 1)**2) + 0.3 * (1 - T))
    
    # Plot volatility surface
    print("\n=== Generating Volatility Surface Plot ===")
    K_range = np.linspace(70, 130, 20)
    T_range = np.linspace(0.1, 2.0, 20)
    vol_fig = bs_model.plot_volatility_surface(S, K_range, T_range, r, sample_vol_func)
    vol_fig.savefig('black_scholes_volatility_surface.png')
    print("Volatility surface plot saved as 'black_scholes_volatility_surface.png'")