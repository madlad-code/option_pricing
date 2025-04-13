import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import networkx as nx

class BinomialTreeModel:
    def __init__(self):
        """Initialize the Binomial Tree Option Pricing Model"""
        pass
    
    def price_option(self, S, K, T, r, sigma, N, option_type='call', american=False):
        """
        Binomial Tree Option Pricing Model
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free interest rate (annual)
        sigma: Volatility of the underlying asset (annual)
        N: Number of time steps
        option_type: 'call' or 'put'
        american: Whether the option is American (can be exercised early)
        
        Returns:
        option_price: Price of the option
        stock_tree: The tree of stock prices
        option_tree: The tree of option values
        """
        # Time step
        dt = T / N
        
        # Up and down factors
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        
        # Risk-neutral probability
        p = (np.exp(r * dt) - d) / (u - d)
        
        # Initialize stock price tree
        stock_tree = np.zeros((N+1, N+1))
        
        # Fill in the stock price tree
        for i in range(N+1):
            for j in range(i+1):
                stock_tree[j, i] = S * (u ** (i-j)) * (d ** j)
        
        # Initialize option value tree
        option_tree = np.zeros((N+1, N+1))
        
        # Fill in the option values at expiration (time step N)
        for j in range(N+1):
            if option_type.lower() == 'call':
                option_tree[j, N] = max(0, stock_tree[j, N] - K)
            else:  # put
                option_tree[j, N] = max(0, K - stock_tree[j, N])
        
        # Work backwards through the tree
        for i in range(N-1, -1, -1):
            for j in range(i+1):
                # Calculate discounted expected option value
                option_value = np.exp(-r * dt) * (p * option_tree[j, i+1] + (1-p) * option_tree[j+1, i+1])
                
                # Check for early exercise if American option
                if american:
                    if option_type.lower() == 'call':
                        intrinsic_value = max(0, stock_tree[j, i] - K)
                    else:  # put
                        intrinsic_value = max(0, K - stock_tree[j, i])
                    
                    option_tree[j, i] = max(option_value, intrinsic_value)
                else:
                    option_tree[j, i] = option_value
        
        # Return the option price at the root node and the trees
        return option_tree[0, 0], stock_tree, option_tree
    
    def plot_tree(self, stock_tree, option_tree, N, stock_price, strike_price, option_type, american=False, max_nodes=5):
        """
        Plot the binomial tree visualization (limited to a reasonable number of nodes)
        
        Parameters:
        stock_tree: Array of stock prices at each node
        option_tree: Array of option values at each node
        N: Number of time steps
        stock_price: Initial stock price
        strike_price: Option strike price
        option_type: 'call' or 'put'
        american: Whether the option is American
        max_nodes: Maximum number of nodes to show for clarity
        """
        # Limit the visualization to a manageable number of nodes
        display_steps = min(max_nodes, N)
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges for the limited tree
        for i in range(display_steps+1):
            for j in range(i+1):
                # Node attributes
                node_id = f"{i}_{j}"
                stock_value = stock_tree[j, i]
                option_value = option_tree[j, i]
                
                # Check if early exercise happened (for American options)
                early_exercise = False
                if american and i < N:
                    if option_type.lower() == 'call':
                        intrinsic = max(0, stock_value - strike_price)
                    else:
                        intrinsic = max(0, strike_price - stock_value)
                    
                    early_exercise = np.isclose(option_value, intrinsic) and option_value > 0 and intrinsic > 0
                
                # Add node with attributes
                G.add_node(node_id, 
                          stock=f"${stock_value:.2f}", 
                          option=f"${option_value:.2f}",
                          pos=(i, -j + (i/2)),  # Position for plotting
                          early_exercise=early_exercise)
                
                # Add edges from previous level
                if i > 0:
                    G.add_edge(f"{i-1}_{j-1}", node_id)  # Down edge
                    if j > 0:
                        G.add_edge(f"{i-1}_{j}", node_id)  # Up edge
        
        # Get positions and create figure
        pos = nx.get_node_attributes(G, 'pos')
        plt.figure(figsize=(12, 8))
        
        # Draw nodes
        default_nodes = [n for n, attrs in G.nodes(data=True) if not attrs['early_exercise']]
        exercise_nodes = [n for n, attrs in G.nodes(data=True) if attrs['early_exercise']]
        
        nx.draw_networkx_nodes(G, pos, nodelist=default_nodes, node_color='lightblue', 
                              node_size=1000, alpha=0.8)
        
        if exercise_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=exercise_nodes, node_color='lightgreen', 
                                  node_size=1000, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15)
        
        # Draw labels
        stock_labels = {n: f"S: {attrs['stock']}\nO: {attrs['option']}" 
                      for n, attrs in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=stock_labels, font_size=8)
        
        # Set title and remove axes
        option_description = f"{'American' if american else 'European'} {option_type.capitalize()}"
        plt.title(f"Binomial Tree for {option_description} Option (S={stock_price}, K={strike_price}, Steps={N})")
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()  # Return the figure

    def compare_convergence(self, S, K, T, r, sigma, max_steps=50, option_type='call', american=False):
        """
        Compare option prices with increasing number of time steps
        
        Parameters:
        S, K, T, r, sigma: Standard option parameters
        max_steps: Maximum number of time steps to analyze
        option_type: 'call' or 'put'
        american: Whether to price American options
        
        Returns:
        Figure showing convergence of option prices
        """
        steps = range(5, max_steps+1)
        prices = []
        
        for n in steps:
            price, _, _ = self.price_option(S, K, T, r, sigma, n, option_type, american)
            prices.append(price)
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, prices, 'bo-')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Number of Time Steps')
        plt.ylabel('Option Price ($)')
        
        option_description = f"{'American' if american else 'European'} {option_type.capitalize()}"
        plt.title(f"Convergence of {option_description} Option Price with Increasing Steps")
        
        # Add horizontal line for final price
        plt.axhline(y=prices[-1], color='r', linestyle='--', alpha=0.5, 
                   label=f"Converged Price: ${prices[-1]:.4f}")
        plt.legend()
        
        return plt.gcf()  # Return the figure


if __name__ == "__main__":
    # Create a Binomial Tree model instance
    bt_model = BinomialTreeModel()
    
    # Example parameters
    S = 100      # Current stock price
    K = 100      # Strike price
    T = 1.0      # Time to maturity (1 year)
    r = 0.05     # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)
    N = 20       # Number of time steps
    
    # Price European options
    euro_call_price, euro_call_stock_tree, euro_call_option_tree = bt_model.price_option(
        S, K, T, r, sigma, N, 'call', False)
    
    euro_put_price, euro_put_stock_tree, euro_put_option_tree = bt_model.price_option(
        S, K, T, r, sigma, N, 'put', False)
    
    # Price American options
    amer_call_price, amer_call_stock_tree, amer_call_option_tree = bt_model.price_option(
        S, K, T, r, sigma, N, 'call', True)
    
    amer_put_price, amer_put_stock_tree, amer_put_option_tree = bt_model.price_option(
        S, K, T, r, sigma, N, 'put', True)
    
    # Display results
    print("\n=== Binomial Tree Option Pricing ===")
    print(f"Stock Price (S): ${S}")
    print(f"Strike Price (K): ${K}")
    print(f"Time to Maturity (T): {T} years")
    print(f"Risk-free Rate (r): {r*100}%")
    print(f"Volatility (σ): {sigma*100}%")
    print(f"Time Steps (N): {N}")
    
    print(f"\nEuropean Call Option Price: ${euro_call_price:.4f}")
    print(f"European Put Option Price: ${euro_put_price:.4f}")
    print(f"American Call Option Price: ${amer_call_price:.4f}")
    print(f"American Put Option Price: ${amer_put_price:.4f}")
    
    # Check for early exercise premium
    call_premium = amer_call_price - euro_call_price
    put_premium = amer_put_price - euro_put_price
    
    print(f"\nCall Early Exercise Premium: ${call_premium:.6f}")
    print(f"Put Early Exercise Premium: ${put_premium:.6f}")
    
    # Plot the binomial trees (limited to a reasonable number of nodes)
    print("\n=== Generating Tree Visualizations ===")
    
    # European Call
    euro_call_fig = bt_model.plot_tree(euro_call_stock_tree, euro_call_option_tree, 
                                      N, S, K, 'call', False)
    euro_call_fig.savefig('binomial_euro_call_tree.png')
    print("European call option tree saved as 'binomial_euro_call_tree.png'")
    
    # American Put (more interesting due to early exercise)
    amer_put_fig = bt_model.plot_tree(amer_put_stock_tree, amer_put_option_tree, 
                                     N, S, K, 'put', True)
    amer_put_fig.savefig('binomial_amer_put_tree.png')
    print("American put option tree saved as 'binomial_amer_put_tree.png'")
    
    # Analyze convergence
    print("\n=== Analyzing Convergence ===")
    conv_euro_call_fig = bt_model.compare_convergence(S, K, T, r, sigma, 50, 'call', False)
    conv_euro_call_fig.savefig('binomial_euro_call_convergence.png')
    print("European call convergence plot saved as 'binomial_euro_call_convergence.png'")
    
    conv_amer_put_fig = bt_model.compare_convergence(S, K, T, r, sigma, 50, 'put', True)
    conv_amer_put_fig.savefig('binomial_amer_put_convergence.png')
    print("American put convergence plot saved as 'binomial_amer_put_convergence.png'")