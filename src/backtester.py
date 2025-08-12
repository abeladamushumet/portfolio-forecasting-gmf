import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class PortfolioBacktester:
    def __init__(self, price_data, weights, initial_cash=100000):
        """
        Args:
            price_data (pd.DataFrame): Historical prices indexed by date, columns are tickers.
            weights (dict): Portfolio weights by ticker (sum to 1).
            initial_cash (float): Starting capital.
        """
        self.price_data = price_data.sort_index()
        self.weights = weights
        self.initial_cash = initial_cash
        self.tickers = list(weights.keys())
        self._validate_inputs()

    def _validate_inputs(self):
        # Ensure weights sum to 1
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1):
            raise ValueError(f"Weights must sum to 1. Current sum: {total_weight}")

        # Ensure tickers exist in price_data
        missing = [t for t in self.tickers if t not in self.price_data.columns]
        if missing:
            raise ValueError(f"Tickers missing from price data: {missing}")

    def run_backtest(self):
        """
        Simulate portfolio value over time given weights and price data.
        Assumes buy-and-hold strategy with initial cash allocation.

        Returns:
            pd.DataFrame: Portfolio value over time.
        """
        # Calculate normalized prices (start at 1)
        normalized_prices = self.price_data / self.price_data.iloc[0]

        # Calculate weighted portfolio value
        weighted_prices = normalized_prices * pd.Series(self.weights)
        portfolio_norm = weighted_prices.sum(axis=1)

        # Portfolio value in dollars
        portfolio_value = portfolio_norm * self.initial_cash

        return portfolio_value.to_frame(name="Portfolio Value")

    def plot_performance(self, portfolio_value):
        """
        Plot portfolio value over time.

        Args:
            portfolio_value (pd.DataFrame): Portfolio value time series.
        """
        plt.figure(figsize=(12,6))
        plt.plot(portfolio_value.index, portfolio_value["Portfolio Value"], label="Portfolio Value")
        plt.title("Backtested Portfolio Performance")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def calculate_returns(self, portfolio_value):
        """
        Calculate portfolio returns.

        Args:
            portfolio_value (pd.DataFrame): Portfolio value time series.

        Returns:
            pd.Series: Daily returns.
        """
        returns = portfolio_value.pct_change().dropna()
        return returns

    def performance_metrics(self, returns):
        annualized_return = returns.mean() * 252
        annualized_vol = returns.std() * np.sqrt(252)

    # Convert to scalar if single-value Series
        if isinstance(annualized_return, pd.Series) and len(annualized_return) == 1:
            annualized_return = annualized_return.item()
        if isinstance(annualized_vol, pd.Series) and len(annualized_vol) == 1:
            annualized_vol = annualized_vol.item()

    # Safely compute sharpe ratio
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else np.nan

        return {
            "Annualized Return": annualized_return,
            "Annualized Volatility": annualized_vol,
            "Sharpe Ratio": sharpe_ratio
       }



if __name__ == "__main__":
    # Example usage (replace with your real data & weights)
    DATA_DIR = "data/processed"
    tickers = ["TSLA", "BND", "SPY"]
    
    # Load price data
    price_data = pd.DataFrame()
    for ticker in tickers:
        file_path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        price_data[ticker] = df['Adj Close']
    
    # Example portfolio weights (must sum to 1)
    weights = {"TSLA": 0.5, "BND": 0.3, "SPY": 0.2}

    backtester = PortfolioBacktester(price_data, weights)
    portfolio_value = backtester.run_backtest()
    backtester.plot_performance(portfolio_value)
    
    returns = backtester.calculate_returns(portfolio_value)
    metrics = backtester.performance_metrics(returns)
    
    print("Backtest Performance Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
