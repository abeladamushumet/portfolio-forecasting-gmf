import os
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
import numpy as np
import json

def load_price_data(tickers, data_dir="data/processed"):
    price_data = pd.DataFrame()
    for ticker in tickers:
        file_path = os.path.join(data_dir, f"{ticker}_processed.csv")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        prices = df['Adj Close']

        # Replace zeros and non-positive prices with NaN (invalid for returns)
        prices = prices.replace(0, np.nan)
        prices[prices <= 0] = np.nan

        # Fill missing values forward and backward to avoid gaps
        prices = prices.ffill().bfill()

        price_data[ticker] = prices

    # Remove any remaining infinite or NaN values across the whole dataframe
    price_data = price_data.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any')

    return price_data


def optimize_portfolio(price_data, risk_free_rate=0.02):
    # Basic data checks
    print("Checking for NaNs and infinite values:")
    print(price_data.isnull().sum())
    print("Any infinite values?", np.isfinite(price_data).all())

    # Remove zero-variance columns
    variances = price_data.var()
    zero_var_cols = variances[variances == 0].index.tolist()
    if zero_var_cols:
        print(f"Dropping zero-variance columns: {zero_var_cols}")
        price_data = price_data.drop(columns=zero_var_cols)

    mu = expected_returns.mean_historical_return(price_data)
    S = risk_models.sample_cov(price_data)

    # Regularize covariance matrix by adding small value to diagonal
    S += np.eye(S.shape[0]) * 1e-4

    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    try:
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=True, risk_free_rate=risk_free_rate)
    except Exception as e:
        print(f"Optimization failed: {e}")
        return None, None

    return weights, performance

def save_weights(weights, save_path="results/portfolio_weights.json"):
    with open(save_path, "w") as f:
        json.dump(weights, f, indent=4)
    print(f"Saved portfolio weights to {save_path}")

if __name__ == "__main__":
    TICKERS = ["TSLA", "BND", "SPY"]
    price_data = load_price_data(TICKERS)

    weights, performance = optimize_portfolio(price_data)

    if weights is not None:
        print("Optimized portfolio weights:")
        for ticker, weight in weights.items():
            print(f"  {ticker}: {weight:.4f}")
        
        print("\nPortfolio performance:")
        print(f"  Expected annual return: {performance[0]*100:.2f}%")
        print(f"  Annual volatility: {performance[1]*100:.2f}%")
        print(f"  Sharpe Ratio: {performance[2]:.2f}")

        save_weights(weights)
    else:
        print("Portfolio optimization did not complete successfully.")
