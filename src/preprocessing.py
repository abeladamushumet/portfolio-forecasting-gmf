import os
import pandas as pd
import numpy as np
from src import data_loader as dl

# ------------------ Preprocessing Functions ------------------

def check_and_fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')  
    df = df.reindex(full_idx)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df

def add_daily_returns(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.DataFrame:
    df = df.copy()
    df['Daily_Return'] = df[price_col].pct_change() * 100
    return df

def add_rolling_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    df[f'Rolling_Mean_{window}'] = df['Adj Close'].rolling(window=window).mean()
    df[f'Rolling_Volatility_{window}'] = df['Daily_Return'].rolling(window=window).std()
    return df

def detect_outliers(df: pd.DataFrame, col: str = 'Daily_Return', threshold: float = 3) -> pd.DataFrame:
    df = df.copy()
    mean = df[col].mean()
    std = df[col].std()
    df['Z_Score'] = (df[col] - mean) / std
    df['Outlier'] = df['Z_Score'].abs() > threshold
    return df

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = check_and_fill_missing(df)
    df = add_daily_returns(df)
    df = add_rolling_features(df)

    # Add log adjusted close price column
    df['log_adjclose'] = np.log(df['Adj Close'])

    df = detect_outliers(df)
    return df

def generate_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary_stats = df.describe().T
    summary_stats['missing_values'] = df.isna().sum()
    summary_stats['missing_percent'] = (df.isna().mean() * 100).round(2)
    return summary_stats

# ------------------ Run for Multiple Tickers ------------------

if __name__ == "__main__":
    TICKERS = ["TSLA", "BND", "SPY"]
    START_DATE = "2015-07-01"
    END_DATE = "2025-07-31"

    processed_dir = os.path.join("data", "processed")
    eda_dir = os.path.join("data", "eda")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(eda_dir, exist_ok=True)

    for ticker in TICKERS:
        print(f"\nProcessing {ticker}...")
        raw_df = dl.get_data(ticker, START_DATE, END_DATE)
        processed_df = preprocess_pipeline(raw_df)

        # Save processed data
        save_path = os.path.join(processed_dir, f"{ticker}_processed.csv")
        processed_df.to_csv(save_path)
        print(f"✅ Saved processed {ticker} data to {save_path} "
              f"({processed_df.shape[0]} rows)")

        # Save EDA summary
        summary_df = generate_summary(processed_df)
        summary_path = os.path.join(eda_dir, f"{ticker}_eda_summary.csv")
        summary_df.to_csv(summary_path)
        print(f"📊 Saved EDA summary for {ticker} to {summary_path}")
