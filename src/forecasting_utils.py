"""
Utility functions for evaluating, comparing, and visualizing forecasts
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Metrics calculation
def evaluate_forecast(y_true, y_pred):
    """
    Calculate MAE and RMSE for a forecast.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

# 2. Save metrics to CSV
def save_metrics(metrics_list, save_path):
    """
    Save list of metrics dicts to CSV.
    """
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(save_path, index=False)
    print(f"Metrics saved to {save_path}")

# 3. Save forecasts to CSV
def save_forecasts(forecast_df, save_path):
    """
    Save combined forecast DataFrame.
    """
    forecast_df.to_csv(save_path)
    print(f"Forecasts saved to {save_path}")

# 4. Plot actual vs forecast for one ticker
def plot_forecast(df, ticker, save_dir=None):
    """
    Plot actual vs forecast for a given ticker DataFrame.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Actual'], label='Actual', color='black')
    plt.plot(df.index, df['Forecast'], label='Forecast', color='red', linestyle='--')
    plt.title(f"{ticker} - Actual vs Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{ticker}_forecast_plot.png")
        plt.savefig(file_path, dpi=300)
        print(f"Plot saved to {file_path}")
    else:
        plt.show()

# 5. Compare ARIMA vs LSTM metrics
def compare_models(arima_metrics_path, lstm_metrics_path, save_path=None):
    """
    Load metrics CSVs for ARIMA and LSTM, merge for comparison.
    """
    arima_df = pd.read_csv(arima_metrics_path)
    lstm_df = pd.read_csv(lstm_metrics_path)

    merged_df = arima_df.merge(lstm_df, on="Ticker", suffixes=("_ARIMA", "_LSTM"))

    if save_path:
        merged_df.to_csv(save_path, index=False)
        print(f"Comparison saved to {save_path}")

    return merged_df
