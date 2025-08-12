import os
import itertools
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src import data_loader as dl
from src import preprocessing as prep


# Train-test split function
def train_test_split(ts: pd.Series, test_size: int = 30):
    """
    Split time series into train and test sets by taking the last `test_size` points as test.
    """
    train = ts.iloc[:-test_size]
    test = ts.iloc[-test_size:]
    return train, test


# Manual ARIMA parameter search
def fit_arima_model(ts: pd.Series, p_range=(0, 4), d_range=(0, 3), q_range=(0, 4)):
    """
    Finds the best (p,d,q) parameters using AIC score and fits an ARIMA model.
    """
    best_aic = float("inf")
    best_order = None
    best_model = None

    for p, d, q in itertools.product(range(*p_range), range(*d_range), range(*q_range)):
        try:
            model = ARIMA(ts, order=(p, d, q))
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_order = (p, d, q)
                best_model = model_fit
        except:
            continue

    print(f"Best ARIMA order: {best_order} with AIC={best_aic:.2f}")
    return best_model


# Forecast using statsmodels ARIMA model
def forecast_arima_model(model, steps=30):
    forecast_res = model.get_forecast(steps=steps)
    forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()
    conf_int.columns = ['Lower_CI', 'Upper_CI']
    return forecast, conf_int


# Forecast evaluation
def evaluate_forecast(true_values: pd.Series, predicted_values: pd.Series):
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    return {"MAE": mae, "RMSE": rmse}


if __name__ == "__main__":
    TICKERS = ["TSLA", "BND", "SPY"]
    START_DATE = "2015-07-01"
    END_DATE = "2025-07-31"
    TEST_SIZE = 30  # Last 30 days for test

    results_dir = os.path.join("results")
    os.makedirs(results_dir, exist_ok=True)

    all_forecasts = []

    for ticker in TICKERS:
        print(f"\nProcessing ARIMA for {ticker}...")

        # Load and preprocess data
        raw_df = dl.get_data(ticker, START_DATE, END_DATE)
        processed_df = prep.preprocess_pipeline(raw_df)

        ts = processed_df['log_adjclose'].dropna()

        # Split train-test
        train_ts, test_ts = train_test_split(ts, test_size=TEST_SIZE)

        # Fit ARIMA
        model = fit_arima_model(train_ts)

        # Forecast for test period
        forecast, conf_int = forecast_arima_model(model, steps=TEST_SIZE)

        # Align forecast index with test dates
        forecast.index = test_ts.index
        conf_int.index = test_ts.index

        # Evaluate forecast
        metrics = evaluate_forecast(test_ts, forecast)
        print(f"{ticker} Test MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")

        # Save forecast DataFrame
        forecast_df = pd.DataFrame({
            "Forecast": forecast.values,
            "Lower_CI": conf_int['Lower_CI'].values,
            "Upper_CI": conf_int['Upper_CI'].values,
            "Ticker": ticker
        }, index=test_ts.index)

        save_path = os.path.join(results_dir, f"forecast_arima_{ticker}_test.csv")
        forecast_df.to_csv(save_path)
        print(f"Saved test forecast for {ticker} to {save_path}")

        all_forecasts.append(forecast_df)

    combined_forecasts = pd.concat(all_forecasts)
    combined_save_path = os.path.join(results_dir, "forecasts_arima_all_test.csv")
    combined_forecasts.to_csv(combined_save_path)
    print(f"\nSaved combined test forecasts to {combined_save_path}")
