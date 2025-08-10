import os
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src import data_loader as dl
from src import preprocessing as prep

def fit_arima_model(ts: pd.Series, order=(1,1,1), seasonal_order=(0,0,0,0), verbose=False):
    model = SARIMAX(ts, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    if verbose:
        print(model_fit.summary())
    return model_fit

def forecast_arima_model(model_fit, steps=30):
    forecast_res = model_fit.get_forecast(steps=steps)
    forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()
    return forecast, conf_int

def evaluate_forecast(true_values: pd.Series, predicted_values: pd.Series):
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    return {"MAE": mae, "RMSE": rmse}

if __name__ == "__main__":
    TICKERS = ["TSLA", "BND", "SPY"]
    START_DATE = "2015-07-01"
    END_DATE = "2025-07-31"
    FORECAST_STEPS = 30

    results_dir = os.path.join("results")
    os.makedirs(results_dir, exist_ok=True)

    all_forecasts = []

    for ticker in TICKERS:
        print(f"\nProcessing ARIMA for {ticker}...")
        
        # Load and preprocess data
        raw_df = dl.get_data(ticker, START_DATE, END_DATE)
        processed_df = prep.preprocess_pipeline(raw_df)
        
        ts = processed_df['log_adjclose'].dropna()
        
        # Fit ARIMA model
        arima_order = (1, 1, 1)  # You can customize per ticker if needed
        model_fit = fit_arima_model(ts, order=arima_order, verbose=False)
        
        # Forecast next n steps
        forecast, conf_int = forecast_arima_model(model_fit, steps=FORECAST_STEPS)
        
        # Prepare forecast DataFrame
        forecast_df = pd.DataFrame({
            "Date": forecast.index,
            "Forecast": forecast.values,
            "Lower_CI": conf_int.iloc[:, 0].values,
            "Upper_CI": conf_int.iloc[:, 1].values,
            "Ticker": ticker
        })
        forecast_df.set_index("Date", inplace=True)

        # Save per ticker forecast csv
        save_path = os.path.join(results_dir, f"forecast_arima_{ticker}.csv")
        forecast_df.to_csv(save_path)
        print(f"Saved forecast for {ticker} to {save_path}")

        all_forecasts.append(forecast_df.assign(Ticker=ticker))

    # Optionally combine all forecasts into one CSV
    combined_forecasts = pd.concat(all_forecasts)
    combined_save_path = os.path.join(results_dir, "forecasts_arima_all.csv")
    combined_forecasts.to_csv(combined_save_path)
    print(f"\nSaved combined forecasts to {combined_save_path}")
