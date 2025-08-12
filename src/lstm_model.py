import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src import data_loader as dl
from src import preprocessing as prep

def create_sequences(data, seq_length=30):
    """
    Convert 1D array into sequences for LSTM input.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32):
    early_stop = EarlyStopping(monitor='loss', patience=5)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])
    return history

def forecast_lstm(model, X_input, n_steps):
    """
    Predict next n_steps sequentially, using previous predictions as input.
    """
    preds = []
    input_seq = X_input.copy()
    for _ in range(n_steps):
        pred = model.predict(input_seq[np.newaxis, :, :])[0,0]
        preds.append(pred)
        input_seq = np.append(input_seq[1:], pred)
        input_seq = input_seq.reshape(-1, 1)
    return np.array(preds)

def evaluate_forecast(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    return {"MAE": mae, "RMSE": rmse}

if __name__ == "__main__":
    TICKERS = ["TSLA", "BND", "SPY"]
    START_DATE = "2015-07-01"
    END_DATE = "2025-07-31"
    TEST_SIZE = 30
    SEQ_LENGTH = 30

    results_dir = os.path.join("results")
    os.makedirs(results_dir, exist_ok=True)

    for ticker in TICKERS:
        print(f"\nProcessing LSTM for {ticker}...")

        # Load and preprocess data
        raw_df = dl.get_data(ticker, START_DATE, END_DATE)
        processed_df = prep.preprocess_pipeline(raw_df)

        ts = processed_df['log_adjclose'].dropna().values.reshape(-1, 1)

        # Scale data
        scaler = MinMaxScaler()
        ts_scaled = scaler.fit_transform(ts)

        # Train-test split
        train, test = ts_scaled[:-TEST_SIZE], ts_scaled[-TEST_SIZE - SEQ_LENGTH:]

        # Create sequences
        X_train, y_train = create_sequences(train, seq_length=SEQ_LENGTH)
        X_test, y_test = create_sequences(test, seq_length=SEQ_LENGTH)

        # Reshape for LSTM [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Build and train model
        model = build_lstm_model(input_shape=(SEQ_LENGTH, 1))
        train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32)

        # Forecast
        preds_scaled = model.predict(X_test).flatten()

        # Inverse scale predictions and true values
        preds = scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
        y_true = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

        # Evaluate
        metrics = evaluate_forecast(y_true, preds)
        print(f"{ticker} Test MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")

        # Save predictions
        pred_df = pd.DataFrame({
            "True": y_true,
            "Predicted": preds,
            "Ticker": ticker
        })

        save_path = os.path.join(results_dir, f"forecast_lstm_{ticker}_test.csv")
        pred_df.to_csv(save_path)
        print(f"Saved LSTM test forecast for {ticker} to {save_path}")
