import os
import yfinance as yf
import pandas as pd

RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")

def fetch_asset_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical data for a given ticker from yfinance and save clean CSV.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    data.index.name = "Date"
    
    # Save cleaned CSV without extra headers
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    file_path = os.path.join(RAW_DATA_DIR, f"{ticker}_raw.csv")
    data.to_csv(file_path)
    print(f"Saved clean raw data for {ticker} to {file_path}")
    return data


def save_raw_data(df: pd.DataFrame, ticker: str) -> None:
    """
    Save raw downloaded data to CSV in data/raw/.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    file_path = os.path.join(RAW_DATA_DIR, f"{ticker}_raw.csv")
    df.to_csv(file_path)
    print(f"Saved raw data for {ticker} to {file_path}")


def load_raw_data(ticker: str) -> pd.DataFrame:
    file_path = os.path.join(RAW_DATA_DIR, f"{ticker}_raw.csv")
    
    if not os.path.exists(file_path):
        print(f"No raw data file found for {ticker} at {file_path}")
        return None

    print(f"Loading raw data for {ticker} from {file_path}")

    try:
        # Try loading normally
        df = pd.read_csv(file_path, parse_dates=["Date"])
    except ValueError as e:
        if "Missing column provided to 'parse_dates'" in str(e):
            print("⚠ Detected messy header — attempting to clean...")
            raw = pd.read_csv(file_path, header=None)

            # Actual data starts at row 2
            df = raw.iloc[2:].copy()

            # Force expected column names
            expected_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
            if len(df.columns) >= len(expected_cols):
                df.columns = expected_cols + list(df.columns[len(expected_cols):])
            else:
                df.columns = expected_cols[:len(df.columns)]

            # Parse Date column
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            # Convert all numeric columns
            numeric_cols = [c for c in df.columns if c != "Date"]
            df[numeric_cols] = df[numeric_cols].apply(
                lambda col: pd.to_numeric(col.astype(str).str.replace(",", ""), errors="coerce")
            )
        else:
            raise e

    df.set_index("Date", inplace=True)
    return df


def get_data(ticker: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """
    Main function to get data:
      - Load from CSV if exists and refresh=False
      - Otherwise fetch from yfinance and save
    """
    if not refresh:
        df = load_raw_data(ticker)
        if df is not None:
            return df

    df = fetch_asset_data(ticker, start_date, end_date)
    save_raw_data(df, ticker)
    return df


if __name__ == "__main__":
    # Example usage
    start = "2015-07-01"
    end = "2025-07-31"
    tickers = ["TSLA", "BND", "SPY"]

    for ticker in tickers:
        df = get_data(ticker, start, end, refresh=False)
        print(f"{ticker} data sample:")
        print(df.head())
