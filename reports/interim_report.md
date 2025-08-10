# Interim Report: Time Series Forecasting for Portfolio Management Optimization


## Introduction

This interim report details the progress made on the 

Time Series Forecasting for Portfolio Management Optimization project, specifically addressing **Task 1: Preprocess and Explore the Data**. The objective of this task is to load, clean, and understand historical financial data for key assets to prepare it for subsequent modeling and analysis. This report will cover the methodologies employed for data extraction, cleaning, and exploratory data analysis (EDA), highlighting key insights gained from the process.




## Task 1: Preprocess and Explore the Data

### 1.1 Data Extraction and Loading

As per the project requirements, historical financial data for three key assets—Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY)—was sourced from YFinance. The data covers the period from July 1, 2015, to July 31, 2025. The `src/data_loader.py` module is responsible for this crucial step. This module contains functions to fetch data from YFinance and save it locally, ensuring that the raw data is consistently available for further processing.

The `fetch_asset_data` function within `data_loader.py` utilizes the `yfinance` library to download historical stock prices, including Open, High, Low, Close, Adjusted Close, and Volume. A key aspect of this function is its ability to save the fetched data into a structured format within the `data/raw/` directory, ensuring data persistence and reproducibility. The `get_data` function acts as a wrapper, first attempting to load data from the local raw data directory and, if not found or if a refresh is explicitly requested, it proceeds to fetch the data from YFinance. This mechanism prevents redundant API calls and facilitates efficient data management.

For instance, the `main` block of `data_loader.py` demonstrates how to iterate through the specified tickers (TSLA, BND, SPY) and retrieve their respective historical data, which is then saved as `_raw.csv` files in the `data/raw` directory. This systematic approach ensures that all necessary raw data is available before any preprocessing steps are initiated.




### 1.2 Data Cleaning and Understanding

Data cleaning and understanding are critical steps to ensure the quality and reliability of the dataset for modeling. The `src/preprocessing.py` module is designed to handle these aspects. The `preprocess_pipeline` function orchestrates a series of cleaning and feature engineering steps, which are essential for preparing the time series data.

#### 1.2.1 Basic Statistics and Data Types

Before any transformations, it is crucial to understand the basic statistics and data types of the raw data. The `generate_summary` function in `preprocessing.py` provides a descriptive summary of the DataFrame, including count, mean, standard deviation, min, max, and quartile values. It also calculates the number and percentage of missing values for each column. This summary helps in quickly identifying potential issues such as skewed distributions, outliers, or columns with a high proportion of missing data.

For example, after loading the raw data for TSLA, BND, and SPY, the `generate_summary` function would reveal insights into the distribution of 'Open', 'High', 'Low', 'Close', 'Adj Close', and 'Volume' columns. It also helps confirm that 'Date' is correctly parsed as a datetime object and other numerical columns are indeed numeric. Any discrepancies in data types or unexpected statistical values would prompt further investigation and targeted cleaning.

#### 1.2.2 Handling Missing Values

Financial time series data often contains missing values due to non-trading days (weekends, holidays) or data collection issues. The `check_and_fill_missing` function in `preprocessing.py` addresses this by:

1.  **Reindexing to a full date range**: It creates a complete business day (`B`) index from the minimum to the maximum date in the dataset. This ensures that all potential trading days are accounted for, even if no data was originally present.
2.  **Forward-fill (`ffill`)**: Missing values are filled using the last valid observation. This is a common practice in time series to carry forward the last known price or value.
3.  **Backward-fill (`bfill`)**: Any remaining missing values (e.g., at the very beginning of the series if the first few days are missing) are filled using the next valid observation. This ensures that there are no `NaN` values remaining after the process.

This robust approach to handling missing values ensures a continuous time series, which is a prerequisite for many time series forecasting models like ARIMA and LSTM.

#### 1.2.3 Normalization or Scaling

While the current `preprocess_pipeline` in `preprocessing.py` does not explicitly include normalization or scaling steps, it is a crucial consideration, especially for machine learning models like LSTMs. The `log_adjclose` transformation (`df['log_adjclose'] = np.log(df['Adj Close'])`) is a form of transformation that can help stabilize variance and make the series more stationary, which is beneficial for many time series models. For deep learning models, further scaling (e.g., Min-Max scaling or Standardization) would typically be applied to bring features to a similar range, preventing features with larger values from dominating the learning process.




### 1.3 Conduct Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is a critical phase for understanding the underlying patterns, trends, and anomalies within the financial data. The `preprocessing.py` module, particularly through its `preprocess_pipeline` function, incorporates several steps that contribute to a comprehensive EDA.

#### 1.3.1 Visualization of Closing Price, Daily Returns, and Volatility

The `preprocess_pipeline` function calculates `Daily_Return` and `Rolling_Mean` and `Rolling_Volatility` features. While the `preprocessing.py` script itself does not generate visualizations, these calculated features are fundamental for understanding the time series behavior. Visualizing the `Adj Close` price over time for each asset (TSLA, BND, SPY) would reveal overall trends, such as growth in TSLA, stability in BND, and the general market movement represented by SPY. Such visualizations are typically performed in Jupyter notebooks, as indicated by the presence of `eda_preprocessing.ipynb` in the `notebooks` directory.

Plotting the `Daily_Return` would highlight periods of high and low volatility. The `add_daily_returns` function computes the percentage change in the adjusted closing price, providing a clear measure of daily price fluctuations. Analyzing these plots helps in identifying significant price movements and understanding the risk profile of each asset.

Furthermore, the `add_rolling_features` function calculates rolling mean and rolling volatility (standard deviation of daily returns) over a specified window (defaulting to 20 days). Visualizing these rolling metrics provides insights into short-term trends and fluctuations, smoothing out daily noise and revealing underlying patterns. For instance, an increasing rolling volatility for TSLA would indicate periods of heightened risk, while a stable rolling mean for BND would confirm its role as a low-risk asset.

#### 1.3.2 Outlier Detection

Identifying outliers is crucial in financial data as they can represent significant events (e.g., market crashes, unexpected news) or data errors. The `detect_outliers` function in `preprocessing.py` implements a simple Z-score based method to identify outliers in the `Daily_Return` column. A `threshold` (defaulting to 3) is used to flag observations whose Z-score exceeds this value as outliers. Visualizing these outliers on a time series plot of daily returns can help pinpoint specific dates or events that led to extreme price movements. Analyzing days with unusually high or low returns can provide valuable context for understanding market dynamics and potential risks.




### 1.4 Seasonality and Trends

Understanding seasonality and trends is fundamental for time series forecasting. While visual inspection of time series plots can provide initial insights, statistical tests are necessary to formally assess these properties, particularly stationarity. A stationary time series is one whose statistical properties (mean, variance, autocorrelation) do not change over time. Many time series models, such as ARIMA, assume stationarity.

#### 1.4.1 Statistical Test for Stationarity

For financial time series, raw closing prices are typically non-stationary, exhibiting trends. Daily returns, however, are often stationary or can be made stationary through differencing. The `preprocess_pipeline` in `src/preprocessing.py` calculates `Daily_Return` and `log_adjclose`. The `log_adjclose` transformation can help in achieving stationarity by stabilizing variance, and further differencing (e.g., `df['log_adjclose'].diff().dropna()`) would be applied to make the series mean-stationary if needed. The `arima_model.py` implicitly handles differencing through the `d` parameter in the ARIMA order, which accounts for the integrated (I) part of the model, making non-stationary series stationary for modeling.

While the provided `preprocessing.py` does not explicitly include an ADF test implementation, it is a crucial analytical step that would typically be performed in the `eda_preprocessing.ipynb` notebook. The results of such a test would inform the choice of differencing order (`d`) for ARIMA models. A non-stationary series requires differencing to become stationary, which is a prerequisite for models like ARIMA.




### 1.5 Analyze Volatility

Volatility analysis is a cornerstone of financial time series analysis, providing insights into the degree of variation of a trading price series over time. High volatility often indicates higher risk. The `preprocessing.py` module facilitates this analysis through the calculation of rolling statistics.

#### 1.5.1 Rolling Means and Standard Deviations

The `add_rolling_features` function in `src/preprocessing.py` computes rolling mean and rolling volatility (standard deviation) for a given window (defaulting to 20 days). These metrics are invaluable for understanding short-term trends and fluctuations, as they smooth out daily noise and highlight underlying patterns. For instance, a rising rolling standard deviation for TSLA would signal increasing risk, while a consistently low rolling standard deviation for BND would confirm its role as a stable asset. These rolling metrics are essential for visualizing how risk and return characteristics evolve over time.




### 1.6 Documenting Key Insights and Foundational Risk Metrics

Beyond the raw data and preprocessing steps, the objective of Task 1 also includes documenting key insights derived from the data and calculating foundational risk metrics. While the current Python scripts (`data_loader.py`, `preprocessing.py`, `arima_model.py`) lay the groundwork by providing processed data and basic statistical measures, the comprehensive documentation of insights and the calculation of advanced risk metrics like Value at Risk (VaR) and the Sharpe Ratio would typically be performed in the analytical phase, likely within the Jupyter notebooks (e.g., `eda_preprocessing.ipynb` or `forecasting_analysis.ipynb`).

#### 1.6.1 Overall Direction and Fluctuations

Analyzing the overall direction of asset prices, such as Tesla’s stock price, involves observing long-term trends from the `Adj Close` series. Fluctuations in daily returns and their impact are assessed by examining the `Daily_Return` series and its volatility. Periods of high volatility, as identified by the rolling standard deviation and outlier detection, indicate heightened risk and potential for significant price swings. These observations are crucial for understanding the historical behavior of each asset and informing future forecasting efforts.

#### 1.6.2 Value at Risk (VaR) and Sharpe Ratio

**Value at Risk (VaR)** is a widely used risk metric that quantifies the potential loss of an investment over a specified period at a given confidence level. For example, a 95% VaR of $1 million means there is a 5% chance that the investment could lose more than $1 million over the defined period. Calculating VaR typically involves analyzing the historical distribution of returns, often using methods like historical simulation, parametric (e.g., variance-covariance), or Monte Carlo simulation. While the `preprocessing.py` provides the `Daily_Return` series, the actual calculation of VaR would involve statistical analysis on these returns, which is not explicitly present in the provided Python scripts but is a standard practice in financial analysis.

**The Sharpe Ratio** is a measure of risk-adjusted return, indicating the average return earned in excess of the risk-free rate per unit of volatility (standard deviation). A higher Sharpe Ratio implies a better risk-adjusted performance. It is calculated as: 

$$ \text{Sharpe Ratio} = \frac{E(R_p) - R_f}{\sigma_p} $$

Where:
-   $E(R_p)$ is the expected portfolio return.
-   $R_f$ is the risk-free rate.
-   $\sigma_p$ is the standard deviation of the portfolio’s excess return.

To calculate the Sharpe Ratio, one would need the historical daily returns (already computed by `preprocessing.py`), the standard deviation of these returns, and a chosen risk-free rate. These calculations would be performed during the analysis phase to assess the historical risk-adjusted performance of individual assets or a portfolio. The `portfolio_optimization.ipynb` notebook would likely be the place where such metrics are integrated to evaluate different portfolio compositions.

These foundational risk metrics provide quantitative insights into the risk-return profile of the assets, complementing the qualitative observations from trend and volatility analysis. Their inclusion is vital for making informed decisions regarding portfolio adjustments and risk management.




## Conclusion

Task 1, focusing on data preprocessing and exploration, has successfully laid the groundwork for the subsequent phases of the Portfolio Forecasting with GMF project. The data extraction process, handled by `data_loader.py`, ensures reliable access to historical financial data. The `preprocessing.py` module effectively addresses data cleaning, including handling missing values and generating essential features like daily returns and rolling statistics. While the current implementation provides the necessary data transformations, the detailed visualization and in-depth statistical analysis, particularly for stationarity testing and advanced risk metrics like VaR and Sharpe Ratio, are intended to be performed within the Jupyter notebooks.

The insights gained from this initial phase are crucial for understanding the characteristics of TSLA, BND, and SPY, and for preparing the data for robust time series modeling. The next steps will involve developing and evaluating forecasting models (Task 2) and subsequently optimizing portfolios based on these forecasts (Task 3 and 4).



