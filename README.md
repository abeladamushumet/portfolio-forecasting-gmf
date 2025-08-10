# Portfolio Forecasting with GMF (Generative Modeling Framework)

This project provides a comprehensive framework for forecasting financial asset prices and optimizing investment portfolios using advanced time series models, including ARIMA and LSTM. It covers the entire data science lifecycle from data acquisition and preprocessing to model training, forecasting, and backtesting.




## Features

- **Automated Data Acquisition**: Fetches historical stock data from Yahoo Finance for specified tickers and date ranges.
- **Robust Data Preprocessing**: Handles missing values, calculates daily returns, adds rolling statistics, and detects outliers.
- **ARIMA Modeling**: Implements SARIMAX models for time series forecasting, including fitting, forecasting, and evaluation.
- **LSTM Modeling**:  Utilizes Long Short-Term Memory networks for advanced time series prediction.
- **Portfolio Optimization**:  Tools for optimizing asset allocation based on forecasted returns and risk.
- **Backtesting Framework**: Evaluates model performance on historical data.
- **Modular Codebase**: Organized into `src` directory with dedicated modules for data loading, preprocessing, modeling, and utilities.
- **Jupyter Notebooks**: Provides interactive notebooks for exploratory data analysis (EDA), model development, and result visualization.




## Project Structure

The project is organized into the following directories:

```
portfolio-forecasting-gmf/
├── data/
│   ├── eda/              # Exploratory Data Analysis summaries
│   ├── processed/        # Cleaned and preprocessed data
│   └── raw/              # Raw historical data fetched from yfinance
├── notebooks/            # Jupyter notebooks for analysis and model development
├── reports/              # Generated reports and visualizations
├── results/              # Model forecasts and backtesting results
│   └── figures/          # Plots and figures generated from analysis
├── src/                  # Source code for data handling, modeling, and utilities
│   ├── __pycache__/
│   ├── arima_model.py    # ARIMA model implementation
│   ├── backtester.py     # Backtesting framework
│   ├── data_loader.py    # Functions for data fetching and loading
│   ├── forecasting_utils.py # Utility functions for forecasting
│   ├── lstm_model.py     # LSTM model implementation
│   ├── optimizer.py      # Portfolio optimization algorithms
│   └── preprocessing.py  # Data preprocessing pipeline
├── .git/                 # Git version control (if applicable)
├── LICENSE               # Project license
└── requirements.txt      # Python dependencies
```




## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository (if applicable):**
    ```bash
    git clone https://github.com/abeladamushumet/portfolio-forecasting-gmf.git
    cd portfolio-forecasting-gmf
    ```
    


2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```




## Usage

This project can be used by running the Jupyter notebooks or by executing the Python scripts directly. Below is a general workflow:

1.  **Data Preparation:**
    The `data_loader.py` script can fetch historical data. You can run it directly to download data for specified tickers.
    ```bash
    python src/data_loader.py
    ```
    The `eda_preprocessing.ipynb` notebook provides an interactive way to perform initial data exploration and preprocessing.

2.  **Model Training and Forecasting:**
    -   **ARIMA Models:** The `arima_model.py` script demonstrates how to train and forecast using ARIMA models. You can run it to generate forecasts for the configured tickers.
        ```bash
        python src/arima_model.py
        ```
        The `arima_modeling.ipynb` notebook offers a more detailed walkthrough of ARIMA model development.
    -   **LSTM Models:**  The `lstm_model.py` would contain the LSTM model implementation, and `lstm_modeling.ipynb` would guide through its usage.

3.  **Portfolio Optimization:**
    The `portfolio_optimization.ipynb` notebook (and potentially `src/optimizer.py`) would be used for optimizing asset allocations based on the generated forecasts.

4.  **Backtesting:**
    The `backtesting.ipynb` notebook (and `src/backtester.py`) would be used to evaluate the performance of the forecasting and optimization strategies on historical data.

5.  **Analysis and Visualization:**
    The `forecasting_analysis.ipynb` notebook is designed for analyzing the results and creating visualizations.




## Dependencies

The project relies on the following key Python libraries, as specified in `requirements.txt`:

-   **Data Handling**: `pandas`, `numpy`
-   **Finance Data Fetching**: `yfinance`
-   **Statistical Modeling**: `statsmodels`, `pmdarima` (for ARIMA)
-   **Machine Learning / Deep Learning**: `tensorflow` (for LSTM)
-   **Portfolio Optimization**: `PyPortfolioOpt`
-   **Visualization**: `matplotlib`, `seaborn`
-   **Testing & Utilities**: `scipy`, `tqdm` (for progress bars), `jupyterlab` (for notebooks)


## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
