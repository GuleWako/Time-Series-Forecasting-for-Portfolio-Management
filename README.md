# Time Series Forecasting and Portfolio Optimization for Guide Me in Finance (GMF) Investments

This project focuses on time series forecasting to optimize portfolio management strategies for Guide Me in Finance (GMF) Investments, using historical financial data from YFinance. By forecasting future market trends, we aim to enhance asset allocation and provide investment strategies that maximize returns and minimize risk.

## Project Overview

 **Goal** : Utilize time series forecasting to predict stock trends and optimize portfolio allocations for a balanced risk-return profile.

### Key Assets:

* **TSLA (Tesla)** : High-growth, high-risk stock.
* **BND (Vanguard Total Bond Market ETF)** : Stability-focused bond ETF.
* **SPY (S&P 500 ETF)** : Broad market exposure for diversification.

## Tasks and Approach

### Data Preprocessing and Exploration

* **Data Sources** : Historical data for TSLA, BND, and SPY (2015–2024).
* **Cleaning** : Handled missing values and checked data consistency.
* **EDA** : Analyzed trends, volatility, and seasonality using rolling averages and outlier detection.

### Model Development

* **Models Used** :
* **ARIMA** : For univariate, non-seasonal time series forecasting.
* **SARIMA** : For time series with seasonal patterns.
* **LSTM** : Deep learning model for capturing long-term dependencies.
* **Evaluation** : Assessed models using MAE, RMSE, and MAPE to determine best performance.

### Forecasting and Trend Analysis

* **Forecast Horizon** : Predicted Tesla's future stock prices for 6-12 months.
* **Forecast Insights** : Visualized forecasts with confidence intervals and interpreted expected market trends, volatility, and risk levels.

### Portfolio Optimization

* **Portfolio Composition** : Combined TSLA, BND, and SPY.
* **Risk-Return Analysis** :
* **Annual Returns** : Calculated returns based on forecasted trends.
* **Covariance & Volatility** : Analyzed how asset returns move together.
* **Sharpe Ratio** : Optimized weights to maximize risk-adjusted returns.
