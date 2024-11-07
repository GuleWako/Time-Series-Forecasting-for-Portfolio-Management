import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import numpy as np # type: ignore
import yfinance as yf # type: ignore
from statsmodels.tsa.seasonal import seasonal_decompose # type: ignore

def loadData():
    tickers = ["TSLA", "BND", "SPY"]
    start_date = "2015-01-01"
    end_date = "2024-10-31"
    data_frames = {}
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] 
        data_frames[ticker] = data

    tsla_data = data_frames["TSLA"]
    bnd_data = data_frames["BND"]
    spy_data = data_frames["SPY"]
    return tsla_data, bnd_data, spy_data
def preprocess_data(data,ticker):
    print(f"{ticker} Missing values:\n{data.isnull().sum()}")
    data.reset_index(inplace=True)
    return data
def closePriceOverTime(stockData,tickers):
    # Plot Close Price Trend
    for data, ticker in zip(stockData, tickers):
        plt.figure(figsize=(12, 6))
        plt.plot(data['Date'], data['Close'], label=f'{ticker} Close Price')
        plt.title(f'{ticker} Close Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.show()
    
def dailyReturn(stockData,tickers):
    # Calculate daily percentage change for volatility analysis
    for data, ticker in zip(stockData, tickers):
        data['Daily_Return'] = data['Close'].pct_change()
        data['Daily_Return'].fillna(0, inplace=True)
        
        # Plot daily returns
        plt.figure(figsize=(12, 6))
        plt.plot(data['Date'], data['Daily_Return'], label=f'{ticker} Daily Returns')
        plt.title(f'{ticker} Daily Returns Over Time')
        plt.xlabel('Date')
        plt.ylabel('Daily Return')
        plt.legend()
        plt.show()

def rollingAvgAndStd(stockData,tickers):
    # Calculate rolling averages and standard deviations
    for data, ticker in zip(stockData,tickers):
        data['Rolling_Mean'] = data['Close'].rolling(window=30).mean()
        data['Rolling_Std'] = data['Close'].rolling(window=30).std()
        data['Rolling_Mean'].fillna(0, inplace=True) 
        data['Rolling_Std'].fillna(0, inplace=True)
        
        # Plot rolling mean and std
        plt.figure(figsize=(12, 6))
        plt.plot(data['Date'], data['Close'], label='Close Price')
        plt.plot(data['Date'], data['Rolling_Mean'], label='30-Day Rolling Mean')
        plt.plot(data['Date'], data['Rolling_Std'], label='30-Day Rolling Std', linestyle='--')
        plt.title(f'{ticker} Rolling Mean & Standard Deviation')
        plt.xlabel('Date')
        plt.ylabel('Price / Volatility')
        plt.legend()
        plt.show()
    
def timeSeriesDecomposition(stockData,tickers):
    # Time Series Decomposition
    for data, ticker in zip(stockData,tickers):
        decomposition = seasonal_decompose(data['Close'], model='additive', period=252)
        plt.figure(figsize=(12,6))
        decomposition.plot()
        plt.suptitle(f'{ticker} Time Series Decomposition')
        plt.show()
def varAndSharpeRatio(stockData,tickers):
    for data, ticker in zip(stockData,tickers):
        # VaR and Sharpe Ratio
        VaR = data['Daily_Return'].quantile(0.05)
        print(f"Value at Risk (VaR) at 5% confidence level for {ticker}: {VaR}")
        
        plt.figure(figsize=(12, 6))
        plt.hist(data['Daily_Return'], bins=50, color='skyblue', edgecolor='black')
        plt.axvline(VaR, color='red', linestyle='--', label=f'VaR at 5%: {VaR:.2%}')
        plt.xlabel('Daily Return')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Daily Returns for {ticker} with VaR')
        plt.legend()
        plt.show()
    
        # Sharpe Ratio
        mean_return = data['Daily_Return'].mean()
        std_dev_return = data['Daily_Return'].std()
        sharpe_ratio = mean_return / std_dev_return * np.sqrt(252)
        print(f"Sharpe Ratio for {ticker}: {sharpe_ratio}")
        
        plt.figure(figsize=(6, 6))
        plt.bar(ticker, sharpe_ratio, color='purple')
        plt.xlabel('Ticker')
        plt.ylabel('Sharpe Ratio')
        plt.title(f'Sharpe Ratio for {ticker}')
        plt.show()
    

def outlierDetection(stockData,tickers):
    # Get the highest and lowest returns
    for data, ticker in zip(stockData,tickers):
        high_returns = data.nlargest(30, 'Daily_Return') 
        low_returns = data.nsmallest(30, 'Daily_Return') 
        plt.figure(figsize=(10,6))
        # Plot highest returns
        plt.plot(data['Daily_Return'], label=f'{ticker} Daily Returns')
        plt.scatter(high_returns.index, high_returns['Daily_Return'], color='green', label='Highest Returns')
        # Plot lowest returns
        plt.scatter(low_returns.index, low_returns['Daily_Return'], color='red', label='Lowest Returns')
        # Adding labels and title
        plt.xlabel('Date')
        plt.ylabel('Daily Return')
        plt.title(f'Top 30 Highest and Lowest Returns For {ticker}')
        plt.legend()
        # Display the plot
        plt.xticks(rotation=45) 
        plt.show()