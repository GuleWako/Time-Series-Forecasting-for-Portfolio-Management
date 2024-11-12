import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import numpy as np # type: ignore
import yfinance as yf # type: ignore
from statsmodels.tsa.seasonal import seasonal_decompose # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # type: ignore
import math
from matplotlib.dates import DateFormatter
import scipy.optimize as sco
from scipy.stats import norm

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
        data.set_index("Date",inplace=True)
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


def arimaModel(train,test):
    try:
        p, d, q = 1, 1, 1 
        arima_model = ARIMA(train, order=(p, d, q))
        arima_result = arima_model.fit()
        arima_forecast = arima_result.forecast(steps=len(test))
        
        # Evaluate ARIMA
        mae_arima = mean_absolute_error(test, arima_forecast)
        rmse_arima = math.sqrt(mean_squared_error(test, arima_forecast))
        mape_arima = mean_absolute_percentage_error(test, arima_forecast)
        return arima_model,mae_arima,rmse_arima,mape_arima
    except Exception as e:
        print(f"ARIMA Model Error: {e}")

def sarimaModel(train,test):
    try:
        p, d, q = 1, 1, 1
        P, D, Q, s = 1, 1, 1, 12
        sarima_model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s))
        sarima_result = sarima_model.fit()
        sarima_forecast = sarima_result.forecast(steps=len(test))
        
        # Evaluate SARIMA
        mae_sarima = mean_absolute_error(test, sarima_forecast)
        rmse_sarima = math.sqrt(mean_squared_error(test, sarima_forecast))
        mape_sarima = mean_absolute_percentage_error(test, sarima_forecast)
        return sarima_model,mae_sarima,rmse_sarima,mape_sarima
        
    except Exception as e:
        print(f"SARIMA Model Error: {e}")

def lstmModel(train,test):
    train_values = train.values
    test_values = test.values

    # Normalize data 
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_values.reshape(-1, 1))
    test_scaled = scaler.transform(test_values.reshape(-1, 1))

    # Create TimeSeriesGenerator for LSTM
    sequence_length = 30 
    batch_size = 1

    train_generator = TimeseriesGenerator(train_scaled, train_scaled, length=sequence_length, batch_size=batch_size)
    test_generator = TimeseriesGenerator(test_scaled, test_scaled, length=sequence_length, batch_size=batch_size)

    # Build LSTM model
    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
        Dense(1)
    ])

    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(train_generator, epochs=10, verbose=1)

    # Forecast with LSTM
    lstm_predictions = lstm_model.predict(test_generator)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)

    # Evaluate LSTM
    mae_lstm = mean_absolute_error(test_values[sequence_length:], lstm_predictions)
    rmse_lstm = math.sqrt(mean_squared_error(test_values[sequence_length:], lstm_predictions))
    mape_lstm = mean_absolute_percentage_error(test_values[sequence_length:], lstm_predictions)
    return lstm_model, lstm_predictions, sequence_length,mae_lstm,rmse_lstm,mape_lstm

def actual_VS_lstm_prediced(test,sequence_length,lstm_predictions):
    plt.figure(figsize=(14, 7))
    plt.plot(test.index, test.values, label='Actual', color='blue')
    plt.plot(test.index[sequence_length:], lstm_predictions, label='LSTM Predictions', color='orange')
    plt.legend()
    plt.show()

def forecastFutureMarketTrend(test_tsla,lstm_model_tsla,tsla_data):
    forecast_length = 180 
    confidence_interval = 0.10 
    sequence_length = 60 
    forecast_input = test_tsla[-sequence_length:].to_numpy().reshape((1, sequence_length, 1))
    lstm_forecast_prices = []

    # Forecasting loop
    for _ in range(forecast_length):
        lstm_forecast = lstm_model_tsla.predict(forecast_input, batch_size=1)
        lstm_forecast_prices.append(lstm_forecast[0, 0])

        # Update forecast_input with the new prediction
        forecast_input = np.append(forecast_input[:, 1:, :], lstm_forecast.reshape(1, 1, 1), axis=1)

    # Convert forecast list to numpy array for further processing
    lstm_forecast_prices = np.array(lstm_forecast_prices)

    # Prepare forecast dates starting from the last date in the dataset
    last_date = pd.to_datetime(test_tsla.index[0])  
    forecast_dates = pd.date_range(start=tsla_data['Date'].iloc[-1], periods=forecast_length+1, freq='D')[1:]
    # Convert historical and forecasted prices to the same numerical scale
    historical_prices = test_tsla.values.flatten()
    forecasted_prices = lstm_forecast_prices.flatten()

    plt.figure(figsize=(14, 7))
    # Plot historical data
    plt.plot(tsla_data['Date'], tsla_data['Close'], label='Historical Data', color='green')
    plt.plot(forecast_dates, lstm_forecast_prices,label="LSTM",color='blue')


    # Add confidence intervals only on the forecast portion
    upper_bound = forecasted_prices * (1 + confidence_interval)
    lower_bound = forecasted_prices * (1 - confidence_interval)
    plt.fill_between(forecast_dates, lower_bound, upper_bound, color='orange', alpha=0.3, label="Confidence Interval")

    # Format x-axis for better date visibility
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.legend()
    plt.title("LSTM Stock Price Forecast for Tesla (6 Months from 2024)")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.grid(True)
    plt.show()


    # Trend Analysis
    print("Trend Analysis:")
    if lstm_forecast_prices[-1] > lstm_forecast_prices[0]:
        print("The forecast indicates a potential upward trend over the next 6 months.")
    else:
        print("The forecast suggests a stable or declining trend in the upcoming months.")

    # Volatility and Risk Analysis
    print("\nVolatility and Risk Analysis:")
    print(f"The confidence interval suggests a {confidence_interval*100}% uncertainty in predicted prices.")
    print("Periods with wider intervals may indicate increased volatility.")

    # Market Opportunities and Risks
    print("\nMarket Opportunities and Risks:")
    print("1. Opportunities: If the forecast shows a stable upward trend, there may be potential gains.")
    print("2. Risks: High volatility or a downward trend could pose risks, especially if confidence intervals widen significantly.")

# Portfolio Optimization Functions
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = returns / volatility
    return returns, volatility, sharpe_ratio

def neg_sharpe_ratio(weights, mean_returns, cov_matrix):
    return -portfolio_performance(weights, mean_returns, cov_matrix)[2]

def optimize_portfolio(df_returns):
    # Compute annualized mean returns and covariance matrix
    mean_daily_returns = df_returns.mean()
    annualized_mean_returns = mean_daily_returns * 252 
    cov_matrix = df_returns.cov() * 252  

    # Set constraints and bounds for weights
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) 
    bounds = tuple((0, 1) for _ in range(len(mean_daily_returns)))  
    initial_weights = len(mean_daily_returns) * [1 / len(mean_daily_returns)]  

    # Optimize portfolio to maximize Sharpe Ratio
    opt_results = sco.minimize(neg_sharpe_ratio, initial_weights, args=(mean_daily_returns, cov_matrix),
                            method='SLSQP', bounds=bounds, constraints=constraints)

    # Retrieve the optimal weights and performance metrics
    optimal_weights = opt_results.x
    optimal_return, optimal_volatility, optimal_sharpe = portfolio_performance(optimal_weights, mean_daily_returns, cov_matrix)

    # Print optimal allocations and performance metrics
    print("Optimal Portfolio Allocation:")
    for asset, weight in zip(['TSLA', 'BND', 'SPY'], optimal_weights):
        print(f"{asset}: {weight:.2%}")

    print(f"\nExpected Portfolio Return: {optimal_return:.2%}")
    print(f"Portfolio Volatility (Risk): {optimal_volatility:.2%}")
    print(f"Sharpe Ratio: {optimal_sharpe:.2f}")

    # Calculate Value at Risk (VaR) for Tesla
    confidence_level = 0.05  # 95% confidence level
    tsla_volatility = df_returns['TSLA'].std() * np.sqrt(252)
    tsla_mean_return = df_returns['TSLA'].mean() * 252
    VaR = tsla_mean_return - tsla_volatility * norm.ppf(1 - confidence_level)
    print(f"\nValue at Risk (VaR) for Tesla at 95% confidence level: {VaR:.2%}")

    # Calculate daily portfolio returns with optimal weights
    df_returns["Portfolio"] = (df_returns["TSLA"] * optimal_weights[0] + 
                            df_returns["BND"] * optimal_weights[1] + 
                            df_returns["SPY"] * optimal_weights[2])

    # Calculate cumulative returns for each asset and portfolio
    df_cum_returns = (1 + df_returns).cumprod()

    # Plot cumulative returns for each asset and the optimized portfolio
    plt.figure(figsize=(14, 7))
    plt.plot(df_cum_returns.index, df_cum_returns['TSLA'], label='TSLA')
    plt.plot(df_cum_returns.index, df_cum_returns['BND'], label='BND')
    plt.plot(df_cum_returns.index, df_cum_returns['SPY'], label='SPY')
    plt.plot(df_cum_returns.index, df_cum_returns['Portfolio'], label='Optimized Portfolio', linewidth=2, linestyle='--', color='black')
    plt.title("Cumulative Returns for TSLA, BND, SPY, and Optimized Portfolio")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend(loc='upper left')
    plt.show()


    # Summary of Portfolio Performance
    print("\nPortfolio Performance Summary:")
    print(f"Expected Annual Return: {optimal_return:.2%}")
    print(f"Annual Volatility (Risk): {optimal_volatility:.2%}")
    print(f"Sharpe Ratio: {optimal_sharpe:.2f}")
    print(f"Value at Risk (VaR) for TSLA: {VaR:.2%}")
