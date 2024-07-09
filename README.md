# DoubleTouch-HFT
import pandas as pd
import yfinance as yf
import numpy as np

def download_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

def generate_signals(data):
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0

    # Create a short simple moving average over the short window
    signals['short_mavg'] = data['Close'].rolling(window=40, min_periods=1, center=False).mean()

    # Create a long simple moving average over the long window
    signals['long_mavg'] = data['Close'].rolling(window=100, min_periods=1, center=False).mean()

    # Create signals
    signals['signal'][40:] = np.where(signals['short_mavg'][40:] > signals['long_mavg'][40:], 1.0, 0.0)

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    return signals

def backtest_strategy(data, signals, initial_capital=100000):
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['stock'] = 100 * signals['signal']   # Buy 100 shares on each buy signal

    # Initialize the portfolio with value owned
    portfolio = positions.multiply(data['Adj Close'], axis=0)

    # Store the difference in shares owned
    pos_diff = positions.diff()

    # Add 'cash' to portfolio
    portfolio['cash'] = initial_capital - (pos_diff.multiply(data['Adj Close'], axis=0)).cumsum()

    # Add 'total' to portfolio
    portfolio['total'] = portfolio['cash'] + portfolio['stock']

    return portfolio

if __name__ == "__main__":
    symbol = 'AAPL'
    start_date = '2022-01-01'
    end_date = '2023-01-01'

    # Download historical stock data
    data = download_stock_data(symbol, start_date, end_date)

    # Generate trading signals
    signals = generate_signals(data)

    # Backtest the trading strategy
    portfolio = backtest_strategy(data, signals)

    # Print the portfolio
    print(portfolio.tail())

    # Plot the portfolio
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    plt.plot(portfolio['total'], label='Total Portfolio Value')
    plt.title(f'Backtest of Trading Strategy for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.show()
