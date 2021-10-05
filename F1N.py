#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 01:06:27 2021

@author: michelmaalouli
"""
import os
from datetime import datetime
from pytz import timezone
import pandas_datareader.data as pdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix

# Alpha Vantage API Key
os.environ["ALPHAVANTAGE_API_KEY"] = "AXMXNHRUJ00L5N93"

# apple info from alpha vantage
apple = pdr.get_data_yahoo("AAPL", start = datetime(2006, 10, 1), end = datetime.now())
#apple = pdr.DataReader("AAPL", "av-daily-adjusted", start = datetime(2006, 10, 1), end = datetime(2012, 1, 1))
apple.index = pd.to_datetime(apple.index)

# basic commands to checkup on data
# print(apple.head())
# print(apple.tail())
# print(apple.describe())
# print(apple.index)
# print(apple.columns)
# print(apple["close"][-10:])

# resample data to monthly
# apple_m = web.DataReader("AAPL", "av-monthly", start = datetime(2006, 10, 1), end = datetime(2012, 1, 1))
# amm = apple.resample("M").mean

# add column in apple dataframe for difference between opening and closing
apple["Diff"] = apple.Open - apple.Close

# OR
# apple["diff"] = apple["open"] - apple["close"]

# plot the closing prices for apple
apple["Close"].plot(grid = True)

# show the plot
plt.show()

# Common Financial Analysis

# assign adjusted close to daily close
daily_close = apple[["Adj Close"]]

# daily returns
daily_pct_change = daily_close.pct_change()

# replace NA values with 0
daily_pct_change.fillna(0, inplace = True)

# inspect daily returns
print(daily_pct_change)

# daily log returns
daily_log_returns = np.log(daily_pct_change+1)

# print daily log returns
print(daily_log_returns)

# resample apple to business months, take last observation as value
bmonth = apple.resample("BM").apply(lambda x: x[-1])

# calculate monthly percentage change
print(bmonth.pct_change())

# resample apple to quarters, take the lmean as value per quarter
quarter = apple.resample("4M").mean()

# calculate the quarterly percentage change
print(quarter.pct_change())

# plot the distribution of daily_pct_change
daily_pct_change.hist(bins = 50)

# show the plot
plt.show()

# pull up summary statistics
print(daily_pct_change.describe())

# calculate cumulative daily return
cum_daily_return = (1 + daily_pct_change).cumprod()

# print cum_daily_return
print(cum_daily_return)

# plot cum_daily_return
cum_daily_return.plot(figsize = (12, 8))
plt.show()

# resample cumulative daily return to cumulative monthly return
cum_monthly_return = cum_daily_return.resample("M").mean()
print(cum_monthly_return)
cum_monthly_return.plot(figsize = (12, 8))
plt.show()

def get(tickers, startdate, enddate):
    def data(ticker):
        return (pdr.get_data_yahoo(ticker, start = startdate, end = enddate))
    datas = map (data, tickers)
    return (pd.concat(datas, keys = tickers, names = ["Ticker", "Date"]))

tickers = ["AAPL", "MSFT", "IBM", "GOOG"]
all_data = get(tickers, startdate = datetime(2006, 10, 1), enddate = datetime.now())


# isolate adj close values and transform dataframe
daily_close_px = all_data[["Adj Close"]].reset_index().pivot("Date", "Ticker", "Adj Close")
daily_pxpct_change = daily_close_px.pct_change()

# plot the distributions
daily_pxpct_change.hist(bins = 50, sharex = True, figsize = (12, 8))
plt.show()

# plot scatter matrix with daily pct change data
pd.plotting.scatter_matrix(daily_pxpct_change, diagonal = "kde", alpha = 0.1, figsize = (12, 12))
plt.show()

# adjusted closing prices
adj_close_px = apple["Adj Close"]









