5# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 15:31:23 2025

@author: wilfb
"""

# project

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
'''
print("yfinance:", yf.__version__)
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("matplotlib:", matplotlib.__version__)
print("scipy:", scipy.__version__)
'''
# choose assets - various sectors
# chose from FTSE 100 to keep consistent currency, trading hours 
# with sector diversification

assets = ["HSBA.L", "BP.L", "SHEL.L","AZN.L", "GSK.L","BA.L" ] #FTSE 100

#fetch datafrom yahoo finance - 5 years of data
# gives adjusted prices give correct price returns over time

data = yf.download(assets, start = "2020-01-01", end="2025-01-01")["Close"]
# ['Adj Close'] pulls out adjusted closing prices (accounting for dividends & stock splits)
#daily returns

data = data.dropna() # ensure no NaN rows
returns = data.pct_change().dropna() #takes % change from previous day - decimal
returns_pct= returns *100

'''
print(data.head()) # first 5 rows
print(data.tail()) # last 5 rows
print("Data Shape:", data.shape)  #rows = trading days, cols = stocks
print("Returns shape:", returns.shape)
print(returns.head())
print('Returns Dimensions:', returns.shape)
print(returns.tail())
'''
returns_2024= returns_pct["2024-01-01":"2024-12-31"] #select only for 2024 returns
returns_Jan_2024 =returns_pct["2024-01-01":"2024-01-31"]
#returns.index.years gives a numpy array of years for each row-> filter with normal boolean mask
#returns_pct.index.year==2024 = array([False, False,...True,True...])
#only rows showing True are shown
#could also string splice- returns_2024=returns_pct["2024-01-01":"2024-12-31"]

returns.index = pd.to_datetime(returns.index)
#print(returns.index[:5]) #ensure panda is converting strings to readable dates

returns_2024.plot(figsize=(12,6), title= 'Daily % Returns for 2024')
plt.xlabel('Date')
plt.ylabel('Daily % returns')
plt.show() #no parenthesis- no calling the function

returns_Jan_2024.plot(figsize=(12,6), title= 'Daily % Returns Jan 2024')
#plots time series of Jan 2025
plt.xlabel('Date')
plt.ylabel('% Return')
plt.show()

returns_pct.plot(figsize=(10,6))
plt.title('Daily percentage returns of selected stocks')
plt.xlabel('Date')
plt.ylabel('Daily % Returns')
plt.legend(loc="upper right")
plt.show()

#step 4 - equal weight portfolio - to use as benchmark

trading_days= 252

mean_returns = returns.mean()  # average daily return for each stock
cov_matrix = returns.cov()
n_assets = len(assets)
mean_ann = returns.mean() * trading_days #expected annual return per asset
cov_ann= returns.cov() * trading_days   #annual covariance matrix

w_equal= np.array([1/n_assets]* n_assets)  #equal weighting

port_return_daily = np.dot(w_equal, mean_returns)  #weighted average of expected annual returns
port_vol = np.sqrt(np.dot(w_equal.T,np.dot(cov_matrix,w_equal))) # standard dev
port_return_ann = np.dot(w_equal, mean_ann)

print("Equal-weight weights:\n", pd.Series(w_equal, index=returns.columns).round(4))
print("Expected annual return (Equal-weight):", round(port_return_ann,4)) 
#round to 4 dp
print("Expected annual volatility (equal-weight):", round(port_vol,4)) # risk

# step 5 - optimised portfolio using modern portfolio theory
# goal - find 'best' mix of assets
# 1. Minimum variance portfolio 2. Maximum sharpe portfolio (best risk-adjusted return)
#constraints: all weights sum to 1 (fully invested), and weights>0 - no shorts

from scipy.optimize import minimize

rf = 0.02 #assume 2 percent risk free rate
#helper functions
 
def portfolio_return(weights):
    # weighted sum of expected returns - portfolio return
    return np.dot(weights, mean_ann)
def portfolio_volatility(weights):
    # std of portfolio = sqrt(w.T *Cov*w)  - portfolio risk
    return np.sqrt(np.dot(weights.T, np.dot(cov_ann,weights)))
def sharpe_ratio(weights, rf=0.02):
    #Sharpe = (Return - Risk free rate)/ volatility
    return(portfolio_return(weights)-rf)/ portfolio_volatility(weights)

#constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w)-1}) 
# ensures sum to 1 - 'eq' must equal 1 exactly
bounds = tuple((0,1) for asset in range(n_assets))
# each weight must be between 0 and 1
init_guess = n_assets*[1./n_assets]

def min_var_portfolio():      # safest portfolio
    result= minimize(portfolio_volatility,
                    init_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints)
    return result.x #optimal weights

w_minvar = min_var_portfolio()

# 2. Max Sharpe Portfolio
# Obj: Maximise Sharpe

def max_sharpe_portfolio():
    result = minimize(lambda w: -sharpe_ratio(w),
                      init_guess,
                      method='SLSQP',
                      bounds=bounds,
                      constraints= constraints)
    return result.x #optimal weights

w_maxsharpe= max_sharpe_portfolio()

def summarise(weights,name):
    print(f"\n{name}")
    print(pd.Series(weights, index=assets).round(4)) #weights by stock
    print("Expected return:", round(portfolio_return(weights),4))
    print("Volatility:", round(portfolio_volatility(weights),4))
    print("Sharpe:", round(sharpe_ratio(weights), 4))
    
summarise(w_minvar, "Minimum Variance Portfolio")
summarise(w_maxsharpe, "Max Sharpe ratio Portfolio")
    
# fix a target return 
# efficient frontier - shows best possible portfolio for each level of risk

def efficient_return(target_return):
    #minimise volatility given target return
    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w)-1},  # sum to 1
        {'type': 'eq', 'fun': lambda w: portfolio_return(w) - target_return}
        )   # only consider portfolios with exactly the expected return
    bounds = tuple((0,1) for i in range(n_assets)) #restrict weights to 0-1
    result = minimize(portfolio_volatility, #find least risky portfolio under above conditions
                      w_equal,
                      method = 'SLSQP',  # seqeuntial least squares programming
                      bounds=bounds,
                      constraints=constraints)
    return result

# Generate frontier curve 
# Range of returns from min variance to max sharpe

rets_to_try= np.linspace(portfolio_return(w_minvar), #50 evenly spaced numbers
                         portfolio_return(w_maxsharpe),
                         50)

frontier_returns = [] # y coordinates
frontier_vols = []  # x coordinates

for r in rets_to_try:
    res= efficient_return(r)
    if res.success:  # only store valid solutions
        frontier_returns.append(portfolio_return(res.x))
        frontier_vols.append(portfolio_volatility(res.x))
        
# plot everything

plt.figure(figsize=(10,6))
plt.plot(frontier_vols, frontier_returns, 'g--', label= 'Efficient Frontier')

# mark special portfolios

plt.scatter(port_vol, port_return_ann, c='blue', marker='o', label= 'equal weight')
plt.scatter(portfolio_volatility(w_minvar), portfolio_return(w_minvar),
            c='red', marker='o', label=' minimum variance')
plt.scatter(portfolio_volatility(w_maxsharpe), portfolio_return(w_maxsharpe),
            c='gold', marker='*', s=200, label = 'Max Sharpe')

#chart labels

plt.xlabel('Volatility (risk)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.legend()
plt.show()

    
    