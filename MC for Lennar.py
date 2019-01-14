import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as web
from scipy.stats import norm

ticker = "LEN"

data = web.DataReader(ticker, 'yahoo',
                      dt.datetime(2014,1,1), dt.date.today())['Adj Close']

log_returns = np.log(1 + data.pct_change())
u = log_returns.mean()
var = log_returns.var()
drift = u - (0.5 * var ** 2)

simulations = 1000
t_intervals = 360

std = log_returns.std()
daily_r = np.exp(drift +
                 std * norm.ppf(np.random.rand(t_intervals,simulations)))

S0 = data.iloc[-1]
price_list = np.zeros_like(daily_r)
price_list[0] = S0

for t in range(1,t_intervals):
     price_list[t] = price_list[t - 1] * daily_r[t]

intrinsic_val = (price_list[t_intervals-1].mean()) / (1.0931)
print(intrinsic_val)
plt.figure(figsize=(10,5))
#plt.plot(price_list)
plt.title("Lennar Monte Carlo")

plt.hist((price_list[t_intervals-1]/1.0931), bins=100)
plt.show()
