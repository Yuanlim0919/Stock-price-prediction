import pandas as pd
import yfinance
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
stock_data = yfinance.download('2330.TW',start='2016-01-01',end='2022-12-01')
stock_data_close = stock_data[['Close']]

components = seasonal_decompose(stock_data_close,model='additive',period=28,extrapolate_trend='freq')
components.plot()
pyplot.show()