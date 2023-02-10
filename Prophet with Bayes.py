import yfinance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import prophet
from bayes_opt import BayesianOptimization

TSLA_train = yfinance.download("TSLA",start="2018-01-01",end='2021-12-31')
TSLA_test = yfinance.download("TSLA",start="2022-01-01",end='2022-11-01')

TSLA_train.reset_index(inplace=True)
TSLA_test.reset_index(inplace=True)


TSLA_train = TSLA_train.rename(columns={'Date':'ds','Close':'y'})
TSLA_test = TSLA_test.rename(columns={'Date':'ds','Close':'y'})
TSLA_train.drop(columns=['High','Low','Open','Adj Close','Volume'],inplace=True)
TSLA_test.drop(columns=['High','Low','Open','Adj Close','Volume'],inplace=True)
TSLA_train['ds'] = pd.to_datetime(TSLA_train['ds'])
TSLA_test['ds'] = pd.to_datetime(TSLA_test['ds'])

def split_data(data,num_day):
    X,y=[],[]
    y_true,y_component=[],[]
    for i in range(len(data)-num_day-1):
        X.append(np.array(data[i:i+num_day])) # use trend and resid of time series as x
        y.append(np.array(data[i+num_day]))   # close price of next trade day
    return np.array(X),np.array(y)




def prophet_cv(changepoint_prior_scale,seasonality_prior_scale,holidays_prior_scale):
  model = prophet.Prophet(
      weekly_seasonality=True,
      daily_seasonality=True,
      changepoint_prior_scale=changepoint_prior_scale,
      seasonality_prior_scale=seasonality_prior_scale,
      holidays_prior_scale=holidays_prior_scale
  )
  def cross_val_pred(model,X,y):
    model.fit(X)
    future = model.make_future_dataframe(periods=1)
    pred = model.predict(future)


  pass

def optimize_prophet(data,targets): #data:X, targets:y

  def prophet_crossval(changepoint_prior_scale,seasonality_prior_scale,holidays_prior_scale): #input is parameters able to hypertune
    return prophet_cv(
      changepoint_prior_scale,
      seasonality_prior_scale,
      holidays_prior_scale
    )
  optimizer = BayesianOptimization(
      f = prophet_crossval,
      pbounds = {
          "changepoint_prior_scale":(0.001,0.5),
          "seasonality_prior_scale":(0.01,10),
          "holidays_prior_scale":(0.01,10)
      },
      random_state=1234,
      verbose=2

  )      
  optimizer.maximize(n_iter=10)
  print("Final:",optimizer.max)

X_in = TSLA_train['y'].values.tolist() # 不要 tolist 直接让他吃df with ds column

X,y = split_data(X_in,30)
optimize_prophet(X,y)
