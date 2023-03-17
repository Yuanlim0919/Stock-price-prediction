# Stock-price-prediction

This is a project that predicts stock price of TSMC with LSTM, Prophet and their variation by using STL-Decomposition.

## Introduction
Stock price prediction is a topic that oftenly discussed, and it was tried by many people before. It is a time-series prediction problem and we are trying to do it by using the popular methods in solving time series problem.

## Methods
We used Prophet model and LSTM in our study. The data is acquired by yfinance API. For Prophet model, we fine tune the model based on the power spectrum of stock price. We analyze the data by using Fast Fourier Transform (FFT), and extract 5 of the most influential waves as the input of seasonality in Prophet model.

By the inspiration of Prophet model, we decompose the stock price by using STL-Decomposition method. This is a method that able to decompose a single time series into three components, which is trend, seasonal and residue respectively. After we separate the time series, we sum up the trend and residue series to decrease influences of residue component in latter prediction. Afterwards, we predict these two series separately and adding them in the end.
