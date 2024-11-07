# stock-market-analysis-and-prediction

This project involves analyzing historical stock data to identify trends and build predictive models for stock price forecasting. Using technical indicators, statistical analysis, and machine learning models like ARIMA, LSTM, and hybrid approaches, this project explores both price trend analysis and predictive trading strategies. It includes data preprocessing, feature engineering, model training, and backtesting on simulated trades to evaluate performance. Tools used include Python, scikit-learn, TensorFlow, and Backtrader. This project serves as a foundation for building data-driven financial insights and automated trading systems.

## LSTM Neural Network for Time Series Prediction

LSTM built using the Keras Python package to predict time series steps and sequences. Includes sine wave and stock market data.

[Full article write-up for this code](https://www.altumintelligence.com/articles/a/Time-Series-Prediction-Using-LSTM-Deep-Neural-Networks)

[Video on the workings and usage of LSTMs and run-through of this code](https://www.youtube.com/watch?v=2np77NOdnwk)

## Requirements

Install requirements.txt file to make sure correct versions of libraries are being used.

* Python 3.5.x
* TensorFlow 1.10.0
* Numpy 1.15.0
* Keras 2.2.2
* Matplotlib 2.2.2

Output for sine wave sequential prediction:

![Output for sin wave sequential prediction](https://www.altumintelligence.com/assets/time-series-prediction-using-lstm-deep-neural-networks/sinwave_full_seq.png)

Output for stock market multi-dimensional multi-sequential predictions:

![Output for stock market multiple sequential predictions](https://www.altumintelligence.com/assets/time-series-prediction-using-lstm-deep-neural-networks/sp500_multi_2d.png)
