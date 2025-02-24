import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(filename='task2_analysis.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class BrentOilAnalysis:
    def __init__(self, data: pd.DataFrame):
        """Initialize with Brent Oil Price Data"""
        self.data = data.copy()
        self.data = self.data.sort_values(by="Date")
        self.data.set_index("Date", inplace=True)
        logging.info("Data initialized and sorted.")
    
    def fit_lstm(self, lookback=30, epochs=20, batch_size=16):
        """Train LSTM model for time series forecasting."""
        logging.info("Training LSTM model.")

        # Normalize Data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(self.data[['Price']])

        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - lookback):
            X.append(scaled_data[i:i+lookback])
            y.append(scaled_data[i+lookback])
        self.X_train, self.y_train = np.array(X), np.array(y)

        # Define LSTM Model
        self.lstm_model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(lookback, 1)),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        self.lstm_model.compile(optimizer='adam', loss='mse')
        
        # Train Model (verbose=1 to show training progress)
        self.lstm_model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        
        self.lookback = lookback
        logging.info("LSTM model trained successfully.")

    def forecast_lstm(self, future_days=60):
        """Forecast Brent Oil Prices using LSTM and plot results."""
        logging.info(f"Forecasting next {future_days} days using LSTM.")

        # Get last sequence for forecasting
        last_sequence = self.data['Price'].values[-self.lookback:].reshape(-1, 1)
        last_sequence = self.scaler.transform(last_sequence)  # Normalize
        
        # Generate future predictions
        future_preds = []
        for _ in range(future_days):
            pred = self.lstm_model.predict(last_sequence.reshape(1, self.lookback, 1), verbose=0)
            future_preds.append(pred[0][0])
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = pred  # Append prediction
        
        future_preds = np.array(future_preds).reshape(-1, 1)
        future_preds = self.scaler.inverse_transform(future_preds)  # Denormalize

        # Plot Results
        plt.figure(figsize=(12, 6))

        # Plot actual prices
        plt.plot(self.data.index, self.data['Price'], label="Actual Prices", color='blue')

        # Plot model predictions on training data
        train_preds = self.lstm_model.predict(self.X_train)
        train_preds = self.scaler.inverse_transform(train_preds)
        plt.plot(self.data.index[self.lookback:], train_preds, label="Model Predictions (Train)", color='orange')

        # Plot forecasted future prices
        future_dates = pd.date_range(start=self.data.index[-1], periods=future_days + 1, freq='D')[1:]
        plt.plot(future_dates, future_preds, label="Future Predictions (60 Days)", color='red', linestyle='dashed')

        plt.title("LSTM Forecast for Brent Oil Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()

        logging.info("LSTM forecast plotted successfully.")
        return future_preds
    def fit_markov_switching(self, k_regimes=2):
            """Fit a Markov-Switching Model for Brent Oil Prices"""
            logging.info(f"Fitting Markov-Switching ARIMA with {k_regimes} regimes.")
            
            # Ensure the Date index is removed
            model_data = self.data.copy()
            model_data = model_data[['Price']].astype(float)  # Keep only numeric data
            
            # Fit the Markov-Switching Model
            model = MarkovRegression(model_data, k_regimes=k_regimes, trend='c', switching_variance=True)
            result = model.fit()
            
            logging.info("Markov-Switching ARIMA model fitted successfully.")
            return result.summary()

    def fit_markov_switching(self, k_regimes=2):
        """Fit a Markov-Switching Model for Brent Oil Prices"""
        logging.info(f"Fitting Markov-Switching ARIMA with {k_regimes} regimes.")
        
        # Ensure the Date index is removed
        model_data = self.data.copy()
        model_data = model_data[['Price']].astype(float)  # Keep only numeric data
        
        # Fit the Markov-Switching Model
        model = MarkovRegression(model_data, k_regimes=k_regimes, trend='c', switching_variance=True)
        result = model.fit()
        
        logging.info("Markov-Switching ARIMA model fitted successfully.")
        return result.summary()
    
    def check_stationarity(self):
        """Perform Augmented Dickey-Fuller test."""
        result = sm.tsa.adfuller(self.data['Price'])
        logging.info(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
        return result[1] < 0.05  # Returns True if stationary
    
    def fit_var(self, lags=5):
        """Fit a VAR model for multivariate time series analysis."""
        logging.info(f"Fitting VAR model with {lags} lags.")
        model = VAR(self.data)
        self.var_result = model.fit(lags)
        return self.var_result.summary()
    
    def fit_markov_switching(self, k_regimes=2):
        """Fit a Markov Switching ARIMA model."""
        logging.info(f"Fitting Markov Switching model with {k_regimes} regimes.")
        model = MarkovRegression(self.data['Price'], k_regimes=k_regimes, trend='c', switching_variance=True)
        self.ms_result = model.fit()
        return self.ms_result.summary()
  

# Usage Example:
# analysis = BrentOilAnalysis(data)  # Assuming `data` is already loaded
# analysis.fit_lstm(lookback=30, epochs=20, batch_size=16)
# future_predictions = analysis.forecast_lstm(future_days=60)

