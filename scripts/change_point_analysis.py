import logging
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import ruptures as rpt
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Configure logging
logging.basicConfig(filename='analysis.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class SalesChangePointAnalysis:
    def __init__(self, data: pd.DataFrame):
        """Initialize with Sales Data."""
        self.data = data.copy()
        self.data = self.data.sort_values(by="Date")  # Ensure data is sorted
        self.data.set_index("Date", inplace=True)
        logging.info("Data initialized, sorted, and Date set as index.")
    
    def data_summary(self):
        """Provide basic statistics and insights about the data."""
        logging.info("Generating data summary.")
        return self.data.describe()
    
    def check_stationarity(self):
        """Perform Augmented Dickey-Fuller test to check stationarity."""
        result = adfuller(self.data['Price'])
        logging.info(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
        return result[1] < 0.05  # Returns True if data is stationary
    
    def detect_change_points(self, method='normal'):
        """Detect change points in the sales data."""
        algo = rpt.Pelt(model="rbf").fit(self.data['Price'].values)
        change_points = algo.predict(pen=10)
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Price'], label="Sales Data")
        for cp in change_points:
            plt.axvline(self.data.index[cp-1], color='r', linestyle='--', label="Change Point" if cp == change_points[0] else "")
        plt.legend()
        plt.title("Change Point Detection (Normal)")
        plt.show()
        logging.info(f"Detected change points: {change_points}")
        return change_points
    
    def detect_change_points_bayesian(self):
        """Detect change points using a Bayesian method."""
        algo = rpt.Binseg(model="l2").fit(self.data['Price'].values)
        change_points = algo.predict(n_bkps=5)
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Price'], label="Sales Data")
        for cp in change_points:
            plt.axvline(self.data.index[cp-1], color='g', linestyle='--', label="Bayesian Change Point" if cp == change_points[0] else "")
        plt.legend()
        plt.title("Change Point Detection (Bayesian)")
        plt.show()
        logging.info(f"Bayesian detected change points: {change_points}")
        return change_points
    
    def fit_arima(self, order=(1,1,1)):
        """Fit an ARIMA model to the data."""
        logging.info(f"Fitting ARIMA model with order {order}.")
        model = ARIMA(self.data['Price'], order=order)
        self.arima_result = model.fit()
        logging.info("ARIMA model fitted successfully.")
        return self.arima_result.summary()
    
    def forecast_arima(self, steps=30):
        """Forecast future prices using the fitted ARIMA model."""
        logging.info(f"Forecasting next {steps} days using ARIMA.")
        forecast = self.arima_result.get_forecast(steps=steps)
        return forecast.predicted_mean
    
    def fit_garch(self, p=1, q=1):
        """Fit a GARCH model to capture volatility."""
        logging.info(f"Fitting GARCH model with (p={p}, q={q}).")
        model = arch_model(self.data['Price'], vol='Garch', p=p, q=q)
        self.garch_result = model.fit(disp='off')
        logging.info("GARCH model fitted successfully.")
        return self.garch_result.summary()
    
    def forecast_garch(self, steps=30):
        """Forecast volatility using the fitted GARCH model."""
        logging.info(f"Forecasting volatility for next {steps} days using GARCH.")
        forecast = self.garch_result.forecast(horizon=steps)
        return forecast.variance.iloc[-1]  # Fix empty DataFrame issue
