import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OilPriceAnalysis:
    def __init__(self, data):
        self.data = data
        self.models = {}

    def exploratory_data_analysis(self):
        logging.info("Performing Exploratory Data Analysis.")
        
        self.data['Year'] = self.data['Year'].astype(int)
        print(self.data.describe())
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Brent Oil Price', color='blue')
        ax1.plot(self.data['Year'], self.data['Price'], color='blue', label='Brent Oil Price')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Ethiopian Inflation Rate (%)', color='red')
        ax2.plot(self.data['Year'], self.data['Inflation_%'], color='red', linestyle='dashed', label='Inflation Rate')
        ax2.tick_params(axis='y', labelcolor='red')
        
        years = sorted(self.data['Year'].unique())
        ax1.set_xticks(years[::5])
        
        fig.legend(loc='upper left')
        plt.title('Brent Oil Price vs Ethiopian Inflation Rate (%)')
        plt.grid()
        plt.show()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm')
        plt.title("Feature Correlation Heatmap")
        plt.show()
        
        logging.info("EDA completed.")

    def prepare_data(self):
        logging.info("Preparing data for training.")
        X = self.data[['Year']]
        y_price = self.data['Price']
        y_inflation = self.data['Inflation_%']
        X_train, X_test, y_price_train, y_price_test, y_inflation_train, y_inflation_test = train_test_split(
            X, y_price, y_inflation, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_price_train, y_price_test, y_inflation_train, y_inflation_test

    def train_models(self, X_train, y_price_train):
        logging.info("Training models using MLflow.")
        with mlflow.start_run():
            
            # Decision Tree Model
            dt = DecisionTreeRegressor()
            dt.fit(X_train, y_price_train)
            self.models['Decision Tree'] = dt
            mlflow.sklearn.log_model(dt, "decision_tree")
            
            # Random Forest Model
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_price_train)
            self.models['Random Forest'] = rf
            mlflow.sklearn.log_model(rf, "random_forest")
        
        logging.info("Model training completed.")

    def train_lstm(self, X_train, y_price_train, y_inflation_train):
        logging.info("Training LSTM model using MLflow.")
        with mlflow.start_run():
            X_train_seq = np.expand_dims(X_train, axis=-1)
            
            model = Sequential([
                LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
                LSTM(50, activation='relu'),
                Dense(2)  # Predicting both price and inflation
            ])
            model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
            model.fit(X_train_seq, np.column_stack((y_price_train, y_inflation_train)), epochs=100, verbose=0)
            self.models['LSTM'] = model
            
            mlflow.tensorflow.log_model(model, "lstm_model")
            logging.info("LSTM model training completed.")

    def evaluate_models(self, X_test, y_price_test, y_inflation_test):
        logging.info("Evaluating models.")
        for name, model in self.models.items():
            if name == 'LSTM':
                X_test_seq = np.expand_dims(X_test, axis=-1)
                predictions = model.predict(X_test_seq)
                y_price_pred, y_inflation_pred = predictions[:, 0], predictions[:, 1]
            else:
                y_price_pred = model.predict(X_test)
                y_inflation_pred = None  

            mae = mean_absolute_error(y_price_test, y_price_pred)
            mse = mean_squared_error(y_price_test, y_price_pred)
            r2 = r2_score(y_price_test, y_price_pred)
            
            logging.info(f"{name} Model - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")

    def forecast_future(self, years_ahead=4):
        logging.info(f"Forecasting for next {years_ahead} years.")
        future_years = np.array(range(self.data['Year'].max() + 1, self.data['Year'].max() + 1 + years_ahead)).reshape(-1, 1)
        model = self.models['LSTM']
        future_years_seq = np.expand_dims(future_years, axis=-1)
        predictions = model.predict(future_years_seq)
        
        forecast_df = pd.DataFrame({
            'Year': future_years.flatten(),
            'Predicted_Price': predictions[:, 0],
            'Predicted_Inflation': predictions[:, 1]
        })
        fig, ax1 = plt.subplots(figsize=(12, 6))
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Year'], self.data['Price'], color='blue', label='Actual Oil Price')
        plt.plot(forecast_df['Year'], forecast_df['Predicted_Price'], color='blue', linestyle='dashed', label='Predicted Oil Price')
        plt.ylabel('Brent Oil Price', color='blue')
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Year'], self.data['Inflation_%'], color='red', label='Actual Inflation')
        plt.plot(forecast_df['Year'], forecast_df['Predicted_Inflation'], color='red', linestyle='dashed', label='Predicted Inflation')
        plt.ylabel('Inflation Rate (%)', color='red')
        plt.legend()
        plt.show()

# # Usage Example
# data = pd.read_csv("aligned_data.csv")
# analysis = OilPriceAnalysis(alligned_data)
# analysis.exploratory_data_analysis()
# X_train, X_test, y_price_train, y_price_test, y_inflation_train, y_inflation_test = analysis.prepare_data()
# analysis.train_models(X_train, y_price_train)
# analysis.train_lstm(X_train, y_price_train, y_inflation_train)
# analysis.evaluate_models(X_test, y_price_test, y_inflation_test)
# analysis.forecast_future()
