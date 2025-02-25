from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load data
df = pd.read_csv("../week 10 data/Data/alligned_price_and_inflation_eth.csv")

# Ensure correct data types
df['Year'] = df['Year'].astype(int)

# Function to calculate summary statistics
def get_summary_statistics():
    summary = {
        "average_price": round(df["Price"].mean(), 2),
        "average_inflation": round(df["Inflation_%"].mean(), 2),
        "price_volatility": round(df["Price"].std(), 2),
        "inflation_volatility": round(df["Inflation_%"].std(), 2)
    }
    return summary

@app.route("/api/data", methods=["GET"])
def get_data():
    """ Serve the historical dataset """
    return jsonify(df.to_dict(orient="records"))

@app.route("/api/summary", methods=["GET"])
def get_summary():
    """ Serve summary statistics """
    return jsonify(get_summary_statistics())

@app.route("/api/forecast", methods=["GET"])
def get_forecast():
    """ Simulate future forecast using a simple linear extrapolation """
    years_ahead = 5
    max_year = df["Year"].max()
    future_years = np.array(range(max_year + 1, max_year + 1 + years_ahead))

    # Simple linear trend projection based on last 5 years' growth
    price_trend = np.polyfit(df["Year"], df["Price"], 1)
    inflation_trend = np.polyfit(df["Year"], df["Inflation_%"], 1)

    future_prices = np.polyval(price_trend, future_years)
    future_inflation = np.polyval(inflation_trend, future_years)

    forecast_data = pd.DataFrame({
        "Year": future_years,
        "Predicted_Price": future_prices,
        "Predicted_Inflation": future_inflation
    })

    return jsonify(forecast_data.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)
