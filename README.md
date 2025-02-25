# Brent Oil Price Analysis

## Overview
This project focuses on analyzing Brent oil prices using time series methods and machine learning models. It includes data collection, preprocessing, exploratory data analysis (EDA), predictive modeling, and interactive visualization. The analysis explores relationships between Brent oil prices and economic indicators such as inflation.

## Project Structure
```
├── .github/workflows     # GitHub Actions workflows for CI/CD
├── .vscode               # VS Code settings and extensions
├── dashboard app         # Task 3: Interactive Dashboard for Analysis Results
│   ├── oil-price-dashboard/
│       ├── node_modules/
│           ├── app.js    # React app
│           ├── ....
│       ├── public
│   ├── app.py           # Flask backend
├── logs                  # Logs for monitoring outputs
│   ├── logs.log
├── notebooks             # Jupyter notebooks for data processing & insights
│   ├── Task_1_Defining_Data_Analysis_Workflow.ipynb       # Define analysis workflow
│   ├── Task_2_Oil_Analysis_with_AI_and_Stat_Models.ipynb  # Predictive modeling
│   ├── Task_3_Dashboard_Visualization.ipynb              # Interactive visualizations
├── scripts               # Python scripts for automation
│   ├── data_load_clean_transform.py       # Load, clean & transform data
│   ├── change_point_analysis.py           # Change point detection
│   ├── model_training.py                   # Model training and evaluation
│   ├── dashboard_api.py                    # Flask API for dashboard
├── tests                 # Unit tests for data validation & pipeline integrity
├── .gitignore            # Ignore unnecessary files
├── README.md             # Project documentation
├── requirements.txt      # Dependencies
```

## Tasks Breakdown
### Task 1: Defining Data Analysis Workflow
- Establish a structured approach for analyzing Brent oil prices.
- Define steps for data collection, preprocessing, modeling, and evaluation.

### Task 2: Data Analysis & Predictive Modeling
1. **Data Collection**
   - Gather Brent oil price data and economic indicators (e.g., inflation).
2. **Data Preprocessing**
   - Clean data, handle missing values, and align time series.
3. **Exploratory Data Analysis (EDA)**
   - Identify patterns and relationships using visualizations.
4. **Model Building**
   - Train models (Random Forest, Decision Tree, LSTM) for price prediction.
   - LSTM used for forecasting oil sales and Ethiopian inflation for 4 years.
5. **Model Evaluation**
   - Evaluate models using RMSE, MAE, and R-squared.
   - Compare models to determine the best performer.
6. **Insight Generation**
   - Analyze results and provide recommendations based on findings.

### Task 3: Developing an Interactive Dashboard
1. **Backend (Flask)**
   - Serve processed data and model results through API endpoints.
2. **Frontend (React)**
   - Display EDA insights with interactive charts (Recharts, Chart.js, D3.js).
   - Provide filters, year ranges, and correlation analysis.
   - Highlight key events impacting Brent oil prices.
3. **Key Features**
   - Present historical trends, forecasts, and correlations.
   - Enable users to explore specific events and their impact on oil prices.
   - Display key indicators like volatility and price changes.

## Setup Instructions
### 1. Clone the Repository
```sh
git clone https://github.com/zereay74/Kaim-week-10.git
cd brent-oil-analysis
```

### 2. Create a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Run Jupyter Notebook
```sh
jupyter notebook
```

### 5. Start Flask Backend
```sh
python app.py
```

### 6. Start React Frontend
```sh
cd oil-price-dashboard
npm install
npm start
```

## Features
- **Change Point Detection**: Identifies key trend shifts in oil prices.
- **Forecasting Models**: Uses ARIMA for price forecasting and GARCH for volatility estimation.
- **Machine Learning Models**: Random Forest, Decision Tree, and LSTM for predictive analytics.
- **Automated Data Processing**: Scripts for data cleaning and transformation.
- **Interactive Dashboard**: Visualizes insights, trends, and correlations.
- **CI/CD Pipeline**: GitHub Actions for automation and testing.

