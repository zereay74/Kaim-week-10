# Brent Oil Price Analysis

## Overview
This project focuses on analyzing Brent oil prices using time series methods. Key components include data cleaning, change point detection, and forecasting models like ARIMA and GARCH. The project is structured for reproducibility, automation, and CI/CD integration.

## Project Structure
```
├── .github/workflows     # GitHub Actions workflows for CI/CD
├── .vscode               # VS Code settings and extensions
├── logs                  # Logs for monitoring outputs
│   ├── logs.log
├── notebooks             # Jupyter notebooks for data processing & insights
│   ├── Task_1_Defining_Data_Analysis_Workflow.ipynb # Define analysis workflow
├── scripts               # Python scripts for automation
│   ├── data_load_clean_transform.py       # Load, clean & transform data
│   ├── change_point_analysis.py           # Change point detection
├── tests                 # Unit tests for data validation & pipeline integrity
├── .gitignore            # Ignore unnecessary files
├── README.md             # Project documentation
├── requirements.txt      # Dependencies
```

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

## Features
- **Change Point Detection**: Identifies key trend shifts in oil prices.
- **Forecasting Models**: Uses ARIMA for price forecasting and GARCH for volatility estimation.
- **Automated Data Processing**: Scripts for data cleaning and transformation.
- **CI/CD Pipeline**: GitHub Actions for automation and testing.


