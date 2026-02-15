# Natural Gas Market Analytics & Contract Valuation System

A Python-based analytics platform for natural gas price forecasting and storage contract valuation. This project combines time series modeling with financial derivatives pricing for energy trading and risk management.

## Project Overview

This repository contains two integrated projects:

1. Natural Gas Price Forecasting Model - Predicts future gas prices using machine learning and statistical methods
2. Storage Contract Valuation System - Calculates the net present value of gas storage contracts with constraint validation

## Features

### Price Forecasting Module
- Historical price analysis and trend visualization
- Seasonal pattern detection using Fourier transformations
- Polynomial regression with seasonal features
- Facebook Prophet implementation for advanced forecasting
- 12-month forward price predictions
- 95% confidence intervals for uncertainty quantification
- Interactive price estimator for custom date queries

### Contract Valuation Module
- Automated storage contract pricing
- Injection and withdrawal schedule optimization
- Real-time constraint validation for injection rates, withdrawal rates, storage capacity limits, and volume tracking
- Detailed cash flow analysis
- Storage cost calculations
- Profitability metrics and P&L breakdown

## Technologies Used

- Python 3.x
- Pandas - Data manipulation and analysis
- NumPy - Numerical computing
- Scikit-learn - Machine learning models and preprocessing
- Prophet - Time series forecasting
- Matplotlib - Data visualization

## Project Structure
```
Natural-Gas-Market-Analytics/
│
├── task_1_chase.py              # Price forecasting model
├── task_2_chase.py              # Contract valuation system
├── Nat_Gas.csv                  # Historical price data
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Installation

Clone the repository:
```bash
git clone https://github.com/abdsalam25/Natural-Gas-Market-Analytics-Contract-Valuation-System.git
cd Natural-Gas-Market-Analytics-Contract-Valuation-System
```

Install required packages:
```bash
pip install -r requirements.txt
```

Make sure Nat_Gas.csv is in the root directory.

## Usage

### Price Forecasting

Run the forecasting model:
```bash
python task_1_chase.py
```

The script will load historical data, train the model, and provide an interactive interface to estimate prices for any future date.

Example output:
```
Model RMSE: $0.45
Model R²: 0.8234

Estimated price for June 2025: $3.12
95% Confidence Interval: $2.67 to $3.57
```

### Contract Valuation

Run the contract valuation system:
```bash
python task_2_chase.py
```

Example usage in code:
```python
from task_2_chase import GasStorageContract

pricer = GasStorageContract('Nat_Gas.csv')

injection_schedule = [
    {'date': '2021-05-31', 'volume': 100000}
]

withdrawal_schedule = [
    {'date': '2021-12-31', 'volume': 100000}
]

result = pricer.price_contract(
    injection_schedule=injection_schedule,
    withdrawal_schedule=withdrawal_schedule,
    injection_rate=150000,
    withdrawal_rate=150000,
    max_volume=500000,
    storage_cost_per_unit=0.005,
    verbose=True
)
```

Example output:
```
CONTRACT VALUE CALCULATION
Withdrawal Revenue:        $312,000.00
Injection Cost:          - $287,500.00
Storage Cost:            -  $10,250.00
NET CONTRACT VALUE:         $14,250.00
```

## Methodology

### Forecasting Approach
The model uses polynomial regression with seasonal features extracted through sine and cosine transformations. Facebook's Prophet algorithm is also implemented for comparison. The approach includes:
- Feature engineering with temporal components
- Time series cross-validation
- Confidence interval estimation using residual standard deviation

### Valuation Approach
The contract valuation system calculates net present value through:
- Historical price matching for injection and withdrawal dates
- Cash flow calculation: injection costs, withdrawal revenue, and storage costs
- Constraint validation for physical and operational limits
- Net contract value calculation

## Key Results

Natural gas prices show strong seasonal patterns with higher prices in winter months. Storage contracts achieve profitability by capturing the seasonal price spread between summer injection periods and winter withdrawal periods. Storage costs typically represent 5-10% of total contract value.

## Requirements

Create a requirements.txt file with:
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.5.0
scikit-learn>=1.2.0
prophet>=1.1.0
```

## Future Work

- Add volatility forecasting using GARCH models
- Implement Monte Carlo simulation for uncertainty analysis
- Integrate real-time price data APIs
- Build interactive dashboard for visualization
- Optimize injection and withdrawal schedules using dynamic programming
- Include weather data as additional predictive features

## License

This project is available under the MIT License.

## Author

Abdsalam
GitHub: github.com/abdsalam25
