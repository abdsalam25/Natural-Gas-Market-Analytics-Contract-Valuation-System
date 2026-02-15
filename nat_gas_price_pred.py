import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from prophet import Prophet

#Load the data
df = pd.read_csv('Nat_Gas.csv')
df['Dates'] = pd.to_datetime(df['Dates'])
df['Prices'] = df['Prices'].astype(float)

#check thw data
print(df.head())
print(f"Date range:{df['Dates'].min()} to {df['Dates'].max()}")
print(f"Price range:${df['Prices'].min():.2f} to ${df['Prices'].max():.2f}")

#plotting garph for price over time
plt.figure(figsize=(12,6))
plt.plot(df['Dates'], df['Prices'], marker='o')
plt.xlabel ('Date')
plt.ylabel ('Price ($)')
plt.title ('Natural Gas Prices over Time')
plt.grid(True)
plt.show()

#check for season
df['Month'] = df['Dates'].dt.month
monthly_avg = df.groupby('Month')['Prices'].mean()
print(monthly_avg)

first_date = df['Dates'].min()
df['days_since_start'] = (df['Dates'] - first_date).dt.days

#seasonal cycle
df['month'] = df['Dates'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12) #using sin and cose to have months in a circle

#features and target
features = ['days_since_start', 'month_sin', 'month_cos']
X = df[features].values
y = df['Prices'].values

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

predictions = model.predict(X_poly)
rmse = np.sqrt(mean_squared_error(y, predictions))
r2 = r2_score(y, predictions)
print(f"Model RMSE: ${rmse:.2f}")
print(f"Model R²: {r2:.4f}")

def estimate_price(date_input, model, poly, first_date):
  date_obj = pd.to_datetime(date_input)
  days_since_start = (date_obj - first_date).days
  month = date_obj.month
  month_sin = np.sin(2 * np.pi * month / 12)
  month_cos = np.cos(2 * np.pi * month / 12)
  X_new = np.array([[days_since_start, month_sin, month_cos]])
  X_new_poly = poly.transform(X_new)
  price = model.predict(X_new_poly)[0]
  return price

#example
price_2025 = estimate_price('2025-06-30', model, poly, first_date)
print(f"Estimated price for June 2025: ${price_2025:.2f}")

#forecast for next 12 months
last_date = df['Dates'].max()
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='M')
#pred for each date
forecast_prices = [estimate_price(date, model, poly, first_date) for date in future_dates]

forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Price': forecast_prices})
print(forecast_df)

#uncertainities
residuals = y - predictions
std_dev = np.std(residuals)
lower_bound = predictions - 1.96 * std_dev
upper_bound = predictions + 1.96 * std_dev

#trying facebooks forecasting tool
df_prophet = df.rename(columns={'Dates': 'ds', 'Prices': 'y'})
model_prophet = Prophet(yearly_seasonality=True)
model_prophet.fit(df_prophet)
future = model_prophet.make_future_dataframe(periods=365)
forecast = model_prophet.predict(future)
model_prophet.plot(forecast)
plt.title('Prophet Forecast')
plt.show()
plt.figure(figsize=(14,7))
plt.plot(df['Dates'], y, label='Historical Prices', marker='o')
plt.plot(df['Dates'], predictions, label='Model Fit', linestyle='--')
plt.fill_between(df['Dates'], lower_bound, upper_bound, alpha=0.3, label='95% Confidence Interval')
plt.plot(future_dates, forecast_prices, 'ro-', label='12-Month Forecast')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.title('Natural Gas Price Forecast')
plt.legend()
plt.grid(True)
plt.show()

print("\n=== MODEL SUMMARY ===")
print(f"Training period: {df['Dates'].min()} to {df['Dates'].max()}")
print(f"Forecast period: {future_dates[0]} to {future_dates[-1]}")
print(f"\nModel Performance:")
print(f"  RMSE: ${rmse:.2f}")
print(f"  R²: {r2:.4f}")

print("\n=== NATURAL GAS PRICE ESTIMATOR ===")
print("Enter a date to get a price estimate")
print("Format: YYYY-MM-DD (e.g., 2025-12-31)")

while True:
    user_date = input("\nEnter date (or 'quit' to exit): ")

    if user_date.lower() == 'quit':
        print("Thank you for using the price estimator!")
        break

    try:
        estimated_price = estimate_price(user_date, model, poly, first_date)

        # Calculate confidence interval for this prediction
        date_obj = pd.to_datetime(user_date)
        lower = estimated_price - 1.96 * std_dev
        upper = estimated_price + 1.96 * std_dev

        print(f"\nEstimated price for {user_date}: ${estimated_price:.2f}")
        print(f"95% Confidence Interval: ${lower:.2f} to ${upper:.2f}")

    except Exception as e:
        print(f"Error: {e}")
        print("Please enter a valid date in YYYY-MM-DD format")




tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
