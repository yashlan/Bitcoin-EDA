import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

# Load data
data = pd.read_csv("export/BTC_USD_Historical_Data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.sort_index(inplace=True)
data = data[['Close']]

# Train ARIMA model
arima_model = ARIMA(data['Close'], order=(3, 1, 3)) # p, d, q
arima_fit = arima_model.fit()

# Forecast and evaluation function for different forecast periods
def forecast_and_evaluate(ax, forecast_days, color):
    # Generate forecast
    forecast = arima_fit.forecast(steps=forecast_days)
    forecast_index = [data.index[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
    
    # Select the last 'forecast_days' actual prices for evaluation
    y_true_arima = data['Close'].iloc[-forecast_days:].values
    mse_arima = mean_squared_error(y_true_arima, forecast)
    mae_arima = mean_absolute_error(y_true_arima, forecast)
    
    # Calculate MAPE and Accuracy
    mape_arima = np.mean(np.abs((y_true_arima - forecast) / y_true_arima)) * 100
    accuracy_arima = 100 - mape_arima
    
    print(f'ARIMA ({forecast_days} days) - MSE: {mse_arima:.2f}, MAE: {mae_arima:.2f}, MAPE: {mape_arima:.2f}%, Akurasi: {accuracy_arima:.2f}%')
    
    # Limit data to the last 30 days for plotting
    recent_data = data[-30:]
    
    # Plotting actual vs forecasted prices in the specified subplot
    ax.plot(recent_data.index, recent_data['Close'], label='Harga Aktual', color='magenta')
    ax.plot(forecast_index, forecast, label=f' Harga Prediksi {forecast_days} Hari', color=color, linestyle='--', linewidth=2)  # Gaya garis diperbarui
    ax.set_title(f'Prediksi {forecast_days} Hari')
    ax.set_ylabel('Harga (USD)')
    ax.legend()
    ax.set_xlim([recent_data.index[0], forecast_index[-1]])  # Show only the last 30 days of data + forecast

# Set up 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Run forecasts for 7, 14, and 30 days on different subplots
forecast_and_evaluate(axs[0, 0], 7, 'red')
forecast_and_evaluate(axs[0, 1], 14, 'green')
forecast_and_evaluate(axs[1, 0], 30, 'blue')

# Adjust layout
fig.suptitle('Prediksi Harga Bitcoin dengan Model ARIMA', fontsize=16)
axs[1, 1].axis('off')  # Disable the last (empty) subplot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the main title
plt.show()
