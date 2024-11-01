import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import timedelta

# Load data
data = pd.read_csv("export/BTC_USD_Historical_Data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.sort_index(inplace=True)
data = data[['Close']]

# Create lagged features for SVR
def create_lagged_features(data, lag=1):
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = data['Close'].shift(i)
    return data.dropna()

# Prepare data with lagged features
lag_days = 30  # Use the last 30 days as features
data_lagged = create_lagged_features(data, lag=lag_days)

# Split data into training and test sets
X = data_lagged.drop('Close', axis=1)
y = data_lagged['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train SVR model
svr_model = SVR(kernel='rbf')  # Using RBF kernel
svr_model.fit(X_train, y_train)

# Forecast and evaluation function for different forecast periods
def forecast_and_evaluate(ax, forecast_days, color):
    # Prepare input for forecasting
    last_features = data_lagged.iloc[-1].drop('Close').values.reshape(1, -1)  # Use .values.reshape(1, -1)
    forecast = []
    
    for _ in range(forecast_days):
        pred = svr_model.predict(last_features)[0]
        forecast.append(pred)
        
        # Update features for the next prediction
        last_features = np.append(last_features[:, 1:], pred).reshape(1, -1)  # Append prediction and reshape

    forecast_index = [data.index[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
    
    # Select the last 'forecast_days' actual prices for evaluation
    y_true_svr = data['Close'].iloc[-forecast_days:].values
    mse_svr = mean_squared_error(y_true_svr, forecast)
    mae_svr = mean_absolute_error(y_true_svr, forecast)
    
    # Calculate MAPE and Accuracy
    mape_svr = np.mean(np.abs((y_true_svr - forecast) / y_true_svr)) * 100
    accuracy_svr = 100 - mape_svr
    
    print(f'SVR ({forecast_days} days) - MSE: {mse_svr:.2f}, MAE: {mae_svr:.2f}, MAPE: {mape_svr:.2f}%, Akurasi: {accuracy_svr:.2f}%')
    
    # Limit data to the last 30 days for plotting
    recent_data = data[-30:]
    
    # Plotting actual vs forecasted prices in the specified subplot
    ax.plot(recent_data.index, recent_data['Close'], label='Harga Aktual', color='magenta')
    ax.plot(forecast_index, forecast, label=f' Harga Prediksi {forecast_days} Hari', color=color, linestyle='--', linewidth=2)
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
fig.suptitle('Prediksi Harga Bitcoin dengan Model Support Vector Regression', fontsize=16)
axs[1, 1].axis('off')  # Disable the last (empty) subplot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the main title
plt.show()
