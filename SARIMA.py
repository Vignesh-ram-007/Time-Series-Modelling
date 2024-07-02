import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

np.random.seed(42)
date_range = pd.date_range(start='2020-01-01', end='2021-01-01', freq='D')
temperature = np.random.normal(loc=20, scale=5, size=len(date_range))
temperature_series = pd.Series(temperature, index=date_range)

temperature_interpolated = temperature_series.interpolate(method='linear')

window_size = 7
temperature_sliding = temperature_series.rolling(window=window_size).mean()

temperature_expanding = temperature_series.expanding().mean()

temperature_rolling = temperature_series.rolling(window=window_size).mean()

temperature_upsampled = temperature_series.resample('H').interpolate(method='linear')

temperature_downsampled = temperature_series.resample('W').mean()

def fit_sarima_model(data, order=(1,1,1), seasonal_order=(1,1,1,12)):
    train, test = train_test_split(data, test_size=0.2, shuffle=False)
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    predictions = model_fit.forecast(steps=len(test))
    mse = mean_squared_error(test, predictions)
    return model_fit, predictions, mse

models = {
    'Original': temperature_series,
    'Interpolated': temperature_interpolated,
    'Sliding Window': temperature_sliding,
    'Expanding Window': temperature_expanding,
    'Rolling Window': temperature_rolling,
    'Upsampled': temperature_upsampled,
    'Downsampled': temperature_downsampled
}

results = {}
for name, data in models.items():
    model_fit, predictions, mse = fit_sarima_model(data.dropna())
    results[name] = {
        'model': model_fit,
        'predictions': predictions,
        'mse': mse
    }

fig, axs = plt.subplots(len(models), 1, figsize=(12, 24), sharex=True)
for i, (name, result) in enumerate(results.items()):
    axs[i].plot(result['model'].data.endog, label='Train')
    axs[i].plot(result['model'].predict(start=result['model'].nobs, end=result['model'].nobs+len(result['predictions'])-1), label='Fit')
    axs[i].plot(result['predictions'], label='Forecast')
    axs[i].legend()
    axs[i].set_title(f'SARIMA Model - {name} (MSE: {result["mse"]:.2f})')

plt.tight_layout()
plt.show()
