import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# Generate synthetic multivariate time series data
date_range = pd.date_range(start='2020-01-01', end='2021-01-01', freq='D')
data1 = np.random.normal(loc=20, scale=5, size=len(date_range))
data2 = np.random.normal(loc=15, scale=3, size=len(date_range))

# Create a DataFrame for multivariate time series data
data = pd.DataFrame({'Data1': data1, 'Data2': data2}, index=date_range)

# Apply various feature engineering steps
data_interpolated = data.interpolate(method='linear')

window_size = 7
data_sliding = data.rolling(window=window_size).mean()

data_expanding = data.expanding().mean()

data_upsampled = data.resample('H').interpolate(method='linear')

data_downsampled = data.resample('W').mean()

def fit_varmax_model(data, order=(1, 0), seasonal_order=(0, 0, 0, 0)):
    train, test = train_test_split(data, test_size=0.2, shuffle=False)
    model = VARMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    predictions = model_fit.forecast(steps=len(test))
    mse = mean_squared_error(test, predictions)
    return model_fit, predictions, mse

models = {
    'Original': data,
    'Interpolated': data_interpolated,
    'Sliding Window': data_sliding,
    'Expanding Window': data_expanding,
    'Upsampled': data_upsampled,
    'Downsampled': data_downsampled
}

results = {}
for name, data in models.items():
    model_fit, predictions, mse = fit_varmax_model(data.dropna())
    results[name] = {
        'model': model_fit,
        'predictions': predictions,
        'mse': mse
    }

fig, axs = plt.subplots(len(models), 1, figsize=(12, 24), sharex=True)
for i, (name, result) in enumerate(results.items()):
    axs[i].plot(result['model'].endog[:, 0], label='Train Data1')
    axs[i].plot(result['model'].endog[:, 1], label='Train Data2')
    axs[i].plot(result['predictions'][:, 0], label='Forecast Data1')
    axs[i].plot(result['predictions'][:, 1], label='Forecast Data2')
    axs[i].legend()
    axs[i].set_title(f'VARMAX Model - {name} (MSE: {result["mse"]:.2f})')

plt.tight_layout()
plt.show()
