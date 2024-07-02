import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)
date_range = pd.date_range(start='2020-01-01', end='2021-01-01', freq='D')
temperature = np.random.normal(loc=20, scale=5, size=len(date_range))
temperature[::10] = np.nan
temperature_series = pd.Series(temperature, index=date_range)

temperature_interpolated = temperature_series.interpolate(method='linear')

window_size = 7
temperature_sliding = temperature_series.rolling(window=window_size).mean()

temperature_expanding = temperature_series.expanding().mean()

temperature_rolling = temperature_series.rolling(window=window_size).mean()

fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

axs[0].plot(temperature_series, label='Original')
axs[0].plot(temperature_interpolated, label='Interpolated')
axs[0].legend()
axs[0].set_title('Linear Interpolation')

axs[1].plot(temperature_series, label='Original')
axs[1].plot(temperature_sliding, label='Sliding Window')
axs[1].legend()
axs[1].set_title(f'Sliding Window (Window Size = {window_size})')

axs[2].plot(temperature_series, label='Original')
axs[2].plot(temperature_expanding, label='Expanding Window')
axs[2].legend()
axs[2].set_title('Expanding Window')

axs[3].plot(temperature_series, label='Original')
axs[3].plot(temperature_rolling, label='Rolling Window')
axs[3].legend()
axs[3].set_title(f'Rolling Window (Window Size = {window_size})')

plt.tight_layout()
plt.show()
