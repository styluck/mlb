# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:39:27 2024

@author: lich5
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

# Data preprocessing
t = np.linspace(0, 2 * np.pi, 500)
y = 100 * np.sin(t)

# Generate noise
noise = np.random.normal(0, 15, 500)
y_hat = y + noise

# Plot the data
plt.figure()
plt.plot(t, y_hat, 'b', linewidth=2, label='y_hat (noisy signal)')
plt.plot(t, y, 'r', linewidth=3, label='y (original signal)')
plt.legend()
plt.show()

# mean filtering 均值滤波
window_size = 30  # Define the size of the moving window
y_denoised = np.convolve(y_hat, np.ones(window_size)/window_size, mode='same')

# Plot the data
plt.figure()
plt.plot(t, y_hat, 'b', linewidth=2, label='y_hat (noisy signal)')
plt.plot(t, y, 'r', linewidth=3, label='y (original signal)')
plt.plot(t, y_denoised, 'g', linewidth=2, label='y_denoised (mean filtered)')
plt.legend()
plt.show()

# Median filtering
kernel_size = 29  # Define the size of the filtering window (must be odd)
y_denoised = medfilt(y_hat, kernel_size=kernel_size)

# Plot the data
plt.figure()
plt.plot(t, y_hat, 'b', linewidth=2, label='y_hat (noisy signal)')
plt.plot(t, y, 'r', linewidth=3, label='y (original signal)')
plt.plot(t, y_denoised, 'g', linewidth=2, label='y_denoised (median filtered)')
plt.legend()
plt.show()

# [EOF]
