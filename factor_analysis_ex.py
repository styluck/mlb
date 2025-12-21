# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:08:45 2024

@author: lich5
"""
# import tensorflow as tf
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

# Load MNIST dataset from OpenML
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Extract data and labels
X_train = mnist.data  # Pixel values (70,000 samples x 784 features)
y_train = mnist.target  # Labels (digits 0-9)

# Normalize the data to [0, 1]
X_train = X_train / 255.0

print(f"Data shape: {X_train.shape}")  
print(f"Labels shape: {y_train.shape}")  

# (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize and flatten the data
X_train = X_train / 255.0  # Normalize to range [0, 1]
X_train_flat = X_train.reshape(-1, 28 * 28)  # Flatten to (num_samples, 784)

# Perform Factor Analysis: 50 factors
# X = AF+e
n_factors = 50
fa = FactorAnalysis(n_components=50)
X_factors = fa.fit_transform(X_train) # F


# Get factor loadings (components)
loadings = fa.components_ # A


# Visualize all 50 factors
fig, axes = plt.subplots(5, 10, figsize=(15, 7))  # Create a 5x10 grid of subplots
for i, ax in enumerate(axes.flat):
    ax.imshow(loadings[i].reshape(28, 28), cmap='gray')  # Reshape each factor into a 28x28 image
    ax.axis('off')  # Turn off axis display
    ax.set_title(f"Factor {i+1}", fontsize=8)  # Add factor title
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# Reconstruct data
X_reconstructed = X_factors @ loadings + fa.mean_


#%% Visualize original vs reconstructed images
fig, axes = plt.subplots(2, 10, figsize=(15, 5))
for i in range(10):
    # Original
    axes[0, i].imshow(X_train_flat[i].reshape(28, 28), cmap='gray')
    axes[0, i].axis('off')
    axes[0, i].set_title("Original")
    
    # Reconstructed
    axes[1, i].imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')
    axes[1, i].axis('off')
    axes[1, i].set_title("Reconstructed")
plt.tight_layout()
plt.show()

# Calculate reconstruction error (MSE)
reconstruction_error = np.mean((X_train_flat - X_reconstructed) ** 2)
print(f"Reconstruction Error (MSE): {reconstruction_error:.4f}")

# Calculate variance explained
total_variance = np.var(X_train_flat, axis=0).sum()
explained_variance = np.var(X_factors @ loadings, axis=0).sum()
variance_explained_ratio = explained_variance / total_variance
print(f"Variance Explained Ratio: {variance_explained_ratio:.2%}")

# [EOF]