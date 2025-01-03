# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:21:38 2024

@author: lich5
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load the MNIST dataset
print("Loading the MNIST dataset...")
mnist = fetch_openml("mnist_784", version=1)
X, y = mnist.data, mnist.target.astype(int)

# Reduce the data for faster processing (optional)
# X, y = X[:20000], y[:20000]  # Use a subset (20,000 samples)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% Distance Discriminant Classifier

# Compute class centroids
class_centroids = {}
for label in np.unique(y_train):
    class_centroids[label] = X_train[y_train == label].mean(axis=0)


def predict_using_centroids(X, centroids):
    """
    Predict the class for each sample in X based on the closest centroid.
    """
    predictions = []
    for sample in X:
        distances = {label: np.linalg.norm(sample - centroid) for label, centroid in centroids.items()}
        predicted_label = min(distances, key=distances.get)  # Find the closest centroid
        predictions.append(predicted_label)
    return np.array(predictions)

# Make predictions on the test set
y_pred = predict_using_centroids(X_test, class_centroids)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

#%% 加载模型
# 随机选择9张图片进行展示
idx = np.random.randint(4e3,size=9)
images = X_test.squeeze()[idx,:]
y_ = y_test[idx]
# 测试模型
def plot_mnist_3_3(images, y_, y=None):
    assert images.shape[0] == len(y_)
    fig, axes = plt.subplots(3, 3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape([28,28]), cmap='binary')
        if y is None:
            xlabel = 'True: {}'.format(y_[i])
        else:
            xlabel = 'True: {0}, Pred: {1}'.format(y_[i], y[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


plot_mnist_3_3(images, y_, y_pred[idx])
# [EOF]
