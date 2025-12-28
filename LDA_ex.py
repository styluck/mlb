# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:40:58 2024

@author: lich5
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the MNIST dataset
print("Loading the MNIST dataset...")
mnist = fetch_openml("mnist_784", version=1)
X, y = mnist.data, mnist.target.astype(int)

# Reduce the data for faster processing (optional)
# X, y = X[:20000], y[:20000]  # Use a subset (20,000 samples)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#########################
'''# Standardize the data'''
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#########################

#%% Evaluate classification performance on LDA-transformed data


#########################
'''Apply Linear Discriminant Analysis (LDA)'''
lda = LinearDiscriminantAnalysis(n_components=9)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)


#########################

print(f"Explained variance ratio (components): {lda.explained_variance_ratio_}")

# Visualize the first two LDA components
plt.figure(figsize=(10, 8))
for label in np.unique(y_train):
    plt.scatter(X_train_lda[y_train == label, 0], X_train_lda[y_train == label, 1], label=str(label), alpha=0.7)

plt.title("LDA: First Two Discriminant Components of MNIST")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend(loc="best")
plt.grid()
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train a logistic regression classifier on the LDA-transformed data
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_lda, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test_lda)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on LDA-transformed MNIST data: {accuracy:.4f}")


# # 加载模型
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

'''利用predict命令，输入x_test生成测试样本的测试值'''
plot_mnist_3_3(images, y_, y_pred[idx])



# [EOF]