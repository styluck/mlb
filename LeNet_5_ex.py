# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 10:55:58 2024

@author: lich5
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns


# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 加载Fashion MNIST数据集
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# 对数据进行预处理
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = tf.reshape(x_train, [-1, 28, 28, 1])
x_train = tf.pad(x_train, [[0,0],[2,2],[2,2],[0,0]], 'CONSTANT')
x_test = tf.reshape(x_test, [-1, 28, 28, 1])
x_test = tf.pad(x_test, [[0,0],[2,2],[2,2],[0,0]], 'CONSTANT')

train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train))
test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test))

train_ds = train_ds.take(10000).shuffle(60000).batch(100)
test_ds = test_ds.shuffle(60000).batch(100)

'''# 创建模型'''


'''# 训练模型'''
start = time.perf_counter()


end = time.perf_counter() # time.process_time()
c=end-start 
print("程序运行总耗时:%0.4f"%c, 's') 


'''利用predict命令，输入x_test生成测试样本的测试值'''


# %% evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# model = tf.keras.models.load_model('my_model.h5')
idx = np.random.randint(1e4,size=9)
images = x_test.numpy().squeeze()[idx,:]
y_ = y_test[idx]
# 测试模型
def plot_mnist_3_3(images, y_, y=None):
    assert images.shape[0] == len(y_)
    fig, axes = plt.subplots(3, 3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape([32,32]), cmap='binary')
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


