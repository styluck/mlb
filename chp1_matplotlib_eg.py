# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 19:09:39 2024

@author: 6
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 07:22:59 2024

@author: 6
"""
import numpy as np
import matplotlib.pyplot as plt

#%% 绘制[0, 2*pi]区间的sin曲线，设置线段颜色、类型、marker类型、
# 并添加轴标题。


#%% 绘制单位圆、[0, 2*pi]区间的sin、cos曲线，设置线段颜色、类型、
# marker类型、并添加标签。

#%% 绘制3x² + 2xy + 4y² = 5的曲线


#%% 生成1000个服从正态分布随机样本，满足均值为10，方差为20，并绘制的直方图

#%% 用以下数据绘制饼图
x = [10, 10, 20, 25, 35]
labels = ['A', 'B', 'C', 'D', 'E']


#%% 用以下数据绘制三维的表面图
t = np.linspace(-np.pi, np.pi, 20)
X, Y = np.meshgrid(t, t)
Z = np.cos(X) * np.sin(Y)
