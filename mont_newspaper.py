# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 09:18:58 2024

@author: lich5
"""
import numpy as np


# 定义参数
p = 10  # 售价
c = 6   # 进价
v = 2   # 回收价
mu = 50  # 需求分布的均值
sigma = 10  # 需求分布的标准差
N = 10000  # 模拟次数


# 生成订购量范围
Q_range = np.arange(30, 80)
max_mean_profit = -np.inf
optimal_Q = None


for Q in Q_range:
    profit_samples = []
    for _ in range(N):
        demand = np.random.normal(mu, sigma)
        if demand >= Q:
            profit = (p - c) * Q
        else:
            profit = (p - v) * demand+(v - c) * (Q - demand)
        profit_samples.append(profit)
    mean_profit = np.mean(profit_samples)
    if mean_profit > max_mean_profit:
        max_mean_profit = mean_profit
        optimal_Q = Q
print("最优订购量:", optimal_Q)
print("最大期望利润:", max_mean_profit)

# [EOF]