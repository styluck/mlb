# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:31:54 2024

@author: lich5
"""
import numpy as np

# 基础资产当前价格
S0 = 100
# 行权价格
K = 105
# 无风险利率
r = 0.05
# 波动率
sigma = 0.2
# 到期时间（以年为单位）
T = 1
# 模拟的路径数量
N = int(1e5)
# 时间步数
n = 52

# 计算时间步长
dt = T / n

# 生成随机路径并计算期权收益
option_payoffs = []
for _ in range(N):
    St = S0
    for _ in range(n):
        epsilon = np.random.normal()
        St = St + r * St * dt + sigma * St * np.sqrt(dt) * epsilon
    # payoff = max(St - K, 0)
    payoff = max(K - St, 0)
    option_payoffs.append(payoff)

# 贴现到期收益并计算期权价格
option_price = np.mean([payoff * np.exp(-r * T) for payoff in option_payoffs])

print("看涨期权价格:", option_price)