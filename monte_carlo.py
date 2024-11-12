# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:26:50 2024

@author: lich5
"""
import numpy as np
import matplotlib.pyplot as plt

def mont_int_avg(f, x_min, x_max, N=1000000):
    Range = x_max - x_min
    x = np.random.rand(N, 1) * Range + x_min
    sum_avg = 0
    for i in range(N):
        area_i = f(x[i]) * Range
        sum_avg += area_i
    return sum_avg / N

def mont_int(f, x_min, x_max, N=1000000):
    # 蒙特卡罗方法计算定积分（随机投点法）
    rnd = np.random.rand(N, 2)
    xx = np.arange(x_min, x_max + 0.01, 0.01)
    x = x_min + (x_max - x_min) * rnd[:, 0]
    y_min = np.min(f(xx))
    y_max = np.max(f(xx))
    y = y_min + (y_max - y_min) * rnd[:, 1]
    i = np.where(y < f(x))
    outp = np.sum(i[0]) / N * (x_max - x_min) * (y_max - y_min) + y_min * (x_max - x_min)
    # 画图
    plt.figure()
    plt.plot(x, y, 'go', x[i], y[i], 'bo')
    plt.axis([x_min, x_max, y_min, y_max])
    plt.plot(xx, f(xx), 'r-', linewidth=2)
    return outp

if __name__=='__main__':
    import numpy as np
    from scipy.integrate import quad
    
    x_min = 0.001
    x_max = np.pi
    
    def f(x):
        return 1 / x
        # return np.power(x, 3) - np.power(x, 2)
    
    estimate = mont_int_avg(f, x_min, x_max)
    # estimate = mont_int(f, x_min, x_max)
    p, _ = quad(f, x_min, x_max)
    print(f'Real integral: {p:.4f}, Monte Carlo Estimate: {estimate[0]:.4f}')
    
    
    
# [eof]