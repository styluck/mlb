# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:28:49 2024

@author: lich5
"""
import numpy as np
import matplotlib.pyplot as plt


def PiMonteCarlo(n):
    """
    PiMonteCarlo(n)，用随机投点法模拟圆周率pi，作出模拟图。n为投点次数，可以是非负整数标量或向量。

    piva = PiMonteCarlo(n)，用随机投点法模拟圆周率pi，返回模拟值piva。若n为标量（向量），则piva也为标量（向量）。

    """
    m = len(n)  # 求变量n的长度
    pivalue = np.zeros((m, 1))  # 为变量pivalue赋初值

    # 通过循环用投点法模拟圆周率pi
    for i in range(m):
        x = 2 * np.random.rand(n[i], 1) - 1
        y = 2 * np.random.rand(n[i], 1) - 1
        d = x ** 2 + y ** 2
        pivalue[i] = 4 * np.sum(d <= 1) / n[i]  # 圆周率的模拟值

    if len(pivalue) == 1:
        piva = pivalue[0]
    else:
        piva = pivalue

    if plt.get_fignums() == []:
        # 不输出圆周率的模拟值，返回模拟图
        if m > 1:
            # 如果n为向量，则返回圆周率的模拟值与投点个数的散点图
            plt.figure()  # 新建一个图形窗口
            plt.plot(n, pivalue, 'k.')  # 绘制散点图
            h = plt.axhline(y=np.pi, c='k', linewidth=2)  # 添加参考线
            plt.text(1.05 * n[-1], np.pi, r'$\pi$', fontsize=15)  # 添加文本信息
            plt.xlabel('num of points')
            plt.ylabel(r'simulated $\pi$ value')  # 添加坐标轴标签
        else:
            # 如果n为标量，则返回投点法模拟圆周率的示意图
            plt.figure()  # 新建一个图形窗口
            plt.plot(x, y, 'k.')  # 绘制散点图
            plt.axhline(y=np.pi, c='k', linewidth=2)  # 添加参考线
            plt.text(1.05 * n[-1], np.pi, r'$\pi$', fontsize=15)  # 添加文本信息
            plt.xlabel('X')
            plt.ylabel('Y')  # 添加坐标轴标签
            plt.title(f'Simulated Pi ： {pivalue[0]}')  # 添加标题
            plt.axis([-1.1, 1.1, -1.1, 1.1])
            plt.axis('equal')  # 设置坐标轴属性

    return piva

PiMonteCarlo(np.arange(0,200000,100))
