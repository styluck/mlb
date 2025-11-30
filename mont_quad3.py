# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:33:58 2024

@author: lich5
"""
import numpy as np
from scipy.integrate import tplquad


def quad3mont(n):
    """
    [V0, Vm] = quad3mont(n), 蒙特卡洛方法计算3重积分，返回理论值V0和模拟值Vm.
    输入参数n是随机投点的个数，可以是正整数标量或向量.
    
    """

    # 计算理论积分值（传统数值算法）
    def fun(x, y, z):
        return x * y * z

    def ymin(x):
        return x

    def ymax(x):
        return 2 * x

    def zmin(x, y):
        return x * y

    def zmax(x, y):
        return 2 * x * y


    V0, _ = tplquad(fun, 1, 2, ymin, ymax, lambda x, y: x* y, lambda x, y: zmax(x, y))

    # 构造被积函数
    def fun_vec(x):
        return np.prod(x, axis=0)

    Vm = np.zeros(len(n))
    # 求体积的蒙特卡洛模拟值
    for i in range(len(n)):
        # 在立方体（1<=x<=1, 1<=y<=4, 1<=z<=16）内随机投n(i)个点
        x = np.random.uniform(1, 2, n[i])  # x坐标
        y = np.random.uniform(1, 4, n[i])  # y坐标
        z = np.random.uniform(1, 16, n[i])  # z坐标
        X = np.vstack((x, y, z))
        id = (y >= x) & (y <= 2 * x) & (z >= x * y) & (z <= 2 * x * y)  # 落入积分区域内点的坐标索引
        Vm[i] = (4 - 1) * (16 - 1) * np.sum(fun_vec(X[:, id])) / n[i]  # 求积分的模拟值

    return V0, Vm

if __name__ == "__main__":
    n_values = np.array([10, 100, 1000, 10000, 100000, 1000000])

    V0, Vm = quad3mont(n_values)

    print(f"理论积分值 V0 = {V0:.8f}")
    print("不同随机投点数 n 下的蒙特卡洛模拟结果：")
    print("-" * 60)
    print(f"{'n':>10s} | {'Vm(模拟值)':>16s} | {'误差 Vm - V0':>16s}")
    print("-" * 60)
    for n_i, vm_i in zip(n_values, Vm):
        err = vm_i - V0
        print(f"{n_i:10d} | {vm_i:16.8f} | {err:16.8e}")
        
# [EOF]