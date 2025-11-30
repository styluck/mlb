# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 16:06:58 2025

@author: lich5
"""
import numpy as np

try:
    from scipy import integrate
except ImportError:
    integrate = None


def quad2mont1(n, use_scipy=True, random_state=None):
    """
    Python 版 quad2mont1(n)

    计算球面 x^2 + y^2 + z^2 = 4 被圆柱面 x^2 + y^2 = 2*x 所截得的、
    含在圆柱面内部分立体的体积的：
      - 理论值 V0
      - Monte Carlo 模拟值 Vm

    参数
    ----
    n : int 或 一维数组/列表
        随机投点的个数，可以是标量或向量。
    use_scipy : bool, optional
        若为 True 且已安装 SciPy，则使用 dblquad 做数值积分求 V0；
        否则使用解析公式求 V0。
    random_state : int 或 numpy.random.Generator, optional
        用于控制随机种子，便于复现。

    返回
    ----
    V0 : float
        体积的理论值（数值解或解析公式）。
    Vm : float 或 numpy.ndarray
        Monte Carlo 模拟得到的体积估计值。
        如果 n 是标量，则返回 float；如果 n 是向量，则返回同长度的一维数组。
    """

    # ---------- 1. 理论体积 V0 ----------
    if use_scipy and (integrate is not None):
        # 对应 MATLAB:
        # V0 = 4*quad2d(@(x,y)sqrt(4-x.^2-y.^2), 0, 2, 0, @(x)sqrt(1-(1-x).^2));
        def integrand(y, x):
            # 注意 dblquad 的参数顺序是 (func, x_min, x_max, y_min, y_max)
            # func(y, x)
            return np.sqrt(4 - x**2 - y**2)

        def y_min(x):
            return 0.0

        def y_max(x):
            return np.sqrt(1 - (1 - x)**2)

        # 只积分 z >= 0, y >= 0 的部分，再乘 4 利用对称性
        res, err = integrate.dblquad(integrand, 0.0, 2.0, y_min, y_max)
        V0 = 4.0 * res
    else:
        # 对应 MATLAB 里注释掉的解析公式：
        # V0 = 32*(pi/2-2/3)/3;
        V0 = 32 * (np.pi / 2 - 2 / 3) / 3

    # ---------- 2. Monte Carlo 模拟 Vm ----------
    # 统一把 n 转成一维整数数组，方便逐个模拟
    n_arr = np.atleast_1d(n).astype(int)

    # 随机数发生器
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    Vm = np.empty_like(n_arr, dtype=float)

    for idx, ni in enumerate(n_arr):
        # 对应 MATLAB:
        # x = 2*rand(n(i),1);   % x in [0, 2]
        # y = rand(n(i),1);     % y in [0, 1]
        # z = 2*rand(n(i),1);   % z in [0, 2]
        x = 2.0 * rng.random(ni)
        y = 1.0 * rng.random(ni)
        z = 2.0 * rng.random(ni)

        # 球内：x^2 + y^2 + z^2 <= 4
        inside_sphere = x**2 + y**2 + z**2 <= 4.0

        # 圆柱内：(x-1)^2 + y^2 <= 1  <=>  x^2 + y^2 = 2x
        inside_cylinder = (x - 1.0) ** 2 + y**2 <= 1.0

        # 同时落在球和圆柱内
        m = np.count_nonzero(inside_sphere & inside_cylinder)

        # 对应 MATLAB:
        # Vm(i) = 16*m/n(i);
        # 16 是完整对称区域的外包盒体积：
        # x in [0,2], y in [-1,1], z in [-2,2] => 2 * 2 * 4 = 16
        Vm[idx] = 16.0 * m / ni

    # 若输入是标量，就返回标量；否则返回数组
    if np.isscalar(n):
        Vm = Vm.item()

    return V0, Vm


# ---------- 3. 简单演示 ----------
if __name__ == "__main__":
    # 可以改成你想试验的样本量
    n_values = np.array([10, 100, 1000, 10000, 100000, 1000000])

    V0, Vm = quad2mont1(n_values, random_state=0)

    print(f"理论积分值 V0 = {V0:.8f}")
    print("不同随机投点数 n 下的蒙特卡洛模拟结果：")
    print("-" * 60)
    print(f"{'n':>10s} | {'Vm(模拟值)':>16s} | {'误差 Vm - V0':>16s}")
    print("-" * 60)
    for n_i, vm_i in zip(n_values, Vm):
        err = vm_i - V0
        print(f"{n_i:10d} | {vm_i:16.8f} | {err:16.8e}")
        
# [EOF]