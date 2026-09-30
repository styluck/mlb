# -*- coding: utf-8 -*-
"""第三章 Part 1 课堂练习。按 # %% 单元从上到下逐步补全。"""

# %% 准备数据：df 是保存三张宽表的字典（日期 × 股票）
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from data_io.io_framework import load_data

startdate, enddate = "2023-12-01", "2024-03-31"
# TODO: 1. 用 load_data 分别加载 close、adj_factor、total_mv、vol 
# 四个数据表，时间范围为 startdate 到 enddate。
# close = load_data("close", startdate, enddate) # 股票的收盘价 
# factor = load_data("adj_factor", startdate, enddate)  # 股票价格的复权因子（什么是复权？）
# total_mv = load_data("total_mv", startdate, enddate)  # 股票的总市值
# volume = load_data("vol", startdate, enddate)  # 股票的成交量


codes = close.columns  # 使用全部股票
adjusted_close = close[codes] * factor[codes] # 为什么要用复权因子调整收盘价？（提示：考虑分红、配股等因素对价格的影响）
daily_return = adjusted_close.pct_change(fill_method=None)
daily_return = daily_return.where(close[codes].notna() & close[codes].shift(1).notna())

df = {
    "return": daily_return.loc["2024-01-01":"2024-03-31"],
    "total_mv": total_mv.loc["2024-01-01":"2024-03-31", codes],
    "volume": volume.loc["2024-01-01":"2024-03-31", codes],
}
print({name: data.shape for name, data in df.items()})

# %% TODO 1：描述统计
# 依照课件中的代码顺序完成：
# 1. df["return"].iloc[-1]：计算均值、中位数、众数和 describe()。
# 2. 同一收益率截面：计算极差、样本方差、样本标准差和 IQR（ddof=1）。
# 3. df["total_mv"].iloc[-1]：计算偏度并画直方图，标出均值和中位数。
# 4. 收益率截面：计算超额峰度及普通峰度（超额峰度 + 3）。
# 5. 收益率截面：左图只显示 1%—99% 分位数范围的直方图 + KDE；
#    右侧箱线图仍使用完整截面，保留极端值信息。


# %% TODO 2：分布估计与诊断
# 对最后一日的 return 截面：用样本矩估计正态参数，再用 stats.norm.fit 和 stats.t.fit 拟合。
# 画直方图 + 两条拟合密度、正态与 t 分布的 Q-Q 图；计算 JB 检验。
# Shapiro-Wilk 对大样本的 p 值不稳定，超过 5000 个观测时固定随机种子抽取 5000 个。


# %% TODO 3：变换与标准化
# 最后一日 volume 截面：Z-score、Min-Max；total_mv 截面：稳健标准化、log1p。
# 保存变换参数；分母为 0 时输出 NaN；保留原始列。


# %% TODO 4：异常值
# 最后一日 return 截面：按 1.5×IQR 标记异常值；按 1%/99% 分位数分别缩尾、截尾。
# 比较原始、缩尾和截尾后的样本量与标准差。


# %% TODO 5：面板数据的两个方向
# volume.iloc[-1]：当日股票之间的横截面 Z-score。
# volume.shift(1).rolling(20)：逐只股票用前 20 期历史计算最后一日 Z-score。
# 比较 z_cross 与 z_time 的含义和非空样本数。
