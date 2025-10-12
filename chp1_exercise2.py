# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 09:36:46 2024

@author: lich5
"""

#%% 数据清洗
# 载入adj_factor、total_mv、close中两个市场的数据。

# 使用 concat() 合并两个市场的数据。

# 计算adjusted_close数据。

# 删除数据中，全为缺失值的行，并删除重复的行、列。

# 将日期作为DataFrame的索引，并按照时间从前到后排序。

# 删除有效长度不超过400行的列数据。
missing_values = closeprice.isna().sum()
for i in dataset.keys():
    dataset[i] = dataset[i][:,missing_values<len(closeprice.index)-400]
# *计算Benchmark的VWAP数据。

#%% 时间序列数据
# 从 日期 列中提取年份、月份信息。

# 将时间序列数据按日、月或季度重新采样。

# *统计每个月股票的涨跌幅比例，绘制柱状图。

#%% 数据可视化
# 使用 pandas 内置的 plot() 方法绘制benchmark close数据的折线图与volume数据的柱状图。

# 绘制benchmark收益率数据的直方图、散点图。

# *使用 groupby() 对市值数据按（30亿以下，30-50亿，50-100亿，100亿-500亿，500亿以上）
# 进行分组后，绘制每个组的平均值图表。

# *对最后一期的市值数据，使用 boxplot() 绘制分组后的市值数据的箱线图。

#%% 高级操作
# 使用 shift() 函数对 total_mv数据进行移动。

# *使用重采样的周度数据计算adjusted_close收益率数据与total_mv(-1)中两个数值列之间的相关系数。
# 这里的(-1)指的是上一期的数据。

# *使用 rolling() 计算adjusted_close数据的（5日，10日，20日）移动平均价格。
