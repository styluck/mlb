# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:21:45 2024

@author: 6
"""
#%% 数据导入与创建
# 创建一个 pandas 的 DataFrame，包含一列是从 1 到 10 的整数。

# 使用字典数据来创建一个 DataFrame，列名为 ['A', 'B']，数据为 A=[1,2,3], B=[4,5,6]。

# 从benchmark.csv文件中导入数据，并将其转换为 DataFrame。

# 从close_price.xlsx文件的close_price_sh工作表中读取数据，并存储为 DataFrame。

# 将 closeprice 转换为 CSV 文件并保存到本地。
#%% 数据处理
# 添加一列到benchmark，新列的数据为'close'和'open'两列数据之差，命名为'change'。

# 将'Unnamed: 0'重命名为'time'。

# 对 benchmark 按'time'的值进行排序。

# 将'time'列设为索引。

# 删除 benchmark 中的'amount'数据。

# 根据close计算收益率

# 计算 pct_chg 中某列的最大值、最小值和平均值。


#%% 数据选择与过滤
# 选择 benchmark 中的'close'数据。

# 选择 DataFrame 中的多列数据：'close','open','high','low'。

# 从 benchmark 中选择行号为 3 到 7 的数据。

# 找出 benchmark 中开盘价比收盘价高的数据。

# 使用 loc 和 iloc 获取特定行和列的数据。

#%% 将某列的数据替换为指定的值（例如将所有 0 替换为 NaN）。
# 计算 closeprice 中每一列的缺失值数量。

# 填充 closeprice 中的缺失值。

# 对 benchmark 中的数据进行去重操作。

# 使用 apply() 方法对 benchmark 应用自定义函数。

