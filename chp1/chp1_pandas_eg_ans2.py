# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:21:45 2024

@author: 6
"""
import pandas as pd
import numpy as np
#%% 数据导入与创建
# 创建一个 pandas 的 DataFrame，包含一列是从 1 到 10 的整数。
df = pd.DataFrame({'Numbers': range(1, 11)})
df = pd.DataFrame()
df['Numbers'] = range(1,11)
# 使用字典数据来创建一个 DataFrame，列名为 ['A', 'B']，数据为 A=[1,2,3], B=[4,5,6]。
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)
# 从benchmark.csv文件中导入数据，并将其转换为 DataFrame。
benchmark = pd.read_csv('chp1_2_data\\benchmark.csv')
benchmark.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
# 查看前几行数据
print(benchmark.head())

# 简单统计信息
print(benchmark.describe())

# 按年龄排序
print(benchmark.sort_values('time', ascending=False))
# 从close_price.xlsx文件的close_price_sh工作表中读取数据，并存储为 DataFrame。
closeprice = pd.read_excel('chp1_2_data\\close_price.xlsx', 
                           sheet_name='close_price_sh')
# 将 closeprice 转换为 CSV 文件并保存到本地。
closeprice.to_csv('close_price.csv', index=False)

#%% 数据处理
# 添加一列到benchmark，新列的数据为'close'和'open'两列数据之差。
benchmark['change'] = benchmark['close'] - benchmark['open']
# 对 benchmark 按'Unnamed: 0'的值进行排序。
df_sorted = benchmark.sort_values(by='time')
# 将'Unnamed: 0'列设为索引。
benchmark = benchmark.set_index('time')
# 删除 benchmark 中的'amount'数据。
benchmark = benchmark.drop(columns=['amount'])
# 根据close计算收益率
benchmark['pct_chg'] = benchmark['close'].pct_change()
# 计算 pct_chg 中某列的最大值、最小值和平均值。
max_value = benchmark['pct_chg'].max()
min_value = benchmark['pct_chg'].min()
mean_value = benchmark['pct_chg'].mean()

# 将行列进行交换
df_transposed = df.T

#%% 数据选择与过滤
# 选择 benchmark 中的'close'数据。
benchmark_close = benchmark['close']
# 选择 DataFrame 中的多列数据：'close','open','high','low'。
benchmark_cohl = benchmark[['close','open','high','low']]
# 从 benchmark 中选择行号为 3 到 7 的数据。
# 注意索引从 0 开始，所以是第 4 到第 8 行
benchmark_3_7 = benchmark.iloc[3:8]
# 找出 benchmark 中开盘价比收盘价高的数据。
filtered_df = benchmark[benchmark['close'] > benchmark['open']]
# 使用 loc 和 iloc 获取特定行和列的数据。
data_iloc = benchmark.iloc[1, 0]  # 第2行第1列
data_loc = benchmark.loc[2, 'close'] 

#%% 将某列的数据替换为指定的值（例如将所有 0 替换为 NaN）。
# 计算 closeprice 中每一列的缺失值数量。
missing_values = closeprice.isna().sum()
# 填充 closeprice 中的缺失值。
df_filled = closeprice.fillna(method='ffill')
df_filled = closeprice.fillna(0)
# 对 benchmark 中的数据进行去重操作。
benchmark = benchmark.loc[~benchmark.index.duplicated(),:]
# 使用 apply() 方法对 benchmark 应用自定义函数。
benchmark['ma10'] = benchmark.rolling.apply(lambda x:np.mean(x))
