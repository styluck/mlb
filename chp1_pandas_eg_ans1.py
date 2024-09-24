# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:21:45 2024

@author: 6
"""
import numpy as np
array = np.zeros(10)
array = np.ones(10)
array = np.ones(10)*2

array = np.arange(1, 11)
print(array)
array = array[1::2]
print(array)

array = np.random.rand(10,10)
array = np.random.randn(10,10)
array = np.random.randint(10,1,100)

array = array.T
dotmultiply = np.dot(array, array)
array @ array
array * array

_, s,_ = np.linalg.svd(array)
np.linalg.det(array)
np.linalg.norm(array)

[np.abs(i) for i in array]

(np.abs(i) for i in array)
#%% 数据导入与创建
import pandas as pd
# 创建一个 pandas 的 DataFrame，包含一列是从 1 到 10 的整数。
dct = {'Numbers':np.arange(1,11)}
df = pd.DataFrame(dct)
# 使用字典数据来创建一个 DataFrame，列名为 ['A', 'B']，数据为 A=[1,2,3], B=[4,5,6]。
A=[1,2,3]
B=[4,5,6]
dct = {'A':A, 'B':B}
df = pd.DataFrame(dct)

df = pd.DataFrame()
df['A'] = A
df['B'] = B
# 从benchmark.csv文件中导入数据，并将其转换为 DataFrame。
benchmark = pd.read_csv('chp1_2_data\\benchmark.csv')

# 将'Unnamed: 0'重命名为'time'。
benchmark.rename(columns={'Unnamed: 0':'time'}, inplace = True)

# 对 benchmark 按'time'的值进行排序。
# benchmark.sort_values(by = 'time', inplace = True)

# 将'time'列设为索引。
benchmark.set_index('time', inplace = True)

# 从close_price.xlsx文件的close_price_sh工作表中读取数据，并存储为 DataFrame。
close_price_sh = pd.read_excel('chp1_2_data\\close_price.xlsx',
                               sheet_name='close_price_sh')
# 将 closeprice 转换为 CSV 文件并保存到本地。
close_price_sh.to_csv('close_price.csv',index=False)
print(benchmark.describe())

#%% 数据处理
# 添加一列到benchmark，新列的数据为'close'和'open'两列数据之差，命名为'change'。
benchmark['change'] = benchmark['close'] - benchmark['open']
benchmark['change'] = benchmark['close'] - benchmark['close'].shift(1)

# 删除 benchmark 中的'amount'数据。
b1 =benchmark.drop(columns=['amount'])
b1 =benchmark.drop(['amount'], axis = 1)

# 根据close计算收益率
benchmark['pct_chg'] = benchmark['close'].pct_change()

# 计算 pct_chg 中某列的最大值、最小值和平均值。
max_value = benchmark['close'].max()
min_value = benchmark['close'].min()
mean_value = benchmark['close'].mean()

# 将行和列进行交换
benchmark_trans = benchmark.T

#%% 数据选择与过滤
# 选择 benchmark 中的'close'数据。
benchmar_close = benchmark['close']
benchmar_close1 = pd.DataFrame(benchmark['close'])

# 选择 DataFrame 中的多列数据：'close','open','high','low'。
benchmar_cohl = benchmark[['close','open','high','low']]

# 从 benchmark 中选择行号为 3 到 7 的数据。
benchmark_value = benchmark.values
benchmark_value_3_7 =benchmark_value[4:8]
benchmark_value_3_7 = benchmark.iloc[4:8]

# 找出 benchmark 中开盘价比收盘价高的数据。
filter_df = benchmark[benchmark['close']<benchmark['open']]

# 使用 loc 和 iloc 获取特定行和列的数据。
data_iloc = benchmark.iloc[1,0]
data_loc = benchmark.loc['03/06/2013','close']

#%% 将某列的数据替换为指定的值（例如将所有 NaN 替换为 0）。

# 填充 closeprice 中的缺失值。
close_price_sh_filled = close_price_sh.fillna(method = 'ffill')
# 对 benchmark 中的数据进行去重操作。
benchmark = benchmark.loc[~benchmark.index.duplicated(), :]
a1 = ~benchmark.index.duplicated()

# 使用 apply() 方法对 benchmark 应用自定义函数。
ma5 = benchmark['close'].rolling(5).mean()

def mean1(x):
    return np.nanmean(x)/100

ma51 = benchmark['close'].rolling(5).apply(mean1)