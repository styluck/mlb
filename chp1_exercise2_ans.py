# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 09:36:46 2024

@author: lich5
"""
import pandas as pd
import numpy as np

#%% 数据清洗 
# 载入adj_factor、total_mv、close_price中两个市场的数据。

import os
dir_head = 'chp1_2_data\\'
list_name = {os.path.splitext(fileName)[0]:os.path.splitext(fileName)[1]
             for fileName in os.listdir(dir_head)}

dataset = {}
for fileName in list_name.keys():
    if list_name[fileName] == '.csv':
        dataset[fileName] = pd.read_csv(dir_head+fileName+'.csv')
        
    elif list_name[fileName] == '.xlsx':
        dataset[fileName] = pd.read_excel(dir_head+fileName+'.xlsx')
    
    dataset[fileName].set_index(dataset[fileName].columns[0], inplace = True)
    try:
        dataset[fileName].index = pd.to_datetime(dataset[fileName].index, format= '%Y-%m-%d')
    except:
        dataset[fileName].index = pd.to_datetime(dataset[fileName].index, format= '%d/%m/%Y')
        
    dataset[fileName].sort_index(inplace = True)
    print(fileName+' is loaded.')
        
#%% 使用 concat() 合并两个市场的数据。
mkt = ['_sh','_sz']
field = ['adj_factor', 'close','pb','total_mv']
combined_data = {}
for f in field:
        combined_data[f] = pd.concat([dataset[f+mkt[0]], dataset[f+mkt[1]]], axis = 1)
        
        # 删除重复的行、列。
        combined_data[f] = combined_data[f].loc[:,~combined_data[f].columns.duplicated()]
        combined_data[f] = combined_data[f].loc[~combined_data[f].index.duplicated(),:]
        
        # 删除数据中，全为缺失值的行，
        combined_data[f].dropna(how='all', inplace = True)
        
        # 将日期作为DataFrame的索引，并按照时间从前到后排序。
        # 注：日期需要先转化为datetime格式才可以用于正确排序
        
# 计算adjusted_close数据。
# 注：由于adj_factor中存在多余的column数据，这些数据在close_price没有。可以直接取close_price中已有的column
combined_data['adj_factor'] = combined_data['adj_factor'][combined_data['close'].columns]
combined_data['adj_close'] = combined_data['adj_factor'] * combined_data['close']


# 删除有效长度不超过400行的列数据。
# 注：只需要对adj_close做一次索引，其他所有数据集引用该索引即可
valid_lengths = combined_data['adj_close'].notna().sum()
columns_to_keep = valid_lengths[valid_lengths > 400].index
for f in combined_data.keys():
    combined_data[f] = combined_data[f][columns_to_keep]

# *计算Benchmark的VWAP数据。
# 注：VWAP = amount/vol
dataset['benchmark']['VWAP'] = dataset['benchmark']['amount']/dataset['benchmark']['vol']


#%% 时间序列数据
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
# 从 日期 列中提取年份、月份信息。
benchmark = dataset['benchmark'].copy()
benchmark['time'] = benchmark.index 
benchmark['time'] = pd.to_datetime(benchmark['time'], format= '%d/%m/%Y')
calendars = {}
calendars['year'] = benchmark['time'].dt.year
calendars['month'] = benchmark['time'].dt.month
# 将时间序列数据按日、月或季度重新采样。
monthly_data = {}
for f in combined_data.keys():
    monthly_data[f] = combined_data[f].resample('M').apply('last')
    
quarterly_data = {}
for f in combined_data.keys():
    quarterly_data[f] = combined_data[f].resample('Q').apply('last')

# *统计每个月股票的涨跌幅比例，绘制柱状图。
monthly_data['pct_chg'] = monthly_data['adj_close'].pct_change()

win_loss = pd.DataFrame()
win_loss['gain'] = np.sum(monthly_data['pct_chg'] > 0, axis = 1)
win_loss['loss'] = -np.sum(monthly_data['pct_chg'] < 0, axis = 1)
win_loss.plot(kind='bar', title='win loss')

#%% 数据可视化
# 使用 pandas 内置的 plot() 方法绘制benchmark close数据的折线图与volume数据的柱状图。
import matplotlib.pyplot as plt
benchmark = benchmark.set_index('time')
benchmark_monthly = benchmark.resample('M').apply('last')

plt.subplot(1, 3, 1)
(benchmark_monthly['vol']).plot(kind='bar', title='vol')

plt.subplot(1, 3, 2)
(benchmark_monthly['close']).plot(title='close price')

# 绘制benchmark收益率数据的直方图、散点图。
benchmark['pct_chg'] = benchmark['close'].pct_change()
benchmark['pct_chg_lag'] = benchmark['pct_chg'].shift(1)

plt.subplot(1, 3, 3)
benchmark['pct_chg'].plot.hist(bins=100,)

benchmark.plot(x='pct_chg_lag',y='pct_chg',style = '.')

# *使用 groupby() 对市值数据按（30亿以下，30-50亿，50-100亿，100亿-500亿，500亿以上）
# 进行分组后，绘制每个组的平均值图表。
# 提取最新的股票市值
current_mv = pd.DataFrame(combined_data['total_mv'].iloc[-1])
current_mv.columns = ['total_mv']
# 标注类别。注：原始数据以（万元）为单位。
def categorize_mv(total_mv):
    if total_mv < 3e5:
        return '<3bil'
    elif 3e5 <= total_mv < 5e5:
        return '3~5bil'
    elif 5e5 <= total_mv < 1e6:
        return '5~10bil'
    elif 1e6 <= total_mv < 5e6:
        return '10~50bil'
    else:
        return '>50bil'
    
current_mv['mv_category'] = current_mv['total_mv'].apply(categorize_mv)
grouped = current_mv.groupby('mv_category') # 使用 groupby() 进行分组
stats = grouped.describe()

stats.plot(y=[('total_mv',  'mean'),('total_mv',  'count')])

# *对最后一期的市值数据，使用 boxplot() 绘制分组后的市值数据的箱线图。
current_mv['log_mv'] = np.log(current_mv['total_mv'])
current_mv.boxplot(column='log_mv', by='mv_category')

#%% 高级操作
# 使用 shift() 函数对 total_mv数据进行移动。
total_mv_shift = combined_data['total_mv'].shift(1)

# *使用重采样的周度数据计算adjusted_close收益率数据与total_mv(-1)中两个数值列之间的相关系数。
# 这里的(-1)指的是上一期的数据。
adjusted_close_weekly = combined_data['adj_close'].resample('W').apply('last')
total_mv_weekly = combined_data['total_mv'].resample('W').apply('last').shift(1)
def row_corr(row1, row2):
    return np.corrcoef(row1, row2)[0, 1]

corr = pd.DataFrame(index=adjusted_close_weekly.index, columns=['Correlation'])
for index in adjusted_close_weekly.index:
    corr.loc[index] = row_corr(adjusted_close_weekly.loc[index], 
                               total_mv_weekly.loc[index])
    
# *使用 rolling() 计算adjusted_close数据的（5日，10日，20日）移动平均价格。
adj_close_ma5 = combined_data['adj_close'].rolling(5).mean()
adj_close_ma10 = combined_data['adj_close'].rolling(10).mean()
adj_close_ma20 = combined_data['adj_close'].rolling(20).mean()
