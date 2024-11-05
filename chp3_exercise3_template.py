# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:38:20 2024

@author: lich5
"""
import pandas as pd
import numpy as np
import os

#%% 数据载入
dir_head = '../chp3_data/'
list_name = {os.path.splitext(fileName)[0]:os.path.splitext(fileName)[1]
             for fileName in os.listdir(dir_head)}
dataset = {}
for fileName in list_name.keys():
    if list_name[fileName] == '.csv':
        dataset[fileName] = pd.read_csv(dir_head+fileName+'.csv')
        dataset[fileName].set_index(dataset[fileName].columns[0], inplace = True)
        
        print(fileName+' is loaded.')

# 索引调整
for fileName in dataset.keys():
    if fileName == 'stk_company_info':
        continue
    dataset[fileName].index = pd.to_datetime(dataset[fileName].index, format= '%Y-%m-%d')
    
#%% 计算复权价格及收益率

pricename = ['close','open','high','low']
for fileName in pricename:
    dataset['adj_'+fileName] = dataset[fileName]*dataset['adj_factor']
    
dataset['pct_chg'] = np.log(dataset['adj_close']).diff()

bp = (1./dataset['pb']).replace([-np.inf, np.inf], np.nan)
mv = dataset['total_mv'].replace([-np.inf, np.inf], np.nan)

#%% 预处理
from scipy.stats import boxcox
def my_boxcox(r_data):
    row_data = r_data.values
    nonempty = np.where(~np.isnan(row_data))[0]
    a, _ = boxcox(row_data[nonempty])
    row_data[nonempty] = a
    return row_data

def preprocess(data):
    data.fillna(method='ffill', inplace = True)
    
    # box-cox
    n, s = np.shape(data)
    for i in range(n):
        data.iloc[i] = my_boxcox(data.iloc[i])
        
    # 去极值
    mu = np.nanmean(data, axis = 1)
    std = np.nanstd(data, axis = 1)
    upper = mu + 3*std
    lower = mu - 3*std
    data.clip(lower = lower, 
              upper = upper,
              axis = 0, inplace = True)
    # 标准化
    mu = np.nanmean(data, axis = 1)
    std = np.nanstd(data, axis = 1)
    data_normalized = (data - mu[:, np.newaxis])/std[:, np.newaxis]
    return data_normalized

bp_normalized = preprocess(bp)
mv_normalized = preprocess(mv)

#%% 行业中性化
from chp3_data.utils import industry_neutral

neutralizer = industry_neutral('l1_code')

dataset['pb_neutralized'] = neutralizer(bp_normalized)
dataset['mv_neutralized'] = mv_normalized

#%% 新上市股票剔除
from chp3_data.utils import get_listdate, filter_matrix_spl
times = dataset['pb_neutralized'].index
col = dataset['pb_neutralized'].columns
listdate = get_listdate()
filt = filter_matrix_spl(times, col, listdate)

benchmark_return = dataset['benchmark']['close'].pct_change()
dataset['excess_return'] = dataset['pct_chg']*filt - (benchmark_return.values)[:,np.newaxis]
dataset['pb_neutralized'] = dataset['pb_neutralized']*filt
dataset['mv_neutralized'] = dataset['mv_neutralized']*filt

#%% 组合构建 这部分代码需要自己编写
from chp3.FactorTest import regression_analysis, factor_test
from stockselection import stock_selection


result = regression_analysis(dataset['pb_neutralized'], dataset['excess_return'])
result2 = factor_test(dataset['pb_neutralized'], dataset['excess_return'])
holdings = result2['Code_G10'][:,:30]

mat = stock_selection(col, holdings, times)

#%% 组合业绩表现回测
from chp3_data.utils import calc_nav, plot_equity

outputs = calc_nav(dataset['pct_chg'].fillna(0), mat)
plot_equity(outputs['nav'], dataset['benchmark'])
