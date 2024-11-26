# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:33:02 2024

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

bp = (1./dataset['pb']).replace([np.inf, -np.inf], np.nan)
n, s = np.shape(bp)

# 去空值
bp.fillna(method='ffill',inplace = True)

# 去极值
mu = np.nanmean(bp, axis = 1)
std = np.nanstd(bp, axis = 1)
upper = mu + 3*std
lower = mu - 3*std

bp.clip(lower=lower, upper = upper, axis = 0, inplace = True)
c = bp.copy()
c = np.where(c>upper[:,np.newaxis], upper[:,np.newaxis], c)
c = np.where(c<lower[:,np.newaxis], lower[:,np.newaxis], c)
np.linalg.norm((bp).fillna(0))

# 标准化
mu = np.nanmean(bp, axis = 1)
std = np.nanstd(bp, axis = 1)
bp_normalized = (bp - mu[:,np.newaxis])/std[:,np.newaxis]
np.linalg.norm(bp_normalized.fillna(0))

#%% 行业中性化
from industry_neutral import industry_neutral

neutralizer = industry_neutral('l1_code')
dataset['pb_neutralized'] = neutralizer(bp_normalized)


from FactorTest import factor_test
result = factor_test(dataset['pb_neutralized'], dataset['pct_chg'])

# [EOF]
