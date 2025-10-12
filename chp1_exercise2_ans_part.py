# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 16:18:19 2025

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


        