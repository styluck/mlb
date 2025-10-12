# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 16:18:19 2025

@author: lich5
"""

import os
dir_head = 'chp1_2_data\\'
list_name = {os.path.splitext(fileName)[0]:os.path.splitext(fileName)[1]
             for fileName in os.listdir(dir_head)}

dataset = {}
for fileName in list_name.keys():
    if list_name[fileName] == '.csv':
        dataset[fileName] = pd.read_csv(dir_head+fileName+'.csv')
        
    elif list_name[fileName] == '.xlsx':
        excel_file = pd.ExcelFile(dir_head+fileName+'.xlsx')
        sheet_names = excel_file.sheet_names
        
        for sheets in sheet_names:
            dataset[sheets] = excel_file.parse(sheets)
            
        del excel_file #clear memory
    print(fileName+' is loaded.')
        
# 使用 concat() 合并两个市场的数据。
mkt = ['_sh','_sz']
field = ['adj_factor', 'close_price','pb','total_mv']
combined_data = {}
for f in field:
        combined_data[f] = pd.concat([dataset[f+mkt[0]], dataset[f+mkt[1]]], axis = 1)
        
        # 删除重复的行、列。
        combined_data[f] = combined_data[f].loc[:,~combined_data[f].columns.duplicated()]
        combined_data[f] = combined_data[f].loc[~combined_data[f].index.duplicated(),:]
        combined_data[f].set_index(combined_data[f].columns[0], inplace = True)
        
        # 删除数据中，全为缺失值的行，
        combined_data[f].dropna(how='all', inplace = True)
        
        # 将日期作为DataFrame的索引，并按照时间从前到后排序。
        # 注：日期需要先转化为datetime格式才可以用于正确排序
        try:
            combined_data[f].index = pd.to_datetime(combined_data[f].index, format= '%Y-%m-%d')
        except:
            combined_data[f].index = pd.to_datetime(combined_data[f].index, format= '%d/%m/%Y')
            
        combined_data[f].sort_index(inplace = True)
        