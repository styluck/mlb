# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 16:16:57 2025

@author: lich5
"""
import numpy as np
from scipy.stats import boxcox

def _my_boxcox(r_data, return_lambda=False):
    """对非缺失正值做 Box-Cox 变换；可选返回估计的 lambda。"""
    row_data = r_data.to_numpy(dtype=float, copy=True)
    nonempty = np.where(~np.isnan(row_data))[0]
    a, lam = boxcox(row_data[nonempty])
    row_data[nonempty] = a
    if return_lambda:
        return row_data, lam
    return row_data

def preprocess(data):
    data.fillna(method='ffill', inplace = True)
    
    # box-cox
    n, s = np.shape(data)
    for i in range(n):
        data.iloc[i] = _my_boxcox(data.iloc[i])
        
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
