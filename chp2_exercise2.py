# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 21:46:53 2024

@author: 6
"""
'''
程序模块化练习
请按照下列步骤分析换手率因子。
换手率类因子主要反映的是过去一段时间内资产的流通性强弱，是一类非常重要的风格因
子。
'''

from data_io import data_input
from data_preprocess import winsorize, standardize, fillna
from factors_analysis import wls_analysis, ic_analysis


# [EOF]

