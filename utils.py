# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:40:30 2024

@author: lich5
"""
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates


def calc_nav(Pctchg, w, **kwargs):
    '''
        计算净值、pnl、换手率
    '''
    
    if 'comsn' in kwargs.keys():
        comsn = kwargs['comsn'] #加入交易费用
    else:
        comsn = 0
        
    Times = pd.to_datetime(Pctchg.index)
    # 时间期数
    T = len(Times)
    # 从价格序列得到的收益矩阵
    Pctchg = Pctchg.values
    w = w.fillna(0)
    w = w.values
    # 定义pnl
    pnl = np.zeros((T, ))
    # 定义净值
    nav = np.ones((T, ))
    # 定义换手率
    turnover = np.zeros((T, ))
    # 循环计算每期净值以及换手
    
    for i in range(1, T):
        turnover[i] = np.sum(np.abs(w[i] - ((1 + Pctchg[i]) * w[i-1]))) / 2
        pnl[i] = np.dot(w[i-1], Pctchg[i]) - (turnover[i] * comsn)
        nav[i] = (1 + pnl[i]) * nav[i-1]
        
    # 转换成pandas格式
    nav = pd.Series(nav, index = Times)
    pnl = pd.Series(pnl, index = Times)
    turnover = pd.Series(turnover, index = Times)
    
    return {'nav':nav, 'pnl':pnl, 'turnover':turnover}


def plot_equity(nav, bench_stats=None):

    def format_two_dec(x, pos):
        return '%.2f' % x
    
    equity = nav
    
    plt.figure()
    ax = plt.gca()

    y_axis_formatter = FuncFormatter(format_two_dec)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    ax.xaxis.set_tick_params(reset=True)
    ax.yaxis.grid(linestyle=':')
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.grid(linestyle=':')
    
    equity.plot(lw=2, color='blue', alpha=0.6, x_compat=False,
                label='Strategy', ax=ax)

    benchmark = bench_stats.pct_change()
    benchmark.iloc[0] = 0
    benchmark_nav = (1 + benchmark).cumprod()
    benchmark_nav = benchmark_nav.reindex(pd.date_range(benchmark_nav.index[0], benchmark_nav.index[-1], freq='D'))
    benchmark_nav = benchmark_nav.fillna(method = 'ffill')
    benchmark_nav = benchmark_nav.dropna()
    
    benchmark_nav.plot(lw=2, color='gray', alpha=0.6, x_compat=False,
                   label='Benchmark', ax=ax)
    ax.axhline(1.0, linestyle='--', color='black', lw=1)
    ax.set_ylabel('Cumulative returns')
    ax.legend(loc='best')
    ax.set_xlabel('')
    plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha='center')
    return ax
    



def get_industries(ind = 'l1_code'):
    '''
    载入行业数据
    ind = 'l1_code', 'l2_code', 'l3_code'
    ind = 'l1_name', 'l2_name', 'l3_name'
    '''
    outp = pd.read_csv('../chp3_data/stk_company_info.csv')
    outp.index = outp['ts_code']
    outp = outp[[ind,'in_date']]
    return outp


def filter_matrix_spl(times, codes, listdate, days = 52):

    T = len(times)
    S = len(codes)
#    idx = np.zeros(S)
    outs = np.ones((T, S))
    listdate = listdate[~listdate.duplicated(keep='first')]
    codeidx = pd.Series(np.nan, index=codes)
    for i in list(listdate.index):
        if (i in codes):
            codeidx[i] = listdate[i]
            
    # get the dates one year (or otherwise defined in 'days') after stocks' listing date
    valid_dates = pd.to_datetime([str(int(i)) if i == i else np.nan for i in codeidx]) + pd.Timedelta('%s days'%(days))
    
    for i in range(S):
        
        Match1 = (np.abs(times - valid_dates[i])).argmin()
        if not Match1: # Match1 == 0: valid date before the startdate
            continue
        else:
            outs[:Match1, i] = np.nan
            
    return outs


def get_listdate(ind = 'l1_code'):
    '''
    载入行业数据
    ind = 'l1_code', 'l2_code', 'l3_code'
    ind = 'l1_name', 'l2_name', 'l3_name'
    '''
    outp = pd.read_csv('../chp3_data/securities_info.csv')
    outp.index = outp['ts_code']
    outp = outp['list_date']
    return outp


class industry_neutral:
    def __init__(self, ind:str = 'l1_code'):
        self.industry = get_industries(ind)
        self.ind = ind
    def __call__(self, factor):
        '''
        input:
            factor: 因子值，可以为dict或者dataframe
            ind2： (default：False)， 是否采用二级行业进行中性化
        output:
            中性化后的因子值
        '''
        ind = self.ind
        industry = self.industry
        
        facs = factor.copy()
        codeid = industry.index.tolist()

        a = industry.values[:,0]
        
        a = np.where(a == None, '0', a)
        a = np.where(a != a, '0', a) # replace nan value        
        unique_industries = np.unique(a)
        unique_industries = unique_industries[unique_industries!='0']
        
        if isinstance(facs, dict):
            for i in facs.keys():
                code = facs[i].columns
                set_industries = pd.DataFrame(index = code, columns = [ind])
                
                for j in set_industries.index:
                    if j in codeid:
                        ind_loc = industry.loc[j]
                        if type(ind_loc) == pd.core.frame.DataFrame and (len(ind_loc) > 1):
                            ind_loc = ind_loc[ind_loc['in_date'] == np.max(ind_loc['in_date'])]
                        set_industries.loc[j] = ind_loc[ind]
                    
                set_industries = set_industries.fillna('0').values
                
                # calculate the mean and std of each industries
                fac_values = facs[i].values
                for k in unique_industries:
                    
                    match_i = np.where(set_industries == k)[0]
                    mean = np.nan_to_num(np.nanmean(fac_values[:,match_i],axis=1))
                    std = np.nan_to_num(np.nanstd(fac_values[:,match_i],axis=1),nan=1)
                    std = np.where(std == 0, 1, std)
                    
                    facs[i].iloc[:, match_i] = (facs[i].iloc[:, match_i] - mean[:,np.newaxis]
                                                )/std[:,np.newaxis]
                print('factor "'+i+'" neutralized.')
                
        elif isinstance(facs, pd.core.frame.DataFrame):
            code = facs.columns
            set_industries = pd.DataFrame(index = code, columns = [ind])
            
            for j in set_industries.index:
                if j in codeid:
                    ind_loc = industry.loc[j]
                    if type(ind_loc) == pd.core.frame.DataFrame and (len(ind_loc) > 1):
                        ind_loc = ind_loc[ind_loc['in_date'] == np.max(ind_loc['in_date'])]
                    set_industries.loc[j] = ind_loc[ind]
                
            set_industries = set_industries.fillna('0').values
            
            # calculate the mean and std of each industries
            fac_values = facs.values
            for k in unique_industries:
                
                match_i = np.where(set_industries == k)[0]
                mean = np.nan_to_num(np.nanmean(fac_values[:,match_i],axis=1))
                std = np.nan_to_num(np.nanstd(fac_values[:,match_i],axis=1),nan=1)
                std = np.where(std == 0, 1, std)
                
                facs.iloc[:, match_i] = (facs.iloc[:, match_i] - mean[:,np.newaxis]
                                         )/std[:,np.newaxis]
                
        return facs
                
                