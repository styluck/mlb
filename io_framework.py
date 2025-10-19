# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 15:29:52 2025

@author: lich5
"""
# io.py
from __future__ import annotations

from pathlib import Path
from typing import Optional 
import pandas as pd



def load_data(field: str, startdate: str, enddate: str) -> pd.DataFrame:
    """
    载入指定 field 的上/深两个市场数据，并在列上合并。

    输入参数
    --------
    field : str
        字段名（例如：'open' 或 'close'），会尝试读取 `field_sh.csv` 与 `field_sz.csv`
    startdate : str
        开始日期（含），如 '2020-01-01'
    enddate : str
        结束日期（含），如 '2024-12-31'

    输出
    ----
    pandas.DataFrame
        index 为日期；列名是“资产名”（不含市场层级）。
        如同名资产同时存在于两市，按 `_MARKETS` 的优先顺序做空值合并（先 sh，后 sz）。
 
    """

 
    return df


def write_data(df: pd.DataFrame, field: str) -> None:
    """
    将最近一次通过 load_data/load_all 得到并缓存的指定 field 的 DataFrame 写回 CSV。

    CSV 格式
    --------
    - 第1行写列名（第1列为索引名 'date'）
    - 第1列为日期字符串（按 YYYY-MM-DD 导出）
    - 第2列起为资产数据（float）
    """ 

        
#%% ------------------------- main -------------------------      
if __name__=='__main__': 
    
    # 载入单个字段（如 'close'），并合并 sh/sz 两市
    df_close = load_data("close", "2020-01-01", "2024-12-31")
    df_open = load_data("open", "2020-01-01", "2024-12-31")
    print(df_close.columns)  # MultiIndex: level0 为 'sh'/'sz'，level1 为资产名
    
    # 载入所有字段（目录下凡是 *_sh.csv 或 *_sz.csv 的都会被识别为字段）
    # all_data = load_all("2020-01-01", "2024-12-31")
    # all_data 形如：{'open': df_open, 'close': df_close, ...}
    
    # 将最近一次缓存的 close 写回到 close_sh.csv 和 close_sz.csv
    write_field("close")

    
    
# [EOF]
