"""Part 3 学生版：按课件顺序运行准备代码，补全相应 TODO。"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from data_io.io_framework import load_data

# %% 单日数据
date, next_date = "2024-03-28", "2024-03-29"
pb = load_data("pb", date, date)
bp = (1 / pb).rename("bp")
close = load_data("close", date, next_date)
adj = load_data("adj_factor", date, next_date)
price = close * adj
return_next = (100 * (price.loc[next_date] / price.loc[date] - 1)).rename("return_next")

# %% 一元回归样本：一行是一只股票
single = pd.concat([bp, return_next], axis=1)
single = single.replace([np.inf, -np.inf], np.nan)
simple_cross = single.loc[single.bp > 0].dropna().copy()

# %% TODO 1：一元回归
# TODO：拟合 return_next ~ bp，保存为 simple；解释 BP 每增加 1 时的预测收益变化。

# %% 多元回归样本：加入市值
mv = load_data("total_mv", date, date).loc[date]
log_mv = np.log(mv.where(mv > 0)).rename("log_mv")
cross = simple_cross.join(log_mv).dropna().copy()

# %% TODO 2：多元回归
# TODO：拟合 return_next ~ bp + log_mv，保存为 model；另拟合 simple_same，使用同一个 cross 比较 BP 系数。

# %% TODO 3：统计推断与拟合优度
# TODO：输出 model 的系数、标准误、t、p、区间、R2、调整 R2 和整体 F。

# %% TODO 3a：市值的 Box-Cox 变换
# TODO：筛选 mv 中有限的正值，计算自然对数，并调用 stats.boxcox 估计幂变换参数。
# 用同一组股票绘制原始市值、对数市值和 Box-Cox 变换值的三个直方图，解释各自尺度。
# 这里只展示分布变化，不替换回归模型中的 log_mv。

# %% TODO 4：残差与影响点
# TODO：画残差图与 Q-Q 图；用 get_influence() 查看 Cook 距离等诊断量。

# %% TODO 5：异方差与 HC3
# TODO：进行 Breusch-Pagan 检验，计算 HC3 标准误，对比系数与推断结果。

# %% TODO 6：共线性
# TODO：用 model 的设计矩阵计算 VIF，不解释常数项的 VIF。

# %% 跨期数据：讲完单日分析后才读取整个季度
start, end = "2024-01-01", "2024-03-31"
fields = {f: load_data(f, start, end)
          for f in ["pb", "total_mv", "close", "adj_factor"]}
dates = fields["close"].index
prices = fields["close"] * fields["adj_factor"]
forward = 100 * (prices.shift(-1) / prices - 1)

# %% 构造 panel：逐日重复已经讲过的单日整理
parts = []
for j, day in enumerate(dates[:-1]):
    g = pd.DataFrame({
        "bp": 1 / fields["pb"].loc[day],
        "log_mv": np.log(fields["total_mv"].loc[day].where(
            fields["total_mv"].loc[day] > 0)),
        "return_next": forward.loc[day],
    }).replace([np.inf, -np.inf], np.nan)
    g = g.loc[g.bp > 0].copy()
    g["trade_date"], g["target_date"] = day, dates[j + 1]
    g.index.name = "ts_code"
    parts.append(g.reset_index())
panel = pd.concat(parts, ignore_index=True)

# %% TODO 7：逐期回归
# TODO：用 for 循环逐日拟合 return_next ~ bp + log_mv，保存 beta、t_value、n 到 daily。

# %% TODO 8：跨期平均系数的检验
# TODO：对 daily.beta 拟合只有截距的模型，使用 HAC(maxlags=5) 检验平均系数。

# %% 行业中性化的数据准备：直接复用 utils
from utils.utils import _get_industries, industry_neutral

industry = _get_industries("l1_name", date)["l1_name"].rename("industry")
factor_cross = bp.to_frame().join(industry)
factor_cross = factor_cross.replace([np.inf, -np.inf], np.nan)
factor_cross = factor_cross.loc[factor_cross.bp > 0]
print("当日未匹配行业的股票数：", factor_cross.industry.isna().sum())
industry_cross = factor_cross.dropna().copy()
bp_matrix = pd.DataFrame([industry_cross.bp], index=[pd.Timestamp(date)])
neutralizer = industry_neutral("l1_name")

# %% TODO 9a：行业内去均值
# TODO：调用 neutralizer(bp_matrix, standardize=False)，将首行存为 bp_demean。
# 检查 industry_cross 中各行业 bp_demean 的均值是否接近 0。

# %% TODO 9b：行业内标准化
# TODO：调用 neutralizer(bp_matrix)，将首行存为 bp_industry_z。
# 检查各行业均值与标准差（ddof=0）；有波动的行业应分别接近 0 和 1。
# 两次调用都使用原始 bp_matrix，不覆盖原始 BP；函数已在 utils 中提供。

# %% 加入市值：不按次日收益是否缺失筛选中性化样本
neutral_cross = industry_cross.join(log_mv).dropna(subset=["log_mv"]).copy()

# %% TODO 9c：单日行业与规模回归中性化
# TODO：拟合 bp ~ log_mv + C(industry)，把残差存为 bp_neutral。
# 检查残差与 log_mv 的相关系数及各行业的残差均值，不用标准化 BP 替换回归中的 BP。

# %% TODO 10：IC 与 Rank IC；不额外要求市值完整
# TODO：逐日计算 BP 与次日收益的 Pearson IC、Spearman Rank IC，保存为 ic；只按这两个变量筛选有效样本。

# %% TODO 11：汇总与绘图
# TODO：汇总 ic 的均值、标准差、正值比例和 ICIR，并画逐日 IC 和累计 IC。
