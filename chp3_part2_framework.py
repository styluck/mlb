# -*- coding: utf-8 -*-
"""第三章 Part 2 课堂练习。按 # %% 顺序补全 TODO；参考答案在课件中。

直接运行本文件，或在 IDE 中以本文件目录为工作目录逐单元运行。
依赖：numpy、pandas、matplotlib、scipy、statsmodels。
准备部分已写好，运行后得到 df、groups、bank、computer 和 wide。
"""

# %% 准备：导入工具（已完成）
from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from data_io.io_framework import load_data

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
alpha = 0.05

# %% 准备：同一交易日的 PB（市净率）与行业匹配（已完成）
data_dir = Path(__file__).resolve().parent / "dataset"
try:
    pb_panel = load_data("pb", "2024-01-01", "2024-03-31")
except OSError as exc:
    raise OSError(
        "无法读取 PB 数据。若 dataset 位于 OneDrive，请先将该目录设为“始终保留在此设备上”。"
    ) from exc
if pb_panel.empty:
    raise ValueError("指定区间没有 PB 数据，请检查 dataset 和日期。")
sample_date = pb_panel.index[-1]
pb = pb_panel.loc[sample_date].rename("pb")
pb.index.name = "ts_code"

info = pd.read_csv(data_dir / "stk_company_info.csv", dtype=str)
info = info.dropna(subset=["ts_code", "l1_name"]).copy()
for col in ["in_date", "out_date"]:
    info[col] = pd.to_datetime(info[col], format="%Y%m%d", errors="coerce")
# 按样本日匹配行业；退出日按不再属于该行业处理。
# 缺失进入日期的记录不用于历史匹配；不能直接用 is_new 筛历史行业。
active = info.loc[
    (info["in_date"] <= sample_date)
    & (info["out_date"].isna() | (sample_date < info["out_date"]))
].copy()
# 同一股票可能有多条行业记录，选当时已生效且进入日期最晚的一条。
active = active.sort_values("in_date").drop_duplicates("ts_code", keep="last")
matched = pb.to_frame().join(active.set_index("ts_code")[["l1_name"]],
                            validate="one_to_one")
print("样本日：", sample_date.date(), "；未匹配行业：", matched.l1_name.isna().sum())

industry_order = ["银行", "计算机", "房地产"]
selected = matched.loc[matched.l1_name.isin(industry_order)].copy()
selected["pb"] = pd.to_numeric(selected["pb"], errors="coerce")
valid_pb = np.isfinite(selected["pb"])
print("选定行业中剔除的缺失/无穷 PB 数：", (~valid_pb).sum())
# PB 可能非正；保留有限的零值、负值和极端值，不预先取对数或缩尾。
df = selected.loc[valid_pb].rename(columns={"l1_name": "industry"}).reset_index()
df["market"] = df["ts_code"].str.rsplit(".", n=1).str[-1]
groups = [df.loc[df.industry == name, "pb"].to_numpy() for name in industry_order]
if any(len(g) < 3 or np.var(g) == 0 for g in groups):
    raise ValueError("每个行业至少需要 3 个有效观测，且组内方差不能为零。")
bank, computer, real_estate = groups
print("各组样本量：", dict(zip(industry_order, map(len, groups))))
print("有限且非正的 PB 数：", int((df.pb <= 0).sum()))
print("行业信息以所给文件为准，可能不是完整的历史分类档案。")

# %% 准备：Friedman 拓展练习的配对模拟数据（已完成）
# 24 个对象各接受 A/B/C 三种条件，行表示同一对象；不是股票 PB 实测数据。
rng = np.random.default_rng(20260930)
subject_effect = rng.normal(0, 1, size=(24, 1))
wide = pd.DataFrame(subject_effect + np.array([0., 0.3, 0.8])
                    + rng.normal(0, 0.5, size=(24, 3)),
                    columns=["condition_A", "condition_B", "condition_C"])
wide.index.name = "subject"

# %% TODO 1：描述统计与图形
# 对 df 按 industry 分组，计算 n、mean、median、std（ddof=1）和 IQR。
# 按 industry_order 画三组 PB 的箱线图与分组直方图，保留全部有效观测。
# 观察样本量、偏态、离群点；不要为了显著性而删除真实极端值。


# %% TODO 2：银行与计算机的两组均值比较
# 写出 H0 与双侧 H1；用 bank、computer 做 Welch t 检验。
# 输出均值差（银行减计算机）、t、p 和均值差的 95% 置信区间。
# 再用等方差 t 检验与两组经典 ANOVA 验证 F=t^2、p 相同。
# 等价验证只是数学关系，不代表这些数据满足等方差假设。


# %% TODO 3：三组经典单因素 ANOVA 与效应量
# 用 groups 手算 k、N、总均值、SSB、SSE、F 与右尾 p 值。
# 用 stats.f_oneway 验证；拟合 pb ~ C(industry)，保存为 model。
# 用 sm.stats.anova_lm 输出 ANOVA 表，并计算 eta2=SSB/(SSB+SSE)。
# 写出三组的 H0/H1；整体显著是否表示每两个行业都不同？


# %% TODO 4：检查前提假设
# 用 TODO 3 的 model 画残差-拟合值图和正态 Q-Q 图。
# 对残差做 Shapiro 检验；用 groups 做中心为中位数的 Levene 检验。
# 若残差超过 5000 个，固定随机种子抽取 5000 个用于 Shapiro。
# 独立性不能靠这两个检验证明；讨论同日股票的共同市场冲击。


# %% TODO 5：Welch ANOVA 与 Kruskal-Wallis
# 用 df.pb 和 df.industry 做 use_var="unequal" 的 anova_oneway。
# 用 groups 做 stats.kruskal，并与经典 ANOVA 比较。
# 区分均值差异与秩/分布差异，不能将 Kruskal 的拒绝解释为均值不同。


# %% TODO 6：三组事后比较
# 教学对照：运行 Tukey HSD，说明其经典等方差前提。
# 异方差方案：枚举三对行业做双侧 Welch t 检验，再对全部 3 个 p 值作 Holm 校正。
# 输出行业对、均值差、原始 p、校正 p 和是否拒绝 H0。
# 不要根据未校正 p 值挑选行业对；选择与整体分析假设相符的结果解释。


# %% TODO 7：双因素 ANOVA（行业 × 上市市场）
# 查看 industry × market 的样本量与均值表，确认没有空单元格。
# 用 pb ~ C(industry, Sum) * C(market, Sum) 拟合 model2。
# 不平衡设计下用 typ=3 输出含交互作用的 ANOVA 表，先看交互作用。
# 画三行业在 SH/SZ 两个市场的均值折线图；检查 model2 的残差。
# 这是观察性数据；主效应依赖所用对比与加权方式，不能解释为行业的因果作用。


# %% TODO 8（拓展）：Friedman 配对检验
# 对 wide 的三列做 Friedman 检验；每行必须对应同一对象。
# 为什么不能把长度不同的 bank/computer/real_estate 数组当成配对样本？


# %% 最后提交：用自己的运行结果写结论
# 报告日期、行业匹配口径、样本量、均值差/效应量、统计量与 p 值。
# 比较经典与稳健方法；解释实际意义、独立性限制及数据清洗的影响。
# 不要只写“显著”，也不要把未拒绝 H0 写成“证明相同”。
