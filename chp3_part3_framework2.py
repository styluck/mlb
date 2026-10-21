"""Part 3 学生版：按课件顺序运行准备代码，补全相应 TODO。"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from data_io.io_framework import load_data

OUT = Path(__file__).resolve().parent / "outputs/chp3_part3_framework_current"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False})

# %% 单日数据
date, next_date = "2024-03-28", "2024-03-29"
pb = load_data("pb", date, date).loc[date]
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
# TODO：拟合 return_next ~ bp，保存为 simple；说明 BP 每增加 1 时，预测的次日收益变化多少。

simple = smf.ols("return_next ~ bp", data=simple_cross).fit()
print(simple.summary())
fig_simple, ax_simple = plt.subplots(figsize=(8, 3.6))
ax_simple.scatter(simple_cross.bp, simple_cross.return_next, s=14, alpha=.5)
grid = np.linspace(simple_cross.bp.min(), simple_cross.bp.max(), 150)
ax_simple.plot(grid, simple.params.iloc[0] + simple.params.bp * grid, color="#176B4D")
ax_simple.set(xlabel="BP", ylabel="Next-day return (%)")
fig_simple.tight_layout()
fig_simple.savefig(OUT / "simple_regression.png", dpi=180)

# %% 多元回归样本：加入市值
mv = load_data("total_mv", date, date).loc[date]
log_mv = np.log(mv.where(mv > 0)).rename("log_mv")
cross = simple_cross.join(log_mv).dropna().copy()

# %% TODO 2：多元回归
# TODO：拟合 return_next ~ bp + log_mv，保存为 model；再在同一 cross 上拟合 simple_same，比较 BP 系数。

model = smf.ols("return_next ~ bp + log_mv", data=cross).fit()
simple_same = smf.ols("return_next ~ bp", data=cross).fit()
print("一元、多元样本数：", len(simple_cross), len(cross))
print("同样本 BP 系数：", simple_same.params.bp, model.params.bp)

# %% TODO 3：统计推断与拟合优度
# TODO：输出 model 的系数、标准误、t、p、区间、R2、调整 R2 和整体 F。

report = model.summary2().tables[1]
print(report)
print("R2, adjusted R2, F, p:", model.rsquared, model.rsquared_adj,
      model.fvalue, model.f_pvalue)

# %% TODO 3a：变量变换——原始、对数和 Box-Cox 市值
# TODO：筛选 mv 中有限的正值，计算自然对数，并调用 stats.boxcox 估计幂变换参数。
# 用同一组股票绘制原始市值、对数市值和 Box-Cox 变换值三个直方图，比较三种尺度。
# 本单元只观察分布变化，回归模型仍使用 log_mv。

mv_valid = mv[np.isfinite(mv) & (mv > 0)]
mv_boxcox, lam = stats.boxcox(mv_valid.to_numpy())
fig_boxcox, axes_boxcox = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, values, title, label in zip(
        axes_boxcox, [mv_valid, np.log(mv_valid), mv_boxcox],
        ["Raw market cap", "Log market cap", f"Box-Cox, lambda={lam:.3f}"],
        ["Market cap (10,000 CNY)", "ln(market cap in 10,000 CNY)", "Box-Cox value"]):
    ax.hist(values, bins=50)
    ax.set(title=title, xlabel=label, ylabel="Count")
plt.tight_layout()
fig_boxcox.savefig(OUT / "market_cap_transformations.png", dpi=180)

# %% TODO 4：残差与影响点
# TODO：画残差图与 Q-Q 图；用 get_influence() 查看 Cook 距离等诊断量。

# %% TODO 5：异方差与 HC3
# TODO：用 Breusch-Pagan 检验 H0：同方差；输出检验统计量与 p 值。
# 用 HC3（use_t=True）调整标准误，比较 coef、OLS_SE、OLS_p、HC3_SE、HC3_p。
# 区分：BP 检验判断同方差，后两列 p 值检验对应回归系数为零；HC3 本身不是检验。

# %% TODO 6：共线性
# TODO：用 model 的设计矩阵计算 VIF，常数项的 VIF 不解释。

# %% 逐日横截面回归练习（参照课件的七个步骤独立完成）
start, end = "2024-01-01", "2024-03-31"

# TODO 7.1：读取 pb、total_mv、close、adj_factor；检查重复记录、日期范围及缺失比例。
# 按 close 的完整交易日序列和股票代码对齐，保留数据检查表。

fields = {f: load_data(f, start, end)
          for f in ["pb", "total_mv", "close", "adj_factor"]}
checks = []
for name, data in fields.items():
    if data.index.has_duplicates or data.columns.has_duplicates:
        raise ValueError(f"{name} 有重复日期或股票代码，请先核查")
    checks.append({"field": name, "start": data.index.min(), "end": data.index.max(),
                   "trading_days": len(data), "stocks": data.shape[1],
                   "missing_rate": data.isna().to_numpy().mean(),
                   "dates_not_in_close": len(data.index.difference(fields["close"].index)),
                   "stocks_not_in_close": len(data.columns.difference(fields["close"].columns))})
data_checks = pd.DataFrame(checks).set_index("field")
dates = fields["close"].index.sort_values()
codes = fields["close"].columns
fields = {name: data.reindex(index=dates, columns=codes) for name, data in fields.items()}

# TODO 7.2：构造 BP、对数市值和下一交易日复权收益率（百分比）。
# 先按完整交易日序列计算次日收益，不能先删除缺失日期再移位。
# 选取几条记录列出两日价格、变量值与收益率，并手工核对计算。

prices = fields["close"] * fields["adj_factor"]
valid_prices = prices.where(np.isfinite(prices) & (prices > 0))
forward = 100 * (valid_prices.shift(-1) / valid_prices - 1)

# TODO 7.3：构造 panel，每行是一只股票在一天的观测。
# 列：trade_date、target_date、ts_code、bp、log_mv、return_next。
# 要求 BP 为正、两日价格及市值为正、回归变量为有限值；统计各日有效回归样本量。

parts, audit_parts = [], []
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
    if j < 2:
        audit = g.dropna().head(3).copy()
        audit["price_t"] = valid_prices.loc[day].reindex(audit.index)
        audit["price_next"] = valid_prices.loc[dates[j + 1]].reindex(audit.index)
        audit_parts.append(audit.reset_index())
panel = pd.concat(parts, ignore_index=True)
variable_checks = pd.concat(audit_parts, ignore_index=True)
variable_checks["return_manual"] = 100 * (variable_checks.price_next / variable_checks.price_t - 1)
assert np.allclose(variable_checks.return_next, variable_checks.return_manual)
regression_panel = panel.dropna(subset=["bp", "log_mv", "return_next"]).copy()
sample_counts = regression_panel.groupby("trade_date").size().reindex(dates[:-1], fill_value=0)
sample_counts.name = "n"

# TODO 7.4：先复现 2024-03-28 的模型，与前面单日案例核对；再逐日估计 return_next ~ bp + log_mv。
# 使用 HC3、use_t=True。至少 30 只股票且设计矩阵满列秩才估计，跳过日期记录原因。

rows, skipped = {}, []
for day in dates[:-1]:
    g = panel.loc[panel.trade_date == day]
    g = g.dropna(subset=["return_next", "bp", "log_mv"])
    X_day = sm.add_constant(g[["bp", "log_mv"]], has_constant="add")
    if len(g) < 30 or np.linalg.matrix_rank(X_day.to_numpy()) < 3:
        skipped.append({"trade_date": day, "n": len(g),
                        "reason": "样本量不足30" if len(g) < 30 else "设计矩阵不满列秩"})
        continue
    fit = smf.ols("return_next ~ bp + log_mv", data=g).fit(cov_type="HC3", use_t=True)
    ci_day = fit.conf_int().loc["bp"]
    rows[day] = {"target_date": g.target_date.iloc[0], "n": int(fit.nobs),
                 "alpha": fit.params["Intercept"], "beta": fit.params["bp"],
                 "gamma": fit.params["log_mv"], "se_hc3": fit.bse["bp"],
                 "t_value": fit.tvalues["bp"], "p_value": fit.pvalues["bp"],
                 "ci_lower": ci_day.iloc[0], "ci_upper": ci_day.iloc[1], "r_squared": fit.rsquared}

# TODO 7.5：把逐日结果存为 daily，日期作索引，列名如下：
# target_date、n、alpha、beta、gamma、se_hc3、t_value、p_value、ci_lower、ci_upper、r_squared。
# 绘制每日有效样本量折线图，检查样本量骤降的日期。

daily = pd.DataFrame.from_dict(rows, orient="index").sort_index()
daily.index.name = "trade_date"
skipped_dates = pd.DataFrame(skipped, columns=["trade_date", "n", "reason"])
if daily.empty:
    raise ValueError("没有满足条件的回归日期")
if pd.Timestamp(date) in daily.index:
    single_hc3 = model.get_robustcov_results(cov_type="HC3", use_t=True)
    bp_index = model.model.exog_names.index("bp")
    assert np.allclose(daily.loc[date, ["beta", "se_hc3"]].astype(float),
                       [model.params["bp"], single_hc3.bse[bp_index]])

fig_daily_n, ax_n = plt.subplots(figsize=(10, 3.5))
sample_counts.plot(ax=ax_n, color="#2457A7")
ax_n.set(xlabel="因子日期", ylabel="有效股票数", title="每日回归样本量")

# TODO 7.6：分别绘制 BP 系数与 95% HC3 置信区间、HC3 t 值的时间序列。
# 系数图加零线，t 值图加 0 和正负 2 参考线；横轴用因子日期。

fig_daily_beta, ax_beta = plt.subplots(figsize=(10, 3.5))
beta_plot = daily.reindex(dates[:-1])
ax_beta.plot(beta_plot.index, beta_plot.beta, color="#176B4D", label="BP系数")
ax_beta.fill_between(beta_plot.index, beta_plot.ci_lower, beta_plot.ci_upper,
                     color="#176B4D", alpha=.18, label="95% HC3置信区间")
ax_beta.axhline(0, color="gray", lw=.8)
ax_beta.set(xlabel="因子日期", ylabel="BP系数（百分点）", title="逐日BP系数及置信区间")
ax_beta.legend()
fig_daily_t, ax_t = plt.subplots(figsize=(10, 3.5))
ax_t.plot(beta_plot.index, beta_plot.t_value, color="#2457A7")
for reference in [0, 2, -2]:
    ax_t.axhline(reference, color="gray", lw=.8, ls="--")
ax_t.set(xlabel="因子日期", ylabel="HC3 t值", title="逐日BP系数的t值")
for fig_daily, name in [(fig_daily_n, "daily_sample_size"),
                        (fig_daily_beta, "daily_beta_ci"), (fig_daily_t, "daily_t_values")]:
    fig_daily.tight_layout()
    fig_daily.savefig(OUT / f"{name}.png", dpi=180)

# TODO 8：汇总系数均值、标准差、正值比例、绝对 t 值均值和 p<0.05 的日期比例。
# 对按日期排序的 daily.beta 拟合只含截距的模型，用 HAC(maxlags=5)、use_t=True。
# 输出均值的 HAC 标准误、t、p 及 95% 置信区间；数据、逐日结果和汇总表存 CSV，图形存 PNG。

beta = daily["beta"].dropna()
mean_fit = sm.OLS(beta, np.ones((len(beta), 1))).fit(
    cov_type="HAC", cov_kwds={"maxlags": 5}, use_t=True)
mean_ci = mean_fit.conf_int().iloc[0]
daily_summary = pd.Series({"valid_dates": len(beta), "mean_beta": beta.mean(),
                          "std_beta": beta.std(ddof=1), "positive_rate": beta.gt(0).mean(),
                          "mean_abs_t": daily.t_value.abs().mean(),
                          "significant_rate_5pct": daily.p_value.lt(.05).mean(),
                          "hac_se": mean_fit.bse.iloc[0], "hac_t": mean_fit.tvalues.iloc[0],
                          "hac_p": mean_fit.pvalues.iloc[0],
                          "hac_ci_lower": mean_ci.iloc[0], "hac_ci_upper": mean_ci.iloc[1]})
print(daily_summary)
for table_name, table in [("data_checks", data_checks), ("variable_checks", variable_checks),
                           ("panel", panel), ("regression_panel", regression_panel),
                           ("daily_sample_size", sample_counts), ("daily_results", daily),
                           ("skipped_dates", skipped_dates), ("daily_summary", daily_summary)]:
    table.to_csv(OUT / f"{table_name}.csv", encoding="utf-8-sig")

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

# %% TODO 10：课后练习——行业中性化前后的逐日回归对比
# TODO 10.1：读取行业历史，按各因子日期匹配当天有效的申万一级行业。
# 每日使用 BP、log_mv、return_next、industry 均有效的共同股票样本；沿用 BP>0 等筛选。
# 两组模型至少有30只股票且各自设计矩阵满列秩才估计；保存跳过日期及原因。
# TODO 10.2：在当天共同样本上拟合 bp ~ C(industry)，取残差为 bp_industry_neutral。
# 检查各行业残差均值接近0；这是去均值，不除以行业标准差，也不控制规模。
# TODO 10.3：分别拟合 return_next ~ bp + log_mv 和
# return_next ~ bp_industry_neutral + log_mv，均使用 HC3、use_t=True。
# 保存 homework_daily：日期、model（raw/industry_neutral）、样本量、系数、HC3标准误、
# t、p、95%区间及R2；每个日期必须同时包含两组结果。
# TODO 10.4：比较两组系数均值、标准差、正值比例、|t|均值及p<0.05比例。
# 叠加绘制两组系数及95%置信区间；相对于完整交易日序列保留日期缺口。
# TODO 10.5：两组均使用共同有效日期，对系数做仅截距回归，HAC最大滞后阶数为5。
# 保存均值检验的标准误、t、p和95%区间，核查日期缺口并比较maxlags=1、5、10。
# 若有日期缺口，常规HAC滞后是有效观测间隔，不能直接称为交易日滞后。
# 说明：仅中性化BP一般不等于在收益回归中加入行业指标；这里比较的是两种因子定义。
# TODO 10.6：保存共同样本、逐日结果、跳过原因、汇总表、HAC敏感性表及叠加图。

# %% 最后提交：
# 报告样本期、有效交易日数、平均 BP 系数及其 HAC 检验结果。
# 比较行业中性化前后的系数与检验结果，说明平均关系及其时间稳定性。
# 不要只写"显著"；未拒绝 H0 不等于证明了系数为零。
