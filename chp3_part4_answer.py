# -*- coding: utf-8 -*-
"""第三章 Part4：滚动 IC 合成选股与资金配置。"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 复用第三章的选股、权重矩阵生成函数。
sys.path.append(str(Path(__file__).resolve().parents[1]))
from chp3.stockselection import fac_selection, stock_selection
from data_io.io_framework import load_data, load_benchmark
from utils.FactorTest import normal_ic, rank_ic, factor_test
from utils.utils import calc_nav

#%% 参数
start = "2022-10-01"
end = "2025-03-31"
test_start = "2024-01-01"
window = 20
top_n = 30
cost_bps = 10
annual_days = 252
risk_free_rate = 0.0  # 年化无风险利率，夏普率使用日度超额收益计算
FEATURES = ["value", "size"]
DISPLAY = {
    "cfg_equal": "等权配置", "cfg_mv": "市值加权",
    "cfg_rp": "逆波动率配置", "cfg_mvo": "最小方差配置",
}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

#%% 数据载入
close = load_data("close", start, end).sort_index()
adj = load_data("adj_factor", start, end).reindex_like(close)
pb = load_data("pb", start, end).reindex_like(close)
mv = load_data("total_mv", start, end).reindex_like(close)
vol = load_data("vol", start, end).reindex_like(close)
price = close * adj
raw_return = price.ffill().pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

#%% 特征预处理
bp = (1 / pb.where(pb > 0)).replace([np.inf, -np.inf], np.nan)   # BP = 1 / PB
size = -np.log(mv.where(mv > 0))                                  # 规模 = -log(MV)
size = size.replace([np.inf, -np.inf], np.nan)

# 数据中的无成交股票可能沿用旧价格；仅用当日已知成交量筛选，不看次日收益。
tradable = (vol > 0) & price.notna() & (price > 0)
valid = bp.notna() & size.notna() & tradable
bp, size = bp.where(valid), size.where(valid)

# 简单收益：t 日收盘 -> t+1 日收盘
forward = (price.shift(-1) / price - 1).replace([np.inf, -np.inf], np.nan)

raw = {"value": bp, "size": size}
features = {}
for name, x in raw.items():
    lo = x.quantile(0.01, axis=1)
    hi = x.quantile(0.99, axis=1)
    clipped = x.clip(lo, hi, axis=0)
    z = clipped.sub(clipped.mean(axis=1), axis=0).div(
        clipped.std(axis=1, ddof=0).replace(0, np.nan), axis=0)
    features[name] = z.replace([np.inf, -np.inf], np.nan)

# 最后一期没有次日收益，仅从 IC 样本中去掉；保留当天特征用于调仓。
forward = forward.iloc[:-1]
return_dates = price.index[1:]

#%% 每日 IC 与 Rank IC
rows = []
for date, ret_date in zip(forward.index, return_dates):
    x = pd.DataFrame({name: features[name].loc[date] for name in FEATURES})
    x["r"] = forward.loc[date]
    x = x.dropna()  # 两个特征与下一期收益共同有效的股票
    row = {"factor_date": date, "return_date": ret_date, "n": len(x)}
    for name in FEATURES:
        good = len(x) >= 30 and x[name].nunique() > 1 and x.r.nunique() > 1
        row[f"{name}_ic"] = normal_ic(x[name].to_numpy(), x.r.to_numpy()) if good else np.nan
        row[f"{name}_rank_ic"] = rank_ic(x[name].to_numpy(), x.r.to_numpy()) if good else np.nan
    rows.append(row)
daily = pd.DataFrame(rows).set_index("factor_date")

#%% 滚动 IC 评分与持仓矩阵
# t 日收盘时，最近已实现的是 t-1 日特征对应的 Rank IC。
past = daily[[f"{name}_rank_ic" for name in FEATURES]].reindex(price.index).shift(1)
means = past.rolling(window, min_periods=window).mean()
means.columns = FEATURES
ready = means.notna().all(axis=1)
denom = means.abs().sum(axis=1)
factor_weights = means.div(denom.where(denom > 1e-12), axis=0)
factor_weights.loc[ready & (denom <= 1e-12), :] = 0.5
score = sum(features[name].mul(factor_weights[name], axis=0) for name in FEATURES)
score = score.where(ready, axis=0)

# 保留评价期前一个交易日的持仓，供首个评价日计算收益。
first_eval = price.index.searchsorted(pd.Timestamp(test_start))
if first_eval == 0 or first_eval >= len(price.index) or not ready.any():
    raise ValueError("需要评价期前的历史数据及有效滚动权重。")
signal_start = max(price.index[first_eval - 1], ready[ready].index[0])
times = price.index[price.index >= signal_start]
if len(times) < 2:
    raise ValueError("有效回测日期不足。")
col = price.columns
holdings = fac_selection(score.loc[times], top_n)
matrices = {
    "cfg_equal": stock_selection(col, holdings, times),
    "cfg_mv": stock_selection(col, holdings, times, cap=mv),
}

#%% 风险配置
# 以下两种配置使用相同的入选股票：逆波动率加权、全局最小方差。
for name in ["cfg_rp", "cfg_mvo"]:
    matrices[name] = pd.DataFrame(0.0, index=times, columns=col)

for date in times:
    members = col[matrices["cfg_equal"].loc[date] > 0]
    if len(members) == 0:
        continue
    # 当天收盘调仓，协方差只用截至当天已实现的最近 60 日收益。
    hist = raw_return.loc[:date, members].tail(60)
    cov = hist.cov().reindex(index=members, columns=members).fillna(0).to_numpy()
    inv_vol = 1 / np.sqrt(np.maximum(np.diag(cov), 1e-12))
    matrices["cfg_rp"].loc[date, members] = inv_vol / inv_vol.sum()

    # 沿用原配置公式：允许做空的全局最小方差权重。
    try:
        w = np.linalg.solve(cov + 1e-8 * np.eye(len(members)), np.ones(len(members)))
        if abs(w.sum()) < 1e-12:
            w = np.ones(len(members))
        w = w / w.sum()
    except np.linalg.LinAlgError:
        w = np.ones(len(members)) / len(members)
    matrices["cfg_mvo"].loc[date, members] = w

#%% 组合回测
# calc_nav 内部用上一交易日权重乘当日收益，这里不再传 forward。
daily_return = raw_return.reindex(index=times, columns=col).fillna(0)
ret_dates = times[(times >= pd.Timestamp(test_start)) & (times > signal_start)]
result = {}
backtest_outputs = {}
for name, mat in matrices.items():
    outputs = calc_nav(daily_return, mat, comsn=cost_bps / 10000)
    backtest_outputs[name] = outputs
    base_nav = outputs["nav"].shift(1).loc[ret_dates[0]]
    result[name] = {
        "nav": outputs["nav"].loc[ret_dates] / base_nav,
        "pnl": outputs["pnl"].loc[ret_dates],
        "turnover": outputs["turnover"].loc[ret_dates],
        "weights": mat.loc[ret_dates],
        "return": daily_return.loc[ret_dates],
    }

#%% IC 检验与组合表现
# IC 按收益实现日期筛选，与组合评价期保持一致。
eval_dates = daily.index[daily["return_date"].isin(ret_dates)]
rows = []
sub = daily.loc[daily.index.isin(eval_dates)]
for name in FEATURES:
    for kind in ["ic", "rank_ic"]:
        x = sub[f"{name}_{kind}"].dropna()
        if len(x) < 10:
            raise ValueError("IC 检验日期不足。")
        fit = sm.OLS(x.to_numpy(), np.ones((len(x), 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": 5}, use_t=False)
        rows.append({"feature": name, "metric": "Rank IC" if kind == "rank_ic" else "IC",
                     "n": len(x), "mean": x.mean(), "std": x.std(),
                     "positive_share": (x > 0).mean(),
                     "icir": x.mean() / x.std() if x.std() > 0 else np.nan,
                     "hac_p": fit.pvalues[0]})
ic_summary = pd.DataFrame(rows)

rows = []
for c, res in result.items():
    nav = res["nav"]
    r = res["pnl"]
    rows.append({
        "strategy": DISPLAY.get(c, c),
        "key": c,
        "days": len(nav),
        "total_return": nav.iloc[-1] - 1,
        "annual_vol": r.std() * np.sqrt(252),
        "max_drawdown": (nav / nav.cummax().clip(lower=1.0) - 1).min(),
        "mean_daily_traded": res["turnover"].mean(),
    })
perf = pd.DataFrame(rows)

#%% 因子检验
# factor_test 约定：Factors[i] 与 Return[i+1] 配对，故 Return 传原始日收益矩阵
ret = raw_return.reindex(index=features["value"].index).fillna(0)

factors = {"value": features["value"], "size": features["size"], "rolling": score}
reports = {}
for name, fac in factors.items():
    if name == "rolling":
        fac = fac.loc[ready]
    reports[name] = factor_test(
        fac, ret.loc[fac.index], name=name, times=fac.index, comsn=cost_bps / 10000,
        PlotFlag=True, DispFlag=False)

rows = []
for name, ft in reports.items():
    rows.append({
        "factor": name,
        "rank_ic_mean": ft["Rank_stats"][0],
        "rank_ic_std": ft["Rank_stats"][1],
        "rank_ic_ir": ft["Rank_stats"][2],
        "rank_ic_winrate": ft["Rank_stats"][3],
        "normal_ic_mean": ft["Normal_stats"][0],
        "normal_ic_std": ft["Normal_stats"][1],
        "normal_ic_ir": ft["Normal_stats"][2],
        "normal_ic_winrate": ft["Normal_stats"][3],
        "longshort_sharpe": ft["Sharpe"],
    })
factor_reports = reports
factor_stats = pd.DataFrame(rows)

#%% 资金配置与市场基准对比
benchmark = load_benchmark(start, end)["close"].reindex(times).ffill()
benchmark_nav = benchmark / benchmark.iloc[0]
benchmark_nav = benchmark_nav.loc[ret_dates]

# 最后一个子图中的策略与基准采用相同评价日期、相同初始净值。
benchmark_return = benchmark.pct_change(fill_method=None).loc[ret_dates]
comparison_returns = pd.DataFrame({
    "滚动 IC 策略（股票等权）": result["cfg_equal"]["pnl"],
    "市场基准": benchmark_return,
})
if comparison_returns.isna().any().any():
    raise ValueError("策略与市场基准需要共同有效的评价日期。")
comparison_rows = []
daily_rf = (1 + risk_free_rate) ** (1 / annual_days) - 1
for name, r in comparison_returns.items():
    nav = (1 + r).cumprod()
    annual_return = nav.iloc[-1] ** (annual_days / len(r)) - 1 if nav.iloc[-1] > 0 else np.nan
    annual_vol = r.std(ddof=1) * np.sqrt(annual_days)
    drawdown = nav / nav.cummax().clip(lower=1.0) - 1  # 包含期初净值 1
    max_drawdown = -drawdown.min()
    sharpe = (r.mean() - daily_rf) / r.std(ddof=1) * np.sqrt(annual_days) if annual_vol > 0 else np.nan
    comparison_rows.append({
        "组合": name,
        "累计收益率": nav.iloc[-1] - 1,
        "年化收益率": annual_return,
        "年化波动率": annual_vol,
        "最大回撤": max_drawdown,
        "夏普率": sharpe,
        "卡玛比率": annual_return / max_drawdown if max_drawdown > 0 else np.nan,
        "日胜率": (r > 0).mean(),
    })
benchmark_comparison = pd.DataFrame(comparison_rows).set_index("组合")
percent_columns = ["累计收益率", "年化收益率", "年化波动率", "最大回撤", "日胜率"]
formatters = {name: "{:.2%}".format for name in percent_columns}
formatters.update({"夏普率": "{:.3f}".format, "卡玛比率": "{:.3f}".format})
print(f"\n评价期：{ret_dates[0]:%Y-%m-%d} 至 {ret_dates[-1]:%Y-%m-%d}，共 {len(ret_dates)} 个交易日") 
print(benchmark_comparison.to_string(formatters=formatters))

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for name in ["cfg_equal", "cfg_mv", "cfg_rp", "cfg_mvo"]:
    axes[0].plot(ret_dates, result[name]["nav"].to_numpy(), label=DISPLAY[name])
axes[0].set_title("滚动 IC 合成选股：资金配置对比")

axes[1].plot(ret_dates, result["cfg_equal"]["nav"].to_numpy(),
             label="滚动 IC 评分 + 股票等权配置")
axes[1].plot(ret_dates, benchmark_nav.to_numpy(), label="市场基准", color="gray")
axes[1].set_title("滚动 IC 合成选股：等权配置与市场基准")
for ax in axes:
    ax.set_ylabel("净值")
    ax.grid(alpha=0.3)
    ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
plt.show()
