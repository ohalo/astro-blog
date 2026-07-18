#!/usr/bin/env python3
"""
为文章「Bootstrap 回测显著性：用重抽样给你的夏普率算出 p 值」
(bootstrap-backtest-pvalue) 生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（自洽合成，仅用于演示方法；真实落地见文末路径）：
  * 策略收益 r_t ~ 月度,真实 Sharpe = mean/std*sqrt(12)
  * 对收益做 iid bootstrap 重抽样 B 次,重建「零假设(夏普=0)」下的分布:
        构造去均值的残差 r0_t = r_t - mean(r),再重抽样得 null 分布
  * p 值(单侧) = P(null 夏普 >= 观测夏普);越小越不可能「纯噪声」
  * 功效分析:在真夏普 = {0.0,0.3,0.6,0.9} 下统计 5% 拒绝率
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "bootstrap-backtest-pvalue")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "x": "#4C72B0", "y": "#DD8452", "base": "#55A868",
      "band": "#C44E52", "eq": "#2F4B7C", "bh": "#CCB974", "mk": "#8172B3",
      "null": "#999999"}


def sharpe(x, ann=12.0):
    x = np.asarray(x, float)
    if x.std(ddof=1) == 0:
        return 0.0
    return x.mean() / x.std(ddof=1) * np.sqrt(ann)


def gen_returns(true_sharpe, T=120, ann=12.0, seed=0):
    r = np.random.default_rng(seed)
    # 目标月度夏普 = true_sharpe/sqrt(ann);用正态收益反推均值/波动
    sd = 0.04
    mu = true_sharpe / np.sqrt(ann) * sd
    return r.normal(mu, sd, size=T)


def bootstrap_pvalue(ret, B=2000, seed=1):
    r = np.random.default_rng(seed)
    obs = sharpe(ret)
    resid = ret - ret.mean()           # 去均值 -> 零假设(夏普=0)下的残差
    nulls = np.empty(B)
    T = len(ret)
    idx = r.integers(0, T, size=(B, T))
    boots = resid[idx]                 # 重抽样残差
    # 向量化计算每组的夏普
    m = boots.mean(axis=1)
    s = boots.std(axis=1, ddof=1)
    nulls = np.where(s > 0, m / s * np.sqrt(12), 0.0)
    p = np.mean(nulls >= obs)
    return obs, nulls, p


# ---------- 图 1：策略净值 + 月度收益（真 alpha vs 纯噪声） ----------
ret_alpha = gen_returns(0.8, T=120, seed=1)      # 真夏普 0.8 的策略
ret_noise = gen_returns(0.0, T=120, seed=21)     # 纯噪声策略
eq_alpha = np.cumprod(1.0 + ret_alpha)
eq_noise = np.cumprod(1.0 + ret_noise)
obs_a, nulls_a, p_a = bootstrap_pvalue(ret_alpha, seed=3)
obs_n, nulls_n, p_n = bootstrap_pvalue(ret_noise, seed=9)
print("alpha 策略: 观测月度夏普=%.3f 年化=%.2f  p=%.4f" % (obs_a / np.sqrt(12), obs_a, p_a))
print("noise 策略: 观测月度夏普=%.3f 年化=%.2f  p=%.4f" % (obs_n / np.sqrt(12), obs_n, p_n))

fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax[0].plot(eq_alpha, color=C["eq"], lw=1.6, label="真 alpha 策略（年化 SR=%.2f）" % obs_a)
ax[0].plot(eq_noise, color=C["null"], lw=1.4, ls="--", label="纯噪声策略（年化 SR=%.2f）" % obs_n)
ax[0].set_ylabel("净值（起始=1）", fontsize=11)
ax[0].set_title("两个「看起来都不错」的净值曲线", fontsize=12)
ax[0].legend(loc="upper left", fontsize=9)
ax[0].grid(True, color=C["grid"])
ax[1].bar(np.arange(len(ret_alpha)), ret_alpha, color=C["x"], width=0.8, label="真 alpha 月度收益")
ax[1].bar(np.arange(len(ret_noise)), ret_noise, color=C["null"], width=0.8, alpha=0.6, label="纯噪声 月度收益")
ax[1].axhline(0, color="#333333", lw=0.8)
ax[1].set_ylabel("月度收益", fontsize=11)
ax[1].set_xlabel("月份", fontsize=11)
ax[1].legend(loc="upper right", fontsize=9)
ax[1].grid(True, color=C["grid"], axis="y")
plt.tight_layout()
fig.savefig(os.path.join(D, "strategy_equity_compare.png"), dpi=130)
plt.close(fig)

# ---------- 图 2：零假设下夏普的 bootstrap 分布 + p 值 ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for axx, nulls, obs, p, title, col in [
    (ax[0], nulls_a, obs_a, p_a, "真 alpha 策略", C["eq"]),
    (ax[1], nulls_n, obs_n, p_n, "纯噪声策略", C["null"]),
]:
    axx.hist(nulls, bins=60, color=col, alpha=0.75, density=True)
    axx.axvline(0.0, color="#333333", lw=1.0, ls=":", label="H0: 夏普=0")
    axx.axvline(obs, color=C["band"], lw=2.0, label="观测夏普=%.2f" % obs)
    axx.set_title("%s\np 值=%.3f" % (title, p), fontsize=11)
    axx.set_xlabel("bootstrap 夏普", fontsize=11)
    axx.set_ylabel("密度", fontsize=11)
    axx.legend(fontsize=8)
    axx.grid(True, color=C["grid"])
plt.tight_layout()
fig.savefig(os.path.join(D, "bootstrap_null_distribution.png"), dpi=130)
plt.close(fig)

# ---------- 图 3：功效分析（真夏普 -> 5% 拒绝率） ----------
true_srs = [0.0, 0.3, 0.6, 0.9, 1.2]
reject = []
for ts in true_srs:
    rr = 0
    for k in range(60):
        ret = gen_returns(ts, T=120, seed=1000 + k)
        _, _, p = bootstrap_pvalue(ret, seed=2000 + k)
        if p < 0.05:
            rr += 1
    reject.append(rr / 60.0)
print("功效(5%% 拒绝率) by 真夏普:", dict(zip(true_srs, np.round(reject, 3))))

fig, ax = plt.subplots(figsize=(9, 4.6))
bars = ax.bar([str(s) for s in true_srs], reject, color=C["y"], width=0.6)
for b, v in zip(bars, reject):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, "%.0f%%" % (v * 100),
            ha="center", fontsize=10)
ax.axhline(0.05, color=C["band"], ls="--", lw=1.2, label="纯噪声基准 5%")
ax.set_xlabel("策略真实年化夏普", fontsize=11)
ax.set_ylabel("5% 显著性下被拒绝的比例", fontsize=11)
ax.set_title("功效分析：真 alpha 越强，bootstrap 越容易抓到你", fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)
ax.grid(True, color=C["grid"], axis="y")
plt.tight_layout()
fig.savefig(os.path.join(D, "bootstrap_power_analysis.png"), dpi=130)
plt.close(fig)

print("saved 3 figures to", D)
