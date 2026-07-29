# -*- coding: utf-8 -*-
"""日内动量尾盘效应配图：受控模拟复现 Gao-Han-Li-Zhou (2018) 风格结果"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/intraday-momentum-last30min"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

# ---------- 模拟设置 ----------
# 每个交易日 13 个半小时区间（美股 6.5h）。r1 = 第一个半小时（含隔夜），r13 = 最后半小时。
# 生成机制：r13 = gamma * r1 + 少量 r12 贡献 + 噪声；日间加入波动率聚集（GARCH 风格）
N_DAYS = 2500  # ~10 年
GAMMA1 = 0.08   # r1 -> r13 的传导系数
GAMMA12 = 0.14  # r12 -> r13
BASE_VOL = 0.006  # 半小时基础波动

def simulate(n_days, gamma1=GAMMA1, gamma12=GAMMA12, seed_rng=rng):
    # 日级波动状态（波动聚集）
    vol = np.zeros(n_days)
    vol[0] = BASE_VOL
    for t in range(1, n_days):
        vol[t] = 0.92 * vol[t - 1] + 0.08 * BASE_VOL * (1 + 2.5 * seed_rng.random())
    R = np.zeros((n_days, 13))
    for t in range(n_days):
        # 第一个半小时含隔夜信息，波动更大
        R[t, 0] = seed_rng.normal(0, vol[t] * 1.8)
        for j in range(1, 12):
            R[t, j] = 0.00012 + seed_rng.normal(0, vol[t] * 0.8)
        R[t, 12] = gamma1 * R[t, 0] + gamma12 * R[t, 11] + seed_rng.normal(0, vol[t] * 1.1)
    return R, vol

R, vol = simulate(N_DAYS)
r1, r12, r13 = R[:, 0], R[:, 11], R[:, 12]

# ---------- 图1：13 个半小时区间对 r13 的预测能力 ----------
print("图1：各区间预测 r13 的 t 统计量")
tstats, betas = [], []
for j in range(12):
    x = R[:, j]
    beta = np.cov(x, r13)[0, 1] / np.var(x)
    resid = r13 - beta * x
    se = np.sqrt(np.var(resid) / (N_DAYS * np.var(x)))
    tstats.append(beta / se)
    betas.append(beta)
    print(f"  r{j+1}: beta={beta:.4f}, t={beta/se:.2f}")

plt.figure(figsize=(9.5, 5))
colors = ["#d62728" if abs(t) > 2 else "#8ab4d8" for t in tstats]
plt.bar(range(1, 13), tstats, color=colors)
plt.axhline(2, color="gray", ls="--", lw=1, label="t = ±2")
plt.axhline(-2, color="gray", ls="--", lw=1)
plt.axhline(0, color="black", lw=0.8)
plt.xlabel("用第 j 个半小时收益预测最后半小时收益（j = 1…12）")
plt.ylabel("回归 t 统计量")
plt.title("只有第 1 个（含隔夜）和第 12 个半小时对尾盘收益有显著预测力")
plt.xticks(range(1, 13))
plt.legend()
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/intraday-tstats.png", dpi=130)
plt.close()

# ---------- 图2：策略净值 ----------
# 策略：sign(r1) 决定最后半小时方向；对照：始终做多最后半小时；buy-and-hold 全天
print("图2：策略净值")
sig = np.sign(r1)
strat = sig * r13
long_only = r13.copy()
allday = R.sum(axis=1)

def perf(r, label, periods=252):
    ann = r.mean() * periods
    sh = r.mean() / r.std() * np.sqrt(periods)
    eq = np.cumprod(1 + r)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    win = (r[r != 0] > 0).mean()
    print(f"  {label}: 年化 {ann*100:.1f}%, Sharpe {sh:.2f}, MaxDD {dd*100:.1f}%, 胜率 {win*100:.1f}%")
    return eq, ann, sh, dd, win

eq_s, ann_s, sh_s, dd_s, win_s = perf(strat, "尾盘动量")
eq_l, ann_l, sh_l, dd_l, _ = perf(long_only, "始终做多尾盘")
eq_a, ann_a, sh_a, dd_a, _ = perf(allday, "全天持有")

plt.figure(figsize=(9.5, 5.2))
plt.plot(eq_s, label=f"尾盘动量 sign(r1)×r13（Sharpe {sh_s:.2f}）", color="#d62728", lw=1.8)
plt.plot(eq_a, label=f"全天持有（Sharpe {sh_a:.2f}）", color="#1f77b4", lw=1.4)
plt.plot(eq_l, label=f"始终做多尾盘（Sharpe {sh_l:.2f}）", color="#999999", lw=1.2)
plt.yscale("log")
plt.xlabel("交易日")
plt.ylabel("净值（对数轴）")
plt.title(f"尾盘动量策略净值：{N_DAYS} 日模拟（每天只持仓最后 30 分钟）")
plt.legend()
plt.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(f"{OUT}/intraday-equity.png", dpi=130)
plt.close()

# ---------- 图3：条件收益 - 按 r1 分位数 ----------
print("图3：条件收益")
qs = np.quantile(r1, np.linspace(0, 1, 6))
means, stds, labels = [], [], []
for i in range(5):
    mask = (r1 >= qs[i]) & (r1 <= qs[i + 1])
    means.append(r13[mask].mean() * 1e4)
    stds.append(r13[mask].std() / np.sqrt(mask.sum()) * 1e4)
    labels.append(f"Q{i+1}")
    print(f"  Q{i+1}: E[r13]={means[-1]:.2f}bp")

plt.figure(figsize=(9, 5))
plt.bar(labels, means, yerr=np.array(stds) * 1.96, capsize=5,
        color=["#2166ac", "#67a9cf", "#cccccc", "#ef8a62", "#b2182b"])
plt.axhline(0, color="black", lw=0.8)
plt.xlabel("按第一个半小时收益 r1 分成五组（Q1 最跌 → Q5 最涨）")
plt.ylabel("最后半小时平均收益（基点，95% 置信区间）")
plt.title("条件期望单调递增：早盘方向在尾盘延续")
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/intraday-conditional.png", dpi=130)
plt.close()

# ---------- 图4：成本敏感性 + 信号消失对照 ----------
print("图4：成本敏感性")
costs_bp = np.array([0, 1, 2, 3, 5, 8, 10])
sharpes = []
for c in costs_bp:
    net = strat - 2 * c / 1e4  # 每天进出各一次
    sharpes.append(net.mean() / net.std() * np.sqrt(252))
    print(f"  单边成本 {c}bp: Sharpe {sharpes[-1]:.2f}")

# 对照：gamma=0 的世界里同样跑策略（假信号检验）
R0, _ = simulate(N_DAYS, gamma1=0.0, gamma12=0.0, seed_rng=np.random.default_rng(99))
strat0 = np.sign(R0[:, 0]) * R0[:, 12]
sh0 = strat0.mean() / strat0.std() * np.sqrt(252)
print(f"  无效应对照世界: Sharpe {sh0:.2f}")

plt.figure(figsize=(9, 5))
plt.plot(costs_bp, sharpes, "o-", color="#d62728", lw=2, ms=7, label="尾盘动量（真实效应世界）")
plt.axhline(sh0, color="gray", ls="--", lw=1.5, label=f"对照：无效应世界 Sharpe = {sh0:.2f}")
plt.axhline(0, color="black", lw=0.8)
plt.xlabel("单边交易成本（基点）")
plt.ylabel("年化 Sharpe")
plt.title("成本敏感性：日频翻仓策略的 Sharpe 随成本线性坍塌")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/intraday-costs.png", dpi=130)
plt.close()

print("完成，输出目录：", OUT)
