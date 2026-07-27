#!/usr/bin/env python3
"""巴塞尔交通灯检验配图生成"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "public/images/var-traffic-light-basel"
os.makedirs(OUT, exist_ok=True)

T = 250
p = 0.01

# ---------- 图1：二项分布 + 交通灯分区 ----------
k = np.arange(0, 16)
pmf = stats.binom.pmf(k, T, p)
cdf = stats.binom.cdf(k, T, p)

fig, ax = plt.subplots(figsize=(9, 5.2))
colors = ["#55A868" if x <= 4 else ("#E8B93C" if x <= 9 else "#C44E52") for x in k]
bars = ax.bar(k, pmf, color=colors, edgecolor="white")
for x, c, prob in zip(k, cdf, pmf):
    if prob > 0.005:
        ax.text(x, prob + 0.002, f"{c:.3f}", ha="center", fontsize=7.5, color="#555")
ax.axvline(4.5, color="#E8B93C", ls="--", lw=1.5)
ax.axvline(9.5, color="#C44E52", ls="--", lw=1.5)
ax.text(1.8, 0.20, "绿区\n0–4 次\n乘数 3.0", ha="center", fontsize=11, color="#2E6B3E", weight="bold")
ax.text(7, 0.20, "黄区\n5–9 次\n乘数 3.4–3.85", ha="center", fontsize=11, color="#8A6D1D", weight="bold")
ax.text(12.5, 0.20, "红区\n≥10 次\n乘数 4.0 + 整改", ha="center", fontsize=11, color="#8B2E38", weight="bold")
ax.set_xlabel("250 个交易日内 99% VaR 突破次数")
ax.set_ylabel("概率（模型正确时，Binomial(250, 1%)）")
ax.set_title("巴塞尔交通灯：突破次数的三色分区（柱上数字为累积概率）")
plt.tight_layout()
plt.savefig(f"{OUT}/traffic-light-zones.jpg", dpi=110)
plt.close()

# 分区累积概率
print("P(X<=4) =", stats.binom.cdf(4, T, p))
print("P(X<=9) =", stats.binom.cdf(9, T, p))
print("P(X>=10) =", 1 - stats.binom.cdf(9, T, p))
print("P(X>=5) =", 1 - stats.binom.cdf(4, T, p))

# ---------- 图2：两类错误的权衡 ----------
# 如果模型真实覆盖率是 q（低估风险 => q > 1%），落在各区的概率
q_list = np.linspace(0.005, 0.05, 100)
p_green = stats.binom.cdf(4, T, q_list)
p_yellow = stats.binom.cdf(9, T, q_list) - p_green
p_red = 1 - stats.binom.cdf(9, T, q_list)

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.stackplot(q_list * 100, p_green, p_yellow, p_red,
             colors=["#55A868", "#E8B93C", "#C44E52"], alpha=0.85,
             labels=["落入绿区", "落入黄区", "落入红区"])
ax.axvline(1.0, color="black", ls="--", lw=1.2)
ax.text(1.05, 0.5, "模型正确\n(真实覆盖率=1%)", fontsize=9)
ax.axvline(2.0, color="black", ls=":", lw=1)
ax.text(2.05, 0.75, "风险被低估一倍\n仍有约22%概率进绿区", fontsize=9)
ax.set_xlabel("模型的真实突破概率（%）")
ax.set_ylabel("概率")
ax.set_xlim(0.5, 5); ax.set_ylim(0, 1)
ax.set_title("交通灯的功效：真实覆盖率偏离 1% 时落入各区的概率")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/traffic-light-power.jpg", dpi=110)
plt.close()

print("q=2%: green prob =", stats.binom.cdf(4, T, 0.02),
      " red prob =", 1 - stats.binom.cdf(9, T, 0.02))
print("q=1%: yellow+red =", 1 - stats.binom.cdf(4, T, 0.01))
print("q=3%: red prob =", 1 - stats.binom.cdf(9, T, 0.03))

# ---------- 图3：资本乘数阶梯 ----------
breaches = np.arange(0, 13)
mult = np.where(breaches <= 4, 3.0,
        np.where(breaches == 5, 3.40,
        np.where(breaches == 6, 3.50,
        np.where(breaches == 7, 3.65,
        np.where(breaches == 8, 3.75,
        np.where(breaches == 9, 3.85, 4.0))))))

fig, ax = plt.subplots(figsize=(8.5, 5))
colors3 = ["#55A868"]*5 + ["#E8B93C"]*5 + ["#C44E52"]*3
ax.bar(breaches, mult, color=colors3, edgecolor="white")
for x, m in zip(breaches, mult):
    ax.text(x, m + 0.03, f"{m:.2f}", ha="center", fontsize=9)
ax.set_ylim(2.8, 4.3)
ax.set_xlabel("250 日内突破次数")
ax.set_ylabel("市场风险资本乘数 k")
ax.set_title("突破次数 → 资本乘数：模型差 = 资本贵（MRC = k × VaR 均值）")
plt.tight_layout()
plt.savefig(f"{OUT}/traffic-light-multiplier.jpg", dpi=110)
plt.close()

# ---------- 图4：模拟银行组合的年度回测 ----------
rng = np.random.default_rng(11)
nu = 5.0; scale = np.sqrt((nu - 2) / nu)
n_years = 10
results = {"good": [], "bad": []}
for y in range(n_years):
    # 好模型：t 分布 VaR；坏模型：正态但方差低估 25%
    r = rng.standard_t(nu, T) * scale
    var_good = stats.t.ppf(p, nu) * scale
    var_bad = stats.norm.ppf(p) * 0.75
    results["good"].append(int((r < var_good).sum()))
    results["bad"].append(int((r < var_bad).sum()))

fig, ax = plt.subplots(figsize=(9, 5))
xs = np.arange(n_years)
w = 0.36
def zone_color(x):
    return "#55A868" if x <= 4 else ("#E8B93C" if x <= 9 else "#C44E52")
b1 = ax.bar(xs - w/2, results["good"], w, color=[zone_color(v) for v in results["good"]],
            edgecolor="black", lw=0.6, label="合格模型（t 尾部）")
b2 = ax.bar(xs + w/2, results["bad"], w, color=[zone_color(v) for v in results["bad"]],
            edgecolor="black", lw=0.6, hatch="//", label="低估波动 25% 的模型")
ax.axhline(4.5, color="#E8B93C", ls="--", lw=1)
ax.axhline(9.5, color="#C44E52", ls="--", lw=1)
ax.set_xticks(xs); ax.set_xticklabels([f"第{y+1}年" for y in xs], fontsize=9)
ax.set_ylabel("年度突破次数")
ax.set_title("十年模拟：同一分区规则下，好模型偶尔黄，坏模型反复红")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/traffic-light-simulation.jpg", dpi=110)
plt.close()

print("good:", results["good"])
print("bad:", results["bad"])
print("done, images at", OUT)
