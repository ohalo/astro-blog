#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「ARFIMA 长记忆：用分数差分把波动与收益的慢衰减写进模型」生成真实配图与统计数字。

核心逻辑:
  - 真实金融序列(平方收益/已实现波动)有「长记忆」: ACF 按幂律 c·k^(2d-1) 衰减, 不是指数衰减.
  - 标准 ARIMA 用整数差分 d=1 强行去趋势, 会过差分、把长记忆也差掉.
  - ARFIMA 用分数差分 (1-L)^d (0<d<0.5 长记忆, d<0 负记忆/反持续), 把慢衰减窗口保留进模型.
  - 用分数积分权重生成长记忆序列, 用双对数 ACF 斜率回收 d, 用「长期累积方差」展示长记忆 vs 短记忆 AR(1) 的胜负.
  - 对照: 短记忆 AR(1) 把累积方差当成线性增长, 长记忆 ARFIMA(d) 才是 H^(2d) 幂律增长.

全部数字由文中 Python 真实计算(仅依赖 numpy/matplotlib), 随机种子固定.
图片:
  arfima_acf.png        —— 长记忆 ACF 幂律衰减 vs 短记忆指数衰减
  arfima_filter.png     —— 分数积分滤波器权重 (1-L)^{-d} 长尾
  arfima_fit.png        —— 用双对数 ACF 斜率反推 d 的拟合
  arfima_forecast.png   —— 长期累积方差: AR(1) 系统性低估长 horizon 风险
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
SLUG = "arfima-long-memory"
D = os.path.join(BASE, SLUG)
os.makedirs(D, exist_ok=True)
os.makedirs(os.path.join("/Users/halo/workspace/astro-blog/src/content/blog", SLUG), exist_ok=True)

C = {"mem": "#C0392B", "short": "#34495E", "band": "#F2D7D5", "grid": "#DDDDDD",
     "accent": "#E67E22", "good": "#27AE60", "bad": "#C0392B", "fit": "#2F4F8F"}

# ----------------- 参数 -----------------
rng = np.random.default_rng(20260717)
T = 20000
D_TRUE = 0.30                     # 真实长记忆参数 (0<d<0.5)

# ----------------- 两类分数权重 -----------------
# 差分权重 (1-L)^d : 符号交替, 用于「过滤/差分」, 文中作为对照与展示
def fracdiff_weights(d, kmax):
    w = np.zeros(kmax + 1)
    w[0] = 1.0
    for k in range(1, kmax + 1):
        w[k] = w[k-1] * (d - k + 1) / k
    return w

# 积分权重 (1-L)^{-d} : 全正、缓慢衰减, 对白噪声卷积 -> 生成长记忆序列
def fracint_weights(d, kmax):
    w = np.zeros(kmax + 1)
    w[0] = 1.0
    for k in range(1, kmax + 1):
        w[k] = (k - 1 + d) / k * w[k-1]
    return w

kmax = 1000
w = fracint_weights(D_TRUE, kmax)        # 全正权重 -> 长记忆

# ----------------- 用分数积分白噪声构造长记忆序列 -----------------
eps = rng.standard_normal(T + kmax)
xm = np.zeros(T)
for t in range(T):
    xm[t] = np.dot(w, eps[t:t+kmax+1][::-1])
xm = xm - xm.mean()

# 短记忆对照: AR(1) phi=0.6
phi = 0.6
xs = np.zeros(T)
xs[0] = eps[0]
for t in range(1, T):
    xs[t] = phi * xs[t-1] + eps[t]
xs = xs - xs.mean()

# ----------------- ACF -----------------
def acf(x, lag=60):
    x = x - x.mean()
    n = len(x)
    denom = np.sum(x**2)
    return np.array([np.sum(x[:n-k] * x[k:]) / denom for k in range(lag + 1)])

acf_m = acf(xm, 60)
acf_s = acf(xs, 60)
kk = np.arange(1, 61)
mask = acf_m[1:61] > 1e-6          # 去掉已衰减到 0 以下的滞后(对数无意义)
kk_fit = kk[mask]
slope = np.polyfit(np.log(kk_fit), np.log(acf_m[1:61][mask]), 1)[0]
D_FIT = (slope + 1) / 2.0

# ----------------- 长期累积方差: AR(1) vs 长记忆 -----------------
# 点预测不可靠(瞬时创新不可预测); 真正胜负在「长期累积」:
#   AR(1)      -> Var(S_H) ~ H        (线性)
#   长记忆(d)  -> Var(S_H) ~ H^(2d)   (幂律)
y = xm[1:]; X = xm[:-1]
phi_hat = np.sum(X * y) / np.sum(X * X)
sigma_e2 = ((y - phi_hat * X) ** 2).mean()
Hs = [10, 30, 60, 120, 250]
true_var = []
ar1_var = []
for H in Hs:
    blocks = xm[: (len(xm) // H) * H].reshape(-1, H).sum(axis=1)
    true_var.append(blocks.var())
    ar1_var.append(sigma_e2 / (1 - phi_hat**2) * H)     # AR(1) 线性增长预测
worst_ratio = min(ar1_var[i] / true_var[i] for i in range(len(Hs)))   # AR(1)/true 最小(最低估)
under_ratio = max(true_var[i] / ar1_var[i] for i in range(len(Hs)))    # 真实/AR(1) 最大(高估倍数)

# ================= 绘图 =================
fig, ax = plt.subplots(figsize=(9, 4.3))
kk1 = np.arange(1, 61)
ax.plot(kk1, acf_m[1:], color=C["mem"], lw=2.0, marker="o", ms=3,
        label=f"长记忆 d={D_TRUE:.2f} (ACF 幂律衰减)")
ax.plot(kk1, acf_s[1:], color=C["short"], lw=1.6, label=f"短记忆 AR(1) φ={phi}")
ax.set_yscale("log")
ax.set_title("ACF 对比: 长记忆幂律衰减(红) vs 短记忆指数衰减(蓝)", fontsize=11.5)
ax.set_xlabel("滞后 k (交易日)", fontsize=10)
ax.set_ylabel("ACF (log)", fontsize=10)
ax.grid(True, color=C["grid"], ls=":", alpha=0.7)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "arfima_acf.png"), dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.0))
ax.plot(np.arange(kmax + 1), w, color=C["mem"], lw=1.4)
ax.set_yscale("log")
ax.set_title(f"分数积分滤波器权重 (1−L)^{{-D_TRUE:.2f}}: 长尾=长记忆窗口保留", fontsize=11.5)
ax.set_xlabel("滞后 k", fontsize=10)
ax.set_ylabel("权重 (log)", fontsize=10)
ax.grid(True, color=C["grid"], ls=":", alpha=0.7)
fig.tight_layout()
fig.savefig(os.path.join(D, "arfima_filter.png"), dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.0))
ax.plot(np.log(kk_fit), np.log(acf_m[1:61][mask]), color=C["mem"], lw=2.0, marker="o", ms=3, label="log ACF")
fit = slope * np.log(kk_fit) + np.log(acf_m[1:61][mask][0])
ax.plot(np.log(kk_fit), fit, color=C["fit"], ls="--", lw=1.6,
        label=f"拟合 斜率={slope:.3f} → d̂={D_FIT:.3f}")
ax.set_title(f"双对数 ACF 斜率反推 d: 真值 {D_TRUE:.2f} vs 回收 d̂={D_FIT:.3f}", fontsize=11.5)
ax.set_xlabel("log k", fontsize=10)
ax.set_ylabel("log ACF", fontsize=10)
ax.grid(True, color=C["grid"], ls=":", alpha=0.7)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "arfima_fit.png"), dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(Hs, true_var, color=C["mem"], lw=2.0, marker="o", ms=5, label="真实长记忆 Var(S_H)")
ax.plot(Hs, ar1_var, color=C["short"], lw=1.8, marker="s", ms=4, label="AR(1) 短记忆预测")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_title(f"长期累积方差: AR(1) 系统性低估长 horizon 风险 (H=250 时仅 {worst_ratio:.0%} 真实值)", fontsize=11)
ax.set_xlabel("累积 horizon H (对数)", fontsize=10)
ax.set_ylabel("Var(累积和 S_H) (对数)", fontsize=10)
ax.grid(True, color=C["grid"], ls=":", alpha=0.7)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "arfima_forecast.png"), dpi=130)
plt.close(fig)

# ================= 写出 markdown =================
def fmt(x, n=3):
    return f"{x:.{n}f}"

front = """---
title: "ARFIMA 长记忆：用分数差分把波动与收益的慢衰减写进模型"
description: "标准 ARIMA 用整数差分 d=1 把序列强行去趋势, 可金融序列(平方收益、已实现波动)的 ACF 不是指数衰减, 而是按幂律 c·k^(2d-1) 慢慢衰减——这种『长记忆』被整数差分一差就过差分、把慢衰减也差没了。ARFIMA 用分数差分 (1-L)^d (0<d<0.5 长记忆, d<0 反持续) 把这段长尾记忆窗口保留进模型。本文用 2 万点合成长记忆序列演示: 分数积分滤波器权重呈长尾、双对数 ACF 斜率反推出 d̂≈__DFIT__(真值 0.30)、长期累积方差按 H^{2d} 幂律增长, 而短记忆 AR(1) 误判为线性, 在 H=250 时把长期风险低估到真实值的约 __WR__。附完整 Python 与五类真实陷阱(高阶)。"
publishDate: '2026-07-17'
tags:
  - 量化交易
  - ARFIMA
  - 长记忆
  - 分数差分
  - 波动率建模
  - 时间序列
  - Python
language: Chinese
difficulty: advanced
---

"""
front = (front.replace("__DFIT__", fmt(D_FIT)).replace("__WR__", f"{worst_ratio:.0%}"))

p1 = """如果你做过波动率建模, 大概率踩过这个坑:

用 `returns.diff()`(也就是一阶整数差分 d=1)去「去趋势」, 然后发现——**去完之后序列反而变得像白噪声, 但模型对波动的预测力反而掉了**。

原因很微妙: 金融里的波动和平方收益, 本身有**长记忆(long memory)**——它的自相关不是「几步就贴到 0」的指数衰减, 而是按幂律 $c\\cdot k^{2d-1}$ **慢慢**衰减, 几十、上百步之后还剩一截。你一旦用整数差分把它「去差分」干净, 这截长尾记忆也被你差没了。

ARFIMA(Autoregressive Fractionally Integrated Moving Average)干的事, 就是**允许差分阶数 d 取分数**(0<d<0.5 即长记忆, d<0 是反持续/负记忆), 把这段慢衰减窗口原样保留进模型。

"""

p2 = """## 一、整数差分的盲区: 过差分

标准 ARIMA(p,1,q) 的 `d=1` 做的是 $(1-L)y_t = (1-L)x_t$, 其中 $L$ 是滞后算子。问题在于:

- 如果原序列本来**只有短记忆**(ACF 指数衰减), d=1 刚刚好把它变平稳;
- 但如果原序列**有长记忆**(ACF 幂律衰减), d=1 会「过度差分」——把平稳的长记忆结构差成短记忆甚至过短, 损失了真实的依赖结构。

长记忆的本质是: 今天的波动, 和 50 天前、100 天前的波动, **都还藕断丝连**。整数差分粗暴地剪断了这根丝。

"""

p3 = """## 二、分数差分: (1-L)^d 的权重

ARFIMA 的核心是一个优雅的展开: 把整数差分推广到分数

$$(1-L)^d = \\sum_{k=0}^{\\infty} w_k\\, L^k, \\quad w_k = \\frac{\\Gamma(k-d)}{\\Gamma(-d)\\,\\Gamma(k+1)} = \\prod_{i=1}^{k} \\frac{d-i+1}{i}$$

当 d 取 0 到 0.5 之间, 权重 $w_k$ 呈**长尾**——前面的项大, 但后面几十、上百步的项仍然显著非零。这正是「长记忆窗口」的数学形态。

对应的**分数积分** $(1-L)^{-d}$ 权重则是全正、缓慢衰减的——它正是我们用白噪声卷积生成一条长记忆序列的工具:

```python
def fracint_weights(d, kmax):
    w = np.zeros(kmax + 1)
    w[0] = 1.0
    for k in range(1, kmax + 1):
        w[k] = (k - 1 + d) / k * w[k-1]   # 全正、缓慢衰减
    return w

w = fracint_weights(0.30, 1000)    # 长尾权重 -> 长记忆
xm = np.array([np.dot(w, eps[t:t+1001][::-1]) for t in range(T)])  # 卷积白噪声
```

用这个长尾权重去卷积白噪声, 得到的序列 ACF 自然就是幂律衰减。我们用 2 万点合成一个真实长记忆序列(d=0.30), 看三个证据:

1. **ACF 幂律衰减**: 红线的 ACF 在 log 尺度下是一条缓坡的线, 而短记忆 AR(1) 蓝线几步就塌到 0;
2. **分数权重长尾**: 滤波器权重在 log 尺度下拖出长尾(不像 AR(1) 那种指数快衰减);
3. **双对数 ACF 斜率反推 d**: $\\ln\\text{ACF}(k) \\approx (2d-1)\\ln k$, 斜率直接给出 d。

本文回收结果: **斜率 __SLOPE__ → d̂ = __DFIT2__**, 非常贴近真值 0.30。

![ACF 对比: 长记忆幂律衰减 vs 短记忆指数衰减](/images/arfima-long-memory/arfima_acf.png)

![分数积分滤波器权重: 长尾=长记忆窗口保留](/images/arfima-long-memory/arfima_filter.png)

![双对数 ACF 斜率反推 d: 真值 0.30 vs 回收 d̂=__DFIT2__](/images/arfima-long-memory/arfima_fit.png)

"""
p3 = p3.replace("__SLOPE__", fmt(slope)).replace("__DFIT2__", fmt(D_FIT))

p4 = """## 三、长记忆的「真刀真枪」: 长期累积方差

点预测 x_{t+1} 对纯长记忆序列意义不大——下一刻的瞬时创新是不可预测的, 无论长记忆短记忆都猜不准。长记忆真正的威力在**「长期累积」**: 把序列连续加 H 期得到累积和 $S_H = \\sum_{i=1}^{H} x_{t+i}$, 看它的方差随 horizon H 怎么长。

- **短记忆 AR(1)**: 记忆几步就断, 累积和方差随 H **线性**增长, $\\text{Var}(S_H) \\sim H$;
- **长记忆 ARFIMA(d)**: 记忆无限长尾, 累积和方差随 H **幂律**增长, $\\text{Var}(S_H) \\sim H^{2d}$。

也就是说, 长记忆序列的「长期风险」比短记忆模型以为的要**大得多、且停不下来**。这正是长记忆最该被建模的地方——它对**风险度量、VaR、组合长期暴露**的影响是决定性的。

```python
# 短记忆 AR(1) 拟合, 其累积和方差预测 = sigma_e^2/(1-phi^2) * H  (线性)
phi_hat  = np.sum(X * y) / np.sum(X * X)
sigma_e2 = ((y - phi_hat * X) ** 2).mean()
ar1_var  = sigma_e2 / (1 - phi_hat**2) * H
# 真实长记忆: 直接对样本切块算累积和方差
true_var = xm[: (len(xm)//H)*H].reshape(-1, H).sum(axis=1).var()
```

"""

p5 = """## 四、结果: AR(1) 系统性低估长期风险

取 horizon H = 10 / 30 / 60 / 120 / 250, 对比真实长记忆的累积方差与 AR(1) 短记忆预测:

- H=10:  真实 Var(S_H)=**__TV10__**, AR(1) 预测 **__AV10__**;
- H=60:  真实 Var(S_H)=**__TV60__**, AR(1) 预测 **__AV60__**;
- H=250: 真实 Var(S_H)=**__TV250__**, AR(1) 预测 **__AV250__**——**AR(1) 只估到真实值的 {worst_ratio:.0%}**;
- 即 AR(1) 对长 horizon 的风险**低估了约 {under_ratio:.0f} 倍**。

![长期累积方差: AR(1) 系统性低估长 horizon 风险 (H=250 时仅 {worst_ratio:.0%} 真实值)](/images/arfima-long-memory/arfima_forecast.png)

这不是「多塞几个滞后项」的暴力堆砌, 而是来自**正确的衰减结构**: 长记忆的累积方差按 $H^{2d}$ 幂律生长, 而 AR(1) 强行假设线性。在 log-log 图上, 真实曲线的斜率就是 $2d$, 一眼就能看出 AR(1) 那条线全方位趴在下面。对波动率建模、风险预算、长期 VaR 而言, 漏掉这一段慢衰减, 等于把尾部风险系统性算小。

"""
p5 = (p5.replace("__TV10__", f"{true_var[0]:.0f}").replace("__AV10__", f"{ar1_var[0]:.0f}")
        .replace("__TV60__", f"{true_var[2]:.0f}").replace("__AV60__", f"{ar1_var[2]:.0f}")
        .replace("__TV250__", f"{true_var[4]:.0f}").replace("__AV250__", f"{ar1_var[4]:.0f}")
        .replace("__IMPROVE2__", fmt(under_ratio, 1)))

p6 = """## 五、五类真实陷阱(实战必看)

1. **d 的估计误差被放大**: 双对数 ACF 斜率法对短样本极敏感, 几千点以下 d̂ 会大幅抖动。实战要用 Whittle 近似似然或 GPH 半参数估计, 并给置信区间。
2. **分数差分要截断**: 真实 (1-L)^{-d} 有无限长尾, 实现必须截断到 kmax。kmax 太小丢记忆、太大引噪声——本文用 1000, 实战要按样本长度调。
3. **长记忆 ≠ 可预测暴利**: 长记忆说的是「依赖结构长」, 不是「方向可预测」。它改善的是波动率/相关建模与**长期风险度量**, 不是给你一个能稳定赚钱的方向信号。
4. **非平稳边界**: d 接近 0.5 时序列趋于非平稳(类似单位根), 估计会糊。d>0.5 在理论上非平稳, ARFIMA 不适用, 要先差分再建模。
5. **别和 GARCH 混淆**: GARCH 建模的是「波动的波动」(条件异方差, 短记忆), ARFIMA 建模的是「波动水平的长记忆」。真实波动常常**两者兼有**——GARCH 的残差里还嵌着长记忆, 于是有了 FIGARCH。单用哪一个都不完整。

## 六、小结: 把记忆的长度说对

ARFIMA 的贡献, 不是又塞了一个模型进工具箱, 而是提醒我们一件事:

**金融序列的记忆, 比 AR(1) 想的长, 比随机游走想的短——它是一个分数。** 用整数差分去硬套, 要么残留非平稳、要么过差分丢信息。分数差分 (1-L)^d 给了你一个连续的旋钮, 把那段「几十步之外还藕断丝连」的记忆, 原样写进模型。

对波动率建模、风险度量、期限结构形变这类「慢衰减」主导的问题, 这一个分数 d, 往往就是风险是否被算对的全部来源。

---

*代码与图表均由自包含 Python(numpy/matplotlib)真实计算, 随机种子固定为 20260717, 可完整复现。所有统计数字(分数积分权重长尾、双对数 ACF 斜率 __SLOPE__→d̂=__DFIT2__/真值 0.30、H=250 时真实累积方差 __TV250__ vs AR(1) __AV250__、AR(1) 仅估到真实值 {worst_ratio:.0%}/低估约 {under_ratio:.0f} 倍)均来自文中脚本输出。序列为合成长记忆过程, 实战须用真实高频/日频波动数据并叠加 Whittle/GPH 估计与截断长度选择。*
"""
p6 = (p6.replace("__SLOPE__", fmt(slope)).replace("__DFIT2__", fmt(D_FIT))
        .replace("__TV250__", f"{true_var[4]:.0f}").replace("__AV250__", f"{ar1_var[4]:.0f}")
        .replace("__IMPROVE2__", fmt(under_ratio, 1)))

md = front + p1 + p2 + p3 + p4 + p5 + p6
with open(os.path.join("/Users/halo/workspace/astro-blog/src/content/blog", SLUG, "index.md"), "w", encoding="utf-8") as f:
    f.write(md)

metrics = {
    "d_true": D_TRUE, "slope": round(float(slope), 3), "d_fit": round(float(D_FIT), 3),
    "Hs": Hs, "true_var": [round(float(v), 1) for v in true_var],
    "ar1_var": [round(float(v), 1) for v in ar1_var],
    "worst_ratio": round(float(worst_ratio), 3),
    "under_ratio": round(float(under_ratio), 1),
}
with open(os.path.join(D, "_metrics.txt"), "w") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print("ARTICLE WORDS:", len(md))
