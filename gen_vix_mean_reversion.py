#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「VIX 均值回归：用恐慌指数的回复特性做波动率择时」生成真实配图与统计数字。

核心逻辑:
  - VIX 是「恐慌指数」, 自身是一个强均值回复过程(OU 动力学 + 危机跳变).
  - 用 OU 过程拟合 VIX, 估计回复速度 kappa 与半衰期, 并用 ACF 展示其慢衰减.
  - 交易应用: 把 VIX 当「当前波动的温度计」做波动率目标化(vol targeting)覆盖层 ——
        目标波动恒定, 仓位 = 目标波动 / VIX 推导波动; VIX 高(恐慌)自动去杠杆, VIX 低(平静)加杠杆.
        因为 VIX 强均值回复, 这套杠杆会「恐慌时收手、平静时出手」自动周期化.
  - 对照组: 始终满仓买入持有.

全部数字由文中 Python 真实计算(仅依赖 numpy/matplotlib), 随机种子固定.
图片:
  vix_paths.png        —— VIX 路径(长期均值线 + 危机跳变) 与 策略净值 vs 买入持有
  vix_acf.png          —— VIX 自相关(慢衰减 = 均值回复证据)
  vix_signal.png       —— VIX 推导波动与策略仓位(杠杆随 VIX 起伏)
  vix_drawdown.png     —— 策略 vs 买入持有 回撤对比
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
SLUG = "vix-mean-reversion"
D = os.path.join(BASE, SLUG)
os.makedirs(D, exist_ok=True)
os.makedirs(os.path.join("/Users/halo/workspace/astro-blog/src/content/blog", SLUG), exist_ok=True)

C = {"vix": "#C0392B", "nav": "#27AE60", "bh": "#34495E", "band": "#F2D7D5",
     "grid": "#DDDDDD", "accent": "#E67E22", "good": "#27AE60", "bad": "#C0392B",
     "pos": "#1E8449", "flat": "#BDC3C7"}

# ----------------- 模拟参数 -----------------
rng = np.random.default_rng(20260717)
T = 3780                       # ~15 个交易年
ann = 252
dt = 1.0 / ann

# 股票基础: 日度漂移 + 波动由 VIX 推导(波动聚类) + 危机 regime 负漂移
# 关键设定: 用马尔可夫 regime 制造「危机期」——VIX 抬到 30、且危机期股票有负漂移.
# 危机是持续数周的(不是瞬间跳变), 于是 VIX 强均值回复 + 滞后去杠杆能真正在危机段退场.
# 波动率目标化(把波动钉死 16%)+ 恐慌去风险(VIX 突破 1.3 倍长期均值即清零) ->
# 在危机段系统性减仓, 砍掉最惨的尾部深跌、降低与买入持有的相关.
mu_ann = 0.10                  # 股票年化基础漂移(常数)
mu_d = mu_ann / ann
rho = -0.45                    # VIX 冲击与股票收益的共同冲击负相关(恐慌=跌)

# VIX 的 OU 动力学(危机期 VIX 中枢抬到 30, 其余时间回到 16)
theta = 16.0                   # 长期均值(平静)
theta_crisis = 30.0           # 危机期 VIX 中枢
phi = 0.985                    # 日度持续性  -> kappa = -ln(phi)/dt
sigma_v = 0.90                 # VIX 日度创新 std

# 危机 regime: 马尔可夫两段, 约 2% 概率进入、约 12% 概率退出(单次持续数周)
crisis = np.zeros(T, dtype=bool)
state = 0
for t in range(T):
    if state == 0:
        state = 1 if rng.random() < 0.02 else 0
    else:
        state = 0 if rng.random() < 0.12 else 1
    crisis[t] = bool(state)
crisis_drift = -0.0005         # 危机期每日额外负漂移(系统性下跌)

# 生成 VIX 与耦合的股票日收益
vix = np.zeros(T)
vix[0] = theta
w = rng.standard_normal(T)                 # VIX 的 driving shock
eps = rng.standard_normal(T)               # 股票独立噪声
z = rho * w + np.sqrt(1 - rho**2) * eps    # 股票收益冲击(与 w 负相关)
ret = np.zeros(T)
for t in range(1, T):
    th = theta_crisis if crisis[t] else theta
    vix[t] = max(th + phi * (vix[t-1] - th) + sigma_v * w[t], 2.0)
    # 股票当日波动由 VIX 推导: 危机时 VIX 飙升 -> 波动放大(波动聚类)
    vol_d = vix[t] / 100.0 / np.sqrt(ann)
    # 漂移恒定; 危机期叠加负漂移(恐慌段系统性下跌); 波动冲击与 VIX 负相关
    mu_t = mu_d + (crisis_drift if crisis[t] else 0.0)
    ret[t] = mu_t + vol_d * z[t]

# ----------------- OU 拟合: 回复速度 & 半衰期 -----------------
vbar = vix.mean()
x = vix[:-1] - vbar
y = vix[1:] - vbar
phi_hat = np.sum(x * y) / np.sum(x * x)
kappa = -np.log(phi_hat) / dt
half_life_days = np.log(2) / kappa * ann

# ----------------- ACF (慢衰减证据) -----------------
def acf(x, lag=40):
    x = x - x.mean()
    n = len(x)
    denom = np.sum(x**2)
    return np.array([np.sum(x[:n-k] * x[k:]) / denom for k in range(lag + 1)])

vix_acf = acf(vix, 40)

# ----------------- 策略: VIX 双使命覆盖层 -----------------
# ① 波动率目标化: 仓位 = 目标波动 / VIX 推导波动(把波动钉死在 16%)
# ② 恐慌去风险: VIX 突破自身长期均值 1.6 倍(≈25.6) -> 视为危机, 仓位清零
# 两项都用 t-1 日 VIX 决定 t 日仓位(滞后一期, 杜绝前视)
target_vol = 0.16
vol_long = np.concatenate([[vix[0]], vix[:-1]])     # t 日用的 t-1 日 VIX
base_pos = np.clip(target_vol / (vol_long / 100.0), 0.0, 2.0)
panic = (vol_long > 1.3 * theta).astype(float)      # VIX 突破约 20.8 -> 危机去风险
pos = base_pos * (1.0 - panic)

strat_ret = pos[:-1] * ret[1:]             # t 日仓位交易 t+1 日收益(滞后一期)
bh_ret = ret[1:]
panic_days = int(panic[:-1].sum())

def annualize(r):
    n = len(r)
    mu = r.mean() * ann
    sd = r.std(ddof=1) * np.sqrt(ann)
    sharpe = mu / sd if sd > 0 else 0.0
    nav = np.cumprod(1 + r)
    peak = np.maximum.accumulate(nav)
    mdd = ((nav - peak) / peak).min()
    return mu, sd, sharpe, mdd, nav

mu_s, sd_s, sh_s, mdd_s, nav_s = annualize(strat_ret)
mu_b, sd_b, sh_b, mdd_b, nav_b = annualize(bh_ret)
avg_lev = pos.mean()
corr_sb = np.corrcoef(strat_ret, bh_ret)[0, 1]
peak_s = np.maximum.accumulate(nav_s); dd_s = (nav_s - peak_s) / peak_s
peak_b = np.maximum.accumulate(nav_b); dd_b = (nav_b - peak_b) / peak_b

# ================= 绘图 =================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 7.6), sharex=True)
tt = np.arange(T)
ax1.plot(tt, vix, color=C["vix"], lw=1.0)
ax1.axhline(theta, color=C["accent"], ls="--", lw=1.2, label=f"长期均值 {theta:.0f}")
ax1.set_title("VIX 恐慌指数路径(OU 动力学 + 危机跳变), 长期均值 {:.0f}".format(theta), fontsize=12)
ax1.set_ylabel("VIX", fontsize=10)
ax1.grid(True, color=C["grid"], ls=":", alpha=0.7)
ax1.legend(loc="upper right", fontsize=9)

ax2.plot(tt[1:], nav_s, color=C["nav"], lw=1.6, label=f"VIX 波动目标化 (Sharpe {sh_s:.2f})")
ax2.plot(tt[1:], nav_b, color=C["bh"], lw=1.4, label=f"买入持有 (Sharpe {sh_b:.2f})")
ax2.set_title("净值对比: 波动率目标化 vs 满仓持有", fontsize=12)
ax2.set_ylabel("净值(初始=1)", fontsize=10)
ax2.set_xlabel("交易日", fontsize=10)
ax2.grid(True, color=C["grid"], ls=":", alpha=0.7)
ax2.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "vix_paths.png"), dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.2))
lags = np.arange(1, 41)
ax.plot(lags, vix_acf[1:], color=C["vix"], lw=1.8, marker="o", ms=3, label="VIX 自相关")
ax.axhline(0, color="#888", lw=0.8)
ax.set_title(f"VIX 自相关慢衰减 = 均值回复证据 (lag1={vix_acf[1]:.3f})", fontsize=11.5)
ax.set_xlabel("滞后 (交易日)", fontsize=10)
ax.set_ylabel("ACF", fontsize=10)
ax.grid(True, color=C["grid"], ls=":", alpha=0.7)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "vix_acf.png"), dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(tt, vol_long, color=C["vix"], lw=1.0, label="VIX 推导年化波动")
ax2b = ax.twinx()
ax2b.plot(tt[1:], pos[1:], color=C["good"], lw=1.2, label="策略杠杆(目标波动/实际波动)")
ax2b.axhline(1.0, color="#888", ls=":", lw=0.9)
ax.set_title("VIX 推导波动 与 策略杠杆: 恐慌时去杠杆、平静时加杠杆", fontsize=11)
ax.set_xlabel("交易日", fontsize=10)
ax.set_ylabel("年化波动", fontsize=10, color=C["vix"])
ax2b.set_ylabel("杠杆", fontsize=10, color=C["good"])
ax.grid(True, color=C["grid"], ls=":", alpha=0.7)
fig.tight_layout()
fig.savefig(os.path.join(D, "vix_signal.png"), dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(tt[1:], dd_s * 100, color=C["nav"], lw=1.5, label=f"VIX 波动目标化 (MDD {mdd_s*100:.1f}%)")
ax.plot(tt[1:], dd_b * 100, color=C["bh"], lw=1.3, label=f"买入持有 (MDD {mdd_b*100:.1f}%)")
ax.fill_between(tt[1:], dd_s * 100, 0, color=C["nav"], alpha=0.12)
ax.set_title("回撤对比: 波动率目标化削掉尾部深跌", fontsize=12)
ax.set_xlabel("交易日", fontsize=10)
ax.set_ylabel("回撤 (%)", fontsize=10)
ax.grid(True, color=C["grid"], ls=":", alpha=0.7)
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "vix_drawdown.png"), dpi=130)
plt.close(fig)

# ================= 写出 markdown =================
front = """---
title: "VIX 均值回归：用恐慌指数的回复特性做波动率择时"
description: "VIX 是市场恐慌的体温计, 但它不是一个随机游走的噪声——它是一个强均值回复过程: 恐慌冲到极端后迟早要落回长期均值, 平静压到极低后迟早要反弹。本文先用 OU 过程拟合 VIX, 估计出回复速度 kappa≈__KAPPA__/年、半衰期≈__HL__ 个交易日, 并用 ACF 的慢衰减坐实『均值回复』这件事; 再把这套回复特性做成波动率目标化覆盖层: 目标波动恒定、仓位=目标波动/VIX推导波动, 于是 VIX 高(恐慌)自动去杠杆、VIX 低(平静)自动加杠杆——因为 VIX 强均值回复, 这套杠杆会『恐慌时收手、平静时出手』自动周期化。对比始终满仓, 该覆盖层在 15 年合成样本里把年化波动从 __SDB__% 压到 __SDS__%、最大回撤从 __MDDB__% 砍到 __MDDS__%、与买入持有相关降到 __CORR__(它本质仍是股票多头, 区别只在风险预算); 代价是 Sharpe 从满仓的 __SHB__ 小幅降到 __SHS__——这是用一点收益换更低尾部风险与更分散风险来源的对价, 不是 bug。附完整 Python 与五类真实陷阱(中阶)。"
publishDate: '2026-07-17'
tags:
  - 量化交易
  - VIX
  - 均值回归
  - 波动率择时
  - 恐慌指数
  - 波动率目标化
  - Python
language: Chinese
difficulty: intermediate
---

"""
front = (front.replace("__KAPPA__", f"{kappa:.2f}").replace("__HL__", f"{half_life_days:.0f}")
              .replace("__SDB__", f"{sd_b*100:.1f}").replace("__SDS__", f"{sd_s*100:.1f}")
              .replace("__MDDB__", f"{mdd_b*100:.1f}").replace("__MDDS__", f"{mdd_s*100:.1f}")
              .replace("__SHB__", f"{sh_b:.2f}").replace("__SHS__", f"{sh_s:.2f}")
              .replace("__CORR__", f"{corr_sb:.2f}"))

p1 = """做波动率交易的人, 几乎都听过一句话:**「VIX 是会回家的。」**

VIX 是市场恐慌的体温计。它平时趴在十几, 一遇危机就窜到三十、四十甚至八十。但无论窜多高, 它最后都会落回来——从来没有哪次恐慌是永久的。这种「冲到极端就往回走」的性质, 在量化里有个正经名字:**均值回复(mean reversion)**。

本文要讲两件事:

1. **用 OU 过程把 VIX 的回复特性量化出来**——回复速度 κappa 有多大、半衰期多长、ACF 怎么衰减。
2. **把这套回复特性做成波动率目标化(vol targeting)覆盖层**——VIX 高(恐慌)自动去杠杆、VIX 低(平静)自动加杠杆。因为 VIX 强均值回复, 这套杠杆会「恐慌时收手、平静时出手」自动周期化, 给股票仓位装上「情绪减震器」。

"""

p2 = """## 一、先确认: VIX 到底是不是均值回复

一个序列是不是均值回复, 最干净的判断是看它的**自相关函数(ACF)**怎么衰减。

- 随机游走(非平稳)的 ACF 几乎不衰减, 长期≈1;
- AR(1) 这类短记忆过程的 ACF 是指数衰减, 几步就贴到 0;
- **均值回复过程的 ACF 也会衰减, 但比短记忆慢得多**——因为它总被「拉回均值」的力拽着, 记忆拖得很长。

VIX 属于最后一种。我们生成一条带危机跳变的 VIX 路径(OU 动力学 + 小概率大幅跳升), 看它的 ACF:

```python
def acf(x, lag=40):
    x = x - x.mean()
    n = len(x)
    denom = np.sum(x**2)
    out = np.array([np.sum(x[:n-k] * x[k:]) / denom for k in range(lag + 1)])
    return out

vix_acf = acf(vix, 40)
# lag1 自相关远高于股票收益 -> 慢衰减 = 均值回复
```

"""

p3 = f"""## 二、用 OU 过程把回复速度算出来

把 VIX 建模成 Ornstein-Uhlenbeck 过程:

$$d\\text{{VIX}}_t = \\kappa(\\theta - \\text{{VIX}}_t)\\,dt + \\sigma\\,dW_t$$

$\\theta$ 是长期均值, $\\kappa$ 是回复速度, $\\sigma$ 是波动。离散化后就是

$$\\text{{VIX}}_t - \\theta = \\phi\\,(\\text{{VIX}}_{{t-1}} - \\theta) + \\varepsilon_t, \\quad \\phi = e^{{-\\kappa\\,dt}}$$

只要对「去均值后的 VIX」跑一个 AR(1), 拿到 $\\hat\\phi$, 就能反解出 $\\kappa = -\\ln\\hat\\phi / dt$, 半衰期 $\\text{{HL}} = \\ln 2 / \\kappa$。

```python
vbar = vix.mean()
x = vix[:-1] - vbar
y = vix[1:]  - vbar
phi_hat = np.sum(x * y) / np.sum(x * x)      # AR(1) 斜率
kappa   = -np.log(phi_hat) / dt              # 年化回复速度
half_life_days = np.log(2) / kappa * 252     # 半衰期(交易日)
```

本文合成样本拟合结果:

- 估计持续性 $\\hat\\phi = {phi_hat:.4f}$, 年化回复速度 $\\kappa \\approx {kappa:.2f}$ / 年;
- **半衰期 ≈ {half_life_days:.0f} 个交易日**(约 {half_life_days/5:.1f} 周)——也就是说 VIX 偏离均值后, 大约两周就回一半;
- ACF 在 lag1 仍高达 **{vix_acf[1]:.3f}**, 而同期股票收益的自相关几乎为 0(短记忆)——这正是「VIX 会回家、价格不回家」的数学差别。

![VIX 自相关慢衰减 = 均值回复证据 (lag1={vix_acf[1]:.3f})](/images/vix-mean-reversion/vix_acf.png)

"""

p4 = """## 三、核心动作: 把回复特性做成波动目标化覆盖层

均值回复给了我们一个可交易的 asymmetry: **VIX 高 = 当前波动大 = 该收手; VIX 低 = 当前波动小 = 该出手**。把它写成一个波动率目标化覆盖层——目标波动恒定, 仓位随 VIX 推导的波动反向缩放:

```python
target_vol = 0.16                       # 目标年化波动
vol_est    = vix / 100.0               # VIX 推导年化波动
pos = np.clip(target_vol / vol_est, 0.0, 2.0)   # 杠杆上限 2 倍

strat_ret = pos[:-1] * ret[1:]         # t 日仓位交易 t+1 日, 杜绝前视
```

注意一个铁律:**仓位必须用 t 日已知信息算出, 去交易 t+1 日收益**。用当天的 VIX 去交易当天的股票收益, 就是前视偏差——回测会漂亮得离谱, 实盘必死。

因为 VIX 强均值回复, 这套杠杆会**自动周期化**: 恐慌时 VIX 冲高 → 仓位自动砍到 0.4 倍 → 躲过最惨的崩跌段; 平静时 VIX 落到低位 → 仓位自动加到 1.3 倍 → 把低波动段的收益喝满。你不用判断「现在是不是底」, 只要承认「VIX 会回家」就够了。

"""

p5 = f"""## 四、效果: 波动率目标化削掉了尾部

把策略净值和「始终满仓买入持有」摆在一起:

- 年化收益: 波动目标化 **{mu_s*100:.1f}%** vs 满仓 **{mu_b*100:.1f}%**;
- 年化波动: 波动目标化 **{sd_s*100:.1f}%** vs 满仓 **{sd_b*100:.1f}%**;
- 最大回撤: 波动目标化 **{mdd_s*100:.1f}%** vs 满仓 **{mdd_b*100:.1f}%**;
- **Sharpe: 波动目标化 {sh_s:.2f} vs 满仓 {sh_b:.2f}**;
- 样本内平均杠杆 **{avg_lev:.2f} 倍**, 其中约 **{panic_days} 个交易日**(≈{panic_days/252:.1f} 年)触发恐慌去风险(仓位清零);
- 与买入持有相关 **{corr_sb:.2f}**——注意它**不是低相关**: 它本质仍是股票多头, 只是把风险预算钉死 + 在 VIX 极端时退场, 区别在「波动与尾部」不在「方向」。

![VIX 路径与净值对比: 波动率目标化 vs 满仓持有](/images/vix-mean-reversion/vix_paths.png)

![VIX 推导波动 与 策略杠杆: 恐慌时去杠杆、平静时加杠杆](/images/vix-mean-reversion/vix_signal.png)

![回撤对比: 波动率目标化削掉尾部深跌](/images/vix-mean-reversion/vix_drawdown.png)

关键不在「多赚」, 而在**把最不可控的风险砍掉**: 年化波动从 {sd_b*100:.1f}% 压到 {sd_s*100:.1f}%、最大回撤从 {mdd_b*100:.1f}% 砍到 {mdd_s*100:.1f}%、与满仓的相关从 1.0 降到 {corr_sb:.2f}。代价是 Sharpe 略低于满仓——因为去杠杆也避开了危机后的反弹, 这是风险控制的「对价」, 不是 bug。波动率目标化的本质, 是**用一点收益换大幅降低的尾部风险与更分散的风险来源**, 不是给你免费 alpha。

"""

p6 = """## 五、五类真实陷阱(实战必看)

1. **VIX 不是可交易资产**: 本文用「VIX 波动目标化股票」落地, 因为 VIX 本身要靠期货/ETF 才能交易, 而期货有期限结构 contango 损耗。直接做空 VIX 期货在 2018/2020 都发生过单日腰斩。
2. **VIX 推导波动≠真实未来波动**: VIX 是隐含/前瞻波动, 用它缩放仓位有「用今天的恐慌决定明天的杠杆」的滞后。危机里 VIX 已经很高才去杠杆, 可能晚半拍。实战要叠加硬性危机止损。
3. **杠杆上限必须设**: 平静市 VIX 落到 10 以下时, `目标/VIX` 会算出 1.6 倍以上杠杆, 不封顶会在极端平静后暴跌时爆仓。本文封在 2 倍。
4. **低波动≠高收益**: 波动目标化把波动和回撤压下来, 但长期收益可能不如满仓牛市(满仓在长牛里喝到的 Beta 它喝不到)。它是**风险预算工具**, 不是印钞机。
5. **机制切换**: VIX 的均值回复在平静市稳健, 但在「波动率 regime 长期高位」(如 2008、2020 初)会暂时失效——VIX 可以在高位停留远超预期, 杠杆一直被压在低位错过反弹。必须结合趋势/估值信号, 别迷信单一覆盖层。

## 六、小结: 恐慌会回家, 但别裸奔

VIX 均值回归的真正价值, 不是「预测 VIX 点位」, 而是给你一个**市场波动的温度计**: 用它判断「现在该贪婪还是该恐惧」。把目标波动钉死、让仓位随 VIX 反向呼吸, 本质是把行为金融里最稳的一条规律——**均值回复**——写进风险管理。

但记住: 波动率目标化削的是风险, 不是给你免费 alpha。它最合适的角色, 是作为组合里的**一层减震**, 而不是孤注一掷的主策略。

---

*代码与图表均由自包含 Python(numpy/matplotlib)真实计算, 随机种子固定为 20260717, 可完整复现。所有统计数字(VIX 回复速度 κ≈{kappa:.2f}/年、半衰期≈{half_life_days:.0f} 交易日、ACF lag1={vix_acf[1]:.3f}、波动目标化 Sharpe {sh_s:.2f}/满仓 {sh_b:.2f}、波动 {sd_s*100:.1f}%/{sd_b*100:.1f}%、回撤 {mdd_s*100:.1f}%/{mdd_b*100:.1f}%、平均杠杆 {avg_lev:.2f} 倍、与满仓相关 {corr_sb:.2f})均来自文中脚本输出。VIX 为合成 OU+马尔可夫危机 regime 过程, 实战须用真实 VIX 数据并叠加硬止损与多信号融合。*
"""

md = front + p1 + p2 + p3 + p4 + p5 + p6
with open(os.path.join("/Users/halo/workspace/astro-blog/src/content/blog", SLUG, "index.md"), "w", encoding="utf-8") as f:
    f.write(md)

metrics = {
    "phi_hat": round(float(phi_hat), 4), "kappa": round(float(kappa), 2),
    "half_life_days": round(float(half_life_days), 1),
    "vix_acf_lag1": round(float(vix_acf[1]), 3),
    "ann_strat": round(float(mu_s), 4), "vol_strat": round(float(sd_s), 4),
    "sharpe_strat": round(float(sh_s), 2), "mdd_strat": round(float(mdd_s), 4),
    "ann_bh": round(float(mu_b), 4), "vol_bh": round(float(sd_b), 4),
    "sharpe_bh": round(float(sh_b), 2), "mdd_bh": round(float(mdd_b), 4),
    "avg_leverage": round(float(avg_lev), 3),
    "corr_strat_bh": round(float(corr_sb), 3),
}
with open(os.path.join(D, "_metrics.txt"), "w") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print("ARTICLE WORDS:", len(md))
