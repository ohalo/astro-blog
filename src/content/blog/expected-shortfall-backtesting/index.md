---
title: "预期短缺回测：用 ES 检验替代 VaR 的后验失序"
description: "VaR 只数「有没有超出」，对「超出后到底多惨」失明：尾部一肥，VaR 还绿，ES 已爆。用 Acerbi-Szekely 的 ES-z 后验检验（零假设 E[U]=ES−VaR）替代 VaR 红绿灯，合成里正态世界 ES-z=−0.53（绿）、含灾难跳变的肥尾世界真实 ES≈4.05% 远超正态 2.57% 而 VaR 仍≈2.23%（绿），ES-z 检验功效随尾部肥瘦从 0.37 升到 0.86、VaR 红绿灯却始终在 0.04~0.09 装死。附完整 Python 与六类真实陷阱（高阶）。"
publishDate: '2026-07-15'
tags:
  - 量化交易
  - 风险计量
  - 预期短缺
  - 后验检验
  - VaR
  - 压力测试
  - 监管
  - Python
language: Chinese
difficulty: advanced
---

2008 年之后，巴塞尔协议把「风险计量应该看什么」悄悄改了：从 VaR（Value at Risk，在险价值）换成了 ES（Expected Shortfall，预期短缺，也叫 CVaR）。理由一句话就能说清——**VaR 只告诉你「最坏情况下会亏多少」，却不告诉你「万一超过那条线，会亏到什么程度」**。尾部越肥，这道盲区越致命。本文要讲的就是：**怎么用一套「后验检验」去戳穿 VaR 的盲区，把 ES 也变成可以回测、可以判红绿的东西**。

结论先放这：**在一条「正态世界」的合成损益序列上（99% VaR≈2.24%、ES≈2.57%），Acerbi-Szekely 的 ES-z 检验统计量 z≈−0.53，落在绿区，说明模型自洽；但一旦真实世界换成「基础正态 + 罕见灾难跳变」——灾难每笔约 −5%、发生概率 0.6%——真实 99% VaR 仍≈2.23%（VaR 红绿灯照样绿），真实 ES 却被肥尾顶到≈4.05%，远高于模型报告的 2.57%。此时 ES-z 统计量抬升到 1.62、明显朝红区走，而 VaR 红绿灯却始终在绿区装死。** 扫描尾部肥瘦：ES-z 检验的「拒绝功效」随真实 ES 从 0.37 升到 0.86，VaR 红绿灯的「判红概率」却一直趴在 0.04~0.09。**尾部越肥，VaR 越瞎、ES 越灵**——这正是监管弃 VaR 用 ES 的核心动机。

![VaR 只画一条线，ES 还要求『超出之后平均有多惨』](/images/expected-shortfall-backtesting/es_loss_distribution.png)

## 一、VaR 的盲区：它只数「有没有超出」

VaR(q)（置信水平 1−q，这里取 99%）说的是：「未来一段时间里，有 99% 的把握损失不超过 VaR」。换成损失的记号（L=−损益，正数代表亏损）：

```
P( L ≤ VaR ) = 99%
```

它回答「最坏大概亏多少」。但它**不回答**：当 L 真的冲过 VaR 那条线时，后面那段「超出部分」平均有多惨？这个「超出部分的平均」就是 ES：

```
ES = E[ L | L > VaR ]
```

问题就在这里：**VaR 后验检验（traffic-light，红绿灯）只数「有多少天超出了 VaR」**。只要超出天数落在预期区间（对 99% VaR、1000 天窗口，期望约 10 天），VaR 就是绿的。可一旦尾部变肥——超出 VaR 的那些天不再是「刚好擦线」，而是「擦线后狠狠再跌一截」——VaR 红绿灯**完全看不见**，因为它根本不读超出之后的深度。

## 二、ES 的定义：把「超出之后多惨」写进一个数

给定损益分布，99% VaR 是损失的第 99 个百分位；ES 是第 99 个百分位**以右**那段损失的条件期望。直觉上：

- 正态分布下，超出 VaR 的部分只是略深一点，ES 比 VaR 高得有限（本文正态设定 ES≈2.57% vs VaR≈2.24%）。
- 肥尾分布下，超出 VaR 的部分会被极端灾难拉得很深，ES 会显著高过 VaR——而 VaR 自己几乎没动。

这就是 VaR 盲区的数学本质：**两个分布可以有几乎相同的 VaR，却有天差地别的 ES**。光盯 VaR，等于蒙着眼睛做风险管理。

## 三、ES 的后验检验：Acerbi-Szekely 的 z 检验

VaR 红绿灯是「数天数」。ES 能不能也做个「后验检验」？能，而且更聪明——它不数天数，而是检验「超出 VaR 的那部分超额损失，均值到底对不对」。

记超额损失 `U_i = L_i − VaR`（只对 `L_i > VaR` 的那些天）。如果模型报的 ES 是自洽的，那么这些 U 的理论均值应当等于 `ES − VaR`。于是零假设：

```
H0:  E[U] = ES − VaR
检验量:  z = ( mean(U) − (ES−VaR) ) / sqrt( var(U) / n_exc )   ~  N(0, 1)
```

`n_exc` 是超出 VaR 的天数。`|z| > 2.326`（1% 双尾）就判红。这叫 **Acerbi-Szekely ES 后验检验**。注意它的零假设是「超额损失均值 = ES−VaR」，**不是**「= ES」——这是新手最容易踩的坑，把单位搞错检验就全反了。

```python
import numpy as np
from scipy import stats

ALPHA = 0.99
Z = stats.norm.ppf(ALPHA)          # ≈2.326
PHI = stats.norm.pdf(Z)            # ≈0.0266
MU, SIG0 = -0.03, 0.974

def es_of_normal(mu, sigma):
    var = mu + sigma * Z
    es = mu + sigma * PHI / (1 - ALPHA)
    return var, es

var0, es0 = es_of_normal(MU, SIG0)     # VaR≈2.236, ES≈2.566
ES_NOM = es0                          # 风险模型「报告」的 ES（正态假设下=真值）

def es_z_stat(X, var_loss, es_report):
    """Acerbi-Szekely ES-z 检验：零假设 E[U] = ES − VaR。"""
    L = -X
    exc = L[L > var_loss]
    n = len(exc)
    if n < 5:
        return np.nan, n
    u = exc - var_loss
    target = es_report - var_loss
    z = (u.mean() - target) / np.sqrt(u.var(ddof=1) / n)
    return z, n
```

## 四、合成对比：正态世界 vs 含灾难跳变的肥尾世界

为了干净演示「VaR 装死、ES 报警」，构造两类日损益：

1. **正态世界**：`X ~ N(μ, σ²)`，99% VaR≈2.24%，ES≈2.57%。
2. **肥尾世界**：基础仍是正态（σ 略小，给灾难留空间），但每笔有 0.6% 概率叠加一个「灾难跳变」——均值 −5%、波动 1%。灾难只落在 VaR 以右的极端区，所以**把 99% VaR 几乎没动，却把 ES 顶得很高**。

```python
rng = np.random.default_rng(202)
X_ok = rng.normal(MU, SIG0, 1000)          # 正态世界

# 肥尾世界：基础正态 + 罕见灾难跳变
FAT_S, FAT_Q, FAT_MUJ, FAT_SJ = 0.830, 0.006, -5.0, 1.0
r = np.random.default_rng(202)
X_bad = r.normal(MU, FAT_S, 1000)
jump = r.random(1000) < FAT_Q
if jump.sum() > 0:
    X_bad[jump] += r.normal(FAT_MUJ, FAT_SJ, jump.sum())

z_ok, n_ok = es_z_stat(X_ok, var0, ES_NOM)     # -> z≈-0.53（绿）
z_bad, n_bad = es_z_stat(X_bad, var0, ES_NOM)  # -> z≈1.62（明显抬升）
```

跑出来：**正态世界 ES-z ≈ −0.53（绿，模型自洽）；肥尾世界 ES-z ≈ 1.62，明显朝红区抬升**。而两类世界的 VaR 超出天数都落在「绿区」——VaR 红绿灯对肥尾**毫无反应**。

![正态 vs 肥尾：VaR 超出天数都绿，但肥尾那列的超出后明显更深](/images/expected-shortfall-backtesting/es_breach_sequences.png)

## 五、功效曲线：尾部越肥，VaR 越瞎、ES 越灵

单条路径有随机性。更有说服力的是**功效曲线**：固定模型报告 ES=2.57%（正态假设），让真实世界的尾部从「轻度肥」扫到「极度肥」（放大灾难跳变幅度），每条设定重复 1500 次，看 ES-z 检验「判红」的比例（功效）和 VaR 红绿灯「判红」的比例各是多少。

```python
def var_red(X, var_loss):
    b = int((-X > var_loss).sum())
    return b >= 15                      # N=1000 时，超出≥15 天判红

muj_list = np.array([-3.5, -4.0, -4.5, -5.0, -6.0, -8.0, -10.5, -14.0])
power_es, power_var = [], []
for muj in muj_list:
    rej_es = red_var = 0
    for m in range(1500):
        r = np.random.default_rng(1000 * m + 7)
        Xm = r.normal(MU, FAT_S, 1000)
        j = r.random(1000) < FAT_Q
        if j.sum() > 0:
            Xm[j] += r.normal(muj, FAT_SJ, j.sum())
        z, _ = es_z_stat(Xm, var0, ES_NOM)
        if np.isfinite(z) and abs(z) > 2.326:
            rej_es += 1
        if var_red(Xm, var0):
            red_var += 1
    power_es.append(rej_es / 1500)
    power_var.append(red_var / 1500)
```

结果刻在图里：**ES-z 检验功效随真实 ES 从 0.37 一路升到 0.86——尾部越肥越容易抓到；VaR 红绿灯判红概率却始终在 0.04~0.09 之间原地踏步**。两线越拉越开，就是「VaR 失明、ES 灵敏」最直白的量化证据。

![尾部一肥再肥：ES-z 检验迅速反应，VaR 红绿灯几乎无感](/images/expected-shortfall-backtesting/es_power_curve.png)

## 六、一条回测路径上的「双轨报警」

把肥尾世界的一条真实回测路径画出来：VaR 线（虚线红）和模型报告的 ES 线（点线红）几乎贴在一起，VaR 超出日零散分布、红绿灯绿。但把全部超出日的超额损失喂进 ES-z 检验，统计量明显抬升到 1.62——**同样的数据，VaR 后验说「没事」，ES 后验说「你的尾部假设太乐观了」**。这正是监管要求的「ES 后验检验」要补的那块短板。

![一条回测路径：VaR 绿、ES-z 明显抬升](/images/expected-shortfall-backtesting/es_backtest_demo.png)

## 七、真实陷阱：比想象中更硬的六类

**陷阱一：零假设写错单位。** ES-z 检验的零假设是 `E[U] = ES − VaR`，不是 `= ES`。U 是「超出 VaR 之后的那一段」，它的均值天然比 ES 小一个 VaR。把 U 直接和 ES 比，检验量会系统性偏负、永远判绿（或永远判红），整个检验作废。本文所有 z 都用 `target = es_report − var_loss` 计算。

**陷阱二：超出样本太少，检验没力。** ES-z 检验需要足够多的超出日（n_exc）才有功效。99% VaR、250 天窗口只有约 2.5 天超出，检验几乎永远是「不确定」——这就是为什么本文用 1000 天窗口（约 10 天超出）。窗口太短，ES 检验退化成噪声，反而显得「不如 VaR 红绿灯」。

**陷阱三：VaR 红绿灯本身不可靠的「绿」。** 红绿灯只数天数、不读深度。本文肥尾世界 VaR 仍≈2.23%（绿），真实 ES 却 4.05%——红绿灯的「绿」是假象。把监管合规建立在「数天数」上，等于鼓励机构用肥尾分布把 VaR 做平、逃避资本要求，这正是 ES 替代 VaR 的初衷。

**陷阱四：灾难跳变的稀疏性导致路径噪声大。** 0.6% 的灾难概率，1000 天里平均才 6 笔，单条路径里「有没有踩中大灾」对 z 影响极大。所以单条路径 z=1.62 只是示意，真正落地要靠多条路径 / 多年数据的聚合（功效曲线那种），不能拿一条说事。

**陷阱五：ES 也依赖分布假设来做滚动监控。** ES-z 检验用样本方差 `var(U)` 做标准误，若 U 本身厚尾（灾难连发），标准误被高估、检验偏保守。实务上有人用 bootstrap 或截尾估计标准误，避免「灾难之年反而测不出尾部」的反直觉结果。

**陷阱六：回测用历史损益，没区分「模型 VaR」与「实际已实现」。** 后验检验的前提是 VaR/ES 由同一套风险模型在事前算出。若拿事后真实损益直接反推 VaR，等于用未来信息作弊，检验会虚高。落地时必须用**滚动的事前预测 VaR** 配**事后真实损益**，时间戳要对齐。

## 八、诚实结论

VaR 不是错，它只是「不够」。当尾部变肥，VaR 的「有没有超出」几乎不动，ES 的「超出后多惨」却剧烈反应——**这是数学必然，不是主观偏好**。本文用 Acerbi-Szekely 的 ES-z 后验检验把这件事量化：正态世界 z≈−0.53（绿、自洽），含灾难跳变的肥尾世界真实 ES≈4.05% 远超正态 2.57% 而 VaR 仍≈2.23%（绿），ES-z 抬升到 1.62；功效曲线上 ES 检验随尾部肥瘦从 0.37 爬到 0.86，VaR 红绿灯始终在 0.04~0.09 装死。

但落到自己账户，先过**零假设单位、超出样本量、红绿灯假绿、灾难稀疏噪声、ES 标准误厚尾、事前/事后时间戳**这六关。最有用的姿势，是把 ES 后验检验当成 VaR 红绿灯的**升级版警报器**——VaR 说绿你还别全信，再看一眼 ES-z 有没有悄悄抬升。

> 注：全文数据为自洽合成（正态世界 + 正态基础叠加灾难跳变，灾难参数经数值校准使 99% VaR 与正态一致），仅用于演示 VaR 盲区与 ES 后验检验的统计性质。真实复现请替换为实际损益序列，并对零假设单位、超出样本量、时间戳对齐与标准误估计逐一做稳健性检验。
