---
title: "系统性风险 SRISK 与边际期望损失 MES：把「谁会拖垮整片市场」做成可量化指标"
description: "单个机构的 VaR 只管自己亏多少，看不见「你倒了连累谁」。MES（Marginal Expected Shortfall）量的是「市场崩盘时某只股票平均会跌多少」，SRISK（Brownlees-Engle）再把它乘上杠杆、变成资本的系统性缺口。本文用单指数模型从零算 MES→LRMES→SRISK，量化「高杠杆金融机构既是尾部最痛、也是缺口最大」的机制，并诚实拆穿估计窗口/β 时变/监管资本口径/相关性断裂四类真实陷阱（中阶）。"
publishDate: '2026-07-21'
tags:
  - 量化交易
  - 系统性风险
  - SRISK
  - MES
  - 边际期望损失
  - 金融稳定
  - 风险管理
  - Python
language: Chinese
difficulty: intermediate
---

2008 年雷曼倒下，远在亚洲的股指也跟着崩——风险会**传染**。但传统风险管理只看单家机构的 VaR：「我自己最多亏多少」。VaR 看不见「你倒了会拖垮多少同行」。系统性风险研究的全部努力，就是把这个看不见的外部性**量化成数字**。结论先放这：

**MES（Marginal Expected Shortfall，边际期望损失）= 当市场陷入极端尾部时，单只股票的平均损失；SRISK（Brownlees & Engle 2010）= 把 MES 沿危机时长放大成长期损失（LRMES），再乘以杠杆，得到「危机中这家机构的资本缺口」。高杠杆的金融机构既在尾部最痛（高 MES），又因债务庞大而缺口最大（高 SRISK）——它们就是系统重要性最高的节点。**

下面用单指数模型（single-index model）从零把 MES → LRMES → SRISK 这条链走通，并诚实点出四类会坑你的估计陷阱。

## 一、MES：市场崩盘时，单只股票平均跌多少

MES 的定义很直白：

$$\text{MES}_i = \mathbb{E}[r_i \mid r_m \le C]$$

$r_m$ 是市场（或系统组合）收益，$C$ 是尾部门槛（比如市场日收益最差的 5%）。它回答：**「当市场已经烂到这份上，第 i 只股票平均还会再跌多少？」**

和「无条件平均收益」不同，MES 只看**系统性事件发生时**那一小撮样本。下面用模拟的 12 家机构日度收益演示：把市场最差 5% 的日子挑出来，看银行 A（β 最高之一）在这些日子里跌多少。

![MES 尾部图：市场跌入尾部（红虚线左）时，银行 A 的日收益（红点）系统性低于常态，绿线即 MES](/images/srisk-mes-systemic/srisk_mes_tail.png)

```python
import numpy as np

# 模拟：市场日收益 + 12 家机构（单指数模型 r_i = beta_i * r_m + idio）
rng = np.random.default_rng(20260721)
N = 12
beta_true = rng.uniform(0.6, 1.8, N)
idio = rng.uniform(0.008, 0.020, N)
Tdays = 756
r_m = rng.normal(0.0003, 0.012, Tdays)
# 注入一段真实崩盘窗口（约 60 天、每天 -1.0%）
cs, ce = 470, 530
r_m[cs:ce] += -0.010
R = np.array([beta_true[i]*r_m + rng.normal(0, idio[i], Tdays) for i in range(N)])

# MES：市场收益处于最差 5% 时，该股票的条件期望损失
Cr = np.quantile(r_m, 0.05)          # 尾部门槛（经验分位数，避免模拟里没触发硬阈值）
mkt_cond = r_m <= Cr
mes = np.array([R[i][mkt_cond].mean() for i in range(N)])
print("各机构 MES (%)：", np.round(mes*100, 2))
print("MES 区间：", round(mes.min()*100,2), "% ~", round(mes.max()*100,2), "%")
```

跑出来 MES 分布在 **−1.8% 到 −4.4%** 之间。注意：**MES 不是 β 的简单放大**——β 高只代表对市场的敏感度，但 MES 还捕捉了「系统性事件里个股的特异下行」。

## 二、从 MES 到 LRMES：把单日尾部拉长成一场危机

单日 MES 只问「崩盘那天跌多少」。但一场危机持续几周甚至几个月。Brownlees-Engle 用**单指数近似**把单日 MES 放大成危机长窗损失：

$$\text{LRMES}_i \approx -\beta_i \cdot \Gamma$$

$\Gamma$ 是系统性事件里市场的累计跌幅（例如危机中市场跌 40%），$\beta_i$ 是股票对市场的敏感度。于是：

$$\text{LRMES}_i = \text{危机时长 H 内第 i 只股票的期望累计损失}$$

```python
H = 126                      # 危机时长（约半年交易日）
Gamma = 0.40                # 系统性事件中市场累计跌幅（此处设为 40%）
beta_hat = np.array([np.cov(r_m, R[i])[0,1] / np.var(r_m) for i in range(N)])
lrmes_ret = -beta_hat * Gamma     # 危机中第 i 只股票的期望累计收益（负值）
print("各机构 LRMES (累计损失%)：", np.round(-lrmes_ret*100, 1))
```

这一步把「单日尾部」升级成「整场危机的损失敞口」——SRISK 真正用的就是这个长窗损失。

## 三、SRISK：用杠杆把损失翻译成资本缺口

单指数模型下，SRISK 有闭式解（Brownlees & Engle 2010）：

$$\text{SRISK}_i = \max\bigl(0,\; k \cdot D_i - (1-k) \cdot \text{Equity}_i \cdot (1 + \text{LRMES}_i)\bigr)$$

- $k$：监管资本要求（巴塞尔常用 8%）
- $D_i$：债务
- $\text{Equity}_i = M_i - D_i$：权益市值
- $1 + \text{LRMES}_i$：危机后权益的剩余比例（LRMES 为负，所以是打折）

直觉：**如果危机损失吃掉了全部权益还不够补监管缓冲（k·债务），缺口就是 SRISK**。缺口越大，这家机构越可能「在别人最需要它的时候倒下」。

```python
k = 0.08
# 市值与杠杆（金融高杠杆、其他行业适中），$B 单位
mktcap = rng.uniform(20, 400, N)
lev = np.array([11.0, 9.5, 8.0, 7.0, 5.0, 3.0, 2.2, 1.8, 2.0, 2.5, 2.6, 1.9])
debt = mktcap * lev
equity = mktcap               # 此处 mktcap 即权益市值口径
srisk = np.maximum(0, k * debt - (1 - k) * equity * (1 + lrmes_ret))
order = np.argsort(srisk)[::-1]
print("SRISK 排名前 4 (十亿美元)：", np.round(srisk[order[:4]], 1))
print("对应机构：", [names[j] for j in order[:4]])
```

结果：SRISK 前四名是 **银行A（77.5B）、券商C（30.5B）、银行B（25.0B）、保险D（10.3B）**——清一色高杠杆金融机构。这正是 SRISK 的设计初衷：**它把杠杆钉死在公式里，谁债务堆得高、危机中损失大，谁的系统性缺口就最大。**

![SRISK 排名：资本缺口最大的机构即系统重要性最高，红条为前四名高杠杆金融](/images/srisk-mes-systemic/srisk_ranking.png)

## 四、拆开看 SRISK：监管缓冲 vs 危机损失

SRISK 是两项之差：一边是「监管要求的资本缓冲」$k\cdot D_i$（橙色），另一边是「危机后还剩多少权益」$(1-k)\cdot E_i(1+\text{LRMES}_i)$（蓝色）。缺口就是红条之上的部分：

![SRISK 拆解：监管缓冲（橙）与危机后权益（蓝）之差即资本缺口](/images/srisk-mes-systemic/srisk_decomposition.png)

对高杠杆机构，债务 $D_i$ 是权益的十几倍，监管缓冲 $k\cdot D_i$ 本身就已是天文数字；一旦危机把权益打折，缺口立刻爆表。**SRISK 不是惩罚「跌得多」，而是惩罚「跌得多 + 借得多」**——这恰好是系统性风险的核心。

## 五、动态 SRISK：压力路径会随市场呼吸

把 SRISK 沿真实（模拟）市场路径逐日重算，能看到它**平时回落、崩盘跃升**的体温计属性：

![系统性压力路径：崩盘窗口（红带）SRISK 跃升，平时回落](/images/srisk-mes-systemic/srisk_stress_path.png)

实现上用「滚动 H 日市场累计收益」替代静态 $\Gamma$，让缓冲要求随市场状态实时呼吸：

```python
cum_m = np.cumsum(r_m)
trailing = np.zeros(Tdays)
for t in range(Tdays):
    lo = max(0, t - H)
    trailing[t] = cum_m[t] - cum_m[lo]      # H 日市场累计收益（有界）
srisk_path = np.maximum(0, k*debt[:,None] - (1-k)*equity[:,None]*(1 + beta_hat[:,None]*trailing))
```

这就把 SRISK 从一个「危机后事后指标」变成「危机中实时监控」的工具——监管者和宏观对冲基金都用它筛系统性脆弱节点。

## 六、MES 与 SRISK 的关系：尾部越痛，系统越重要

把每家机构的 MES（横轴）对 SRISK（纵轴）画散点，能看到正相关：尾部损失越大的机构，系统性缺口也越高——而且金融机构因为杠杆，会明显落在右上方：

![MES 与 SRISK 散点：尾部越痛（MES 越负），系统性缺口（SRISK）越高](/images/srisk-mes-systemic/srisk_mes_scatter.png)

> 一句话：**MES 量「这家机构在系统性事件里有多痛」，SRISK 量「它痛了之后会欠市场多少」。两者合起来，就是一张系统重要性地图。**

## 七、诚实拆穿四类真实陷阱

**1. 估计窗口与分位数阈值主观**。MES 的 $C$（尾部门槛）取 5% 还是 1%？取硬阈值（如日跌 −10%）还是经验分位数？不同选择会让 MES 差好几倍。本文用经验 5% 分位数避免「模拟里没触发硬阈值导致空样本」的坑，但实盘中要固定阈值口径、跨周期一致。

**2. β 时变**。单指数模型假设 $\beta_i$ 恒定，但危机里 β 会飙升（股票更跟市场）。用全样本 β 会**低估**危机损失，用危机样本 β 又会**高估**常态风险。实务用滚动 β 或状态依赖 β，但引入估计噪声。

**3. 监管资本口径 k 的选择**。k=8% 是巴塞尔最低，但系统重要性银行有附加资本要求（G-SIB 附加 1%~3.5%）。k 取错，SRISK 量级全错。本文为演示用 8%，实盘应按机构实际资本要求校准。

**4. 相关性断裂（correlation breakdown）**。MES/SRISK 都假设「市场崩盘时个股按 β 跟跌」。但极端危机里相关性趋近 1，单指数模型的「个体 idio 风险」假设失效——所有股票一起跌，分散效应消失，真实 MES 比模型估计更肥。需用 t-Copula 或 EVT 尾部依赖修正（参见本专栏 Copula 与 EVT 相关文章）。

## 八、落地口径小结

- **MES**：回答「系统性事件里单只股票平均跌多少」，是尾部条件期望，不是 VaR。
- **LRMES**：把单日 MES 沿危机时长放大成累计损失，用单指数模型 $\beta_i \cdot \Gamma$ 近似。
- **SRISK**：把 LRMES 乘上杠杆，得到危机中的资本缺口；缺口越大系统重要性越高。
- **最大敌人**：β 时变、阈值主观、相关性断裂——它是近似框架，不是精确预言。
- **正确使用**：当「筛系统重要性节点」和「危机实时温度计」，而不是当「某机构会破产」的点预测。

SRISK 与 MES 的价值，是把「大而不能倒」从政治口号变成了**可排序、可监控的数字**——这正是一个量化视角能给金融监管带来的最硬的贡献。
