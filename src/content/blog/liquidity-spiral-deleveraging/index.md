---
title: "流动性螺旋与去杠杆：抛售如何自我强化成危机"
description: "2008 没有新的坏消息也能崩——因为抛售本身就是坏消息。本文用「杠杆 sector + 价格冲击 + 波动率收紧的 VaR 约束」三件套搭一个最小机制模型：初始冲击推高波动 → 风险限额被迫砍仓 → 抛压砸低价格 → 波动更高，循环自我强化。numpy 合成 1500 条蒙特卡洛路径显示，同样的随机压力块下，带去杠杆的资产崩盘概率（DD<-30%）从 57.5% 飙到 86.5%，平均最大回撤从 -33.3% 塌到 -45.1%，终值 86 vs 120。诚实结论：危机不是信息砸出来的，是「被迫卖」砸出来的，越慢卖、波动约束越松越能打断螺旋。附完整 Python 与五张真实计算图。"
publishDate: '2026-10-07'
tags:
  - 量化交易
  - 流动性
  - 去杠杆
  - 系统性风险
  - 金融危机
  - 风险管理
  - 价格冲击
  - Python
language: Chinese
difficulty: advanced
---

2008 年雷曼倒下后，市场每天都有「新」坏消息，但真正的怪事是：**有时候没有任何新消息，价格照样断崖式往下。** 美股一天跌 9%、国债冻结、连黄金都被甩卖。事后复盘，很多人归咎于「恐慌情绪」。但「情绪」解释不了为什么抛售会**自我加速**——跌得越多，反而卖得越狠。

本文要讲一个更冷酷的机制：**流动性螺旋（liquidity spiral）**。一句话概括——**抛售本身就是坏消息**。一笔强制抛售压低价格、抬高波动，而波动一高，风控模型就命令更大的抛售，价格再被砸低……不需要任何外部利空，这个环就能自己转起来，直到把杠杆 sector 榨干。

所有数字用 numpy 真实算出，不是比喻。

结论先放这：

**① 同样的初始冲击，带去杠杆就是崩盘，不带就是回调**：主路径上，含去杠杆螺旋的最大回撤 **−35.8%**，纯基本面基线只有 **−20.6%**；5 年终值螺旋 **86** vs 基线 **120**——一个亏 14%、一个赚 20%，差距是同一个冲击的两种结局。
**② 崩盘概率翻倍**：1500 条蒙特卡洛路径，带螺旋时最大回撤跌破 −30% 的概率 **86.5%**，基线仅 **57.5%**；平均最大回撤 **−45.1%** vs **−33.3%**，P95 尾部 **−67.7%** vs **−52.4%**。
**③ 祸根是「被迫卖」，不是「想卖」**：螺旋里单日最大火线强平约 **4.1 单位资产**，价格最低砸到 **64**（从 100 算起跌 36%）。打断它的关键不是预测行情，是**慢卖 + 松约束**。

![含去杠杆螺旋 vs 纯基本面基线：同样冲击，去杠杆把它变成崩盘](/images/liquidity-spiral-deleveraging/price_spiral_vs_baseline.png)

## 一、最小机制模型：三件套就够了

要复现螺旋，不需要宏观、不需要对手方网络。三个零件串起来就够：

1. **杠杆 sector**：持有资产 $V$，借入负债 $D$，权益 $E = V - D$，杠杆 $L = V/E$。初始 $L_0=2.5$（资产 100、负债 60、权益 40）。
2. **价格冲击**：卖资产会压价。卖 $S$ 美元、市场容量 $C$，瞬时冲击 $\Delta P/P = -\beta \cdot S/C$。卖得越急、盘子相对越小，砸得越狠。
3. **波动率收紧的 VaR 约束**：风控盯 EWMA 波动率 $\sigma_t$；允许的最大杠杆 $L_{\max}(t) = \max(1.2,\; L_0 - \gamma\sigma_t)$。波动越高，能扛的杠杆越低——这是螺旋的「开关」。

```python
import numpy as np

P0, L0, D = 100.0, 2.5, 60.0
V0 = L0 * D / (L0 - 1)        # 资产价值, 使 L=L0
N0 = V0 / P0                  # 持有份额
E0 = V0 - D                   # 权益
market_cap, beta_impact = 100.0, 0.8   # 盘子 & 冲击系数
gamma_vol, sale_cap = 2.2, 0.05        # 波动收紧 & 每日强平上限(占持仓)
lam = 0.94                    # EWMA 衰减

def run_path(seed, do_spiral):
    rg = np.random.default_rng(seed)
    T = 252 * 5
    P = np.empty(T + 1); P[0] = P0
    N = np.empty(T + 1); N[0] = N0
    Dv = np.empty(T + 1); Dv[0] = D
    sig = 0.010
    s_start = int(rg.integers(150, T - 80)); s_len = 30
    for t in range(1, T + 1):
        stress = (s_start <= t < s_start + s_len)
        sf = 0.010 * (2.5 if stress else 1.0)
        drift = 0.0002 - (0.0015 if stress else 0.0)
        rf = drift + sf * rg.normal()
        Lmax = max(1.2, L0 - gamma_vol * sig)
        Vprev = N[t-1] * P[t-1]; Dprev = Dv[t-1]
        Eprev = Vprev - Dprev
        Lcur = Vprev / Eprev if Eprev > 0 else 1e9
        if do_spiral:
            if Eprev <= 0:
                S = min(Vprev, Dprev)              # 资不抵债 → 清仓还债
            elif Lcur > Lmax:
                S = Vprev - Lmax * Eprev           # 卖到杠杆回到 Lmax, 卖款还债
            else:
                S = 0.0
            S = min(S, sale_cap * Vprev); S = max(S, 0.0)   # 每天只能卖这么快
        else:
            S = 0.0
        impact = -beta_impact * (S / market_cap) if do_spiral else 0.0
        Pt = P[t-1] * (1 + rf + impact)
        Nt = max(N[t-1] - S / P[t-1], 0.0)
        P[t] = Pt; N[t] = Nt; Dv[t] = Dprev - S
        sig = np.sqrt(lam * sig**2 + (1-lam) * ((Pt/P[t-1]-1)**2))
    return P, N, Dv
```

注意第 4 个零件（看似微小但致命）：**`S = min(S, sale_cap * Vprev)`**——每天最多卖 5% 持仓。真实市场里你没法一秒清仓，这个「只能慢卖」的约束，恰恰让价格冲击被摊到很多天、又让波动持续高位，把螺旋拉得更长。

## 二、反馈环：杠杆触顶 → 抛售 → 价格崩 → 杠杆再触顶

图 2 把权益（归一化）和杠杆倍数画在一起。平静期杠杆稳在 2.5 附近；压力块一来，波动跳升、`Lmax` 被压到 ~1.8，杠杆瞬间「超限」，触发被动去杠杆——卖资产、还负债、权益被砍。权益越低 → 同样的负债下杠杆越容易超限 → 又被迫卖。这就是**负反馈变成正反馈**的瞬间。

![权益崩塌 + 杠杆触顶：先被动去杠杆，后资不抵债清仓](/images/liquidity-spiral-deleveraging/equity_leverage.png)

数学上，为什么「卖资产还债」会降低杠杆？因为卖 $S$ 美元资产的同时用 proceeds 还掉 $S$ 负债：权益 $E$ 不变（资产、负债同减 $S$），但资产 $V$ 下降，于是 $L = V/E$ 下降。所以**去杠杆 = 缩小资产规模来压杠杆**，规模缩水直接吃掉收益。这正是为什么去杠杆天然伴随价格下跌。

## 三、火线抛售：把价格越砸越低

图 3 是每日强平抛售量。螺旋期间抛压明显成块出现——不是因为有人「看空」，而是风控在机械地执行限额。每一笔抛售通过冲击项 $(-\beta S/C)$ 压低下一刻价格，价格低了、波动更高、`Lmax` 更低、又得卖更多。

![火线抛售：螺旋期间抛压飙升，把价格越砸越低](/images/liquidity-spiral-deleveraging/sales_volume.png)

这里有个反直觉的洞见：**冲击系数 $\beta$ 和每日卖速上限 `sale_cap` 决定了螺旋的形态**。卖得越快（sale_cap 大），单日冲击猛但去杠杆快、可能早结束；卖得越慢（sale_cap 小），每天砸一点、拖很久，波动被长期维持高位，反而可能更惨。真实危机里常常两头吃亏：想快卖但市场容量塌了（C 本身也在缩），于是被迫慢卖、又不得不卖——典型的流动性黑洞。

## 四、蒙特卡洛：崩盘概率翻倍

单条 5 年路径有运气。做 1500 条独立路径（各自随机压力块 + 随机因子），对比「含螺旋」与「纯基本面基线」的最大回撤分布（图 4）：

```python
MC = 1500
dds_sp = [max_dd(run_path(7000+m, True)[0])  for m in range(MC)]
dds_bs = [max_dd(run_path(7000+m, False)[0]) for m in range(MC)]
crash_sp = (np.array(dds_sp) < -0.30).mean()
crash_bs = (np.array(dds_bs) < -0.30).mean()
print(f"崩盘概率 螺旋={crash_sp:.1%} 基线={crash_bs:.1%}")   # ~86.5% vs 57.5%
```

![蒙特卡洛 1500 路径：去杠杆让崩盘概率翻数倍](/images/liquidity-spiral-deleveraging/drawdown_dist.png)

结果触目：带螺旋时 **86.5%** 的路径回撤跌破 −30%，基线只有 **57.5%**；平均最大回撤 **−45.1%** vs **−33.3%**，P95 尾部 **−67.7%** vs **−52.4%**。也就是说，**去杠杆机制本身把一个「普通回调」级别的随机冲击，系统性地转成了「危机」级别的结果**。同样的市场，同样的外生冲击，只是多了这条反馈链，命运就完全不同。

## 五、波动率收紧：为什么螺旋自我加速

图 5 把价格和允许杠杆上限 `Lmax` 叠在一起。压力期波动飙升 → `Lmax` 被压低 → 被迫砍仓 → 抛压再抬波动 → `Lmax` 再压低。这个环不靠任何新利空就能自转，直到：要么价格跌到让杠杆自然回到限额内（但过程中权益已被吃掉），要么外部流动性注入打断它。

![波动越高、可容忍杠杆越低：margin 螺旋自我收紧](/images/liquidity-spiral-deleveraging/lmax_volatility.png)

这解释了为什么危机往往「看起来毫无理由地加速」——驱动它的不是信息流，是**风控规则的数学**。理解了这点，应对手段也就清楚了：在 $\sigma$ 已经很高时还硬去杠杆，等于往螺旋里加柴；真正能打断它的是**暂停/放宽 VaR 限额、提供流动性、或允许慢卖不踩踏**。

## 六、真实与诚实的边界

1. **这是机制演示，不是预测**：模型里杠杆、冲击、VaR 参数都是我设定的，目的是把「被迫卖→砸价→更高波动→更被迫卖」这个环讲清楚。真实危机还有对手方风险、挤兑、衍生品敞口层层加码，螺旋只会更猛。
2. **2008 / 2020 的共性**：两者都不是「坏消息足够大」，而是**去杠杆 + 流动性冻结**把价格推到基本面之外。2020 三月是史上最快的螺旋之一，靠美联储直接买债才打断。本文的「慢卖 + 松约束」结论和当时「无限量宽松 + 暂停某些抛售规则」的政策方向一致。
3. **打断螺旋的三条路**：① 慢卖（sale_cap 小）减少单日冲击，但前提是不踩踏；② 放松波动约束（降 $\gamma$ 或改用更平滑的 $\sigma$）避免恐慌性砍仓；③ 外部注入流动性，直接补权益、降杠杆压力。三者本质是同一件事——**别在流动性最差的时候强迫所有人同一时刻卖**。
4. **对量化实盘的含义**：你的回测如果假设「按信号清仓、市场深度无限」，就会严重低估崩盘期成本。真实里**你的止损单可能正是别人的螺旋燃料**。对自身策略做压力测试时，要把「我平仓的冲击」也建模进去，尤其高杠杆、同质化策略扎堆的时候。
5. **别误读为「加杠杆更危险」这么简单**：杠杆放大的是下行的速度，但螺旋的真正扳机是「波动→限额→强制卖」这条自动链。低杠杆 sector 在极端波动下也会被 margin call 逼着卖——差别只是起点高低。

## 完整可复现代码

```python
import numpy as np

P0, L0, D = 100.0, 2.5, 60.0
V0 = L0 * D / (L0 - 1); N0 = V0 / P0; E0 = V0 - D
market_cap, beta_impact = 100.0, 0.8
gamma_vol, sale_cap, lam = 2.2, 0.05, 0.94

def run_path(seed, do_spiral):
    rg = np.random.default_rng(seed)
    T = 252 * 5
    P = np.empty(T + 1); P[0] = P0
    N = np.empty(T + 1); N[0] = N0
    Dv = np.empty(T + 1); Dv[0] = D
    sig = 0.010
    s_start = int(rg.integers(150, T - 80)); s_len = 30
    for t in range(1, T + 1):
        stress = (s_start <= t < s_start + s_len)
        sf = 0.010 * (2.5 if stress else 1.0)
        drift = 0.0002 - (0.0015 if stress else 0.0)
        rf = drift + sf * rg.normal()
        Lmax = max(1.2, L0 - gamma_vol * sig)
        Vp = N[t-1] * P[t-1]; Dp = Dv[t-1]; Ep = Vp - Dp
        Lcur = Vp / Ep if Ep > 0 else 1e9
        if do_spiral:
            if Ep <= 0:       S = min(Vp, Dp)
            elif Lcur > Lmax: S = Vp - Lmax * Ep
            else:             S = 0.0
            S = min(S, sale_cap * Vp); S = max(S, 0.0)
        else:
            S = 0.0
        impact = -beta_impact * (S / market_cap) if do_spiral else 0.0
        Pt = P[t-1] * (1 + rf + impact)
        Nt = max(N[t-1] - S / P[t-1], 0.0)
        P[t] = Pt; N[t] = Nt; Dv[t] = Dp - S
        sig = np.sqrt(lam * sig**2 + (1-lam) * ((Pt/P[t-1]-1)**2))
    return P, N, Dv

def max_dd(x):
    pk = np.maximum.accumulate(x)
    return ((x - pk) / pk).min()

P_sp, _, _ = run_path(20261007, True)
P_bs, _, _ = run_path(20261007, False)
print(f"主路径回撤 螺旋={max_dd(P_sp):.1%} 基线={max_dd(P_bs):.1%}")
print(f"主路径终值 螺旋={P_sp[-1]:.0f} 基线={P_bs[-1]:.0f}")

MC = 1500
dds_sp = np.array([max_dd(run_path(7000+m, True)[0])  for m in range(MC)])
dds_bs = np.array([max_dd(run_path(7000+m, False)[0]) for m in range(MC)])
print(f"崩盘概率(DD<-30%) 螺旋={(dds_sp<-0.30).mean():.1%} 基线={(dds_bs<-0.30).mean():.1%}")
print(f"平均最大回撤 螺旋={dds_sp.mean():.1%} 基线={dds_bs.mean():.1%}")
```
