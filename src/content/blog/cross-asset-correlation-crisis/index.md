---
title: "跨资产相关性危机：恐慌时相关性冲向 1 的组合含义"
description: "分散投资的前提是「资产不全都一起跌」。但恐慌一来，跨资产相关性会冲向 1——股票、商品、REITs、新兴市场齐刷刷同涨同跌，只有国债还倔强独立。本文用「单因子+特异波动」模型合成 10 年（含危机块）日度数据：风险资产间相关性从平静期 0.07 飙到危机期 0.74；分散比率（Σwᵢσᵢ/σ_p）从 1.79 塌到 1.11；蒙特卡洛 2500 路径显示风险 4 资产组合 P95 回撤 −8%（均值 −17%），而含国债组合 −7%、60/40 −6%。诚实结论：相关性危机=相关结构塌缩（不是单纯波动变大），分散化红利在你需要它时蒸发，且只有 beta 真正独立的资产（国债）还能对冲。附完整 Python 与五张真实计算图。"
publishDate: '2026-10-07'
tags:
  - 量化交易
  - 资产配置
  - 相关性
  - 分散化
  - 风险管理
  - 危机
  - 组合构建
  - Python
language: Chinese
difficulty: intermediate
---

「不要把鸡蛋放在同一个篮子里」——分散投资是投资界唯一的「免费午餐」。但这句话有个隐藏前提：**篮子们不能在同一时刻一起摔。

现实里，平静时股票、商品、REITs 各涨各的，相关性很低，分散确实免费。可一旦恐慌来袭——2008、2020 疫情、2022 加息——你会眼睁睁看到一件诡异的事：**所有风险资产突然开始同涨同跌，相关性冲向 1**。你以为持有 5 个篮子，危机时发现它们焊成了一个篮子。

本文把这件事量化：**相关性危机到底是什么、它怎么让分散化红利蒸发、以及什么样的资产在危机里还能真正对冲**。所有数字用「单因子 + 特异波动」合成模型 + numpy 真实算出，不是直觉。

结论先放这：

**① 风险资产相关性暴冲**：平静期风险资产间平均相关仅 **0.07**，危机期飙到 **0.74**——几乎同向运动。
**② 分散比率塌缩**：组合「分散比率」(diversification ratio = Σwᵢσᵢ/σ_p) 从平静期 **1.79** 塌到危机期 **1.11**：分散带来的波动折扣几乎消失。
**③ 只有独立 beta 的资产还能对冲**：国债（对共同因子 β≈0）危机期仍独立，含国债组合 P95 回撤 −7% vs 纯风险 4 资产 −8%（均值 −17%）。分散化不是没用，是**依赖你有没有真·独立的资产**。

![平静期 vs 危机期相关矩阵：风险资产相关性冲向 1，仅国债仍独立](/images/cross-asset-correlation-crisis/corr_matrix.png)

## 一、为什么相关性会冲向 1：一个因子模型就够

直觉上，「相关性冲向 1」很邪门——股票和黄金的基本面差那么多，凭什么一起跌？答案在一个极简的结构里：

$$r_{i,t} = \beta_i \cdot F_t + \varepsilon_{i,t}$$

- $F_t$：所有风险资产共同暴露在上的**共同因子**（宏观、流动性、风险偏好）。
- $\beta_i$：资产 $i$ 对共同因子的敏感度。
- $\varepsilon_{i,t}$：资产自身的特异波动（公司/行业层面）。

平时，共同因子波动小、特异波动大 → 各资产的「个性」盖过「共性」→ 相关性低、分散有效。

**危机时，共同因子波动爆炸（$\sigma_F$ 从 5% 飙到 28%），而特异波动不变**。于是所有 $\beta_i>0$ 的资产都被同一个 $F_t$ 驱动，收益近似 $\beta_i F_t$——它们变成同一个东西的不同倍数，**相关性自然冲向 1**。这就是「一起摔」的数学本质。

```python
import numpy as np

NAMES = ["股票","商品","REITs","新兴市场","国债(对冲)"]
sigma_idio = np.array([0.14, 0.16, 0.13, 0.20, 0.05])  # 年化特异波动
beta       = np.array([1.00, 0.85, 0.90, 1.10, 0.05])  # 前4风险资产正, 国债≈0
mu_ann     = np.array([0.08, 0.05, 0.06, 0.09, 0.025])
sigma_F_calm, sigma_F_crisis = 0.05, 0.28

T = 252*10
rng = np.random.default_rng(20261007)
# 构造 regime：危机块插入
regime = np.zeros(T, dtype=int); pos = 0
while pos < T:
    if rng.random() < 0.12 and pos < T-40:
        L = int(rng.integers(20,40)); regime[pos:pos+L]=1
        pos += L + int(rng.integers(30,90))
    else: pos += 1
F = np.where(regime==0, rng.normal(0, sigma_F_calm/np.sqrt(252), T),
                     rng.normal(0, sigma_F_crisis/np.sqrt(252), T))
eps = rng.normal(0,1,(5,T)) * (sigma_idio/np.sqrt(252))[:,None]
R = beta[:,None]*F + eps + (mu_ann/252)[:,None]
```

注意国债的 $\beta=0.05$：它几乎不暴露在共同因子上，所以无论因子怎么炸，它的收益都来自自己的特异波动——**危机里它依然独立**。这就是后面能对冲的关键。

## 二、相关矩阵：从「花花绿绿」到「全红」

把平静块和危机块的样本相关矩阵画出来（图 1），差别触目惊心：

- 平静期：风险资产间相关性低且分化（有些微正、有些近 0）。
- 危机期：「股票—商品—REITs—新兴市场」四个格子全红，相关性冲到 0.6~0.8；唯独它们和「国债」那一行仍是蓝（负相关/独立）。

```python
calm = regime == 0; crisis = regime == 1
corr_calm   = np.corrcoef(R[:, calm])
corr_crisis = np.corrcoef(R[:, crisis])
risky = [0,1,2,3]
off = ~np.eye(4, dtype=bool)
avg_risky_calm   = corr_calm[np.ix_(risky,risky)][off].mean()
avg_risky_crisis = corr_crisis[np.ix_(risky,risky)][off].mean()
print(f"风险资产相关性: 平静={avg_risky_calm:.2f} 危机={avg_risky_crisis:.2f}")
```

算出来：**平静 0.07 → 危机 0.74**。这就是「一起摔」的量化证据。

## 三、分散比率：分散化红利蒸发了多少

相关冲 1，对组合意味着什么？用 **分散比率**（Choueifaty & Coignard 2008）：

$$DR = \frac{\sum_i w_i \sigma_i}{\sigma_p}$$

- $DR > 1$：分散有效（组合波动低于加权平均资产波动，有红利）。
- $DR → 1$：分散失效（组合波动≈加权平均，等于把资产等权堆在一起）。

```python
w5 = np.ones(5)/5
w4 = np.array([0.25,0.25,0.25,0.25,0.0])   # 风险4资产
def port_vol(cov, w): return np.sqrt(w @ cov @ w)
cov_c, cov_k = np.cov(R[:,calm]), np.cov(R[:,crisis])
sig_c = np.sqrt(np.diag(cov_c))*np.sqrt(252)
sig_k = np.sqrt(np.diag(cov_k))*np.sqrt(252)
pv5_c, pv5_k = port_vol(cov_c,w5)*np.sqrt(252), port_vol(cov_k,w5)*np.sqrt(252)
dr5_c = (w5*sig_c).sum()/pv5_c; dr5_k = (w5*sig_k).sum()/pv5_k
print(f"含国债5资产 组合波动: 平静={pv5_c:.1%} 危机={pv5_k:.1%}")
print(f"分散比率: 平静={dr5_c:.2f} → 危机={dr5_k:.2f}")
```

含国债 5 资产组合：波动从平静期 7.4% 翻到危机期 22.9%；分散比率从 **1.91 塌到 1.14**。风险 4 资产更惨：分散比率 **1.79 → 1.11**——几乎没有任何分散红利了。

![组合波动翻数倍 + 分散比率塌缩（风险4资产最惨）](/images/cross-asset-correlation-crisis/diversification.png)

**关键认知**：相关性危机不是「波动变大」那么简单。波动变大你早有预期；真正致命的是**相关结构塌缩**——它让组合波动近似等于 $\sum w_i\sigma_i$，也就是「分散化」这一项直接归零。你为分散付的所有配置努力，在危机里蒸发。

## 四、累计净值：危机段齐跌，国债独自撑住

把等权 5 资产（含国债）和等权 4 风险资产的累计净值画出来（图 3，红阴影是危机段）：

![危机段风险资产齐跌，国债独自撑住净值](/images/cross-asset-correlation-crisis/equity_curve.png)

肉眼可见：危机段里，红色的风险 4 资产净值断崖式齐跌；而含国债的蓝线，跌幅明显更浅——因为国债那块在危机里不跟跌（甚至微微涨），成了组合里唯一的减震器。这正呼应第一节的 β≈0 设计。

## 五、蒙特卡洛：危机相关下的真实回撤分布

单条 10 年路径有运气成分。我们做 2500 条蒙特卡洛路径（每条独立随机 regime + 因子路径），对比三类组合的**年度最大回撤**分布：

```python
def sim_dd(seed, weights):
    rg = np.random.default_rng(seed)
    reg = np.zeros(252, dtype=int); p = 0
    while p < 252:
        if rg.random() < 0.14 and p < 227:
            L = int(rg.integers(20,35)); reg[p:p+L]=1
            p += L + int(rg.integers(30,80))
        else: p += 1
    Fs = np.where(reg==0, rg.normal(0,0.05/np.sqrt(252),252),
                         rg.normal(0,0.28/np.sqrt(252),252))
    Rs = beta[:,None]*Fs + rg.normal(0,1,(5,252))*(sigma_idio/np.sqrt(252))[:,None] \
         + (mu_ann/252)[:,None]
    eq = np.cumprod(1 + (Rs*weights[:,None]).sum(axis=0))
    peak = np.maximum.accumulate(eq)
    return ((eq-peak)/peak).min()

dd4   = np.array([sim_dd(5000+m, w4)   for m in range(2500)])
dd5   = np.array([sim_dd(8000+m, w5)   for m in range(2500)])
dd6040= np.array([sim_dd(11000+m, np.array([0.6,0,0,0,0.4])) for m in range(2500)])
print(f"风险4资产 P95={np.percentile(dd4,95):.0%} 均值={dd4.mean():.0%}")
print(f"含国债5资产 P95={np.percentile(dd5,95):.0%} 均值={dd5.mean():.0%}")
print(f"60/40 P95={np.percentile(dd6040,95):.0%} 均值={dd6040.mean():.0%}")
```

![三类组合年度最大回撤分布：风险4资产最惨，含国债/60-40 更浅](/images/cross-asset-correlation-crisis/drawdown_dist.png)

结果：

| 组合 | P95 最大回撤 | 平均回撤 |
|---|---|---|
| 风险 4 资产等权 | **−8%** | −17% |
| 含国债 5 资产等权 | −7% | −14% |
| 经典 60/40 | −6% | −13% |

**含国债和 60/40 的回撤明显更浅**——因为它们都含「β 独立的国债」。风险 4 资产因为全是正 β、危机里全绑在一起，回撤最深。这把「分散化依赖独立资产」从口号变成了数字。

## 六、滚动分散比率：危机来临时骤降

把 10 年切成滚动 60 日窗口，逐窗算分散比率（图 5），会看到一条清晰的规律：**平静期分散比率高（红利在），危机期骤降到 1 附近（红利蒸发），危机过后慢慢恢复**。

![滚动分散比率：危机来临时骤降，分散红利蒸发](/images/cross-asset-correlation-crisis/rolling_div.png)

这给实战一个直接启示：分散比率本身是个**危机领先指标**。当它开始塌，说明共同因子在主导市场，该减风险敞口或加独立对冲资产了——而不是等净值已经跌了才反应。

## 七、落地要点与诚实边界

1. **相关性危机 = 相关结构塌缩，不是单纯波动变大**：波动变大你早定价了，塌缩让分散化归零才是杀手。看组合风险别只看 σ，要看 $\Sigma$ 的 off-diagonal。
2. **只有「β 真正独立」的资产能对冲**：本文国债 β≈0.05 所以有用。但真实里「避险资产」会漂移——极端危机时连黄金、国债都可能短暂同向（流动性挤兑下什么都卖）。对冲资产要动态校验，不能假设它永远独立。
3. **分散化红利会随 regime 漂移**：平静期 DR≈1.9 看着美，危机期→1.1。配置时用「危机相关矩阵」而不是「全样本相关矩阵」去测压力，否则严重低估尾部风险。
4. **本文是单因子合成演示**：真实跨资产有多元因子（增长、通胀、流动性、美元），相关冲 1 的速度/幅度取决于危机类型（2008 信用、2020 流动性、2022 利率各有不同）。但「共同因子主导 → 相关冲 1 → 分散失效」的机制是普适的。
5. **应对工具**：危机前降杠杆、加真正低 β 资产（国债/现金）、用相关性掉期或尾部对冲（看跌价差、VIX 类）；更重要的是**用滚动分散比率/相关冲击指数做预警**，别等回撤发生了才动。
6. **不要因此放弃分散化**：本文结论是「分散化在危机时打折」，不是「分散化无用」。含国债组合和 60/40 的回撤确实更浅——只是你要对「打折幅度」有清醒预期，并为它配置独立的对冲资产。

## 完整可复现代码

```python
import numpy as np

NAMES = ["股票","商品","REITs","新兴市场","国债(对冲)"]
sigma_idio = np.array([0.14,0.16,0.13,0.20,0.05])
beta       = np.array([1.00,0.85,0.90,1.10,0.05])
mu_ann     = np.array([0.08,0.05,0.06,0.09,0.025])
sigma_F_calm, sigma_F_crisis = 0.05, 0.28

# 生成 10 年（含危机块）
T = 252*10; rng = np.random.default_rng(20261007)
regime = np.zeros(T, dtype=int); pos = 0
while pos < T:
    if rng.random() < 0.12 and pos < T-40:
        L = int(rng.integers(20,40)); regime[pos:pos+L]=1
        pos += L + int(rng.integers(30,90))
    else: pos += 1
F = np.where(regime==0, rng.normal(0,sigma_F_calm/np.sqrt(252),T),
                     rng.normal(0,sigma_F_crisis/np.sqrt(252),T))
R = beta[:,None]*F + rng.normal(0,1,(5,T))*(sigma_idio/np.sqrt(252))[:,None] \
    + (mu_ann/252)[:,None]

calm, crisis = regime==0, regime==1
corr_c = np.corrcoef(R[:,calm]); corr_k = np.corrcoef(R[:,crisis])
risky = [0,1,2,3]; off = ~np.eye(4, dtype=bool)
print(f"风险资产相关性 平静={corr_c[np.ix_(risky,risky)][off].mean():.2f} "
      f"危机={corr_k[np.ix_(risky,risky)][off].mean():.2f}")

w5 = np.ones(5)/5
def dr(cov, w):
    pv = np.sqrt(w@cov@w)*np.sqrt(252)
    return (w*np.sqrt(np.diag(cov))*np.sqrt(252)).sum()/pv
print(f"含国债5资产 分散比率 平静={dr(np.cov(R[:,calm]),w5):.2f} "
      f"危机={dr(np.cov(R[:,crisis]),w5):.2f}")
```
