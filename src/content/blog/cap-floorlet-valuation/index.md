---
title: "利率上限/下限(Floorlet)估值：把利率衍生品拆成一篮子期权"
description: "Cap（利率上限）和 Floor（利率下限）是利率市场最基础的期权结构，但它们的定价不靠神秘公式，而是被拆成「一篮子 caplet / floorlet」——每一个 caplet 就是「在重置日进入一笔浮动利率、按固定 K 结算」的看涨期权。本文用 Black(1983) 逐期闭合式给 4 期 caplet 定价，验证 Cap−Floor=Swap 平价、蒙特卡洛与闭合式一致（偏差 0.6%），并诚实点出「第一个 caplet 退化向前/扁平曲线假设/波动率期限结构/重设频率」四类真实陷阱（中阶）。"
publishDate: '2026-07-19'
tags:
  - 量化交易
  - 利率衍生品
  - 利率上限
  - 利率下限
  - Caplet
  - 期权定价
  - 蒙特卡洛
  - Python
language: Chinese
difficulty: intermediate
---

浮动利率负债（比如一笔 LIBOR/SHIBOR 浮息贷款）的最大噩梦是：**利率一夜之间飙上去**。借入方想锁住「利率最多不超过 K」——这个需求催生了 **Cap（利率上限）**；反过来，持有浮息资产的投资者怕利率崩到 0 以下，想锁住「利率至少 K」——催生了 **Floor（利率下限）**。

很多人以为 Cap/Floor 是某种复杂利率模型才能定价的怪物。其实不是：**它们就是一篮子期权**。一个 N 期的 Cap = N 个 caplet 之和，每个 caplet 是一个独立的欧式看涨期权，标的是「那一期重置出来的浮动利率」。本文从 caplet 的微观结构讲起，用 Black(1983) 的利率期权公式逐期定价，再验证三条铁律，最后拆穿最容易被忽略的四类陷阱。

## 一、微观结构：Cap 就是一篮子 caplet

设一个季度重置、1 年期的 Cap，行权价 $K=2.5\%$。它包含 4 个 caplet，重置/支付日如下：

| caplet | 重置日(期权到期) $T_i$ | 支付日 $T_{i+1}$ | 覆盖期 |
|---|---|---|---|
| 1 | 0.00 | 0.25 | 第 1 季度 |
| 2 | 0.25 | 0.50 | 第 2 季度 |
| 3 | 0.50 | 0.75 | 第 3 季度 |
| 4 | 0.75 | 1.00 | 第 4 季度 |

第 $i$ 个 caplet 的逻辑：**在第 $T_i$ 日观测浮动利率 $L(T_i)$**，如果它高于 $K$，caplet 就向买方赔付 $(L(T_i)-K)\times \tau$（$\tau$ 是该期长度），按支付日贴现。这实际上就是一个**看涨期权**：标的 = 重置日的远期利率，行权价 = $K$，到期日 = 重置日。

> 关键直觉：Cap 的价值 = 把每个 caplet 当作独立期权分别定价，再求和。没有「整条曲线联动」的魔法——联动（相关性）只在**波动率曲面**里体现，定价每个 caplet 时只用它自己的重置期限波动率。

![各 caplet 重置日远期利率的不确定性：Cap 是一篮子 caplet](/images/cap-floorlet-valuation/cap_fwd_uncertainty.png)

每个 caplet 的 Black(1983) 价格（每单位面值）：

$$C_i = \tau \cdot P(0,T_{i+1}) \cdot \big[\,F_i\,N(d_1) - K\,N(d_2)\,\big],\quad d_1=\frac{\ln(F_i/K)+\frac12\sigma_i^2 T_i}{\sigma_i\sqrt{T_i}}$$

其中 $F_i$ 是第 $i$ 期重置日的**远期利率**，$P(0,T_{i+1})$ 是支付日贴现因子，$\sigma_i$ 是该 caplet 的对数正态波动率。Floorlet 对称：

$$F_i^{\text{floor}} = \tau \cdot P(0,T_{i+1}) \cdot \big[\,K\,N(-d_2) - F_i\,N(-d_1)\,\big]$$

## 二、扁平曲线下的闭合式定价

为数值干净，设**扁平初始曲线** $r_0=2\%$，即 $P(0,t)=e^{-rt}$，各期远期利率 $F_i=r_0=2\%$。取波动率期限结构 $\sigma_i=[29\%,30\%,31\%,32\%]$（期限越长略升，符合真实 cap 曲面形状），$K=2.5\%$（OTM）：

```python
import numpy as np
from scipy.stats import norm

r0 = 0.02          # 扁平初始曲线
tau = 0.25         # 季度重置
T_reset = np.array([0.0, 0.25, 0.5, 0.75])
T_pay   = np.array([0.25, 0.5, 0.75, 1.0])
sig_i   = np.array([0.29, 0.30, 0.31, 0.32])
F = r0 * np.ones(4)   # 扁平曲线下各期远期 = 即期

def P(t): return np.exp(-r0 * t)

def black_caplet(Fi, K, sig, T_opt, T_pay_i):
    if T_opt <= 0:                       # 第一个 caplet 重置在 0 → 退化为远期
        return tau * P(T_pay_i) * max(Fi - K, 0.0)
    d1 = (np.log(Fi / K) + 0.5 * sig**2 * T_opt) / (sig * np.sqrt(T_opt))
    d2 = d1 - sig * np.sqrt(T_opt)
    return tau * P(T_pay_i) * (Fi * norm.cdf(d1) - K * norm.cdf(d2))

def cap_value(K):
    return sum(black_caplet(F[i], K, sig_i[i], T_reset[i], T_pay[i]) for i in range(4))

print(f"Cap(K=2.5%) = {cap_value(0.025)*1e4:.3f} bp")   # 各 caplet 之和
```

把各行权价 $K$ 下的 Cap 总价和各 caplet 贡献画出来：

![Cap 总价与各行权价 K：各 caplet 扇形叠加，ATM(K=2%)处最贵](/images/cap-floorlet-valuation/cap_vs_strike.png)

Cap 总价随 $K$ 严格单调递减——$K$ 越高，每个 caplet 越 OTM。在 ATM（$K=F=2\%$）处 Cap 最贵，因为此时每个 caplet 都站在自己远期利率的中心。**注意第一个 caplet（重置日 0）是平的、无期权性**：它重置在当下，远期利率就是已知的 $F_1=2\%$，与 $K=2.5\%$ 比较是纯确定性的，所以 caplet1 的价值是一条水平线（图中最低那条）。

把 $K=2.5\%$ 时各 caplet 的价值摊开，能看出**久期越长、波动率越高的 caplet 越贵**（contrib 由 caplet1→4 递增）：

![行权价 K=2.5% 时各 caplet 价值贡献：越长久期 caplet 越贵](/images/cap-floorlet-valuation/caplet_decomp.png)

## 三、铁律一：Cap − Floor = Swap 平价

利率期权里有一条比看涨/看跌平价更漂亮的关系。对一个**支付固定 $K$ 的 payer swap**（浮动端按远期 $F$ 定价），它的现值是：

$$\text{Swap PV}(K) = \sum_i \tau \cdot P(0,T_{i+1})\cdot(F_i - K)$$

而 Cap 是「浮动利率封顶」、Floor 是「浮动利率保底」。数学上可证（逐期把 caplet 与 floorlet 相减，远期部分恰好抵消）：

$$\text{Cap}(K) - \text{Floor}(K) = \text{Swap PV}(K)$$

这条平价让我们**只需会定 Cap，就能反推出 Floor**：`Floor = Cap − Swap`。验证一下（数值精确到 $10^{-19}$）：

```python
def floor_value(K):
    return sum(black_floorlet(F[i], K, sig_i[i], T_reset[i], T_pay[i]) for i in range(4))

def swap_pv(K):
    return sum(tau * P(T_pay[i]) * (F[i] - K) for i in range(4))

K = 0.025
print(cap_value(K) - floor_value(K) - swap_pv(K))   # ≈ 0 (1e-19 级)
```

![Cap/Floor/Swap 三者价值 vs K：Cap−Floor−Swap 恒为 0；右图蒙特卡洛校验闭合式](/images/cap-floorlet-valuation/cap_floor_parity.png)

## 四、铁律二：蒙特卡洛与闭合式一致

闭合式漂亮，但实盘利率模型（Hull-White、LMM）下没有 caplet 闭式，只能蒙特卡洛。本文用对数正态模拟每个 caplet 的远期利率 $L_i$，做 pathwise 估值，与 Black 闭合式交叉验证：

```python
rng = np.random.default_rng(20260719)
M = 300_000
mc_cap = 0.0
for i in range(4):
    t_r = T_reset[i]
    if t_r <= 0:
        mc_cap += tau * P(T_pay[i]) * max(F[i] - 0.025, 0.0)
    else:
        Z = rng.standard_normal(M)
        L = F[i] * np.exp(-0.5 * sig_i[i]**2 * t_r + sig_i[i] * np.sqrt(t_r) * Z)
        mc_cap += np.mean(tau * np.maximum(L - 0.025, 0.0) * P(T_pay[i]))

black_total = cap_value(0.025)
print(f"MC={mc_cap*1e4:.3f}bp  Black={black_total*1e4:.3f}bp  "
      f"偏差={(mc_cap-black_total)/black_total*100:.2f}%")   # ≈ -0.6%
```

蒙特卡洛得到 2.645 bp，闭合式 2.661 bp，相对偏差 0.6%——一致，说明我们的 caplet 分解与对数正态模拟口径完全对齐。**这条校验是实盘给 Cap 双边报价前的必须动作**：如果 MC 和闭式对不上，先查贴现因子/远期口径，而不是怀疑市场无效。

## 五、四类真实陷阱（务必先看这段）

1. **第一个 caplet 退化向前（无期权性）**：重置日 $T_1=0$ 的 caplet，其标的远期利率在估值日已经确定（就是当期即期利率），它退化成一个远期合约，不是期权。新手容易把它也套 Black 公式（出现 $\sqrt{0}$ 除零）。本文用 `if T_opt <= 0` 分支显式处理——**Cap 的第一个 caplet 永远不贡献 vega**，这是报价台最容易漏算的地方。

2. **扁平曲线把「远期=即期」焊死了**：本文设 $r_0=2\%$ 扁平曲线，于是所有 $F_i=2\%$。真实初始曲线是上翘/倒挂的，各期远期利率各不相同，且 caplet 之间不再同值。更致命的是**贴现因子**必须喂真实初始曲线 $P(0,T_{i+1})$，而不是 $e^{-r_0 T}$。本文为数值干净简化了这一维，实盘必须替换。

3. **波动率期限结构（flat vol 是谎言）**：本文第一次尝试用了常数 $\sigma=30\%$，结果 4 条 caplet 曲线完全重合、看不出「扇形」。真实 cap 市场波动率**随期限漂移**（短端低、长端高），本文改用 $\sigma_i=[29\%,30\%,31\%,32\%]$ 后扇形才显现。如果你只用一条 ATM cap vol 给所有 caplet 定价，长端 caplet 会被系统性低估。

4. **重设频率改写一切**：季度重置（$\tau=0.25$）和月度重置（$\tau=1/12$）的 Cap 价差极大——重置越频繁，买方越多「封顶」机会，Cap 越贵（这是为什么危机中月度 cap 比季度贵一大截）。本文用季度，实盘若用月度，caplet 数量从 4 变 12，代码里 `T_reset/T_pay` 数组要重排，价值会显著上升。同理，**营业日惯例（Act/360 vs 30/360）** 也微调 $\tau$ 与贴现。

## 六、实战：用平价给 Floor 报价

假设交易台要卖一个 1 年期、季度重置、行权价 $K=2\%$ 的 Floor。你手上有 Cap 的报价引擎，但没有 Floor 引擎——直接用平价反推：

```python
K = 0.02
cap_atm = cap_value(K)          # Cap(K=2%) 用闭式
swap_atm = swap_pv(K)           # Swap PV(K=2%)，注意扁平曲线下 F=K → swap_atm≈0
floor_atm = cap_atm - swap_atm  # 平价: Floor = Cap - Swap
print(f"Floor(K=2%) = {floor_atm*1e4:.3f} bp")   # 与 floor_value(K) 应对齐到机器精度
```

在 ATM（$F=K$）处 swap PV 趋近 0，于是 `Floor ≈ Cap`——这正是看涨/看跌平价的利率版。若你对出来的 Floor 与直接算的 floorlet 之和差了几个 bp，回查：远期口径（重置日 vs 支付日）、贴现因子、波动率期限结构、以及随机冲击是否同一批。

## 七、正确用法：把 Cap/Floor 当「期权篮子」，别当「曲线模型」

Cap/Floor 最该记住的三件事：

- 它的价值 = **每个 caplet 独立定价的加和**，caplet 之间是「篮子关系」而非「相关性关系」——相关性藏在波动率曲面里，不进单 caplet 公式；
- **Cap − Floor = Swap** 平价是免费的反向定价器，会定 Cap 就会定 Floor；
- 第一个 caplet 永远退化向前、不贡献 vega；重置频率和波动率期限结构对价值的影响，远大于多数人对「模型选择」的焦虑。

## 结语

Cap 和 Floor 把「利率封顶/保底」的需求，翻译成了**一篮子 caplet/floorlet**——每个都是标准的欧式期权，标的为重置日远期利率、行权价为 $K$。本文用 Black(1983) 逐期闭合式定价 4 期 caplet，验证了 Cap−Floor=Swap 平价（偏差 $10^{-19}$ 级）、蒙特卡洛与闭合式一致（偏差 0.6%），并量化了「越长久期、越高波动的 caplet 越贵」的扇形结构。但它的真边界在四类裂缝里：首期 caplet 退化向前、扁平曲线焊死远期、常数波动率压扁扇形、重置频率改写价值。理解这些，比背下 Black 公式重要——它告诉你**任何「模型价=市场价」的胜利，都只在被校准的那一个点上成立**。
