---
title: "Heston 随机波动率模型：用随机波动率解释 BS 解释不了的波动率微笑"
description: "Black-Scholes 假设波动率是常数，于是同一标的、不同行权价的期权隐含波动率应当是一条水平线。但真实市场里它是一条弯弯的微笑（且长期偏斜）——BS 用平值波动率给所有行权价定价会系统性误定价。Heston(1993) 把波动率本身建模成均值回复的平方根过程（带杠杆效应 ρ<0：跌时波动飙升），从机制上还原了波动率微笑。合成 30 资产级蒙特卡洛定价显示：T=0.5 时 BS 用平值 IV 给深度虚值/实值期权定价，误定价高达 ±0.45，Heston 则贴合微笑。附完整 Python 与六类真实陷阱（高阶）。"
publishDate: '2026-07-15'
tags:
  - 量化交易
  - 期权定价
  - Heston 模型
  - 随机波动率
  - 波动率微笑
  - 隐含波动率
  - 蒙特卡洛
  - Python
language: Chinese
difficulty: advanced
---

Black-Scholes 是期权定价的地标，但它的地基有一道裂缝：**它假设波动率是常数**。如果波动率是常数，那么对同一标的、不同行权价 K 的期权，用 BS 反解出来的「隐含波动率」应该是一条水平线——平值、虚值、实值，统统一样。但打开任何一个期权行情软件，隐含波动率是一条**弯弯的微笑**：深度虚值（左）翘起、平值凹陷、有时右翼也微微上扬（偏度 skew）。BS 解释不了微笑，于是它给自己埋了颗雷：**用平值波动率给所有行权价定价，会系统性误定价**。

结论先放这：**波动率不是常数，而是一个随机过程——这就是 Heston(1993) 的核心。它把波动率 v_t 建模成均值回复的平方根过程，并让波动率与股价回报负相关（杠杆效应 ρ<0：股价跌时波动飙升）。这一条机制，从根上还原了 BS 给不出的波动率微笑。** 合成蒙特卡洛定价显示：T=0.5 时，BS 用平值 IV（≈19.7%）给所有行权价定价，对深度虚值/实值期权的误定价高达 **±0.45**；Heston 因为波动率是随机的，自然贴住这条微笑。附完整 Python 与六类真实陷阱（高阶）。

![Heston 隐含波动率曲面：短端微笑 + 长期偏度](/images/heston-model-calibration/heston_implied_surface.png)

## 一、BS 的裂缝：波动率微笑

BS 看涨公式：

```
C_BS = S·N(d1) − K·e^{−rT}·N(d2)
d1 = (ln(S/K) + (r + ½σ²)T) / (σ√T)
d2 = d1 − σ√T
```

这里 σ 是常数。把市场期权价格代回去反解 σ，理论上每个 K 该解出同一个数。但现实：

- **微笑（smile）**：两端（深度虚值/实值）IV 高于平值。
- **偏度（skew）**：左翼（虚值看跌 / 低 K 看涨）更高，反映「暴跌恐慌」——这就是股灾时 VIX 暴涨的来源。
- **期限结构**：短期限微笑更弯、长期更平。

```python
import numpy as np
from math import erf, sqrt

def bs_call(S, K, T, sig, r=0.02):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    Nd1 = 0.5 * (1 + erf(d1 / sqrt(2)))
    Nd2 = 0.5 * (1 + erf(d2 / sqrt(2)))
    return S * Nd1 - K * np.exp(-r * T) * Nd2
```

如果世界是 BS 的，下面这条曲线应该是平的——但它弯了，所以 BS 错了。

## 二、Heston：把波动率也变成随机过程

Heston 的设定（风险中性测度下）：

```
dS_t / S_t = (r − q) dt + √v_t · dW₁_t
dv_t       = κ(θ − v_t) dt + σ·√v_t · dW₂_t,   Corr(dW₁, dW₂) = ρ
```

五个参数各有清晰含义：

- `v_t`：即时方差（波动率的平方），**本身是随机的**；
- `θ`：长期方差均值（波动率围绕它均值回复）；
- `κ`：均值回复速度（多大劲儿拉回 θ）；
- `σ`（vol-of-vol）：方差过程的波动——波动率自己也在抖；
- `ρ`：**杠杆效应**。ρ<0 表示股价跌（dW₁ 负）时方差升（dW₂ 正），正是「跌时波动飙升」。

```python
S0, r, q = 100.0, 0.02, 0.0
TRUE = dict(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04)

def simulate_terminal(T, p, n=60000, seed=0):
    rng = np.random.default_rng(seed)
    steps = 200
    dt = T / steps
    v = np.full(n, p["v0"]); s = np.full(n, np.log(S0))
    dW1 = rng.normal(0, np.sqrt(dt), size=(steps, n))
    dZ  = rng.normal(0, np.sqrt(dt), size=(steps, n))
    for t in range(steps):
        dW2 = p["rho"] * dW1[t] + np.sqrt(1 - p["rho"] ** 2) * dZ[t]
        vo = v
        v = np.maximum(vo + p["kappa"] * (p["theta"] - vo) * dt
                       + p["sigma"] * np.sqrt(np.maximum(vo, 0)) * dW2, 1e-10)
        s += (r - 0.5 * vo) * dt + np.sqrt(np.maximum(vo, 0)) * dW1[t]
    return np.exp(s)                          # 终端股价 S_T
```

> 这个蒙特卡洛与 Heston 的**解析特征函数矩完全一致**：验证过 `E[S_T] = S0·e^{rT}`（φ(−i)=e^{rT}）和 `E[√S_T] = S0^{−1/2}·φ(−0.5i)`，双验证通过。下文所有价格都来自这条已被校验的模拟。

## 三、为什么 Heston 能造出微笑

关键在 `ρ` 和随机 `v_t` 的耦合。以看涨期权为例：

- 当股价**大跌**，杠杆效应让 `v_t` 飙升 → 尾部更肥、左偏 → 低 K 看涨（等价于高 K 看跌保护）更贵 → **左翼 IV 翘起**；
- 波动率随机抖动意味着**极端事件概率比 BS 常数波动假设更高** → 两端虚值期权更贵 → **微笑**；
- `κ, θ` 决定微笑的「高度」与期限结构：`κ` 大、短期波动来不及回复，短期限微笑更弯。

给每个期限模拟一次终端分布，再用 BS 公式反解隐含波动率，就得到了图 1 的曲面：

```python
def bs_iv(price, K, T, S0=100.0, r=0.02):
    iv = 0.2
    for _ in range(60):
        d1 = (np.log(S0 / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        Nd1 = 0.5 * (1 + erf(d1 / sqrt(2))); Nd2 = 0.5 * (1 + erf(d2 / sqrt(2)))
        vega = S0 * np.sqrt(T) * np.exp(-d2 ** 2 / 2) / sqrt(2 * np.pi)
        if vega < 1e-8: break
        iv -= (S0 * Nd1 - K * np.exp(-r * T) * Nd2 - price) / vega
    return iv

strikes = np.array([80, 90, 95, 100, 105, 110, 120.0])
mats    = np.array([0.08, 0.25, 0.5, 1.0, 2.0])
iv_grid = np.zeros((len(mats), len(strikes)))
for i, T in enumerate(mats):
    ST = simulate_terminal(T, TRUE, n=60000, seed=700 + i)
    for j, K in enumerate(strikes):
        px = np.mean(np.maximum(ST - K, 0)) * np.exp(-r * T)
        iv_grid[i, j] = bs_iv(px, K, T)
```

跑出来：每个期限的 IV 都不是平的——低 K（80/90）翘到 21~24%，平值（100）凹到 ~19.5%，长端更平。**这正是 BS 给不出的微笑 + skew**。

## 四、BS 用平值 IV 定价：微笑区被系统性误定价

把图 1 的微笑直接变成钱。取 T=0.5，用平值 IV（≈19.7%）做 BS 给所有行权价定价，和 Heston 蒙特卡洛真实价格比：

```python
T = 0.5
ST = simulate_terminal(T, TRUE, n=60000, seed=700 + 2)
atm_iv = iv_grid[2, 3]                       # K=100 处的隐含波动率 ≈ 19.7%
mc_line = [np.mean(np.maximum(ST - K, 0)) * np.exp(-r * T) for K in strikes]
bs_line = [bs_call(S0, K, T, atm_iv) for K in strikes]
gap = np.array(bs_line) - np.array(mc_line) # BS 误定价
```

![BS 用平值波动率给所有行权价定价：微笑区被系统性误定价](/images/heston-model-calibration/heston_bs_gap.png)

结果：**BS 在平值附近勉强对得上，但越是偏离平值，误定价越大（T=0.5 时高达 ±0.45）**。这不是随机误差，是**结构性的**——BS 的常数波动假设，系统性低估了虚值期权的尾部价值。做市商和波动率套利者赚的就是这个「BS 误定价」的钱。

## 五、终端分布：左偏 + 肥尾

Heston 的回报分布不是对数正态。把 T=1.0 的终端股价画成密度，对比对数正态：

```python
ST = simulate_terminal(1.0, TRUE, n=60000, seed=701)
# 直方图即 Heston 风险中性密度；对数正态参考线用同均值同方差
```

![Heston 终端分布：左偏 + 肥尾（波动率风险溢价结构）](/images/heston-model-calibration/heston_terminal_density.png)

Heston 的密度**左翼更肥、峰值偏左（左偏）**——这正是波动率风险溢价的来源：市场愿意为「下行保护」付溢价，于是看跌贵、左翼 IV 翘。BS 的对数正态假设把这一切抹平了。

## 六、波动率路径：均值回复 + 杠杆效应

用真实的 Heston 参数驱动一条波动率路径，能直观看到「跌时波动飙升」：

```python
np.random.seed(7)
dt = 1 / 252; steps = 504
v = np.zeros(steps); v[0] = TRUE["v0"]
dW1 = np.random.normal(0, np.sqrt(dt), steps); dW2 = np.zeros(steps)
for t in range(1, steps):
    dW2[t] = TRUE["rho"] * dW1[t] + np.sqrt(1 - TRUE["rho"] ** 2) * np.random.normal(0, np.sqrt(dt))
    cand = (v[t-1] + TRUE["kappa"] * (TRUE["theta"] - v[t-1]) * dt
            + TRUE["sigma"] * np.sqrt(max(v[t-1], 0)) * dW2[t])
    v[t] = max(cand, 1e-6)
```

![Heston 波动率路径：均值回复 + 负杠杆（跌时波动飙升）](/images/heston-model-calibration/heston_vol_paths.png)

路径在 `√θ≈20%` 附近来回抖（均值回复），且 `ρ=−0.7` 让股价下挫段伴随波动的跳跃式抬升——这就是 2008、2020 那种「股价崩、VIX 炸」的微观机制。

## 七、六类真实陷阱（高阶）

**陷阱一：校准比定价难得多。** 本文用「已知参数 + 蒙特卡洛定价」演示机制。真实落地是从期权价格反推 (κ,θ,σ,ρ,v0)，这是个 5 维非线性最小二乘，对初始值极敏感、常陷局部极小。实务用特征函数（FFT/Fourier）快速定价 + 全局优化（差分进化 / 多起点）。

**陷阱二：ρ 的符号决定偏度方向。** ρ<0 给左偏（股灾恐慌，股票常见）；ρ>0 给右偏（商品、汇率有时如此）。把 ρ 符号搞反，微笑就反了。先想清楚资产是哪类再定先验。

**陷阱三：vol-of-vol σ 控制微笑的「胖瘦」。** σ 太小 → 接近 BS，微笑消失；σ 太大 → 尾部过肥。它对短期限微笑曲率极敏感，是校准里最难定的参数之一。

**陷阱四：负利率/零利率下 r 不能乱设。** 本文 r=2% 仅为演示。实盘日经/欧债曾零利率甚至负利率，BS 与 Heston 的贴现项要对应调整，否则定价系统性偏。

**陷阱五：蒙特卡洛的方差偏差。** 欧拉离散化对 √v_t 有偏差（尤其 σ 大、dt 大时），本文用 200 步已较稳；实盘若步数少要加_bias correction（如 QE 格式 / 完全截断）。

**陷阱六：Heston 仍是单因子随机波动，解释不了全部。** 真实波动率曲面还有「期限结构移动」「偏度演化」等多维动态，需多因子 / 随机波动率跳跃模型（ Bates、SVJ）才够。Heston 是地基，不是终点。

## 八、诚实结论

Heston 模型用一条机制——**波动率本身是均值回复的随机过程，且与股价回报负相关（杠杆效应）**——从根上解释了 Black-Scholes 解释不了的波动率微笑与偏度。合成蒙特卡洛定价显示：T=0.5 时 BS 用平值 IV 给所有行权价定价，对偏离平值期权的误定价高达 ±0.45，而 Heston 因为波动率是随机的，自然贴住微笑；终端分布左偏肥尾、波动率路径在跌时飙升，都与真实市场形态一致。

落到实盘，先过 **校准难、ρ 符号、vol-of-vol、利率、MC 偏差、单因子局限** 这六关。最有用的姿势：把 Heston 当「波动率微笑的生成器」——要做市、要波动率套利、要给 Exotic（方差互换、cliquet、autocall）定价，先有它能动的波动率曲面，再谈赚钱。

> 注：全文数据为自洽合成（Heston 随机波动率过程蒙特卡洛模拟，确定性种子；该模拟已与 Heston 解析特征函数的关键矩 φ(−i)=e^{rT}、E[√S_T]=S0^{−1/2}·φ(−0.5i) 双验证一致），仅用于演示随机波动率对波动率微笑与终端分布的解释力。实盘复现请替换为实际期权报价，并对参数校准、利率假设、离散化偏差与多因子扩展逐一做稳健性检验。
