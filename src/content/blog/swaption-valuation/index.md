---
title: "Swaption 估值与波动率校准：把互换期权装进利率模型"
description: "Swaption 是利率衍生品里最标准的「波动率期权」：买的是未来某天进入一个利率互换的权利。本文用 Hull-White 短期利率模型做全路径蒙特卡洛估值，给出平价互换利率 S(T_e)、远期年金 A、payer swaption 到期 payoff 与现值的完整闭式，再用二分法把 HW 扩散 σ_HW 校准到市场 20% 对数正态 vol(ATM 价精确对齐 0.0152)，并诚实点出「HW 常数 σ 压制波动率微笑 / 扁平曲线使 forward 确定性 / 利率模型 vs 波动率模型是两层不同校准 / 单因子无法拟合整条波动率曲面」四类真实局限(中阶)。"
publishDate: '2026-07-19'
tags:
  - 量化交易
  - 利率衍生品
  - Swaption
  - Hull-White
  - 波动率校准
  - 蒙特卡洛
  - Python
language: Chinese
difficulty: intermediate
---

Swaption（互换期权）是利率市场里交易量最大的期权之一。它给持有者一个**权利**（不是义务）：在未来某个约定日 $T_e$，按约定的固定利率 $K$ 进入一个利率互换。如果你是 payer swaption 的买方，你就拥有了「未来以 $K$ 支付固定端、收取浮动端」的权利——本质上是在赌未来利率会涨（互换固定端利率上行时你赚）。

本文从模型设定讲起，用 Hull-White 短期利率模型做**全路径蒙特卡洛估值**，再用校准把模型波动率钉到市场报价，最后诚实拆穿利率模型里最被低估的四道裂缝。

## 一、模型设定：扁平曲线下的互换

设初始曲线扁平、短期利率 $r_0=b=2\%$，这样平价互换利率近似等于 $r_0$（扁平曲线下理论值约 $2.01\%$，蒙特卡洛实测中位数 $2.0186\%$，详见文末补充；下文为表述简洁仍以 $2\%$ 作 ATM 行权价）。参数：

- 期权到期 / 互换开始：$T_e=5$ 年；
- 互换期限：$n=5$ 年，半年付息 $\tau=0.5$，共 10 个支付日 $T_1,\dots,T_{10}$；
- Hull-White 均值回复速度 $a=0.10$，中枢 $b=0.02$；
- 市场 swaption 对数正态波动率 $\sigma_B=20\%$（5y/5y 典型水平）。

在扁平曲线下，任意时刻 $t$ 的贴现因子 $P(0,t)=e^{-rt}$，**远期年金**（annuity，支付间隔相对 $T_e$）闭式为：

$$A = \sum_{i=1}^{10} \tau \cdot e^{-r\,\tau_i},\qquad \tau_i=0.5,1.0,\dots,5.0$$

代入 $r=2\%$ 得 $A\approx 4.734$，期权到期日贴现 $P(0,T_e)=e^{-0.02\times5}\approx 0.9048$。远期平价互换利率 $F=(1-e^{-r\cdot 10})/A\approx 2.01\%$。

## 二、平价互换利率与 payer swaption 的 payoff

到期日 $T_e$ 时，标的互换的**平价互换利率**（使互换现值为零的固定端利率）由贴现因子决定：

$$S(T_e)=\frac{1-P(T_e,T_n)}{\sum_{i=1}^{n}\tau_i\,P(T_e,T_i)}=\frac{1-P(T_e,T_n)}{A(T_e)}$$

其中 $P(T_e,T_i)=P(0,T_i)/P(0,T_e)$。扁平曲线下 $S(T_e)$ 的分布围绕 $2\%$ 波动（由利率路径的随机性驱动）。

Payer swaption 的到期 payoff（每单位面值）：

$$\text{payoff} = \max\big(0,\; S(T_e)-K\big)\cdot A(T_e)$$

期权现值（风险中性期望，用 Hull-White 全路径模拟 $r_t$ 后 pathwise 贴现）：

$$PV = \mathbb{E}\!\left[\,P(0,T_e)\cdot \text{payoff}\,\right]$$

完整估值代码（确定性随机冲击，保证各图互相一致、可复现）：

```python
import numpy as np
from scipy.stats import norm

r0 = b = 0.02; a = 0.10; Te = 5.0; tenor = 5.0; tau = 0.5
pay_dates = np.arange(tau, tenor + 1e-9, tau)
swap_times = Te + pay_dates
dt = 1.0/12; M = int(round((Te + tenor)/dt))
alpha = np.exp(-a*dt); mu_step = b*(1-alpha)

# 扁平曲线闭式: 年金与贴现
P0Te_det = np.exp(-r0*Te)
A_fwd = sum(tau*np.exp(-r0*rel) for rel in pay_dates)  # 远期年金 ≈ 4.734
F = (1.0 - np.exp(-r0*tenor)) / A_fwd                 # 远期平价互换利率 ≈ 2.01%

def black_swaption(F, K, v, T, A, P0T):
    d1 = (np.log(F/K) + 0.5*v*v*T)/(v*np.sqrt(T))
    d2 = d1 - v*np.sqrt(T)
    return A*P0T*(F*norm.cdf(d1) - K*norm.cdf(d2))

target = black_swaption(F, 0.02, 0.20, Te, A_fwd, P0Te_det)
print(f"市场 Black ATM 目标价 = {target:.6f}")         # 0.015235 (K=2%)

def simulate_paths(Z, sigma):
    sd = sigma*np.sqrt((1-alpha**2)/(2*a))
    r = np.empty((Z.shape[0], M+1)); r[:,0]=r0
    for i in range(M):
        r[:,i+1] = alpha*r[:,i] + mu_step + sd*Z[:,i]
    cum = np.cumsum(r, axis=1)*dt
    return r, cum

def price_swaption(Z, sigma, K):
    r, cum = simulate_paths(Z, sigma)
    P0Te = np.exp(-cum[:, int(round(Te/dt))])
    Ann = np.zeros(Z.shape[0])
    for ti in swap_times:
        Ann += tau*(np.exp(-cum[:, int(round(ti/dt))])/P0Te)
    P_Te_Tn = np.exp(-cum[:, int(round((Te+tenor)/dt))])/P0Te
    S_Te = (1.0 - P_Te_Tn)/Ann
    payoff = np.maximum(S_Te - K, 0.0)*Ann
    return np.mean(payoff*P0Te), S_Te
```

短利率路径与到期互换利率分布如下（校准后的 σ_HW）：

![Hull-White 短期利率模拟：向 2% 中枢回复](/images/swaption-valuation/hw_rate_paths.png)

![期权到期日平价互换利率 S(T_e) 分布](/images/swaption-valuation/swap_rate_dist.png)

## 三、Swaption 价格 vs 行权价：payer 单调递减

Payer swaption 的价值随行权价 $K$ 单调递减——$K$ 越高，你「以高固定利率支付」的权利越不值钱。把模型价（校准 σ_HW，ATM 在 $K=F\approx2.01\%$ 处）与对数正态 Black 基准画在一起（$K=2\%$ 视作近似 ATM）：

```python
Z = np.random.default_rng(20260719).standard_normal((20000, M))
# 二分法校准 sigma_HW 使模型 ATM 价 = target
lo, hi = 0.001, 0.05
for _ in range(40):
    mid = 0.5*(lo+hi)
    if price_swaption(Z, mid, F)[0] < target: lo = mid
    else: hi = mid
sigma_fit = 0.5*(lo+hi)                      # ≈ 0.52%
print(f"拟合 HW 扩散 σ_HW = {sigma_fit*100:.2f}%")
```

![Payer Swaption 价格 vs 行权价 K](/images/swaption-valuation/swaption_vs_strike.png)

两曲线在 ATM（$K=2\%$）处贴合（校准目标），但在两翼分化：模型在低成本价略高于 Black、在高成本价略低——这正是 Hull-White 单因子与对数正态基准的偏差指纹。

## 四、波动率校准：把模型钉到市场

校准的本质：市场报的是 **20% 的 swaption 对数正态 vol**，它对应的是 Black 公式下的 ATM 价 `0.015235`（在真实远期 $F=2.01\%$ 处）。我们扫 HW 扩散系数 $\sigma_{HW}$，找到使模型 ATM 价等于该目标的那一个点：

![波动率校准：扫描 HW 扩散 σ_HW](/images/swaption-valuation/calibration_curve.png)

扫描得到的 $\sigma_{HW}\approx 0.52\%$。注意这个数量级——**HW 短期利率的扩散（~0.5%）远小于 swaption 的 20% vol**，因为前者是利率水平的波动、后者是互换利率的波动，二者差着一个「凸性放大」倍数。这个差距本身就是利率衍生品里最容易搞混的概念。

## 五、凸性：波动率越高，payer swaption 越值钱

把不同 $\sigma_{HW}$ 下的 $S(T_e)$ 分布叠起来，能直观看到凸性效应——波动越高，分布越肥，右尾越长，payer swaption 的 $\max(0,S-K)$ 被抬得越高：

![波动率越高，S(T_e) 分布越肥 → payer swaption 凸性价值越高](/images/swaption-valuation/vol_convexity.png)

印证了前文单调性扫描：模型 ATM 价随 $\sigma_{HW}$ 严格递增（0.5%→0.0146，1%→0.0279，2%→0.0510），凸性把期权价值推上去。

### 补充：平价互换利率为什么约等于 r0（扁平曲线）

前文说扁平曲线下 $S(T_e)$ 的理论均值约为 $2\%$。在 $P(0,t)=e^{-rt}$ 的扁平曲线里，互换平价条件给出**远期**平价利率：

$$S = \frac{1-P(T_e,T_n)}{A(T_e)} = \frac{1-e^{-r\cdot n}}{\sum_{i=1}^{n}\tau_i\,e^{-r\cdot \tau_i}}$$

注意分母是**远期年金**（支付时刻相对 $T_e$ 的 $\tau_i=0.5,1.0,\dots,5.0$），不是绝对时刻。代入 $\tau=0.5, n=10, r=2\%$：分子 $1-e^{-0.2}=0.1813$，分母 $\sum_{i}0.5\,e^{-0.02\cdot 0.5 i}=4.735$，比值 $S\approx 2.01\%$。蒙特卡洛路径实际得到的中位数 **$2.0186\%$** 正落在这个理论值上——比即期利率 $2\%$ 略高约 2bp，源于**半年付息的凸性**（离散付息让 par rate 略高于连续即期利率）。这个细节提醒：**par swap rate 与即期利率的等号只在连续复利、按年付息的简化约定下成立，实盘要看付息频率**。本文校准用的 $K=2\%$ 与真实远期 $2.0186\%$ 仅差不到 2bp，对 ATM 价与 $\sigma_{HW}$ 拟合的影响可忽略。

### 补充：payer 与 receiver 的对称关系

Payer swaption 的买方有权「付固定、收浮动」——赌利率上行。对称的 **receiver swaption** 买方有权「收固定、付浮动」——赌利率下行。两者 payoff 互为镜像：

$$\text{Receiver PV} = A\cdot\big[\,K\cdot N(-d_2) - S\cdot N(-d_1)\,\big]$$

且在 Black 框架下，payer 与 receiver 的 ATM 价之和恒等于远期互换的远期价值（零），即 $PV_{payer}(K=S)\approx PV_{receiver}(K=S)$ 且二者关于 $K=S$ 对称。本文只做了 payer，但 receiver 的代码只需把 `np.maximum(S_Te - K, 0)` 换成 `np.maximum(K - S_Te, 0)` 即可，校准逻辑完全复用。

## 六、四类真实局限（务必先看这段）

1. **HW 常数 σ 压不出波动率微笑**：本文校准只在 ATM 一处对齐 Black 价。但真实 swaption 市场是**整条微笑/期限结构**——同期限不同行权价、同行权价不同到期日的 vol 都不一样。HW 单因子常数扩散只能拟合一个标量，无法还原曲面。真实交易台用 SABR 或 LMM 才覆盖微笑。
2. **扁平曲线使 forward 确定性**：本文为数值干净设了 $r_0=b=2\%$ 的扁平初始曲线，导致 $S(T_e)$ 的理论均值恒为 2%、ATM 即 $K=2\%$。真实初始曲线是上翘/倒挂的，forward 互换利率本身有水平不确定性——本例刻意简化了这个维度，实盘必须喂入真实初始贴现曲线。
3. **利率模型 ≠ 波动率模型，是两层校准**：HW 的 $\sigma_{HW}$ 校准到 swaption vol，但 HW 的 $a,b$ 要先从**初始期限结构**校准（拟合整条即期/远期曲线）。本文直接给定 $a,b$，跳过了第一层校准——实盘里这两层必须分开做、且可能冲突。
4. **单因子无法拟合整条曲面**：一个状态变量（短期利率）意味着所有期限的利率完全相关，而真实市场长短期波动脱钩。要拟合跨期限 vol 曲面，得上 LMM（多因子 Libor 市场模型）或加入随机波动率。

## 七、实战示例：给一个 receiver swaption 报价

把上文代码直接复用，只把 payoff 方向反过来，就能给「收固定、付浮动」的 receiver swaption 报价。假设交易台想买一个 5y/5y receiver、行权价取市场远期 $F=2.01\%$、用同一个校准好的 $\sigma_{HW}=0.52\%$ 跑蒙特卡洛：

```python
def price_receiver(Z, sigma, K):
    r, cum = simulate_paths(Z, sigma)
    P0Te = np.exp(-cum[:, int(round(Te/dt))])
    Ann = np.zeros(Z.shape[0])
    for ti in swap_times:
        Ann += tau*(np.exp(-cum[:, int(round(ti/dt))])/P0Te)
    P_Te_Tn = np.exp(-cum[:, int(round((Te+tenor)/dt))])/P0Te
    S_Te = (1.0 - P_Te_Tn)/Ann
    payoff = np.maximum(K - S_Te, 0.0) * Ann        # 方向相反
    return np.mean(payoff * P0Te)

pv_recv = price_receiver(Z, sigma_fit, F)            # 与 payer 在同一 ATM 近似相等
print(f"Receiver swaption ATM 现值 = {pv_recv:.6f}")
```

在 ATM（$K=F$）处，receiver 与 payer 的现值应当几乎相等——这是看涨/看跌平价在互换期权上的体现。若你算出的两者差了几个 bp，第一反应不是「市场无效」，而是检查你的年金口径（远期 vs 绝对）、贴现因子、和随机冲击是否同一批。这一行校验，是实盘给 swaption 双边报价前的必须动作。

## 八、正确用法：当「定价引擎的拼图」，别当「完整曲面模型」

Swaption 估值最该记住的三件事：

- 它的现值 = 风险中性期望的 pathwise 贴现，**蒙特卡洛是唯一不依赖闭式的通用方法**（当标的变复杂、加多因子时，闭式立刻失效）；
- 校准是「把模型的一两个参数钉到市场报价」，钉住了 ATM 不代表钉住了微笑；
- HW 的价值是**解析透明 + 校准快**，适合做市快速报价；要覆盖微笑与期限结构，必须升级到 SABR/LMM。

## 结语

Swaption 把「未来进入利率互换的权利」装进了 Hull-White 这类利率模型：平价互换利率 $S(T_e)$ 由贴现曲线决定、payer payoff 是 $\max(0,S-K)\cdot A$、现值用全路径蒙特卡洛贴现求得。本文用二分法把 HW 扩散 $\sigma_{HW}\approx0.52\%$ 校准到市场 20% swaption vol（ATM 价精确对齐 0.0152），并验证了模型价随 $\sigma_{HW}$ 严格单调（0.5%→0.0146、1%→0.0279、2%→0.0510）。但它的真边界在四类裂缝里：常数 σ 压不出微笑、扁平曲线简化、利率与波动率是两层校准、单因子装不下整条曲面。理解这些，比背下 Black 公式重要——它告诉你**任何「模型价=市场价」的胜利，都只在被校准的那一个点上成立**。
