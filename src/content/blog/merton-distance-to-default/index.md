---
title: "Merton 违约距离模型：用股价反推信用风险"
description: "股权是公司对资产的看涨期权——这是 Merton 1974 结构化信用模型的核心洞见。既然期权价格（股价）和波动率都可观测，就能反推出不可观测的公司资产价值与资产波动率，进而算出「资产离违约边界还有几个标准差」（违约距离 DD），再把 DD 变成违约概率 PD = N(−DD)。本文从 BS 方程出发完整实现迭代求解器，用三家合成公司（健康/中等/困境）演示 DD 如何随杠杆与波动变化，并用 200 条蒙特卡洛路径校验 PD，附完整 Python。"
publishDate: '2026-08-27'
tags:
  - 量化交易
  - 信用风险
  - Merton 模型
  - 结构化信用模型
  - 违约距离
  - 期权定价
  - 固定收益
  - Python
language: Chinese
difficulty: advanced
---

一家公司会不会违约，市场其实每天都在「投票」——票就是它的股价。Merton (1974) 的伟大之处，是把这个直觉变成了一个可计算的公式：**把公司股权看成公司资产价值 $V$ 对债务面值 $D$ 的欧式看涨期权**。资产是底层标的，债务面值是执行价，到期时资产低于债务就违约（期权作废）。

结论先放这：**股价和股价波动率都是可观测的，而真正决定违约的资产价值 $V$ 和资产波动率 $\sigma_V$ 不可观测；但用两条 Black-Scholes 方程联立，可以把 $V$ 和 $\sigma_V$ 反解出来，然后算「资产离违约边界还有几个标准差」——违约距离 DD，再映射到违约概率 $\mathrm{PD}=N(-DD)$**。本文用真实计算的迭代求解器，得到三家合成公司：健康公司 DD=7.39（PD≈0%）、中等公司 DD=3.63（PD≈0.014%）、困境公司 DD=1.91（PD≈2.8%）；并用 200 条蒙特卡洛路径校验困境公司经验 PD=2.5%，与理论值吻合。附完整 Python 与四张真实计算图（高阶）。

![Merton 结构化模型：到期资产价值分布，资产跌破债务面值 D 的左侧面积就是违约概率](/images/merton-distance-to-default/mdd_structure.png)

## 一、核心思想：股权 = 看涨期权

设公司资产价值为 $V$，债务面值为 $D$，到期为 $T$，无风险利率 $r$。到期时：

- 若 $V_T > D$，股东行使「期权」，拿走 $V_T - D$；
- 若 $V_T \le D$，股东放弃，公司违约，拿走 0。

所以股权市值 $E = \max(V_T - D, 0)$，正是执行价 $D$、到期 $T$ 的看涨期权。资产价值 $V$ 服从几何布朗运动，于是 $E$ 满足标准 BS 公式：

$$
E = V\,N(d_1) - D e^{-rT} N(d_2), \quad
d_1 = \frac{\ln(V/D) + (r + \tfrac{1}{2}\sigma_V^2)T}{\sigma_V\sqrt{T}}, \quad
d_2 = d_1 - \sigma_V\sqrt{T}
$$

另一条件来自 **Itô 引理**：股权波动率 $\sigma_E$ 与资产波动率 $\sigma_V$ 之间满足

$$
\sigma_E \cdot E = N(d_1)\,\sigma_V \cdot V
$$

这两个方程里，$E$（股价）、$\sigma_E$（股价年化波动率）、$D$（债务面值）都是可观测的，未知量是 $V$ 和 $\sigma_V$。两个方程、两个未知数——可解。

## 二、联立求解：牛顿式迭代反解 V 与 σ_V

直接解闭式很难，但迭代收敛极快。思路：给定 $(V, \sigma_V)$，用 BS 公式算出理论股权 $E^{\text{bs}}(V,\sigma_V)$，逼它等于观测 $E$。我们固定 $\sigma_V$ 反解 $V$，再用第二个方程更新 $\sigma_V$，交替迭代：

```python
import numpy as np

def norm_cdf(x):
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    z = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    a = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]
    erf = 1.0 - (((((a[4]*t + a[3])*t) + a[2])*t + a[1])*t + a[0])*t*np.exp(-z*z)
    return 0.5 * (1.0 + sign * erf)

def solve_merton(E, sigma_E, D_face, r, T, tol=1e-9, max_iter=1000):
    """由可观测股权 E, σ_E 反解资产 V, σ_V"""
    V = E + D_face                  # 初值：资产 ≈ 股权 + 债务
    sigma_V = sigma_E * E / V
    for _ in range(max_iter):
        d1 = (np.log(V / D_face) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
        d2 = d1 - sigma_V * np.sqrt(T)
        # 由 BS 反解 V：E = V·N(d1) − D·e^{−rT}·N(d2)
        V_new = (E + D_face * np.exp(-r * T) * norm_cdf(d2)) / norm_cdf(d1)
        sigma_V_new = sigma_E * E / (V_new * norm_cdf(d1))
        if abs(V_new - V) < tol and abs(sigma_V_new - sigma_V) < tol:
            V, sigma_V = V_new, sigma_V_new
            break
        V, sigma_V = V_new, sigma_V_new
    d1 = (np.log(V / D_face) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)
    return V, sigma_V, d1, d2
```

## 三、违约距离 DD 与违约概率 PD

解出 $V, \sigma_V$ 后，违约距离定义为「资产对数收益离违约边界的标准差个数」：

$$
\mathrm{DD} = \frac{\ln(V/D) + (\mu - \tfrac{1}{2}\sigma_V^2)T}{\sigma_V\sqrt{T}}
$$

其中 $\mu$ 是资产漂移率（预期增长率）。DD 越大越安全。把 DD 映射到概率：

$$
\mathrm{PD} = N(-\mathrm{DD})
$$

注意这里用的是 **$\mu$（真实漂移）** 而不是无风险利率 $r$——因为 PD 是真实世界违约概率，不是风险中性定价。这是 Merton 模型实务里最容易踩的坑：用 $r$ 算出来的 DD 偏乐观。

## 四、三家公司的真实计算结果

取 $r=3\%, T=1, \mu=8\%$，三家合成公司（健康/中等/困境）的可观测输入与反解结果：

```python
r, T, mu = 0.03, 1.0, 0.08
firms = [
    # 名称, 股权市值E, 股权波动σ_E, 债务面值D
    ("健康公司 A", 800.0, 0.25, 300.0),
    ("中等公司 B", 400.0, 0.40, 400.0),
    ("困境公司 C", 120.0, 0.65, 500.0),
]
for name, E, sE, Df in firms:
    V, sV, d1, d2 = solve_merton(E, sE, Df, r, T)
    DD = (np.log(V / Df) + (mu - 0.5 * sV**2) * T) / (sV * np.sqrt(T))
    PD = norm_cdf(-DD)
    print(f"{name}: V={V:.1f} σ_V={sV:.3f} 杠杆D/V={Df/V:.2f} DD={DD:.2f} PD={PD*100:.3f}%")
```

真实输出：

```
健康公司 A: V=1091.1 σ_V=0.183 杠杆D/V=0.27 DD=7.39 PD=0.000%
中等公司 B: V=788.2  σ_V=0.203 杠杆D/V=0.51 DD=3.63 PD=0.014%
困境公司 C: V=603.6  σ_V=0.136 杠杆D/V=0.83 DD=1.91 PD=2.801%
```

![三家公司 DD 与 PD 对比：DD 越高 PD 越低（对数坐标），困境公司 PD 比健康公司高几个数量级](/images/merton-distance-to-default/mdd_dd_pd.png)

三个关键点：

1. **杠杆率决定一切**：健康公司债务只占资产 27%，DD=7.39，几乎不会违约；困境公司债务占 83%，DD 只有 1.91，两年内违约概率接近 2.8%。**资产波动反而不是主因**——困境公司 σ_V 只有 0.136（最低），但它的债务快压垮资产了。
2. **股权波动 ≠ 资产波动**：困境公司股价波动 0.65，但反解出的资产波动只有 0.136。因为高杠杆把股权变成了一个深度虚值的期权，期权的高波动主要来自「距离执行价近」，而不是资产本身乱动。这正是 Merton 模型「用股价反推」的魅力——它把杠杆的放大效应显式拆了出来。
3. **PD 对 DD 极度非线性**：DD 从 3.63 掉到 1.91（腰斩），PD 从 0.014% 跳到 2.8%（200 倍）。信用风险不是线性恶化的，是加速恶化的。

## 五、DD 对杠杆与波动率的敏感性

把 DD 对（杠杆率 D/V、资产波动率 σ_V）画成曲线，能看到违约的边界在哪里：

![违约距离对杠杆与资产波动率的敏感性：杠杆越高、波动越大，DD 越快穿过 0（即违约）](/images/merton-distance-to-default/mdd_sensitivity.png)

- 三条曲线都随杠杆上升而下降，且在 D/V 接近 1 时快速穿过 DD=0（违约线）。
- 同一杠杆下，资产波动越大（红线），DD 越低。但注意横轴是**资产**波动——现实中你观测到的是**股权**波动，中间隔着杠杆的放大，所以「股价波动高」往往同时意味着「杠杆高或资产波动高」，需要模型才能区分。

实务含义：**Merton DD 是一个优秀的「早期预警」指标，但它对杠杆率极其敏感**。一家公司如果只是短期股价跌（股权价值降），只要债务没变，DD 会下降、PD 上升——模型会立刻把违约风险调高，这正是它比静态评级有用的地方（评级滞后，模型日更）。

## 六、蒙特卡洛校验：理论 PD 对不对

把反解出的资产参数拿去跑真实 GBM 路径，看到期资产低于债务的比例，应该和 $N(-DD)$ 吻合：

```python
rng = np.random.default_rng(42)
V0, sV, Df = 603.6, 0.136, 500.0   # 困境公司
n_paths, n_steps, T = 200, 252, 1.0
dt = T / n_steps
paths = np.zeros((n_paths, n_steps + 1)); paths[:, 0] = V0
for i in range(n_steps):
    z = rng.standard_normal(n_paths)
    paths[:, i+1] = paths[:, i] * np.exp((mu - 0.5*sV**2)*dt + sV*np.sqrt(dt)*z)
defaulted = paths[:, -1] < Df
emp_pd = defaulted.mean()
print(f"经验 PD={emp_pd*100:.2f}%  理论 PD=2.80%")
```

真实输出：**经验 PD=2.50%，理论 PD=2.80%**，量级一致（200 条路径的统计误差内）。

![困境公司 200 条资产蒙特卡洛路径：红线到期跌破债务边界 D，经验违约率 2.5% 与理论 PD 吻合](/images/merton-distance-to-default/mdd_paths.png)

这条校验链很重要：它证明我们的迭代求解器没写错——反解出的 $(V, \sigma_V)$ 真的能复现理论违约概率。很多网上的 Merton 实现只在「健康公司」上自洽，一到高杠杆就发散；本文求解器在 D/V=0.83 的困境公司上依然收敛且校验通过。

## 七、已知的边界（诚实地说）

Merton 模型不是万能的，三个硬伤要记牢：

1. **单一债务、零息假设**：真实公司有多笔不同期限债务，用单一面值 $D$ 是近似。实务用「债务账面价值 × 折让」或「短期债务 + 0.5 长期债务」做 $D$。
2. **资产收益正态假设**：真实资产收益有肥尾和跳跃，DD 会低估极端违约。改进是跳扩散或局部波动率。
3. **μ 不可观测**：PD 对 $\mu$ 敏感，而 $\mu$ 只能估计。常见做法是用行业/历史资产增长率，或干脆只报 DD（不报 PD），把 PD 映射留给校准。

最后给一个可直接落地的结论：**如果你只想每天监控持仓的信用风险，DD 比 PD 更稳**——它不依赖 $\mu$ 的估计，且对杠杆变化灵敏、对评级滞后免疫。PD 适合做组合层面的违约损失预期，但务必记得它来自一个正态近似。本文所有数字均为真实计算，代码可直接复用到任意「股价 + 债务」可观测的公司上。
