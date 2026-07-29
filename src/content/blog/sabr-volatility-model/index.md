---
title: "SABR 随机波动率模型：把波动率微笑写进四个参数"
description: "SABR 用四个参数 (α, β, ρ, ν) 就能拟合一整条波动率微笑，而且 Hagan 闭式近似让校准快到可以放进实时报价系统——这让它成为利率期权市场二十多年的事实标准。本文拆解四个参数各自控制微笑的哪一段（水平/骨架/偏斜/曲率），用 Python 完成一次完整校准（最小二乘 3 参数拟合 11 个市场报价），验证 SABR 相对局部波动率模型最关键的优势——正确的微笑动态（sticky-delta），并用蒙特卡洛揭示 Hagan 近似在长期限与深度 OTM 下的失真边界。附完整代码与四类实盘陷阱（中高阶）。"
publishDate: '2026-07-29'
tags:
  - 量化交易
  - SABR
  - 随机波动率
  - 波动率微笑
  - 期权定价
  - 利率衍生品
  - 模型校准
  - Python
language: Chinese
difficulty: advanced
---

Black-Scholes 说波动率是常数，市场用一条弯曲的**波动率微笑**回应：同一标的、同一到期日，不同行权价的期权隐含波动率各不相同。要给非标准行权价的期权报价、要算对冲比率，你需要一个能拟合并**正确外推**这条微笑的模型。

结论先放这：**SABR（Stochastic Alpha Beta Rho）用四个参数就能拟合绝大多数市场的微笑形状，而且每个参数各管一件事：α 定水平、β 定骨架、ρ 定偏斜、ν 定曲率。** 更关键的是 Hagan 等人 2002 年给出的闭式隐含波动率近似——不用解 PDE、不用蒙特卡洛，一个公式直接从参数映射到隐含波动率，校准快到毫秒级。这让 SABR 成为利率期权（swaption、cap/floor）市场二十多年的事实标准。本文用 Python 把它的拟合、校准、微笑动态和近似失真边界全部跑一遍。

## 一、模型结构：两个随机过程，四个参数

SABR 在远期测度下对远期价格 $F_t$ 和瞬时波动率 $\alpha_t$ 建模：

$$
dF_t = \alpha_t F_t^\beta \, dW_t^1, \qquad d\alpha_t = \nu \, \alpha_t \, dW_t^2, \qquad dW_t^1 dW_t^2 = \rho \, dt
$$

四个参数的分工：

| 参数 | 含义 | 控制微笑的哪一段 |
|------|------|------------------|
| $\alpha$ | 初始波动率水平 | 整条曲线上下平移（≈ ATM 波动率） |
| $\beta$ | CEV 弹性指数 ∈ [0,1] | "骨架"斜率：β=1 对数正态（无偏斜），β=0 正态（天然负偏） |
| $\rho$ | 价格与波动率的相关性 | 偏斜方向：ρ<0 左偏（股票典型），ρ>0 右偏 |
| $\nu$ | 波动率的波动率（vol-of-vol） | 曲率：ν 越大微笑两翼翘得越高 |

![SABR四参数各自的作用](/images/sabr-volatility-model/sabr-params-effect.png)

这张图是理解 SABR 的钥匙：**四个旋钮几乎正交**。α 只平移，β 只改骨架斜率，ρ 只倾斜，ν 只弯曲。校准时你能清楚知道每个参数在干什么——这种可解释性是它打败更"精确"但参数纠缠的模型（如 Heston）的重要原因。

## 二、Hagan 近似：从参数到隐含波动率的直达公式

SABR 本身没有欧式期权的精确闭式解，它的杀手锏是 Hagan et al. (2002) 的奇异摄动展开——直接给出 Black 隐含波动率的近似：

```python
import numpy as np

def sabr_iv(F, K, T, alpha, beta, rho, nu):
    """Hagan 2002 对数正态 SABR 隐含波动率近似"""
    K = np.asarray(K, dtype=float)
    atm = np.isclose(K, F)
    FK = (F * K) ** ((1 - beta) / 2)
    logFK = np.log(F / K, where=~atm, out=np.zeros_like(K))

    z = nu / alpha * FK * logFK
    xz = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))

    A = alpha / (FK * (1 + (1-beta)**2/24 * logFK**2
                         + (1-beta)**4/1920 * logFK**4))
    B = 1 + T * ((1-beta)**2/24 * alpha**2 / FK**2
                 + rho*beta*nu*alpha / (4*FK)
                 + (2 - 3*rho**2)/24 * nu**2)
    ratio = np.where(atm, 1.0,
                     np.divide(z, xz, where=~atm, out=np.ones_like(K)))
    return A * ratio * B
```

整个公式只有初等函数，对一条 50 个行权价的微笑求值是微秒级。**这就是 SABR 能放进实时报价系统的原因**：交易员改一个参数，整条微笑立刻重画。

## 三、校准实战：3 个参数拟合 11 个报价

实务中 β 通常不参与校准——它和 ρ 在拟合意义上高度共线（都能产生偏斜），一般按市场惯例固定（股票/外汇用 β=1，利率市场常用 β=0.5 或 0）。剩下 (α, ρ, ν) 三个参数用最小二乘拟合：

```python
from scipy.optimize import least_squares

F0, T = 100.0, 0.5
# 合成"市场报价"：真实参数 α=0.28, ρ=-0.55, ν=1.1 + 30bp 噪声
K_mkt = np.array([70, 80, 85, 90, 95, 100, 105, 110, 115, 120, 130], float)
iv_mkt = sabr_iv(F0, K_mkt, T, 0.28, 1.0, -0.55, 1.1) \
         + np.random.default_rng(7).normal(0, 0.003, len(K_mkt))

def resid(x):
    a, r_, n_ = x
    return sabr_iv(F0, K_mkt, T, a, 1.0, r_, n_) - iv_mkt

sol = least_squares(resid, x0=[0.2, -0.2, 0.5],
                    bounds=([1e-4, -0.999, 1e-4], [2, 0.999, 5]))
print(sol.x)   # -> α=0.279, ρ=-0.541, ν=1.113
```

![SABR校准拟合](/images/sabr-volatility-model/sabr-calibration-fit.png)

11 个带噪声的报价，3 个自由参数，拟合出的曲线穿过所有点，恢复的参数（α=0.279, ρ=−0.54, ν=1.11）与真值（0.28, −0.55, 1.1）几乎一致。实务加速技巧：**用 ATM 波动率反解 α**（给定 ρ、ν 时 α 满足一个三次方程），把三维搜索降成二维——这是生产系统的标准做法。

## 四、微笑动态：SABR 真正的护城河

拟合微笑很多模型都行——局部波动率（Dupire）模型甚至能**完美**拟合任意无套利微笑。SABR 胜出的地方不是拟合，是**预测微笑如何随标的移动**。

Hagan 在原论文里指出了局部波动率模型的致命伤：当标的价格上涨时，局部波动率模型预测微笑**向左移动**（与市场观察相反），导致 delta 对冲系统性错误。而 SABR 的微笑跟着远期价一起平移——即所谓 **sticky-delta 动态**，与股票、外汇市场的实际行为一致：

![SABR微笑动态](/images/sabr-volatility-model/sabr-smile-dynamics.png)

F 从 90 移到 110，整条微笑形状不变、随远期平移，ATM 点（圆点）始终落在微笑的同一相对位置。这意味着 SABR 的 delta 里正确包含了"标的动、微笑跟着动"的调整项。**对冲账本上的差异是真金白银**：用局部波动率模型对冲 skew 明显的市场，会持续在 vega 再平衡上漏钱。

## 五、近似的边界：Hagan 公式什么时候会骗你

Hagan 公式是 $T$ 的一阶摄动展开，不是精确解。用蒙特卡洛（20 万条路径，Euler 离散）对照：

```python
def sabr_mc(F, K_arr, T, alpha, beta, rho, nu,
            n_paths=200_000, n_steps=250, seed=1):
    rg = np.random.default_rng(seed)
    dt = T / n_steps
    f = np.full(n_paths, F); a = np.full(n_paths, alpha)
    for _ in range(n_steps):
        z1 = rg.standard_normal(n_paths)
        z2 = rho*z1 + np.sqrt(1-rho**2)*rg.standard_normal(n_paths)
        f = np.maximum(f + a * f**beta * np.sqrt(dt) * z1, 1e-8)
        a = a * np.exp(nu*np.sqrt(dt)*z2 - 0.5*nu**2*dt)  # 精确对数正态步
    # 期权均价 -> Black 隐波（brentq 反解），略
```

![Hagan近似vs蒙特卡洛](/images/sabr-volatility-model/sabr-hagan-vs-mc.png)

T=0.5 年时近似和蒙特卡洛几乎重合；T=5 年、ν=1.0 时，深度 OTM 两翼的近似开始明显偏离——**期限越长、vol-of-vol 越大、离 ATM 越远，误差越大**。更隐蔽的问题：低行权价区域 Hagan 近似可能给出**负的隐含概率密度**（对行权价的二阶导为负），即公式本身蕴含蝶式套利。利率市场在负利率时代被迫直面这个问题，催生了 shifted SABR、free-boundary SABR 和 Antonov 精确解等修补方案。

## 六、实盘陷阱清单

**陷阱一：β 和 ρ 一起校准。** 两者在单一期限的拟合上近乎共线，联合估计会得到不稳定的参数组合，今天 β=0.7/ρ=−0.3、明天 β=0.3/ρ=−0.6，报价没变但对冲比率跳变。正确做法：β 按市场惯例或用长期回归（log ATM vol 对 log F）定死，只校准三个。

**陷阱二：把 Hagan 公式用到超长期限。** 5 年以上、高 ν 的组合，近似误差可达数个波动率点，且两翼可能隐含负密度。长期限报价要么用精确方法（Antonov et al.）、要么至少做蒙特卡洛交叉验证。

**陷阱三：每天独立重新校准，不看参数时序。** SABR 参数应该是缓变的。如果 ν 隔天从 0.8 跳到 1.5，大概率不是市场变了，而是优化器掉进了另一个局部极小。生产系统要用前一天参数做初值 + 参数变动惩罚。

**陷阱四：拿 SABR 定价强路径依赖产品。** SABR 的波动率没有均值回复，长期方差发散。它是**单一到期日微笑的插值/外推工具**，不是完整的期限结构模型。定价 cliquet、autocallable 这类跨期限产品，需要 Heston、rough vol 或 LMM-SABR 这类全动态模型。

## 七、总结

SABR 的成功是"工程上够好"击败"理论上更对"的经典案例：

1. **四个参数正交分工**（水平/骨架/偏斜/曲率），交易员能直觉地读懂和手调每个参数；
2. **Hagan 闭式近似让校准毫秒级完成**——3 参数最小二乘拟合 11 个报价，恢复参数误差 <2%；
3. **正确的 sticky-delta 微笑动态**是它相对局部波动率模型的本质优势，直接体现在 delta 对冲的长期损益上;
4. **边界要心里有数**：长期限、深 OTM、高 ν 下近似失真甚至隐含负密度——知道模型什么时候会骗你，和会用模型同样重要。

从 2002 年到今天，比 SABR 更精确的模型出现了几十个，但利率期权屏幕上跳动的报价背后，大多仍然是这四个参数。**在拟合能力、校准速度、参数可解释性和对冲正确性的四维权衡里，SABR 至今仍站在效率前沿上。**
