---
title: "门限GARCH与杠杆效应：用TGARCH/EGARCH捕捉跌比涨更恐慌的不对称波动"
description: "经典 GARCH 假设「好消息和坏消息对称地推高波动」，但股灾里「跌比涨更恐慌」是铁律：同样 3% 的跌幅带来的波动跳升，往往大于 3% 涨幅。本文用一套合成的真实波动数据，从零实现 TGARCH(1,1) 与 EGARCH 的 MLE 拟合，量化「同等 ±3% 冲击，跌幅比涨幅多推高 9% 条件方差」，并给出把杠杆效应写进波动率预测、风险预算与期权对冲的完整 Python（中阶）。"
publishDate: '2026-07-19'
tags:
  - 量化交易
  - 波动率建模
  - GARCH
  - 门限GARCH
  - 杠杆效应
  - EGARCH
  - 风险管理
  - Python
language: Chinese
difficulty: intermediate
---

你每天看盘都见过这个画面:

- 某天大盘 **+3%**,第二天风平浪静,波动率没怎么动;
- 另某天大盘 **−3%**,第二天却像打开了恐慌开关,波动率猛地蹿上去。

经典 GARCH 模型**捕捉不到这个差别**。它把昨日收益平方项 $e_{t-1}^2$ 直接搬进方差方程——而平方项把正负号抹平了,于是「涨 3%」和「跌 3%」对波动的贡献**完全一样**。这跟市场常识对着干。

> 这就是 **杠杆效应(leverage effect)** / 非对称波动(asymmetric volatility):坏消息对波动的冲击大于同幅好消息。Black(1976) 最早发现,随后 Nelson(1991) 的 EGARCH、Zakoian(1994) 的 TGARCH(也叫 GJR-GARCH)把这种不对称写进了模型。本文用合成数据从零实现它们,并量化「到底不对称多少」。

## 一、先看清对称 GARCH 哪里「瞎」

对称 GARCH(1,1) 的方差方程是:

$$\sigma_t^2 = \omega + \alpha\,e_{t-1}^2 + \beta\,\sigma_{t-1}^2$$

注意 $e_{t-1}^2$ 这个平方项:**符号被平方吃掉了**。无论昨天涨 3% 还是跌 3%,进公式的都是同一个 $0.03^2=9\times10^{-4}$。模型说:「好消息和坏消息同等可怕」。

真实市场显然不是这样。我们要在方差方程里把「负收益」单独拎出来,给它额外加权。两个主流做法:

**TGARCH / GJR-GARCH(门限型)**——加一个只在 $e_{t-1}<0$ 时生效的杠杆项:

$$\sigma_t^2 = \omega + (\alpha + \gamma\,\mathbf{1}_{e_{t-1}<0})\,e_{t-1}^2 + \beta\,\sigma_{t-1}^2$$

$\gamma>0$ 就是杠杆效应:下跌时冲击系数从 $\alpha$ 跳到 $\alpha+\gamma$。

**EGARCH(指数型)**——在**对数方差**上建模,用标准化残差 $z_{t-1}=e_{t-1}/\sigma_{t-1}$ 的**非对称**形式:

$$\ln\sigma_t^2 = \omega + \beta\ln\sigma_{t-1}^2 + \alpha\big(|z_{t-1}|-\sqrt{2/\pi}\big) + \gamma\,z_{t-1}$$

EGARCH 的好处:(1) 对数变换天然保证 $\sigma_t^2>0$,不用像 GARCH 那样加非负约束;(2) $\gamma<0$ 直接对应「负 $z$ 推高波动」的杠杆效应,而且它对大小冲击是非线性的。

## 二、造一段「真有杠杆效应」的数据

我们用 TGARCH(1,1) 模拟 4000 个交易日的日收益,真实参数 $\omega=2\times10^{-5},\ \alpha=0.08,\ \gamma=0.06,\ \beta=0.85$(持久性 $\alpha+\gamma/2+\beta=0.96$,符合真实股指波动的聚集特征)。

```python
import numpy as np

def simulate_tgarch(T, omega=2e-5, alpha=0.08, gamma=0.06, beta=0.85):
    e = np.zeros(T); sigma = np.zeros(T); z = np.random.default_rng(20260719).normal(0, 1, T)
    sigma[0] = np.sqrt(omega / (1 - alpha - 0.5*gamma - beta))
    for t in range(1, T):
        g = alpha + (gamma if e[t-1] < 0 else 0.0)
        sigma[t] = np.sqrt(max(omega + g*e[t-1]**2 + beta*sigma[t-1]**2, 1e-10))
        e[t] = sigma[t] * z[t]
    return e, sigma

T = 4000
e, sigma_true = simulate_tgarch(T)   # 真实波动 sigma_true 已知
```

关键点:因为是我们**自己造的数据**,我们既知道「真实波动」,也知道「杠杆参数 $\gamma$ 真值」,这正是检验后续拟合是否靠谱的金标准。

## 三、用 MLE 把三个模型都拟合一遍

对数似然(高斯假设):

$$\ell = \sum_{t}\Big[-\tfrac12\ln(2\pi)-\tfrac12\ln\sigma_t^2-\tfrac12\,\frac{e_t^2}{\sigma_t^2}\Big]$$

TGARCH 和对称 GARCH 用同一个递归,差别只是「是否加 $\gamma$ 项」。EGARCH 用对数方差递归。直接上 `scipy.optimize.minimize`:

```python
from scipy.optimize import minimize

def garch_ll(params, e, asymmetric=False):
    omega, alpha, beta = params[0], params[1], params[2]
    gamma = params[3] if asymmetric else 0.0
    if omega <= 0 or alpha <= 0 or beta <= 0 or beta >= 0.999:
        return 1e12
    s2 = np.zeros_like(e); s2[0] = np.var(e); ll = 0.0
    for t in range(1, len(e)):
        g = alpha + (gamma if e[t-1] < 0 else 0.0)
        s2[t] = omega + g*e[t-1]**2 + beta*s2[t-1]
        s2[t] = max(s2[t], 1e-12)
        ll += -0.5*(np.log(2*np.pi) + np.log(s2[t]) + e[t]**2/s2[t])
    return -ll

# 对称 GARCH 与 TGARCH 拟合
p_sym  = minimize(garch_ll, [2e-5, 0.10, 0.85],     args=(e, False),
                  method="L-BFGS-B", bounds=[(1e-8,1e-3),(1e-4,0.4),(1e-3,0.98)]).x
p_asym = minimize(garch_ll, [2e-5, 0.08, 0.85, 0.06], args=(e, True),
                  method="L-BFGS-B",
                  bounds=[(1e-8,1e-3),(1e-4,0.4),(1e-3,0.98),(0.0,0.3)]).x
```

拟合结果(真实 $\gamma=0.06$):

| 模型 | $\omega$ | $\alpha$ | $\gamma$ | $\beta$ | 对数似然 |
|---|---|---|---|---|---|
| 真实(造数据) | $2.00\times10^{-5}$ | 0.080 | **0.060** | 0.850 | — |
| 对称 GARCH | $2.23\times10^{-5}$ | 0.1066 | — | 0.8529 | 9546.1 |
| TGARCH | $2.26\times10^{-5}$ | 0.0792 | **0.0605** | 0.8503 | **9551.7** |

TGARCH 把 $\gamma$ 几乎精确还原成真值 0.0605,而且对数似然比对称 GARCH **高了 5.6**——在 4000 个样本上,这是杠杆效应存在的清晰统计证据(似然比检验)。

![条件方差对正负冲击的响应：跌弯得更陡](/images/threshold-garch-asymmetry/tgarch_leverage_effect.png)

## 四、量化「到底不对称多少」

把拟合好的两个模型摆在一起,给它们喂**同一个昨日条件方差 $\sigma_{t-1}^2$**,然后分别注入一个 −3% 崩盘和一个 +3% 反弹,看下一步条件方差被推高多少:

- 对称 GARCH:跌和涨完全一样,因为公式里只有平方项;
- TGARCH:跌时用系数 $\alpha+\gamma$,涨时用 $\alpha$,差一个 $\gamma$。

代入拟合参数算出来:

$$\frac{\sigma^2_{\text{跌}}}{\sigma^2_{\text{涨}}}-1 = \frac{\omega+(\alpha+\gamma)(0.03)^2+\beta\sigma_{t-1}^2}{\omega+\alpha(0.03)^2+\beta\sigma_{t-1}^2}-1 \approx 9.0\%$$

**同等 ±3% 冲击,跌幅比涨幅多推高约 9% 的条件方差。** 这就是杠杆效应的体量——不是文学修辞,是可写进预测公式的数字。图 1 左图更直观:响应曲线在负半轴「弯得更陡」,对称 GARCH 则是一条左右对称的 U。

## 五、不靠模型,用数据直接看不对称

想绕开 MLE、直接验证不对称?做个**非对称响应回归**:把次日 |e_t| 对「前日下跌幅度」和「前日上涨幅度」分别回归,

$$|e_t| = c + \beta^-(-\min(e_{t-1},0)) + \beta^+(+\max(e_{t-1},0)) + \varepsilon$$

斜率 $\beta^-$ 和 $\beta^+$ 谁大,谁就更「可怕」:

```python
neg_ret = -np.minimum(e[:-1], 0)   # 下跌幅度(正数)
pos_ret =  np.maximum(e[:-1], 0)   # 上涨幅度(正数)
y_abs   =  np.abs(e[1:])
X = np.column_stack([neg_ret, pos_ret, np.ones_like(neg_ret)])
coef, *_ = np.linalg.lstsq(X, y_abs, rcond=None)
sl_neg, sl_pos, _ = coef
```

![非对称响应：跌幅斜率 > 涨幅斜率](/images/threshold-garch-asymmetry/tgarch_asymmetric_slope.png)

跑出来:下跌幅度斜率 $\beta^-=0.2396$,上涨幅度斜率 $\beta^+=0.2259$,**下跌斜率更大**——和 TGARCH 的 $\gamma>0$ 结论一致(两者是同一个不对称性的两种度量视角)。注意这张图的 R² 只有 0.054,这**不说明模型烂**:日收益 |e_t| 本身高度离散,哪怕真实存在杠杆效应,单日冲击也淹没了它;要看清结构,还是得靠 GARCH 类模型把「条件方差」抽出来再比较。

## 六、TGARCH 还原波动到底比对称 GARCH 好多少

用拟合参数把两条波动率路径都递推出来,和「真实波动」比 RMSE:

| 模型 | 波动拟合 RMSE |
|---|---|
| 对称 GARCH | 0.0011 |
| TGARCH | **0.0004** |

TGARCH 的还原误差只有对称 GARCH 的约 1/3。图 3 把三段采样窗口摆出来看更清楚:在波动急升急降的拐点,对称 GARCH 的波动线「钝」了一截,因为它无法对下跌做额外放大;TGARCH 的线更贴着真实波动走。

![TGARCH 还原波动更贴真实](/images/threshold-garch-asymmetry/tgarch_egarch_fit.png)

(本文 EGARCH 也实现了对数方差递归,拟合思路同 TGARCH;篇幅所限正文聚焦 TGARCH 与不对称量化,EGARCH 的 $\gamma<0$ 符号含义与 TGARCH 的 $\gamma>0$ 完全等价,读者可把 `garch_ll` 换成对数方差版本自行验证。)

## 七、五个必须知道的坑

1. **$\gamma$ 不是「跌一定更可怕」的保证,只是参数估计。** 真实数据上 $\gamma$ 偶尔估成 0 甚至负(某些商品、外汇在样本期呈现「涨比跌慌」)。永远先看似然比检验 $\Delta\ell$ 和 bootstrap 置信区间,再下「存在杠杆效应」的结论。
2. **MLE 方差递归是串行的,4000 点跑循环慢。** 生产上要向量化或上 `numba`;`scipy` 循环版适合教学和小样本。本文用循环只为可读性。
3. **持久性 $\alpha+\gamma/2+\beta$ 必须 < 1。** 一旦逼近 1,波动率变成近似单位根,样本外预测会「冻结」在近期水平,IG 估计极不稳定。拟合后务必打印这个和。
4. **杠杆效应会改变你的风险预算。** 如果你用波动率做仓位/风险平价,对称 GARCH 会**系统性低估下跌后的波动**,从而在暴跌次日给过高的仓位——这正是 2008 年许多风险平价策略踩的坑之一。用 TGARCH/EGARCH 至少把不对称建模进去。
5. **期权对冲要用它。** 崩盘后波动率跳升最快,用对称 GARCH 做 Vega 对冲会在股灾里欠一大截 Gamma 对冲成本。把 $\gamma$ 写进你的波动率预测,是做波动率曲面/恐慌指数类策略的基本功。

## 八、完整可复现脚本

本文全部图表与数字由 `gen_threshold_garch.py` 生成(已附仓库)。核心三件事:① 用 `simulate_tgarch` 造带杠杆的真实数据;② 用 `garch_ll` 做对称 / TGARCH 的 MLE;③ 用同一 $\sigma_{t-1}^2$ 喂 ±3% 冲击量化不对称。把 `rng` 种子换成你的真实收益序列,就能直接对沪深 300、中证 500 或个股做杠杆效应体检。

> 下一篇可以接着看「波动率之波动率(VVIX)」或「粗糙波动率模型」——当 $\gamma$ 本身也随时间变,杠杆效应就成了时变的非对称。
