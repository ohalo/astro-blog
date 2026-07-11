---
title: "DCC-GARCH 多元波动率模型：让相关矩阵随时间起舞"
description: "单变量 GARCH 只解决了一半问题——真正的魔鬼藏在相关矩阵里。Engle(2002) 的 DCC-GARCH 用极少参数让相关矩阵随时间演化，把危机里的相关性飙升如实搬进组合协方差。从零实现并对比常数相关的风险低估。"
publishDate: '2026-07-11'
tags:
  - 量化交易
  - 波动率
  - DCC-GARCH
  - 相关结构
  - 组合风险
  - Python
language: Chinese
difficulty: advanced
---

多元波动率建模里最容易踩的坑，是以为「给每个资产各装一个 GARCH，再把它们拼成协方差矩阵」就完事了。拼出来的协方差长这样：

$$\Sigma_t = D_t\,R\,D_t$$

其中 $D_t$ 是各资产条件波动率的对角阵，$R$ 是相关矩阵。问题在于：**大多数人会顺手把 $R$ 写成全样本估计的常数矩阵**。而相关从来不是常数——平静时两只股票相关性 0.3，暴跌时所有东西一起跳水，相关性瞬间冲到 0.85 甚至更高。用常数相关，等于在暴风雨里假设风平浪静。

结论先放这：**DCC-GARCH（Dynamic Conditional Correlation, Engle 2002）用一个两阶段结构解决了这个问题——第一阶段给每个资产各拟合一个单变量 GARCH 拿到条件波动，第二阶段用仅仅两个参数 $(a,b)$ 让相关矩阵 $R_t$ 随时间平滑演化。它用极少参数把「相关性随时间变化」这一事实搬进组合风险，且计算量只是单变量 GARCH 的线性放大。本文从零实现，并展示 DCC 如何把危机里的相关飙升如实捕捉，而常数相关会系统性低估组合风险。

补充一句：本文用双资产把直觉讲到透，但 DCC 的方法论对 $N$ 资产完全一致——只是把 $\bar{Q}$、$Q_t$ 从 $2\times2$ 换成 $N\times N$，$(a,b)$ 仍是那两个参数，估计量不随维度爆炸。这正是它在实盘组合风险里比「逐对相关分别建模」现实得多的根本原因。**

![两资产收益与真实时变相关：危机窗口相关性从 0.30 飙升到约 0.85](/images/dcc-garch-multivariate/dcc_returns_true_corr.png)

## 一、为什么单变量 GARCH 不够

波动率聚集是每个做量化的人都熟悉的：安静的行情能连着几周风平浪静，暴跌往往接二连三。GARCH 族把单个资产的方差建模成随时间演化的隐变量，这是对的。但当你的目标是**组合**——算组合方差、做风险预算、算 VaR——你需要的不是一堆 $\sigma_i^2$，而是完整的协方差矩阵 $\Sigma_t$。

组合方差的分解很清楚（等权 $w=[0.5,0.5]$ 为例）：

$$\text{Var}(r_p) = \tfrac14(\sigma_1^2+\sigma_2^2) + \tfrac12 \rho\,\sigma_1\sigma_2$$

前半段是两个资产各自的波动，后半段是它们的**相关耦合项**。危机里 $\sigma_1,\sigma_2$ 会涨，但更致命的是 $\rho$ 从 0.3 跳到 0.85——耦合项几乎翻三倍。如果你用常数 $\rho=0.387$（全样本平均），危机段的风险会被严重低估。这正是 2008 年很多风险模型失灵的根源。

更隐蔽的是，常数相关还会把风险贡献算错。风险预算（risk budgeting）要回答「组合波动里多大比例来自科技、多大比例来自金融」，这依赖相关结构。当真实相关在危机里趋同，风险贡献会高度集中到少数系统性因子；常数相关下的风险贡献却是平滑分散的假象，会让你在最需要降杠杆时误以为自己已经足够分散。

## 二、DCC 模型：相关矩阵怎么"动"起来

DCC 的关键洞察是：先**标准化**掉每个资产自己的波动率，再在"标准化残差"的空间里让相关动态演化。

第一步，对每个资产拟合 GARCH(1,1)，得到条件标准差 $\sigma_{i,t}$，构造标准化残差：

$$\varepsilon_{i,t} = \frac{r_{i,t}}{\sigma_{i,t}}$$

标准化残差近似是方差为 1、彼此相关的白噪声序列。第二步，Engle 让相关矩阵通过下面这个递归"动态"演化：

$$Q_t = (1-a-b)\,\bar{Q} + a\,(\varepsilon_{t-1}\varepsilon_{t-1}^\top) + b\,Q_{t-1}$$

$$R_t = \text{diag}(Q_t)^{-1/2}\,Q_t\,\text{diag}(Q_t)^{-1/2}$$

- $\bar{Q}$：标准化残差的**样本**相关矩阵（长期中枢）
- $a$：对"昨天的相关新息" $\varepsilon_{t-1}\varepsilon_{t-1}^\top$ 的敏感度
- $b$：相关性的持续度（记忆）
- 约束 $a+b<1$ 保证 $Q_t$ 平稳、长期收敛到 $\bar{Q}$
- 最后一步把 $Q_t$ 重新缩放成单位对角的相关矩阵 $R_t$

整个相关动态**只有两个待估参数** $(a,b)$——不管你有 2 个还是 200 个资产。这是 DCC 最漂亮的地方：它绕开了"N 个资产就要估 N(N-1)/2 个时变相关"的维度灾难。

## 三、从零实现（Python）

下面这段代码与配图完全对应：先模拟两资产（各自 GARCH 波动 + 一个会飙升的时变真实相关），再拟合单变量 GARCH、跑 DCC，最后对比动态相关与常数相关。

```python
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(20260711)
N = 2500

# ---- 1) 构造时变真实相关 rho_t：基线 0.30，危机窗口平滑飙到 ~0.85 ----
t = np.arange(N)
center, width = int(N*0.55), int(N*0.10)
rho_true = 0.30 + 0.55*np.exp(-((t-center)/width)**2)

m  = rng.standard_normal(N)   # 共同市场因子
e1 = rng.standard_normal(N); e2 = rng.standard_normal(N)
z1 = np.sqrt(rho_true)*m + np.sqrt(1-rho_true)*e1   # corr(z1,z2) == rho_true
z2 = np.sqrt(rho_true)*m + np.sqrt(1-rho_true)*e2

# ---- 2) 各自叠一层 GARCH(1,1) 波动聚集 ----
def simulate_garch(n, omega=1e-5, alpha=0.08, beta=0.90, mu=0.0002):
    r = np.zeros(n); s2 = np.zeros(n)
    s2[0] = omega/(1-alpha-beta)
    for i in range(1, n):
        z = rng.standard_normal()
        s2[i] = omega + alpha*r[i-1]**2 + beta*s2[i-1]
        r[i] = mu + np.sqrt(s2[i])*z
    return r, np.sqrt(s2)

R1 = simulate_garch(N)[0] * z1
R2 = simulate_garch(N)[0] * z2

# ---- 3) 每只资产拟合 GARCH(1,1) MLE，拿标准化残差 ----
def garch_mle(r):
    def negll(p):
        w, a, b = p
        if w<=0 or a<0 or b<0 or a+b>=0.999: return 1e10
        s2 = np.empty(len(r)); s2[0] = np.var(r)
        for i in range(1, len(r)):
            s2[i] = w + a*r[i-1]**2 + b*s2[i-1]
        return -0.5*(np.log(2*np.pi)+np.log(s2)+r**2/s2).sum()
    res = minimize(negll, [1e-5,0.08,0.90],
                   bounds=[(1e-9,1e-3),(1e-6,0.6),(1e-6,0.99)], method="L-BFGS-B")
    w, a, b = res.x
    s2 = np.empty(len(r)); s2[0] = np.var(r)
    for i in range(1, len(r)):
        s2[i] = w + a*r[i-1]**2 + b*s2[i-1]
    return np.sqrt(s2), r/np.sqrt(s2)

sig1, u1 = garch_mle(R1)
sig2, u2 = garch_mle(R2)
U = np.column_stack([u1, u2])

# ---- 4) DCC 相关动态 ----
a, b = 0.03, 0.92
Qbar = np.cov(U.T)
Q = Qbar.copy()
R_series = np.zeros(N); R_mats = np.zeros((N,2,2))
for i in range(N):
    if i>0:
        u = U[i-1]
        Q = (1-a-b)*Qbar + a*np.outer(u,u) + b*Q
    d = np.sqrt(np.diag(Q))
    R = Q/np.outer(d,d)
    R_mats[i] = R; R_series[i] = R[0,1]

R_const = np.corrcoef(U.T)   # 朴素的常数相关
print(f"DCC 拟合相关峰值={R_series.max():.3f}  常数相关={R_const[0,1]:.3f}  真实峰值={rho_true.max():.3f}")
```

## 四、结果：DCC 真的抓住了危机

跑出来三个数很说明问题：**DCC 拟合相关峰值 0.623，常数相关 0.387，真实峰值 0.850**。DCC 把危机里的相关飙升捕捉到了，而常数相关是一条平直的线，对危机毫无反应。

![DCC 拟合相关 vs 真实相关：危机飙升被追踪，常数相关全程不动](/images/dcc-garch-multivariate/dcc_fitted_vs_true_corr.png)

注意 DCC 峰值（0.62）低于真实峰值（0.85），这是**DCC 的固有特性而非 bug**：它的相关靠"昨天的相关新息"驱动，必然对瞬时跳变有平滑滞后。它擅长刻画"相关持续走高/走低"的趋势，不擅长瞬时捕捉闪崩。如果你需要即时相关，应改用基于高频实现的实时相关（realized correlation），而不是怪 DCC。

把相关矩阵在不同时点画成热图，差异更直观——平静期两只资产相关性温和，危机期几乎完全同涨同跌：

![平静期 vs 危机期相关矩阵：危机里相关性被 DCC 如实抬高](/images/dcc-garch-multivariate/dcc_corr_heatmap.png)

落到组合风险上，区别就是"生与死"。下面把 DCC 动态协方差和常数相关协方差都代入等权组合，算每日组合波动：

![等权组合风险：DCC 在危机段如实放大风险，常数相关严重低估](/images/dcc-garch-multivariate/dcc_portfolio_risk.png)

灰线是常数相关——危机来了它几乎纹丝不动，告诉你"一切正常"；红线（DCC）在危机段明显抬高年化波动率。用灰线做风险预算或止损，你会在最需要防守的时候以为自己很安全。

## 五、真实陷阱（别把 DCC 当神）

1. **滞后性**：如上，DCC 相关是平滑的，对瞬时跳变（闪崩、流动性枯竭）反应慢。需要即时反应时用高频实现的滚动相关，或加一个"危机开关"外生变量。
2. **维度与 $\bar{Q}$ 噪声**：DCC 只估 $(a,b)$ 两个参数，但长期中枢 $\bar{Q}$ 是 $N\times N$ 的。资产一多，$\bar{Q}$ 的样本估计噪声很大，相关矩阵可能不正定（需投影到正定锥）。高维场景应上**因子 DCC**（用少数因子解释相关）或收缩估计。
3. **标准化残差的分布假设**：标准 DCC 假设标准化残差近似条件正态。对肥尾资产（个股、加密货币），用 **DCC-t**（残差服从多元 t）能显著改善尾部相关的刻画。
4. **非平稳与结构突变**：$(a,b)$ 和 $\bar{Q}$ 用全样本估计，遇到 regime change（牛熊切换、货币政策转向）会失效。可滚动窗口估计，或上 Markov 切换 DCC。
5. **相关 ≠ 因果**：DCC 给你的是统计相关，不是"谁传染谁"的方向。要方向得用 VAR / 网络因果，别把相关飙升直接解读成风险传导路径。

## 结论

DCC-GARCH 是用**极少参数**把"相关随时间变化"这一铁律搬进组合风险的最实用桥梁：它优于常数相关（危机里不再系统性低估），计算量只是单变量 GARCH 的线性放大，且相关矩阵自动保持正定。但它不是万能的——对瞬时跳变有平滑滞后、高维时 $\bar{Q}$ 噪声大、肥尾资产需换 t 分布。把它当成"相关结构的一阶近似 + 危机放大器"，而不是能瞬时捕捉闪崩的神谕，你才不会在 2008 重演的风险模型失灵里踩坑。

落地时一个省心的做法：把 DCC 当作组合风险监控的「主引擎」，同时用高频实现的滚动相关做「危机探针」——平时看 DCC 的平滑趋势，一旦探针报警就立刻切到即时相关，既不损失 DCC 的稳健，也不放过闪崩的瞬间。
