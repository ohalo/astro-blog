---
title: "向量自回归 VAR：用矩阵把多资产联动写成一个模型"
description: "单变量模型看不见股债跷跷板与波动率聚集这类系统级联动。向量自回归 VAR 用矩阵把 N 个变量平等写进一个方程，从零实现 OLS 估计、Granger 因果、正交化脉冲响应与预测误差方差分解。本文用 2000 个交易日的合成三变量金融系统（股指收益、国债收益率变化、波动率变化），植入『equity 单向带动 bond 与 vol』的因果链：A 矩阵恢复 RMSE 仅 0.018；equity→bond 的 Wald F=59.1(p≈2e-14)、equity→vol 的 F=314(p≈0) 显著，反向均不显著；10 日窗口里 vol 预测误差的 24.9% 来自 equity 冲击。附完整 numpy 实现与五张真实计算图。"
publishDate: '2026-10-26'
tags:
  - 量化交易
  - 时间序列
  - 向量自回归
  - Granger 因果
  - Python
language: Chinese
difficulty: advanced
---

做多变量时间序列的人迟早会遇到一个尴尬：你拟合了十个单变量 ARIMA，却发现十个序列之间的联动全是黑箱。股债跷跷板、波动率聚集、危机时的相关性飙升——这些"系统级"现象，单变量模型天生看不见。向量自回归（Vector Autoregression, VAR）就是为此而生的：它不做任何结构假设，只是把所有变量"平等"地写进一个矩阵方程，让数据自己说出谁领先、谁滞后、谁被谁带动。

这篇我们用一个三变量金融系统——股指日收益、国债收益率变化、波动率变化——从零跑一遍 VAR(1)：OLS 估计、Granger 因果检验、正交化脉冲响应（IRF）、预测误差方差分解（FEVD），并刻意植入一条"equity 带动 vol 与 bond"的因果链，看 VAR 能不能把它找回来。数据是 2000 个交易日的合成市场，参数已知，方便我们核对恢复精度。

结论先放这：

① 参数恢复很干净。OLS 把 A 矩阵 9 个参数恢复的 RMSE 只有 0.018，植入的非零项（equity→bond 0.15、equity→vol 0.40）估计为 0.135 与 0.411，植入的零项（bond→equity、vol→equity）都落回 0 附近。线性联立方程在平稳数据上就是这么好估。

② Granger 因果方向正确。equity 对 bond 的 Wald F=59.1（p≈2×10⁻¹⁴）、对 vol 的 F=314（p≈0），显著；反向 bond→equity、vol→equity 都不显著。我们埋的结构被原样挖出。

③ 脉冲响应与方差分解给出相同故事：1 单位 equity 冲击让 vol 当期抬升约 0.4 个标准差并逐期衰减；10 日窗口里，vol 预测误差的 24.9% 来自 equity 冲击，自身只占 75.0%——波动的"自身惯性"很强，但 equity 是第二大来源。

## 模型：一个矩阵装下整个系统

VAR(p) 的形式极简：

Y_t = c + A_1 Y_{t-1} + … + A_p Y_{t-p} + ε_t,    ε_t ~ WN(0, Σ)

其中 Y_t 是 N×1 向量，A_i 是 N×N 系数矩阵，A_i[i,j] 度量"变量 j 的滞后一阶对变量 i 当期的影响"。当 p=1 时它等价于一个向量化的 AR(1)；更高阶可以靠"伴随形式"压成 VAR(1) 来统一处理。它之所以称为"无约束"（reduced-form），是因为它不区分内生的因果机制，只描述联合动态——好处是估计闭式可得，坏处是系数本身没有结构含义。

下面就是本文全部数值的实验设置：三变量、VAR(1)、T=2000，并设定 true A 使 equity 单向带动 bond 与 vol：

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(20261026)
N, T = 3, 2000
# 设计：equity 单向 Granger 带动 bond 与 vol
A_true = np.array([[0.10, 0.00, 0.00],
                   [0.15, 0.20, 0.00],
                   [0.40, 0.00, 0.30]])
c_true = np.array([0.0003, -0.0001, 0.0002])
Sigma = np.array([[1.00, 0.10, 0.20],
                  [0.10, 0.60, 0.05],
                  [0.20, 0.05, 1.00]]) * 1e-3
L = np.linalg.cholesky(Sigma)
Y = np.zeros((T, N))
eps = rng.standard_normal((T, N)) @ L.T
for t in range(1, T):
    Y[t] = c_true + A_true @ Y[t-1] + eps[t]
```

![三变量金融系统的合成序列](/images/vector-autoregression-var/fig1_sim.png)

## 估计：用 OLS 一次解出全部系数

VAR(1) 的估计没有任何魔法——它就是每个方程各自做一次带常数项的多元回归，再把结果拼成矩阵。把 Y_{t-1}（含截距）横向铺开成设计矩阵 X，Y_t 纵向铺开成响应，最小二乘闭式求解：

```python
Ylag = Y[:-1]            # Y_{t-1}
Ynow = Y[1:]             # Y_t
X = np.column_stack([np.ones(Ylag.shape[0]), Ylag])     # (T-1)×(1+N)
B = np.linalg.lstsq(X, Ynow, rcond=None)[0]             # (1+N)×N
A_est = B[1:].T          # 转置回来：A_est[i,j] = 变量j滞后对变量i当期的效应
resid = Ynow - X @ B
Sig_est = resid.T @ resid / (X.shape[0] - X.shape[1])
```

注意一个容易翻车的细节：最小二乘返回的系数矩阵是"按响应变量分块"的，要转置一次才能得到 A[i,j] 的标准语义。我第一版就漏了这步，把 A 当成它的转置去画图，Granger 方向全反。估计结果 RMSE(A)=0.018，下图的左/右两块热图对比真实与估计，肉眼几乎一致。

![A 矩阵参数恢复](/images/vector-autoregression-var/fig2_recovery.png)

## Granger 因果：谁真的"领先"谁

Granger 因果不声称哲学意义上的因果，只问"把 X 的滞后加进 Y 的方程，能否显著降低残差方差"。做法是对 Y_i 的方程做 Wald 检验：约束掉 X 的滞后项，看受限与无约束的残差平方和之比（F 统计量）。

```python
def granger_stat(Ynow, Ylag, i, j):
    Tn = Ylag.shape[0]
    X1 = np.column_stack([np.ones(Tn), Ylag])
    b1 = np.linalg.lstsq(X1, Ynow[:, i], rcond=None)[0]
    rss1 = Ynow[:, i] @ Ynow[:, i] - b1 @ X1.T @ Ynow[:, i]
    cols = [0, 1, 2, 3]; cols.remove(1 + j)   # 去掉 X 的滞后列
    X0 = X1[:, cols]
    b0 = np.linalg.lstsq(X0, Ynow[:, i], rcond=None)[0]
    rss0 = (Ynow[:, i] - X0 @ b0) @ (Ynow[:, i] - X0 @ b0)
    F = ((rss0 - rss1) / 1) / (rss1 / (Tn - X1.shape[1]))
    p = 1 - stats.f.cdf(F, 1, Tn - X1.shape[1])
    return F, p
```

结果（F, p）：equity→bond 59.1 / 2.3e-14，equity→vol 314 / ≈0，bond→equity 1.12 / 0.29，vol→equity 0.13 / 0.72，bond→vol 0.65 / 0.42。正向显著、反向不显著，和我们植入的结构完全一致——这正是 Granger 检验的用武之地。

![Granger 因果 Wald 检验](/images/vector-autoregression-var/fig3_granger.png)

## 脉冲响应：一个冲击要多久才散去

IRF 回答"给变量 j 一记标准差冲击，变量 i 在 h 期后变成什么样"。先把 VAR(1) 展开成滑动平均 Ψ_k = A^k，再对残差协方差做 Cholesky 正交化，得到互相排斥的冲击：

```python
P = np.linalg.cholesky(Sig_est)          # Σ = P P'
Psi = np.zeros((21, N, N)); Psi[0] = np.eye(N)
for k in range(1, 21):
    Psi[k] = A_est @ Psi[k-1]
Psi_o = np.einsum("kij,jl->kil", Psi, P) # 正交化 IRF
```

下图是一个 equity 冲击对三个变量的响应：vol 当期就被推高约 0.4 个标准差，随后以 A[2,2]=0.31 的速度衰减；bond 也被带动约 0.13；equity 自身则快速回归（A 对角线都小）。这正是"黑天鹅来自股市、波动率的痛苦更久"的量化画像。

![正交化脉冲响应：equity 冲击](/images/vector-autoregression-var/fig4_irf.png)

## 方差分解：谁在解释谁的预测误差

FEVD 把 h 步预测误差按冲击来源切分。10 日窗口上，vol 误差的 75.0% 来自自身冲击、24.9% 来自 equity、几乎 0 来自 bond；bond 误差 94.7% 来自自身、5.3% 来自 equity；equity 几乎 100% 来自自身。也就是说，在这个系统里波动率的"自惯性"最强，但 equity 是它第二重要的驱动——单看 vol 的 AR 模型会系统性漏掉这一点。

![波动率预测误差分解](/images/vector-autoregression-var/fig5_fevd.png)

## 诚实的结论与边界

VAR 的优点是"无结构假设、闭式估计、因果/响应/分解一条龙"；但它有清楚的边界：① 它假设序列平稳，非平稳（含单位根）必须先差分或用 VECM，否则系数是假的；② 正交化 IRF 依赖 Cholesky 分解的变量排序，对顺序敏感，稳健做法是做排列敏感性分析；③ 它只刻画线性联动，危机时相关性的非线性飙升（regime switch）不在它的语言里；④ 高维（N 大）时自由度迅速吃紧，需要收缩或因子结构。下一篇我们用协整与误差修正模型，专门处理"几个 I(1) 序列长期锁在一起"的那种平稳化。

---
本内容仅为信息整理与分析参考，不构成投资建议，投资有风险，决策需谨慎。
