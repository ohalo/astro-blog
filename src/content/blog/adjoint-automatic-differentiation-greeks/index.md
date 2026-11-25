---
title: "伴随自动微分 AAD 计算 Greeks：为什么一次反向扫描就能同时拿到 Delta 与 Vega、成本不随参数个数增长，又在路径不连续处失效"
description: "24 万条路径、16 步 GBM 的实测：AAD 一次前向加一次反向同时给出 Delta 0.61658、Vega 38.3580、Rho 50.9359，与 bump-and-revalue 的 0.61656 / 38.3582 对齐到 1.3e-5；而有限差分要 2P 次前向，算 16 个 Greeks 花 1746ms，AAD 恒定 108ms。但数字看涨的 pathwise 导数是零测集上的 0，AAD 报 0.0000，真值 0.01738 只能用似然比救回 0.01743。附完整 numpy 反向扫描代码与四张真实计算图。"
publishDate: '2026-11-26T03:50:00Z'
tags:
  - 量化
  - 期权定价
  - 自动微分
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

期权 Greek 的本质是价格对参数的导数。教科书教的做法是 bump-and-revalue：把 $S_0$ 往上挪一点、往下挪一点，各重跑一遍蒙特卡洛，再相减除以两倍的扰动。一个有 16 个风险因子的簿记，这意味着 32 次完整定价。当一次定价要模拟几十万条路径时，这笔算力账很难看。

伴随自动微分（Adjoint Automatic Differentiation，AAD）换了个思路：把定价程序看成一个计算图，前向算一遍价格并**记住每一步的中间量**，然后从输出往输入反向扫一遍，用链式法则把所有参数的导数一次性累积出来。关键结论是——反向扫描的成本与参数个数无关，只与前向的成本同阶。

我在固定种子 `20261126` 的合成实验里得到几个可以直接对拍的数字：

- $S_0=100,\ K=100,\ r=4\%,\ \sigma=22\%,\ T=1$，16 步、24 万条路径：AAD 的 Delta 为 **0.61658**，与中心差分的 **0.61656** 只差 **1.3e-5**；Vega 为 **38.3580**，中心差分 **38.3582**；Rho 为 **50.9359**；
-  Black-Scholes 解析 Delta 是 0.61479，AAD 高出 0.00179，相当于 **1.48 个标准误**（路径法 Delta 的 SE 为 0.00121），是抽样噪声而非方法偏差；MC 价格 10.7220 对解析 10.6888，也在 1.02 个标准误内；
-  单次前向定价用时中位数 **54.56ms**，一次反向扫描 **53.44ms**，AAD 合计约 **108ms**。中心差分算 2 个 Greek 要 109ms，和 AAD 持平；但算 16 个 Greek 要 32 次前向、约 **1746ms**，是 AAD 的 **16.2 倍**——这就是"成本不随参数个数增长"的具体含义；
-  扰动步长 $h$ 从 0.1 一路缩到 1e-4，中心差分的 Delta 在 0.6165645 到 0.6166354 之间几乎不动，说明连续 payoff 下 bump 窗口不是主要误差源；真正的问题是离散 payoff；
-  数字看涨（到期高于 $K$ 付 1 元）的 pathwise 导数是示性函数的导数，**几乎处处为 0**。AAD 老老实实给出 **0.0000**，而解析真值 **0.01738**。用似然比加权把示性函数的跳变"解析掉"，估到 **0.01743**（SE 5.2e-5），才把洞补上。

![价格与 Delta 的收敛](/images/adjoint-automatic-differentiation-greeks/fig1_convergence.png)

风险系统里 Greeks 不是装饰品。做市商盯 Delta 决定对冲多少现货，交易台盯 Vega 决定波动率敞口，风控盯 Rho 评估利率冲击。一张几百个合约的簿记，每个合约都要报出十几个希腊字母，而且必须在分钟级刷新。传统的 bump-and-revalue 每一次扰动都要重跑整条定价路径；当路径数上到十万量级、步长还要为不同参数分别设置时，计算量迅速变成瓶颈。更糟的是扰动步长本身是个玄学：设大了引入截断偏差，设小了被蒙特卡洛噪声淹没。AAD 的吸引力正在于它把求导变成一次确定性的反向遍历，没有步长要调，也没有额外的抽样噪声。

## 二、前向路径与反向扫描

把风险中性 GBM 离散成 16 步，每一步

$$
S_{i+1}=S_i\exp\!\Big[(r-\tfrac12\sigma^2)\Delta t+\sigma\sqrt{\Delta t}\,Z_i\Big]\equiv S_i E_i,\qquad \Delta t=T/16.
$$

前向循环只需保存每一步的 $S_i$ 与 $E_i$（或用 $Z_i$ 现算），内存是 $O(MN)$，反过来逐时刻回收。对一个可微 payoff $P(S_T)$，把折扣因子也记进种子：

$$
\bar S_T=\frac{e^{-rT}}{M}\,\frac{\partial P}{\partial S_T}.
$$

反向从第 $N-1$ 步走到第 0 步。因为 $\partial S_{i+1}/\partial S_i=E_i$，对 $S_i$ 的伴随量只要乘回去：

$$
\bar S_i \mathrel{+}= \bar S_{i+1}E_i.
$$

参数导数则在同一个循环里顺手累积。注意 $\log E_i=(r-\frac12\sigma^2)\Delta t+\sigma\sqrt{\Delta t}Z_i$，所以 $\partial S_{i+1}/\partial r=S_{i+1}\Delta t$，$\partial S_{i+1}/\partial\sigma=S_{i+1}(-\sigma\Delta t+\sqrt{\Delta t}Z_i)$：

$$
\bar r \mathrel{+}= \bar S_{i+1}S_{i+1}\Delta t,\qquad
\bar\sigma \mathrel{+}= \bar S_{i+1}S_{i+1}\big(-\sigma\Delta t+\sqrt{\Delta t}Z_i\big).
$$

循环结束时 $\bar S_0$ 就是 Delta，$\bar r$、$\bar\sigma$ 就是 Rho 和 Vega——**三条导数共用同一次反向扫描**。还有一个容易漏掉的项：折扣因子 $e^{-rT}$ 自己依赖 $r$，必须额外减去 $T\cdot \text{price}$，否则 Rho 会虚高约一个价格（本例会变成 61.76 而不是 50.94）。

```python
import numpy as np

rng = np.random.default_rng(20261126)
M, N = 240_000, 16
S0, K, r, sig, T = 100., 100., 0.04, 0.22, 1.
dt = T / N
Z = rng.standard_normal((M, N))

def forward(s0, v):
    E = np.exp((r - .5*v*v)*dt + v*np.sqrt(dt)*Z)
    S = np.empty((M, N+1)); S[:, 0] = s0
    for i in range(N):
        S[:, i+1] = S[:, i] * E[:, i]
    return S, E

def aad(S, E):
    # 种子：e^{-rT}/M * dP/dS_T，看涨为示性函数
    g = np.full(M, np.exp(-r*T)/M) * (S[:, -1] > K)
    acc_r = acc_v = 0.0
    for i in range(N-1, -1, -1):
        Si1 = S[:, i+1]
        acc_r += np.dot(g, Si1) * dt
        acc_v += np.dot(g*Si1, -sig*dt + np.sqrt(dt)*Z[:, i])
        g = g * E[:, i]                       # 链式法则回传
    acc_r -= T * np.exp(-r*T) * np.mean(np.maximum(S[:, -1]-K, 0))  # 折扣因子
    return g.sum(), acc_r, acc_v              # Delta, Rho, Vega
```

这段代码里没有一次"扰动再定价"，导数全部由链式法则解析累积。

理解反向扫描的关键是把它和计算图联系起来。前向时每一步 $S_{i+1}=S_iE_i$ 都是一个乘法节点，把 $S_i$ 和 $E_i$ 两个输入接到输出；反向时，每个节点把自己输出的伴随量乘以局部导数，再分别回传给两个输入。因为定价路径是一条链，回传不会分叉，所以一次反向就能把这条路径对所有输入参数的导数全部累积完——这和反向传播训练神经网络是同一套数学，只是这里的"网络"是沿时间展开的定价循环。代价是要保存 $M\times(N+1)$ 个状态供反向使用（本例 24 万×17 个 double，约 33MB），是典型的内存换时间。如果路径数再上一个量级，就得用 checkpointing：只存每隔若干步的状态，反向时局部重算，把内存压到 $\sqrt N$ 量级而多付一倍前向成本。

顺便说清 AAD 在自动微分谱系里的位置。符号微分直接对公式做代数化简，表达式会指数膨胀，对 16 步循环展开后的路径根本不实用；前向模式（tangent mode）每行代码带一个导数分量，求 $P$ 个参数就要跑 $P$ 遍前向，成本和 bump-and-revalue 同阶，只是没有截断偏差；反向模式（adjoint mode）只需两遍——一遍前向记中间量、一遍反向累积——成本与参数个数无关，这正是 Greeks 场景需要的。唯一的代价是必须保留中间量或做 checkpointing，而前向模式流式一遍过。所以当只对一两个参数求导时前向模式反而更省内存，只有当参数多起来，AAD 的优势才压倒性地显现。

## 三、与 bump-and-revalue 对拍

中心差分用同一组随机数（common random numbers）重跑前向，把噪声抵消掉：

```python
def price(S):
    return np.exp(-r*T) * np.mean(np.maximum(S[:, -1]-K, 0.))

hS, hv = 0.05, 1e-4
Su, _ = forward(S0+hS, sig); Sd, _ = forward(S0-hS, sig)
Vu, _ = forward(S0, sig+hv); Vd, _ = forward(S0, sig-hv)
fd_delta = (price(Su)-price(Sd)) / (2*hS)
fd_vega  = (price(Vu)-price(Vd)) / (2*hv)
```

![扰动步长敏感性](/images/adjoint-automatic-differentiation-greeks/fig2_bumps.png)

两种算法在 Delta 上差 1.3e-5，在 Vega 上差 2.7e-4，可以认为互为验证。Rho 的 50.9359 也落入解析值（$Ke^{-rT}T\Phi(d_2)$，约 50.8）附近。这里要提醒一句：CRN 让有限差分看起来"很准"，是因为它把抽样噪声消掉了；实盘里如果对扰动用了不同的随机种子，差分估计的方差会大得多，而 AAD 天然没有这个问题。

为什么 CRN 如此重要？中心差分的分子是两个高度相关的蒙特卡洛估计量之差，若两次定价共享同一组 $Z_i$，路径层面的噪声在相减时大部分抵消，剩下的只是 payoff 对扰动的局部变化；若用独立随机数，相减反而把两份噪声叠加，标准误可能比单个价格还大。代价是 CRN 要求你能控制底层随机流——在有些闭源定价库里做不到，这时有限差分的方差会成为真实瓶颈，AAD 的相对优势比本例还要大。

还有一个容易被忽视的数字：AAD 的 Delta 0.61658 比 BS 解析值 0.61479 高 0.00179，看起来"偏了"。但路径法 Delta 的标准误是 0.00121，偏差只有 1.48 倍标准误——这是抽样波动，不是方法误差。判断 Monte Carlo 类估计器是否可信，永远要看标准误或 z 分数，不要只看小数点后几位的绝对差。

代价对比更值得看。前向 54.56ms、反向 53.44ms，AAD 合计约等于两次前向；中心差分每多一个参数就多两次前向。当风险因子 $P$ 从 1 增到 16，差分成本线性涨到 1746ms，AAD 仍是 108ms。对一张有几十个风险维度的衍生品簿记，这个差别决定了一分钟风险快照能不能算完。

![Greeks 数量与计算成本](/images/adjoint-automatic-differentiation-greeks/fig3_cost.png)

## 四、失效边界：不连续 payoff

AAD 的前提是 payoff 对状态变量可微。数字看涨的 payoff 是 $\mathbf 1\{S_T>K\}$，它在 $S_T=K$ 这一点的导数不存在，其余地方导数为 0。反向扫描把 0 一路乘回输入端，结果自然是 0——**这不是 bug，而是 pathwise 导数本身在几乎处处等于 0**。

![数字期权的失效与似然比修正](/images/adjoint-automatic-differentiation-greeks/fig4_digital.png)

修正办法是换估计器：不要对 payoff 求导，而是对转移密度求导。令 $X=\log S_T$，它服从均值 $\log S_0+(r-\frac12\sigma^2)T$、方差 $\sigma^2T$ 的正态，对数密度的 $S_0$ 分数为 $(X-\mu)/(\sigma^2T S_0)$。于是

$$
\Delta_{\text{LR}}=e^{-rT}\,\mathbb E\!\left[\mathbf 1\{S_T>K\}\,
\frac{X-\mu}{\sigma^2 T S_0}\right].
$$

实测得到 **0.01743**（SE 5.2e-5），与解析 0.01738 吻合。障碍期权、触碰期权等带指示函数的产品都落在同一类：要么用似然比，要么把 payoff 平滑化，否则 AAD 会安静地报 0。

## 五、实践清单

1. 前向必须保存每一步的 $S_i$ 和 $E_i$（或 $Z_i$），内存换时间；
2. 反向循环里参数导数与状态伴随量**同时**累积，不要分开两遍扫；
3. 折扣因子、股息率、时间参数都依赖 $r$ 或其他输入，别只盯路径内部；
4. 对指示型 payoff 先做可微性检查，必要时切换似然比或平滑近似；
5. 用 CRN 差分做单元测试是可行的，但不要把它的低方差当成 AAD 也会有的性质。

工程上还有一层：AAD 的反向扫描要求前向把每一步的状态都留在内存里，这对长路径、大组合的内存占用并不友好。生产系统通常做两件事——一是把 $Z_i$ 按固定种子确定性生成，反向时直接重算而不存储；二是分块处理路径，把 $M$ 拆成若干批，每批独立做前向与反向，只在最后汇总导数均值。这样既保留了一次反向拿全部 Greeks 的性质，又把峰值内存控制在可控范围。

再往深一层看，AAD 真正改变的是"导数供给"的边际成本。当 Greeks 便宜到近乎免费，很多此前算不起的东西变得可行：整张隐含波动率曲面同时对几十个期权反解校准参数、组合层面直接对风险因子做敏感度归因、把 Greek 约束塞进优化目标做动态对冲。反过来，如果只在乎一个 Delta，手工公式或一次前向差分永远更简单——工具的价值要看问题的规模。

AAD 不是魔法：它只是把链式法则写进了循环。但当 Greeks 的数量乘以路径数把计算预算吃光时，"一次前向 + 一次反向"确实是唯一能把成本钉住的做法。
