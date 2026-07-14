---
title: "CoVaR 与系统性风险：一只机构倒下会拖垮多少同行"
description: "CoVaR（Adrian-Brünnermeier）把「系统性风险」从一句话变成可量化指标：CoVaR_i = 机构 i 陷入自身 VaR 困境时，系统的条件风险价值；ΔCoVaR_i = CoVaR_i − 常态下系统 VaR，即机构 i 对系统的边际贡献。用 6 家机构的因子关联仿真：银行A（关联度最高）ΔCoVaR 最大(−3.34)，信托F 最小(−2.55)，与系统关联度严丝合缝。附完整 Python 与六类真实陷阱（高阶）。"
publishDate: '2026-07-15'
tags:
  - 量化交易
  - 系统性风险
  - CoVaR
  - 风险管理
  - 条件风险价值
  - 压力测试
  - 监管
  - Python
language: Chinese
difficulty: advanced
---

2008 年雷曼倒下时，没人问「雷曼自己亏多少」，大家问的是「雷曼倒了，会连累多少家」。**传统的 VaR 只回答前者**——一家机构自身的潜在损失。它天生是「单体视角」，对「我的亏损会传染给谁、别人倒了我又亏多少」完全失明。CoVaR（Conditional VaR，Adrian & Brünnermeier, 2016）干的就是这件事：把风险从「单体」拉到「系统」，量化**「当机构 i 陷入困境时，整个系统的风险价值是多少」**。

结论先放这：**用 6 家机构（银行A/B、券商C、保险D、基金E、信托F）、各与系统因子的关联度 β=(0.85,0.78,0.65,0.55,0.45,0.30) 仿真 4000 期（含 15% 危机 regime）。系统自身 VaR(5%) = −1.70；当机构 i 跌到自身 VaR 困境时，系统的条件 VaR 即 CoVaR_i。算出边际贡献 ΔCoVaR_i = CoVaR_i − 常态系统 VaR 为 [−3.34, −3.23, −3.20, −3.08, −3.07, −2.55]（银行A→信托F），与关联度 β 严丝合缝——关联越紧、越系统重要。** 顺序恰好是 ΔCoVaR：银行A(−3.34) > 银行B(−3.23) > 券商C(−3.20) > 保险D(−3.08) > 基金E(−3.07) > 信托F(−2.55)。两两 CoVaR 矩阵进一步显示：银行A 倒下时，其他银行的系统性损失最深（−5.64），而信托F 倒下对同行冲击最浅（−2.98）。分位回归的 β(τ) 在极端分位仍为正（尾端 0.64 vs 中位 0.60），印证传染在危机态不减弱。

![ΔCoVaR 排序：谁倒了最要命，银行A 系统性最强](/images/covar-systemic-risk/covar_delta_bars.png)

## 一、VaR 的盲区：它只管自己

VaR(q) 的经典定义：在置信水平 1−q 下，未来某段时间内最大损失不超过 VaR。对机构 i：

```
P( R_i ≤ VaR_i(q) ) = q
```

它回答「我亏多少」。但它**不回答**：当我亏到 VaR 时，市场其他部分怎样？别人亏到 VaR 时，我又怎样？这正是系统性风险的要害——风险在机构之间是**相关、且时变、且危机时飙升**的。2008 的教训就是：单体 VaR 全绿，系统却塌了。

## 二、CoVaR 的定义：把「条件」钉进 VaR

记系统（或市场组合）收益为 `R_sys`，机构 i 收益为 `R_i`。CoVaR 的定义是：

```
CoVaR_i(q) = VaR_q( R_sys | R_i = VaR_i(q) )
```

翻译：**「在机构 i 恰好处于它自己 VaR 困境（左尾 q 分位）的条件下，系统的风险价值」**。注意 CoVaR 量的是**系统**的损失，不是机构 i 自己的。

更干净的是**边际贡献 ΔCoVaR**：

```
ΔCoVaR_i = CoVaR_i(q) − VaR_q( R_sys | 常态 )
```

`ΔCoVaR_i` 就是「机构 i 陷入困境，给系统额外增添了多少风险」——这正是监管想要的「系统重要性」单一数字。

## 三、仿真数据：用因子关联度制造 contagion

为了干净演示，构造 6 家机构收益，全部由一个共同系统因子 F 加特质噪声驱动，关联度 β 递减（银行最高、信托最低）：

```python
import numpy as np

rng = np.random.default_rng(20260715)
N, T = 6, 4000
beta = np.array([0.85, 0.78, 0.65, 0.55, 0.45, 0.30])   # 与系统因子的关联度
names = ["银行A", "银行B", "券商C", "保险D", "基金E", "信托F"]

# regime switching：85% 常态 + 15% 危机（系统因子左移且更肥尾）
regime = rng.choice([0, 1], size=T, p=[0.85, 0.15])
F = np.where(regime == 0, rng.normal(0, 1, T), rng.normal(-1.6, 2.2, T))
eps = rng.normal(0, 1, (T, N))
R = beta * F[:, None] + np.sqrt(1 - beta**2) * eps     # 每家机构收益
R_sys = R.mean(axis=1)                                  # 等权系统指数
```

关键点：β 越大，机构越「随系统共舞」。危机 regime 里 F 整体左移，高 β 机构跌得更狠——这就是 contagion 的来源。下面所有结论都应能被 β 的顺序预测。

## 四、计算 CoVaR：经验条件分位（AB 原始定义）

最稳健的实现就是 Adrian-Brünnermeier 的**经验法**：先把机构 i 陷入困境的样本筛出来（R_i ≤ 自身 VaR），再在这些样本里取系统收益的分位。

```python
q = 0.05
var_i = np.quantile(R, q, axis=0)                 # 各机构自身 VaR（左尾 5%）
sys_var = np.quantile(R_sys, q)                   # 系统自身 VaR

covar_q, covar_med = [], []
for i in range(N):
    distress = R[:, i] <= var_i[i]                # 机构 i 陷入困境的样本
    covar_q.append(np.quantile(R_sys[distress], q))   # 困境态下系统的条件 VaR
    normal = np.abs(R[:, i] - np.median(R[:, i])) <= 0.15 * R[:, i].std()
    covar_med.append(np.quantile(R_sys[normal], q))   # 常态下系统的条件 VaR

delta = np.array(covar_q) - np.array(covar_med)  # ΔCoVaR
```

结果（ΔCoVaR，按机构 β 顺序）：

```
银行A  −3.34    银行B  −3.23    券商C  −3.20
保险D  −3.08    基金E  −3.07    信托F  −2.55
```

**完全符合 β 的顺序**：关联最紧的银行A 系统性最强，关联最松的信托F 最弱。这就是 CoVaR 想抓的信号——「大而不能倒」被翻译成了一个单调、可比的数字。

![条件分布：当银行A 跌到 VaR 时，系统收益整体左移，CoVaR 远小于系统 VaR](/images/covar-systemic-risk/covar_conditional_distribution.png)

## 五、条件分布图：困境时系统整条左移

把「银行A 困境」那批样本的系统收益画成直方图，和全样本对比：全样本系统收益居中（VaR=−1.70，蓝虚线），而条件样本明显左移——CoVaR=−4.03（红实线），比系统自身 VaR 深得多。这说明**一家机构的困境不是孤立事件，而是把整个系统的损失分布往尾部拽**。

## 六、两两 CoVaR：传染的方向与强度

监管不只关心「系统重要性」，还关心「谁传染谁」。把「行机构 j 陷入困境」作为条件，看「列机构 k 收益」的条件 VaR，得到 N×N 矩阵：

```python
pair = np.zeros((N, N))
for j in range(N):
    mask = R[:, j] <= var_i[j]
    for k in range(N):
        pair[j, k] = np.quantile(R[mask, k], q)
```

![两两 CoVaR 矩阵：行机构倒下时，列机构的系统性损失（越蓝越深）](/images/covar-systemic-risk/covar_network_heatmap.png)

读图：银行A 倒下时，同行银行/券商的损失最深（约 −5.6），信托F 倒下对同行冲击最浅（约 −3.0）。**对角线最浅（自己困境时自己本来就最惨，但「系统视角」看的是对「其他」的溢出），而非对角线的深浅揭示了传染网络的结构性不对称**——银行是网络的「震中」，信托是「边缘节点」。

## 七、分位回归视角：CoVaR 的态依赖与非线性

CoVaR 也可用**分位回归**估计：`R_sys = a(τ) + b(τ)·R_i + ε`，取 τ=q 分位回归的拟合值即 CoVaR。好处是能量出**敏感度 β(τ) 随分位的变化**——如果 β(τ) 在极端 τ 更大，说明传染在危机态更猛（非线性）。

```python
def pinball(u, tau):
    return np.where(u >= 0, tau * u, (tau - 1) * u)

def quantile_regression(y, x, tau, rng):
    from scipy.optimize import minimize
    loss = lambda p: np.sum(pinball(y - (p[0] + p[1]*x), tau))
    best, bv = None, 1e18
    for x0 in [[np.median(y), 0.0], [np.quantile(y, tau), 0.0]]:
        r = minimize(loss, x0, method="Nelder-Mead")
        if r.fun < bv: best, bv = r.x, r.fun
    return best

i = 0  # 银行A
taus = np.linspace(0.05, 0.95, 19)
slopes = [quantile_regression(R_sys, R[:, i], tau, rng)[1] for tau in taus]
```

模拟里 β(τ) 在尾端约 0.64、中位约 0.60，始终为正——印证传染是普遍且危机不减弱的。这条曲线本身就是监管仪表盘的好素材。

![分位回归斜率 β(τ)：银行A 对系统的传导随分位变化（非线性）](/images/covar-systemic-risk/covar_quantile_regression.png)

## 九、从 CoVaR 到 SRISK：监管怎么用这套语言

CoVaR 的直系后代是 **SRISK**（Brownlees & Engle, 2017，纽约联储实际采用）。它把 ΔCoVaR 的思路推进成「一家机构在危机时的资本短缺」：

```
SRISK_i = max( k·D_i + (1−k)·E_i − (1−Pr)·A_i·(μ_m + β_i·(C_m − μ_m)) , 0 )
```

其中 `D/E/A` 是负债/权益/资产，`β_i` 是机构对市场的敏感度，`C_m` 是危机期市场损失，`Pr` 是资本充足率。可见 SRISK 的本质仍是 CoVaR 的灵魂——**用 β（关联度）把「单体」放大成「系统在危机下的溢出」**。我们仿真里 β 的顺序（银行>信托），正是 SRISK 排序的主驱动：高 β 机构在 `C_m` 大跌时缺口最大。

这给基金经理一个直接可用的漏斗：**先看 ΔCoVaR 排序挑出系统重要性最高的几家 → 再查其两两 CoVaR 矩阵确认传染方向 → 把这几家放进压力测试的核心情景**。比「凭名声列系统性重要银行」更可量化。

```python
# 把本篇的 delta（ΔCoVaR 向量）直接当成 SRISK 的代理排序即可
order = np.argsort(delta)   # 系统重要性由高到低
```

## 十、真实陷阱（六类）

1. **把 CoVaR 当机构自身损失**：CoVaR_i 量的是**系统**在「i 困境」下的风险，不是 i 自己的。汇报 ΔCoVaR 时务必说清「这是给系统增添的风险」，否则监管报告会误导。
2. **分位回归的共线性与方差**：QR 在 τ 极端（如 1%）时样本极少、系数方差爆炸，β(τ) 估计噪声大。本文用 N=4000 还算稳，真实月频数据（几十年×12）样本薄，尾端 QR 极易失真——务必用 bootstrap 给 QR 系数加置信带。
3. **条件筛选的稀疏性**：「R_i ≤ VaR_i(q)」在 q=5% 时只有 5% 样本，CoVaR 估计方差大。可放宽到 q=10% 或对困境样本做平滑（如 R_i 落在 [VaR, VaR+ε] 邻域），但要在报告里标注口径。
4. **因子关联度的内生性**：我们用手设 β 制造 contagion，真实世界里 β 本身在危机时飙升（杠杆循环、margin call）。若用全样本 β 估 CoVaR，会**低估**危机传染。正确做法是分 regime 或用时变 β（如 DCC-GARCH），否则「平静期 CoVaR」在危机里完全失效。
5. **幸存者偏差**：用的是 6 家「还在牌桌上」的机构。系统性风险恰恰是「谁会先出局」——若样本剔除了已倒闭者，CoVaR 系统性低估。监管口径必须含困境/退市的边际机构。
6. **非线性与跳跃未被捕获**：纯线性 QR 假设「i 跌 1% 系统跌 β%」处处成立。但真实传染有门槛（某机构 CDS 触发后才引爆连锁）和跳跃（闪崩）。CoVaR 是线性近似，对**跳跃式 contagion** 不敏感——这时应补一个「网络/跳跃」模型（见两两矩阵里的非对角不对称，已经露出端倪）。

---

**小结**：CoVaR 把「系统重要性」从一个形容词变成了一个可比、可排序、可网络化的数字。它的灵魂是「条件」——不是你亏多少，而是**你亏的时候，别人亏多少**。分支比 α 决定 Hawkes 订单流会不会自激失控（上篇），而 ΔCoVaR 决定哪只机构倒下会让系统最疼（本篇）——两者一个管「微观传染的速度」，一个管「宏观传染的结构」。
