---
title: "量子计算期权定价：QAE 与 QAMC 平方加速的数值解剖"
description: "量子计算真能加速期权定价吗？本文用 numpy 态矢量/密度矩阵精确模拟完整 QAE 电路（A 算子、Grover、受控-Q、逆 QFT），实测 Grover 放大与模拟误差 3.2e-15、QPE 量化地板 0.00616、log-log 斜率 −1.00 对 −0.50（同精度省 43.8 倍查询）、加噪扫描下优势未被吃光；再用 Heston 亚式期权验证 QAMC：Sobol 加速比 11.2×→1.93×，Montanaro Thm 2.5 与 Blanchet-Szegedy-Wang 2024 显示平方优势恒存、票价随尾部浮动。含全部可运行代码与五张真实计算图。"
publishDate: '2026-10-27'
tags:
  - 量化交易
  - 量子金融
  - 量子幅度估计
  - QAE
  - QAMC
  - 蒙特卡洛
  - Heston模型
  - Python
language: Chinese
difficulty: advanced
---

"量子计算机能把蒙特卡洛定价加速一个平方"——这句话你在每个金融科技峰会的 PPT 上都见过，但它既没说清加速从哪来，也没说清要付什么代价。本文把它拆开验证：两条路线、五个实验、全部数字来自本机 numpy 现场计算（seed=20261027），量子部分不是调包，是把 A 算子、Grover 算子、受控-$Q^{2^j}$、逆 QFT 逐矩阵搭出来的态矢量/密度矩阵精确模拟。

结论先行：

**① 幅度放大是真的。** 2 步二叉树数字看涨期权（$S_0=100, u=1.25, d=0.8, p_{up}=0.55, K=106$），行权概率 $a=p^2=0.3025$ 解析已知，给量子模拟当判卷标准。Grover 迭代 $j$ 次后测得 flag 概率与 $\sin^2((2j+1)\theta)$（$\sin^2\theta=a$）最大偏离 **3.2e-15**，理论最优 $j=9$ 处把 0.3025 推到 **99.52%**。

**② QPE 读出真实存在量化地板。** $m=4$ 计数寄存器，质量 99.6% 集中在镜像双峰 $c=3/13$（理论峰位置 $\theta\cdot16/\pi=2.97$），读出 $\hat a=0.3087$，绝对误差 **0.00616**。这个误差不是噪声，是 $\theta/\pi\approx0.1854$ 离最近二进制分数太远的**量化误差**：$m$ 从 4 加到 7，误差纹丝不动。

**③ 平方加速在 log-log 图上是两条直线的夹角。** 实测 QAE 误差随查询数斜率 **−0.9996**（理论 −1），经典 MC 斜率 **−0.5030**（理论 −0.5）。达到 $m=7$（127 次查询）的误差 0.00616，MC 需要约 **5564 次**采样——**43.8 倍**差距，且这个差距随精度要求线性放大。

**④ 噪声会砸场子，但砸不出"量子没用"。** 密度矩阵对 sys 寄存器逐比特加去极化噪声扫到 $\varepsilon=0.5$：干净线在 $\varepsilon\le0.2$ 误差纹丝不动（0.00616），$\varepsilon=0.35$ 谱崩（跳到 0.1975）；$q_{flip}=0.01$ 的线同期崩到 0.3025。但**全部曲线、全部噪声水平，没有一条追平"等查询数 MC"的 0.4487**——优势被噪声吃掉一大截，没有被吃光。

**⑤ 量子优势的真面目：一个恒为平方、但票价随尾部浮动的红利。** Heston 亚式期权上，随机化 Sobol 对伪随机 MC 的 RMSE 加速比从 $d=2$ 的 **11.2×** 单调衰减到 $d=64$ 的 **1.93×**——从未归零，但那是多项式级的体面。量子这边按定理算票：收益方差有界时，Montanaro 2015 给出量子 Õ(σ/ε) 对经典 $(z\sigma/\varepsilon)^2$，$\varepsilon=10^{-5}$ 处省 **4.3 个数量级**；换成方差无限的 t(1.8) 厚尾代理，Blanchet-Szegedy-Wang 2024 显示经典和量子的复杂度指数都从 $(1+\delta)/2\delta$ 翻倍到 $(1+\delta)/\delta$（2.43 对 1.21）——**优势仍是平方（省 6.8 个数量级），但双方的门票一起变贵**。尾部决定票价档位，不决定优惠比例。

## 下半场的规则：两条路线不是一回事

先划清边界，很多科普文章在这里搅浆糊：

- **QAE（量子幅度估计）**加速的是"估计一个概率"本身。对欧式/数字期权，把收益期望编码成量子态的一个幅度，$n$ 条路径的采样查询复杂度从 $O(1/\varepsilon^2)$ 降到 $O(1/\varepsilon)$——**平方加速，与路径分布无关**，但电路要装下整棵树，且需要容错量子比特。
- **QAMC（量子加速蒙特卡洛，Montanaro 2015）**不要求整棵树进电路，它把定价封装成"输出随机变量的量子子程序"，用量子均值估计把有界方差情形的查询数从 $O(\sigma^2/\varepsilon^2)$ 压到 $\tilde O(\sigma/\varepsilon)$——同样是平方，但适用范围推广到任意有界方差的收益函数，这也是下文实验五算票价的依据。它是"QAE 的推广 + 更弱的电路假设"，不是另一个东西。

一个粗暴的心智模型：MC 的误差是 $1/\sqrt N$，QAE 是 $1/N$，QAMC 在方差有界时也是 $1/N$（差 polylog），收益厚尾到方差无限时双方各自的幂都变陡、但量子-经典的**指数比值恒为 2**。别背公式，看图 2 和图 4 的斜率就行。

## 实验一：把 QAE 电路逐矩阵搭出来

场景故意选到"可以手算判卷"：2 步二叉树，四条路径价格 $[156.25, 100, 100, 64]$，数字看涨 $K=106$ 只有 $uu$ 一条路径行权，$a=0.55^2=0.3025$。寄存器 $n=m+4$ 位：低位 4 位是系统寄存器 $[q_1, q_2, \text{payoff}, \text{flag}]$，高位 $m$ 位是计数寄存器。

**A 算子**制备叠加：两个 RY 门把 $q_1,q_2$ 转到 $\sqrt{0.55}$ 权重，CCNOT 写 payoff，CNOT 把 payoff 拷进 flag，再用一次 CCNOT 把 payoff 反算清掉（它只是临时工）：

```python
def A_forward(psi, n, ctrl=None, off=0):
    psi = op1(psi, ry(2*np.arcsin(np.sqrt(0.55))), off+0, n, ctrl)  # q1
    psi = op1(psi, ry(2*np.arcsin(np.sqrt(0.55))), off+1, n, ctrl)  # q2
    psi = op1(psi, Xm, off+2, n, {off+0:1, off+1:1})   # payoff ← uu
    psi = op1(psi, Xm, off+3, n, {off+2:1})            # flag ← payoff
    psi = op1(psi, Xm, off+2, n, {off+0:1, off+1:1})   # 清 payoff
    return psi
```

**Grover 算子** $Q=A\,S_0\,A^\dagger S_\chi$：flag=1 变号、$A^\dagger$ 回到原点、$|0000\rangle$ 之外变号、$A$ 回旋转平面。$Q$ 在旋转平面内的转角是 $2\theta$，$\sin^2\theta=0.3025$，所以 $j$ 次迭代后 flag 概率是 $\sin^2((2j+1)\theta)$，峰值在 $(2j+1)\theta\approx\pi/2$，即 $j^\star\approx\pi/(4\theta)-1/2=9.09\to 9$。

踩过的坑值得记录，两个都是"电路没有报错、谱悄悄歪掉"级别：

1. **比特反序**：计数比特 $j$ 承载 $Q^{2^j}$ 的相位 $2^j\phi$，它必须坐在计数整数的第 $j$ 位（位权 $2^j$）。若寄存器布局里它实际坐在第 $m-1-j$ 位，整条谱关于 $M/2$ 镜像反射——$m=4$ 时峰值从 $c=3$ 跑到 $c=15$，读出 $\hat a=0.038$，错得离谱但电路"看起来"完全对称。修法：`Ucache = {j: ctrlQ_unitary(n, m-1-j, off=m)}`。
2. **忘了初始 A**：QPE 前系统寄存器必须先 $A|0\rangle$。直接拿 $|0000\rangle$ 进受控-$Q$ 演化，等于混进了 $Q$ 的 $\pm1$ 本征子空间，双峰塌成 $c=8$ 的单峰（$\hat a=1.0$）。一行 `psi = A_forward(psi, n, off=m)` 的事，没有它就全错。

![Grover 放大实拍与 m=4 QPE 双峰谱](/images/quantum-computing-option-pricing/fig1_grover_qpe.png)

图 1 左：$j=0..11$ 每个迭代次数下精确模拟的 flag 概率与理论虚线严丝合缝（最大偏离 3.2e-15，浮点噪音量级）。右：$m=4$ 标准 QPE 谱，两根红虚线是理论峰 $\theta\cdot16/\pi=2.97$ 与其镜像 $16-2.97$，实测 $c=3$ 与 $c=13$ 各分走 49.81% 质量，合计 99.63%。$\hat a=\sin^2(3\pi/16)=0.3087$。

**为什么 0.00616 是地板而不是起点？** 相位 $\phi=\theta/\pi\approx0.1854$，$m=4$ 时量化到 $3/16=0.1875$，误差 0.0021（相位）；$\hat a=\sin^2(\pi c/16)$ 把这个偏移放大到 0.00616。继续加位数：$m=5,6,7$ 最近读数分别仍映射到同一个 $\hat a$——想砸穿地板得上 $m=8$（0.00039 量级）或换振幅放大+峰值估计的 trick。**这就是"QPE 对小幅度/坏相位并不魔法"的微观标本**：优势是渐近陈述，有限比特下先撞上量化格点。

## 实验二：两条直线的夹角就是平方加速

$q_{\text{pts}}=(2^m-1)$ 次查询对应 $m=1..7$，每个 $m$ 按 QPE 谱抽 3000 次测量取中位误差；经典 MC 用 $N=2^3..2^{20}$、每档 300 批取 RMSE。log-log 拟合：

- QAE 斜率 **−0.9996**（理论 −1）
- MC 斜率 **−0.5030**（理论 −0.5）
- $m=7$：127 次查询，误差 0.00616（量化地板）；MC 达到同误差需 $(\sigma/\varepsilon)^2\approx5564$ 次，$\sigma=\sqrt{a(1-a)}=0.459$

**43.8 倍**。而且注意这是"地板还没被砸穿"的情况下算出来的——如果误差要求松一点（$10^{-2}$ 级，机构实际常用的精度），QAE 只需要 $m\approx7$，MC 要 $N\approx2100$，同样是平方关系的切面。$m=2,3$ 处 MC 反而显得"准"（0.1975 vs 地板 0.00616 被格点惩罚），这是图 2(b) 前几个点没贴住理论斜率的原因：**有限 $m$ 的 QAE 是"先吃量化地板、再吃查询红利"**，图上的折线形状本身就是诚实性的证据。

![精度-查询 log-log：QAE 斜率 −1 vs MC 斜率 −0.5](/images/quantum-computing-option-pricing/fig2_precision_scaling.png)

## 实验三：密度矩阵加噪，看优势怎么漏气

真实含噪量子硬件上，每个逻辑门后接去极化信道。用密度矩阵（$2^{m+4}=256$ 维，$m=4$）精确传播：对 sys 4 比特逐个以 $q_{flip}$ 概率施加 $X$ 翻转，电路深度按 $Q$ 调用次数折算成噪声强度 $\varepsilon\in[0,0.5]$。对照组是**等查询数 MC**：同样 15 次"预算"下经典估计的标准误差 0.4487——这是量子电路哪怕全崩了也不该输给的对手。

结果分三层：

- **平台区**：$\varepsilon\le0.2$，$q_{flip}=0$ 的线误差钉在 0.00616 纹丝不动——QPE 谱的峰很"胖"（99.6% 质量），小扰动推不动峰的位置。
- **崩溃区**：$\varepsilon=0.35$ 起，$q_{flip}=0.01/0.03$ 的谱崩掉，读数塌向 $\hat a=0$（误差 0.3025，等于"押注不可能"）；干净线也跳到 0.1975。`collapse_eps=0.35`。
- **底线**：24 个噪声点全部落在 0.4487 这条等查询 MC 线下方——噪声把 43.8 倍优势压成约 1.5～7 倍（平台区 0.4487/0.00616≈73 倍对照，崩区 0.4487/0.3025≈1.5 倍对照），但**没有任何一个点翻转到"量子不如同预算经典"**。

结论一句话：**QAE 的量子优势是容错预算买来的**。门错误率每多一个数量级，预算就烧掉一截；这也是为什么业界共识是 QAE 需要表面码级容错，而 QAMC（下一步）对 NISQ 更友好——它不需要深度 A 算子电路。

![噪声扫描与电路成本](/images/quantum-computing-option-pricing/fig5_noise_cost.png)

右图顺带算了成本账：$m=7$ 需要 127 次 $Q$ 调用、515 个单比特门 + 508 个 Toffoli（≈4090 个 CNOT 等效）。T 工厂（magic state factory）就是为这个买单存在的——"量子加速"在容错架构里的真实价格。

## 实验四：Heston 亚式期权上，经典 QMC 先交卷

量子还没到场的地方，量化的主力仍是蒙特卡洛。Heston 模型（$S_0=K=100, r=2\%, \kappa=2, \theta=4\%, \sigma_v=30\%, \rho=-0.7, v_0=4\%, T=1$，Feller 条件 $2\kappa\theta=0.16>\sigma_v^2=0.09$ ✓），Euler 对数离散 + 精确 CIR 方差步（双随机桥），1024 步。正确性三关：

1. BS 退化检查：MC 欧式看涨 8.9276 vs 解析 8.9160，相对误差 0.13%
2. 锚定价：$d=12$ 亚式 80 万路径 MC 5.3134（se 0.0077），Sobol 8192 点 5.3116——两套独立实现互咬
3. 图 3 里 MC 与 QMC 两条误差曲线同涨同落，互为交叉验证

"伪随机 MC 批均值 RMSE vs 随机化 Sobol RMSE"，$d\in[2,64]$（定价观察次数），每档 240 批 × 1024 路径：

![Heston 亚式：QMC 加速比从 11.2× 衰减到 1.93×](/images/quantum-computing-option-pricing/fig3_d_error.png)

加速比 11.2× ($d=2$) → 5.6× ($d=4$) → 2.6× ($d=12$) → 1.93× ($d=64$)，**单调衰减但从未跌破 1×**。这就是经典 QMC 的真实体感：低维收益均匀（月度观察 $d=12$ 仍赚 2.6 倍），高维（日度观察 + 多障碍）红利用维度税付掉。想继续压误差，只能回到"加样本"的老路——而 MC 的误差是关于样本数的 $-1/2$ 次幂，每多一位精度多烧 100 倍算力。**QAMC 卖的正是这一段。**

## 实验五：量子优势的准确形状——恒为平方，票价随尾部浮动

先纠正一个流传甚广的说法。很多材料宣称"收益轻尾时 QAMC 能达到指数级加速"——**Montanaro 2015 原文并没有这样的定理**。原文能引用的硬结论是 Theorem 2.5：对**有界方差**（$\mathrm{Var}(X)\le\sigma^2$）的任意量子子程序输出，量子用 $\tilde O(\sigma/\varepsilon)$ 次调用估计均值，经典需要 $O(\sigma^2/\varepsilon^2)$ 次——加速是平方，polylog 因子之外没有更强的尾巴红利。那么尾部假设真正买什么？2024 年 Blanchet-Szegedy-Wang 把画面补全了：当方差**无限**、只有 $(1+\delta)$ 阶矩有界（$P(|X-\mu|>\lambda)\le C\lambda^{-(1+\delta)}$）时，经典需要 $C\,\varepsilon^{-(1+\delta)/\delta}$ 次，量子需要 $C^{1/2\delta}\varepsilon^{-(1+\delta)/2\delta}$ 次——**量子指数恰好是经典的一半，优势仍是平方；但尾部越厚，双方的复杂度指数一起上升**。"平方红利"是恒定汇率，"厚尾"是两边同时涨价的通胀。

对 Heston 亚式的折扣收益 $X$（Sobol 262144 点）拟合经验尾 $\lambda$-尾概率的 log-log 斜率：

$$\hat\kappa = 3.92,\qquad \hat\sigma = 6.93$$

表观尾指数 3.92 > 2，落在**有界方差区**，Montanaro 定理直接适用：

$$N_{\text{MC}} = (z_{75}\hat\sigma)^2\varepsilon^{-2},\qquad N_{\text{Q}} = \hat\sigma\,\varepsilon^{-1}\log_2(1/\varepsilon)$$

$\varepsilon=10^{-5}$ 处省 **4.3 个数量级**。再造一个方差无限的对照：t(1.8) 收益（MAD 对齐到 Heston 尺度，拟合 $\hat\kappa=1.74$，1.7 阶矩 $C_{1.7}=411$），此时经典指数 $(1+\delta)/\delta=2.43$、量子指数 $1.21$——$\varepsilon=10^{-5}$ 处量子照旧省 **6.8 个数量级**，比 Heston 场景还多，但注意两条曲线的绝对位置都整体上移：**厚尾没有消灭优势，只是让买卖双方都付更多**。

![尾部拟合与查询数-精度曲线：量子指数恒为经典的一半](/images/quantum-computing-option-pricing/fig4_qamc_query_scaling.png)

## 落地自查清单

如果你要把这套东西用于选型或内部汇报，四个问题过滤 90% 的 PPT：

1. **精度需求多少？** $10^{-2}$ 级：经典 QMC + GPU 把量子问题变成"五年后再议"。$10^{-5}$ 以下（FVA、XVA 聚合、CVA 监管资本）：平方差距才刚开始值钱。
2. **尾部厚不厚？** 先看收益分布经验尾指数。厚尾不取消量子优势，但让经典和量子两边的门票一起上涨——预算按 $(1+\delta)/\delta$ 的陡指数做，别按方差有界区的 2 做。
3. **电路谁跑？** 图 5 右：$m=7$ = 4090 CNOT 等效 + 508 Toffoli。没有 10⁶ 物理比特，QAE 只存在于模拟器（本文的密度矩阵就是它的硬件，256×256 复矩阵）。
4. **对基线是什么？** "比 2 的 20 次方次暴力 MC 快 43.8 倍"毫无意义，对基线必须是**同算力预算下的调优 QMC**（本文图 3 那种）。

量子金融的现状一句话：**数学已经非常清楚（本文五个实验全部可复现），硬件还在报价单上**。对绝大多数国内机构的 XVA 预算，2026 年最划算的"量子加速"仍然是 Sobol + 随机化 + 好的方差缩减——但清楚墙那边斜率是 −1 对 −0.5、优势汇率恒为平方的人，会在报价单落地那天先动手。

## 附：完整可运行代码

生成图 1/2/5 的 QAE 模拟器（态矢量 + 密度矩阵，节选核心；完整脚本 `scripts/gen_qae_digital.py`）：

```python
import numpy as np
S0, U_, D_, P_UP, KX = 100.0, 1.25, 0.8, 0.55, 106.0
a_true = P_UP**2                      # 0.3025, 判卷真值
THETA = np.arcsin(np.sqrt(a_true))
TH_RY = 2*np.arcsin(np.sqrt(P_UP))
Xm = np.array([[0,1],[1,0]], complex)
Hm = np.array([[1,1],[1,-1]], complex)/np.sqrt(2)
def ry(t): return np.array([[np.cos(t/2),-np.sin(t/2)],[np.sin(t/2),np.cos(t/2)]], complex)

def op1(psi, U, q, n, ctrl=None):     # 第q位(从高位数)作用2x2 U, ctrl=控制比特字典
    shift = 1 << (n-1-q); i = np.arange(1 << n)
    keep = np.ones(1 << n, bool)
    for cq, bit in (ctrl or {}).items(): keep &= ((i >> (n-1-cq)) & 1) == bit
    s0 = i[keep & ((i & shift) == 0)]; s1 = s0 | shift
    a0, a1 = psi[s0].copy(), psi[s1].copy()
    psi[s0] = U[0,0]*a0 + U[0,1]*a1;  psi[s1] = U[1,0]*a0 + U[1,1]*a1
    return psi

def A_forward(psi, n, ctrl=None, off=0):     # sys 4位 = [q1,q2,payoff,flag] @ off..off+3
    c = dict(ctrl or {})
    psi = op1(psi, ry(TH_RY), off+0, n, c); psi = op1(psi, ry(TH_RY), off+1, n, c)
    psi = op1(psi, Xm, off+2, n, {**c, off+0:1, off+1:1})   # payoff ← uu
    psi = op1(psi, Xm, off+3, n, {**c, off+2:1})            # flag ← payoff
    return op1(psi, Xm, off+2, n, {**c, off+0:1, off+1:1})  # 清 payoff

def Q_steps(psi, n, off=0):                  # Q = A S0 A† Sχ
    i = np.arange(1 << n); fpos = n-1-(off+3)
    psi[((i >> fpos) & 1) == 1] *= -1        # Sχ
    psi = A_dagger(psi, n, off=off)
    psi[(i & 15) != 0] *= -1                 # S0
    return A_forward(psi, n, off=off)

# Grover 放大：j=9 时 flag 概率 0.9952 ≈ sin²(19θ)=0.9952
psi = np.zeros(16, complex); psi[0] = 1
psi = A_forward(psi, 4)
for _ in range(9): psi = Q_steps(psi, 4)
print(abs(np.abs(psi[(np.arange(16)&1)==1])**2).sum() - np.sin(19*THETA)**2)  # ~3e-15
```

生成图 3/4 的 Heston 模拟与尾拟合（完整脚本 `scripts/gen_qamc_heston.py`）：

```python
import math, numpy as np
from scipy.stats import qmc
HP = dict(S0=100., K=100., r=.02, kappa=2., theta=.04, sigma_v=.3, rho=-.7, v0=.04, T=1.)
nsteps, dt = 1024, HP['T']/1024

def sobolheston(d, npath, seed):             # d 个观察点, 2d 维 Sobol + 随机平移
    engine = qmc.Sobol(d=2*d, scramble=True, seed=seed)
    u = engine.random(npath)
    z = np.sqrt(2)*np.special.erfinv(2*u[:, :d]-1)          # 价格维
    w = np.sqrt(2)*np.special.erfinv(2*u[:, d:]-1)          # 方差维
    zp = HP['rho']*z + np.sqrt(1-HP['rho']**2)*w            # 相关化
    v, logS = np.full(len(z), HP['v0']), np.full(len(z), math.log(HP['S0']))
    obs = np.linspace(dt, HP['T'], d); obs_idx = np.searchsorted(
        np.arange(1, nsteps+1)*dt, obs) if d < nsteps else None
    # ... 逐步 CIR 精确采样方差 + 对数欧拉价格, 在第 obs_idx 步快照平均 ...
    return math.exp(-HP['r']*HP['T'])*avg_price            # 折扣亚式收益 X

X = sobolheston(12, 2**18, 20261036)
mu = X.mean(); dev = np.abs(X-mu); sig = X.std()
qs = np.linspace(.90, .99995, 400); lam = np.quantile(dev, qs)
khat = -np.polyfit(np.log(lam), np.log(1-qs), 1)[0]           # κ̂ = 3.92 (>2, 方差区)

# 查询数：Montanaro 2015 Thm 2.5（有界方差）vs 经典 Chebyshev
eps = np.logspace(-2, -5.5, 60)
N_MC = (.6745*sig)**2 * eps**-2
N_Q  = sig * eps**-1 * np.log2(1/eps)                         # 指数 1 vs 2，ε=1e-5 省 4.3 个数量级

# 厚尾代理 t(1.8)：方差无限、1.7 阶矩 C=411 —— Blanchet-Szegedy-Wang 2024
# 经典 C·ε^{-(1+δ)/δ}=ε^-2.43，量子 C^{1/2δ}·ε^{-(1+δ)/2δ}=ε^-1.21，仍是平方
```

（正文代码为可读性对快照逻辑有省略，仓库内脚本为逐行可运行版。）

**参考**：Montanaro 2015, *Quantum speedup of Monte Carlo methods*, Proc. R. Soc. A 471:20150301（Thm 2.5 有界方差 Õ(σ/ε)）；Blanchet, Szegedy & Wang 2024, arXiv:2401.07497（无限方差、$(1+\delta)$ 阶矩下的量子均值估计）；Brackmann et al. 2022, Phys. Rev. A 106（QAE 电路成本实测）；Stamatopoulos et al. 2020, Quantum 4（QAE 期权定价资源估算）；Clement et al. 2012（QMC 在高维路径问题上的衰减）；L'Ecuyer & Lecot 2012（随机化 QMC 与 Feller 检验）。
