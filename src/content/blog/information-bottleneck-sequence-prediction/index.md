---
title: "信息瓶颈在序列预测中的应用：压缩预测什么，什么就能被预测"
description: "Tishby 的信息瓶颈把「好表示」定义成一个权衡：尽量少保留输入信息 I(X;T)，尽量少丢目标信息 I(T;Y)。本文不用任何深度学习框架，纯 numpy 手搓两级实验验证这句话：先用确定性 IB 在精确已知的 24→6 类联合分布上做贪心状态合并，画出信息-预测权衡曲线（H(Y)=2.58 bit，K=24 全保留也只有 1.58 bit 预测力）；再对市场状态序列预测做变分信息瓶颈 VIB——3 态粘性马尔可夫链生成信号，混入最多 48 个 AR(1) 结构化干扰维。实测结果很诚实：VIB 买到的不是准确率，是码率——β∈[3e-4,1e-2] 区间准确率与无正则版持平，保留码率却从 15.9 bit 压到 1.1 bit（15 倍）；β 再大 10 倍模型直接「失忆」塌缩到 0.366≈最优单类基线。干扰税主要咬 plain（0.71→0.41），但消融实验判决两个模型谁都没学会无视干扰。诚实边界写在文末。"
publishDate: '2026-10-26'
tags:
  - 量化交易
  - 信息瓶颈
  - 变分信息瓶颈
  - 表示学习
  - 市场状态预测
  - Python
language: Chinese
difficulty: advanced
---

## 一句话纲领：压缩即预测

Tishby 的信息瓶颈（Information Bottleneck, IB）对"什么是好的特征"给出了一个非常干脆的定义。给定输入 $X$（比如过去 10 帧行情）和目标 $Y$（下一期市场状态），找一个中间表示 $T$，最大化

$$
\max_{p(t|x)} \; I(T;Y) - \beta\, I(X;T), \qquad \beta \ge 0
$$

前半句要 $T$ 尽量带上 $Y$ 的信息，后半句惩罚 $T$ 里保留的 $X$ 的信息。$\beta$ 是汇率：一 bit 输入信息值多少钱。直觉是——$X$ 里既有与 $Y$ 有关的信号，也有噪声和细节，**逼着表示变小，噪声就装不下**，剩下的自然是预测性成分。于是有了那句口号：压缩预测什么，什么就能被预测。

听起来很美。但它在真实（半真实）的序列预测任务上到底兑现什么、不兑现什么？本文做两个从零实验，把这笔账算清楚。

## 实验一：确定性 IB——在精确分布上量"比特的价格"

第一个实验不需要任何近似：构造一个联合分布精确已知的世界。$Y$ 均匀取 6 类；给定 $Y=y$，$X$ 落在 24 个值上，质量集中在第 $y$ 组和第 $(y+3)\bmod 6$ 组（组内再按 Dirichlet 扰动展开），类间共享第二个模式——这样"合并哪两个值"才不亏预测力，才有结构可挖。$H(Y)=2.585$ bit。

对确定性映射 $T = t(X)$，问题退化为组合优化：把 24 个 $X$ 值合并成 $K$ 个状态，每一步贪心合并使 $I(T;Y)$ 损失最小的一对，记录整条 $(I(X;T), I(T;Y))$ 曲线。

![确定性信息瓶颈权衡曲线](/images/information-bottleneck-sequence-prediction/ib-tradeoff.png)

几个实测读数值得记住：

| 状态数 K | 保留输入 I(X;T) | 预测信息 I(T;Y) |
|---|---|---|
| 2 | 0.789 bit | 0.002 bit |
| 8 | 2.978 bit | 0.115 bit |
| 10 | 3.128 bit | 0.251 bit |
| 24 | 4.560 bit | 1.578 bit |

两个反直觉的事实。第一，**第一个 bit 往往最不值钱**：K 从 1 到 2 花了 0.79 bit 保留码率，只换来 0.002 bit 预测力——贪心最先合并的当然无关紧要，但最先"保留差异"的方向也不保证与 $Y$ 有关。第二，K=24 即完全不压缩，$I(T;Y)$ 也只有 1.58 bit < $H(Y)$ = 2.585——$X \to Y$ 这条含噪通道本身就丢了近一个 bit 的信息，任何表示都找不回来。数据处理不等式在预测问题里的长相就是这样：**你能预测的天花板在数据生成时就被定死了**。

![信息阶梯与每状态的边际售价](/images/information-bottleneck-sequence-prediction/info-staircase.png)

信息阶梯图（右）给出每个状态跃迁 $K \to K+1$ 的"预测力售价"$\Delta I(T;Y)$：$K \le 6$ 的五个跃迁每步只买到 0.004–0.019 bit，是纯买输入细节；到 $K=7\to8$ 一步 0.075 bit，是真正的预测结构。IB 的 $\beta$ 扫描就是在这些价格上划一条线：售价低于 $\beta$ 的比特，不值得留。

## 实验二：变分 IB——市场状态序列预测上的真章

确定性 IB 依赖精确联合分布，行情显然没有。变分信息瓶颈（VC-Net/Alemi et al. 2017）把 $T$ 松弛成连续高斯瓶颈：编码器输出 $q(z|x)=\mathcal N(\mu(x),\sigma^2(x))$，训练目标

$$
\mathcal L = \mathbb E\big[-\log p(y|z)\big] + \beta\, \mathrm{KL}\big(q(z|x)\,\|\,\mathcal N(0,I)\big)
$$

KL 项关于 $(\mu,\log\sigma^2)$ 有解析梯度，采样用重参数化 $z = \mu + \sigma \odot \epsilon$，剩下全是普通反向传播。我们纯 numpy 手搓（Adam + lr 衰减 + β 线性退火），不用 torch。

数据生成过程贴近"市场"二字：隐藏状态 $r_t \in \{熊, 震荡, 牛\}$ 是粘性马尔可夫链（停留概率 0.85，50 万步链上实测 0.8494）；价格信号 $s_t = \text{REG\_MU}[r_t] + 0.15 \cdot \text{noise}$，观测取过去 10 帧的 **3 路含噪视角**（各加 0.25 观测噪声），再拼上 $d_{nuis}$ 个 **AR(1)($\varphi=0.8$) 干扰序列**窗口。目标 $y = r_{t+1}$。干扰是自相关 0.8 的"假信号"——它长得像因子，与 $Y$ 却机械无关。这个 DGP 的好处是：贝叶斯天花板可以单独测出来（在 50 万步链上"识破当前状态就预测不变"，准确率 0.8494），瞎猜是 1/3。

### Rate–Distortion 前沿：前两三个 bit 最值钱

固定 32 个干扰维，扫 7 个 β，每档 4 个种子：

![信息-失真前沿](/images/information-bottleneck-sequence-prediction/rd-frontier.png)

实测（4 种子均值）：

| 配置 | 保留码率 I(X;Z) | 测试准确率 |
|---|---|---|
| plain（β=0） | 15.87 bit | 0.4125 |
| β=3e-4 | 3.78 bit | 0.4129 |
| β=3e-3 | 1.91 bit | 0.4158 |
| β=6e-3 | 1.55 bit | 0.4240 |
| β=1e-2 | 1.09 bit | 0.4040 |
| β=3e-2 | 0.0014 bit | 0.3660 |
| β=1e-1 | 0.0001 bit | 0.3660 |
| 最优单类基线 | 0 bit | 0.3717 |

这张图是全文核心。横向看：**从 plain 的 15.9 bit 砍到 1.1 bit（15 倍压缩），准确率纹丝不动**（0.40–0.42 区间内，种子间波动远大于组间差异，β=6e-3 甚至略优于 plain）；纵向看，$\beta \ge 3\times10^{-2}$ 时 KL 项压倒似然项，瓶颈塌缩（rate→0.00，模型把 $\mu$ 整个拖到零附近），准确率 0.366，**和"永远猜多数类"的 0.372 难分彼此——这就是 IB 的"失忆"**。中间地带的教训很具体：0→1 bit 买到绝大部分可用预测力，1→15 bit 买的基本是噪声和细节。给序列预测模型喂更长窗口、更多因子之前，先问问你的比特卖出了什么价。

### 干扰税：咬的是 plain

把 $d_{nuis}$ 从 0 扫到 48，对比 plain 与 VIB（β=3e-3，退火 warm=800）：

![干扰税与码率解耦](/images/information-bottleneck-sequence-prediction/nuisance-tax.png)

左图：plain 的准确率从 0.711（无干扰）一路被 AR(1) 假因子拖到 0.413（48 干扰维），**损失 0.30，几近腰斩**——600 个训练样本、510 维输入，无正则网络把干扰序列当成了背诵材料（训练集准确率实测 1.0）。VIB 在同点位（0.460 / 0.416 / 0.403）与 plain 持平或略好，互有胜负，没有神话可讲，也没有崩坏。右图才是 IB 真正兑现的承诺：**VIB 的保留码率钉死在 1.6–2.3 bit，完全不随输入维度增长**；plain 的自发保留码率则从 9.9 bit 涨到 15.9 bit——它照单全收地把每个干扰维都编码进了瓶颈。做量化的人对这张右图应该有体感：你的特征库每个月都在膨胀，模型的"输入依赖度"也在无声膨胀，而 IB 给你一个可审计的、与特征数解耦的压力阀。

### 解剖：谁在真正依赖干扰？

码率低不等于"看懂了"。两个终审实验（48 干扰维，4 种子均值）：

![权重与消融解剖](/images/information-bottleneck-sequence-prediction/vib-weights.png)

**首层权重能量**：plain 的 $|W_1|$ 在信号列（3 列）与干扰列（48 列）上的均值分别是 0.1246 / 0.1233，比值 0.99；VIB 是 0.1809 / 0.1715，比值 0.95。热图上看不出谁"剪掉"了干扰列——Adam 逐坐标归一化更新，600 样本又足以双双背完训练集（实测训练准确率同为 1.0），权重大小在这里不是可靠证词。

**消融判决**：测试时把全部 48 个干扰维置零（等价于"假如你能提前知道它们无关"），plain 从 0.413 **反弹**到 0.635（+0.222），VIB 从 0.403 反弹到 0.617（+0.214）。两个数字几乎一样大，结论也就清楚了：**两个模型谁都没学会在推理时无视干扰**，干扰是通过推理通路持续注入噪声的，不是靠训练期"学过又忘掉"。VIB 的红利是表示层的压缩与稳定性（码率、对维度膨胀的免疫力），不是对混杂源的因果识别——那是下一期 PC 算法的活。真想让准确率反弹，把干扰维度从输入里物理删掉，比任何正则都直接。

## 可运行的核心代码

VIB 的骨架真的只有几十行（完整 DGP 与出图代码见本文配图脚本思路，逐行可复现）：

```python
import numpy as np

def train_vib(Xtr, ytr, Xte, yte, beta, iters=2400, batch=64,
              D=None, HID=64, ZD=20, K=3, lr0=0.02, warm=800, seed=0):
    rng = np.random.default_rng(seed)
    n = len(Xtr); D = Xtr[0].size
    Xf = (Xtr.reshape(n, D) - Xtr.reshape(n, D).mean(0)) / (Xtr.reshape(n, D).std(0) + 1e-9)
    Xtef = (Xte.reshape(len(Xte), D) - Xtr.reshape(n, D).mean(0)) / (Xtr.reshape(n, D).std(0) + 1e-9)
    shp = [(D+1,HID),(HID+1,ZD),(HID+1,ZD),(ZD+1,K)]
    Ws  = [rng.normal(0, np.sqrt(2/s[0]), s) for s in shp]
    Ms, Vs = [np.zeros(w.shape) for w in Ws], [np.zeros(w.shape) for w in Ws]
    b1, b2, eps = .9, .999, 1e-8
    def fwd(Xb, do_sample):                      # 前向 + 重参数化
        x1 = np.c_[Xb, np.ones(len(Xb))]
        h1 = np.tanh(x1 @ Ws[0]); h1x = np.c_[h1, np.ones(len(Xb))]
        mu = h1x @ Ws[1]; lv = np.clip(h1x @ Ws[2], -8, 8)
        z = mu + np.exp(.5*lv)*rng.normal(size=mu.shape) if do_sample else mu
        zx = np.c_[z, np.ones(len(z))]; lg = zx @ Ws[3]
        lg -= lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True)
        return x1, h1, h1x, mu, lv, z, zx, lg, p
    sample = beta > 0
    for it in range(1, iters+1):
        bt = beta*min(1., it/warm) if sample else 0.   # β 线性退火防塌缩
        idx = rng.integers(0, n, batch)
        x1,h1,h1x,mu,lv,z,zx,lg,p = fwd(Xf[idx], sample)
        dp = p.copy(); dp[np.arange(batch), ytr[idx]] -= 1; dp /= batch  # CE 梯度
        dz  = dp @ Ws[3][:-1].T
        dmu = dz + bt*mu
        dlv = (dz*(.5*(z-mu)) if sample else 0.) + bt*.5*(np.exp(lv)-1)  # KL 解析梯度
        g = [x1.T @ ((dmu@Ws[1][:-1].T + dlv@Ws[2][:-1].T)*(1-h1**2)),
             h1x.T @ dmu, h1x.T @ dlv, zx.T @ dp]
        lr = lr0/(1 + 8.*it/iters)                     # lr 衰减，后期稳到瓶颈上
        for W,M,V,gi in zip(Ws,Ms,Vs,g):
            M[:] = b1*M + (1-b1)*gi; V[:] = b2*V + (1-b2)*gi*gi
            W -= lr*(M/(1-b1**it))/(np.sqrt(V/(1-b2**it))+eps)
    _,_,_,mu,lv,_,_,lg,_ = fwd(Xtef, False)
    acc = (np.argmax(lg,1)==yte).mean()
    rate = .5*np.sum(np.log2(1 + mu.var(0)/np.maximum(np.exp(lv).mean(0), 1e-9)))  # 码率估计
    return acc, float(rate)
```

两处手搓细节最容易翻车：`z` 采样时 $\sigma^2$ 走 `logvar` 并 clip 到 `[-8, 8]`；KL 对 `logvar` 的梯度 $\frac12(\sigma^2-1)$ 里的 $-\frac12 \cdot \frac{1}{\sigma^2}$ 分项被重参数化噪声自动吸收，写成上式即可。码率用离散化近似 $R \approx \frac12 \sum_j \log_2\!\big(1 + \widehat{\mathrm{var}}(\mu_j)/\overline{e^{lv_j}}\big)$ 估计，就是文中"1.9 bit vs 15.9 bit"的来源。

## 给序列预测建模的三条操作结论

1. **先画你自己的 rate–distortion 前沿**。把 β（或瓶颈维度）扫一遍，如果发现砍掉 80% 码率不掉准确率——你之前的输入冗余度就是实锤的；如果砍 10% 就掉，说明任务信息密度高，值得继续喂数据。本文任务是前者：1.1 bit ≈ 15.9 bit 的预测力。
2. **警惕"失忆"塌缩**。β 过大的信号是 rate→0 且准确率落到最优单类基线附近（本文 0.366 vs 0.372）。上线前测一下模型输出是否已退化成常数策略——这比看损失曲线可靠。
3. **特征膨胀用码率审计**。VIB 码率不随输入维度涨（1.6–2.3 bit 平线），plain 会（9.9→15.9 bit）。每月加因子时顺手看一眼这个数，加进去的是信号还是背诵材料，一目了然。

## 诚实边界

必须说清楚本文**没能**证明什么。其一，**VIB 不是准确率增强器**：在与 plain 的全部对比中，VIB 准确率互有胜负、差距均在种子噪声内（dn=0 时 VIB 反而低 0.03）；它买单的是码率压缩、训练稳定性与可审计性。市面上"信息瓶颈提升预测精度"的叙事，在我们的受控复现里没有得到支持。其二，**低码率 ≠ 学到因果**：消融实验显示 VIB 与 plain 对干扰的推理期依赖几乎相同（反弹 +0.222 vs +0.214），KL 正则剪不掉推理通路上的混杂输入，只让它"存得更省"。其三，实验是合成 DGP：粘性马尔可夫状态 + 线性高斯观测 + AR(1) 干扰，真实行情的厚尾、波动聚集、状态非平稳都会更苛刻；β 天花板（0.003–0.006 甜蜜区）不具移植性，换任务必须重扫。其四，H(Y) 天花板、贝叶斯极限都依赖 DGP 已知——真实数据上这些只能估计，估错方向整个结论会反。信息瓶颈是个干净的记账框架，不是点石成金的咒语。

下次见。压缩你的比特，然后按价付钱。
