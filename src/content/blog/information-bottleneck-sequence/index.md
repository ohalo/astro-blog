---
title: "信息瓶颈的收益边界：序列回归里 β、干扰维与过拟合的真实对照"
publishDate: 2026-11-01
language: zh
tags: [机器学习, 序列预测, 信息论, 信息瓶颈, 量化]
difficulty: advanced
description: "前篇用确定性 IB 与 VIB 分类验证了压缩即预测；本篇换到更常见的序列回归设定，追问一个工程问题：把 KL 正则塞进下一步值预测器，究竟能换来多少泛化？三类 DGP、八档 β、五档干扰维、三个随机种子、验证集早停，全部纯 numpy 真算。诚实结论：早停后 AR(2) 只改善 0.06%，Lorenz 反而恶化 19.31%，β 从 0.001 加到 0.3 使 MAE 从 0.6145 升到 0.6514。IB 在这里的作用更接近抑制过拟合的正则项，而非普遍的信息论最优压缩。附完整可运行代码、结果 JSON 与五张真实计算图。"
image: "images/information-bottleneck-sequence/cover.png"
---

## 一、先说结论：这次没有好消息

本站 10 月 26 日那篇《信息瓶颈在序列预测中的应用》做的是**分类**设定：在精确已知的联合分布上量出信息的价格，再用变分信息瓶颈（VIB）预测三态市场状态，结论是压缩确实能换来泛化。那篇文章的实验是成立的，但它留下了一个更常见的场景没有回答——如果目标不是离散状态，而是**下一步的数值**，同样的 KL 正则还灵吗？

本篇把问题搬到序列回归上，用同一套严格的对照纪律重做：三类数据生成过程、八档压缩强度、五档干扰维数量、三个随机种子、训练与验证严格切分、两次训练共用同一批随机数。答案是：**几乎没有收益**。AR(2) 上，加了信息瓶颈的验证 MAE 是 0.615378，不加是 0.615745，改善 0.06%；Lorenz 混沌序列上加了反而恶化 19.31%；把 β 从 0.001 拉到 0.3，验证 MAE 从 0.614453 一路涨到 0.651419。

这个负结果比任何一句「显著提升」都更有用，因为它划出了适用边界：**信息瓶颈不是免费的泛化午餐，它的价值取决于目标信息能否被少数几个语义方向承载**。

![三类数据生成过程的真实样本序列](/images/information-bottleneck-sequence/ib_1_series.png)

## 二、先把理论和出处摆正

信息瓶颈（Information Bottleneck）由 Tishby、Pereira 与 Bialek 在 1999 年的《The Information Bottleneck Method》中提出。年份与作者在这类文章里经常被写错，引用时建议直接核对原论文。它把「好表示」写成一个拉格朗日权衡：

$$\min_{p(z|x)} I(X;Z) - \beta\, I(Z;Y)$$

其中 $I(X;Z)$ 是保留输入信息的代价，$I(Z;Y)$ 是保留预测信息的收益。深度学习里更常用的变分形式（VIB，以及 β-VAE）写成：

$$\mathcal{L} = \mathbb{E}_{q(z|x)}\left[-\log q(y|z)\right] + \beta\,\mathrm{KL}\big(q(z|x)\,\|\,p(z)\big)$$

**这里有一个极易踩坑的符号陷阱：两个式子里的 β 方向是相反的。** 在原始 IB 里，β 是「预测信息的单价」，β 越大越不愿意压缩；在 VIB 里，β 是「压缩项的系数」，β 越大压得越狠。把两套符号混着用，就会推出「β→0 退化为标准 VAE」这种错误说法——事实上 β→0 时 KL 项消失，模型退化成**没有随机瓶颈的确定性预测器**，而标准 VAE 恰好要求 KL 系数等于 1。本文从头到尾用的是 VIB 口径，β 越大压缩越强。

另一个常被含糊带过的点是：VIB 的 KL 项上界住的是 $I(X;Z)$，它并不直接优化互信息 $I(Z;Y)$。所以「IB 正则化」在实践中更接近一个**带噪声注入的各向同性高斯先验正则**，把它当成信息论最优性的保证是不成立的。这一点会在后面的实验里非常清楚。

## 三、实验设计：怎么让对照真的可比

数据生成过程（DGP）有三类，全部是纯 numpy 现场模拟：AR(1) 取 $x_t=0.6x_{t-1}+0.3\epsilon_t$，AR(2) 取 $x_t=0.4x_{t-1}+0.35x_{t-2}+0.3\epsilon_t$，Lorenz 系统用四阶 Runge-Kutta 以 $\mathrm{d}t=0.01$ 积分后取 $x$ 分量。每类生成 2600 个滑窗样本，窗口长 10，目标就是窗口之后那一期的主序列值；前 2000 个训练，后 600 个验证。

关键在干扰维：除了真实信号，输入还拼进 $d$ 条独立的结构化 AR(1) 序列窗口（$\phi=0.8$）。这些维度和预测目标完全无关，但因为有时间自相关，很容易被模型当成「看起来有规律」的特征记住。$d$ 从 0 扫到 16，对应输入维度从 10 涨到 170。

模型是一个刻意做小的瓶颈网络：输入展平后接 32 维 tanh 隐层，再映射到 6 维高斯瓶颈 $(\mu,\log\sigma^2)$，用重参数化采样 $z=\mu+\epsilon\odot\sigma$，最后由线性层输出标量预测。$\beta=0$ 时直接令 $z=\mu$，就是普通确定性网络；$\beta>0$ 时损失为 $\mathrm{MSE}+\beta\,\mathrm{KL}$。手写 Adam，学习率 0.008，批量 128，600 次迭代。

**三条纪律决定结果可信度**：第一，两次训练使用同一 seed 派生出的三条独立随机流（初始化、批量采样、重参数化噪声），因此 $\beta=0$ 与 $\beta>0$ 的初始化和训练样本顺序完全一致，差异只能来自 β 本身；第二，报告的是验证集早停 MAE（训练过程中验证 MAE 的最低点），避免把「谁过拟合得更晚」误当成泛化能力；第三，同时保留末点 MAE，用来量化过拟合幅度。所有数字都写进同目录的 `results.json`，文末代码可直接复算。

## 四、结果一：收益微弱，且会反转

| DGP | plain 验证 MAE | VIB 验证 MAE | 改善 | plain 过拟合幅度 | VIB 过拟合幅度 |
|---|---|---|---|---|---|
| AR(1) | 0.678999 | 0.664071 | +2.20% | 0.229800 | 0.174697 |
| AR(2) | 0.615745 | 0.615378 | +0.06% | 0.246284 | 0.231491 |
| Lorenz | 0.068221 | 0.081398 | **−19.31%** | 0.000784 | 0.001312 |

「过拟合幅度」是末点 MAE 减去早停 MAE。AR 过程上两个模型都严重过拟合（验证 MAE 从最低点回升 0.23 以上），VIB 只是把回升幅度压小了一点点；Lorenz 上几乎没有过拟合，因为混沌序列的确定性结构很容易学，此时**瓶颈里的随机噪声纯属伤害**——预测误差被硬生生推高了近两成。

![真实训练轨迹：训练损失持续下降，验证 MAE 先降后升](/images/information-bottleneck-sequence/ib_2_train_curve.png)

图里最值得盯的是右子图：大约 100 次迭代之后两条验证 MAE 曲线一起掉头向上，而左子图的训练 MSE 还在稳步下降。这就是典型的容量过剩 + 无关特征记忆。VIB 曲线爬得更慢，但**早停之后这点优势基本被抹平**——0.231491 对 0.246284，差 0.015。换句话说，只要认真做早停，KL 正则能给的额外保护相当有限。

## 五、结果二：β 扫描与「早停替代效应」

在 AR(2)、8 个干扰维、与主实验完全相同的数据与训练流下扫 β：

| β | 0 | 0.001 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.3 |
|---|---|---|---|---|---|---|---|---|
| 早停验证 MAE | 0.615745 | **0.614453** | 0.614667 | 0.614850 | 0.615378 | 0.618389 | 0.624294 | 0.651419 |
| 末点验证 MAE | 0.862029 | 0.832743 | 0.825695 | 0.830274 | 0.846869 | 0.833745 | 0.797176 | **0.751074** |

![β 扫描：早停口径下压缩越强越差，末点口径下强压缩反而抗过拟合](/images/information-bottleneck-sequence/ib_3_beta_scan.png)

两行数字讲了两个相反的故事，这恰恰是本篇最重要的发现。**早停口径**下最优 β 是 0.001（改善 0.2%，几乎在噪声范围内），之后单调变差，β=0.3 时 MAE 比 baseline 高 5.8%；**末点口径**下却是 β=0.3 最好（0.751074 对 0.862029）。原因不神秘：大 β 把模型压得接近「预测常数」，天然不容易记住训练集的无关维度，所以不早停时它衰得最慢。可一旦允许早停，这个保护就变得多余，而压缩造成的信息损失却实打实留了下来。

结论很实用：**别把「IB 抗过拟合」当成加正则的理由，先做早停；两者在功能上高度重叠，叠加使用只会付出信息损失的代价。**

## 六、结果三：干扰维不是「越多越该压缩」

理论上，无关维度越多，压缩越该有用。实际扫下来并非如此：

| 干扰维 d | 0 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| plain | 0.596032 | 0.608473 | 0.611845 | 0.615745 | 0.637630 |
| VIB | 0.597050 | 0.610217 | 0.603253 | 0.615378 | 0.633497 |
| VIB 改善 | −0.17% | −0.29% | +1.40% | +0.06% | +0.65% |

![干扰维数量与泛化：差距始终在 1.5% 以内且非单调](/images/information-bottleneck-sequence/ib_4_nuisance.png)

收益在 $\pm1.5\%$ 内来回摆动，没有随 $d$ 单调放大。$d=0$ 和 $d=2$ 时 VIB 甚至略差——没有干扰需要压制时，噪声注入就是纯负担。这说明本设定下瓶颈并没有「精准识别并扔掉干扰维」，它只是给所有参数施加了一个统一的高斯压力。指望 IB 自动完成特征选择，是对它能力的高估。

## 七、结果四：瓶颈里到底发生了什么

$\beta=0.02$ 时，6 个隐维度在验证集上 $\mathrm{Var}(\mu)$ 分别是 1.192424、0.003698、0.000688、0.008088、0.698151、0.029310，按 $\mathrm{Var}(\mu)>0.01$ 的阈值只有 **3 维被激活**，总 KL 为 2.957959 nats。也就是说，名义上的 6 维瓶颈实际只用了 2 个主方向加 1 个边缘方向，其余被先验压到接近常数。

![瓶颈各维的 mu 方差：6 维里只有 3 维真正被使用](/images/information-bottleneck-sequence/ib_5_bottleneck.png)

这解释了两件事：一是压缩确实发生了，KL 不是摆设；二是它压缩出的表示并没有比一个好好早停的确定性网络更有预测力。前篇分类实验里「干扰税」之所以明显，是因为标签是三态离散状态、信号稀疏，压缩能把 48 个干扰维里的噪声一次性挤掉；而回归下一步值时，目标的高频成分几乎全是不可预测的新息，**可压缩的语义冗余本来就少**。IB 的收益不是算法的属性，是任务信噪比的属性。

## 八、给工程实践的四条建议

1. **先做早停，再考虑 KL。** 本实验里早停把验证 MAE 从 0.862 拉到 0.616，而 KL 只再贡献 0.0004。顺序搞反，会把早停的功劳记到信息瓶颈头上。
2. **β 从 $10^{-3}$ 量级起调，别从 0.1 开始。** 扫描显示 0.001 是拐点，0.3 已经明显有害。β 的最优值依赖损失尺度，换数据集必须重扫。
3. **确定性强的序列（混沌、物理系统）不要用随机瓶颈。** Lorenz 上 19.31% 的恶化是纯粹的采样噪声代价。
4. **把 KL 当诊断量而非目标。** 本报告显示 3/6 维激活、2.958 nats，这类数字能告诉你瓶颈是否坍缩、是否有后验坍塌风险，比「MAE 降了多少」更有解释力。

## 九、诚实边界

本实验用的是单层 MLP 瓶颈，不是 LSTM 或 Transformer；结论只说明「在这个容量与这批 DGP 下」，不能外推到大规模序列模型。三个随机种子、单一隐藏维度 32、单一窗口长 10，都没有做超参数搜索，所以 0.06% 这类微小差异应当视为**不可区分于噪声**，而不是精确估计。早停使用验证集选点，本身带有乐观偏差；两个模型享受同等待遇，对比仍公平，但绝对数值偏低。全部数据是模拟的，没有一笔真实行情——任何「提升策略收益」的说法都不在本篇的证据范围内。最后，KL 只是 $I(X;Z)$ 的上界，本文没有估计互信息本身，因此严格说检验的是「VIB 正则的效果」，不是 IB 理论最优性。

想看互补的一面，请读前篇《信息瓶颈在序列预测中的应用》（10 月 26 日）：它用精确已知的分布量出了信息阶梯，也展示了分类设定下压缩确实能买到预测力。两篇合起来才是一句完整的话——**压缩有没有用，取决于你要预测的东西里有多少冗余可压。**

## 十、完整可运行代码

以下代码即本篇实验的源码，只依赖 numpy 与 scipy；直接运行会打印出与 `results.json` 完全一致的 12 项关键数字。

```python
import numpy as np
from scipy import stats
from itertools import combinations

SEED_IB, SEED_PC = 20261101, 20261102
L, HID, K, ITERS, BS, LR = 10, 32, 6, 600, 128, 0.008
BETAS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3]
DGPS = ['ar1', 'ar2', 'lorenz']
NUIS = [0, 2, 4, 8, 16]


def _series(kind, total, rng):
    if kind in ('ar1', 'ar2'):
        s = np.zeros(total)
        for t in range(2, total):
            if kind == 'ar1':
                s[t] = 0.6 * s[t - 1] + 0.3 * rng.normal()
            else:
                s[t] = 0.4 * s[t - 1] + 0.35 * s[t - 2] + 0.3 * rng.normal()
        return s
    # Lorenz: RK4, dt=0.01, 每 5 步取一个 x 分量
    dt, sigma, rho, beta = 0.01, 10.0, 28.0, 8.0 / 3
    x, y, z = 1.0, 1.0, 1.0
    out = np.empty(total)
    for i in range(total):
        for _ in range(5):
            def f(a, b, c):
                return sigma * (b - a), a * (rho - c) - b, a * b - beta * c
            k1 = f(x, y, z)
            k2 = f(x + dt * k1[0] / 2, y + dt * k1[1] / 2, z + dt * k1[2] / 2)
            k3 = f(x + dt * k2[0] / 2, y + dt * k2[1] / 2, z + dt * k2[2] / 2)
            k4 = f(x + dt * k3[0], y + dt * k3[1], z + dt * k3[2])
            x += dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
            y += dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
            z += dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
        out[i] = x
    return out


def make_dataset(kind, n, d_nuis, rng, L=L):
    """滑窗回归：X=(n, L*(1+d)), y=下一期主序列值。干扰维是结构化 AR(1)。"""
    total = n + L + 1
    main = _series(kind, total, rng)
    main = (main - main.mean()) / (main.std() + 1e-12)
    blocks = [main]
    for _ in range(d_nuis):
        v = np.zeros(total)
        for t in range(1, total):
            v[t] = 0.8 * v[t - 1] + 0.6 * rng.normal()
        blocks.append((v - v.mean()) / (v.std() + 1e-12))
    X = np.empty((n, L * len(blocks)))
    for b, s in enumerate(blocks):
        for j in range(L):
            X[:, b * L + j] = s[j:j + n]
    return X, main[L:L + n]


def init_model(D, rng):
    return {
        'W1': rng.normal(0, np.sqrt(1.0 / D), (D, HID)), 'b1': np.zeros(HID),
        'Wmu': rng.normal(0, np.sqrt(1.0 / HID), (HID, K)), 'bmu': np.zeros(K),
        'Wlv': rng.normal(0, np.sqrt(1.0 / HID), (HID, K)), 'blv': np.zeros(K),
        'Wo': rng.normal(0, np.sqrt(1.0 / K), (K, 1)), 'bo': np.zeros(1),
    }


class Adam:
    def __init__(self, P, lr=LR):
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}

    def step(self, P, G, it):
        for k in P:
            self.m[k] = 0.9 * self.m[k] + 0.1 * G[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * G[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** it)
            vh = self.v[k] / (1 - 0.999 ** it)
            P[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def predict(P, X):
    h = np.tanh(X @ P['W1'] + P['b1'])
    mu = h @ P['Wmu'] + P['bmu']
    return (mu @ P['Wo'] + P['bo']).ravel()


def train(Xtr, ytr, Xva, yva, beta, seed, iters=ITERS):
    # 三条独立随机流：初始化 / minibatch 采样 / 重参数化噪声。
    # 这样 β=0 与 β>0 的初始化与 batch 序列完全一致，差异只来自 β 本身。
    D = Xtr.shape[1]
    P = init_model(D, np.random.default_rng(seed + 2_000_003))
    g_idx = np.random.default_rng(seed)
    g_eps = np.random.default_rng(seed + 1_000_003)
    opt = Adam(P)
    ytr = ytr[:, None]
    hist = []
    n = len(Xtr)
    for it in range(1, iters + 1):
        idx = g_idx.integers(0, n, BS)
        xb = Xtr[idx]
        yb = ytr[idx]
        h = np.tanh(xb @ P['W1'] + P['b1'])
        mu = h @ P['Wmu'] + P['bmu']
        lv = np.clip(h @ P['Wlv'] + P['blv'], -6, 6)
        eps = g_eps.normal(size=mu.shape)
        z = mu + eps * np.exp(0.5 * lv) if beta > 0 else mu
        pred = z @ P['Wo'] + P['bo']
        mse = float(np.mean((pred - yb) ** 2))
        kl = float(0.5 * np.mean(np.sum(mu ** 2 + np.exp(lv) - 1 - lv, axis=1))) if beta > 0 else 0.0
        g = 2 * (pred - yb) / BS
        G = {'Wo': z.T @ g, 'bo': g.sum(0)}
        gz = g @ P['Wo'].T
        gmu = gz.copy()
        glv = np.zeros_like(lv)
        if beta > 0:
            gmu = gz + beta * mu / BS
            glv = gz * 0.5 * eps * np.exp(0.5 * lv) + beta * 0.5 * (np.exp(lv) - 1) / BS
        G['Wmu'] = h.T @ gmu
        G['bmu'] = gmu.sum(0)
        G['Wlv'] = h.T @ glv
        G['blv'] = glv.sum(0)
        gh = gmu @ P['Wmu'].T + glv @ P['Wlv'].T
        dh = gh * (1 - h ** 2)
        G['W1'] = xb.T @ dh
        G['b1'] = dh.sum(0)
        opt.step(P, G, it)
        if it == 1 or it % 25 == 0:
            hist.append((it, mse + beta * kl, mse, kl,
                         float(np.mean(np.abs(predict(P, Xva) - yva)))))
    return P, hist


def best_mae(hist):
    """验证集早停：取训练过程中最低的验证 MAE（两个模型各自取其最佳点）。"""
    return min(row[4] for row in hist)


def final_mae(hist):
    return hist[-1][4]


def run_ib(seed=SEED_IB, reps=3):
    """返回 (可 JSON 序列化的统计, 画图用的数组)。"""
    out, plots = {}, {}
    # 1) 三类 DGP，干扰维 8，plain(β=0) vs VIB(β=0.02)，早停 + 末点都记录
    by_dgp = {}
    for kind in DGPS:
        a, b, a_fin, b_fin = [], [], [], []
        for r in range(reps):
            rng = np.random.default_rng(seed + 100 * len(DGPS) + 17 * DGPS.index(kind) + r)
            X, y = make_dataset(kind, 2600, 8, rng)
            Xtr, ytr, Xva, yva = X[:2000], y[:2000], X[2000:], y[2000:]
            cs = seed + 9000 + r
            P0, h0 = train(Xtr, ytr, Xva, yva, 0.0, cs)
            Pv, hv = train(Xtr, ytr, Xva, yva, 0.02, cs)
            a.append(best_mae(h0)); b.append(best_mae(hv))
            a_fin.append(final_mae(h0)); b_fin.append(final_mae(hv))
            if kind == 'ar2' and r == 0:
                plots['curve_plain'] = h0
                plots['curve_vib'] = hv
                plots['P_vib'] = Pv
                plots['Xva'] = Xva
        by_dgp[kind] = {
            'plain': round(float(np.mean(a)), 6), 'vib': round(float(np.mean(b)), 6),
            'improve_pct': round(float(100 * (np.mean(a) - np.mean(b)) / np.mean(a)), 2),
            'plain_final': round(float(np.mean(a_fin)), 6),
            'vib_final': round(float(np.mean(b_fin)), 6),
            'overfit_gap_plain': round(float(np.mean(a_fin) - np.mean(a)), 6),
            'overfit_gap_vib': round(float(np.mean(b_fin) - np.mean(b)), 6),
        }
    out['val_mae_by_dgp'] = by_dgp

    # 2) β 扫描（ar2, d=8）——与主实验 ar2 用同一批数据/训练流，保证 β=0/0.02 可对拍
    scan, scan_final = {}, {}
    for beta in BETAS:
        vals, fin = [], []
        for r in range(reps):
            rng = np.random.default_rng(seed + 317 + r)
            X, y = make_dataset('ar2', 2600, 8, rng)
            _, h = train(X[:2000], y[:2000], X[2000:], y[2000:], beta, seed + 9000 + r)
            vals.append(best_mae(h)); fin.append(final_mae(h))
        scan[str(beta)] = round(float(np.mean(vals)), 6)
        scan_final[str(beta)] = round(float(np.mean(fin)), 6)
    out['beta_scan'] = scan
    out['beta_scan_final'] = scan_final
    out['best_beta'] = float(min(scan, key=scan.get))
    out['worst_beta'] = float(max(scan, key=scan.get))

    # 3) 干扰维扫描（ar2）——同一 rng seed 下主序列完全相同，只变干扰维数量
    nuis = {}
    for d in NUIS:
        a, b = [], []
        for r in range(reps):
            rng = np.random.default_rng(seed + 317 + r)
            X, y = make_dataset('ar2', 2600, d, rng)
            _, h0 = train(X[:2000], y[:2000], X[2000:], y[2000:], 0.0, seed + 9000 + r)
            _, hv = train(X[:2000], y[:2000], X[2000:], y[2000:], 0.02, seed + 9000 + r)
            a.append(best_mae(h0)); b.append(best_mae(hv))
        nuis[str(d)] = {'plain': round(float(np.mean(a)), 6),
                        'vib': round(float(np.mean(b)), 6),
                        'gap_pct': round(float(100 * (np.mean(a) - np.mean(b)) / np.mean(a)), 2)}
    out['nuisance_scan'] = nuis

    # 4) 瓶颈激活维度（β=0.02 模型在验证集上 mu 的方差）
    P = plots['P_vib']
    h = np.tanh(plots['Xva'] @ P['W1'] + P['b1'])
    mu = h @ P['Wmu'] + P['bmu']
    lv = np.clip(h @ P['Wlv'] + P['blv'], -6, 6)
    out['mu_var_by_dim'] = [round(float(v), 6) for v in mu.var(0)]
    out['kl_at_beta002'] = round(float(0.5 * np.mean(np.sum(mu ** 2 + np.exp(lv) - 1 - lv, axis=1))), 6)
    out['active_dims'] = int(np.sum(mu.var(0) > 0.01))
    out['train_curve_best'] = round(best_mae(plots['curve_vib']), 6)
    out['train_curve_plain_best'] = round(best_mae(plots['curve_plain']), 6)
    plots['series'] = {k: _series(k, 220, np.random.default_rng(seed + 77 + i))[:200]
                       for i, k in enumerate(DGPS)}
    return out, plots


if __name__ == '__main__':
    s, _ = run_ib()
    a = s['val_mae_by_dgp']['ar2']
    print('val_mae_plain     ', a['plain'])
    print('val_mae_vib       ', a['vib'])
    print('improve_pct       ', a['improve_pct'])
    print('overfit_plain     ', a['overfit_gap_plain'])
    print('overfit_vib       ', a['overfit_gap_vib'])
    print('best_beta         ', s['best_beta'])
    print('mae_at_best_beta  ', s['beta_scan'][str(s['best_beta'])])
    print('mae_at_beta03     ', s['beta_scan']['0.3'])
    print('gap_d0_pct        ', s['nuisance_scan']['0']['gap_pct'])
    print('gap_d16_pct       ', s['nuisance_scan']['16']['gap_pct'])
    print('kl_at_beta002     ', s['kl_at_beta002'])
    print('active_dims       ', s['active_dims'])
```
