---
title: "隐马尔可夫模型(HMM)市场状态识别：让策略知道现在是牛是熊"
description: "市场不是一条平稳的随机游走，而是牛、熊、高波动三段「性格」轮流登场。本文从零实现高斯 HMM（Baum-Welch EM + Viterbi 解码），把日收益切成可解释的隐藏状态，给出每个状态的均值/波动/转移矩阵，并诚实列出标签切换、只能描述不能预测等真实陷阱。"
publishDate: '2026-07-11'
tags:
  - 量化交易
  - 隐马尔可夫模型
  - 市场状态
  -  regime switching
  - 波动率建模
  - Python
language: Chinese
difficulty: advanced
---

几乎所有入门级量化模型都偷偷做了一个假设：**市场是平稳的（stationary）**。均值回归策略假定均值恒定，动量策略假定趋势会延续，连最朴素的均线金叉都默认「过去的平均关系」未来还成立。可你我都清楚：2020 年 3 月的美股和 2021 年的美股，根本不是同一个市场；2015 年的 A 股杠杆牛和之后的股灾，也不是同一套规律在跑。

更麻烦的是，绝大多数策略的「失效」都不是因为信号本身错了，而是因为**信号 born 于某种环境、却被用在了另一种环境里**。一个在慢牛里夏普 2 的策略，放到高波动的恐慌段可能亏得比随机游走还快。问题的根源只有一句话：传统模型把「现在到底是个什么环境」这个变量，悄悄假设成了常数。

结论先放这：**市场其实是一个「 regime-switching（状态切换）」系统——几段性格完全不同的 hidden state 轮流登场，我们观察到的价格只是它们投下的影子。** 隐马尔可夫模型（Hidden Markov Model, HMM）就是一套把「看不见的状态」从「看得见的收益」里反推出来的标准工具。在我们的 2400 天模拟数据上，自实现的 3 状态高斯 HMM 干净地切出：牛市（日均 +0.069%、年化波动仅 10.5%）、熊市（日均 −0.010%、波动 15.4%）、高波动（日均 +0.014%、波动高达 24.2%），三者的转移矩阵对角线全在 0.9 以上——状态非常「黏」。

![HMM 解码：同一根价格曲线被切成牛/熊/高波动三段](/images/hmm-market-regime/hmm_price_regime.png)

## 一、HMM 到底在建模什么

普通模型直接建模「收益 → 收益」。HMM 多了一层：**收益由当前隐藏状态决定，而状态之间按一个转移矩阵随机游走**。

- **隐藏状态 $S_t$**：你看不见，比如「牛市 / 熊市 / 高波动」；
- **观测 $X_t$**：你看得见的，比如当日收益率、滚动波动率；
- **发射概率 $B$**：给定状态，观测服从的分布（我们用对角高斯）；
- **转移矩阵 $A$**：$A_{ij}=P(S_{t+1}=j\mid S_t=i)$；
- **初始分布 $\pi$**。

这里要划清一个容易混的概念：**HMM 和「高斯混合模型（GMM）」不一样**。GMM 也是用多个高斯叠加去拟合数据，但它是「各状态独立采样」——抽一个样本，随机落到某个高斯里，样本之间毫无时序关联。可市场状态明显是有记忆、会黏连的：今天牛市，明天大概率还是牛市；一旦转熊，往往要跌一阵才回头。这种「状态之间有先后顺序」的依赖，只有带转移矩阵的 HMM 能建模，GMM 天生做不到。所以当你想捕捉的是**会持续一段时间的环境**，必须用 HMM，而不是把每天独立地丢进混合模型。

学习的目标只有一件事：给定一长串观测 $X_{1:T}$，反推出 $(A,B,\pi)$ 和最可能的状态序列。这件事没有闭式解，要靠 **EM（期望最大化）** 迭代，具体叫 **Baum-Welch 算法**；解码最可能序列则靠 **Viterbi 算法**。

## 二、从零实现：Baum-Welch EM + Viterbi

不依赖任何第三方 HMM 库，纯 numpy 实现。核心是带缩放的前向-后向（数值稳定），以及 M 步里用「状态后验」做加权平均。

为什么用 EM 而不是梯度下降？因为 HMM 的对数似然关于参数是**非凸**的，梯度法极易卡在糟糕的局部最优；EM 的「先根据当前参数算出状态后验 $\gamma$、再用后验加权更新参数」的迭代，虽也不保证全局最优，但配上多次随机重启和对数域运算，实践中稳定得多。代码里 `n_init=6` 次重启、保留对数似然最高的那次，正是为了躲开坏局部最优。

```python
import numpy as np

def fit_gaussian_hmm(X, K, n_init=6, n_iter=200, tol=1e-6, seed=0):
    N, Dd = X.shape
    rnd = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):                      # 多次随机初始化，保住最优解
        pi = np.full(K, 1.0 / K)
        A = rnd.dirichlet(np.ones(K), size=K)
        idx = rnd.choice(N, K, replace=False)
        mu = X[idx].copy()
        var = np.tile(X.var(axis=0, ddof=1) + 1e-6, (K, 1))
        logp_prev = -np.inf
        for _ in range(n_iter):
            # ---- E-step：前向-后向（带缩放 c_t 防下溢）----
            logB = np.array([ -0.5*np.sum(np.log(2*np.pi*var[k]))
                              -0.5*np.sum((X-mu[k])**2 / var[k], axis=1)
                              for k in range(K)]).T
            logpi, logA = np.log(pi+1e-300), np.log(A+1e-300)
            alpha = np.zeros((N, K)); c = np.zeros(N)
            alpha[0] = np.exp(logpi + logB[0]); c[0] = alpha[0].sum(); alpha[0] /= c[0]
            for t in range(1, N):
                alpha[t] = np.exp(logB[t]) * (alpha[t-1] @ np.exp(logA))
                c[t] = alpha[t].sum(); alpha[t] /= c[t]
            beta = np.ones(N)
            for t in range(N-2, -1, -1):
                beta[t] = (np.exp(logA) @ (np.exp(logB[t+1]) * beta[t+1])) / c[t+1]
            gamma = alpha * beta; gamma /= gamma.sum(1, keepdims=True)
            xi = np.zeros((N-1, K, K))
            for t in range(N-1):
                xi[t] = (alpha[t][:, None] * np.exp(logA)
                         * np.exp(logB[t+1])[None, :] * beta[t+1][None, :])
                xi[t] /= xi[t].sum()
            # ---- M-step：用状态后验做加权平均 ----
            pi = gamma[0].copy()
            A = xi.sum(0); A /= A.sum(1, keepdims=True)
            w = gamma.sum(0)
            for k in range(K):
                mu[k] = (gamma[:, k] @ X) / w[k]
                var[k] = (gamma[:, k] @ (X-mu[k])**2) / w[k] + 1e-6
            logp = np.sum(np.log(c))
            if abs(logp - logp_prev) < tol:
                break
            logp_prev = logp
        if best is None or logp_prev > best["logp"]:
            best = dict(logp=logp_prev, pi=pi, A=A, mu=mu, var=var)
    return best

def viterbi(X, pi, A, mu, var):
    N, K = X.shape[0], mu.shape[0]
    def logpdf(x, m, v):
        return -0.5*np.sum(np.log(2*np.pi*v)) - 0.5*np.sum((x-m)**2/v)
    delta = np.zeros((N, K)); psi = np.zeros((N, K), dtype=int)
    for k in range(K):
        delta[0, k] = np.log(pi[k]+1e-300) + logpdf(X[0], mu[k], var[k])
    for t in range(1, N):
        for k in range(K):
            seq = delta[t-1] + np.log(A[:, k]+1e-300)
            psi[t, k] = np.argmax(seq)
            delta[t, k] = np.max(seq) + logpdf(X[t], mu[k], var[k])
    path = np.zeros(N, dtype=int); path[-1] = np.argmax(delta[-1])
    for t in range(N-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
    return path
```

注意几个工程细节：**前向-后向必须带缩放因子 $c_t$**，否则几千步连乘概率直接下溢成 0；`n_init` 多次重启很关键，HMM 是非凸的，坏初始化会卡在局部最优；对角协方差把维度间解耦，既省参数又抗过拟合，对 2 维观测足够。

## 三、喂数据：为什么观测用「收益 + 滚动波动率」两维

只喂当日收益（1 维）时，三个高斯很容易「彼此重叠」——熊市和波动市的均值都接近 0，EM 分不清。加入 **20 日滚动波动率** 作为第二维，相当于给了模型一个「恐慌温度计」，状态立刻可分：

```python
roll_vol = np.array([r[max(0,t-20):t].std(ddof=1) for t in range(len(r))])
X = np.column_stack([r, roll_vol * 100.0])   # 收益 + 波动率（缩放对齐量纲）
hmm = fit_gaussian_hmm(X, K=3, seed=42)
path = viterbi(X, hmm["pi"], hmm["A"], hmm["mu"], hmm["var"])
```

拟合后，我们按「波动最大者为高波动、余下两者收益高者为牛市、低者为熊市」这个**后验语义规则**给状态贴标签（不是靠模型自己认名字——模型只认编号，标签是人的事后解释，详见第五节陷阱）。三个状态的真实画像：

![各状态统计：牛市低波动正收益，高波动状态波动是牛市的 ~3 倍](/images/hmm-market-regime/hmm_regime_stats.png)

| 状态 | 日均收益 | 年化波动 | 解读 |
|---|---|---|---|
| 牛市 | +0.069% | 10.5% | 慢牛，波动小 |
| 熊市 | −0.010% | 15.4% | 阴跌，略负 |
| 高波动 | +0.014% | 24.2% | 上蹿下跳，波动是牛市的约 3 倍 |

## 四、转移矩阵：状态有多「黏」

解码出的转移矩阵直接告诉我们「当前状态会持续多久」：

```python
A = hmm["A"]                      # 已按语义重排行列
print(np.round(A, 2))
# [[0.96 0.02 0.02]   牛市 → 96% 留在牛市
#  [0.02 0.93 0.05]   熊市 → 93% 留在熊市
#  [0.05 0.04 0.91]]   高波动 → 91% 留在高波动
```

![估计的转移矩阵：对角线全部 >0.9，状态非常黏](/images/hmm-market-regime/hmm_transition.png)

对角线全在 0.9 以上，意味着**每个状态平均要持续 10~50 个交易日才切换**——这正解释了为什么「追涨杀跌」在牛市里赚钱、在熊市里送钱：你赚的不是信号，是 regime 的持续性。落地用法有三：
1. **状态依赖仓位**：牛市给满仓、熊市砍到半仓、高波动降到 1/3；
2. **信号过滤**：只在「当前状态 == 训练该信号时的状态」时放单；
3. **风险预算**：高波动状态自动收紧止损、降低杠杆。

### 一段可直接用的 regime-aware 仓位代码

把上面三件事写成一个函数，每个交易日喂入当日观测，输出目标仓位——这才是 HMM 真正的落地形态，而不是只画一张漂亮的彩色曲线图：

```python
# 用「当前状态」决定目标杠杆：牛市满仓、熊市半仓、高波动 1/3
TARGET_LEV = {0: 1.0, 1: 0.5, 2: 0.33}
cur = path[-1]                       # Viterbi 解码的当日状态（0=牛,1=熊,2=高波动）
target = TARGET_LEV[cur]
# 高波动状态再叠加波动率目标：把组合波动钉在 12%
if cur == 2:
    ann_vol_now = r[-20:].std(ddof=1) * np.sqrt(252)
    target = min(target, 0.12 / max(ann_vol_now, 1e-6))
print(f"当前状态={labels[cur]}, 目标仓位={target:.2f}")
```

这套开关的核心价值**不是预测拐点**，而是承认环境在变、并据此调整风险预算：牛市里少赚一点无所谓，熊市和高波动里少亏一点，才是你活到下一个牛市的关键。它把一个模糊的「我感觉现在行情不太对」，变成了可回测、可监控、可问责的明确数字。

## 五、真实陷阱（诚实版）

**1. 标签切换（label switching）**：EM 对状态编号是任意的，每次跑可能 0/1/2 含义互换。生产里必须加一道「确定性重排」（如按均值/波动排序）才能保证可复现，否则今天的「牛市」可能是昨天的「熊市」。

**2. 描述 ≠ 预测**：HMM 给你的是「这段历史属于哪种状态」，不是「明天一定切换」。平滑后验（smoothed）用到了未来信息，只能回测；Viterbi 才是可实时的。

**3. 状态数 $K$ 要拍**：$K=2$ 太粗、$K=6$ 会硬凑出不存在的 regime 并过拟合。用 BIC 在 $K=2..5$ 上选，但别迷信——BIC 也只是近似。

**4. 对初始化和数据量敏感**：2400 天尚可，几百天的数据 EM 经常不收敛或掉进坏的局部最优。

**5. 高斯假设太「乖」**：真实收益的厚尾、波动聚集，单高斯拟合不了一整段危机。更稳的做法是用学生-t 发射分布，或干脆只把 HMM 当「状态探测仪」，下游再接 regime-specific 模型。

## 结语

HMM 不是「圣杯策略」，而是一架**状态显微镜**：它把「哪个环境」这个被传统模型忽视的维度，显式地摆到你面前。只要记住——它描述的是过去的环境划分，真正的 alpha 在于「识别环境后，你决定怎么做」。把标签切换和前视偏差这两道坑迈过去，它就是你策略开关里最便宜、也最被低估的那个旋钮。

> 本文数据与图表均由 `generate_hmm_market_regime_images.py` 用 numpy 从零计算生成（模拟 regime-switching 收益 + 自实现 EM/Viterbi），非占位图。把 `r` 换成你自己的日收益序列，整套代码可直接复用。
