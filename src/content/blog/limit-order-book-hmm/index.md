---
title: "限价单簿隐马尔可夫建模：用 HMM 把盘口状态分成安静与挤兑"
description: "限价单簿不是一直一个样——平静时价差薄、成交稀疏，受压时（ toxic flow、挤兑、闪崩前）订单流失衡、价差瞬间拉开。这个『状态切换』恰恰是马尔可夫的：今天的状态只取决于上一刻。本文用 numpy 从零实现 2 状态高斯 HMM（缩放 Baum-Welch EM + Viterbi 解码），在 1500 步合成盘口数据上把隐藏状态恢复出来，Viterbi 解码准确率 99.4%、转移矩阵 [[0.99,0.01],[0.055,0.945]] 几乎还原真值；并把状态后验概率接到一个做市商策略上——受压时减仓+加宽价差，库存风险（σ）从 50.3 砍到 19.6（−61%）、最大回撤从 −22.4% 收到 −8.6%（−62%）。附完整 Python 与四张真实仿真图。"
publishDate: '2026-08-31'
tags:
  - 量化交易
  - 市场微观结构
  - 隐马尔可夫模型
  - 限价单簿
  - 做市商
  - 状态空间
  -  Baum-Welch
  -  Python
language: Chinese
difficulty: advanced
---

做市商挣的是 spread，赔的是库存。最要命的是：盘口不是一直一个样。平静时买卖价差薄、成交稀疏，你挂个窄价差稳稳吃；可一旦有毒流（toxic flow）进来、或者闪崩前夜，订单流失衡（order flow imbalance, OFI）突然拉满、价差瞬间撕开，这时候还按平静期的窄价差接单，库存会被单向行情砸穿。

这个「状态切换」恰恰是马尔可夫的：**下一刻盘口是平静还是受压，基本只取决于这一刻是什么状态**，而不是取决于三个月前。隐马尔可夫模型（HMM）就是为这种「观测得到盘口、看不见背后状态」的场景生的：观测是 OFI 和中间价收益，隐藏状态是「平静 / 受压」。

本文用 numpy 从零实现 2 状态高斯 HMM（缩放 Baum-Welch EM 估计 + Viterbi 解码），在 1500 步合成盘口上把隐藏状态恢复出来，再把状态后验概率接到一个做市商策略上，量化「知道现在是牛是熊」值多少钱。

![中间价路径，红色背景标出真实受压状态：受压段 OFI 拉满、中间价抖动更剧烈](/images/limit-order-book-hmm/mid_price_states.png)

## 一、为什么是 HMM 而不是阈值规则

最直接的做法是给 OFI 设个阈值：超过就当受压。但这有三个坑：(1) 阈值拍脑袋，敏感度高；(2) 观测有噪声，单步 OFI 会乱跳，阈值规则会抖；(3) 它不利用「状态持续性」——真正的受压会持续几十步，单看一步容易误判。

HMM 把这三点一起解决：它同时估计「状态怎么转移」（转移矩阵 $A$）、「每个状态下观测长什么样」（发射高斯 $\mathcal{N}(\mu_k,\Sigma_k)$）、以及「当前最可能是哪个状态」（前向-后向平滑出的后验概率）。平滑后的状态概率不会因单步噪声乱跳，因为状态持续性被 $A$ 显式建模了。

## 二、从零实现：缩放 Baum-Welch EM

标准前向-后向在金融高频数据上会数值下溢（概率乘 1500 步直接归零），必须做缩放。核心是用对数概率 + 缩放因子 $c_t$：

```python
import numpy as np

def log_gauss(x, mu, cov):
    d = x - mu
    sign, logdet = np.linalg.slogdet(cov)
    return -0.5 * (d @ np.linalg.solve(cov, d) + logdet + 2*np.log(2*np.pi))

def baum_welch(obs, K=2, n_iter=80, seed=0):
    rng = np.random.default_rng(seed)
    Tn = len(obs)
    pi = np.array([0.7, 0.3])
    A = np.array([[0.95, 0.05], [0.05, 0.95]])
    mu = np.array([[0.01, 0.0003], [0.06, 0.0]])
    cov = np.array([[[0.001, 0], [0, 0.001]], [[0.002, 0], [0, 0.004]]])
    for _ in range(n_iter):
        logB = np.stack([np.array([log_gauss(obs[t], mu[k], cov[k])
                                   for k in range(K)]) for t in range(Tn)])
        # 缩放前向
        c = np.zeros(Tn); alpha = np.zeros((Tn, K))
        la0 = np.log(pi) + logB[0]; c[0] = np.logaddexp.reduce(la0)
        alpha[0] = np.exp(la0 - c[0])
        for t in range(1, Tn):
            num = np.logaddexp.reduce(np.log(alpha[t-1])[:, None] + np.log(A), 0) + logB[t]
            c[t] = np.logaddexp.reduce(num); alpha[t] = np.exp(num - c[t])
        # 缩放后向
        beta = np.zeros((Tn, K)); beta[Tn-1] = 1.0
        for t in range(Tn-2, -1, -1):
            beta[t] = (A @ (beta[t+1] * np.exp(logB[t+1]))) / np.exp(c[t+1])
            beta[t] /= beta[t].sum()
        gamma = alpha * beta; gamma /= gamma.sum(1, keepdims=True)
        # M-step
        pi = gamma[0] / gamma[0].sum()
        xi = np.zeros((Tn-1, K, K))
        for t in range(Tn-1):
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = alpha[t, i]*A[i, j]*np.exp(logB[t+1, j])*beta[t+1, j]
            xi[t] /= xi[t].sum()
        A = xi.sum(0); A /= A.sum(1, keepdims=True)
        for k in range(K):
            w = gamma[:, k]
            mu[k] = (w[:, None] * obs).sum(0) / w.sum()
            d = obs - mu[k]
            cov[k] = (w[:, None, None] * (d[:, :, None] @ d[:, None, :])).sum(0) / w.sum()
            cov[k] += 1e-8*np.eye(2)
    return pi, A, mu, cov, gamma
```

解码用 Viterbi（对数域动态规划）得到最可能的状态序列；因为 HMM 的状态标签是任意的（0/1 可交换），最后用一个暴力置换匹配把预测状态对齐到真实标签，再报准确率。

## 三、合成盘口：两种状态，两种分布

我造 1500 步 2 状态 HMM：状态 0（平静，占比约 85%）OFI 均值≈0、中间价收益小；状态 1（受压，占比约 15%）OFI 均值拉到 +0.09、中间价抖动放大 5 倍。观测是 `[OFI, 中间价收益]` 二维向量。

```python
A_true = np.array([[0.99, 0.01], [0.06, 0.94]])
mean0 = np.array([0.00, 0.0002]); cov0 = np.array([[0.0004, 0], [0, 0.0008]])
mean1 = np.array([0.09, 0.0000]); cov1 = np.array([[0.0025, 0], [0, 0.0040]])
# 用 A_true 生成隐藏状态序列，再按状态抽观测 ...
```

训练出来（部分关键输出）：

- **转移矩阵**估计 `[[0.99, 0.01], [0.055, 0.945]]`，几乎还原真值 `[[0.99, 0.01], [0.06, 0.94]]`——说明状态持续性学得对。
- **发射均值**估计 `[[0.0006, 0.0016], [0.0915, -0.0028]]`，受压态 OFI 均值 0.0915 精准命中真值 0.09。
- **Viterbi 解码准确率 99.4%**——HMM 几乎把每一步的真实状态都猜对了。

![订单流失衡 OFI 的两种分布：HMM 把平静簇（蓝）和受压簇（红）的参数都估了出来](/images/limit-order-book-hmm/ofi_states.png)

## 四、平滑后的状态概率：比单步阈值稳

上图是发射分布，但 HMM 真正比阈值规则强的地方在**平滑后验概率**。下图是过滤出的「受压」状态后验概率：它在真实受压段稳稳贴到 1，在平静段贴到 0，过渡带平滑收敛，而不是单步 OFI 那样乱跳。这个平滑概率才是可以放心接进交易逻辑的「状态置信度」。

![HMM 过滤出的『受压』状态后验概率：受压段贴 1、平静段贴 0，过渡平滑](/images/limit-order-book-hmm/stressed_prob.png)

## 五、接到做市商策略上：状态感知值多少钱

把状态后验概率（取 0.5 为阈值得离散状态）接到一个简单的做市商仿真：每步按 OFI 符号逆势接单（吃 spread、承担库存方向风险），价差 s0=0.0008。

- **固定策略**：不管什么状态，永远吃满 1 单位、价差不变。
- **状态感知策略**：预测为受压时，只吃 0.4 单位 + 把价差加宽到 1.6 倍（少接毒单、多收风险补偿）。

```python
for t in range(Tn):
    trade = np.sign(obs[t, 0]) if abs(obs[t, 0]) > 0.005 else 0.0
    mid_ret = obs[t, 1]
    stressed = pred_states[t] == 1
    # 固定策略
    eq_fixed += s0 * 1.0 - inv_f * mid_ret; inv_f += 1.0 * trade
    # 状态感知：受压时减仓 + 加宽价差
    fill_s = 0.4 if stressed else 1.0
    spread_s = s0 * (1.6 if stressed else 1.0)
    eq_state += spread_s * fill_s - inv_s * mid_ret; inv_s += fill_s * trade
```

结果：

| 指标 | 固定价差 | 状态感知 | 变化 |
|---|---|---|---|
| 库存标准差 σ | 50.3 | 19.6 | **−61.0%** |
| 最大回撤 | −22.45% | −8.58% | **−61.8%** |
| 期末净值（累计） | −129.7 | −34.7 | 少亏 73% |

![做市商净值：状态感知策略（红）库存风险显著更低、回撤更浅](/images/limit-order-book-hmm/mm_pnl.png)

受压段占全样本才 15%，可它贡献了绝大部分的逆风——状态感知策略在 85% 的平静期照样吃 spread，只在那 15% 的受压段收手加价，就把库存风险砍掉六成、回撤砍掉六成。这就是「知道现在是牛是熊」的实打实价值。

## 六、三个落地坑（诚实版）

1. **状态数要拍**。本文用 2 状态是因为场景干净。实盘常要 3–4 个（平静/正常/受压/危机），状态多了 EM 更难收敛、BIC 选状态数是必修课。
2. **高斯发射是近似**。真实 OFI/收益是尖峰厚尾的，正态发射会低估极端事件。可以换学生 t 发射或把观测做分箱，但 Baum-Welch 要改 E-step 积分。
3. **非平稳**。市场状态的定义会漂移（牛市的「受压」和熊市的「受压」不是一回事），滚动窗口或在线 EM 比全样本估计稳。本文用的是固定生成分布，所以恢复得好；实盘要加这个机制。

## 七、结论

限价单簿的「状态切换」是马尔可夫的，而 HMM 是把这个结构从观测里挖出来的标准工具：缩放 Baum-Welch 无溢出地估计转移矩阵与发射分布，Viterbi 解码出最可能的状态序列，前向-后向平滑出可信的后验概率。本文的合成实验里，HMM 把隐藏状态恢复到 99.4% 准确率、转移矩阵几乎还原真值，并证明把它接到做市商策略上能在受压段减仓加价，把库存风险和最大回撤各砍六成。它不预测价格方向，只回答一个更朴素也更有用的问题：**现在盘口，是安全的还是危险的**。

完整生成脚本与四张仿真图见 `gen_two_articles_aug31.py`（仓库内），所有数值固定 seed 可复现。
