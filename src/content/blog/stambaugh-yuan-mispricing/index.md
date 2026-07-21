---
title: "Stambaugh-Yuan 误定价因子：两个因子吃掉 17 个异象的 alpha"
description: "Stambaugh-Yuan (2017) 发现学术界几百个「异象」大多不是独立 alpha，而是两坨系统性误定价的症状。本文从零用 PCA 把 17 个异象坍缩成 MGMT 与 PERF 两个误定价因子，并用时间序列回归证明：FF3 下异象平均留 4.28% 的年化 alpha，加入 MGMT+PERF 后塌到 0.58%。附完整 Python 与五类真实陷阱（中阶）。"
publishDate: '2026-07-22'
tags:
  - 量化交易
  - Stambaugh-Yuan
  - 误定价因子
  - 资产定价
  - 因子动物园
  - PCA
  - 异象
  - Python
language: Chinese
difficulty: intermediate
---

量化圈有个公开的尴尬：**因子动物园（zoo of factors）**。几十年来论文里冒出几百个「异象」——应计、资产增长、净发行、动量崩溃、盈利意外……每一个都号称能挖到别人挖不到的 alpha。可把它们放进同一个回归里，很多 alpha 就互相踩踏、消失大半。Stambaugh 和 Yuan（2017）给了一个干脆的解释：**这些异象里，有一大坨根本不是独立 alpha，而是「系统性误定价」的两种症状。**

结论先放这：**学术界几百个异象，可以被两个「误定价因子」管住——MGMT（管理行为异象簇，11 个）和 PERF（业绩交易异象簇，6 个）。** 本文用纯 numpy 从零合成真实定价的资产宇宙（FF3 + 两个潜在误定价因子），把 17 个异象造出来，再用 PCA 从它们的相关结构里把 MGMT/PERF 还原出来，最后用时间序列回归证明：FF3 模型下这 17 个异象平均还留 **4.28%** 的年化 alpha，而一旦加入 MGMT + PERF，平均 alpha 直接塌到 **0.58%**。也就是说，那 4.28% 里绝大部分是「误定价因子的伪装」，不是新 alpha。附完整 Python 与五类真实陷阱（中阶）。

![四个传统因子 vs 两个误定价因子的月均风险溢价：MGMT/PERF 量级堪比 SMB/HML](/images/stambaugh-yuan-mispricing/cover.png)

## 一、为什么「异象」会扎堆

先想一个反直觉的事实。如果几百个异象**真的是各自独立的定价错误**，那它们应该彼此没什么关系——你做空「高资产增长」和做多「低应计」是两件不搭界的事。可现实是：这些异象的因子模拟组合（FMP，factor mimic portfolio）之间**高度相关**。

Stambaugh-Yuan 的洞见是：把几百个异象按「它到底在赚哪种错误定价」分成两大家族——

- **MGMT 簇（管理行为）**：管理层在搞什么动作带来的错误定价。典型成员：资产增长（NOA）、投资（Invest）、净股票发行（NSI）、应计（Accrual）、杠杆变化、外部融资……一共 11 个。
- **PERF 簇（业绩交易）**：投资者对短期业绩过度反应带来的错误定价。典型成员：净营业资产、毛利、困境、O-Score、动量崩溃、低波动……一共 6 个。

这两族内部高度相关、跨族弱相关。于是 Stambaugh-Yuan 干脆用 PCA（带符号约束）从几十个 FMP 里抽出前两个主成分，命名成 MGMT 和 PERF，再用它们做资产定价。结果：加入这两个因子后，绝大多数异象的 alpha 归零。

**关键含义**：你费劲发现的「新异象」，可能只是 MGMT 或 PERF 的又一个线性组合。它能在你自己的回测里赚钱，但放进定价模型就什么都不剩——那它就不是 alpha，只是「已被定价的误定价」的代理变量。

---

## 二、从零合成真实定价的资产宇宙

为了干净地演示，我们造一个「真 alpha = 0」的世界：所有收益都只由 5 个因子驱动——

$$\text{收益} = \beta_{Mkt}Mkt + \beta_{SMB}SMB + \beta_{HML}HML + \beta_{MGMT}MGMT + \beta_{PERF}PERF + \text{特质噪声}$$

MGMT 和 PERF 是两个潜在因子，带现实量级的风险溢价（合成数据，量级参考历史）：

```python
import numpy as np

T = 360  # 360 个月
FF_LAM = {"Mkt": 0.006, "SMB": 0.0025, "HML": 0.004}
FF_SD  = {"Mkt": 0.044, "SMB": 0.026, "HML": 0.026}
MGMT_LAM, MGMT_SD = 0.0045, 0.030
PERF_LAM, PERF_SD = 0.0035, 0.034

def gen_factors(seed=20260722):
    rng = np.random.default_rng(seed)
    # FF3 之间轻度相关
    corr = np.array([[1.00, 0.20, 0.15],
                     [0.20, 1.00, 0.30],
                     [0.15, 0.30, 1.00]])
    L = np.linalg.cholesky(corr)
    z = rng.normal(0, 1, size=(T, 3)) @ L.T
    F = np.empty((T, 3))
    for j, name in enumerate(["Mkt", "SMB", "HML"]):
        F[:, j] = FF_LAM[name] + FF_SD[name] * z[:, j]
    # MGMT / PERF：彼此弱相关，与 FF3 弱相关
    zm = rng.normal(0, 1, T)
    zp = 0.25 * zm + np.sqrt(1 - 0.25**2) * rng.normal(0, 1, T)
    MGMT = MGMT_LAM + MGMT_SD * zm
    PERF = PERF_LAM + PERF_SD * zp
    return F, MGMT, PERF
```

注意：MGMT 与 PERF 之间相关 0.25，模拟「两种误定价有时同源、但大体独立」。这正是它们能当两个不同因子的前提。

---

## 三、把 17 个异象「生」出来

Stambaugh-Yuan 论文里有 11 个 MGMT 异象、6 个 PERF 异象。我们按「载荷归属」造出 17 个因子模拟组合（FMP）：前 11 个主要吃 MGMT 载荷，后 6 个主要吃 PERF 载荷，再叠一点点 FF3 和特质噪声。

```python
N_ANOM = 17

def anomaly_loadings():
    rng = np.random.default_rng(42)
    loads = np.zeros((N_ANOM, 5))  # (b_mkt, b_smb, b_hml, b_mgmt, b_perf)
    for i in range(N_ANOM):
        if i < 11:                      # MGMT 簇
            b_mgmt = 0.6 + 0.3 * rng.random()
            b_perf = 0.1 * rng.normal()
        else:                           # PERF 簇
            b_perf = 0.6 + 0.3 * rng.random()
            b_mgmt = 0.1 * rng.normal()
        loads[i] = [0.1*rng.normal(), 0.2*rng.normal(),
                    0.2*rng.normal(), b_mgmt, b_perf]
    return loads

def gen_anomalies(F, MGMT, PERF, loads, seed=7):
    rng = np.random.default_rng(seed)
    A = np.zeros((T, N_ANOM))
    for i in range(N_ANOM):
        b_mkt, b_smb, b_hml, b_mgmt, b_perf = loads[i]
        idio = rng.normal(0, 0.012, T)
        A[:, i] = (b_mkt*F[:,0] + b_smb*F[:,1] + b_hml*F[:,2]
                   + b_mgmt*MGMT + b_perf*PERF + idio)
    return A
```

这一步是整篇的「诚实前提」：**异象是真 alpha = 0 的，因为它们只是 MGMT/PERF 的线性组合 + 噪声。** 真实世界里你当然无法直接看到这两个因子，但你可以从 17 个异象的相关结构里把它们「反推」出来——这就是下一节的 PCA。

---

## 四、数据驱动还原 MGMT 与 PERF：带符号约束的 PCA

我们假装不知道 MGMT/PERF 的存在，只拿到 17 个异象的时间序列。对它们的相关矩阵做 PCA，看前两个主成分能不能还原出那两个潜在因子。

```python
def pca_sign(factor_matrix):
    X = factor_matrix - factor_matrix.mean(0)
    cov = X.T @ X / (len(X) - 1)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    pc1 = X @ eigvec[:, 0]
    pc2 = X @ eigvec[:, 1]
    return pc1, pc2, eigval
```

还原效果（见散点图与碎石图）：

![17 个异象 FMP 相关矩阵 + 前 2 个主成分解释力：PC1+PC2 吃掉 80.1% 截面方差](/images/stambaugh-yuan-mispricing/sy_anomaly_corr.png)

| 指标 | 数值 |
|---|---|
| 前 2 个 PC 解释方差 | **80.1%**（PC1=50.4%，PC2=29.7%） |
| PC1 与 MGMT 真实因子相关 | **0.892** |
| PC2 与 PERF 真实因子相关 | **0.728** |

**读这张图**：17 个异象的相关矩阵里有清晰的结构——前两个主成分就吃掉八成截面方差，说明它们背后真的只有「两股力量」。而这两股力量，和我们造数据时埋进去的 MGMT/PERF 高度吻合（相关 0.89 / 0.73）。这正是 Stambaugh-Yuan 方法的核心：**不用主观挑异象，让数据自己告诉你「到底有几坨错误定价」**。

> 符号约束：PCA 抽出的主成分方向是任意的，我们用「与 MGMT/PERF 的相关系数最大化」决定正负号，保证还原的因子和潜在因子同向。

---

## 五、核心检验：两个因子吃掉了异象的 alpha

现在做最关键的检验。对每一个异象，分别跑两次时间序列回归：

1. **FF3 模型**：收益对 Mkt/SMB/HML 回归，看截距（年化 alpha）；
2. **SY 模型**：收益对 Mkt/SMB/HML/MGMT/PERF 回归，再看截距。

```python
def ts_alpha(y, X):
    n = len(y)
    Xd = np.column_stack([np.ones(n), X])
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return coef[0] * 12.0   # alpha 年化

ff3_alpha, full_alpha = [], []
for i in range(N_ANOM):
    a3 = ts_alpha(A[:, i], F)
    a5 = ts_alpha(A[:, i], np.column_stack([F, MGMT, PERF]))
    ff3_alpha.append(a3); full_alpha.append(a5)
```

结果（见条形图对照）：

![FF3 下异象平均留 4.28% 年化 alpha，加入 MGMT+PERF 后塌到 0.58%](/images/stambaugh-yuan-mispricing/sy_factor_absorb.png)

| 模型 | 平均 |α| | 最大 |α| |
|---|---|---|
| FF3（仅 Mkt/SMB/HML） | **4.28%** | 6.29% |
| FF3 + MGMT + PERF | **0.58%** | 1.57% |

**这就是全文的拳头结论**：FF3 漏掉了 MGMT/PERF 这两种误定价，所以把异象当成 alpha 留了下来（平均 4.28%）；而一旦把这两个误定价因子加进定价模型，异象的 alpha 几乎被吃光（0.58%，且主要是特质噪声）。

现实中你没法直接拿到 MGMT/PERF（它们是论文作者用 PCA 从几百个异象里抽的）。但结论的方法是通用的：**每当你发现一个新异象，先问它是不是已有因子动物园成员的线性组合。** 是的话，它的 alpha 在定价模型里会归零，不值得单开一个策略。

---

## 六、五类真实陷阱（不拆穿就是自欺）

**陷阱 1：把「误定价因子的伪装」当成 alpha（最致命）。**
这是全文主题。你回测出一个异象赚 5%/年，激动地以为挖到金矿——但若它只是 MGMT 的代理，放进 SY 模型就归零。对策：**任何新因子都要跑「加入庞大因子集合后 alpha 还在不在」的检验**（ spanning test / GRS），而不是只和市场比。

**陷阱 2：PCA 的「成分神话」——主成分不等于可交易因子。**
抽出的 PC1/PC2 是统计构造，权重可正可负、可超过 100% 空头。要变成真实可投资的 MGMT/PERF 因子，得把成分投影成多空组合（或直接用论文定义的 11+6 个异象做多空），否则只是数学玩具。本文为演示用真实因子，真实落地要走这一步。

**陷阱 3：数据窥探 / 因子动物园的 p-hacking。**
几百个异象里总有几个「碰巧」显著。Stambaugh-Yuan 的 PCA 把冗余压成 2 个，本身就是一道抗过拟合——它不让你为每个异象单独讲故事。对策：把「因子数量」当成要估计的量（碎石图断点），而不是先入为主定死。

**陷阱 4：MGMT/PERF 自身的可交易性与换手。**
异象组合大多需要卖空小盘/低质股，而 A 股做空约束会让 MGMT 簇（重仓小盘、ST、高资产增长）根本建不起空头。对策：用沪深 300 成分股做中性化、或只做多头倾斜（long-only tilt），但要接受 alpha 打折。

**陷阱 5：样本内还原度是自证（circular）的。**
本文造数据时埋了 MGMT/PERF，PCA 当然能还原——这是演示不是证据。真实检验要用**样本外**：用一段数据估 PCA 载荷，在另一段数据上验证「加入前 2 PC 后异象 alpha 归零」。不跨样本，还原度毫无意义。

---

## 七、小结

Stambaugh-Yuan 给因子动物园泼了一盆冷水，但泼得很建设性：

- 几百个异象里，有一大坨只是 **MGMT（管理行为误定价）** 和 **PERF（业绩交易误定价）** 两种系统性错误的症状；
- 用 **PCA + 带符号约束** 可以从异象的相关结构里把它们还原出来（本文还原度 0.89 / 0.73，前 2 PC 吃 80% 方差）；
- 加入这两个因子后，17 个异象的 alpha 从平均 **4.28%** 塌到 **0.58%**——其余都是伪装；
- 对你我做策略的启示：**新异象先过「因子动物园 spanning 检验」**，别急着宣布挖到金矿。

一句话：**异象扎堆不是因为市场错了几百次，而是因为市场犯了两类错，被几百篇论文各自取了一个名字。**

> 代码与数据完全可复现：本文所有图由 `gen_sy_mispricing.py` 生成（纯 numpy，无 sklearn/torch 依赖）。FF3 下平均 |α|=4.28%、SY 下=0.58%；前 2 PC 解释 80.1% 方差；PC1-MGMT 相关 0.892、PC2-PERF 相关 0.728；组合误定价因子年化溢价≈5.49%，均直接打印在脚本输出里。
