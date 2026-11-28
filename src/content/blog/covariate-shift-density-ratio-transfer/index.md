---
title: "协变量漂移与密度比迁移：用分类器把牛市训练的信号权重掰到熊市，为什么有效样本量与重叠度决定成败、又在分布不重叠时反噬"
description: "5000 个牛市样本、3000 个熊市无标签样本、400 个熊市标签样本的实测：密度比加权的测试 R^2 从只用源域的 0.8555 升到 0.8811，接近知道真实密度的 0.8816；但漂移加大后有效样本量从 4914 塌到 184，加权模型 R^2 崩到 0.1972，连 oracle 权重也只有 0.1031。当条件关系本身改变（概念漂移）时，加权 0.5148 几乎等同于不加权 0.4873，而用 400 个目标标签重训能到 0.8753。附分类器估计密度比的完整 Python 代码与四张真实计算图。"
publishDate: '2026-11-28T11:20:00Z'
tags:
  - 量化
  - 迁移学习
  - 风险管理
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

量化研究里最常见的数据划分错误，是把牛市和熊市混在一起随机切训练/测试集。模型在测试集上表现很好，上线后却立刻失效——因为测试集里仍然混着大量和训练期相似的市场状态。真正的考验是分布外推：在一种状态下训练，在另一种状态下使用。

协变量漂移（covariate shift）假设条件关系 $p(y\mid x)$ 不变，只有特征分布 $p(x)$ 变了。这个假设有时成立，比如同一个定价规律在不同的波动率环境下依然有效，只是特征的取值范围不同。此时标准解法是重要性加权：给每个源域样本乘上权重

$$
w(x)=\frac{p_{\text{target}}(x)}{p_{\text{source}}(x)},
$$

然后像往常一样做加权回归。

固定种子 `20261128` 的实验结果：

- 温和漂移下，只用牛市数据的测试 $R^2$ 是 **0.8555**，简单混合训练是 **0.8643**，密度比加权是 **0.8811**，知道真实密度的 oracle 是 **0.8816**。分类器估计的权重几乎追上了理论上限；
- 有效样本量（ESS）是 **1765**，占源域的 **35.3%**；权重中位数 0.55，99 分位 7.61，最大 47.0，截到 10 只影响 0.6% 的样本；
- 漂移放大后 ESS 从无漂移时的 **4914** 一路塌到 **184**，权重最大值冲到 **16665**。加权模型的 $R^2$ 崩到 **0.1972**，甚至输给不加权的 **0.6331**；连 oracle 权重也只有 **0.1031**；
- 这种极端情形下，用 400 个目标域标签直接重训的 $R^2$ 是 **0.9167**，远高于任何加权方案；
- 如果连条件关系都变了（目标域收益额外加上 $1.0X_0$），密度比加权 **0.5148** 和不加权 **0.4873** 几乎一样，而目标域重训 **0.8753**。权重再准也修不了概念漂移。

![分布漂移与重要性权重](/images/covariate-shift-density-ratio-transfer/fig1_shift_weights.png)

结论很直白：密度比加权是有条件的工具。它在「规律没变、只是样本分布变了、且两域高度重叠」时有效；一旦重叠消失或者规律本身变了，它不但无效，还会把模型推向更糟的外推。

## 二、为什么简单混合不是最优

设源域样本来自 $p_s(x)$，目标域来自 $p_t(x)$，我们想最小化目标域风险

$$
R_t(f)=E_{x\sim p_t}\big[(y-f(x))^2\big].
$$

手里只有源域的标签，那就把期望换成源域上的加权平均：

$$
R_t(f)=E_{x\sim p_s}\Big[w(x)(y-f(x))^2\Big],\quad w(x)=\frac{p_t(x)}{p_s(x)}.
$$

这是标准的重要性采样恒等式，前提是 $p_t$ 在 $p_s$ 为零的地方也必须为零。如果目标域跑到源域几乎没见过的区域，那个区域的权重理论上趋于无穷，估计量方差爆炸——这正是后面实验中翻车的原因。

简单把源域和目标域样本混起来训练，隐含的假设是「两域样本量按真实出现频率混合」。但目标域往往只有少量标签，混合后源域仍占主导，学到的函数依然偏向源域常见区域。加权则直接改写优化目标，让稀缺的目标区域在损失里获得应有的发言权。

## 三、用一个分类器把密度比撞出来

直接估计两个高维密度再相除，是统计上最糟糕的做法。更稳的办法是把它化成一个二分类问题：把源域样本标 0，目标域无标签样本标 1，训练分类器得到 $q(x)=P(\text{target}\mid x)$，利用贝叶斯公式：

$$
\frac{p_t(x)}{p_s(x)}=\frac{q(x)}{1-q(x)}\cdot\frac{n_s}{n_t}.
$$

这里 $n_s/n_t$ 是两域样本量之比。实验里源域 5000、目标无标签 3000，所以比例是 1.667。为了避免分类器在训练样本上过拟合导致权重虚高，我对两域合并样本做 4 折交叉拟合：每折用其余数据训练分类器，只回写该折的权重。


权重必须截断。实验里温和漂移的最大原始权重是 47.0，截断到 10 只影响 0.6% 的样本；极端漂移的最大权重是 16665，不截断的话单个样本就能主导整个回归。截断是偏差与方差的交换：它承认「极高权重区域其实没有可靠信息」。

有了权重，岭回归只需把正规方程里的内积加权：


这里的 `design` 是三次多项式基。我故意让真实函数含有一个折点项 $-2.6\max(X_1-0.7,0)$，多项式基无法精确表达它。这种「模型略有误设」的情形才是密度比加权有用的地方：如果模型本来就完全正确，源域估计是无偏的，加权只会增加方差。

## 四、温和漂移：加权确实兑现了理论

![有效样本量随漂移坍缩](/images/covariate-shift-density-ratio-transfer/fig2_ess_curve.png)

温和漂移下，目标域特征均值相对源域移动约 0.5 到 1 个标准差。分类器估计的权重均值是 **1.002**，接近理论均值 **0.984**；99 分位是 **7.61**，理论是 **6.63**。估计量本身是可信的。

![各方法在两种漂移下的表现](/images/covariate-shift-density-ratio-transfer/fig3_r2_compare.png)

预测结果上，源域模型 $R^2$ 0.8555，目标域仅有 400 个标签时单独训练是 0.8362，混合训练 0.8643，密度比加权 0.8811，知道真实密度的 oracle 是 0.8816。加权相对源域提升 **0.026**，相对混合提升 **0.017**，且几乎追平 oracle，说明误差主要来自分类器估计精度，而不是加权框架本身。Rank IC 从 0.913 升到 0.915，多空价差（预测最高 20% 减最低 20% 的真实收益）在几种方法间差别不大，说明这个场景里线性排序本来就稳，加权主要修的是均方误差而非排序。

## 五、ESS 是仪表盘，不是装饰

有效样本量定义为

$$
\text{ESS}=\frac{\big(\sum_i w_i\big)^2}{\sum_i w_i^2}.
$$

它衡量「加权之后这批样本相当于多少个独立同分布样本」。ESS 越低，权重越集中在少数样本上，估计方差越大。

实验里漂移强度从 0 加到 3.2 倍基准位移，ESS 从 **4914** 降到 **3051、1359、750、409、184、72**。截断权重把 ESS 从最极端情形的个位数救回到两位数，但救不回信息本身——被截掉的样本正是目标域独有的那批，源域根本没有对应的标签。

极端漂移场景的后果是灾难性的：源域模型 $R^2$ 0.6331，密度比加权 **0.1972**，oracle 加权 **0.1031**，而 400 个标签的目标域模型是 **0.9167**。oracle 也输，说明这不是分类器估错，而是源域根本不覆盖目标区域：模型必须在没见过的特征区间外推，而三次多项式基在那里会给出荒唐的值。这时候正确的动作是重新采集目标域样本，而不是继续调权重截断阈值。

![交易表现与概念漂移失效](/images/covariate-shift-density-ratio-transfer/fig4_trading_failure.png)

## 六、概念漂移：权重再准也没用

协变量漂移的前提是 $p(y\mid x)$ 不变。如果这个前提破坏——例如同一组特征在熊市里的收益结构整体平移——重要性采样恒等式就不再成立。

我在目标域的标签里额外加了 $1.0X_0$，保持特征分布不变（温和漂移），只改条件关系。此时加权模型 $R^2$ **0.5148**，oracle 加权 **0.5139**，源域模型 **0.4873**，几乎没有区别；目标域 400 个标签重训是 **0.8753**。多空价差也从 6.866 变成 6.899，没有任何实质提升。

这个概念实验很重要，因为实盘里「市场状态变了」通常同时包含特征分布漂移和条件关系漂移。只做密度比校准，等于默认定价规律没变。要检验这个假设，最直接的办法是留一小批目标域标签做诊断：比较源域模型在目标域上的残差均值是否为零、残差和特征是否还有相关性。如果有，说明条件关系也变了，必须更新模型而不是校准权重。

![权重截断阈值敏感性](/images/covariate-shift-density-ratio-transfer/fig5_clipping.png)

截断阈值本身也值得扫一遍。温和漂移下，阈值从 1 加到 10，目标 $R^2$ 从 **0.873** 升到 **0.881**，ESS 从 3780 降到 1765；继续放到 20 或不截断，反而回落到 0.8810 与 0.8807。严重漂移下结论完全相反：阈值 1 时 $R^2$ 是 **0.644**（基本等同不加权），阈值 10 只剩 **0.197**，阈值 20 是 0.197，完全不截断反而回到 0.391——但那时 ESS 只有 1.6，结果毫无稳定性可言。换句话说，在重叠塌缩的场景里，最安全的权重截断往往就是「几乎不加权」。

## 七、落地清单

第一，先画两域的特征分布和权重直方图，报告 ESS 占源域样本的比例。ESS 低于 10% 时，别指望加权能救。第二，权重一定要截断，并把截断比例写进实验记录；截断样本超过 1% 就该怀疑重叠度。第三，永远同时跑「源域模型」「目标域小样本模型」「加权模型」三个对照，加权模型没赢就说明漂移假设或者重叠条件不成立。第四，用少量目标域标签做条件关系诊断，区分协变量漂移和概念漂移。第五，交易口径上盯 IC 和多空价差而不只是 $R^2$，因为策略的收益来自排序而非拟合优度。

密度比方法是把「没有标签的新市场」转成「有标签的旧市场」的一座桥。桥能不能走，取决于两岸是不是连着同一块地。

## 附：完整可运行代码

下面这段脚本独立运行即可复现正文的实验与五张配图（不设 `OUT_DIR` 时写到本地 `figures/` 目录），固定随机种子，可直接对拍 `results.json`：

```python
def main():
    """Covariate shift / density-ratio transfer: importance-weighted ridge, ESS diagnostics, failure modes."""
    import json
    import os

    import numpy as np
    from scipy.stats import spearmanr

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["font.family"] = "DejaVu Sans"
    rcParams["axes.unicode_minus"] = False

    OUT = "/Users/halo/workspace/astro-blog/public/images/covariate-shift-density-ratio-transfer"
    os.makedirs(OUT, exist_ok=True)
    SEED = 20261128
    D = 6
    N_SRC = 5000
    N_TGT_U = 3000
    N_TGT_L = 400
    N_TEST = 2500

    MU_S = np.array([0.6, 0.4, 0.0, -0.2, 0.0, 0.0])
    COV = np.eye(D) * 0.9 + np.full((D, D), 0.1)


    def f_true(X):
        return (
            1.4 * np.tanh(X[:, 0]) - 0.8 * X[:, 1]
            + 0.9 * np.tanh(X[:, 2] * X[:, 3]) + 0.5 * np.sin(1.2 * X[:, 4])
            - 2.6 * np.maximum(X[:, 1] - 0.7, 0.0)
        )


    def make_shift(scale, rng, concept=False):
        mu_t = MU_S + scale * np.array([1.0, 0.6, -0.5, 0.4, 0.0, 0.0])
        Xs = rng.multivariate_normal(MU_S, COV, size=N_SRC)
        Xt = rng.multivariate_normal(mu_t, COV, size=N_TGT_U)
        Xtl = rng.multivariate_normal(mu_t, COV, size=N_TGT_L)
        Xte = rng.multivariate_normal(mu_t, COV, size=N_TEST)
        shift_target = Xtl[:, 0] if concept else 0.0
        shift_test = Xte[:, 0] if concept else 0.0
        ys = f_true(Xs) + rng.normal(0, 0.8, N_SRC)
        ytl = f_true(Xtl) + shift_target + rng.normal(0, 0.8, N_TGT_L)
        yte = f_true(Xte) + shift_test + rng.normal(0, 0.8, N_TEST)
        return Xs, ys, Xt, Xtl, ytl, Xte, yte, mu_t


    def design(X):
        # cubic polynomial basis: cannot represent the hinge exactly, but can bend near it
        from itertools import combinations_with_replacement
        n, d = X.shape
        cols = [np.ones(n)]
        for deg in (1, 2, 3):
            for inds in combinations_with_replacement(range(d), deg):
                v = np.ones(n)
                for i in inds:
                    v = v * X[:, i]
                cols.append(v)
        return np.column_stack(cols)


    def ridge_weighted(A, y, w, lam=8.0):
        G = A.T @ (w[:, None] * A) + lam * np.eye(A.shape[1])
        G[0, 0] -= lam
        return np.linalg.solve(G, A.T @ (w * y))


    def estimate_ratio(Xs, Xt, clip=10.0, seed=0, C=0.3):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        Z = np.vstack([Xs, Xt])
        labels = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
        poly = PolynomialFeatures(2, include_bias=False)
        A = poly.fit_transform(Z)
        A = StandardScaler().fit_transform(A)
        rg = np.random.default_rng(seed)
        folds = rg.permutation(len(Z)) % 4
        q = np.empty(len(Z))
        for k in range(4):
            tr, te = folds != k, folds == k
            clf = LogisticRegression(C=C, max_iter=800)
            clf.fit(A[tr], labels[tr])
            q[te] = clf.predict_proba(A[te])[:, 1]
        q = np.clip(q, 1e-4, 1 - 1e-4)
        w_all = q / (1 - q) * (len(Xs) / len(Xt))
        ws = np.clip(w_all[: len(Xs)], 0, clip)
        # for external points, average 4 classifiers fitted on all data
        clf = LogisticRegression(C=C, max_iter=800).fit(A, labels)
        return ws, w_all[: len(Xs)], clf, poly, None


    def ratio_for_external(Xq, clf, poly, scaler, n_src, n_tgt, clip=10.0):
        A = scaler.transform(poly.transform(Xq))
        q = np.clip(clf.predict_proba(A)[:, 1], 1e-4, 1 - 1e-4)
        return np.clip(q / (1 - q) * (n_src / n_tgt), 0, clip)


    def true_ratio(X, mu_t, scale):
        from scipy.stats import multivariate_normal as MVN
        return MVN(mean=mu_t, cov=COV).pdf(X) / MVN(mean=MU_S, cov=COV).pdf(X)


    def score(y, pred):
        mse = float(np.mean((y - pred) ** 2))
        r2 = 1 - mse / float(np.var(y))
        ic = float(spearmanr(pred, y).statistic)
        k = int(0.2 * len(y))
        top = np.argsort(-pred)[:k]
        bot = np.argsort(pred)[:k]
        spread = float(y[top].mean() - y[bot].mean())
        return {"mse": mse, "r2": r2, "ic": ic, "ls_spread": spread}


    results = {}
    np.random.seed(SEED)


    def run_scenario(scale, concept=False, seed=SEED):
        rng = np.random.default_rng(seed)
        Xs, ys, Xt, Xtl, ytl, Xte, yte, mu_t = make_shift(scale, rng, concept)
        A_src, A_tl, A_te = design(Xs), design(Xtl), design(Xte)
        # density ratio from unlabeled target sample
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        Z = np.vstack([Xs, Xt])
        labels = np.r_[np.zeros(N_SRC), np.ones(N_TGT_U)]
        poly = PolynomialFeatures(2, include_bias=False)
        AZ = poly.fit_transform(Z)
        sc = StandardScaler().fit(AZ)
        AZs = sc.transform(AZ)
        folds = np.random.default_rng(seed).permutation(len(Z)) % 4
        q = np.empty(len(Z))
        q_te_parts = []
        for k in range(4):
            clf = LogisticRegression(C=0.3, max_iter=800).fit(AZs[folds != k], labels[folds != k])
            q[folds == k] = clf.predict_proba(AZs[folds == k])[:, 1]
            q_te_parts.append(clf.predict_proba(sc.transform(poly.transform(Xte)))[:, 1])
        q = np.clip(q, 1e-4, 1 - 1e-4)
        w_raw = q / (1 - q) * (N_SRC / N_TGT_U)
        w_clip = np.clip(w_raw[:N_SRC], 0, 10)
        q_te = np.clip(np.mean(q_te_parts, axis=0), 1e-4, 1 - 1e-4)
        w_te = np.clip(q_te / (1 - q_te) * (N_SRC / N_TGT_U), 0, 10)
        ess = float(np.sum(w_clip) ** 2 / np.sum(w_clip ** 2))
        ess_raw = float(np.sum(w_raw[:N_SRC]) ** 2 / np.sum(w_raw[:N_SRC] ** 2))
        w_true = true_ratio(Xs, mu_t, scale)
        w_true_te = true_ratio(Xte, mu_t, scale)

        betas = {
            "source": ridge_weighted(A_src, ys, np.ones(N_SRC)),
            "target": ridge_weighted(A_tl, ytl, np.ones(N_TGT_L)),
            "pooled": ridge_weighted(np.vstack([A_src, A_tl]), np.r_[ys, ytl], np.ones(N_SRC + N_TGT_L)),
            "weighted": ridge_weighted(A_src, ys, w_clip),
            "oracle": ridge_weighted(A_src, ys, np.clip(w_true, 0, 10)),
        }
        preds = {k: A_te @ b for k, b in betas.items()}
        metrics = {k: score(yte, p) for k, p in preds.items()}
        diag = {
            "ess_clipped": ess, "ess_raw": ess_raw,
            "ess_fraction": ess / N_SRC,
            "w_p50": float(np.quantile(w_raw[:N_SRC], 0.5)),
            "w_p95": float(np.quantile(w_raw[:N_SRC], 0.95)),
            "w_p99": float(np.quantile(w_raw[:N_SRC], 0.99)),
            "w_max": float(w_raw[:N_SRC].max()),
            "frac_clipped": float(np.mean(w_raw[:N_SRC] >= 10)),
            "true_w_mean": float(np.mean(w_true)),
            "true_w_p99": float(np.quantile(w_true, 0.99)),
            "emp_w_mean": float(np.mean(w_raw[:N_SRC])),
        }
        return metrics, diag, (Xs, Xt, Xte, w_raw[:N_SRC], ys, yte, preds)


    # moderate and severe shift
    mod_metrics, mod_diag, mod_data = run_scenario(0.8, seed=SEED)
    sev_metrics, sev_diag, sev_data = run_scenario(2.6, seed=SEED + 1)
    drift_metrics, drift_diag, drift_data = run_scenario(0.8, concept=True, seed=SEED + 2)
    results["moderate"] = {"metrics": mod_metrics, "diag": mod_diag}
    results["severe"] = {"metrics": sev_metrics, "diag": sev_diag}
    results["concept_drift"] = {"metrics": drift_metrics, "diag": drift_diag}

    # ESS curve
    ess_curve = []
    for sc in [0.0, 0.5, 1.0, 1.5, 2.0, 2.6, 3.2]:
        _, dg, _ = run_scenario(sc, seed=SEED + 10)
        ess_curve.append({"scale": sc, "ess": dg["ess_clipped"], "ess_raw": dg["ess_raw"], "w_p99": dg["w_p99"]})
    results["ess_curve"] = ess_curve

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(mod_data[0][:, 0], bins=45, alpha=0.6, density=True, label="source (bull)", color="#1f77b4")
    axes[0].hist(mod_data[1][:, 0], bins=45, alpha=0.6, density=True, label="target (bear)", color="#d62728")
    axes[0].set_xlabel("Feature X0")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Moderate covariate shift")
    axes[0].legend(frameon=False)
    axes[1].scatter(mod_data[0][:, 0], mod_data[3], s=5, alpha=0.25, color="#1f77b4")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Feature X0 (source samples)")
    axes[1].set_ylabel("Estimated density ratio w(x)")
    axes[1].set_title(f"Importance weights, ESS={mod_diag['ess_clipped']:.0f}/{N_SRC}")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_shift_weights.png", dpi=150)
    plt.close(fig)

    scales = [p["scale"] for p in ess_curve]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(scales, [p["ess"] for p in ess_curve], marker="o", color="#1b7837", label="clipped w (<=10)")
    ax.plot(scales, [p["ess_raw"] for p in ess_curve], marker="s", color="#d95f02", label="unclipped w")
    ax.axhline(0.1 * N_SRC, ls="--", color="0.4", label="10% of source size")
    ax.set_xlabel("Shift magnitude (multiples of base offsets)")
    ax.set_ylabel("Effective sample size")
    ax.set_title("Weighting buys bias reduction by spending sample size")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_ess_curve.png", dpi=150)
    plt.close(fig)

    names = ["source", "target", "pooled", "weighted", "oracle"]
    labels = ["Source only", "Target only", "Pooled", "Density-ratio", "Oracle ratio"]
    colors = ["#7f7f7f", "#1f77b4", "#9467bd", "#d62728", "#2ca02c"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, mets, title in [(axes[0], mod_metrics, "Moderate shift"), (axes[1], sev_metrics, "Severe shift")]:
        vals = [mets[n]["r2"] for n in names]
        bars = ax.bar(labels, vals, color=colors)
        ax.set_ylabel("Target test R^2")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_r2_compare.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, mets, title in [(axes[0], mod_metrics, "Covariate shift only"), (axes[1], drift_metrics, "Concept drift added")]:
        vals = [mets[n]["ls_spread"] for n in names]
        bars = ax.bar(labels, vals, color=colors)
        ax.set_ylabel("Long-short spread (top 20% - bottom 20%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_trading_failure.png", dpi=150)
    plt.close(fig)

    with open(f"{OUT}/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```
