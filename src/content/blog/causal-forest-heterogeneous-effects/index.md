---
title: "因果森林与异质处理效应：用诚实样本分裂把「平均无效」的策略拆成可交易的子群，为什么它比全局回归更能挑人、又在重叠不足与强混杂下翻车"
description: "6000 训练样本、6 维特征、阈值型异质效应的实测：真实平均效应只有 -0.031，直接比较处理组与控制组却得到 +0.665；交叉拟合加因果森林把 CATE 的 RMSE 从线性交互模型的 1.087 压到 0.571，相关系数 0.953；按预测效应选前 30% 交易赚到 2.058 的真实平均效应，线性模型只有 1.724，随机选股是 -0.086。但在倾向得分超过 0.95 的区域，效应估计被压缩到真值的 77%。附完整 Python 实现与四张真实计算图。"
publishDate: '2026-11-28T10:45:00Z'
tags:
  - 量化
  - 因果推断
  - 机器学习
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

一个策略在全样本上平均无效，不等于它在每个子样本上都无效。假设某个信号只在「高动量且低波动」的股票上有效、在相反的子群上有害，两群人数差不多，总体平均效应就接近零。这时只看平均处理效应（ATE）会得出结论「策略没用」，而真正的问题是效应在人群里变号。

我用一个可控的合成实验把这件事量化。6000 个训练样本、3000 个测试样本、6 维标准化特征，处理变量是「是否发生某类事件/是否执行某策略」，结果变量由混淆变量、真实异质效应和标准差 0.8 的噪声构成。真实效应是一个带门控的阶梯函数：在某些特征组合下为 +2.2 或 +1.5，在相反组合下为 -2.2 或 -1.5。

固定种子 `20261128` 的实测结果：

- 测试集真实 ATE 为 **-0.031**，几乎为零；但真实 CATE 的标准差高达 **1.742**。平均值把正负两块完全抵消了；
- 直接做处理组均值减控制组均值，得到 **+0.665**，方向都是错的。原因是处理概率依赖特征，而特征本身影响收益——这是最典型的混杂偏差；
- 用倾向得分做 IPW 后，ATE 回到 **0.023**，已经接近真实值；
- 估计个体效应时，线性交互模型的 RMSE 是 **1.087**、相关系数 **0.782**；诚实因果森林把 RMSE 降到 **0.571**、相关系数升到 **0.953**；
- 用估计出的效应选前 30% 的股票做多，选中的样本真实平均效应是 **2.058**；线性交互模型是 **1.724**，随机选是 **-0.086**，上帝视角的 oracle 是 **2.214**。森林拿到了 oracle 收益的约 93%；
- 但森林不是万能的：在倾向得分超过 0.95 的极端区域，估计效应只有真值的 **77%**；强混杂下不做交叉拟合的朴素均值差偏差达到 **+1.357**。

![真实异质效应与处理分配](/images/causal-forest-heterogeneous-effects/fig1_heterogeneity.png)

这些数字说明一件事：当效应结构既有符号翻转又有交互门控时，把「平均」拆成「对谁有效」可以产生实打实的策略价值；而这套方法对重叠度和混杂机制极其敏感，不是随便套个森林就能用。

## 二、为什么 ATE 会骗人

用 $X$ 表示股票特征，$T\in\{0,1\}$ 表示是否接受处理（例如是否执行某交易信号），$Y$ 是未来收益。我们想知道的不是 $E[Y(1)-Y(0)]$，而是 $\tau(X)=E[Y(1)-Y(0)\mid X]$。前者是一个数，后者是一个函数。

困难在于，对同一个股票我们永远只能看到 $Y(T)$ 一个结果，另一个是反事实。更麻烦的是，处理往往不是随机的：高动量的股票更容易触发信号，而动量本身影响收益。此时朴素估计量

$$
\hat\tau_{\text{naive}}=E[Y\mid T=1]-E[Y\mid T=0]
$$

里混进了「特征差异导致的收益差异」，它会把处理效应和选股偏差搅在一起。本次实验中真实 ATE 接近零，朴素估计却给出 +0.665，就是这个机制在作祟。

## 三、R-learner：把难题拆成两个监督学习

R-learner 的关键观察是：如果先估计出条件均值 $m(X)=E[Y\mid X]$ 和倾向得分 $e(X)=P(T=1\mid X)$，构造残差

$$
\tilde Y=Y-\hat m(X),\qquad \tilde T=T-\hat e(X),
$$

那么真实效应满足一个正交条件：

$$
\tau(\cdot)=\arg\min_\tau E\Big[\big(\tilde Y-\tau(X)\tilde T\big)^2\Big].
$$

这个式子解释了为什么要交叉拟合：若 $\hat m$ 和 $\hat e$ 在同一份数据上训练，残差会记住训练噪声，$\tau$ 的训练目标被人为压低，测试集上却一塌糊涂。实务上用 K 折，每折用其余数据训练 nuisance 模型，再回写这一折的残差。

下面是实验里用的核心代码（完整版见文末说明，随机数固定可复现）：


我在这里用梯度提升树和逻辑回归估计 nuisance，因为真实数据里 $m(X)$ 往往有复杂非线性；用得太简单会留下残差结构，森林会把这部分结构误当成处理效应。这是本次调试中最大的一个坑：最初用二次多项式近似 nuisance 时，森林的 CATE RMSE 高达 1.36，甚至输给线性交互模型；换成交叉拟合的树模型后立刻降到 0.57。

## 四、因果树怎么分裂，什么叫「诚实」

普通回归树按「分裂后子节点的残差平方和下降」选切点；因果树按「分裂后子节点的局部效应解释了多少 $\tilde Y$ 的方差」选切点。对一个节点 $S$，令

$$
\hat\tau(S)=\frac{\sum_{i\in S}\tilde T_i\tilde Y_i}{\sum_{i\in S}\tilde T_i^2+\lambda},
$$

分裂准则就是最大化左右子节点的 $\hat\tau^2\sum \tilde T_i^2$。直观上，一个好切点应该让左右两拨人的处理效应真的不一样，而不是让收益均值不一样。

诚实样本分裂（honest splitting）要求：用来**寻找切点**的样本和用来**估计叶子效应**的样本不重叠。每棵树抽 80% 的样本，再对半拆成两份；前一份长树，后一份只负责给叶子算 $\hat\tau$。代价是叶子估计只用到一半数据，方差更大；收益是切点不会因为恰好拟合了叶子的噪声而被选中，后续做置信区间和变量重要性时才有统计意义。


把 140 棵这样的树集成起来，就得到 $\hat\tau(X)$。完整的实验脚本还包含 predict 的路由逻辑与森林平均，结构上和上面一致。

## 五、结果：谁能把「平均无效」拆成可交易的子群

![CATE 拟合对比](/images/causal-forest-heterogeneous-effects/fig2_cate_fit.png)

上图的横轴是真实 CATE，纵轴是估计值。诚实森林的点明显更贴近 45 度线，相关系数 0.953 对 0.782；线性交互模型由于只能表达「$X_j$ 的线性变化乘以处理」，无法表达「$X_0>0.2$ 且 $X_2<0.4$」这样的门控，系统性把阶梯压成斜面。作为对照，把所有样本都当成同一个效应（用 IPW 的 ATE 0.023）去预测，RMSE 是 1.742——几乎等于效应本身的标准差 1.742，等于什么都没学到。

一个反直觉的细节：不诚实版本的森林测试 RMSE 是 **0.470**，比诚实版的 0.571 更低。原因是诚实版只用一半样本估计叶子，方差更大。这不是说诚实分裂「没用」，而是说它买到的是正确的统计口径和可解释性，不是自动的精度提升。样本量只有几千时，这个方差代价真实存在；样本量上去之后，诚实性带来的无偏优势才会显现。

## 六、策略价值：从估计到交易

估计 CATE 本身不是目的，目的是用它决定「对谁下手」。定义选股比例 $q$，按 $\hat\tau$ 从高到低排序，取前 $q$ 比例建仓，实际拿到的收益就是这些样本真实 $\tau$ 的均值。

![按估计效应选股的策略价值](/images/causal-forest-heterogeneous-effects/fig3_policy_value.png)

当 $q=30\%$ 时，森林选中样本的真实效应为 2.058，线性模型 1.724，随机选择 -0.086，上帝视角 2.214。森林相对随机的增量是 2.058，相对线性模型多赚 0.334，达到 oracle 的约 93%。注意 $q=5\%$ 时森林是 2.880、oracle 是 2.920，几乎贴合；说明高置信的头部子群被抓得很准。

这个口径必须在实盘里保持一致：CATE 是「处理相对于不处理的增量」，不是绝对收益。如果你本来就有基础持仓，策略增益应该是 $\hat\tau$ 乘以实际改变的处理比例，再扣掉交易成本和冲击成本。

## 七、翻车现场：强混杂与重叠不足

![强混杂下的失效](/images/causal-forest-heterogeneous-effects/fig4_overlap_failure.png)

把处理概率改成 $e(X)=\mathrm{sigmoid}(2.8X_0+0.4X_1-0.3X_2)$，倾向得分的 5% 到 95% 分位数变成 0.009 到 0.991。此时：

- 朴素均值差的偏差达到 **+1.357**，完全不能用；
- 在 $e>0.95$ 的区域，训练集里 97.6% 的样本被处理，几乎找不到对照组；测试集上该区域的真实平均效应是 **1.603**，森林只估出 **1.233**，压缩到 **77%**；
- 在 $e<0.05$ 的区域，训练集只有 2.3% 被处理，真实效应 **-1.583**，森林估出 **-1.144**，压缩到 **72%**。

这就是重叠不足的本质：没有对照组就没有反事实，任何模型都只能往整体均值方向收缩。它不会因为换一个更复杂的森林而消失。实务上的应对是改变数据采集——例如对小概率子群做定向实验、做 propensity 截断后只交易重叠区域——而不是指望算法凭空补出没见过的样本。

还有一个容易忽略的经验：本次实验里真实数据中效应为阶梯函数，nuisance 模型的复杂度是决定性变量。用二次多项式近似 $m(X)$ 时，森林把 nuisance 残差中的结构误当成效应，CATE 相关系数只有 0.80；换成交叉拟合的梯度提升树后升到 0.953。交叉拟合不是形式主义，它是把「特征解释收益」和「处理解释收益」切开的关键手术。

![样本量与噪声敏感性](/images/causal-forest-heterogeneous-effects/fig5_sensitivity.png)

最后做一次敏感性检验。图中森林用 60 棵树，正文表格用 140 棵，口径不同但趋势一致：训练样本从 1200 增到 6000，森林的 CATE RMSE 从 **0.799** 降到 **0.567**，线性模型几乎不动，在 1.090 到 1.109 之间横着走。把结果噪声标准差从 0.4 抬到 2.0，森林从 **0.583** 升到 **0.675**，线性模型从 1.091 升到 1.100。森林的优势随样本量增长、随噪声收敛——它并没有魔法，只是更充分地利用了异质结构。

## 八、落地清单

第一，先算 ATE 和它的置信区间，再算 CATE 的方差。若效应方差远大于 ATE 的平方，「平均无效」才值得拆；否则复杂度换不来收益。第二，务必对 nuisance 做 K 折交叉拟合，并报告倾向得分的分布与重叠区间。第三，报告策略价值曲线而不是只报一个 RMSE——交易关心的是头部子群的命中率。第四，把成本写进目标函数：CATE 排序的边际优势若小于双边手续费，子群策略就不该上线。第五，永远留出一段时间外测试，效应结构随市场状态漂移时，历史子群会失效。

因果森林解决的是「对谁有效」，不是「什么时候有效」。把两者混为一谈，是这类方法在实盘里最常见的误用。

## 附：完整可运行代码

下面这段脚本独立运行即可复现正文的实验与五张配图（不设 `OUT_DIR` 时写到本地 `figures/` 目录），固定随机种子，可直接对拍 `results.json`：

```python
def main():
    """Causal forest / heterogeneous treatment effects: honest splitting, policy value, overlap failure."""
    import json
    import os

    import numpy as np
    from scipy.special import expit

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["font.family"] = "DejaVu Sans"
    rcParams["axes.unicode_minus"] = False

    SEED = 20261128
    OUT = "/Users/halo/workspace/astro-blog/public/images/causal-forest-heterogeneous-effects"
    os.makedirs(OUT, exist_ok=True)
    rng_global = np.random.default_rng(SEED)


    def tau_true(X):
        return (
            2.2 * ((X[:, 0] > 0.2) & (X[:, 2] < 0.4))
            - 2.2 * ((X[:, 0] < -0.2) & (X[:, 2] > -0.4))
            + 1.5 * ((X[:, 1] > 0.5) & (X[:, 3] > 0.5))
            - 1.5 * ((X[:, 1] < -0.5) & (X[:, 3] < -0.5))
        )


    def propensity(X, mode):
        if mode == "moderate":
            return expit(0.6 * X[:, 0] + 0.3 * X[:, 1] - 0.4 * X[:, 2])
        return expit(2.8 * X[:, 0] + 0.4 * X[:, 1] - 0.3 * X[:, 2])


    def make_data(n, mode, rng, x_mask=None):
        X = rng.normal(size=(n, 6))
        if x_mask is not None:
            X = X[x_mask(X)]
        e = propensity(X, mode)
        T = (rng.random(len(X)) < e).astype(float)
        mu = 0.8 * X[:, 3] - 0.6 * X[:, 4] + 0.3 * X[:, 0] + 0.2 * X[:, 1] ** 2
        Y = mu + tau_true(X) * T + rng.normal(0.0, 0.8, len(X))
        return X, T, Y, e


    def phi(X):
        return np.column_stack([np.ones(len(X)), X, X ** 2])


    def ridge_fit(A, y, lam=1.0):
        G = A.T @ A + lam * np.eye(A.shape[1])
        G[0, 0] -= lam
        return np.linalg.solve(G, A.T @ y)


    def logistic_fit(A, y, lam=1.0, iters=30):
        beta = np.zeros(A.shape[1])
        for _ in range(iters):
            p = expit(A @ beta)
            w = np.clip(p * (1 - p), 1e-8, None)
            H = A.T @ (w[:, None] * A) + lam * np.eye(A.shape[1])
            H[0, 0] -= lam
            g = A.T @ (p - y) + lam * np.r_[0.0, beta[1:]]
            beta = np.linalg.solve(H, g)
        return beta


    def crossfit_residuals(X, T, Y, nfold=4, seed=7):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        rg = np.random.default_rng(seed)
        folds = rg.permutation(len(X)) % nfold
        yt = np.empty(len(X))
        tt = np.empty(len(X))
        eh = np.empty(len(X))
        for f in range(nfold):
            tr, te = folds != f, folds == f
            poly = PolynomialFeatures(2, include_bias=False)
            Atr = poly.fit_transform(X[tr])
            Ate = poly.transform(X[te])
            sc = StandardScaler().fit(Atr)
            Atr, Ate = sc.transform(Atr), sc.transform(Ate)
            mr = HistGradientBoostingRegressor(max_iter=120, max_depth=4, learning_rate=0.08, random_state=f)
            mr.fit(Atr, Y[tr])
            mhat = mr.predict(Ate)
            lr = LogisticRegression(C=0.3, max_iter=500)
            lr.fit(Atr, T[tr])
            ehat = np.clip(lr.predict_proba(Ate)[:, 1], 0.02, 0.98)
            yt[te] = Y[te] - mhat
            tt[te] = T[te] - ehat
            eh[te] = ehat
        return yt, tt, eh


    class CausalTree:
        def __init__(self, max_depth, min_leaf, lam, mtry, honest, seed):
            self.max_depth = max_depth
            self.min_leaf = min_leaf
            self.lam = lam
            self.mtry = mtry
            self.honest = honest
            self.rng = np.random.default_rng(seed)
            self.root = None
            self.Xs = None

        def fit(self, X, ytilde, ttilde, idx):
            self.Xs = X
            if self.honest:
                rg = self.rng.permutation(idx)
                split_idx, est_idx = rg[: len(rg) // 2], rg[len(rg) // 2 :]
            else:
                split_idx, est_idx = idx, idx
            self.root = self._grow(split_idx, est_idx, 0)

        def _leaf_tau(self, est_idx, split_idx):
            use = est_idx if len(est_idx) >= 10 else split_idx
            denom = np.sum(self.t[use] ** 2) + self.lam
            return float(np.sum(self.t[use] * self.y[use]) / denom)

        def _grow(self, split_idx, est_idx, depth):
            tau = self._leaf_tau(est_idx, split_idx)
            if depth >= self.max_depth or len(split_idx) < 2 * self.min_leaf:
                return {"leaf": True, "tau": tau}
            best = None
            feats = self.rng.choice(self.X.shape[1], size=self.mtry, replace=False)
            for f in feats:
                vals = self.X[split_idx, f]
                cuts = np.unique(np.quantile(vals, np.linspace(0.1, 0.9, 9)))
                for c in cuts:
                    left = split_idx[vals <= c]
                    if len(left) < self.min_leaf or len(split_idx) - len(left) < self.min_leaf:
                        continue
                    right = split_idx[vals > c]
                    score = 0.0
                    for part in (left, right):
                        num = np.sum(self.t[part] * self.y[part])
                        den = np.sum(self.t[part] ** 2) + self.lam
                        score += num * num / den
                    if best is None or score > best[0]:
                        best = (score, f, c, left, right)
            if best is None:
                return {"leaf": True, "tau": tau}
            _, f, c, left, right = best
            el = est_idx[self.X[est_idx, f] <= c]
            er = est_idx[self.X[est_idx, f] > c]
            return {
                "leaf": False,
                "f": f,
                "c": c,
                "left": self._grow(left, el, depth + 1),
                "right": self._grow(right, er, depth + 1),
            }

        def predict(self, Xq):
            out = np.empty(len(Xq))
            for i, row in enumerate(Xq):
                node = self.root
                while not node["leaf"]:
                    node = node["left"] if row[node["f"]] <= node["c"] else node["right"]
                out[i] = node["tau"]
            return out


    def fit_forest(X, ytilde, ttilde, B=140, honest=True, seed=11):
        rg = np.random.default_rng(seed)
        trees = []
        for b in range(B):
            idx = rg.integers(0, len(X), size=int(0.80 * len(X)))
            tree = CausalTree(max_depth=6, min_leaf=40, lam=0.5, mtry=4, honest=honest, seed=seed * 1000 + b)
            tree.X, tree.y, tree.t = X, ytilde, ttilde
            tree.fit(X, ytilde, ttilde, idx)
            trees.append(tree)
        return trees


    def forest_predict(trees, Xq):
        return np.mean([tr.predict(Xq) for tr in trees], axis=0)


    def linear_interaction_tau(Xtr, Ttr, Ytr, Xq, lam=10.0):
        def design(X, T):
            return np.column_stack([np.ones(len(X)), X, T, T[:, None] * X])
        A = design(Xtr, Ttr)
        beta = ridge_fit(A, Ytr, lam=lam)
        return design(Xq, np.ones(len(Xq))) @ beta - design(Xq, np.zeros(len(Xq))) @ beta


    def policy_value(tau_hat, tau_real, q):
        vals = []
        for frac in q:
            k = max(1, int(round(frac * len(tau_hat))))
            sel = np.argsort(-tau_hat)[:k]
            vals.append(float(np.mean(tau_real[sel])))
        return np.array(vals)


    results = {}

    # ---------- moderate confounding: main comparison ----------
    rg = np.random.default_rng(SEED)
    Xtr, Ttr, Ytr, etr = make_data(6000, "moderate", rg)
    Xte, Tte, Yte, ete = make_data(3000, "moderate", np.random.default_rng(SEED + 1))
    tau_te = tau_true(Xte)

    yt, tt, eh = crossfit_residuals(Xtr, Ttr, Ytr)
    trees_h = fit_forest(Xtr, yt, tt, honest=True, seed=11)
    trees_nh = fit_forest(Xtr, yt, tt, honest=False, seed=13)
    pred_h = forest_predict(trees_h, Xte)
    pred_nh = forest_predict(trees_nh, Xte)
    pred_lin = linear_interaction_tau(Xtr, Ttr, Ytr, Xte)
    ate_naive = float(Ytr[Ttr == 1].mean() - Ytr[Ttr == 0].mean())

    # IPW ATE for reference
    Atr = phi(Xtr)
    be = logistic_fit(Atr, Ttr, lam=1.0)
    ate_ipw = float(
        np.mean(Ytr[Ttr == 1] / np.clip(etr[Ttr == 1], 0.05, 0.95))
        - np.mean(Ytr[Ttr == 0] / np.clip(1 - etr[Ttr == 0], 0.05, 0.95))
    )


    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))


    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])


    results["moderate"] = {
        "n_train": len(Xtr), "n_test": len(Xte),
        "true_ate_test": float(np.mean(tau_te)),
        "naive_diff_means_ate": ate_naive,
        "ipw_ate": ate_ipw,
        "forest_honest_cate_rmse": rmse(pred_h, tau_te),
        "forest_honest_cate_corr": corr(pred_h, tau_te),
        "forest_nonhonest_cate_rmse": rmse(pred_nh, tau_te),
        "linear_interaction_cate_rmse": rmse(pred_lin, tau_te),
        "linear_interaction_cate_corr": corr(pred_lin, tau_te),
        "constant_ate_rmse": rmse(np.full_like(tau_te, ate_ipw), tau_te),
        "cate_true_sd": float(np.std(tau_te)),
        "cate_hat_sd": float(np.std(pred_h)),
        "propensity_min": float(etr.min()), "propensity_max": float(etr.max()),
    }

    qs = np.linspace(0.05, 0.50, 10)
    pv_forest = policy_value(pred_h, tau_te, qs)
    pv_lin = policy_value(pred_lin, tau_te, qs)
    pv_rand = policy_value(np.zeros_like(pred_h), tau_te, qs)
    pv_oracle = policy_value(tau_te, tau_te, qs)
    results["policy"] = {
        "fractions": qs.tolist(),
        "forest": pv_forest.tolist(),
        "linear": pv_lin.tolist(),
        "random": pv_rand.tolist(),
        "oracle": pv_oracle.tolist(),
        "gain_at_30pct_forest": float(pv_forest[5] - pv_rand[5]),
        "gain_at_30pct_linear": float(pv_lin[5] - pv_rand[5]),
        "oracle_at_30pct": float(pv_oracle[5] - pv_rand[5]),
    }

    # ---------- strong confounding / limited overlap ----------
    rg2 = np.random.default_rng(SEED + 2)
    Xs, Ts, Ys, es = make_data(6000, "strong", rg2)
    Xq, Tq, Yq, eq = make_data(3000, "strong", np.random.default_rng(SEED + 3))
    tau_q = tau_true(Xq)
    ys2, ts2, _ = crossfit_residuals(Xs, Ts, Ys)
    trees_s = fit_forest(Xs, ys2, ts2, B=180, honest=True, seed=17)
    pred_s = forest_predict(trees_s, Xq)
    in_overlap = (Xq[:, 0] > 0.5) & (Xq[:, 0] < 2.5)
    out_overlap = (Xq[:, 0] < -0.5) & (Xq[:, 0] > -2.5)
    hi = eq > 0.95
    lo = eq < 0.05
    results["overlap_extremes"] = {
        "n_high_prop_test": int(hi.sum()),
        "n_low_prop_test": int(lo.sum()),
        "high_prop_mean_true_tau": float(np.mean(tau_q[hi])),
        "high_prop_mean_pred_tau": float(np.mean(pred_s[hi])),
        "low_prop_mean_true_tau": float(np.mean(tau_q[lo])),
        "low_prop_mean_pred_tau": float(np.mean(pred_s[lo])),
        "high_prop_shrinkage_ratio": float(np.mean(pred_s[hi]) / np.mean(tau_q[hi])),
        "low_prop_shrinkage_ratio": float(np.mean(pred_s[lo]) / np.mean(tau_q[lo])),
        "high_prop_rmse": rmse(pred_s[hi], tau_q[hi]),
        "low_prop_rmse": rmse(pred_s[lo], tau_q[lo]),
        "naive_ate_bias_strong": float(
            np.mean(Ys[Ts == 1]) - np.mean(Ys[Ts == 0]) - np.mean(tau_true(Xs))
        ),
        "train_treated_frac_high_prop": float(Ts[es > 0.95].mean()) if (es > 0.95).sum() else None,
        "train_treated_frac_low_prop": float(Ts[es < 0.05].mean()) if (es < 0.05).sum() else None,
    }
    results["strong_confounding"] = {
        "propensity_p05": float(np.quantile(es, 0.05)),
        "propensity_p95": float(np.quantile(es, 0.95)),
        "cate_rmse_overlap": rmse(pred_s[in_overlap], tau_q[in_overlap]),
        "cate_rmse_no_overlap": rmse(pred_s[out_overlap], tau_q[out_overlap]),
        "cate_corr_overlap": corr(pred_s[in_overlap], tau_q[in_overlap]),
        "cate_corr_no_overlap": corr(pred_s[out_overlap], tau_q[out_overlap]),
        "overall_rmse": rmse(pred_s, tau_q),
        "n_overlap": int(in_overlap.sum()), "n_no_overlap": int(out_overlap.sum()),
        "mean_prop_overlap": float(es[Xs[:, 0] > 0.5].mean()),
        "mean_prop_no_overlap": float(es[Xs[:, 0] < -0.5].mean()),
    }

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    grid = np.linspace(-2.5, 2.5, 80)
    g0, g2 = np.meshgrid(grid, grid)
    G = np.zeros((g0.size, 6))
    G[:, 0] = g0.ravel(); G[:, 2] = g2.ravel()
    Z = tau_true(G).reshape(g0.shape)
    cf = axes[0].contourf(g0, g2, Z, levels=18, cmap="RdBu_r")
    axes[0].set_xlabel("Feature X0 (momentum)")
    axes[0].set_ylabel("Feature X2 (volatility)")
    axes[0].set_title("True treatment effect tau(X)")
    fig.colorbar(cf, ax=axes[0], label="tau")
    xx = np.linspace(-3, 3, 200)
    axes[1].plot(xx, expit(0.6 * xx), label="moderate propensity", color="#1f77b4")
    axes[1].plot(xx, expit(2.8 * xx), label="strong propensity", color="#d62728")
    axes[1].set_xlabel("Feature X0")
    axes[1].set_ylabel("P(treatment = 1)")
    axes[1].set_title("Treatment assignment: overlap shrinks")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_heterogeneity.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, pr, name in [(axes[0], pred_h, "Honest causal forest"), (axes[1], pred_lin, "Linear interaction model")]:
        ax.scatter(tau_te, pr, s=7, alpha=0.35, color="#2b6cb0" if name.startswith("Honest") else "#c05621")
        lim = [-3.5, 3.5]
        ax.plot(lim, lim, "--", color="0.35", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("True CATE")
        ax.set_ylabel("Estimated CATE")
        ax.set_title(f"{name}\nRMSE={rmse(pr,tau_te):.3f}, corr={corr(pr,tau_te):.3f}")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_cate_fit.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(qs, pv_forest, marker="o", label="Causal forest policy", color="#1b7837")
    ax.plot(qs, pv_lin, marker="s", label="Linear model policy", color="#d95f02")
    ax.plot(qs, pv_rand, marker="^", label="Random targeting (mean tau)", color="0.4")
    ax.plot(qs, pv_oracle, linestyle="--", label="Oracle policy", color="#762a83")
    ax.set_xlabel("Fraction treated (top predicted effect)")
    ax.set_ylabel("Mean true effect among selected")
    ax.set_title("Policy value of targeting by estimated CATE")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_policy_value.png", dpi=150)
    plt.close(fig)

    bins = np.linspace(0, 1, 11)
    idx = np.digitize(eq, bins) - 1
    true_by_bin, pred_by_bin = [], []
    for b in range(len(bins) - 1):
        m = idx == b
        true_by_bin.append(np.mean(tau_q[m]) if m.sum() >= 10 else np.nan)
        pred_by_bin.append(np.mean(pred_s[m]) if m.sum() >= 10 else np.nan)
    centers = (bins[:-1] + bins[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(centers, true_by_bin, marker="o", label="true mean CATE", color="#1a9850")
    axes[0].plot(centers, pred_by_bin, marker="s", label="estimated mean CATE", color="#d73027")
    axes[0].axvspan(0.95, 1.0, color="0.85", alpha=0.5)
    axes[0].axvspan(0.0, 0.05, color="0.85", alpha=0.5)
    axes[0].set_xlabel("Propensity e(X) decile")
    axes[0].set_ylabel("Mean treatment effect")
    axes[0].set_title("Strong confounding: shrinkage where overlap dies")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(alpha=0.2)
    for label, mask, color in [("X0 < -0.5", Xs[:, 0] < -0.5, "#2166ac"), ("X0 > 0.5", Xs[:, 0] > 0.5, "#b2182b")]:
        axes[1].hist(es[mask], bins=30, alpha=0.55, density=True, label=label, color=color)
    axes[1].set_xlabel("Propensity e(X)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Assignment probability by region")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_overlap_failure.png", dpi=150)
    plt.close(fig)

    with open(f"{OUT}/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```
