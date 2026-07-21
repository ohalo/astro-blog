---
title: "双重机器学习 Double ML 因果估计：把「被混淆的因果」剥离干净"
description: "你算出「低波动股跑赢」就敢说低波动「导致」高收益吗？两者可能都被某个隐藏变量驱动——这是因果推断的死穴：混淆。双重机器学习 Double ML (Chernozhukov et al. 2018) 用两阶段机器学习（随机森林纯 numpy 从零实现）估计「处理→结果」之外的所有混淆路径 g(X)/f(X)，再用交叉拟合 cross-fitting 残差化，把处理效应 θ 从相关里撬成因果。自洽合成面板（Y=θ·D+g(X)+ε, D=f(X)+ν, g 与 f 共享非线性项）上：朴素 OLS 估到 1.88（真值 1.50，偏差 +0.38）、线性残差化 Double ML 仍 1.90（偏差 +0.40，抓不住非线性混淆）、随机森林 Double ML 压到 1.60（偏差 +0.10，蒙特卡洛 100 次 SD≈0.02）；并诚实拆穿 RF 收缩偏误/无交叉拟合必过拟合/混淆变量遗漏/模型误设/PLM 线性假设五类真实陷阱（中阶）。"
publishDate: '2026-07-22'
tags:
  - 量化交易
  - 因果推断
  - 双重机器学习
  - Double ML
  - 机器学习因果
  - 混淆变量
  - 随机森林
  - Python
language: Chinese
difficulty: intermediate
cover: "/images/double-ml-causal/cover.png"
---

你做因子研究，算出「低波动股票长期跑赢高波动」(低波动异象)，于是很自然地下结论：**低波动「导致」了高收益**。但这话藏着因果推断最大的坑——你确定是低波动*导致*高收益，而不是「两者都被某个第三变量驱动」？

那个第三变量，行话叫**混淆变量 (confounder)**。最可能的候选是「杠杆约束」：受杠杆限制的机构被迫买高 beta 股、把低 beta 股压便宜，于是低波动≠因果、只是和某种系统性风险偏好被同一个因子驱动。朴素回归 `return ~ low_vol` 把这条「从混淆变量漏进来的相关」一并算成低波动的贡献，**因果效应被高估**。

**结论先放这**：要回答「D 到底导致 Y 多少」，必须把混淆变量 X 对 D 和 Y 的"双重驱动"剥掉。双重机器学习 Double ML (Chernozhukov et al. 2018) 的聪明之处在于——它不要求你手写出 g(X)、f(X) 的精确形式，而是用机器学习(本文用纯 numpy 从零实现的随机森林)去*拟合*这两个 nuisance 函数，再用**交叉拟合 (cross-fitting)** 把过拟合偏差打散，最后对"残差对残差"做正交回归，得到几乎无偏的处理效应 θ。

**在我们的自洽合成面板（Y = 1.5·D + g(X) + ε，D = f(X) + ν，g 与 f 故意共享一个 X² 非线性项制造混淆）上**：朴素 OLS 把 θ 估到 1.877（偏差 +0.377），线性残差化版 Double ML 仍 1.900（偏差 +0.400——因为它抓不住非线性混淆），而随机森林 Double ML 把偏差压到 +0.096（蒙特卡洛 100 次均值 1.596，SD≈0.020）。诚实地说：**+0.096 还没到 0，这恰恰是随机森林"收缩偏误"的教科书级现身说法**——见文末陷阱第一条。但相比朴素 OLS 的 +0.38，它已经把混淆几乎吃干净。

![Double ML 概念：X 同时驱动 D 与 Y，残差化剥离混淆后回归得 θ](/images/double-ml-causal/cover.png)

---

## 1. 偏线性模型与正交化：把"因果"写成一个干净的公式

Double ML 最常用的设定是**偏线性模型 (Partially Linear Model, PLM)**：

```
Y = θ·D + g(X) + ε
D = f(X) + ν
```

- `D` 是处理变量（比如"是否是低波动股"，0/1 或连续）。
- `X` 是混淆变量（所有同时影响 D 和 Y 的可观测特征）。
- `g(X)` 是 X 对 Y 的"直接效果"，`f(X)` 是 X 对 D 的效果。
- `θ` 是我们真正想要的**因果效应**：在控制住 X 后，D 每动一单位，Y 动多少。

朴素 OLS 直接跑 `Y ~ D + X` 会怎样？只要 `g(X)` 或 `f(X)` 不是线性的（现实里几乎一定不是），OLS 就会把 `Cov(g(X), f(X))` 漏进来，于是 `θ̂` 系统性偏离真值。

**正交化 (orthogonalization)** 的思路干净得像做手术：

```
第一步：用 ML 分别估出 ĝ(X) 和 f̂(X)，得到残差
        Y_res = Y − ĝ(X)          （Y 里 X 解释不掉的部分）
        D_res = D − f̂(X)          （D 里 X 解释不掉的部分）
第二步：在正交后的残差上做简单回归
        θ̂ = Cov(D_res, Y_res) / Var(D_res)
             = Σ D_res·Y_res / Σ D_res²
```

直觉是：`D_res` 已经不含 X 的信息，`Y_res` 也只含"D 的效果 + 纯噪声"。两者相乘，X 的贡献被减掉了，**这步回归得到的斜率就是 θ**。

---

## 2. 从零实现：纯 numpy 随机森林 + 交叉拟合

关键工程点有两个：(1) 用随机森林而非线性回归去估 `g`/`f`，才能吃掉非线性混淆；(2) **交叉拟合 (cross-fitting)**——用"训练集之外的样本"去生成残差，否则 ML 过拟合会让残差里混进噪声、污染 θ。

下面是一棵纯 numpy 实现的随机森林回归器（决策树用方差下降划分、bagging + 特征子采样，无任何外部依赖）：

```python
import numpy as np

class SimpleRF:
    def __init__(self, n_trees=60, max_depth=8, min_samples_leaf=20,
                 subsample=0.7, seed=0):
        self.n_trees, self.max_depth = n_trees, max_depth
        self.min_samples_leaf, self.subsample, self.seed = min_samples_leaf, subsample, seed
        self.trees_ = []

    def _fit_tree(self, X, y, rng):
        n = X.shape[0]
        idx = rng.integers(0, n, size=n)          # bootstrap 子采样
        Xb, yb = X[idx], y[idx]
        feat_sample = max(1, int(np.sqrt(X.shape[1])))
        return self._build(Xb, yb, rng, feat_sample, 0)

    def _build(self, X, y, rng, feat_sample, depth):
        n = X.shape[0]
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            return {"leaf": True, "val": float(np.mean(y))}
        feats = rng.choice(X.shape[1], size=feat_sample, replace=False)
        ymean, yvar = np.mean(y), np.sum((y - ymean)**2)
        best = None
        for f in feats:                           # 候选特征里贪心找最优切分
            xcol = X[:, f]
            uniq = np.unique(xcol)
            if len(uniq) < 2: continue
            thr = (uniq[:-1] + uniq[1:]) / 2.0
            if len(thr) > 25: thr = rng.choice(thr, size=25, replace=False)
            for t in thr:
                left = xcol <= t
                nl = int(left.sum())
                if nl < self.min_samples_leaf or (n - nl) < self.min_samples_leaf: continue
                yl, yr = y[left], y[~left]
                sl = np.sum((yl - yl.mean())**2) + np.sum((yr - yr.mean())**2)
                if best is None or sl < best[0]:
                    best = (sl, int(f), float(t))
        if best is None or best[0] >= yvar - 1e-12:
            return {"leaf": True, "val": float(ymean)}
        f, t = best[1], best[2]
        left = X[:, f] <= t
        return {"leaf": False, "feat": f, "thr": t,
                "left": self._build(X[left], y[left], rng, feat_sample, depth + 1),
                "right": self._build(X[~left], y[~left], rng, feat_sample, depth + 1)}

    def _pred_tree(self, tree, X):
        if tree["leaf"]: return np.full(X.shape[0], tree["val"])
        out = np.empty(X.shape[0])
        left = X[:, tree["feat"]] <= tree["thr"]
        out[left] = self._pred_tree(tree["left"], X[left])
        out[~left] = self._pred_tree(tree["right"], X[~left])
        return out

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.trees_ = [self._fit_tree(X, y, rng) for _ in range(self.n_trees)]
        return self

    def predict(self, X):
        return np.mean([self._pred_tree(t, X) for t in self.trees_], axis=0)
```

接下来是 Double ML 的灵魂——**K 折交叉拟合**：

```python
def double_ml(Y, D, X, kind="rf", K=5, seed=0):
    n = len(Y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, K)
    Dres = np.empty(n); Yres = np.empty(n)
    for k, fk in enumerate(folds):
        mask = np.ones(n, bool); mask[fk] = False     # 训练集 = 折叠外
        if kind == "rf":
            g_hat = SimpleRF(seed=seed + k).fit(X[mask], Y[mask]).predict(X[fk])
            f_hat = SimpleRF(seed=seed + k + 1000).fit(X[mask], D[mask]).predict(X[fk])
        # 线性对照：普通 OLS 残差化（抓不住非线性混淆，故意保留做对比）
        else:
            Xd = np.column_stack([np.ones(mask.sum()), X[mask]])
            g_hat = (np.column_stack([np.ones(len(fk)), X[fk]]) @
                     np.linalg.lstsq(Xd, Y[mask], rcond=None)[0])
            f_hat = (np.column_stack([np.ones(len(fk)), X[fk]]) @
                     np.linalg.lstsq(Xd, D[mask], rcond=None)[0])
        Yres[fk] = Y[fk] - g_hat                     # 残差化：剥掉 X 的贡献
        Dres[fk] = D[fk] - f_hat
    theta = np.sum(Dres * Yres) / np.sum(Dres ** 2)  # 正交回归
    return theta, Yres, Dres

def naive_ols(Y, D):                                 # 朴素 OLS（有偏基准）
    Xd = np.column_stack([np.ones(len(D)), D])
    return np.linalg.lstsq(Xd, Y, rcond=None)[0][1]
```

注意 `double_ml` 里 `ĝ`/`f̂` 是在**折叠外**训练的——这正是 cross-fitting。如果你用同一批数据既训练又生成残差，随机森林会把样本内噪声"记住"，残差被压扁、θ 被严重低估（陷阱二）。

---

## 3. 合成数据：故意制造混淆，看谁剥得干净

为了可复现，我们构造一个"明确知道真值"的面板——这样估出来的 θ 离 1.50 多远，一眼看出偏差：

```python
THETA_TRUE = 1.5

def gen_data(n, seed, theta=THETA_TRUE):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, 2))
    # 故意让 g 与 f 共享 X0² 项 → 二者相关，制造混淆
    g = 0.6 * X[:, 0]**2 - 0.4 * X[:, 1]**2 + 0.3 * X[:, 0] * X[:, 1]
    f = 0.5 * X[:, 0]**2 + 0.3 * X[:, 1]
    D = f + rng.normal(0, 1.0, n)
    Y = theta * D + g + rng.normal(0, 1.0, n)
    return X, D, Y
```

因为 `g` 和 `f` 都含 `X0²`，所以 `D` 和 `Y` 通过 X 高度相关——这正是现实里"混淆变量同时驱动处理与结果"的缩影。

跑一轮看单条数据的残差化效果：

![Double ML 残差化：Y_res 对 D_res 正交回归，斜率≈θ](/images/double-ml-causal/dml_residualization.png)

图上 `Y_res` 对 `D_res` 的正交回归斜率 = 1.580，紧贴真值 1.50 的虚线。残差之所以围绕原点散开、且和 D_res 无明显弯曲，说明 ML 已经把 X 的非线性影响吃掉了——如果还在用朴素 OLS，这条线会被"从原点斜拉出去"的混淆趋势带偏。

---

## 4. 蒙特卡洛：三种估计器的偏差对决

单条数据有随机性，跑 100 次蒙特卡洛看均值与波动：

```python
naive, lin, rf = [], [], []
for b in range(100):
    X, D, Y = gen_data(4000, b * 7)
    naive.append(naive_ols(Y, D))
    lin.append(double_ml(Y, D, X, kind="linear", K=5, seed=b)[0])
    rf.append(double_ml(Y, D, X, kind="rf", K=5, seed=b)[0])
```

![蒙特卡洛 100 次：Double ML(随机森林) 把偏差压回真值附近](/images/double-ml-causal/dml_bias_comparison.png)

| 估计器 | θ 均值 | 偏差 | 蒙特卡洛 SD |
|---|---|---|---|
| 朴素 OLS | 1.877 | **+0.377** | 0.019 |
| 线性残差化 Double ML | 1.900 | **+0.400** | 0.019 |
| 随机森林 Double ML | 1.596 | **+0.096** | 0.021 |

三句话解读：

1. **朴素 OLS 偏差 +0.38**——它把混淆变量 X 通过 D 漏进来的相关全算成 θ 的贡献，估计值虚高 25%。
2. **线性残差化反而更糟 (+0.40)**——这是最反直觉也最重要的一课：Double ML 的框架对，但 nuisance learner 用错（线性回归）就彻底失效。因为 `g`/`f` 是非线性的，线性残差化根本没把 X 的影响剥干净，反而在残差里留下了 D 和 Y 仍被 X 相关的尾巴。
3. **随机森林 Double ML 把偏差压到 +0.096**——终于把混淆几乎吃干净。但这 +0.096 依然不是 0，它就是下面要拆的第一类陷阱。

---

## 5. 诚实拆穿：五类真实陷阱

**① 随机森林的"收缩偏误" (regularization bias)——我们的 +0.096 从哪来。**
树模型是*收缩*估计量：它把极端预测往中心拉，导致 `ĝ(X)` 系统性欠估、`D_res` 被轻微放大，θ 被向上偏。这不是我们实现错了，而是 RF 本身的属性。工业级 Double ML (如 Doubleml 库) 用 **RF 偏差校正 (bias correction) 或换成梯度提升树 (GBM)** 把这项压到接近 0。所以严谨说法是：Double ML 的*理论*无偏，但*你选的 learner* 可能带来有限样本偏误——选 learner 也是方法论的一部分。

**② 不做交叉拟合必过拟合。**
如果 `ĝ`/`f̂` 用同一批数据训练和预测残差，RF 会把样本内噪声记进去，残差平方和变小、θ 被严重低估（常被压成接近 0 甚至反向）。cross-fitting 用"折叠外"预测，是 Double ML 无偏性的工程基石，丢不得。

**③ 混淆变量 X 没列全 (omitted confounder)。**
Double ML 只能控制*你观测到的* X。如果真有个驱动 D 和 Y 的隐藏变量没进 X，偏差原封不动留着。因果识别永远依赖"已观测混淆"这个不可证伪的假设——文献里叫"条件独立性 / 无未观测混淆假设 (CIA)"。做不出来的部分，得靠研究设计（自然实验、工具变量）补。

**④ nuisance 函数学不出来 (model misspecification)。**
Double ML 无偏的前提是 `ĝ→g`、`f̂→f`（consistency）。如果你的 learner 容量太小、或 X 和 Y/D 的关系本身极复杂，残差没剥干净，θ 仍偏。 learner 越强 → 偏差越小，但 learner 越强 → 越依赖交叉拟合防过拟合，二者耦合。

**⑤ 偏线性假设 (PLM) 本身可能不成立。**
PLM 假设 θ 对 D 是线性的（"D 每多一单位，Y 多 θ"）。但真实世界里处理效应可能随 X 变化（异质处理效应）、甚至 D 和 X 有交互。这时 PLM 给出的只是"平均处理效应 (ATE)"，想看 CATE（条件平均处理效应）得上 **因果森林 (Causal Forest / GRF)**，那是另一个话题。

---

## 6. 落到量化：它能回答什么、不能回答什么

Double ML 在量化里最适合的问题是**"因子有没有因果 alpha"**这一类：

- 你想验证"加入北向资金流之后，某因子的超额收益还剩多少"——把北向流、市值、行业等当 X，因子暴露当 D，未来收益当 Y，Double ML 给你"因子暴露的纯因果贡献"。
- 你想测"某事件（如纳入指数）的真实价格效应"——把同期宏观/风格当 X，事件指示当 D，事件窗收益当 Y。

它**不能**回答"明天涨不涨"（那是预测，不是因果）、也**不能**绕开 CIA 假设凭空造出因果。它的价值是把"我观察到 D 和 Y 相关"升级成"我控制了一篮子可观测混淆后，D 仍对 Y 有 θ 的净效应"——这已经是绝大多数因子研究里最稀缺的一步。

> 真实落地时，用 `DoubleML` (Python) 或 `Doubleml` (R) 库替你管交叉拟合、偏差校正与聚类标准误；本文纯 numpy 实现只为把"残差化 + 正交回归"的每一步摊开给你看。数据用你自己的面板，但记住：**X 的选取比 learner 的选型更决定成败。**
