#!/usr/bin/env python3
"""
为文章「双重机器学习 Double ML 因果估计」(double-ml-causal) 生成真实配图 + 可复现指标。

所有图表都由文中代码真实计算生成（纯 numpy，无 sklearn 依赖）：

  1) cover.png               —— 概念图：混淆变量 X 同时驱动处理 D 与结果 Y，Double ML 用 ML 残差化剥离
  2) dml_residualization.png —— 单条数据集上 RF 残差化的 (Y_res vs D_res) 散点 + 正交回归线（斜率≈θ）
  3) dml_bias_comparison.png —— 蒙特卡洛 100 次：朴素 OLS / 线性残差化 / Double ML(RF) 三种估计的偏差对比

机制（数据自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 偏线性模型 Y = θ·D + g(X) + ε，D = f(X) + ν；X 是混淆变量，g、f 含非线性项
  - 朴素 OLS 把 Cov(g, f) 漏进来 → 有偏；线性残差化抓不到非线性 → 仍有偏
  - Double ML 用随机森林(纯 numpy 实现)估计 g、f，交叉拟合(cross-fitting)去过拟合，正交化后 θ 无偏
"""
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

for _cand in ["/System/Library/Fonts/STHeiti Medium.ttc",
              "/System/Library/Fonts/Supplemental/Songti SC.ttf"]:
    try:
        fm.fontManager.addfont(_cand)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Heiti SC", "Songti SC", "STHeiti", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight",
})

BASE = "/Users/halo/workspace/astro-blog/public/images"
IMG = os.path.join(BASE, "double-ml-causal")
os.makedirs(IMG, exist_ok=True)

C = {"treat": "#4C72B0", "ctrl": "#9E9E9E", "gold": "#E1A100",
     "pos": "#55A868", "neg": "#C44E52", "gap": "#C44E52", "purple": "#8172B3"}

THETA_TRUE = 1.5

# ---------------------------------------------------------------------------
# 纯 numpy 随机森林回归器（Double ML 的 nuisance learner）
# ---------------------------------------------------------------------------
class SimpleRF:
    def __init__(self, n_trees=60, max_depth=8, min_samples_leaf=20, subsample=0.7, seed=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.seed = seed
        self.trees_ = []

    def _fit_tree(self, X, y, rng):
        n = X.shape[0]
        idx = rng.integers(0, n, size=n)
        Xb, yb = X[idx], y[idx]
        feat_sample = max(1, int(np.sqrt(X.shape[1])))
        return self._build(Xb, yb, rng, feat_sample, 0)

    def _build(self, X, y, rng, feat_sample, depth):
        n = X.shape[0]
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            return {"leaf": True, "val": float(np.mean(y))}
        feats = rng.choice(X.shape[1], size=feat_sample, replace=False)
        ymean = np.mean(y)
        yvar = np.sum((y - ymean) ** 2)
        best = None
        for f in feats:
            xcol = X[:, f]
            uniq = np.unique(xcol)
            if len(uniq) < 2:
                continue
            thr = (uniq[:-1] + uniq[1:]) / 2.0
            if len(thr) > 25:
                thr = rng.choice(thr, size=25, replace=False)
            for t in thr:
                left = xcol <= t
                nl = int(left.sum())
                if nl < self.min_samples_leaf or (n - nl) < self.min_samples_leaf:
                    continue
                yl, yr = y[left], y[~left]
                sl = np.sum((yl - yl.mean()) ** 2) + np.sum((yr - yr.mean()) ** 2)
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
        if tree["leaf"]:
            return np.full(X.shape[0], tree["val"])
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


def ml_predict(kind, Xtr, ytr, Xte, seed):
    if kind == "linear":
        Xd = np.column_stack([np.ones(len(Xtr)), Xtr])
        beta, *_ = np.linalg.lstsq(Xd, ytr, rcond=None)
        return np.column_stack([np.ones(len(Xte)), Xte]) @ beta
    # rf
    rf = SimpleRF(n_trees=30, max_depth=4, min_samples_leaf=25, seed=seed)
    rf.fit(Xtr, ytr)
    return rf.predict(Xte)


def gen_data(n, seed, theta=THETA_TRUE):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, 2))
    # 混淆变量 X 的非线性函数：g(X) 与 f(X) 共享 X0^2 项 → 二者相关
    g = 0.6 * X[:, 0] ** 2 - 0.4 * X[:, 1] ** 2 + 0.3 * X[:, 0] * X[:, 1]
    f = 0.5 * X[:, 0] ** 2 + 0.3 * X[:, 1]
    D = f + rng.normal(0, 1.0, n)
    Y = theta * D + g + rng.normal(0, 1.0, n)
    return X, D, Y


def naive_ols(Y, D):
    Xd = np.column_stack([np.ones(len(D)), D])
    beta, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    return beta[1]


def double_ml(Y, D, X, kind="rf", K=4, seed=0):
    n = len(Y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, K)
    Dres = np.empty(n)
    Yres = np.empty(n)
    for k, fk in enumerate(folds):
        mask = np.ones(n, bool); mask[fk] = False
        g_hat = ml_predict(kind, X[mask], Y[mask], X[fk], seed=seed + k)
        f_hat = ml_predict(kind, X[mask], D[mask], X[fk], seed=seed + k + 1000)
        Yres[fk] = Y[fk] - g_hat
        Dres[fk] = D[fk] - f_hat
    theta = np.sum(Dres * Yres) / np.sum(Dres ** 2)
    return theta, Yres, Dres


# ---------------------------------------------------------------------------
# 1) 封面概念图
# ---------------------------------------------------------------------------
def make_cover():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    def box(x, y, w, h, text, fc, tc="white", fs=12):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                        fc=fc, ec="none"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=tc, fontsize=fs, fontweight="bold")

    box(4.0, 4.4, 2.0, 1.0, "混淆变量 X", C["ctrl"], "black", 13)
    box(0.8, 1.6, 2.2, 1.1, "处理 D", C["treat"], "white", 13)
    box(7.0, 1.6, 2.2, 1.1, "结果 Y", C["gold"], "black", 13)
    # X -> D, X -> Y (confounded arrows)
    ax.annotate("", xy=(1.9, 2.7), xytext=(4.7, 4.4),
                arrowprops=dict(arrowstyle="->", color=C["ctrl"], lw=2))
    ax.annotate("", xy=(7.2, 2.7), xytext=(5.3, 4.4),
                arrowprops=dict(arrowstyle="->", color=C["ctrl"], lw=2))
    # D -> Y (causal, dashed)
    ax.annotate("", xy=(7.2, 2.7), xytext=(3.0, 2.7),
                arrowprops=dict(arrowstyle="->", color=C["neg"], lw=2.5,
                                linestyle=(0, (5, 3))))
    ax.text(5.0, 3.05, "θ (因果效应)", color=C["neg"], fontsize=11,
            ha="center", fontweight="bold")

    ax.text(2.9, 1.15, "被 X 混淆 → OLS 有偏", color=C["neg"], fontsize=10,
            ha="center")
    ax.text(5.0, 0.55, "Double ML：用随机森林估计 g(X)、f(X)，交叉拟合后残差化\n"
                        "Y_res 与 D_res 正交 ⇒ 正交回归得无偏 θ", color="#333333",
            fontsize=10.5, ha="center")
    ax.set_title("双重机器学习：把「被混淆的因果」剥离干净", fontsize=14,
                 fontweight="bold", pad=10)
    fig.savefig(os.path.join(IMG, "cover.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2) 残差化散点 + 正交回归线（单条代表性数据）
# ---------------------------------------------------------------------------
def make_residualization(seed=20260722):
    X, D, Y = gen_data(4000, seed)
    _, Yres, Dres = double_ml(Y, D, X, kind="rf", K=4, seed=seed)
    b = np.sum(Dres * Yres) / np.sum(Dres ** 2)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.scatter(Dres, Yres, s=8, alpha=0.25, color=C["treat"], label="残差样本")
    xs = np.linspace(Dres.min(), Dres.max(), 50)
    ax.plot(xs, b * xs, color=C["neg"], lw=2.5, label=f"正交回归 斜率={b:.3f}")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("处理残差  D − f̂(X)")
    ax.set_ylabel("结果残差  Y − ĝ(X)")
    ax.set_title(f"Double ML 残差化：Y_res 对 D_res 回归，斜率≈θ={THETA_TRUE}",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(os.path.join(IMG, "dml_residualization.png"))
    plt.close(fig)
    return b


# ---------------------------------------------------------------------------
# 3) 蒙特卡洛偏差对比
# ---------------------------------------------------------------------------
def make_bias_comparison(B=100, seed=0):
    naive, lin, rf = [], [], []
    for b in range(B):
        X, D, Y = gen_data(4000, seed + b * 7)
        naive.append(naive_ols(Y, D))
        lin.append(double_ml(Y, D, X, kind="linear", K=4, seed=seed + b)[0])
        rf.append(double_ml(Y, D, X, kind="rf", K=4, seed=seed + b)[0])
    naive, lin, rf = map(np.array, (naive, lin, rf))
    means = [naive.mean(), lin.mean(), rf.mean()]
    sds = [naive.std(), lin.std(), rf.std()]
    biases = [m - THETA_TRUE for m in means]
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["朴素 OLS\n(有偏)", "线性残差化\nDouble ML (仍有偏)", "Double ML\n随机森林 (无偏)"]
    cols = [C["neg"], C["purple"], C["pos"]]
    bars = ax.bar(labels, means, yerr=sds, capsize=5, color=cols, alpha=0.9,
                  width=0.6)
    ax.axhline(THETA_TRUE, color="black", lw=2, ls="--", label=f"真值 θ={THETA_TRUE}")
    for bar, m, sd in zip(bars, means, sds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + sd + 0.02,
                f"{m:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("θ 估计值（蒙特卡洛均值 ± 1 SD）")
    ax.set_ylim(0.8, 2.3)
    ax.set_title("蒙特卡洛 100 次：Double ML(随机森林) 把偏差压回真值",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(IMG, "dml_bias_comparison.png"))
    plt.close(fig)
    return dict(naive=means[0], lin=means[1], rf=means[2],
                naive_sd=sds[0], lin_sd=sds[1], rf_sd=sds[2],
                naive_bias=biases[0], lin_bias=biases[1], rf_bias=biases[2])


if __name__ == "__main__":
    make_cover()
    b = make_residualization()
    res = make_bias_comparison(B=100)
    print("=" * 50)
    print(f"真实 θ        = {THETA_TRUE}")
    print(f"残差化回归斜率 = {b:.4f}")
    print(f"朴素 OLS 均值  = {res['naive']:.4f}  (偏差 {res['naive_bias']:+.4f}, SD {res['naive_sd']:.4f})")
    print(f"线性DML 均值   = {res['lin']:.4f}  (偏差 {res['lin_bias']:+.4f}, SD {res['lin_sd']:.4f})")
    print(f"RF-DML 均值    = {res['rf']:.4f}  (偏差 {res['rf_bias']:+.4f}, SD {res['rf_sd']:.4f})")
    print("=" * 50)
