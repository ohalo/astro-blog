#!/usr/bin/env python3
"""可解释 boosting 因子归因：从零实现梯度提升 + 精确 TreeSHAP，生成 4 张真实图。

- 手写 CART 回归树（MSE 分裂，叶节点存经验分裂概率 p_left）
- 手写梯度提升（GBM）预测股票下一期收益
- 手写精确 TreeSHAP：对每一棵树用「子集枚举 + 缺失特征边际化」计算
  每个特征的 Shapley 贡献（interventional 期望），再按加性求和到森林
- 生成：SHAP 摘要 beeswarm / 单样本瀑布 / 特征重要性 / 按 SHAP 打分十分位 IC
"""
import os, math, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 注册中文字体（macOS 自带 Hiragino Sans GB）
_CJK = "/System/Library/Fonts/Hiragino Sans GB.ttc"
if os.path.exists(_CJK):
    font_manager.fontManager.addfont(_CJK)
    _name = font_manager.FontProperties(fname=_CJK).get_name()
    plt.rcParams["font.family"] = _name
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/interpretable-boosting-shap"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2026)

# ====================== 1. 合成横截面因子数据 ======================
FEATURES = ["价值", "动量", "规模", "质量", "成长", "波动率", "流动性", "Beta"]
M = len(FEATURES)
N_TR, N_TE = 1400, 500
N = N_TR + N_TE
coef = np.array([0.55, 0.40, -0.30, 0.45, 0.25, -0.20, 0.15, -0.10])  # 真实因子暴露
X = rng.standard_normal((N, M))
signal = X @ coef
y = signal + rng.standard_normal(N) * 0.6          # 含噪声的真实收益
Xtr, ytr = X[:N_TR], y[:N_TR]
Xte, yte = X[N_TR:], y[N_TR:]                       # 测试集用于 SHAP 与十分位

# ====================== 2. 从零实现 CART 回归树 ======================
class Node:
    __slots__ = ("feature", "thr", "left", "right", "value", "is_leaf", "p_left")
    def __init__(self):
        self.feature = None; self.thr = None
        self.left = self.right = None
        self.value = 0.0; self.is_leaf = False; self.p_left = 0.5

def build_tree(X, y, depth, max_depth, min_samples):
    node = Node()
    n = X.shape[0]
    if depth >= max_depth or n < 2 * min_samples or np.var(y) < 1e-9:
        node.is_leaf = True; node.value = float(np.mean(y)); return node
    best, best_err = None, np.inf
    for f in range(X.shape[1]):
        vals = np.unique(X[:, f])
        if len(vals) < 2:
            continue
        thrs = (vals[:-1] + vals[1:]) / 2.0
        for thr in thrs:
            left = X[:, f] <= thr
            nl, nr = left.sum(), (~left).sum()
            if nl < min_samples or nr < min_samples:
                continue
            err = np.var(y[left]) * nl + np.var(y[~left]) * nr
            if err < best_err:
                best_err = err; best = (f, thr, left)
    if best is None:
        node.is_leaf = True; node.value = float(np.mean(y)); return node
    f, thr, left = best
    node.feature = f; node.thr = thr
    node.p_left = float(left.sum() / n)
    node.left = build_tree(X[left], y[left], depth + 1, max_depth, min_samples)
    node.right = build_tree(X[~left], y[~left], depth + 1, max_depth, min_samples)
    return node

# ====================== 3. 从零实现梯度提升 ======================
def fit_gbm(X, y, n_trees=60, lr=0.1, max_depth=3, min_samples=20):
    F = np.full(X.shape[0], np.mean(y))
    trees = []
    for t in range(n_trees):
        resid = y - F
        tree = build_tree(X, resid, 0, max_depth, min_samples)
        pred = predict_tree(tree, X)
        F = F + lr * pred
        trees.append(tree)
        if t % 20 == 0:
            print(f"  [gbm] tree {t:3d}  train-R2={1-np.var(y-F)/np.var(y):.3f}")
    return trees, F

def predict_tree(tree, X):
    N = X.shape[0]
    def rec(node):
        if node.is_leaf:
            return np.full(N, node.value)
        go_left = X[:, node.feature] <= node.thr
        res = np.empty(N)
        res[go_left] = rec(node.left)[go_left]
        res[~go_left] = rec(node.right)[~go_left]
        return res
    return rec(tree)

def gbm_predict(trees, X, lr=0.1):
    out = np.zeros(X.shape[0])
    for tr in trees:
        out += lr * predict_tree(tr, X)
    return out

print("训练梯度提升因子模型 ...")
trees, Ftr = fit_gbm(Xtr, ytr, n_trees=70, lr=0.1, max_depth=3, min_samples=20)

# ====================== 4. 从零实现精确 TreeSHAP ======================
# 对每棵树：枚举所有固定特征子集 mask（bit j=1 表示特征 j 固定为 x 值），
# 缺失特征按经验分裂概率 p_left 边际化 -> 得到该子集下的期望预测 val[mask]（对所有样本并行）。
# 再用 Shapley 公式汇总：phi_i = Σ_{S⊆M\{i}} w(S)·(val[S∪i] − val[S])，按树加性求和。

def eval_tree_mask(tree, X, fixed_mask):
    N = X.shape[0]
    def rec(node):
        if node.is_leaf:
            return np.full(N, node.value)
        if fixed_mask[node.feature]:
            go_left = X[:, node.feature] <= node.thr
            res = np.empty(N)
            res[go_left] = rec(node.left)[go_left]
            res[~go_left] = rec(node.right)[~go_left]
            return res
        else:
            return node.p_left * rec(node.left) + (1.0 - node.p_left) * rec(node.right)
    return rec(tree)

def shap_values(trees, X, lr=0.1):
    N = X.shape[0]
    # 预计算每个树的 val[mask] (2^M, N)
    n_masks = 1 << M
    per_tree = []
    for tree in trees:
        V = np.empty((n_masks, N))
        for mask in range(n_masks):
            fm = np.array([bool((mask >> j) & 1) for j in range(M)])
            V[mask] = eval_tree_mask(tree, X, fm)
        per_tree.append(V)
    # 汇总到森林
    phi = np.zeros((N, M))
    fact = math.factorial
    for i in range(M):
        others = [j for j in range(M) if j != i]
        for k in range(len(others) + 1):
            for S in itertools.combinations(others, k):
                maskS = 0
                for j in S:
                    maskS |= (1 << j)
                maskSi = maskS | (1 << i)
                w = fact(k) * fact(M - k - 1) / fact(M)
                # lr 缩放（每个树贡献乘 lr）
                for V in per_tree:
                    phi[:, i] += w * lr * (V[maskSi] - V[maskS])
    return phi

print("计算精确 TreeSHAP（子集枚举）...")
phi = shap_values(trees, Xte, lr=0.1)
phi0 = np.mean(gbm_predict(trees, Xte, lr=0.1)) - phi.sum(1).mean()  # 期望基线
print(f"  SHAP 一致性检查：phi0 + Σphi ≈ pred?  max|err|={np.max(np.abs(phi0 + phi.sum(1) - gbm_predict(trees,Xte,lr=0.1))):.2e}")

# ====================== 5. 衍生指标 ======================
mean_abs = np.mean(np.abs(phi), axis=0)
order = np.argsort(-mean_abs)
imp = mean_abs[order]
# 按 SHAP 总打分排十分位，看真实收益单调性
score = phi.sum(1)
edges = np.quantile(score, np.linspace(0, 1, 11))
ranks = np.clip(np.digitize(score, edges[1:-1]), 0, 9)   # 0..9 共 10 组
dec = np.array([yte[ranks == d].mean() for d in range(10)])
print("  特征重要性（均值|SHAP|）:", dict(zip([FEATURES[o] for o in order], np.round(imp,3))))
print("  十分位平均真实收益:", np.round(dec,3))

# ========================= 绘图 =========================
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 110, "savefig.bbox": "tight"})
Xz = (Xte - Xte.mean(0)) / (Xte.std(0) + 1e-9)

# 图1：SHAP 摘要 beeswarm
fig, ax = plt.subplots(figsize=(9.5, 5.5))
idx = np.arange(M)[::-1]
colors = plt.cm.coolwarm(Xz[:, order].T)
for row, f in enumerate(order[::-1]):
    col = f
    yv = np.full(N_TE, row) + rng.standard_normal(N_TE) * 0.08
    ax.scatter(phi[:, col], yv, c=Xz[:, col], cmap="coolwarm", s=14, alpha=0.55,
               vmin=-2.5, vmax=2.5, edgecolors="none")
ax.set_yticks(range(M)); ax.set_yticklabels([FEATURES[o] for o in order][::-1])
ax.set_xlabel("SHAP 值（对预测收益的贡献）")
ax.set_title("SHAP 摘要图：每个因子的边际贡献分布")
fig.colorbar(plt.cm.ScalarMappable(cmap="coolwarm",
            norm=plt.Normalize(-2.5, 2.5)), ax=ax, label="特征标准化取值（红=高）")
fig.tight_layout(); fig.savefig(f"{OUT}/shap_summary.png"); plt.close(fig)

# 图2：单样本瀑布
ex = 37
vals_w = phi[ex].copy()
order_w = np.argsort(vals_w)
labels = [FEATURES[order_w[k]] for k in range(M)]
cum = np.concatenate([[phi0], phi0 + np.cumsum(vals_w[order_w])])
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.barh(range(M), vals_w[order_w], left=cum[:-1],
        color=["#264653" if k == 0 else "#2a9d8f" for k in range(M)])
for k in range(M + 1):
    ax.text(cum[k] + (0.02 if k < M else -0.02), min(k, M - 1), f"{cum[k]:.2f}",
            va="center", ha="left" if k < M else "right", fontsize=8)
ax.axvline(phi0, color="#264653", ls="--", lw=1, alpha=0.7)
ax.text(phi0, M - 0.4, f" 基线 E[f]={phi0:.2f}", color="#264653", fontsize=8, va="bottom")
ax.set_yticks(range(M)); ax.set_yticklabels(labels)
ax.set_xlabel("累积预测收益"); ax.set_title(f"SHAP 瀑布图（个股 #{ex}）：从基线到预测 {cum[-1]:.2f}")
fig.tight_layout(); fig.savefig(f"{OUT}/shap_waterfall.png"); plt.close(fig)

# 图3：特征重要性
fig, ax = plt.subplots(figsize=(9.0, 4.8))
ax.barh(range(M), imp, color="#e76f51")
ax.set_yticks(range(M)); ax.set_yticklabels([FEATURES[o] for o in order])
ax.invert_yaxis(); ax.set_xlabel("平均 |SHAP|（全局重要性）")
ax.set_title("全局特征重要性：均值 |SHAP|")
fig.tight_layout(); fig.savefig(f"{OUT}/feature_importance.png"); plt.close(fig)

# 图4：按 SHAP 打分十分位的平均真实收益
fig, ax = plt.subplots(figsize=(9.0, 4.8))
ax.plot(np.arange(1, 11), dec, "-o", color="#2a9d8f", lw=2, label="十分位平均真实收益")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xticks(range(1, 11)); ax.set_xlabel("按 SHAP 总打分排序的十分位（低→高）")
ax.set_ylabel("平均下一期真实收益"); ax.set_title("单调可分性：SHAP 打分越高，真实收益越高")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/shap_decile_return.png"); plt.close(fig)

print("✅ 已生成 4 张图表到", OUT)
print(os.listdir(OUT))
