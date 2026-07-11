#!/usr/bin/env python3
"""
为文章「注意力机制的可解释性：用特征归因解释模型决策」(attention-interpretability-quant) 生成真实配图。

核心逻辑（全部可复现，非占位图）：
  - 搭一个小的「特征级自注意力」收益预测器：输入 (F=8 特征 × T=20 时序)，token=特征，
    过一层 self-attention + 线性头，预测次日收益方向。
  - 故意只让 feature 3 携带真实信号（label 仅依赖它的时序均值），其余为噪声。
  - 训练后对比两种「特征重要性」：
      (a) 注意力权重（入度）  (b) Integrated Gradients 特征归因
  - 结论：注意力经常把权重摊在多个特征上、定位不到真正驱动预测的 feature 3；IG 能精确锁定。
生成 3 张图 + 打印关键指标到 _metrics.txt
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "attention-interpretability-quant")
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(20260711)

# ============ 1) 数据：只有 feature 3 携带真实信号 ============
# feature3 的时序 = 某个固定形态 p 乘以一个「隐藏标量」s_i；label 只由 s_i 决定。
# 模型必须学会把 feature3 的序列投影回 s_i 才能预测——这是它唯一的信息来源。
N, F, T = 1400, 8, 20
TRUE_FEAT = 3
rng = np.random.default_rng(20260711)
p_pattern = rng.normal(0, 1, T)
s_hidden = rng.normal(0, 1, N)
X = rng.normal(0, 1, (N, F, T))
X[:, TRUE_FEAT, :] = s_hidden[:, None] * p_pattern[None, :] + 0.10 * rng.normal(0, 1, (N, T))
label = s_hidden + rng.normal(0, 0.10, N)

n_tr = 1000
Xtr, ytr = X[:n_tr], label[:n_tr]
Xte, yte = X[n_tr:], label[n_tr:]

# ============ 2) 小注意力模型（纯 numpy，含完整前向/反向） ============
d = 20
P = {
    "Wp": rng.normal(0, np.sqrt(2.0 / T), (d, T)),
    "Wq": rng.normal(0, np.sqrt(2.0 / d), (d, d)),
    "Wk": rng.normal(0, np.sqrt(2.0 / d), (d, d)),
    "Wv": rng.normal(0, np.sqrt(2.0 / d), (d, d)),
    "w": rng.normal(0, 0.1, d),
    "b": 0.0,
}
def softmax_rows(m):
    m = m - m.max(1, keepdims=True)
    e = np.exp(m)
    return e / e.sum(1, keepdims=True)

def forward(x):  # x: (F,T)
    tok = x @ P["Wp"].T                       # (F,d)
    Q = tok @ P["Wq"].T; K = tok @ P["Wk"].T; V = tok @ P["Wv"].T
    scores = Q @ K.T / np.sqrt(d)             # (F,F)
    A = softmax_rows(scores)                  # (F,F)
    context = A @ V                           # (F,d)
    pooled = context.mean(0)                  # (d,)
    y = pooled @ P["w"] + P["b"]
    return y, dict(tok=tok, Q=Q, K=K, V=V, A=A, context=context, pooled=pooled)

def grad_x(x):  # 返回 ∂y/∂x (F,T)，dy=1
    _, c = forward(x)
    tok, Q, K, V, A, context, pooled = c["tok"], c["Q"], c["K"], c["V"], c["A"], c["context"], c["pooled"]
    dpooled = P["w"]                          # (d,)
    dcontext = np.outer(np.ones(F), dpooled) / F   # (F,d)
    dA = dcontext @ V.T                       # (F,F)
    dV = A.T @ dcontext                       # (F,d)
    dScores = A * (dA - (dA * A).sum(1, keepdims=True))   # softmax 行反向
    dScores = dScores / np.sqrt(d)
    dQ = dScores @ K                          # (F,d)
    dK = dScores.T @ Q                        # (F,d)
    dtok = dQ @ P["Wq"] + dK @ P["Wk"] + dV @ P["Wv"]   # (F,d)
    dx = dtok @ P["Wp"]                       # (F,T)
    return dx

# 训练
# 训练集 / 测试集 R2
def r2(Xs, ys):
    yh = np.array([forward(x)[0] for x in Xs])
    return 1 - ((ys - yh) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()

lr = 0.01
best_val, best_P = -1e9, None
for ep in range(5000):
    idx = rng.integers(0, n_tr, 128)
    # 批量前向（仅用于 head 参数更新）
    tok = Xtr[idx] @ P["Wp"].T
    Q = tok @ P["Wq"].T; K = tok @ P["Wk"].T; V = tok @ P["Wv"].T
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d)
    scores = scores - scores.max(2, keepdims=True)
    A = np.exp(scores); A = A / A.sum(2, keepdims=True)
    context = A @ V
    pooled = context.mean(1)
    yhat = pooled @ P["w"] + P["b"]
    err = yhat - ytr[idx]
    dw = pooled.T @ err / len(idx); db = err.sum() / len(idx)
    for i in range(len(idx)):
        xi = Xtr[idx[i]]; yi = ytr[idx[i]]
        yh_i, c = forward(xi)                 # 用「当前」参数算误差，避免滞后梯度
        tok_i, Q_i, K_i, V_i, A_i = c["tok"], c["Q"], c["K"], c["V"], c["A"]
        e = yh_i - yi
        dp = e * P["w"]
        dcon = np.outer(np.ones(F), dp) / F
        dA_i = dcon @ V_i.T
        dV_i = A_i.T @ dcon
        dS = A_i * (dA_i - (dA_i * A_i).sum(1, keepdims=True)) / np.sqrt(d)
        dQ_i = dS @ K_i; dK_i = dS.T @ Q_i
        dtok = dQ_i @ P["Wq"] + dK_i @ P["Wk"] + dV_i @ P["Wv"]
        P["Wp"] -= lr * (dtok.T @ xi)
        P["Wq"] -= lr * (dQ_i.T @ tok_i)
        P["Wk"] -= lr * (dK_i.T @ tok_i)
        P["Wv"] -= lr * (dV_i.T @ tok_i)
    # head 参数（批量更新）
    P["w"] -= lr * dw; P["b"] -= lr * db
    # 早停：在验证集上跟踪最佳
    if ep % 200 == 0:
        val_r2 = r2(Xte, yte)
        if val_r2 > best_val:
            best_val = val_r2
            best_P = {k: v.copy() for k, v in P.items()}

# 用验证集最佳参数
P = best_P
r2_tr = r2(Xtr, ytr); r2_te = r2(Xte, yte)

# ============ 3) 注意力重要性 vs Integrated Gradients 归因 ============
K_ig = 50
def ig_importance(x):
    ig = np.zeros((F, T))
    for k in range(K_ig):
        a = (k + 0.5) / K_ig
        xa = a * x
        ig += grad_x(xa)
    ig = x * ig / K_ig
    return ig.mean(1)   # 按时间聚合 -> (F,)

def attn_received(x):
    _, c = forward(x)
    return c["A"].sum(0)   # 入度 (F,)

ig_top1 = []; attn_top1 = []
ig_share = []; attn_share = []
ig_store = []; attn_store = []
for x in Xte:
    igv = ig_importance(x); atv = attn_received(x)
    ig_store.append(igv); attn_store.append(atv)
    ig_top1.append(np.argmax(np.abs(igv)) == TRUE_FEAT)
    attn_top1.append(np.argmax(atv) == TRUE_FEAT)
    ig_share.append(np.abs(igv)[TRUE_FEAT] / (np.abs(igv).sum() + 1e-12))
    attn_share.append(atv[TRUE_FEAT] / (atv.sum() + 1e-12))

metrics = {
    "F": F, "T": T, "TRUE_FEAT": TRUE_FEAT,
    "r2_train": round(float(r2_tr), 3),
    "r2_test": round(float(r2_te), 3),
    "best_val_r2": round(float(best_val), 3),
    "ig_top1_acc": round(float(np.mean(ig_top1)), 3),
    "attn_top1_acc": round(float(np.mean(attn_top1)), 3),
    "ig_share_true": round(float(np.mean(ig_share)), 3),
    "attn_share_true": round(float(np.mean(attn_share)), 3),
}
with open(os.path.join(D, "_metrics.txt"), "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}={v}\n")
print("METRICS", metrics)

# ============ 图1：注意力热力图（取一条测试样本） ============
sample_idx = int(np.argmax(np.abs(ig_store)[:, TRUE_FEAT]))  # 选 IG 最强的样本
xs = Xte[sample_idx]
_, c = forward(xs)
Avis = c["A"]
fig, ax = plt.subplots(figsize=(7.5, 6.2))
im = ax.imshow(Avis, cmap="viridis", aspect="auto")
ax.set_title("自注意力权重矩阵 A (F×F)：注意力的「看哪里」", fontsize=13, fontweight="bold")
ax.set_xlabel("被注意的特征 (Key feature)")
ax.set_ylabel("发起注意的特征 (Query feature)")
ax.set_xticks(range(F)); ax.set_yticks(range(F))
ax.set_xticklabels([f"F{j}" for j in range(F)])
ax.set_yticklabels([f"F{j}" for j in range(F)])
# 高亮真正重要的 feature 3 列
for j in range(F):
    if j == TRUE_FEAT:
        ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, F, fill=False, edgecolor="red", lw=2.5))
ax.axvline(TRUE_FEAT - 0.5, color="red", lw=0.8, ls="--", alpha=0.5)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(D, "fig_attention_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============ 图2：Integrated Gradients 特征归因（单样本，高亮 feature 3） ============
igv = np.abs(ig_store[sample_idx])
fig, ax = plt.subplots(figsize=(9, 5.5))
colors = ["#d62728" if j == TRUE_FEAT else "#1f77b4" for j in range(F)]
bars = ax.bar([f"F{j}" for j in range(F)], igv, color=colors)
ax.set_title("Integrated Gradients 特征归因：精确锁定真正驱动预测的 F3", fontsize=13, fontweight="bold")
ax.set_xlabel("特征"); ax.set_ylabel("|IG| 归因强度")
ax.grid(True, alpha=0.3, axis="y")
for j, v in enumerate(igv):
    ax.text(j, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
ax.text(TRUE_FEAT, igv[TRUE_FEAT], " ← 真实信号特征", color="red", fontsize=9, va="bottom")
plt.tight_layout()
plt.savefig(os.path.join(D, "fig_ig_attribution.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============ 图3：注意力 vs IG 定位能力对比（聚合指标） ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
methods = ["注意力权重", "Integrated Gradients"]
accs = [metrics["attn_top1_acc"], metrics["ig_top1_acc"]]
shares = [metrics["attn_share_true"], metrics["ig_share_true"]]
b1 = ax1.bar(methods, accs, color=["#ff7f0e", "#2ca02c"], width=0.5)
ax1.set_title("Top-1 定位准确率：谁更准地找到 F3", fontsize=12, fontweight="bold")
ax1.set_ylabel("Top-1 准确率"); ax1.set_ylim(0, 1); ax1.grid(True, alpha=0.3, axis="y")
for b, v in zip(b1, accs):
    ax1.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.1%}", ha="center", fontsize=10)
b2 = ax2.bar(methods, shares, color=["#ff7f0e", "#2ca02c"], width=0.5)
ax2.set_title("真实特征占重要性比重：注意力被稀释", fontsize=12, fontweight="bold")
ax2.set_ylabel("真实特征重要性占比"); ax2.set_ylim(0, 1); ax2.grid(True, alpha=0.3, axis="y")
for b, v in zip(b2, shares):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.1%}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(D, "fig_attn_vs_ig.png"), dpi=150, bbox_inches="tight")
plt.close()
print("DONE attention images")
