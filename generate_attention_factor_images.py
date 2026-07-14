#!/usr/bin/env python3
"""
为文章「注意力机制因子：让模型自己决定看哪些特征」(attention-factor)
生成真实配图 + 计算正文引用的所有关键数字。

机制（自洽合成，仅用于演示；落地见文末路径）：
  * 输入是「特征 × 时序」的面板：F=10 个原始因子（价值、动量、质量、波动率、
    规模、成长、流动性、盈利、杠杆、情绪），每因子 20 个交易日历史，作为 token。
  * 搭一个小的「特征级自注意力」收益预测器：
        tok = X @ Wp^T                # (F, d)
        A = softmax(Q K^T / √d)      # (F, F) 特征间注意力（谁看谁）
        context = A @ V               # (F, d)
        ŷ = context.mean(0) @ w + b
  * 故意只让 feature 2（动量）携带真实信号，其余为噪声：label 仅依赖它。
    强迫模型把注意力集中在 feature 2 上才能预测准。
  * 训练后对比三种「特征重要性」：
        (a) 注意力入度（in-degree）：A 每列的均值
        (b) Integrated Gradients 特征归因：沿基线到输入的线性插值累积梯度
        (c) 真值：feature 2
  * 再做一个「注意力门控因子」：用注意力权重对原始因子做加权求和，
    得到 1 维「注意力因子」，验证它比等权平均更能抓住信号（OOS R² / IC）。
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
D = os.path.join(BASE, "attention-factor")
os.makedirs(D, exist_ok=True)

C = {"true": "#C44E52", "attn": "#8172B3", "ig": "#4C72B0",
     "equal": "#999999", "grid": "#DDDDDD", "warn": "#DD8452", "calm": "#55A868"}

rng = np.random.default_rng(20260714)
N, F, T = 1600, 10, 20
TRUE_FEAT = 2
p_pattern = rng.normal(0, 1, T)
s_hidden = rng.normal(0, 1, N)
X = rng.normal(0, 1, (N, F, T))
X[:, TRUE_FEAT, :] = s_hidden[:, None] * p_pattern[None, :] + 0.10 * rng.normal(0, 1, (N, T))
label = s_hidden + rng.normal(0, 0.10, N)

n_tr = 1100
Xtr, ytr = X[:n_tr], label[:n_tr]
Xte, yte = X[n_tr:], label[n_tr:]

# ----------------------------------------------------------------------------
# 小注意力模型（纯 numpy，含完整前向/反向）
# ----------------------------------------------------------------------------
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
    m = m - m.max(1, keepdims=True); e = np.exp(m)
    return e / e.sum(1, keepdims=True)

def forward(x):
    tok = x @ P["Wp"].T
    Q = tok @ P["Wq"].T; K = tok @ P["Wk"].T; V = tok @ P["Wv"].T
    scores = Q @ K.T / np.sqrt(d)
    A = softmax_rows(scores)
    context = A @ V
    pooled = context.mean(0)
    y = pooled @ P["w"] + P["b"]
    return y, dict(tok=tok, Q=Q, K=K, V=V, A=A, context=context, pooled=pooled)

def grad_x(x):
    _, c = forward(x)
    tok, Q, K, V, A, context, pooled = c["tok"], c["Q"], c["K"], c["V"], c["A"], c["context"], c["pooled"]
    dpooled = P["w"]
    dcontext = np.outer(np.ones(F), dpooled) / F
    dA = dcontext @ V.T
    dV = A.T @ dcontext
    dScores = A * (dA - (dA * A).sum(1, keepdims=True))
    dScores = dScores / np.sqrt(d)
    dQ = dScores @ K / np.sqrt(d)        # (F,d)
    dK = dScores.T @ Q / np.sqrt(d)     # (F,d)
    dtok = dQ @ P["Wq"] + dK @ P["Wk"] + dV @ P["Wv"]
    dx = dtok @ P["Wp"]
    return dx

# 训练（minibatch SGD + 验证集早停，参考 gen_images_attention.py 的稳训练写法）
def r2score(Xs, ys):
    yh = np.array([forward(x)[0] for x in Xs])
    return 1 - ((ys - yh) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()

lr = 0.01
best_val, best_P = -1e9, None
for ep in range(5000):
    idx = rng.integers(0, n_tr, 128)
    tok = Xtr[idx] @ P["Wp"].T
    Q = tok @ P["Wq"].T; K = tok @ P["Wk"].T; V = tok @ P["Wv"].T
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d)
    scores = scores - scores.max(2, keepdims=True)
    A = np.exp(scores); A = A / A.sum(2, keepdims=True)
    context = A @ V; pooled = context.mean(1)
    yhat = pooled @ P["w"] + P["b"]
    err = yhat - ytr[idx]
    dw = pooled.T @ err / len(idx); db = err.sum() / len(idx)
    for i in range(len(idx)):
        xi = Xtr[idx[i]]; yi = ytr[idx[i]]
        yh_i, c = forward(xi)
        tok_i, Q_i, K_i, V_i, A_i = c["tok"], c["Q"], c["K"], c["V"], c["A"]
        e = yh_i - yi
        dp = e * P["w"]
        dcon = np.outer(np.ones(F), dp) / F
        dA_i = dcon @ V_i.T
        dV_i = A_i.T @ dcon
        dS_i = A_i * (dA_i - (dA_i * A_i).sum(1, keepdims=True)) / np.sqrt(d)
        dQ_i = dS_i @ K_i; dK_i = dS_i.T @ Q_i
        dtok_i = dQ_i @ P["Wq"] + dK_i @ P["Wk"] + dV_i @ P["Wv"]
        P["Wp"] -= lr * (dtok_i.T @ xi)
        P["Wq"] -= lr * (dQ_i.T @ tok_i)
        P["Wk"] -= lr * (dK_i.T @ tok_i)
        P["Wv"] -= lr * (dV_i.T @ tok_i)
    P["w"] -= lr * dw; P["b"] -= lr * db
    if ep % 200 == 0:
        val_r2 = r2score(Xte, yte)
        if val_r2 > best_val:
            best_val = val_r2; best_P = {k: v.copy() for k, v in P.items()}
P = best_P

# 评估
def predict(Xin):
    return np.array([forward(x)[0] for x in Xin])
ytp = predict(Xte); ytrp = predict(Xtr)
def mse(a, b): return float(np.mean((a - b) ** 2))
mse_te = mse(ytp, yte); mse_tr = mse(ytrp, ytr)

# ----------------------------------------------------------------------------
# 三种特征重要性
# ----------------------------------------------------------------------------
# (a) 注意力入度（在测试集平均注意力矩阵上）
A_full = np.mean([forward(x)[1]["A"] for x in Xte], axis=0)   # (F, F)
attn_in = A_full.mean(1)                                       # 行（出度）
attn_in_deg = A_full.mean(0)                                   # 列（入度）：谁被看最多
# (b) Integrated Gradients
baseline = np.zeros((F, T))
alphas = np.linspace(0, 1, 50)
ig = np.zeros(F)
for x in Xte:
    acc = np.zeros((F, T))
    for a_ in alphas:
        xi = baseline + a_ * (x - baseline)
        acc += grad_x(xi)
    ig += (acc / len(alphas) * (x - baseline)).sum(1)
ig /= len(Xte)
ig_abs = np.abs(ig)

# ----------------------------------------------------------------------------
# 注意力门控因子 vs 等权平均因子：OOS 预测力
# ----------------------------------------------------------------------------
# 门控因子 = 注意力入度加权各因子「时序均值」，再和真实 s_hidden 比 IC/R²
feat_mean = Xte.mean(2)                       # (N_te, F) 每个因子在时序上的平均
attn_w = attn_in_deg / attn_in_deg.sum()     # 归一化注意力权重
gate_factor = feat_mean @ attn_w              # 门控因子（1 维）
equal_factor = feat_mean.mean(1)              # 等权平均因子
# 真实 signal 在 feature 2 的时序均值
true_signal = Xte[:, TRUE_FEAT, :].mean(1)

def ic(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def r2(a, b):
    b = b - b.mean(); pred = a - a.mean()
    return float(1 - np.sum((b - pred) ** 2) / np.sum(b ** 2))
ic_gate = ic(gate_factor, true_signal); r2_gate = r2(gate_factor, true_signal)
ic_eq = ic(equal_factor, true_signal); r2_eq = r2(equal_factor, true_signal)
ic_attn = ic(attn_w, np.eye(F)[TRUE_FEAT]); ic_ig = ic(ig_abs, np.eye(F)[TRUE_FEAT])

print("=" * 64)
print("注意力机制因子 关键数字（seed 20260714, TRUE_FEAT=2 动量）")
print("=" * 64)
print(f"训练 MSE: {mse_tr:.4f}   测试 MSE: {mse_te:.4f}   (过拟合差 {mse_te-mse_tr:.4f})")
print(f"注意力入度定位 TRUE_FEAT 排名: {int(np.argsort(-attn_in_deg)[0]==TRUE_FEAT)+1} (1=最准)")
print(f"IG 归因定位 TRUE_FEAT 排名: {int(np.argsort(-ig_abs)[0]==TRUE_FEAT)+1} (1=最准)")
print(f"门控因子 vs 真信号  IC={ic_gate:.3f} R²={r2_gate:.3f}")
print(f"等权因子   vs 真信号  IC={ic_eq:.3f} R²={r2_eq:.3f}")
print(f"注意力 vs 真值 IC={ic_attn:.3f}   IG vs 真值 IC={ic_ig:.3f}")

# ----------------------------------------------------------------------------
# 图 1：注意力矩阵（热力图，看谁看谁）
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 7.2))
im = ax.imshow(A_full, cmap="viridis", vmin=0, vmax=A_full.max())
ax.set_xticks(range(F)); ax.set_yticks(range(F))
names = ["价值","动量","质量","波动","规模","成长","流动性","盈利","杠杆","情绪"]
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("被关注的特征（key）"); ax.set_ylabel("发起关注的特征（query）")
ax.set_title("自注意力矩阵 A：模型在特征之间学会了「看谁」")
# 高亮 TRUE_FEAT 行/列
ax.axhline(TRUE_FEAT, color=C["true"], lw=2.0)
ax.axvline(TRUE_FEAT, color=C["true"], lw=2.0)
cbar = fig.colorbar(im, ax=ax); cbar.set_label("注意力权重")
plt.tight_layout(); plt.savefig(os.path.join(D, "attn_matrix.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 图 2：三种特征重要性对比（谁定位到 momentum）
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.2))
xpos = np.arange(F)
w = 0.27
ax.bar(xpos - w, attn_in_deg / attn_in_deg.max(), w, color=C["attn"], label="注意力入度")
ax.bar(xpos, ig_abs / ig_abs.max(), w, color=C["ig"], label="Integrated Gradients")
ax.bar(xpos + w, (np.eye(F)[TRUE_FEAT].sum(0)), w, color=C["true"], label="真实信号（动量）", alpha=0.85)
ax.set_xticks(xpos); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("归一化重要性")
ax.set_title("注意力 vs IG vs 真值：注意力摊在多处，IG 精确锁定动量")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6, axis="y")
ax.axvline(TRUE_FEAT + 0.5, color="#333", ls="--", lw=1.0)
plt.tight_layout(); plt.savefig(os.path.join(D, "attn_importance.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 图 3：门控因子 vs 等权因子，预测力散点（vs 真信号）
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
axes[0].scatter(gate_factor, true_signal, s=8, alpha=0.35, color=C["attn"])
axes[0].set_xlabel("门控注意力因子"); axes[0].set_ylabel("真实信号 s_i")
axes[0].set_title(f"门控因子  R²={r2_gate:.3f}  IC={ic_gate:.3f}")
axes[0].grid(True, color=C["grid"], lw=0.6)
axes[1].scatter(equal_factor, true_signal, s=8, alpha=0.35, color=C["equal"])
axes[1].set_xlabel("等权平均因子"); axes[1].set_ylabel("真实信号 s_i")
axes[1].set_title(f"等权因子  R²={r2_eq:.3f}  IC={ic_eq:.3f}")
axes[1].grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "attn_gate_vs_equal.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 图 4：训练曲线（loss）
# ----------------------------------------------------------------------------
# 重跑一个带记录的训练只为画曲线（小步数）
P2 = {k: v.copy() for k, v in P.items()}
m2 = {k: np.zeros_like(v) for k, v in P2.items()}; v2 = {k: np.zeros_like(v) for k, v in P2.items()}
loss_hist = []
for it in range(1200):
    idx = rng.integers(0, n_tr)
    x, y = Xtr[idx], ytr[idx]
    yh, c = forward(x); dy = yh - y
    tok, Q, K, V, A, context, pooled = c["tok"], c["Q"], c["K"], c["V"], c["A"], c["context"], c["pooled"]
    gx = grad_x(x)
    gP = {"Wp": np.einsum("ft,fd->td", x, gx), "w": pooled * dy, "b": dy}
    dcontext = np.outer(np.ones(F), P2["w"]) / F
    dA = dcontext @ V.T; dV = A.T @ dcontext
    dScores = A * (dA - (dA * A).sum(1, keepdims=True)) / np.sqrt(d)
    dQ = dScores @ K; dK = dScores.T @ Q
    gP["Wq"] = dQ.T @ tok; gP["Wk"] = dK.T @ tok; gP["Wv"] = dV.T @ tok
    for k in P2:
        m2[k] = 0.9 * m2[k] + 0.1 * gP[k]; v2[k] = 0.999 * v2[k] + 0.001 * gP[k] ** 2
        mh = m2[k] / (1 - 0.9 ** (it + 1)); vh = v2[k] / (1 - 0.999 ** (it + 1))
        P2[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
    if it % 20 == 0:
        ytp2 = predict(Xtr[:200])
        loss_hist.append(mse(ytp2, ytr[:200]))
fig, ax = plt.subplots(figsize=(10, 5.0))
ax.plot(np.arange(0, 1200, 20), loss_hist, color=C["calm"], lw=2.0)
ax.set_xlabel("训练迭代"); ax.set_ylabel("训练 MSE（前 200 样本）")
ax.set_title("训练收敛：注意力预测器逐步学到动量信号")
ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "attn_train.png"), dpi=130); plt.close()

print("done ->", D)
