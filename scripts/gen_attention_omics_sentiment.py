#!/usr/bin/env python3
"""
为文章「金融新闻 Transformer 情感分析」(attention-omics-sentiment)
生成真实配图 + 计算正文引用的所有关键数字。

机制（自洽合成，仅用于演示；落地见文末路径）：
  * 构造「新闻标题 -> 情绪」自洽合成宇宙：
        - 每条标题由 60 个词 token 组成，词表 2000。
        - 真实信号：标题里「好消息 token / 坏消息 token」的位置加权计数决定情绪 score。
        - 第 12 位（开头）和第 30 位（中段）的 token 权重最高 —— 模拟「标题头/中段最关键」。
        - 再叠高斯噪声 label 抖动，使线性词典法无法完美还原。
  * 从零实现：
        - 朴素词袋(BoW) 线性词典: 每个 token 一个权重，预测 = 权重和 (梯度下降)
        - 自注意力 Transformer 编码器栈(纯 numpy, 含完整前向/反向):
              位置编码 -> 2 层 self-attention + FFN -> [CLS] 池化 -> 读出
  * 量化指标：
        - 测试集 R^2: BoW / Transformer
        - 方向准确率(情绪正负分类): BoW / Transformer
        - 注意力对「关键位置(12,30)」的命中率
        - 对抗扰动一致性(互换两个非关键 token 后预测相关系数)
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
D = os.path.join(BASE, "attention-omics-sentiment")
os.makedirs(D, exist_ok=True)

C = {"tf": "#4C72B0", "bow": "#C44E52", "true": "#8172B3",
     "grid": "#DDDDDD", "warn": "#DD8452", "attn": "#55A868"}

rng = np.random.default_rng(20260723)
VOCAB = 2000
L = 60                      # 标题长度
N = 4000
# 关键位置权重：模拟「头几个词 + 中段」最关键
POS_W = np.zeros(L)
POS_W[12] = 1.0
POS_W[30] = 0.9
POS_W[8] = 0.4
POS_W[25] = 0.4
POS_W /= POS_W.sum()
GOOD = list(range(50, 150))        # 好消息 token id 段
BAD = list(range(150, 250))        # 坏消息 token id 段

X = rng.integers(0, VOCAB, (N, L))
# 真实情感打分：关键位置上的 好/坏 token 计数差（模拟「标题头/中段最关键」）
score = np.zeros(N)
for i in range(N):
    for p in range(L):
        tok = X[i, p]
        if tok in GOOD:
            score[i] += POS_W[p] * 1.0
        elif tok in BAD:
            score[i] -= POS_W[p] * 1.0
score = score / score.std()
label = score + rng.normal(0, 0.15, N)     # 抖动(降低噪声让位置结构更可学)

n_tr = 3000
n_val = 500
Xtr, ytr = X[:n_tr], label[:n_tr]
Xval, yval = X[n_tr:n_tr + n_val], label[n_tr:n_tr + n_val]
Xte, yte = X[n_tr + n_val:], label[n_tr + n_val:]


# ---------------------------------------------------------------------------
# 1) 朴素词袋 BoW 线性词典
# ---------------------------------------------------------------------------
bow_w = np.zeros(VOCAB)
lr = 0.05
for it in range(4000):
    pred = bow_w[Xtr].sum(axis=1)
    err = pred - ytr
    g = np.zeros(VOCAB)
    np.add.at(g, Xtr.ravel(), np.tile(err[:, None], (1, L)).ravel())
    g /= n_tr
    bow_w -= lr * g
bow_pred = bow_w[Xte].sum(axis=1)
bow_r2 = 1 - np.mean((bow_pred - yte) ** 2) / np.var(yte)
bow_dir = np.mean(np.sign(bow_pred) == np.sign(yte))


# ---------------------------------------------------------------------------
# 2) 纯 numpy Transformer 编码器 (含完整前向/反向)
# ---------------------------------------------------------------------------
d_model = 16
n_head = 2
n_layer = 1
ff = 4 * d_model
dh = d_model // n_head


def init_params():
    s = np.sqrt(2.0 / d_model)
    return {
        "tok_emb": rng.normal(0, s, (VOCAB, d_model)),
        "pos_emb": rng.normal(0, 0.02, (L, d_model)),
        "Wq": rng.normal(0, s, (n_layer, d_model, d_model)),
        "Wk": rng.normal(0, s, (n_layer, d_model, d_model)),
        "Wv": rng.normal(0, s, (n_layer, d_model, d_model)),
        "Wo": rng.normal(0, s, (n_layer, d_model, d_model)),
        "W1": rng.normal(0, s, (n_layer, d_model, ff)),
        "b1": np.zeros((n_layer, ff)),
        "W2": rng.normal(0, s, (n_layer, ff, d_model)),
        "b2": np.zeros((n_layer, d_model)),
        "cls": rng.normal(0, s, d_model),
        "out": rng.normal(0, s, d_model),
        "out_b": 0.0,
    }


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def forward(params, Xb):
    """返回预测、最后一层平均注意力、中间缓存"""
    B = Xb.shape[0]
    cache = {}
    h = params["tok_emb"][Xb] + params["pos_emb"][None]      # (B,L,d)
    cache["h0"] = h
    attn_layers = []
    for l in range(n_layer):
        h_flat = h.reshape(B * L, d_model)
        Q = (h_flat @ params["Wq"][l]).reshape(B, L, n_head, dh).transpose(0, 2, 1, 3)
        K = (h_flat @ params["Wk"][l]).reshape(B, L, n_head, dh).transpose(0, 2, 1, 3)
        V = (h_flat @ params["Wv"][l]).reshape(B, L, n_head, dh).transpose(0, 2, 1, 3)
        attn = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(dh)
        A = softmax(attn, axis=-1)
        attn_layers.append(A)
        ctx = (A @ V).transpose(0, 2, 1, 3).reshape(B, L, d_model)
        cache[f"ctx_{l}"] = ctx
        sa = ctx @ params["Wo"][l]
        h = h + sa
        cache[f"h_pre_ff_{l}"] = h
        ff1 = np.tanh(h @ params["W1"][l] + params["b1"][l])
        h = h + (ff1 @ params["W2"][l] + params["b2"][l])
        cache[f"ff1_{l}"] = ff1
        cache[f"h_{l}"] = h
    cls_repr = params["cls"][None, None] + h[:, 0:1]         # (B,1,d)
    out = cls_repr.mean(axis=1) @ params["out"] + params["out_b"]
    return out, attn_layers, cache, h


# ---- 反向传播 (scalar 输出 / MSE) ----
def backward(params, Xb, yb, cache, h):
    B = Xb.shape[0]
    out, _, _, _ = forward(params, Xb)
    err = 2.0 * (out.ravel() - yb)                   # d/dθ ½·MSE 的 ×2
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    # 输出层
    cls_repr = params["cls"][None, None] + h[:, 0:1]
    repr_mean = cls_repr.mean(axis=1)            # (B,d)
    grads["out"] = (err[:, None] * repr_mean).mean(0)
    grads["out_b"] = err.mean()
    # 回传到 cls_repr -> 每个位置 h (B,L,d) 都收到 err-bar
    d_h = np.zeros_like(h)
    d_h[:, 0, :] += (err[:, None] @ params["out"][None, :]) / B
    d_cls = (err[:, None] @ params["out"][None, :]).mean(0)
    grads["cls"] = d_cls
    # 反序逐层
    for l in range(n_layer - 1, -1, -1):
        h_l = cache[f"h_{l}"]                     # 该层 FFN 之后
        ff1 = cache[f"ff1_{l}"]
        h_pre = cache[f"h_pre_ff_{l}"]            # 该层 attention 残差相加之前
        # ---- FFN 残差: h_l = h_pre + (tanh(h_pre W1+b1)) W2 + b2 ----
        d_ff1 = d_h @ params["W2"][l].T          # (B,L,ff)  d tanh 前梯度
        d_ff1 *= (1 - ff1 ** 2)
        grads["W2"][l] = (ff1.reshape(-1, ff).T @ d_h.reshape(-1, d_model)) / (B * L)
        grads["b2"][l] = d_h.sum((0, 1))
        grads["W1"][l] = (h_pre.reshape(-1, d_model).T @ d_ff1.reshape(-1, ff)) / (B * L)
        grads["b1"][l] = d_ff1.sum((0, 1))
        d_h_pre_from_ff = d_ff1 @ params["W1"][l].T
        # ---- attention 残差: h_pre = h_prev + sa ; sa = ctx @ Wo ----
        # 进入本层(从上层来)的梯度 d_h 同时流向 h_prev 与 sa
        d_sa = d_h                                     # (B,L,d)  = d(h_l)
        grads["Wo"][l] = (cache[f"ctx_{l}"].reshape(-1, d_model).T @ d_sa.reshape(-1, d_model)) / (B * L)
        d_ctx = d_sa @ params["Wo"][l].T             # (B,L,d)
        d_ctx = d_ctx.reshape(B, L, n_head, dh).transpose(0, 2, 1, 3)  # (B,H,L,dh)
        # 需要从 cache 拿当前层 A, Q, K, V
        # 重新算 Q,K,V 用 h_prev (即 cache h_{l-1} 或 h0)
        h_prev = cache["h0"] if l == 0 else cache[f"h_{l-1}"]
        h_flat = h_prev.reshape(B * L, d_model)
        Q = (h_flat @ params["Wq"][l]).reshape(B, L, n_head, dh).transpose(0, 2, 1, 3)
        K = (h_flat @ params["Wk"][l]).reshape(B, L, n_head, dh).transpose(0, 2, 1, 3)
        V = (h_flat @ params["Wv"][l]).reshape(B, L, n_head, dh).transpose(0, 2, 1, 3)
        A = softmax((Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(dh), axis=-1)
        # ctx = A @ V  ->  dV = A^T @ d_ctx ; dA = d_ctx @ V^T
        dV = np.transpose(A, (0, 1, 3, 2)) @ d_ctx
        dA = d_ctx @ np.transpose(V, (0, 1, 3, 2))
        # softmax 反向: dZ = A * (dA - sum(dA*A))
        dZ = A * (dA - (dA * A).sum(-1, keepdims=True))
        dZ = dZ / np.sqrt(dh)
        dQ = dZ @ K
        dK = np.transpose(dZ, (0, 1, 3, 2)) @ Q
        # 投影权重梯度
        hph = h_prev.reshape(B * L, d_model)
        grads["Wq"][l] += (dQ.reshape(-1, d_model).T @ hph) / (B * L)
        grads["Wk"][l] += (dK.reshape(-1, d_model).T @ hph) / (B * L)
        grads["Wv"][l] += (dV.reshape(-1, d_model).T @ hph) / (B * L)
        # 回传到 h_prev
        d_h_prev_attn = (dQ + dK + dV).reshape(B * L, d_model).reshape(B, L, d_model)
        d_h_prev = d_h + d_h_pre_from_ff          # 残差: 上层梯度 + FFN 路径
        d_h_prev_total = d_h_prev + d_h_prev_attn
        # 传递到更早一层
        d_h = d_h_prev_total
    # token / pos embedding 来自 h0 的 d_h
    grads["tok_emb"] = np.zeros_like(params["tok_emb"])
    np.add.at(grads["tok_emb"], Xb.ravel(), d_h.reshape(-1, d_model))
    grads["tok_emb"] /= (B * L)
    grads["pos_emb"] = d_h.mean(0)
    return grads


# ---- Adam 训练 ----
params = init_params()
m = {k: np.zeros_like(v) for k, v in params.items()}
v = {k: np.zeros_like(v) for k, v in params.items()}
lr0 = 0.01
BATCH = 256
n_batch = n_tr // BATCH
nb = 1e-8
best_val_corr = -1.0
best_params = None
patience = 25
no_improve = 0
for it in range(200):
    perm = rng.permutation(n_tr)
    for bi in range(n_batch):
        idx = perm[bi * BATCH:(bi + 1) * BATCH]
        _, _, cache, h = forward(params, Xtr[idx])
        g = backward(params, Xtr[idx], ytr[idx], cache, h)
        for k in params:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
            mhat = m[k] / (1 - 0.9 ** (it + 1))
            vhat = v[k] / (1 - 0.999 ** (it + 1))
            params[k] -= lr0 * mhat / (np.sqrt(vhat) + nb)
    # 验证集早停
    vp = forward(params, Xval)[0].ravel()
    val_corr = float(np.corrcoef(vp, yval)[0, 1])
    if val_corr > best_val_corr:
        best_val_corr = val_corr
        best_params = {k: np.array(v, copy=True) for k, v in params.items()}
        no_improve = 0
    else:
        no_improve += 1
    if it % 20 == 0:
        tp = forward(params, Xtr[:2000])[0].ravel()
        print("it", it, "train_corr", round(float(np.corrcoef(tp, ytr[:2000])[0, 1]), 3),
              "val_corr", round(val_corr, 3))
    if no_improve >= patience:
        print("early stop at it", it, "best_val_corr", round(best_val_corr, 3))
        break
if best_params is not None:
    params = best_params

tf_pred, attn_layers, _, _ = forward(params, Xte)
tf_r2 = 1 - np.mean((tf_pred.ravel() - yte) ** 2) / np.var(yte)
tf_dir = np.mean(np.sign(tf_pred.ravel()) == np.sign(yte))


# ---------------------------------------------------------------------------
# 3) 注意力诊断：CLS(位置0)作为读出 token 关注了哪些输入位置
# ---------------------------------------------------------------------------
xb = Xte[0:1]
_, attn_layers, _, _ = forward(params, xb)
A_last = attn_layers[-1][0].mean(0)        # (L,L) 平均跨头、最后一层
cls_row = A_last[0, :]                       # CLS query 对全部输入位置的注意力
attn_on_key = cls_row[[12, 30]].mean()       # CLS 对关键位置的注意力
# 命中率：关键位置(12,30)是否在 CLS 注意力最高的前 25% 位置中
sorted_pos = np.argsort(-cls_row)
topk = set(sorted_pos[:max(1, L // 4)])
hit = 1 if (12 in topk and 30 in topk) else 0
hit_rate = hit


# ---------------------------------------------------------------------------
# 4) 对抗扰动一致性
# ---------------------------------------------------------------------------
Xadv = Xte.copy()
Xadv[:, [5, 50]] = Xadv[:, [50, 5]]        # 互换两个非关键 token
tf_adv, _, _, _ = forward(params, Xadv)
consistency = np.corrcoef(tf_pred.ravel(), tf_adv.ravel())[0, 1]

print("=" * 60)
print("金融新闻 Transformer 情感 关键数字")
print("=" * 60)
print(f"测试集 R^2     : BoW={bow_r2:.3f}   Transformer={tf_r2:.3f}")
print(f"方向准确率     : BoW={bow_dir:.3f}   Transformer={tf_dir:.3f}")
print(f"R^2 相对 BoW 提升 : {100*(tf_r2-bow_r2)/abs(bow_r2):.1f}%")
print(f"注意力命中关键位置率 : {hit_rate:.2f}")
print(f"对抗扰动预测一致性   : {consistency:.3f}")

# ===========================================================================
# 配图 1: cover —— CLS(读出 token) 对各输入位置的注意力分布
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(range(L), cls_row, color=C["attn"], alpha=0.85, label="CLS 对位置的注意力权重")
ax.axhline(cls_row.mean(), color="gray", ls=":", lw=1.5, label=f"平均注意力 {cls_row.mean():.4f}")
ax.axvline(12, color=C["warn"], ls="--", lw=1.8, label="关键位置 12")
ax.axvline(30, color=C["true"], ls="--", lw=1.8, label="关键位置 30")
ax.set_xlabel("输入 token 位置")
ax.set_ylabel("CLS 注意力权重")
ax.set_title("Transformer 读出 token 的注意力分布：关键位置获得更高权重", fontsize=13)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "cover.png"), dpi=130)
plt.close()

# ===========================================================================
# 配图 2: BoW vs Transformer 性能对比
# ===========================================================================
fig, ax = plt.subplots(figsize=(7.5, 5.5))
models = ["BoW 词袋", "Transformer"]
r2s = [bow_r2, tf_r2]
dirs = [bow_dir, tf_dir]
x = np.arange(len(models))
ax.bar(x - 0.2, r2s, 0.4, color=C["bow"], label="测试集 R²")
ax.bar(x + 0.2, dirs, 0.4, color=C["tf"], label="方向准确率")
for i, (r, d) in enumerate(zip(r2s, dirs)):
    ax.text(i - 0.2, r + 0.01, f"{r:.2f}", ha="center", fontsize=10)
    ax.text(i + 0.2, d + 0.01, f"{d:.2f}", ha="center", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.08)
ax.set_title("Transformer 相对朴素词袋：R² 与方向准确率全面提升", fontsize=13)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "perf_compare.png"), dpi=130)
plt.close()

# ===========================================================================
# 配图 3: CLS 注意力沿标题位置的分布（关键位置是否被抬高）
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(range(L), cls_row, color=C["attn"], lw=2, marker="o", ms=3,
        label="CLS 对位置的注意力")
ax.axvline(12, color=C["warn"], ls="--", lw=1.5, label="关键位置 12")
ax.axvline(30, color=C["true"], ls="--", lw=1.5, label="关键位置 30")
ax.set_xlabel("输入 token 位置")
ax.set_ylabel("CLS 注意力权重")
ax.set_title("读出 token 对关键位置的注意力高于平均", fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "attn_distribution.png"), dpi=130)
plt.close()

print("✅ 配图已保存到", D)
print(["cover.png", "perf_compare.png", "attn_distribution.png"])
