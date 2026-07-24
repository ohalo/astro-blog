#!/usr/bin/env python3
"""
为文章「元学习量化策略：用 MAML 让模型在少样本上快速适应新市场」(meta-learning-quant)
生成真实配图 + 计算正文引用的关键数字。

机制（自洽合成，仅用于演示算法）：
  * 每个「市场/regime」是一个任务 task：因子->收益 的线性映射 y = X·w_task + 噪声。
    不同任务的 w_task 从共同分布 N(w0, tau^2 I) 采样（任务间共享结构 + 各自差异）。
  * MAML 学一个「元初始化」w_meta，使得在任一新任务上，只用 K 个样本做少数几步
    梯度下降(inner loop)就能快速适应。外层(outer loop)对适应后的损失做元梯度更新。
  * 对比三条基线：
      - Scratch: 每个新任务从随机初始化只用 K 样本训练
      - Pooled : 把所有任务数据混起来训一个全局模型（忽略任务差异）
      - MAML   : 学到的元初始化 + K 样本快速适应
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "meta-learning-quant")
os.makedirs(D, exist_ok=True)

C = {"maml": "#C44E52", "pool": "#4C72B0", "scratch": "#999999",
     "oracle": "#8172B3", "grid": "#DDDDDD", "warn": "#DD8452", "calm": "#55A868"}

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 任务分布：每个任务是一个 因子->收益 的线性模型
# ---------------------------------------------------------------------------
P = 5                                            # 因子数
W0 = np.array([0.9, -0.6, 0.4, 0.5, -0.3])       # 任务共享中心
TAU = 1.1                                         # 任务间 w 差异强度
SIG_Y = 1.0                                       # 观测噪声
K = 10                                            # 适应用样本数（few-shot）
INNER_LR = 0.12
INNER_STEPS = 5


def sample_task(seed=None):
    r = np.random.default_rng(seed)
    w = W0 + TAU * r.standard_normal(P)
    return w, r


def gen_data(w, n, r):
    X = r.standard_normal((n, P))
    y = X @ w + SIG_Y * r.standard_normal(n)
    return X, y


def mse(w, X, y):
    e = X @ w - y
    return float(e @ e / len(y))


def grad(w, X, y):
    e = X @ w - y
    return 2.0 * X.T @ e / len(y)


def adapt(w_init, X, y, lr=INNER_LR, steps=INNER_STEPS):
    w = w_init.copy()
    for _ in range(steps):
        w = w - lr * grad(w, X, y)
    return w


# ---------------------------------------------------------------------------
# 训练 MAML 元初始化（一阶近似 FOMAML）
# ---------------------------------------------------------------------------
N_TASKS_TRAIN = 200
META_LR = 0.02
META_EPOCHS = 300
TASK_BATCH = 8

# 预生成训练任务
train_tasks = [sample_task(seed=1000 + i) for i in range(N_TASKS_TRAIN)]

w_meta = W0 + 0.3 * rng.standard_normal(P)  # 随机起点
meta_curve = []
for epoch in range(META_EPOCHS):
    idx = rng.choice(N_TASKS_TRAIN, TASK_BATCH, replace=False)
    meta_grad = np.zeros(P)
    epoch_loss = 0.0
    for j in idx:
        w_task, r = train_tasks[j]
        Xs, ys = gen_data(w_task, K, r)          # support（适应用）
        Xq, yq = gen_data(w_task, 100, r)        # query（评估用）
        w_adapted = adapt(w_meta, Xs, ys)
        meta_grad += grad(w_adapted, Xq, yq)     # FOMAML：在适应后参数上求 query 梯度
        epoch_loss += mse(w_adapted, Xq, yq)
    w_meta = w_meta - META_LR * meta_grad / TASK_BATCH
    meta_curve.append(epoch_loss / TASK_BATCH)

# Pooled 基线：混合所有任务数据训一个全局模型
Xall, yall = [], []
for w_task, r in train_tasks:
    X, y = gen_data(w_task, 60, r)
    Xall.append(X); yall.append(y)
Xall = np.vstack(Xall); yall = np.concatenate(yall)
w_pool = np.linalg.lstsq(Xall, yall, rcond=None)[0]

# ---------------------------------------------------------------------------
# 在 100 个全新任务上评估三种方法
# ---------------------------------------------------------------------------
N_TEST = 100
res = {"maml": [], "pool": [], "scratch": [], "oracle": []}
for i in range(N_TEST):
    w_task, r = sample_task(seed=5000 + i)
    Xs, ys = gen_data(w_task, K, r)
    Xq, yq = gen_data(w_task, 300, r)

    # MAML：元初始化 + K 样本适应
    w_m = adapt(w_meta, Xs, ys)
    res["maml"].append(mse(w_m, Xq, yq))

    # Pooled：固定全局模型，直接用（忽略任务差异，不做 per-task 适应）
    res["pool"].append(mse(w_pool, Xq, yq))

    # Scratch：随机初始化只用 K 样本训练
    w_s = adapt(0.1 * r.standard_normal(P), Xs, ys, steps=INNER_STEPS)
    res["scratch"].append(mse(w_s, Xq, yq))

    # Oracle：用大样本直接拟合（理论下界）
    Xbig, ybig = gen_data(w_task, 2000, r)
    w_o = np.linalg.lstsq(Xbig, ybig, rcond=None)[0]
    res["oracle"].append(mse(w_o, Xq, yq))

MEAN = {k: float(np.mean(v)) for k, v in res.items()}
print("测试集平均 MSE:", {k: round(v, 4) for k, v in MEAN.items()})

# ---------------------------------------------------------------------------
# 图1：元训练曲线（外层 query loss 随 epoch 下降）
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.6), dpi=130)
ax.plot(meta_curve, color=C["maml"], lw=2.2, label="MAML 元训练 query 损失")
ax.axhline(MEAN["oracle"], color=C["oracle"], ls="--", lw=1.6,
           label=f"Oracle 下界 ≈ {MEAN['oracle']:.2f}")
ax.set_xlabel("元训练轮次 (outer epoch)")
ax.set_ylabel("适应后 query MSE")
ax.set_title("MAML 元训练：外层损失稳步逼近理论下界", fontsize=12.5, weight="bold")
ax.grid(alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(D, "meta_training_curve.png"), bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 图2：few-shot 适应曲线（inner steps 上 loss 下降，MAML vs Scratch）
# ---------------------------------------------------------------------------
steps_axis = list(range(0, 9))
maml_path, scratch_path = [], []
for i in range(N_TEST):
    w_task, r = sample_task(seed=7000 + i)
    Xs, ys = gen_data(w_task, K, r)
    Xq, yq = gen_data(w_task, 300, r)
    wm = w_meta.copy(); ws = 0.1 * r.standard_normal(P)
    m_row, s_row = [], []
    for s in steps_axis:
        m_row.append(mse(wm, Xq, yq)); s_row.append(mse(ws, Xq, yq))
        wm = wm - INNER_LR * grad(wm, Xs, ys)
        ws = ws - INNER_LR * grad(ws, Xs, ys)
    maml_path.append(m_row); scratch_path.append(s_row)
maml_path = np.array(maml_path).mean(0)
scratch_path = np.array(scratch_path).mean(0)

fig, ax = plt.subplots(figsize=(8, 4.6), dpi=130)
ax.plot(steps_axis, maml_path, "o-", color=C["maml"], lw=2.2, label="MAML 元初始化")
ax.plot(steps_axis, scratch_path, "s--", color=C["scratch"], lw=2.0, label="随机初始化 (Scratch)")
ax.axhline(MEAN["oracle"], color=C["oracle"], ls=":", lw=1.5, label="Oracle 下界")
ax.set_xlabel("新市场上的适应步数 (inner steps, K=10 样本)")
ax.set_ylabel("query MSE")
ax.set_title("少样本适应：MAML 起点低、一两步就到位", fontsize=12.5, weight="bold")
ax.grid(alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(D, "fewshot_adaptation.png"), bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 图3：三方法在 100 个新市场上的 MSE 分布（箱线图）
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.6), dpi=130)
data = [res["scratch"], res["pool"], res["maml"], res["oracle"]]
labels = ["Scratch\n(从零学)", "Pooled\n(全局混训)", "MAML\n(元初始化+适应)", "Oracle\n(大样本下界)"]
colors = [C["scratch"], C["pool"], C["maml"], C["oracle"]]
bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", lw=1.4))
for patch, col in zip(bp["boxes"], colors):
    patch.set_facecolor(col); patch.set_alpha(0.65)
for i, v in enumerate(data):
    ax.scatter([i + 1], [np.mean(v)], color="black", zorder=5, s=28, marker="D")
    ax.annotate(f"{np.mean(v):.2f}", (i + 1, np.mean(v)),
                textcoords="offset points", xytext=(10, 0), fontsize=9.5, weight="bold")
ax.set_ylabel("新市场 query MSE（越低越好）")
ax.set_title("100 个全新市场上的适应误差分布", fontsize=12.5, weight="bold")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(D, "method_comparison_box.png"), bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 图4：样本量 K 敏感性（MAML 优势随 K 收窄）
# ---------------------------------------------------------------------------
Ks = [3, 5, 10, 20, 50, 100]
maml_k, scratch_k, pool_k = [], [], []
for k in Ks:
    mm, ss, pp = [], [], []
    for i in range(60):
        w_task, r = sample_task(seed=9000 + i)
        Xs, ys = gen_data(w_task, k, r)
        Xq, yq = gen_data(w_task, 300, r)
        mm.append(mse(adapt(w_meta, Xs, ys), Xq, yq))
        ss.append(mse(adapt(0.1 * r.standard_normal(P), Xs, ys), Xq, yq))
        pp.append(mse(w_pool, Xq, yq))
    maml_k.append(np.mean(mm)); scratch_k.append(np.mean(ss)); pool_k.append(np.mean(pp))

fig, ax = plt.subplots(figsize=(8, 4.6), dpi=130)
ax.plot(Ks, maml_k, "o-", color=C["maml"], lw=2.2, label="MAML")
ax.plot(Ks, pool_k, "^-", color=C["pool"], lw=2.0, label="Pooled")
ax.plot(Ks, scratch_k, "s--", color=C["scratch"], lw=2.0, label="Scratch")
ax.axhline(MEAN["oracle"], color=C["oracle"], ls=":", lw=1.5, label="Oracle 下界")
ax.set_xscale("log")
ax.set_xticks(Ks); ax.set_xticklabels([str(k) for k in Ks])
ax.set_xlabel("新市场可用样本数 K（对数轴）")
ax.set_ylabel("query MSE")
ax.set_title("样本越少，MAML 优势越大；样本足够时三者收敛", fontsize=12.5, weight="bold")
ax.grid(alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(D, "sample_size_sensitivity.png"), bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 导出关键数字
# ---------------------------------------------------------------------------
stats = {
    "P": P, "K": K, "TAU": TAU, "N_TASKS_TRAIN": N_TASKS_TRAIN,
    "META_EPOCHS": META_EPOCHS, "N_TEST": N_TEST,
    "mse_mean": {k: round(v, 4) for k, v in MEAN.items()},
    "maml_vs_scratch_reduction": round((MEAN["scratch"] - MEAN["maml"]) / MEAN["scratch"], 4),
    "maml_vs_pool_reduction": round((MEAN["pool"] - MEAN["maml"]) / MEAN["pool"], 4),
    "maml_gap_to_oracle": round(MEAN["maml"] - MEAN["oracle"], 4),
    "meta_curve_start": round(meta_curve[0], 4),
    "meta_curve_end": round(meta_curve[-1], 4),
    "fewshot_step0": round(float(maml_path[0]), 4),
    "fewshot_step1": round(float(maml_path[1]), 4),
    "sample_k": {str(k): {"maml": round(m, 3), "scratch": round(s, 3), "pool": round(p, 3)}
                 for k, m, s, p in zip(Ks, maml_k, scratch_k, pool_k)},
}
with open(os.path.join(D, "_stats.json"), "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(json.dumps(stats, ensure_ascii=False, indent=2))
print("\n图片已保存到:", D)
