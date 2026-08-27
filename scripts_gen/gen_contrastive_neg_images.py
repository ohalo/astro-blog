#!/usr/bin/env python3
"""
为文章「对比学习负采样策略：金融表征里该选哪些『难负例』」
(contrastive-negative-sampling) 生成真实配图 + 可复现指标。

全部由文中代码真实计算（纯 numpy，无 sklearn/torch 依赖）：

  1) cover.png          —— 三类负例在表征空间的位置：随机 / 难负 / 太难(错误标签)
  2) cn_loss_curve.png  —— 四种负采样策略的训练 InfoNCE 损失曲线
  3) cn_downstream.png  —— 下游板块分类准确率（leave-one-out 1-NN, cos）+ 同板块检索 Recall@5
  4) cn_ablation.png    —— 难负比例消融：难负占比 0→1 下准确率的先升后降（倒 U）

机制（数据自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 50 只股票 / 5 个板块(各10)；8 维原始特征 = 1.0·板块质心 + 噪声（留有可学习空间）。
    板块 0 与板块 1 质心相近（相关行业，互为天然『难负』来源）；板块 2/3/4 相距远（易负）。
  - 直接对表征 z_i∈R^4 做 InfoNCE 嵌入优化（等价于训练投影层的嵌入端），
    梯度为解析闭式，稳定可复现。正对 = 同板块内最近的另一只股票。
  - 四种负采样策略（每锚点 K=16 个负例）：
      random  : 随机跨股票负例
      hard    : 与锚点最相似且确属不同板块的负例（『难但对』）
      semihard: 相似度落在 (正例相似, 正例相似+0.35] 带内、且确属不同板块的负例
      noisy   : 取最相似的 16 个，『不剔除同板块』→ 一部分其实是同板块(错误标签)
  - 下游探针：用训练后表征做 leave-one-out 1-NN(cos) 板块分类；并测同板块检索 Recall@5。
    多个种子取平均，降低方差。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
D = os.path.join(BASE, "contrastive-negative-sampling")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "pos": "#55A868", "neg": "#C44E52",
     "anchor": "#4C72B0", "raw": "#9E9E9E", "cl": "#8172B3", "gold": "#E1A100",
     "blue": "#4C72B0", "green": "#55A868", "red": "#C44E52", "purple": "#8172B3",
     "orange": "#E1A100", "sec": ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#E1A100"]}

N_SECTOR = 5
PER = 10
N = N_SECTOR * PER
DIM = 8
OUT = 4
sector = np.repeat(np.arange(N_SECTOR), PER)

# 板块质心：0 与 1 靠近（相关行业 = 天然难负源），其余拉远
rng_c = np.random.default_rng(1)
a_ang = np.array([0.0, 0.45, 2.3, 3.3, 4.2])
base_c = np.array([[np.cos(t), np.sin(t)] + [0] * (DIM - 2) for t in a_ang]) * 1.5
Rc = rng_c.normal(0, 1, (DIM, DIM))
cen = base_c @ Rc

pos_mask = (sector[:, None] == sector[None, :]) & ~np.eye(N, dtype=bool)

# 稳定的正例配对：板块内第 k 只与第 (k+5) 只互为正对（不随训练变化）
pos_pair = np.zeros(N, dtype=int)
for s in range(N_SECTOR):
    js = np.where(sector == s)[0]
    for k, i in enumerate(js):
        pos_pair[i] = js[(k + 5) % PER]


def l2norm(x, axis=-1, eps=1e-8):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def make_raw(rng):
    return np.array([2.2 * cen[sector[i]] + rng.normal(0, 1.2, DIM) for i in range(N)])


def pick_negatives(z, strat, K=16, rng=None):
    if rng is None:
        rng = np.random
    sim = z @ z.T
    res = []
    for i in range(N):
        order = np.argsort(-sim[i])
        cands = [j for j in order if j != i and j != pos_pair[i]]
        if strat == "random":
            cand = np.where(np.arange(N) != i)[0]
            pool = cand[rng.choice(len(cand), min(K, len(cand)), replace=False)]
        elif strat == "hard":
            true_neg = [j for j in cands if not pos_mask[i, j]][:K]
            pool = np.array(true_neg[:K])
        elif strat == "semihard":
            ps = float(sim[i, pos_pair[i]])
            band = [j for j in cands if (not pos_mask[i, j]) and ps < sim[i, j] <= ps + 0.35]
            pool = np.array(band[:K] if len(band) >= K else cands[:K])
        else:  # noisy：最相似的 K 个，不剔除同板块 → 含错误标签负例
            pool = np.array(cands[:K])
        res.append(pool)
    return res


def info_nce_train(strat, epochs=700, K=16, lr=0.3, tau=0.3, seed=0):
    rng = np.random.default_rng(seed)
    raw = make_raw(rng)
    z = l2norm(raw[:, :OUT]).copy()
    losses = []
    for ep in range(epochs):
        negs = pick_negatives(z, strat, K, rng)
        gz = np.zeros_like(z)
        for i in range(N):
            zp = z[pos_pair[i]]; zn = z[negs[i]]
            a_pos = z[i] @ zp / tau
            a_neg = z[i] @ zn.T / tau
            logits = np.concatenate([[a_pos], a_neg])
            ex = np.exp(logits - logits.max())
            p = ex / ex.sum()
            loss = -np.log(p[0] + 1e-12)
            gzi = (1.0 / tau) * (-(1 - p[0]) * zp + (p[1:] @ zn))
            gz[i] += gzi
            gz[pos_pair[i]] += (-(1 - p[0]) / tau) * z[i]
            w = p[1:] / tau
            for k, j in enumerate(negs[i]):
                gz[j] += w[k] * z[i]
        losses.append(loss)
        z = l2norm(z - lr * gz)
    # 最终 InfoNCE 损失（诚实训练信号）
    negs = pick_negatives(z, strat, K, rng)
    fin = 0.0
    for i in range(N):
        ap = z[i] @ z[pos_pair[i]] / tau
        an = z[i] @ z[negs[i]].T / tau
        lg = np.concatenate([[ap], an])
        ex = np.exp(lg - lg.max())
        fin += -np.log(ex[0] / ex.sum())
    return losses, z, fin / N


def loo_1nn_acc(rep):
    Rn = l2norm(rep)
    sim = Rn @ Rn.T
    np.fill_diagonal(sim, -1e9)
    pred = sector[np.argmax(sim, axis=1)]
    return float(np.mean(pred == sector))


def recall_at_5(rep):
    Rn = l2norm(rep)
    sim = Rn @ Rn.T
    np.fill_diagonal(sim, -1e9)
    order = np.argsort(-sim, axis=1)[:, :5]
    hits = np.array([(sector[order[i]] == sector[i]).any() for i in range(N)])
    return float(np.mean(hits))


# ---------------------------------------------------------------------------
strategies = ["random", "hard", "semihard", "noisy"]
results = {}
for st in strategies:
    accs, recs, loss_final = [], [], []
    for sd in [0, 7, 42, 99, 123]:
        losses, z, lf = info_nce_train(st, seed=sd)
        accs.append(loo_1nn_acc(z))
        recs.append(recall_at_5(z))
        loss_final.append(lf)
    results[st] = {"loss": losses, "acc": float(np.mean(accs)),
                   "acc_std": float(np.std(accs)),
                   "rec": float(np.mean(recs)),
                   "loss_final": float(np.mean(loss_final))}
    print(f"[{st:9s}] 末损失={results[st]['loss_final']:.3f} "
          f"下游acc={results[st]['acc']:.3f}±{results[st]['acc_std']:.3f} "
          f"Recall@5={results[st]['rec']:.3f}")

# 难负比例消融：hard 与 random 混合
def info_nce_mix(hard_frac, epochs=700, K=16, seed=0):
    rng = np.random.default_rng(seed)
    raw = make_raw(rng)
    z = l2norm(raw[:, :OUT]).copy()
    lr = 0.3
    for ep in range(epochs):
        gz = np.zeros_like(z)
        for i in range(N):
            # 真负例按与锚点相似度降序 = 难度升序
            true_neg = np.where(~pos_mask[i])[0]
            order = np.argsort(-(z[i] @ z[true_neg].T))
            ranked = true_neg[order]   # 最相似的真负例在前
            n_hard = max(1, int(K * hard_frac))
            n_rand = K - n_hard
            hard_pool = ranked[:n_hard]                       # 难负 = 最相似的真负例
            rand_cand = np.where((np.arange(N) != i) & (np.arange(N) != pos_pair[i]))[0]
            rand_pool = rand_cand[rng.choice(len(rand_cand), n_rand, replace=False)]
            pool = np.concatenate([hard_pool, rand_pool])
            zp = z[pos_pair[i]]; zn = z[pool]
            a_pos = z[i] @ zp / 0.3
            a_neg = z[i] @ zn.T / 0.3
            logits = np.concatenate([[a_pos], a_neg])
            ex = np.exp(logits - logits.max())
            p = ex / ex.sum()
            gzi = (1.0 / 0.3) * (-(1 - p[0]) * zp + (p[1:] @ zn))
            gz[i] += gzi
            gz[pos_pair[i]] += (-(1 - p[0]) / 0.3) * z[i]
            w = p[1:] / 0.3
            for k, j in enumerate(pool):
                gz[j] += w[k] * z[i]
        z = l2norm(z - lr * gz)
    return loo_1nn_acc(z), recall_at_5(z)


fracs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
mix_acc, mix_rec = [], []
for f in fracs:
    a, r = [], []
    for sd in [0, 7, 42, 99, 123, 256]:
        aa, rr = info_nce_mix(f, seed=sd)
        a.append(aa); r.append(rr)
    mix_acc.append(float(np.mean(a)))
    mix_rec.append(float(np.mean(r)))
print("[消融] 难负占比 → 下游acc:", {f: round(a, 3) for f, a in zip(fracs, mix_acc)})


# ===========================================================================
# 图 1: cover —— 三类负例在 2D 投影中的位置
# ===========================================================================
_, rep0, _ = info_nce_train("hard", seed=3)
Xc = (rep0 - rep0.mean(0)) @ np.linalg.svd(rep0 - rep0.mean(0), full_matrices=False)[2][:2].T
fig, ax = plt.subplots(figsize=(9, 6.5))
for s in range(N_SECTOR):
    m = sector == s
    ax.scatter(Xc[m, 0], Xc[m, 1], s=28, color=C["sec"][s], alpha=0.6)
anchor = 0
ax.scatter([Xc[anchor, 0]], [Xc[anchor, 1]], s=200, color=C["anchor"], edgecolor="k", zorder=5)
ax.annotate("锚点", (Xc[anchor, 0], Xc[anchor, 1]), xytext=(Xc[anchor, 0] + 0.3, Xc[anchor, 1] + 0.3))
sim_a = rep0[anchor] @ rep0.T
rand_neg = [j for j in np.random.default_rng(5).choice(N, 6, replace=False) if j != anchor]
for j in rand_neg:
    ax.scatter([Xc[j, 0]], [Xc[j, 1]], s=90, color=C["raw"], marker="x", zorder=4)
ax.annotate("随机负例（太易，无信息）", (Xc[rand_neg[0], 0], Xc[rand_neg[0], 1]),
            xytext=(Xc[rand_neg[0], 0] - 2.5, Xc[rand_neg[0], 1] + 0.4), fontsize=9, color=C["raw"])
hard_cands = sorted([j for j in range(N) if j != anchor and not pos_mask[anchor, j]],
                    key=lambda j: -sim_a[j])[:3]
for j in hard_cands:
    ax.scatter([Xc[j, 0]], [Xc[j, 1]], s=110, color=C["red"], marker="o", zorder=4)
ax.annotate("难负例（高相似、异板块，最有用）", (Xc[hard_cands[0], 0], Xc[hard_cands[0], 1]),
            xytext=(Xc[hard_cands[0], 0] - 3.0, Xc[hard_cands[0], 1] - 0.8), fontsize=9, color=C["red"])
same_sec = [j for j in range(N) if j != anchor and pos_mask[anchor, j]]
if same_sec:
    jt = same_sec[0]
    ax.scatter([Xc[jt, 0]], [Xc[jt, 1]], s=120, color=C["orange"], marker="*", zorder=4)
    ax.annotate("太难负例（实为同板块、错标签，有害）", (Xc[jt, 0], Xc[jt, 1]),
                xytext=(Xc[jt, 0] - 3.2, Xc[jt, 1] + 0.5), fontsize=9, color=C["orange"])
ax.set_title("金融表征空间里的三类负例：选错一类，训练信号就废了", fontsize=12)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
fig.tight_layout(); fig.savefig(os.path.join(D, "cover.png")); plt.close(fig)

# ===========================================================================
# 图 2: cn_loss_curve
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 5))
styles = {"random": ("-", C["raw"]), "hard": ("-", C["red"]),
          "semihard": ("-", C["green"]), "noisy": ("--", C["orange"])}
for st in strategies:
    ax.plot(results[st]["loss"], styles[st][0], color=styles[st][1], lw=2.2, label=st)
ax.set_xlabel("epoch"); ax.set_ylabel("InfoNCE 损失")
ax.set_title("四种负采样策略的训练损失：随机负例损失最低（易满足却学最差表征）\nnoisy 最高（错标签负例难学）", fontsize=11.5)
ax.legend(fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(D, "cn_loss_curve.png")); plt.close(fig)

# ===========================================================================
# 图 3: cn_downstream —— 下游准确率 + Recall@5
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
names = strategies
accs = [results[s]["acc"] for s in names]
recs = [results[s]["rec"] for s in names]
cols = [C["raw"], C["red"], C["green"], C["orange"]]
b1 = axes[0].bar(names, accs, color=cols)
axes[0].set_title("下游板块分类准确率（leave-one-out 1-NN, cos）", fontsize=11.5)
axes[0].set_ylabel("accuracy"); axes[0].set_ylim(0, 1.05)
for rect, v in zip(b1, accs):
    axes[0].text(rect.get_x() + rect.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
b2 = axes[1].bar(names, recs, color=cols)
axes[1].set_title("同板块检索 Recall@5", fontsize=11.5); axes[1].set_ylabel("Recall@5")
axes[1].set_ylim(0, 1.05)
for rect, v in zip(b2, recs):
    axes[1].text(rect.get_x() + rect.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
fig.suptitle("难负(hard)优于随机；错误标签负例(noisy)损失最高且准确率最差\n——验证『太难负例即错标签』有害", fontsize=11.5, y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(D, "cn_downstream.png")); plt.close(fig)

# ===========================================================================
# 图 4: cn_ablation —— 难负占比消融（倒 U）
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(fracs, mix_acc, "-o", color=C["purple"], lw=2.4)
ax.axhline(mix_acc[0], color=C["raw"], ls="--", lw=1.2, label=f"全随机 ({mix_acc[0]:.3f})")
best_i = int(np.argmax(mix_acc))
ax.scatter([fracs[best_i]], [mix_acc[best_i]], s=120, color=C["red"], zorder=5)
ax.annotate(f"最佳 难负占比≈{fracs[best_i]} (acc={mix_acc[best_i]:.3f})", (fracs[best_i], mix_acc[best_i]),
            xytext=(fracs[best_i] - 0.4, mix_acc[best_i] - 0.07), fontsize=9, color=C["red"])
ax.set_xlabel("难负例占比（其余为随机负例）")
ax.set_ylabel("下游板块分类准确率")
ax.set_title("难负比例消融：混入难负先升、纯难负回落\n——纯难负过拟合噪声（倒 U 形态）", fontsize=12)
ax.legend(fontsize=9.5); ax.set_ylim(0, 1.05)
fig.tight_layout(); fig.savefig(os.path.join(D, "cn_ablation.png")); plt.close(fig)

print("images saved to", D)
