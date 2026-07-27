# -*- coding: utf-8 -*-
"""回测过拟合概率 PBO 配图（Bailey & López de Prado 2014, CSCV）"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/pbo-overfitting-probability"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(7)

# ---------------------------------------------------------------
# 模拟：N 个策略变体的收益矩阵 (T 期 x N 策略)
# 情形 A：全是噪声（真实 alpha=0）——应得高 PBO
# 情形 B：少数有真 alpha ——应得低 PBO
# ---------------------------------------------------------------
T = 1000
N_STRAT = 40

def make_returns(n_true_alpha, seed):
    r = np.random.default_rng(seed)
    R = r.normal(0, 0.01, (T, N_STRAT))
    # 给前 n_true_alpha 个策略注入持续 alpha
    for k in range(n_true_alpha):
        R[:, k] += 0.01 * 0.09   # 稳定日度 alpha
    return R

def sharpe(x):
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + 1e-12
    return mu / sd * np.sqrt(252)

def cscv_pbo(R, S=14):
    """CSCV：把时间切成 S 块，一半训练一半测试的所有组合"""
    T_ = R.shape[0]
    idx = np.array_split(np.arange(T_), S)
    logits = []
    is_oos = []
    for combo in combinations(range(S), S // 2):
        train_blocks = list(combo)
        test_blocks = [b for b in range(S) if b not in train_blocks]
        tr = np.concatenate([idx[b] for b in train_blocks])
        te = np.concatenate([idx[b] for b in test_blocks])
        sr_is = sharpe(R[tr])
        sr_oos = sharpe(R[te])
        n_star = np.argmax(sr_is)          # 样本内最优策略
        # 该策略在样本外的相对排名
        rank = (sr_oos < sr_oos[n_star]).sum() / (len(sr_oos) - 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
        is_oos.append((sr_is[n_star], sr_oos[n_star]))
    logits = np.array(logits)
    pbo = (logits < 0).mean()
    return logits, pbo, np.array(is_oos)

# 情形 A：纯噪声
R_noise = make_returns(0, 101)
log_A, pbo_A, iso_A = cscv_pbo(R_noise)
# 情形 B：8 个真 alpha
R_alpha = make_returns(8, 202)
log_B, pbo_B, iso_B = cscv_pbo(R_alpha)

print(f"PBO(纯噪声)={pbo_A:.3f}   PBO(含真alpha)={pbo_B:.3f}   组合路径数={len(log_A)}")

# ===============================================================
# 图 1：CSCV 分块示意（S=8 简化版做可视化）
# ===============================================================
S_vis = 8
combos = list(combinations(range(S_vis), S_vis // 2))
M = np.zeros((len(combos), S_vis))
for i, c in enumerate(combos):
    for b in range(S_vis):
        M[i, b] = 0 if b in c else 1   # 0=训练 1=测试
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.imshow(M, cmap="coolwarm", aspect="auto", vmin=0, vmax=1)
for i in range(len(combos)):
    for b in range(S_vis):
        ax.text(b, i, "训" if M[i, b] == 0 else "测",
                ha="center", va="center", fontsize=8,
                color="white" if M[i, b] == 1 else "#1a1a1a")
ax.set_xticks(range(S_vis))
ax.set_xticklabels([f"块{b+1}" for b in range(S_vis)])
ax.set_yticks(range(0, len(combos), 5))
ax.set_yticklabels([f"#{i+1}" for i in range(0, len(combos), 5)])
ax.set_xlabel("时间分块（按时序切成 S 块）")
ax.set_ylabel(f"组合划分（共 C(8,4)={len(combos)} 种）")
ax.set_title("CSCV：一半块训练、一半块测试，穷举所有对称组合", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/cscv-partition.jpg", dpi=130)
plt.close(fig)

# ===============================================================
# 图 2：logit 分布对比（噪声 vs 真 alpha）
# ===============================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
a1.hist(log_A, bins=22, color="#4C72B0", alpha=0.78, edgecolor="white")
a1.axvline(0, color="#C44E52", ls="--", lw=2.0)
a1.set_title(f"纯噪声策略池：PBO ≈ {pbo_A:.2f}", fontsize=12.5)
a1.set_xlabel("logit(样本外相对排名)")
a1.set_ylabel("路径数")
a1.grid(alpha=0.25)
a2.hist(log_B, bins=22, color="#55A868", alpha=0.82, edgecolor="white")
a2.axvline(0, color="#C44E52", ls="--", lw=2.0)
a2.set_title(f"含 8 个真 alpha：PBO ≈ {pbo_B:.2f}", fontsize=12.5)
a2.set_xlabel("logit(样本外相对排名)")
a2.grid(alpha=0.25)
fig.suptitle("logit<0（红线左侧）= 样本内最优在样本外掉到中位数以下 → PBO", fontsize=12.5)
fig.tight_layout()
fig.savefig(f"{OUT}/pbo-logit-distribution.jpg", dpi=130)
plt.close(fig)

# ===============================================================
# 图 3：样本内 vs 样本外 夏普散点（退化图）
# ===============================================================
fig, ax = plt.subplots(figsize=(9, 5.4))
ax.scatter(iso_A[:, 0], iso_A[:, 1], s=42, color="#C44E52", alpha=0.6, label="纯噪声池")
ax.scatter(iso_B[:, 0], iso_B[:, 1], s=42, color="#55A868", alpha=0.6, label="含真 alpha 池")
lo = min(iso_A[:, 0].min(), iso_B[:, 0].min(), iso_A[:, 1].min())
hi = max(iso_A[:, 0].max(), iso_B[:, 0].max())
ax.plot([lo, hi], [lo, hi], "--", color="gray", label="IS=OOS（理想）")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("样本内最优策略的夏普 (IS)")
ax.set_ylabel("同一策略的样本外夏普 (OOS)")
ax.set_title("纯噪声池里 IS 高的策略 OOS 掉回零附近；真 alpha 池才守在对角线上", fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/is-oos-degradation.jpg", dpi=130)
plt.close(fig)

print("PBO images done:", os.listdir(OUT))
