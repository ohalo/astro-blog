#!/usr/bin/env python3
"""生成「生成对抗模仿学习：从优秀交易员轨迹里学策略」配图。

纯 numpy 从零实现 GAIL（Generative Adversarial Imitation Learning, Ho & Ermon 2016）：
  判别器 D 区分『专家轨迹 vs 生成器(策略)轨迹』，生成器(策略) 用 D 的奖励信号
  做策略优化，形成对抗博弈。本文诚实演示：(a) GAIL 能从随机策略经对抗训练逼近专家；
  (b) 判别器最终被"骗到分不清"(acc→0.5)，奖赏信号耗尽；(c) 在单步 iid 模仿里，
  BC(行为克隆) 直接回归动作更高效，GAIL 仍逼近专家——并诚实标注 GAIL 的真正优势场在
  序贯/误差累积场景。
"""
import numpy as np
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False
fm._load_fontmanager()

SEED = 20260828
rng = np.random.default_rng(SEED)

# ============ 1. 受控环境：状态 s 是 4 维特征，专家策略 π_E 输出 1 维动作 ============
D_S = 4
D_A = 1
N_EXPERT_FULL = 1500

W_EXPERT = np.array([1.4, -0.8, 0.6, 0.3])

def expert_policy(S):
    return S @ W_EXPERT + rng.standard_normal(S.shape[0]) * 0.15

def true_reward(S, A):
    target = S @ W_EXPERT
    fit = -((A - target) ** 2)
    smooth = -0.1 * (A ** 2)
    return fit + smooth

def sample_state(n, ood=False):
    if ood:
        return rng.standard_normal((n, D_S)) * 1.8 + 1.2
    return rng.standard_normal((n, D_S))

# 完整专家池
S_pool = sample_state(N_EXPERT_FULL)
A_pool = expert_policy(S_pool)
R_expert = float(true_reward(S_pool, A_pool).mean())

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ============ 2. 行为克隆 BC（OLS 回归专家动作） ============
def train_bc(S_demo, A_demo):
    N = len(S_demo)
    w = np.zeros(D_S)
    for _ in range(400):
        p = S_demo @ w
        w -= 0.02 * (S_demo.T @ (p - A_demo) / N)
    return w

def eval_policy(w, ood=False):
    S = sample_state(3000, ood=ood)
    return float(true_reward(S, S @ w).mean())

# ============ 3. GAIL：判别器 + 生成器(策略) 对抗 ============
def train_gail(S_demo, A_demo, n_iter=2500, lr_d=0.05, lr_g=0.0005, batch=256):
    N = len(S_demo)
    bs = min(batch, N)
    w_d = rng.standard_normal(D_S + D_A) * 0.1
    b_d = 0.0
    rng_g = np.random.default_rng(SEED + 7)
    w_g = rng_g.standard_normal(D_S) * 0.1
    hist_reward, hist_acc = [], []
    for it in range(n_iter):
        idx = rng.choice(N, bs)
        Se = S_demo[idx]; Ae = A_demo[idx]
        Sg = sample_state(bs)
        Ag = Sg @ w_g + rng.standard_normal(bs) * 0.05
        fe_e = np.hstack([Se, Ae[:, None]])
        fe_g = np.hstack([Sg, Ag[:, None]])
        p_e = sigmoid(fe_e @ w_d + b_d)
        p_g = sigmoid(fe_g @ w_d + b_d)
        grad_d = (fe_e * p_e[:, None]).mean(0) - (fe_g * (1 - p_g)[:, None]).mean(0)
        w_d += lr_d * grad_d
        b_d += lr_d * (p_e.mean() - (1 - p_g).mean())
        wdn = np.linalg.norm(w_d)
        if wdn > 3.0:
            w_d = w_d / wdn * 3.0
        logD = np.log(p_g + 1e-8)
        adv = logD - logD.mean()
        dlogD = (1 - p_g) * w_d[-D_A:]
        grad_g = (adv[:, None] * dlogD[:, None] * Sg).mean(0)
        gn = np.linalg.norm(grad_g)
        if gn > 1.0:
            grad_g = grad_g / gn * 1.0
        w_g += lr_g * grad_g
        if it % 100 == 0:
            hist_reward.append(float(true_reward(Sg, Ag).mean()))
            hist_acc.append(float(0.5 * (p_e > 0.5).mean() + 0.5 * (p_g < 0.5).mean()))
    return w_g, hist_reward, hist_acc

# ---- 充足演示下 BC vs GAIL（诚实对照：单步 iid 下 BC 更直接、更高效） ----
W_BC_full = train_bc(S_pool, A_pool)
W_GAIL_full, hg_full, ha_full = train_gail(S_pool, A_pool)
R_BC_full = eval_policy(W_BC_full)
R_GAIL_full = eval_policy(W_GAIL_full)

# ============ 4. 绘图 ============
outdir = "public/images/gail-imitation-trading"
os.makedirs(outdir, exist_ok=True)

# 图1 cover：GAIL 生成器回报随对抗训练上升（充足演示）
fig, ax = plt.subplots(figsize=(11, 5.2))
xs = np.arange(len(hg_full)) * 100
ax.plot(xs, hg_full, "o-", color="#4393c3", lw=2, label="GAIL 生成器回报（真实环境评估）")
ax.axhline(R_expert, color="#1a9850", ls="--", lw=1.6, label=f"专家回报 R={R_expert:.3f}")
ax.axhline(eval_policy(np.zeros(D_S)), color="gray", ls=":", lw=1.3, label="随机起点")
ax.set_xlabel("GAIL 对抗迭代")
ax.set_ylabel("生成器在真实环境的回报")
ax.set_title(f"GAIL：从随机策略经对抗训练逼近专家（最终 R={R_GAIL_full:.3f}，专家 {R_expert:.3f}）",
             fontsize=12.0, fontweight="bold", color="#1f3a5f")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{outdir}/gail_convergence.png", dpi=130)
plt.close(fig)

# 图2：判别器混淆度（准确率从能分清 -> 0.5）
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.plot(xs, ha_full, "s-", color="#762a83", lw=2, label="判别器准确率（专家 vs 生成）")
ax.axhline(0.5, color="gray", ls="--", lw=1.2, label="随机猜（0.5）= 完全混淆")
ax.set_xlabel("GAIL 对抗迭代")
ax.set_ylabel("判别器准确率")
ax.set_title("对抗收敛：判别器从能分清降到分不清（奖赏信号被耗尽）",
             fontsize=12.0, fontweight="bold", color="#1f3a5f")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{outdir}/discriminator_confusion.png", dpi=130)
plt.close(fig)

# 图3：诚实对照 —— 同一批干净演示下 BC vs GAIL（单步 iid 下 BC 更直接高效，GAIL 仍逼近专家）
fig, ax = plt.subplots(figsize=(11, 5.0))
labels = ["专家", "BC(行为克隆)", "GAIL"]
vals = [R_expert, R_BC_full, R_GAIL_full]
colors = ["#1a9850", "#d73027", "#4393c3"]
ax.bar(labels, vals, color=colors, width=0.55)
ax.axhline(0, color="gray", lw=1.0)
ax.set_ylabel("真实环境回报（越高越好）")
ax.set_title(f"同一批演示下：BC 直接回归更高效({R_BC_full:.2f})，GAIL 经对抗也逼近专家({R_GAIL_full:.2f})",
             fontsize=11.0, fontweight="bold", color="#1f3a5f")
for i, v in enumerate(vals):
    ax.text(i, v + (0.05 if v >= 0 else -0.12), f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(f"{outdir}/bc_vs_gail.png", dpi=130)
plt.close(fig)

# 图4：动作分布对比（ID 样本上）
fig, ax = plt.subplots(figsize=(11, 5.0))
S_id = sample_state(800)
ax.hist(expert_policy(S_id), bins=40, alpha=0.5, color="#1a9850", density=True, label="专家动作")
ax.hist(S_id @ W_BC_full, bins=40, alpha=0.5, color="#d73027", density=True, label="BC 生成动作")
ax.hist(S_id @ W_GAIL_full, bins=40, alpha=0.5, color="#4393c3", density=True, label="GAIL 生成动作")
ax.set_xlabel("动作 a（线性策略输出）")
ax.set_ylabel("密度")
ax.set_title("动作分布：GAIL 还原出贴近专家的分布形态",
             fontsize=10.5, fontweight="bold", color="#1f3a5f")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{outdir}/action_distribution.png", dpi=130)
plt.close(fig)

# ============ 5. stats ============
stats = {
    "D_S": D_S, "N_expert_full": N_EXPERT_FULL,
    "R_expert": round(R_expert, 4),
    "R_bc_full": round(R_BC_full, 4), "R_gail_full": round(R_GAIL_full, 4),
    "gail_conv_start": round(float(hg_full[0]), 4), "gail_conv_end": round(float(hg_full[-1]), 4),
    "disc_acc_start": round(float(ha_full[0]), 4), "disc_acc_end": round(float(ha_full[-1]), 4),
    "w_expert": [round(float(x), 3) for x in W_EXPERT],
    "w_bc_full": [round(float(x), 3) for x in W_BC_full],
    "w_gail_full": [round(float(x), 3) for x in W_GAIL_full],
}
with open(f"{outdir}/stats.json", "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print("=== GAIL imitation metrics ===")
print(f"  R_expert      = {R_expert:.4f}")
print(f"  充足: BC={R_BC_full:.3f}  GAIL={R_GAIL_full:.3f}  (conv {hg_full[0]:.2f}->{hg_full[-1]:.2f}, disc {ha_full[0]:.2f}->{ha_full[-1]:.2f})")
print(f"  w_expert={np.round(W_EXPERT,2)}  w_bc={np.round(W_BC_full,2)}  w_gail={np.round(W_GAIL_full,2)}")
print("GAIL images written.")
