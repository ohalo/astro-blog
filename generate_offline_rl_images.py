#!/usr/bin/env python3
"""
为文章「深度强化学习的离线批处理(Offline RL)在交易中的应用」(offline-rl-trading)
生成真实配图（matplotlib，非占位图，全部可复现）。

核心对照（在固定历史数据集上训练，不与环境交互）：
  - Oracle：已知最优策略（上界）；
  - 行为策略(Behavior)：生成数据集的策略，本身就不完美；
  - 行为克隆(BC)：监督学数据集里的动作，把策略锁在「数据支撑集」内，安全；
  - 朴素离线 Q 学习：用一个灵活函数直接回归 Q，再贪心部署 —— 因分布偏移
    给「数据从未支持的动作」虚高估值，策略幻想，收益崩塌；
  - 保守 Q 学习(CQL)：在 Q 上外加 logsumexp 惩罚以压低分布外(OOD)动作的高估
    （本演示为浅层线性模型，局部惩罚受共享权重限制，故收益与朴素法相近；
     生产环境需高容量 Q 网 + 独立策略提取才显威力 —— 文中诚实说明）。

用 1-D 状态(归一化动量) + 3 个离散动作(空仓/多/空)的可解析环境演示，
真实 Q* 已知，便于诚实度量「未支持动作的高估」与「真实收益」。
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
D = os.path.join(BASE, "offline-rl-trading")
os.makedirs(D, exist_ok=True)

rng = np.random.default_rng(20260711)

# ---------------- 环境与真实 Q* ----------------
def Qstar(x):
    z = np.sin(3.0 * x)
    q0 = np.zeros_like(x)
    q1 = 0.6 * np.maximum(z, 0.0) + 0.1      # 多：动量正向时好
    q2 = 0.6 * np.maximum(-z, 0.0) + 0.1     # 空：动量负向时好
    return np.stack([q0, q1, q2], axis=-1)   # (n,3)

# ---------------- 特征(多项式 + RBF) ----------------
centers = np.linspace(-1.2, 1.2, 14)
sigma = 0.18
def basis(x):
    x = np.atleast_1d(x)
    poly = np.stack([np.ones_like(x), x, x*x, x*x*x], axis=-1)
    rbf = np.exp(-((x[:, None] - centers[None, :]) ** 2) / (2 * sigma**2))
    return np.concatenate([poly, rbf], axis=-1)

nb = 4 + 14
na = 3

def build_Z(X):
    B = basis(X)
    Z = np.zeros((X.shape[0], na * nb))
    for a in range(na):
        Z[:, a*nb:(a+1)*nb] = B
    return Z, B

def Qmat_from_theta(theta, B):
    return B @ theta.reshape(na, nb).T

# ---------------- 偏置行为策略（制造分布偏移 / 伪相关） ----------------
# 多(a=1)只在 x>0.35 出现；空(a=2)只在 x<-0.35 出现；中间永远空仓。
# 于是「多=高收益」的伪相关被记入数据；在 x<-0.35（多其实很差）等区域，
# 多这一动作从未被采样 -> 灵活模型会把「多很好」外推过去 -> 高估。
def sample_behavior(x):
    a = np.where(x > 0.35, 1, np.where(x < -0.35, 2, 0))
    m = rng.random(x.shape[0]) < 0.15
    a[m] = rng.integers(0, na, size=m.sum())
    return a

# 覆盖区域：缺口 [-0.2, 0.2] 完全没有状态数据（协变量偏移）
def sample_x_covered(m):
    left = rng.uniform(-1.0, -0.2, m // 2)
    right = rng.uniform(0.2, 1.0, m - m // 2)
    return np.concatenate([left, right])

M = 2500
X = sample_x_covered(M)
A = sample_behavior(X)
R = Qstar(X)[np.arange(M), A] + rng.normal(0.0, 0.02, M)
Z, B = build_Z(X)

# ---------------- 朴素离线 Q 学习（共享线性模型，监督回归 + 贪心部署） ----------------
lam = 1e-2
theta_naive = np.linalg.solve(Z.T @ Z + lam * np.eye(na*nb), Z.T @ R)

# ---------------- 行为克隆(BC) ----------------
W = np.zeros((nb, na))
lr_bc = 0.05
for _ in range(2000):
    logits = B @ W
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits); prob = e / e.sum(axis=1, keepdims=True)
    g = B.T @ (prob - np.eye(na)[A]) / M
    W -= lr_bc * g

# ---------------- 保守 Q 学习(CQL)：共享模型 + logsumexp 惩罚（稳定版） ----------------
def softmax_rows(m):
    m = m - m.max(axis=1, keepdims=True)
    e = np.exp(m)
    return e / e.sum(axis=1, keepdims=True)

theta_cql = theta_naive.copy()
lr = 0.0015
alpha = 0.6
lam2 = 1e-3
steps = 2500
clip = 0.8
for _ in range(steps):
    g = 2 * Z.T @ (Z @ theta_cql - R) + 2 * lam2 * theta_cql
    p = softmax_rows(Qmat_from_theta(theta_cql, B))
    g += (alpha / M) * np.concatenate([B.T @ p[:, k] for k in range(na)])
    nrm = np.linalg.norm(g)
    if nrm > clip:
        g = g * clip / nrm
    theta_cql -= lr * g

# ---------------- 真实收益评估（在完整 [-1,1] 上 rollout 贪心策略） ----------------
Xt = np.linspace(-1.0, 1.0, 400)
Bt = basis(Xt)
Qstar_t = Qstar(Xt)
oracle_a = np.argmax(Qstar_t, axis=1)
oracle_ret = Qstar_t[np.arange(400), oracle_a].mean()

Ab = sample_behavior(Xt)
behavior_ret = Qstar_t[np.arange(400), Ab].mean()

bc_a = np.argmax(Bt @ W, axis=1)
bc_ret = Qstar_t[np.arange(400), bc_a].mean()

naive_a = np.argmax(Qmat_from_theta(theta_naive, Bt), axis=1)
naive_ret = Qstar_t[np.arange(400), naive_a].mean()

cql_a = np.argmax(Qmat_from_theta(theta_cql, Bt), axis=1)
cql_ret = Qstar_t[np.arange(400), cql_a].mean()

# ---------------- 未支持动作的高估（朴素离线 Q 把没见过的动作估高了） ----------------
# 下尾 x<-0.35：多(a=1)从未被采样 -> 看朴素 Q 给多赋了多少值，相对真实值
low_mask = Xt < -0.35
naive_long_low = Qmat_from_theta(theta_naive, basis(Xt[low_mask]))[:, 1].mean()
true_long_low = Qstar_t[low_mask][:, 1].mean()
# 上尾 x>0.35：空(a=2)从未被采样
high_mask = Xt > 0.35
naive_short_high = Qmat_from_theta(theta_naive, basis(Xt[high_mask]))[:, 2].mean()
true_short_high = Qstar_t[high_mask][:, 2].mean()

# ================= 图 1：策略真实收益对比 =================
fig, ax = plt.subplots(figsize=(8.2, 5.0))
names = ["Oracle(最优)", "行为策略", "行为克隆 BC", "朴素离线 Q", "保守 Q(CQL)"]
vals = [oracle_ret*100, behavior_ret*100, bc_ret*100, naive_ret*100, cql_ret*100]
cols = ["#333", "#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
bars = ax.bar(names, vals, color=cols, alpha=0.85)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.1f}", ha="center", fontsize=9.5)
ax.set_ylabel("真实平均收益（rollout）", fontsize=11.5)
ax.set_title("离线训练后贪心部署：朴素离线 Q 因分布偏移崩塌，BC 锁在支撑集内", fontsize=11.5)
ax.grid(axis="y", alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(D, "fig_policy_return.png"), dpi=130); plt.close(fig)

# ================= 图 2：未支持动作的高估 =================
fig, ax = plt.subplots(figsize=(8.2, 5.0))
groups = ["下尾 x<-0.35（多从未出现）", "上尾 x>0.35（空从未出现）"]
xpos = np.arange(2)
w = 0.35
naive_vals = [naive_long_low*100, naive_short_high*100]
true_vals = [true_long_low*100, true_short_high*100]
ax.bar(xpos - w/2, naive_vals, w, color="#d62728", alpha=0.85, label="朴素离线 Q 赋值")
ax.bar(xpos + w/2, true_vals, w, color="#2ca02c", alpha=0.85, label="真实 Q*")
for i, v in enumerate(naive_vals):
    ax.text(i - w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
for i, v in enumerate(true_vals):
    ax.text(i + w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
ax.set_xticks(xpos); ax.set_xticklabels(groups, fontsize=10)
ax.set_ylabel("动作 Q 值", fontsize=11.5)
ax.set_title("分布外动作被虚高赋值：朴素离线 Q 给「没见过」的动作估高了 ~6pp", fontsize=11.5)
ax.legend(fontsize=9.5); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "fig_q_overestimation.png"), dpi=130); plt.close(fig)

# ================= 图 3：Q 表面：朴素 vs 真实（下尾多/上尾空） =================
xs = Xt
qn = Qmat_from_theta(theta_naive, Bt)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.6))
a1.plot(xs, qn[:, 1], "-", color="#d62728", lw=2, label="朴素离线 Q(多)")
a1.plot(xs, Qstar_t[:, 1], "--", color="#333", lw=1.8, label="真实 Q*(多)")
a1.axvspan(-1.0, -0.35, color="#888", alpha=0.08)
a1.text(-0.67, a1.get_ylim()[1]*0.5, "多从未出现\n被外推虚高", ha="center", fontsize=8.5)
a1.set_xlabel("状态 x（归一化动量）", fontsize=11); a1.set_ylabel("Q(多)", fontsize=11)
a1.set_title("下尾：多被虚高", fontsize=11.5); a1.legend(fontsize=8.5); a1.grid(alpha=0.3)

a2.plot(xs, qn[:, 2], "-", color="#d62728", lw=2, label="朴素离线 Q(空)")
a2.plot(xs, Qstar_t[:, 2], "--", color="#333", lw=1.8, label="真实 Q*(空)")
a2.axvspan(0.35, 1.0, color="#888", alpha=0.08)
a2.text(0.67, a2.get_ylim()[1]*0.5, "空从未出现\n被外推虚高", ha="center", fontsize=8.5)
a2.set_xlabel("状态 x（归一化动量）", fontsize=11); a2.set_ylabel("Q(空)", fontsize=11)
a2.set_title("上尾：空被虚高", fontsize=11.5); a2.legend(fontsize=8.5); a2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "fig_q_surface.png"), dpi=130); plt.close(fig)

# 保存数值
with open(os.path.join(D, "_metrics.txt"), "w") as f:
    f.write(f"M={M} nb={nb} na={na}\n")
    f.write(f"oracle_ret={oracle_ret:.4f}\n")
    f.write(f"behavior_ret={behavior_ret:.4f}\n")
    f.write(f"bc_ret={bc_ret:.4f}\n")
    f.write(f"naive_ret={naive_ret:.4f}\n")
    f.write(f"cql_ret={cql_ret:.4f}\n")
    f.write(f"naive_long_low={naive_long_low:.4f}\n")
    f.write(f"true_long_low={true_long_low:.4f}\n")
    f.write(f"naive_short_high={naive_short_high:.4f}\n")
    f.write(f"true_short_high={true_short_high:.4f}\n")

print("✅ offline-rl 配图生成完成")
print(f"oracle={oracle_ret*100:.2f} behavior={behavior_ret*100:.2f} bc={bc_ret*100:.2f} naive={naive_ret*100:.2f} cql={cql_ret*100:.2f}")
print(f"下尾多: 朴素={naive_long_low*100:.2f} 真实={true_long_low*100:.2f} (高估 {(naive_long_low-true_long_low)*100:.2f}pp)")
print(f"上尾空: 朴素={naive_short_high*100:.2f} 真实={true_short_high*100:.2f} (高估 {(naive_short_high-true_short_high)*100:.2f}pp)")
