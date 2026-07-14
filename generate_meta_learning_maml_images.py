#!/usr/bin/env python3
"""
为文章「元学习(MAML)跨市场快速适应：让策略学会『学会』」(meta-learning-maml)
生成真实配图 + 计算正文引用的所有关键数字。

机制（自洽合成，仅用于演示算法；落地见文末路径）：
  * 每个「市场/regime」是一个任务 task：因子->收益 的线性映射 y = X·θ_task + 噪声，
    不同任务的 θ_task 从共同分布 N(θ0, τ²I) 采样（任务间共享结构 + 各自差异）。
  * MAML 学一个「元初始化」θ_meta，使得在任一新任务上，只用 K 个样本做 1~few 步
    梯度下降(inner loop)，就能快速适应。外层(outer loop)对适应后的损失做元梯度更新。
  * 对比三条基线：
      - Scratch: 每个新任务从随机初始化只用 K 样本训练（学不动）
      - Pooled : 把所有任务数据混起来训一个全局模型（忽略任务差异）
      - MAML   : 学到的元初始化 + K 样本快速适应
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
D = os.path.join(BASE, "meta-learning-maml")
os.makedirs(D, exist_ok=True)

C = {"maml": "#C44E52", "pool": "#4C72B0", "scratch": "#999999", "oracle": "#8172B3",
     "grid": "#DDDDDD", "warn": "#DD8452", "calm": "#55A868"}

# ----------------------------------------------------------------------------
# 任务分布：每个任务是一个因子->收益 线性模型
# ----------------------------------------------------------------------------
P = 5             # 因子数
THETA0 = np.array([0.8, -0.5, 0.3, 0.6, -0.4])   # 任务共享中心
TAU = 0.5         # 任务间 θ 差异强度
SIG_Y = 1.0       # 观测噪声
K = 10            # 适应用样本数（few-shot）
Q = 200           # 每任务评估样本数


def sample_task(rng):
    theta = THETA0 + rng.normal(0, TAU, P)
    return theta


def gen_data(theta, n, rng):
    X = rng.normal(0, 1, (n, P))
    y = X @ theta + rng.normal(0, SIG_Y, n)
    return X, y


def mse(theta, X, y):
    r = X @ theta - y
    return float((r ** 2).mean())


def grad(theta, X, y):
    r = X @ theta - y
    return 2.0 / len(y) * (X.T @ r)


# ----------------------------------------------------------------------------
# MAML 训练（回归，闭式内层用梯度步；外层元梯度）
# ----------------------------------------------------------------------------
def train_maml(n_tasks=2000, inner_lr=0.05, outer_lr=0.02, inner_steps=1,
               K=K, seed=0):
    rng = np.random.default_rng(seed)
    theta_meta = np.zeros(P)
    hist = []
    for it in range(n_tasks):
        theta_task = sample_task(rng)
        # support（适应集）与 query（评估集）
        Xs, ys = gen_data(theta_task, K, rng)
        Xq, yq = gen_data(theta_task, K, rng)
        # inner loop：从 theta_meta 出发几步梯度下降
        phi = theta_meta.copy()
        for _ in range(inner_steps):
            phi = phi - inner_lr * grad(phi, Xs, ys)
        # outer loop：对 query 损失关于 theta_meta 的元梯度
        # d L_q(phi)/d theta_meta = (I - inner_lr*H_s) @ grad_q(phi)
        gq = grad(phi, Xq, yq)
        # 一阶近似 Hessian（H_s = 2/K X_s^T X_s）
        Hs = 2.0 / K * (Xs.T @ Xs)
        meta_grad = (np.eye(P) - inner_lr * Hs) @ gq
        theta_meta = theta_meta - outer_lr * meta_grad
        if it % 20 == 0:
            hist.append(np.linalg.norm(theta_meta - THETA0))
    return theta_meta, hist


def train_pooled(n_tasks=2000, K=K, seed=1):
    """把所有任务的 support 数据混起来 OLS。"""
    rng = np.random.default_rng(seed)
    Xall, yall = [], []
    for _ in range(n_tasks):
        theta_task = sample_task(rng)
        X, y = gen_data(theta_task, K, rng)
        Xall.append(X); yall.append(y)
    X = np.vstack(Xall); y = np.concatenate(yall)
    theta = np.linalg.lstsq(X, y, rcond=None)[0]
    return theta


theta_meta, meta_hist = train_maml()
theta_pool = train_pooled()

print("=" * 60)
print("MAML 元初始化 vs Pooled 全局模型")
print("=" * 60)
print("θ0 (任务共享中心) :", np.round(THETA0, 3))
print("θ_meta (MAML)     :", np.round(theta_meta, 3), " ‖θ_meta-θ0‖=", round(np.linalg.norm(theta_meta - THETA0), 3))
print("θ_pool (Pooled)   :", np.round(theta_pool, 3), " ‖θ_pool-θ0‖=", round(np.linalg.norm(theta_pool - THETA0), 3))


# ----------------------------------------------------------------------------
# 评估：新任务上 few-shot 适应
# ----------------------------------------------------------------------------
def adapt(theta_init, Xs, ys, lr=0.05, steps=1):
    phi = theta_init.copy()
    for _ in range(steps):
        phi = phi - lr * grad(phi, Xs, ys)
    return phi


def evaluate(theta_meta, theta_pool, n_eval=400, K=K, steps=1, seed=999):
    rng = np.random.default_rng(seed)
    mse_maml, mse_pool, mse_scratch, mse_oracle = [], [], [], []
    for _ in range(n_eval):
        theta_task = sample_task(rng)
        Xs, ys = gen_data(theta_task, K, rng)
        Xq, yq = gen_data(theta_task, Q, rng)
        # MAML：元初始化 + K 样本适应
        phi_m = adapt(theta_meta, Xs, ys, steps=steps)
        mse_maml.append(mse(phi_m, Xq, yq))
        # Pooled：直接用全局模型（不适应）
        mse_pool.append(mse(theta_pool, Xq, yq))
        # Scratch：随机初始化 + K 样本适应（多步）
        phi_s = adapt(rng.normal(0, 0.1, P), Xs, ys, steps=steps)
        mse_scratch.append(mse(phi_s, Xq, yq))
        # Oracle：用真 θ_task
        mse_oracle.append(mse(theta_task, Xq, yq))
    return (np.array(mse_maml), np.array(mse_pool),
            np.array(mse_scratch), np.array(mse_oracle))


m_maml, m_pool, m_scratch, m_oracle = evaluate(theta_meta, theta_pool)
print("-" * 60)
print(f"400 个新任务 (K={K} shot, 1 步适应) 平均 query MSE:")
print(f"  MAML    : {m_maml.mean():.3f}")
print(f"  Pooled  : {m_pool.mean():.3f}")
print(f"  Scratch : {m_scratch.mean():.3f}")
print(f"  Oracle  : {m_oracle.mean():.3f} (真 θ 下界)")
print(f"  MAML 相对 Pooled 降 MSE: {(1 - m_maml.mean()/m_pool.mean())*100:.1f}%")
print(f"  MAML 相对 Scratch 降 MSE: {(1 - m_maml.mean()/m_scratch.mean())*100:.1f}%")

# ----------------------------------------------------------------------------
# 图 1：元训练收敛（θ_meta -> θ0）
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(np.arange(len(meta_hist)) * 20, meta_hist, color=C["maml"], lw=2.0)
ax.axhline(np.linalg.norm(theta_pool - THETA0), color=C["pool"], ls="--", lw=1.6,
           label=f"Pooled ‖θ-θ0‖={np.linalg.norm(theta_pool-THETA0):.3f}")
ax.set_xlabel("元训练任务数"); ax.set_ylabel("‖θ_meta − θ0‖（到任务中心的距离）")
ax.set_title("MAML 元训练收敛：元初始化收敛到任务分布的『共享中心』")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "maml_convergence.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 图 2：few-shot MSE 对比（箱线）
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.2))
data = [m_maml, m_pool, m_scratch, m_oracle]
labels = [f"MAML\n{m_maml.mean():.2f}", f"Pooled\n{m_pool.mean():.2f}",
          f"Scratch\n{m_scratch.mean():.2f}", f"Oracle\n{m_oracle.mean():.2f}"]
bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
for patch, col in zip(bp["boxes"], [C["maml"], C["pool"], C["scratch"], C["oracle"]]):
    patch.set_facecolor(col); patch.set_alpha(0.55)
ax.set_ylabel("新任务 query MSE（越低越好）")
ax.set_title(f"{K}-shot 适应：MAML 快速逼近 Oracle，碾压 Scratch 与 Pooled")
ax.grid(True, axis="y", color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "maml_fewshot.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 图 3：MSE 随 shot 数 K 变化
# ----------------------------------------------------------------------------
K_grid = [2, 3, 5, 8, 10, 15, 20, 40]
maml_k, pool_k, scr_k = [], [], []
for k in K_grid:
    mm, mp, ms, mo = evaluate(theta_meta, theta_pool, n_eval=300, K=k, steps=1, seed=77)
    maml_k.append(mm.mean()); pool_k.append(mp.mean()); scr_k.append(ms.mean())
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(K_grid, maml_k, "o-", color=C["maml"], lw=2.0, label="MAML")
ax.plot(K_grid, pool_k, "s--", color=C["pool"], lw=1.8, label="Pooled（全局，不适应）")
ax.plot(K_grid, scr_k, "^--", color=C["scratch"], lw=1.8, label="Scratch（随机初始化）")
ax.axhline(m_oracle.mean(), color=C["oracle"], ls=":", lw=1.6, label="Oracle 下界")
ax.set_xlabel("适应样本数 K（shot）"); ax.set_ylabel("新任务 query MSE")
ax.set_title("样本效率：MAML 在极少样本(K小)时优势最大")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "maml_shots.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 图 4：适应轨迹（单个新任务，MSE 随内层步数下降）
# ----------------------------------------------------------------------------
rng = np.random.default_rng(2026)
theta_task = sample_task(rng)
Xs, ys = gen_data(theta_task, K, rng)
Xq, yq = gen_data(theta_task, Q, rng)
steps = 8
tr_maml = [mse(theta_meta, Xq, yq)]
tr_scr = [mse(np.zeros(P), Xq, yq)]
phi_m = theta_meta.copy(); phi_s = np.zeros(P)
for _ in range(steps):
    phi_m = phi_m - 0.05 * grad(phi_m, Xs, ys); tr_maml.append(mse(phi_m, Xq, yq))
    phi_s = phi_s - 0.05 * grad(phi_s, Xs, ys); tr_scr.append(mse(phi_s, Xq, yq))
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(range(steps + 1), tr_maml, "o-", color=C["maml"], lw=2.0, label="从 MAML 元初始化适应")
ax.plot(range(steps + 1), tr_scr, "s--", color=C["scratch"], lw=1.8, label="从零初始化适应")
ax.axhline(mse(theta_task, Xq, yq), color=C["oracle"], ls=":", lw=1.6, label="Oracle（真 θ）")
ax.set_xlabel("内层梯度步数"); ax.set_ylabel("新任务 query MSE")
ax.set_title("单任务适应轨迹：MAML 一步就贴近最优，从零要走很多步")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "maml_trajectory.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 鲁棒性：多种子
# ----------------------------------------------------------------------------
win = 0; gaps = []
for sd in range(10):
    tm, _ = train_maml(n_tasks=1500, seed=sd)
    tp = train_pooled(n_tasks=1500, seed=sd + 100)
    mm, mp, _, _ = evaluate(tm, tp, n_eval=300, seed=sd + 500)
    gaps.append((mp.mean() - mm.mean()) / mp.mean() * 100)
    if mm.mean() < mp.mean():
        win += 1
print("-" * 60)
print(f"10 种子: MAML 相对 Pooled 平均降 MSE {np.mean(gaps):.1f}%±{np.std(gaps):.1f}%  "
      f"MAML<Pooled: {win}/10")
print("done ->", D)
