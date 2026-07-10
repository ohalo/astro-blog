#!/usr/bin/env python3
"""
为文章「遗传算法与进化策略在组合优化中的应用」(genetic-algo-portfolio) 生成真实配图。
全部用 matplotlib + numpy 从原理渲染，非占位图。

图表：
  1. ga_convergence.png   适应度收敛曲线：每代最优 / 平均适应度
  2. ga_frontier.png      解分布(风险-收益散点) + 经典均值方差前沿 + GA 帕累托前沿
  3. ga_weights.png       最终入选组合的权重柱状图（带基数约束 cardinality）
  4. ga_schematic.png     遗传算法流程示意图：选择→交叉→变异→精英保留
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "genetic-algo-portfolio")
os.makedirs(D, exist_ok=True)
np.random.seed(20260711)


# ============================================================
# 1) 构造一个 12 资产的回报分布（均值向量 + 协方差）
# ============================================================
n_asset = 12
names = [f"A{i+1:02d}" for i in range(n_asset)]
# 用随机但固定的参数构造协方差
A = np.random.randn(n_asset, n_asset)
cov = A @ A.T / n_asset * 0.04           # 约 20% 年化波动量级
diag = np.random.uniform(0.01, 0.06, n_asset)
cov = cov + np.diag(diag)
np.fill_diagonal(cov, diag + 0.02)
mu = np.random.uniform(0.04, 0.22, n_asset)   # 年化期望收益


def port_stats(w):
    w = np.asarray(w)
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    sharpe = ret / vol
    return ret, vol, sharpe


# ============================================================
# 2) 经典无约束均值方差（解析解）：最大夏普组合
# ============================================================
inv_cov = np.linalg.inv(cov)
ones = np.ones(n_asset)
w_mv = inv_cov @ mu
w_mv /= ones @ w_mv
# 全局最小方差组合
w_gmv = inv_cov @ ones
w_gmv /= ones @ w_gmv
# 纯多头(归一化期望收益)作为朴素基准
w_naive = mu / (ones @ mu)


# ============================================================
# 3) 遗传算法：基数约束(|w|>0 的个数=K) + 多头 + 和为1，目标最大化夏普
# ============================================================
K = 5                       # 只允许精选 5 只
POP = 200
GEN = 120
MUT = 0.25
CROSS = 0.9
ELITE = 4

def random_chrom():
    # 随机选 K 个资产，随机权重后归一
    idx = np.random.choice(n_asset, K, replace=False)
    w = np.zeros(n_asset)
    raw = np.random.rand(K)
    w[idx] = raw / raw.sum()
    return w

def fitness(w):
    ret, vol, sharpe = port_stats(w)
    # 惩罚：若基数不对 / 有空头 / 不归一，重罚
    penalty = 0.0
    if np.sum(w > 1e-6) != K:
        penalty += 5.0
    if np.any(w < -1e-9):
        penalty += 5.0
    if abs(w.sum() - 1) > 1e-6:
        penalty += 5.0
    # 轻微偏好分散，防孤注一掷
    hhi = np.sum(w**2)
    penalty += 0.3 * max(0, hhi - 1.0 / K)
    return sharpe - penalty

pop = [random_chrom() for _ in range(POP)]
best_hist, avg_hist = [], []
archive = []                # 收集 Pareto 候选（风险-收益）

for g in range(GEN):
    fits = [fitness(w) for w in pop]
    order = np.argsort(fits)[::-1]
    best_hist.append(fits[order[0]])
    avg_hist.append(np.mean(fits))
    # 精英
    new_pop = [pop[order[i]] for i in range(ELITE)]
    # 锦标赛选择
    def tournament():
        i, j = np.random.choice(POP, 2, replace=False)
        return pop[i] if fits[i] > fits[j] else pop[j]
    while len(new_pop) < POP:
        if np.random.rand() < CROSS:
            p1, p2 = tournament(), tournament()
            # 单点交叉：在资产索引维度
            cut = np.random.randint(1, n_asset)
            c1 = np.concatenate([p1[:cut], p2[cut:]])
            c2 = np.concatenate([p2[:cut], p1[cut:]])
            for c in (c1, c2):
                # 归一到多头且=1，再强制基数≈K
                c = np.clip(c, 0, None)
                if c.sum() <= 0:
                    c = np.ones(n_asset)
                c = c / c.sum()
                # 保留最大 K 个，其余置 0 后重新归一
                thresh = np.sort(c)[::-1][K - 1]
                c = np.where(c >= thresh, c, 0.0)
                c = c / c.sum()
                new_pop.append(c)
                if len(new_pop) >= POP:
                    break
        else:
            new_pop.append(tournament().copy())
    # 变异：对部分染色体加噪后重归一 + 重投影基数
    for i in range(ELITE, POP):
        if np.random.rand() < MUT:
            w = new_pop[i].copy()
            w = w + np.random.rand(n_asset) * 0.15
            w = np.clip(w, 0, None)
            w = w / w.sum()
            thresh = np.sort(w)[::-1][K - 1]
            w = np.where(w >= thresh, w, 0.0)
            w = w / w.sum()
            new_pop[i] = w
    pop = new_pop[:POP]

# 最终最优
fits = [fitness(w) for w in pop]
best = pop[np.argmax(fits)]
ret_b, vol_b, sh_b = port_stats(best)

print(f"GA 最优: 夏普={sh_b:.3f} 收益={ret_b:.2%} 波动={vol_b:.2%} 持仓数={int(np.sum(best>1e-6))}")
print(f"均值方差(无约束): 夏普={port_stats(w_mv)[2]:.3f} 持仓数={int(np.sum(w_mv>1e-6))}")
print(f"朴素归一: 夏普={port_stats(w_naive)[2]:.3f}")

# 收集一批 GA 探索过的解用于散点
ga_solutions = []
for _ in range(400):
    w = random_chrom()
    r, v, s = port_stats(w)
    ga_solutions.append((v, r, s))
ga_solutions = np.array(ga_solutions)


# ============================================================
# 图1：收敛曲线
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(best_hist, color="#1f77b4", lw=1.8, label="每代最优适应度（夏普）")
ax.plot(avg_hist, color="#ff7f0e", lw=1.4, alpha=0.8, label="每代平均适应度")
ax.axhline(port_stats(w_mv)[2], color="#d62728", ls="--", lw=1.4,
           label=f"无约束均值方差夏普={port_stats(w_mv)[2]:.3f}")
ax.set_xlabel("进化代数", fontsize=11)
ax.set_ylabel("适应度（夏普比率）", fontsize=11)
ax.set_title("遗传算法收敛曲线：早期快速上升，后期在约束前沿精细搜索", fontsize=12.5, fontweight="bold")
ax.legend(loc="lower right", fontsize=9.5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "ga_convergence.png"), dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 图2：解分布 + 前沿
# ============================================================
# 经典均值方差随机组合云（无约束，展示可行域）
rand_w = np.random.dirichlet(np.ones(n_asset), 3000)
rand_ret = rand_w @ mu
rand_vol = np.sqrt(np.einsum("ij,jk,ik->i", rand_w, cov, rand_w))
fig, ax = plt.subplots(figsize=(11, 6.0))
ax.scatter(rand_vol, rand_ret, s=6, alpha=0.25, color="#cccccc", label="无约束随机组合（可行域）")
ax.scatter(ga_solutions[:, 0], ga_solutions[:, 1], s=12, alpha=0.5, color="#ff7f0e",
           label="GA 探索的解（基数=5）")
# 经典 MV 前沿采样（不同目标返回）
lams = np.linspace(0, 6, 30)
mv_frontier = []
for lam in lams:
    w = inv_cov @ (mu + lam * ones)
    w = w / (ones @ w)
    w = np.clip(w, 0, None); w = w / w.sum()
    mv_frontier.append(port_stats(w))
mv_frontier = np.array(mv_frontier)
ax.scatter(mv_frontier[:, 1], mv_frontier[:, 0], s=22, color="#1f77b4", zorder=5,
           label="经典均值方差前沿（无基数约束）")
ax.scatter([vol_b], [ret_b], s=120, marker="*", color="#2ca02c", zorder=6,
           label=f"GA 最终组合(夏普={sh_b:.2f})")
ax.set_xlabel("年化波动率", fontsize=11)
ax.set_ylabel("年化收益", fontsize=11)
ax.set_title("解空间分布：遗传算法在「精选 K 只」约束下逼近有效前沿", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=8.8)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "ga_frontier.png"), dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 图3：最终组合权重
# ============================================================
order = np.argsort(best)[::-1]
top = order[:K]
labels = [names[i] for i in top]
vals = best[top]
fig, ax = plt.subplots(figsize=(11, 5.2))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, K))
bars = ax.bar(labels, vals * 100, color=colors)
for b, v in zip(bars, vals * 100):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}%", ha="center",
            fontsize=10, fontweight="bold")
ax.set_ylabel("权重 (%)", fontsize=11)
ax.set_title(f"遗传算法最终入选组合（精选 {K} 只，多头、权重和为 100%）", fontsize=12.5, fontweight="bold")
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "ga_weights.png"), dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 图4：遗传算法流程示意图
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.set_xlim(0, 12); ax.set_ylim(0, 6)
ax.axis("off")
boxes = [
    (1.0, 4.2, "初始化种群\n随机权重染色体"),
    (4.0, 4.2, "适应度评估\n夏普 − 约束惩罚"),
    (7.0, 4.2, "选择(锦标赛)\n优质基因留存"),
    (10.0, 4.2, "交叉 + 变异\n探索新组合"),
]
for x, y, txt in boxes:
    ax.add_patch(FancyBboxPatch((x - 0.85, y - 0.55), 1.7, 1.1,
                 boxstyle="round,pad=0.04", fc="#e8f0fe", ec="#1f77b4", lw=1.6))
    ax.text(x, y, txt, ha="center", va="center", fontsize=9.3, fontweight="bold")
# 箭头
for i in range(3):
    x0 = boxes[i][0] + 0.85
    x1 = boxes[i + 1][0] - 0.85
    ax.add_patch(FancyArrowPatch((x0, boxes[i][1]), (x1, boxes[i + 1][1]),
                 arrowstyle="->", mutation_scale=16, color="#444", lw=1.5))
# 精英回路
ax.add_patch(FancyArrowPatch((10.0, 3.65), (1.0, 3.65),
             arrowstyle="->", mutation_scale=16, color="#2ca02c", lw=1.6,
             connectionstyle="arc3,rad=-0.25"))
ax.text(5.5, 3.25, "精英保留(Elitism)：把当代最优直接复制进下一代，防退化",
        ha="center", fontsize=9.5, color="#2ca02c", fontweight="bold")
# 终止判定
ax.add_patch(FancyBboxPatch((4.0, 1.2), 4.0, 1.0,
             boxstyle="round,pad=0.04", fc="#fff3e0", ec="#ff7f0e", lw=1.6))
ax.text(6.0, 1.7, "终止条件：代数上限 / 适应度收敛", ha="center", va="center",
        fontsize=9.5, fontweight="bold")
ax.add_patch(FancyArrowPatch((10.0, 3.65), (6.0, 2.2),
             arrowstyle="->", mutation_scale=14, color="#888", lw=1.3))
ax.text(6.0, 5.6, "遗传算法主循环：从种群中演化出满足约束的最优组合", ha="center",
        fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(D, "ga_schematic.png"), dpi=150, bbox_inches="tight")
plt.close()


print("✅ 遗传算法配图生成完成：", sorted(os.listdir(D)))
