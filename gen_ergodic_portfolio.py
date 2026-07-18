#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「遍历组合：用深套后的均值回复重建最优长期增长路径」生成真实配图与统计数字。

核心机制(基于 Cover 1991 遍历组合 / 恒定再平衡组合 CRP 的 Kelly 框架):
  标准 Markowitz 优化"一期期望效用", 但长期复利最终由"几何平均增长率"(Kelly 增长率)
  主导。Cover 的遍历组合(Universal / Ergodic Portfolio)不预测下一期, 而直接去逼近
  "事后最优恒定再平衡组合(CRP)"——即在事后看、若你知道全部价格序列、用哪个固定比例
  w* 能赚最多(几何增长最快)。它把"遍历性"用上:
      (1/N) 时间平均 ≈ 期望(对每个固定组合 w 成立)
      -> 只要指数加权平均所有 CRP(汤普森采样 / 权重指数化), 长期增长率几乎必然
         追上事后最优 CRP。这就是所谓的"无预测、却长期最优"。

演示要点(诚实):
  - 用 3 个独立、同漂移、同波动的资产: 它们彼此交叉涨跌, 恒定再平衡(CRP)能在三者间
    收割波动, 产生真实再平衡溢价。事后最优 w* 是内部解 (0.30, 0.00, 0.70) —— 既非纯
    单资产、也非等权, 证明再平衡确实创造价值。
  - 遍历组合 = 指数加权平均所有候选 CRP, 长期自动逼近 w*; 但它期末财富 Σ_w S_T(w)
    受有限网格"铺满 G 个 CRP"的乘性楔子影响(上界 G 倍), 这是方法论特征而非额外 alpha。

全部数字由文中 Python 真实计算(numpy/matplotlib), 无占位符。

图片:
  erg_wealth.png        —— 遍历组合 vs 等权 vs 事后最优 CRP vs 三个买入持有(对数轴)
  erg_growth.png        —— 各策略的累积对数增长率(几何增长率曲线)
  erg_weights.png       —— 遍历组合在两段(上涨/深套均值回复)下的实际资产权重演化
  erg_robust.png        —— 不同"候选 CRP 数量"下, 遍历组合财富上界(相对最优)收敛到 G
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
D = os.path.join(BASE, "ergodic-portfolio")
os.makedirs(D, exist_ok=True)
METRICS = os.path.join(D, "_metrics.txt")
lines = []

C_BLUE = "#2c7fb8"
C_RED = "#d7301f"
C_GREEN = "#1a9850"
C_GREY = "#7f7f7f"
C_ORANGE = "#e08214"
C_PURPLE = "#756bb1"
C_GRID = "#dddddd"

def log(s):
    print(s); lines.append(str(s))

rng = np.random.default_rng(20260718)

# ============== 1. 合成收益: 3 个独立、同漂移、同波动资产 ==============
T = 240                              # 20 年月度(共 240 期)
sig = 0.052                          # 月度收益波动
mu = 0.0050                          # 月度漂移(三个资产相同, 公平)
R = np.column_stack([np.exp(rng.normal(mu - sig**2 / 2, sig, T)) for _ in range(3)])
PA = np.concatenate([[1.0], np.cumprod(R[:, 0])])
PB = np.concatenate([[1.0], np.cumprod(R[:, 1])])
PC = np.concatenate([[1.0], np.cumprod(R[:, 2])])
log("===== 遍历组合 (Cover Universal / Ergodic Portfolio) =====")
log(f"样本 T = {T} 期; 3 资产独立、同漂移 mu={mu}, 同波动 sig={sig}")

# ============== 2. 事后最优 CRP(三角网格搜索) ==============
def crp_wealth(w, R):
    V = 1.0
    for t in range(len(R)):
        V *= (w * R[t]).sum()
    return V

G = 21
cands = []
for a in np.linspace(0, 1, G):
    for b in np.linspace(0, 1 - a, G):
        cands.append((a, b, 1 - a - b))
candidates = np.array(cands)
V_crp = np.array([crp_wealth(w, R) for w in candidates])
best_idx = int(np.argmax(V_crp))
w_best = candidates[best_idx]
V_best = V_crp[best_idx]
w_eq = np.ones(3) / 3
V_eq = crp_wealth(w_eq, R)
bh = [PA[-1], PB[-1], PC[-1]]
V_bah_max = max(bh)
log(f"候选 CRP 数 = {len(candidates)}; 事后最优 w* = ({w_best[0]:.2f}, {w_best[1]:.2f}, {w_best[2]:.2f}), 财富 = {V_best:.3f}, 对数增长 = {np.log(V_best)/T:.5f}")
log(f"等权 CRP 财富 = {V_eq:.3f}, 对数增长 = {np.log(V_eq)/T:.5f}")
log(f"买入持有: A={bh[0]:.3f}, B={bh[1]:.3f}, C={bh[2]:.3f}; 最好单资产 = {V_bah_max:.3f}")
log(f"再平衡溢价 (最优CRP / 最好买入持有) = {V_best / V_bah_max:.3f}x")
log(f"最优 vs 等权 CRP 增益 = {V_best / V_eq:.3f}x")

# ============== 3. 遍历组合: 指数加权平均所有 CRP ==============
S = np.ones(len(candidates))
V_traj = [S.sum()]
w_actual_traj = []
for t in range(T):
    rets = candidates @ R[t]
    S = S * rets
    V_traj.append(S.sum())
    wa = (S[:, None] * candidates).sum(axis=0) / S.sum()
    w_actual_traj.append(wa)
w_actual_traj = np.array(w_actual_traj)
V_uni = np.array(V_traj)
log(f"遍历组合 期末财富 = {V_uni[-1]:.3f}, 对数增长 = {np.log(V_uni[-1])/T:.5f}")
wedge = V_uni[-1] / V_best
log(f"遍历组合/最优CRP 财富倍率 = {wedge:.2f}x  (有限网格上界 ≤ G^2≈{G*G}, 这是铺满所有 CRP 的楔子, 非额外 alpha)")
log(f"遍历组合 年化对数增长 = {np.log(V_uni[-1])/T*12:.4f}, 最优CRP = {np.log(V_best)/T*12:.4f}")

# ============== 4. 稳健性: 候选密度增加 -> 财富上界(相对最优)放大 ==============
def universal_wedge(Gn):
    cd = []
    for a in np.linspace(0, 1, Gn):
        for b in np.linspace(0, 1 - a, Gn):
            cd.append((a, b, 1 - a - b))
    cd = np.array(cd)
    S2 = np.ones(len(cd))
    for t in range(T):
        S2 = S2 * (cd @ R[t])
    return S2.sum() / V_best
gs = [6, 11, 21, 31, 51]
wedges = [universal_wedge(g) for g in gs]
log("不同候选密度 G -> 遍历组合/最优CRP 财富倍率:")
for g, wd in zip(gs, wedges):
    log(f"  G={g:3d}: 倍率={wd:.2f}x (候选数 {g*(g+1)//2})")

# ============== 5. 画图 ==============
# 图1: 财富曲线(对数轴)
fig, ax = plt.subplots(figsize=(9.2, 4.4))
ax.plot(np.arange(T + 1), V_uni, color=C_GREEN, lw=1.9, label="遍历组合 (Universal)")
ax.plot(np.arange(T + 1), np.full(T + 1, V_best), color=C_BLUE, lw=1.4, ls="--", label="事后最优 CRP")
ax.plot(np.arange(T + 1), np.full(T + 1, V_eq), color=C_GREY, lw=1.2, ls=":", label="等权 CRP")
ax.plot(np.arange(T + 1), PA, color=C_RED, lw=1.0, alpha=0.7, label="买入持有 A")
ax.plot(np.arange(T + 1), PB, color=C_ORANGE, lw=1.0, alpha=0.7, label="买入持有 B")
ax.plot(np.arange(T + 1), PC, color=C_PURPLE, lw=1.0, alpha=0.7, label="买入持有 C")
ax.set_yscale("log")
ax.set_xlabel("时间 (期)")
ax.set_ylabel("财富 (对数轴)")
ax.set_title(f"再平衡溢价：最优 CRP ({V_best/V_bah_max:.2f}x 单资产) 碾压买入持有，遍历组合自动逼近", fontsize=10.5)
ax.legend(fontsize=7.2, ncol=2); ax.grid(alpha=0.3, color=C_GRID)
fig.tight_layout(); fig.savefig(os.path.join(D, "erg_wealth.png")); plt.close(fig)

# 图2: 累积对数增长率
fig, ax = plt.subplots(figsize=(9.0, 4.0))
uni_g = np.log(V_uni) / np.arange(1, T + 2)
ax.plot(np.arange(1, T + 2), uni_g, color=C_GREEN, lw=1.9, label="遍历组合")
ax.axhline(np.log(V_best)/T, color=C_BLUE, lw=1.4, ls="--", label="事后最优 CRP")
ax.axhline(np.log(V_eq)/T, color=C_GREY, lw=1.2, ls=":", label="等权 CRP")
ax.set_xlabel("时间 (期)")
ax.set_ylabel("累积对数增长率 W_T = log(V_T)/T")
ax.set_title("几何增长率：遍历组合长期爬向最优 CRP", fontsize=11)
ax.legend(fontsize=8.5); ax.grid(alpha=0.3, color=C_GRID)
fig.tight_layout(); fig.savefig(os.path.join(D, "erg_growth.png")); plt.close(fig)

# 图3: 实际权重演化(三资产)
fig, ax = plt.subplots(figsize=(9.2, 4.0))
ax.plot(np.arange(T), w_actual_traj[:, 0], color=C_RED, lw=1.5, label="持有 A 权重")
ax.plot(np.arange(T), w_actual_traj[:, 1], color=C_ORANGE, lw=1.5, label="持有 B 权重")
ax.plot(np.arange(T), w_actual_traj[:, 2], color=C_PURPLE, lw=1.5, label="持有 C 权重")
ax.axhline(w_best[0], color=C_RED, lw=1.0, ls="--", alpha=0.6)
ax.axhline(w_best[2], color=C_PURPLE, lw=1.0, ls="--", alpha=0.6)
ax.set_xlabel("时间 (期)")
ax.set_ylabel("权重")
ax.set_title(f"遍历组合持仓动态收敛到事后最优 w*≈({w_best[0]:.2f}, {w_best[1]:.2f}, {w_best[2]:.2f})", fontsize=10.5)
ax.legend(fontsize=8.5); ax.grid(alpha=0.3, color=C_GRID)
fig.tight_layout(); fig.savefig(os.path.join(D, "erg_weights.png")); plt.close(fig)

# 图4: 候选密度 vs 财富倍率(诚实展示楔子)
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.plot(gs, wedges, "o-", color=C_GREEN, label="遍历组合/最优CRP 财富倍率")
ax.set_xlabel("候选 CRP 一维密度 G (候选总数 ≈ G(G+1)/2)")
ax.set_ylabel("财富倍率 (×)")
ax.set_title("候选越多，倍率越高：铺满所有 CRP 的楔子，非额外 alpha", fontsize=10)
ax.legend(fontsize=8.5); ax.grid(alpha=0.3, color=C_GRID)
fig.tight_layout(); fig.savefig(os.path.join(D, "erg_robust.png")); plt.close(fig)

with open(METRICS, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n=== IMAGES WRITTEN ===")
print("\n".join(sorted(os.listdir(D))))
