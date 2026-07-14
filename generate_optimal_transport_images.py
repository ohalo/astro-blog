#!/usr/bin/env python3
"""
为文章「最优传输(Wasserstein)资产配置：用 Earth Mover's Distance 把分布差异变成权重」
生成真实配图 + 计算正文引用的所有关键数字。

机制（自洽合成，仅用于演示；落地见文末路径）：
  * 5 个资产的日收益分布当作各自的「签名」。其中 asset0 是质量标杆（最高 Sharpe），
    asset4 在样本内被人为注入一串正向外生跳点（幸运的肥尾），样本外不再出现。
  * 用 1-Wasserstein(W1, Earth Mover's Distance) 度量每个资产收益分布与「目标分布」
    （asset0 的样本内分布）的形状距离；W1 比较的是整个分布的形状（位置/尺度/偏度/尾部），
    所以 asset4 那个只在样本内出现的双峰/肥尾会被判定「离目标很远」。
  * 软分配：w_i = softmax(−β·W1_i)。离目标越近权重越大；β 控制集中度。
  * 对比四种分配在样本外(OOS)的表现：OT 分配、等权、均值贪婪(按样本内均值)、标杆独仓。
  * 关键论点：均值贪婪把权重压到样本内幸运的 asset4，OOS 崩；OT 看整个分布，避开它。
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
D = os.path.join(BASE, "optimal-transport-portfolio")
os.makedirs(D, exist_ok=True)

C = {"ot": "#4C72B0", "mean": "#C44E52", "equal": "#999999",
     "target": "#55A868", "grid": "#DDDDDD", "warn": "#DD8452", "calm": "#8172B3"}

rng = np.random.default_rng(20260714)
T = 1000
INS = 500
N = 5
names = ["质量标杆", "低波稳健", "均衡", "低收益", "样本内肥尾(陷阱)"]
mu = np.array([0.0007, 0.0003, 0.0004, 0.0002, 0.0003])
sig = np.array([0.0080, 0.0100, 0.0120, 0.0090, 0.0110])

R = np.zeros((T, N))
for i in range(N):
    if i == 4:
        # 陷阱资产：样本内被注入一串正向外生跳点（幸运肥尾），样本外消失
        R[:, i] = rng.normal(-0.0002, 0.0090, T)
    else:
        R[:, i] = rng.normal(mu[i], sig[i], T)
# 仅在样本内给 asset4 注入正向外生跳点（幸运肥尾），样本外不再有
spike_prob = 0.10
spike_mask = (np.arange(T) < INS) & (rng.random(T) < spike_prob)
R[spike_mask, 4] += rng.normal(0.060, 0.020, spike_mask.sum())

Rin = R[:INS]
Roos = R[INS:]

def wasserstein1(x, y):
    x = np.sort(x); y = np.sort(y)
    n = max(len(x), len(y))
    xs = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)
    ys = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(y)), y)
    return float(np.mean(np.abs(xs - ys)))

def soft_alloc(w, beta):
    z = -beta * np.asarray(w, float)
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()

def softmax(x):
    z = np.asarray(x, float); z -= z.max()
    e = np.exp(z); return e / e.sum()

def metrics(ret):
    r = np.asarray(ret, float)
    ann = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() / (r.std() + 1e-12) * np.sqrt(252)
    cvar5 = -np.percentile(r, 5) * 252
    # 最大回撤
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    mdd = float(((eq - peak) / peak).min())
    return ann, vol, sharpe, cvar5, mdd

# 目标分布 = 标杆(asset0) 样本内分布
target = Rin[:, 0]
W1 = np.array([wasserstein1(Rin[:, i], target) for i in range(N)])
beta = 300.0
w_ot = soft_alloc(W1, beta)
w_mean = softmax(Rin.mean(0) * 4000.0)          # 均值贪婪
w_equal = np.ones(N) / N
best = int(np.argmax(Rin.mean(0) / Rin.std(0))) # 样本内 Sharpe 最优（恰好是陷阱 asset4）
w_target = np.zeros(N); w_target[best] = 1.0

# OOS 表现
strat = {"OT 分配": w_ot, "等权": w_equal, "均值贪婪": w_mean, "样本内冠军独仓": w_target}
oos_ret = {k: Roos @ w for k, w in strat.items()}
oos_met = {k: metrics(oos_ret[k]) for k in strat}

# 样本内各资产特征
ins_mean = Rin.mean(0)
ins_sr = Rin.mean(0) / Rin.std(0) * np.sqrt(252)
oos_sr_simple = Roos.mean(0) / Roos.std(0) * np.sqrt(252)

print("=" * 70)
print("最优传输资产配置 关键数字 (seed 20260714)")
print("=" * 70)
print(f"样本内均值(日): {np.round(ins_mean,5)}")
print(f"样本内 Sharpe:   {np.round(ins_sr,2)}")
print(f"样本外 Sharpe(单资产): {np.round(oos_sr_simple,2)}")
print(f"\nW1 距离(对标杆): {np.round(W1,5)}")
print(f"OT 权重:   {np.round(w_ot,3)}")
print(f"均值贪婪权重: {np.round(w_mean,3)}")
print(f"等权权重:   {np.round(w_equal,3)}")
print(f"样本内冠军独仓: 资产{best}")
print("\nOOS 指标 (年化收益 / 年化波动 / Sharpe / 年化CVaR5% / 最大回撤):")
for k in strat:
    a, v, s, c, m = oos_met[k]
    print(f"  {k:8s}  ann={a:+.3f} vol={v:.3f} Sharpe={s:+.3f} CVaR5={c:.3f} MDD={m:+.3f}")

# ----------------------------------------------------------------------------
# 图 1：资产间两两 W1 距离矩阵（看谁和谁分布相近）
# ----------------------------------------------------------------------------
M = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        M[i, j] = wasserstein1(Rin[:, i], Rin[:, j])
fig, ax = plt.subplots(figsize=(7.6, 6.4))
im = ax.imshow(M, cmap="viridis")
ax.set_xticks(range(N)); ax.set_yticks(range(N))
ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
ax.set_yticklabels(names, fontsize=8)
ax.set_title("资产间两两 1-Wasserstein 距离：谁和谁收益分布更像")
for i in range(N):
    for j in range(N):
        ax.text(j, i, f"{M[i,j]*1000:.1f}", ha="center", va="center",
                color="white" if M[i, j] > M.mean() else "black", fontsize=7)
cbar = fig.colorbar(im, ax=ax); cbar.set_label("W1 (×10⁻³ 日收益)")
plt.tight_layout(); plt.savefig(os.path.join(D, "w1_matrix.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 图 2：三种分配权重对比（看清 OT vs 均值贪婪 的分歧）
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.4))
xpos = np.arange(N); w = 0.26
ax.bar(xpos - w, w_ot, w, color=C["ot"], label="OT 分配(Wasserstein)")
ax.bar(xpos, w_mean, w, color=C["mean"], label="均值贪婪(样本内均值)")
ax.bar(xpos + w, w_equal, w, color=C["equal"], label="等权")
ax.set_xticks(xpos); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("权重")
ax.set_title("三种分配：OT 锁定标杆，均值贪婪被样本内肥尾(asset4)骗走权重")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(D, "alloc_compare.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 图 3：分布叠加 —— 标杆 / 陷阱(asset4) / OT 组合 OOS 收益分布
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.2))
bins = 60
ax.hist(target * 100, bins=bins, density=True, alpha=0.55, color=C["target"],
        label="标杆 asset0 分布")
ax.hist(R[:, 4] * 100, bins=bins, density=True, alpha=0.45, color=C["warn"],
        label="陷阱 asset4 分布(含样本内肥尾)")
ax.hist(oos_ret["OT 分配"] * 100, bins=bins, density=True, alpha=0.6,
        histtype="step", lw=2.2, color=C["ot"], label="OT 组合 OOS 收益")
ax.set_xlabel("日收益 (%)")
ax.set_ylabel("密度")
ax.set_title("分布叠加：OT 组合 OOS 收益贴着标杆，避开了 asset4 的肥尾")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "dist_overlay.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 图 4：OOS 累计净值曲线（四种策略）
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.2))
for k, c in [("OT 分配", C["ot"]), ("等权", C["equal"]),
             ("均值贪婪", C["mean"]), ("样本内冠军独仓", C["target"])]:
    eq = np.cumprod(1 + oos_ret[k])
    ax.plot(eq, color=c, lw=1.8, label=f"{k} (Sharpe={oos_met[k][2]:+.2f})")
ax.set_xlabel("样本外交易日")
ax.set_ylabel("累计净值 (起始=1)")
ax.set_title("样本外累计净值：OT 分配接近标杆，均值贪婪被陷阱拖垮")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "oos_equity.png"), dpi=130); plt.close()

print("done ->", D)
