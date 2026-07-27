#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 fractional-kelly-drawdown 文章配图（纯 numpy 模拟）。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/fractional-kelly-drawdown"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

# ---------------- 参数 ----------------
mu, sig = 0.10, 0.20          # 年化
f_star = mu / sig**2          # 全 Kelly 杠杆 = 2.5
g_star = mu**2 / (2 * sig**2) # 全 Kelly 增长率 = 0.125
DT = 1/252
T_YEARS = 20
T = int(252 * T_YEARS)
N_PATHS = 4000

print(f"f* = {f_star:.2f}, g* = {g_star:.4f}")

c_grid = np.array([0.10, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])

def simulate(c, n_paths, T, rng):
    """对数财富路径 + 最大回撤（几何回撤）"""
    f = c * f_star
    # GBM 对数收益：log-wealth increment = (f*mu - f^2 sig^2/2) dt + f sig sqrt(dt) Z
    drift = (f * mu - 0.5 * f**2 * sig**2) * DT
    vol = f * sig * np.sqrt(DT)
    z = rng.standard_normal((n_paths, T))
    logw = np.cumsum(drift + vol * z, axis=1)
    logw = np.hstack([np.zeros((n_paths, 1)), logw])
    running_max = np.maximum.accumulate(logw, axis=1)
    dd = 1 - np.exp(logw - running_max)   # 从峰值的几何回撤
    maxdd = dd.max(axis=1)
    min_logw = logw.min(axis=1)           # 相对初始资金的最深亏损
    loss_from_start = 1 - np.exp(min_logw)
    growth = logw[:, -1] / T_YEARS        # 年化对数增长
    return growth, maxdd, loss_from_start

results = {}
for c in c_grid:
    g, mdd, lfs = simulate(c, N_PATHS, T, rng)
    results[c] = (g, mdd, lfs)
    print(f"c={c:5.3f}  median_g={np.median(g):+.4f}  mean_g={g.mean():+.4f}  "
          f"P(loss>50%)={np.mean(lfs>0.5):.4f}  P(loss>30%)={np.mean(lfs>0.3):.4f}  "
          f"medianMDD={np.median(mdd):.3f}")

# 理论值
c_fine = np.linspace(0.05, 2.0, 200)
g_theory = g_star * (2*c_fine - c_fine**2)
def p_dd_theory(c, D):
    # P(财富曾跌破初始资金的 (1-D) 倍) = (1-D)^(2/c - 1)，无限期
    return (1 - D) ** (2.0/c - 1.0)

# ---------- 图1：增长率抛物线 ----------
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(c_fine, g_theory*100, color="#1f2a44", lw=2, label="理论：$g(c)=g^*(2c-c^2)$")
sim_g = [np.median(results[c][0])*100 for c in c_grid]
ax.scatter(c_grid, sim_g, color="#d1495b", zorder=5, s=42, label="模拟（4000 条路径中位数）")
ax.axvline(1.0, color="#888", ls="--", lw=1)
ax.axvline(0.5, color="#3a7d44", ls="--", lw=1)
ax.annotate("全 Kelly：g = 12.5%", xy=(1.0, 12.5), xytext=(1.12, 13.0),
            fontsize=10, color="#1f2a44")
ax.annotate("半 Kelly：g = 9.4%\n（保留 75% 增长）", xy=(0.5, 9.375), xytext=(0.12, 10.6),
            fontsize=10, color="#3a7d44")
ax.annotate("c=2：增长归零", xy=(2.0, 0), xytext=(1.62, 2.0),
            fontsize=10, color="#d1495b",
            arrowprops=dict(arrowstyle="->", color="#d1495b"))
ax.set_xlabel("Kelly 分数 c（实际杠杆 = c × 2.5）")
ax.set_ylabel("年化对数增长率（%）")
ax.set_title("增长率是 c 的抛物线：右半边全是纯亏损区")
ax.legend(loc="lower center")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/kelly-growth-parabola.jpg", dpi=110)
plt.close(fig)

# ---------- 图2：回撤概率 vs c ----------
fig, ax = plt.subplots(figsize=(9, 5))
for D, color in [(0.5, "#d1495b"), (0.3, "#e0a458"), (0.2, "#3a7d44")]:
    ax.plot(c_fine, p_dd_theory(c_fine, D)*100, color=color, lw=2,
            label=f"理论 P(亏掉初始资金的 {int(D*100)}%)")
    sim_p = [np.mean(results[c][2] > D)*100 for c in c_grid]
    ax.scatter(c_grid, sim_p, color=color, s=36, zorder=5)
ax.set_xlabel("Kelly 分数 c")
ax.set_ylabel("触及概率（%）")
ax.set_title("亏损概率对 c 是指数级敏感：理论线（无限期）vs 20 年模拟散点")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/kelly-drawdown-prob.jpg", dpi=110)
plt.close(fig)

# ---------- 图3：同一组冲击下的三条净值 ----------
rng2 = np.random.default_rng(7)
z = rng2.standard_normal(T)
fig, ax = plt.subplots(figsize=(10, 5))
for c, color in [(1.0, "#d1495b"), (0.5, "#1f6091"), (0.25, "#3a7d44")]:
    f = c * f_star
    logw = np.cumsum((f*mu - 0.5*f**2*sig**2)*DT + f*sig*np.sqrt(DT)*z)
    logw = np.concatenate([[0], logw])
    mdd = (1 - np.exp(logw - np.maximum.accumulate(logw))).max()
    ax.plot(np.arange(T+1)/252, np.exp(logw), color=color, lw=1.4,
            label=f"c={c}（杠杆 {c*f_star:.2f}）  最大回撤 {mdd*100:.0f}%")
    print(f"sample path c={c}: final={np.exp(logw[-1]):.2f}, maxdd={mdd:.3f}")
ax.set_yscale("log")
ax.set_xlabel("年")
ax.set_ylabel("净值（对数刻度）")
ax.set_title("同一组随机冲击：全 Kelly、半 Kelly、四分之一 Kelly")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(f"{OUT}/kelly-sample-paths.jpg", dpi=110)
plt.close(fig)

# ---------- 图4：增长-回撤交换前沿 ----------
fig, ax = plt.subplots(figsize=(9, 5.5))
p50 = p_dd_theory(c_fine[c_fine <= 1.0], 0.5)*100
g50 = g_star*(2*c_fine[c_fine <= 1.0] - c_fine[c_fine <= 1.0]**2)*100
ax.plot(p50, g50, color="#1f2a44", lw=2)
marks = [0.25, 0.375, 0.5, 0.75, 1.0]
for c in marks:
    x = p_dd_theory(c, 0.5)*100
    y = g_star*(2*c - c**2)*100
    ax.scatter([x], [y], color="#d1495b", zorder=5, s=48)
    ax.annotate(f"c={c}", xy=(x, y), xytext=(x+1.2, y-0.55), fontsize=10)
ax.set_xlabel("P(曾亏掉初始资金的 50%)（%）")
ax.set_ylabel("年化对数增长率（%）")
ax.set_title("讨价还价的菜单：每一点增长要用多少亏损概率来买")
ax.grid(alpha=0.3)
# 标注边际交换率
ax.annotate("从 c=1.0 → 0.5：\n增长 12.5%→9.4%（-25%）\n腰斩概率 50%→12.5%（-75%）",
            xy=(12.5, 9.375), xytext=(22, 5.2), fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#555"))
fig.tight_layout()
fig.savefig(f"{OUT}/kelly-tradeoff-frontier.jpg", dpi=110)
plt.close(fig)

# ---------- 附加数值：给定亏损约束反解 c ----------
for D, pmax in [(0.5, 0.10), (0.3, 0.10), (0.2, 0.05)]:
    # (1-D)^(2/c-1) = pmax  =>  2/c - 1 = ln(pmax)/ln(1-D)
    k = np.log(pmax)/np.log(1-D)
    c_sol = 2.0/(k+1.0)
    g_sol = g_star*(2*c_sol - c_sol**2)
    print(f"约束 P(loss>{int(D*100)}%)<={pmax:.0%} -> c={c_sol:.3f}, 杠杆={c_sol*f_star:.2f}, g={g_sol:.4f} ({g_sol/g_star:.1%} of g*)")

# ---------- 从峰值回撤的分位数（有限期描述统计） ----------
print("\n20年内从峰值最大回撤中位数：")
for c in [0.25, 0.5, 1.0]:
    mdd = results[c][1]
    print(f"  c={c}: median={np.median(mdd):.3f}, q90={np.quantile(mdd,0.9):.3f}")

print("done")
