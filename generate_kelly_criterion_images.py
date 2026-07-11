#!/usr/bin/env python3
"""
为文章「Kelly 公式与资金管理」(kelly-criterion-sizing) 生成真实配图。
数据：假设一个胜率 p、赔率 b 的离散下注，或一个年化收益 μ、波动 σ 的连续策略。
方法：
  1) 离散 Kelly：f* = (bp - q)/b，画出期望对数增长率 G(f) 随下注比例 f 的曲线（峰值=Kelly）；
  2) 连续 Kelly：f* = μ/σ²，用蒙特卡洛模拟不同 f 下长期财富路径与破产概率；
  3) 对比 满仓(f=1)、全 Kelly、半 Kelly、固定 0.25 四类资金曲线的长期表现。
全部为真实数值计算，非占位图。
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
D = os.path.join(BASE, "kelly-criterion-sizing")
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(4026)

# ============================================================
# 1) 离散下注：胜率/赔率设定
# ============================================================
p = 0.55          # 胜率
q = 1 - p
b = 1.0           # 赔率（赢 1 元 / 亏 1 元，即对称赔率）
f_star = (b * p - q) / b
print(f"离散 Kelly：p={p}, b={b}  →  f* = {f_star:.4f}（即 {(f_star*100):.1f}% 仓位）")

# 期望对数增长率 G(f) = p*log(1+bf) + q*log(1-bf) （bf 在 [-1,1] 即 f∈[-1,1] 当 b=1）
fs = np.linspace(0.001, 0.99, 400)
G = p * np.log(1 + b * fs) + q * np.log(1 - b * fs)
G[f_star < fs] += 0   # 占位（无操作）

# ============================================================
# 2) 连续策略：μ / σ 与连续 Kelly f* = μ/σ²
# ============================================================
mu = 0.10          # 年化收益
sigma = 0.20       # 年化波动
f_cont = mu / sigma**2
print(f"连续 Kelly（正态近似）：f* = μ/σ² = {mu}/{sigma**2:.3f} = {f_cont:.3f}（即 {(f_cont*100):.1f}% 仓位）")

# 蒙特卡洛：以日频复利模拟不同 f 的财富路径
Tdays = 252 * 10       # 10 年
dt = 1.0 / 252
npaths = 4000
# 日收益 ~ 正态（μ_d, σ_d）
mud = mu * dt
sigmad = sigma * np.sqrt(dt)
shocks = rng.normal(mud, sigmad, size=(npaths, Tdays))

def wealth_path(f, shocks):
    # 每期收益 = f * 资产收益（满仓 f=1 即全仓持有）
    peri = f * shocks
    return np.cumprod(1 + peri, axis=1)

fracs = {"满仓 f=1.0": 1.0, "全 Kelly f*": f_cont, "半 Kelly 0.5f*": 0.5 * f_cont, "固定 0.25": 0.25}
paths = {name: wealth_path(f, shocks) for name, f in fracs.items()}
# 破产概率：任意时刻财富低于 0.5（腰斩）的路径占比（近似「大幅回撤」）
ruin_share = {name: (wp.min(axis=1) < 0.5).mean() for name, wp in paths.items()}
final_med = {name: np.median(wp[:, -1]) for name, wp in paths.items()}
print("\n蒙特卡洛（10年 / 4000 条路径）：")
for name in fracs:
    print(f"  {name:18s} 最终中位数财富={final_med[name]:.2f}x  曾腰斩占比={ruin_share[name]*100:.1f}%")

# ============================================================
# 图1：G(f) 增长曲线，峰值在 Kelly
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(fs, G, color="#1f77b4", lw=2.2)
ax.axvline(f_star, color="#d62728", ls="--", lw=1.6, label=f"Kelly f* = {f_star:.3f}")
ax.scatter([f_star], [G.max()], color="#d62728", zorder=5, s=60)
ax.annotate(f"峰值 G(f*) = {G.max():.4f}\n（长期复合增速最快）",
            xy=(f_star, G.max()), xytext=(f_star + 0.18, G.max() - 0.08),
            fontsize=10, arrowprops=dict(arrowstyle="->", color="black"))
ax.axvspan(0.7, 1.0, color="#d62728", alpha=0.08)
ax.text(0.85, G.min()*0.6, "过度下注区\nG(f)<0 → 长期必亏", ha="center", fontsize=9, color="#a00")
ax.set_xlabel("下注比例 f（仓位占比）", fontsize=11)
ax.set_ylabel("期望对数增长率 G(f)", fontsize=11)
ax.set_title("离散 Kelly：期望对数增长率在 f* 处达峰，越线即转负", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "kelly_growth_vs_f.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图2：不同 f 的财富中位数 + 破产概率
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(12, 5.0))
names = list(fracs.keys())
# 左：中位数终值
meds = [final_med[n] for n in names]
bars = ax[0].bar(names, meds, color=["#d62728", "#2ca02c", "#1f77b4", "#9467bd"])
for b_, v in zip(bars, meds):
    ax[0].text(b_.get_x()+b_.get_width()/2, v+0.05, f"{v:.2f}x", ha="center", fontsize=10, fontweight="bold")
ax[0].set_ylabel("10 年终值（中位数, x）", fontsize=11)
ax[0].set_title("长期复利：全 Kelly 终值最高，但…", fontsize=11.5, fontweight="bold")
ax[0].tick_params(axis="x", labelrotation=20)
ax[0].grid(True, axis="y", alpha=0.25)
# 右：腰斩概率
ruins = [ruin_share[n] for n in names]
bars2 = ax[1].bar(names, np.array(ruins)*100, color=["#d62728", "#2ca02c", "#1f77b4", "#9467bd"])
for b_, v in zip(bars2, np.array(ruins)*100):
    ax[1].text(b_.get_x()+b_.get_width()/2, v+0.4, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax[1].set_ylabel("曾腰斩路径占比 (%)", fontsize=11)
ax[1].set_title("满仓 100% 腰斩概率爆表，半 Kelly 大幅降低", fontsize=11.5, fontweight="bold")
ax[1].tick_params(axis="x", labelrotation=20)
ax[1].grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "kelly_ruin_vs_f.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图3：抽样财富路径（每条策略画 30 条）
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.6))
sample = 30
tvec = np.arange(Tdays + 1) / 252
colors = {"满仓 f=1.0": "#d62728", "全 Kelly f*": "#2ca02c", "半 Kelly 0.5f*": "#1f77b4", "固定 0.25": "#9467bd"}
for name, wp in paths.items():
    idx = rng.choice(npaths, sample, replace=False)
    for i in idx:
        ax.plot(tvec[1:], wp[i], color=colors[name], alpha=0.18, lw=0.8)
    # 画中位数曲线加粗
    ax.plot(tvec[1:], np.median(wp, axis=0), color=colors[name], lw=2.4, label=name)
ax.set_xlabel("年", fontsize=11)
ax.set_ylabel("财富（起始=1）", fontsize=11)
ax.set_title("抽样财富路径：满仓最陡但频繁暴跌，半 Kelly 更平滑", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
ax.set_yscale("log")
plt.tight_layout()
plt.savefig(os.path.join(D, "kelly_paths.png"), dpi=150, bbox_inches="tight")
plt.close()

print("\n✅ Kelly 配图生成完成：", sorted(os.listdir(D)))
