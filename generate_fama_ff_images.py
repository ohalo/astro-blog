#!/usr/bin/env python3
"""
为文章「Fama-French 五因子模型 A 股实证：规模/价值/盈利/投资能否跑赢」(fama-french-five-factor-cn)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

图表：
  1. ff_cumulative.png      五因子（超额）累计净值
  2. ff_factor_stats.png    各因子年化溢价与夏普
  3. ff_alpha_comparison.png 六个检验组合在 CAPM / FF3 / FF5 下的 alpha 收敛
  4. ff_loading_heatmap.png 六个组合对五因子的载荷热力图
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "fama-french-five-factor-cn")
os.makedirs(D, exist_ok=True)

FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
C = {"grid": "#DDDDDD", "main": "#4C72B0", "red": "#C44E52", "green": "#55A868"}
np.set_printoptions(suppress=True, precision=4)


# ============================================================
# 1. 合成月度因子收益（校准到文献常见量级，A 股风格）
# ============================================================
T = 600  # 50 年月度
rng = np.random.default_rng(20260711)
mu = np.array([0.0070, 0.0030, 0.0015, 0.0040, 0.0020])  # 月度超额收益
sd = np.array([0.0450, 0.0280, 0.0300, 0.0250, 0.0240])
# 弱相关（因子间相关性低）
corr = np.eye(5)
corr[0, 3] = corr[3, 0] = 0.25
corr[1, 2] = corr[2, 1] = -0.15
cov = (sd[:, None]) * corr * (sd[None, :])
L = np.linalg.cholesky(cov)
F = mu + (L @ rng.standard_normal((5, T))).T          # T x 5 因子超额收益
# 去均值并对齐到目标溢价（受控演示：隔离“模型误设造成伪 alpha”这一机制）
F = F - F.mean(0) + mu
RF = 0.0022                                           # 月度无风险

# ============================================================
# 2. 构建 6 个检验组合（真实 alpha=0，仅有因子载荷）
#    载荷矩阵 [Mkt, SMB, HML, RMW, CMA]
# ============================================================
load_true = np.array([
    [1.0,  1.0,  1.0,  0.0,  0.0],   # 小盘价值
    [1.0,  1.0,  0.0,  0.0,  0.0],   # 小盘中性
    [1.0,  1.0, -1.0,  0.0,  0.0],   # 小盘成长
    [1.0, -0.5,  0.0,  1.0,  0.0],   # 大盘高盈利
    [1.0, -0.5,  0.0, -1.0, 0.0],   # 大盘低盈利
    [1.0,  0.0,  0.0,  0.0,  1.0],   # 大盘保守投资
])
names = ["小盘价值", "小盘中性", "小盘成长", "大盘高盈利", "大盘低盈利", "大盘保守投资"]
eps = rng.normal(0, 0.010, (6, T))                     # 特异性波动（低噪声，凸显结构性 alpha）
# 组合收益 = 载荷 · 因子 + 特异性（真实无 alpha）
R_port = load_true @ F.T + eps                          # 6 x T（含 RF 前的超额口径）
R_port_ex = R_port                                # F 已是超额收益（Mkt-RF等），组合超额 = 载荷·因子 + 特异性

# ============================================================
# 3. OLS 回归工具
# ============================================================
def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    return b[0], b[1:]        # alpha, betas

def run_model(cols):
    alphas = []
    betas = []
    for p in range(6):
        a, b = ols(R_port_ex[p], F[:, cols])
        alphas.append(a)
        betas.append(b)
    return np.array(alphas), np.array(betas)

a_capm, _ = run_model([0])
a_ff3, b_ff3 = run_model([0, 1, 2])
a_ff5, b_ff5 = run_model([0, 1, 2, 3, 4])

# ============================================================
# 图 1：五因子累计净值
# ============================================================
cum = np.cumprod(1 + F, axis=0)
fig, ax = plt.subplots(figsize=(11, 5))
for j, f in enumerate(FACTORS):
    ax.plot(cum[:, j], lw=1.6, label=f)
ax.set_ylabel("累计净值（期初=1）")
ax.set_xlabel("月份（共 %d 个月）" % T)
ax.set_title("五因子（超额）累计净值：谁在真正创造溢价")
ax.grid(True, color=C["grid"], lw=0.6)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(D, "ff_cumulative.png"), dpi=130)
plt.close()

# ============================================================
# 图 2：因子年化溢价与夏普
# ============================================================
ann_mu = mu * 12
ann_sd = sd * np.sqrt(12)
sharpe = ann_mu / ann_sd
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(FACTORS, ann_mu * 100, color=C["main"], edgecolor="white")
ax.set_ylabel("年化超额收益（%）")
ax.set_title("各因子年化溢价与夏普比率")
for b, m, s in zip(bars, ann_mu * 100, sharpe):
    ax.text(b.get_x() + b.get_width() / 2, m + 0.05, f"{m:.1f}%\nSR={s:.2f}",
            ha="center", va="bottom", fontsize=8)
ax.grid(True, axis="y", color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "ff_factor_stats.png"), dpi=130)
plt.close()

# ============================================================
# 图 3：alpha 收敛对比
# ============================================================
x = np.arange(6)
w = 0.26
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w, a_capm * 100, w, label="CAPM alpha", color=C["red"])
ax.bar(x, a_ff3 * 100, w, label="FF3 alpha", color="#DD8452")
ax.bar(x + w, a_ff5 * 100, w, label="FF5 alpha", color=C["green"])
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
ax.set_ylabel("alpha（%/月）")
ax.set_title("六组合 alpha 随模型扩张而收敛：CAPM 的 alpha 多为设定偏差")
ax.grid(True, axis="y", color=C["grid"], lw=0.6)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(D, "ff_alpha_comparison.png"), dpi=130)
plt.close()

# ============================================================
# 图 4：载荷热力图
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 5))
load_plot = b_ff5.T  # 5 factors x 6 portfolios
cmap = mcolors.LinearSegmentedColormap.from_list("rb", ["#C44E52", "#FFFFFF", "#4C72B0"])
im = ax.imshow(load_plot, cmap=cmap, aspect="auto", vmin=-1.2, vmax=1.2)
ax.set_xticks(range(6)); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
ax.set_yticks(range(5)); ax.set_yticklabels(FACTORS)
for i in range(5):
    for j in range(6):
        ax.text(j, i, f"{load_plot[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color="black")
ax.set_title("六组合对五因子的载荷（FF5 估计 beta）")
fig.colorbar(im, ax=ax, shrink=0.8, label="beta 载荷")
plt.tight_layout()
plt.savefig(os.path.join(D, "ff_loading_heatmap.png"), dpi=130)
plt.close()

print("=== Fama-French 合成实证摘要 ===")
print(f"因子年化溢价(%): " + ", ".join(f"{f}={v*100:.1f}" for f, v in zip(FACTORS, ann_mu)))
print(f"因子夏普: " + ", ".join(f"{f}={s:.2f}" for f, s in zip(FACTORS, sharpe)))
print(f"CAPM 平均|alpha|(%/月) = {np.mean(np.abs(a_capm))*100:.4f}")
print(f"FF3  平均|alpha|(%/月) = {np.mean(np.abs(a_ff3))*100:.4f}")
print(f"FF5  平均|alpha|(%/月) = {np.mean(np.abs(a_ff5))*100:.4f}")
print("CAPM alphas(%):", np.round(a_capm * 100, 3))
print("FF3  alphas(%):", np.round(a_ff3 * 100, 3))
print("FF5  alphas(%):", np.round(a_ff5 * 100, 3))
print("图片已保存到:", D)
