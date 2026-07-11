#!/usr/bin/env python3
"""
为文章「Barra 风险模型实战拆解：把你持仓的每一个风险都摊开来」(barra-risk-model-practice) 生成真实配图。

核心逻辑（Barra 基本面风险模型）：
  r = X f + u
  V = X F X' + Δ          (F = 因子收益协方差, Δ = 特异性方差对角)
  σ_p² = w' V w = a' F a + w'Δw ,  a = X' w（组合因子暴露）
  每个因子 k 对总风险的贡献 C_k = a_k * (F a)_k （精确分解，Σ_k C_k = a'F a）
  全部为合成但贴合真实结构的数值计算，非占位图。
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
D = os.path.join(BASE, "barra-risk-model-practice")
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(20260711)

# ---------- 构建合成但真实感强的因子结构 ----------
industries = ["科技", "金融", "消费", "工业"]
n_ind = len(industries)
style_names = ["规模 Size", "价值 Value", "动量 Momentum", "Beta"]
n_style = len(style_names)
n_stocks = 24
stocks_per_ind = n_stocks // n_ind

# 暴露矩阵 X (n_stocks x (n_ind + n_style))
X = np.zeros((n_stocks, n_ind + n_style))
stock_names = []
for i in range(n_stocks):
    ind = i // stocks_per_ind
    X[i, ind] = 1.0                      # 行业哑变量
    # 风格暴露：每行业给一个合理中枢再叠加个股扰动
    center = np.array([
        0.6 if ind == 0 else -0.4,       # 科技偏大、金融偏小
        0.3 if ind == 2 else -0.2,       # 消费偏价值
        (0.4 if ind == 0 else -0.3) + (0.2 if ind == 3 else 0.0),
        0.2 if ind == 1 else -0.1,       # 金融 Beta 偏高
    ])
    X[i, n_ind:] = center + rng.normal(0, 0.5, n_style)
    stock_names.append(f"{industries[ind][0]}{i % stocks_per_ind + 1}")

# 因子协方差 F（日频近似）：对角波动 + 相关性
f_vol = np.array([0.011, 0.009, 0.010, 0.008,           # 行业（科技/金融/消费/工业）
                  0.006, 0.005, 0.007, 0.004])           # 风格（规模/价值/动量/Beta）
corr = np.eye(n_ind + n_style)
# 行业间适度正相关
corr[0, 1] = corr[1, 0] = 0.45
corr[0, 2] = corr[2, 0] = 0.30
corr[1, 2] = corr[2, 1] = 0.25
corr[2, 3] = corr[3, 2] = 0.35
corr[0, 3] = corr[3, 0] = 0.20
# 风格间：规模-价值负相关、动量-规模负相关、Beta-价值略负
corr[4, 5] = corr[5, 4] = -0.30
corr[4, 6] = corr[6, 4] = -0.25
corr[5, 7] = corr[7, 5] = -0.20
corr[6, 7] = corr[7, 6] = 0.15
F = (f_vol[:, None] * f_vol[None, :]) * corr

# 特异性方差（日频）：每只股票 18%~26% 年化 -> 日方差
idio_annual = rng.uniform(0.18, 0.26, n_stocks)
delta = (idio_annual ** 2) / 252.0

# ---------- 组合构建：一个偏科技+动量的主动组合 vs 市值加权基准 ----------
cap = rng.uniform(0.5, 3.0, n_stocks)
w_bench = cap / cap.sum()
# 主动组合：超配科技、超配动量、低配金融，行业略集中
w主动 = w_bench.copy()
w主动 += np.array([0.10 if i // stocks_per_ind == 0 else -0.02 for i in range(n_stocks)])
w主动 += 0.5 * X[:, 4 + 2] / 100.0    # 按动量暴露加权（极轻）
w主动 = np.clip(w主动, 0.001, None)
w_port = w主动 / w主动.sum()

factor_labels = industries + style_names


def risk_decomp(w):
    a = X.T @ w                                  # 组合因子暴露（8 维）
    Fa = F @ a
    C = a * Fa                                   # 每个因子的总贡献（含交叉项），Σ=C = a'Fa
    idio = (w ** 2) * delta                      # 特异性贡献（每只）
    var_factor = C.sum()
    var_idio = idio.sum()
    total = var_factor + var_idio
    return a, C, idio, total, var_factor, var_idio


a_p, C_p, idio_p, total_p, vf_p, vi_p = risk_decomp(w_port)
ann_vol = np.sqrt(total_p * 252) * 100
print(f"主动组合年化波动 ≈ {ann_vol:.2f}%")
print(f"因子风险占比 = {vf_p/total_p*100:.1f}%，特异性风险占比 = {vi_p/total_p*100:.1f}%")

# ============================================================
# 图1：风险贡献分解（行业 + 风格 + 特异性）
# ============================================================
contrib = np.concatenate([C_p, [vi_p]])
# 特异性写成一个汇总项
labels1 = factor_labels + ["特异性 Idiosyncratic"]
pct1 = contrib / total_p * 100
colors1 = ["#4c72b0"] * n_ind + ["#dd8452"] * n_style + ["#55a868"]
fig, ax = plt.subplots(figsize=(11, 6.2))
y = np.arange(len(labels1))[::-1]
bars = ax.barh(y, pct1, color=colors1)
ax.set_yticks(y)
ax.set_yticklabels(labels1, fontsize=10)
for b, v in zip(bars, pct1):
    ax.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)
ax.set_xlabel("对组合总方差的贡献占比 (%)", fontsize=11)
ax.set_title("把组合总风险拆开：行业、风格、特异性各占多少", fontsize=12.5, fontweight="bold")
ax.grid(True, axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "barra_risk_decomp.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图2：组合相对基准的主动风格暴露
# ============================================================
a_bench = X.T @ w_bench
active = a_p - a_bench
# 只用风格因子（4 个）画主动暴露
fig, ax = plt.subplots(figsize=(11, 5.2))
xpos = np.arange(n_style)
bars = ax.bar(xpos, active[n_ind:], color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"])
ax.axhline(0, color="black", lw=1.0)
for b, v in zip(bars, active[n_ind:]):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.02 if v >= 0 else -0.04),
            f"{v:+.2f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(xpos)
ax.set_xticklabels(["规模 Size", "价值 Value", "动量 Momentum", "Beta"], fontsize=11)
ax.set_ylabel("主动因子暴露 (组合 − 基准)", fontsize=11)
ax.set_title("组合相对基准的主动暴露：你在偷偷押注什么", fontsize=12.5, fontweight="bold")
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "barra_active_exposure.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图3：因子协方差热图（哪些风险源在联手放大波动）
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 7.2))
im = ax.imshow(F * 100, cmap="RdBu_r", vmin=-0.01, vmax=0.01)
ax.set_xticks(range(len(factor_labels)))
ax.set_yticks(range(len(factor_labels)))
ax.set_xticklabels(factor_labels, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(factor_labels, fontsize=9)
for i in range(len(factor_labels)):
    for j in range(len(factor_labels)):
        val = F[i, j] * 100
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.5,
                color="black" if abs(val) < 0.006 else "white")
ax.set_title("因子协方差热图（日频 ×100）：哪些风险源在联手放大波动",
             fontsize=11.5, fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="协方差 ×100")
plt.tight_layout()
plt.savefig(os.path.join(D, "barra_cov_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

print("\n✅ Barra 配图生成完成：", sorted(os.listdir(D)))
print("因子贡献占比(%):", dict(zip(factor_labels, (C_p / total_p * 100).round(1))))
print(f"特异性占比: {vi_p/total_p*100:.1f}%")
