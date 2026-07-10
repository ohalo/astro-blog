#!/usr/bin/env python3
"""
为文章「ESG 因子量化：可持续数据能否转化为 Alpha」(esg-factor-quant) 生成真实配图。
数据：模拟 300 只股票、含 E/S/G 分项 + 行业 + 真实 ESG 信号（弱）+ ESG 改善(ΔESG)信号（较强）。
图表：
  1. esg_distribution.png    ESG 综合分行业分布 + E/S/G 相关矩阵
  2. esg_forward_scatter.png 当期 ESG 分 vs 未来 12 月收益（弱相关）
  3. esg_tilted_portfolio.png 高 ESG / 低 ESG / 等权 组合累计收益 + ESG 改善因子 IC
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
D = os.path.join(BASE, "esg-factor-quant")
os.makedirs(D, exist_ok=True)
np.random.seed(20260711)

# ============================================================
# 1) 模拟个股面板数据
# ============================================================
N = 300
sectors = ["能源", "材料", "工业", "金融", "消费", "医药", "科技", "公用"]
sec = np.random.choice(sectors, N, p=[0.10, 0.12, 0.14, 0.12, 0.16, 0.10, 0.16, 0.10])

# E/S/G 分项（0-100），行业有系统性偏移
sec_e_bias = {"能源": -18, "材料": -8, "工业": -4, "金融": 2, "消费": 6, "医药": 8, "科技": 4, "公用": 14}
E = np.array([np.clip(np.random.normal(55 + sec_e_bias[s], 12), 5, 99) for s in sec])
S = np.array([np.clip(np.random.normal(58 + 0.5 * sec_e_bias[s], 13), 5, 99) for s in sec])
G = np.array([np.clip(np.random.normal(60 + 0.3 * sec_e_bias[s], 11), 5, 99) for s in sec])

# 综合分：G 权重偏低（数据可得性最差）
esg = 0.35 * E + 0.30 * S + 0.35 * G
# ESG 改善（ΔESG，过去 12 月）：与当期分独立的增量信息（真实数据里也常正交）
d_esg = np.clip(np.random.normal(0, 1, N), -3, 3)

# 未来 12 月收益：弱 ESG 水平信号 + 较强 ΔESG 信号 + 行业/噪声
sector_ret = {"能源": 0.06, "材料": 0.09, "工业": 0.08, "金融": 0.05,
              "消费": 0.11, "医药": 0.10, "科技": 0.13, "公用": 0.04}
base = np.array([sector_ret[s] for s in sec])
signal_level = 0.010 * (esg - esg.mean()) / esg.std()      # 弱
# 注：d_esg 与 esg 独立，故下方 IC 分解才干净：水平 IC 仅来自 signal_level，Δ IC 仅来自 signal_delta
signal_delta = 0.040 * d_esg                              # 较强
fwd_ret = base + signal_level + signal_delta + np.random.normal(0, 0.18, N)

# ============================================================
# 图1：ESG 行业分布 + E/S/G 相关矩阵
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.2))
order = sorted(sectors, key=lambda s: np.mean(esg[sec == s]))
means = [np.mean(esg[sec == s]) for s in order]
ax1.bar(order, means, color="#2ca02c", alpha=0.85)
ax1.set_ylabel("ESG 综合分均值", fontsize=11)
ax1.set_title("ESG 评分的行业系统性偏差（能源低、公用高）", fontsize=11.5, fontweight="bold")
ax1.tick_params(axis="x", rotation=40, labelsize=9)
ax1.grid(True, axis="y", alpha=0.25)
for i, m in enumerate(means):
    ax1.text(i, m + 0.4, f"{m:.0f}", ha="center", fontsize=8.5)

C = np.corrcoef(np.vstack([E, S, G]))
im = ax2.imshow(C, cmap="RdYlGn", vmin=-1, vmax=1)
ax2.set_xticks([0, 1, 2]); ax2.set_yticks([0, 1, 2])
ax2.set_xticklabels(["E", "S", "G"], fontsize=11); ax2.set_yticklabels(["E", "S", "G"], fontsize=11)
for i in range(3):
    for j in range(3):
        ax2.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=11,
                 color="black" if abs(C[i, j]) < 0.7 else "white")
ax2.set_title("E/S/G 分项相关性（交叉重叠→信息冗余）", fontsize=11.5, fontweight="bold")
fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(D, "esg_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图2：ESG 分 vs 未来收益（散点 + 回归）
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.scatter((esg - esg.mean()) / esg.std(), fwd_ret, s=18, alpha=0.45, color="#1f77b4")
# 回归线
z = np.polyfit((esg - esg.mean()) / esg.std(), fwd_ret, 1)
xs = np.linspace(((esg - esg.mean()) / esg.std()).min(), ((esg - esg.mean()) / esg.std()).max(), 50)
ax.plot(xs, np.polyval(z, xs), color="#d62728", lw=2.2, label=f"拟合斜率≈{z[0]:.3f}（弱）")
ax.set_xlabel("ESG 综合分（标准化）", fontsize=11)
ax.set_ylabel("未来 12 月收益", fontsize=11)
ax.set_title("当期 ESG 分 vs 未来收益：截面相关很弱（拥挤 + 数据噪声）", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9.5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "esg_forward_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图3：高/低 ESG 组合累计收益 + ΔESG 因子 IC
# ============================================================
# 用月度面板模拟验证组合表现
Tm = 120
mu_level_h = 0.010 + 0.0009   # 高 ESG 微弱超额
mu_level_l = 0.010 - 0.0006   # 低 ESG 微弱落后
mu_delta = 0.010 + 0.0020     # ESG 改善组合更明显
mu_ew = 0.010                 # 等权基准
rng = np.random.default_rng(20260711)
r_h = rng.normal(mu_level_h, 0.030, Tm)
r_l = rng.normal(mu_level_l, 0.030, Tm)
r_d = rng.normal(mu_delta, 0.030, Tm)
r_ew = rng.normal(mu_ew, 0.030, Tm)
nav_h = np.cumprod(1 + r_h)
nav_l = np.cumprod(1 + r_l)
nav_d = np.cumprod(1 + r_d)
nav_ew = np.cumprod(1 + r_ew)

def cagr(nav):
    return nav[-1] ** (12.0 / len(nav)) - 1
def sharpe(r):
    return r.mean() / (r.std() + 1e-12) * np.sqrt(12)
print(f"高ESG CAGR={cagr(nav_h):.2%} Sharpe={sharpe(r_h):.2f}")
print(f"低ESG CAGR={cagr(nav_l):.2%} Sharpe={sharpe(r_l):.2f}")
print(f"ESG改善 CAGR={cagr(nav_d):.2%} Sharpe={sharpe(r_d):.2f}")
print(f"等权  CAGR={cagr(nav_ew):.2%} Sharpe={sharpe(r_ew):.2f}")

# ΔESG 因子 IC（用单一截面相关近似月度 IC）
ic_delta = np.corrcoef(d_esg, fwd_ret)[0, 1]
ic_level = np.corrcoef(esg, fwd_ret)[0, 1]
print(f"IC(ΔESG)={ic_delta:.3f}, IC(ESG水平)={ic_level:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.2))
months = np.arange(Tm)
ax1.plot(months, nav_h, color="#2ca02c", lw=1.8, label=f"高 ESG (Sharpe={sharpe(r_h):.2f})")
ax1.plot(months, nav_l, color="#d62728", lw=1.8, label=f"低 ESG (Sharpe={sharpe(r_l):.2f})")
ax1.plot(months, nav_d, color="#1f77b4", lw=1.8, label=f"ESG 改善 (Sharpe={sharpe(r_d):.2f})")
ax1.plot(months, nav_ew, color="#888", lw=1.4, ls="--", label=f"等权 (Sharpe={sharpe(r_ew):.2f})")
ax1.set_xlabel("月", fontsize=11); ax1.set_ylabel("净值（起始=1）", fontsize=11)
ax1.set_title("ESG 分组组合累计净值：改善因子领先，水平因子优势有限", fontsize=11.5, fontweight="bold")
ax1.legend(fontsize=8.5); ax1.grid(True, alpha=0.25)

ics = [ic_level, ic_delta]
labels = ["ESG 水平", "ESG 改善(ΔESG)"]
bars = ax2.bar(labels, ics, color=["#d62728", "#2ca02c"], alpha=0.85)
ax2.axhline(0, color="#444", lw=1.1)
ax2.set_ylabel("信息系数 IC", fontsize=11)
ax2.set_title("因子 IC：ΔESG 显著强于 ESG 水平", fontsize=11.5, fontweight="bold")
for b, v in zip(bars, ics):
    ax2.text(b.get_x() + b.get_width() / 2, v + (0.01 if v >= 0 else -0.03),
             f"{v:.3f}", ha="center", fontsize=10)
ax2.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "esg_tilted_portfolio.png"), dpi=150, bbox_inches="tight")
plt.close()

print("✅ esg-factor-quant 配图生成完成：", sorted(os.listdir(D)))
