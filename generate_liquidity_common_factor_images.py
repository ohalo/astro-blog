#!/usr/bin/env python3
"""
为文章「流动性共同因子：Amihud 溢价的系统性来源与危机放大」(liquidity-common-factor) 生成真实配图。

核心逻辑：
  - 个股月度 Amihud ILLIQ = 特异基准 × 时变噪声 + 共同因子载荷 × 系统性流动性因子 L_t + 特异扰动
  - 用 PCA 在 (股票 × 时间) 的 ILLIQ 矩阵上抽出 PC1 = 流动性共同因子
  - 证明：(1) PC1 解释绝大部分截面方差 → 流动性溢价有系统性来源
         (2) 危机期截面相关性飙升 → 共同因子被放大
         (3) 按共同因子载荷排序的组合，危机期溢价被显著放大
  全部为合成但贴合真实结构的数值（流动性天然分层、危机期系统性枯竭、相关性上升），非占位图。
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
D = os.path.join(BASE, "liquidity-common-factor")
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(20260715)

N, T = 60, 120  # 60 只股票 × 120 个月（10 年）

# ---------- 1) 构造系统性流动性共同因子 L_t ----------
# L_t 是「全市场非流动性水平」：常态温和、危机窗口系统性飙升（流动性枯竭）
L = np.zeros(T)
trend = 0.0
for t in range(T):
    trend += rng.normal(0, 0.02)
    L[t] = 0.5 + 0.15 * trend + rng.normal(0, 0.10)   # 常态均值 ~0.5
# 注入 3 个危机窗口（系统性流动性枯竭）
crises = [(30, 36), (66, 72), (102, 110)]
for a, b in crises:
    L[a:b] += np.linspace(1.2, 2.4, b - a) + rng.normal(0, 0.15, b - a)

# ---------- 2) 个股特异流动性档位 + 共同因子载荷 ----------
log_base = rng.normal(2.0, 0.8, N)        # 个股流动性档位（基准 ILLIQ 量级）
loading = rng.uniform(0.3, 1.0, N)        # 对共同因子 L_t 的敏感度（截面异质）
idio_sd = rng.uniform(0.25, 0.45, N)      # 特异流动性噪声

illiq = np.zeros((N, T))
for i in range(N):
    # ILLIQ 必须为正数：用 log 空间构造后再指数化（Amihud ILLIQ 本质为正）
    log_illiq_i = (log_base[i]
                   + loading[i] * L
                   + rng.normal(0, idio_sd[i], T))
    illiq[i] = np.exp(log_illiq_i)

# 个股 ILLIQ 取对数便于 PCA（右偏太严重）
log_illiq = np.log(illiq)

# ---------- 3) PCA 抽共同因子（协方差特征分解，严格归一化）----------
X = log_illiq.T                      # (T, N)：每行一个时间截面，每列一只股票
Xc = X - X.mean(axis=0)
Cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)   # (N, N) 协方差矩阵
eigval, eigvec = np.linalg.eigh(Cov)
order = np.argsort(eigval)[::-1]
eigval = eigval[order]
eigvec = eigvec[:, order]
evr = eigval / eigval.sum()          # 严格归一化，和=1
pc1 = eigvec[:, 0]                    # 各股票在 PC1 上的载荷（共同因子暴露）
common_factor = Xc @ pc1              # 共同因子的时间序列（PC1 得分），(T,)
# 把共同因子变成「非流动性水平」方向（与 L 同向，便于解读）
if np.corrcoef(common_factor, L)[0, 1] < 0:
    common_factor = -common_factor
    pc1 = -pc1

# ---------- 图1：PCA 解释方差（PC1 主导 = 系统性来源）----------
fig, ax = plt.subplots(figsize=(10, 6))
xs = np.arange(1, 11)
bars = ax.bar(xs, evr[:10] * 100, color="#4c72b0", alpha=0.85, label="单主成分解释方差")
ax.set_ylabel("解释方差占比 (%)", fontsize=11)
ax.set_xlabel("主成分序号", fontsize=11)
ax2 = ax.twinx()
ax2.plot(xs, np.cumsum(evr[:10]) * 100, color="#c44e52", marker="o", lw=2.2,
         label="累计解释方差")
ax2.set_ylabel("累计解释方差 (%)", fontsize=11)
ax.set_title(f"流动性共同因子是系统性的：PC1 单独解释 {evr[0]*100:.1f}% 的截面方差",
             fontsize=12.5, fontweight="bold")
ax.axvline(1.5, color="#888", ls="--", lw=1)
ax.text(1.7, evr[0]*100*0.6, "PC1 = 共同因子", color="#c44e52", fontsize=10)
ax.legend(loc="center right", fontsize=9)
ax2.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(D, "lcf_pca_variance.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------- 图2：共同因子时间序列 + 危机阴影 ----------
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(common_factor, color="#2f4b7c", lw=2.0, label="流动性共同因子 (PC1)")
for a, b in crises:
    ax.axvspan(a, b, color="#c44e52", alpha=0.18)
ax.set_xlabel("月份", fontsize=11)
ax.set_ylabel("共同因子取值", fontsize=11)
ax.set_title("系统性流动性因子：常态温和、危机窗口系统性飙升",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "lcf_common_factor.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------- 图3：危机放大 —— 截面相关性飙升 ----------
def avg_pairwise_corr(mat, idx):
    sub = mat[:, idx]
    c = np.corrcoef(sub)
    iu = np.triu_indices(c.shape[0], 1)
    return np.nanmean(c[iu])

calm_idx = [t for t in range(T) if not any(a <= t < b for a, b in crises)]
crisis_idx = [t for t in range(T) if any(a <= t < b for a, b in crises)]
corr_calm = avg_pairwise_corr(log_illiq, calm_idx)
corr_crisis = avg_pairwise_corr(log_illiq, crisis_idx)

fig, ax = plt.subplots(figsize=(8, 5.5))
bars = ax.bar(["常态期", "危机期"], [corr_calm * 100, corr_crisis * 100],
              color=["#4c72b0", "#c44e52"], width=0.5)
for b, v in zip(bars, [corr_calm, corr_crisis]):
    ax.text(b.get_x() + b.get_width()/2, v*100 + 1, f"{v*100:.1f}%",
            ha="center", fontsize=12, fontweight="bold")
ax.set_ylabel("截面平均两两相关性 (%)", fontsize=11)
ax.set_title(f"危机放大机制：流动性同步性 {corr_calm*100:.1f}% → {corr_crisis*100:.1f}%\n"
             f"（共同因子把个股拧成一股绳）",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, max(corr_calm, corr_crisis)*100 * 1.25)
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "lcf_crisis_corr.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------- 图4：按共同因子载荷排序的组合，危机期溢价被放大 ----------
# 构造收益：高共同因子暴露的股票承担流动性风险 → 危机期（L 高）获得更高补偿
beta_m = rng.uniform(0.6, 1.4, N)
rM = rng.normal(0.008, 0.04, T)
# 个股收益：市场Beta + 共同因子暴露×L（正相关=稀缺时补偿多）+ 特异
ret = np.zeros((N, T))
for i in range(N):
    ret[i] = beta_m[i] * rM + 0.05 * pc1[i] * (L - L.mean()) + rng.normal(0, 0.03, T)

# 每个月按 pc1（共同因子暴露）排序，分成 5 组，记录下月收益
spread_cal_m, spread_cr_m = [], []
for t in range(T - 1):
    order = np.argsort(pc1)
    q = np.array_split(order, 5)
    r_low = ret[q[0], t + 1].mean()
    r_high = ret[q[4], t + 1].mean()
    if any(a <= t < b for a, b in crises):
        spread_cr_m.append(r_high - r_low)
    else:
        spread_cal_m.append(r_high - r_low)

prem_calm = np.mean(spread_cal_m) * 12 * 100
prem_crisis = np.mean(spread_cr_m) * 12 * 100

fig, ax = plt.subplots(figsize=(8, 5.5))
bars = ax.bar(["常态期", "危机期"], [prem_calm, prem_crisis],
              color=["#4c72b0", "#c44e52"], width=0.5)
for b, v in zip(bars, [prem_calm, prem_crisis]):
    ax.text(b.get_x() + b.get_width()/2, v + (1 if v >= 0 else -2),
            f"{v:+.1f}%", ha="center", fontsize=12, fontweight="bold")
ax.axhline(0, color="#333", lw=1)
ax.set_ylabel("高暴露−低暴露组合 年化溢价 (%)", fontsize=11)
ax.set_title(f"危机放大溢价：共同因子暴露溢价 {prem_calm:+.1f}% → {prem_crisis:+.1f}%\n"
             f"（危机期稀缺性补偿被显著放大）",
             fontsize=12, fontweight="bold")
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "lcf_premium.png"), dpi=150, bbox_inches="tight")
plt.close()

print("=== 流动性共同因子 诊断 ===")
print(f"PC1 解释方差: {evr[0]*100:.1f}%  | 前三主成分累计: {evr[:3].sum()*100:.1f}%")
print(f"共同因子与注入 L 的相关性: {np.corrcoef(common_factor, L)[0,1]:.3f}")
print(f"常态期截面相关性: {corr_calm*100:.1f}%  | 危机期: {corr_crisis*100:.1f}%  (放大 {(corr_crisis/corr_calm-1)*100:.0f}%)")
print(f"共同因子暴露溢价 常态: {prem_calm:+.1f}%  | 危机: {prem_crisis:+.1f}%")
print(f"生成图片: {sorted(os.listdir(D))}")
