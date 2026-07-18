#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「LSV 流动性风险：用流动性调整给资产定价加一道折扣」生成真实配图与统计数字。

核心机制(基于 Pastor & Stambaugh 2003 流动性调整 CAPM):
  标准 CAPM 的定价只认系统性市场风险 beta_mkt, 但很多资产(小盘、低换手、信用债)
  对"市场整体流动性收紧"极度敏感——危机时一起跌、流动性一起干涸。Pástor-Stambaugh
  把"市场层面的流动性创新" LIQ_t 当成第二个风险因子, 给出流动性调整定价:
      E[R_i] - R_f = beta_mkt_i * E[R_mkt - R_f] + lambda * beta_liq_i
  其中 beta_liq_i = Cov(R_i, LIQ_t)/Var(LIQ_t), lambda 是"流动性风险的价格"。
  含义: 一个资产即使 beta_mkt 不高, 只要它在流动性冲击波里跟市场一起变脸
  (beta_liq 高), 它就该拿更高的预期收益——这是流动性给它贴的"折扣"对应的补偿。

两步估计(经典两遍回归):
  (1) 时间序列: 对每个资产 R_i = a + b_mkt*R_mkt + b_liq*LIQ + e, 得到 (b_mkt_i, b_liq_i)
  (2) 横截面:   mean(R_i) - R_f = g0 + g_mkt*b_mkt_i + g_liq*b_liq_i + u
      g_liq 就是 lambda 的估计(流动性风险价格), g_mkt 是市场溢价。

全部数字由文中 Python 真实计算(numpy/scipy/matplotlib), 无占位符。

图片:
  lsv_liq_ts.png          —— 市场层面流动性创新 LIQ_t 时间序列(含流动性枯竭冲击)
  lsv_xs_fit.png          —— 横截面: 预期超额收益 vs 流动性 beta(拟合线斜率 = lambda)
  lsv_quintile.png        —— 按流动性 beta 五分位排序, 各组平均超额收益(流动性溢价)
  lsv_lambda_bar.png      —— 估计出的市场溢价 g_mkt vs 流动性价格 lambda(g_liq)
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
D = os.path.join(BASE, "lsv-liquidity-risk")
os.makedirs(D, exist_ok=True)
METRICS = os.path.join(D, "_metrics.txt")
lines = []

C_BLUE = "#2c7fb8"
C_RED = "#d7301f"
C_GREEN = "#1a9850"
C_GREY = "#7f7f7f"
C_GRID = "#dddddd"

def log(s):
    print(s); lines.append(str(s))

rng = np.random.default_rng(20260718)

# ============== 1. 合成因子: 市场收益 Rm 与 市场流动性创新 LIQ ==============
T = 360                      # 30 年月度
mu_mkt = 0.006              # 市场月度溢价
sig_mkt = 0.045
Rm = rng.normal(mu_mkt, sig_mkt, T)          # 市场超额收益
# LIQ: 市场层面流动性创新(零均值冲击因子!) 平时小波动, 偶发"流动性枯竭"大负冲击
# 注意: 流动性因子必须是零均值创新, 否则会污染预期收益的横截面结构
liq_base = rng.normal(0.0, 0.020, T)
shock_mask = rng.random(T) < 0.06            # 约 6% 月份出现流动性冲击
liq_base[shock_mask] -= rng.uniform(0.06, 0.14, shock_mask.sum())   # 深度负冲击
liq_base[~shock_mask] += rng.uniform(0.0, 0.03, (~shock_mask).sum()) * 0.4  # 少量正流动性宽松对冲
shock_pos = rng.random(T) < 0.04
liq_base[shock_pos] += rng.uniform(0.05, 0.12, shock_pos.sum())    # 偶发流动性宽松正冲击
LIQ = liq_base - liq_base.mean()             # 严格零均值创新
log("===== 流动性调整 CAPM (Pastor-Stambaugh 2003) =====")
log(f"样本 T = {T} 月; 市场月度溢价 mu_mkt = {mu_mkt:.4f}, 波动 = {sig_mkt:.4f}")
log(f"LIQ 均值 = {LIQ.mean():.6f} (严格零均值), 波动 = {LIQ.std():.4f}, 冲击月份数 = {int(shock_mask.sum())}")

# ============== 2. 造 N 个资产: 给每个资产真实 mkt/liq beta + 特质噪声 ==============
N = 60
beta_mkt_true = rng.uniform(0.55, 1.45, N)     # 各资产市场 beta
beta_liq_true = rng.uniform(0.10, 1.60, N)     # 各资产流动性 beta(差异很大)
lambda_true = 0.0080                           # 真实流动性风险价格(月度, 正数!)
# 关键: 流动性溢价必须嵌入预期收益的横截面 —— 高流动性 beta 资产拿正 alpha 补偿
# alpha_i = lambda_true * beta_liq_true  (即 E[R_i]-R_f = beta_mkt*mu_mkt + lambda*beta_liq)
alpha_i = lambda_true * beta_liq_true + rng.normal(0.0, 0.0008, N)  # 微弱特质扰动
R = np.zeros((T, N))
for i in range(N):
    eps = rng.normal(0.0, 0.022, T)
    R[:, i] = (alpha_i[i]
               + beta_mkt_true[i] * Rm
               + beta_liq_true[i] * LIQ
               + eps)
# 真实预期超额收益(理论): E[R_i]-R_f = beta_mkt*mu_mkt + lambda_true*beta_liq
E_excess_true = beta_mkt_true * mu_mkt + lambda_true * beta_liq_true
log(f"资产数 N = {N}; 真实 lambda(流动性价格) = {lambda_true:.4f}")
log(f"真实 beta_liq 范围 [{beta_liq_true.min():.2f}, {beta_liq_true.max():.2f}]")

# ============== 3. 第一遍回归: 时间序列估计每资产 (b_mkt, b_liq) ==============
X = np.column_stack([np.ones(T), Rm, LIQ])
b_mkt_est, b_liq_est = np.zeros(N), np.zeros(N)
for i in range(N):
    coef, *_ = np.linalg.lstsq(X, R[:, i], rcond=None)
    b_mkt_est[i] = coef[1]
    b_liq_est[i] = coef[2]
# 横截面平均超额收益
mean_excess = R.mean(axis=0)
log(f"第一遍回归: b_mkt 相关(beta_mkt_true, b_mkt_est) corr = {np.corrcoef(beta_mkt_true, b_mkt_est)[0,1]:.3f}")
log(f"第一遍回归: b_liq  相关(beta_liq_true, b_liq_est) corr = {np.corrcoef(beta_liq_true, b_liq_est)[0,1]:.3f}")

# ============== 4. 第二遍回归: 横截面估计 lambda ==============
Z = np.column_stack([np.ones(N), b_mkt_est, b_liq_est])
gamma, *_ = np.linalg.lstsq(Z, mean_excess, rcond=None)
g0, g_mkt, g_liq = gamma
pred = Z @ gamma
ss_res = np.sum((mean_excess - pred) ** 2)
ss_tot = np.sum((mean_excess - mean_excess.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
log(f"第二遍横截面回归: g0={g0:.5f}, g_mkt(市场溢价)={g_mkt:.5f}, g_liq=lambda={g_liq:.5f}")
log(f"横截面 R^2 = {r2:.3f}")
log(f"lambda 估计误差 = {abs(g_liq - lambda_true):.5f}")

# 与"只看市场 beta 的标准 CAPM"对比: 只用 b_mkt 做横截面
Z1 = np.column_stack([np.ones(N), b_mkt_est])
g1, *_ = np.linalg.lstsq(Z1, mean_excess, rcond=None)
pred1 = Z1 @ g1
ss_res1 = np.sum((mean_excess - pred1) ** 2)
r2_capm = 1 - ss_res1 / ss_tot
log(f"对比 标准 CAPM(只用市场 beta) 横截面 R^2 = {r2_capm:.3f}  vs  流动性调整 R^2 = {r2:.3f}")

# ============== 5. 排序验证: 按流动性 beta 五分位 ==============
order = np.argsort(b_liq_est)
quint = np.array_split(order, 5)
q_ret = [mean_excess[q].mean() for q in quint]
q_liq = [b_liq_est[q].mean() for q in quint]
log("流动性 beta 五分位(由低到高) —— 平均流动性 beta / 平均超额收益:")
for k in range(5):
    log(f"  Q{k+1}: beta_liq={q_liq[k]:.3f}, 超额收益={q_ret[k]:.4f}")
liq_premium = q_ret[4] - q_ret[0]
log(f"流动性溢价(最高-最低五分位) = {liq_premium:.4f} 月 = {liq_premium*12:.2%} 年")

# ============== 6. 画图 ==============
# 图1: LIQ 时间序列(市场流动性创新)
fig, ax = plt.subplots(figsize=(9.0, 3.6))
colors = [C_RED if v < -0.04 else C_BLUE for v in LIQ]
ax.bar(np.arange(T), LIQ, color=colors, width=1.0)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("时间 (月)")
ax.set_ylabel("LIQ_t (流动性创新)")
ax.set_title("市场层面流动性创新 LIQ_t：平时小波动，偶发深度枯竭冲击", fontsize=11)
ax.grid(alpha=0.3, axis="y", color=C_GRID)
fig.tight_layout(); fig.savefig(os.path.join(D, "lsv_liq_ts.png")); plt.close(fig)

# 图2: 横截面拟合 (预期超额收益 vs 流动性 beta)
fig, ax = plt.subplots(figsize=(7.4, 5.0))
ax.scatter(b_liq_est, mean_excess, s=22, color=C_BLUE, alpha=0.6, label="个股 (估计 beta_liq, 平均超额收益)")
xs = np.linspace(b_liq_est.min(), b_liq_est.max(), 50)
ax.plot(xs, g0 + g_mkt * np.mean(b_mkt_est) + g_liq * xs, color=C_RED, lw=2,
        label=f"流动性调整拟合: slope=lambda={g_liq:.4f}")
ax.set_xlabel("流动性 beta (beta_liq, 对 LIQ 的敏感度)")
ax.set_ylabel("平均超额收益")
ax.set_title(f"横截面：预期超额收益 vs 流动性 beta (R²={r2:.2f})", fontsize=11)
ax.legend(fontsize=8.5); ax.grid(alpha=0.3, color=C_GRID)
fig.tight_layout(); fig.savefig(os.path.join(D, "lsv_xs_fit.png")); plt.close(fig)

# 图3: 五分位流动性溢价
fig, ax = plt.subplots(figsize=(7.0, 4.4))
bars = ax.bar([f"Q{k+1}\n(b={q_liq[k]:.2f})" for k in range(5)], q_ret,
              color=[C_GREEN if k >= 3 else C_GREY for k in range(5)])
for bar, val in zip(bars, q_ret):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.0008, f"{val*100:.2f}%",
            ha="center", fontsize=9)
ax.set_xlabel("按流动性 beta 五分位 (低 → 高)")
ax.set_ylabel("平均超额收益 (月)")
ax.set_title(f"流动性溢价：高流动性beta组多赚 {liq_premium*100:.2f}%/月", fontsize=10.5)
ax.grid(alpha=0.3, axis="y", color=C_GRID)
fig.tight_layout(); fig.savefig(os.path.join(D, "lsv_quintile.png")); plt.close(fig)

# 图4: 市场溢价 vs 流动性价格
fig, ax = plt.subplots(figsize=(6.0, 4.2))
bars = ax.bar(["市场溢价\ng_mkt", "流动性价格\nlambda (g_liq)"],
              [g_mkt, g_liq], color=[C_BLUE, C_RED])
for bar, val in zip(bars, [g_mkt, g_liq]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.0003, f"{val:.4f}", ha="center", fontsize=10)
ax.set_ylabel("横截面系数 (月)")
ax.set_title("两遍回归：市场溢价 vs 流动性风险价格", fontsize=11)
ax.grid(alpha=0.3, axis="y", color=C_GRID)
fig.tight_layout(); fig.savefig(os.path.join(D, "lsv_lambda_bar.png")); plt.close(fig)

with open(METRICS, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n=== IMAGES WRITTEN ===")
print("\n".join(sorted(os.listdir(D))))
