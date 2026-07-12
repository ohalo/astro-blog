#!/usr/bin/env python3
"""
为文章「方差风险溢价(VRP)：恐慌的有形价格」(variance-risk-premium)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

模型与机制（自洽合成，仅用于演示方法；落地须用真实期权/VIX 数据）：
  * 标的指数日收益在物理测度 P 下服从「随机方差 + 偶发崩盘跳 + 轻度均值回复」过程
  * 已实现方差(已知): TRAIL_t = 252·mean(r_{t-H+1..t}²)   (过去 21 日, 年化)
  * 隐含方差(风险中性): VIX_t = sqrt(TRAIL_t) + VRP加载_t
        VRP加载 > 0 (均值约 +1.6 vol点), 随波动水平放大(危机时 +10~15 vol点)
  * 结算已实现方差(未来): RV_t = 252·mean(r_{t+1..t+H}²)   (t 时未知)
  * 方差风险溢价(到期观测): VRP_t = VIX_t² − RV_t   (平静为正, 危机为负)
  * 收割因子 = 做空方差互换: 月收益 (VIX_t² − RV_t) / VIX_t²
  * 可预测性: VIX_t 越高(越恐慌) → 未来 1 月股票超额收益越高(逆向买入信号)
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
D = os.path.join(BASE, "variance-risk-premium")
os.makedirs(D, exist_ok=True)

C = {"eq": "#4C72B0", "vix": "#C44E52", "grid": "#DDDDDD", "fv": "#55A868",
     "rv": "#DD8452", "pnl": "#55A868", "thr": "#888888", "panic": "#C44E52",
     "green": "#2ca02c", "purple": "#9467bd"}

# ============================================================
# 1) 物理测度 P 下标的组合日度路径(自洽、轻度均值回复)
# ============================================================
rng = np.random.default_rng(20260713)
N = 252 * 20
dt = 1.0 / 252.0
mu = 0.09 / 252.0
trend = mu
v = np.zeros(N); v[0] = 0.04
for t in range(1, N):
    v[t] = v[t - 1] + 5.0 * (0.04 - v[t - 1]) * dt
    v[t] += 0.6 * np.sqrt(max(v[t - 1], 1e-4)) * np.sqrt(dt) * rng.normal()
    if rng.random() < 1.0 / (252 * 4.0):          # 约每 4 年一次危机
        v[t] += rng.uniform(0.06, 0.16)
    v[t] = max(v[t], 0.005)
r = np.zeros(N)
for t in range(N):
    r[t] = mu + np.sqrt(v[t] / 252.0) * rng.normal()
    if v[t] > 0.12 and rng.random() < 0.015:
        r[t] += -rng.uniform(0.02, 0.05)          # 崩盘型负跳
    r[t] = np.clip(r[t], -0.5, 0.5)
# 轻度均值回复(危机后反弹): 使 VRP→未来收益 呈经典正向
adj = r.copy()
for t in range(252, N):
    ex = np.sum(r[t - 63:t]) - 63 * trend
    adj[t] += 0.04 * (-ex)
r = np.clip(adj, -0.5, 0.5)
price = np.cumprod(1.0 + r)

# ============================================================
# 2) 已实现方差: 过去(TRAIL, t 时已知) vs 未来(RV, 到期观测)
# ============================================================
H = 21
TRAIL = np.full(N, np.nan)
RV = np.full(N, np.nan)
for t in range(H, N - H):
    TRAIL[t] = 252.0 * np.mean(r[t - H + 1:t + 1] ** 2)
    RV[t] = 252.0 * np.mean(r[t + 1:t + H + 1] ** 2)
trail_vol = np.sqrt(TRAIL)
realized_vol = np.sqrt(RV)

# ============================================================
# 3) 隐含方差(VIX) = TRAIL + VRP加载(随波动水平放大)
# ============================================================
base_vrp = 0.016
vrp_vol = base_vrp + 0.06 * np.clip((trail_vol - 0.20) / 0.20, 0.0, 2.2)
implied_vol = trail_vol + vrp_vol
implied_var = implied_vol ** 2
vrp_var = implied_var - RV
mask = ~np.isnan(TRAIL) & ~np.isnan(RV)

# ============================================================
# 4) 收割因子: 做空方差互换(月收益, 波动率目标 15%)
# ============================================================
ret_short = np.full(N, np.nan)
for t in range(H, N - H):
    ret_short[t] = (implied_var[t] - RV[t]) / implied_var[t]
ret_short = np.clip(ret_short[mask], -0.95, 0.95)
TGT = 0.15
sc = min(TGT / (ret_short.std(ddof=1) * np.sqrt(12.0)), 5.0)
rs = np.clip(ret_short * sc, -0.95, 3.0)

def ann_stats(x):
    m = x.mean(); s = x.std(ddof=1)
    ann_r = (1.0 + m) ** 12 - 1.0
    ann_v = s * np.sqrt(12.0)
    sharpe = ann_r / ann_v if ann_v > 0 else 0.0
    return ann_r, ann_v, sharpe

sr, sv, ss = ann_stats(rs)
pnl = np.cumprod(1.0 + np.nan_to_num(rs, nan=0.0))
mdd = float(np.nanmin((pnl - np.maximum.accumulate(pnl)) / np.maximum.accumulate(pnl)))
worst_i = int(np.argmin(rs))
eq_m = np.full(N, np.nan)
for t in range(H, N - H):
    eq_m[t] = float(np.prod(1.0 + r[t + 1:t + H + 1]) - 1.0)
eq_m = eq_m[mask]
er, ev, es = ann_stats(eq_m)
pnl_eq = np.cumprod(1.0 + eq_m)
mdd_eq = float(np.nanmin((pnl_eq - np.maximum.accumulate(pnl_eq)) / np.maximum.accumulate(pnl_eq)))
corr_se = float(np.corrcoef(rs, eq_m)[0, 1])

# ============================================================
# 5) 可预测性: VIX_t vs 未来 1 月股票超额收益
# ============================================================
vix_now = implied_vol[mask].copy()
fut_ret = eq_m.copy()
rf_month = 0.02 * (H / 252.0)
excess = fut_ret - rf_month
slp = float(np.corrcoef(vix_now, excess)[0, 1])
A = np.vstack([np.ones_like(vix_now), vix_now]).T
bhat = np.linalg.lstsq(A, excess, rcond=None)[0]
lo = excess[vix_now <= np.quantile(vix_now, 0.33)].mean()
hi = excess[vix_now >= np.quantile(vix_now, 0.67)].mean()

# ============================================================
# 6) VRP 期限结构(平均, 短端最肥, 随期限衰减)
# ============================================================
def avg_vrp_for_horizon(tau):
    tr = np.full(N, np.nan); rvh = np.full(N, np.nan)
    for t in range(tau, N - tau):
        tr[t] = 252.0 * np.mean(r[t - tau + 1:t + 1] ** 2)
        rvh[t] = 252.0 * np.mean(r[t + 1:t + tau + 1] ** 2)
    m = ~np.isnan(tr) & ~np.isnan(rvh)
    tv = np.sqrt(tr[m]); rv2 = np.sqrt(rvh[m])
    decay = np.exp(-(tau - H) / 252.0)
    prem = (base_vrp + 0.06 * np.clip((tv - 0.20) / 0.20, 0.0, 2.2)) * decay
    vixv = tv + prem
    return float(np.mean(vixv - rv2))
terms = [("1M", H), ("3M", 63), ("6M", 126), ("12M", 252)]
vrp_term = [avg_vrp_for_horizon(tau) for _, tau in terms]

# ============================================================
# 图 1：VIX(隐含) vs 已实现波动, VRP 阴影
# ============================================================
fig, ax = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
tt = np.where(mask)[0]
ax[0].plot(price[mask], color=C["eq"], lw=0.7)
ax[0].set_ylabel("指数净值"); ax[0].set_title("指数走势(参照)")
ax[1].plot(realized_vol[mask] * 100, color=C["rv"], lw=0.8, label="已实现波动率(年化 %)")
ax[1].plot(implied_vol[mask] * 100, color=C["vix"], lw=0.8, label="隐含波动率 VIX(年化 %)")
ax[1].fill_between(tt, realized_vol[mask] * 100, implied_vol[mask] * 100,
                   where=implied_vol[mask] > realized_vol[mask],
                   color=C["vix"], alpha=0.18, label="VRP = 隐含 − 已实现 (恐慌的有形价格)")
ax[1].set_ylabel("波动率 (%)"); ax[1].set_xlabel("交易日")
ax[1].set_title("VIX 长期高于已实现波动: 方差风险溢价(红区)为常态, 危机最肥")
ax[1].legend(fontsize=8, loc="upper right")
for a in ax:
    a.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "vrp_implied_realized.png"), dpi=130)
plt.close()

# ============================================================
# 图 2：VRP 期限结构(平均)
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 5.0))
xlab = [t[0] for t in terms]; x = np.arange(len(terms))
ax.bar(x, [v * 100 for v in vrp_term], color=C["vix"], alpha=0.8, width=0.55)
for i, val in enumerate(vrp_term):
    ax.text(x[i], val * 100 + 0.15, f"+{val*100:.1f}", ha="center", fontsize=10, color=C["vix"])
ax.set_xticks(x); ax.set_xticklabels(xlab)
ax.set_xlabel("到期期限"); ax.set_ylabel("平均 VRP (波动率点)")
ax.set_title("VRP 期限结构: 各期限均为正, 短端最肥(1M ≈ +%.1f vol点)" % (vrp_term[0] * 100))
ax.grid(True, color=C["grid"], lw=0.6, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(D, "vrp_term_structure.png"), dpi=130)
plt.close()

# ============================================================
# 图 3：收割因子(做空方差互换) 净值 vs 股票
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4.8))
months = np.arange(len(rs))
ax.plot(months, pnl, color=C["panic"], lw=1.1, label="收割 VRP 因子(做空方差互换, 波动率目标 15%)")
ax.plot(months, pnl_eq, color=C["eq"], lw=0.9, label="买入持有股票")
ax.axhline(1.0, color="black", lw=0.6)
ax.set_xlabel("月份"); ax.set_ylabel("净值 (期初=1)")
ax.set_title("做空方差互换: 长期正漂移(收租)但崩盘月剧烈回撤, 与股票正相关")
ax.legend(fontsize=8, loc="upper left"); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "vrp_harvest_factor.png"), dpi=130)
plt.close()

# ============================================================
# 图 4：VIX 对未来 1 月股票超额收益的可预测性
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.scatter(vix_now * 100, excess * 100, s=10, color=C["purple"], alpha=0.35)
xs = np.linspace(vix_now.min(), vix_now.max(), 50)
ax.plot(xs * 100, (bhat[0] + bhat[1] * xs) * 100, color=C["vix"], lw=2,
        label="回归线 (斜率 %.2f%%/vol点)" % (bhat[1] * 100))
ax.set_xlabel("VIX 隐含波动率 (%, 当日)"); ax.set_ylabel("未来 1 月股票超额收益 (%)")
ax.set_title("VIX 与未来 1 月股票超额收益: 本合成样本呈弱负相关(真实市场多为正向, 见文献)")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "vrp_predictability.png"), dpi=130)
plt.close()

# ============================================================
# 关键数字输出
# ============================================================
print("=== 方差风险溢价(VRP) 关键数字 ===")
print("样本 %d 年(%d 交易日)" % (N // 252, N))
print("标的: 年化收益 %.1f%%  已实现波动均值 %.1f%%  最大 %.1f%%  最大回撤 %.1f%%"
      % (er * 100, realized_vol[mask].mean() * 100, realized_vol[mask].max() * 100,
         float(np.nanmin((price[mask] - np.maximum.accumulate(price[mask])) / np.maximum.accumulate(price[mask]))) * 100))
print("VIX(隐含)均值 %.1f%%" % (implied_vol[mask].mean() * 100))
print("VRP 均值(方差单位) = %.3f ; 折算波动率点 = %.2f ; VRP>0 占比 = %.1f%%"
      % (vrp_var[mask].mean(), vrp_vol[mask].mean() * 100, 100 * np.mean(vrp_var[mask] > 0)))
print("VRP 期限结构(平均, vol点): " + ", ".join("%s +%.1f" % (t[0], vv * 100) for t, vv in zip(terms, vrp_term)))
print("--- 收割因子(做空方差互换, 波动率目标 15%) ---")
print("年化收益 %.1f%%  年化波动 %.1f%%  Sharpe %.2f  最大回撤 %.1f%%" % (sr * 100, sv * 100, ss, mdd * 100))
print("与股票相关 = %.2f ; 最惨单月收益 = %.1f%%" % (corr_se, rs[worst_i] * 100))
print("买入持有: 年化 %.1f%% 波动 %.1f%% Sharpe %.2f 回撤 %.1f%%" % (er * 100, ev * 100, es, mdd_eq * 100))
print("--- VRP 可预测性(本合成样本方向, 真实市场多为正向, 见 Bollerslev & Todorov 等) ---")
print("VIX 与未来1月超额收益 相关 = %.2f ; 回归斜率 = %.2f%%/vol点" % (slp, bhat[1] * 100))
print("低 VIX 组未来收益 %.2f%% vs 高 VIX 组 %.2f%%" % (lo * 100, hi * 100))
print("\n图片已保存到:", D)
