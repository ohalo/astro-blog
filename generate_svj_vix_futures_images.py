#!/usr/bin/env python3
"""
为文章「随机波动率带跳跃与 VIX 期货定价：把恐慌也做成因子」(svj-vix-futures)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

模型与机制（自洽合成，仅用于演示方法；落地见文末路径）：
  * 标的指数收益服从 Bates 型 SVJ：Heston 随机方差 v_t + 复合泊松跳（崩盘型负跳）。
    - 方差 CIR:  dv = κ(θ-v)dt + σ_v√v dW
    - 收益:      d lnS = (r - 0.5 v - λm_J)dt + √v dZ + J_t,  ρ(J,Z)=-0.7
    - 跳: 强度 λ/年, 单跳对数收益 ~ N(μJ, σJ²), m_J=E[e^J-1]
  * VIX(模型公允) = 100·√(30日远期方差互换率), 在 Heston 下对 v_t 仿射解析:
        VS30(v) = (1/T30)[ θ·T30 + (v-θ)(1-e^{-κT30})/κ ],  T30=30/252
  * VIX 期货 F_t(τ) = E^Q[ VIX_{t+τ} ] + VRP楔差(τ); 楔差随期限衰减→0（近端最肥的波动率风险溢价）。
  * 「恐慌因子」= 做空 VIX 期货(收割 VRP)的月度收益: r = (F_t(21) - VIX_{t+21}) / F_t(21)。
    均值>0(溢价)但崩盘月 VIX 飙升→巨额负收益(你承担的恐慌风险)；与股票负相关→可分散的风险因子。
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
D = os.path.join(BASE, "svj-vix-futures")
os.makedirs(D, exist_ok=True)

C = {"eq": "#4C72B0", "vix": "#C44E52", "grid": "#DDDDDD", "fv": "#55A868",
     "rv": "#DD8452", "pnl": "#55A868", "thr": "#888888", "panic": "#C44E52"}

# ============================================================
# 1) 参数
# ============================================================
S0 = 100.0
r = 0.02
v0, kappa, theta, sigma_v, rho = 0.04, 3.0, 0.04, 0.4, -0.7   # Heston(SV)
lam, muJ, sigJ = 0.6, -0.09, 0.11                             # 崩盘型负跳
mJ = np.exp(muJ + 0.5 * sigJ ** 2) - 1.0
MU_EQ = 0.09                # 股票股权风险溢价(年化, 使指数有正漂移)
T30 = 30.0 / 252.0
VRP0 = 1.6          # 波动率风险溢价楔差总量(vol 点, 近端)
TAU_VRP = 63.0      # 楔差衰减时标(交易日)

YEARS = 20
NDAY = 252 * YEARS
dt = 1.0 / 252.0


def fair_vix(v):
    """模型公允 VIX = 100·√(30日远期方差, 仿射于 v)。"""
    vs = (theta * T30 + (v - theta) * (1.0 - np.exp(-kappa * T30)) / kappa) / T30
    return 100.0 * np.sqrt(np.maximum(vs, 1e-6))


def vrp_wedge(tau_days):
    """期限 τ(交易日) 上的 VRP 楔差(vol 点), 短端最肥、τ→0 趋于 0。"""
    return VRP0 * (1.0 - np.exp(-tau_days / TAU_VRP))


# ============================================================
# 2) 模拟 SVJ 日度路径
# ============================================================
def simulate(seed=20260712):
    rng = np.random.default_rng(seed)
    steps = NDAY
    v = np.empty(steps + 1); v[0] = v0
    lnS = np.empty(steps + 1); lnS[0] = np.log(S0)
    jump_day = np.zeros(steps + 1, dtype=bool)
    Z1 = rng.standard_normal(steps); Z2 = rng.standard_normal(steps)
    e = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2
    for t in range(steps):
        vs = max(v[t], 0.0)
        v[t + 1] = v[t] + kappa * (theta - vs) * dt + sigma_v * np.sqrt(vs) * np.sqrt(dt) * e[t]
        v[t + 1] = max(v[t + 1], 1e-6)
        j = 0.0
        Nj = rng.poisson(lam * dt)
        if Nj > 0:
            j = Nj * muJ + np.sqrt(Nj) * sigJ * rng.standard_normal()
            jump_day[t + 1] = True
        drift = (MU_EQ - 0.5 * vs - lam * mJ) * dt
        lnS[t + 1] = lnS[t] + drift + np.sqrt(vs) * np.sqrt(dt) * Z1[t] + j
    return lnS, v, jump_day


lnS, v, jump_day = simulate()
ret = np.diff(lnS)                     # 日收益
price = np.exp(lnS)
vix_fair = fair_vix(v)                 # 模型公允 VIX(含跳前的扩散隐含)
vix_fair_t = vix_fair[1:]              # 长度 5040, 与 ret/fv21/rv 对齐

# ============================================================
# 3) 蒙特卡洛: E^Q[ VIX_{t+τ} | v_t ]  (CIR 模拟方差, 取终态 VIX 平均)
# ============================================================
def mc_expected_vix(v0_scalar, tau_days, n_paths=1500, seed=7):
    if tau_days <= 0:
        return fair_vix(v0_scalar)
    n_steps = max(1, int(round(tau_days)))
    dts = tau_days / n_steps / 252.0
    rng = np.random.default_rng(seed)
    vv = np.full(n_paths, v0_scalar)
    Z = rng.standard_normal((n_steps, n_paths))
    for s in range(n_steps):
        vs = np.maximum(vv, 0.0)
        vv = vv + kappa * (theta - vs) * dts + sigma_v * np.sqrt(vs) * np.sqrt(dts) * Z[s]
        vv = np.maximum(vv, 1e-6)
    return float(np.mean(fair_vix(vv)))


# 远端锚(长期中枢 VIX): 用于期限结构图的基准线
vix_longrun = fair_vix(theta)          # ≈ 100·√θ·... = 100·√(θ) 的等价(实际是 30日远期在 v=θ)


# ============================================================
# 4) 「恐慌因子」(做空 VIX 期货收割 VRP) 月度收益
# ============================================================
H = 21                                 # 1 个月 = 21 交易日
entry_idx = list(range(0, NDAY - H, H))
r_short, r_long, eq_m, vix_settle_m = [], [], [], []
for d in entry_idx:
    entry = mc_expected_vix(v[d], H, n_paths=1500, seed=11 + (d // H) % 97) + vrp_wedge(H)
    settle = vix_fair[d + H]          # 到期时 = 即月 VIX(楔差在 τ→0 趋于0)
    rs = (entry - settle) / entry     # 做空 VIX 期货收益(收割 VRP)
    r_short.append(rs)
    r_long.append(-rs)
    eq_m.append(float(np.prod(1.0 + ret[d:d + H]) - 1.0))   # 同期股票月度收益
    vix_settle_m.append(settle)
r_short = np.clip(np.array(r_short), -0.99, 3.0)
r_long = -r_short
eq_m = np.array(eq_m)
# 波动率目标缩放(标准做法: 原始头寸波动极高, 按 15% 年化缩放, Sharpe 不变)
TGT = 0.15
sc = TGT / (r_short.std(ddof=1) * np.sqrt(12.0))
r_short_s = r_short * sc          # 缩放后做空 VIX 期货(VRP 因子)
r_long_s = r_long * sc           # 缩放后做多 VIX 期货(恐慌对冲因子)


def ann_stats(x):
    m = x.mean(); s = x.std(ddof=1)
    ann_r = (1.0 + m) ** 12 - 1.0
    ann_v = s * np.sqrt(12.0)
    sharpe = (ann_r - 0.0) / ann_v if ann_v > 0 else 0.0
    return ann_r, ann_v, sharpe


sr, sv, ss = ann_stats(r_short_s)
er, ev, es = ann_stats(eq_m)
corr_ev = np.corrcoef(r_short_s, eq_m)[0, 1]
# 恐慌因子最大回撤(做空版净值, 缩放后)
pnl_short = np.cumprod(1.0 + r_short_s)
mdd_short = float(np.min((pnl_short - np.maximum.accumulate(pnl_short))
                         / np.maximum.accumulate(pnl_short)))
# 崩盘月: 找出做空因子最惨的一个月 & 股票最惨月
worst_short_idx = int(np.argmin(r_short_s))
best_short_idx = int(np.argmax(r_short_s))
# 合并组合 90% 股票 + 10% 恐慌对冲因子(保险, 缩放后)
w_eq, w_pf = 0.90, 0.10
comb_m = w_eq * eq_m + w_pf * r_long_s
cr, cv, cs = ann_stats(comb_m)
pnl_comb = np.cumprod(1.0 + comb_m)
mdd_comb = float(np.min((pnl_comb - np.maximum.accumulate(pnl_comb))
                        / np.maximum.accumulate(pnl_comb)))
pnl_eq = np.cumprod(1.0 + eq_m)
mdd_eq = float(np.min((pnl_eq - np.maximum.accumulate(pnl_eq))
                      / np.maximum.accumulate(pnl_eq)))


# ============================================================
# 图 1：指数价格路径(标崩盘点) + 模型 VIX 路径
# ============================================================
fig, ax = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
jd = np.where(jump_day)[0]
ax[0].plot(price, color=C["eq"], lw=0.8)
ax[0].scatter(jd, price[jd], color=C["panic"], s=12, zorder=5, label="跳空(崩盘型)")
ax[0].set_ylabel("指数点位"); ax[0].legend(loc="upper left", fontsize=8)
ax[0].set_title("SVJ 模拟：随机波动(曲曲折折) + 崩盘跳(红点) 同时存在")
ax[1].plot(vix_fair, color=C["vix"], lw=0.8)
ax[1].set_ylabel("模型 VIX (%)"); ax[1].set_xlabel("交易日")
ax[1].axhline(vix_longrun, color=C["thr"], ls="--", lw=1, label="长期中枢 ≈%.0f" % vix_longrun)
ax[1].legend(loc="upper right", fontsize=8)
for a in ax:
    a.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "svj_equity_vol_paths.png"), dpi=130)
plt.close()


# ============================================================
# 图 2：VIX 期货期限结构(平静日 vs 恐慌日)
# ============================================================
mats = np.array([21, 42, 63, 84, 105, 126])
# 平静日: v 处于约 12% 分位(低波动但不极端); 恐慌日: v 最高(跳后)
v_all = v[H:(NDAY - H)]
tgt = np.quantile(v_all, 0.12)
calm_idx = int(H + np.argmin(np.abs(v_all - tgt)))
panic_idx = int(np.argmax(v[(H):(NDAY - H)])) + H
xlab = ["1M", "2M", "3M", "4M", "5M", "6M"]; x = np.arange(len(mats))
curves = []
for idx, tag, col in [(calm_idx, "平静日(低波动, Contango)", C["fv"]),
                     (panic_idx, "恐慌日(高波动, Backwardation)", C["panic"])]:
    ys = [mc_expected_vix(v[idx], tau, n_paths=1500, seed=3) + vrp_wedge(tau) for tau in mats]
    curves.append((ys, tag, col, idx))
    # 同时画公允(无楔差)曲线做对照
    ys_fair = [mc_expected_vix(v[idx], tau, n_paths=1500, seed=3) for tau in mats]
    axp = None
fig, ax = plt.subplots(figsize=(10, 5.2))
for ys, tag, col, idx in curves:
    ax.plot(x, ys, color=col, lw=2, marker="o", label="%s  VIX=%.1f" % (tag, vix_fair[idx]))
ax.axhline(vix_longrun, color=C["thr"], ls="--", lw=1, label="长期中枢 ≈%.0f" % vix_longrun)
ax.set_xticks(x); ax.set_xticklabels(xlab)
ax.set_xlabel("到期期限"); ax.set_ylabel("VIX 期货价格 (波动率 %)")
ax.set_title("VIX 期货期限结构：平静日整体升水(Contango), 恐慌日近端倒挂(Backwardation)")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "svj_vix_futures_curve.png"), dpi=130)
plt.close()


# ============================================================
# 图 3：VIX 期货相对即期的溢价(Contango/Backwardation) 时序
# ============================================================
# 实现波动: 21日滚动年化(含跳) — 仅作对照展示
rv = np.full(NDAY, np.nan)
sq = ret ** 2
for t in range(H, NDAY):
    rv[t] = 100.0 * np.sqrt(252.0 * sq[t - H:t].mean())
# 近端期货 F_t(21) 与即期公允 VIX 的价差 = VRP(隐含溢价)
fv21 = np.full(NDAY, np.nan)
step = 10
for t in range(H, NDAY, step):
    fv21[t] = mc_expected_vix(v[t], H, n_paths=800, seed=5) + vrp_wedge(H)
mask = ~np.isnan(rv) & ~np.isnan(fv21)
vrp_ts = fv21[mask] - vix_fair_t[mask]      # 期货 − 即期: >0 即 Contango(做空赚)
pos_share = float(np.mean(vrp_ts > 0))
fig, ax = plt.subplots(figsize=(11, 4.6))
tt = np.where(mask)[0]
ax.plot(tt, fv21[mask], color=C["fv"], lw=0.8, label="VIX 期货(近端 1M)")
ax.plot(tt, vix_fair_t[mask], color=C["vix"], lw=0.8, label="即期公允 VIX")
ax.fill_between(tt, vix_fair_t[mask], fv21[mask], where=fv21[mask] > vix_fair_t[mask],
               color=C["fv"], alpha=0.20, label="Contango: 期货>即期, 做空赚溢价")
ax.set_xlabel("交易日"); ax.set_ylabel("波动率 (%)")
ax.set_title("VIX 期货相对即期长期溢价(Contango): %.0f%% 交易日期货高于即期, 均值 +%.2f vol点"
             % (100 * pos_share, vrp_ts.mean()))
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "svj_vrp_timeseries.png"), dpi=130)
plt.close()


# ============================================================
# 图 4：恐慌(波动率风险溢价)因子净值 vs 股票 + 合并组合
# ============================================================
# VRP 因子 = 做空 VIX 期货(收割溢价): 顺周期, 与股票正相关, 但崩盘月同跌
pnl_short = np.cumprod(1.0 + r_short_s)
mdd_short = float(np.min((pnl_short - np.maximum.accumulate(pnl_short))
                         / np.maximum.accumulate(pnl_short)))
corr_ev = np.corrcoef(r_short_s, eq_m)[0, 1]
# 恐慌对冲因子 = 做多 VIX 期货(保险): 与股票负相关, 崩盘月飙升, 平静月微 bleed
pnl_long = np.cumprod(1.0 + r_long_s)
mdd_long = float(np.min((pnl_long - np.maximum.accumulate(pnl_long))
                        / np.maximum.accumulate(pnl_long)))
corr_el = np.corrcoef(r_long_s, eq_m)[0, 1]      # 应为负(对冲)
# 合并组合 A: 90% 股票 + 10% VRP 因子(把它当"赚取溢价的因子"配进去)
comb_vrp = 0.90 * eq_m + 0.10 * r_short_s
cr, cv, cs = ann_stats(comb_vrp)
pnl_comb = np.cumprod(1.0 + comb_vrp)
mdd_comb = float(np.min((pnl_comb - np.maximum.accumulate(pnl_comb))
                        / np.maximum.accumulate(pnl_comb)))
# 合并组合 B(对照): 90% 股票 + 10% 恐慌对冲因子(保险)
comb_hedge = 0.90 * eq_m + 0.10 * r_long_s
chr_, chv_, chs_ = ann_stats(comb_hedge)
mdd_hedge = float(np.min((np.cumprod(1.0 + comb_hedge) - np.maximum.accumulate(np.cumprod(1.0 + comb_hedge)))
                            / np.maximum.accumulate(np.cumprod(1.0 + comb_hedge))))
pnl_eq = np.cumprod(1.0 + eq_m)
mdd_eq = float(np.min((pnl_eq - np.maximum.accumulate(pnl_eq))
                      / np.maximum.accumulate(pnl_eq)))
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(pnl_short, color=C["panic"], lw=1.2, label="恐慌因子(做空VIX期货, 收割VRP) 净值")
ax.plot(pnl_eq, color=C["eq"], lw=1.0, label="买入持有股票 净值")
ax.plot(pnl_comb, color=C["fv"], lw=1.0, ls="--", label="组合 90%%股+10%%恐慌因子")
ax.axhline(1.0, color="black", lw=0.6)
ax.set_xlabel("月份"); ax.set_ylabel("净值 (期初=1)")
ax.set_title("恐慌因子：顺周期(与股票正相关)但有正溢价; 配 10%% 提升组合 Sharpe")
ax.legend(fontsize=8, loc="upper left"); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "svj_panic_factor.png"), dpi=130)
plt.close()


# ============================================================
# 关键数字输出
# ============================================================
print("=== SVJ + VIX 期货定价 关键数字 ===")
print("样本 %d 年(%d 交易日), Heston: v0=%.3f κ=%.1f θ=%.3f σv=%.2f ρ=%.2f"
      % (YEARS, NDAY, v0, kappa, theta, sigma_v, rho))
print("跳: λ=%.2f/年 μJ=%.3f σJ=%.3f (崩盘型负跳)" % (lam, muJ, sigJ))
print("长期中枢 VIX(fair) ≈ %.1f ; 实测 VIX 均值 %.1f / 最大 %.1f"
      % (vix_longrun, vix_fair.mean(), vix_fair.max()))
print("跳空次数(全样本) = %d, 占比 %.2f%%" % (int(jump_day.sum()), 100 * jump_day.mean()))
# 期货曲线快照数字
calm_curve = [mc_expected_vix(v[calm_idx], tau, 1500, 3) + vrp_wedge(tau) for tau in mats]
panic_curve = [mc_expected_vix(v[panic_idx], tau, 1500, 3) + vrp_wedge(tau) for tau in mats]
print("平静日(第%d) VIX=%.1f, 期货1M→6M: %s"
      % (calm_idx, vix_fair[calm_idx], ", ".join("%.1f" % x for x in calm_curve)))
print("恐慌日(第%d) VIX=%.1f, 期货1M→6M: %s"
      % (panic_idx, vix_fair[panic_idx], ", ".join("%.1f" % x for x in panic_curve)))
print("VRP 时序(期货−即期)均值 = %.2f vol点, %.1f%% 交易日 Contango(期货>即期)"
      % (vrp_ts.mean(), 100 * pos_share))
print("--- 恐慌风险溢价因子(做空VIX期货, 月度) ---")
print("年化收益 %.1f%%  年化波动 %.1f%%  Sharpe %.2f  最大回撤 %.1f%%"
      % (sr * 100, sv * 100, ss, mdd_short * 100))
print("与股票相关 = %.2f (正: 它是顺周期的风险因子, 危机关头与你同跌)" % corr_ev)
print("最惨月(第%d月): 因子收益 %.1f%%, 当股票收益 %.1f%%"
      % (worst_short_idx, r_short_s[worst_short_idx] * 100, eq_m[worst_short_idx] * 100))
print("最好月(第%d月): 因子收益 %.1f%%" % (best_short_idx, r_short_s[best_short_idx] * 100))
print("--- 恐慌对冲因子(做多VIX期货, 镜像) ---")
lr, lv, ls = ann_stats(r_long_s)
print("年化收益 %.1f%%  年化波动 %.1f%%  Sharpe %.2f  最大回撤 %.1f%%  与股票相关 %.2f"
      % (lr * 100, lv * 100, ls, mdd_long * 100, corr_el))
print("--- 组合对照: 90%%股 + 10%%恐慌因子 ---")
print("纯股:            年化 %.1f%% 波动 %.1f%% Sharpe %.2f 回撤 %.1f%%"
      % (er * 100, ev * 100, es, mdd_eq * 100))
print("  +10%% VRP因子(做空):   年化 %.1f%% 波动 %.1f%% Sharpe %.2f 回撤 %.1f%%"
      % (cr * 100, cv * 100, cs, mdd_comb * 100))
print("  +10%% 对冲因子(做多):   年化 %.1f%% 波动 %.1f%% Sharpe %.2f 回撤 %.1f%%"
      % (chr_ * 100, chv_ * 100, chs_, mdd_hedge * 100))
print("\n图片已保存到:", D)
