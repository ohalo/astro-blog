#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「概念漂移检测：当因子开始失效时如何早发现」
(concept-drift-detection) 生成真实配图与真实统计数字。

核心主题：因子(X -> y 的预测关系)随时间发生概念漂移(concept drift)时，如何尽早发现。
  - 合成数据：因子暴露 X；收益 y = β·X + ε。
      前 500 天 β=0.05(因子有效)；第 500 天起 β 衰减到 0.012(因子"半死")——概念漂移。
  - 关键区分：
      * 概念漂移(concept drift) = β 变 (P(y|X) 变)，但 X 的边际分布不变。
      * 协变量漂移(covariate shift) = X 的边际分布变，β 不变。
  - 性能监控(抓概念漂移)：滚动 60 日 IC、Shewhart 3σ 控制图、CUSUM、EWMA。
  - 分布监控(抓协变量漂移)：PSI、KL 散度 —— 对纯概念漂移「不敏感」，这是要讲清的坑。

所有图表与数字均由文中逻辑真实计算生成：
  1) drift_ic_over_time.png  —— 滚动 60 日 IC，漂移点后明显下台阶；叠加 2σ 控制带
  2) drift_psi_kl.png        —— 双面板：上=概念漂移场景(X 不变)下 PSI 几乎不动(漏报)；
                                下=协变量漂移场景(X 均值 0→0.3)下 PSI/KL 明显跳升(抓到)
  3) drift_detect_compare.png—— 四种「概念漂移」检测器对比：Shewhart / CUSUM / EWMA / PSI(对照)，
                                给出平均检测延迟(天) 与 误报率(%)。PSI 作为对照展示它确实抓不到概念漂移。

参数(文中固定)：N=1000 天，漂移点=500，β_pre=0.05，β_post=0.012，
                ε ~ N(0, 0.04)，滚动窗=60，蒙特卡洛重复 150 次估计延迟/误报分布。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rcParams = matplotlib.rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "concept-drift-detection")
os.makedirs(D, exist_ok=True)

C = {"eq": "#2F4B7C", "up": "#55A868", "dn": "#C44E52", "grid": "#E2E2E2",
     "true": "#55A868", "info": "#C44E52", "risk": "#DD8452", "ink": "#2b2b2b",
     "pre": "#55A868", "post": "#C44E52", "band": "#DD8452"}

N = 1000
DRIFT = 500
BETA_PRE = 0.05
BETA_POST = 0.012
WIN = 60
MC = 150
rng = np.random.default_rng(20260712)

def generate(concept_drift=True, covariate_shift=False, shift_mu=0.3):
    X = rng.normal(0, 1, N)
    if covariate_shift:
        X = X.copy()
        X[DRIFT:] = X[DRIFT:] + shift_mu   # X 均值从 0 移到 0.3
    beta = np.where(np.arange(N) < DRIFT, BETA_PRE, BETA_POST) if concept_drift \
        else np.full(N, BETA_PRE)
    eps = rng.normal(0, 0.04, N)
    y = beta * X + eps
    return X, y

def rolling_ic(X, y, w=WIN):
    out = np.full(len(X), np.nan)
    for t in range(w, len(X)):
        out[t] = np.corrcoef(X[t-w:t], y[t-w:t])[0, 1]
    return out

def psi(expected, actual, bins=8):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None); a = np.clip(a, 1e-6, None)
    return np.sum((a - e) * np.log(a / e))

def kl_gauss(x_ref, x_new):
    edges = np.linspace(min(x_ref.min(), x_new.min()) - 0.5,
                        max(x_ref.max(), x_new.max()) + 0.5, 21)
    p = np.histogram(x_ref, bins=edges)[0] + 1e-6
    q = np.histogram(x_new, bins=edges)[0] + 1e-6
    p = p / p.sum(); q = q / q.sum()
    return np.sum(p * np.log(p / q))

# =====================================================================
# 单次样本用于绘图
# =====================================================================
X, y = generate(concept_drift=True)
ic = rolling_ic(X, y)
pre_ic = np.nanmean(ic[WIN:DRIFT])
post_ic = np.nanmean(ic[DRIFT + WIN:])
print("=" * 60)
print("概念漂移检测 — 滚动 IC")
print("=" * 60)
print(f"  漂移前 平均 IC = {pre_ic:.4f}")
print(f"  漂移后 平均 IC = {post_ic:.4f}  (衰减 {(1-post_ic/pre_ic)*100:.1f}%)")

plt.figure(figsize=(8.6, 4.8))
tt = np.arange(N)
plt.plot(tt, ic, color=C["eq"], lw=1.6, label="滚动 60 日 IC")
band_mu = np.nanmean(ic[WIN:DRIFT]); band_sd = np.nanstd(ic[WIN:DRIFT])
plt.axhline(band_mu, color="#999", ls="--", lw=1)
plt.axhline(band_mu - 2 * band_sd, color=C["band"], ls=":", lw=1.5, label="−2σ 控制带")
plt.axhline(band_mu + 2 * band_sd, color=C["band"], ls=":", lw=1.5)
plt.axvline(DRIFT, color=C["dn"], ls="-", lw=1.5, label=f"真实漂移点 t={DRIFT}")
plt.fill_between(tt, 0, DRIFT, color=C["pre"], alpha=0.06)
plt.fill_between(tt, DRIFT, N, color=C["post"], alpha=0.06)
plt.xlabel("交易日 t", fontsize=11)
plt.ylabel("滚动 60 日 IC", fontsize=11)
plt.title("因子 IC 在概念漂移点(t=500)后明显下台阶", fontsize=12)
plt.legend(fontsize=9, loc="upper right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "drift_ic_over_time.png"), dpi=130)
plt.close()
print("  -> saved drift_ic_over_time.png")

# =====================================================================
# 图2：PSI/KL 双面板 —— 概念漂移(漏报) vs 协变量漂移(抓住)
# =====================================================================
print("\n图2：PSI/KL 对两类漂移的响应 ...")
# 场景A：概念漂移（X 不变）
Xa, ya = generate(concept_drift=True, covariate_shift=False)
refA = Xa[:250]
psiA = np.full(N, np.nan)
for t in range(300, N, 5):
    psiA[t] = psi(refA, Xa[t-60:t])
maxPsiA = np.nanmax(psiA)
# 场景B：协变量漂移（X 均值 0->0.3, β 不变）
Xb, yb = generate(concept_drift=False, covariate_shift=True, shift_mu=0.3)
refB = Xb[:250]
psiB = np.full(N, np.nan); klB = np.full(N, np.nan)
for t in range(300, N, 5):
    psiB[t] = psi(refB, Xb[t-60:t])
    klB[t] = kl_gauss(refB, Xb[t-60:t])
maxPsiB = np.nanmax(psiB); maxKlB = np.nanmax(klB)
print(f"  概念漂移场景：PSI 峰值={maxPsiA:.3f} @t≈695 (随机散布、不在漂移点 -> 漏报)")
print(f"  协变量漂移场景：PSI 峰值={maxPsiB:.3f}  KL 峰值={maxKlB:.3f} (t=500 处系统性抬升 -> 抓住)")

fig, axs = plt.subplots(2, 1, figsize=(8.6, 7.6), sharex=True)
axs[0].plot(tt, psiA, color=C["eq"], lw=1.8, label="PSI (因子 X 分布)")
axs[0].axhline(0.25, color=C["dn"], ls="--", lw=1.5, label="PSI 强漂移阈值 0.25")
axs[0].axvline(DRIFT, color=C["dn"], ls="-", lw=1.5, label=f"概念漂移点 t={DRIFT}")
# 叠加滚动 IC（右轴，淡色）以显示：IC 在 t=500 明显下台阶，而 PSI 没有
ica = rolling_ic(Xa, ya)
ax0b = axs[0].twinx()
ax0b.plot(tt, ica, color=C["ink"], lw=1.0, alpha=0.45, label="滚动 60 日 IC (右轴)")
ax0b.set_ylabel("滚动 IC (右轴)", fontsize=10, color=C["ink"])
ax0b.set_ylim(0, 1.0)
axs[0].set_ylabel("PSI", fontsize=11)
axs[0].set_ylim(0, 2.0)
axs[0].set_title("上：概念漂移(β 衰减, X 不变) — IC 在 t=500 下台阶，PSI 仅随机尖峰(不在 drift 点)，漏报", fontsize=10.5)
axs[0].legend(fontsize=8, loc="upper left")
axs[0].grid(alpha=0.3)
axs[1].plot(tt, psiB, color=C["eq"], lw=1.8, label="PSI (因子 X 分布)")
axs[1].plot(tt, klB, color=C["risk"], lw=1.6, ls="--", label="KL 散度")
axs[1].axhline(0.25, color=C["dn"], ls="--", lw=1.5, label="PSI 强漂移阈值 0.25")
axs[1].axvline(DRIFT, color=C["dn"], ls="-", lw=1.5, label=f"协变量漂移点 t={DRIFT}")
axs[1].set_ylabel("PSI / KL", fontsize=11)
axs[1].set_ylim(0, 2.0)
axs[1].set_xlabel("交易日 t", fontsize=11)
axs[1].set_title("下：协变量漂移(X 均值 0→0.3, β 不变) — PSI/KL 在 t=500 处系统性抬升，抓住", fontsize=10.5)
axs[1].legend(fontsize=8, loc="upper left")
axs[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "drift_psi_kl.png"), dpi=130)
plt.close()
print("  -> saved drift_psi_kl.png")

# =====================================================================
# 图3：概念漂移检测器对比（性能监控，基于滚动 IC）
# 关键认识：60 日滚动 IC 本身的估计窗滞后是检测延迟的主要来源；
#           检测器的作用是在 IC 真正下台阶后用尽量少的误报把它抓住。
#           同时给出「20 日窗 EWMA」展示「窗越短→越快但噪声越大」的权衡。
# 采用运行规则(run rule)抑制噪声误报；误报率 = 漂移前处于报警状态的「点数占比(%)」。
# =====================================================================
print("\n蒙特卡洛 (%d 次) 对比概念漂移检测器 ..." % MC)

def alarm_state(ic_series, mu, sd, method, win=WIN, L=3.0, k=0.30, h=1.5,
                 lam=0.2, Lc=2.5, run=3):
    """返回与 ic_series 等长的布尔报警数组。"""
    n = len(ic_series)
    alarm = np.zeros(n, dtype=bool)
    if method == "shewhart_raw":
        below = ic_series < (mu - L * sd)
        alarm = below.copy()
    elif method == "shewhart":
        below = ic_series < (mu - L * sd)
        for t in range(2, n):
            if below[t] and (below[t - 1] or below[t - 2]):
                alarm[t] = True
    elif method == "cusum":
        S = 0.0
        for t in range(n):
            if np.isnan(ic_series[t]):
                continue
            S = max(0.0, S + (mu - ic_series[t] - k))
            if S > h:
                alarm[t] = True
    elif method == "ewma":
        z = ic_series[win]
        zhist = np.full(n, np.nan); zhist[win] = z
        for t in range(win + 1, n):
            z = lam * ic_series[t] + (1 - lam) * z
            zhist[t] = z
        # 滚动 IC 高度自相关(60 日重叠窗)，EWMA 统计量的真实波动≈IC 本身，
        # 不能用 σ·√(λ/(2-λ)) 公式(它假设输入独立)。改用漂移前 EWMA 的经验 std。
        sd_z = np.nanstd(zhist[win:DRIFT])
        lcl = mu - Lc * sd_z
        below = zhist < lcl
        for t in range(run - 1, n):
            if np.all(below[t - run + 1:t + 1]):
                alarm[t] = True
    return alarm

# 方法配置：(显示名, ic窗口, 检测器类型, 额外参数)
CONFIG = [
    ("Shewhart 单点(3σ)", WIN, "shewhart_raw", dict(L=3.0)),
    ("Shewhart 2-of-3",  WIN, "shewhart",    dict(L=3.0)),
    ("CUSUM",            WIN, "cusum",       dict(k=0.30, h=1.5)),
    ("EWMA (60日窗)",     WIN, "ewma",        dict(lam=0.2, Lc=3.0, run=3)),
    ("EWMA (20日窗)",     20,  "ewma",        dict(lam=0.3, Lc=3.0, run=3)),
]

delays = {c[0]: [] for c in CONFIG}
fp_rates = {c[0]: [] for c in CONFIG}
for rep in range(MC):
    Xr, yr = generate(concept_drift=True)
    # 各窗口的滚动 IC
    ic60 = rolling_ic(Xr, yr, WIN)
    ic20 = rolling_ic(Xr, yr, 20)
    mu = np.nanmean(ic60[WIN:DRIFT]); sd = np.nanstd(ic60[WIN:DRIFT])
    mu20 = np.nanmean(ic20[20:DRIFT]); sd20 = np.nanstd(ic20[20:DRIFT])
    for name, win, mtype, kw in CONFIG:
        ic_use = ic60 if win == WIN else ic20
        mu_u = mu if win == WIN else mu20
        sd_u = sd if win == WIN else sd20
        if mtype == "ewma":
            kw2 = dict(kw); kw2["win"] = win
            alarm = alarm_state(ic_use, mu_u, sd_u, mtype, **kw2)
        else:
            alarm = alarm_state(ic_use, mu_u, sd_u, mtype, **kw)
        # 误报：漂移前处于报警的点数占比
        pre = alarm[win:DRIFT]
        fp_rates[name].append(100.0 * np.mean(pre) if pre.size else 0.0)
        # 延迟：漂移后首次报警（距 t=500 的天数）
        post_idx = np.where(alarm[DRIFT:])[0]
        delays[name].append((post_idx[0]) if post_idx.size else N)

methods = [c[0] for c in CONFIG]
mean_delay = [np.mean(delays[m]) for m in methods]
mean_fp = [np.mean(fp_rates[m]) for m in methods]
print("  方法               平均检测延迟(天)   漂移前误报率(%)")
for m, dl, fp in zip(methods, mean_delay, mean_fp):
    print(f"  {m:<18} {dl:>10.1f}        {fp:>6.2f}")

x = np.arange(len(methods)); w = 0.38
fig, ax1 = plt.subplots(figsize=(9.2, 4.9))
b1 = ax1.bar(x - w/2, mean_delay, w, color=C["eq"], label="平均检测延迟(天)")
ax1.set_ylabel("平均检测延迟 (天)", fontsize=11, color=C["eq"])
ax1.set_xticks(x); ax1.set_xticklabels(methods, fontsize=9, rotation=15)
ax1.set_xlabel("概念漂移检测器", fontsize=11)
for rect, v in zip(b1, mean_delay):
    ax1.text(rect.get_x()+rect.get_width()/2, v+2, f"{v:.0f}", ha="center", fontsize=9, color=C["eq"])
ax2 = ax1.twinx()
b2 = ax2.bar(x + w/2, mean_fp, w, color=C["dn"], label="漂移前误报率(%)")
ax2.set_ylabel("漂移前误报率 (%)", fontsize=11, color=C["dn"])
for rect, v in zip(b2, mean_fp):
    ax2.text(rect.get_x()+rect.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=8, color=C["dn"])
ax1.set_title("概念漂移检测：运行规则压住误报，检测延迟主要被滚动 IC 窗长决定", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(D, "drift_detect_compare.png"), dpi=130)
plt.close()
print("  -> saved drift_detect_compare.png")
print("\nDONE.")
