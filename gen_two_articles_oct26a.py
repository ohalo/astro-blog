#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""oct26a (v3 final): 10 figures + key stats for
A) monte-carlo-variance-reduction  蒙特卡洛方差缩减：从对偶变量到重要性采样的收敛加速
B) cpt-portfolio-choice            累积前景理论组合选择：损失厌恶与概率加权如何改写最优配置

v3 修正:
- A: 分层抽样用正确的 Brownian bridge（减样本均值 + 加 c/n）；稀有事件改到 99.99 分位
     （p≈1e-4），θ 扫描选优（P_q(event)≈3-12% 区间最优，过度倾斜反而崩溃）。
- B: 市场定为 mid 配置（3%@+18% 彩票），λ=1→股票 / λ=2.25→彩票 / λ=5→杠铃 的非单调
     模式；新增受限 CPT（权重上限 25%）作为实用解；经验 CPT 噪声用 5 条独立路径。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import qmc, norm
from scipy.optimize import minimize

for f in ["Heiti TC", "PingFang SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130
rcParams["figure.facecolor"] = "white"
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.3

BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT_A = os.path.join(BASE, "monte-carlo-variance-reduction")
OUT_B = os.path.join(BASE, "cpt-portfolio-choice")
os.makedirs(OUT_A, exist_ok=True)
os.makedirs(OUT_B, exist_ok=True)

RED, GREEN, BLUE, GRAY, ORANGE, BLACK, PURPLE = "#d63031", "#00b894", "#0984e3", "#636e72", "#e17055", "#2d3436", "#6c5ce7"
rng = np.random.default_rng(20261026)


# ============================================================
# Article A: Monte Carlo variance reduction
# ============================================================
S0, K, r_rate, sigma, T = 100.0, 100.0, 0.05, 0.30, 1.0
N_STEPS, DT = 12, 1 / 12
MU = r_rate - 0.5 * sigma ** 2
DISC = np.exp(-r_rate * T)
SQDT = np.sqrt(DT)

mG = np.log(S0) + MU * T * (N_STEPS + 1) / (2 * N_STEPS)
sG2 = sigma ** 2 * T * (N_STEPS + 1) * (2 * N_STEPS + 1) / (6 * N_STEPS ** 2)
sG = np.sqrt(sG2)
d1g = (mG - np.log(K) + sG2) / sG
d2g = d1g - sG
GEO_PRICE = DISC * (np.exp(mG + sG2 / 2) * norm.cdf(d1g) - K * norm.cdf(d2g))
GEO_PAYOFF = GEO_PRICE / DISC
print(f"[A] geometric Asian: price={GEO_PRICE:.6f}, E[payoff]={GEO_PAYOFF:.6f}")


def paths_from_Z(Z):
    return np.log(S0) + np.cumsum(MU * DT + sigma * SQDT * Z, axis=1)


def arith_avg(Z):
    return np.exp(paths_from_Z(Z)).mean(axis=1)


def arith_payoff(Z):
    return np.maximum(arith_avg(Z) - K, 0.0)


def geom_payoff(Z):
    return np.maximum(np.exp(paths_from_Z(Z).mean(axis=1)) - K, 0.0)


# ---- pilot (CV beta, payoff std, A-W_T correlation) ----
N_PILOT = 1 << 18
Zp = rng.standard_normal((N_PILOT, N_STEPS))
Xp, Yp = arith_payoff(Zp), geom_payoff(Zp)
Ap = arith_avg(Zp)
BETA_CV = np.cov(Xp, Yp)[0, 1] / np.var(Yp)
RHO_CV = np.corrcoef(Xp, Yp)[0, 1]
PAYOFF_STD = Xp.std()
WT = paths_from_Z(Zp)[:, -1]
R2_WT = np.corrcoef(Ap, WT)[0, 1] ** 2
print(f"[A] corr(arith,geom)={RHO_CV:.6f}, beta*={BETA_CV:.6f}, payoff std={PAYOFF_STD:.3f}, "
      f"R2(A|W_T)={R2_WT:.3f}")

# ---- reference price (CV, 同口径), N=2^23 chunked ----
N_REF, CHUNK = 1 << 23, 1 << 18
sx = sy = n = 0.0
while n < N_REF:
    m = min(CHUNK, N_REF - n)
    Zc = rng.standard_normal((m, N_STEPS))
    Xc, Yc = arith_payoff(Zc), geom_payoff(Zc)
    sx += Xc.sum(); sy += Yc.sum(); n += m
REF_PRICE = DISC * (sx / n - BETA_CV * (sy / n - GEO_PAYOFF))
print(f"[A] reference arithmetic Asian price = {REF_PRICE:.6f}")

# ---- convergence study ----
NS = [256, 1024, 4096, 16384, 65536]
R_REP, N_STRATA = 48, 16


def mc_naive(N):
    return DISC * arith_payoff(rng.standard_normal((N, N_STEPS))).mean()


def mc_antithetic(N):
    Z = rng.standard_normal((N // 2, N_STEPS))
    return DISC * (0.5 * (arith_payoff(Z) + arith_payoff(-Z))).mean()


def mc_cv(N):
    Z = rng.standard_normal((N, N_STEPS))
    X, Y = arith_payoff(Z), geom_payoff(Z)
    return DISC * (X.mean() - BETA_CV * (Y.mean() - GEO_PAYOFF))


def mc_stratified(N, k=N_STRATA):
    per = N // k
    tot = 0.0
    for j in range(k):
        U = (j + rng.random(per)) / k
        WTj = norm.ppf(U) * np.sqrt(T)
        Zf = rng.standard_normal((per, N_STEPS))
        Zf = Zf - Zf.mean(axis=1, keepdims=True) + ((WTj / SQDT) / N_STEPS)[:, None]
        tot += arith_payoff(Zf).mean()
    return DISC * tot / k


def mc_rqmc(N, m_pow):
    sob = qmc.Sobol(d=N_STEPS, scramble=True, seed=int(rng.integers(1 << 31)))
    return DISC * arith_payoff(norm.ppf(sob.random_base2(m_pow))).mean()


conv = {m: [] for m in ["naive", "anti", "cv", "strat", "rqmc"]}
for N in NS:
    m_pow = int(round(np.log2(N)))
    for rep in range(R_REP):
        conv["naive"].append((N, mc_naive(N)))
        conv["anti"].append((N, mc_antithetic(N)))
        conv["cv"].append((N, mc_cv(N)))
        conv["strat"].append((N, mc_stratified(N)))
        conv["rqmc"].append((N, mc_rqmc(N, m_pow)))

rmse = {}
for m, vals in conv.items():
    arr = np.array(vals)
    rmse[m] = [float(np.sqrt(np.mean((arr[arr[:, 0] == N, 1] - REF_PRICE) ** 2))) for N in NS]
for m in ["naive", "anti", "cv", "strat", "rqmc"]:
    slope = np.polyfit(np.log(NS[-3:]), np.log(rmse[m][-3:]), 1)[0]
    print(f"[A] {m:6s} RMSE: {[f'{v:.2e}' for v in rmse[m]]} slope={slope:.2f}")

vr = {}
arr = np.array(conv["naive"]); v0 = np.var(arr[arr[:, 0] == 4096, 1])
for m in ["anti", "cv", "strat", "rqmc"]:
    a = np.array(conv[m]); vr[m] = v0 / np.var(a[a[:, 0] == 4096, 1])
print(f"[A] VR@4096: anti={vr['anti']:.1f} cv={vr['cv']:.1f} (theory {1/(1-RHO_CV**2):.0f}) "
      f"strat={vr['strat']:.1f} (theory {1/(1-R2_WT):.1f}) rqmc={vr['rqmc']:.1f}")

# --- Figure A1: convergence ---
fig, ax = plt.subplots(figsize=(7.6, 5.2))
labels = {"naive": ("朴素 MC", RED), "anti": ("对偶变量", ORANGE), "cv": ("控制变量", BLUE),
          "strat": ("分层抽样(W_T)", PURPLE), "rqmc": ("Sobol RQMC", GREEN)}
for m, (lab, c) in labels.items():
    ax.loglog(NS, rmse[m], "o-", color=c, label=lab, lw=1.8, ms=5)
nn = np.array(NS, dtype=float)
ax.loglog(nn, rmse["naive"][0] * (nn / NS[0]) ** -0.5, "--", color=GRAY, lw=1.2, label="N$^{-1/2}$ 参考线")
ax.loglog(nn, rmse["rqmc"][0] * (nn / NS[0]) ** -1.0, ":", color=GRAY, lw=1.2, label="N$^{-1}$ 参考线")
ax.set_xlabel("路径数 N"); ax.set_ylabel("价格估计 RMSE（48 次重复）")
ax.set_title("算术亚式期权定价：五种估计量的收敛速度（参考价 %.4f）" % REF_PRICE)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT_A, "convergence_comparison.png")); plt.close(fig)

# --- Figure A2: control variate scatter ---
idx = rng.choice(N_PILOT, 3000, replace=False)
fig, ax = plt.subplots(figsize=(6.6, 5.6))
ax.scatter(Yp[idx], Xp[idx], s=6, alpha=0.45, color=BLUE, edgecolors="none")
xs = np.linspace(0, float(Yp[idx].max()), 50)
ax.plot(xs, BETA_CV * xs + (Xp[idx].mean() - BETA_CV * Yp[idx].mean()), color=RED, lw=2,
        label=f"回归斜率 β*={BETA_CV:.3f}")
ax.plot(xs, xs, color=GRAY, lw=1.2, ls="--", label="45° 参考线")
ax.set_xlabel("几何亚式回报（控制变量，有闭式解）"); ax.set_ylabel("算术亚式回报（目标）")
ax.set_title(f"控制变量生效的根源：两条收益几乎同向（ρ={RHO_CV:.4f}）")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT_A, "control_variate_scatter.png")); plt.close(fig)

# --- Figure A3: Sobol vs random ---
sob2 = qmc.Sobol(d=2, scramble=True, seed=7)
sob_pts = sob2.random(256)
rnd_pts = rng.random((256, 2))
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
axes[0].scatter(rnd_pts[:, 0], rnd_pts[:, 1], s=10, color=RED, alpha=0.7)
axes[0].set_title("伪随机点（256 个）"); axes[0].set_xlabel("第 1 维"); axes[0].set_ylabel("第 2 维")
axes[1].scatter(sob_pts[:, 0], sob_pts[:, 1], s=10, color=BLUE, alpha=0.7)
axes[1].set_title("Sobol 序列（256 个）"); axes[1].set_xlabel("第 1 维")
for a in axes:
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.grid(alpha=0.25)
fig.suptitle("低差异序列：每个小方格都被均匀拜访（RQMC 更快收敛的几何直觉）", y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(OUT_A, "qmc_vs_random.png"), bbox_inches="tight"); plt.close(fig)

# ---- rare event: P(A > H), H = 99.99th percentile ----
N_PILOT_R = 1 << 22
ApR = arith_avg(rng.standard_normal((N_PILOT_R, N_STEPS)))
H = round(float(np.quantile(ApR, 0.9999)), 1)
p_true = float(np.mean(ApR > H))
print(f"[A] rare event: H={H}, p_true={p_true:.3e} (pilot N={N_PILOT_R})")


def mc_rare_naive(N):
    return float(np.mean(arith_avg(rng.standard_normal((N, N_STEPS))) > H))


def mc_rare_is(N, theta):
    Y = rng.standard_normal((N, N_STEPS)) + theta
    hit = arith_avg(Y) > H
    lr = np.exp(-theta * Y.sum(axis=1) + 0.5 * N_STEPS * theta ** 2)
    return float(np.mean(lr * hit)), float(np.mean(hit))


# theta scan at N=16384
THETAS = [0.3, 0.6, 1.0, 1.4]
N_SCAN, R_SCAN = 16384, 16
scan = {}
base_var = np.var([mc_rare_naive(N_SCAN) for _ in range(R_SCAN)])
for th in THETAS:
    ests, evs = [], []
    for _ in range(R_SCAN):
        e, ev = mc_rare_is(N_SCAN, th)
        ests.append(e); evs.append(ev)
    scan[th] = (float(np.mean(ests)), float(np.var(ests)), float(np.mean(evs)))
    print(f"[A] theta scan: theta={th}: mean={scan[th][0]:.3e} var={scan[th][1]:.3e} "
          f"P_q(event)={scan[th][2]:.3f} VR={base_var/max(scan[th][1],1e-30):.0f}x")
THETA = min(THETAS, key=lambda t: scan[t][1])
print(f"[A] chosen theta={THETA}")

NS_R = [1024, 4096, 16384, 65536, 262144]
R_R = 24
rare = {"naive": {N: [] for N in NS_R}, "is": {N: [] for N in NS_R}}
for N in NS_R:
    for _ in range(R_R):
        rare["naive"][N].append(mc_rare_naive(N))
        e, _ = mc_rare_is(N, THETA)
        rare["is"][N].append(e)

N_CMP = 16384
vn, vi = np.var(rare["naive"][N_CMP]), np.var(rare["is"][N_CMP])
print(f"[A] rare N={N_CMP}: naive mean={np.mean(rare['naive'][N_CMP]):.3e} | "
      f"IS mean={np.mean(rare['is'][N_CMP]):.3e} | VR={vn/max(vi,1e-30):.0f}x")

FLOOR = p_true / 50
fig, ax = plt.subplots(figsize=(7.6, 5.0))
for lab, key, c in [("朴素 MC", "naive", RED), (f"重要性采样（指数倾斜 θ={THETA}）", "is", GREEN)]:
    means = [max(np.mean(rare[key][N]), FLOOR) for N in NS_R]
    stds = [np.std(rare[key][N]) / np.sqrt(R_R) for N in NS_R]
    ax.errorbar(NS_R, means, yerr=2 * np.array(stds), fmt="o-", color=c, label=lab, lw=1.8, ms=5, capsize=3)
ax.axhline(p_true, color=GRAY, ls="--", lw=1.2, label=f"pilot 真值 ≈ {p_true:.1e}")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(FLOOR / 2, p_true * 30)
ax.set_xlabel("样本数 N"); ax.set_ylabel(f"P(均值 > {H}) 估计（±2SE，{R_R} 次重复）")
ax.set_title(f"稀有事件 p≈{p_true:.0e}：朴素 MC 在 N≤65536 多数重复估计为 0，指数倾斜后稳步收敛")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT_A, "rare_event_is.png")); plt.close(fig)

# --- Figure A5: VR summary ---
fig, ax = plt.subplots(figsize=(7.2, 4.6))
names = ["对偶变量", "控制变量", "分层抽样(W_T)", "Sobol RQMC"]
vals = [vr["anti"], vr["cv"], vr["strat"], vr["rqmc"]]
cols = [ORANGE, BLUE, PURPLE, GREEN]
bars = ax.bar(names, vals, color=cols, width=0.55)
ax.set_yscale("log")
for b_, v_ in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v_ * 1.15, f"{v_:.0f}×", ha="center", fontsize=10)
ax.axhline(1.0, color=GRAY, lw=1)
ax.set_ylabel("方差缩减倍数（N=4096，对数轴）")
ax.set_title("同一预算下各方法的方差缩减倍数：控制变量逼近理论极限 1/(1−ρ²)")
fig.tight_layout(); fig.savefig(os.path.join(OUT_A, "vr_factor_summary.png")); plt.close(fig)
print("[A] figures done:", sorted(os.listdir(OUT_A)))


# ============================================================
# Article B: CPT portfolio choice
# ============================================================
ALPHA_C, LAMBDA, GAMMA_G, GAMMA_L = 0.88, 2.25, 0.61, 0.69
NAMES = ["股票", "债券", "黄金", "彩票"]
C_COL = [RED, BLUE, ORANGE, PURPLE]


def gen_history(seed, months):
    """参数化市场：危机块(~11%) + 单因子 M + 4 资产（股票溢价/债券黄金保险/彩票零均值正偏）。"""
    g = np.random.default_rng(seed)
    crisis = np.zeros(months, dtype=bool)
    t = 0
    while t < months:
        if g.random() < 0.03:
            L = int(g.integers(2, 7))
            crisis[t:t + L] = True
            t += L
        else:
            t += 1
    M = np.empty(months)
    M[~crisis] = g.standard_normal((~crisis).sum()) * 0.042
    M[crisis] = -0.010 + g.standard_normal(crisis.sum()) * 0.085
    bM = np.where(crisis, -0.50, -0.25)
    gM = np.where(crisis, -0.90, -0.45)
    eq = 0.011 + M + g.standard_normal(months) * 0.020
    bd = 0.0020 + bM * M + g.standard_normal(months) * 0.008
    gd = 0.0010 + gM * M + g.standard_normal(months) * 0.030
    jack = g.random(months) < 0.03
    lot = np.where(jack, 0.18 + g.standard_normal(months) * 0.08,
                   -0.006 + g.standard_normal(months) * 0.012)
    return np.column_stack([eq, bd, gd, lot]), crisis


def cpt_value(x, lam=LAMBDA, gg=GAMMA_G, gl=GAMMA_L, alpha=ALPHA_C):
    xs = np.sort(x)
    n = xs.size
    p = np.arange(1, n + 1) / n
    wp = p ** gg / (p ** gg + (1 - p) ** gg) ** (1 / gg)
    wm = p ** gl / (p ** gl + (1 - p) ** gl) ** (1 / gl)
    dpg = np.diff(np.concatenate([[0.0], wp]))
    dpl = np.diff(np.concatenate([[0.0], wm]))
    av = np.abs(xs)
    v = np.where(xs >= 0, av ** alpha, -lam * av ** alpha)
    return float(np.sum(np.where(xs >= 0, dpg, dpl) * v))


def cpt_linear(x, lam=LAMBDA, alpha=ALPHA_C):
    av = np.abs(x)
    v = np.where(x >= 0, av ** alpha, -lam * av ** alpha)
    return float(v.mean())


def grid_weights(R, obj, step=0.05, cap=None):
    bw, bv = None, -np.inf
    for w1 in np.arange(0, 1 + 1e-9, step):
        for w2 in np.arange(0, 1 - w1 + 1e-9, step):
            for w3 in np.arange(0, 1 - w1 - w2 + 1e-9, step):
                w4 = 1 - w1 - w2 - w3
                if w4 < -1e-9:
                    continue
                w = np.array([w1, w2, w3, max(w4, 0.0)])
                if cap is not None and w.max() > cap + 1e-9:
                    continue
                v = obj(R @ w)
                if v > bv:
                    bv, bw = v, w
    return bw, bv


def mv_max_sharpe(R):
    n = R.shape[1]

    def ns(w):
        s = (R @ w).std()
        return -((R @ w).mean() / s) if s > 0 else 0.0

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bnds = [(0.0, 1.0)] * n
    best = None
    for _ in range(6):
        w0 = np.random.default_rng(int(rng.integers(1 << 30))).dirichlet(np.ones(n))
        res = minimize(ns, w0, method="SLSQP", bounds=bnds, constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-12})
        if res.success and (best is None or -res.fun > best[0]):
            best = (-res.fun, res.x)
    return best[1] / best[1].sum()


R_big, _ = gen_history(424242, 200000)
R_bt, crisis_bt = gen_history(777, 480)
print("[B] big means(%)", np.round(R_big.mean(0) * 100, 3), "vols(%)", np.round(R_big.std(0) * 100, 3))
print("[B] big corr\n", np.round(np.corrcoef(R_big.T), 3))
print("[B] backtest path: crisis=%.3f means(%%)=" % crisis_bt.mean(), np.round(R_bt.mean(0) * 100, 3))

# 彩票 CPT 分解 + 尾部决策权重
lot = R_big[:, 3]
print(f"[B] lottery: linearCPT={cpt_linear(lot):+.5f} | weighted CPT={cpt_value(lot):+.5f}")
w97p = 0.97 ** GAMMA_G / (0.97 ** GAMMA_G + 0.03 ** GAMMA_G) ** (1 / GAMMA_G)
w97m = 0.97 ** GAMMA_L / (0.97 ** GAMMA_L + 0.03 ** GAMMA_L) ** (1 / GAMMA_L)
print(f"[B] top 3% gain-domain decision weight = {1 - w97p:.4f} ({(1 - w97p)/0.03:.2f}x overweight); "
      f"bottom 97% loss-domain weight = {w97m:.4f}")
for nm, w in [("EW", [.25, .25, .25, .25]), ("60/40", [.6, .4, 0, 0])]:
    print(f"[B] CPT value {nm} = {cpt_value(R_big @ np.array(w)):+.5f}")

# 单资产 CPT（λ 扫描）
for lam in [1.0, 2.25, 5.0]:
    vals = [cpt_value(R_big[:, i], lam=lam) for i in range(4)]
    print(f"[B] single-asset CPT lam={lam}: " + " ".join(f"{n}={v:+.5f}" for n, v in zip(NAMES, vals)))

# CPT vs MV 权重
w_cpt, v_cpt = grid_weights(R_big, cpt_value)
w_mv = mv_max_sharpe(R_big)
w_cpt_cap, v_cpt_cap = grid_weights(R_big, cpt_value, cap=0.30)
print(f"[B] CPT-opt: {np.round(w_cpt, 3)} (v={v_cpt:.5f})")
print(f"[B] CPT-cap30: {np.round(w_cpt_cap, 3)} (v={v_cpt_cap:.5f})")
print(f"[B] MV max-Sharpe: {np.round(w_mv, 3)}")

# λ / γ 敏感性
lams = [1.0, 2.25, 3.5, 5.0]
lam_w = {lm: grid_weights(R_big, lambda x: cpt_value(x, lam=lm))[0] for lm in lams}
for lm in lams:
    print(f"[B] lambda={lm}: {np.round(lam_w[lm], 3)}")
gam_w = {}
for gg in [0.40, 0.61, 0.85]:
    gam_w[gg] = grid_weights(R_big, lambda x: cpt_value(x, gg=gg, gl=min(gg + 0.08, 0.99)))[0]
    print(f"[B] gamma={gg}: {np.round(gam_w[gg], 3)}")


def agg(R, h):
    n = (R.shape[0] // h) * h
    return (1 + R[:n].reshape(-1, h, R.shape[1])).prod(axis=1) - 1


horizons = [1, 3, 6, 12, 24]
hz_w = {}
for h in horizons:
    hz_w[h] = grid_weights(agg(R_big, h), cpt_value)[0]
    print(f"[B] horizon={h}m: {np.round(hz_w[h], 3)}")


def perf(R, w):
    pr = R @ w
    ann = (1 + pr).prod() ** (12 / len(pr)) - 1
    vol = pr.std() * np.sqrt(12)
    sh = pr.mean() / pr.std() * np.sqrt(12) if pr.std() > 0 else 0.0
    cum = np.cumprod(1 + pr)
    dd = (cum / np.maximum.accumulate(cum) - 1).min()
    return dict(cagr=ann, vol=vol, sharpe=sh, mdd=dd, cum=cum,
                worst=pr.min(), down=float(np.mean(pr < 0)))


portfolios = {"MV 最大夏普": w_mv, "CPT 最优": w_cpt, "CPT 受限(≤30%)": w_cpt_cap,
              "60/40 股债": np.array([0.60, 0.40, 0.0, 0.0])}
stats = {n: perf(R_bt, w) for n, w in portfolios.items()}
for n, s in stats.items():
    print(f"[B] BT {n}: CAGR={s['cagr']*100:.2f}% vol={s['vol']*100:.2f}% Sharpe={s['sharpe']:.2f} "
          f"MDD={s['mdd']*100:.2f}% worst_m={s['worst']*100:.2f}% down={s['down']*100:.0f}%")

# 经验 CPT 噪声：5 条独立 360 个月路径
emp = []
for sd in [11, 22, 33, 44, 55]:
    Re, _ = gen_history(sd, 360)
    w, _ = grid_weights(Re, cpt_value)
    emp.append(w)
    print(f"[B] empirical path {sd}: {np.round(w, 3)}")
emp = np.array(emp)
print(f"[B] empirical spread: equity {emp[:,0].min():.2f}-{emp[:,0].max():.2f}, "
      f"lottery {emp[:,3].min():.2f}-{emp[:,3].max():.2f}")

# --- Figure B1: value function ---
fig, ax = plt.subplots(figsize=(6.8, 5.0))
xs = np.linspace(-0.15, 0.15, 400)
ax.plot(xs * 100, np.where(xs >= 0, np.abs(xs) ** ALPHA_C, -LAMBDA * np.abs(xs) ** ALPHA_C) * 100,
        color=RED, lw=2.4, label=f"CPT 价值函数 v(x)（λ={LAMBDA}, α={ALPHA_C}）")
ax.plot(xs * 100, xs * 100, color=GRAY, ls="--", lw=1.4, label="风险中性（线性）")
ax.plot(xs * 100, np.sign(xs) * np.sqrt(np.abs(xs)) * 100, color=BLUE, ls=":", lw=1.6,
        label="效用函数（√ 凹性）")
ax.axvline(0, color=BLACK, lw=0.8)
ax.set_xlabel("收益 / 亏损（%）"); ax.set_ylabel("心理价值 v(x)")
ax.set_title("参考点在 0：亏损比等量收益疼 %.1f 倍" % LAMBDA)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT_B, "value_function.png")); plt.close(fig)

# --- Figure B2: probability weighting ---
fig, ax = plt.subplots(figsize=(6.8, 5.4))
ps = np.linspace(0.001, 0.999, 400)
wp = ps ** GAMMA_G / (ps ** GAMMA_G + (1 - ps) ** GAMMA_G) ** (1 / GAMMA_G)
wm = ps ** GAMMA_L / (ps ** GAMMA_L + (1 - ps) ** GAMMA_L) ** (1 / GAMMA_L)
ax.plot(ps, wp, color=RED, lw=2.2, label=f"收益域 w+(p)（γ={GAMMA_G}）")
ax.plot(ps, wm, color=GREEN, lw=2.2, label=f"亏损域 w-(p)（γ={GAMMA_L}）")
ax.plot(ps, ps, color=GRAY, ls="--", lw=1.2, label="恒等线（无加权）")
marks = [(0.03, RED, "p=0.03 → w≈0.10（3.4 倍）"), (0.10, GREEN, "p=0.10 → w≈0.17（1.7 倍）"),
         (0.95, BLUE, "p=0.95 → w≈0.86（低估）")]
for p0, c, lab in marks:
    wv = (p0 ** GAMMA_G / (p0 ** GAMMA_G + (1 - p0) ** GAMMA_G) ** (1 / GAMMA_G) if c == RED else
          p0 ** GAMMA_L / (p0 ** GAMMA_L + (1 - p0) ** GAMMA_L) ** (1 / GAMMA_L) if c == GREEN else
          p0 ** GAMMA_G / (p0 ** GAMMA_G + (1 - p0) ** GAMMA_G) ** (1 / GAMMA_G))
    ax.plot([p0], [wv], "o", color=c, ms=8)
    ax.annotate(lab, (p0, wv), textcoords="offset points", xytext=(10, -16), fontsize=8.5, color=c)
ax.set_xlabel("真实概率 p"); ax.set_ylabel("决策权重 w(p)")
ax.set_title("逆 S 形概率加权：小概率被放大、中大概率被压缩")
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT_B, "probability_weighting.png")); plt.close(fig)

# --- Figure B3: weights vs lambda ---
fig, ax = plt.subplots(figsize=(8.0, 5.0))
xpos = np.arange(len(lams))
bottom = np.zeros(len(lams))
for i, nm in enumerate(NAMES):
    vals = np.array([lam_w[lm][i] for lm in lams])
    ax.bar(xpos, vals, bottom=bottom, color=C_COL[i], label=nm, width=0.5)
    for xp_, v_, b_ in zip(xpos, vals, bottom):
        if v_ > 0.04:
            ax.text(xp_, b_ + v_ / 2, f"{v_*100:.0f}%", ha="center", va="center", fontsize=8.5, color="white")
    bottom += vals
for i, nm in enumerate(NAMES):
    ax.plot(xpos, [w_mv[i]] * len(lams), "D", color=C_COL[i], ms=6, mec=BLACK, mew=0.8,
            label=f"{nm}（MV 基准）" if i == 0 else None)
    ax.annotate(f"{w_mv[i]*100:.0f}%", (xpos[-1] + 0.30, w_mv[i]), fontsize=8.5, color=C_COL[i], va="center")
ax.set_xticks(xpos); ax.set_xticklabels([f"λ={lm}" for lm in lams])
ax.set_ylabel("组合权重"); ax.set_ylim(0, 1.08)
ax.set_title("CPT 最优配置随损失厌恶 λ 的变化（菱形 = MV 最大夏普权重）")
ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(OUT_B, "weights_lambda.png")); plt.close(fig)

# --- Figure B4: evaluation horizon ---
fig, ax = plt.subplots(figsize=(7.6, 5.0))
for i, nm in enumerate(NAMES):
    ax.plot(horizons, [hz_w[h][i] for h in horizons], "o-", color=C_COL[i], label=nm, lw=1.9, ms=5)
ax.set_xscale("log"); ax.set_xticks(horizons); ax.set_xticklabels([str(h) for h in horizons])
ax.set_xlabel("评估期长度（月，对数轴）"); ax.set_ylabel("CPT 最优权重")
ax.set_title("短视的损失厌恶：评估期越长，越敢持有股票")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT_B, "evaluation_horizon.png")); plt.close(fig)

# --- Figure B5: backtest ---
fig, ax = plt.subplots(figsize=(9.2, 5.2))
style = {"MV 最大夏普": (BLUE, "-"), "CPT 最优": (RED, "-"), "CPT 受限(≤30%)": (PURPLE, "-."),
         "60/40 股债": (GREEN, ":")}
for name, (c, ls) in style.items():
    s = stats[name]
    lab = (f"{name}（CAGR {s['cagr']*100:.1f}%、波动 {s['vol']*100:.1f}%、MDD {s['mdd']*100:.1f}%）")
    ax.plot(np.arange(1, 481), s["cum"], color=c, ls=ls, lw=1.8, label=lab)
d = np.diff(crisis_bt.astype(int))
for s0, e0 in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
    ax.axvspan(s0, e0, color=ORANGE, alpha=0.12)
ax.set_yscale("log")
ax.set_xlabel("月份"); ax.set_ylabel("累计财富（对数轴，期初 = 1）")
ax.set_title("独立 480 个月路径：CPT 最优（95% 彩票）、受限 CPT、MV 与经典 60/40")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(OUT_B, "backtest_cpt_vs_mv.png")); plt.close(fig)
print("[B] figures done:", sorted(os.listdir(OUT_B)))
print("ALL DONE")
