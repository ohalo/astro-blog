#!/usr/bin/env python3
"""PIN model simulation: EKOP (1996) MLE estimation + cross-sectional backtest."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.special import logsumexp
from scipy.optimize import minimize
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)
OUT = "/Users/halo/workspace/astro-blog/public/images/pin-informed-trading-probability"
os.makedirs(OUT, exist_ok=True)

# ---------- 1. simulate one stock's daily buy/sell counts from EKOP tree ----------
def simulate_days(alpha, delta, mu, eb, es, n_days, rng):
    B = np.zeros(n_days, dtype=int)
    S = np.zeros(n_days, dtype=int)
    day_type = np.zeros(n_days, dtype=int)  # 0 none, 1 bad, 2 good
    for t in range(n_days):
        if rng.random() < alpha:
            if rng.random() < delta:  # bad news -> informed sellers
                B[t] = rng.poisson(eb); S[t] = rng.poisson(es + mu); day_type[t] = 1
            else:
                B[t] = rng.poisson(eb + mu); S[t] = rng.poisson(es); day_type[t] = 2
        else:
            B[t] = rng.poisson(eb); S[t] = rng.poisson(es); day_type[t] = 0
    return B, S, day_type

def neg_loglik(params, B, S):
    a, d, mu, eb, es = params
    if not (0 < a < 1 and 0 < d < 1 and mu > 0 and eb > 0 and es > 0):
        return 1e12
    l0 = np.log(1 - a) + poisson.logpmf(B, eb) + poisson.logpmf(S, es)
    l1 = np.log(a * d) + poisson.logpmf(B, eb) + poisson.logpmf(S, es + mu)
    l2 = np.log(a * (1 - d)) + poisson.logpmf(B, eb + mu) + poisson.logpmf(S, es)
    ll = logsumexp(np.vstack([l0, l1, l2]), axis=0)
    return -np.sum(ll)

def estimate_pin(B, S):
    best = None
    for a0 in (0.2, 0.4):
        for mu0 in (np.mean(B + S) * 0.3, np.mean(B + S) * 0.6):
            x0 = [a0, 0.5, mu0, np.mean(B) * 0.8, np.mean(S) * 0.8]
            res = minimize(neg_loglik, x0, args=(B, S), method="Nelder-Mead",
                           options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-4})
            if best is None or res.fun < best.fun:
                best = res
    a, d, mu, eb, es = best.x
    pin = a * mu / (a * mu + eb + es)
    return pin, best.x

# demo single stock
true_p = dict(alpha=0.35, delta=0.5, mu=180, eb=350, es=330)
B, S, dt = simulate_days(**true_p, n_days=250, rng=rng)
pin_hat, est = estimate_pin(B, S)
true_pin = true_p["alpha"] * true_p["mu"] / (true_p["alpha"] * true_p["mu"] + true_p["eb"] + true_p["es"])
print(f"single stock: true PIN={true_pin:.4f}, est PIN={pin_hat:.4f}")
print("est params:", np.round(est, 3))

# fig1: buy/sell scatter colored by day type
fig, ax = plt.subplots(figsize=(8, 6))
colors = {0: "#8899aa", 1: "#d62728", 2: "#2ca02c"}
labels = {0: "无信息日", 1: "坏消息日（知情卖出）", 2: "好消息日（知情买入）"}
for k in (0, 2, 1):
    m = dt == k
    ax.scatter(B[m], S[m], s=22, alpha=0.75, c=colors[k], label=labels[k], edgecolors="none")
ax.set_xlabel("日买单笔数 B")
ax.set_ylabel("日卖单笔数 S")
ax.set_title(f"EKOP 混合结构：250 个交易日的 (B, S) 散点\n真实 PIN={true_pin:.3f}，MLE 估计 PIN={pin_hat:.3f}")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/ekop-scatter.png", dpi=110); plt.close(fig)

# ---------- 2. cross-section: 300 stocks ----------
N = 300
alphas = rng.uniform(0.10, 0.60, N)
mus = rng.uniform(60, 260, N)
ebs = rng.uniform(150, 600, N)
ess = ebs * rng.uniform(0.85, 1.15, N)
true_pins = alphas * mus / (alphas * mus + ebs + ess)

est_pins = np.zeros(N)
for i in range(N):
    Bi, Si, _ = simulate_days(alphas[i], 0.5, mus[i], ebs[i], ess[i], 250, rng)
    est_pins[i], _ = estimate_pin(Bi, Si)
corr = np.corrcoef(true_pins, est_pins)[0, 1]
print(f"cross-section corr(true, est) = {corr:.3f}")

fig, ax = plt.subplots(figsize=(7.5, 6))
ax.scatter(true_pins, est_pins, s=18, alpha=0.6, c="#1f77b4", edgecolors="none")
lim = [0, max(true_pins.max(), est_pins.max()) * 1.05]
ax.plot(lim, lim, "k--", lw=1, label="45° 线")
ax.set_xlabel("真实 PIN"); ax.set_ylabel("MLE 估计 PIN")
ax.set_title(f"300 只股票：估计 PIN vs 真实 PIN（相关 {corr:.2f}）")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/pin-estimation.png", dpi=110); plt.close(fig)

# ---------- 3. monthly cross-sectional returns with PIN premium ----------
T = 180  # 15y months
lam_annual = 0.055  # premium per unit PIN spread... define monthly premium coefficient
lam_m = 0.032  # monthly return per unit of PIN
mkt = rng.normal(0.007, 0.045, T)
betas = rng.uniform(0.8, 1.2, N)
idio = rng.normal(0, 0.07, (T, N))
rets = np.outer(mkt, betas) + lam_m * (true_pins - true_pins.mean())[None, :] + idio

# quintile sort by ESTIMATED pin (static)
order = np.argsort(est_pins)
q = np.array_split(order, 5)
q_rets = np.array([rets[:, idx].mean(axis=1) for idx in q])  # 5 x T
ann = (1 + q_rets.mean(axis=1)) ** 12 - 1
ls = q_rets[4] - q_rets[0]
ls_ann = (1 + ls.mean()) ** 12 - 1
ls_t = ls.mean() / ls.std(ddof=1) * np.sqrt(T)
ls_sharpe = ls.mean() / ls.std(ddof=1) * np.sqrt(12)
print("quintile ann:", np.round(ann, 4))
print(f"LS ann={ls_ann:.4f}, monthly mean={ls.mean()*100:.3f}%, t={ls_t:.2f}, sharpe={ls_sharpe:.2f}")

fig, ax = plt.subplots(figsize=(8, 5.5))
bars = ax.bar([f"G{i+1}" for i in range(5)], ann * 100, color=["#aec7e8"] * 4 + ["#1f77b4"])
for b, v in zip(bars, ann * 100):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}%", ha="center", fontsize=10)
ax.set_ylabel("年化收益（%）")
ax.set_title(f"按估计 PIN 五分组的年化收益（G5 最高 PIN）\n多空 G5−G1 年化 {ls_ann*100:.1f}%，t={ls_t:.1f}")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/quintile-returns.png", dpi=110); plt.close(fig)

nav = np.cumprod(1 + ls)
navs_q = np.cumprod(1 + q_rets, axis=1)
fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(T) / 12
for i in (0, 4):
    ax.plot(x, navs_q[i], lw=1.6, label=f"G{i+1}（{'最低' if i==0 else '最高'} PIN）")
ax.plot(x, nav, lw=2.2, c="#d62728", label=f"多空 G5−G1（Sharpe {ls_sharpe:.2f}）")
ax.set_xlabel("年"); ax.set_ylabel("累计净值")
ax.set_title("PIN 分组累计净值：15 年月度再平衡")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/long-short-nav.png", dpi=110); plt.close(fig)

# boundary failure stat: fraction of days where B+S very large -> pin estimate bias with trade intensity
# report correlation of est_pin with total intensity (spurious size link)
intensity = ebs + ess + alphas * mus
print(f"corr(est_pin, intensity) = {np.corrcoef(est_pins, intensity)[0,1]:.3f}")
print(f"corr(true_pin, intensity) = {np.corrcoef(true_pins, intensity)[0,1]:.3f}")
