#!/usr/bin/env python3
"""Trend Scanning labeling experiment.

1. Generate synthetic price with known regime segments (up-trend / down-trend / sideways).
2. Apply trend-scanning labels (max |t-value| over forward horizons).
3. Compare against fixed-horizon labels (sign of forward return over fixed h).
4. Train a small classifier on identical features, compare label quality downstream.
Outputs: PNGs + printed metrics.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

OUT = "/Users/halo/workspace/astro-blog/public/images/trend-scanning-labels"
import os
os.makedirs(OUT, exist_ok=True)

# ---------------- 1. synthetic market with known regimes ----------------
# regimes: +1 uptrend, -1 downtrend, 0 sideways; each segment random length
segments = []
true_regime = []
n_target = 2400
drift_map = {1: 0.0012, -1: -0.0012, 0: 0.0}
state_pool = [1, -1, 0, 0, 1, -1]  # sideways twice as likely-ish
logp = [0.0]
while len(logp) < n_target:
    s = rng.choice(state_pool)
    L = int(rng.integers(60, 180))
    vol = 0.008 if s == 0 else 0.007
    for _ in range(L):
        if len(logp) >= n_target:
            break
        r = drift_map[s] + vol * rng.standard_normal()
        if s == 0:
            # mean-reverting sideways: pull back to segment start
            r += -0.03 * (logp[-1] - logp[max(0, len(logp) - 40)])
        logp.append(logp[-1] + r)
        true_regime.append(s)
logp = np.array(logp[:n_target])
true_regime = np.array(true_regime[:n_target - 1])
price = 100 * np.exp(logp)
n = len(price)

# ---------------- 2. trend scanning ----------------
def tval_ols(y):
    """t-value of slope in y = a + b*t + e."""
    T = len(y)
    x = np.arange(T, dtype=float)
    xm, ym = x.mean(), y.mean()
    Sxx = ((x - xm) ** 2).sum()
    b = ((x - xm) * (y - ym)).sum() / Sxx
    a = ym - b * xm
    resid = y - a - b * x
    dof = T - 2
    s2 = (resid ** 2).sum() / dof
    se = np.sqrt(s2 / Sxx)
    if se == 0:
        return 0.0
    return b / se

def trend_scanning(close, spans):
    n = len(close)
    tmax = np.full(n, np.nan)
    hbest = np.full(n, -1)
    hi = max(spans)
    for i in range(n - hi):
        best_t, best_h = 0.0, -1
        for h in spans:
            tv = tval_ols(close[i : i + h])
            if abs(tv) > abs(best_t):
                best_t, best_h = tv, h
        tmax[i] = best_t
        hbest[i] = best_h
    return tmax, hbest

spans = list(range(10, 61, 5))
tvals, hbest = trend_scanning(logp, spans)
valid = ~np.isnan(tvals)

# --- calibrate threshold on a pure random-walk null ---
# OLS t-values on price levels are wildly inflated (autocorrelated residuals),
# so the textbook |t|>2 is meaningless. Use the null distribution of max-|t|
# from a driftless random walk with matched vol.
null_ts = []
sig = np.diff(logp).std()
for _ in range(60):
    rw = np.cumsum(sig * rng.standard_normal(400))
    tv, _ = trend_scanning(rw, spans)
    null_ts.append(np.abs(tv[~np.isnan(tv)]))
null_ts = np.concatenate(null_ts)
THR = float(np.quantile(null_ts, 0.90))
print(f"null-calibrated threshold (90% of random walk max-|t|): {THR:.2f}")

ts_label = np.sign(tvals)          # raw sign labels
# thresholded version: |t|>THR trend, else neutral
ts_label3 = np.where(np.abs(tvals) > THR, np.sign(tvals), 0)

# ---------------- fixed horizon labels ----------------
H = 20
fwd = np.full(n, np.nan)
fwd[: n - H] = logp[H:] - logp[: n - H]
fh_label = np.sign(fwd)

# ---------------- 3. label quality vs true regime ----------------
idx = np.arange(n - max(spans))
tr = true_regime[: len(idx)]
def agree(labels, name):
    m = (tr != 0) & valid[idx]
    acc = (labels[idx][m] == tr[m]).mean()
    # noise on sideways: fraction of nonzero labels inside sideways
    ms = (tr == 0) & valid[idx]
    return acc, m.sum(), ms

acc_ts, ntrend, ms = agree(ts_label, "ts")
acc_fh, _, _ = agree(fh_label, "fh")
# sideways behavior
side_ts_strong = (np.abs(tvals[idx][ms]) > THR).mean()
side_fh_flip = np.abs(np.diff(fh_label[idx][ms])).mean() / 2  # flip rate per bar
side_ts_flip = np.abs(np.diff(np.sign(tvals[idx][ms]))).mean() / 2

print(f"trend bars n={ntrend}")
print(f"trend-scanning sign acc on true-trend bars: {acc_ts:.3f}")
print(f"fixed-horizon (h=20) sign acc on true-trend bars: {acc_fh:.3f}")
print(f"sideways: frac |t|>THR (false trend) = {side_ts_strong:.3f}")
print(f"sideways label flip-rate: fixed-horizon {side_fh_flip:.3f} vs trend-scan {side_ts_flip:.3f}")

# label3 coverage
cov3 = (ts_label3[idx] != 0).mean()
acc3_m = (tr != 0) & valid[idx] & (ts_label3[idx] != 0)
acc3 = (ts_label3[idx][acc3_m] == tr[acc3_m]).mean()
print(f"|t|>THR labels: coverage={cov3:.3f}, acc on true-trend={acc3:.3f}")

# ---------------- 4. downstream classifier comparison ----------------
# features: past returns/momentum/vol; predict label; evaluate strategy PnL out of sample
def make_features(logp):
    r = np.diff(logp, prepend=logp[0])
    f = {}
    for w in (5, 10, 20, 40):
        mom = np.convolve(r, np.ones(w), "full")[: len(r)]
        f[f"mom{w}"] = mom
        vol = np.array([r[max(0, i - w) : i + 1].std() for i in range(len(r))])
        f[f"vol{w}"] = vol
    X = np.column_stack(list(f.values()))
    return X

X = make_features(logp)

from numpy.linalg import lstsq
def train_eval(labels, weights=None, name=""):
    # simple ridge-like linear classifier via least squares on {-1,+1}
    m = valid & (labels != 0) & (np.arange(n) < n - max(spans))
    ii = np.where(m)[0]
    split = int(len(ii) * 0.6)
    tr_i, te_i = ii[:split], ii[split:]
    Xtr = X[tr_i]; ytr = labels[tr_i]
    w = weights[tr_i] if weights is not None else np.ones(len(tr_i))
    Xw = Xtr * w[:, None]
    A = Xw.T @ Xtr + 1e-3 * np.eye(X.shape[1])
    b = Xw.T @ ytr
    beta = np.linalg.solve(A, b)
    # out-of-sample: trade sign of prediction next day
    te_mask = np.arange(n - 1) >= ii[split]
    pred = np.sign(X[:-1] @ beta)
    r1 = np.diff(logp)
    pnl = pred[te_mask] * r1[te_mask]
    sr = pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(252)
    hit = (labels[te_i] == np.sign(X[te_i] @ beta)).mean()
    print(f"{name}: OOS label-hit={hit:.3f}, strategy Sharpe={sr:.2f}, cum logret={pnl.sum():.3f}")
    return sr, hit

sr_fh, hit_fh = train_eval(fh_label, None, "fixed-horizon labels")
sr_ts, hit_ts = train_eval(ts_label, None, "trend-scan sign labels")
wts = np.abs(np.nan_to_num(tvals))
sr_tsw, hit_tsw = train_eval(ts_label3, wts, "trend-scan |t|>2 + t-weighted")

# ---------------- plots ----------------
# P1: price colored by trend-scanning label
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
ax = axes[0]
ii = np.where(valid)[0]
colors = np.where(tvals[ii] > THR, "#d62728", np.where(tvals[ii] < -THR, "#2ca02c", "#bbbbbb"))
ax.scatter(ii, price[ii], c=colors, s=4)
ax.set_title(f"趋势扫描标注：|t|>{THR:.1f}（null 校准阈值）上涨(红) / 下跌(绿) / 无趋势(灰)")
ax.set_ylabel("价格")
# shade true regimes
ax2 = axes[1]
ax2.plot(tvals, lw=0.8, color="#1f77b4")
ax2.axhline(THR, color="#d62728", ls="--", lw=0.8)
ax2.axhline(-THR, color="#2ca02c", ls="--", lw=0.8)
ax2.fill_between(np.arange(len(true_regime)), -8, 8, where=true_regime == 0, color="#f0f0f0")
ax2.set_ylim(-9, 9)
ax2.set_ylabel("t 值")
ax2.set_xlabel("交易日（灰色底纹 = 真实震荡段）")
ax2.set_title("t 值序列 vs 真实 regime")
plt.tight_layout()
plt.savefig(f"{OUT}/tvalue-labels.png", dpi=110)
plt.close()

# P2: label comparison on a zoom window (sideways segment)
# find a long sideways stretch
zs = None
run = 0
for i, s in enumerate(true_regime):
    if s == 0:
        run += 1
        if run > 120:
            zs = i - run + 1
            break
    else:
        run = 0
if zs is None:
    zs = 200
z0, z1 = zs, min(zs + 160, n - max(spans))
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
xs = np.arange(z0, z1)
axes[0].plot(xs, price[z0:z1], color="k", lw=1)
axes[0].set_title("震荡段放大：价格")
axes[1].step(xs, fh_label[z0:z1], color="#ff7f0e", lw=1)
axes[1].set_ylim(-1.5, 1.5); axes[1].set_title(f"固定视野标签（h={H}）：震荡段内高频翻转")
axes[2].step(xs, ts_label3[z0:z1], color="#1f77b4", lw=1.2)
axes[2].set_ylim(-1.5, 1.5); axes[2].set_title(f"趋势扫描标签（|t|>{THR:.1f}）：震荡段大多保持 0")
axes[2].set_xlabel("交易日")
plt.tight_layout()
plt.savefig(f"{OUT}/label-comparison.png", dpi=110)
plt.close()

# P3: distribution of t-values + chosen horizon
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].hist(tvals[valid], bins=60, color="#1f77b4", alpha=0.85)
axes[0].axvline(THR, color="#d62728", ls="--"); axes[0].axvline(-THR, color="#2ca02c", ls="--")
axes[0].set_title(f"t 值分布（虚线 = ±{THR:.1f} null 校准阈值）"); axes[0].set_xlabel("t 值")
hb = hbest[valid & (np.abs(tvals) > THR)]
axes[1].hist(hb, bins=np.arange(7.5, 65, 5), color="#ff7f0e", alpha=0.85)
axes[1].set_title("显著趋势样本的最优回看窗口分布"); axes[1].set_xlabel("窗口长度（天）")
plt.tight_layout()
plt.savefig(f"{OUT}/tvalue-horizon-dist.png", dpi=110)
plt.close()

# P4: downstream bars
fig, ax = plt.subplots(figsize=(9, 4.2))
names = ["固定视野 h=20", "趋势扫描 sign(t)", "趋势扫描 |t|>阈值\n+ t 值加权"]
srs = [sr_fh, sr_ts, sr_tsw]
bars = ax.bar(names, srs, color=["#ff7f0e", "#1f77b4", "#2ca02c"], width=0.55)
for b, v in zip(bars, srs):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.03 * np.sign(v), f"{v:.2f}", ha="center")
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("样本外策略年化 Sharpe")
ax.set_title("同一特征、同一模型：只换标签方案的样本外表现")
plt.tight_layout()
plt.savefig(f"{OUT}/downstream-sharpe.png", dpi=110)
plt.close()

print("PLOTS DONE")
