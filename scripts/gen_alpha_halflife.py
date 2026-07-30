#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""换手率衰减与 Alpha 半衰期：模拟实验 + 配图"""
import numpy as np, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc"]:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)
OUT = "/Users/halo/workspace/astro-blog/public/images/turnover-decay-alpha-halflife"
os.makedirs(OUT, exist_ok=True)

# ---------- 模拟：N 只股票，T 天，信号 AR(1) 衰减 ----------
N, T = 200, 2000
HL_true = 8.0                      # 真实 alpha 半衰期（天）
rho = 0.5 ** (1.0 / HL_true)       # 信号自相关
ic0 = 0.012                        # 即期 IC
sigma_r = 0.02                     # 日收益波动

# 信号面板：横截面标准化的 AR(1)
S = np.zeros((T, N))
S[0] = rng.standard_normal(N)
for t in range(1, T):
    S[t] = rho * S[t-1] + np.sqrt(1 - rho**2) * rng.standard_normal(N)
# 收益 = ic0 * 昨日信号 + 噪声  (信号预测次日收益)
R = np.zeros((T, N))
R[1:] = ic0 * S[:-1] * sigma_r / 1.0 + sigma_r * np.sqrt(1 - ic0**2) * rng.standard_normal((T-1, N))

def zscore(x):
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)

Sz = zscore(S)

# ---------- 1) IC 衰减曲线与半衰期 ----------
lags = np.arange(1, 41)
ics = []
for h in lags:
    # 用 t 日信号预测 t+h 日收益
    x = Sz[:-h].ravel(); y = R[h:].ravel()
    ics.append(np.corrcoef(x, y)[0, 1])
ics = np.array(ics)
# 拟合指数衰减 ic(h) = a * exp(-lambda h)：非线性最小二乘（网格），避免只用正值点的选择偏差
best_sse, lam_hat, a_hat = np.inf, None, None
for lam_try in np.linspace(0.01, 0.6, 600):
    x = np.exp(-lam_try * lags)
    a_try = (x @ ics) / (x @ x)
    sse = np.sum((ics - a_try * x) ** 2)
    if sse < best_sse:
        best_sse, lam_hat, a_hat = sse, lam_try, a_try
hl_hat = np.log(2) / lam_hat
# 对比：只用正值点的对数拟合（有选择偏差，文章用作反面教材）
mask = ics > 0
logfit = np.polyfit(lags[mask], np.log(ics[mask]), 1)
hl_logfit = np.log(2) / (-logfit[0])

fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
ax.plot(lags, ics * 100, "o", ms=4, color="#1f77b4", label="实测 IC")
ax.plot(lags, a_hat * np.exp(-lam_hat * lags) * 100, "-", color="#d62728",
        label=f"指数拟合  半衰期={hl_hat:.1f} 天")
ax.axhline(0, color="gray", lw=0.8)
ax.axvline(hl_hat, color="#d62728", ls="--", lw=1)
ax.set_xlabel("预测滞后 h（天）"); ax.set_ylabel("IC（%）")
ax.set_title("信号 IC 随预测滞后的衰减（真实半衰期 8 天）")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/ic-decay-halflife.png"); plt.close(fig)

# ---------- 2) 不同持有期的组合：毛/净 Sharpe vs 换手 ----------
WARM = 100
def backtest(hold, cost_bps):
    """每 hold 天再平衡一次的多空十分位组合"""
    w = np.zeros(N); pnl = []; turns = []
    for t in range(WARM, T-1):
        if (t - WARM) % hold == 0:
            s = Sz[t]
            q_hi = np.quantile(s, 0.9); q_lo = np.quantile(s, 0.1)
            w_new = np.where(s >= q_hi, 1.0, np.where(s <= q_lo, -1.0, 0.0))
            n_long = (w_new > 0).sum(); n_short = (w_new < 0).sum()
            w_new = np.where(w_new > 0, w_new / n_long, np.where(w_new < 0, w_new / n_short, 0.0))
            turn = np.abs(w_new - w).sum() / 2
            turns.append(turn)
            pnl.append(w_new @ R[t+1] - turn * cost_bps * 1e-4)
            w = w_new
        else:
            pnl.append(w @ R[t+1]); turns.append(0.0)
    pnl = np.array(pnl)
    sharpe = pnl.mean() / pnl.std() * np.sqrt(252)
    ann_turn = np.sum(turns) / len(pnl) * 252
    return sharpe, ann_turn

holds = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60]
res = {}
for cb in [0, 10, 30, 60]:
    res[cb] = [backtest(h, cb) for h in holds]

fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
colors = {0: "#2ca02c", 10: "#1f77b4", 30: "#ff7f0e", 60: "#d62728"}
best = {}
for cb in [0, 10, 30, 60]:
    sh = [r[0] for r in res[cb]]
    ax.plot(holds, sh, "o-", ms=4, color=colors[cb], label=f"单边成本 {cb}bp")
    b = holds[int(np.argmax(sh))]; best[cb] = (b, max(sh))
    ax.plot(b, max(sh), "*", ms=15, color=colors[cb])
ax.set_xscale("log"); ax.set_xticks(holds); ax.set_xticklabels(holds)
ax.set_xlabel("持有期（天，对数轴）"); ax.set_ylabel("净 Sharpe")
ax.set_title("净 Sharpe vs 持有期：成本越高，最优持有期越长")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/sharpe-vs-holding.png"); plt.close(fig)

# ---------- 3) 换手率 vs 持有期 ----------
fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
ann_turns = [r[1] for r in res[0]]
ax.plot(holds, ann_turns, "s-", color="#9467bd")
ax.set_xscale("log"); ax.set_xticks(holds); ax.set_xticklabels(holds)
ax.set_xlabel("持有期（天，对数轴）"); ax.set_ylabel("年化单边换手（倍）")
ax.set_title("换手率随持有期的衰减")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/turnover-vs-holding.png"); plt.close(fig)

# ---------- 4) EMA 平滑信号 vs 直接拉长持有期 ----------
def backtest_ema(span, cost_bps, hold=1):
    alpha = 2 / (span + 1)
    Se = np.zeros_like(Sz); Se[0] = Sz[0]
    for t in range(1, T):
        Se[t] = (1 - alpha) * Se[t-1] + alpha * Sz[t]
    Sez = zscore(Se)
    w = np.zeros(N); pnl = []
    for t in range(WARM, T-1):
        s = Sez[t]
        q_hi = np.quantile(s, 0.9); q_lo = np.quantile(s, 0.1)
        w_new = np.where(s >= q_hi, 1.0, np.where(s <= q_lo, -1.0, 0.0))
        n_long = (w_new > 0).sum(); n_short = (w_new < 0).sum()
        w_new = np.where(w_new > 0, w_new / n_long, np.where(w_new < 0, w_new / n_short, 0.0))
        turn = np.abs(w_new - w).sum() / 2
        pnl.append(w_new @ R[t+1] - turn * cost_bps * 1e-4)
        w = w_new
    pnl = np.array(pnl)
    return pnl.mean() / pnl.std() * np.sqrt(252)

spans = [1, 2, 3, 5, 8, 12, 20, 30]
fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
for cb in [10, 30, 60]:
    sh_ema = [backtest_ema(s, cb) for s in spans]
    ax.plot(spans, sh_ema, "o-", color=colors[cb], label=f"EMA 平滑，成本 {cb}bp")
    sh_hold = [r[0] for r in res[cb]]
    ax.plot(holds, sh_hold, "--", color=colors[cb], alpha=0.45, label=f"拉长持有期，成本 {cb}bp")
ax.set_xscale("log"); ax.set_xticks(spans); ax.set_xticklabels(spans)
ax.set_xlabel("EMA span / 持有期（天，对数轴）"); ax.set_ylabel("净 Sharpe")
ax.set_title("降换手的两条路：信号平滑（实线） vs 拉长持有期（虚线）")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/ema-vs-holding.png"); plt.close(fig)

# ---------- 5) 多种子：半衰期估计量的抽样分布 ----------
def estimate_hl(seed):
    r2 = np.random.default_rng(seed)
    S2 = np.zeros((T, N)); S2[0] = r2.standard_normal(N)
    for t in range(1, T):
        S2[t] = rho * S2[t-1] + np.sqrt(1 - rho**2) * r2.standard_normal(N)
    R2 = np.zeros((T, N))
    R2[1:] = ic0 * S2[:-1] * sigma_r + sigma_r * np.sqrt(1 - ic0**2) * r2.standard_normal((T-1, N))
    S2z = zscore(S2)
    ics2 = np.array([np.corrcoef(S2z[:-h].ravel(), R2[h:].ravel())[0, 1] for h in lags])
    bs, lh = np.inf, None
    for lt in np.linspace(0.01, 0.6, 300):
        x = np.exp(-lt * lags); at = (x @ ics2) / (x @ x)
        sse = np.sum((ics2 - at * x) ** 2)
        if sse < bs: bs, lh = sse, lt
    return np.log(2) / lh

hls = [estimate_hl(s) for s in range(100, 120)]
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
ax.hist(hls, bins=10, color="#1f77b4", alpha=0.75, edgecolor="white")
ax.axvline(HL_true, color="#d62728", lw=2, label=f"真实半衰期 = {HL_true:.0f} 天")
ax.axvline(np.mean(hls), color="#2ca02c", lw=2, ls="--", label=f"20 种子均值 = {np.mean(hls):.1f} 天")
ax.set_xlabel("估计半衰期（天）"); ax.set_ylabel("频数")
ax.set_title("半衰期估计量的抽样分布（20 个独立种子）")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/halflife-sampling.png"); plt.close(fig)

summary = {
    "hl_true": HL_true, "hl_hat": round(float(hl_hat), 2), "hl_logfit": round(float(hl_logfit), 2),
    "hl_mc_mean": round(float(np.mean(hls)), 2), "hl_mc_std": round(float(np.std(hls)), 2),
    "hl_mc_min": round(float(np.min(hls)), 2), "hl_mc_max": round(float(np.max(hls)), 2),
    "ic1": round(float(ics[0]) * 100, 2), "a_hat": round(float(a_hat) * 100, 2),
    "best_hold": {str(k): {"hold": v[0], "sharpe": round(v[1], 2)} for k, v in best.items()},
    "gross_sharpe_h1": round(res[0][0][0], 2),
    "net10_h1": round(res[10][0][0], 2), "net30_h1": round(res[30][0][0], 2),
    "net60_h1": round(res[60][0][0], 2),
    "ann_turn_h1": round(ann_turns[0], 1), "ann_turn_h8": round([r[1] for r in res[0]][holds.index(8)], 1),
    "ann_turn_h20": round([r[1] for r in res[0]][holds.index(20)], 1),
    "ema_best": {str(cb): {"span": spans[int(np.argmax([backtest_ema(s, cb) for s in spans]))],
                            "sharpe": round(max(backtest_ema(s, cb) for s in spans), 2)} for cb in [10, 30, 60]},
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
with open("/Users/halo/workspace/astro-blog/scripts/_halflife_results.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
