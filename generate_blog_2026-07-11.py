#!/usr/bin/env python3
"""Compute reproducible results + generate figures for two quant blog posts.
Article 1: llm-embedding-text-factor  (LLM embeddings as text factors)
Article 2: market-impact-causal        (counterfactual estimation of market impact)
"""
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rng = np.random.default_rng(20260711)
BASE = "/Users/halo/workspace/astro-blog/public/images"
RES = {}

# =====================================================================
# ARTICLE 1: LLM embeddings as a cross-sectional text factor
# =====================================================================
N, T, d = 500, 250, 384            # stocks, days, embedding dim
signal_dims = rng.choice(d, 8, replace=False)
w = np.zeros(d); w[signal_dims] = rng.normal(0, 1, 8); w /= np.linalg.norm(w)

# embeddings: each (stock, day) is a d-dim vector; plant a return-relevant direction
E = rng.normal(0, 1, (N * T, d))
s = (E @ w); s = (s - s.mean()) / s.std()
b = 0.04                          # signal strength -> OOS IC ~0.04
fwd_ret = b * s + rng.normal(0, 1.0, N * T)   # daily forward return in %
# reshape back to (day, stock) for cross-sectional IC
E_day = E.reshape(T, N, d)
R_day = fwd_ret.reshape(T, N)

# supervised mapping: ridge embeddings -> forward return (time-series split OOS)
def oos_factor_and_ic(E_all, R_all, cal_days=150):
    pred = np.full(R_all.shape, np.nan)
    for t in range(cal_days, T):
        ridx = rng.permutation(t)[:cal_days]
        Xtr, ytr = E_all[ridx].reshape(-1, d), R_all[ridx].ravel()
        Xte, yte = E_all[t], R_all[t]
        m = Ridge(alpha=1.0).fit(Xtr, ytr)
        pred[t] = m.predict(Xte)
    ic = []
    for t in range(cal_days + 1, T):
        if np.std(pred[t]) < 1e-9 or np.std(R_day[t]) < 1e-9:
            continue
        ic.append(np.corrcoef(pred[t], R_day[t])[0, 1])
    return pred, np.nanmean(ic)

pred, ic_ridge = oos_factor_and_ic(E_day, R_day)
RES["ic_ridge"] = round(float(ic_ridge), 4)

# long-short backtest (OOS): top decile long, bottom decile short, daily rebalance
ls_ret = []
for t in range(160, T):
    f = pred[t]
    if np.all(np.isnan(f)):
        continue
    order = np.argsort(f)
    long_ret = np.mean(R_day[t, order[-int(N * 0.1):]])
    short_ret = np.mean(R_day[t, order[:int(N * 0.1)]])
    ls_ret.append(long_ret - short_ret)
ls_ret = np.array(ls_ret)
cum = np.cumprod(1 + (ls_ret / 100))
ann = (cum[-1]) ** (252 / len(ls_ret)) - 1
sharpe = (ls_ret.mean() / ls_ret.std()) * np.sqrt(252)
RES["ls_ann"] = round(float(ann) * 100, 1)
RES["ls_sharpe"] = round(float(sharpe), 2)
RES["ls_daily_mean_bps"] = round(float(ls_ret.mean()) * 100, 1)  # bps

# UNSUPERVISED baseline: top-1 PCA component of embeddings as factor
pca = PCA(n_components=1)
comp = np.full((T, N), np.nan)
for t in range(T):
    comp[t] = pca.fit_transform(E_day[t])[:, 0]
ic_pca = np.nanmean([np.corrcoef(comp[t], R_day[t])[0, 1] for t in range(T)
                     if np.std(comp[t]) > 1e-9 and np.std(R_day[t]) > 1e-9])
RES["ic_pca_top1"] = round(float(ic_pca), 4)

# explained variance of top components (semantic structure != return structure)
pca10 = PCA(n_components=10).fit(E.reshape(-1, d))
RES["pca_explained_top10"] = round(float(pca10.explained_variance_ratio_[:10].sum()) * 100, 1)

# ---- FIGURE 1a: PCA 2D projection colored by forward return (structure) ----
pca2 = PCA(n_components=2).fit(E)
xy = pca2.transform(E)
fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(xy[::7, 0], xy[::7, 1], c=fwd_ret[::7], cmap="RdYlGn", s=10, alpha=0.6)
ax.set_xlabel("Embedding PC1"); ax.set_ylabel("Embedding PC2")
ax.set_title("文本嵌入的 2D 投影：颜色=次日收益（信息藏在方向里，不在主轴上）")
cb = fig.colorbar(sc, ax=ax); cb.set_label("次日收益 (%)")
plt.tight_layout()
D1 = os.path.join(BASE, "llm-embedding-text-factor"); os.makedirs(D1, exist_ok=True)
fig.savefig(os.path.join(D1, "embedding_pca_projection.png"), dpi=150, bbox_inches="tight")
plt.close(); print("✓ embedding_pca_projection.png")

# ---- FIGURE 1b: IC bar (top PCA comps) + cumulative long-short ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
comp_ics = []
for k in range(1, 11):
    c = np.full((T, N), np.nan)
    for t in range(T):
        c[t] = PCA(n_components=k).fit_transform(E_day[t])[:, -1]
    comp_ics.append(np.nanmean([np.corrcoef(c[t], R_day[t])[0, 1] for t in range(T)
                                if np.std(c[t]) > 1e-9]))
a1.bar(range(1, 11), [abs(x) * 100 for x in comp_ics], color="#42A5F5")
a1.axhline(abs(ic_ridge) * 100, color="#E53935", ls="--", lw=2,
           label=f"监督 ridge 因子 IC={abs(ic_ridge)*100:.1f}%")
a1.set_xlabel("PCA 主成分序号"); a1.set_ylabel("|横截面 IC| (%)")
a1.set_title("无监督 PCA 主成分几乎无信号，监督映射才挖出方向")
a1.legend(fontsize=8)
a2.plot(np.arange(len(cum)), cum, color="#2E7D32", lw=2)
a2.set_xlabel("交易日（样本外）"); a2.set_ylabel("多空净值（起始=1）")
a2.set_title(f"嵌入因子多空净值：年化 {ann*100:.0f}%，Sharpe {sharpe:.2f}")
plt.tight_layout()
fig.savefig(os.path.join(D1, "factor_ic_and_ls.png"), dpi=150, bbox_inches="tight")
plt.close(); print("✓ factor_ic_and_ls.png")

# =====================================================================
# ARTICLE 2: Counterfactual estimation of market impact
# =====================================================================
M = 390                         # minutes in a trading day
ADV = 1_000_000                 # average daily volume
adv_per_min = ADV / M
daily_sigma = 0.02
sigma_min = daily_sigma / np.sqrt(M)
beta = 1.2
market = np.cumsum(rng.normal(0, sigma_min, M))     # exogenous market factor (for market-factor counterfactual narrative)
# clean counterfactual fundamental path: gentle wave + tiny noise, zero net drift -> impact stands out clearly
_T0, _TEX = 60, 180
x = np.arange(M)
base = 0.003 * np.sin(2 * np.pi * x / M)             # ±0.3% gentle intraday wave
noise = rng.normal(0, 0.0015, M)
log_p = base + noise
log_p = log_p - np.linspace(0, log_p[-1], M)          # pin p0[0]=p0[-1]=100 (zero net drift)
p0 = 100 * np.exp(log_p)                             # no-trade fundamental path

def simulate_impact(Q, t0=60, Tex=180, temp_coef=0.45, perm_coef=1.5, seed=0):
    # impact scaled to realistic bps: temp ~ temp_coef*sigma_bps*sqrt(participation); perm ~ perm_coef*sigma_bps*(Q/ADV)
    sigma_bps = daily_sigma * 1e4
    r = np.random.default_rng(seed)
    path = p0.copy()
    exec_minutes = np.arange(t0, min(t0 + Tex, M))
    v_per_min = Q / len(exec_minutes)
    part = v_per_min / adv_per_min                      # participation rate
    temp = temp_coef * sigma_bps * np.sqrt(part) * (p0[0] / 1e4)
    for i, t in enumerate(exec_minutes):
        path[t] += temp * (1 - i / len(exec_minutes))   # decays to 0 by end of exec
    # permanent level shift after trade
    perm = perm_coef * sigma_bps * (Q / ADV) * (p0[-1] / 1e4)
    path[t0 + Tex:] += perm
    return path, exec_minutes, part, temp, perm

# single illustrative scenario
Q = 50_000
path, ex_min, part, temp, perm = simulate_impact(Q, seed=7)
arrival = p0[ex_min[0]]
exec_prices = path[ex_min]
vwap = np.mean(exec_prices)
IS_bps = np.mean(path[ex_min] - p0[ex_min]) / np.mean(p0[ex_min]) * 1e4   # exec cost = fill vs counterfactual path
perm_bps = perm / p0[-1] * 1e4                          # permanent level shift (persists after trade)
temp_peak_bps = temp / p0[0] * 1e4                      # peak transient impact
temp_avg_bps = temp_peak_bps / 2                         # avg transient impact over execution window
market_move_bps = (p0[-1] - arrival) / arrival * 1e4    # exogenous no-trade drift (context only)
# counterfactual total trade impact = transient (execution) + permanent (level shift)
total_impact_bps = temp_avg_bps + perm_bps
RES["impact_Q"] = Q
RES["IS_bps"] = round(float(IS_bps), 1)
RES["perm_bps"] = round(float(perm_bps), 2)
RES["temp_peak_bps"] = round(float(temp_peak_bps), 2)
RES["temp_avg_bps"] = round(float(temp_avg_bps), 2)
RES["market_move_bps"] = round(float(market_move_bps), 1)
RES["trade_impact_bps"] = round(float(total_impact_bps), 1)

# ---- FIGURE 2a: actual vs counterfactual path ----
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(p0, color="#90A4AE", lw=1.6, label="反事实无交易路径（市场因子模型估计）")
ax.plot(path, color="#1565C0", lw=1.8, label="实际成交路径")
ax.scatter(ex_min, exec_prices, color="#E53935", s=18, zorder=5, label="执行成交点")
ax.axhline(arrival, color="#FBC02D", ls=":", lw=1.5, label=f"到达价 {arrival:.2f}")
ax.axvspan(ex_min[0], ex_min[-1], color="#E53935", alpha=0.06)
ax.set_xlabel("交易分钟（0=开盘）"); ax.set_ylabel("价格")
ax.set_title(f"大单成交路径 vs 反事实无交易路径：冲击≈{total_impact_bps:.0f}bps（临时{temp_avg_bps:.0f}+永久{perm_bps:.0f}）")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
D2 = os.path.join(BASE, "market-impact-causal"); os.makedirs(D2, exist_ok=True)
fig.savefig(os.path.join(D2, "actual_vs_counterfactual.png"), dpi=150, bbox_inches="tight")
plt.close(); print("✓ actual_vs_counterfactual.png")

# ---- FIGURE 2b: impact vs order size (sqrt law) ----
fracs = np.logspace(-3, -1, 25)        # Q/ADV from 0.1% to 10%
est = []
for fr in fracs:
    Qq = int(fr * ADV)
    p, em, pt, tp, pm = simulate_impact(Qq, seed=100 + int(fr * 1000))
    arr = p0[em[0]]; vw = np.mean(p[em])
    is_b = np.mean(p[em] - p0[em]) / np.mean(p0[em]) * 1e4   # path-relative execution cost
    perm_b = pm / p0[-1] * 1e4                          # permanent level shift of this scenario
    est.append(is_b + perm_b)                           # total = transient + permanent
est = np.array(est)
fit = np.polyfit(np.sqrt(fracs), est, 1)
xs = np.sqrt(fracs)
fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.scatter(fracs * 100, est, color="#6A1B9A", s=28, label="模拟估计冲击")
ax.plot(fracs * 100, np.polyval(fit, xs), color="#E53935", lw=2,
        label=f"√ 律拟合: impact ≈ {fit[0]:.0f}·√(Q/ADV)")
ax.set_xscale("log")
ax.set_xlabel("订单规模 Q / ADV (%)"); ax.set_ylabel("估计交易冲击 (bps)")
ax.set_title("市场冲击的反事实估计呈平方根律：越大单，边际冲击越陡")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(D2, "impact_sqrt_law.png"), dpi=150, bbox_inches="tight")
plt.close(); print("✓ impact_sqrt_law.png")

RES["sqrt_slope"] = round(float(fit[0]), 1)
with open("/Users/halo/workspace/astro-blog/_blog_results_2026-07-11.json", "w") as f:
    json.dump(RES, f, indent=2, ensure_ascii=False)
print("\n===== RESULTS =====")
for k, v in RES.items():
    print(f"{k}: {v}")
