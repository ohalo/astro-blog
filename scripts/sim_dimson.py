#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""非同步交易与 Dimson Beta 受控模拟"""
import numpy as np, os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for f in ["PingFang SC", "Heiti SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    if any(f == x.name for x in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]; break
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/nonsynchronous-trading-dimson-beta"
os.makedirs(OUT, exist_ok=True)
R = {}

N, T = 300, 1500
rng = np.random.default_rng(20260801)

def build_market(rng, T):
    return rng.normal(0.0003, 0.011, T)

def simulate(rng, N, T, mkt, ptrade_vec, beta_true, sig_idio):
    """返回 观测收益矩阵 (T,N)"""
    true_r = beta_true[None, :] * mkt[:, None] + rng.normal(0, 1, (T, N)) * sig_idio[None, :]
    logp_true = np.cumsum(true_r, axis=0)
    trades = rng.random((T, N)) < ptrade_vec[None, :]
    trades[0, :] = True; trades[-1, :] = True
    obs_logp = np.empty((T, N))
    last = logp_true[0].copy()
    for t in range(T):
        m = trades[t]
        last = np.where(m, logp_true[t], last)
        obs_logp[t] = last
    obs_r = np.diff(obs_logp, axis=0, prepend=obs_logp[:1])
    return obs_r

def ols_beta(y, x):
    xc = x - x.mean(); yc = y - y.mean()
    return float(xc @ yc / (xc @ xc))

def dimson_beta(y, mkt, k=1):
    T = len(y)
    cols = [mkt[k + j : T - k + j] for j in range(-k, k + 1)]
    X = np.column_stack([np.ones(T - 2 * k)] + cols)
    yy = y[k : T - k]
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ coef
    dof = len(yy) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    w = np.zeros(X.shape[1]); w[1:] = 1.0
    se = float(np.sqrt(w @ cov @ w))
    return float(coef[1:].sum()), se

def ols_beta_se(y, x):
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = resid @ resid / (len(y) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    return float(coef[1]), float(np.sqrt(cov[1, 1]))

def scholes_williams(y, mkt):
    T = len(y)
    yy = y[1:T-1]
    b0 = ols_beta(yy, mkt[1:T-1])
    bm = ols_beta(yy, mkt[0:T-2])
    bp = ols_beta(yy, mkt[2:T])
    rho = ols_beta(mkt[1:T-1] - mkt[1:T-1].mean(), mkt[0:T-2])
    return (b0 + bm + bp) / (1 + 2 * rho)

# ============ 主实验 ============
mkt = build_market(rng, T)
beta_true = rng.uniform(0.4, 1.8, N)
sig_idio = rng.uniform(0.012, 0.030, N)
# 流动性分层：5 组，非交易概率 0 / 0.1 / 0.25 / 0.4 / 0.55
tiers = np.repeat(np.arange(5), N // 5)
nontrade = np.array([0.0, 0.10, 0.25, 0.40, 0.55])[tiers]
ptrade = 1.0 - nontrade

obs = simulate(rng, N, T, mkt, ptrade, beta_true, sig_idio)

b_ols = np.array([ols_beta(obs[:, i], mkt) for i in range(N)])
dim1 = np.array([dimson_beta(obs[:, i], mkt, 1) for i in range(N)])
b_d1, se_d1 = dim1[:, 0], dim1[:, 1]
dim3 = np.array([dimson_beta(obs[:, i], mkt, 3) for i in range(N)])
b_d3, se_d3 = dim3[:, 0], dim3[:, 1]
b_sw = np.array([scholes_williams(obs[:, i], mkt) for i in range(N)])
se_ols = np.array([ols_beta_se(obs[:, i], mkt)[1] for i in range(N)])

R["tier_table"] = []
for t in range(5):
    m = tiers == t
    R["tier_table"].append(dict(
        nontrade=float(nontrade[m][0]),
        true=float(beta_true[m].mean()),
        ols=float(b_ols[m].mean()),
        bias_pct=float((b_ols[m] / beta_true[m] - 1).mean() * 100),
        dimson1=float(b_d1[m].mean()),
        d1_bias=float((b_d1[m] / beta_true[m] - 1).mean() * 100),
        dimson3=float(b_d3[m].mean()),
        d3_bias=float((b_d3[m] / beta_true[m] - 1).mean() * 100),
        sw=float(b_sw[m].mean()),
        sw_bias=float((b_sw[m] / beta_true[m] - 1).mean() * 100),
        se_ols=float(se_ols[m].mean()),
        se_d1=float(se_d1[m].mean()),
        se_d3=float(se_d3[m].mean()),
    ))

R["rmse"] = dict(
    ols=float(np.sqrt(np.mean((b_ols - beta_true) ** 2))),
    d1=float(np.sqrt(np.mean((b_d1 - beta_true) ** 2))),
    d3=float(np.sqrt(np.mean((b_d3 - beta_true) ** 2))),
    sw=float(np.sqrt(np.mean((b_sw - beta_true) ** 2))),
)
R["corr_true"] = dict(
    ols=float(np.corrcoef(b_ols, beta_true)[0, 1]),
    d1=float(np.corrcoef(b_d1, beta_true)[0, 1]),
    d3=float(np.corrcoef(b_d3, beta_true)[0, 1]),
)

# ============ 对照 1：全流动（无非同步）Dimson 是否过度修正 ============
obs_liq = simulate(rng, N, T, mkt, np.ones(N), beta_true, sig_idio)
bo = np.array([ols_beta(obs_liq[:, i], mkt) for i in range(N)])
bd1 = np.array([dimson_beta(obs_liq[:, i], mkt, 1)[0] for i in range(N)])
bd3 = np.array([dimson_beta(obs_liq[:, i], mkt, 3)[0] for i in range(N)])
R["control_liquid"] = dict(
    ols_bias=float((bo / beta_true - 1).mean() * 100),
    d1_bias=float((bd1 / beta_true - 1).mean() * 100),
    d3_bias=float((bd3 / beta_true - 1).mean() * 100),
    ols_rmse=float(np.sqrt(np.mean((bo - beta_true) ** 2))),
    d1_rmse=float(np.sqrt(np.mean((bd1 - beta_true) ** 2))),
    d3_rmse=float(np.sqrt(np.mean((bd3 - beta_true) ** 2))),
)

# ============ 对照 2：时间聚合能否替代 Dimson ============
def aggregate(r, k):
    n = (len(r) // k) * k
    return r[:n].reshape(-1, k, r.shape[1] if r.ndim > 1 else 1).sum(axis=1).squeeze()

agg = {}
for k in [1, 5, 10, 21]:
    n = (T // k) * k
    ra = obs[:n].reshape(-1, k, N).sum(axis=1)
    ma = mkt[:n].reshape(-1, k).sum(axis=1)
    ba = np.array([ols_beta(ra[:, i], ma) for i in range(N)])
    agg[k] = dict(
        illiq_bias=float((ba[tiers == 4] / beta_true[tiers == 4] - 1).mean() * 100),
        all_bias=float((ba / beta_true - 1).mean() * 100),
        se=float(np.array([ols_beta_se(ra[:, i], ma)[1] for i in range(N)]).mean()),
        nobs=int(len(ma)),
    )
R["aggregation"] = agg

# ============ 对照 3：低 beta 异象是否可由测量误差伪造 ============
# 真实世界：预期收益严格 CAPM（无异象），alpha=0
ann_true = beta_true * 0.06   # 真实年化预期收益 = beta * 市场溢价 6%
q_ols = np.argsort(b_ols)
q_true = np.argsort(beta_true)
q_d3 = np.argsort(b_d3)
def quintile_stats(order, val, bt):
    out = []
    for q in range(5):
        idx = order[q * (N // 5):(q + 1) * (N // 5)]
        out.append(dict(ret=float(val[idx].mean() * 100),
                        beta_true=float(bt[idx].mean()),
                        illiq=float(nontrade[idx].mean())))
    return out
R["low_beta"] = dict(
    sort_ols=quintile_stats(q_ols, ann_true, beta_true),
    sort_true=quintile_stats(q_true, ann_true, beta_true),
    sort_d3=quintile_stats(q_d3, ann_true, beta_true),
)
# 按 OLS beta 排序后，Q1 组合的真实 beta 与其实现收益不匹配 -> 伪 alpha
q1 = q_ols[:N // 5]
R["fake_alpha"] = dict(
    q1_ols_beta=float(b_ols[q1].mean()),
    q1_true_beta=float(beta_true[q1].mean()),
    q1_ret=float(ann_true[q1].mean() * 100),
    implied_alpha=float((ann_true[q1].mean() - b_ols[q1].mean() * 0.06) * 100),
    q1_illiq_share=float((nontrade[q1] >= 0.40).mean() * 100),
)

# ============ 对照 4：置换 —— 打乱流动性归属，偏误-流动性关系应消失 ============
perm_slopes = []
for s in range(200):
    rg = np.random.default_rng(9000 + s)
    perm = rg.permutation(N)
    sl = np.polyfit(nontrade, b_ols[perm] / beta_true[perm] - 1, 1)[0]
    perm_slopes.append(sl)
perm_slopes = np.array(perm_slopes)
real_slope = np.polyfit(nontrade, b_ols / beta_true - 1, 1)[0]
R["permutation"] = dict(real=float(real_slope),
                        perm_mean=float(perm_slopes.mean()),
                        perm_p5=float(np.percentile(perm_slopes, 5)),
                        n_beat=int((perm_slopes <= real_slope).sum()))

# ============ 对照 5：多种子稳健性 ============
seed_bias, seed_d3 = [], []
for s in range(20):
    rg = np.random.default_rng(4000 + s)
    mk = build_market(rg, T)
    bt = rg.uniform(0.4, 1.8, N)
    si = rg.uniform(0.012, 0.030, N)
    ob = simulate(rg, N, T, mk, ptrade, bt, si)
    m4 = tiers == 4
    bb = np.array([ols_beta(ob[:, i], mk) for i in np.where(m4)[0]])
    dd = np.array([dimson_beta(ob[:, i], mk, 3)[0] for i in np.where(m4)[0]])
    seed_bias.append((bb / bt[m4] - 1).mean() * 100)
    seed_d3.append((dd / bt[m4] - 1).mean() * 100)
R["seeds"] = dict(ols_mean=float(np.mean(seed_bias)), ols_std=float(np.std(seed_bias)),
                  ols_min=float(np.min(seed_bias)), ols_max=float(np.max(seed_bias)),
                  d3_mean=float(np.mean(seed_d3)), d3_std=float(np.std(seed_d3)))

# ============ 对照 6：Dimson 各滞后项系数分布（信息从哪来） ============
lag_coefs = []
for i in np.where(tiers == 4)[0]:
    y = obs[:, i]; k = 3
    cols = [mkt[k + j: T - k + j] for j in range(-k, k + 1)]
    X = np.column_stack([np.ones(T - 2 * k)] + cols)
    c, *_ = np.linalg.lstsq(X, y[k:T - k], rcond=None)
    lag_coefs.append(c[1:] / beta_true[i])
lag_coefs = np.array(lag_coefs)
R["lag_profile"] = dict(labels=[f"t{j:+d}" if j else "t" for j in range(-3, 4)],
                        mean=[float(x) for x in lag_coefs.mean(axis=0)])

# ============ 绘图 ============
c_true, c_ols, c_d1, c_d3, c_sw = "#2c3e50", "#e74c3c", "#f39c12", "#27ae60", "#8e44ad"

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
x = np.array([r["nontrade"] for r in R["tier_table"]]) * 100
ax[0].plot(x, [r["bias_pct"] for r in R["tier_table"]], "o-", c=c_ols, lw=2, ms=8, label="普通 OLS beta")
ax[0].plot(x, [r["d1_bias"] for r in R["tier_table"]], "s-", c=c_d1, lw=2, ms=7, label="Dimson (1 阶)")
ax[0].plot(x, [r["d3_bias"] for r in R["tier_table"]], "^-", c=c_d3, lw=2, ms=7, label="Dimson (3 阶)")
ax[0].plot(x, [r["sw_bias"] for r in R["tier_table"]], "d-", c=c_sw, lw=2, ms=7, label="Scholes-Williams")
ax[0].axhline(0, c="k", ls="--", lw=1)
ax[0].set_xlabel("非交易日占比 (%)"); ax[0].set_ylabel("beta 估计偏误 (%)")
ax[0].set_title("非同步交易把 beta 系统性压低"); ax[0].legend(); ax[0].grid(alpha=.3)

ax[1].scatter(beta_true[tiers == 4], b_ols[tiers == 4], s=22, c=c_ols, alpha=.65, label="OLS（最不流动组）")
ax[1].scatter(beta_true[tiers == 4], b_d3[tiers == 4], s=22, c=c_d3, alpha=.65, label="Dimson 3 阶")
lim = [0.2, 2.0]
ax[1].plot(lim, lim, "k--", lw=1.2, label="无偏参考线")
ax[1].set_xlabel("真实 beta"); ax[1].set_ylabel("估计 beta")
ax[1].set_title("最不流动组：估计值 vs 真实值"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/bias-by-liquidity.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
labels = ["OLS", "Dimson1", "Dimson3", "S-W"]
rm = [R["rmse"]["ols"], R["rmse"]["d1"], R["rmse"]["d3"], R["rmse"]["sw"]]
rl = [R["control_liquid"]["ols_rmse"], R["control_liquid"]["d1_rmse"], R["control_liquid"]["d3_rmse"], np.nan]
xx = np.arange(4)
ax[0].bar(xx - .19, rm, .38, color=c_ols, label="含非同步交易")
ax[0].bar(xx + .19, rl, .38, color="#95a5a6", label="全流动对照组")
ax[0].set_xticks(xx); ax[0].set_xticklabels(labels)
ax[0].set_ylabel("对真实 beta 的 RMSE"); ax[0].set_title("修正不是免费的：方差换偏误")
ax[0].legend(); ax[0].grid(alpha=.3, axis="y")

ax[1].bar(R["lag_profile"]["labels"], R["lag_profile"]["mean"], color=c_d3, alpha=.85)
ax[1].axhline(0, c="k", lw=1)
ax[1].set_ylabel("系数 / 真实 beta")
ax[1].set_title("最不流动组：Dimson 各期系数占比")
ax[1].grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig(f"{OUT}/rmse-lag-profile.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
qs = ["Q1\n(最低)", "Q2", "Q3", "Q4", "Q5\n(最高)"]
o = [q["ret"] for q in R["low_beta"]["sort_ols"]]
tq = [q["ret"] for q in R["low_beta"]["sort_true"]]
d = [q["ret"] for q in R["low_beta"]["sort_d3"]]
xx = np.arange(5)
ax[0].bar(xx - .26, tq, .26, color=c_true, label="按真实 beta 排序")
ax[0].bar(xx, o, .26, color=c_ols, label="按 OLS beta 排序")
ax[0].bar(xx + .26, d, .26, color=c_d3, label="按 Dimson beta 排序")
ax[0].set_xticks(xx); ax[0].set_xticklabels(qs)
ax[0].set_ylabel("组合真实预期年化收益 (%)")
ax[0].set_title("同一个无异象世界，三种排序给出三种结论")
ax[0].legend(); ax[0].grid(alpha=.3, axis="y")

ax2 = ax[1]
ax2.bar(xx, [q["illiq"] * 100 for q in R["low_beta"]["sort_ols"]], color="#16a085", alpha=.85)
ax2.set_xticks(xx); ax2.set_xticklabels(qs)
ax2.set_ylabel("组内平均非交易日占比 (%)")
ax2.set_title("按 OLS beta 排序 = 顺手按流动性排序")
ax2.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig(f"{OUT}/low-beta-artifact.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ks = [1, 5, 10, 21]
ax[0].plot(ks, [agg[k]["illiq_bias"] for k in ks], "o-", c=c_ols, lw=2, ms=8, label="最不流动组偏误")
ax[0].axhline(0, c="k", ls="--", lw=1)
ax[0].set_xlabel("收益聚合窗口（交易日）"); ax[0].set_ylabel("beta 偏误 (%)")
ax[0].set_title("拉长收益频率能治标"); ax[0].legend(); ax[0].grid(alpha=.3)
ax0b = ax[0].twinx()
ax0b.plot(ks, [agg[k]["se"] for k in ks], "s--", c="#7f8c8d", lw=1.6, label="平均标准误")
ax0b.set_ylabel("平均标准误", color="#7f8c8d")

ax[1].hist(perm_slopes, bins=30, color="#bdc3c7", edgecolor="w")
ax[1].axvline(real_slope, c=c_ols, lw=2.4, label=f"真实斜率 {real_slope:.3f}")
ax[1].axvline(np.percentile(perm_slopes, 5), c="#34495e", ls="--", lw=1.5, label="置换 5 分位")
ax[1].set_xlabel("偏误对非交易概率的回归斜率"); ax[1].set_ylabel("频次")
ax[1].set_title("置换检验（200 次打乱流动性归属）")
ax[1].legend(); ax[1].grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig(f"{OUT}/aggregation-permutation.png", dpi=130); plt.close()

print(json.dumps(R, ensure_ascii=False, indent=1))
