#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""波动率的波动率 (vol-of-vol) 因子受控模拟"""
import numpy as np, os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for f in ["PingFang SC", "Heiti SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    if any(f == x.name for x in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]; break
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/volatility-of-volatility-factor"
os.makedirs(OUT, exist_ok=True)
R = {}

N, T = 400, 2520          # 400 只股票, 10 年
rng = np.random.default_rng(20260801)

def gen_market(rng, T, vov_market=0.28):
    """市场层：对数波动率 OU + vol-of-vol"""
    logv = np.zeros(T); logv[0] = np.log(0.011)
    kappa, theta = 0.02, np.log(0.011)
    for t in range(1, T):
        logv[t] = logv[t-1] + kappa*(theta - logv[t-1]) + vov_market*np.sqrt(1/252)*rng.normal()
    v = np.exp(logv)
    r = rng.normal(0, 1, T) * v
    return r, v

def gen_panel(rng, N, T, mkt_r, mkt_v, vov_spread=True, price_vov=True, lam=0.0):
    """
    每只股票有自己的 vol-of-vol 参数 xi_i。
    price_vov=True 时，高 xi 股票被赋予负的预期收益补偿（vov 是风险，投资者愿意付溢价规避）
    lam: vov 风险溢价强度（年化收益 = -lam * xi）
    """
    xi = rng.uniform(0.10, 0.75, N) if vov_spread else np.full(N, 0.35)
    beta = rng.uniform(0.6, 1.4, N)
    base_vol = rng.uniform(0.012, 0.028, N)

    logv = np.zeros((T, N))
    logv[0] = np.log(base_vol)
    kappa = 0.02
    theta = np.log(base_vol)
    dt = np.sqrt(1/252)
    shocks = rng.normal(0, 1, (T, N))
    for t in range(1, T):
        logv[t] = logv[t-1] + kappa*(theta - logv[t-1]) + xi*dt*shocks[t]
    vol = np.exp(logv)

    drift = (-lam * xi / 252.0) if price_vov else np.zeros(N)
    idio = rng.normal(0, 1, (T, N)) * vol
    r = beta[None, :]*mkt_r[:, None] + idio + drift[None, :]
    return r, vol, xi, beta, base_vol

# ---------- 估计量 ----------
def realized_vol(r, w=21):
    """滚动已实现波动（按窗口）"""
    T, N = r.shape
    out = np.full((T, N), np.nan)
    c = np.cumsum(np.vstack([np.zeros(N), r**2]), axis=0)
    s = c[w:] - c[:-w]
    out[w-1:] = np.sqrt(s / w)
    return out

def rolling_std_of_logvol(rv, w=63):
    """vol-of-vol 代理：log 已实现波动的滚动标准差（年化）"""
    lv = np.log(rv)
    T, N = lv.shape
    out = np.full((T, N), np.nan)
    for t in range(w, T):
        seg = lv[t-w+1:t+1]
        if np.isnan(seg).any():
            continue
        out[t] = np.std(np.diff(seg, axis=0), axis=0) * np.sqrt(252)
    return out

mkt_r, mkt_v = gen_market(rng, T)
LAM = 0.06
r, vol, xi, beta, base_vol = gen_panel(rng, N, T, mkt_r, mkt_v, lam=LAM)

rv21 = realized_vol(r, 21)
vov_est = rolling_std_of_logvol(rv21, 63)

# ---------- 1. 估计量能否还原真实 xi ----------
valid = ~np.isnan(vov_est).all(axis=1)
vov_mean = np.nanmean(vov_est[valid], axis=0)
R["estimator"] = dict(
    corr_xi=float(np.corrcoef(vov_mean, xi)[0, 1]),
    slope=float(np.polyfit(xi, vov_mean, 1)[0]),
    intercept=float(np.polyfit(xi, vov_mean, 1)[1]),
    mean_est=float(vov_mean.mean()), mean_true=float(xi.mean()),
    bias_pct=float((vov_mean.mean()/xi.mean()-1)*100),
)

# 与常规波动率的相关（vov 是不是只是 vol 的马甲？）
avg_vol = np.nanmean(rv21[valid], axis=0)
R["vov_vs_vol"] = dict(
    corr_est=float(np.corrcoef(vov_mean, avg_vol)[0, 1]),
    corr_true=float(np.corrcoef(xi, base_vol)[0, 1]),
)

# ---------- 2. 单变量分组 vs 波动率中性分组 ----------
ann = r.mean(axis=0) * 252 * 100     # 各股实现年化收益(%)

def quintiles(score, val, n=5):
    o = np.argsort(score); k = len(score)//n
    return [float(val[o[i*k:(i+1)*k]].mean()) for i in range(n)]

R["sort_raw"] = quintiles(vov_mean, ann)
R["sort_raw_xi"] = quintiles(vov_mean, xi)
R["sort_raw_vol"] = quintiles(vov_mean, avg_vol*np.sqrt(252)*100)

# 波动率中性：先按 vol 分 5 组，组内再按 vov 分 5 组
o_vol = np.argsort(avg_vol); k = N//5
neutral = [[] for _ in range(5)]
for g in range(5):
    idx = o_vol[g*k:(g+1)*k]
    oo = idx[np.argsort(vov_mean[idx])]
    kk = len(oo)//5
    for q in range(5):
        neutral[q].extend(oo[q*kk:(q+1)*kk])
R["sort_neutral"] = [float(ann[np.array(g)].mean()) for g in neutral]
R["sort_neutral_vol"] = [float((avg_vol[np.array(g)]*np.sqrt(252)*100).mean()) for g in neutral]
R["sort_neutral_xi"] = [float(xi[np.array(g)].mean()) for g in neutral]

def tstat(a, b):
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
    return float(d), float(d/se)

d_raw, t_raw = tstat(ann[np.argsort(vov_mean)[:N//5]], ann[np.argsort(vov_mean)[-N//5:]])
d_neu, t_neu = tstat(ann[np.array(neutral[0])], ann[np.array(neutral[4])])
R["spread"] = dict(raw=d_raw, raw_t=t_raw, neutral=d_neu, neutral_t=t_neu)

# ---------- 3. 对照：无 vov 溢价的纯净市场 ----------
rng2 = np.random.default_rng(777)
mk2, _ = gen_market(rng2, T)
r0, vol0, xi0, _, bv0 = gen_panel(rng2, N, T, mk2, None, price_vov=False)
rv0 = realized_vol(r0, 21); vov0 = rolling_std_of_logvol(rv0, 63)
v0 = ~np.isnan(vov0).all(axis=1)
vm0 = np.nanmean(vov0[v0], axis=0); ann0 = r0.mean(axis=0)*252*100
av0 = np.nanmean(rv0[v0], axis=0)
o0 = np.argsort(av0); neu0 = [[] for _ in range(5)]
for g in range(5):
    idx = o0[g*k:(g+1)*k]; oo = idx[np.argsort(vm0[idx])]; kk = len(oo)//5
    for q in range(5): neu0[q].extend(oo[q*kk:(q+1)*kk])
d0, t0 = tstat(ann0[np.array(neu0[0])], ann0[np.array(neu0[4])])
R["control_null"] = dict(spread=d0, t=t0,
                         quints=[float(ann0[np.array(g)].mean()) for g in neu0],
                         err=[float(ann0[np.array(g)].std(ddof=1)/np.sqrt(len(g))) for g in neu0],
                         corr_xi=float(np.corrcoef(vm0, xi0)[0, 1]))
R["neutral_err"] = [float(ann[np.array(g)].std(ddof=1)/np.sqrt(len(g))) for g in neutral]

# ---------- 4. 估计窗口敏感性 ----------
win = {}
for w_rv, w_vov in [(10, 42), (21, 63), (21, 126), (42, 126), (63, 252)]:
    rvx = realized_vol(r, w_rv); vx = rolling_std_of_logvol(rvx, w_vov)
    vv = ~np.isnan(vx).all(axis=1)
    if vv.sum() < 100: continue
    vmx = np.nanmean(vx[vv], axis=0)
    avx = np.nanmean(rvx[vv], axis=0)
    ox = np.argsort(avx); nn = [[] for _ in range(5)]
    for g in range(5):
        idx = ox[g*k:(g+1)*k]; oo = idx[np.argsort(vmx[idx])]; kk = len(oo)//5
        for q in range(5): nn[q].extend(oo[q*kk:(q+1)*kk])
    dd, tt = tstat(ann[np.array(nn[0])], ann[np.array(nn[4])])
    win[f"{w_rv}/{w_vov}"] = dict(corr_xi=float(np.corrcoef(vmx, xi)[0,1]),
                                  spread=dd, t=tt)
R["window"] = win

# ---------- 5. 时序稳定性：vov 排名的持续性 ----------
half = T//2
def vov_by_period(sl):
    rvp = realized_vol(r[sl], 21); vp = rolling_std_of_logvol(rvp, 63)
    m = ~np.isnan(vp).all(axis=1)
    return np.nanmean(vp[m], axis=0)
v_h1 = vov_by_period(slice(0, half)); v_h2 = vov_by_period(slice(half, T))
R["persistence"] = dict(
    corr=float(np.corrcoef(v_h1, v_h2)[0,1]),
    rank_corr=float(np.corrcoef(np.argsort(np.argsort(v_h1)), np.argsort(np.argsort(v_h2)))[0,1]),
    top_stay=float(np.isin(np.argsort(v_h2)[-N//5:], np.argsort(v_h1)[-N//5:]).mean()*100),
)
# 对比：普通波动率的持续性
def vol_by_period(sl):
    rvp = realized_vol(r[sl], 21); m = ~np.isnan(rvp).all(axis=1)
    return np.nanmean(rvp[m], axis=0)
a1, a2 = vol_by_period(slice(0, half)), vol_by_period(slice(half, T))
R["persistence"]["vol_corr"] = float(np.corrcoef(a1, a2)[0,1])
R["persistence"]["vol_top_stay"] = float(np.isin(np.argsort(a2)[-N//5:], np.argsort(a1)[-N//5:]).mean()*100)

# ---------- 6. 多种子 ----------
seeds = []
for s in range(15):
    rg = np.random.default_rng(3000+s)
    mkx, _ = gen_market(rg, T)
    rx, _, xix, _, _ = gen_panel(rg, N, T, mkx, None, lam=LAM)
    rvx = realized_vol(rx, 21); vx = rolling_std_of_logvol(rvx, 63)
    vv = ~np.isnan(vx).all(axis=1)
    vmx = np.nanmean(vx[vv], axis=0); avx = np.nanmean(rvx[vv], axis=0)
    annx = rx.mean(axis=0)*252*100
    ox = np.argsort(avx); nn = [[] for _ in range(5)]
    for g in range(5):
        idx = ox[g*k:(g+1)*k]; oo = idx[np.argsort(vmx[idx])]; kk = len(oo)//5
        for q in range(5): nn[q].extend(oo[q*kk:(q+1)*kk])
    dd, tt = tstat(annx[np.array(nn[0])], annx[np.array(nn[4])])
    seeds.append((dd, tt, float(np.corrcoef(vmx, xix)[0,1])))
seeds = np.array(seeds)
R["seeds"] = dict(spread_mean=float(seeds[:,0].mean()), spread_std=float(seeds[:,0].std()),
                  spread_min=float(seeds[:,0].min()), spread_max=float(seeds[:,0].max()),
                  t_mean=float(seeds[:,1].mean()), n_pos=int((seeds[:,0]>0).sum()),
                  corr_mean=float(seeds[:,2].mean()))

# ---------- 6a. 噪声来源诊断 + 改进估计量 ----------
# 无重叠 RV 窗口：避免重叠窗口人为制造的平滑/噪声结构
def vov_nonoverlap(r_mat, w_rv=21, n_blocks=None):
    T_, N_ = r_mat.shape
    nb = T_ // w_rv
    rv = np.sqrt((r_mat[:nb*w_rv]**2).reshape(nb, w_rv, N_).mean(axis=1))
    lv = np.log(rv)
    return np.std(np.diff(lv, axis=0), axis=0) * np.sqrt(252/w_rv), nb

vov_no, nb_ = vov_nonoverlap(r, 21)
R["nonoverlap"] = dict(
    corr_xi=float(np.corrcoef(vov_no, xi)[0,1]),
    mean_est=float(vov_no.mean()), bias_pct=float((vov_no.mean()/xi.mean()-1)*100),
    n_blocks=int(nb_))

# RV 自身估计误差的理论噪声：log(RV_w) 的方差 ≈ 1/(2w)
for w_rv in [21, 63, 126]:
    vn, nbx = vov_nonoverlap(r, w_rv)
    # 理论：var(dlogRV) = var(真实变动) + 2*var(估计误差), 误差方差≈1/(2w)
    noise_var = 2*(1/(2*w_rv))
    corrected = np.sqrt(np.maximum((vn**2)*(w_rv/252) - noise_var, 1e-9))*np.sqrt(252/w_rv)
    R.setdefault("denoise", {})[w_rv] = dict(
        raw_mean=float(vn.mean()), raw_corr=float(np.corrcoef(vn, xi)[0,1]),
        corrected_mean=float(corrected.mean()), corrected_corr=float(np.corrcoef(corrected, xi)[0,1]),
        n_blocks=int(nbx), true_mean=float(xi.mean()))

# ---------- 6a2. 用最优估计量重跑分组 ----------
def neutral_sort(score, vol_ctrl, val, n=5):
    ov = np.argsort(vol_ctrl); kk2 = len(score)//n; g = [[] for _ in range(n)]
    for gi in range(n):
        idx = ov[gi*kk2:(gi+1)*kk2]; oo = idx[np.argsort(score[idx])]; q = len(oo)//n
        for qi in range(n): g[qi].extend(oo[qi*q:(qi+1)*q])
    return g, [float(val[np.array(x)].mean()) for x in g]

vov_best, _ = vov_nonoverlap(r, 63)
g_best, q_best = neutral_sort(vov_best, avg_vol, ann)
d_b, t_b = tstat(ann[np.array(g_best[0])], ann[np.array(g_best[4])])
R["best_estimator"] = dict(quints=q_best, spread=d_b, t=t_b,
                           corr_xi=float(np.corrcoef(vov_best, xi)[0,1]),
                           capture_vs_oracle=None)

# ---------- 6b. 上界：如果能观测到真实 xi ----------
o_true = np.argsort(avg_vol); neu_t = [[] for _ in range(5)]
for g in range(5):
    idx = o_true[g*k:(g+1)*k]; oo = idx[np.argsort(xi[idx])]; kk = len(oo)//5
    for q in range(5): neu_t[q].extend(oo[q*kk:(q+1)*kk])
d_t, t_t = tstat(ann[np.array(neu_t[0])], ann[np.array(neu_t[4])])
R["oracle"] = dict(spread=d_t, t=t_t,
                   quints=[float(ann[np.array(g)].mean()) for g in neu_t],
                   theoretical=float(LAM*100*(xi[np.array(neu_t[4])].mean()-xi[np.array(neu_t[0])].mean())),
                   capture=float(d_neu/d_t*100) if d_t != 0 else 0.0)
R["best_estimator"]["capture_vs_oracle"] = float(d_b/d_t*100) if d_t != 0 else 0.0
R["oracle"]["xi_spread_oracle"] = float(xi[np.array(neu_t[4])].mean()-xi[np.array(neu_t[0])].mean())
R["oracle"]["xi_spread_best"] = float(xi[np.array(g_best[4])].mean()-xi[np.array(g_best[0])].mean())
R["oracle"]["xi_spread_rolling"] = float(xi[np.array(neutral[4])].mean()-xi[np.array(neutral[0])].mean())

# ---------- 6c. 信噪比分解 ----------
# 估计量 = a + b*xi + noise ; 求噪声占比
b_, a_ = np.polyfit(xi, vov_mean, 1)
fit = a_ + b_*xi
R["snr"] = dict(
    var_signal=float(np.var(fit)), var_total=float(np.var(vov_mean)),
    signal_share=float(np.var(fit)/np.var(vov_mean)*100),
    noise_std=float(np.std(vov_mean-fit)), true_std=float(np.std(xi)),
)
# 样本长度对估计质量的影响
len_scan = {}
for yrs in [2, 5, 10, 20]:
    TT = yrs*252
    if TT <= T:
        vs = vov_est[:TT]
    else:
        rg = np.random.default_rng(555)
        mkl, _ = gen_market(rg, TT)
        rl, _, _, _, _ = gen_panel(rg, N, TT, mkl, None, lam=LAM)
        # 复用同一批 xi 不可行(重新抽), 仅用于噪声量级参考
        rvl = realized_vol(rl, 21); vs = rolling_std_of_logvol(rvl, 63)
    m = ~np.isnan(vs).all(axis=1)
    if m.sum() < 50: continue
    vmx = np.nanmean(vs[m], axis=0)
    len_scan[yrs] = float(np.corrcoef(vmx, xi)[0,1]) if TT <= T else None
R["len_scan"] = len_scan

# ---------- 7. 交易成本 ----------
# 月度重构，多低 vov / 空高 vov，等权
def build_ls(vov_series, rv_series, r_mat, rebal=21, cost_bp=0.0):
    T_, N_ = r_mat.shape
    pnl = np.zeros(T_); prev_w = np.zeros(N_); turn = 0.0
    for t in range(300, T_):
        if (t - 300) % rebal == 0:
            sc = vov_series[t]; vl = rv_series[t]
            if np.isnan(sc).any() or np.isnan(vl).any():
                w = prev_w
            else:
                ov = np.argsort(vl); grp = [[] for _ in range(5)]
                kk2 = N_//5
                for g in range(5):
                    idx = ov[g*kk2:(g+1)*kk2]; oo = idx[np.argsort(sc[idx])]
                    q = len(oo)//5
                    grp[0].extend(oo[:q]); grp[4].extend(oo[-q:])
                w = np.zeros(N_)
                w[np.array(grp[0])] = 1.0/len(grp[0])
                w[np.array(grp[4])] = -1.0/len(grp[4])
            turn += np.abs(w - prev_w).sum()
            pnl[t] -= np.abs(w - prev_w).sum() * cost_bp/1e4
            prev_w = w
        pnl[t] += float(prev_w @ r_mat[t])
    return pnl, turn

res_cost = {}
for cb in [0, 5, 15, 30]:
    p, tn = build_ls(vov_est, rv21, r, cost_bp=cb)
    seg = p[300:]
    res_cost[cb] = dict(ann=float(seg.mean()*252*100),
                        sharpe=float(seg.mean()/seg.std()*np.sqrt(252)) if seg.std()>0 else 0.0,
                        turnover=float(tn/((T-300)/252)))
R["cost"] = res_cost

# ---------- 绘图 ----------
c1, c2, c3, c4 = "#2c3e50", "#e74c3c", "#27ae60", "#f39c12"

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(xi, vov_mean, s=16, c=c3, alpha=.6)
z = np.polyfit(xi, vov_mean, 1); xs = np.linspace(xi.min(), xi.max(), 50)
ax[0].plot(xs, np.polyval(z, xs), c=c2, lw=2,
           label=f"斜率 {z[0]:.3f}，相关 {R['estimator']['corr_xi']:.3f}")
ax[0].plot(xs, xs, "k--", lw=1.2, label="理想 45 度线")
ax[0].set_xlabel("真实 vol-of-vol 参数 ξ"); ax[0].set_ylabel("估计的 vol-of-vol")
ax[0].set_title("估计量几乎括不住真实 vol-of-vol"); ax[0].legend(); ax[0].grid(alpha=.3)

ws = [21, 63, 126]
xw = np.arange(len(ws))
ax[1].bar(xw-.27, [R["denoise"][w]["raw_mean"] for w in ws], .27, color=c2, label="原始估计均值")
ax[1].bar(xw, [R["denoise"][w]["corrected_mean"] for w in ws], .27, color=c3, label="扇除噪声后")
ax[1].bar(xw+.27, [R["denoise"][w]["true_mean"] for w in ws], .27, color=c1, label="真实 ξ 均值")
ax[1].set_xticks(xw); ax[1].set_xticklabels([f"{w} 日" for w in ws])
ax[1].set_xlabel("已实现波动估计窗口"); ax[1].set_ylabel("vol-of-vol")
ax[1].set_title("偏误主要来自 RV 自身的估计误差")
ax[1].legend(); ax[1].grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig(f"{OUT}/estimator-validation.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
qs = ["Q1\n(低vov)", "Q2", "Q3", "Q4", "Q5\n(高vov)"]
xx = np.arange(5)
ax[0].bar(xx-.27, R["oracle"]["quints"], .27, color=c1, label="上界：按真实 ξ 排序")
ax[0].bar(xx, R["best_estimator"]["quints"], .27, color=c3, label="最优估计量（63日无重叠）")
ax[0].bar(xx+.27, R["sort_neutral"], .27, color=c2, label="常规估计量（21/63滚动）")
ax[0].set_xticks(xx); ax[0].set_xticklabels(qs)
ax[0].set_ylabel("实现年化收益 (%)")
ax[0].set_title("同一个世界，三种估计量三种结论")
ax[0].legend(fontsize=9); ax[0].grid(alpha=.3, axis="y")

ax[1].plot(xx, R["sort_raw_vol"], "o-", c=c2, lw=2, ms=8, label="单变量排序组的平均波动率")
ax[1].plot(xx, R["sort_neutral_vol"], "s-", c=c3, lw=2, ms=8, label="中性排序组的平均波动率")
ax[1].set_xticks(xx); ax[1].set_xticklabels(qs)
ax[1].set_ylabel("组内平均年化波动率 (%)")
ax[1].set_title("两种排序下的波动率暴露")
ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/sorting-comparison.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].bar(xx, R["control_null"]["quints"], color="#95a5a6",
          yerr=R["control_null"]["err"], capsize=5, ecolor="#2c3e50", error_kw=dict(lw=1.4))
ax[0].axhline(0, c="k", lw=1)
ax[0].set_xticks(xx); ax[0].set_xticklabels(qs)
ax[0].set_ylabel("实现年化收益 (%)")
ax[0].set_title(f"纯净对照：无 vov 溢价世界\n价差 {R['control_null']['spread']:.2f}%，t={R['control_null']['t']:.2f}（误差棒全重叠）", fontsize=11)
ax[0].grid(alpha=.3, axis="y")

wl = list(win.keys())
ax[1].bar(np.arange(len(wl))-.2, [win[w]["corr_xi"] for w in wl], .4, color=c3, label="与真实 ξ 相关")
ax2 = ax[1].twinx()
ax2.bar(np.arange(len(wl))+.2, [win[w]["t"] for w in wl], .4, color=c2, label="价差 t 值")
ax[1].set_xticks(np.arange(len(wl))); ax[1].set_xticklabels(wl)
ax[1].set_xlabel("已实现波动窗口 / vov 窗口（交易日）")
ax[1].set_ylabel("与真实 ξ 相关系数", color=c3)
ax2.set_ylabel("Q1−Q5 价差 t 值", color=c2)
ax[1].set_title("窗口选择：短窗噪声大，长窗丢时变性")
ax[1].grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig(f"{OUT}/control-window.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
cbs = [0, 5, 15, 30]
ax[0].plot(cbs, [res_cost[c]["sharpe"] for c in cbs], "o-", c=c2, lw=2.2, ms=9)
ax[0].axhline(0, c="k", ls="--", lw=1)
ax[0].set_xlabel("单边成本 (bp)"); ax[0].set_ylabel("多空组合 Sharpe")
ax[0].set_title(f"成本侵蚀（年化换手 {res_cost[0]['turnover']:.1f} 倍）")
ax[0].grid(alpha=.3)

ax[1].scatter(v_h1, v_h2, s=16, c=c1, alpha=.6, label=f"vov 前后半期相关 {R['persistence']['corr']:.3f}")
ax[1].scatter(a1*np.sqrt(252)*3, a2*np.sqrt(252)*3, s=10, c=c4, alpha=.35,
              label=f"对比：波动率本身相关 {R['persistence']['vol_corr']:.3f}（已缩放）")
ax[1].set_xlabel("前 5 年估计的 vol-of-vol"); ax[1].set_ylabel("后 5 年估计的 vol-of-vol")
ax[1].set_title(f"排名持续性：高 vov 组留存率 {R['persistence']['top_stay']:.1f}%")
ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/cost-persistence.png", dpi=130); plt.close()

print(json.dumps(R, ensure_ascii=False, indent=1))
