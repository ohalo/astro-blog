"""MCS 补充实验：势均力敌设定（RV 噪声 s=0.19，GARCH 与 HAR 接近平手）"""
import os, json, numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
OUT = 'public/images/model-confidence-set'
os.makedirs(OUT, exist_ok=True)

OM_T, A_T, B_T, G_T = 4e-6, 0.06, 0.88, 0.04

def simulate(T, s=0.19, seed=0):
    r_ = np.random.default_rng(seed)
    z = r_.standard_normal(T)
    h = np.empty(T); r = np.empty(T)
    h[0] = OM_T / (1 - A_T - B_T - 0.5 * G_T); r[0] = np.sqrt(h[0]) * z[0]
    for t in range(1, T):
        lev = G_T * (r[t-1] < 0) * r[t-1]**2
        h[t] = OM_T + A_T * r[t-1]**2 + B_T * h[t-1] + lev
        r[t] = np.sqrt(h[t]) * z[t]
    u = r_.normal(-0.5 * s**2, s, T)
    return r, h, h * np.exp(u)

def garch_nll(p, r, model):
    om, al, be = np.exp(p[0]), 1/(1+np.exp(-p[1])), 1/(1+np.exp(-p[2]))
    ga = np.exp(p[3]) if model == 'gjr' else 0.0
    if al + be + 0.5 * ga > 0.999: return 1e12
    n = len(r); h = np.empty(n); h[0] = np.var(r); lev = ga * (r < 0) * r**2
    for t in range(1, n):
        h[t] = om + al * r[t-1]**2 + be * h[t-1] + lev[t-1]
        if not np.isfinite(h[t]): return 1e12
    h = np.maximum(h, 1e-14)
    return float(np.sum(np.log(h) + r**2 / h))

def fit_garch(r, model='garch'):
    p0 = np.array([np.log(np.var(r) * 0.04), -2.7, 2.0, np.log(0.03)])
    p = minimize(garch_nll, p0, args=(r, model), method='Nelder-Mead',
                 options={'maxiter': 1500, 'maxfev': 1500, 'xatol': 1e-5, 'fatol': 1e-4}).x
    return (np.exp(p[0]), 1/(1+np.exp(-p[1])), 1/(1+np.exp(-p[2])),
            np.exp(p[3]) if model == 'gjr' else 0.0)

def garch_forecast(r, oos_start, n_oos, model='garch', refit_every=50, win=1250):
    fc = np.empty(n_oos); om = al = be = ga = None; h_prev = None
    lev_all = (r < 0) * r**2
    for i in range(n_oos):
        t = oos_start + i
        if om is None or (i % refit_every == 0):
            om, al, be, ga = fit_garch(r[max(0, t-win):t], model); h_prev = None
        if h_prev is None:
            x = r[max(0, t-win):t]; lv = ga * (x < 0) * x**2
            hh = np.empty(len(x)); hh[0] = np.var(x)
            for k in range(1, len(x)):
                hh[k] = om + al * x[k-1]**2 + be * hh[k-1] + lv[k-1]
            h_prev = hh[-1]
        fc[i] = om + al * r[t-1]**2 + be * h_prev + ga * lev_all[t-1]
        h_prev = fc[i]
    return fc

def past_mean(x, w):
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan); out[w:] = (cs[w+1:] - cs[:len(x)-w]) / w
    return out

def har_forecast(rv, oos_start, n_oos, refit_every=50):
    pm5 = past_mean(rv, 5); pm22 = past_mean(rv, 22)
    fc = np.empty(n_oos); beta = None
    for i in range(n_oos):
        t = oos_start + i
        if beta is None or (i % refit_every == 0):
            y = rv[23:t]
            X = np.column_stack([np.ones(t-23), rv[22:t-1], pm5[23:t], pm22[23:t]])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        fc[i] = beta[0] + beta[1]*rv[t-1] + beta[2]*pm5[t] + beta[3]*pm22[t]
    return fc

def ewma_fc(x, lam, oos_start, n_oos):
    fc = np.empty(n_oos)
    for i in range(n_oos):
        t = oos_start + i
        w = lam ** np.arange(t)[::-1]; w /= w.sum(); fc[i] = np.dot(w, x[:t])
    return fc

def ma_fc(x, w, oos_start, n_oos):
    return np.array([x[oos_start+i-w: oos_start+i].mean() for i in range(n_oos)])

def const_fc(x, oos_start, n_oos):
    cs = np.concatenate([[0.0], np.cumsum(x)])
    return cs[oos_start: oos_start+n_oos] / np.arange(oos_start, oos_start+n_oos)

def ar1_log_fc(rv, oos_start, n_oos, refit_every=50):
    l = np.log(np.clip(rv, 1e-12, None)); fc = np.empty(n_oos); beta = None; sv = None
    for i in range(n_oos):
        t = oos_start + i
        if beta is None or (i % refit_every == 0):
            y = l[1:t]; X = np.column_stack([np.ones(len(y)), l[:t-1]])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None); sv = np.var(y - X @ beta, ddof=2)
        fc[i] = np.exp(beta[0] + beta[1]*l[t-1] + 0.5*sv)
    return fc

def qlike(y, f):
    f = np.clip(f, 1e-13, None)
    return np.log(f) + y / f

def stat_boot_idx(T, p, r_):
    idx = np.empty(T, dtype=np.int64); t = 0
    while t < T:
        start = r_.integers(0, T); Lb = r_.geometric(p); end = min(t + Lb, T)
        idx[t:end] = (start + np.arange(end - t)) % T; t = end
    return idx

def mcs(L, B=1500, Lb=20, alpha=0.10, seed=7):
    T, K = L.shape; mu = L.mean(0)
    r_ = np.random.default_rng(seed); p = 1.0 / Lb
    M = np.empty((B, K))
    for bb in range(B):
        M[bb] = L[stat_boot_idx(T, p, r_)].mean(0)
    D = M - mu
    alive = list(range(K)); order = []; pvals = {}
    while len(alive) > 1:
        A = np.array(alive); m = len(A); iu = np.triu_indices(m, 1)
        d_obs = mu[A][:, None] - mu[A][None, :]
        db = D[:, A]; vb = db[:, :, None] - db[:, None, :]
        var = (vb**2).mean(0) + 1e-18
        t_ij = d_obs / np.sqrt(var)
        Tstat = float(np.abs(t_ij[iu]).max())
        Tb = np.abs(vb[:, iu[0], iu[1]] / np.sqrt(var[iu])).max(1)
        pv = float((Tb > Tstat).mean())
        if pv >= alpha:
            for k in A: pvals[int(k)] = pv
            break
        e = np.array([t_ij[i][np.arange(m) != i].max() for i in range(m)])
        worst = int(A[int(np.argmax(e))]); order.append(worst); pvals[worst] = pv
        alive.remove(worst)
    return order, pvals, alive, None

T = 4200; OOS_START = 2200; N_OOS = 2000
NAMES = ['GARCH(1,1)', 'GJR-GARCH', 'HAR-RV', 'EWMA-RV(0.94)', 'EWMA-RV(0.97)',
         'EWMA-r2(0.94)', 'MA22-RV', 'AR(1)-logRV', '常数方差']
r, h, rv = simulate(T, s=0.19, seed=11)
F = {'GARCH(1,1)': garch_forecast(r, OOS_START, N_OOS, 'garch'),
     'GJR-GARCH': garch_forecast(r, OOS_START, N_OOS, 'gjr'),
     'HAR-RV': har_forecast(rv, OOS_START, N_OOS),
     'EWMA-RV(0.94)': ewma_fc(rv, 0.94, OOS_START, N_OOS),
     'EWMA-RV(0.97)': ewma_fc(rv, 0.97, OOS_START, N_OOS),
     'EWMA-r2(0.94)': ewma_fc(r**2, 0.94, OOS_START, N_OOS),
     'MA22-RV': ma_fc(rv, 22, OOS_START, N_OOS),
     'AR(1)-logRV': ar1_log_fc(rv, OOS_START, N_OOS),
     '常数方差': const_fc(rv, OOS_START, N_OOS)}
y = rv[OOS_START:]
LQ = np.column_stack([qlike(y, F[n]) for n in NAMES])
res = {'s': 0.19, 'mean_qlike': {n: float(LQ[:, i].mean()) for i, n in enumerate(NAMES)}}
print('=== 势均力敌设定 s=0.19（2000 天 OOS）===')
for i in np.argsort(LQ.mean(0)):
    print(f'  {NAMES[i]:16s} {LQ[:,i].mean():.6f}')
print(f"  GARCH 与 HAR 的平均损失差 = {LQ[:,0].mean()-LQ[:,2].mean():+.6f}")
res['gap_garch_har'] = float(LQ[:, 0].mean() - LQ[:, 2].mean())

o_, p_, m_, _ = mcs(LQ, B=2000, Lb=20, alpha=0.10, seed=7)
res['mcs2000'] = [NAMES[k] for k in m_]; res['pvals'] = {NAMES[k]: float(v) for k, v in p_.items()}
print('  MCS(2000天):', res['mcs2000'])
print('  p 值:', {NAMES[k]: round(v, 3) for k, v in p_.items()})

print('=== 样本长度扫描 ===')
lens = [250, 500, 1000, 1500, 2000]; sw = {}
for Ln in lens:
    o2, p2, m2, _ = mcs(LQ[:Ln], B=1500, Lb=20, alpha=0.10, seed=29)
    sw[Ln] = dict(size=len(m2), best=NAMES[int(np.argmin(LQ[:Ln].mean(0)))],
                  mcs=[NAMES[k] for k in m2])
    print(f'  T={Ln:5d}  MCS 大小 {len(m2)}  单次最优 {sw[Ln]["best"]}  {[NAMES[k] for k in m2]}')
res['len_sweep'] = sw

print('=== 30 次重复（每次 500 天 OOS）===')
best_cnt = {n: 0 for n in NAMES}; mcs_cnt = {n: 0 for n in NAMES}; mcs_size = []
for rep in range(30):
    rr, hh, rvv = simulate(T, s=0.19, seed=100 + rep)
    Fr = {'GARCH(1,1)': garch_forecast(rr, OOS_START, 500, 'garch', refit_every=100),
          'GJR-GARCH': garch_forecast(rr, OOS_START, 500, 'gjr', refit_every=100),
          'HAR-RV': har_forecast(rvv, OOS_START, 500),
          'EWMA-RV(0.94)': ewma_fc(rvv, 0.94, OOS_START, 500),
          'EWMA-RV(0.97)': ewma_fc(rvv, 0.97, OOS_START, 500),
          'EWMA-r2(0.94)': ewma_fc(rr**2, 0.94, OOS_START, 500),
          'MA22-RV': ma_fc(rvv, 22, OOS_START, 500),
          'AR(1)-logRV': ar1_log_fc(rvv, OOS_START, 500),
          '常数方差': const_fc(rvv, OOS_START, 500)}
    LL = np.column_stack([qlike(rvv[OOS_START:OOS_START+500], Fr[n]) for n in NAMES])
    best_cnt[NAMES[int(np.argmin(LL.mean(0)))]] += 1
    o3, p3, m3, _ = mcs(LL, B=800, Lb=20, alpha=0.10, seed=200 + rep)
    for k in m3: mcs_cnt[NAMES[k]] += 1
    mcs_size.append(len(m3))
    print(f'  rep{rep}: best={NAMES[int(np.argmin(LL.mean(0)))]}  mcs={len(m3)} '
          f'{[NAMES[k] for k in m3]}', flush=True)
res['rep_best'] = best_cnt; res['rep_mcs'] = mcs_cnt
res['rep_mcs_size_mean'] = float(np.mean(mcs_size))
print('单次最优频次:', best_cnt)
print('MCS 入选率:', mcs_cnt)
print('平均 MCS 大小:', np.mean(mcs_size))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
axes[0].plot(lens, [sw[l]['size'] for l in lens], 'o-', color='#2980b9', label='MCS 集合大小')
axes[0].plot(lens, [9 - sw[l]['size'] for l in lens], 's--', color='#c0392b', label='被剔除模型数')
axes[0].set_xlabel('样本外长度（交易日）'); axes[0].set_ylabel('数量 / 9 个候选')
axes[0].set_title('势均力敌设定下：样本越长，MCS 越有底气砍人')
axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
xs = np.arange(len(NAMES))
axes[1].bar(xs - .2, [best_cnt[n] for n in NAMES], .4, color='#e67e22', label='30 次里"单次最优"')
axes[1].bar(xs + .2, [mcs_cnt[n] for n in NAMES], .4, color='#27ae60', label='30 次里入选 MCS')
axes[1].set_xticks(xs); axes[1].set_xticklabels(NAMES, rotation=60, fontsize=7)
axes[1].set_ylabel('次数（共 30 次）'); axes[1].legend(fontsize=8)
axes[1].set_title('500 天样本：选冠军像抽奖，MCS 只剪输家'); axes[1].grid(alpha=.3, axis='y')
plt.tight_layout(); plt.savefig(f'{OUT}/robustness.png', dpi=130); plt.close()

with open('mcs2_results.json', 'w') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print('=== 完成 ===')
