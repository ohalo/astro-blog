"""模型置信集（Model Confidence Set, Hansen-Lunde-Nason 2003）数值实验。
生成 5 张真实计算图 + 打印关键数值。"""
import os, json, numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = 'public/images/model-confidence-set'
os.makedirs(OUT, exist_ok=True)

OM_T, A_T, B_T, G_T = 4e-6, 0.06, 0.88, 0.04      # 真值：ω, α, β, γ(杠杆)
LRV = OM_T / (1 - A_T - B_T - 0.5 * G_T)          # 长期日方差 = 1e-4

# ---------------- DGP: GARCH(1,1)+杠杆，RV 带测量噪声 ----------------
def simulate(T, s=0.35, seed=0, om=OM_T, a=A_T, b=B_T, gam=G_T):
    r_ = np.random.default_rng(seed)
    z = r_.standard_normal(T)
    h = np.empty(T); r = np.empty(T)
    h[0] = om / (1 - a - b - 0.5 * gam)
    r[0] = np.sqrt(h[0]) * z[0]
    for t in range(1, T):
        lev = gam * (r[t-1] < 0) * r[t-1]**2
        h[t] = om + a * r[t-1]**2 + b * h[t-1] + lev
        r[t] = np.sqrt(h[t]) * z[t]
    u = r_.normal(-0.5 * s**2, s, T)
    rv = h * np.exp(u)                             # E[RV|h] = h
    return r, h, rv

# ---------------- GARCH QMLE ----------------
def garch_nll(p, r, model):
    om, al, be = np.exp(p[0]), 1/(1+np.exp(-p[1])), 1/(1+np.exp(-p[2]))
    ga = np.exp(p[3]) if model == 'gjr' else 0.0
    if al + be + 0.5 * ga > 0.999: return 1e12
    n = len(r); h = np.empty(n); h[0] = np.var(r)
    lev = ga * (r < 0) * r**2
    for t in range(1, n):
        h[t] = om + al * r[t-1]**2 + be * h[t-1] + lev[t-1]
        if not np.isfinite(h[t]): return 1e12
    return float(np.sum(np.log(np.maximum(h, 1e-14)) + r**2 / np.maximum(h, 1e-14)))

def fit_garch(r, model='garch'):
    p0 = np.array([np.log(np.var(r) * 0.04), -2.7, 2.0, np.log(0.03)])
    res = minimize(garch_nll, p0, args=(r, model), method='Nelder-Mead',
                   options={'maxiter': 1500, 'maxfev': 1500, 'xatol': 1e-5, 'fatol': 1e-4})
    p = res.x
    return (np.exp(p[0]), 1/(1+np.exp(-p[1])), 1/(1+np.exp(-p[2])),
            np.exp(p[3]) if model == 'gjr' else 0.0)

def garch_forecast(r, oos_start, n_oos, model='garch', refit_every=50, win=1250):
    """滚动窗口重估 + 增量递推。fc[i] = 用 t-1 及之前信息对 rv[t] 的一步预测"""
    fc = np.empty(n_oos); om = al = be = ga = None; h_prev = None
    lev_all = (r < 0) * r**2
    for i in range(n_oos):
        t = oos_start + i                            # 目标 rv[t]
        if om is None or (i % refit_every == 0):
            om, al, be, ga = fit_garch(r[max(0, t-win):t], model)   # 只用 t 之前的数据
            h_prev = None
        if h_prev is None:                           # 重估后做一次全递推，得 h_{t-1}
            x = r[max(0, t-win):t]; lv = ga * (x < 0) * x**2
            hh = np.empty(len(x)); hh[0] = np.var(x)
            for k in range(1, len(x)):
                hh[k] = om + al * x[k-1]**2 + be * hh[k-1] + lv[k-1]
            h_prev = hh[-1]
        fc[i] = om + al * r[t-1]**2 + be * h_prev + ga * lev_all[t-1]
        h_prev = fc[i]                               # 下一轮的 h_{t-1}
    return fc

# ---------------- 其他模型 ----------------
def past_mean(x, w):
    """pm[i] = mean(x[i-w:i])，即 i 之前 w 个观测的均值"""
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    out[w:] = (cs[w+1:] - cs[:len(x)-w]) / w
    return out

def har_forecast(rv, oos_start, n_oos, refit_every=50):
    pm5 = past_mean(rv, 5); pm22 = past_mean(rv, 22)
    fc = np.empty(n_oos); beta = None
    for i in range(n_oos):
        t = oos_start + i
        if beta is None or (i % refit_every == 0):
            y = rv[23:t]                             # 目标 rv[k], k=23..t-1
            X = np.column_stack([np.ones(t-23), rv[22:t-1], pm5[23:t], pm22[23:t]])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        fc[i] = beta[0] + beta[1]*rv[t-1] + beta[2]*pm5[t] + beta[3]*pm22[t]
    return fc

def ewma_fc(x, lam, oos_start, n_oos):
    fc = np.empty(n_oos)
    for i in range(n_oos):
        t = oos_start + i
        w = lam ** np.arange(t)[::-1]; w /= w.sum()
        fc[i] = np.dot(w, x[:t])
    return fc

def ma_fc(x, win, oos_start, n_oos):
    return np.array([x[oos_start+i-win: oos_start+i].mean() for i in range(n_oos)])

def const_fc(x, oos_start, n_oos):
    cs = np.concatenate([[0.0], np.cumsum(x)])
    # 目标 rv[t]（t=oos_start+i）用 rv[0..t-1] 的扩展窗口均值
    return cs[oos_start: oos_start+n_oos] / np.arange(oos_start, oos_start+n_oos)

def ar1_log_fc(rv, oos_start, n_oos, refit_every=50):
    l = np.log(np.clip(rv, 1e-12, None)); fc = np.empty(n_oos); beta = None; sv = None
    for i in range(n_oos):
        t = oos_start + i
        if beta is None or (i % refit_every == 0):
            y = l[1:t]; X = np.column_stack([np.ones(len(y)), l[:t-1]])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            sv = np.var(y - X @ beta, ddof=2)
        fc[i] = np.exp(beta[0] + beta[1]*l[t-1] + 0.5*sv)
    return fc

def qlike(y, f):
    f = np.clip(f, 1e-13, None)
    return np.log(f) + y / f

def mse(y, f):
    return (y - f)**2

# ---------------- 平稳 bootstrap + MCS ----------------
def stat_boot_idx(T, p, r_):
    idx = np.empty(T, dtype=np.int64); t = 0
    while t < T:
        start = r_.integers(0, T)
        Lb = r_.geometric(p)
        end = min(t + Lb, T); n = end - t
        idx[t:end] = (start + np.arange(n)) % T
        t = end
    return idx

def boot_means(L, B, Lb, seed):
    T, K = L.shape
    r_ = np.random.default_rng(seed); p = 1.0 / Lb
    M = np.empty((B, K))
    for bb in range(B):
        M[bb] = L[stat_boot_idx(T, p, r_)].mean(0)
    return M

def mcs(L, B=2000, Lb=20, alpha=0.10, seed=7):
    T, K = L.shape
    mu = L.mean(0)
    D = boot_means(L, B, Lb, seed) - mu
    alive = list(range(K)); order = []; pvals = {}; Tb_first = None
    while len(alive) > 1:
        A = np.array(alive); m = len(A)
        iu = np.triu_indices(m, 1)
        d_obs = mu[A][:, None] - mu[A][None, :]
        db = D[:, A]
        vb = db[:, :, None] - db[:, None, :]
        var = (vb**2).mean(0) + 1e-18
        t_ij = d_obs / np.sqrt(var)
        Tstat = float(np.abs(t_ij[iu]).max())
        Tb = np.abs(vb[:, iu[0], iu[1]] / np.sqrt(var[iu])).max(1)
        if Tb_first is None: Tb_first = Tb
        pv = float((Tb > Tstat).mean())
        if pv >= alpha:
            for k in A: pvals[int(k)] = pv
            break
        e = np.array([t_ij[i][np.arange(m) != i].max() for i in range(m)])
        worst = int(A[int(np.argmax(e))])
        order.append(worst); pvals[worst] = pv; alive.remove(worst)
    return order, pvals, alive, Tb_first

# ================= 主实验 =================
T = 4200; OOS_START = 2200; N_OOS = T - OOS_START
r, h, rv = simulate(T, s=0.35, seed=11)
print(f'DGP 检查：日收益 std={r.std():.5f}  长期方差={LRV:.2e}  '
      f'RV 均值={rv.mean():.3e}  真值方差均值={h.mean():.3e}', flush=True)

NAMES = ['GARCH(1,1)', 'GJR-GARCH', 'HAR-RV', 'EWMA-RV(0.94)', 'EWMA-RV(0.97)',
         'EWMA-r2(0.94)', 'MA22-RV', 'AR(1)-logRV', '常数方差']
print('=== 生成各模型 OOS 预测 ===', flush=True)
F = {}
F[NAMES[0]] = garch_forecast(r, OOS_START, N_OOS, 'garch')
F[NAMES[1]] = garch_forecast(r, OOS_START, N_OOS, 'gjr')
F[NAMES[2]] = har_forecast(rv, OOS_START, N_OOS)
F[NAMES[3]] = ewma_fc(rv, 0.94, OOS_START, N_OOS)
F[NAMES[4]] = ewma_fc(rv, 0.97, OOS_START, N_OOS)
F[NAMES[5]] = ewma_fc(r**2, 0.94, OOS_START, N_OOS)
F[NAMES[6]] = ma_fc(rv, 22, OOS_START, N_OOS)
F[NAMES[7]] = ar1_log_fc(rv, OOS_START, N_OOS)
F[NAMES[8]] = const_fc(rv, OOS_START, N_OOS)

y = rv[OOS_START:]; K = len(NAMES)
LQ = np.column_stack([qlike(y, F[n]) for n in NAMES])
LM = np.column_stack([mse(y, F[n]) for n in NAMES])

print('=== MCS ===', flush=True)
order_q, pvals_q, mcs_q, Tb = mcs(LQ, B=2000, Lb=20, alpha=0.10, seed=7)
order_m, pvals_m, mcs_m, _ = mcs(LM, B=2000, Lb=20, alpha=0.10, seed=7)
res = {'mean_qlike': {NAMES[k]: float(LQ[:, k].mean()) for k in range(K)},
       'mean_mse': {NAMES[k]: float(LM[:, k].mean()*1e8) for k in range(K)},
       'elim_order_q': [NAMES[k] for k in order_q],
       'pvals_q': {NAMES[k]: float(v) for k, v in pvals_q.items()},
       'mcs_q': [NAMES[k] for k in mcs_q],
       'mcs_m': [NAMES[k] for k in mcs_m]}
print('\n--- 平均 QLIKE（越低越好）---')
for k in np.argsort(LQ.mean(0)):
    print(f'  {NAMES[k]:16s} {LQ[:,k].mean():.6f}   MCS p={pvals_q.get(k, float("nan")):.3f}')
print('剔除顺序:', res['elim_order_q'])
print('MCS 集合 (QLIKE):', res['mcs_q'])
print('MCS 集合 (MSE):', res['mcs_m'])

# 图1：预测轨迹
fig, ax = plt.subplots(figsize=(11, 4.6))
seg = slice(200, 400); xx = np.arange(200, 400)
ax.plot(xx, np.sqrt(rv[OOS_START:][seg]) * np.sqrt(252) * 100, color='#333',
        lw=1.1, label='已实现波动 RV（年化 %）', alpha=.85)
for nm, c, ls in [('GARCH(1,1)', '#c0392b', '-'), ('HAR-RV', '#2980b9', '-'),
                  ('EWMA-r2(0.94)', '#27ae60', '--'), ('常数方差', '#8e44ad', ':')]:
    ax.plot(xx, np.sqrt(F[nm][seg]) * np.sqrt(252) * 100, color=c, lw=1.0, ls=ls, label=nm)
ax.set_xlabel('样本外交易日'); ax.set_ylabel('年化波动率 %')
ax.set_title('图1：一步预测 vs 已实现波动（样本外局部 200 天）')
ax.legend(fontsize=8, ncol=3); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{OUT}/forecast-path.png', dpi=130); plt.close()

# 图2：累积损失优势
fig, ax = plt.subplots(figsize=(11, 4.6))
for nm, c in [('GARCH(1,1)', '#c0392b'), ('GJR-GARCH', '#e67e22'), ('HAR-RV', '#2980b9'),
              ('EWMA-RV(0.94)', '#27ae60'), ('EWMA-r2(0.94)', '#16a085'),
              ('MA22-RV', '#8e44ad'), ('AR(1)-logRV', '#7f8c8d')]:
    d = np.cumsum(LQ[:, NAMES.index('常数方差')] - LQ[:, NAMES.index(nm)])
    ax.plot(d, color=c, lw=1.2, label=nm)
ax.axhline(0, color='k', lw=.8)
ax.set_xlabel('样本外交易日'); ax.set_ylabel('相对常数基准的累积 QLIKE 优势')
ax.set_title('图2：累积损失优势——领先者随时间反复换手')
ax.legend(fontsize=8, ncol=4); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{OUT}/cumulative-loss.png', dpi=130); plt.close()

# 图3：损失 + p 值
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
o = np.argsort(LQ.mean(0))
axes[0].barh([NAMES[i] for i in o], LQ.mean(0)[o], color='#34495e')
axes[0].set_xlabel('平均 QLIKE'); axes[0].set_title('样本外平均损失（越低越好）')
axes[0].grid(alpha=.3, axis='x')
pv = [pvals_q.get(i, 1.0) for i in o]
axes[1].barh([NAMES[i] for i in o], pv,
             color=['#c0392b' if v < 0.10 else '#27ae60' for v in pv])
axes[1].axvline(0.10, color='k', ls='--', lw=1)
axes[1].set_xlabel('MCS p 值'); axes[1].set_title('MCS p 值（虚线左侧=被剔除）')
axes[1].grid(alpha=.3, axis='x')
plt.tight_layout(); plt.savefig(f'{OUT}/loss-and-pvalues.png', dpi=130); plt.close()

# 图4：t 矩阵 + bootstrap 分布
mu = LQ.mean(0)
D_ = boot_means(LQ, 2000, 20, 7) - mu
vb = D_[:, :, None] - D_[:, None, :]
var = (vb**2).mean(0) + 1e-18
t_ij = (mu[:, None] - mu[None, :]) / np.sqrt(var)
tri = np.triu_indices(K, 1)
Tstat = float(np.abs(t_ij[tri]).max())
Tb = np.abs(vb[:, tri[0], tri[1]] / np.sqrt(var[tri])).max(1)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
im = axes[0].imshow(np.abs(t_ij), cmap='YlOrRd')
axes[0].set_xticks(range(K)); axes[0].set_xticklabels(NAMES, rotation=60, fontsize=7)
axes[0].set_yticks(range(K)); axes[0].set_yticklabels(NAMES, fontsize=7)
axes[0].set_title('|t| 统计量矩阵（行模型 − 列模型）')
plt.colorbar(im, ax=axes[0], fraction=.046)
axes[1].hist(Tb, bins=60, color='#95a5a6', label='bootstrap $T^*$ 分布')
axes[1].axvline(Tstat, color='#c0392b', lw=2, label=f'观测 T={Tstat:.2f}')
axes[1].set_xlabel('max|t| 统计量'); axes[1].set_ylabel('频数')
axes[1].set_title(f'$H_0$：全部模型等价（p={np.mean(Tb>Tstat):.3f}）')
axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{OUT}/tmatrix-bootstrap.png', dpi=130); plt.close()
res['T_obs'] = Tstat; res['p_first'] = float(np.mean(Tb > Tstat))

# 块长敏感性
print('=== 块长敏感性 ===', flush=True)
blk = {}
for Lb in [1, 5, 10, 20, 50, 100]:
    o_, p_, m_, _ = mcs(LQ, B=1500, Lb=Lb, alpha=0.10, seed=13)
    blk[Lb] = {'mcs': [NAMES[k] for k in m_], 'size': len(m_), 'n_elim': len(o_)}
    print(f'  Lb={Lb:4d} 剔除 {len(o_)} 个  MCS: {[NAMES[k] for k in m_]}', flush=True)
res['block_sens'] = blk

# 噪声敏感性
print('=== 噪声敏感性 ===', flush=True)
noise = {}
for s in [0.15, 0.35, 0.60, 0.90]:
    r2, h2, rv2 = simulate(T, s=s, seed=23)
    F2 = {'GARCH(1,1)': garch_forecast(r2, OOS_START, N_OOS, 'garch'),
          'HAR-RV': har_forecast(rv2, OOS_START, N_OOS),
          'EWMA-RV(0.94)': ewma_fc(rv2, 0.94, OOS_START, N_OOS),
          'MA22-RV': ma_fc(rv2, 22, OOS_START, N_OOS),
          '常数方差': const_fc(rv2, OOS_START, N_OOS)}
    nn = list(F2)
    LL = np.column_stack([qlike(rv2[OOS_START:], F2[n]) for n in nn])
    o_, p_, m_, _ = mcs(LL, B=1200, Lb=20, alpha=0.10, seed=17)
    noise[s] = {'mean': {n: float(LL[:, i].mean()) for i, n in enumerate(nn)},
                'mcs': [nn[k] for k in m_]}
    print(f'  s={s}: ' + ', '.join(f'{n}={LL[:,i].mean():.4f}' for i, n in enumerate(nn)))
    print(f'        MCS: {[nn[k] for k in m_]}', flush=True)
res['noise_sens'] = noise

# 短样本（500 天）下的块长敏感性
print('=== 块长敏感性（500 天短样本）===', flush=True)
blk5 = {}
for Lb in [1, 5, 10, 20, 50, 100]:
    o_, p_, m_, _ = mcs(LQ[:500], B=1500, Lb=Lb, alpha=0.10, seed=13)
    blk5[Lb] = {'size': len(m_), 'n_elim': len(o_), 'mcs': [NAMES[k] for k in m_]}
    print(f'  Lb={Lb:4d} 剔除 {len(o_)} 个  MCS: {[NAMES[k] for k in m_]}', flush=True)
res['block_sens_500'] = blk5

# 样本长度扫描
print('=== 样本长度扫描 ===', flush=True)
lens = [250, 500, 1000, 1500, 2000]
sw = {}
for Ln in lens:
    o_, p_, m_, _ = mcs(LQ[:Ln], B=1500, Lb=20, alpha=0.10, seed=29)
    sw[Ln] = {'size': len(m_), 'best': NAMES[int(np.argmin(LQ[:Ln].mean(0)))],
              'mcs': [NAMES[k] for k in m_]}
    print(f'  T={Ln:5d}  MCS 大小 {len(m_)}  单次最优 {sw[Ln]["best"]}  '
          f'MCS: {[NAMES[k] for k in m_]}', flush=True)
res['len_sweep'] = sw

# 30 次独立重复（500 天 OOS，模拟"两年实盘比较"）
print('=== 30 次重复实验（每次 500 天 OOS）===', flush=True)
best_cnt = {n: 0 for n in NAMES}; mcs_cnt = {n: 0 for n in NAMES}
mcs_size = []; gap_dist = []
for rep in range(30):
    rr, hh, rvv = simulate(T, s=0.35, seed=100 + rep)
    Fr = {'GARCH(1,1)': garch_forecast(rr, OOS_START, 500, 'garch', refit_every=100),
          'GJR-GARCH': garch_forecast(rr, OOS_START, 500, 'gjr', refit_every=100),
          'HAR-RV': har_forecast(rvv, OOS_START, 500),
          'EWMA-RV(0.94)': ewma_fc(rvv, 0.94, OOS_START, 500),
          'EWMA-RV(0.97)': ewma_fc(rvv, 0.97, OOS_START, 500),
          'EWMA-r2(0.94)': ewma_fc(rr**2, 0.94, OOS_START, 500),
          'MA22-RV': ma_fc(rvv, 22, OOS_START, 500),
          'AR(1)-logRV': ar1_log_fc(rvv, OOS_START, 500),
          '常数方差': const_fc(rvv, OOS_START, 500)}
    nn = list(Fr)
    LL = np.column_stack([qlike(rvv[OOS_START:OOS_START+500], Fr[n]) for n in nn])
    bm = int(np.argmin(LL.mean(0))); best_cnt[nn[bm]] += 1
    o_, p_, m_, _ = mcs(LL, B=800, Lb=20, alpha=0.10, seed=200 + rep)
    for k in m_: mcs_cnt[nn[k]] += 1
    mcs_size.append(len(m_))
    so = np.sort(LL.mean(0)); gap_dist.append(so[1] - so[0])
    print(f'  rep{rep}: best={nn[bm]}  mcs_size={len(m_)}', flush=True)
res['rep_best'] = best_cnt; res['rep_mcs'] = mcs_cnt
res['rep_mcs_size_mean'] = float(np.mean(mcs_size))
res['rep_gap_mean'] = float(np.mean(gap_dist))
print('单次最优频次:', best_cnt)
print('MCS 入选率:', mcs_cnt)
print('平均 MCS 大小:', np.mean(mcs_size), '平均最优-次优差距:', np.mean(gap_dist))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
axes[0].plot(lens, [sw[l]['size'] for l in lens], 'o-', color='#2980b9', label='MCS 集合大小')
axes[0].plot(lens, [K - sw[l]['size'] for l in lens], 's--', color='#c0392b', label='被剔除模型数')
axes[0].set_xlabel('样本外长度（交易日）'); axes[0].set_ylabel('数量 / 9 个候选')
axes[0].set_title('样本越长，MCS 越有底气砍人')
axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
xs = np.arange(K)
axes[1].bar(xs - .2, [best_cnt[n] for n in NAMES], .4, color='#e67e22', label='30 次里"单次最优"')
axes[1].bar(xs + .2, [mcs_cnt[n] for n in NAMES], .4, color='#27ae60', label='30 次里入选 MCS')
axes[1].set_xticks(xs); axes[1].set_xticklabels(NAMES, rotation=60, fontsize=7)
axes[1].set_ylabel('次数（共 30 次）'); axes[1].legend(fontsize=8)
axes[1].set_title('500 天样本：选冠军像抽奖，MCS 只剪输家'); axes[1].grid(alpha=.3, axis='y')
plt.tight_layout(); plt.savefig(f'{OUT}/robustness.png', dpi=130); plt.close()

with open('mcs_results.json', 'w') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print('\n=== 完成 ===')
