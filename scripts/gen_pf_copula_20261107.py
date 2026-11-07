#!/usr/bin/env python3
"""
生成 2 篇量化博客文章的真实计算数据与配图（v2 严格版，2026-11-07）

v1 的问题已全部修掉：
  - 尾部依赖系数改用正确公式，并用蒙特卡洛经验值对拍
  - 所有曲线来自真实模拟/拟合，不再用"乘系数"伪造
  - t-copula MLE 的似然返回标量，不再 try/except 回退真值矩阵
  - 粒子滤波重采样条件统一为 ESS < N/2，文章代码与本脚本同源
  - 图 4 真实保存 t=50/200/400 三个时点的粒子快照
  - Heston 用年化参数 (dt=1/252)，RV/EWMA 同步年化口径

输出：public/images/{slug}/ 下 5 张计算图 + cover.png + results.json
根目录同时保留 _pf_results.json / _copula_results.json（gitignore）
"""

import json
import os
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = Path('/Users/halo123456/workspace/astro-blog')
SEED_PF = 20261107          # 文章 PF 代码块用同一个 seed 可精确复现主实验
SEED_CP = 20261108          # 文章 Copula 代码块用同一个 seed 可精确复现主实验
SEED_PF_AUX = 20261117      # 辅助实验（粒子数敏感性 / 分位带 / 错配）独立流
SEED_CP_AUX = 20261118      # 大样本蒙特卡洛独立流

for fp in ['/System/Library/Fonts/Heiti TC.ttc', '/System/Library/Fonts/STHeiti Medium.ttc',
           '/System/Library/Fonts/Heiti TC.ttf']:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        break
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
DPI = 150


def save_json(slug, data):
    d = ROOT / 'public' / 'images' / slug
    (d / 'results.json').write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tag = 'pf' if 'particle' in slug else 'copula'
    (ROOT / f'_{tag}_results.json').write_text(json.dumps(data, indent=2, ensure_ascii=False))


# =====================================================================
# 文章 1：粒子滤波波动率估计
# =====================================================================

PF_PARAMS = dict(kappa=3.0, theta=0.04, sigma_v=0.5, v0=0.04, mu=0.05, dt=1/252)
T_PF, N_PF, W_RV = 500, 500, 20


def simulate_heston(T, rho, rng, kappa=3.0, theta=0.04, sigma_v=0.5,
                    v0=0.04, mu=0.05, dt=1/252):
    """Heston 随机波动率（年化参数）。z1 同时驱动方差与价格 => 杠杆效应。"""
    v = np.empty(T); logp = np.empty(T)
    v[0], logp[0] = v0, np.log(100.0)
    z1 = rng.standard_normal(T); z2 = rng.standard_normal(T)
    for t in range(1, T):
        vp = v[t-1]
        v[t] = max(vp + kappa*(theta-vp)*dt + sigma_v*np.sqrt(vp*dt)*z1[t], 1e-8)
        shock = rho*z1[t] + np.sqrt(1-rho**2)*z2[t]
        logp[t] = logp[t-1] + (mu - 0.5*v[t])*dt + np.sqrt(v[t]*dt)*shock
    return np.exp(logp), np.sqrt(v)


def particle_filter(prices, N=500, rng=None, ess_frac=0.5, snapshots=None,
                    kappa=3.0, theta=0.04, sigma_v=0.5, v0=0.04, mu=0.05, dt=1/252):
    """Bootstrap 粒子滤波。返回：后验波动率、ESS 序列、重采样次数、快照、一步预测。"""
    if rng is None:
        rng = np.random.default_rng(0)
    T = len(prices)
    r = np.diff(np.log(prices))
    vp = np.full(N, v0); w = np.full(N, 1.0/N)
    est = np.empty(T); pred = np.full(T, np.nan); ess_hist = np.empty(T)
    band_lo = np.full(T, np.nan); band_hi = np.full(T, np.nan)
    est[0] = np.sqrt(v0); ess_hist[0] = N
    resamples = 0; snap = {}

    for t in range(1, T):
        # 1) 状态转移（bootstrap：不利用当前观测引导粒子）
        z = rng.standard_normal(N)
        v_new = np.maximum(vp + kappa*(theta-vp)*dt
                           + sigma_v*np.sqrt(vp*dt)*z, 1e-8)
        # 2) 似然加权：r_t ~ N((mu - v/2)dt, v*dt)
        mean = (mu - 0.5*v_new)*dt
        sd = np.sqrt(v_new*dt)
        logw = -0.5*((r[t-1]-mean)/sd)**2 - np.log(sd)
        w = w*np.exp(logw-logw.max()); w /= w.sum()
        # 3) 后验估计 + 加权分位带（与 est 同一批粒子，完全自洽）
        est[t] = np.sqrt(np.dot(v_new, w))
        order = np.argsort(v_new)
        cw = np.cumsum(w[order]); sv = v_new[order]
        band_lo[t] = np.sqrt(np.interp(0.05, cw, sv))
        band_hi[t] = np.sqrt(np.interp(0.95, cw, sv))
        # 4) 一步预测：状态转移的条件期望（去掉零均值随机项）
        v_forecast = np.dot(v_new + kappa*(theta-v_new)*dt, w)
        pred[t] = np.sqrt(max(v_forecast, 1e-8))
        # 5) ESS 诊断 + 系统重采样
        ess = 1.0/np.dot(w, w); ess_hist[t] = ess
        if snapshots and t in snapshots:
            snap[t] = (v_new.copy(), w.copy())
        if ess < ess_frac*N:
            pos = (rng.random() + np.arange(N))/N
            cum = np.cumsum(w); cum[-1] = 1.0
            vp = v_new[np.searchsorted(cum, pos)]
            w = np.full(N, 1.0/N)
            resamples += 1
        else:
            vp = v_new
    return est, ess_hist, resamples, snap, pred, band_lo, band_hi


def realized_vol(prices, window=20, dt=1/252):
    r = np.diff(np.log(prices))
    out = np.full(len(prices), np.nan)
    for t in range(window, len(prices)):
        out[t] = np.sqrt(np.mean(r[t-window:t]**2)/dt)
    return out


def ewma_vol(prices, lam=0.94, dt=1/252):
    r = np.diff(np.log(prices))
    var = np.full(len(prices), np.nan)
    var[0] = r[0]**2
    for t in range(1, len(r)):
        var[t] = lam*var[t-1] + (1-lam)*r[t]**2
    return np.sqrt(var/dt)


def mae(est, tv, lo, hi):
    return float(np.mean(np.abs(est[lo:hi]-tv[lo:hi])))


def gen_particle_filter():
    slug = 'particle-filter-volatility'
    outdir = ROOT/'public'/'images'/slug
    outdir.mkdir(parents=True, exist_ok=True)
    HI = T_PF-1   # EWMA 末位无有效值，统一排除最后一个点

    # ---------- 主实验 A：模型匹配 (rho=0)，与文章代码块完全同源 ----------
    rng = np.random.default_rng(SEED_PF)
    px, tv = simulate_heston(T_PF, rho=0.0, rng=rng, **PF_PARAMS)
    pf, ess, nres, snap, pred_pf, band_lo, band_hi = particle_filter(
        px, N=N_PF, rng=np.random.default_rng(SEED_PF+1),
        snapshots={50, 200, 400}, **PF_PARAMS)
    rv = realized_vol(px, W_RV)
    ew = ewma_vol(px)

    pf_mae, rv_mae, ew_mae = [mae(x, tv, W_RV, HI) for x in (pf, rv, ew)]
    pf_corr = float(np.corrcoef(pf[W_RV:HI], tv[W_RV:HI])[0, 1])
    rv_corr = float(np.corrcoef(rv[W_RV:HI], tv[W_RV:HI])[0, 1])
    ew_corr = float(np.corrcoef(ew[W_RV:HI], tv[W_RV:HI])[0, 1])
    # 一步预测：pred[t] 与 rv[t] 都预测 t+1 时刻，对齐同一个真值
    actual_next = tv[W_RV+1:T_PF]
    pf_pred_mae = float(np.mean(np.abs(pred_pf[W_RV:HI]-actual_next)))
    rv_pred_mae = float(np.mean(np.abs(rv[W_RV:HI]-actual_next)))

    # 图 1：真值 + PF 后验均值 + 由同一批粒子真实算出的 5%–95% 分位带
    x = np.arange(T_PF)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(x, band_lo, band_hi, color='#e74c3c', alpha=.18,
                    label='PF 后验 5%–95% 分位带（由粒子实算）')
    ax.plot(x, tv, 'k-', lw=1.4, label='真值波动率')
    ax.plot(x, pf, color='#c0392b', lw=1.4, label='粒子滤波后验均值')
    ax.set_xlabel('交易日'); ax.set_ylabel('年化波动率')
    ax.set_ylim(0, max(tv.max(), np.nanmax(band_hi))*1.08)
    ax.set_title(f'模型匹配 (ρ=0)：PF MAE={pf_mae:.4f}')
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(outdir/'pf_1_paths.png', dpi=DPI); plt.close()

    # 图 2：三条基线同框
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, tv, 'k-', lw=1.5, label='真值')
    ax.plot(x, pf, color='#c0392b', lw=1.4, label=f'PF (MAE={pf_mae:.4f})')
    ax.plot(x, rv, color='#2980b9', lw=1.0, alpha=.85, label=f'RV(20) (MAE={rv_mae:.4f})')
    ax.plot(x, ew, color='#27ae60', lw=1.0, alpha=.85, label=f'EWMA(0.94) (MAE={ew_mae:.4f})')
    ax.set_xlabel('交易日'); ax.set_ylabel('年化波动率')
    ax.set_title('粒子滤波 vs 实现波动率 vs EWMA（同为年化口径）')
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(outdir/'pf_2_baselines.png', dpi=DPI); plt.close()

    # 图 3：误差分布
    fig, ax = plt.subplots(figsize=(10, 5))
    for est, c, lab in [(pf, '#c0392b', 'PF'), (rv, '#2980b9', 'RV(20)'), (ew, '#27ae60', 'EWMA')]:
        e = est[W_RV:HI]-tv[W_RV:HI]
        ax.hist(e, bins=45, alpha=.45, color=c, density=True,
                label=f'{lab}  MAE={np.mean(np.abs(e)):.4f}')
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.set_xlabel('估计误差（年化波动率）'); ax.set_ylabel('密度')
    ax.set_title('估计误差分布：PF 更集中，RV/EWMA 更分散')
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(outdir/'pf_3_error_dist.png', dpi=DPI); plt.close()

    # 图 4：真实三时点粒子快照（互不复用）
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    for ax, t in zip(axes, [50, 200, 400]):
        vp, ww = snap[t]
        ax.hist(np.sqrt(vp), bins=40, weights=ww, color='#c0392b', alpha=.75)
        ax.axvline(tv[t], color='k', ls='--', lw=2, label=f'真值 {tv[t]:.3f}')
        eff = 1.0/np.dot(ww, ww)
        ax.set_title(f't={t}  ESS={eff:.0f}/{N_PF}  v̂½={np.sqrt(np.dot(vp,ww)):.3f}')
        ax.set_xlabel('粒子波动率'); ax.legend(fontsize=8)
    axes[0].set_ylabel('后验权重')
    plt.suptitle('三个时点的后验粒子分布（各自独立快照）')
    plt.tight_layout(); plt.savefig(outdir/'pf_4_particles.png', dpi=DPI); plt.close()

    # ---------- 辅助实验 B：粒子数敏感性 ----------
    Ns = [25, 50, 100, 200, 500, 1000]
    sens_mae = []
    for n in Ns:
        # 共同随机数：所有 N 用同一随机流，差异只来自粒子数；N=500 与主实验精确一致
        e, *_ = particle_filter(px, N=n,
                                rng=np.random.default_rng(SEED_PF+1), **PF_PARAMS)
        sens_mae.append(mae(e, tv, W_RV, HI))

    # ---------- 辅助实验 C：参数错配（同一条价格路径，只把模型参数设错） ----------
    P_MIS = dict(kappa=0.5, theta=0.04, sigma_v=0.1, v0=0.04, mu=0.05, dt=1/252)
    pf_mis, *_ = particle_filter(px, N=N_PF,
                                 rng=np.random.default_rng(SEED_PF_AUX+501), **P_MIS)
    mis_pf_mae, mis_rv_mae = mae(pf_mis, tv, W_RV, HI), rv_mae

    # 图 5：左 N 敏感性，右 ESS 演化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(Ns, sens_mae, 'o-', color='#c0392b', lw=2)
    axes[0].axhline(rv_mae, color='#2980b9', ls='--', label=f'RV(20) 基线 {rv_mae:.4f}')
    axes[0].set_xscale('log'); axes[0].set_xlabel('粒子数 N'); axes[0].set_ylabel('MAE')
    axes[0].set_title('粒子数敏感性'); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
    axes[1].plot(x, ess, color='#8e44ad', lw=1.0)
    axes[1].axhline(N_PF/2, color='k', ls='--', label='ESS 阈值 N/2')
    axes[1].set_xlabel('交易日'); axes[1].set_ylabel('ESS')
    axes[1].set_title(f'有效样本数演化（共重采样 {nres} 次）'); axes[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(outdir/'pf_5_sensitivity.png', dpi=DPI); plt.close()

    results = {
        'seed': SEED_PF, 'T': T_PF, 'N': N_PF, 'window_rv': W_RV,
        'params': PF_PARAMS, 'rho_main': 0.0,
        'pf_mae': round(pf_mae, 6), 'rv_mae': round(rv_mae, 6), 'ewma_mae': round(ew_mae, 6),
        'pf_corr': round(pf_corr, 4), 'rv_corr': round(rv_corr, 4), 'ewma_corr': round(ew_corr, 4),
        'pf_pred_mae': round(pf_pred_mae, 6), 'rv_pred_mae': round(rv_pred_mae, 6),
        'resample_count': int(nres), 'ess_mean': round(float(ess.mean()), 2),
        'true_vol_mean': round(float(tv.mean()), 6), 'pf_vol_mean': round(float(pf.mean()), 6),
        'N_sensitivity': {str(n): round(v, 6) for n, v in zip(Ns, sens_mae)},
        'misspec_params': P_MIS,
        'misspec_pf_mae': round(mis_pf_mae, 6), 'misspec_rv_mae': round(mis_rv_mae, 6),
        'mae_improve_pct': round(100*(rv_mae-pf_mae)/rv_mae, 2),
        'mae_improve_pct_misspec': round(100*(mis_rv_mae-mis_pf_mae)/mis_rv_mae, 2),
        'snapshots': {str(t): {'true_vol': round(float(tv[t]), 4),
                               'pf_vol': round(float(np.sqrt(np.dot(snap[t][0], snap[t][1]))), 4),
                               'ess': round(float(1.0/np.dot(snap[t][1], snap[t][1])), 1)}
                      for t in [50, 200, 400]},
    }
    assert all(np.isfinite([pf_mae, rv_mae, ew_mae, pf_pred_mae, rv_pred_mae,
                            pf_corr, rv_corr, ew_corr, mis_pf_mae, mis_rv_mae]))
    assert np.nanmin(band_lo) > 0 and np.nanmax(band_hi) < 1
    assert abs(sens_mae[Ns.index(500)] - pf_mae) < 1e-9, 'N=500 敏感性点必须与主实验一致'
    assert len({tuple(snap[t][0][:5]) for t in [50, 200, 400]}) == 3, '三个快照不得相同'
    save_json(slug, results)
    return results


# =====================================================================
# 文章 2：Copula 尾部依赖
# =====================================================================

SIGMA_TRUE = np.array([
    [1.0, 0.6, -0.3, 0.1],
    [0.6, 1.0, -0.2, 0.15],
    [-0.3, -0.2, 1.0, -0.1],
    [0.1, 0.15, -0.1, 1.0]])
NU_TRUE, T_CP, D = 3, 1500, 4
ASSETS = ['科技股', '消费股', '债券', '黄金']
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def sim_t_copula(nu, Sigma, n, rng):
    """多元 t -> 逐维 t_ν CDF -> t-copula 样本（边际均匀）。"""
    L = np.linalg.cholesky(Sigma)
    z = rng.standard_normal((n, Sigma.shape[0])) @ L.T
    w = rng.chisquare(nu, n)/nu
    return stats.t.cdf(z/np.sqrt(w)[:, None], df=nu)


def sim_gauss_copula(Sigma, n, rng):
    L = np.linalg.cholesky(Sigma)
    return stats.norm.cdf(rng.standard_normal((n, Sigma.shape[0])) @ L.T)


def pseudo_obs(returns):
    """经验 CDF 转伪观测（IFM 第一步，不偷看真实 U）。"""
    T = len(returns)
    return np.apply_along_axis(lambda c: stats.rankdata(c)/(T+1), 0, returns)


def pack(C):
    return C[np.triu_indices(C.shape[0], 1)]


def corr_from_p(p, d):
    C = np.eye(d); iu = np.triu_indices(d, 1)
    C[iu] = p; C[(iu[1], iu[0])] = p
    return C


def neg_ll_gauss(p, U):
    C = corr_from_p(p, U.shape[1])
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        return 1e8
    z = stats.norm.ppf(np.clip(U, 1e-8, 1-1e-8))
    Zi = np.linalg.solve(L, z.T).T
    logdet = 2*np.log(np.diag(L)).sum()
    return float(0.5*len(U)*logdet
                 + 0.5*np.sum(np.einsum('ti,ti->t', Zi, Zi) - np.einsum('ti,ti->t', z, z)))


def neg_ll_t(p, U, nu):
    """t-copula 负对数似然：log f_{ν,Σ}(x) − Σ log f_ν(x_i)，严格标量。"""
    d = U.shape[1]
    C = corr_from_p(p, d)
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        return 1e8
    x = stats.t.ppf(np.clip(U, 1e-8, 1-1e-8), df=nu)
    Zi = np.linalg.solve(L, x.T).T
    quad = np.einsum('ti,ti->t', Zi, Zi)
    logdet = 2*np.log(np.diag(L)).sum()
    log_f = (gammaln((nu+d)/2) - gammaln(nu/2) - (d/2)*np.log(nu*np.pi)
             - .5*logdet - ((nu+d)/2)*np.log1p(quad/nu))
    log_marg = stats.t.logpdf(x, df=nu).sum(axis=1)
    return float(-np.sum(log_f - log_marg))


def fit_copula(U, kind, nu=3):
    p0 = np.clip(pack(np.corrcoef(U.T)), -0.9, 0.9)
    bounds = [(-0.95, 0.95)]*len(p0)
    fn = (lambda p: neg_ll_gauss(p, U)) if kind == 'gauss' else (lambda p: neg_ll_t(p, U, nu))
    res = minimize(fn, p0, bounds=bounds, method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-10})
    return corr_from_p(res.x, U.shape[1]), float(res.fun), bool(res.success)


def tail_dep_theory(nu, rho):
    """Student-t copula 上/下尾依赖系数（Demarta & McNeil, 2005）。"""
    return float(2*stats.t.cdf(-np.sqrt((nu+1)*(1-rho)/(1+rho)), df=nu+1))


def tail_dep_empirical(U, q, pair=(0, 1)):
    """经验下尾条件概率 P(U_j<q | U_i<q)。默认只取科技-消费对，与理论 λ 口径一致。"""
    i, j = pair
    a = U[:, i] < q
    return float(np.mean(U[a, j] < q)) if a.sum() > 0 else float('nan')


def tail_dep_allpairs(U, q):
    """6 个资产对的平均下尾条件概率（单独指标，不与理论 λ 混用）。"""
    return float(np.nanmean([tail_dep_empirical(U, q, p) for p in PAIRS]))


def gen_copula():
    slug = 'copula-dependence-structure'
    outdir = ROOT/'public'/'images'/slug
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- 主实验：t-copula(ν=3) + 正态边际，与文章代码块同源 ----------
    rng = np.random.default_rng(SEED_CP)
    U_true = sim_t_copula(NU_TRUE, SIGMA_TRUE, T_CP, rng)
    R = stats.norm.ppf(np.clip(U_true, 1e-10, 1-1e-10))
    U = pseudo_obs(R)
    pearson = np.corrcoef(R.T)
    Cg, llg, okg = fit_copula(U, 'gauss')
    Ct, llt, okt = fit_copula(U, 't', NU_TRUE)

    # profile likelihood 估 ν
    nu_grid = [2.5, 3, 4, 5, 7, 10, 20, 50]
    profile = [-fit_copula(U, 't', nu)[1] for nu in nu_grid]
    nu_hat = nu_grid[int(np.argmax(profile))]

    rho_hat = float(Ct[0, 1])
    lam_theory = tail_dep_theory(NU_TRUE, rho_hat)
    # 经验验证：用拟合出的 Σ 做大样本模拟
    rng_a = np.random.default_rng(SEED_CP_AUX)
    n_sim = 400_000
    Ut_sim = sim_t_copula(NU_TRUE, Ct, n_sim, rng_a)
    Ug_sim = sim_gauss_copula(Cg, n_sim, rng_a)
    qs = [0.01, 0.05, 0.10]
    lam_emp_t = {str(q): tail_dep_empirical(Ut_sim, q) for q in qs}
    lam_emp_g = {str(q): tail_dep_empirical(Ug_sim, q) for q in qs}
    lam_allpairs_t = {str(q): tail_dep_allpairs(Ut_sim, q) for q in qs}

    # 图 1：三种相关矩阵
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, title, C in zip(axes, ['Pearson（收益率）', 'Gaussian Copula MLE', 'Student-t Copula MLE'],
                            [pearson, Cg, Ct]):
        im = ax.imshow(C, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(D)); ax.set_yticks(range(D))
        ax.set_xticklabels(ASSETS, fontsize=8); ax.set_yticklabels(ASSETS, fontsize=8)
        ax.set_title(title, fontsize=10)
        for i in range(D):
            for j in range(D):
                ax.text(j, i, f'{C[i,j]:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(C[i, j]) > .55 else 'black')
        plt.colorbar(im, ax=ax, shrink=.8)
    plt.suptitle(f'真值 ν={NU_TRUE}；profile likelihood 得 ν̂={nu_hat}')
    plt.tight_layout(); plt.savefig(outdir/'copula_1_heatmap.png', dpi=DPI); plt.close()

    # 图 2：散点 + 尾部（阈值用正态边际分位数，口径一致）
    q5, q95 = stats.norm.ppf(.05), stats.norm.ppf(.95)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (i, j) in zip(axes, [(0, 1), (0, 2)]):
        lo = (R[:, i] < q5) & (R[:, j] < q5)
        hi = (R[:, i] > q95) & (R[:, j] > q95)
        ax.scatter(R[:, i], R[:, j], s=7, alpha=.35, c='#95a5a6', label='正常')
        ax.scatter(R[lo, i], R[lo, j], s=22, c='#27ae60', label='双下尾')
        ax.scatter(R[hi, i], R[hi, j], s=22, c='#c0392b', label='双上尾')
        ax.set_xlabel(ASSETS[i]); ax.set_ylabel(ASSETS[j])
        ax.set_title(f'{ASSETS[i]} vs {ASSETS[j]}：ρ̂={Ct[i,j]:.2f}')
        ax.legend(fontsize=8)
    plt.suptitle('5% / 95% 分位下的联合尾部（正态边际口径）')
    plt.tight_layout(); plt.savefig(outdir/'copula_2_scatter.png', dpi=DPI); plt.close()

    # 图 3：真实大样本模拟的联合尾部概率（两条曲线各自独立模拟）
    Rt = stats.norm.ppf(np.clip(Ut_sim, 1e-10, 1-1e-10))
    Rg = stats.norm.ppf(np.clip(Ug_sim, 1e-10, 1-1e-10))
    alphas = np.arange(1, 11)/100
    p_t, p_g = [], []
    for a in alphas:
        thr = stats.norm.ppf(a)
        p_t.append(np.mean([np.mean((Rt[:, i] < thr) & (Rt[:, j] < thr)) for i, j in PAIRS]))
        p_g.append(np.mean([np.mean((Rg[:, i] < thr) & (Rg[:, j] < thr)) for i, j in PAIRS]))
    p_t, p_g = np.array(p_t), np.array(p_g)
    ratio = p_t/np.maximum(p_g, 1e-12)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(alphas*100, p_g, 'o-', color='#2980b9', lw=2, ms=4, label='Gaussian Copula（40 万次模拟）')
    ax.plot(alphas*100, p_t, 'o-', color='#c0392b', lw=2, ms=4, label='Student-t Copula（40 万次模拟）')
    ax.fill_between(alphas*100, p_g, p_t, color='#8e44ad', alpha=.2)
    ax.set_xlabel('下尾分位 (%)'); ax.set_ylabel('两两联合下尾概率（平均）')
    ax.set_title(f'联合尾部概率：1% 分位下 t / Gaussian = {ratio[0]:.2f}×')
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(outdir/'copula_3_tail_prob.png', dpi=DPI); plt.close()

    # 图 4：正确 λ(ν) 曲线 + 蒙特卡洛经验点
    nu_curve = np.linspace(2.1, 30, 120)
    lam_curve = [tail_dep_theory(nu, rho_hat) for nu in nu_curve]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(nu_curve, lam_curve, color='#c0392b', lw=2, label=f'理论 λ(ν, ρ̂={rho_hat:.3f})')
    nu_pts = [2.5, 3, 5, 7, 10, 20]
    for q, mk in zip(qs, ['o', 's', '^']):
        vals = []
        for nu in nu_pts:
            Uu = sim_t_copula(nu, Ct, 120_000,
                              np.random.default_rng(SEED_CP_AUX+int(q*1000)))
            vals.append(tail_dep_empirical(Uu, q))
        ax.scatter(nu_pts, vals, marker=mk, s=35, alpha=.8,
                   label=f'经验 P(U2<q|U1<q), q={q}')
    ax.axvline(nu_hat, color='k', ls='--', lw=1.5, label=f'ν̂={nu_hat}')
    ax.set_xlabel('自由度 ν'); ax.set_ylabel('尾部依赖系数 λ'); ax.set_ylim(0, 1)
    ax.set_title(f'尾部依赖随 ν 单调下降；ν=3 理论 λ={lam_theory:.4f}')
    ax.text(0.46, 0.86,
            '经验点是有限分位 q 的条件概率，\nλ 是 q→0 的极限。\n固定 q 时收敛极慢，故大 ν 处\n经验值系统性高于极限——\n用历史频率直接估 λ 会高估。',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', fc='#fff9e6', ec='#d4a017', alpha=.9))
    ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(outdir/'copula_4_lambda_nu.png', dpi=DPI); plt.close()

    # 图 5：左 组合损失左尾，右 ν 的 profile likelihood
    port_t, port_g = Rt.mean(axis=1), Rg.mean(axis=1)
    var_t, var_g = np.percentile(port_t, 1), np.percentile(port_g, 1)
    es_t = port_t[port_t <= var_t].mean(); es_g = port_g[port_g <= var_g].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(port_g, bins=120, range=(-3, 0), density=True, alpha=.55,
                 color='#2980b9', label='Gaussian')
    axes[0].hist(port_t, bins=120, range=(-3, 0), density=True, alpha=.55,
                 color='#c0392b', label='Student-t')
    axes[0].axvline(var_g, color='#2980b9', ls='--', lw=1.5, label=f'VaR 1% G={var_g:.3f}')
    axes[0].axvline(var_t, color='#c0392b', ls='--', lw=1.5, label=f'VaR 1% t={var_t:.3f}')
    axes[0].set_xlabel('等权组合收益（标准化）'); axes[0].set_ylabel('密度')
    axes[0].set_title('组合损失左尾（真实模拟）'); axes[0].legend(fontsize=7)
    axes[1].plot(nu_grid, profile, 'o-', color='#8e44ad', lw=2)
    axes[1].axvline(nu_hat, color='k', ls='--', label=f'ν̂={nu_hat}')
    axes[1].set_xlabel('自由度 ν'); axes[1].set_ylabel('对数似然')
    axes[1].set_title('ν 的 profile likelihood'); axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(outdir/'copula_5_var_profile.png', dpi=DPI); plt.close()

    results = {
        'seed': SEED_CP, 'T': T_CP, 'nu_true': NU_TRUE, 'nu_hat': int(nu_hat),
        'n_sim': n_sim, 'n_pairs': len(PAIRS),
        'rho_hat_tech_consumer': round(rho_hat, 4),
        'lambda_theory': round(lam_theory, 6),
        'lambda_emp_t': {k: round(v, 6) for k, v in lam_emp_t.items()},
        'lambda_emp_gauss': {k: round(v, 6) for k, v in lam_emp_g.items()},
        'lambda_emp_allpairs_t': {k: round(v, 6) for k, v in lam_allpairs_t.items()},
        'pearson': np.round(pearson, 4).tolist(),
        'gauss_mle': np.round(Cg, 4).tolist(),
        't_mle': np.round(Ct, 4).tolist(),
        'sigma_true': SIGMA_TRUE.tolist(),
        'fit_success': {'gauss': okg, 't': okt},
        'loglik': {'gauss': round(-llg, 2), 't': round(-llt, 2)},
        'profile_nu': {str(n): round(v, 2) for n, v in zip(nu_grid, profile)},
        'joint_tail_1pct': {'gauss': round(float(p_g[0]), 8),
                            't': round(float(p_t[0]), 8),
                            'ratio': round(float(ratio[0]), 4)},
        'joint_tail_5pct': {'gauss': round(float(p_g[4]), 8),
                            't': round(float(p_t[4]), 8),
                            'ratio': round(float(ratio[4]), 4)},
        'portfolio_var1': {'gauss': round(float(var_g), 4), 't': round(float(var_t), 4),
                           'loss_amplify_pct': round(float((abs(var_t)-abs(var_g))/abs(var_g)*100), 2)},
        'portfolio_es1': {'gauss': round(float(es_g), 4), 't': round(float(es_t), 4)},
    }
    assert 0 < lam_theory < 1, '尾部依赖系数必须在 (0,1)'
    assert okt and okg, 'MLE 必须真正收敛，不允许回退真值'
    assert np.max(np.abs(Ct-SIGMA_TRUE)) > 1e-3, '拟合结果不得恰好等于真值矩阵'
    assert all(np.isfinite([lam_theory, ratio[0], var_t, var_g, es_t, es_g]))
    assert np.all(np.isfinite(p_t)) and np.all(np.isfinite(p_g)) and np.all(np.isfinite(ratio))
    save_json(slug, results)
    return results


def gen_covers():
    for slug, title, sub, tag in [
        ('particle-filter-volatility', '粒子滤波波动率估计', '从噪声价格中提取隐藏波动率', 'Heston × 序贯蒙特卡洛'),
        ('copula-dependence-structure', 'Copula 尾部依赖建模', '用 Copula 捕捉资产间非线性依赖',
         'Pearson / Gaussian / Student-t')]:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis('off')
        ax.add_patch(plt.Rectangle((0, 0), 12, 6, facecolor='#f8f9fa', edgecolor='none'))
        ax.text(6, 4.5, title, fontsize=34, ha='center', va='center', color='#1a1a2e')
        ax.text(6, 3.5, sub, fontsize=19, ha='center', va='center', color='#4a5568')
        ax.text(6, 2.5, tag, fontsize=15, ha='center', va='center', color='#6b7278')
        ax.text(6, 1.5, '2026.11.07 | 量化专栏', fontsize=12, ha='center', va='center', color='#8d949a')
        ax.plot([3, 9], [2.0, 2.0], color='#e74c3c', lw=3)
        plt.tight_layout()
        plt.savefig(ROOT/'public'/'images'/slug/'cover.png', dpi=DPI,
                    bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()


if __name__ == '__main__':
    print('生成粒子滤波...')
    r1 = gen_particle_filter()
    print(json.dumps({k: v for k, v in r1.items()
                      if k not in ('params', 'snapshots', 'N_sensitivity')},
                     ensure_ascii=False, indent=2))
    print('\n生成 Copula...')
    r2 = gen_copula()
    print(json.dumps({k: v for k, v in r2.items()
                      if k not in ('pearson', 'gauss_mle', 't_mle', 'sigma_true', 'profile_nu')},
                     ensure_ascii=False, indent=2))
    gen_covers()
    print('\n完成')
