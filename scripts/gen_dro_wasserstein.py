# -*- coding: utf-8 -*-
"""
分布鲁棒优化 (DRO)：用 Wasserstein 球 (W2) 定义最坏情况分布族
手写全部计算，无 POT / cvxpy 依赖。seed = 20261029。

真分布：带肥左尾的日收益混合分布
  R ~ 0.8 * N(mu1, s1^2) + 0.2 * N(mu2, s2^2)   (mu2<0 偶发回撤)
用 n=400 个历史样本估计均值 mu_hat，Wasserstein-2 球 B_eps(P_n) 内
最坏情况期望收益 = mu_hat - eps（对手把整族分布向下平移 eps）。
据此做均值-方差下的最优仓位 w*，并做样本外蒙特卡洛验证。

输出：public/images/distributionally-robust-optimization/*.png + _dro_results.json
"""
import numpy as np
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Heiti TC', 'PingFang SC', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

SEED = 20261029
rng = np.random.default_rng(SEED)

SLUG = 'distributionally-robust-optimization'
OUTDIR = f'public/images/{SLUG}'
os.makedirs(OUTDIR, exist_ok=True)

# ---------- 真分布参数（带肥左尾）----------
W1, MU1, S1 = 0.80, 0.0012, 0.012
W2, MU2, S2 = 0.20, -0.0040, 0.035
MU0 = W1 * MU1 + W2 * MU2                       # 真均值
VAR0 = (W1 * (S1**2 + MU1**2) + W2 * (S2**2 + MU2**2)) - MU0**2   # 真方差
SIG0 = np.sqrt(VAR0)

N = 400                                          # 样本量
GAMMA = 2.0                                      # 风险厌恶（均值-方差）

def draw_sample(n, seed=None):
    r = np.random.default_rng(seed) if seed is not None else rng
    comp = r.random(n) < W1
    s = np.where(comp, r.normal(MU1, S1, n), r.normal(MU2, S2, n))
    return s

# 固定一次"主样本"用于单点可视化与决策
SAMPLE = draw_sample(N, seed=SEED)
MU_HAT = SAMPLE.mean()
VAR_HAT = SAMPLE.var(ddof=1)
SIG_HAT = np.sqrt(VAR_HAT)
STD_MU = SIG0 / np.sqrt(N)                        # mu_hat 的估计误差标准差

# 真最优仓位 / 样本最优仓位 / DRO 仓位  (w* = mu / (gamma * var))
def opt_pos(mu, var):
    return mu / (GAMMA * var)

W_TRUE = opt_pos(MU0, VAR0)
W_SAA = opt_pos(MU_HAT, VAR_HAT)
# DRO 最坏情况期望收益 = mu_hat - eps
EPS_SWEEP = np.linspace(0.0, 0.004, 41)
W_DRO = np.array([opt_pos(MU_HAT - e, VAR_HAT) for e in EPS_SWEEP])

# ---------- W2 精确计算：平移分布的 W2 ----------
def w2_shift(empirical, shift):
    """Q = 每个原子平移 shift，W2^2 = mean((shift)^2) = shift^2"""
    return abs(shift)

# ---------- 实验 1：模糊集可视化（W2 球内的极端成员）----------
EPS_VIZ = STD_MU                               # 用估计误差尺度做可视化
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(-0.12, 0.06, 50)
ax.hist(SAMPLE, bins=bins, density=True, alpha=0.45, color='#4C78A8',
        label=f'经验分布 P_n (n={N})')
ax.hist(SAMPLE + EPS_VIZ, bins=bins, density=True, histtype='step', linewidth=2,
        color='#E45756', label=f'最坏情况 +eps (均值上移, W2=eps)')
ax.hist(SAMPLE - EPS_VIZ, bins=bins, density=True, histtype='step', linewidth=2,
        color='#54A24B', label=f'最坏情况 -eps (均值下移, W2=eps)')
ax.axvline(MU_HAT, color='#4C78A8', ls='--', lw=1.5, label=f'mu_hat={MU_HAT:.5f}')
ax.axvline(MU0, color='black', ls=':', lw=1.5, label=f'mu0(真)={MU0:.5f}')
ax.set_xlabel('日收益 R')
ax.set_ylabel('概率密度')
ax.set_title('图1  Wasserstein-2 模糊集：经验分布周围半径 eps 内的整族分布')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig1_ambiguity_set.png', dpi=130)
plt.close(fig)

# ---------- 实验 2：最坏情况期望收益带 vs eps ----------
fig, ax = plt.subplots(figsize=(8, 5))
eps2 = np.linspace(0, 0.004, 100)
wc_max = MU_HAT + eps2
wc_min = MU_HAT - eps2
ax.fill_between(eps2, wc_min, wc_max, color='#F2A93B', alpha=0.25,
                label='最坏情况期望收益带 [mu_hat-eps, mu_hat+eps]')
ax.plot(eps2, wc_max, color='#E45756', lw=1.5)
ax.plot(eps2, wc_min, color='#54A24B', lw=1.5)
ax.axhline(MU0, color='black', ls=':', lw=1.5, label=f'mu0(真)={MU0:.5f}')
ax.axvline(abs(MU_HAT - MU0), color='#888', ls='--', lw=1.2,
           label=f'覆盖真值所需 eps = |mu_hat-mu0| = {abs(MU_HAT-MU0):.5f}')
ax.set_xlabel('模糊半径 eps (Wasserstein-2)')
ax.set_ylabel('期望日收益 E[R]')
ax.set_title('图2  最坏情况期望收益的 eps 带：eps 够大即覆盖真值（有限样本保证）')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig2_worstcase_band.png', dpi=130)
plt.close(fig)

# ---------- 实验 3：DRO 对最优仓位的收缩 ----------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(EPS_SWEEP, W_SAA * np.ones_like(EPS_SWEEP), color='#4C78A8', lw=2,
        label=f'SAA 仓位 w*={W_SAA:.4f}（不随 eps 变）')
ax.plot(EPS_SWEEP, W_DRO, color='#E45756', lw=2,
        label='DRO 最坏情况仓位 w*(eps)= (mu_hat-eps)/(gamma*sig_hat^2)')
ax.axhline(W_TRUE, color='black', ls=':', lw=1.5, label=f'真最优仓位 w*_true={W_TRUE:.4f}')
ax.axvline(STD_MU, color='#888', ls='--', lw=1.2,
           label=f'eps*=估计误差 std(mu_hat)={STD_MU:.5f}')
ax.set_xlabel('模糊半径 eps')
ax.set_ylabel('最优仓位 w* (占资比例)')
ax.set_title('图3  Wasserstein 模糊让最优仓位向 0 收缩（鲁棒性 = 谨慎）')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig3_dro_shrinkage.png', dpi=130)
plt.close(fig)

# ---------- 实验 4 & 5：样本外蒙特卡洛（聚焦"过度下注"与杠杆封顶）----------
MC = 600
w_saa_mc = np.zeros(MC)
w_dro_mc = np.zeros((len(EPS_SWEEP), MC))
mu_hat_mc = np.zeros(MC)
for k in range(MC):
    s = draw_sample(N, seed=SEED + 1000 + k)
    mh = s.mean(); vh = s.var(ddof=1)
    mu_hat_mc[k] = mh
    w_saa_mc[k] = opt_pos(mh, vh)
    for j, e in enumerate(EPS_SWEEP):
        w_dro_mc[j, k] = opt_pos(mh - e, vh)

j_star = int(np.argmin(np.abs(EPS_SWEEP - STD_MU)))
max_w_saa = w_saa_mc.max()
max_w_dro_star = w_dro_mc[j_star].max()
# 最坏情况（下界）覆盖真值的比例： mu0 >= mu_hat - eps  ?
frac_cover = np.mean(MU0 >= (mu_hat_mc - STD_MU))
# DRO 杠杆严格 <= SAA 杠杆的比例（构造上恒为 1）
frac_le_cap = np.mean(w_dro_mc[j_star] <= w_saa_mc + 1e-12)

# 图4：SAA vs DRO(eps*) 的"最优杠杆"经验 CDF（DRO 杠杆一阶随机占优于 SAA）
fig, ax = plt.subplots(figsize=(8, 5))
xs_cdf = np.linspace(-4, 7, 400)
for arr, c, lab in [(w_saa_mc, '#4C78A8', 'SAA 杠杆'),
                    (w_dro_mc[j_star], '#E45756', 'DRO(eps*) 杠杆')]:
    cdf = np.array([np.mean(arr <= x) for x in xs_cdf])
    ax.plot(xs_cdf, cdf, color=c, lw=2, label=lab)
ax.axvline(W_TRUE, color='black', ls=':', lw=1.5, label=f'真最优杠杆 w*_true={W_TRUE:.3f}')
ax.set_xlabel('最优杠杆 w* (占资比例，>1 为加杠杆)')
ax.set_ylabel('P(w* <= x)  经验 CDF')
ax.set_title('图4  杠杆经验 CDF：DRO 曲线整体在左，最坏下注被系统性压低')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig4_leverage_cdf.png', dpi=130)
plt.close(fig)

# 图5：最坏情况(最大)杠杆 vs eps —— 模糊半径直接控制你的"最坏下注"
max_w_dro = w_dro_mc.max(axis=1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(EPS_SWEEP, max_w_saa * np.ones_like(EPS_SWEEP), color='#4C78A8', lw=2,
        label=f'SAA 最大杠杆(固定)={max_w_saa:.3f}')
ax.plot(EPS_SWEEP, max_w_dro, color='#E45756', lw=2,
        label='DRO 最大杠杆 = max_k (mu_hat_k - eps)/(gamma*sig_hat^2)')
ax.axhline(W_TRUE, color='black', ls=':', lw=1.5, label=f'真最优杠杆={W_TRUE:.3f}')
ax.axvline(STD_MU, color='#888', ls='--', lw=1.2,
           label=f'校准半径 eps*=std(mu_hat)={STD_MU:.5f}')
ax.set_xlabel('模糊半径 eps')
ax.set_ylabel('最坏情况(最大)杠杆')
ax.set_title('图5  模糊半径越大，最坏情况杠杆上限越低（鲁棒性 = 给下注上锁）')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig5_max_leverage.png', dpi=130)
plt.close(fig)

# ---------- 汇总 ----------
results = {
    'seed': SEED,
    'true': {'mu0': float(MU0), 'var0': float(VAR0), 'sig0': float(SIG0),
             'w_true': float(W_TRUE)},
    'sample': {'n': N, 'mu_hat': float(MU_HAT), 'var_hat': float(VAR_HAT),
               'w_saa': float(W_SAA), 'std_mu': float(STD_MU)},
    'worstcase_mean': {'wc_max_at_eps1e-3': float(MU_HAT + 1e-3),
                       'wc_min_at_eps1e-3': float(MU_HAT - 1e-3),
                       'coverage_eps': float(abs(MU_HAT - MU0))},
    'dro_shrinkage': {'w_dro_at_eps_star': float(opt_pos(MU_HAT - STD_MU, VAR_HAT))},
    'oos': {'mc': MC, 'max_w_saa': float(max_w_saa), 'max_w_dro_star': float(max_w_dro_star),
            'p90_w_saa': float(np.percentile(w_saa_mc, 90)),
            'p90_w_dro_star': float(np.percentile(w_dro_mc[j_star], 90)),
            'frac_cover_mu0_ge_muhat_minus_eps': float(frac_cover),
            'frac_dro_le_saa': float(frac_le_cap), 'eps_star': float(STD_MU)},
    'w2_checks': {'w2_shift_plus_eps': float(w2_shift(SAMPLE, EPS_VIZ)),
                  'eps_viz': float(EPS_VIZ)},
}
with open('_dro_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('===== DRO RESULTS =====')
print(f'true mu0={MU0:.6f}  var0={VAR0:.6f}  sig0={SIG0:.6f}  w_true={W_TRUE:.5f}')
print(f'sample n={N} mu_hat={MU_HAT:.6f} var_hat={VAR_HAT:.6f} w_saa={W_SAA:.5f} std(mu_hat)={STD_MU:.6f}')
print(f'worst-case mean band at eps=1e-3: [{MU_HAT-1e-3:.6f}, {MU_HAT+1e-3:.6f}]')
print(f'coverage eps (|mu_hat-mu0|)={abs(MU_HAT-MU0):.6f}')
print(f'W2 of +eps shifted dist = {w2_shift(SAMPLE, EPS_VIZ):.6f}  (== eps_viz {EPS_VIZ:.6f})')
print(f'DRO w* at eps*=std(mu_hat) = {opt_pos(MU_HAT-STD_MU, VAR_HAT):.5f}  (SAA={W_SAA:.5f}, true={W_TRUE:.5f})')
print(f'OOS MC={MC}: max leverage SAA={max_w_saa:.4f}  DRO(eps*)={max_w_dro_star:.4f}  (封顶 {(1-max_w_dro_star/max_w_saa)*100:.1f}%)')
print(f'  p90 leverage SAA={np.percentile(w_saa_mc,90):.4f}  DRO(eps*)={np.percentile(w_dro_mc[j_star],90):.4f}')
print(f'  最坏情况下界覆盖真值比例 = {frac_cover:.3f}  (eps=1 SE 理论 0.841)')
print(f'  DRO 杠杆<=SAA 比例 = {frac_le_cap:.3f} (构造恒成立)')
print('figures:', os.listdir(OUTDIR))
