"""随机贴现因子与 Hansen-Jagannathan 界：数值实验（5 张真实计算图）"""
import os, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = 'public/images/sdf-hansen-jagannathan'
os.makedirs(OUT, exist_ok=True)
res = {}

N, T = 12, 600
LAM = np.array([0.0060, 0.0050, 0.0040])          # 月度因子风险溢价：市场/规模/价值
SF = np.array([[0.045**2, 0.00040, 0.00020],
               [0.00040, 0.030**2, 0.00030],
               [0.00020, 0.00030, 0.025**2]])
CF = np.linalg.cholesky(SF)

def make_B(r, N):
    return np.column_stack([r.uniform(0.7, 1.3, N),
                            r.uniform(-1.2, 1.2, N),
                            r.uniform(-1.0, 1.0, N)])

def simulate(T=T, N=N, seed=1, B=None, lam=None):
    """lam=None 用真实三因子溢价；lam=LAM0 时规模/价值溢价归零（CAPM 世界）"""
    lam = LAM if lam is None else lam
    r = np.random.default_rng(seed)
    if B is None: B = make_B(r, N)
    idi = r.uniform(0.030, 0.060, N)
    f = lam + r.standard_normal((T, 3)) @ CF.T
    e = r.standard_normal((T, N)) * idi
    R = f @ B.T + e
    fsp = r.normal(0.002, 0.030, (T, 2))          # 两个与资产无关的伪因子
    return R, f, B, idi, fsp

def sdf_from_factors(R, Fm):
    Gff = Fm.T @ Fm / len(Fm)
    b = np.linalg.solve(Gff, Fm.mean(0))
    m = 1.0 - Fm @ b
    alpha = R.T @ m / len(R)
    G = R.T @ R / len(R)
    return m, alpha, float(np.sqrt(alpha @ np.linalg.solve(G, alpha))), \
        float(m.std(ddof=1) / m.mean())

def tan_sharpe(R):
    mu = R.mean(0)
    return float(np.sqrt(mu @ np.linalg.solve(np.cov(R.T), mu)))

MODELS = [('CAPM（市场）', [0]), ('市场+规模', [0, 1]), ('三因子 FF3（真）', [0, 1, 2]),
          ('四因子（+噪声1）', [0, 1, 2, 3]), ('五因子（+噪声1,2）', [0, 1, 2, 3, 4])]
names = [m[0] for m in MODELS]
B0 = make_B(np.random.default_rng(7), N)

# ============ 1. 总体（T=20 万）============
Rp, Fp, _, _, FSp = simulate(T=200000, seed=3, B=B0)
POP = {'S_tan': tan_sharpe(Rp)}
POOLp = np.column_stack([Fp, FSp])
for nm, cols in MODELS:
    _, _, d, ra = sdf_from_factors(Rp, POOLp[:, cols])
    POP[nm] = dict(delta=d, ratio=ra)
POP['mkt_sharpe'] = float(Fp[:, 0].mean() / Fp[:, 0].std(ddof=1))
print('=== 总体（T=200000）===')
print(f"  HJ 界 S_tan={POP['S_tan']:.4f}   市场夏普={POP['mkt_sharpe']:.4f}")
for nm in names:
    print(f'  {nm:16s} ratio={POP[nm]["ratio"]:.4f}  delta={POP[nm]["delta"]:.4f}')
res['pop'] = POP

# ============ 2. 主样本 T=600（现实长度）============
R, F, Bm, IDIO, FS = simulate(T=T, seed=11, B=B0)
POOL = np.column_stack([F, FS])
S_tan = tan_sharpe(R)
lam_v = np.linalg.solve(np.cov(R.T), R.mean(0))
m_star = 1.0 - R @ lam_v
a_star = R.T @ m_star / T
res['S_tan'] = S_tan; res['S_tan_ann'] = S_tan * np.sqrt(12)
res['ratio_star'] = float(m_star.std(ddof=1) / m_star.mean())
res['delta_star'] = float(np.sqrt(a_star @ np.linalg.solve(R.T @ R / T, a_star)))
res['mkt_sharpe'] = float(F[:, 0].mean() / F[:, 0].std(ddof=1))
res['mkt_sharpe_ann'] = res['mkt_sharpe'] * np.sqrt(12)
res['noise_floor'] = float(np.sqrt(N / T))

mrows = {}
print('\n=== 主样本 T=600 ===')
print(f'  HJ 界(样本)={S_tan:.4f}（总体 {POP["S_tan"]:.4f}，样本上偏 {S_tan/POP["S_tan"]-1:+.1%}）')
print(f'  无约束最小方差 SDF: ratio={res["ratio_star"]:.4f}  delta={res["delta_star"]:.4f}')
for nm, cols in MODELS:
    m, al_, d, ra = sdf_from_factors(R, POOL[:, cols])
    mrows[nm] = dict(delta=d, ratio=ra, alpha_absmax=float(np.abs(al_).max()),
                     m_mean=float(m.mean()), m_std=float(m.std(ddof=1)),
                     alpha_rms=float(np.sqrt((al_**2).mean())))
    print(f'  {nm:16s} ratio={ra:.4f}(总体 {POP[nm]["ratio"]:.4f})  '
          f'delta={d:.4f}(总体 {POP[nm]["delta"]:.4f})  '
          f'{"低于界→被拒" if ra < S_tan else "界以上"}')
res['models'] = mrows; res['S_tan_sample_bias'] = S_tan / POP['S_tan'] - 1

# ---------- 图 1：HJ 界锥形图 ----------
fig, ax = plt.subplots(figsize=(9.8, 6))
Em = np.linspace(0, 1.1, 100)
ax.fill_between(Em, 0, S_tan * Em, color='#c0392b', alpha=.12)
ax.plot(Em, S_tan * Em, color='#c0392b', lw=2.4, label=f'样本 HJ 界 = {S_tan:.4f}')
ax.plot(Em, POP['S_tan'] * Em, color='#c0392b', lw=1.3, ls=':',
        label=f'总体 HJ 界 = {POP["S_tan"]:.4f}')
ax.text(0.60, S_tan * 0.60 - 0.025, '界以下：不可能定价全部资产\n（模型被拒绝）',
        color='#c0392b', fontsize=9, ha='center')
cols = {'CAPM（市场）': '#e67e22', '市场+规模': '#f1c40f', '三因子 FF3（真）': '#27ae60',
        '四因子（+噪声1）': '#2980b9', '五因子（+噪声1,2）': '#8e44ad'}
for nm in names:
    d = mrows[nm]
    ax.plot(Em, d['ratio'] * Em, color=cols[nm], lw=1.6, ls='--' if '噪声' in nm else '-',
            label=f'{nm}：{d["ratio"]:.4f}')
    ax.plot([d['m_mean']], [d['m_std']], 'o', color=cols[nm], ms=9)
ax.set_xlabel('E[m]（SDF 均值）'); ax.set_ylabel('σ(m)（SDF 标准差，月度）')
ax.set_title('图1：Hansen-Jagannathan 界——候选 SDF 必须落在锥形区域内')
ax.set_xlim(0, 1.1); ax.set_ylim(0, max(d['ratio'] for d in mrows.values()) * 1.15)
ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{OUT}/hj-bound.png', dpi=130); plt.close()

# ============ 3. 长样本 T=4800：样本内/样本外拐点 ============
TL, seedL = 4800, 21
RL, FL, _, _, FSL = simulate(T=TL, seed=seedL, B=B0)
POOLL = np.column_stack([FL, FSL])
S_tan_L = tan_sharpe(RL)
rowsL, oosL, ratioL = {}, {}, {}
half = TL // 2
for nm, cols in MODELS:
    Fm = POOLL[:, cols]
    _, _, d, ra = sdf_from_factors(RL, Fm)
    rowsL[nm], ratioL[nm] = d, ra
    b = np.linalg.solve(Fm[:half].T @ Fm[:half] / half, Fm[:half].mean(0))
    m2 = 1.0 - Fm[half:] @ b; R2 = RL[half:]
    a2 = R2.T @ m2 / half
    oosL[nm] = float(np.sqrt(a2 @ np.linalg.solve(R2.T @ R2 / half, a2)))
res['long'] = dict(S_tan=S_tan_L, ins=rowsL, oos=oosL, ratio=ratioL,
                   floor=float(np.sqrt(N / (TL / 2))))
print(f'\n=== 长样本 T={TL} ===（HJ 界 {S_tan_L:.4f}，样本外噪声地板 √N/(T/2)={np.sqrt(N/(TL/2)):.4f}）')
for nm in names:
    print(f'  {nm:16s} 样本内 δ={rowsL[nm]:.4f}  样本外 δ={oosL[nm]:.4f}  '
          f'总体 {POP[nm]["delta"]:.4f}  ratio={ratioL[nm]:.4f}')

fig, ax = plt.subplots(figsize=(10.5, 4.6))
x = np.arange(len(names))
ax.bar(x - .27, [rowsL[n] for n in names], .26, color='#34495e', label='样本内 δ')
ax.bar(x, [oosL[n] for n in names], .26, color='#e74c3c', label='样本外 δ（前半段定 b）')
ax.plot(x, [POP[n]['delta'] for n in names], 'g^--', ms=8, label='总体 δ')
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel('HJ 距离 δ（月度）')
ax.set_title(f'图2：长样本 T={TL} —— 样本内 δ 单调下降，样本外 δ 在三因子处触底')
ax.legend(fontsize=8); ax.grid(alpha=.3, axis='y')
plt.tight_layout(); plt.savefig(f'{OUT}/delta-in-out.png', dpi=130); plt.close()

# ============ 4. T=600 下 δ 的零分布 vs 备选分布（功效不足）============
LAM0 = np.array([0.0060, 0.0, 0.0])                # CAPM 成立的世界
B_mc = 400
null_d, alt_d = np.empty(B_mc), np.empty(B_mc)
null_g, alt_g = np.empty(B_mc), np.empty(B_mc)
for rep in range(B_mc):
    R0, F0, _, _, FS0 = simulate(T=T, seed=6000 + rep, B=B0, lam=LAM0)
    _, _, d0, ra0 = sdf_from_factors(R0, np.column_stack([F0, FS0])[:, [0]])
    null_d[rep] = d0; null_g[rep] = tan_sharpe(R0) - ra0
    R1, F1, _, _, FS1 = simulate(T=T, seed=8000 + rep, B=B0)
    _, _, d1, ra1 = sdf_from_factors(R1, np.column_stack([F1, FS1])[:, [0]])
    alt_d[rep] = d1; alt_g[rep] = tan_sharpe(R1) - ra1
thr_d, thr_g = np.quantile(null_d, .95), np.quantile(null_g, .95)
pow_d = float((alt_d > thr_d).mean()); pow_g = float((alt_g > thr_g).mean())
obs_d = mrows['CAPM（市场）']['delta']; obs_g = S_tan - mrows['CAPM（市场）']['ratio']
print(f'\n=== T=600 的检验功效（CAPM 为原假设）===')
print(f'  δ 检验：零分布 95% 分位 {thr_d:.4f}，备选世界超出比例（功效）{pow_d:.3f}；'
      f'本样本观测 δ={obs_d:.4f}，p={(null_d >= obs_d).mean():.3f}')
print(f'  界违反检验：零分布 95% 分位 {thr_g:.4f}，功效 {pow_g:.3f}；'
      f'本样本观测 gap={obs_g:.4f}，p={(null_g >= obs_g).mean():.3f}')
res['power600'] = dict(thr_d=float(thr_d), pow_d=pow_d, obs_d=float(obs_d),
                       p_d=float((null_d >= obs_d).mean()),
                       thr_g=float(thr_g), pow_g=pow_g, obs_g=float(obs_g),
                       p_g=float((null_g >= obs_g).mean()),
                       null_d_mean=float(null_d.mean()), alt_d_mean=float(alt_d.mean()),
                       null_g_mean=float(null_g.mean()), alt_g_mean=float(alt_g.mean()))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
axes[0].hist(null_d, bins=35, color='#7f8c8d', alpha=.7, label='CAPM 成立世界下的 δ̂')
axes[0].hist(alt_d, bins=35, color='#e67e22', alpha=.7, label='三因子真实世界下的 δ̂')
axes[0].axvline(thr_d, color='k', ls='--', label=f'零分布 95% 分位 {thr_d:.3f}')
axes[0].axvline(obs_d, color='#c0392b', lw=2.2, label=f'本样本观测 {obs_d:.3f}')
axes[0].set_xlabel('CAPM 的 HJ 距离 δ̂（月度）'); axes[0].set_ylabel('频数（400 次）')
axes[0].set_title(f'图3a：T=600 时 δ 检验功效仅 {pow_d:.0%}（分布严重重叠）')
axes[0].legend(fontsize=7.5); axes[0].grid(alpha=.3)
axes[1].hist(null_g, bins=35, color='#7f8c8d', alpha=.7, label='CAPM 成立世界')
axes[1].hist(alt_g, bins=35, color='#27ae60', alpha=.7, label='三因子真实世界')
axes[1].axvline(thr_g, color='k', ls='--', label=f'零分布 95% 分位 {thr_g:.3f}')
axes[1].axvline(obs_g, color='#c0392b', lw=2.2, label=f'本样本观测 {obs_g:.3f}')
axes[1].set_xlabel('界违反缺口 gap = HJ 界 − SDF 比率')
axes[1].set_title(f'图3b：同一个零假设，「界违反」检验功效 {pow_g:.0%}')
axes[1].legend(fontsize=7.5); axes[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{OUT}/mc-null.png', dpi=130); plt.close()

# ============ 5. N 扫描 ============
print('\n=== N 扫描（正确模型的样本内 δ，T=600）===')
Ns = [4, 8, 12, 20, 30, 50, 80]; scan = {}
for n_ in Ns:
    ins = np.empty(150)
    for rep in range(150):
        Rm, Fm_, _, _, _ = simulate(T=T, N=n_, seed=9000 + rep * 13 + n_)
        _, _, d, _ = sdf_from_factors(Rm, Fm_[:, :3])
        ins[rep] = d
    scan[n_] = dict(mean=float(ins.mean()), q05=float(np.quantile(ins, .05)),
                    q95=float(np.quantile(ins, .95)))
    print(f'  N={n_:3d}  δ 均值 {ins.mean():.4f}  [5%,95%]=[{np.quantile(ins,.05):.4f},'
          f'{np.quantile(ins,.95):.4f}]  √(N/T)={np.sqrt(n_/T):.4f}')
res['nscan'] = scan
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(Ns, [scan[n_]['mean'] for n_ in Ns], 'o-', color='#2980b9', label='样本内 δ 均值（模型正确）')
ax.fill_between(Ns, [scan[n_]['q05'] for n_ in Ns], [scan[n_]['q95'] for n_ in Ns],
                color='#2980b9', alpha=.2, label='5%–95%')
ax.plot(Ns, [np.sqrt(n_/T) for n_ in Ns], 'k--', label=r'理论噪声地板 $\sqrt{N/T}$')
ax.set_xlabel('检验资产个数 N（T=600）'); ax.set_ylabel('HJ 距离 δ（月度）')
ax.set_title('图4：模型完全正确，样本内 δ 仍随 N 膨胀——δ 不能跨 N 比大小')
ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{OUT}/n-scan.png', dpi=130); plt.close()

# ============ 6. 功效 vs 样本长度 ============
print('\n=== 功效 vs 样本长度（200/200 次）===')
Ts = [600, 1200, 2400, 4800, 9600]
pw = {}
for tt in Ts:
    nd, ad = np.empty(200), np.empty(200); ng, ag = np.empty(200), np.empty(200)
    for rep in range(200):
        R0, F0, _, _, FS0 = simulate(T=tt, seed=11000 + rep * 31 + tt, B=B0, lam=LAM0)
        _, _, d0, ra0 = sdf_from_factors(R0, np.column_stack([F0, FS0])[:, [0]])
        nd[rep], ng[rep] = d0, tan_sharpe(R0) - ra0
        R1, F1, _, _, FS1 = simulate(T=tt, seed=13000 + rep * 31 + tt, B=B0)
        _, _, d1, ra1 = sdf_from_factors(R1, np.column_stack([F1, FS1])[:, [0]])
        ad[rep], ag[rep] = d1, tan_sharpe(R1) - ra1
    pw[tt] = dict(pow_d=float((ad > np.quantile(nd, .95)).mean()),
                  pow_g=float((ag > np.quantile(ng, .95)).mean()),
                  floor=float(np.sqrt(N / tt)))
    print(f'  T={tt:5d}  δ 检验功效 {pw[tt]["pow_d"]:.3f}   '
          f'界违反检验功效 {pw[tt]["pow_g"]:.3f}   噪声地板 {pw[tt]["floor"]:.4f}')
res['power_scan'] = pw

_, alpha_capm, _, _ = sdf_from_factors(R, POOL[:, [0]])
_, alpha_ff3, _, _ = sdf_from_factors(R, POOL[:, [0, 1, 2]])
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
axes[0].plot(Ts, [pw[t]['pow_d'] for t in Ts], 'o-', color='#e67e22', label='δ 检验')
axes[0].plot(Ts, [pw[t]['pow_g'] for t in Ts], 's-', color='#27ae60', label='界违反检验')
axes[0].axhline(0.05, color='k', ls=':', lw=1)
axes[0].set_xlabel('样本长度 T（月度）'); axes[0].set_ylabel('检验功效（5% 尺寸）')
axes[0].set_xscale('log'); axes[0].set_title('图5a：δ 检验要靠样本长度堆功效，界违反检验不怎么需要')
axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
axes[1].scatter(alpha_capm * 100, B0[:, 1], color='#e67e22', s=48, label='CAPM 定价误差 α')
axes[1].scatter(alpha_ff3 * 100, B0[:, 1], color='#27ae60', s=48, label='FF3 定价误差 α')
z = np.polyfit(B0[:, 1], alpha_capm, 1)
xs_ = np.linspace(B0[:, 1].min(), B0[:, 1].max(), 20)
axes[1].plot(np.polyval(z, xs_) * 100, xs_, color='#e67e22', lw=1.2, ls='--', alpha=.8)
axes[1].axvline(0, color='k', lw=.8)
axes[1].set_xlabel('定价误差 α（月度 %）'); axes[1].set_ylabel('规模因子载荷 $B_{i,2}$')
axes[1].set_title(f'图5b：CAPM 的 α 由遗漏载荷解释（斜率 {z[0]:.4f}）')
axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{OUT}/power-scan.png', dpi=130); plt.close()
res['capm_alpha_slope'] = float(z[0])
res['alpha_capm_rms'] = float(np.sqrt((alpha_capm**2).mean())) * 100
res['alpha_ff3_rms'] = float(np.sqrt((alpha_ff3**2).mean())) * 100
res['alpha_capm_max'] = float(np.abs(alpha_capm).max()) * 100

with open('sdf_results.json', 'w') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print('\n=== 完成 ===')
print(f"CAPM α RMS {res['alpha_capm_rms']:.3f}%/月，|α|max {res['alpha_capm_max']:.3f}%；"
      f"FF3 α RMS {res['alpha_ff3_rms']:.3f}%")
print(f"市场夏普 {res['mkt_sharpe']:.4f}  样本 HJ 界 {S_tan:.4f}  总体界 {POP['S_tan']:.4f}")
