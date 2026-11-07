#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""11-01 两篇重写版：真实计算配图 + results.json + 文章代码块（与 core 逐字同源）。"""
import json
import inspect
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ib_pc_core as C

ROOT = Path('/Users/halo123456/workspace/astro-blog')
OUT_IB = ROOT / 'public/images/information-bottleneck-sequence'
OUT_PC = ROOT / 'public/images/pc-algorithm-causal-factor'
for d in (OUT_IB, OUT_PC):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
RED, BLUE, GREEN, GRAY, ORANGE = '#c0392b', '#2980b9', '#27ae60', '#7f8c8d', '#e67e22'


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print('  saved', path.name)


def cover(path, title, subtitle, tag):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), 12, 6, facecolor='#f8f9fa', edgecolor='none'))
    ax.text(6, 4.5, title, fontsize=32, fontweight='bold', ha='center', va='center', color='#1a1a2e')
    ax.text(6, 3.4, subtitle, fontsize=18, ha='center', va='center', color='#4a5568')
    ax.text(6, 2.5, tag, fontsize=15, ha='center', va='center', color='#6b7278')
    ax.plot([3, 9], [1.9, 1.9], color=RED, linewidth=3)
    ax.text(6, 1.3, '2026-11-01 | Quant Column', fontsize=12, ha='center', va='center', color='#8d949a')
    save(fig, path)


# ---------------------------------------------------------------- IB 图
def draw_ib(stats, plots):
    # 图1：三类 DGP 的真实样本序列
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    labels = ['AR(1): x_t = 0.6 x_{t-1} + 0.3 e_t',
              'AR(2): x_t = 0.4 x_{t-1} + 0.35 x_{t-2} + 0.3 e_t',
              'Lorenz system (x component, RK4 dt=0.01)']
    for ax, kind, lab in zip(axes, C.DGPS, labels):
        s = plots['series'][kind]
        ax.plot(np.arange(len(s)), s, color=BLUE, lw=1.0)
        ax.set_ylabel('x (standardized)', fontsize=9)
        ax.set_title(lab, fontsize=10, loc='left')
        ax.grid(alpha=.3)
    axes[-1].set_xlabel('time step')
    fig.suptitle('Three simulated DGPs used in the regression experiment', fontsize=12)
    save(fig, OUT_IB / 'ib_1_series.png')

    # 图2：真实训练曲线（MSE 与验证 MAE，展示过拟合）
    hp, hv = plots['curve_plain'], plots['curve_vib']
    itp = [r[0] for r in hp]; itv = [r[0] for r in hv]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(itp, [r[2] for r in hp], color=GRAY, lw=1.6, label='plain (beta=0) train MSE')
    axes[0].plot(itv, [r[2] for r in hv], color=RED, lw=1.6, label='VIB (beta=0.02) train MSE')
    axes[0].plot(itv, [r[3] for r in hv], color=ORANGE, lw=1.3, ls=':', label='VIB KL (nats)')
    axes[0].set_xlabel('iteration'); axes[0].set_ylabel('loss')
    axes[0].set_title('Training loss (minibatch)'); axes[0].legend(fontsize=8)
    axes[1].plot(itp, [r[4] for r in hp], color=GRAY, lw=1.8, label='plain (beta=0)')
    axes[1].plot(itv, [r[4] for r in hv], color=RED, lw=1.8, label='VIB (beta=0.02)')
    axes[1].axhline(min(r[4] for r in hp), color=GRAY, ls='--', lw=1)
    axes[1].axhline(min(r[4] for r in hv), color=RED, ls='--', lw=1)
    axes[1].set_xlabel('iteration'); axes[1].set_ylabel('validation MAE')
    axes[1].set_title('Validation MAE rises after ~100 iters (overfitting)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Real training trajectories, AR(2) with 8 nuisance dimensions (seed 20261101)', fontsize=11)
    save(fig, OUT_IB / 'ib_2_train_curve.png')

    # 图3：β 扫描（早停 vs 末点）
    bs = np.array([float(b) for b in C.BETAS])
    eb = np.array([stats['beta_scan'][str(b)] for b in C.BETAS])
    fb = np.array([stats['beta_scan_final'][str(b)] for b in C.BETAS])
    x = np.where(bs == 0, 5e-5, bs)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.semilogx(x, eb, 'o-', color=BLUE, lw=1.8, label='validation MAE (early stopping)')
    ax.semilogx(x, fb, 's--', color=GRAY, lw=1.5, label='validation MAE (final iteration)')
    ax.axhline(eb[0], color=BLUE, ls=':', lw=1, label=f'plain baseline = {eb[0]:.4f}')
    ax.set_xlabel('beta (log scale; 0 plotted at 5e-5)'); ax.set_ylabel('validation MAE')
    ax.set_title('Beta sweep: mild regularization helps, strong compression hurts')
    ax.legend(fontsize=8)
    save(fig, OUT_IB / 'ib_3_beta_scan.png')

    # 图4：干扰维扫描
    ds = [int(d) for d in C.NUIS]
    pl = [stats['nuisance_scan'][str(d)]['plain'] for d in ds]
    vb = [stats['nuisance_scan'][str(d)]['vib'] for d in ds]
    gaps = [stats['nuisance_scan'][str(d)]['gap_pct'] for d in ds]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(ds, pl, 'o-', color=GRAY, lw=1.8, label='plain (beta=0)')
    ax.plot(ds, vb, 's-', color=RED, lw=1.8, label='VIB (beta=0.02)')
    ax.set_xlabel('number of structured AR(1) nuisance dimensions'); ax.set_ylabel('validation MAE')
    ax.set_title('Nuisance dimensions vs generalization')
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.bar(ds, gaps, alpha=.18, color=GREEN, width=1.2)
    ax2.set_ylabel('VIB improvement (%)', color=GREEN)
    ax2.axhline(0, color=GREEN, lw=.8, ls=':')
    ax.legend(loc='upper left', fontsize=8)
    save(fig, OUT_IB / 'ib_4_nuisance.png')

    # 图5：瓶颈各维 mu 方差
    mv = stats['mu_var_by_dim']
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bars = ax.bar(np.arange(1, len(mv) + 1), mv, color=[GREEN if v > 0.01 else GRAY for v in mv])
    ax.axhline(0.01, color=RED, ls='--', lw=1.2, label='active threshold (var > 0.01)')
    ax.set_xlabel('latent dimension'); ax.set_ylabel('Var(mu) on validation set')
    ax.set_title(f'Bottleneck usage: {stats["active_dims"]}/{len(mv)} dims active, '
                 f'KL = {stats["kl_at_beta002"]:.3f} nats at beta=0.02')
    ax.legend(fontsize=8)
    save(fig, OUT_IB / 'ib_5_bottleneck.png')


# ---------------------------------------------------------------- PC 图
def _draw_graph(ax, edges, p, title, highlight=None, missing=None):
    ang = np.linspace(0, 2 * np.pi, p, endpoint=False) + np.pi / 2
    pos = np.column_stack([np.cos(ang), np.sin(ang)])
    for e in edges:
        i, j = sorted(e)
        col = GRAY
        ls = '-'
        if highlight is not None and e in highlight.get('fp', set()):
            col, ls = RED, '-'
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color=col, lw=1.6, ls=ls, zorder=1)
    for e in (missing or set()):
        i, j = sorted(e)
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                color=ORANGE, lw=2.0, ls='--', zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], s=520, c='#dfe6e9', edgecolors='#2d3436', zorder=2)
    for i in range(p):
        ax.text(pos[i, 0], pos[i, 1], str(i), ha='center', va='center', fontsize=11, zorder=3)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal'); ax.axis('off'); ax.grid(False)
    ax.set_title(title, fontsize=10)


def draw_pc(stats, plots):
    truth, found = plots['demo_truth'], plots['demo_found']
    fp, fn = found - truth, truth - found
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    _draw_graph(axes[0], truth, 8, f'True skeleton ({len(truth)} edges)')
    _draw_graph(axes[1], found, 8,
                f'PC-stable recovered ({len(found)} edges: {len(fp)} FP, {len(fn)} FN; '
                'orange dashed = missed edge)',
                {'fp': fp}, missing=fn)
    fig.suptitle('PC skeleton recovery, p=8, n=2000, alpha=0.05 '
                 f'(precision={stats["demo"]["precision"]}, recall={stats["demo"]["recall"]})',
                 fontsize=11)
    save(fig, OUT_PC / 'pc_1_dag.png')

    # 图2：样本量功效
    ns = [int(n) for n in stats['n_scan']]
    rec = [stats['n_scan'][n]['recall'] for n in stats['n_scan']]
    prec = [stats['n_scan'][n]['precision'] for n in stats['n_scan']]
    shd = [stats['n_scan'][n]['shd'] for n in stats['n_scan']]
    rstd = [stats['n_scan'][n]['recall_std'] for n in stats['n_scan']]
    rec, rstd = np.array(rec), np.array(rstd)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.semilogx(ns, rec, 'o-', color=BLUE, lw=1.8, label='recall (edge detection)')
    ax.fill_between(ns, rec - rstd, rec + rstd, color=BLUE, alpha=.15)
    ax.semilogx(ns, prec, 's-', color=GREEN, lw=1.8, label='precision')
    ax.set_xlabel('sample size n (log scale)'); ax.set_ylabel('precision / recall')
    ax.set_ylim(0.4, 1.05)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.semilogx(ns, shd, '^--', color=RED, lw=1.5, label='SHD (right)')
    ax2.set_ylabel('structural Hamming distance', color=RED)
    ax.set_title('Power curve: recall rises with n, SHD never reaches 0')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='lower right')
    save(fig, OUT_PC / 'pc_2_n_power.png')

    # 图3：α 敏感性
    als = [float(a) for a in stats['alpha_scan']]
    prec = [stats['alpha_scan'][a]['precision'] for a in stats['alpha_scan']]
    rec = [stats['alpha_scan'][a]['recall'] for a in stats['alpha_scan']]
    shd = [stats['alpha_scan'][a]['shd'] for a in stats['alpha_scan']]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.semilogx(als, prec, 's-', color=GREEN, lw=1.8, label='precision')
    ax.semilogx(als, rec, 'o-', color=BLUE, lw=1.8, label='recall')
    ax.set_xlabel('significance level alpha (log scale)'); ax.set_ylabel('precision / recall')
    ax.set_ylim(0.4, 1.05)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.semilogx(als, shd, '^--', color=RED, lw=1.5, label='SHD (right)')
    ax2.set_ylabel('SHD', color=RED)
    ax.axvline(float(stats['best_alpha']), color=ORANGE, ls=':', lw=1.5,
               label=f'best alpha = {stats["best_alpha"]}')
    ax.set_title('Alpha sweep at n=1000, p=8: conservative alpha wins')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    save(fig, OUT_PC / 'pc_3_alpha.png')

    # 图4：维度灾难（检验次数 + SHD）
    ps = [int(p) for p in stats['p_scan']]
    nt = [stats['p_scan'][p]['n_tests'] for p in stats['p_scan']]
    sh = [stats['p_scan'][p]['shd'] for p in stats['p_scan']]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(ps, nt, color=[BLUE, BLUE, ORANGE, RED], alpha=.75, width=2.6)
    ax.set_xlabel('number of variables p'); ax.set_ylabel('conditional independence tests')
    ax.set_title('Dimension cost: test count explodes with p (max_k=2)')
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(ps, sh, 'o--', color='#2d3436', lw=1.8, label='SHD (right)')
    ax2.set_ylabel('SHD')
    ax2.legend(fontsize=8)
    save(fig, OUT_PC / 'pc_4_dimension.png')

    # 图5：collider 反转
    ns = [int(n) for n in stats['collider']]
    mr = [abs(stats['collider'][n]['marginal_r']) for n in stats['collider']]
    pr = [abs(stats['collider'][n]['partial_r']) for n in stats['collider']]
    x = np.arange(len(ns)); w = 0.34
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - w / 2, mr, w, color=GRAY, label='|marginal corr(X1,X2)|')
    ax.bar(x + w / 2, pr, w, color=RED, label='|partial corr(X1,X2 | X3)|')
    ax.set_xticks(x); ax.set_xticklabels([f'n={n}' for n in ns])
    ax.set_ylabel('absolute correlation'); ax.set_ylim(0, 1.18)
    ax.set_title('Collider reversal: X1 _||_ X2 marginally, strongly dependent given X3')
    for i, n in enumerate(ns):
        d = stats['collider'][str(n)]
        pc = d['partial_p']
        ptext = 'p_cond < 1e-300' if pc == 0 else f'p_cond={pc:.1e}'
        ax.text(i, max(mr[i], pr[i]) + .05,
                f'p_marg={d["marginal_p"]:.2f}\n{ptext}',
                ha='center', va='bottom', fontsize=8)
    ax.legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=2)
    save(fig, OUT_PC / 'pc_5_collider.png')


# ---------------------------------------------------------------- 文章代码块（与 core 逐字同源）
MAIN_IB = '''if __name__ == '__main__':
    s, _ = run_ib()
    a = s['val_mae_by_dgp']['ar2']
    print('val_mae_plain     ', a['plain'])
    print('val_mae_vib       ', a['vib'])
    print('improve_pct       ', a['improve_pct'])
    print('overfit_plain     ', a['overfit_gap_plain'])
    print('overfit_vib       ', a['overfit_gap_vib'])
    print('best_beta         ', s['best_beta'])
    print('mae_at_best_beta  ', s['beta_scan'][str(s['best_beta'])])
    print('mae_at_beta03     ', s['beta_scan']['0.3'])
    print('gap_d0_pct        ', s['nuisance_scan']['0']['gap_pct'])
    print('gap_d16_pct       ', s['nuisance_scan']['16']['gap_pct'])
    print('kl_at_beta002     ', s['kl_at_beta002'])
    print('active_dims       ', s['active_dims'])
'''

MAIN_PC = '''if __name__ == '__main__':
    s, _ = run_pc()
    print('recall_n100       ', s['n_scan']['100']['recall'])
    print('recall_n5000      ', s['n_scan']['5000']['recall'])
    print('shd_n100          ', s['n_scan']['100']['shd'])
    print('shd_n5000         ', s['n_scan']['5000']['shd'])
    print('best_alpha        ', s['best_alpha'])
    print('shd_at_alpha01    ', s['alpha_scan']['0.1']['shd'])
    print('ntests_p16        ', s['p_scan']['16']['n_tests'])
    print('demo_precision    ', s['demo']['precision'])
    print('demo_recall       ', s['demo']['recall'])
    print('demo_n_tests      ', s['demo']['n_tests'])
    print('collider_marg_r   ', s['collider']['1000']['marginal_r'])
    print('collider_partial_r', s['collider']['1000']['partial_r'])
'''

CONSTS = f'''import numpy as np
from scipy import stats
from itertools import combinations

SEED_IB, SEED_PC = {C.SEED_IB}, {C.SEED_PC}
L, HID, K, ITERS, BS, LR = {C.L}, {C.HID}, {C.K}, {C.ITERS}, {C.BS}, {C.LR}
BETAS = {C.BETAS!r}
DGPS = {C.DGPS!r}
NUIS = {C.NUIS!r}
'''

IB_FUNCS = ['_series', 'make_dataset', 'init_model', 'Adam', 'predict', 'train',
            'best_mae', 'final_mae', 'run_ib']
PC_FUNCS = ['gen_dag', 'ci_test', 'pc_skeleton', 'score', 'run_pc']


def build_code(funcs, main):
    parts = [CONSTS]
    for name in funcs:
        parts.append(inspect.getsource(getattr(C, name)))
    parts.append(main)
    return '\n\n'.join(parts)


def main():
    print('=== 运行核心实验 ===')
    sib, pib = C.run_ib()
    spc, ppc = C.run_pc()

    print('\n=== 生成 IB 配图 ===')
    draw_ib(sib, pib)
    cover(OUT_IB / 'cover.png', '信息瓶颈的收益边界',
          '序列回归里 β、干扰维与过拟合的真实对照', 'VIB Regression | Beta Sweep | Honest Bounds')
    print('\n=== 生成 PC 配图 ===')
    draw_pc(spc, ppc)
    cover(OUT_PC / 'cover.png', 'PC 算法的功效边界',
          '样本量、维度与显著性水平的系统检验', 'Fisher-z | PC-stable | Power Analysis')

    (OUT_IB / 'results.json').write_text(json.dumps(sib, ensure_ascii=False, indent=1))
    (OUT_PC / 'results.json').write_text(json.dumps(spc, ensure_ascii=False, indent=1))

    ib_code = build_code(IB_FUNCS, MAIN_IB)
    pc_code = build_code(PC_FUNCS, MAIN_PC)
    (ROOT / 'scripts/_article_code_ib.py').write_text(ib_code)
    (ROOT / 'scripts/_article_code_pc.py').write_text(pc_code)
    print('\n=== 文章代码块 ===')
    print('  IB 代码块行数:', len(ib_code.splitlines()))
    print('  PC 代码块行数:', len(pc_code.splitlines()))

    # 自检
    assert abs(sib['val_mae_by_dgp']['ar2']['plain'] - sib['beta_scan']['0.0']) < 1e-9
    assert abs(sib['val_mae_by_dgp']['ar2']['vib'] - sib['beta_scan']['0.02']) < 1e-9
    assert abs(sib['nuisance_scan']['8']['plain'] - sib['beta_scan']['0.0']) < 1e-9
    assert all(np.isfinite(v) for v in sib['beta_scan'].values())
    assert spc['collider']['1000']['partial_p'] < 1e-50
    print('  自检通过（共同随机数一致、无 NaN、collider 反转显著）')


if __name__ == '__main__':
    main()
