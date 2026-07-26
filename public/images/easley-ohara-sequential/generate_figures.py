"""
Generate figures for Easley-O'Hara sequential trade model blog post.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os

OUT = '/Users/halo/workspace/astro-blog/public/images/easley-ohara-sequential'

# ─── Font setup ────────────────────────────────────────────────────────────────
try:
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Arial Unicode MS', 'SimHei']
except Exception:
    pass
plt.rcParams['axes.unicode_minus'] = False

# ─── Shared palette ────────────────────────────────────────────────────────────
BLUE   = '#2563EB'
RED    = '#DC2626'
GREEN  = '#16A34A'
AMBER  = '#D97706'
PURPLE = '#7C3AED'
GRAY   = '#6B7280'
ALPHA  = 0.85


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: PIN composition – bar chart of parameter contributions
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_pin_composition():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=130)

    # Left: bar chart – how alpha, delta, mu, epsilon influence PIN
    params = ['α\n(信息事件概率)', 'δ\n(坏消息条件概率)', 'μ\n(知情下单率)', 'ε\n(噪声下单率)']
    # Sensitivity direction (↑ means larger value → larger PIN)
    direction = [1, 1, 1, -1]  # epsilon inverse
    colors = [BLUE, PURPLE, RED, GREEN]

    bars = axes[0].bar(params, [0.6, 0.5, 0.8, 0.4], color=colors, alpha=ALPHA, edgecolor='white', linewidth=1.5)
    axes[0].axhline(0.5, color=GRAY, linestyle='--', linewidth=1.2, label='基准值')
    axes[0].set_title('PIN 各参数的敏感性方向', fontsize=14, fontweight='bold', pad=12)
    axes[0].set_ylabel('参数相对影响系数', fontsize=11)
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)

    # Annotate direction
    for bar, d in zip(bars, direction):
        y = bar.get_height()
        arrow = '↑ 增大 PIN' if d == 1 else '↓ 增大 ε → PIN 减小'
        axes[0].text(bar.get_x() + bar.get_width()/2, y + 0.02, arrow,
                     ha='center', va='bottom', fontsize=9, color=bar.get_facecolor())

    # Right: PIN formula decomposition diagram
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('PIN = αμ / (αμ + 2ε) 的几何直觉', fontsize=14, fontweight='bold', pad=12)

    # Boxes
    def box(ax, x, y, w, h, label, color, fontsize=11):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                                        facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=color)

    box(ax2, 0.5, 7.5, 3.5, 1.5, 'α (信息事件概率)', BLUE, 10)
    box(ax2, 4.5, 7.5, 3.5, 1.5, 'μ (知情下单率)', RED, 10)
    box(ax2, 0.5, 5.0, 7.5, 1.5, 'α·μ  (知情下单总量)', PURPLE, 11)

    box(ax2, 0.5, 2.5, 3.5, 1.5, 'ε (噪声买单率)', GREEN, 10)
    box(ax2, 4.5, 2.5, 3.5, 1.5, 'ε (噪声卖单率)', GREEN, 10)
    box(ax2, 0.5, 0.5, 7.5, 1.5, '2ε  (噪声下单总量)', GRAY, 11)

    # Arrows
    for (x0, y0, x1, y1) in [(2.25, 7.5, 4.25, 5.0), (6.25, 7.5, 6.25, 5.0),
                              (2.25, 2.5, 4.25, 3.5), (6.25, 2.5, 6.25, 3.5)]:
        ax2.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))

    # PIN formula
    ax2.text(5, 4.0, 'PIN = αμ / (αμ + 2ε)', ha='center', va='center',
             fontsize=14, fontweight='bold', color=RED,
             bbox=dict(facecolor='#FEE2E2', edgecolor=RED, boxstyle='round,pad=0.4', linewidth=2))

    # Numerator/denominator labels
    ax2.annotate('分子：\n知情下单', xy=(3.5, 5.5), xytext=(1.2, 9.0),
                 fontsize=9, color=PURPLE,
                 arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.2))
    ax2.annotate('分母：\n知情 + 噪声', xy=(5.0, 3.0), xytext=(8.0, 1.2),
                 fontsize=9, color=GRAY,
                 arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.2))

    plt.tight_layout(pad=2)
    path = os.path.join(OUT, 'fig1-pin-composition.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    sz = os.path.getsize(path) / 1024
    print(f'fig1: {path}  {sz:.1f} KB')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: EM / MLE parameter convergence
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_mle_convergence():
    np.random.seed(42)

    # True parameters
    alpha_true, delta_true = 0.25, 0.40
    mu_true, eps_true      = 0.80, 1.20

    # Function: PIN-like "score" for a given (alpha, mu, eps) given fixed delta
    # We simulate EM iterations by doing numerical MLE on a growing dataset
    def neg_log_likelihood_single(params, B, S):
        alpha, mu, eps = params
        if alpha <= 0 or alpha >= 1 or mu <= 0 or eps <= 0:
            return 1e12
        # Simplified PIN likelihood for one observation window
        theta = alpha * (delta_true * mu + (1 - delta_true) * mu)
        # Very rough approximation: log-likelihood proportional to B*log(alpha*mu) + S*log(2*eps) ...
        # We use the EKOP full likelihood approximation
        ll = 0.0
        for b, s in zip(B, S):
            # Pr(B=b, S=s) = (1-delta)*alpha*exp(-alpha*mu)* (alpha*mu)^b / b! * ...
            # Full EKOP below
            try:
                log_p = _ekop_ll(alpha, delta_true, mu, eps, int(b), int(s))
            except:
                return 1e12
            ll += log_p
        return -ll

    def _ekop_ll(alpha, delta, mu, eps, B, S):
        """EKOP log-likelihood for one (B,S) observation."""
        import math
        lam = alpha * mu
        # Pr(B,S | no info) = Poisson(B|eps) * Poisson(S|eps)
        # Pr(B,S | good info) = Poisson(B|lam+eps) * Poisson(S|eps)
        # Pr(B,S | bad info)  = Poisson(B|eps) * Poisson(S|lam+eps)
        p0 = (math.exp(-eps) * eps**B / math.factorial(B) *
              math.exp(-eps) * eps**S / math.factorial(S))
        pg = (math.exp(-(lam+eps)) * (lam+eps)**B / math.factorial(B) *
              math.exp(-eps) * eps**S / math.factorial(S))
        pb = (math.exp(-eps) * eps**B / math.factorial(B) *
              math.exp(-(lam+eps)) * (lam+eps)**S / math.factorial(S))
        L  = (1-delta)*alpha*p0 + delta*(1-alpha)*pg + delta*alpha*pb
        if L <= 0:
            return -300.0
        return math.log(L)

    def simulate_bs(alpha, delta, mu, eps, n_days, seed=99):
        rng = np.random.default_rng(seed)
        B_list, S_list = [], []
        for _ in range(n_days):
            has_info = rng.random() < alpha
            if not has_info:
                B = rng.poisson(eps)
                S = rng.poisson(eps)
            else:
                bad = rng.random() < delta
                if bad:
                    B = rng.poisson(eps)
                    S = rng.poisson(mu + eps)
                else:
                    B = rng.poisson(mu + eps)
                    S = rng.poisson(eps)
            B_list.append(B)
            S_list.append(S)
        return np.array(B_list), np.array(S_list)

    # Vary sample sizes and run MLE via scipy-style grid + scipy
    from scipy.optimize import minimize
    from scipy.special import gammaln

    def ekop_ll_full(params, B, S):
        """Full EKOP log-likelihood (vectorized over observations)."""
        alpha, delta, mu, eps = params
        if alpha <= 1e-6 or alpha >= 1-1e-6 or mu <= 1e-6 or eps <= 1e-6 or delta <= 1e-6 or delta >= 1-1e-6:
            return 1e10
        lam = alpha * mu
        total = 0.0
        for b, s in zip(B, S):
            p0 = (np.exp(-eps) * eps**b / _fact(b) *
                  np.exp(-eps) * eps**s / _fact(s))
            pg = (np.exp(-(lam+eps)) * (lam+eps)**b / _fact(b) *
                  np.exp(-eps) * eps**s / _fact(s))
            pb = (np.exp(-eps) * eps**b / _fact(b) *
                  np.exp(-(lam+eps)) * (lam+eps)**s / _fact(s))
            L  = (1-delta)*alpha*p0 + (1-alpha)*delta*pg + alpha*delta*pb
            if L <= 1e-300:
                total += -700.0
            else:
                total += np.log(L)
        return -total

    def _fact(n):
        import math
        if n > 170:
            return np.inf
        return math.factorial(n)

    # Run MLE for different sample sizes
    sample_sizes = [50, 100, 200, 500, 1000, 2000]
    alpha_hats, delta_hats, mu_hats, eps_hats = [], [], [], []
    alpha_stds, delta_stds, mu_stds, eps_stds = [], [], [], []

    for n in sample_sizes:
        B, S = simulate_bs(alpha_true, delta_true, mu_true, eps_true, n, seed=42)
        # Bootstrap to get std
        ah, dh, mh, eh = [], [], [], []
        for _boot in range(50):
            idx = np.random.randint(0, n, n)
            Bb, Sb = B[idx], S[idx]
            res = minimize(ekop_ll_full,
                           x0=[0.3, 0.5, 1.0, 1.0],
                           args=(Bb, Sb),
                           method='L-BFGS-B',
                           bounds=[(0.01, 0.99), (0.01, 0.99), (0.01, 5.0), (0.01, 5.0)])
            if res.success:
                ah.append(res.x[0])
                dh.append(res.x[1])
                mh.append(res.x[2])
                eh.append(res.x[3])
        alpha_hats.append(np.mean(ah))
        alpha_stds.append(np.std(ah))
        delta_hats.append(np.mean(dh))
        delta_stds.append(np.std(dh))
        mu_hats.append(np.mean(mh))
        mu_stds.append(np.std(mh))
        eps_hats.append(np.mean(eh))
        eps_stds.append(np.std(eh))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=130)
    true_vals  = [alpha_true, delta_true, mu_true, eps_true]
    est_vals   = [alpha_hats, delta_hats, mu_hats, eps_hats]
    std_vals   = [alpha_stds, delta_stds, mu_stds, eps_stds]
    titles     = ['α (信息事件概率)', 'δ (坏消息条件概率)', 'μ (知情下单率)', 'ε (噪声下单率)']
    colors     = [BLUE, PURPLE, RED, GREEN]
    ylims      = [(0, 0.8), (0, 0.8), (0, 2.5), (0, 3.0)]

    for ax, tv, ev, sv, title, col, ylim in zip(axes.flat, true_vals, est_vals, std_vals, titles, colors, ylims):
        ax.errorbar(sample_sizes, ev, yerr=[1.96*s for s in sv],
                    fmt='o-', color=col, ecolor=col, capsize=4, linewidth=2,
                    markersize=6, label='估计值 ± 95% CI')
        ax.axhline(tv, color='black', linestyle='--', linewidth=1.5, label=f'真值 = {tv}')
        ax.fill_between(sample_sizes,
                        [tv - 1.96*s for s in sv],
                        [tv + 1.96*s for s in sv],
                        alpha=0.1, color=col)
        ax.set_xscale('log')
        ax.set_xlabel('样本天数 N', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(*ylim)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle('EKOP / PIN 参数最大似然估计：估计值随样本量收敛到真值',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT, 'fig2-mle-convergence.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    sz = os.path.getsize(path) / 1024
    print(f'fig2: {path}  {sz:.1f} KB')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Order imbalance distribution – info day vs no-info day
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_oi_distribution():
    np.random.seed(7)

    def simulate_oi(alpha, delta, mu, eps, n_days=500):
        rng = np.random.default_rng(7)
        oi_informed = []   # OI = (B-S)/(B+S)
        oi_uninformed = []
        for _ in range(n_days):
            has_info = rng.random() < alpha
            if not has_info:
                B = rng.poisson(eps)
                S = rng.poisson(eps)
                oi_uninformed.append((B - S) / (B + S + 1e-9))
            else:
                bad = rng.random() < delta
                if bad:
                    B = rng.poisson(eps)
                    S = rng.poisson(mu + eps)
                else:
                    B = rng.poisson(mu + eps)
                    S = rng.poisson(eps)
                oi_informed.append((B - S) / (B + S + 1e-9))
        return np.array(oi_informed), np.array(oi_uninformed)

    oi_inf, oi_uninf = simulate_oi(alpha=0.25, delta=0.40, mu=0.80, eps=1.20)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=130)

    bins = np.linspace(-1, 1, 41)
    axes[0].hist(oi_uninf, bins=bins, color=GREEN, alpha=0.6, label=f'无信息日 (N={len(oi_uninf)})',
                density=True, edgecolor='white', linewidth=0.8)
    axes[0].hist(oi_inf,   bins=bins, color=RED,   alpha=0.5, label=f'有信息日 (N={len(oi_inf)})',
                density=True, edgecolor='white', linewidth=0.8, hatch='//')
    axes[0].axvline(np.mean(oi_uninf), color=GREEN, linestyle='--', linewidth=2,
                    label=f'无信息均值={np.mean(oi_uninf):.3f}')
    axes[0].axvline(np.mean(oi_inf),   color=RED,   linestyle='--', linewidth=2,
                    label=f'有信息均值={np.mean(oi_inf):.3f}')
    axes[0].set_xlabel('买卖单不平衡 OI = (B−S)/(B+S)', fontsize=11)
    axes[0].set_ylabel('密度', fontsize=11)
    axes[0].set_title('买卖单不平衡分布：有信息日 vs 无信息日', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].text(0.05, 0.92,
                 f'无信息日 OI 标准差: {np.std(oi_uninf):.3f}\n'
                 f'有信息日 OI 标准差: {np.std(oi_inf):.3f}',
                 transform=axes[0].transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor=GRAY))

    # Right: box plot
    data_box = [oi_uninf, oi_inf]
    colors_box = [GREEN, RED]
    bp = axes[1].boxplot(data_box,
                         patch_artist=True, widths=0.5,
                         medianprops=dict(color='black', linewidth=2))
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(['无信息日', '有信息日'])
    for patch, col in zip(bp['boxes'], colors_box):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    for flier in bp['fliers']:
        flier.set(marker='o', alpha=0.4, markersize=3)
    axes[1].set_ylabel('买卖单不平衡 OI', fontsize=11)
    axes[1].set_title('OI 分布对比：箱线图', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    # Annotation
    diff = np.abs(np.mean(oi_inf)) - np.abs(np.mean(oi_uninf))
    axes[1].annotate(f'|均值差异|: {diff:.3f}\n有信息日 OI 更偏向\n方向极端值',
                     xy=(1.5, np.mean(oi_inf)), xytext=(1.7, 0),
                     fontsize=10, color=PURPLE,
                     arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.5))

    plt.suptitle('EKOP 模型：信息事件会显著拉偏买卖单不平衡分布',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT, 'fig3-oi-distribution.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    sz = os.path.getsize(path) / 1024
    print(f'fig3: {path}  {sz:.1f} KB')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: PIN sensitivity heatmap
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_pin_heatmap():
    alphas = np.linspace(0.05, 0.80, 50)
    epsilons = np.linspace(0.2, 4.0, 50)
    mu = 0.80   # fix mu

    PIN = np.zeros((len(epsilons), len(alphas)))
    for i, eps in enumerate(epsilons):
        for j, alpha in enumerate(alphas):
            PIN[i, j] = (alpha * mu) / (alpha * mu + 2 * eps)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=130)

    # Heatmap
    im = axes[0].imshow(PIN, extent=[alphas[0], alphas[-1], epsilons[0], epsilons[-1]],
                        origin='lower', aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    axes[0].set_xlabel('α (信息事件概率)', fontsize=12)
    axes[0].set_ylabel('ε (噪声下单率)', fontsize=12)
    axes[0].set_title('PIN = αμ / (αμ + 2ε)  敏感性热力图\n(μ = 0.80 固定)',
                      fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=axes[0], shrink=0.85)
    cbar.set_label('PIN', fontsize=11)

    # Contour lines
    CS = axes[0].contour(alphas, epsilons, PIN,
                          levels=[0.05, 0.10, 0.20, 0.30, 0.50],
                          colors='black', linewidths=1.0, alpha=0.7)
    axes[0].clabel(CS, inline=True, fontsize=8, fmt='%.2f')

    # Right: cross-section curves – fix alpha, vary epsilon, or vice versa
    alpha_vals = [0.10, 0.25, 0.50, 0.75]
    eps_range  = np.linspace(0.2, 4.0, 100)
    cmap_lines = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(alpha_vals)))

    for a_val, col in zip(alpha_vals, cmap_lines):
        pin_curve = (a_val * mu) / (a_val * mu + 2 * eps_range)
        axes[1].plot(eps_range, pin_curve, color=col, linewidth=2.5,
                     label=f'α = {a_val}')

    axes[1].set_xlabel('ε (噪声下单率)', fontsize=12)
    axes[1].set_ylabel('PIN', fontsize=12)
    axes[1].set_title('固定 α，PIN 随 ε 的变化曲线', fontsize=13, fontweight='bold')
    axes[1].legend(title='信息事件概率 α', fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 0.9)
    axes[1].axhline(0.10, color=GRAY, linestyle=':', linewidth=1, label='_nolegend_')
    axes[1].axhline(0.20, color=GRAY, linestyle=':', linewidth=1)
    axes[1].annotate('ε↑ → PIN↓\n噪声掩盖信息',
                     xy=(3.5, 0.07), fontsize=10, color=GRAY,
                     bbox=dict(facecolor='white', alpha=0.7))

    plt.suptitle('PIN 对 α 和 ε 的敏感性：热力图与截面曲线',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT, 'fig4-pin-sensitivity.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    sz = os.path.getsize(path) / 1024
    print(f'fig4: {path}  {sz:.1f} KB')


if __name__ == '__main__':
    print('Generating figures...')
    fig1_pin_composition()
    fig2_mle_convergence()
    fig3_oi_distribution()
    fig4_pin_heatmap()
    print('Done.')
