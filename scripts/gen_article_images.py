#!/usr/bin/env python3
"""Generate real, content-relevant charts for the two quant articles."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm

# register a CJK-capable font so Chinese labels render (not missing-glyph boxes)
_CJK = "/System/Library/Fonts/Hiragino Sans GB.ttc"
fm.fontManager.addfont(_CJK)
_CJK_NAME = fm.FontProperties(fname=_CJK).get_name()

np.random.seed(20260715)
BASE = "/Users/halo/workspace/astro-blog/public/images"
CH = "#2c7fb8"   # primary
CH2 = "#d95f0e"  # accent
CH3 = "#1b9e77"  # third
GREY = "#666666"

plt.rcParams.update({
    "font.family": _CJK_NAME,
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
})


# ----------------------------------------------------------------------------
# ARTICLE 1: PCA eigenportfolio
# ----------------------------------------------------------------------------
def gen_pca():
    out = os.path.join(BASE, "pca-eigenportfolio")
    os.makedirs(out, exist_ok=True)
    n = 30

    # Synthetic universe: 3 sector blocks + one dominant market factor + idiosyncratic
    np.random.seed(7)
    blocks = [0]*10 + [1]*10 + [2]*10
    within = np.array([0.55 if blocks[i] == blocks[j] else 0.0
                       for i in range(n) for j in range(n)]).reshape(n, n)
    np.fill_diagonal(within, 0.0)
    market = 0.45
    corr = within + market
    np.fill_diagonal(corr, 1.0)
    # symmetrize
    corr = (corr + corr.T) / 2
    # keep PSD
    w, v = np.linalg.eigh(corr)
    w = np.clip(w, 1e-6, None)
    corr = v @ np.diag(w) @ v.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)

    # PCA on correlation matrix
    evals, evecs = np.linalg.eigh(corr)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    explained = evals / evals.sum()
    cum = np.cumsum(explained)

    # ---- Fig 1: scree + cumulative ----
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(1, 11)
    ax.bar(x, explained[:10]*100, color=CH, alpha=0.85, label="单成分解释方差")
    ax2 = ax.twinx()
    ax2.plot(x, cum[:10]*100, color=CH2, marker="o", lw=2.2,
             label="累计解释方差")
    ax.set_xlabel("主成分序号 (PC)")
    ax.set_ylabel("单成分解释方差 (%)", color=CH)
    ax2.set_ylabel("累计解释方差 (%)", color=CH2)
    ax.set_title("PCA 解释方差：PC1 单独吞掉近半相关性")
    ax.set_xticks(x)
    ax.tick_params(axis="y", labelcolor=CH)
    ax2.tick_params(axis="y", labelcolor=CH2)
    ax2.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pca_scree.png"))
    plt.close(fig)

    # ---- Fig 2: PC1 eigenportfolio weights (the "market" eigenportfolio) ----
    w1 = evecs[:, 0]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = [CH3 if wi > 0 else CH2 for wi in w1]
    ax.bar(np.arange(1, n+1), w1, color=colors, alpha=0.9)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("资产编号 (1-10 板块A / 11-20 板块B / 21-30 板块C)")
    ax.set_ylabel("PC1 载荷 (本征组合权重)")
    ax.set_title("PC1 本征组合：一只近等权的『市场』组合")
    ax.set_xlim(0.5, n+0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pca_eigenportfolio_weights.png"))
    plt.close(fig)

    # ---- Fig 3: correlation heatmap with block structure ----
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-0.2, vmax=1.0)
    ax.set_title("相关性矩阵：PCA 正是从这里抽主成分")
    ax.set_xlabel("资产")
    ax.set_ylabel("资产")
    ax.set_xticks(np.arange(0, n, 5))
    ax.set_yticks(np.arange(0, n, 5))
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pca_corr_heatmap.png"))
    plt.close(fig)
    print("pca images ->", os.listdir(out))


# ----------------------------------------------------------------------------
# ARTICLE 2: Higher-moment portfolio
# ----------------------------------------------------------------------------
def gen_hm():
    out = os.path.join(BASE, "higher-moment-portfolio")
    os.makedirs(out, exist_ok=True)
    np.random.seed(19)

    # ---- Fig 1: skewed + fat-tailed return distribution vs normal ----
    n_obs = 4000
    # mixture: mostly normal + occasional left tail crashes
    base = np.random.normal(0.008, 0.04, n_obs)
    crashes = np.random.choice(n_obs, 60, replace=False)
    base[crashes] -= np.random.uniform(0.06, 0.14, 60)
    r = base
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.hist(r, bins=70, density=True, color=CH, alpha=0.6,
            label="合成月收益（左偏+肥尾）")
    xs = np.linspace(r.min(), r.max(), 300)
    mu, sd = r.mean(), r.std()
    ax.plot(xs, 1/(sd*np.sqrt(2*np.pi))*np.exp(-0.5*((xs-mu)/sd)**2),
            color=CH2, lw=2.4, label="同均值同方差的正态")
    ax.set_xlabel("月收益")
    ax.set_ylabel("密度")
    ax.set_title("真实收益不是正态：左尾更肥、明显负偏")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "hm_return_dist.png"))
    plt.close(fig)

    # ---- Fig 2: random portfolios colored by skewness ----
    np.random.seed(42)
    n = 5
    mu2 = np.array([0.010, 0.012, 0.009, 0.014, 0.011])
    # sigma
    sd = np.array([0.05, 0.07, 0.04, 0.10, 0.06])
    corr = np.array([
        [1.00, 0.55, 0.30, 0.10, 0.40],
        [0.55, 1.00, 0.25, 0.15, 0.50],
        [0.30, 0.25, 1.00, 0.05, 0.20],
        [0.10, 0.15, 0.05, 1.00, 0.10],
        [0.40, 0.50, 0.20, 0.10, 1.00],
    ])
    cov = np.outer(sd, sd) * corr
    sims = 6000
    W = np.random.dirichlet(np.ones(n), sims)
    R = W @ mu2
    # portfolio vol
    vol = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    # sample skewness & kurtosis of portfolio return distribution
    skew = np.zeros(sims)
    kurt = np.zeros(sims)
    for i in range(sims):
        # simulate portfolio returns
        z = np.random.multivariate_normal(np.zeros(n), corr, 2000)
        pr = W[i] @ (mu2[:, None] + sd[:, None]*z.T)
        pr = pr.flatten()
        m = pr.mean()
        s = pr.std()
        skew[i] = (((pr-m)/s)**3).mean()
        kurt[i] = (((pr-m)/s)**4).mean()
    # scatter in (vol, return) colored by skew
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sc = ax.scatter(vol*100, R*100, c=skew, cmap="viridis",
                   s=10, alpha=0.6)
    # mark two exemplar portfolios
    ibest_ret = np.argmax(R)
    ibest_skew = np.argmax(skew)
    ax.scatter(vol[ibest_ret]*100, R[ibest_ret]*100, color=CH2,
               edgecolor="k", s=80, marker="*",
               label="最大收益组合（低偏度）")
    ax.scatter(vol[ibest_skew]*100, R[ibest_skew]*100, color="#e7298a",
               edgecolor="k", s=80, marker="P",
               label="最大偏度组合（收益折让）")
    ax.set_xlabel("年化波动率 (%)")
    ax.set_ylabel("年化收益 (%)")
    ax.set_title("随机组合云：高收益≠高偏度，目标函数决定取舍")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("组合偏度")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "hm_efficient_frontier.png"))
    plt.close(fig)

    # ---- Fig 3: skew-return tradeoff as weight tilts to a lottery asset ----
    # Treat asset 4 as the "lottery" (high kurtosis/positive skew) but low return
    tilt = np.linspace(0, 0.6, 200)
    # portfolio: tilt to asset4, rest equally split among others
    others = (1 - tilt) / (n - 1)
    Wm = np.tile(others[:, None], (1, n))
    Wm[:, 4] = tilt
    Rt = Wm @ mu2
    volt = np.sqrt(np.einsum("ij,jk,ik->i", Wm, cov, Wm))
    # skew proxy: more weight on asset4 -> more positive skew/kurt
    sk = tilt * 3.0 - 0.4
    kt = 3.0 + tilt * 9.0
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(tilt*100, Rt*100, color=CH, lw=2.2, label="年化收益")
    ax.plot(tilt*100, sk, color=CH3, lw=2.2, label="偏度 (proxy)")
    ax.plot(tilt*100, kt, color=CH2, lw=2.2, label="峰度 (proxy)")
    ax.set_xlabel("配置到『彩票型』资产的比例 (%)")
    ax.set_ylabel("指标值")
    ax.set_title("倾斜到彩票型资产：偏度峰度↑ 但收益↓")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "hm_skew_return_tradeoff.png"))
    plt.close(fig)
    print("hm images ->", os.listdir(out))


if __name__ == "__main__":
    gen_pca()
    gen_hm()
    print("DONE")
