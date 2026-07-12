#!/usr/bin/env python3
"""
为文章「鲁棒投资组合优化：在最坏情形下也能站着」(robust-portfolio-optimization)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

设定(自洽合成, 仅用于演示方法):
  * 真实协方差 Sigma_true = B B' + D (因子模型), 固定
  * 真实预期收益 mu_true = B @ factor_premia + 小量 idio
  * 估计噪声: 从 N(mu_true, Sigma_true) 抽 T 条历史, 得到样本均值 mu_hat(误差协方差 Sigma_mu = Sigma_true/T) 与样本协方差 Sigma_hat
  * 名义均值-方差(MV):      max w'mu_hat - (lam/2) w'Sigma_hat w
  * 鲁棒均值-方差(Robust):   max w'mu_hat - kappa*sqrt(w'Sigma_mu w) - (lam/2) w'Sigma_hat w
        sqrt(w'Sigma_mu w) 正是椭球不确定性集 {(mu-mu_hat)'Sigma_mu^-1(mu-mu_hat) <= kappa^2}
        下 mu 取到的最坏情形偏移量; kappa 越大越保守
  * 评价: 用真实 mu_true, Sigma_true 算样本外 Sharpe = w'mu_true / sqrt(w'Sigma_true w)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "robust-portfolio-optimization")
os.makedirs(D, exist_ok=True)

C = {"eq": "#4C72B0", "vix": "#C44E52", "grid": "#DDDDDD", "fv": "#55A868",
     "rv": "#DD8452", "rob": "#9467bd", "nom": "#C44E52", "thr": "#888888",
     "green": "#2ca02c", "orange": "#FF7F0E", "blue": "#1f77b4"}

rng0 = np.random.default_rng(20260713)
N = 10
B = rng0.normal(0.0, 0.05, size=(N, 2))
Ddiag = np.diag(rng0.uniform(0.004, 0.010, N))
SIGMA_TRUE = B @ B.T + Ddiag
FACT = np.array([0.010, 0.006])
alpha = np.linspace(0.010, 0.045, N)
mu_true = alpha + B @ FACT   # 真实(超额)预期收益

def solve_mv(mu, Sigma, lam, lbound=0.0):
    n = len(mu)
    def neg_obj(w):
        return -(w @ mu - 0.5 * lam * (w @ Sigma @ w))
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(lbound, 1.0)] * n
    res = minimize(neg_obj, np.ones(n) / n, method="SLSQP",
                   bounds=bounds, constraints=cons, options={"ftol": 1e-12})
    return res.x

def solve_robust(mu_hat, Sigma_mu, Sigma_hat, lam, kappa, lbound=0.0):
    n = len(mu_hat)
    def neg_obj(w):
        wc = np.sqrt(max(w @ Sigma_mu @ w, 1e-18))
        return -(w @ mu_hat - kappa * wc - 0.5 * lam * (w @ Sigma_hat @ w))
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(lbound, 1.0)] * n
    res = minimize(neg_obj, np.ones(n) / n, method="SLSQP",
                   bounds=bounds, constraints=cons, options={"ftol": 1e-12})
    return res.x

def oos_sharpe(w):
    return w @ mu_true / np.sqrt(max(w @ SIGMA_TRUE @ w, 1e-18))

# ---------------------------------------------------------------- 图1: 椭球不确定性集 + 最坏情形
def fig_ellipse():
    i, j = 0, 1
    mu2 = mu_true[[i, j]]
    Sm2 = SIGMA_TRUE[[i, j]][:, [i, j]] / 60.0     # Sigma_mu (T=60)
    kappa = 1.5
    L = np.linalg.cholesky(Sm2)
    theta = np.linspace(0, 2 * np.pi, 400)
    circ = np.array([np.cos(theta), np.sin(theta)])
    ell = mu2[:, None] + kappa * (L @ circ)
    # 候选权重(向资产 j 倾斜), 展示最坏情形偏移方向
    w = np.array([0.15, 0.85]); w = w / w.sum()
    off = kappa * (L @ (L.T @ w)) / np.sqrt(max(w @ Sm2 @ w, 1e-18))
    mu_worst = mu2 - off
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.plot(ell[0], ell[1], color=C["eq"], lw=2.2, label=r"椭球不确定性集 $(T=60,\kappa=1.5)$")
    ax.scatter([mu2[0]], [mu2[1]], color=C["fv"], zorder=5, s=80, label=r"估计均值 $\hat\mu$")
    ax.plot([mu2[0], mu_worst[0]], [mu2[1], mu_worst[1]], color=C["vix"], lw=2.0,
            ls="--", label=r"最坏情形均值 $\hat\mu-\kappa\Sigma_\mu^{1/2}u$")
    ax.scatter([mu_worst[0]], [mu_worst[1]], color=C["vix"], zorder=5, s=70)
    ax.annotate(r"资产 A", (mu2[0], mu2[1]), textcoords="offset points", xytext=(-6, 10), fontsize=10)
    ax.annotate(r"资产 B", (mu2[0], mu2[1]), textcoords="offset points", xytext=(30, -16), fontsize=10)
    ax.set_xlabel(r"资产 A 预期收益 $\mu_A$", fontsize=11)
    ax.set_ylabel(r"资产 B 预期收益 $\mu_B$", fontsize=11)
    ax.set_title(r"椭球不确定性集下, 集中下注被推向最坏方向", fontsize=12.5)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True, color=C["grid"])
    fig.tight_layout()
    fig.savefig(os.path.join(D, "robust_uncertainty_ellipse.png"), dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------- 图2: 权重随 kappa 变化
def fig_weights_kappa():
    T = 60
    rng = np.random.default_rng(7)
    R = rng.multivariate_normal(mu_true, SIGMA_TRUE, size=T)
    mu_hat = R.mean(0)
    Sigma_hat = np.cov(R.T)
    Sigma_mu = SIGMA_TRUE / T
    lam = 5.0
    kappas = [0.0, 0.5, 1.0, 2.0, 4.0]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for k, col in zip(kappas, [C["nom"], C["orange"], C["rv"], C["rob"], C["eq"]]):
        w = solve_robust(mu_hat, Sigma_mu, Sigma_hat, lam, k, lbound=0.0)
        ax.plot(range(1, N + 1), w * 100, marker="o", ms=4, lw=1.8,
                color=col, label=fr"$\kappa={k}$")
    ax.axhline(100.0 / N, color=C["thr"], ls=":", lw=1.5, label="等权 12.5%")
    ax.set_xlabel("资产序号", fontsize=11)
    ax.set_ylabel("权重 (%)", fontsize=11)
    ax.set_title(r"鲁棒性参数 $\kappa$ 越大, 权重越分散(向等权靠拢)", fontsize=12.5)
    ax.legend(ncol=3, fontsize=9)
    ax.grid(True, color=C["grid"])
    fig.tight_layout()
    fig.savefig(os.path.join(D, "robust_weights_vs_kappa.png"), dpi=150)
    plt.close(fig)
    return mu_hat, Sigma_hat, Sigma_mu, lam

# ---------------------------------------------------------------- 图3: OOS Sharpe 箱线图(名义 vs 鲁棒)
def fig_oos_box(mu_hat, Sigma_hat, Sigma_mu, lam, kappa=1.5, trials=600):
    rng = np.random.default_rng(2026)
    nom, rob = [], []
    for _ in range(trials):
        R = rng.multivariate_normal(mu_true, SIGMA_TRUE, size=60)
        mh = R.mean(0); Sh = np.cov(R.T); Sm = SIGMA_TRUE / 60.0
        wn = solve_mv(mh, Sh, lam)
        wr = solve_robust(mh, Sm, Sh, lam, kappa)
        nom.append(oos_sharpe(wn)); rob.append(oos_sharpe(wr))
    nom = np.array(nom); rob = np.array(rob)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    bp = ax.boxplot([nom, rob], tick_labels=[r"名义 MV ($\kappa=0$)", fr"鲁棒 ($\kappa={kappa}$)"],
                    patch_artist=True, widths=0.55, showfliers=False)
    for patch, col in zip(bp["boxes"], [C["nom"], C["rob"]]):
        patch.set_facecolor(col); patch.set_alpha(0.35)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_ylabel("样本外 Sharpe (真实参数)", fontsize=11)
    ax.set_title(r"估计误差下: 鲁棒组合的样本外 Sharpe 更稳", fontsize=12.5)
    ax.grid(True, color=C["grid"], axis="y")
    q = (np.percentile(nom, 5), np.percentile(rob, 5))
    print(f"[OOS] nominal Sharpe median={nom.mean():.3f} 5%={q[0]:.3f} | robust median={rob.mean():.3f} 5%={q[1]:.3f}")
    print(f"[OOS] robust 最差情形(5%) 提升 vs 名义: {(q[1]-q[0]):.3f} Sharpe")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "robust_oos_sharpe.png"), dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------- 图4: OOS 表现随样本量 T
def fig_oos_vs_T(kappa=1.5, lam=5.0, trials=400):
    Ts = [30, 45, 60, 90, 120, 180, 240, 360]
    nom_med, rob_med, nom_p5, rob_p5 = [], [], [], []
    rng = np.random.default_rng(99)
    for T in Ts:
        nset, rset = [], []
        for _ in range(trials):
            R = rng.multivariate_normal(mu_true, SIGMA_TRUE, size=T)
            mh = R.mean(0); Sh = np.cov(R.T); Sm = SIGMA_TRUE / T
            nset.append(oos_sharpe(solve_mv(mh, Sh, lam)))
            rset.append(oos_sharpe(solve_robust(mh, Sm, Sh, lam, kappa)))
        nset = np.array(nset); rset = np.array(rset)
        nom_med.append(nset.mean()); rob_med.append(rset.mean())
        nom_p5.append(np.percentile(nset, 5)); rob_p5.append(np.percentile(rset, 5))
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(Ts, nom_med, "o-", color=C["nom"], label="名义 MV 中位数")
    ax.plot(Ts, rob_med, "s-", color=C["rob"], label=fr"鲁棒 $\kappa={kappa}$ 中位数")
    ax.fill_between(Ts, nom_p5, nom_med, color=C["nom"], alpha=0.12)
    ax.fill_between(Ts, rob_p5, rob_med, color=C["rob"], alpha=0.12)
    ax.set_xlabel("估计样本长度 $T$ (交易日)", fontsize=11)
    ax.set_ylabel("样本外 Sharpe", fontsize=11)
    ax.set_title(r"样本越短, 鲁棒优化的优势越大(随 $T$ 增大收敛)", fontsize=12.5)
    ax.legend(fontsize=10)
    ax.grid(True, color=C["grid"])
    fig.tight_layout()
    fig.savefig(os.path.join(D, "robust_oos_vs_T.png"), dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    fig_ellipse()
    mu_hat, Sigma_hat, Sigma_mu, lam = fig_weights_kappa()
    fig_oos_box(mu_hat, Sigma_hat, Sigma_mu, lam, kappa=1.5)
    fig_oos_vs_T()
    print("DONE robust-portfolio-optimization images")
