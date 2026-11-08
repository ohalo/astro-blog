"""函数型主成分（FPCA）分析日内波动率曲线：完整可运行实验。

运行方式（在项目根目录）：
    python3 scripts/article_core_fpca_20261108pm.py

产出：
    public/images/functional-pca-intraday-volatility-curve/fig1..fig5.png
    public/images/functional-pca-intraday-volatility-curve/results.json

核心问题：曲线是函数，不是向量。把基展开系数直接拿去做普通 PCA，
等价于偷偷把 L2 内积换成了欧氏内积，得到的「主成分」并不正交，
解释方差比例也是错的。下面用同一套 Gauss-Legendre 求积权重把所有量
（内积、总方差、重构误差）统一到 L2[0,1] 下，并把错误做法一起算出来对拍。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 20261108
OUT = "public/images/functional-pca-intraday-volatility-curve"
T = 500
N_TRAIN = 350
P_OBS = 48          # 每天 48 个 5 分钟已实现波动点
K = 8               # 高斯 RBF 基个数
M_GRID = 200        # L2 内积求积点数
RBF_S = 0.09
RHO_AR = 0.72
RHO_NOISE = 0.50
LAMBDA = 0.01       # 二阶差分粗糙度惩罚
N_FPC = 3
SIGMA_MAIN = 0.20
SIGMAS = (0.02, 0.05, 0.10, 0.20, 0.35)
LAMBDAS = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
N_KEEP = (48, 36, 24, 16, 12)   # 每天保留的观测点数（不规则采样）


def quad_grid(m=M_GRID):
    """[0,1] 上的 Gauss-Legendre 求积节点与权重（含区间缩放）。"""
    x, w = np.polynomial.legendre.leggauss(m)
    return 0.5 * (x + 1.0), 0.5 * w


def l2_norm(v, w):
    return float(np.sqrt(np.sum(w * v * v)))


def basis(x, centers, s=RBF_S):
    return np.exp(-0.5 * ((x[:, None] - centers[None, :]) / s) ** 2)


def make_true_shapes(xg, wg, xo):
    """三条真实成分：水平、U 型日内季节性、开盘衰减，均归一到 L2 单位范数。"""
    g1 = np.ones_like(xg)
    g2 = 12.0 * (xg - 0.5) ** 2 - 1.0
    g3 = np.exp(-8.0 * xg)
    shapes = [g / l2_norm(g, wg) for g in (g1, g2, g3)]
    o1 = np.ones_like(xo)
    o2 = 12.0 * (xo - 0.5) ** 2 - 1.0
    o3 = np.exp(-8.0 * xo)
    obs_shapes = np.column_stack([
        o1 / l2_norm(g1, wg), o2 / l2_norm(g2, wg), o3 / l2_norm(g3, wg)
    ])
    return np.column_stack(shapes), obs_shapes


def gen_scores(seed=SEED, n=T, rho=RHO_AR):
    """三条成分的分数服从独立 AR(1)，稳态标准差 (0.30, 0.18, 0.12)。"""
    rng = np.random.default_rng(seed)
    sd = np.array([0.30, 0.18, 0.12])
    innov_sd = sd * np.sqrt(1.0 - rho ** 2)
    a = np.zeros((n, 3))
    a[0] = rng.standard_normal(3) * sd
    for t in range(1, n):
        a[t] = rho * a[t - 1] + rng.standard_normal(3) * innov_sd
    return a


def gen_noise(seed=SEED + 1, n=T, p=P_OBS, rho=RHO_NOISE):
    """沿日内方向 AR(1) 相关的微观结构噪声，边际方差 1。"""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((n, p))
    z = np.zeros((n, p))
    z[:, 0] = eps[:, 0]
    for j in range(1, p):
        z[:, j] = rho * z[:, j - 1] + np.sqrt(1.0 - rho ** 2) * eps[:, j]
    return z


def penalty(k=K):
    """二阶差分惩罚矩阵 D^T D。"""
    d = np.zeros((k - 2, k))
    for i in range(k - 2):
        d[i, i:i + 3] = (1.0, -2.0, 1.0)
    return d.T @ d


def fit_coefficients(y, phi_obs, omega, lam):
    """把离散观测投影到基上：带粗糙度惩罚的岭回归，返回 (T, K) 系数。"""
    lhs = phi_obs.T @ phi_obs + lam * omega
    rhs = y @ phi_obs                      # (T, K)
    return np.linalg.solve(lhs, rhs.T).T   # (K, K) \ (K, T) -> (T, K)


def fpca_basis(s, W, m=N_FPC):
    """正确的 FPCA：在 L2 内积下解 S W b = lam b。

    W = L L^T，令 u = L^T b，则 L^T S L u = lam u（对称问题）。
    b = L^{-T} u 自动满足 B^T W B = I，总方差 = trace(S W) = sum(lam)。
    """
    L = np.linalg.cholesky(W)
    sym = L.T @ s @ L
    eig, U = np.linalg.eigh(sym)
    order = np.argsort(eig)[::-1]
    eig, U = eig[order], U[:, order]
    B = np.linalg.solve(L.T, U[:, :m])
    return B, eig


def wrong_fpca_basis(s, m=N_FPC):
    """错误做法：直接对系数协方差 S 做普通特征分解，忽略 Gram 矩阵 W。

    得到的方向只在欧氏内积下正交，在真正的 L2 内积下既不正交也不归一。
    """
    eig, V = np.linalg.eigh(s)
    order = np.argsort(eig)[::-1]
    return V[:, order[:m]], eig[order]


def recon_from_coeff(a, B, W, phi_eval):
    """正确 FPCA 重构：系数 a -> 分数 a^T W B -> 函数 phi_eval @ (B s)。"""
    scores = a @ W @ B
    return scores @ B.T @ phi_eval.T


def recon_from_coeff_wrong(a, B, phi_eval):
    """错误版重构：把系数欧氏内积当成函数内积，分数直接取 a^T B。"""
    scores = a @ B
    return scores @ B.T @ phi_eval.T


def recon_pointwise(y, V, mean):
    """逐点 PCA 的重构（只在观测网格上定义）。"""
    return (y - mean) @ V @ V.T + mean


def one_step_forecast(s, n_train):
    """对分数序列拟合 VAR(1) 并做一步预测，返回 (系数矩阵, 测试段预测)。"""
    x = np.column_stack([np.ones(n_train - 1), s[:n_train - 1]])
    y = s[1:n_train]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    x_all = np.column_stack([np.ones(len(s) - 1), s[:-1]])
    return beta, x_all @ beta


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def run():
    os.makedirs(OUT, exist_ok=True)
    xg, wg = quad_grid()
    xo = (np.arange(P_OBS) + 0.5) / P_OBS
    centers = np.linspace(0.0, 1.0, K)
    phi_g = basis(xg, centers)          # (M, K)
    phi_o = basis(xo, centers)          # (P, K)
    W = phi_g.T @ (wg[:, None] * phi_g)  # L2 Gram 矩阵
    omega = penalty()

    G_grid, G_obs = make_true_shapes(xg, wg, xo)
    scores_true = gen_scores()
    F_grid = 0.9 + scores_true @ G_grid.T        # 无噪真曲线（求积网格）
    F_obs = 0.9 + scores_true @ G_obs.T          # 无噪真曲线（观测网格）
    Z = gen_noise()                              # 共同随机数：各扫描只改 sigma
    Y = F_obs + SIGMA_MAIN * Z

    A = fit_coefficients(Y, phi_o, omega, LAMBDA)
    A_center = A - A[:N_TRAIN].mean(axis=0)
    S = np.cov(A_center[:N_TRAIN].T)
    B_ok, eig_ok = fpca_basis(S, W)
    B_bad, eig_bad = wrong_fpca_basis(S)
    # 为可比性，把错误方向按各自 L2 范数归一化后再量正交性偏离
    d = np.sqrt(np.diag(B_bad.T @ W @ B_bad))
    B_bad_norm = B_bad / d[None, :]

    # 逐点 PCA（对观测矩阵直接 SVD）
    Y_mean = Y[:N_TRAIN].mean(axis=0)
    _, _, Vt = np.linalg.svd(Y[:N_TRAIN] - Y_mean, full_matrices=False)
    V_pca = Vt[:N_FPC].T

    # 系数中心化后的重构要平移回原尺度：用训练段平滑曲线的观测网格均值
    smooth_mean = A[:N_TRAIN].mean(axis=0) @ phi_o.T
    rec_ok = recon_from_coeff(A_center, B_ok, W, phi_o) + smooth_mean
    rec_bad = recon_from_coeff_wrong(A_center, B_bad, phi_o) + smooth_mean
    rec_pca = recon_pointwise(Y, V_pca, Y_mean)

    main = {
        "W_condition_number": float(np.linalg.cond(W)),
        "total_variance_L2": float(np.sum(eig_ok)),
        "trace_SW_check": float(np.trace(S @ W)),
        "explained_top3_correct": float(np.sum(eig_ok[:N_FPC]) / np.sum(eig_ok)),
        "explained_top3_wrong": float(np.sum(eig_bad[:N_FPC]) / np.sum(eig_bad)),
        "L2_orthogonality_err_correct": float(
            np.max(np.abs(B_ok.T @ W @ B_ok - np.eye(N_FPC)))),
        "L2_orthogonality_err_wrong": float(
            np.max(np.abs(B_bad_norm.T @ W @ B_bad_norm - np.eye(N_FPC)))),
        "recon_mse_correct": mse(F_obs[N_TRAIN:], rec_ok[N_TRAIN:]),
        "recon_mse_wrong": mse(F_obs[N_TRAIN:], rec_bad[N_TRAIN:]),
        "recon_mse_pointwise": mse(F_obs[N_TRAIN:], rec_pca[N_TRAIN:]),
    }

    # 噪声扫描（共同随机数，只改噪声幅度）
    scan = []
    for sig in SIGMAS:
        Yi = F_obs + sig * Z
        Ai = fit_coefficients(Yi, phi_o, omega, LAMBDA)
        Ac = Ai - Ai[:N_TRAIN].mean(axis=0)
        Si = np.cov(Ac[:N_TRAIN].T)
        Bi, _ = fpca_basis(Si, W)
        _, _, Vti = np.linalg.svd(Yi[:N_TRAIN] - Yi[:N_TRAIN].mean(axis=0),
                                  full_matrices=False)
        Vi = Vti[:N_FPC].T
        sm = Ai[:N_TRAIN].mean(axis=0) @ phi_o.T
        r_ok = recon_from_coeff(Ac, Bi, W, phi_o) + sm
        r_pca = recon_pointwise(Yi, Vi, Yi[:N_TRAIN].mean(axis=0))
        scan.append({
            "sigma": sig,
            "fpca": mse(F_obs[N_TRAIN:], r_ok[N_TRAIN:]),
            "pointwise": mse(F_obs[N_TRAIN:], r_pca[N_TRAIN:]),
        })

    # 平滑参数扫描
    lam_scan = []
    for lam in LAMBDAS:
        Ai = fit_coefficients(Y, phi_o, omega, lam)
        Ac = Ai - Ai[:N_TRAIN].mean(axis=0)
        Si = np.cov(Ac[:N_TRAIN].T)
        Bi, _ = fpca_basis(Si, W)
        sm = Ai[:N_TRAIN].mean(axis=0) @ phi_o.T
        r_ok = recon_from_coeff(Ac, Bi, W, phi_o) + sm
        lam_scan.append({"lambda": lam,
                         "fpca": mse(F_obs[N_TRAIN:], r_ok[N_TRAIN:])})

    # 不规则采样扫描：每天随机只保留 n_keep 个观测点，且各天位置不同。
    # FPCA 直接在可用点上做基回归；逐点 PCA 必须先把曲线插值补齐到同一网格。
    rng_m = np.random.default_rng(SEED + 2)
    missing_scan = []
    for nk in N_KEEP:
        idx_all = [np.sort(rng_m.choice(P_OBS, nk, replace=False))
                   for _ in range(T)]
        Ai = np.zeros((T, K))
        Yfill = np.zeros((T, P_OBS))
        for t in range(T):
            idx = idx_all[t]
            pm = phi_o[idx]
            Ai[t] = np.linalg.solve(pm.T @ pm + LAMBDA * omega,
                                    pm.T @ Y[t, idx])
            Yfill[t] = np.interp(xo, xo[idx], Y[t, idx])
        Ac = Ai - Ai[:N_TRAIN].mean(axis=0)
        Si = np.cov(Ac[:N_TRAIN].T)
        Bi, _ = fpca_basis(Si, W)
        sm = Ai[:N_TRAIN].mean(axis=0) @ phi_o.T
        r_ok = recon_from_coeff(Ac, Bi, W, phi_o) + sm
        mfill = Yfill[:N_TRAIN].mean(axis=0)
        _, _, Vti = np.linalg.svd(Yfill[:N_TRAIN] - mfill,
                                  full_matrices=False)
        Vi = Vti[:N_FPC].T
        r_pca = recon_pointwise(Yfill, Vi, mfill)
        missing_scan.append({
            "n_keep": nk,
            "fpca": mse(F_obs[N_TRAIN:], r_ok[N_TRAIN:]),
            "pointwise": mse(F_obs[N_TRAIN:], r_pca[N_TRAIN:]),
        })

    # 一步预测：持久性 / 全基 VAR(1) / FPCA-3 / 逐点 PCA-3
    pred = {}
    # 持久性（昨日平滑曲线当明日预测）
    smooth_all = A @ phi_o.T
    pred["persistent"] = mse(F_obs[N_TRAIN:], smooth_all[N_TRAIN - 1:-1])
    # 全基 VAR(1)：K 个系数全部保留，只是平滑、不降维
    _, af = one_step_forecast(A_center, N_TRAIN)
    pred["full_basis_ar1"] = mse(
        F_obs[N_TRAIN:], af[N_TRAIN - 1:] @ phi_o.T + smooth_mean)
    # FPCA-3 分数
    fs = A_center @ W @ B_ok
    _, ff = one_step_forecast(fs, N_TRAIN)
    pred["fpca3_ar1"] = mse(
        F_obs[N_TRAIN:], ff[N_TRAIN - 1:] @ B_ok.T @ phi_o.T + smooth_mean)
    # 逐点 PCA-3 分数
    ps = (Y - Y_mean) @ V_pca
    _, pf = one_step_forecast(ps, N_TRAIN)
    pred["pointwise3_ar1"] = mse(
        F_obs[N_TRAIN:], pf[N_TRAIN - 1:] @ V_pca.T + Y_mean)

    results = {"config": {"T": T, "n_train": N_TRAIN, "p_obs": P_OBS,
                          "K": K, "grid": M_GRID, "sigma": SIGMA_MAIN,
                          "lambda": LAMBDA, "seed": SEED},
               "main": main, "noise_scan": scan, "lambda_scan": lam_scan,
               "missing_scan": missing_scan, "forecast_oos_mse": pred}

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---------------- 五张真实计算图 ----------------
    plt.rcParams.update({"font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 110})

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for t in (20, 60, 100):
        ax[0].plot(xo, F_obs[t], lw=1.8, label=f"true t={t}")
        ax[0].plot(xo, Y[t], lw=0.8, alpha=0.65, label=f"observed t={t}")
    ax[0].set_xlabel("intraday time (normalised)")
    ax[0].set_ylabel("realised volatility")
    ax[0].set_title("(a) 5-min intraday curves, sigma=0.20")
    ax[0].legend(fontsize=7, ncol=2)
    for j, lab in enumerate(("level", "U-shape", "opening decay")):
        ax[1].plot(xg, G_grid[:, j], lw=1.8, label=lab)
    ax[1].set_xlabel("intraday time (normalised)")
    ax[1].set_ylabel("L2-normalised shape")
    ax[1].set_title("(b) three true components")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_curves_and_shapes.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for j in range(N_FPC):
        ax[0].plot(xg, phi_g @ B_ok[:, j], lw=2, label=f"FPC {j + 1}")
        ax[1].plot(xg, phi_g @ B_bad[:, j], lw=2, label=f"wrong {j + 1}")
    ax[0].set_title("(a) correct FPCA (L2-orthogonal)")
    ax[1].set_title("(b) plain eig(S) (Euclidean)")
    for a in ax:
        a.set_xlabel("intraday time (normalised)")
        a.set_ylabel("eigenfunction value")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_eigenfunctions.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot([s["sigma"] for s in scan], [s["fpca"] for s in scan],
               "o-", lw=2, label="FPCA (correct L2)")
    ax[0].plot([s["sigma"] for s in scan], [s["pointwise"] for s in scan],
               "s-", lw=2, label="pointwise PCA")
    ax[0].set_xlabel("observation noise sigma")
    ax[0].set_ylabel("reconstruction MSE")
    ax[0].set_title("(a) noise sweep, full 48-point grid")
    ax[0].legend(fontsize=8)
    ax[1].plot([s["n_keep"] for s in missing_scan],
               [s["fpca"] for s in missing_scan],
               "o-", lw=2, label="FPCA (correct L2)")
    ax[1].plot([s["n_keep"] for s in missing_scan],
               [s["pointwise"] for s in missing_scan],
               "s-", lw=2, label="pointwise PCA + interp")
    ax[1].set_xlabel("observed points kept per day")
    ax[1].set_ylabel("reconstruction MSE")
    ax[1].set_title("(b) irregular / sparse sampling")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_noise_scan.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    bars = ax[0].bar(["correct", "wrong"],
                     [main["explained_top3_correct"],
                      main["explained_top3_wrong"]],
                     color=["#3b6ea5", "#c0654a"], width=0.55)
    ax[0].bar_label(bars, fmt="%.4f", fontsize=9)
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("cumulative explained variance (top 3)")
    ax[0].set_title("(a) variance accounting")
    bars = ax[1].bar(["correct", "wrong"],
                     [main["L2_orthogonality_err_correct"],
                      main["L2_orthogonality_err_wrong"]],
                     color=["#3b6ea5", "#c0654a"], width=0.55)
    ax[1].bar_label(bars, fmt="%.2e", fontsize=9)
    ax[1].set_ylabel("max |<g_i,g_j> - delta_ij|")
    ax[1].set_title("(b) L2 orthogonality violation")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_variance_and_orthogonality.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    names = list(pred)
    vals = [pred[k] for k in names]
    bars = ax[0].barh(range(len(names)), vals, color="#4f7d68")
    ax[0].set_yticks(range(len(names)))
    ax[0].set_yticklabels(names, fontsize=8)
    ax[0].bar_label(bars, fmt="%.5f", fontsize=8)
    ax[0].set_xlabel("OOS one-step forecast MSE")
    ax[0].set_title("(a) denoising beats dimension reduction")
    ax[1].plot([s["lambda"] for s in lam_scan],
               [s["fpca"] for s in lam_scan], "o-", lw=2)
    ax[1].set_xscale("symlog", linthresh=1e-3)
    ax[1].set_xlabel("roughness penalty lambda")
    ax[1].set_ylabel("reconstruction MSE")
    ax[1].set_title("(b) smoothing sensitivity")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_forecast_and_lambda.png")
    plt.close(fig)

    print("FPCA experiment (T=%d, p=%d, K=%d, sigma=%.2f)" %
          (T, P_OBS, K, SIGMA_MAIN))
    print("  W condition number             : %.2f" % main["W_condition_number"])
    print("  trace(SW) vs sum(eigen)        : %.6f / %.6f" %
          (main["trace_SW_check"], main["total_variance_L2"]))
    print("  top3 variance correct / wrong  : %.4f / %.4f" %
          (main["explained_top3_correct"], main["explained_top3_wrong"]))
    print("  L2 orthogonality err ok / bad  : %.2e / %.2e" %
          (main["L2_orthogonality_err_correct"],
           main["L2_orthogonality_err_wrong"]))
    print("  recon MSE correct / wrong / pt : %.5f / %.5f / %.5f" %
          (main["recon_mse_correct"], main["recon_mse_wrong"],
           main["recon_mse_pointwise"]))
    for s in scan:
        print("  sigma=%.2f  FPCA %.5f  pointwise %.5f  ratio %.2fx" %
              (s["sigma"], s["fpca"], s["pointwise"],
               s["pointwise"] / s["fpca"]))
    for s in missing_scan:
        print("  keep=%2d  FPCA %.5f  pointwise %.5f  ratio %.2fx" %
              (s["n_keep"], s["fpca"], s["pointwise"],
               s["pointwise"] / s["fpca"]))
    for k, v in pred.items():
        print("  forecast %-18s : %.5f" % (k, v))
    print("  lambda scan MSE               :",
          ", ".join("%g:%.5f" % (s["lambda"], s["fpca"])
                    for s in lam_scan))
    return results


if __name__ == "__main__":
    run()
