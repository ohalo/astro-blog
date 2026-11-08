"""三维日内面板的 CP 分解：结构假设带来的收益，以及假设错配时的代价。

运行方式（在项目根目录）：
    python3 scripts/article_core_cp_20261108night.py

产出：
    public/images/cp-decomposition-intraday-tensor-panel/fig1..fig5.png
    public/images/cp-decomposition-intraday-tensor-panel/results.json

面板 Y[资产 a, 日 d, 日内时段 h]。真模型为 CP 秩 3：
    Y[a,d,h] = sum_r A[a,r] B[d,r] C[h,r] + 噪声
对照三种建模方式：
  CP-ALS        —— 承认「资产×日×时段」三向可分离（参数最少）
  展平 ALS      —— 展平成 资产 × (日*时段) 后做秩 R 加权最小二乘
  逐时段 ALS    —— 每个时段单独做秩 R 加权最小二乘（参数最多）
三种方法都用「只在观测元素上拟合」的同一套交替最小二乘，
差别只在结构假设，不掺优化器强弱。
除了在真 CP 结构下比精度，更重要的是三组反直觉检验：
  1) 秩误设时 CP 是否还赢；
  2) 真数据掺入「时段特异交互」后，结构假设什么时候开始倒戈；
  3) 非随机缺失（高波动更容易缺）下，随机留出评估会低估多少误差。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 20261108
OUT = "public/images/cp-decomposition-intraday-tensor-panel"
N, D, H = 36, 180, 20          # 资产数、交易日数、日内时段数
R_TRUE = 3
R_FIT = 3
SIGMAS = (0.15, 0.30, 0.60, 1.00, 1.50)
R_HATS = (1, 2, 3, 4, 6, 8)
GAMMAS = (0.0, 0.5, 1.0, 2.0)
SIGMA_MAIN = 0.30
HOLDOUT_FRAC = 0.20
MISS_FRAC = 0.25              # 在训练候选元素中再遮掉的比例
ALS_MAX_ITER = 100
ALS_N_INIT = 3
SVD_ITER = 20               # 矩阵 ALS 的交替轮数


def make_components(seed=SEED):
    """真 CP 成分：资产暴露 A、日度因子 B（AR(1)）、日内形状 C。"""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, R_TRUE))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    B = np.zeros((D, R_TRUE))
    sd = np.array([1.0, 0.6, 0.4])
    for r in range(R_TRUE):
        e = rng.standard_normal(D)
        rho = 0.85
        B[0, r] = e[0]
        for t in range(1, D):
            B[t, r] = rho * B[t - 1, r] + np.sqrt(1 - rho ** 2) * e[t]
        B[:, r] *= sd[r] / np.std(B[:, r])
    xh = np.linspace(0, 1, H)
    shapes = [np.ones(H), 12.0 * (xh - 0.5) ** 2 - 1.0, np.exp(-6.0 * xh)]
    C = np.column_stack([s / np.linalg.norm(s) for s in shapes])
    return A, B, C


def cp_tensor(A, B, C):
    return np.einsum("ar,dr,hr->adh", A, B, C)


def hour_specific(seed, gamma):
    """每时段独立的低秩交互：CP 结构无法用 3 个成分表示它。"""
    rng = np.random.default_rng(seed)
    X = np.zeros((N, D, H))
    for h in range(H):
        U = rng.standard_normal((N, 2))
        V = rng.standard_normal((D, 2))
        X[:, :, h] = U @ V.T / np.sqrt(2)
    X /= np.std(X)
    return gamma * X


def make_masks(seed):
    """固定 holdout 与 MCAR 训练观测；MNAR 用另一套同规模掩码。"""
    rng = np.random.default_rng(seed)
    u = rng.random((N, D, H))
    hold = u < HOLDOUT_FRAC
    cand = ~hold
    idx = np.flatnonzero(cand)
    pick = rng.permutation(idx)
    n_miss = int(len(idx) * MISS_FRAC)
    mcar = cand.copy()
    mcar.ravel()[pick[:n_miss]] = False
    return hold, mcar


def mnar_mask(truth, hold, seed):
    """高波动元素更容易缺失：保持与 MCAR 完全相同的训练观测数量。"""
    rng = np.random.default_rng(seed)
    cand = ~hold
    idx = np.flatnonzero(cand)
    n_miss = int(len(idx) * MISS_FRAC)
    score = np.abs(truth).ravel()[idx] + 1e-6 * rng.random(len(idx))
    drop = idx[np.argsort(score)[-n_miss:]]
    m = cand.copy()
    m.ravel()[drop] = False
    return m


def kr(X, Y):
    """Khatri-Rao 积：(n,r) 与 (m,r) -> (n*m,r)，行主序与前两个维度一致。"""
    return (X[:, None, :] * Y[None, :, :]).reshape(-1, X.shape[1])


def wls(Y2, M2, Z, reg=1e-5):
    """逐行解加权最小二乘，只在观测元素上拟合。"""
    R = Z.shape[1]
    out = np.zeros((Y2.shape[0], R))
    eye = np.eye(R) * reg
    for i in range(Y2.shape[0]):
        idx = np.flatnonzero(M2[i])
        if len(idx) == 0:
            continue
        Zi = Z[idx]
        out[i] = np.linalg.solve(Zi.T @ Zi + eye, Zi.T @ Y2[i, idx])
    return out


def fit_cp(Yobs, mask, R=R_FIT, n_init=ALS_N_INIT, max_iter=ALS_MAX_ITER,
           seed=SEED):
    """带缺失掩码的 CP-ALS；多初值选训练误差最小者，并统计局部解比例。"""
    best = None
    finals = []
    iters_list = []
    for k in range(n_init):
        rng = np.random.default_rng(seed + 101 * k)
        A = rng.standard_normal((N, R)) * 0.3
        B = rng.standard_normal((D, R)) * 0.3
        C = rng.standard_normal((H, R)) * 0.3
        Ym = Yobs.reshape(N, D * H)
        Mm = mask.reshape(N, D * H)
        prev = np.inf
        iters = 0
        for it in range(max_iter):
            iters = it + 1
            A = wls(Ym, Mm, kr(B, C))
            Yd = np.transpose(Yobs, (1, 0, 2)).reshape(D, N * H)
            Md = np.transpose(mask, (1, 0, 2)).reshape(D, N * H)
            B = wls(Yd, Md, kr(A, C))
            Yh = np.transpose(Yobs, (2, 0, 1)).reshape(H, N * D)
            Mh = np.transpose(mask, (2, 0, 1)).reshape(H, N * D)
            C = wls(Yh, Mh, kr(A, B))
            # 列归一化，消除 CP 的尺度不定性
            for r in range(R):
                nb, nc = np.linalg.norm(B[:, r]), np.linalg.norm(C[:, r])
                if nb > 1e-12 and nc > 1e-12:
                    A[:, r] *= nb * nc
                    B[:, r] /= nb
                    C[:, r] /= nc
            Yhat = cp_tensor(A, B, C)
            mse = float(np.mean((Yhat[mask] - Yobs[mask]) ** 2))
            if abs(prev - mse) < 1e-6 * max(prev, 1e-8):
                break
            prev = mse
        finals.append(prev)
        iters_list.append(iters)
        if best is None or prev < best["train_mse"]:
            best = {"Yhat": Yhat, "train_mse": prev, "iters": iters,
                    "A": A, "B": B, "C": C}
    finals = np.array(finals)
    best["swamp_rate"] = float(np.mean(finals > best["train_mse"] * 1.10))
    best["median_iters"] = float(np.median(iters_list))
    best["init_mses"] = [float(v) for v in finals]
    return best


def als_matrix(Y2, M2, R, n_iter=SVD_ITER):
    """矩阵低秩分解：与 CP-ALS 一样，只在观测元素上做加权最小二乘。

    两者唯一的区别是结构假设（矩阵低秩 vs 三向可分离），优化器完全同源，
    这样比较才不是「拿一个弱优化器当靶子」。用零填充 SVD 做确定性好初值。
    """
    nrow, ncol = Y2.shape
    rows = [np.flatnonzero(M2[i]) for i in range(nrow)]
    cols = [np.flatnonzero(M2[:, j]) for j in range(ncol)]
    U0, s, Vt = np.linalg.svd(np.where(M2, Y2, 0.0), full_matrices=False)
    root_s = np.sqrt(np.maximum(s[:R], 1e-12))
    U = U0[:, :R] * root_s
    V = Vt[:R].T * root_s
    eye = np.eye(R) * 1e-4
    for _ in range(n_iter):
        for i in range(nrow):
            idx = rows[i]
            Vi = V[idx]
            U[i] = np.linalg.solve(Vi.T @ Vi + eye, Vi.T @ Y2[i, idx])
        for j in range(ncol):
            idx = cols[j]
            Ui = U[idx]
            V[j] = np.linalg.solve(Ui.T @ Ui + eye, Ui.T @ Y2[idx, j])
    X = U @ V.T
    train_mse = float(np.mean((X[M2] - Y2[M2]) ** 2))
    return X, train_mse


def fit_flat(Yobs, mask, R=R_FIT):
    X, tm = als_matrix(Yobs.reshape(N, D * H), mask.reshape(N, D * H), R)
    return {"Yhat": X.reshape(N, D, H), "train_mse": tm}


def fit_hour(Yobs, mask, R=R_FIT):
    Yh = np.empty((N, D, H))
    tms, ns = [], []
    for h in range(H):
        X, tm = als_matrix(Yobs[:, :, h], mask[:, :, h], R)
        Yh[:, :, h] = X
        tms.append(tm)
        ns.append(int(mask[:, :, h].sum()))
    return {"Yhat": Yh, "train_mse": float(np.average(tms, weights=ns))}


def metrics(Yhat, truth, noise_sd, hold, weights=None):
    err = (Yhat[hold] - truth[hold]) ** 2
    out = {"mse": float(np.mean(err))}
    if weights is not None:
        out["weighted_mse"] = float(np.mean(err * weights[hold]))
    out["noise_floor"] = float(noise_sd ** 2)
    return out


def gmv_vol(Yhat, sigma_eval, reg=1e-3):
    """用去噪面板的日均值估协方差，算 GMV 权重在真协方差下的年化波动。"""
    X = Yhat.mean(axis=2)                       # (N, D)
    S = np.cov(X) + np.eye(N) * reg
    ones = np.ones(N)
    w = np.linalg.solve(S, ones)
    w /= ones @ w
    return float(np.sqrt(252.0 * w @ sigma_eval @ w))


def oracle_vol(sigma_eval):
    """真协方差下的精确最小方差权重（不加正则，它是所有 w 的下界）。"""
    ones = np.ones(N)
    w = np.linalg.solve(sigma_eval, ones)
    w /= ones @ w
    return float(np.sqrt(252.0 * w @ sigma_eval @ w))


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in ["Heiti TC", "PingFang SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
        try:
            plt.rcParams["font.family"] = [f]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    A, B, C = make_components()
    Y0 = cp_tensor(A, B, C)
    rng = np.random.default_rng(SEED + 7)
    E = rng.standard_normal((N, D, H))       # 所有扫描共用同一张噪声表
    hold, mcar = make_masks(SEED + 8)
    w_vol = np.abs(Y0) / np.mean(np.abs(Y0))
    R = {"settings": {"N": N, "D": D, "H": H, "R_true": R_TRUE,
                      "R_fit": R_FIT, "holdout_frac": HOLDOUT_FRAC,
                      "missing_frac": MISS_FRAC, "seed": SEED,
                      "n_train_obs": int(mcar.sum()),
                      "n_holdout": int(hold.sum()),
                      "cp_params": int(R_FIT * (N + D + H)),
                      "flat_params": int(R_FIT * (N + D * H)),
                      "hour_params": int(H * R_FIT * (N + D)),
                      "signal_std": float(np.std(Y0)),
                      "snr_at_main_sigma": float(np.std(Y0) / SIGMA_MAIN)}}

    def dataset(sigma, gamma=0.0):
        truth = Y0 + hour_specific(SEED + 55, gamma)
        Yobs = np.where(mcar, truth + sigma * E, 0.0)
        return truth, Yobs

    # ---------- 1) 噪声水平扫描 ----------
    R["sigma_scan"] = []
    for s in SIGMAS:
        truth, Yobs = dataset(s)
        cp = fit_cp(Yobs, mcar)
        fl = fit_flat(Yobs, mcar)
        ho = fit_hour(Yobs, mcar)
        row = {"sigma": s,
               "cp": metrics(cp["Yhat"], truth, s, hold)["mse"],
               "flat": metrics(fl["Yhat"], truth, s, hold)["mse"],
               "hour": metrics(ho["Yhat"], truth, s, hold)["mse"],
               "noise_floor": float(s ** 2),
               "cp_train": cp["train_mse"],
               "flat_train": fl["train_mse"],
               "hour_train": ho["train_mse"],
               "cp_swamp": cp["swamp_rate"]}
        row["gain_vs_flat"] = float(1.0 - row["cp"] / row["flat"])
        R["sigma_scan"].append(row)

    # ---------- 2) 秩误设扫描 ----------
    R["rank_scan"] = []
    for rh in R_HATS:
        truth, Yobs = dataset(SIGMA_MAIN)
        cp = fit_cp(Yobs, mcar, R=rh)
        fl = fit_flat(Yobs, mcar, R=rh)
        R["rank_scan"].append({
            "r_hat": rh,
            "cp": metrics(cp["Yhat"], truth, SIGMA_MAIN, hold)["mse"],
            "flat": metrics(fl["Yhat"], truth, SIGMA_MAIN, hold)["mse"],
            "cp_train_mse": cp["train_mse"],
            "cp_swamp": cp["swamp_rate"],
            "cp_iters": cp["iters"],
        })

    # ---------- 3) 结构错配扫描 ----------
    R["misspec_scan"] = []
    for g in GAMMAS:
        truth, Yobs = dataset(SIGMA_MAIN, gamma=g)
        cp = fit_cp(Yobs, mcar)
        fl = fit_flat(Yobs, mcar)
        ho = fit_hour(Yobs, mcar)
        var_truth = float(np.var(truth[hold]))
        row = {
            "gamma": g,
            "var_truth": var_truth,
            "cp": metrics(cp["Yhat"], truth, SIGMA_MAIN, hold)["mse"],
            "flat": metrics(fl["Yhat"], truth, SIGMA_MAIN, hold)["mse"],
            "hour": metrics(ho["Yhat"], truth, SIGMA_MAIN, hold)["mse"],
        }
        for k in ("cp", "flat", "hour"):
            row[f"rel_{k}"] = float(row[k] / var_truth)
        row["cp_over_hour"] = float(row["cp"] / row["hour"])
        R["misspec_scan"].append(row)

    # ---------- 4) MNAR 缺失与评估偏差 ----------
    truth, Yobs_mcar = dataset(SIGMA_MAIN)
    mnar = mnar_mask(truth, hold, SEED + 9)
    Yobs_mnar = np.where(mnar, truth + SIGMA_MAIN * E, 0.0)
    dropped = (~mnar) & (~hold)
    R["missing"] = {"mcar_n": int(mcar.sum()), "mnar_n": int(mnar.sum()),
                    "mnar_mean_abs_truth_dropped": float(
                        np.mean(np.abs(truth)[dropped])),
                    "mnar_mean_abs_truth_kept": float(
                        np.mean(np.abs(truth)[mnar])),
                    "mcar_mean_abs_truth_kept": float(
                        np.mean(np.abs(truth)[mcar]))}
    cp_m = fit_cp(Yobs_mcar, mcar)
    cp_n = fit_cp(Yobs_mnar, mnar)
    ho_n = fit_hour(Yobs_mnar, mnar)
    R["missing"]["mcar_cp"] = metrics(cp_m["Yhat"], truth, SIGMA_MAIN, hold,
                                      w_vol)
    R["missing"]["mnar_cp"] = metrics(cp_n["Yhat"], truth, SIGMA_MAIN, hold,
                                      w_vol)
    R["missing"]["mnar_hour"] = metrics(ho_n["Yhat"], truth, SIGMA_MAIN, hold,
                                        w_vol)

    # ---------- 5) 下游 GMV ----------
    # 日度聚合把噪声方差压成 sigma^2/H；信号协方差是秩 3 的，
    # 必须补上这一对角项才是满秩的「真实日度收益协方差」。
    sigma_eval = (np.cov(Y0.mean(axis=2))
                  + (SIGMA_MAIN ** 2 / H) * np.eye(N))
    R["gmv"] = {
        "oracle": oracle_vol(sigma_eval),
        "observed": gmv_vol(truth + SIGMA_MAIN * E, sigma_eval),
        "cp": gmv_vol(cp_m["Yhat"], sigma_eval),
        "flat": gmv_vol(fit_flat(Yobs_mcar, mcar)["Yhat"], sigma_eval),
        "hour": gmv_vol(fit_hour(Yobs_mcar, mcar)["Yhat"], sigma_eval),
    }

    with open(f"{OUT}/results.json", "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)

    # ---------- 图 1：真成分与示例切片 ----------
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].plot(A[:, 0], label="r=1"); ax[0, 0].plot(A[:, 1], label="r=2")
    ax[0, 0].plot(A[:, 2], label="r=3")
    ax[0, 0].set_title("资产暴露 A（36 个资产）"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(B[:, 0], label="r=1"); ax[0, 1].plot(B[:, 1], label="r=2")
    ax[0, 1].plot(B[:, 2], label="r=3")
    ax[0, 1].set_title("日度因子 B（AR(1)，180 天）"); ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(C[:, 0], label="水平"); ax[1, 0].plot(C[:, 1], label="U 型")
    ax[1, 0].plot(C[:, 2], label="开盘衰减")
    ax[1, 0].set_title("日内形状 C（20 个时段）"); ax[1, 0].legend(fontsize=8)
    im = ax[1, 1].imshow(Y0[:, 40, :], aspect="auto", cmap="RdBu_r")
    ax[1, 1].set_title("第 40 天：资产 × 日内时段切片")
    ax[1, 1].set_xlabel("时段 h"); ax[1, 1].set_ylabel("资产 a")
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_true_components.png")
    plt.close(fig)

    # ---------- 图 2：噪声扫描 ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ss = [x["sigma"] for x in R["sigma_scan"]]
    for key, label, c in [("cp", "CP-ALS", "#2b6cb0"),
                          ("flat", "展平 ALS", "#dd6b20"),
                          ("hour", "逐时段 ALS", "#805ad5")]:
        ax[0].plot(ss, [x[key] for x in R["sigma_scan"]], "o-", color=c,
                   label=label)
    ax[0].plot(ss, [x["noise_floor"] for x in R["sigma_scan"]], "k--",
               label="噪声下限 sigma^2")
    ax[0].set_xlabel("噪声标准差 sigma"); ax[0].set_ylabel("留出去噪 MSE")
    ax[0].set_title("真 CP 结构：结构假设最省参数也最准")
    ax[0].legend(fontsize=8)
    main = [x for x in R["sigma_scan"] if abs(x["sigma"] - SIGMA_MAIN) < 1e-9][0]
    xp = np.arange(3)
    ax[1].bar(xp - 0.2, [main["cp_train"], main["flat_train"],
                         main["hour_train"]], 0.4,
              color="#c53030", label="训练 MSE")
    ax[1].bar(xp + 0.2, [main["cp"], main["flat"], main["hour"]], 0.4,
              color="#2b6cb0", label="留出去噪 MSE")
    ax[1].set_yscale("log")
    ax[1].set_xticks(xp)
    ax[1].set_xticklabels(["CP-ALS", "展平 ALS", "逐时段 ALS"], fontsize=8)
    ax[1].set_ylabel("MSE（对数轴）")
    ax[1].set_title("sigma=0.30：训练误差更低，泛化反而更差")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_noise_scan.png")
    plt.close(fig)

    # ---------- 图 3：秩误设 ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    rr = [x["r_hat"] for x in R["rank_scan"]]
    ax[0].plot(rr, [x["cp"] for x in R["rank_scan"]], "o-", color="#2b6cb0",
               label="CP-ALS")
    ax[0].plot(rr, [x["flat"] for x in R["rank_scan"]], "s-", color="#dd6b20",
               label="展平 ALS")
    ax[0].axvline(R_TRUE, ls="--", color="0.4", label="真实秩 = 3")
    ax[0].set_xlabel("设定秩 R_hat"); ax[0].set_ylabel("留出去噪 MSE")
    ax[0].set_title("真秩 3 不是最优拟合秩：弱成分被噪声淹没")
    ax[0].legend(fontsize=8)
    ax2 = ax[1].twinx()
    ax[1].plot(rr, [x["cp_train_mse"] for x in R["rank_scan"]], "s-",
               color="#c53030", label="训练 MSE")
    ax[1].set_xlabel("设定秩 R_hat"); ax[1].set_ylabel("训练 MSE",
                                                   color="#c53030")
    ax2.plot(rr, [x["cp_iters"] for x in R["rank_scan"]], "o-",
             color="#2f855a", label="收敛迭代数")
    ax2.set_ylabel("迭代数（上限 100）", color="#2f855a"); ax2.grid(False)
    ax[1].set_title("秩越大：训练误差继续降，迭代数撞上限")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_rank_misspec.png")
    plt.close(fig)

    # ---------- 图 4：结构错配 ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    gg = [x["gamma"] for x in R["misspec_scan"]]
    for key, label, c in [("rel_cp", "CP-ALS（708 参数）", "#2b6cb0"),
                          ("rel_flat", "展平 ALS", "#dd6b20"),
                          ("rel_hour", "逐时段 ALS（12960 参数）", "#805ad5")]:
        ax.plot(gg, [x[key] for x in R["misspec_scan"]], "o-", color=c,
                label=label)
    ax.set_yscale("log")
    ax.set_xlabel("时段特异交互强度 gamma")
    ax.set_ylabel("归一化误差 MSE / Var(真值)（对数轴）")
    ax.set_title("同一套 CP：gamma=0 赢 28 倍，gamma=2 输 34 倍")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_structure_misspec.png")
    plt.close(fig)

    # ---------- 图 5：MNAR 与下游 ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    mm = R["missing"]
    groups = ["CP / 随机缺失", "CP / 高波动缺失", "逐时段 / 高波动缺失"]
    plain = [mm["mcar_cp"]["mse"], mm["mnar_cp"]["mse"],
             mm["mnar_hour"]["mse"]]
    weighted = [mm["mcar_cp"]["weighted_mse"], mm["mnar_cp"]["weighted_mse"],
                mm["mnar_hour"]["weighted_mse"]]
    xp = np.arange(3)
    ax[0].bar(xp - 0.2, plain, 0.4, color="#2b6cb0", label="随机留出 MSE")
    ax[0].bar(xp + 0.2, weighted, 0.4, color="#c53030", label="波动加权 MSE")
    ax[0].set_xticks(xp); ax[0].set_xticklabels(groups, fontsize=8)
    ax[0].set_ylabel("去噪误差"); ax[0].set_title("随机留出会低估高波动处的误差")
    ax[0].legend(fontsize=8)
    keys = ["oracle", "observed", "cp", "flat", "hour"]
    ax[1].bar(np.arange(len(keys)), [R["gmv"][k] for k in keys],
              color=["#2f855a", "#c53030", "#2b6cb0", "#dd6b20", "#805ad5"])
    ax[1].set_xticks(np.arange(len(keys)))
    ax[1].set_xticklabels(["真值", "含噪", "CP", "展平", "逐时段"], fontsize=8)
    ax[1].set_ylabel("GMV 样本外年化波动")
    ax[1].set_title("元素级 28 倍差距，到 GMV 只剩约 12% 波动差")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_mnar_and_gmv.png")
    plt.close(fig)

    print("=== 噪声扫描（留出去噪 MSE） ===")
    for x in R["sigma_scan"]:
        print(f"  sigma={x['sigma']:.2f}  cp={x['cp']:.5f}  "
              f"flat={x['flat']:.5f}  hour={x['hour']:.5f}  "
              f"gain={100*x['gain_vs_flat']:.1f}%")
    print("=== 秩误设 ===")
    for x in R["rank_scan"]:
        print(f"  R_hat={x['r_hat']}  cp={x['cp']:.5f}  flat={x['flat']:.5f}  "
              f"swamp={100*x['cp_swamp']:.0f}%  iters={x['cp_iters']}")
    print("=== 训练/留出误差背离 (sigma=0.30) ===")
    m30 = [x for x in R["sigma_scan"] if abs(x["sigma"] - SIGMA_MAIN) < 1e-9][0]
    for k in ("cp", "flat", "hour"):
        print(f"  {k:6s} train={m30[k+'_train']:.5f}  holdout={m30[k]:.5f}")
    print("=== 结构错配（含归一化误差） ===")
    for x in R["misspec_scan"]:
        print(f"  gamma={x['gamma']:.1f}  cp={x['cp']:.5f}  "
              f"flat={x['flat']:.5f}  hour={x['hour']:.5f}  "
              f"rel_cp={x['rel_cp']:.4f}  rel_hour={x['rel_hour']:.4f}  "
              f"cp/hour={x['cp_over_hour']:.3f}")
    print("=== MNAR 评估偏差 ===")
    for k in ("mcar_cp", "mnar_cp", "mnar_hour"):
        v = mm[k]
        print(f"  {k:12s} mse={v['mse']:.5f}  weighted={v['weighted_mse']:.5f}")
    print(f"  dropped |truth|={mm['mnar_mean_abs_truth_dropped']:.4f}  "
          f"kept |truth|={mm['mnar_mean_abs_truth_kept']:.4f}")
    print("=== GMV 样本外年化波动 ===")
    for k, v in R["gmv"].items():
        print(f"  {k:10s} {v:.6f}")


if __name__ == "__main__":
    main()
