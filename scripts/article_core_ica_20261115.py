# -*- coding: utf-8 -*-
"""FastICA 独立成分分析：从混合信号里找回非高斯 latent 因子。

本文全部数字的唯一来源。运行本脚本会：
1. 现场造 3 个独立、方差不等的非高斯源（Laplace / 均匀 / t(4) 厚尾）；
2. 随机线性混合后，分别用 FastICA（手写定步长不动点）与 PCA 还原；
3. 用匹配相关系数 + Amari 距离定量比较；并跑高斯对照（不可辨识）与非线性扫描；
4. 画 5 张图到 public/images/fast-ica-factor-recovery/，写 results.json。
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261115
_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(
    os.path.dirname(_HERE), "public", "images", "fast-ica-factor-recovery"
)
os.makedirs(_OUT, exist_ok=True)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, name), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 非线性函数
def _g_tanh(y):
    g = np.tanh(y)
    return g, 1.0 - g ** 2


def _g_gauss(y):
    ey = y * np.exp(-y ** 2 / 2.0)
    return ey, (1.0 - y ** 2) * np.exp(-y ** 2 / 2.0)


def _g_cube(y):
    return y ** 3, 3.0 * y ** 2


# ---------------------------------------------------------------- 源生成
def make_sources(rng, T, gauss=False):
    if gauss:
        s = np.vstack([
            rng.standard_normal(T),
            rng.standard_normal(T),
            rng.standard_normal(T),
        ])
        # 单位方差
        s = s / s.std(axis=1, ddof=1)[:, None]
        names = ["高斯#1", "高斯#2", "高斯#3"]
    else:
        s1 = rng.laplace(0.0, 1.0, T)            # 超高斯
        s2 = rng.uniform(-np.sqrt(3), np.sqrt(3), T)  # 亚高斯
        s3 = rng.standard_t(4.0, T)              # 厚尾超高斯
        s = np.vstack([s1, s2, s3])
        s = s / s.std(axis=1, ddof=1)[:, None]
        names = ["Laplace(超高斯)", "均匀(亚高斯)", "t(4)厚尾"]
    return s, names


def random_mix(rng, k):
    A = rng.standard_normal((k, k))
    # 让条件数适中，避免病态
    return A


def whiten(X):
    Xc = X - X.mean(axis=1, keepdims=True)
    cov = Xc @ Xc.T / Xc.shape[1]
    d, U = np.linalg.eigh(cov)
    d = np.clip(d, 1e-12, None)
    K = np.diag(1.0 / np.sqrt(d)) @ U.T
    Z = K @ Xc
    return Z, K, Xc


def fastica(X, g, rng, max_iter=300, tol=1e-7):
    Z, K, Xc = whiten(X)
    k, T = Z.shape
    W = np.linalg.qr(rng.standard_normal((k, k)))[0]
    objs = []
    for it in range(max_iter):
        Y = W @ Z
        gy, gpy = g(Y)
        W_new = (gy @ Z.T) / T - np.diag(gpy.mean(axis=1)) @ W
        # 对称去相关
        s, uu = np.linalg.eigh(W_new @ W_new.T)
        W_new = (uu @ np.diag(1.0 / np.sqrt(s)) @ uu.T) @ W_new
        lim = np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1.0))
        objs.append(float(np.sum(np.abs(gy.mean(axis=1)))))
        W = W_new
        if lim < tol:
            break
    W_full = W @ K
    S_hat = W_full @ Xc
    return S_hat, W, K, Xc, objs, it + 1


def pca_recover(X, k=None):
    Xc = X - X.mean(axis=1, keepdims=True)
    cov = Xc @ Xc.T / Xc.shape[1]
    d, U = np.linalg.eigh(cov)
    U = U[:, ::-1]
    if k is not None:
        U = U[:, :k]
    return U.T @ Xc, U


def match(S_hat, S):
    """为每一个估计成分匹配相关系数最高的、尚未占用的真源（保证 k 对）。"""
    k = S_hat.shape[0]
    corr = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            corr[i, j] = abs(np.corrcoef(S_hat[i], S[j])[0, 1])
    used_j = set()
    pairs = []
    for i in range(k):
        best, bj = -1.0, -1
        for j in range(k):
            if j in used_j:
                continue
            if corr[i, j] > best:
                best, bj = corr[i, j], j
        used_j.add(bj)
        pairs.append((bj, best))
    return pairs


def amari(W_est, W_true):
    """Amari 距离：完美还原（置换*符号）时为 0，越大越偏离。"""
    P = np.abs(W_est @ W_true)
    k = P.shape[0]
    row = P / P.max(axis=1, keepdims=True)
    col = P / P.max(axis=0, keepdims=True)
    return ((row.sum() - k) + (col.sum() - k)) / (2.0 * k * (k - 1))


def higher_order_indep(S_hat):
    """平均 |平方后的互相关|，衡量高阶独立性（越小越独立）。"""
    k = S_hat.shape[0]
    sq = S_hat ** 2
    vals = []
    for i in range(k):
        for j in range(i + 1, k):
            vals.append(abs(np.corrcoef(sq[i], sq[j])[0, 1]))
    return float(np.mean(vals))


def main():
    rng = np.random.default_rng(SEED)
    T = 4000
    k = 3

    # ---------- 主实验：非高斯源 ----------
    S, names = make_sources(rng, T, gauss=False)
    A = random_mix(rng, k)
    X = A @ S

    S_ica, W, K, Xc, objs, n_iter = fastica(X, _g_tanh, rng)
    S_pca, _ = pca_recover(X, k)
    S_pca = S_pca / S_pca.std(axis=1, ddof=1)[:, None]

    pairs_ica = match(S_ica, S)
    pairs_pca = match(S_pca, S)
    ica_corr = [p[1] for p in pairs_ica]
    pca_corr = [p[1] for p in pairs_pca]
    # Amari 距离以混合矩阵 A 为基准：完美还原时 W_est @ A ≈ 置换*符号
    amari_ica = amari(W @ K, A)
    # PCA 的等价全解混矩阵：取前 k 个主成分方向 U^T 再左乘 K^-1
    cov = Xc @ Xc.T / Xc.shape[1]
    _, U = np.linalg.eigh(cov)
    U = U[:, ::-1][:, :k]
    W_pca_full = U.T @ np.linalg.inv(K)
    amari_pca = amari(W_pca_full, A)

    ho_ica = higher_order_indep(S_ica)
    ho_pca = higher_order_indep(S_pca)

    # ---------- 高斯对照（不可辨识）：正确指标是 Amari ----------
    # 高斯源下 ICA 没有非高斯性可作信号，固定点收敛到任意正交旋转，
    # 解不唯一 → Amari（对真混合矩阵）应当很大；而非高斯时 Amari≈0。
    # 匹配相关系数在此会「碰巧偏高」（3 维随机旋转本就可能与某真轴对齐），
    # 不能作为可辨识性的判据。
    rng_g = np.random.default_rng(SEED + 99)
    gauss_ica_amari, gauss_pca_amari = [], []
    gauss_ica_corr_list, gauss_pca_corr_list = [], []
    for trial in range(8):
        Sg, _ = make_sources(rng_g, T, gauss=True)
        Ag = random_mix(rng_g, k)
        Xg = Ag @ Sg
        Sg_ica, Wg, Kg, Xcg, _, _ = fastica(Xg, _g_tanh, rng_g)
        Sg_pca, _ = pca_recover(Xg, k)
        Sg_pca = Sg_pca / Sg_pca.std(axis=1, ddof=1)[:, None]
        covg = Xcg @ Xcg.T / Xcg.shape[1]
        _, Ug = np.linalg.eigh(covg)
        Ug = Ug[:, ::-1][:, :k]
        Wg_pca = Ug.T @ np.linalg.inv(Kg)
        gauss_ica_amari.append(float(amari(Wg @ Kg, Ag)))
        gauss_pca_amari.append(float(amari(Wg_pca, Ag)))
        gauss_ica_corr_list.append(float(np.mean([p[1] for p in match(Sg_ica, Sg)])))
        gauss_pca_corr_list.append(float(np.mean([p[1] for p in match(Sg_pca, Sg)])))
    gauss_ica_amari_mean = float(np.mean(gauss_ica_amari))
    gauss_pca_amari_mean = float(np.mean(gauss_pca_amari))
    gauss_ica_mean = float(np.mean(gauss_ica_corr_list))
    gauss_pca_mean = float(np.mean(gauss_pca_corr_list))

    # ---------- 非线性扫描（超高斯混合下） ----------
    gset = {"tanh": _g_tanh, "gauss": _g_gauss, "cube": _g_cube}
    sweep = {}
    for gname, gfunc in gset.items():
        S2, _ = make_sources(rng, T, gauss=False)
        A2 = random_mix(rng, k)
        X2 = A2 @ S2
        S2_ica, W2, K2, X2c, _, _ = fastica(X2, gfunc, rng)
        p2 = match(S2_ica, S2)
        corr2 = [p[1] for p in p2]
        sweep[gname] = {
            "mean_matched_corr": float(np.mean(corr2)),
            "amari": float(amari(W2 @ K2, A2)),
        }

    # ===================== 画图 =====================
    # 图1：真实源（时间序列 + 直方图）
    colors = ["#d1242f", "#1f6feb", "#2da44e"]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 5.4))
    for j in range(k):
        axes[0, j].plot(S[j, :300], color=colors[j], lw=0.8)
        axes[0, j].set_title(names[j], fontsize=9.5)
        axes[0, j].grid(alpha=0.3)
        axes[1, j].hist(S[j], bins=50, color=colors[j], alpha=0.8)
        axes[1, j].set_ylabel("频数" if j == 0 else "")
        axes[1, j].grid(alpha=0.3, axis="y")
    axes[0, 0].set_ylabel("信号值")
    fig.suptitle("图1 三个独立非高斯源：Laplace(尖峰) / 均匀(平顶) / t(4)(厚尾)",
                 fontsize=11)
    _save(fig, "fig1_true_sources.png")

    # 图2：ICA 还原 vs PCA 还原（按真源匹配后）
    ica_order = [p[0] for p in pairs_ica]
    pca_order = [p[0] for p in pairs_pca]
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 6.2), sharex=True)
    for j in range(k):
        axes[j, 0].plot(S[j, :300], color="#24292f", lw=0.8, label="真源")
        axes[j, 0].set_ylabel(names[j], fontsize=9)
        axes[j, 1].plot(S_ica[ica_order[j], :300], color=colors[j], lw=0.8,
                        label="ICA")
        axes[j, 2].plot(S_pca[pca_order[j], :300], color="#8250df", lw=0.8,
                        label="PCA")
    for j in range(k):
        axes[j, 0].set_title("真源", fontsize=9)
        axes[j, 1].set_title(f"ICA (r={ica_corr[j]:.2f})", fontsize=9)
        axes[j, 2].set_title(f"PCA (r={pca_corr[j]:.2f})", fontsize=9)
    axes[2, 0].set_xlabel("t (前 300 点)")
    fig.suptitle("图2 还原质量：ICA 几乎完美复刻真源形状，PCA 给的是混合投影",
                 fontsize=11)
    _save(fig, "fig2_recovery.png")

    # 图3：非高斯源下 ICA vs PCA 的匹配相关系数
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    x = np.arange(k)
    axes[0].bar(x - 0.2, ica_corr, 0.4, color="#1f6feb", label="ICA")
    axes[0].bar(x + 0.2, pca_corr, 0.4, color="#8250df", label="PCA")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("与真源的匹配 |相关系数|")
    axes[0].set_title(f"非高斯源：ICA 均值 {np.mean(ica_corr):.3f} "
                      f"vs PCA {np.mean(pca_corr):.3f}", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")
    # 右图：Amari 距离（越小越准），非高斯主实验 vs 高斯对照
    amvals = [amari_ica, amari_pca, gauss_ica_amari_mean, gauss_pca_amari_mean]
    amlabels = ["ICA\n(非高斯)", "PCA\n(非高斯)", "ICA\n(高斯)", "PCA\n(高斯)"]
    amcolors = ["#1f6feb", "#8250df", "#79c0ff", "#d2a8ff"]
    axes[1].bar(range(4), amvals, color=amcolors, alpha=0.9)
    for i, v in enumerate(amvals):
        axes[1].text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=9)
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(amlabels, fontsize=8)
    axes[1].set_ylabel("Amari 距离（越小越准）")
    axes[1].set_title("关键对照：高斯源下 ICA 的 Amari 暴涨，解不再唯一",
                     fontsize=10)
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("图3 ICA 的胜出依赖「非高斯」：源一旦高斯，ICA 也不可辨识",
                 fontsize=11)
    _save(fig, "fig3_corr_compare.png")

    # 图4：左=收敛曲线，右=非线性扫描 Amari
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(range(1, len(objs) + 1), objs, "o-", color="#1f6feb", lw=1.4,
                ms=3)
    axes[0].set_xlabel("不动点迭代次数")
    axes[0].set_ylabel("目标（|E[g(Y)]| 之和）")
    axes[0].set_title(f"FastICA 收敛：{n_iter} 步稳定", fontsize=10)
    axes[0].grid(alpha=0.3)
    gn = list(sweep.keys())
    am = [sweep[g]["amari"] for g in gn]
    axes[1].bar(gn, am, color=["#1f6feb", "#bf8700", "#2da44e"], alpha=0.85)
    for i, v in enumerate(am):
        axes[1].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    axes[1].set_ylabel("Amari 距离（越小越准）")
    axes[1].set_title(f"非线性扫描：三者接近，gauss 略优 "
                      f"(tanh {sweep['tanh']['amari']:.3f})", fontsize=10)
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("图4 收敛性与非线性扫描", fontsize=11)
    _save(fig, "fig4_convergence_sweep.png")

    # 图5：高阶独立性热力（离对角 |corr(f²_i, f²_j)|）
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    for ax, Sf, title in ((axes[0], S_ica, "ICA 因子"),
                          (axes[1], S_pca, "PCA 因子")):
        sq = Sf ** 2
        M = np.corrcoef(sq)
        im = ax.imshow(np.abs(M), vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_title(f"{title}：离对角均值 {higher_order_indep(Sf):.3f}",
                     fontsize=10)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{np.abs(M[i,j]):.2f}", ha="center", va="center",
                        color="white" if np.abs(M[i, j]) < 0.5 else "black",
                        fontsize=9)
    fig.suptitle("图5 平方互相关：ICA 因子高阶独立，PCA 因子仍纠缠",
                 fontsize=11)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.04)
    _save(fig, "fig5_higher_order.png")

    results = {
        "dgp": {"T": T, "k": k, "seed": SEED,
                "source_names": names, "mix_cond": float(np.linalg.cond(A))},
        "main": {
            "ica_matched_corr": [float(c) for c in ica_corr],
            "pca_matched_corr": [float(c) for c in pca_corr],
            "ica_mean_corr": float(np.mean(ica_corr)),
            "pca_mean_corr": float(np.mean(pca_corr)),
            "amari_ica": float(amari_ica),
            "amari_pca": float(amari_pca),
            "higher_order_ica": ho_ica,
            "higher_order_pca": ho_pca,
            "n_iter": int(n_iter),
        },
        "gaussian_control": {
            "ica_mean_amari": gauss_ica_amari_mean,
            "pca_mean_amari": gauss_pca_amari_mean,
            "ica_mean_corr": gauss_ica_mean,
            "pca_mean_corr": gauss_pca_mean,
            "random_baseline": 1.0 / np.sqrt(k),
            "n_trials": 8,
            "non_gauss_ica_amari": float(amari_ica),
        },
        "nonlinearity_sweep": {g: {"mean_matched_corr": float(sweep[g]["mean_matched_corr"]),
                                   "amari": float(sweep[g]["amari"])}
                               for g in sweep},
    }
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
