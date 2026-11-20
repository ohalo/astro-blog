# -*- coding utf-8 -*-
"""最大均值差异 MMD 两样本检验：带宽决定功效，同均值同方差的形状漂移才是它的主场。

本脚本是文章全部数字的唯一来源。运行它会：
1. 在真值已知的五个场景上对决 Welch-t / 两样本 KS / MMD(RBF 三带宽 + median heuristic + IMQ)；
2. 场景包含 H0 校准、纯均值漂移、纯方差漂移、同均值同方差换族(Laplace)、尾部加厚(t3)；
3. 用置换检验给 MMD 算 p 值，报告 size 与各备择 power；
4. 扫描 RBF 带宽看 power 的 U 型曲线，扫描样本量看一致性；
5. 画 5 张图到 public/images/mmd-two-sample-distribution-drift/，统计写入同目录 results.json。
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261120
N_PER = 120          # 每样本量
REPS = 250           # 蒙特卡洛重复
B_PERM = 99          # 置换次数
ALPHA = 0.05
SIGMAS = [0.5, 1.5, 4.0]   # 三档 RBF 带宽
TAIL_SCALE = 1.0 / np.sqrt(3.0)   # t(3) 缩放至单位方差

_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "public", "images", "mmd-two-sample-distribution-drift",
)
os.makedirs(_OUT, exist_ok=True)


def rbf_k(a, b, sigma):
    d2 = cdist(a, b, "sqeuclidean")
    return np.exp(-d2 / (2.0 * sigma ** 2))


def imq_k(a, b, c=1.0):
    d2 = cdist(a, b, "sqeuclidean")
    return 1.0 / (1.0 + d2 / c)


def mmd_u_from_K(K, m, n):
    """无偏 MMD^2：K 为合并 (m+n) 方阵，前 m 行是 X。"""
    A = K[:m, :m].sum() - np.trace(K[:m, :m])
    Bv = K[m:, m:].sum() - np.trace(K[m:, m:])
    C = K[:m, m:].sum()
    return A / (m * (m - 1)) + Bv / (n * (n - 1)) - 2.0 * C / (m * n)


def perm_test(x, y, kern, n_perm, rng):
    """置换检验：合并后随机重排标签，用同一 K 的索引重算 MMD^2。返回 (MMD², p)。"""
    m, n = len(x), len(y)
    N = m + n
    z = np.vstack([x, y])
    K = kern(z, z)
    obs = mmd_u_from_K(K, m, n)
    perms = np.argsort(rng.random((n_perm, N)), axis=1)
    idx = perms[:, :m]
    idx2 = perms[:, m:]
    A = K[idx[:, :, None], idx[:, None, :]].sum(axis=(1, 2)) - K[idx, idx].sum(axis=1)
    Bv = K[idx2[:, :, None], idx2[:, None, :]].sum(axis=(1, 2)) - K[idx2, idx2].sum(axis=1)
    # total_offdiag - A - Bv = 2*cross_sum，故 -2*cross/(mn) = -(total_offdiag - A - Bv)/(mn)
    null = A / (m * (m - 1)) + Bv / (n * (n - 1)) - (K.sum() - np.trace(K) - A - Bv) / (m * n)
    p = (1.0 + np.sum(null >= obs - 1e-12)) / (n_perm + 1.0)
    return float(obs), float(p)


def draw(rng, scenario, n):
    if scenario == "h0":
        return rng.standard_normal(n).reshape(-1, 1), rng.standard_normal(n).reshape(-1, 1)
    if scenario == "mean":
        return rng.standard_normal(n).reshape(-1, 1), (rng.standard_normal(n) + 0.6).reshape(-1, 1)
    if scenario == "var":
        return rng.standard_normal(n).reshape(-1, 1), (rng.standard_normal(n) * 2.2).reshape(-1, 1)
    if scenario == "shape":
        # 同为均值 0、方差 1：Laplace 尺度 1/sqrt(2)
        return rng.standard_normal(n).reshape(-1, 1), (rng.laplace(0, 1 / np.sqrt(2), n)).reshape(-1, 1)
    if scenario == "tail":
        return rng.standard_normal(n).reshape(-1, 1), (rng.standard_t(3, n) * TAIL_SCALE).reshape(-1, 1)
    raise ValueError(scenario)


SCENARIOS = ["h0", "mean", "var", "shape", "tail"]


def run_power_table():
    """返回每个场景下各检验的拒绝率；MMD 用固定 σ 与 median heuristic。"""
    rng = np.random.default_rng(SEED)
    methods = ["welch_t", "ks", "mmd_s0.5", "mmd_s1.5", "mmd_s4.0", "mmd_med", "mmd_imq"]
    out = {s: {k: 0 for k in methods} for s in SCENARIOS}
    meds = {s: [] for s in SCENARIOS}
    for s in SCENARIOS:
        for _ in range(REPS):
            x, y = draw(rng, s, N_PER)
            out[s]["welch_t"] += stats.ttest_ind(x.ravel(), y.ravel(), equal_var=False).pvalue <= ALPHA
            out[s]["ks"] += stats.ks_2samp(x.ravel(), y.ravel()).pvalue <= ALPHA
            z = np.vstack([x, y])
            Dmed = np.median(cdist(z, z, "euclidean")[np.triu_indices(len(z), 1)])
            meds[s].append(Dmed)
            for sig in SIGMAS:
                _, p = perm_test(x, y, lambda a, b, sg=sig: rbf_k(a, b, sg), B_PERM, rng)
                out[s][f"mmd_s{sig}"] += p <= ALPHA
            _, p = perm_test(x, y, lambda a, b, sg=Dmed: rbf_k(a, b, sg), B_PERM, rng)
            out[s]["mmd_med"] += p <= ALPHA
            _, p = perm_test(x, y, imq_k, B_PERM, rng)
            out[s]["mmd_imq"] += p <= ALPHA
        for k in methods:
            out[s][k] = out[s][k] / REPS
    med_stat = {s: {"median": float(np.median(meds[s])), "q25": float(np.quantile(meds[s], .25)),
                    "q75": float(np.quantile(meds[s], .75))} for s in SCENARIOS}
    return out, med_stat


def bandwidth_scan():
    """形状漂移场景下，RBF 带宽从 0.1 到 20 扫描 power，展示 U 型。"""
    rng = np.random.default_rng(SEED + 1)
    sigmas = [0.15, 0.25, 0.4, 0.6, 0.9, 1.3, 1.9, 2.8, 4.0, 6.0, 9.0, 14.0, 20.0]
    powers = {s: [] for s in ["shape", "mean", "tail"]}
    for sig in sigmas:
        for s in powers:
            cnt = 0
            for _ in range(REPS):
                x, y = draw(rng, s, N_PER)
                _, p = perm_test(x, y, lambda a, b, sg=sig: rbf_k(a, b, sg), B_PERM, rng)
                cnt += p <= ALPHA
            powers[s].append(cnt / REPS)
    return {"sigmas": sigmas, "power": powers}


def sample_scan():
    """样本量 n 扫描：形状漂移下 MMD(median) / KS / Welch-t 的 power。"""
    rng = np.random.default_rng(SEED + 2)
    ns = [40, 80, 160, 320, 640]
    res = {"n": ns, "mmd": [], "ks": [], "welch": []}
    for n in ns:
        c_mmd = c_ks = c_t = 0
        for _ in range(REPS):
            x, y = draw(rng, "shape", n)
            z = np.vstack([x, y])
            Dmed = np.median(cdist(z, z, "euclidean")[np.triu_indices(len(z), 1)])
            _, p = perm_test(x, y, lambda a, b, sg=Dmed: rbf_k(a, b, sg), min(B_PERM, 99), rng)
            c_mmd += p <= ALPHA
            c_ks += stats.ks_2samp(x.ravel(), y.ravel()).pvalue <= ALPHA
            c_t += stats.ttest_ind(x.ravel(), y.ravel(), equal_var=False).pvalue <= ALPHA
        res["mmd"].append(c_mmd / REPS)
        res["ks"].append(c_ks / REPS)
        res["welch"].append(c_t / REPS)
    return res


def plot1():
    rng = np.random.default_rng(SEED + 3)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.1))
    labels = {"h0": "H0：同分布", "mean": "均值漂移 +0.6", "var": "方差漂移 ×2.2",
              "shape": "同均值方差换族", "tail": "尾部加厚 t(3)"}
    grid = np.linspace(-5, 5, 300)
    for ax, s in zip(axes, SCENARIOS):
        x, y = draw(rng, s, 4000)
        ax.hist(x.ravel(), bins=60, density=True, alpha=.55, color="#2980b9", label="X")
        ax.hist(y.ravel(), bins=60, density=True, alpha=.55, color="#c0392b", label="Y")
        ax.set_title(labels[s], fontsize=10)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=8)
    fig.suptitle("图1：五种真值场景——KS/t 只看均值方差，形状与尾部差异藏在分布里", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .92])
    p = os.path.join(_OUT, "fig1_scenarios.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig1_scenarios.png"


def plot2(power, med):
    methods = ["welch_t", "ks", "mmd_s0.5", "mmd_s1.5", "mmd_s4.0", "mmd_med", "mmd_imq"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    w = .12
    base = np.arange(len(SCENARIOS))
    colors = ["#7f8c8d", "#2980b9", "#e67e22", "#f39c12", "#d35400", "#27ae60", "#8e44ad"]
    for i, (k, c) in enumerate(zip(methods, colors)):
        vals = [power[s][k] for s in SCENARIOS]
        ax.bar(base + (i - 3) * w, vals, w, label=k, color=c)
    ax.axhline(ALPHA, color="k", ls="--", lw=1)
    ax.text(-.45, ALPHA + .012, "5% 显著性", fontsize=8)
    ax.set_xticks(base); ax.set_xticklabels(SCENARIOS)
    ax.set_ylabel("拒绝率（size / power）"); ax.set_ylim(0, 1.05)
    ax.set_title("图2：H0 列越靠近虚线越校准；shape/tail 两列才是 MMD 的主场")
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig2_power_table.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig2_power_table.png"


def plot3(scan):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for s, c in [("shape", "#27ae60"), ("mean", "#c0392b"), ("tail", "#8e44ad")]:
        ax.semilogx(scan["sigmas"], scan["power"][s], "o-", color=c, lw=2, label=s)
    ax.axhline(ALPHA, color="k", ls="--", lw=1)
    ax.set_xlabel("RBF 带宽 σ"); ax.set_ylabel("power（拒绝率）")
    ax.set_title("图3：带宽扫描——形状/尾部差异的 power 先升后降（倒 U），均值漂移却一路走高")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig3_bandwidth.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig3_bandwidth.png"


def plot4(scan):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(scan["n"], scan["mmd"], "o-", color="#27ae60", lw=2, label="MMD(median)")
    ax.plot(scan["n"], scan["ks"], "s-", color="#2980b9", lw=2, label="KS")
    ax.plot(scan["n"], scan["welch"], "^--", color="#c0392b", lw=2, label="Welch-t")
    ax.axhline(ALPHA, color="k", ls=":", lw=1)
    ax.set_xscale("log", base=2); ax.set_xticks(scan["n"]); ax.set_xticklabels(scan["n"])
    ax.set_xlabel("每样本量 n"); ax.set_ylabel("power")
    ax.set_title("图4：同均值同方差的形状漂移——只有 MMD 随样本量爬起来")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig4_sample_size.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig4_sample_size.png"


def plot5():
    rng = np.random.default_rng(SEED + 4)
    z = np.vstack([rng.standard_normal(N_PER), rng.laplace(0, 1/np.sqrt(2), N_PER)])
    K = rbf_k(z, z, 1.5)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(K, cmap="viridis")
    ax.axvline(N_PER - .5, color="red", lw=1.5); ax.axhline(N_PER - .5, color="red", lw=1.5)
    ax.text(N_PER/2, N_PER + 8, "Y (Laplace)", color="white", ha="center", fontsize=9)
    ax.text(3*N_PER/2, N_PER + 8, "Y", color="white", ha="center", fontsize=9)
    ax.text(N_PER/2, N_PER - 8, "X (Normal)", color="white", ha="center", fontsize=9)
    ax.set_title("图5：RBF 核矩阵——MMD 比较的是两块平均嵌入的距离")
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig5_kernel_matrix.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig5_kernel_matrix.png"


def main():
    power, med = run_power_table()
    scan = bandwidth_scan()
    sscan = sample_scan()
    results = {
        "setup": {"seed": SEED, "n_per": N_PER, "reps": REPS, "n_perm": B_PERM,
                  "alpha": ALPHA, "sigmas": SIGMAS, "tail_scale": TAIL_SCALE},
        "power": power,
        "median_distance": med,
        "bandwidth_scan": scan,
        "sample_scan": sscan,
    }
    results["figs"] = [plot1(), plot2(power, med), plot3(scan), plot4(sscan), plot5()]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
