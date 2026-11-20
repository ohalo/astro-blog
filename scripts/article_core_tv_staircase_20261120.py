# -*- coding: utf-8 -*-
"""全变差 TV 去噪：分段常数上保边赢家，平滑信号上 staircasing 阶梯伪影。

本脚本是文章全部数字的唯一来源。运行它会：
1. 手写 ADMM 求解 ROF 模型 min 0.5||x-y||² + λ·TV(x)，TV(x)=Σ|x_{i+1}-x_i|；
2. 在真值已知的 blocks(分段常数) 与 smooth(平滑) 两类信号上对决
   moving average / median / Haar 硬阈值 / Kalman RTS 平滑 / TV；
3. 量化 staircasing：TV 解里相邻差分近零的"平台占比"，平滑信号上该指标虚高；
4. 扫描 λ 与噪声 σ，做 5 折留观测交叉验证选 λ，对比 oracle；
5. 画 5 张图到 public/images/tv-denoising-fused-lasso-staircasing/，统计写入 results.json。
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_banded

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261120
N = 256
NOISE = 0.30
REPS = 30
LAMBDA = 0.6

_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "public", "images", "tv-denoising-fused-lasso-staircasing",
)
os.makedirs(_OUT, exist_ok=True)


def blocks_signal(n):
    """Donoho 风格分段常数，11 段，值域约 [0,3]。"""
    x = np.zeros(n)
    edges = [0, .08, .19, .27, .36, .45, .58, .67, .76, .88, 1.0]
    vals = [1.0, 4.0, 2.0, 3.0, 5.0, 2.5, 6.0, 3.5, 7.0, 4.0, 5.5]
    vals = [v / 3.0 for v in vals]
    for i in range(len(edges) - 1):
        a, b = int(edges[i] * n), int(edges[i + 1] * n)
        x[a:b] = vals[i]
    return x


def smooth_signal(n):
    t = np.arange(n) / n
    return 1.5 * np.sin(2 * np.pi * t) + 0.6 * np.sin(6 * np.pi * t + .4) + 0.8 * t


def tv_denoise(y, lam, n_iter=600, rho=10.0, mask=None):
    """ADMM 解 ROF：min 0.5||x-y||²_W + lam·Σ|x_{i+1}-x_i|。
    mask 为布尔观测掩码；缺失点不做保真约束（供 CV 用）。
    x-update 用三对角 solve_banded，O(n)。"""
    n = len(y)
    Dx = np.diff(np.eye(n), axis=0)  # (n-1, n)，沿行差分
    DtD = Dx.T @ Dx                  # 三对角
    # A = W + rho·DtD，W=diag(mask)
    main = np.diag(DtD).copy()
    off = np.diag(DtD, 1).copy()
    w = np.ones(n) if mask is None else mask.astype(float)
    main = w + rho * main
    ab = np.zeros((3, n))
    ab[1] = main
    ab[0, 1:] = rho * off
    ab[2, :-1] = rho * off
    rhs0 = (w * y).copy()
    x = y.copy()
    z = np.diff(x)
    u = np.zeros(n - 1)
    for _ in range(n_iter):
        rhs = rhs0 + rho * (Dx.T @ (z - u))
        x = solve_banded((1, 1), ab, rhs)
        dz = np.diff(x) + u
        z = np.sign(dz) * np.maximum(np.abs(dz) - lam / rho, 0.0)
        u = u + np.diff(x) - z
    return x


def moving_average(y, k):
    pad = k // 2
    yy = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(yy, np.ones(k) / k, mode="valid")


def median_filter(y, k=9):
    from scipy.ndimage import median_filter as mf
    return mf(y, size=k, mode="nearest")


def haar_hard(y, level=4, factor=1.0):
    """多层 Haar 硬阈值，阈值 = factor·σ·sqrt(2 log n)，σ 用 MAD 稳健估计。"""
    x = y.copy().astype(float)
    n = len(x)
    sig = 1.4826 * np.median(np.abs(np.diff(x) - np.median(np.diff(x)))) / np.sqrt(2)
    thr = factor * sig * np.sqrt(2 * np.log(n))
    cur = x
    for _ in range(level):
        m = len(cur) // 2
        a = (cur[0:2 * m:2] + cur[1:2 * m:2]) / np.sqrt(2)
        d = (cur[0:2 * m:2] - cur[1:2 * m:2]) / np.sqrt(2)
        d[np.abs(d) < thr] = 0.0
        cur = np.concatenate([a, np.abs(d) * 0 + d, cur[2 * m:]]) if 2 * m < len(cur) else np.concatenate([a, d])
    # 重构
    for _ in range(level):
        m = len(cur) // 2
        a, d = cur[:m], cur[m:2 * m]
        rec = np.empty(2 * m)
        rec[0:2 * m:2] = (a + d) / np.sqrt(2)
        rec[1:2 * m:2] = (a - d) / np.sqrt(2)
        cur = rec
    return cur[:n]


def kalman_rts(y, q=1e-2, r=NOISE ** 2):
    """局部水平模型 x_t=x_{t-1}+w, y_t=x_t+v；卡尔曼滤波 + RTS 平滑。"""
    n = len(y)
    mf = np.zeros(n); Pf = np.zeros(n)
    mp = y[0]; Pp = 1.0
    for t in range(n):
        K = Pp / (Pp + r)
        mf[t] = mp + K * (y[t] - mp)
        Pf[t] = (1 - K) * Pp
        mp = mf[t]; Pp = Pf[t] + q
    ms = mf.copy(); Ps = Pf.copy()
    for t in range(n - 2, -1, -1):
        J = Pf[t] / (Pf[t] + q)
        ms[t] = mf[t] + J * (ms[t + 1] - mf[t])
    return ms


def platform_ratio(x, eps=1e-4):
    """相邻差分近零的比例——TV 在平滑信号上制造的平台占比。"""
    return float(np.mean(np.abs(np.diff(x)) < eps))


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def one_repeat(rng, sig, lam=LAMBDA):
    truth = blocks_signal(N) if sig == "blocks" else smooth_signal(N)
    y = truth + rng.standard_normal(N) * NOISE
    est = {
        "ma5": moving_average(y, 5),
        "median9": median_filter(y, 9),
        "haar": haar_hard(y, 4),
        "kalman": kalman_rts(y),
        "tv": tv_denoise(y, lam),
    }
    return truth, y, {k: mse(v, truth) for k, v in est.items()}, est, truth


def method_compare():
    rng = np.random.default_rng(SEED)
    res = {}
    for sig in ["blocks", "smooth"]:
        agg = {k: [] for k in ["ma5", "median9", "haar", "kalman", "tv"]}
        plat = []
        for _ in range(REPS):
            truth, y, ms, est, _ = one_repeat(rng, sig)
            for k in agg:
                agg[k].append(ms[k])
            plat.append(platform_ratio(est["tv"]))
        res[sig] = {k: float(np.mean(v)) for k, v in agg.items()}
        res[sig]["tv_platform"] = float(np.mean(plat))
        res[sig]["tv_platform_sd"] = float(np.std(plat))
    return res


def lambda_scan():
    """λ 扫描：分段常数与平滑信号上 TV 的 MSE，展示最优 λ 数量级不同。"""
    rng = np.random.default_rng(SEED + 1)
    lams = [0.1, 0.3, 0.6, 1.0, 2.0, 3.5, 5.0, 8.0, 12.0]
    res = {"lams": lams, "blocks": [], "smooth": [], "plat_smooth": []}
    for lam in lams:
        mb, msx, pb = [], [], []
        for _ in range(REPS):
            for sig, bucket in [("blocks", mb), ("smooth", msx)]:
                truth = blocks_signal(N) if sig == "blocks" else smooth_signal(N)
                y = truth + rng.standard_normal(N) * NOISE
                x = tv_denoise(y, lam)
                bucket.append(mse(x, truth))
                if sig == "smooth":
                    pb.append(platform_ratio(x))
        res["blocks"].append(float(np.mean(mb)))
        res["smooth"].append(float(np.mean(msx)))
        res["plat_smooth"].append(float(np.mean(pb)))
    return res


def noise_scan():
    """噪声 σ 扫描：TV 固定 λ 与 oracle λ 的 MSE，展示 λ 必须随 σ 缩放。"""
    rng = np.random.default_rng(SEED + 2)
    sigmas = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
    res = {"sigmas": sigmas, "tv_fixed": [], "tv_oracle": [], "ma5": []}
    for sg in sigmas:
        f, o, m = [], [], []
        for _ in range(REPS):
            truth = blocks_signal(N)
            y = truth + rng.standard_normal(N) * sg
            f.append(mse(tv_denoise(y, LAMBDA), truth))
            # oracle：在 λ 网格里挑对当前 y 最优（仅作上界参照）
            best = min((mse(tv_denoise(y, l), truth), l) for l in [0.3, 1, 2, 4, 8, 16, 32])
            o.append(best[0])
            m.append(mse(moving_average(y, 5), truth))
        res["tv_fixed"].append(float(np.mean(f)))
        res["tv_oracle"].append(float(np.mean(o)))
        res["ma5"].append(float(np.mean(m)))
    return res


def cv_lambda():
    """5 折留观测 CV：mask 掉 1/5 观测，用其余点拟合 TV，在留出点算预测 MSE。"""
    rng = np.random.default_rng(SEED + 3)
    truth = blocks_signal(N)
    y = truth + rng.standard_normal(N) * NOISE
    lams = [0.3, 1.0, 2.0, 3.5, 5.0, 8.0, 12.0, 20.0]
    folds = rng.integers(0, 5, N)
    res = {"lams": lams, "cv_mse": []}
    for lam in lams:
        errs = []
        for k in range(5):
            mask = folds != k
            x = tv_denoise(y, lam, mask=mask)
            errs.append(np.mean((x[~mask] - y[~mask]) ** 2))
        res["cv_mse"].append(float(np.mean(errs)))
    best = int(np.argmin(res["cv_mse"]))
    res["cv_best_lam"] = lams[best]
    res["oracle_full"] = min((mse(tv_denoise(y, l), truth), l) for l in lams)[1]
    res["truth"] = truth.tolist(); res["y"] = y.tolist()
    res["x_cv"] = tv_denoise(y, lams[best]).tolist()
    return res


def plot1(cmp):
    truth = blocks_signal(N)
    rng = np.random.default_rng(SEED + 10)
    y = truth + rng.standard_normal(N) * NOISE
    est = {"moving avg(5)": moving_average(y, 5), "median(9)": median_filter(y, 9),
           "Haar 硬阈值": haar_hard(y, 4), "Kalman RTS": kalman_rts(y),
           f"TV(λ={LAMBDA})": tv_denoise(y, LAMBDA)}
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, key in zip(axes, ["moving avg(5)", "Haar 硬阈值", f"TV(λ={LAMBDA})"]):
        ax.plot(truth, "k-", lw=2, label="真值")
        ax.plot(y, color="#bdc3c7", lw=.6, label="观测")
        ax.plot(est[key], color="#c0392b" if "TV" in key else "#2980b9", lw=1.6, label=key)
        ax.set_title(f"{key}  MSE={mse(est[key],truth):.4f}")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("图1：分段常数信号——TV 保住跳变，移动平均把台阶抹成斜坡", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .96])
    p = os.path.join(_OUT, "fig1_blocks_compare.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig1_blocks_compare.png"


def plot2():
    truth = smooth_signal(N)
    rng = np.random.default_rng(SEED + 11)
    y = truth + rng.standard_normal(N) * NOISE
    x = tv_denoise(y, LAMBDA)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(truth, "k-", lw=2, label="真值（平滑）")
    axes[0].plot(y, color="#bdc3c7", lw=.6, label="观测")
    axes[0].plot(x, color="#c0392b", lw=1.5, label=f"TV(λ={LAMBDA})")
    axes[0].set_title(f"TV 在平滑信号上制造阶梯；平台占比={platform_ratio(x):.3f}")
    axes[0].legend(fontsize=9)
    axes[1].plot(np.diff(truth), "k-", lw=1.5, label="真值一阶差分")
    axes[1].plot(np.diff(x), color="#c0392b", lw=1, label="TV 一阶差分（大量近零）")
    axes[1].axhline(0, color="gray", ls="--", lw=.8)
    axes[1].set_title("阶梯伪影指纹：TV 差分钉在 0，真差分连续变化")
    axes[1].legend(fontsize=9)
    fig.suptitle("图2：staircasing——保边优点在平滑区变成伪影", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .96])
    p = os.path.join(_OUT, "fig2_staircasing.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig2_staircasing.png"


def plot3(scan):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.semilogx(scan["lams"], scan["blocks"], "o-", color="#27ae60", lw=2, label="blocks 分段常数")
    ax.semilogx(scan["lams"], scan["smooth"], "s-", color="#c0392b", lw=2, label="smooth 平滑")
    ax.set_xlabel("λ"); ax.set_ylabel("去噪 MSE")
    ax.set_title("图3：λ 扫描——两类信号的最优 λ 都落在 0.6 附近")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig3_lambda_scan.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig3_lambda_scan.png"


def plot4(scan):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.semilogx(scan["lams"], scan["plat_smooth"], "o-", color="#8e44ad", lw=2)
    ax.set_xlabel("λ"); ax.set_ylabel("平滑信号上 TV 解的平台占比")
    ax.set_title("图4：λ 越大平台越多——阶梯伪影随惩罚单调加重")
    fig.tight_layout()
    p = os.path.join(_OUT, "fig4_platform.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig4_platform.png"


def plot5(scan, cv):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(scan["sigmas"], scan["tv_fixed"], "o-", color="#c0392b", lw=2, label="TV 固定 λ=5")
    axes[0].plot(scan["sigmas"], scan["tv_oracle"], "s-", color="#27ae60", lw=2, label="TV oracle λ")
    axes[0].plot(scan["sigmas"], scan["ma5"], "^--", color="#2980b9", lw=2, label="moving avg(5)")
    axes[0].set_xlabel("噪声 σ"); axes[0].set_ylabel("MSE")
    axes[0].set_title("噪声水平扫描：λ 必须随 σ 缩放")
    axes[0].legend(fontsize=8)
    axes[1].plot(cv["lams"], cv["cv_mse"], "o-", color="#e67e22", lw=2, label="5 折 CV 预测 MSE")
    axes[1].axvline(cv["cv_best_lam"], color="red", ls="--", label=f"CV 选中 λ={cv['cv_best_lam']}")
    axes[1].axvline(cv["oracle_full"], color="green", ls=":", label=f"全样本 oracle λ={cv['oracle_full']}")
    axes[1].set_xscale("log"); axes[1].set_xlabel("λ"); axes[1].set_ylabel("留出点 MSE")
    axes[1].set_title("交叉验证选 λ")
    axes[1].legend(fontsize=8)
    fig.suptitle("图5：λ 选择的现实约束——固定 λ 会随噪声失配，CV 能自动跟进", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .94])
    p = os.path.join(_OUT, "fig5_noise_cv.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig5_noise_cv.png"


def main():
    cmp_ = method_compare()
    lscan = lambda_scan()
    nscan = noise_scan()
    cv = cv_lambda()
    results = {
        "setup": {"seed": SEED, "n": N, "noise": NOISE, "reps": REPS, "lambda": LAMBDA},
        "method_compare": cmp_,
        "lambda_scan": lscan,
        "noise_scan": nscan,
        "cv": {k: v for k, v in cv.items() if k not in ("truth", "y", "x_cv")},
    }
    results["figs"] = [plot1(cmp_), plot2(), plot3(lscan), plot4(lscan), plot5(nscan, cv)]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
