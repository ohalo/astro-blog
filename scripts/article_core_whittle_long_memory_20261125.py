# -*- coding: utf-8 -*-
"""Whittle 似然估长记忆 d：与 exact Gaussian MLE 对拍、短记忆误设、覆盖度与计时。
自包含：数据生成 + 实验 + 画图 + 结果 JSON，可直接 exec 复跑。"""
import os
import json
import time
import numpy as np
from scipy.special import gammaln
from scipy.linalg import toeplitz, cholesky, cho_factor, cho_solve
from scipy.optimize import minimize_scalar, minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20261125)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "whittle-likelihood-long-memory")
os.makedirs(OUT, exist_ok=True)


def acvf(k, d, sigma2=1.0):
    k = np.asarray(k)
    c = np.exp(gammaln(1 - 2 * d) - gammaln(1 - d) - gammaln(d))
    # k=0 时 gamma(0)=sigma^2*Gamma(1-2d)/Gamma(1-d)^2，是过程方差（非新息方差），公式对 k>=0 自洽
    return sigma2 * c * np.exp(gammaln(k + d) - gammaln(k + 1 - d))


def cov_matrix(n, d):
    k = np.arange(n)
    return toeplitz(acvf(k, d))


def gen_arfima(n, d, rng):
    L = cholesky(cov_matrix(n, d), lower=True, check_finite=False)
    return L @ rng.standard_normal(n)


def exact_nll(d, y):
    if d <= 0:  # d<0 的 gamma-ratio 协方差非正定，cho_factor 不校验会给出假的低目标值
        return 1e12
    n = len(y)
    S = cov_matrix(n, d)
    from numpy.linalg import LinAlgError
    try:
        c, low = cho_factor(S, lower=True, check_finite=False)
    except LinAlgError:
        return 1e12
    if np.any(np.diag(c) <= 0):
        return 1e12
    q = cho_solve((c, low), y, check_finite=False)
    return float(n * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(c))) + y @ q)


def exact_mle(y):
    r = minimize_scalar(exact_nll, args=(y,), bounds=(1e-6, 0.49), method="bounded",
                        options={"xatol": 1e-4})
    return float(r.x)


def periodogram(y):
    n = len(y)
    z = np.fft.rfft(y - y.mean()) / np.sqrt(n)
    j = np.arange(1, n // 2)
    lam = 2 * np.pi * j / n
    return lam, (np.abs(z[j]) ** 2), j


def whittle_nll(d, y, phi=0.0):
    lam, I, _ = periodogram(y)
    logf = -2 * d * np.log(2 * np.sin(lam / 2)) - np.log(np.abs(1 - phi * np.exp(-1j * lam)) ** 2)
    return float(np.sum(logf + I * np.exp(-logf)))


def whittle_mle(y):
    r = minimize_scalar(whittle_nll, args=(y,), bounds=(-0.44, 0.49), method="bounded",
                        options={"xatol": 1e-4})
    return float(r.x)


def whittle_mle_ar(y):
    def obj(p):
        return whittle_nll(p[0], y, phi=p[1])
    r = minimize(obj, x0=[0.1, 0.0], bounds=[(-0.44, 0.49), (-0.90, 0.90)], method="L-BFGS-B")
    return float(r.x[0]), float(r.x[1])


def main():
    rng = RNG
    res = {}

    # --- exp1: 样本 ACF / 周期图 vs 理论谱 ---
    n1 = 512
    d1 = 0.35
    y1 = gen_arfima(n1, d1, rng)
    ks = np.arange(1, 41)
    acf_theory = acvf(ks, d1) / acvf(0, d1)
    x1 = y1 - y1.mean()
    acf_emp = np.array([np.dot(x1[:-k], x1[k:]) / np.dot(x1, x1) for k in ks])
    lam, I, _ = periodogram(y1)
    sf = (2 * np.sin(lam / 2)) ** (-2 * d1)
    res["exp1_acf_lag1_theory"] = float(acf_theory[0])
    res["exp1_acf_lag1_emp"] = float(acf_emp[0])
    res["exp1_acf_lag20_theory"] = float(acf_theory[19])
    res["exp1_acf_lag20_emp"] = float(acf_emp[19])
    res["exp1_periodogram_first8"] = [float(v) for v in I[:8]]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.plot(ks, acf_theory, color="#2c7fb8", lw=2, label="theory ACF")
    a1.plot(ks, acf_emp, color="#c0392b", lw=1.5, label="sample ACF")
    a1.set_xlabel("lag"); a1.set_title(f"ARFIMA(0,d=0.35), n=512"); a1.legend(fontsize=8)
    a2.loglog(lam, I, color="#95a5a6", lw=0.8, label="periodogram")
    a2.loglog(lam, sf, color="#c0392b", lw=2, label="theory f(lambda)")
    a2.set_xlabel("Fourier frequency"); a2.set_ylabel("spectral density")
    a2.set_title("Periodogram is noisy but unbiased in log slope")
    a2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig01_acf_spectrum.png"), dpi=130); plt.close()

    # --- exp2: 同一路径的 exact / Whittle 目标函数与最优点 ---
    n2 = 384
    d2 = 0.30
    y2 = gen_arfima(n2, d2, rng)
    gd = np.linspace(-0.15, 0.49, 81)
    ecurve = np.array([exact_nll(float(d), y2) for d in gd])
    wcurve = np.array([whittle_nll(float(d), y2) for d in gd])
    de = exact_mle(y2)
    dw = whittle_mle(y2)
    res["exp2_d_true"] = d2
    res["exp2_d_exact"] = de
    res["exp2_d_whittle"] = dw
    res["exp2_gap"] = abs(de - dw)
    res["exp2_nll_exact_at_true"] = float(exact_nll(d2, y2))
    res["exp2_nll_exact_at_mle"] = float(exact_nll(de, y2))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.plot(gd, ecurve - ecurve.min(), color="#2c7fb8", lw=2, label="exact -log L")
    a1.axvline(de, color="#2c7fb8", ls="--", label=f"exact d={de:.3f}")
    a1.set_xlabel("d"); a1.set_title("Exact Gaussian likelihood"); a1.legend(fontsize=8)
    a2.plot(gd, wcurve - wcurve.min(), color="#c0392b", lw=2, label="Whittle objective")
    a2.axvline(dw, color="#c0392b", ls="--", label=f"Whittle d={dw:.3f}")
    a2.axvline(d2, color="#333", ls=":", label=f"true d={d2}")
    a2.set_xlabel("d"); a2.set_title("Whittle objective uses only the periodogram")
    a2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig02_likelihood_curves.png"), dpi=130); plt.close()

    # --- exp3: 蒙特卡洛 bias / RMSE，三档 n × 三档 d；共同随机数 ---
    ns = [128, 256, 512]
    ds = [0.10, 0.30, 0.40]
    reps_map = {128: 70, 256: 60, 512: 40}
    vals = {}
    for n in ns:
        for d in ds:
            ee, ww = [], []
            rr = np.random.default_rng(7000 + n + int(d * 100))
            for _ in range(reps_map[n]):
                yy = gen_arfima(n, d, rr)
                ee.append(exact_mle(yy))
                ww.append(whittle_mle(yy))
            ee = np.array(ee); ww = np.array(ww)
            key = f"n{n}_d{int(d*100)}"
            vals[key] = {
                "exact_bias": float(np.mean(ee) - d),
                "exact_rmse": float(np.sqrt(np.mean((ee - d) ** 2))),
                "whittle_bias": float(np.mean(ww) - d),
                "whittle_rmse": float(np.sqrt(np.mean((ww - d) ** 2))),
                "whittle_sd": float(np.std(ww)),
                "theory_sd": float(np.sqrt(3 / (n * np.pi ** 2))),
                "reps": reps_map[n],
            }
    res["exp3"] = vals
    res["exp3_worst_whittle_bias"] = max(abs(v["whittle_bias"]) for v in vals.values())
    res["exp3_mean_rmse_ratio"] = float(np.mean([v["whittle_rmse"] / v["exact_rmse"] for v in vals.values()]))

    x = np.arange(len(ds)); w = 0.16
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=False)
    for ax, n in zip(axes, ns):
        for i, key in enumerate(["exact_rmse", "whittle_rmse"]):
            col = "#2c7fb8" if i == 0 else "#c0392b"
            ax.bar(x + (i - 0.5) * w, [vals[f"n{n}_d{int(d*100)}"][key] for d in ds], w,
                   color=col, label="exact MLE" if i == 0 else "Whittle")
        ax.set_xticks(x); ax.set_xticklabels([str(d) for d in ds])
        ax.set_xlabel("true d"); ax.set_title(f"n={n}, RMSE"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig03_mc_rmse.png"), dpi=130); plt.close()

    # --- exp4: 短记忆误设：ARFIMA(0,d,0) 新息过 AR(1) 滤波器 ---
    n4 = 512
    phi_true = [0.0, 0.3, 0.6, 0.85]
    d4 = 0.25
    mis = []
    for ph in phi_true:
        e_hat, w_hat, c_hat, c_phi = [], [], [], []
        rr = np.random.default_rng(9100 + int(ph * 100))
        for _ in range(45):
            eta = gen_arfima(n4, d4, rr)
            u = np.zeros(n4)
            for t in range(1, n4):
                u[t] = ph * u[t - 1] + eta[t]
            yy = u  # 真均值为 0；exact 似然与平稳协方差必须同源，不做去均值
            e_hat.append(exact_mle(yy))
            w_hat.append(whittle_mle(yy))
            cd, cp = whittle_mle_ar(yy)
            c_hat.append(cd); c_phi.append(cp)
        mis.append({
            "phi": ph,
            "exact_mean": float(np.mean(e_hat)),
            "whittle_mis_mean": float(np.mean(w_hat)),
            "whittle_ar_mean": float(np.mean(c_hat)),
            "phi_hat_mean": float(np.mean(c_phi)),
        })
    res["exp4_misspec"] = mis
    res["exp4_bias_at_0.85_mis"] = mis[-1]["whittle_mis_mean"] - d4
    res["exp4_bias_at_0.85_fixed"] = mis[-1]["whittle_ar_mean"] - d4

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    phs = [m["phi"] for m in mis]
    ax.plot(phs, [m["exact_mean"] for m in mis], color="#2c7fb8", marker="o", label="exact ARFIMA(0,d,0)")
    ax.plot(phs, [m["whittle_mis_mean"] for m in mis], color="#c0392b", marker="s", label="Whittle misspecified")
    ax.plot(phs, [m["whittle_ar_mean"] for m in mis], color="#27ae60", marker="^", label="Whittle with AR(1)")
    ax.axhline(d4, color="#555", ls=":", label=f"true d={d4}")
    ax.set_xlabel("AR(1) phi in data"); ax.set_ylabel("mean d_hat")
    ax.set_title("Short memory looks like long memory if you ignore AR(1)")
    ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig04_short_memory_bias.png"), dpi=130); plt.close()

    # --- exp5: 覆盖度 + 计时 ---
    n5 = 256
    d5 = 0.30
    rr = np.random.default_rng(555)
    ds_hat = []
    for _ in range(160):
        yy = gen_arfima(n5, d5, rr)
        ds_hat.append(whittle_mle(yy))
    ds_hat = np.array(ds_hat)
    theory_sd = np.sqrt(3 / (n5 * np.pi ** 2))
    lo, hi = d5 - 1.96 * theory_sd, d5 + 1.96 * theory_sd
    cov = float(np.mean((ds_hat >= lo) & (ds_hat <= hi)))
    res["exp5_coverage"] = cov
    res["exp5_nominal"] = 0.95
    res["exp5_theory_sd"] = float(theory_sd)
    res["exp5_emp_sd"] = float(np.std(ds_hat))
    res["exp5_ci_width"] = float(hi - lo)

    bench = gen_arfima(n5, 0.35, rr)
    t0 = time.perf_counter()
    for _ in range(12):
        exact_mle(bench)
    t_exact = (time.perf_counter() - t0) / 12
    t0 = time.perf_counter()
    for _ in range(12):
        whittle_mle(bench)
    t_whittle = (time.perf_counter() - t0) / 12
    res["t_exact_ms"] = t_exact * 1000
    res["t_whittle_ms"] = t_whittle * 1000
    res["speedup"] = t_exact / t_whittle

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.hist(ds_hat, bins=24, color="#95a5a6", edgecolor="white")
    a1.axvspan(lo, hi, color="#2c7fb8", alpha=0.18, label="theory 95% CI")
    a1.axvline(d5, color="#c0392b", lw=2, label=f"true d={d5}")
    a1.set_xlabel("Whittle d_hat"); a1.set_title(f"coverage {cov:.3f} (nominal 0.95)")
    a1.legend(fontsize=8)
    a2.bar(["exact MLE", "Whittle"], [t_exact * 1000, t_whittle * 1000],
           color=["#2c7fb8", "#c0392b"])
    a2.set_ylabel("ms per fit (n=256)")
    a2.set_title(f"Whittle is {t_exact/t_whittle:.1f}x faster")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig05_coverage_speed.png"), dpi=130); plt.close()

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(json.dumps(res, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
