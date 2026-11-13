# -*- coding: utf-8 -*-
"""两阶段因子回归的测量误差：beta 衰减与 Shanken 修正。

本文所有数字的唯一来源。运行本脚本会：
1. 现场生成全部模拟数据（固定随机种子，可复现）；
2. 画 5 张图到 public/images/beta-attenuation-errors-in-variables/；
3. 把全部统计量写入同目录 results.json。
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261113
_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(
    os.path.dirname(_HERE), "public", "images", "beta-attenuation-errors-in-variables"
)
os.makedirs(_OUT, exist_ok=True)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, name), dpi=150)
    plt.close(fig)


def _ols_slope(x, y):
    """带截距 OLS 斜率。x: (n,), y: (n,)。"""
    xm = x - x.mean()
    ym = y - y.mean()
    return float(xm @ ym / (xm @ xm))


# ---------------------------------------------------------------- 实验 1
# 单资产 EIV：因子带测量噪声时，OLS beta 以 1/(1+lam) 衰减
def exp1_attenuation(reps=400, T=1000, beta_true=0.8, sigma_e=0.5):
    rng = np.random.default_rng(SEED)
    lams = [0.0, 0.1, 0.25, 0.5, 1.0, 1.5]
    means, sds = [], []
    for lam in lams:
        draws = np.empty(reps)
        for k in range(reps):
            f = rng.standard_normal(T)
            u = np.sqrt(lam) * rng.standard_normal(T)
            e = sigma_e * rng.standard_normal(T)
            r = beta_true * f + e
            draws[k] = _ols_slope(f + u, r)
        means.append(float(draws.mean()))
        sds.append(float(draws.std(ddof=1)))
    theory = [beta_true / (1.0 + lam) for lam in lams]
    max_dev = max(abs(m - t) for m, t in zip(means, theory))
    rel_dev = max(abs(m - t) / t for m, t in zip(means, theory))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(
        lams, means, yerr=sds, fmt="o", color="#1f6feb", capsize=3,
        label="模拟均值 ± 1 标准差（400 次重复）",
    )
    ax.plot(lams, theory, "-", color="#d1242f", lw=1.8,
            label=r"理论衰减 $\beta/(1+\lambda)$")
    ax.set_xlabel(r"测量噪声方差比 $\lambda=\mathrm{Var}(u)/\mathrm{Var}(f)$")
    ax.set_ylabel(r"OLS $\hat\beta$")
    ax.set_title(r"图1 因子测量误差使 OLS $\beta$ 沿 $\beta/(1+\lambda)$ 精确衰减")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "fig1_eiv_attenuation.png")
    return {
        "lams": lams,
        "beta_hat_mean": means,
        "beta_hat_sd": sds,
        "theory": theory,
        "beta_true": beta_true,
        "T": T,
        "reps": reps,
        "max_abs_dev": max_dev,
        "max_rel_dev": rel_dev,
    }


# ---------------------------------------------------------------- 实验 2
# 两阶段截面回归：生成的回归元（第一阶段估计的 beta）带来斜率衰减
def exp2_cs_slope(reps=500, N=100, T=120, lam_premium=0.5, beta_mean=1.0,
                  beta_sd=0.25):
    rng = np.random.default_rng(SEED + 1)
    sigmas_e = [0.2, 0.4, 0.6, 0.8, 1.0]
    m1, m2, s1, s2 = [], [], [], []
    var_beta = beta_sd ** 2
    for se in sigmas_e:
        v_eta = se ** 2 / T  # Var(f)=1
        k = var_beta / (var_beta + v_eta)
        d1, d2 = np.empty(reps), np.empty(reps)
        for rep in range(reps):
            beta = rng.normal(beta_mean, beta_sd, N)
            f = rng.standard_normal(T)
            e = se * rng.standard_normal((N, T))
            r = lam_premium * beta[:, None] + beta[:, None] * f[None, :] + e
            # 第一阶段：逐资产时序回归（带截距）
            fc = f - f.mean()
            denom = fc @ fc
            beta_hat = (r @ fc) / denom  # 斜率（截距吸收 lam*beta）
            rbar = r.mean(axis=1)
            d1[rep] = _ols_slope(beta_hat, rbar)
            d2[rep] = _ols_slope(beta_hat, lam_premium * beta)
        m1.append(float(d1.mean())); s1.append(float(d1.std(ddof=1)))
        m2.append(float(d2.mean())); s2.append(float(d2.std(ddof=1)))
    theory = [
        lam_premium * var_beta / (var_beta + se ** 2 / T) for se in sigmas_e
    ]
    max_dev = max(abs(m - t) for m, t in zip(m2, theory))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(sigmas_e, m2, yerr=s2, fmt="o", color="#1f6feb", capsize=3,
                label="因变量=真实期望收益 λβ（500 次重复）")
    ax.errorbar(sigmas_e, m1, yerr=s1, fmt="s", color="#8250df", capsize=3,
                label="因变量=样本均值 r̄（多一层均值估计噪声）")
    ax.plot(sigmas_e, theory, "--", color="#d1242f", lw=1.8,
            label=r"理论 $\lambda\,\mathrm{Var}(\beta)/(\mathrm{Var}(\beta)+\sigma_e^2/T)$")
    ax.axhline(lam_premium, color="gray", lw=1, ls=":")
    ax.text(sigmas_e[0], lam_premium + 0.004, "真风险溢价 λ=0.5", fontsize=9,
            color="gray")
    ax.set_xlabel(r"特异波动 $\sigma_e$")
    ax.set_ylabel("第二阶段截面斜率")
    ax.set_title("图2 生成的回归元让截面斜率朝零衰减，且随特异波动加重")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    _save(fig, "fig2_cs_slope_attenuation.png")
    return {
        "sigmas_e": sigmas_e,
        "slope_mean_rbar": m1,
        "slope_sd_rbar": s1,
        "slope_mean_true": m2,
        "slope_sd_true": s2,
        "theory": theory,
        "lam_premium": lam_premium,
        "N": N,
        "T": T,
        "beta_sd": beta_sd,
        "reps": reps,
        "max_abs_dev_true": max_dev,
    }


# ---------------------------------------------------------------- FM 工具
def _fama_macbeth(r, beta_hat):
    """逐期截面 OLS 斜率（带截距）。r: (N,T), beta_hat: (N,)。"""
    N, T = r.shape
    x = np.vstack([np.ones(N), beta_hat])
    inv = np.linalg.inv(x @ x.T)
    lams = np.empty(T)
    for t in range(T):
        lams[t] = (inv @ x @ r[:, t])[1]
    return lams


# ---------------------------------------------------------------- 实验 3
# Shanken 修正前后的 95% 置信区间覆盖率（蒙特卡洛）
def exp3_shanken_coverage(reps=800, N=100, T=120, lam_true=0.5,
                          beta_mean=1.0, beta_sd=0.25):
    rng = np.random.default_rng(SEED + 2)
    sigmas_e = [0.5, 0.8, 1.2]
    out = {"sigmas_e": sigmas_e, "coverage_naive": [], "coverage_shanken": [],
           "sd_emp": [], "se_naive_avg": [], "se_shanken_avg": [],
           "lam_mean": [], "reps": reps, "N": N, "T": T,
           "lam_true": lam_true, "beta_sd": beta_sd}
    for se in sigmas_e:
        cov_n = cov_s = 0
        se_n_all, se_s_all, lam_bar_all = np.empty(reps), np.empty(reps), np.empty(reps)
        for rep in range(reps):
            beta = rng.normal(beta_mean, beta_sd, N)
            f = rng.standard_normal(T)
            e = se * rng.standard_normal((N, T))
            r = lam_true * beta[:, None] + beta[:, None] * f[None, :] + e
            fc = f - f.mean()
            beta_hat = (r @ fc) / (fc @ fc)
            lams_t = _fama_macbeth(r, beta_hat)
            lam_bar = lams_t.mean()
            se_n = lams_t.std(ddof=1) / np.sqrt(T)
            c = 1.0 + lam_bar ** 2 / f.var(ddof=1)
            se_s = se_n * np.sqrt(c)
            lam_bar_all[rep] = lam_bar
            se_n_all[rep] = se_n
            se_s_all[rep] = se_s
            if abs(lam_bar - lam_true) < 1.96 * se_n:
                cov_n += 1
            if abs(lam_bar - lam_true) < 1.96 * se_s:
                cov_s += 1
        out["coverage_naive"].append(cov_n / reps)
        out["coverage_shanken"].append(cov_s / reps)
        out["sd_emp"].append(float(lam_bar_all.std(ddof=1)))
        out["se_naive_avg"].append(float(se_n_all.mean()))
        out["se_shanken_avg"].append(float(se_s_all.mean()))
        out["lam_mean"].append(float(lam_bar_all.mean()))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    ax = axes[0]
    x = np.arange(len(sigmas_e))
    w = 0.36
    ax.bar(x - w / 2, out["coverage_naive"], w, color="#9a6700",
           label="Fama-MacBeth 朴素 SE")
    ax.bar(x + w / 2, out["coverage_shanken"], w, color="#d1242f",
           label="Shanken 修正 SE")
    ax.axhline(0.95, color="gray", ls=":", lw=1.5)
    ax.text(-0.45, 0.955, "名义 95%", fontsize=9, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sigmas_e])
    ax.set_ylim(0.8, 1.0)
    ax.set_xlabel(r"特异波动 $\sigma_e$")
    ax.set_ylabel(r"$\hat\lambda$ 的 95% CI 覆盖率")
    ax.set_title("图3a Shanken 修正抬升覆盖率，但治不了衰减偏倚")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    ax.plot(x, out["sd_emp"], "o-", color="#1f6feb", label="经验标准差")
    ax.plot(x, out["se_naive_avg"], "s--", color="#9a6700", label="朴素 SE 均值")
    ax.plot(x, out["se_shanken_avg"], "^-", color="#d1242f",
            label="Shanken SE 均值")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sigmas_e])
    ax.set_xlabel(r"特异波动 $\sigma_e$")
    ax.set_ylabel(r"$\hat\lambda$ 的离散度")
    ax.set_title(r"图3b $\sigma_e=1.2$ 时偏倚 $\approx$ 1 个标准差，主导失准")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    _save(fig, "fig3_shanken_coverage.png")
    return out


# ---------------------------------------------------------------- 实验 4
# Shanken 因子的来源：c = 1 + λ²/σ_f²，随溢价水平放大
def exp4_se_inflation(reps=800, N=100, T=120, sigma_e=0.8,
                      beta_mean=1.0, beta_sd=0.25):
    rng = np.random.default_rng(SEED + 3)
    lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    sd_emp, se_n_avg, se_s_avg = [], [], []
    for lam_true in lams:
        lam_bar_all = np.empty(reps)
        se_n_all = np.empty(reps)
        se_s_all = np.empty(reps)
        for rep in range(reps):
            beta = rng.normal(beta_mean, beta_sd, N)
            f = rng.standard_normal(T)
            e = sigma_e * rng.standard_normal((N, T))
            r = lam_true * beta[:, None] + beta[:, None] * f[None, :] + e
            fc = f - f.mean()
            beta_hat = (r @ fc) / (fc @ fc)
            lams_t = _fama_macbeth(r, beta_hat)
            se_n = lams_t.std(ddof=1) / np.sqrt(T)
            lam_bar = lams_t.mean()
            lam_bar_all[rep] = lam_bar
            se_n_all[rep] = se_n
            se_s_all[rep] = se_n * np.sqrt(1.0 + lam_bar ** 2 / f.var(ddof=1))
        sd_emp.append(float(lam_bar_all.std(ddof=1)))
        se_n_avg.append(float(se_n_all.mean()))
        se_s_avg.append(float(se_s_all.mean()))
    theory = [float(np.sqrt(1.0 + lam ** 2)) for lam in lams]
    ratio = [s / n for s, n in zip(se_s_avg, se_n_avg)]
    max_dev = max(abs(rt - t) for rt, t in zip(ratio, theory))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(lams, sd_emp, "o-", color="#1f6feb", label="经验标准差（800 次重复）")
    ax.plot(lams, se_n_avg, "s--", color="#9a6700", label="Fama-MacBeth 朴素 SE 均值")
    ax.plot(lams, se_s_avg, "^-", color="#d1242f",
            label=r"Shanken 修正 SE 均值（$\times\sqrt{1+\hat\lambda^2}$）")
    ax.set_xlabel(r"真风险溢价 $\lambda$")
    ax.set_ylabel(r"$\hat\lambda$ 的标准误")
    ax.set_title(r"图4 Shanken 因子 $c=1+\lambda^2/\sigma_f^2$：溢价越高，低估越狠")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    _save(fig, "fig4_shanken_se_inflation.png")
    return {
        "lams": lams, "sd_emp": sd_emp, "se_naive_avg": se_n_avg,
        "se_shanken_avg": se_s_avg, "ratio_shanken_over_naive": ratio,
        "theory_c": theory, "sigma_e": sigma_e, "reps": reps,
        "max_abs_dev_ratio": max_dev,
    }


# ---------------------------------------------------------------- 实验 5
# 衰减的 beta 制造假 alpha：与真实 beta 成正比的伪异常 + 假发现率膨胀
def exp5_spurious_alpha(reps=500, N=100, T=120, lam_true=0.5, sigma_e=1.2,
                        beta_mean=1.0, beta_sd=0.25):
    rng = np.random.default_rng(SEED + 4)
    var_beta = beta_sd ** 2
    v_eta = sigma_e ** 2 / T
    k = var_beta / (var_beta + v_eta)
    theory_slope = lam_true * (1.0 - k)
    slopes = np.empty(reps)
    fdr = np.empty(reps)
    for rep in range(reps):
        beta = rng.normal(beta_mean, beta_sd, N)
        f = rng.standard_normal(T)
        e = sigma_e * rng.standard_normal((N, T))
        r = lam_true * beta[:, None] + beta[:, None] * f[None, :] + e
        fc = f - f.mean()
        denom = fc @ fc
        beta_hat = (r @ fc) / denom
        resid = r - r.mean(axis=1, keepdims=True) - beta_hat[:, None] * fc[None, :]
        sig_e = resid.std(axis=1, ddof=2)
        lams_t = _fama_macbeth(r, beta_hat)
        lam_bar = lams_t.mean()
        rbar = r.mean(axis=1)
        alpha = rbar - lam_bar * beta_hat
        slopes[rep] = _ols_slope(beta, alpha)  # 诊断用真实 beta
        tstat = alpha / (sig_e / np.sqrt(T))
        fdr[rep] = float((np.abs(tstat) > 1.96).mean())

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    ax = axes[0]
    beta_show = rng.normal(beta_mean, beta_sd, N)
    f_show = rng.standard_normal(T)
    e_show = sigma_e * rng.standard_normal((N, T))
    r_show = lam_true * beta_show[:, None] + beta_show[:, None] * f_show[None, :] + e_show
    fc = f_show - f_show.mean()
    bh = (r_show @ fc) / (fc @ fc)
    lams_t = _fama_macbeth(r_show, bh)
    alpha_show = r_show.mean(axis=1) - lams_t.mean() * bh
    ax.scatter(beta_show, alpha_show, s=14, color="#1f6feb", alpha=0.6)
    xs = np.linspace(beta_show.min(), beta_show.max(), 10)
    ax.plot(xs, theory_slope * xs, color="#d1242f", lw=1.8,
            label=r"理论斜率 $\lambda(1-k)$")
    ax.set_xlabel(r"真实 $\beta_i$")
    ax.set_ylabel(r"估计的 $\hat\alpha_i$")
    ax.set_title("图5a 衰减的 beta 制造与 β 成正比的伪 alpha（单次重复）")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.hist(fdr, bins=24, color="#1f6feb", alpha=0.75)
    ax.axvline(0.05, color="gray", ls=":", lw=1.5)
    ax.text(0.052, ax.get_ylim()[1] * 0.9, "名义 5%", fontsize=9, color="gray")
    ax.axvline(float(fdr.mean()), color="#d1242f", lw=1.8)
    ax.text(float(fdr.mean()) + 0.002, ax.get_ylim()[1] * 0.78,
            f"实测 {fdr.mean():.3f}", fontsize=9, color="#d1242f")
    ax.set_xlabel("单次重复中 |t|>1.96 的资产占比")
    ax.set_ylabel("重复次数")
    ax.set_title("图5b 伪 alpha 把 |t|>1.96 比例抬到远超 5%")
    ax.grid(alpha=0.3)
    _save(fig, "fig5_spurious_alpha_fdr.png")
    return {
        "sigma_e": sigma_e, "T": T, "lam_true": lam_true,
        "attenuation_k": k, "theory_alpha_slope": theory_slope,
        "alpha_slope_mean": float(slopes.mean()),
        "alpha_slope_sd": float(slopes.std(ddof=1)),
        "reps": reps,
        "fdr_mean": float(fdr.mean()),
        "fdr_sd": float(fdr.std(ddof=1)),
        "nominal": 0.05,
    }


def main():
    r1 = exp1_attenuation()
    r2 = exp2_cs_slope()
    r3 = exp3_shanken_coverage()
    r4 = exp4_se_inflation()
    r5 = exp5_spurious_alpha()
    results = {"exp1": r1, "exp2": r2, "exp3": r3, "exp4": r4, "exp5": r5}
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
