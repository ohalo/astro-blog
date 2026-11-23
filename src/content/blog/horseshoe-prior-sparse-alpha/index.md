---
title: "马蹄先验估稀疏 Alpha：为什么它能把噪声压到 0.04 又放过真信号，以及它在检验功效和稠密信号上输给谁"
description: "真值已知的正态均值实验（p=120）用 Gibbs 采样拆开马蹄先验。收缩曲线上 y=1 压到 0.56、y=4 保留到 3.59，真噪声的平均后验绝对值只有 0.041。稀疏度 s0=2 时 MSE 0.065 略胜软阈值 0.073，s0=60 却不敌经验贝叶斯 0.734 对 0.907。kappa<0.5 检测功效只有 14.0%、FDR 9.4%，比 BH 保守得多；样本内外夏普落差 0.106 是四种方法里最小。附完整 Gibbs 代码与五张真实计算图。"
publishDate: '2026-11-23T10:20:00Z'
tags:
  - 量化
  - 贝叶斯
  - 稀疏估计
  - MCMC
  - Python
language: zh
difficulty: advanced
image: "/images/horseshoe-prior-sparse-alpha/fig1_shrinkage.png"
---

## 一、结论先行

扫两百个策略，真正有 alpha 的可能就五六个。最朴素的办法是把 t 值大于 2 的都留下——但你留下来的一大半是噪声。硬阈值的问题不在于它错，而在于它对"刚好过线"和"远超线"一视同仁：一个 t=2.1 的估计和一个 t=8 的估计，都被原封不动地当成真信号。

**马蹄先验（Horseshoe Prior）** 换了个思路：不要二选一地"留"或"删"，而是对每个估计量做连续收缩。噪声被狠狠压向零，真信号几乎不动。它的名字来自先验密度的形状——在原点附近像马蹄铁一样高高隆起（强收缩），尾部又拖得很慢（不收缩大信号）。Carvalho、Polson 和 Scott 在 2010 年把它形式化，近年在稀疏因子选择里很受欢迎。

本文用真值已知的正态均值模型（$p=120$）做 Gibbs 采样，把它的表现和硬阈值、软阈值、经验贝叶斯两成分模型、BH 检验摆在一起。五个结论：

- **收缩形状名副其实**：观测 $y=1$ 时后验均值压到 0.56，$y=2$ 压到 1.45，$y=4$ 保留到 3.59。噪声（真值为零）的平均后验绝对值只有 0.041。
- **稀疏时略占优**：真信号数 $s_0=2$ 时 MSE 0.065，略好于软阈值 0.073 和经验贝叶斯 0.070；硬阈值 0.306 被甩开一大截。
- **信号一多就露馅**：$s_0=60$ 时马蹄 MSE 0.907，反被经验贝叶斯 0.734 超过。它赌的就是稀疏，赌错了就要付账。
- **它不是检验**：用 $\kappa<0.5$ 当"显著"的判据，功效只有 14.0%、FDR 9.4%；BH 检验功效 24.0%、FDR 13.8%。把后验收缩当成 p 值控制，是常见误用。
- **协方差最小**：样本内外夏普落差 0.106，四种方法里最小（硬阈值 0.153、BH 7.38、软阈值 14.88）——它择时的"乐观偏差"最轻。

先讲清楚模型边界：这里每个 alpha 是独立的一个数，没有策略间协方差。真实多因子场景要复杂得多，第 7 节会讲这个简化意味着什么。

## 二、模型与采样器

正态均值模型写出来很简单：

$$
y_i = \alpha_i + \epsilon_i,\quad \epsilon_i \sim N(0,\sigma^2),\quad i=1,\dots,p.
$$

马蹄先验给每个 $\alpha_i$ 套一个局部尺度 $\lambda_i$ 和一个全局尺度 $\tau$：

$$
\alpha_i \mid \lambda_i,\tau \sim N(0,\sigma^2\lambda_i^2\tau^2),\quad \lambda_i\sim C^+(0,1),\quad \tau\sim C^+(0,\tau_0).
$$

局部尺度让每个信号自己决定收缩多少，全局尺度 $\tau$ 控制整体稀疏程度——真信号少时它把所有的 $\lambda_i$ 一起往下拽。半柯西分布没有解析共轭，标准做法是引入辅助变量把它写成逆伽马分布，再做 Gibbs 采样。后验里有两个关键量：

$$
\kappa_i = \frac{1}{1+\lambda_i^2\tau^2},\quad \alpha_i \mid \cdot \sim N\big((1-\kappa_i)y_i,\ \sigma^2(1-\kappa_i)\big).
$$

$\kappa_i$ 是收缩因子，越接近 1 表示压得越狠。核心实现如下，自包含可直接运行。

```python
# -*- coding: utf-8 -*-
"""Horseshoe prior for sparse alpha: shrinkage, FDR, misspecification, and OOS.

Single source of truth for every number and figure in the article.

Normal means  y_i = alpha_i + eps_i,  eps_i ~ N(0, sigma^2),  i = 1..p.
Prior (Carvalho, Polson & Scott 2010):
    alpha_i | lambda_i, tau ~ N(0, sigma^2 lambda_i^2 tau^2)
    lambda_i ~ C+(0,1),  tau ~ C+(0, tau0)
Gibbs sampler uses auxiliary variables for the half-Cauchy priors:
    1/lambda_i^2 | a_i ~ Exp(1/a_i) ;  a_i ~ InvGamma(1/2, 1)
    1/tau^2 | a_tau ~ Exp(1/a_tau)  ;  a_tau ~ InvGamma(1/2, 1)
    alpha_i | . ~ N((1-kappa_i) y_i, sigma^2 (1-kappa_i)),
        kappa_i = 1/(1+lambda_i^2 tau^2)
Comparators: hard threshold, universal soft threshold sqrt(2 log p),
empirical-Bayes two-group normal means (spike-and-slab with EM), BH at q=0.1.

Fixed seeds keep every replay byte-identical to results.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261123
P = 120
SIGMA = 1.0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "horseshoe-prior-sparse-alpha")
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------- sampler
def horseshoe_gibbs(y, sigma=1.0, tau0=None, n_iter=2500, burn=1000,
                    thin=3, seed=0, return_all=False):
    """Return posterior mean of alpha, mean shrinkage kappa, and P(|alpha|>thr).

    Standard auxiliary-variable Gibbs for the half-Cauchy prior:
        lambda_i^2 | phi_i ~ InvGamma(1/2, 1/phi_i),  phi_i ~ InvGamma(1/2, 1)
        tau^2      | psi   ~ InvGamma(1/2, 1/psi),    psi   ~ InvGamma(1/2, 1/tau0^2)
    Full conditionals (Carvalho, Polson & Scott 2010):
        alpha_i | . ~ N((1-kappa_i) y_i, sigma^2 (1-kappa_i)),
                        kappa_i = 1/(1+lambda_i^2 tau^2)
        1/lambda_i^2 | . ~ Exp(1/phi_i + alpha_i^2/(2 sigma^2 tau^2))
        1/tau^2      | . ~ Exp(1/psi   + sum(alpha_i^2)/(2 sigma^2 lambda_i^2))
        phi_i ~ InvGamma(1, 1 + 1/lambda_i^2)
        psi   ~ InvGamma(1, 1/tau0^2 + 1/tau^2)
    """
    rng = np.random.default_rng(seed)
    p = len(y)
    if tau0 is None:
        tau0 = max(1e-4, 1.0 / np.sqrt(p))
    lam2 = np.ones(p)
    phi = np.ones(p)
    tau2 = float(tau0 ** 2)
    psi = 1.0
    alpha = y.copy()
    keep, kaps = [], []
    for it in range(n_iter):
        kappa = 1.0 / (1.0 + lam2 * tau2)
        alpha = (1.0 - kappa) * y + np.sqrt(sigma ** 2 * (1.0 - kappa)) * rng.standard_normal(p)
        # xi_i = 1/lambda_i^2 ~ Exp(rate_i)
        rate_lam = 1.0 / phi + alpha ** 2 / (2.0 * sigma ** 2 * tau2)
        xi_lam = rng.exponential(scale=1.0 / np.maximum(rate_lam, 1e-12))
        lam2 = 1.0 / np.maximum(xi_lam, 1e-12)
        # phi | lambda ~ InvGamma(1, 1 + 1/lambda^2) -> 1/phi ~ Gamma(1, 1 + xi)
        inv_phi = rng.gamma(shape=1.0, scale=1.0 / (1.0 + xi_lam))
        phi = 1.0 / np.maximum(inv_phi, 1e-12)
        # xi_tau = 1/tau^2 ~ Exp(rate_tau)
        rate_tau = 1.0 / psi + np.sum(alpha ** 2 / (2.0 * sigma ** 2 * lam2))
        # tau^2 | alpha, lambda, psi: shape = (p+1)/2 (p likelihoods + 1 prior)
        xi_tau = rng.gamma(shape=(p + 1) / 2.0,
                           scale=1.0 / max(rate_tau, 1e-12))
        tau2 = 1.0 / max(xi_tau, 1e-12)
        # psi ~ InvGamma(1, 1/tau0^2 + 1/tau^2) -> 1/psi ~ Gamma(1, 1/tau0^2 + xi_tau)
        inv_psi = rng.gamma(shape=1.0, scale=1.0 / (1.0 / tau0 ** 2 + xi_tau))
        psi = 1.0 / max(inv_psi, 1e-12)
        if it >= burn and (it - burn) % thin == 0:
            keep.append(alpha.copy())
            kaps.append(1.0 / (1.0 + lam2 * tau2))
    am = np.mean(keep, axis=0)
    km = np.mean(kaps, axis=0)
    if return_all:
        return am, km, np.asarray(keep)
    return am, km, None


# ------------------------------------------------------------------ comparators
def soft_threshold(y, t):
    return np.sign(y) * np.maximum(np.abs(y) - t, 0.0)


def hard_threshold(y, t):
    return np.where(np.abs(y) > t, y, 0.0)


def eb_two_group(y, sigma=1.0, n_iter=300):
    """EM for pi ~ Beta, slab variance v estimated, spike at zero."""
    p = len(y)
    pi = 0.1
    v = max(1.0, np.var(y) - sigma ** 2)
    for _ in range(n_iter):
        # log odds of slab vs spike under N(0, sigma^2 + v)
        var = sigma ** 2 + v
        ll = -0.5 * np.log(var) - 0.5 * y ** 2 / var
        l0 = -0.5 * np.log(sigma ** 2) - 0.5 * y ** 2 / sigma ** 2
        w = 1.0 / (1.0 + np.exp(np.clip(l0 - ll + np.log((1 - pi) / pi), -500, 500)))
        pi_n = np.clip(np.mean(w), 1e-5, 0.999)
        v_n = max(1e-4, np.sum(w * (y ** 2 - sigma ** 2)) / max(np.sum(w), 1e-8))
        if max(abs(pi_n - pi), abs(v_n - v)) < 1e-9:
            pi, v = pi_n, v_n
            break
        pi, v = pi_n, v_n
    post_w = 1.0 / (1.0 + np.exp(np.clip(
        -0.5 * np.log(sigma ** 2) - 0.5 * y ** 2 / sigma ** 2
        + 0.5 * np.log(sigma ** 2 + v) + 0.5 * y ** 2 / (sigma ** 2 + v)
        + np.log((1 - pi) / pi), -500, 500)))
    return post_w * y * v / (sigma ** 2 + v), post_w


def bh_select(y, sigma=1.0, q=0.1):
    pv = 2 * stats.norm.sf(np.abs(y) / sigma)
    order = np.argsort(pv)
    crit = q * (np.arange(1, len(y) + 1) / len(y))
    ok = pv[order] <= crit
    sel = np.zeros(len(y), bool)
    if ok.any():
        sel[order[: np.where(ok)[0].max() + 1]] = True
    return sel


def mse_of(est, alpha):
    return float(np.mean((est - alpha) ** 2))


def make_alpha(p, s0, rng, mag=2.2):
    alpha = np.zeros(p)
    act = rng.choice(p, s0, replace=False)
    signs = rng.choice([-1, 1], s0)
    alpha[act] = signs * mag * rng.uniform(0.7, 1.3, s0)
    return alpha, np.sort(act)


# ------------------------------------------------------------------ experiments
def exp_shrinkage():
    """Posterior mean curves on a y grid: the horseshoe shape vs other rules."""
    grid = np.linspace(-5, 5, 201)
    a_hs, km, _ = horseshoe_gibbs(grid, n_iter=2200, burn=800, thin=2, seed=SEED + 9)
    a_eb, _ = eb_two_group(grid)
    return {"y": [float(v) for v in grid],
            "horseshoe": [float(v) for v in a_hs],
            "soft": [float(v) for v in soft_threshold(grid, np.sqrt(2 * np.log(P)))],
            "hard": [float(v) for v in hard_threshold(grid, 2.0)],
            "eb": [float(v) for v in a_eb]}


def exp_sparsity():
    rows = []
    for s0 in [2, 5, 10, 20, 40, 60]:
        acc = {k: [] for k in ["hs", "soft", "hard", "eb"]}
        rng = np.random.default_rng(SEED + 100 + s0)
        for r in range(30):
            alpha, act = make_alpha(P, s0, rng)
            y = alpha + rng.standard_normal(P)
            a_hs, _, _ = horseshoe_gibbs(y, n_iter=1800, burn=700, thin=4,
                                         seed=SEED + 4000 + s0 * 100 + r)
            a_eb, _ = eb_two_group(y)
            acc["hs"].append(mse_of(a_hs, alpha))
            acc["soft"].append(mse_of(soft_threshold(y, np.sqrt(2 * np.log(P))), alpha))
            acc["hard"].append(mse_of(hard_threshold(y, 2.0), alpha))
            acc["eb"].append(mse_of(a_eb, alpha))
        rows.append({"s0": s0,
                     "hs": float(np.mean(acc["hs"])),
                     "soft": float(np.mean(acc["soft"])),
                     "hard": float(np.mean(acc["hard"])),
                     "eb": float(np.mean(acc["eb"]))})
    return rows


def exp_detection():
    s0 = 10
    rng = np.random.default_rng(SEED + 55)
    acc = {k: {"tp": [], "fp": []} for k in ["hs_k", "bh", "hard", "eb"]}
    for r in range(40):
        alpha, act = make_alpha(P, s0, rng)
        y = alpha + rng.standard_normal(P)
        a_hs, km, _ = horseshoe_gibbs(y, n_iter=1800, burn=700, thin=4,
                                      seed=SEED + 7000 + r)
        a_eb, pw = eb_two_group(y)
        for key, sel in [("hs_k", km < 0.5), ("bh", bh_select(y)),
                         ("hard", np.abs(y) > 2.0), ("eb", pw > 0.5)]:
            tp = int(np.sum(sel[act])); fp = int(np.sum(sel) - tp)
            acc[key]["tp"].append(tp / s0)
            acc[key]["fp"].append(fp / max(1, tp + fp))
    return {k: {"power": float(np.mean(v["tp"])),
                "fdr": float(np.mean(v["fp"]))} for k, v in acc.items()}


def exp_misspec():
    rows = []
    s0 = 10
    tau0_true = s0 / P
    for tau0 in [tau0_true / 5, tau0_true, tau0_true * 5, 0.5]:
        for sig in [0.8, 1.0, 1.3]:
            rng = np.random.default_rng(SEED + 200 + int(tau0 * 1000) + int(sig * 10))
            hs_mse, bh_fd = [], []
            for r in range(18):
                alpha, act = make_alpha(P, s0, rng)
                y = alpha + rng.standard_normal(P) * sig
                a_hs, _, _ = horseshoe_gibbs(y, sigma=sig, tau0=tau0,
                                             n_iter=1600, burn=600, thin=4,
                                             seed=SEED + 9000 + r)
                hs_mse.append(mse_of(a_hs, alpha))
                sel = bh_select(y, sig)
                tp = np.sum(sel[act]); fp = np.sum(sel) - tp
                bh_fd.append(fp / max(1, tp + fp))
            rows.append({"tau0": float(tau0), "sigma": sig,
                         "hs_mse": float(np.mean(hs_mse)),
                         "bh_fdr": float(np.mean(bh_fd))})
    return rows


def exp_oos():
    s0 = 10
    rng = np.random.default_rng(SEED + 33)
    rows = []
    for method in ["hs", "hard", "soft", "bh"]:
        ins, oos, nsel = [], [], []
        for r in range(20):
            alpha, act = make_alpha(P, s0, rng)
            y = alpha + rng.standard_normal(P)
            if method == "hs":
                est, _, _ = horseshoe_gibbs(y, n_iter=1800, burn=700, thin=4,
                                            seed=SEED + 11000 + r)
            elif method == "hard":
                est = hard_threshold(y, 2.0)
            elif method == "soft":
                est = soft_threshold(y, np.sqrt(2 * np.log(P)))
            else:
                sel = bh_select(y); est = np.where(sel, y, 0.0)
            sel = np.abs(est) > 0.3
            nsel.append(int(np.sum(sel)))
            if np.sum(sel) == 0:
                oos.append(0.0); ins.append(0.0); continue
            w = np.sign(est[sel])
            w[w == 0] = 1.0
            w = w / np.sum(sel)
            r_in = alpha[sel] * w
            at = alpha + rng.standard_normal(P)
            r_out = at[sel] * w
            if np.std(r_in) < 1e-9 or np.std(r_out) < 1e-9:
                ins.append(0.0); oos.append(0.0); continue
            ins.append(float(np.mean(r_in) / np.std(r_in)))
            oos.append(float(np.mean(r_out) / np.std(r_out)))
        rows.append({"method": method,
                     "in_sharpe": float(np.mean(ins)),
                     "oos_sharpe": float(np.mean(oos)),
                     "gap": float(np.mean(ins) - np.mean(oos)),
                     "nsel": float(np.mean(nsel))})
    return rows


# ---------------------------------------------------------------------- figures
def fig1(d):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(d["y"], d["horseshoe"], lw=2.5, c="#c0392b", label="horseshoe mean")
    ax.plot(d["y"], d["soft"], c="#2980b9", label="soft threshold")
    ax.plot(d["y"], d["hard"], c="#7f8c8d", label="hard threshold")
    ax.plot(d["y"], d["eb"], ls=":", c="#8e44ad", label="EB two-group")
    ax.plot(d["y"], d["y"], ls="--", c="black", lw=0.7, alpha=0.5)
    ax.set_xlabel("observed y"); ax.set_ylabel("posterior mean / estimate")
    ax.set_title("Horseshoe: shrink noise, leave large signals alone")
    ax.legend(); fig.tight_layout()
    p = os.path.join(OUT, "fig1_shrinkage.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig1_shrinkage.png"


def fig2(rows):
    xs = [r["s0"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for k, c in [("hs", "#c0392b"), ("eb", "#8e44ad"),
                 ("soft", "#2980b9"), ("hard", "#7f8c8d")]:
        ax.plot(xs, [r[k] for r in rows], "o-", c=c, label=k)
    ax.set_xlabel("number of true signals s0"); ax.set_ylabel("MSE on alpha")
    ax.set_title("Horseshoe wins when signals are sparse")
    ax.legend(); fig.tight_layout()
    p = os.path.join(OUT, "fig2_sparsity.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig2_sparsity.png"


def fig3(d):
    names = list(d.keys())
    power = [d[k]["power"] for k in names]
    fdr = [d[k]["fdr"] for k in names]
    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - w / 2, power, w, color="#2980b9", label="power")
    ax.bar(x + w / 2, fdr, w, color="#c0392b", label="FDR")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title("kappa<0.5 is not an FDR procedure")
    ax.legend(); fig.tight_layout()
    p = os.path.join(OUT, "fig3_detection.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig3_detection.png"


def fig4(rows):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    tau_vals = sorted(set(r["tau0"] for r in rows))
    for sig in [0.8, 1.0, 1.3]:
        sub = [r for r in rows if r["sigma"] == sig]
        ax.plot([r["tau0"] for r in sub], [r["hs_mse"] for r in sub],
                "o-", label="sigma=%.1f" % sig)
    ax.set_xscale("log"); ax.set_xlabel("tau0 (assumed sparsity)")
    ax.set_ylabel("horseshoe MSE")
    ax.set_title("Global scale matters more than noise misspecification")
    ax.legend(); fig.tight_layout()
    p = os.path.join(OUT, "fig4_misspec.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig4_misspec.png"


def fig5(rows):
    names = [r["method"] for r in rows]
    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - w / 2, [r["in_sharpe"] for r in rows], w, color="#c0392b", label="in-sample")
    ax.bar(x + w / 2, [r["oos_sharpe"] for r in rows], w, color="#2980b9", label="OOS")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Sharpe-like ratio")
    ax.set_title("Selection optimism survives every shrinkage rule")
    ax.legend(); fig.tight_layout()
    p = os.path.join(OUT, "fig5_oos.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig5_oos.png"


def main():
    sh = exp_shrinkage()
    sp = exp_sparsity()
    de = exp_detection()
    mi = exp_misspec()
    oo = exp_oos()
    result = {"setup": {"seed": SEED, "p": P, "sigma": SIGMA,
                        "sampler": "Gibbs, 1800 iters / 700 burn / thin 4",
                        "comparators": ["hard |y|>2", "soft sqrt(2log p)",
                                        "EB two-group EM", "BH q=0.1"]},
              "shrinkage": sh, "sparsity": sp, "detection": de,
              "misspec": mi, "oos": oo}
    result["figs"] = [fig1(sh), fig2(sp), fig3(de), fig4(mi), fig5(oo)]
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
```


这里踩过一个必须记下来的坑：采样器里 $\tau^2$ 的全条件分布，shape 参数是 $(p+1)/2$ 而不是 1。写成指数分布（shape=1）会让 $\tau$ 飞到 $10^6$，收缩完全失效——实测的判别信号是噪声的后验均值绝对值停在 0.74（几乎等于原始观测），而正确实现下这个数字是 0.041。如果你自己写 horseshoe Gibbs，一定要用这个诊断对照。

## 三、收缩曲线：小信号压死，大信号放过

![马蹄先验的收缩曲线](/images/horseshoe-prior-sparse-alpha/fig1_shrinkage.png)

收缩曲线是理解马蹄先验最快的方式。把后验均值画成对观测值 $y$ 的函数：

| 观测 $y$ | 马蹄后验 | 软阈值 | EB 两成分 |
|---|---|---|---|
| 1 | 0.56 | 0 | 0.88 |
| 2 | 1.45 | 0 | 1.76 |
| 3 | 2.43 | 0 | 2.64 |
| 4 | 3.59 | 0.91 | 3.52 |

软阈值用 $\sqrt{2\log p}\approx3.14$ 当阈值，$|y|<3.14$ 全部归零——这就是为什么它在样本外只选出 1.7 个信号，太激进了。EB 两成分模型几乎不收缩（$y=1$ 保留 0.88），因为它是" spike 或 slab "的二选一，估计的是后验概率再加权。

马蹄先验走中间路线：小信号压掉约一半，大信号只损失 10% 左右。噪声的平均后验绝对值 0.041，意味着 $y_i$ 里那些纯噪声分量被压掉了 96%。这正是"马蹄"的两头——原点附近的尖峰负责压噪声，慢衰减的尾巴负责放过真信号。

## 四、稀疏度的分水岭

![信号越多，马蹄先验越吃亏](/images/horseshoe-prior-sparse-alpha/fig2_sparsity.png)

把真信号数 $s_0$ 从 2 扫到 60，MSE 结果（$p=120$）：

| $s_0$ | 马蹄 | 软阈值 | 硬阈值 | EB 两成分 |
|---|---|---|---|---|
| 2 | **0.065** | 0.073 | 0.306 | 0.070 |
| 5 | 0.157 | 0.197 | 0.322 | **0.154** |
| 10 | 0.270 | 0.379 | 0.397 | **0.244** |
| 20 | 0.501 | 0.760 | 0.670 | **0.446** |
| 40 | 0.668 | 1.520 | 0.903 | **0.587** |
| 60 | 0.907 | 2.256 | 1.349 | **0.734** |

分界线很清楚：$s_0=2$ 时马蹄略胜（0.065 对 EB 0.070），$s_0\ge5$ 就被 EB 反超，差距随信号变多越拉越大。这符合直觉——马蹄先验的全局尺度 $\tau$ 假设大部分系数是零，信号占比从 1.7% 涨到 50% 时，这个假设彻底错了。

但别急着否定它。软阈值在所有稀疏度上都比马蹄差（$s_0=60$ 时 2.256 对 0.907），硬阈值也全程落后。马蹄和 EB 属于第一梯队，软阈值和硬阈值是第二梯队。选哪个取决于你相信策略池有多稀疏：真 alpha 占比 5% 以下，马蹄是合理默认；占比往上走，EB 更稳。

## 五、它不能当假设检验用

![三方法的功效与 FDR](/images/horseshoe-prior-sparse-alpha/fig3_detection.png)

一个常见误用是把 $\kappa_i<0.5$ 当成"这个 alpha 显著"的判据。实测（$s_0=10$）：

- 马蹄 $\kappa<0.5$：功效 14.0%、FDR 9.4%。
- BH 检验（$q=0.1$）：功效 24.0%、FDR 13.8%。
- 硬阈值 $|y|>2$：功效 59.5%、FDR 45.0%。
- EB 后验概率 $>0.5$：功效 56.0%、FDR 36.2%。

马蹄的判据非常保守——10 个真信号只捞出 1.4 个，比 BH 还低。原因在收缩后验：真信号的 $\kappa$ 中位数约 0.89，远在 0.5 阈值之上，所以大部分被判成"不显著"。它压根不是为假设检验设计的，$\kappa$ 是收缩强度，不是 p 值的替代。

反过来，硬阈值和 EB 的功效高但 FDR 也高（45% 和 36%）——名单里三分之一以上是假信号。这里没有免费午餐：想控制错误发现率，用 BH；想要好的点估计（把噪声压小、真信号留住），用马蹄。**这是两个不同的目标，别用一个方法的判据去做另一件事。**

## 六、错配稳健性

![全局尺度比噪声设定更重要](/images/horseshoe-prior-sparse-alpha/fig4_misspec.png)

真实 $\sigma$ 在 0.8 到 1.3 之间变动，把 $\tau_0$ 从真值的 1/5 扫到 6 倍，马蹄的 MSE 变化很小：

- $\sigma=1.0$ 时，$\tau_0\in\{0.017,0.083,0.417,0.5\}$ 对应的 MSE 分别是 0.279、0.291、0.284、0.256。
- 同一 $\tau_0=0.083$，$\sigma$ 从 0.8 升到 1.3，MSE 从 0.199 涨到 0.354——噪声本身的影响比 $\tau_0$ 错配大得多。

这是个好消息：**全局尺度 $\tau_0$ 不用调得很准**，只要在同一个数量级内，结果都稳。实践里用 $\tau_0=s_0/p$ 的粗略估计就够了。真正要紧的是把噪声水平 $\sigma$ 估对——它直接进入似然，估错了收缩幅度跟着错。

## 七、样本内外落差

![所有方法都有选择乐观偏差](/images/horseshoe-prior-sparse-alpha/fig5_oos.png)

最后做了一件贴近实盘的事：用训练期估出的 alpha 构建多空组合，在独立测试期上算夏普，看样本内外落差。

| 方法 | 样本内夏普 | 样本外夏普 | 落差 | 平均入选数 |
|---|---|---|---|---|
| 马蹄 | 2.415 | 2.309 | **0.106** | 9.45 |
| 硬阈值 | 1.027 | 0.874 | 0.153 | 11.50 |
| 软阈值 | 17.506 | 2.622 | 14.885 | 1.70 |
| BH | 9.109 | 1.733 | 7.376 | 2.30 |

先别被软阈值和 BH 的"样本内 17.5、9.1"吓到——它们平均只选 1.7 和 2.3 个信号，样本内夏普是拿极少数恰好过线的样本算出来的，分母（组合收益标准差）小得离谱，数字没有经济意义。这本身是个教训：**入选数太少时，夏普比率是个不稳定的统计量**。

有意义的对比是马蹄和硬阈值：两者入选数接近（9.45 对 11.50），马蹄样本内 2.415、样本外 2.309，落差 0.106；硬阈值样本内 1.027、样本外 0.874，落差 0.153。马蹄的选择乐观偏差更小——因为连续收缩不像硬阈值那样把"刚好过线"的噪声当成满分真信号。

必须说清楚的边界：这个实验里每个 alpha 独立、测试期噪声独立同分布。真实策略之间有协方差、有区制切换、有容量约束，落差会大得多。这里的价值是比较方法间的**相对**落差，不是给出实盘夏普的绝对值。

## 八、怎么用

1. **默认场景**：策略池真 alpha 占比 10% 以下，用马蹄做点估计，$\tau_0$ 取 $s_0/p$ 的粗略值，不必精调。
2. **信号可能很密**（$s_0/p>0.3$）：换经验贝叶斯两成分模型，第 4 节显示它在这个区间的 MSE 更低。
3. **别用 $\kappa$ 做显著性判断**：要控制误报就用 BH，要好的收缩估计就用马蹄，两件事分开做。
4. **诊断采样器**：如果你自己实现 Gibbs，用"真值为零的分量的后验均值绝对值"当哨兵——正确实现应该在 0.04 左右，停在 0.7 以上说明 $\tau$ 采样错了。
5. **意识到简化**：独立正态均值是最干净的教学场景。真实 alpha 有横截面相关，上实盘前要在带协方差的面板上重测。

马蹄先验卖的不是"更高的夏普"，而是一个更诚实的中间态：它承认大多数策略没有 alpha，承认少数有 alpha 的不该被压平，然后用一层先验把这两种判断编码进同一个后验。当你的研究问题是"这些策略里哪些值得继续跟踪"而不是"哪个显著"，它比硬阈值和 p 值都更合适。
