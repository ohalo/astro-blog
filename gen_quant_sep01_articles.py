#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 2026-09-01 两篇量化文章配图 + 核心数值（numpy/scipy 合成，固定 seed 可复现）。
  A. bayesian-option-pricing-mcmc       贝叶斯期权定价 MCMC：给奇异期权定价
  B. jump-diffusion-neural-pricing      跳跃扩散神经网络定价：用深度学习逼近含跳资产价格
所有图表均为真实计算图，数值固定随机种子可复现。
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

for f in ["PingFang SC", "Heiti SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130
plt.rcParams["figure.autolayout"] = True

rng_seed = 20260901


def ncdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def ncdf_scalar(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ============================================================
# 文章 A：贝叶斯期权定价 MCMC（以亚式看涨期权为例，无解析解）
# ============================================================
def gen_article_a():
    slug = "bayesian-option-pricing-mcmc"
    OUT = f"public/images/{slug}"
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(rng_seed)

    # —— 1. 生成「观测到的」标的日度对数收益（真实模型：GBM）——
    S0, mu_true, sig_true, T, n_obs = 100.0, 0.08, 0.20, 1.0, 252
    dt = 1.0 / n_obs
    true_drift = (mu_true - 0.5 * sig_true ** 2) * dt
    true_sd = sig_true * np.sqrt(dt)
    rets = rng.normal(true_drift, true_sd, n_obs)

    # 频率学派点估计（MLE）
    sig_mle = rets.std(ddof=1) / np.sqrt(dt)
    mu_mle = rets.mean() / dt + 0.5 * sig_mle ** 2

    # —— 2. 贝叶斯后验：对 (mu, log sigma) 跑随机游动 Metropolis ——
    def loglik(mu, sig):
        m = (mu - 0.5 * sig ** 2) * dt
        s = sig * np.sqrt(dt)
        return np.sum(-0.5 * ((rets - m) / s) ** 2 - np.log(s))

    def logprior(mu, sig):
        lp = -0.5 * ((mu - 0.05) / 0.5) ** 2          # mu ~ N(0.05, 0.5)
        lp += -0.5 * (sig / 0.3) ** 2 + math.log(sig)  # sigma ~ HalfNormal(0.3) + Jacobian
        return lp

    def logpost(p):
        mu, ls = p
        sig = math.exp(ls)
        if sig <= 0:
            return -1e18
        return loglik(mu, sig) + logprior(mu, sig)

    p0 = np.array([0.05, math.log(0.2)])
    prop_cov = np.diag([0.02, 0.012])
    n_iters = 20000
    chain = np.zeros((n_iters, 2))
    lp_cur = logpost(p0)
    cur = p0.copy()
    n_acc = 0
    for i in range(n_iters):
        prop = cur + rng.multivariate_normal([0, 0], prop_cov)
        lp_prop = logpost(prop)
        if math.log(rng.random()) < (lp_prop - lp_cur):
            cur, lp_cur = prop, lp_prop
            n_acc += 1
        chain[i] = cur
    acc_rate = n_acc / n_iters

    burn, thin = 4000, 8
    post = chain[burn::thin]
    mu_post = post[:, 0]
    sig_post = np.exp(post[:, 1])
    sig_mean_post = sig_post.mean()
    sig_true_val = sig_true
    sig_mle_val = sig_mle

    # —— 3. 定价：亚式（算术平均）看涨期权，无闭式解，必须蒙特卡洛 ——
    K, r = 100.0, 0.02

    def asian_call(mu, sig, n_steps=252, n_paths=400, seed=777):
        rgen = np.random.default_rng(seed)
        d = 1.0 / n_steps
        Z = rgen.standard_normal((n_paths, n_steps))
        logS = np.log(S0) + np.cumsum((mu - 0.5 * sig ** 2) * d + sig * np.sqrt(d) * Z, axis=1)
        avg = np.exp(logS).mean(axis=1)
        pay = np.maximum(avg - K, 0.0)
        return math.exp(-r * T) * pay.mean()

    # 后验预测：取 150 个后验样本，各自跑足量 MC（压低 MC 噪声，让 CI 反映参数不确定性）
    price_post = np.array([asian_call(mu_post[i], sig_post[i], n_paths=8000, seed=1000 + i)
                           for i in range(0, len(mu_post), len(mu_post) // 150)])
    price_mean = price_post.mean()
    ci_lo, ci_hi = np.quantile(price_post, [0.025, 0.975])
    # 点估计（plug-in）：用 MLE 参数，大样本 MC
    plug = asian_call(mu_mle, sig_mle, n_paths=20000, seed=42)

    # ================= 图 1：MCMC 链 trace =================
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.0))
    it = np.arange(len(mu_post))
    axes[0].plot(it, mu_post, lw=0.6, color="#2c6fbb")
    axes[0].set_title(r"MCMC 迹图：漂移 $\mu$")
    axes[0].set_ylabel(r"$\mu$")
    axes[1].plot(it, sig_post, lw=0.6, color="#c0392b")
    axes[1].axhline(sig_true, color="black", ls="--", lw=1, label=f"真值 {sig_true:.2f}")
    axes[1].set_title(r"MCMC 迹图：波动率 $\sigma$")
    axes[1].set_ylabel(r"$\sigma$")
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel("迭代（burn-in 后，间隔抽取）")
    fig.savefig(f"{OUT}/mcmc_trace.png"); plt.close(fig)

    # ================= 图 2：sigma 后验直方图 =================
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(sig_post, bins=45, color="#2c6fbb", alpha=0.85, density=True,
            edgecolor="white", linewidth=0.4)
    ax.axvline(sig_true, color="black", ls="--", lw=1.6, label=f"真值 {sig_true:.2f}")
    ax.axvline(sig_mle, color="#e67e22", ls="-.", lw=1.6, label=f"MLE {sig_mle:.3f}")
    ax.axvline(sig_mean_post, color="#27ae60", ls=":", lw=1.6,
               label=f"后验均值 {sig_mean_post:.3f}")
    ax.set_title(r"波动率 $\sigma$ 的后验分布（随机游动 Metropolis）")
    ax.set_xlabel(r"$\sigma$"); ax.set_ylabel("密度")
    ax.legend(fontsize=8)
    fig.savefig(f"{OUT}/posterior_sigma.png"); plt.close(fig)

    # ================= 图 3：期权价格后验预测分布 =================
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(price_post, bins=40, color="#8e44ad", alpha=0.85,
            edgecolor="white", linewidth=0.4, density=True)
    x_lo, x_hi = ci_lo, ci_hi
    ax.axvspan(x_lo, x_hi, color="#8e44ad", alpha=0.18,
               label=f"95% 可信区间 [{x_lo:.2f}, {x_hi:.2f}]")
    ax.axvline(plug, color="#e67e22", ls="-.", lw=1.8, label=f"点估计(plug-in) {plug:.2f}")
    ax.axvline(price_mean, color="black", ls="--", lw=1.4, label=f"后验均值 {price_mean:.2f}")
    ax.set_title("亚式看涨期权公允价的后验预测分布")
    ax.set_xlabel("期权价格"); ax.set_ylabel("密度")
    ax.legend(fontsize=8)
    fig.savefig(f"{OUT}/option_price_posterior.png"); plt.close(fig)

    # ================= 图 4：点估计 vs 贝叶斯（不确定性对比）=================
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ["Plug-in\n(点估计)", "贝叶斯\n(后验均值)"]
    vals = [plug, price_mean]
    yerr = [[0.0, price_mean - ci_lo], [0.0, ci_hi - price_mean]]
    ax.bar(bars, vals, color=["#e67e22", "#8e44ad"], width=0.5,
           yerr=yerr, capsize=8, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("期权价格")
    ax.set_title("点估计把参数不确定性『藏』掉了：贝叶斯给出可信区间")
    fig.savefig(f"{OUT}/bayes_vs_pointestimate.png"); plt.close(fig)

    print("=== ARTICLE A KEY NUMBERS ===")
    print(f"acc_rate={acc_rate:.3f}")
    print(f"sig_true={sig_true:.3f} sig_mle={sig_mle_val:.3f} sig_post_mean={sig_mean_post:.3f}")
    print(f"price_post_mean={price_mean:.3f} ci=[{ci_lo:.3f},{ci_hi:.3f}]")
    print(f"plug_in_price={plug:.3f}")
    print(f"n_price_samples={len(price_post)}")
    print("===========================")


# ============================================================
# 文章 B：跳跃扩散神经网络定价（Merton 跳扩散欧式看涨，闭式标签）
# ============================================================
def merton_call(S, K, r, T, sigma, lam, muJ, delta, n_max=12):
    """Merton(1976) 跳扩散欧式看涨闭式解，向量化于所有输入数组。"""
    m = np.exp(muJ + 0.5 * delta ** 2)
    k = m - 1.0
    price = np.zeros_like(sigma, dtype=float)
    for n in range(n_max + 1):
        sigma_n = np.sqrt(sigma ** 2 + n * delta ** 2 / T)
        r_n = r - lam * k + n * np.log(m) / T
        d1 = (np.log(S / K) + (r_n + 0.5 * sigma_n ** 2) * T) / (sigma_n * np.sqrt(T))
        d2 = d1 - sigma_n * np.sqrt(T)
        bs = S * ncdf(d1) - K * np.exp(-r_n * T) * ncdf(d2)
        pn = np.exp(-lam * T) * (lam * T) ** n / math.factorial(n)
        price += pn * bs
    return price


def build_mlp(n_in, n_h1, n_h2, n_out=1, seed=0):
    rgen = np.random.default_rng(seed)
    def w(shape):
        return rgen.standard_normal(shape) * np.sqrt(2.0 / shape[0])
    return [w((n_in, n_h1)), np.zeros(n_h1),
            w((n_h1, n_h2)), np.zeros(n_h2),
            w((n_h2, n_out)), np.zeros(n_out)]


def forward(params, X):
    W1, b1, W2, b2, W3, b3 = params
    h1 = np.maximum(0, X @ W1 + b1)
    h2 = np.maximum(0, h1 @ W2 + b2)
    out = h2 @ W3 + b3
    return h1, h2, out


def train_mlp(X, y, n_h1=32, n_h2=16, epochs=220, lr=0.015, batch=256, seed=1):
    N, n_in = X.shape
    # 全局打乱：避免 meshgrid 顺序导致训练/测试分布错位
    shuf = np.random.default_rng(seed + 99).permutation(N)
    X, y = X[shuf], y[shuf]
    params = build_mlp(n_in, n_h1, n_h2, 1, seed)
    W1, b1, W2, b2, W3, b3 = params
    m = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(seed + 5)
    loss_hist, val_hist = [], []
    n_train = int(N * 0.8)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:], y[n_train:]
    for ep in range(epochs):
        perm = rng.permutation(n_train)[:batch]
        Xb, yb = Xtr[perm], ytr[perm]
        # forward
        h1 = np.maximum(0, Xb @ W1 + b1)
        h2 = np.maximum(0, h1 @ W2 + b2)
        out = h2 @ W3 + b3
        loss = np.mean((out.ravel() - yb) ** 2)
        # backward
        dout = 2 * (out.ravel() - yb) / len(yb)
        dW3 = h2.T @ dout
        db3 = dout.sum()
        dh2 = (dout[:, None] @ W3.T) * (h2 > 0)
        dW2 = h1.T @ dh2
        db2 = dh2.sum(0)
        dh1 = (dh2 @ W2.T) * (h1 > 0)
        dW1 = Xb.T @ dh1
        db1 = dh1.sum(0)
        grads = [dW1, db1, dW2, db2, dW3, db3]
        for i, g in enumerate(grads):
            g = np.reshape(np.asarray(g), params[i].shape)
            m[i] = beta1 * m[i] + (1 - beta1) * g
            mhat = m[i] / (1 - beta1 ** (ep + 1))
            params[i] = params[i] - lr * mhat
        W1, b1, W2, b2, W3, b3 = params
        vo = forward(params, Xte)[2].ravel()
        val_hist.append(np.mean((vo - yte) ** 2))
        loss_hist.append(loss)
    return params, loss_hist, val_hist, (Xte, yte)


def gen_article_b():
    slug = "jump-diffusion-neural-pricing"
    OUT = f"public/images/{slug}"
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(rng_seed)

    S, r, q = 100.0, 0.02, 0.0
    # 参数网格
    sigma = np.linspace(0.10, 0.50, 7)
    lam = np.linspace(0.0, 3.0, 7)
    muJ = np.array([-0.10, 0.0])
    delta = np.array([0.05, 0.15])
    money = np.linspace(0.80, 1.20, 9)
    mat = np.linspace(0.25, 2.0, 6)
    g = np.meshgrid(sigma, lam, muJ, delta, money, mat, indexing="ij")
    G = [x.ravel() for x in g]
    sig_v, lam_v, muJ_v, del_v, mon_v, mat_v = G
    K_v = mon_v * S
    price = merton_call(S, K_v, r, mat_v, sig_v, lam_v, muJ_v, del_v, n_max=12)

    # 特征与标准化
    Xraw = np.column_stack([sig_v, lam_v, muJ_v, del_v, mon_v, mat_v])
    yraw = price
    Xmean, Xstd = Xraw.mean(0), Xraw.std(0)
    ymean, ystd = yraw.mean(), yraw.std()
    X = (Xraw - Xmean) / Xstd
    y = (yraw - ymean) / ystd

    # 训练（固定随机切分：按索引前 80%）
    params, loss_hist, val_hist, (Xte, yte) = train_mlp(
        X, y, n_h1=32, n_h2=16, epochs=220, lr=0.015, seed=3)

    pred = forward(params, Xte)[2].ravel()
    pred_price = pred * ystd + ymean
    true_price = yte * ystd + ymean
    ss_res = np.sum((pred_price - true_price) ** 2)
    ss_tot = np.sum((true_price - true_price.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((pred_price - true_price) ** 2))

    # ================= 图 1：Merton 价格曲面（moneyness × sigma）=================
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    # 固定 lam=0.5, muJ=-0.05, delta=0.10, T=1.0
    lam0, muJ0, del0, T0 = 0.5, -0.05, 0.10, 1.0
    ms = np.linspace(0.80, 1.20, 50)
    sg = np.linspace(0.10, 0.50, 50)
    MM, SS = np.meshgrid(ms, sg)
    surf = merton_call(S, MM * S, r, T0, SS, lam0, muJ0, del0, n_max=12)
    c = ax.contourf(MM, SS, surf, levels=24, cmap="viridis")
    fig.colorbar(c, ax=ax, label="看涨期权价格")
    ax.set_xlabel("价外程度  K/S₀（moneyness）")
    ax.set_ylabel("波动率 σ")
    ax.set_title("Merton 跳扩散欧式看涨价格曲面\n(λ=0.5, μ_J=−0.05, δ=0.10, T=1)")
    fig.savefig(f"{OUT}/merton_price_surface.png"); plt.close(fig)

    # ================= 图 2：NN 预测 vs 真值 =================
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    idx = np.random.default_rng(9).choice(len(true_price), size=min(1500, len(true_price)), replace=False)
    ax.scatter(true_price[idx], pred_price[idx], s=8, alpha=0.35, color="#2c6fbb")
    lo_, hi_ = true_price.min(), true_price.max()
    ax.plot([lo_, hi_], [lo_, hi_], color="black", ls="--", lw=1.4, label="完美拟合")
    ax.set_xlabel("真值（Merton 闭式）")
    ax.set_ylabel("神经网络预测")
    ax.set_title(f"测试集：预测 vs 真值  (R²={r2:.4f}, RMSE={rmse:.3f})")
    ax.legend(fontsize=8)
    fig.savefig(f"{OUT}/nn_pred_vs_true.png"); plt.close(fig)

    # ================= 图 3：学习到的跳跃溢价 =================
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    lam_grid = np.linspace(0.0, 3.0, 40)
    fix = dict(S=S, r=r, sigma=0.20, muJ=-0.05, delta=0.10, money=1.0, mat=1.0)
    truth = merton_call(fix["S"], fix["money"] * S, fix["r"], fix["mat"],
                        np.full_like(lam_grid, fix["sigma"]), lam_grid,
                        np.full_like(lam_grid, fix["muJ"]), np.full_like(lam_grid, fix["delta"]),
                        n_max=12)
    Xnn = np.column_stack([
        np.full_like(lam_grid, fix["sigma"]),
        lam_grid,
        np.full_like(lam_grid, fix["muJ"]),
        np.full_like(lam_grid, fix["delta"]),
        np.full_like(lam_grid, fix["money"]),
        np.full_like(lam_grid, fix["mat"]),
    ])
    Xnn = (Xnn - Xmean) / Xstd
    pred_nn = forward(params, Xnn)[2].ravel() * ystd + ymean
    ax.plot(lam_grid, truth, color="black", lw=2, label="Merton 闭式（真值）")
    ax.plot(lam_grid, pred_nn, color="#e67e22", lw=2, ls="--", label="神经网络预测")
    ax.set_xlabel("跳跃强度 λ")
    ax.set_ylabel("看涨期权价格")
    ax.set_title("网络学会了『跳跃溢价』：价格随 λ 单调上升")
    ax.legend(fontsize=8)
    fig.savefig(f"{OUT}/learned_jump_premium.png"); plt.close(fig)

    # ================= 图 4：训练 / 验证损失 =================
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(loss_hist, color="#2c6fbb", lw=1.6, label="训练损失")
    ax.plot(val_hist, color="#c0392b", lw=1.6, label="验证损失")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE（标准化后）")
    ax.set_title("MLP 训练收敛：训练/验证损失同步下降（无过拟合）")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    fig.savefig(f"{OUT}/training_loss.png"); plt.close(fig)

    print("=== ARTICLE B KEY NUMBERS ===")
    print(f"N_samples={len(price)}")
    print(f"R2={r2:.4f} RMSE={rmse:.4f}")
    print(f"price_range=[{price.min():.3f},{price.max():.3f}]")
    print(f"final_train_loss={loss_hist[-1]:.5f} final_val_loss={val_hist[-1]:.5f}")
    print("===========================")


if __name__ == "__main__":
    gen_article_a()
    gen_article_b()
    print("ALL IMAGES GENERATED")
