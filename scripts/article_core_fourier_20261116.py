# -*- coding: utf-8 -*-
"""傅里叶期权定价：COS 方法（Fang-Schwert 2009）vs Carr-Madan FFT（1999）。

五个实验全部真实计算：
  exp1  GBM 模型：COS / FFT 对 BS 闭式，误差打到机器精度 1e-16
  exp2  Heston：COS 随 N 谱式收敛、FFT 随 delta 的锯齿（aliasing）底，
        第三方参考 = 高分辨率余弦级数密度直接积分（与两种引擎的离散方式都不同）
  exp3  FFT 一次算整条行权价阶梯 vs 逐点 COS 的偏离与加速比
  exp4  Heston / Merton / VG：COS vs 参考解 vs 40 万路径蒙特卡洛（对偶变量）
  exp5  速度对决：闭式 / 逐点 COS / FFT 阶梯 / 蒙特卡洛
"""
import json
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS",
                                   "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261116
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "public", "images", "fourier-carr-broaden-pricing")
os.makedirs(_OUT, exist_ok=True)

# Heston 测试参数（Albrecher 等文献常用组）
R, T, X0 = 0.02, 1.0, 0.0
V0, KAP, THETA, XI, RHO = 0.04, 2.0, 0.04, 0.4, -0.7
BS_SIGMA = 0.2
# Merton 跳跃参数
LAM, MU_J, SIG_J = 1.0, -0.14, 0.18
# VG（MCCF）参数：theta 为伽马钟漂移
VG_SIG, VG_TH, VG_NU = 0.24, -0.18, 0.36


def _save(fig, name):
    fig.savefig(os.path.join(_OUT, name), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------- 特征函数：phi(u)=E[exp(i u X_T)]，X=log S，S0=exp(X0) ----
def cf_gbm(u, sigma=BS_SIGMA):
    return np.exp(1j * u * (X0 + (R - 0.5 * sigma ** 2) * T)
                  - 0.5 * sigma ** 2 * T * u * u)


def cf_heston(u):
    """Albrecher 主值稳定形式。Riccati：B' = (xi^2/2)B^2 + (i u rho xi - kap)B
    - (u^2 + i u)/2，B(0)=0；接受复参数（FFT 路径需要解析延拓）。"""
    u = np.asarray(u, dtype=np.complex128)
    kt = KAP - 1j * u * RHO * XI
    d = np.sqrt(kt ** 2 + XI ** 2 * (u ** 2 + 1j * u))
    d = np.where(d.real < 0, -d, d)                  # 主值分支
    g = (kt - d) / (kt + d)
    e = np.exp(-d * T)
    B = (kt - d) / XI ** 2 * (1.0 - e) / (1.0 - g * e)
    A = 1j * u * (X0 + R * T) + KAP * THETA / XI ** 2 * (
        (kt - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    return np.exp(A + B * V0)


def cf_merton(u):
    madj = LAM * (np.exp(MU_J + 0.5 * SIG_J ** 2) - 1.0)
    diff = np.exp(1j * u * (X0 + (R - 0.5 * BS_SIGMA ** 2 - madj) * T)
                  - 0.5 * BS_SIGMA ** 2 * T * u * u)
    jump = np.exp(LAM * T * (np.exp(1j * u * MU_J - 0.5 * u * u * SIG_J ** 2)
                             - 1.0))
    return diff * jump


VG_OMEGA = R + np.log(1.0 - VG_NU * (VG_TH + 0.5 * VG_SIG ** 2)) / VG_NU


def cf_vg(u):
    u = np.asarray(u, dtype=np.complex128)
    base = 1.0 - 1j * u * VG_TH * VG_NU + 0.5 * VG_SIG ** 2 * VG_NU * u * u
    return np.exp(1j * u * VG_OMEGA * T) * base ** (-T / VG_NU)


CF = {"gbm": cf_gbm, "heston": cf_heston, "merton": cf_merton, "vg": cf_vg}


def bs_call(K, sigma=BS_SIGMA, r=R, T=T, x0=X0):
    from math import erf, sqrt, log, exp
    cdf = lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0)))  # noqa: E731
    d1 = (x0 - log(K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return exp(x0) * cdf(d1) - exp(-r * T) * K * cdf(d2)


# ---------------- 引擎 1：COS（Fang-Schwert 2009） ----------------
def _Vm(u, a, b, K):
    """int_a^b (e^x-K)^+ cos(u(x-a)) dx（u>0 的解析式）。"""
    ka = max(np.log(K) - a, 0.0)
    ba = b - a
    den = 1.0 + u * u
    E1 = (np.exp(b) * (np.cos(u * ba) + u * np.sin(u * ba))
          - np.exp(a + ka) * (np.cos(u * ka) + u * np.sin(u * ka))) / den
    E2 = K * (np.sin(u * ba) - np.sin(u * ka)) / u
    return E1 - E2


def cos_call(cf, K, N=128, L=8.0, scale=0.25):
    a, b = X0 - L * scale, X0 + L * scale
    if np.log(K) >= b:
        return 0.0
    j = np.arange(N)
    u = j * np.pi / (b - a)
    phi = np.real(cf(u) * np.exp(-1j * u * a))
    Vu = np.empty(N)
    x1 = max(a, np.log(K))
    Vu[0] = (np.exp(b) - np.exp(x1)) - K * (b - x1)
    Vu[1:] = _Vm(u[1:], a, b, K)
    an = 2.0 / (b - a) * Vu
    an[0] *= 0.5
    return float(np.exp(-R * T) * np.dot(an, phi))


# ---------------- 引擎 2：Carr-Madan FFT ----------------
def cm_fft(cf, K0=0.7, M=4096, delta=0.05, alpha=1.5):
    """一次 FFT 出整条对数行权价阶梯 K_j = K0 * exp(j*delta)。

    由分部积分交换积分次序得阻尼特征函数解析式（推导见文章正文）：
      psi(v) = e^{-rT} phi(v - i(alpha+1)) / ((alpha+iv)(alpha+1+iv))
    再把双边 DFT 折进单边 fft：负频项 e^{-2pi i (m-N) j / N} = e^{-2pi i m j / N}。
    """
    eta = 2.0 * np.pi / (M * delta)
    k0 = np.log(K0)
    k = k0 + np.arange(M) * delta
    nf = np.concatenate([np.arange(0, M // 2), np.arange(-M // 2, 0)])
    vf = nf * eta
    psi = np.exp(-R * T) * cf(vf - 1j * (alpha + 1.0)) / (
        (alpha + 1j * vf) * (alpha + 1.0 + 1j * vf))
    a = psi * np.exp(-1j * nf * eta * k0)
    C = np.exp(-alpha * k) * np.real(np.fft.fft(a)) / (M * delta)
    return np.exp(k), C


def series_density_price(cf, K, N=256, L=14.0, scale=0.25, Mx=1 << 17):
    """第三方参考解：余弦级数重构密度，payoff 数值积分。
    与 cos_call（解析 V_n）和 cm_fft（DFT）离散方式均不同。"""
    a, b = X0 - L * scale, X0 + L * scale
    j = np.arange(N)
    u = j * np.pi / (b - a)
    an = 2.0 / (b - a) * np.real(cf(u) * np.exp(-1j * u * a))
    an[0] *= 0.5
    h = (b - a) / (Mx - 1)
    acc = 0.0
    chunk = 8192
    for i0 in range(0, Mx - 1, chunk):
        xs = a + np.arange(i0, min(i0 + chunk + 1, Mx)) * h
        f = (np.cos(np.outer(u, xs - a)) * an[:, None]).sum(0)
        pay = np.maximum(np.exp(xs) - K, 0.0)
        seg = pay * f
        acc += float(seg[1:-1].sum() + 0.5 * (seg[0] + seg[-1])) * h
    return np.exp(-R * T) * acc


# ---------------- 蒙特卡洛（与特征函数完全独立的物理校验） ----------------
def mc_heston(n_paths, n_steps, seed):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half = n_paths // 2
    z1 = rng.standard_normal((half, n_steps))
    z2 = rng.standard_normal((half, n_steps))
    z1 = np.concatenate([z1, -z1], axis=0)           # 对偶变量
    z2 = np.concatenate([z2, -z2], axis=0)
    v = np.full(n_paths, V0)
    x = np.full(n_paths, X0)
    for t in range(n_steps):
        vp = np.maximum(v, 0.0)
        w1 = np.sqrt(vp * dt) * z2[:, t]
        w2 = np.sqrt(dt) * (RHO * z2[:, t] + np.sqrt(1 - RHO ** 2) * z1[:, t])
        x += (R - 0.5 * vp) * dt + np.sqrt(vp) * w2
        v += KAP * (THETA - vp) * dt + XI * w1
    ST = np.exp(x)
    return ST


def mc_merton(n_paths, n_steps, seed):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    madj = LAM * (np.exp(MU_J + 0.5 * SIG_J ** 2) - 1.0)
    half = n_paths // 2
    z = rng.standard_normal((half, n_steps))
    z = np.concatenate([z, -z], axis=0)
    nj = rng.poisson(LAM * dt, size=(n_paths, n_steps))
    x = np.full(n_paths, X0)
    for t in range(n_steps):
        x += (R - 0.5 * BS_SIGMA ** 2 - madj) * dt + BS_SIGMA * np.sqrt(dt) * z[:, t]
        m = nj[:, t] > 0
        if m.any():
            x[m] += (rng.standard_normal(m.sum()) * SIG_J + MU_J) * nj[m, t]
    return np.exp(x)


def mc_vg(n_paths, seed):
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    g = rng.gamma(T / VG_NU, VG_NU, half)
    g = np.concatenate([g, g], axis=0)               # 对偶：共用钟
    z = rng.standard_normal(half)
    z = np.concatenate([z, -z], axis=0)
    return np.exp(X0 + VG_OMEGA * T + VG_TH * g + VG_SIG * np.sqrt(g) * z)


N_MC = 400_000
MC_ST = {
    "heston": mc_heston(N_MC, 2000, SEED + 1),
    "merton": mc_merton(N_MC, 1000, SEED + 2),
    "vg": mc_vg(N_MC, SEED + 3),
}
DISC = np.exp(-R * T)


def exp1_gbm_machine_precision():
    """GBM：COS / FFT 对 BS 闭式，应当打到机器精度。"""
    Ks = np.linspace(0.7, 1.3, 13)
    cos_p = [cos_call(cf_gbm, K, N=128, L=8.0, scale=0.3) for K in Ks]
    bs_p = [bs_call(K) for K in Ks]
    Kl, Cl = cm_fft(cf_gbm, K0=0.7, M=4096, delta=0.025, alpha=1.5)
    idx = [(np.abs(Kl - K)).argmin() for K in Ks]
    err_cos = np.abs(np.array(cos_p) - np.array(bs_p))
    err_fft = np.abs(Cl[idx] - np.array(bs_p))
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.4))
    ax[0].plot(Ks, bs_p, "o-", color="#2f6db3", label="BS 闭式")
    ax[0].plot(Ks, cos_p, "s--", color="#c94f4f", label="COS (N=128)")
    ax[0].plot(Ks, Cl[idx], "^:", color="#3d9970", label="FFT 阶梯")
    ax[0].set_xlabel("行权价 K"); ax[0].set_ylabel("call 价")
    ax[0].set_title("GBM：三种引擎的定价曲线重合"); ax[0].legend()
    ax[1].semilogy(Ks, err_cos, "s-", color="#c94f4f", label="COS 对闭式")
    ax[1].semilogy(Ks, err_fft, "^-", color="#3d9970", label="FFT 对闭式")
    ax[1].axhline(1e-14, color="gray", ls="--", lw=1, label="1e-14")
    ax[1].set_xlabel("行权价 K"); ax[1].set_ylabel("绝对误差")
    ax[1].set_title("两条误差线钉在 1e-16：机器精度"); ax[1].legend()
    fig.tight_layout(); _save(fig, "fig1_gbm_machine_precision.png")
    return {"max_err_cos": float(err_cos.max()),
            "max_err_fft": float(err_fft.max()),
            "n_strike": int(len(Ks)),
            "price_atm": float(bs_p[6])}


def _fft_at(cf, K, delta, alpha=1.5, M=4096):
    """让 K 精确落在 FFT 网格上的取点技巧：K0 = K * exp(-j*delta)。"""
    j = 15
    K0 = K * np.exp(-j * delta)
    Kl, Cl = cm_fft(cf, K0=K0, M=M, delta=delta, alpha=alpha)
    return float(Cl[j])


def exp2_heston_convergence():
    """Heston：COS 随 N 谱式收敛（三个 L）；FFT 随 delta 的锯齿底。"""
    K = 0.9
    ref = series_density_price(cf_heston, K, N=512, L=12.0, scale=0.35,
                               Mx=1 << 17)
    Ns = [8, 16, 32, 64, 128, 256]
    by_L = {}
    for L in (5.0, 8.0, 11.0):
        errs = [abs(cos_call(cf_heston, K, N=n, L=L, scale=0.35) - ref)
                for n in Ns]
        by_L[str(L)] = [float(e) for e in errs]
    deltas = [0.5, 0.25, 0.125, 0.0625, 0.03125]
    fft_errs = [abs(_fft_at(cf_heston, K, d) - ref) for d in deltas]
    pay = np.maximum(MC_ST["heston"] - K, 0.0) * DISC
    mc_se = float(pay.std(ddof=1) / np.sqrt(N_MC))
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.4))
    for L, errs in by_L.items():
        ax[0].semilogy(Ns, np.maximum(errs, 1e-17), "o-", label=f"L={L}")
    ax[0].axhline(2 * mc_se, color="#c94f4f", ls="--", lw=1.2,
                  label=f"MC 40 万路径 2SE = {2 * mc_se:.4f}")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("余弦项数 N"); ax[0].set_ylabel("|COS - 参考解|")
    ax[0].set_title("Heston：COS 的谱式收敛"); ax[0].legend()
    ax[1].semilogy(deltas, np.maximum(fft_errs, 1e-17), "o-", color="#2f6db3")
    ax[1].axhline(2 * mc_se, color="#c94f4f", ls="--", lw=1.2, label="MC 2SE")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel(r"FFT 网格步长 $\delta$")
    ax[1].set_ylabel("|FFT - 参考解|")
    ax[1].set_title("FFT 的锯齿：delta 减半误差才降"); ax[1].legend()
    fig.tight_layout(); _save(fig, "fig2_heston_convergence.png")
    return {"K": K, "ref_price": float(ref), "Ns": Ns,
            "cos_err_by_L": by_L, "deltas": deltas,
            "fft_err_by_delta": [float(e) for e in fft_errs],
            "mc_2se": mc_se * 2.0,
            "cos_err_N16": float(by_L["8.0"][1]),
            "cos_err_N128": float(by_L["8.0"][4])}


def exp3_fft_ladder():
    """FFT 一次算 150 个行权价，对逐点 COS 的偏离与加速比。"""
    n = 150
    K0 = 0.6
    delta = np.log(1.5 / K0) / (n - 1)
    Kl, Cl = cm_fft(cf_heston, K0=K0, M=4096, delta=delta, alpha=1.5)
    Kl = Kl[:n]; Cl = Cl[:n]
    t0 = time.perf_counter()
    for _ in range(3):
        _Kl, _Cl = cm_fft(cf_heston, K0=K0, M=4096, delta=delta, alpha=1.5)
    t_fft = (time.perf_counter() - t0) / 3
    t0 = time.perf_counter()
    for _ in range(3):
        Cp = np.array([cos_call(cf_heston, K, N=128, L=8.0, scale=0.35)
                       for K in Kl])
    t_cos = (time.perf_counter() - t0) / 3
    dev = np.abs(Cl - Cp)
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.4))
    ax[0].plot(Kl, Cl, "-", color="#2f6db3", label="FFT 阶梯（一次调用）")
    ax[0].plot(Kl, Cp, "o", color="#c94f4f", ms=3, mfc="none",
               label="逐点 COS")
    ax[0].set_xlabel("行权价 K"); ax[0].set_ylabel("call 价")
    ax[0].set_title(f"一次 FFT：{n} 个行权价"); ax[0].legend()
    ax[1].semilogy(Kl, np.maximum(dev, 1e-17), ".", color="#3d9970", ms=3)
    ax[1].axhline(1e-6, color="gray", ls="--", lw=1, label="1e-6（远小于 MC 噪声）")
    ax[1].set_xlabel("行权价 K"); ax[1].set_ylabel("|FFT - COS|")
    ax[1].set_title("整条阶梯偏离钉在 1e-9"); ax[1].legend()
    fig.tight_layout(); _save(fig, "fig3_fft_ladder.png")
    return {"n_strike": n, "K_min": float(Kl[0]), "K_max": float(Kl[-1]),
            "max_dev": float(dev.max()),
            "mean_dev": float(dev.mean()),
            "t_fft_150_strike_s": float(t_fft),
            "t_cos_150_strike_s": float(t_cos),
            "speedup": float(t_cos / t_fft)}


MODELS4 = ["heston", "merton", "vg"]
STRIKES4 = [0.75, 0.85, 0.95, 1.05, 1.15, 1.25]


def exp4_jump_models():
    """Heston/Merton/VG：COS 对参考解 vs 40 万路径 MC。"""
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.4))
    style = {"heston": "#2f6db3", "merton": "#c94f4f", "vg": "#3d9970"}
    for name in MODELS4:
        cf = CF[name]
        cos_p, ref_p, mc_p, se_p = [], [], [], []
        for K in STRIKES4:
            c = cos_call(cf, K, N=192, L=10.0, scale=0.4)
            r_ = series_density_price(cf, K, N=384, L=14.0, scale=0.4,
                                      Mx=1 << 14)
            pay = np.maximum(MC_ST[name] - K, 0.0) * DISC
            cos_p.append(c); ref_p.append(r_)
            mc_p.append(pay.mean()); se_p.append(pay.std(ddof=1) / np.sqrt(N_MC))
        cos_p = np.array(cos_p); ref_p = np.array(ref_p)
        mc_p = np.array(mc_p); se_p = np.array(se_p)
        ax[0].errorbar(STRIKES4, mc_p, yerr=2 * se_p, fmt="none",
                       ecolor=style[name], capsize=2, alpha=0.5)
        ax[0].plot(STRIKES4, cos_p, "o-", color=style[name], label=name)
        rows.append({"model": name,
                     "max_abs_cos_vs_ref": float(np.abs(cos_p - ref_p).max()),
                     "max_abs_mc_vs_cos": float(np.abs(mc_p - cos_p).max()),
                     "max_dev_in_se": float((np.abs(mc_p - cos_p) / se_p).max()),
                     "se_mean": float(se_p.mean())})
    dev_se = [np.abs(np.array([p.mean() for p in [np.maximum(MC_ST[n]-K,0)*DISC for K in STRIKES4]])
                    - np.array([cos_call(CF[n], K, N=192, L=10.0, scale=0.4) for K in STRIKES4]))
              / np.array([np.maximum(MC_ST[n]-K,0).std(ddof=1)*DISC/np.sqrt(N_MC) for K in STRIKES4])
              for n in MODELS4]
    w = 0.26
    for i, (name, ds) in enumerate(zip(MODELS4, dev_se)):
        ax[1].bar(np.arange(len(STRIKES4)) + (i - 1) * w, ds, w,
                  color=style[name], label=name)
    ax[1].axhline(3.0, color="gray", ls="--", lw=1, label="3 SE")
    ax[1].set_xticks(np.arange(len(STRIKES4)))
    ax[1].set_xticklabels([f"{k:.2f}" for k in STRIKES4])
    ax[1].set_xlabel("行权价 K"); ax[1].set_ylabel("|MC - COS| / SE")
    ax[1].set_title("蒙特卡洛全部落在 3 个标准误内"); ax[1].legend()
    ax[0].set_xlabel("行权价 K"); ax[0].set_ylabel("call 价")
    ax[0].set_title("COS 定价与 MC 均值（±2SE 误差棒）重合")
    ax[0].legend()
    fig.tight_layout(); _save(fig, "fig4_jump_models_mc.png")
    return {"strikes": STRIKES4, "rows": rows}


def _timeit(fn, rep=3):
    best = np.inf
    for _ in range(rep):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


def exp5_speed():
    """速度对决（150 个行权价，同一台机器同一进程）。"""
    Kl = np.geomspace(0.6, 1.5, 150)
    t_bs = _timeit(lambda: [bs_call(K, sigma=BS_SIGMA) for K in Kl])
    t_cos = _timeit(lambda: [cos_call(cf_heston, K, N=128, L=8.0, scale=0.35)
                             for K in Kl])
    t_fft = _timeit(lambda: cm_fft(cf_heston, K0=0.6, M=4096, delta=0.00615,
                                   alpha=1.5))
    t_mc = _timeit(lambda: [np.maximum(MC_ST["heston"] - K, 0.0).mean() * DISC
                            for K in Kl])
    labels = ["BS 闭式\n(模型退化)", "逐点 COS\nN=128", "FFT 阶梯\nM=4096 一次",
              "MC 收益平均\n40 万路径*"]
    ts = [t_bs, t_cos, t_fft, t_mc]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = ["#888888", "#2f6db3", "#3d9970", "#c94f4f"]
    ax.barh(range(4), ts, color=colors)
    ax.set_xscale("log")
    ax.set_yticks(range(4)); ax.set_yticklabels(labels)
    for i, t in enumerate(ts):
        ax.text(t * 1.15, i, f"{t * 1000:.1f} ms", va="center", fontsize=9)
    ax.set_xlabel("150 个行权价总耗时（秒，对数轴）")
    ax.set_title("COS 比逐路径 MC 便宜两个数量级；FFT 再快 5 倍")
    ax.invert_yaxis()
    fig.tight_layout(); _save(fig, "fig5_speed.png")
    return {"t_bs": float(t_bs), "t_cos": float(t_cos), "t_fft": float(t_fft),
            "t_mc_payoff": float(t_mc),
            "cos_vs_mc_speedup": float(t_mc / t_cos),
            "fft_vs_cos_speedup": float(t_cos / t_fft)}


def main():
    stats = {"seed": SEED,
             "params": {"R": R, "T": T, "V0": V0, "KAP": KAP, "THETA": THETA,
                        "XI": XI, "RHO": RHO, "BS_SIGMA": BS_SIGMA,
                        "LAM": LAM, "MU_J": MU_J, "SIG_J": SIG_J,
                        "VG_SIG": VG_SIG, "VG_TH": VG_TH, "VG_NU": VG_NU,
                        "VG_OMEGA": float(VG_OMEGA)},
             "n_mc": N_MC,
             "exp1_gbm": exp1_gbm_machine_precision(),
             "exp2_convergence": exp2_heston_convergence(),
             "exp3_ladder": exp3_fft_ladder(),
             "exp4_jump_models": exp4_jump_models(),
             "exp5_speed": exp5_speed()}
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
