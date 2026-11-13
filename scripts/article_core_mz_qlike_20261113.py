# -*- coding: utf-8 -*-
"""波动率预测评估：Mincer-Zarnowitz 回归与 QLIKE 的现场对拍。

本文所有数字的唯一来源。运行本脚本会：
1. 现场生成 GARCH(1,1) 数据（固定随机种子，可复现）；
2. 手写高斯 QMLE 估计 GARCH 参数；
3. 画 5 张图到 public/images/mincer-zarnowitz-vol-forecast-eval/；
4. 把全部统计量写入同目录 results.json。
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261113
_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(
    os.path.dirname(_HERE), "public", "images", "mincer-zarnowitz-vol-forecast-eval"
)
os.makedirs(_OUT, exist_ok=True)

# 数据生成参数（日频 GARCH(1,1)）
OMEGA, ALPHA, BETA = 2e-6, 0.08, 0.90
LRV = OMEGA / (1.0 - ALPHA - BETA)  # 长期方差


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, name), dpi=150)
    plt.close(fig)


def simulate_garch(n, rng, burn=200):
    """返回 (r, h)，先丢弃 burn 期。"""
    r = np.empty(n + burn)
    h = np.empty(n + burn)
    h[0] = LRV
    z = rng.standard_normal(n + burn)
    r[0] = np.sqrt(h[0]) * z[0]
    for t in range(1, n + burn):
        h[t] = OMEGA + ALPHA * r[t - 1] ** 2 + BETA * h[t - 1]
        r[t] = np.sqrt(h[t]) * z[t]
    return r[burn:], h[burn:]


def garch_mle(r):
    """高斯 QMLE 估计 GARCH(1,1)。返回 (omega, alpha, beta)。"""
    r2 = r ** 2
    v0 = r2.var()

    def nll(p):
        om, a, b = p
        if om <= 0 or a < 0 or b < 0 or a + b >= 0.9999:
            return 1e10
        h = np.empty_like(r2)
        h[0] = v0
        for t in range(1, len(r2)):
            h[t] = om + a * r2[t - 1] + b * h[t - 1]
        if np.any(h <= 0):
            return 1e10
        return float(0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + r2 / h))

    best = None
    for a0 in (0.05, 0.10):
        for b0 in (0.85, 0.92):
            res = minimize(nll, x0=[1e-6, a0, b0], method="Nelder-Mead",
                           options={"maxiter": 4000, "xatol": 1e-10,
                                    "fatol": 1e-12})
            if best is None or res.fun < best.fun:
                best = res
    om, a, b = best.x
    return float(om), float(a), float(b)


def filter_garch(r2, om, a, b):
    """用给定参数沿已实现收益滤波出条件方差序列。"""
    h = np.empty_like(r2)
    h[0] = om / (1.0 - a - b) if a + b < 1 else r2.mean()
    for t in range(1, len(r2)):
        h[t] = om + a * r2[t - 1] + b * h[t - 1]
    return h


# ---------------------------------------------------------------- MZ 回归
def mz_regression(proxy, forecast, nw_lags=10):
    """MZ 回归 proxy = a + b*f。返回 (a, b, NW 标准误, t(b=1), R^2)。"""
    x = np.vstack([np.ones(len(forecast)), forecast]).T
    coef, *_ = np.linalg.lstsq(x, proxy, rcond=None)
    resid = proxy - x @ coef
    n, k = x.shape
    bread = np.linalg.inv(x.T @ x)
    s = np.zeros((k, k))
    for lag in range(nw_lags + 1):
        w = 1.0 - lag / (nw_lags + 1.0)
        g = (x[lag:] * resid[lag:, None]).T @ (x[: n - lag] * resid[: n - lag, None])
        s += w * (g + g.T) if lag > 0 else g
    v = bread @ s @ bread
    se_b = float(np.sqrt(max(v[1, 1], 0.0) * n / (n - k)))
    b = float(coef[1])
    t_b1 = float((b - 1.0) / se_b) if se_b > 0 else None
    sst = float(((proxy - proxy.mean()) ** 2).sum())
    r_squared = float(1.0 - (resid ** 2).sum() / sst) if sst > 0 else None
    r1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    return float(coef[0]), b, se_b, t_b1, r_squared, r1


# ---------------------------------------------------------------- 损失
def mse_loss(forecast, proxy):
    return float(np.mean((proxy - forecast) ** 2))


def qlike_loss(forecast, proxy):
    ratio = proxy / forecast
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def main():
    rng = np.random.default_rng(SEED)
    T_TRAIN, T_EVAL = 3000, 2000
    r_full, h_full = simulate_garch(T_TRAIN + T_EVAL, rng, burn=500)
    r_train, h_train = r_full[:T_TRAIN], h_full[:T_TRAIN]
    r_eval, h_eval = r_full[T_TRAIN:], h_full[T_TRAIN:]
    r2_eval = r_eval ** 2

    om_hat, a_hat, b_hat = garch_mle(r_train)
    # 各预测（一步 ahead，只用当期之前的信息）
    h_garch = filter_garch(r_full ** 2, om_hat, a_hat, b_hat)[T_TRAIN:]
    h_ewma = filter_garch(r_full ** 2, 0.0, 1.0 - 0.94, 0.94)[T_TRAIN:]
    h_const = np.full(T_EVAL, om_hat / (1.0 - a_hat - b_hat))
    h_oracle = h_eval
    forecasts = {
        "oracle": h_oracle,
        "garch": h_garch,
        "ewma": h_ewma,
        "const": h_const,
        "bias15": 1.15 * h_garch,
    }

    # ---- 单一切分：MZ 回归（两个 proxy）与损失
    mz = {}
    for pname, proxy in (("h_true", h_eval), ("r2", r2_eval)):
        for fname, fvec in forecasts.items():
            a, b, se, t, rsq, acf1 = mz_regression(proxy, fvec)
            mz[f"{pname}|{fname}"] = {"a": a, "b": b, "se_b": se, "t_b1": t,
                                      "R2": rsq, "resid_acf1": acf1}
    # r² proxy 的 R² 理论值：Var(h) / Var(r²)，与预测好坏无关
    eh = float(h_eval.mean())
    eh2 = float((h_eval ** 2).mean())
    vh = eh2 - eh ** 2
    r2_theory = vh / (3.0 * eh2 - eh ** 2)

    losses = {}
    for pname, proxy in (("h_true", h_eval), ("r2", r2_eval)):
        for fname, fvec in forecasts.items():
            losses[f"{pname}|mse|{fname}"] = mse_loss(fvec, proxy)
            losses[f"{pname}|qlike|{fname}"] = qlike_loss(fvec, proxy)

    # ---- 图1：预测路径对比
    sl = slice(1200, 1800)
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    idx = np.arange(T_EVAL)[sl]
    ax.plot(idx, np.abs(r_eval[sl]), color="#d0d7de", lw=0.7,
            label="|日收益|（噪声基准）")
    ax.plot(idx, np.sqrt(h_oracle[sl]), color="#d1242f", lw=1.6,
            label="真实条件波动（oracle）")
    ax.plot(idx, np.sqrt(h_garch[sl]), color="#1f6feb", lw=1.2,
            label="GARCH(1,1) QMLE 滤波")
    ax.plot(idx, np.sqrt(h_ewma[sl]), color="#8250df", lw=1.2, ls="--",
            label="EWMA (λ=0.94)")
    ax.set_xlabel("评估集交易日")
    ax.set_ylabel("日波动率")
    ax.set_title("图1 评估窗口上的波动率预测路径：EWMA 是滞后峰值的平滑复制")
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(alpha=0.3)
    _save(fig, "fig1_forecast_paths.png")

    # ---- 图2：真值 proxy 下的 MZ 散点
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharex=True, sharey=True)
    for ax, fname, color in ((axes[0], "oracle", "#d1242f"),
                             (axes[1], "garch", "#1f6feb")):
        fvec = forecasts[fname]
        sub = slice(0, 2000)
        ax.scatter(fvec[sub] * 1e4, h_eval[sub] * 1e4, s=4, alpha=0.25,
                   color=color)
        lim = max(fvec.max(), h_eval.max()) * 1e4
        ax.plot([0, lim], [0, lim], color="gray", ls=":", lw=1)
        m = mz[f"h_true|{fname}"]
        xs = np.linspace(1e-6, lim * 1e-4, 10)
        ax.plot(xs * 1e4, (m["a"] + m["b"] * xs) * 1e4, color="black", lw=1.4)
        t_txt = "恒等回归" if m["t_b1"] is None else f"t(b=1)={m['t_b1']:.2f}"
        ax.set_title(
            f"{fname}:  b={m['b']:.4f}  R²={m['R2']:.3f}  {t_txt}",
            fontsize=10)
        ax.set_xlabel("预测 h（×1e-4）")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("真实条件方差 h（×1e-4）")
    fig.suptitle("图2 真值 proxy 下的 MZ 回归：oracle 应当 b=1，GARCH 轻微偏离", fontsize=11)
    _save(fig, "fig2_mz_true_proxy.png")

    # ---- 图3：r² proxy 下 R² 全线塌到理论水平，与预测好坏无关
    names = ["oracle", "garch", "ewma", "const"]
    rsqs = [mz[f"r2|{n}"]["R2"] for n in names]
    bs_r2 = [mz[f"r2|{n}"]["b"] for n in names]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = np.arange(len(names))
    ax.bar(x, rsqs, 0.6, color="#1f6feb", alpha=0.85)
    ax.axhline(r2_theory, color="#d1242f", lw=1.6, ls="--",
               label=r"理论 $\mathrm{Var}(h)/\mathrm{Var}(r^2)$"
                     f" = {r2_theory:.4f}")
    for xi, b, rsq in zip(x, bs_r2, rsqs):
        ax.text(xi, rsq + 0.004, f"b={b:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(r"MZ 回归的 $R^2$（proxy = $r^2$）")
    ax.set_title(r"图3 换成 $r^2$ proxy：连 oracle 的 $R^2$ 都塌到 $\approx$ 9%，"
                 "与预测好坏无关")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig3_mz_r2_proxy.png")

    # ---- MC 实验 1：MZ + NW t 检验的实际 size（b=1 的拒绝率）
    N_MC = 300
    rng_mc = np.random.default_rng(SEED + 1)
    pairs = [("oracle", "r2"), ("garch", "r2"), ("garch", "h_true"),
             ("ewma", "h_true")]
    stats = {f"{f}|{p}": {"rej": 0, "b": [], "R2": []} for f, p in pairs}
    for rep in range(N_MC):
        r_m, h_m = simulate_garch(T_EVAL, rng_mc, burn=200)
        fcands = {
            "oracle": h_m,
            "garch": filter_garch(r_m ** 2, OMEGA, ALPHA, BETA),
            "ewma": filter_garch(r_m ** 2, 0.0, 1.0 - 0.94, 0.94),
        }
        for fname, fvec in fcands.items():
            for pname, proxy in (("r2", r_m ** 2), ("h_true", h_m)):
                key = f"{fname}|{pname}"
                if key not in stats:
                    continue
                _, b, _, t, rsq, _ = mz_regression(proxy, fvec)
                stats[key]["rej"] += int(t is not None and abs(t) > 1.96)
                stats[key]["b"].append(b)
                stats[key]["R2"].append(rsq)
    # 偏倚来源分解：固定 h 路径、独立重抽 z^2，E[b|h]=1 应当恢复
    r_fix, h_fix = simulate_garch(T_EVAL, rng_mc, burn=200)
    bs_fix = []
    for k in range(200):
        z2 = np.random.default_rng(SEED + 7 + k).standard_normal(T_EVAL) ** 2
        x = np.vstack([np.ones(T_EVAL), h_fix]).T
        coef, *_ = np.linalg.lstsq(x, h_fix * z2, rcond=None)
        bs_fix.append(float(coef[1]))
    mc_size = {
        "fixed_h_redraw_z": {"b_mean": float(np.mean(bs_fix)),
                             "b_sd": float(np.std(bs_fix, ddof=1)),
                             "N": 200},
    }
    for key, st in stats.items():
        mc_size[key] = {
            "rej_rate": st["rej"] / N_MC,
            "b_mean": float(np.mean(st["b"])),
            "b_sd": float(np.std(st["b"], ddof=1)),
            "R2_mean": float(np.mean(st["R2"])),
        }

    # ---- MC 实验 2 / 图4：损失区分度——各预测相对 GARCH 的倍数
    multiples = {}
    for pname in ("h_true", "r2"):
        for loss_name, fn in (("mse", mse_loss), ("qlike", qlike_loss)):
            base = fn(forecasts["garch"],
                      h_eval if pname == "h_true" else r2_eval)
            for fname in ("ewma", "const", "bias15"):
                multiples[f"{pname}|{loss_name}|{fname}"] = (
                    fn(forecasts[fname],
                       h_eval if pname == "h_true" else r2_eval) / base)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    groups = ["h_true|mse", "h_true|qlike", "r2|mse", "r2|qlike"]
    width = 0.26
    xg = np.arange(len(groups))
    for i, fname in enumerate(("ewma", "const", "bias15")):
        vals = [multiples[f"{g}|{fname}"] for g in groups]
        ax.bar(xg + (i - 1) * width, vals, width,
               label={"ewma": "EWMA(0.94)", "const": "常数=长期方差",
                      "bias15": "GARCH×1.15（偏高 15%）"}[fname], alpha=0.88)
    ax.set_yscale("log")
    ax.axhline(1.0, color="gray", ls=":", lw=1.2)
    for i, g in enumerate(groups):
        for j, fname in enumerate(("ewma", "const", "bias15")):
            v = multiples[f"{g}|{fname}"]
            ax.text(i + (j - 1) * width, v * 1.08, f"×{v:.2f}", ha="center",
                    fontsize=7.2, rotation=90)
    ax.set_xticks(xg)
    ax.set_xticklabels([gg.replace("h_true", "proxy=h").replace("r2", "proxy=r²")
                        for gg in groups])
    ax.set_ylabel("损失 / GARCH 损失（对数轴）")
    ax.set_title("图4 r² proxy 把损失区分度稀释 10-40 倍；QLIKE 对高估更宽容")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, axis="y", which="both")
    _save(fig, "fig4_loss_discrimination.png")

    # ---- proxy 噪声稳健性扫描
    scales = [0.0, 0.25, 0.5, 0.75, 1.0]
    rng_mc2 = np.random.default_rng(SEED + 2)
    sweep = {"scales": scales, "mse_flip": [], "qlike_flip": []}
    paths = [simulate_garch(T_EVAL, rng_mc2, burn=200) for _ in range(N_MC)]
    for s in scales:
        fm = fq = 0
        rng_n = np.random.default_rng(SEED + 3 + int(s * 1000))
        for r_m, h_m in paths:
            h_filt = filter_garch(r_m ** 2, OMEGA, ALPHA, BETA)
            h_b = 1.15 * h_filt
            z = rng_n.standard_normal(T_EVAL)
            proxy = h_m * np.exp(s * z - 0.5 * s * s)
            fm += mse_loss(h_b, proxy) < mse_loss(h_filt, proxy)
            fq += qlike_loss(h_b, proxy) < qlike_loss(h_filt, proxy)
        sweep["mse_flip"].append(fm / N_MC)
        sweep["qlike_flip"].append(fq / N_MC)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(scales, sweep["mse_flip"], "o-", color="#d1242f",
            label="MSE 排名翻转概率")
    ax.plot(scales, sweep["qlike_flip"], "s-", color="#1f6feb",
            label="QLIKE 排名翻转概率")
    ax.set_xlabel("proxy 对数正态噪声尺度 s")
    ax.set_ylabel("P( +15% 有偏预测胜过无偏滤波 )")
    ax.set_title("图5 proxy 噪声越大 MSE 越容易被骗，QLIKE 全程贴地")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, "fig5_proxy_noise_robustness.png")

    results = {
        "dgp": {"omega": OMEGA, "alpha": ALPHA, "beta": BETA,
                "long_run_var": LRV},
        "mle": {"omega": om_hat, "alpha": a_hat, "beta": b_hat,
                "T_train": T_TRAIN, "T_eval": T_EVAL},
        "mz": mz,
        "r2_theory": r2_theory,
        "mc_size": mc_size,
        "loss_multiples": multiples,
        "losses_single_split": losses,
        "noise_sweep": sweep,
    }
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
