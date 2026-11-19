# -*- coding: utf-8 -*-
"""局部投影脉冲响应：VAR 递推 vs Jordà 逐点回归，设定正确时的效率与误设下的稳健性。

本文全部数字的唯一来源。运行本脚本会：
1. 在真值已知的 AR(1)（VAR 设定正确）与 ARMA(1,1)（VAR 漏掉 MA 项）上做蒙特卡洛；
   结构冲击 eps_t 可观测（如政策意外），另有独立噪声 eta_t，避免完全拟合的平凡情形；
2. 对每个 horizon h 用局部投影 LP 直接回归 y_{t+h} on eps_t + 冲击前控制集估计 IRF；
3. 用单方程 VAR 代理 y_t=a·y_{t-1}+c·eps_t 递推 IRF(h)=c·a^h，对比偏差/RMSE/覆盖率；
4. 扫描样本量 n，展示正确设定下两者接近、误设下 LP 反超的交叉；
5. 画 5 张图到 public/images/local-projection-impulse-response/，
   统计量写入同目录 results.json。
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261119
R = 1200          # 蒙特卡洛重复数
N = 500           # 主实验样本量
H = 12            # 最大 horizon
PHI, B, D, SIG_ETA = 0.6, 1.0, 0.5, 0.5
Z975 = 1.959963984540054

_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "public", "images", "local-projection-impulse-response",
)
os.makedirs(_OUT, exist_ok=True)


def true_irf(h, ma=False):
    """eps_t 对 y_{t+h} 的真 IRF，控制冲击前状态。
    AR(1):  b·phi^h；ARMA(1,1): h=0 为 b，h>=1 为 phi^(h-1)·(phi·b+d)。
    """
    if not ma:
        return B * PHI ** h
    if h == 0:
        return B
    return PHI ** (h - 1) * (PHI * B + D)


def simulate(rng, n, ma=False, reps=R):
    """生成 eps(n,reps) 结构冲击、eta 独立噪声、y(n,reps) 路径。"""
    eps = rng.standard_normal((n, reps))
    eta = rng.standard_normal((n, reps)) * SIG_ETA
    y = np.empty((n, reps))
    prev = np.zeros(reps)
    for t in range(n):
        prev = PHI * prev + B * eps[t] + (D * eps[t - 1] if (ma and t >= 1) else 0.0) + eta[t]
        y[t] = prev
    return eps, eta, y


def lp_irf(eps, y, h):
    """y_{t+h}=a+beta·eps_t+gamma·y_{t-1}+delta·eps_{t-1}，beta 即 LP 的 IRF。
    eps/y 均为 (n, reps)，每个 rep 的控制集不同，批量解 (reps,4,4) 正规方程。"""
    n = len(y)
    # t = 1 .. n-1-h，保证 y_{t+h} 有效且 t-1 >= 0
    yy = y[1 + h:n]         # (T, reps) y_{t+h}
    e0 = eps[1:n - h]       # eps_t
    ym1 = y[0:n - 1 - h]     # y_{t-1}
    em1 = eps[0:n - 1 - h]   # eps_{t-1}
    T = yy.shape[0]
    # X: (T, 4, reps)
    Rn = eps.shape[1]
    X = np.stack([np.ones((T, Rn)), e0, ym1, em1], axis=1)  # (T,4,reps)
    G = np.einsum("tir,tjr->rij", X, X)          # (reps,4,4) = X'X
    Xty = np.einsum("tir,tr->ri", X, yy)         # (reps,4)
    coef = np.linalg.solve(G, Xty[:, :, None])[:, :, 0]   # (reps,4)
    beta = coef[:, 1]
    resid = yy - np.einsum("tir,ri->tr", X, coef)
    s2 = (resid * resid).sum(0) / (T - 4)
    inv_diag = np.linalg.inv(G)[:, 1, 1]
    se = np.sqrt(np.maximum(s2 * inv_diag, 0.0))
    return beta, se


def var_irf(eps, y, h):
    """VAR 代理 y_t=a·y_{t-1}+c·eps_t，IRF(h)=c·a^h；SE 用 delta method。"""
    ylag, ynow, elag = y[:-1], y[1:], eps[1:]
    s11 = (ylag * ylag).sum(0)
    s12 = (ylag * elag).sum(0)
    s22 = (elag * elag).sum(0)
    t1 = (ylag * ynow).sum(0)
    t2 = (elag * ynow).sum(0)
    det = s11 * s22 - s12 * s12
    a = (t1 * s22 - t2 * s12) / det
    c = (t2 * s11 - t1 * s12) / det
    irf = c * a ** h
    resid = ynow - a * ylag - c * elag
    s2 = (resid * resid).sum(0) / (len(ynow) - 2)
    da = c * h * a ** (h - 1) if h >= 1 else 0.0
    dc = a ** h
    var = s2 / det * (s22 * da * da - 2 * s12 * da * dc + s11 * dc * dc)
    return irf, np.sqrt(np.maximum(var, 0.0))


def run_dgp(ma, n=N, reps=R):
    """在一个 DGP 上跑全 horizon，返回各方法的偏差/RMSE/覆盖率与 CI 宽度。"""
    rng = np.random.default_rng(SEED + (11 if ma else 0))
    eps, eta, y = simulate(rng, n, ma=ma, reps=reps)
    out = {"irf_true": [], "lp_mean": [], "lp_bias": [], "lp_rmse": [],
           "lp_cov": [], "lp_ciw": [], "var_mean": [], "var_bias": [],
           "var_rmse": [], "var_cov": [], "var_ciw": []}
    for h in range(H + 1):
        tv = true_irf(h, ma=ma)
        bl, sl = lp_irf(eps, y, h)
        bv, sv = var_irf(eps, y, h)
        out["irf_true"].append(tv)
        out["lp_mean"].append(float(bl.mean()))
        out["lp_bias"].append(float(bl.mean() - tv))
        out["lp_rmse"].append(float(np.sqrt(np.mean((bl - tv) ** 2))))
        out["lp_cov"].append(float(np.mean(np.abs(bl - tv) <= Z975 * sl)))
        out["lp_ciw"].append(float(np.mean(2 * Z975 * sl)))
        out["var_mean"].append(float(bv.mean()))
        out["var_bias"].append(float(bv.mean() - tv))
        out["var_rmse"].append(float(np.sqrt(np.mean((bv - tv) ** 2))))
        out["var_cov"].append(float(np.mean(np.abs(bv - tv) <= Z975 * sv)))
        out["var_ciw"].append(float(np.mean(2 * Z975 * sv)))
    return out


def n_scan():
    """样本量扫描：h=1..H 平均 RMSE 的 VAR/LP 比率，两条 DGP 各一条曲线。"""
    rows = []
    for n in [100, 200, 400, 800, 1600]:
        rec = {"n": n}
        for ma, key in [(False, "correct"), (True, "misspec")]:
            a = run_dgp(ma, n=n, reps=400)
            m_lp = float(np.mean(a["lp_rmse"][1:]))
            m_var = float(np.mean(a["var_rmse"][1:]))
            rec[f"ratio_{key}"] = m_var / m_lp
            rec[f"lp_rmse_{key}"] = m_lp
            rec[f"var_rmse_{key}"] = m_var
        rows.append(rec)
    return rows


def _save(fig, name):
    path = os.path.join(_OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot1(a):
    hs = list(range(H + 1))
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(hs, a["irf_true"], "k-", lw=2.5, label="真 IRF  b·phi^h")
    ax.plot(hs, a["lp_mean"], "o-", color="#c0392b", lw=1.8, ms=5, label="LP 逐点回归均值")
    ax.plot(hs, a["var_mean"], "s--", color="#2980b9", lw=1.8, ms=5, label="VAR 递推均值")
    ax.set_xlabel("horizon h")
    ax.set_ylabel("脉冲响应")
    ax.set_title("图1：AR(1) 设定正确，LP 与 VAR 都贴近真 IRF（n=500, 1200 次）")
    ax.legend(fontsize=9)
    return _save(fig, "fig1_irf_correct.png")


def plot2(a):
    hs = list(range(H + 1))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(hs, a["var_rmse"], "s-", color="#2980b9", lw=2, label="VAR 递推 RMSE")
    ax.plot(hs, a["lp_rmse"], "o-", color="#c0392b", lw=2, label="LP 逐点 RMSE")
    ratio = float(np.mean(a["var_rmse"][1:])) / float(np.mean(a["lp_rmse"][1:]))
    ax.set_yscale("log")
    ax.set_xlabel("horizon h")
    ax.set_ylabel("IRF 估计 RMSE（对数轴）")
    ax.set_title(f"图2：设定正确时 VAR 的 RMSE 更低，全 horizon 均值仅 LP 的 {ratio:.3f} 倍")
    ax.legend(fontsize=9)
    return _save(fig, "fig2_rmse_correct.png")


def plot3(a):
    hs = list(range(H + 1))
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(hs, a["irf_true"], "k-", lw=2.5, label="真 IRF  phi^(h-1)(phi·b+d)")
    ax.plot(hs, a["lp_mean"], "o-", color="#c0392b", lw=1.8, ms=5, label="LP 逐点回归均值")
    ax.plot(hs, a["var_mean"], "s--", color="#2980b9", lw=1.8, ms=5, label="VAR(1) 递推均值（漏 MA 项）")
    ax.set_xlabel("horizon h")
    ax.set_ylabel("脉冲响应")
    ax.set_title("图3：ARMA(1,1) 误设下 VAR 把响应估歪，LP 仍贴合真值")
    ax.legend(fontsize=9)
    return _save(fig, "fig3_irf_misspec.png")


def plot4(a):
    hs = list(range(H + 1))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.bar(np.array(hs) - 0.18, a["var_bias"], 0.36, color="#2980b9", label="VAR 偏差")
    ax.bar(np.array(hs) + 0.18, a["lp_bias"], 0.36, color="#c0392b", label="LP 偏差")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("horizon h")
    ax.set_ylabel("IRF 估计偏差（估计均值 − 真值）")
    ax.set_title("图4：VAR(1) 漏掉 MA(1) 后响应被系统性估歪；LP 偏差近零")
    ax.legend(fontsize=9)
    return _save(fig, "fig4_bias_misspec.png")


def plot5(rows):
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(ns, [r["ratio_correct"] for r in rows], "o-", color="#27ae60", lw=2,
            label="AR(1) 设定正确：VAR/LP RMSE 比")
    ax.plot(ns, [r["ratio_misspec"] for r in rows], "s-", color="#8e44ad", lw=2,
            label="ARMA(1,1) 误设：VAR/LP RMSE 比")
    ax.axhline(1.0, color="k", ls="--", lw=1.2)
    ax.text(ns[0], 1.03, "1 = 两者持平", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.set_xlabel("样本量 n")
    ax.set_ylabel("VAR 平均 RMSE ÷ LP 平均 RMSE")
    ax.set_title("图5：效率与稳健性交叉——设定越对 VAR 越省，误设越大 LP 越稳")
    ax.legend(fontsize=9)
    return _save(fig, "fig5_efficiency_tradeoff.png")


def main():
    a = run_dgp(False)
    b = run_dgp(True)
    scan = n_scan()
    results = {
        "setup": {"seed": SEED, "reps": R, "n": N, "H": H, "phi": PHI,
                  "b": B, "d": D, "sig_eta": SIG_ETA, "z975": Z975},
        "dgpA_correct": a,
        "dgpB_misspec": b,
        "n_scan": scan,
    }
    results["figs"] = [
        plot1(a), plot2(a), plot3(b), plot4(b), plot5(scan),
    ]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))
    return results


if __name__ == "__main__":
    main()
