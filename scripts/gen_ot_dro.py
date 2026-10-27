# -*- coding: utf-8 -*-
"""
最优传输组合优化实验：Sinkhorn 熵正则 Wasserstein barycenter + DRO（软最坏情况）
输出 5 张图到 public/images/optimal-transport-portfolio-optimization/
结果 json: scripts/_tda_ot_out/_ot_results.json
seed=20261027。无 POT 库，Sinkhorn / 1D-W2 / simplex 优化全部手写。
"""
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

matplotlib.rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

SEED = 20261027
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "public/images/optimal-transport-portfolio-optimization")
JDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tda_ot_out")
os.makedirs(OUT, exist_ok=True); os.makedirs(JDIR, exist_ok=True)
R = {}

# ---------------- OT 核心 ----------------

def wasserstein_1d(u, v, p=None, q=None):
    """1D W2^2 精确值（分位数法，merge 数值积分）。"""
    if p is None: p = np.full(len(u), 1.0 / len(u))
    if q is None: q = np.full(len(v), 1.0 / len(v))
    u = np.sort(u); v = np.sort(v)
    grid = np.linspace(0, 1, 20001)
    fu = np.interp(grid, np.cumsum(p) - p / 2, u, left=u[0], right=u[-1])
    fv = np.interp(grid, np.cumsum(q) - q / 2, v, left=v[0], right=v[-1])
    return float(np.mean((fu - fv) ** 2))

def sinkhorn(a, b, C, eps, n_iter=3000, tol=1e-11):
    """log 域 Sinkhorn：min <P,C> + ε Σ P logP，行=a 列=b。返回 (P, 目标值, 迭代数)。"""
    Mc = -C / eps
    f = np.zeros(len(a)); g = np.zeros(len(b))
    for it in range(n_iter):
        g_new = np.log(b) - np.logaddexp.reduce(Mc + f[:, None], axis=0)
        f_new = np.log(a) - np.logaddexp.reduce(Mc + g_new[None, :], axis=1)
        d = np.max(np.abs(f_new - f))
        f, g = f_new, g_new
        if d < tol and it > 5:
            break
    P = np.exp(Mc + f[:, None] + g[None, :])
    val = float((P * C).sum()) + float(eps * (P * np.log(P + 1e-300)).sum())
    return P, val, it + 1

def wasserstein_barycenter_ipot(supports, weights, eps, ref=None, n_iter=300):
    """IPOT：bar_{t+1} = Σ_k w_k · T_k(bar_t)，T_k = 到第 k 个分布的熵传输映射（条件均值）。"""
    if ref is None:
        ref = np.sort(np.mean(supports, axis=0))
    bar = ref.copy(); W = len(bar); hist = []
    for it in range(n_iter):
        newbar = np.zeros(W)
        for wk, sup in zip(weights, supports):
            C = (bar[:, None] - sup[None, :]) ** 2
            P, _, _ = sinkhorn(np.full(W, 1.0 / W), np.full(len(sup), 1.0 / len(sup)), C, eps, n_iter=300)
            cond = (P @ sup) / P.sum(1)
            newbar += wk * cond
        bar_new = newbar / sum(weights)
        shift = float(np.max(np.abs(bar_new - bar)))
        bar = bar_new; hist.append(shift)
        if shift < 1e-8 and it > 3:
            break
    return bar, hist

# ---------------- fig1: 1D W2 与 KL 的对比（平移敏感性） ----------------
rng = np.random.default_rng(SEED)
A = rng.normal(0.0, 0.6, 400)
B = rng.normal(0.03, 0.9, 400)
A_shift = A + 2.0
w_AB = wasserstein_1d(A, B)
w_AAs = wasserstein_1d(A, A_shift)
kl_finite = float(np.sum((A - A_shift) ** 2) / (2 * 0.6 ** 2) * 0)  # 经验测度 KL = ∞（支撑不相交）
R["w2"] = {"W2(A,B)": round(w_AB ** 0.5, 4), "shift": 2.0,
           "W2(A,A+2)": round(w_AAs ** 0.5, 4), "n": 400}

fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.2))
ax[0].hist(A, bins=30, density=True, alpha=0.55, color="#3b6ea5", label="情景 A")
ax[0].hist(B, bins=30, density=True, alpha=0.55, color="#c0504d", label="情景 B")
ax[0].set_title(f"两个收益分布：W₂ = {w_AB**0.5:.3f}", fontsize=11)
ax[0].legend(fontsize=9); ax[0].set_xlabel("日收益")
ax[1].hist(A, bins=30, density=True, alpha=0.55, color="#3b6ea5", label="A")
ax[1].hist(A_shift, bins=30, density=True, alpha=0.55, color="#e8a33d", label="A 平移 +2")
ax[1].set_title(f"经验测度 KL=∞，但 W₂ = {w_AAs**0.5:.3f}（=平移量）", fontsize=11)
ax[1].legend(fontsize=9); ax[1].set_xlabel("日收益")
fig.suptitle("图1  Wasserstein 距离：支撑不相交仍可计算、对平移连续——KL 散度做不到", fontsize=12.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_w2_distance.png", dpi=140); plt.close(fig)

# ---------------- fig2: barycenter ≠ mixture；熵正则坍缩 ----------------
A2 = rng.normal(-1.5, 0.25, 300); B2 = rng.normal(1.5, 0.25, 300)
eps_grid = [0.02, 0.1, 0.5, 2.0]
eps_barys = {}
for epsv in eps_grid:
    bar, _ = wasserstein_barycenter_ipot([A2, B2], [0.5, 0.5], epsv)
    eps_barys[epsv] = bar
gridq = (np.arange(400) + 0.5) / 400
bar_exact = 0.5 * (np.quantile(A2, gridq) + np.quantile(B2, gridq))
gap = float(np.std(bar_exact))
mixture_std = float(math.sqrt(0.5 * (np.var(A2) + np.mean(A2) ** 2) + 0.5 * (np.var(B2) + np.mean(B2) ** 2)))
R["barycenter"] = {"separated_means": [round(float(np.mean(A2)), 3), round(float(np.mean(B2)), 3)],
                   "exact_bary_std": round(gap, 4), "mixture_std": round(mixture_std, 4),
                   "eps_bary_std": {str(k): round(float(np.std(v)), 4) for k, v in eps_barys.items()}}
fig, ax = plt.subplots(figsize=(8.0, 4.6))
xg = np.linspace(-4, 4, 800)
mix = (0.5 * np.exp(-0.5 * ((xg - np.mean(A2)) / np.std(A2)) ** 2) / (np.std(A2) * math.sqrt(2 * math.pi))
       + 0.5 * np.exp(-0.5 * ((xg - np.mean(B2)) / np.std(B2)) ** 2) / (np.std(B2) * math.sqrt(2 * math.pi)))
ax.plot(xg, mix, c="#c0504d", lw=2.0, label="50/50 混合分布（双峰）")
ax.hist(bar_exact, bins=40, density=True, alpha=0.55, color="#666666",
        label=f"W₂ 精确 barycenter（std={gap:.3f}）")
for epsv, col in zip(eps_grid, ["#4472c4", "#4f8a3b", "#e8a33d", "#c0504d"]):
    bv = eps_barys[epsv]
    if np.std(bv) < 0.02:
        ax.axvline(np.mean(bv), color=col, ls="--", lw=1.2, label=f"ε={epsv}：坍缩到均值")
    else:
        ax.hist(bv, bins=40, density=True, alpha=0.35, color=col, label=f"熵正则 ε={epsv}")
ax.axvline(0, c="k", ls=":", lw=1)
ax.set_xlabel("收益率"); ax.set_xlim(-4, 4)
ax.set_title("图2  混合（平均密度，双峰）≠ Wasserstein barycenter（中位测度，保形单峰）；ε≥0.5 时 IPOT 坍缩到均值", fontsize=10.5)
ax.legend(fontsize=8.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_barycenter.png", dpi=140); plt.close(fig)

# ---------------- fig3: Sinkhorn log 域收敛 + 传输映射刚体度 ----------------
Cg = (np.sort(A2)[:, None] - np.sort(B2)[None, :]) ** 2
conv_hist = {}
for epsv in [0.05, 0.2, 1.0]:
    Mc = -Cg / epsv
    a_ = np.full(300, 1 / 300); b_ = np.full(300, 1 / 300)
    f = np.zeros(300); g = np.zeros(300)
    objs = []
    for it in range(60):
        g = np.log(b_) - np.logaddexp.reduce(Mc + f[:, None], axis=0)
        f = np.log(a_) - np.logaddexp.reduce(Mc + g[None, :], axis=1)
        P = np.exp(Mc + f[:, None] + g[None, :])
        objs.append(float((P * Cg).sum()) + epsv * float((P * np.log(P + 1e-300)).sum()))
    conv_hist[epsv] = np.array(objs)
R["sinkhorn"] = {str(k): {"obj_gap_after_10": round(float(v[9] - v[-1]), 8),
                          "obj_gap_after_30": round(float(v[29] - v[-1]), 8)}
                 for k, v in conv_hist.items()}
fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.2))
for epsv, col in [(0.05, "#c0504d"), (0.2, "#e8a33d"), (1.0, "#3b6ea5")]:
    e = conv_hist[epsv] - conv_hist[epsv][-1]
    ax[0].semilogy(np.arange(1, 61), e, lw=1.8, color=col, label=f"ε={epsv}")
ax[0].set_xlabel("迭代次数"); ax[0].set_ylabel("目标值 − 收敛值 (log)")
ax[0].legend(fontsize=9); ax[0].set_title("(a) log 域 Sinkhorn：ε 小收敛慢", fontsize=11)
for epsv, col in [(0.05, "#c0504d"), (1.0, "#3b6ea5")]:
    Cpl = (np.sort(A2)[:, None] - np.sort(B2)[None, :]) ** 2
    Pp, _, _ = sinkhorn(np.full(300, 1 / 300), np.full(300, 1 / 300), Cpl, epsv, n_iter=1500)
    T = (Pp @ np.sort(B2)) / Pp.sum(1)
    ax[1].hist(T, bins=40, density=True, alpha=0.4, color=col,
               label=f"ε={epsv}: T(A) 均值 {np.mean(T):.2f} std {np.std(T):.2f}")
ax[1].hist(B2, bins=30, density=True, alpha=0.25, color="#888888", label="目标 B（μ=1.5, σ=0.25）")
ax[1].set_title("(b) ε 小→映射刚体（保形平移）；ε 大→像被拉向目标均值", fontsize=10)
ax[1].legend(fontsize=8); ax[1].set_xlim(-4, 4)
fig.suptitle("图3  Sinkhorn 的两个 ε 代价：收敛速度 与 计划的“模糊度”", fontsize=12.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_sinkhorn.png", dpi=140); plt.close(fig)

# ---------------- DRO 主实验 ----------------
N_ASSET = 3

def make_truth(rng):
    mu = rng.normal(0, 0.05, N_ASSET)
    L = rng.normal(0, 1, (N_ASSET, N_ASSET)) / math.sqrt(N_ASSET)
    S = L @ L.T + 1e-3 * np.eye(N_ASSET)
    sig = np.sqrt(np.diag(S)); Rr = S / np.outer(sig, sig)
    return mu, sig, Rr

def scenario_draw(mu0, sig0, Rr0, rng, k, n=300):
    mus = mu0[None, :] + rng.normal(0, 0.03, (k, N_ASSET))
    scen = []
    for i in range(k):
        Lr = np.linalg.cholesky(Rr0)
        jitter = rng.normal(0, 0.15, (N_ASSET, N_ASSET)) / math.sqrt(N_ASSET)
        Rj = (Lr + jitter) @ (Lr + jitter).T
        dj = np.sqrt(np.clip(np.diag(Rj), 1e-8, None))
        Rn = Rj / np.outer(dj, dj)
        L2 = np.linalg.cholesky(Rn + 1e-9 * np.eye(N_ASSET))
        X = rng.standard_normal((n, N_ASSET)) @ L2.T * sig0[None, :] + mus[i]
        D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        scen.append({"X": X, "C": np.cov(X.T), "D2": D2})
    return scen, mus

def opt_simplex(mu, S, tau=0.5):
    n = len(mu); w = np.full(n, 1 / n); L = 2 + 2 * np.linalg.norm(S, 2)
    def proj(v):
        u = np.sort(v)[::-1]; t = (np.cumsum(u) - 1) / np.arange(1, n + 1)
        k = np.where(u > t)[0].max(); th = t[k]
        return np.clip(v - th, 0, None)
    for _ in range(800):
        g = mu - 2 * tau * (S @ w)
        wn = proj(w + g / L)
        if np.max(np.abs(wn - w)) < 1e-10: w = wn; break
        w = wn
    return w

def dro_portfolio(scens, eps_dro, tau=0.5):
    """Wasserstein 球内最坏评估的对偶代理（软最坏邻居）：
      每点 y 的最坏值 = -ε log (1/n) Σ_j exp(-(f_j - ||y-y'||²)/ε)
    ε→0 坍缩为逐点最坏邻居（硬最坏情况），ε→∞ 回到场景内平均（SAA）。"""
    def neg_obj(w):
        tot = 0.0
        for sc in scens:
            f = sc["X"] @ w - tau * (w @ sc["C"] @ w)
            z = (-f[None, :] - sc["D2"]) / eps_dro
            zm = z.max(1, keepdims=True)
            worst_i = -eps_dro * (np.log(np.exp(z - zm).mean(1)) + zm[:, 0])
            tot += worst_i.mean()
        return -tot / len(scens)
    w0 = np.full(N_ASSET, 1 / N_ASSET)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    res = minimize(neg_obj, w0, method="SLSQP", bounds=[(0, 1)] * N_ASSET,
                   constraints=cons, options={"maxiter": 120, "ftol": 1e-6})
    return res.x

def saa_portfolio(scens, tau=0.5):
    pool = np.vstack([sc["X"] for sc in scens])
    return opt_simplex(pool.mean(0), np.cov(pool.T), tau)

def oos_eval(w, mu0, sig0, Rr0, rng, n_test=30000):
    L = np.linalg.cholesky(Rr0 + 1e-9 * np.eye(N_ASSET))
    X = rng.standard_normal((n_test, N_ASSET)) @ L.T * sig0 + mu0
    r = X @ w
    return float(r.mean() / (r.std() + 1e-12) * math.sqrt(252))

rng4 = np.random.default_rng(SEED + 4)
REPS = 96
TAU = 0.5
mults = [0.02, 0.1, 0.5, 2.0]   # ε = m × median‖y-y'‖²
eps_list = [f"m{m}" for m in mults]
sharpe = {"SAA": [], "EQ": [], "TRUTH": []}
for e in eps_list: sharpe[e] = []

rep_data = []
for rep in range(REPS):
    mu0, sig0, Rr0 = make_truth(rng4)
    scens, _ = scenario_draw(mu0, sig0, Rr0, rng4, 5)
    iu = np.triu_indices(len(scens[0]["X"]), 1)
    med_d2 = float(np.median(scens[0]["D2"][iu]))
    w_truth = opt_simplex(mu0, (Rr0 * np.outer(sig0, sig0)), TAU)
    w_saa = saa_portfolio(scens, TAU)
    row = {"truth": w_truth, "saa": w_saa}
    for e, m in zip(eps_list, mults):
        row[e] = dro_portfolio(scens, m * med_d2, TAU)
    rep_data.append((mu0, sig0, Rr0, row))

for rep, (mu0, sig0, Rr0, row) in enumerate(rep_data):
    rr = np.random.default_rng(SEED + 5000 + rep)
    sharpe["TRUTH"].append(oos_eval(row["truth"], mu0, sig0, Rr0, rr))
    sharpe["SAA"].append(oos_eval(row["saa"], mu0, sig0, Rr0, rr))
    sharpe["EQ"].append(oos_eval(np.full(N_ASSET, 1 / N_ASSET), mu0, sig0, Rr0, rr))
    for e in eps_list:
        sharpe[e].append(oos_eval(row[e], mu0, sig0, Rr0, rr))

R["dro_oos"] = {"reps": REPS, "tau": TAU, "n_scenario": 5, "n_per_scenario": 300,
                "mults": mults,
                "sharpe_mean": {k: round(float(np.mean(v)), 3) for k, v in sharpe.items()},
                "sharpe_se": {k: round(float(np.std(v, ddof=1) / math.sqrt(len(v))), 3) for k, v in sharpe.items()}}

paired = {}
for e in eps_list:
    d_ = np.array(sharpe[e]) - np.array(sharpe["SAA"])
    paired[e] = {"mean": round(float(d_.mean()), 3),
                 "se": round(float(d_.std(ddof=1) / math.sqrt(len(d_))), 3),
                 "win_rate": round(float((d_ > 0).mean()), 3)}
R["paired_dro_minus_saa"] = paired

fig, ax = plt.subplots(figsize=(8.8, 4.6))
names = ["EQ", "SAA", "TRUTH"] + eps_list
mm = [np.mean(sharpe[n]) for n in names]
ee = [np.std(sharpe[n], ddof=1) / math.sqrt(REPS) for n in names]
lbl = ["等权", "SAA(样本平均)", "MV真值组合", "DRO m=0.02\n(硬最坏情况)", "DRO m=0.1", "DRO m=0.5", "DRO m=2\n(≈SAA)"]
cols = ["#999999", "#c0504d", "#4f8a3b", "#4472c4", "#5b9bd5", "#8faadc", "#c9d7ee"]
ax.bar(range(len(mm)), mm, yerr=ee, capsize=4, color=cols)
ax.set_xticks(range(len(mm))); ax.set_xticklabels(lbl, rotation=0, ha="center", fontsize=8)
ax.axhline(0, c="k", lw=0.8)
ax.set_ylabel(f"样本外年化 Sharpe（{REPS} 次重复均值，误差棒=1 SE）")
ax.set_title("图4  最坏情况评估（软最大 m↓ 更硬）的样本外代价；名义差异多数在 ±2SE 内", fontsize=11.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_dro_oos.png", dpi=140); plt.close(fig)

dev_eq = {e: [] for e in eps_list}
for rep, (mu0, sig0, Rr0, row) in enumerate(rep_data):
    for e in eps_list:
        dev_eq[e].append(float(np.abs(row[e] - 1 / N_ASSET).sum()))
saa_dev = float(np.mean([np.abs(row["saa"] - 1 / N_ASSET).sum() for _, _, _, row in rep_data]))
R["dro_dev_eq"] = {k: round(float(np.mean(v)), 3) for k, v in dev_eq.items()}
R["saa_dev_eq"] = round(saa_dev, 3)
fig, ax = plt.subplots(figsize=(7.4, 4.4))
xs = mults; ys = [np.mean(dev_eq[e]) for e in eps_list]
errs = [np.std(dev_eq[e], ddof=1) / math.sqrt(REPS) for e in eps_list]
ax.errorbar(xs, ys, yerr=errs, marker="o", lw=1.9, c="#c0504d", capsize=4,
            label="DRO 到等权距离（m↓ 单调收缩）")
ax.hlines(saa_dev, 0.015, 3, ls="--", color="#3b6ea5", lw=1.6, label=f"SAA 到等权距离 {saa_dev:.2f}")
ax.axhline(0, c="grey", ls=":", lw=1, label="完全收缩到等权")
ax.set_xscale("log")
ax.set_xlabel("软最大平滑尺度 m（ε = m × median‖y−y'‖²），m↓ = 更硬的最坏情况")
ax.set_ylabel("‖w_DRO − 等权‖₁（权重偏移）")
ax.set_title("图5  最坏情况越硬（m↓）权重越向分散收缩；m↑ 退回 SAA 的集中解", fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_shrinkage.png", dpi=140); plt.close(fig)

with open(f"{JDIR}/_ot_results.json", "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False)[:1600])
print("OK figs:", sorted(os.listdir(OUT)))
