#!/usr/bin/env python3
"""Copula-GARCH 组合 VaR 配图: 4 张"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, optimize
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
OUT = Path(__file__).resolve().parent.parent / "public/images/copula-garch-portfolio-var"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

# ---------- DGP: 2 资产 GARCH(1,1)-t(6) 边缘 + t-copula(nu=4, rho=0.5) ----------
T, BURN = 2000, 300
N = T + BURN
NU_COP, RHO = 4.0, 0.5
NU_MARG = 6.0

# t-copula 抽样
L = np.linalg.cholesky(np.array([[1, RHO], [RHO, 1]]))
Z = rng.standard_normal((N, 2)) @ L.T
W = rng.chisquare(NU_COP, N) / NU_COP
Tcop = Z / np.sqrt(W)[:, None]
U = stats.t.cdf(Tcop, NU_COP)                      # copula 均匀量
eps = stats.t.ppf(U, NU_MARG) / np.sqrt(NU_MARG / (NU_MARG - 2))  # 标准化 t(6) 残差

params = [(0.02, 0.08, 0.90), (0.03, 0.10, 0.87)]  # omega, alpha, beta
r = np.zeros((N, 2)); sig2 = np.zeros((N, 2))
for j, (om, al, be) in enumerate(params):
    sig2[0, j] = om / (1 - al - be)
    r[0, j] = np.sqrt(sig2[0, j]) * eps[0, j]
    for t in range(1, N):
        sig2[t, j] = om + al * r[t-1, j]**2 + be * sig2[t-1, j]
        r[t, j] = np.sqrt(sig2[t, j]) * eps[t, j]
r = r[BURN:]; T = len(r)
w = np.array([0.5, 0.5])
port = r @ w                                        # 组合收益，损失 = -port

# ---------- 估计: 前 500 天估参，后 1500 天滚动预报 ----------
EST = 750
def qml_garch(x):
    """方差目标化 + 多起点 QML，避免 Nelder-Mead 卡在 alpha=0 的退化解"""
    vbar = x.var()
    def nll(p):
        al, be = p
        if al <= 1e-4 or be <= 0 or al + be >= 0.999: return 1e9
        om = vbar * (1 - al - be)
        s2 = np.empty(len(x)); s2[0] = vbar
        for t in range(1, len(x)):
            s2[t] = om + al * x[t-1]**2 + be * s2[t-1]
        return 0.5 * np.sum(np.log(s2) + x**2 / s2)
    best, bestv = None, np.inf
    for a0 in (0.03, 0.08, 0.15):
        for b0 in (0.80, 0.88, 0.94):
            if a0 + b0 >= 0.999: continue
            res = optimize.minimize(nll, [a0, b0], method="Nelder-Mead",
                                    options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 4000})
            if res.fun < bestv:
                bestv, best = res.fun, res.x
    al, be = best
    return np.array([vbar * (1 - al - be), al, be])

fit = [qml_garch(r[:EST, j]) for j in range(2)]
# 全样本条件方差递推（参数用估计值）
s2hat = np.zeros((T, 2)); z = np.zeros((T, 2))
for j in range(2):
    om, al, be = fit[j]
    s2hat[0, j] = r[:EST, j].var()
    for t in range(1, T):
        s2hat[t, j] = om + al * r[t-1, j]**2 + be * s2hat[t-1, j]
    z[:, j] = r[:, j] / np.sqrt(s2hat[:, j])

# 边缘 t 自由度（估计窗）: loc 固定 0, scale 自由 —— 标准化 t 的 scale = sqrt((nu-2)/nu)
marg = [stats.t.fit(z[:EST, j], floc=0) for j in range(2)]
nu_hat = [m[0] for m in marg]; sc_marg = [m[2] for m in marg]
# 伪均匀量
Uhat = np.column_stack([stats.t.cdf(z[:EST, j], nu_hat[j], 0, sc_marg[j]) for j in range(2)])
tau = stats.kendalltau(Uhat[:, 0], Uhat[:, 1]).statistic
rho_hat = np.sin(np.pi * tau / 2)

def tcop_nll(nu, u, rho):
    x = stats.t.ppf(u, nu)
    det = 1 - rho**2
    q = (x[:, 0]**2 - 2 * rho * x[:, 0] * x[:, 1] + x[:, 1]**2) / det
    from scipy.special import gammaln
    c = (gammaln((nu + 2) / 2) + gammaln(nu / 2) - 2 * gammaln((nu + 1) / 2)
         - 0.5 * np.log(det))
    ll = c - (nu + 2) / 2 * np.log1p(q / nu) + (nu + 1) / 2 * (np.log1p(x[:, 0]**2 / nu) + np.log1p(x[:, 1]**2 / nu))
    return -np.sum(ll)

grid = np.arange(2.5, 20.5, 0.5)
nu_cop_hat = grid[np.argmin([tcop_nll(v, Uhat, rho_hat) for v in grid])]

# ---------- 三个模型每日 99%/95% VaR（评估窗 EST..T）----------
NSIM = 20000
def sim_var(t_idx, model):
    """返回组合损失分布的 95/99 分位数"""
    if model == "mvn":       # 方差-协方差: 正态边缘 + 高斯 copula(样本相关)
        rho_n = np.corrcoef(z[:EST].T)[0, 1]
        Ln = np.linalg.cholesky(np.array([[1, rho_n], [rho_n, 1]]))
        zz = rng.standard_normal((NSIM, 2)) @ Ln.T
        rr = zz * np.sqrt(s2hat[t_idx])
    else:
        if model == "gauss_cop":
            rho_n = np.corrcoef(stats.norm.ppf(Uhat).T)[0, 1]
            Ln = np.linalg.cholesky(np.array([[1, rho_n], [rho_n, 1]]))
            uu = stats.norm.cdf(rng.standard_normal((NSIM, 2)) @ Ln.T)
        else:                # t_cop
            Ln = np.linalg.cholesky(np.array([[1, rho_hat], [rho_hat, 1]]))
            zz = rng.standard_normal((NSIM, 2)) @ Ln.T
            ww = rng.chisquare(nu_cop_hat, NSIM) / nu_cop_hat
            uu = stats.t.cdf(zz / np.sqrt(ww)[:, None], nu_cop_hat)
        ee = np.column_stack([stats.t.ppf(uu[:, j], nu_hat[j], 0, sc_marg[j]) for j in range(2)])
        rr = ee * np.sqrt(s2hat[t_idx])
    loss = -(rr @ w)
    return np.quantile(loss, [0.95, 0.99])

models = ["mvn", "gauss_cop", "t_cop"]
names = {"mvn": "方差-协方差(正态)", "gauss_cop": "高斯copula + t边缘", "t_cop": "t-copula + t边缘"}
idx_eval = np.arange(EST, T)
VaR = {m: np.zeros((len(idx_eval), 2)) for m in models}
for k, t_idx in enumerate(idx_eval):
    for m in models:
        VaR[m][k] = sim_var(t_idx, m)

loss_eval = -port[idx_eval]

def kupiec(n_breach, n, p):
    if n_breach == 0:
        lr = -2 * n * np.log(1 - p)
    else:
        ph = n_breach / n
        lr = -2 * (n_breach * np.log(p / ph) + (n - n_breach) * np.log((1 - p) / (1 - ph)))
    return 1 - stats.chi2.cdf(lr, 1)

score = {}
for m in models:
    b95 = int((loss_eval > VaR[m][:, 0]).sum()); b99 = int((loss_eval > VaR[m][:, 1]).sum())
    score[m] = (b95, kupiec(b95, len(idx_eval), 0.05), b99, kupiec(b99, len(idx_eval), 0.01))
    print(names[m], score[m])

# ---------- 图1: 残差散点 vs 两种 copula ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
NS = EST
rho_n = np.corrcoef(stats.norm.ppf(Uhat).T)[0, 1]
Ln = np.linalg.cholesky(np.array([[1, rho_n], [rho_n, 1]]))
g_sim = stats.norm.cdf(rng.standard_normal((NS, 2)) @ Ln.T)
Lt = np.linalg.cholesky(np.array([[1, rho_hat], [rho_hat, 1]]))
zz = rng.standard_normal((NS, 2)) @ Lt.T
ww = rng.chisquare(nu_cop_hat, NS) / nu_cop_hat
t_sim = stats.t.cdf(zz / np.sqrt(ww)[:, None], nu_cop_hat)
for ax, dat, ttl, col in zip(axes, [Uhat, g_sim, t_sim],
                             [f"实际标准化残差 (伪均匀量, n={EST})", f"高斯 copula 模拟 (ρ={rho_n:.2f})",
                              f"t-copula 模拟 (ρ={rho_hat:.2f}, ν={nu_cop_hat:.1f})"],
                             ["#345995", "#8d99ae", "#c1121f"]):
    ax.scatter(dat[:, 0], dat[:, 1], s=7, alpha=0.5, color=col, lw=0)
    q = 0.05
    joint = ((dat[:, 0] < q) & (dat[:, 1] < q)).sum()
    ax.axvline(q, color="k", ls=":", lw=0.8); ax.axhline(q, color="k", ls=":", lw=0.8)
    ax.add_patch(plt.Rectangle((0, 0), q, q, color="orange", alpha=0.35))
    ax.set_title(ttl + f"\n左下 5% 联合角: {joint} 个点", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("资产 1 (U1)"); ax.set_ylabel("资产 2 (U2)")
fig.suptitle("同样的相关系数，尾部完全不同的世界：高斯 copula 的联合尾角是空的", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "copula-scatter.jpg", dpi=110, bbox_inches="tight"); plt.close(fig)

# ---------- 图2: VaR 路径 ----------
fig, ax = plt.subplots(figsize=(11, 4.7))
xs = np.arange(len(idx_eval))
ax.plot(xs, loss_eval, color="0.75", lw=0.6, label="组合日损失")
colors = {"mvn": "#8d99ae", "gauss_cop": "#345995", "t_cop": "#c1121f"}
for m in models:
    ax.plot(xs, VaR[m][:, 1], lw=1.1, color=colors[m], label=f"99% VaR: {names[m]}")
br = loss_eval > VaR["mvn"][:, 1]
ax.scatter(xs[br], loss_eval[br], s=22, color="#e07b00", zorder=5, label="正态模型突破点")
ax.set_title("99% VaR 路径（评估窗 1500 天）：三个模型追同一条损失序列", fontsize=12)
ax.set_xlabel("评估窗交易日"); ax.set_ylabel("损失 (%)")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT / "copula-var-paths.jpg", dpi=110, bbox_inches="tight"); plt.close(fig)

# ---------- 图3: 突破记分板 ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
n_eval = len(idx_eval)
for ax, col_i, p, ttl in zip(axes, [0, 2], [0.05, 0.01], ["95% VaR", "99% VaR"]):
    vals = [score[m][col_i] for m in models]
    pvs = [score[m][col_i + 1] for m in models]
    bars = ax.bar([names[m] for m in models], vals,
                  color=["#8d99ae" if pv >= 0.05 else "#c1121f" for pv in pvs], alpha=0.85)
    ax.axhline(n_eval * p, color="k", ls="--", lw=1, label=f"期望 {n_eval * p:.0f} 次")
    for b, v, pv in zip(bars, vals, pvs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v} 次\nKupiec p={pv:.3f}",
                ha="center", fontsize=9)
    ax.set_title(f"{ttl} 突破次数（n={n_eval}）", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.35 + 4); ax.legend(fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
fig.suptitle("红色 = Kupiec 频率检验 5% 水平拒绝", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "copula-breach-scoreboard.jpg", dpi=110, bbox_inches="tight"); plt.close(fig)

# ---------- 图4: 联合尾部超越概率 ----------
qs = np.linspace(0.005, 0.20, 40)
def joint_tail(u, qs):
    return np.array([((u[:, 0] < q) & (u[:, 1] < q)).mean() / q for q in qs])
NS2 = 400000
g_big = stats.norm.cdf(rng.standard_normal((NS2, 2)) @ Ln.T)
zz = rng.standard_normal((NS2, 2)) @ Lt.T
ww = rng.chisquare(nu_cop_hat, NS2) / nu_cop_hat
t_big = stats.t.cdf(zz / np.sqrt(ww)[:, None], nu_cop_hat)
U_all = np.column_stack([stats.t.cdf(z[:, j], nu_hat[j], 0, sc_marg[j]) for j in range(2)])
lam_t = 2 * stats.t.cdf(-np.sqrt((nu_cop_hat + 1) * (1 - rho_hat) / (1 + rho_hat)), nu_cop_hat + 1)
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(qs, joint_tail(U_all, qs), "o-", ms=4, color="#345995", label=f"实际残差 (n={T})")
ax.plot(qs, joint_tail(g_big, qs), lw=1.6, color="#8d99ae", label="高斯 copula（λ=0）")
ax.plot(qs, joint_tail(t_big, qs), lw=1.6, color="#c1121f", label=f"t-copula（λ≈{lam_t:.2f}）")
ax.axhline(lam_t, color="#c1121f", ls=":", lw=1)
ax.set_xlabel("尾部阈值 q"); ax.set_ylabel("P(U1<q, U2<q) / q")
ax.set_title("条件联合尾概率：q→0 时高斯 copula 归零，t-copula 收敛到 λ>0", fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT / "copula-tail-dependence.jpg", dpi=110, bbox_inches="tight"); plt.close(fig)

# 供正文引用的数字
print("nu_hat marginals:", nu_hat, "scales:", sc_marg, "rho_hat:", rho_hat, "nu_cop:", nu_cop_hat, "lambda:", lam_t)
print("garch fits:", fit)
