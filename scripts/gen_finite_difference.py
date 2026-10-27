#!/usr/bin/env python3
"""
有限差分法期权定价 - 配图生成
生成 4 张真实图表到 public/images/finite-difference-option-pricing/
含显式 / 隐式(向后欧拉) / Crank-Nicolson 三种格式,
并对欧式看涨用 BS 闭式做自洽校验, 对美式看跌做 Early-Exercise 处理。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Hiragino Sans GB", "PingFang SC", "STHeiti", "Songti SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
from scipy.stats import norm
from scipy.linalg import solve_banded

OUT = "public/images/finite-difference-option-pricing"
os.makedirs(OUT, exist_ok=True)

# ---------- 参数 ----------
S0, K, r, sig, T = 100.0, 100.0, 0.05, 0.20, 1.0
Smax = 300.0

def bs_call(S, K, r, sig, T):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, r, sig, T):
    if T <= 0 or sig <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ---------- 有限差分核心 ----------
def fd_price(theta, Ns, Nt, option="call", american=False):
    """theta: 0=显式, 1=向后欧拉, 0.5=Crank-Nicolson"""
    dS = Smax / Ns
    dt = T / Nt
    S_grid = np.linspace(0, Smax, Ns + 1)
    i_arr = np.arange(Ns + 1)
    alpha = 0.5 * sig ** 2 * dt * i_arr ** 2            # ½σ² dt i²
    delta = 0.5 * r * dt * i_arr                         # ½ r dt i
    gamma = r * dt

    # 终值 (τ=0, t=T)
    if option == "call":
        V = np.maximum(S_grid - K, 0.0)
    else:
        V = np.maximum(K - S_grid, 0.0)

    # 系数矩阵 A (三对角), 用 solve_banded
    # A: sub = -theta*(alpha-delta), diag = 1+theta*(2*alpha+gamma), sup = -theta*(alpha+delta)
    sub = -theta * (alpha - delta)
    diag = 1.0 + theta * (2 * alpha + gamma)
    sup = -theta * (alpha + delta)

    ab = np.zeros((3, Ns + 1))
    ab[0, 1:] = sup[:-1]     # 上对角
    ab[1, :] = diag
    ab[2, :-1] = sub[1:]     # 下对角

    for step in range(Nt):
        tau = (step + 1) * dt
        # 右端 b
        if theta == 0.0:
            # 显式
            Vnew = (alpha - delta) * np.concatenate([[0], V[:-1]]) \
                   + (1 - 2 * alpha - gamma) * V \
                   + (alpha + delta) * np.concatenate([V[1:], [0]])
        else:
            b = V.copy()
            b += (1 - theta) * (
                (alpha - delta) * np.concatenate([[0], V[:-1]])
                + (-2 * alpha - gamma) * V
                + (alpha + delta) * np.concatenate([V[1:], [0]])
            )
            Vnew = solve_banded((1, 1), ab, b)
        # 边界
        Vnew[0] = 0.0 if option == "call" else K * np.exp(-r * tau)
        Vnew[Ns] = Smax - K * np.exp(-r * tau) if option == "call" else 0.0
        if american:
            intrinsic = np.maximum(K - S_grid, 0.0)
            Vnew = np.maximum(Vnew, intrinsic)
        V = Vnew
    # 在 S0 处线性插值
    idx = int(S0 / dS)
    idx = min(max(idx, 0), Ns - 1)
    w = (S0 - S_grid[idx]) / dS
    return V[idx] * (1 - w) + V[idx + 1] * w

def interp_profile(theta, Ns, Nt, option="call"):
    dS = Smax / Ns
    dt = T / Nt
    S_grid = np.linspace(0, Smax, Ns + 1)
    i_arr = np.arange(Ns + 1)
    alpha = 0.5 * sig ** 2 * dt * i_arr ** 2
    delta = 0.5 * r * dt * i_arr
    gamma = r * dt
    V = np.maximum(S_grid - K, 0.0) if option == "call" else np.maximum(K - S_grid, 0.0)
    sub = -theta * (alpha - delta); diag = 1 + theta * (2 * alpha + gamma); sup = -theta * (alpha + delta)
    ab = np.zeros((3, Ns + 1)); ab[0, 1:] = sup[:-1]; ab[1, :] = diag; ab[2, :-1] = sub[1:]
    for step in range(Nt):
        tau = (step + 1) * dt
        if theta == 0.0:
            Vnew = (alpha - delta) * np.concatenate([[0], V[:-1]]) + (1 - 2 * alpha - gamma) * V + (alpha + delta) * np.concatenate([V[1:], [0]])
        else:
            b = V + (1 - theta) * ((alpha - delta) * np.concatenate([[0], V[:-1]]) + (-2 * alpha - gamma) * V + (alpha + delta) * np.concatenate([V[1:], [0]]))
            Vnew = solve_banded((1, 1), ab, b)
        Vnew[0] = 0.0 if option == "call" else K * np.exp(-r * tau)
        Vnew[Ns] = Smax - K * np.exp(-r * tau) if option == "call" else 0.0
        V = Vnew
    return S_grid, V

# 自洽校验
bs_c = bs_call(S0, K, r, sig, T)
cn = fd_price(0.5, 200, 2000, "call")
imp = fd_price(1.0, 200, 2000, "call")
exp = fd_price(0.0, 200, 4000, "call")
print(f"BS 欧式看涨闭式 = {bs_c:.4f}")
print(f"Crank-Nicolson   = {cn:.4f}  (偏差 {cn-bs_c:+.4f})")
print(f"向后欧拉(隐式)   = {imp:.4f}  (偏差 {imp-bs_c:+.4f})")
print(f"显式             = {exp:.4f}  (偏差 {exp-bs_c:+.4f})")

bs_p = bs_put(S0, K, r, sig, T)
am_p = fd_price(1.0, 200, 2000, "put", american=True)
print(f"BS 欧式看跌闭式 = {bs_p:.4f}")
print(f"美式看跌(FD+Early-Exercise) = {am_p:.4f}  (美式溢价 {am_p-bs_p:+.4f})")

# ============ 图 1: 收敛性 (误差 vs 网格) ============
Ns_list = [40, 60, 90, 120, 160, 200]
err_exp, err_imp, err_cn = [], [], []
for Ns in Ns_list:
    Nt = max(2000, 4 * Ns)
    err_exp.append(abs(fd_price(0.0, Ns, Nt * 2, "call") - bs_c))
    err_imp.append(abs(fd_price(1.0, Ns, Nt, "call") - bs_c))
    err_cn.append(abs(fd_price(0.5, Ns, Nt, "call") - bs_c))
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.loglog(Ns_list, err_exp, "o-", label="显式 (Explicit)")
ax.loglog(Ns_list, err_imp, "s-", label="向后欧拉 (Implicit)")
ax.loglog(Ns_list, err_cn, "^-", label="Crank-Nicolson")
ax.set_xlabel("空间网格数 $N_S$"); ax.set_ylabel("对 BS 闭式的绝对误差")
ax.set_title("三种格式对欧式看涨的收敛速度")
ax.legend(); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fd_convergence.png", dpi=130); plt.close(fig)

# ============ 图 2: 价格曲线对比 ============
Sg, Vcn = interp_profile(0.5, 200, 2000, "call")
_, Vimp = interp_profile(1.0, 200, 2000, "call")
_, Vexp = interp_profile(0.0, 200, 4000, "call")
Vbs = np.array([bs_call(s, K, r, sig, T) for s in Sg])
mask = Sg <= 200
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(Sg[mask], Vbs[mask], "k-", lw=2.4, label="BS 闭式")
ax.plot(Sg[mask], Vcn[mask], "r--", lw=1.4, label="Crank-Nicolson")
ax.plot(Sg[mask], Vimp[mask], "b:", lw=1.4, label="向后欧拉")
ax.plot(Sg[mask], Vexp[mask], "g-.", lw=1.0, label="显式")
ax.axvline(K, color="gray", ls=":", alpha=0.6)
ax.set_xlabel("标的现价 $S$"); ax.set_ylabel("看涨期权价值 $V(S,T)$")
ax.set_title("欧式看涨价格曲线: 三种 FD 格式 vs BS")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fd_price_curve.png", dpi=130); plt.close(fig)

# ============ 图 3: 价值曲面 (S, τ) ============
Ns, Nt = 120, 600
dS = Smax / Ns; dt = T / Nt
S_grid = np.linspace(0, Smax, Ns + 1); i_arr = np.arange(Ns + 1)
alpha = 0.5 * sig ** 2 * dt * i_arr ** 2; delta = 0.5 * r * dt * i_arr; gamma = r * dt
V = np.maximum(S_grid - K, 0.0)
sub = -0.5 * (alpha - delta); diag = 1 + 0.5 * (2 * alpha + gamma); sup = -0.5 * (alpha + delta)
ab = np.zeros((3, Ns + 1)); ab[0, 1:] = sup[:-1]; ab[1, :] = diag; ab[2, :-1] = sub[1:]
Vmat = np.zeros((Nt + 1, Ns + 1)); Vmat[0] = V
for step in range(Nt):
    tau = (step + 1) * dt
    b = V + 0.5 * ((alpha - delta) * np.concatenate([[0], V[:-1]]) + (-2 * alpha - gamma) * V + (alpha + delta) * np.concatenate([V[1:], [0]]))
    Vnew = solve_banded((1, 1), ab, b)
    Vnew[0] = 0.0; Vnew[Ns] = Smax - K * np.exp(-r * tau)
    V = Vnew; Vmat[step + 1] = V
fig, ax = plt.subplots(figsize=(7.2, 4.6))
Tg = np.linspace(0, T, Nt + 1)
X, Y = np.meshgrid(S_grid, Tg)
c = ax.contourf(X, Y, Vmat, levels=20, cmap="viridis")
ax.set_xlabel("标的 $S$"); ax.set_ylabel("剩余期限 $\\tau=T-t$")
ax.set_title("Crank-Nicolson 看涨期权价值曲面 $V(S,\\tau)$")
fig.colorbar(c, ax=ax, label="V")
fig.tight_layout(); fig.savefig(f"{OUT}/fd_surface.png", dpi=130); plt.close(fig)

# ============ 图 4: 显式格式稳定性 (CFL) ============
Sg_s, Vstable = interp_profile(0.0, 200, 4000, "call")      # 稳定 dt
Sg_u, Vunst = interp_profile(0.0, 200, 250, "call")         # 违反 CFL
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(Sg_s, Vstable, "b-", lw=2, label="显式 (稳定: $N_t=4000$, 满足 CFL)")
ax.plot(Sg_s, Vbs, "k--", lw=1.6, label="BS 闭式")
ax.plot(Sg_u[:120], Vunst[:120], "r-", lw=1.0, alpha=0.8, label="显式 (不稳: $N_t=250$, 违反 CFL 震荡发散)")
ax.set_xlabel("标的现价 $S$"); ax.set_ylabel("看涨期权价值 $V$")
ax.set_title("显式格式的稳定性: CFL 条件不满足即发散")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fd_stability.png", dpi=130); plt.close(fig)

print("✅ 有限差分配图已生成:", os.listdir(OUT))
