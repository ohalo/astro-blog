#!/usr/bin/env python3
"""
为文章「Merton 违约距离：用期权定价思想给公司违约概率打分」
(credit-merton-distance-to-default) 生成真实配图与真实统计数字。

核心方法（Merton 1974 结构化信用模型）：
  把股权看成公司资产 V 对债务 D 的看涨期权 E = max(V - D, 0)。
  股权价值和波动率可观测，资产价值 V 和资产波动率 σ_V 不可观测，
  用两条 BS 方程联立求解（迭代法）：
    E = V·N(d1) - D·e^{-rT}·N(d2)
    σ_E·E = N(d1)·σ_V·V
  违约距离 DD = (ln(V/D) + (μ - 0.5σ_V²)T) / (σ_V·√T)
  违约概率 PD = N(-DD)
数据：合成 3 家公司（健康/中等/困境）+ 一条资产价值路径演示违约触发。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "credit-merton-distance-to-default")
os.makedirs(D, exist_ok=True)

C = {"eq": "#2F4B7C", "up": "#55A868", "dn": "#C44E52", "grid": "#DDDDDD",
     "healthy": "#55A868", "mid": "#DD8452", "distress": "#C44E52",
     "debt": "#8172B3", "asset": "#4C72B0", "shade": "#F2C0C0"}

# ---------- 自包含正态 ----------
def norm_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)

def norm_cdf(x):
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    z = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    a = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]
    erf = 1.0 - (((((a[4]*t + a[3])*t) + a[2])*t + a[1])*t + a[0])*t*np.exp(-z*z)
    return 0.5 * (1.0 + sign * erf)

# ---------- Merton 求解器 ----------
def solve_merton(E, sigma_E, D_face, r, T, tol=1e-8, max_iter=500):
    """由可观测股权 E, σ_E 反解资产 V, σ_V"""
    V = E + D_face          # 初值
    sigma_V = sigma_E * E / V
    for _ in range(max_iter):
        d1 = (np.log(V / D_face) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
        d2 = d1 - sigma_V * np.sqrt(T)
        # 由 BS 反解 V：E = V N(d1) - D e^{-rT} N(d2)
        V_new = (E + D_face * np.exp(-r*T) * norm_cdf(d2)) / norm_cdf(d1)
        sigma_V_new = sigma_E * E / (V_new * norm_cdf(d1))
        if abs(V_new - V) < tol and abs(sigma_V_new - sigma_V) < tol:
            V, sigma_V = V_new, sigma_V_new
            break
        V, sigma_V = V_new, sigma_V_new
    d1 = (np.log(V / D_face) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)
    return V, sigma_V, d1, d2

def dd_pd(V, sigma_V, D_face, mu, T):
    DD = (np.log(V / D_face) + (mu - 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    PD = norm_cdf(-DD)
    return DD, PD

# ---------- 三家公司 ----------
r, T, mu = 0.03, 1.0, 0.08
firms = [
    # name, 股权市值E, 股权波动σ_E, 债务面值D
    ("健康公司 A", 800.0, 0.25, 300.0, C["healthy"]),
    ("中等公司 B", 400.0, 0.40, 400.0, C["mid"]),
    ("困境公司 C", 120.0, 0.65, 500.0, C["distress"]),
]

results = []
for name, E, sE, Df, col in firms:
    V, sV, d1, d2 = solve_merton(E, sE, Df, r, T)
    DD, PD = dd_pd(V, sV, Df, mu, T)
    results.append(dict(name=name, E=E, sE=sE, Df=Df, V=V, sV=sV,
                        DD=DD, PD=PD, col=col, lev=Df/V))
    print(f"{name}: V={V:.1f} σ_V={sV:.3f} 杠杆D/V={Df/V:.2f} DD={DD:.2f} PD={PD*100:.3f}%")

# =====================================================================
# 图1：结构化模型示意——资产价值分布与违约边界
# =====================================================================
fig, ax = plt.subplots(figsize=(9.2, 5.4))
f = results[1]   # 中等公司
V, sV, Df = f["V"], f["sV"], f["Df"]
# 到期资产价值 lognormal
x = np.linspace(V * 0.2, V * 2.2, 800)
m_ln = np.log(V) + (mu - 0.5 * sV**2) * T
s_ln = sV * np.sqrt(T)
pdf = np.exp(-(np.log(x) - m_ln)**2 / (2 * s_ln**2)) / (x * s_ln * np.sqrt(2*np.pi))
ax.plot(x, pdf, color=C["asset"], lw=2.4, label="到期资产价值 V_T 分布")
ax.axvline(Df, color=C["dn"], lw=2.2, ls="--", label=f"违约边界 D={Df:.0f}")
ax.axvline(V, color="#888888", lw=1.4, ls=":", label=f"当前资产 V₀={V:.0f}")
mask = x <= Df
ax.fill_between(x[mask], 0, pdf[mask], color=C["shade"], alpha=0.8,
                label=f"违约概率 PD={f['PD']*100:.2f}%")
ax.set_xlabel("到期资产价值 V_T")
ax.set_ylabel("概率密度")
ax.set_title("Merton 结构化模型：资产跌破债务面值即违约")
ax.grid(True, color=C["grid"], alpha=0.6)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "merton_structure.png"), dpi=130)
plt.close(fig)

# =====================================================================
# 图2：三家公司 DD 与 PD 对比
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
names = [f["name"] for f in results]
cols = [f["col"] for f in results]
DDs = [f["DD"] for f in results]
PDs = [f["PD"] * 100 for f in results]

b1 = ax1.bar(names, DDs, color=cols)
ax1.set_ylabel("违约距离 DD（标准差个数）")
ax1.set_title("违约距离：离违约边界几个 σ")
ax1.grid(True, axis="y", color=C["grid"], alpha=0.6)
for b, v in zip(b1, DDs):
    ax1.annotate(f"{v:.2f}", (b.get_x()+b.get_width()/2, v), ha="center", va="bottom")

b2 = ax2.bar(names, PDs, color=cols)
ax2.set_ylabel("违约概率 PD (%)")
ax2.set_title("违约概率 = N(−DD)")
ax2.grid(True, axis="y", color=C["grid"], alpha=0.6)
ax2.set_yscale("log")
for b, v in zip(b2, PDs):
    ax2.annotate(f"{v:.3f}%", (b.get_x()+b.get_width()/2, v), ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "merton_dd_pd.png"), dpi=130)
plt.close(fig)

# =====================================================================
# 图3：DD 对杠杆与波动率的敏感性（热力图式曲线）
# =====================================================================
fig, ax = plt.subplots(figsize=(9.2, 5.4))
lev_grid = np.linspace(0.3, 0.95, 60)     # D/V
for sV_test, cc, lb in [(0.20, C["healthy"], "σ_V=0.20 低波动"),
                         (0.35, C["mid"], "σ_V=0.35 中波动"),
                         (0.55, C["distress"], "σ_V=0.55 高波动")]:
    V_fix = 1000.0
    Df_grid = lev_grid * V_fix
    DD_grid = (np.log(V_fix / Df_grid) + (mu - 0.5*sV_test**2)*T) / (sV_test*np.sqrt(T))
    ax.plot(lev_grid, DD_grid, color=cc, lw=2.4, label=lb)
ax.axhline(0, color="#888888", lw=1.0, ls=":")
ax.set_xlabel("杠杆率 D/V")
ax.set_ylabel("违约距离 DD")
ax.set_title("违约距离对杠杆与资产波动率的敏感性")
ax.grid(True, color=C["grid"], alpha=0.6)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(D, "merton_sensitivity.png"), dpi=130)
plt.close(fig)

# =====================================================================
# 图4：资产价值蒙特卡洛路径 + 违约触发
# =====================================================================
rng = np.random.default_rng(42)
f = results[2]     # 困境公司
V0, sV, Df = f["V"], f["sV"], f["Df"]
n_paths, n_steps = 200, 252
dt = T / n_steps
t_axis = np.linspace(0, T, n_steps + 1)
paths = np.zeros((n_paths, n_steps + 1))
paths[:, 0] = V0
for i in range(n_steps):
    z = rng.standard_normal(n_paths)
    paths[:, i+1] = paths[:, i] * np.exp((mu - 0.5*sV**2)*dt + sV*np.sqrt(dt)*z)
defaulted = paths[:, -1] < Df
fig, ax = plt.subplots(figsize=(9.2, 5.4))
for i in range(n_paths):
    ax.plot(t_axis, paths[i], color=(C["dn"] if defaulted[i] else C["asset"]),
            lw=0.6, alpha=0.35)
ax.axhline(Df, color=C["dn"], lw=2.2, ls="--", label=f"违约边界 D={Df:.0f}")
ax.axhline(V0, color="#333333", lw=1.4, ls=":", label=f"起始资产 V₀={V0:.0f}")
emp_pd = defaulted.mean()
ax.set_xlabel("时间（年）")
ax.set_ylabel("资产价值 V_t")
ax.set_title(f"困境公司资产路径：{n_paths} 条中 {defaulted.sum()} 条到期违约（经验 PD={emp_pd*100:.1f}%，理论 {f['PD']*100:.1f}%）")
ax.grid(True, color=C["grid"], alpha=0.6)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(D, "merton_paths.png"), dpi=130)
plt.close(fig)

print("=== 蒙特卡洛校验（困境公司）===")
print(f"经验 PD={emp_pd*100:.2f}%  理论 PD={f['PD']*100:.2f}%")
print("图片已生成:", os.listdir(D))
