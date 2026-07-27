# -*- coding: utf-8 -*-
"""熵风险度量 EVaR 配图生成（Ahmadi-Javid 2012）"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize, stats
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/entropic-var-evar"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260727)

# ---------------------------------------------------------------
# 用两成分高斯混合模拟肥尾损失：平静态 + 危机态
# （混合正态有有限 MGF，EVaR 才有定义；t 分布无 MGF 不适用）
# ---------------------------------------------------------------
N = 200000
p_crisis = 0.10
crisis = rng.random(N) < p_crisis
ret = np.where(crisis,
               rng.normal(-0.004, 0.030, N),   # 危机态：负漂移、大波动
               rng.normal(0.0006, 0.009, N))   # 平静态
loss = -ret  # 损失

alpha = 0.95
z = 1 - alpha

# --- 经验 VaR / CVaR ---
var_emp = np.quantile(loss, alpha)
cvar_emp = loss[loss >= var_emp].mean()

# --- EVaR：EVaR_alpha = inf_{t>0} t * ln( E[e^{L/t}] / (1-alpha) ) ---
def evar_obj(t, x, a):
    # 用样本矩近似 MGF
    m = np.log(np.mean(np.exp(x / t))) - np.log(1 - a)
    return t * m

# 为数值稳定，用 loss 的样本
sub = loss[rng.integers(0, N, 40000)]
res = optimize.minimize_scalar(lambda t: evar_obj(t, sub, alpha),
                               bounds=(1e-4, 1.0), method="bounded")
evar_val = res.fun
t_star = res.x

print(f"VaR95={var_emp*100:.3f}%  CVaR95={cvar_emp*100:.3f}%  EVaR95={evar_val*100:.3f}%  t*={t_star:.4f}")

# ===============================================================
# 图 1：损失分布 + 三条风险线（VaR < CVaR < EVaR）
# ===============================================================
fig, ax = plt.subplots(figsize=(9, 5.2))
loss_pct = loss * 100
ax.hist(loss_pct, bins=300, range=(-8, 12), color="#4C72B0", alpha=0.72,
        edgecolor="none", density=True)
for val, c, name in [(var_emp, "#DD8452", "VaR"),
                     (cvar_emp, "#C44E52", "CVaR/ES"),
                     (evar_val, "#8172B3", "EVaR")]:
    ax.axvline(val * 100, color=c, lw=2.4, ls="--",
               label=f"{name}95 = {val*100:.2f}%")
ax.set_xlim(-8, 12)
ax.set_xlabel("损失 (%)")
ax.set_ylabel("概率密度")
ax.set_title("高斯混合（平静+危机）肥尾损失下：VaR ≤ CVaR ≤ EVaR", fontsize=13)
ax.legend(fontsize=11, loc="upper right")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/evar-vs-cvar-var.jpg", dpi=130)
plt.close(fig)

# ===============================================================
# 图 2：EVaR 的变分本质 —— t*g(t) 的下确界
# ===============================================================
ts = np.linspace(0.003, 0.10, 240)
vals = np.array([evar_obj(t, sub, alpha) for t in ts]) * 100
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(ts, vals, color="#8172B3", lw=2.4, label=r"$g(t)=t\cdot\ln\frac{E[e^{L/t}]}{1-\alpha}$")
ax.scatter([t_star], [evar_val * 100], color="#C44E52", s=90, zorder=5,
           label=f"下确界 = EVaR95 = {evar_val*100:.2f}%  (t*={t_star:.4f})")
ax.axhline(cvar_emp * 100, color="#DD8452", ls=":", lw=1.8,
           label=f"CVaR95 = {cvar_emp*100:.2f}%（EVaR 的下界锚）")
ax.set_xlabel("辅助变量 t（风险温度）")
ax.set_ylabel("目标函数 g(t)  (%)")
ax.set_title("EVaR 是一维凸优化的下确界：扫 t 找最小值", fontsize=13)
ax.legend(fontsize=10.5, loc="upper center")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/evar-variational.jpg", dpi=130)
plt.close(fig)

# ===============================================================
# 图 3：置信度 alpha 从 0.90 扫到 0.999，三度量的裂口
# ===============================================================
alphas = np.array([0.90, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999])
vars_, cvars_, evars_ = [], [], []
for a in alphas:
    v = np.quantile(loss, a)
    cv = loss[loss >= v].mean()
    r2 = optimize.minimize_scalar(lambda t: evar_obj(t, sub, a),
                                  bounds=(1e-4, 1.0), method="bounded")
    vars_.append(v * 100); cvars_.append(cv * 100); evars_.append(r2.fun * 100)

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(alphas, vars_, "o-", color="#DD8452", lw=2.2, label="VaR")
ax.plot(alphas, cvars_, "s-", color="#C44E52", lw=2.2, label="CVaR/ES")
ax.plot(alphas, evars_, "^-", color="#8172B3", lw=2.4, label="EVaR")
ax.fill_between(alphas, cvars_, evars_, color="#8172B3", alpha=0.12,
                label="EVaR 相对 CVaR 的保守裕度")
ax.set_xlabel("置信度 α")
ax.set_ylabel("风险度量 (%)")
ax.set_title("置信度越高，EVaR 相对 CVaR 的保守裕度越大", fontsize=13)
ax.legend(fontsize=10.5, loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/evar-confidence-sweep.jpg", dpi=130)
plt.close(fig)

print("EVaR images done:", os.listdir(OUT))
