# -*- coding: utf-8 -*-
"""POT/GPD 实验：GARCH-t(5) 模拟日损失，超阈值建模"""
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# ---- 1. GARCH(1,1)-t(5) 模拟 40 年日收益 ----
n = 252 * 40
nu = 5.0
omega, alpha, beta = 2.0e-6, 0.09, 0.89
z = stats.t.rvs(nu, size=n, random_state=rng) * np.sqrt((nu - 2) / nu)
sig2 = np.empty(n)
sig2[0] = omega / (1 - alpha - beta)
r = np.empty(n)
r[0] = np.sqrt(sig2[0]) * z[0]
for t in range(1, n):
    sig2[t] = omega + alpha * r[t - 1] ** 2 + beta * sig2[t - 1]
    r[t] = np.sqrt(sig2[t]) * z[t]
loss = -r * 100  # 百分比损失，正值=亏损

print(f"样本量 {n}, 日损失均值 {loss.mean():.4f}%, std {loss.std():.3f}%, 峰度 {stats.kurtosis(loss):.2f}")

# ---- 2. 阈值选择：平均超额函数 ----
qs = np.arange(0.80, 0.995, 0.005)
u_grid = np.quantile(loss, qs)
mean_excess = np.array([loss[loss > u].mean() - u for u in u_grid])

u = np.quantile(loss, 0.95)  # 选 95% 分位为阈值
exc = loss[loss > u] - u
n_exc = len(exc)
print(f"阈值 u={u:.3f}% (95%分位), 超额数 {n_exc}, 占比 {n_exc/n:.3%}")

# ---- 3. GPD 拟合 ----
xi, loc_, sc = stats.genpareto.fit(exc, floc=0)
print(f"GPD: xi={xi:.4f}, beta={sc:.4f} (理论 xi=1/5=0.2)")

# bootstrap CI for xi
boot_xi = []
for _ in range(500):
    bs = rng.choice(exc, size=n_exc, replace=True)
    bxi, _, _ = stats.genpareto.fit(bs, floc=0)
    boot_xi.append(bxi)
boot_xi = np.array(boot_xi)
print(f"xi bootstrap 90% CI: [{np.percentile(boot_xi,5):.3f}, {np.percentile(boot_xi,95):.3f}]")

# KS 检验
ks = stats.kstest(exc, "genpareto", args=(xi, 0, sc))
print(f"GPD KS: stat={ks.statistic:.4f}, p={ks.pvalue:.4f}")
# 指数分布（xi=0）对照
loc_e, sc_e = stats.expon.fit(exc, floc=0)
ks_e = stats.kstest(exc, "expon", args=(0, sc_e))
print(f"Expon KS: stat={ks_e.statistic:.4f}, p={ks_e.pvalue:.6f}")

# ---- 4. 尾部分位数外推 ----
zeta = n_exc / n  # P(X>u)
def pot_var(p):
    return u + sc / xi * ((( (1 - p) / zeta) ** (-xi)) - 1)
def pot_es(p):
    v = pot_var(p)
    return (v + sc - xi * u) / (1 - xi)

mu_n, sd_n = loss.mean(), loss.std()
for p in [0.99, 0.995, 0.999, 0.9999]:
    v_pot = pot_var(p)
    v_emp = np.quantile(loss, p)
    v_norm = mu_n + sd_n * stats.norm.ppf(p)
    print(f"p={p}: POT VaR={v_pot:.2f}%  经验={v_emp:.2f}%  正态={v_norm:.2f}%  POT ES={pot_es(p):.2f}%")

# bootstrap CI for 99.9% VaR
boot_v = []
for _ in range(500):
    idx = rng.choice(n_exc, size=n_exc, replace=True)
    bs = exc[idx]
    bxi, _, bsc = stats.genpareto.fit(bs, floc=0)
    if abs(bxi) < 1e-6: continue
    boot_v.append(u + bsc / bxi * (((0.001 / zeta) ** (-bxi)) - 1))
boot_v = np.array(boot_v)
print(f"99.9% VaR bootstrap 90% CI: [{np.percentile(boot_v,5):.2f}, {np.percentile(boot_v,95):.2f}]")

# ---- 5. 阈值敏感性 ----
sens_q = np.arange(0.85, 0.995, 0.01)
sens_xi, sens_var999, sens_n = [], [], []
for q in sens_q:
    uu = np.quantile(loss, q)
    ee = loss[loss > uu] - uu
    sxi, _, ssc = stats.genpareto.fit(ee, floc=0)
    zz = len(ee) / n
    sens_xi.append(sxi)
    sens_var999.append(uu + ssc / sxi * (((0.001 / zz) ** (-sxi)) - 1))
    sens_n.append(len(ee))
print("阈值敏感性 (q, n_exc, xi, VaR99.9):")
for q, ne, sx, sv in zip(sens_q, sens_n, sens_xi, sens_var999):
    print(f"  {q:.2f}  {ne:5d}  {sx:.3f}  {sv:.2f}")

# ================= 图 =================
import os
OUT = "/Users/halo/workspace/astro-blog/public/images/peaks-over-threshold-pot"
os.makedirs(OUT, exist_ok=True)

# 图1：损失序列 + 阈值 + 超额点
fig, ax = plt.subplots(figsize=(11, 4.6))
xx = np.arange(n)
ax.plot(xx, loss, lw=0.3, color="#8899aa", alpha=0.7)
mask = loss > u
ax.scatter(xx[mask], loss[mask], s=6, color="#d64545", zorder=3, label=f"超阈值观测 ({n_exc} 个)")
ax.axhline(u, color="#2a6f97", lw=1.4, ls="--", label=f"阈值 u = {u:.2f}%（95% 分位）")
ax.set_title("日损失序列与超阈值观测：POT 只用尾巴，不浪费块内信息")
ax.set_xlabel("交易日")
ax.set_ylabel("日损失 (%)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUT}/pot-threshold-exceedances.jpg", dpi=110, pil_kwargs={"quality": 88})
plt.close(fig)

# 图2：平均超额图
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.plot(u_grid, mean_excess, "o-", ms=4, color="#2a6f97")
ax.axvline(u, color="#d64545", ls="--", lw=1.2, label=f"选定阈值 u={u:.2f}%")
ax.set_title("平均超额函数：GPD 域内应近似线性，斜率符号 = 尾部性格")
ax.set_xlabel("候选阈值 u (%)")
ax.set_ylabel("平均超额 E[X−u | X>u] (%)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/pot-mean-excess.jpg", dpi=110, pil_kwargs={"quality": 88})
plt.close(fig)

# 图3：GPD 拟合直方图 + QQ
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
xs = np.linspace(0, exc.max(), 400)
axes[0].hist(exc, bins=60, density=True, color="#a8c6df", edgecolor="white", label="超额经验分布")
axes[0].plot(xs, stats.genpareto.pdf(xs, xi, 0, sc), color="#d64545", lw=2,
             label=f"GPD(ξ={xi:.2f}, β={sc:.2f})")
axes[0].plot(xs, stats.expon.pdf(xs, 0, sc_e), color="#888", lw=1.5, ls="--",
             label="指数分布（ξ=0）")
axes[0].set_yscale("log")
axes[0].set_title(f"超额分布拟合（对数纵轴）：GPD p={ks.pvalue:.2f}，指数 p={ks_e.pvalue:.4f}")
axes[0].set_xlabel("超额 (%)"); axes[0].legend(fontsize=8)
th_q = stats.genpareto.ppf(np.arange(1, n_exc + 1) / (n_exc + 1), xi, 0, sc)
axes[1].scatter(th_q, np.sort(exc), s=8, color="#2a6f97", alpha=0.6)
lim = [0, max(th_q.max(), exc.max()) * 1.02]
axes[1].plot(lim, lim, color="#d64545", lw=1.2)
axes[1].set_title("GPD QQ 图：深尾不脱轨")
axes[1].set_xlabel("GPD 理论分位数 (%)"); axes[1].set_ylabel("经验分位数 (%)")
fig.tight_layout()
fig.savefig(f"{OUT}/pot-gpd-fit-qq.jpg", dpi=110, pil_kwargs={"quality": 88})
plt.close(fig)

# 图4：阈值敏感性双图
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
axes[0].plot(sens_q, sens_xi, "o-", ms=4, color="#2a6f97")
axes[0].axhline(0.2, color="#d64545", ls="--", lw=1.2, label="理论 ξ = 1/ν = 0.2")
axes[0].set_title("形状参数 ξ vs 阈值：偏差-方差走廊")
axes[0].set_xlabel("阈值分位数"); axes[0].set_ylabel("ξ 估计"); axes[0].legend()
ax2 = axes[1]
ax2.plot(sens_q, sens_var999, "s-", ms=4, color="#7a5195")
ax2.axhline(np.quantile(loss, 0.999), color="#888", ls=":", lw=1.2, label="经验 99.9% 分位（样本内）")
ax2.set_title("99.9% VaR 外推 vs 阈值：估计应在平台区稳定")
ax2.set_xlabel("阈值分位数"); ax2.set_ylabel("VaR 99.9% (%)"); ax2.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/pot-threshold-sensitivity.jpg", dpi=110, pil_kwargs={"quality": 88})
plt.close(fig)

print("figures saved to", OUT)
