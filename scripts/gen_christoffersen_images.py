#!/usr/bin/env python3
"""Christoffersen 独立性检验配图生成"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "public/images/christoffersen-independence-test"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(23)

# ---------- 模拟 GARCH(1,1)-t 收益 ----------
T = 1000
alpha_g, beta_g, nu = 0.15, 0.83, 6.0
omega = 1.0 * (1 - alpha_g - beta_g)  # 无条件方差归一
h = np.zeros(T)
r = np.zeros(T)
h[0] = 1.0
scale = np.sqrt((nu - 2) / nu)
for t in range(1, T):
    h[t] = omega + alpha_g * r[t-1]**2 + beta_g * h[t-1]
    z = rng.standard_t(nu) * scale
    r[t] = np.sqrt(h[t]) * z

p = 0.05  # 95% VaR：独立性检验在该层级功效最高

# 模型A：无条件正态 VaR（静态）
sigma_uc = r.std()
var_static = np.full(T, stats.norm.ppf(p) * sigma_uc)  # negative number
breach_static = (r < var_static).astype(int)

# 模型B：GARCH-t VaR（动态，正确模型）
var_dyn = stats.t.ppf(p, nu) * scale * np.sqrt(h)
breach_dyn = (r < var_dyn).astype(int)


def kupiec_pof(breach, p):
    T = len(breach); x = breach.sum()
    pi = x / T
    if x == 0:
        lr = -2 * (T * np.log(1 - p))
    else:
        lr = -2 * ((T - x) * np.log(1 - p) + x * np.log(p)
                   - (T - x) * np.log(1 - pi) - x * np.log(pi))
    return x, lr, 1 - stats.chi2.cdf(lr, 1)


def christoffersen_ind(breach):
    """LR_ind：一阶马尔可夫独立性检验"""
    b = breach
    n00 = n01 = n10 = n11 = 0
    for t in range(1, len(b)):
        if b[t-1] == 0 and b[t] == 0: n00 += 1
        elif b[t-1] == 0 and b[t] == 1: n01 += 1
        elif b[t-1] == 1 and b[t] == 0: n10 += 1
        else: n11 += 1
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def safelog(x):
        return np.log(x) if x > 0 else 0.0
    ll_h0 = (n00 + n10) * safelog(1 - pi) + (n01 + n11) * safelog(pi)
    ll_h1 = (n00 * safelog(1 - pi01) + n01 * safelog(pi01)
             + n10 * safelog(1 - pi11) + n11 * safelog(pi11))
    lr = -2 * (ll_h0 - ll_h1)
    return dict(n00=n00, n01=n01, n10=n10, n11=n11,
                pi01=pi01, pi11=pi11, lr=lr,
                pval=1 - stats.chi2.cdf(lr, 1))


xs, lr_pof_s, p_pof_s = kupiec_pof(breach_static, p)
xd, lr_pof_d, p_pof_d = kupiec_pof(breach_dyn, p)
ind_s = christoffersen_ind(breach_static)
ind_d = christoffersen_ind(breach_dyn)
lrcc_s = lr_pof_s + ind_s["lr"]; pcc_s = 1 - stats.chi2.cdf(lrcc_s, 2)
lrcc_d = lr_pof_d + ind_d["lr"]; pcc_d = 1 - stats.chi2.cdf(lrcc_d, 2)

print("=== 静态正态 VaR ===")
print(f"突破 {xs} 次, POF LR={lr_pof_s:.2f} p={p_pof_s:.3f}")
print(f"ind: {ind_s}")
print(f"LRcc={lrcc_s:.2f} p={pcc_s:.4f}")
print("=== GARCH-t VaR ===")
print(f"突破 {xd} 次, POF LR={lr_pof_d:.2f} p={p_pof_d:.3f}")
print(f"ind: {ind_d}")
print(f"LRcc={lrcc_d:.2f} p={pcc_d:.4f}")

# ---------- 图1：突破时间线对比 ----------
fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
ax = axes[0]
ax.plot(r, lw=0.5, color="#4C72B0", alpha=0.75, label="日收益")
ax.plot(var_static, lw=1.2, color="#DD8452", label="静态正态 VaR(95%)")
bi = np.where(breach_static == 1)[0]
ax.scatter(bi, r[bi], color="crimson", s=28, zorder=5, label=f"突破 ({xs} 次)")
ax.set_title("静态 VaR：突破在高波动期扎堆")
ax.legend(loc="lower left", fontsize=8)
ax = axes[1]
ax.plot(r, lw=0.5, color="#4C72B0", alpha=0.75)
ax.plot(var_dyn, lw=1.0, color="#55A868", label="GARCH-t VaR(95%)")
bi = np.where(breach_dyn == 1)[0]
ax.scatter(bi, r[bi], color="crimson", s=28, zorder=5, label=f"突破 ({xd} 次)")
ax.set_title("动态 VaR：突破均匀散布")
ax.set_xlabel("交易日")
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/christoffersen-breach-timeline.jpg", dpi=110)
plt.close()

# ---------- 图2：马尔可夫转移概率对比 ----------
fig, ax = plt.subplots(figsize=(8, 5))
labels = ["静态正态 VaR", "GARCH-t VaR"]
pi01_vals = [ind_s["pi01"], ind_d["pi01"]]
pi11_vals = [ind_s["pi11"], ind_d["pi11"]]
xpos = np.arange(2); w = 0.3
b1 = ax.bar(xpos - w/2, pi01_vals, w, label=r"$\pi_{01}$（昨天没破→今天破）", color="#4C72B0")
b2 = ax.bar(xpos + w/2, pi11_vals, w, label=r"$\pi_{11}$（昨天破→今天又破）", color="#C44E52")
ax.axhline(p, color="gray", ls="--", lw=1, label="名义突破率 5%")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.004,
            f"{b.get_height():.3f}", ha="center", fontsize=9)
ax.set_xticks(xpos); ax.set_xticklabels(labels)
ax.set_ylabel("条件突破概率")
ax.set_title("独立性检验的核心证据：突破后的第二天有多危险")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/christoffersen-transition-prob.jpg", dpi=110)
plt.close()

# ---------- 图3：三个检验的 p 值汇总 ----------
fig, ax = plt.subplots(figsize=(8.5, 5))
tests = ["POF\n(次数)", "IND\n(独立性)", "CC\n(联合)"]
pv_s = [p_pof_s, ind_s["pval"], pcc_s]
pv_d = [p_pof_d, ind_d["pval"], pcc_d]
xpos = np.arange(3); w = 0.32
b1 = ax.bar(xpos - w/2, pv_s, w, label="静态正态 VaR", color="#DD8452")
b2 = ax.bar(xpos + w/2, pv_d, w, label="GARCH-t VaR", color="#55A868")
ax.axhline(0.05, color="crimson", ls="--", lw=1.2, label="5% 显著性线")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012,
            f"{b.get_height():.3f}", ha="center", fontsize=9)
ax.set_xticks(xpos); ax.set_xticklabels(tests)
ax.set_ylabel("p 值")
ax.set_title("同样的突破次数，独立性检验拆穿静态模型")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/christoffersen-pvalue-summary.jpg", dpi=110)
plt.close()

# ---------- 图4：功效实验：波动聚集强度 vs 拒绝率 ----------
def run_power(persistence_list, n_sim=300, T=1000):
    """对不同 GARCH 持续性，静态 VaR 的 LR_ind 拒绝率"""
    rej_ind, rej_pof = [], []
    for pers in persistence_list:
        a = min(0.13, pers); b = max(pers - a, 0.0)
        cnt_i = cnt_p = 0
        for s in range(n_sim):
            rg = np.random.default_rng(1000 + s)
            h = np.zeros(T); rr = np.zeros(T)
            om = 1.0 * (1 - pers)  # unconditional var = 1
            om = max(om, 1e-4)
            h[0] = 1.0
            for t in range(1, T):
                h[t] = om + a * rr[t-1]**2 + b * h[t-1]
                rr[t] = np.sqrt(h[t]) * rg.standard_t(6.0) * scale
            v = stats.norm.ppf(0.05) * rr.std()
            br = (rr < v).astype(int)
            res = christoffersen_ind(br)
            if res["pval"] < 0.05: cnt_i += 1
            _, _, pp = kupiec_pof(br, 0.05)
            if pp < 0.05: cnt_p += 1
        rej_ind.append(cnt_i / n_sim); rej_pof.append(cnt_p / n_sim)
    return rej_ind, rej_pof

pers_list = [0.0, 0.5, 0.8, 0.9, 0.95, 0.97]
rej_ind, rej_pof = run_power(pers_list)
print("persistence:", pers_list)
print("rej_ind:", rej_ind)
print("rej_pof:", rej_pof)

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(pers_list, rej_ind, "o-", color="#C44E52", lw=2, label="IND 独立性检验拒绝率")
ax.plot(pers_list, rej_pof, "s--", color="#4C72B0", lw=2, label="POF 次数检验拒绝率")
ax.axhline(0.05, color="gray", ls=":", lw=1, label="名义5%（无聚集时应回到这里）")
ax.set_xlabel(r"波动持续性 $\alpha+\beta$")
ax.set_ylabel("对静态 VaR 的拒绝率")
ax.set_title("波动聚集越强，独立性检验越容易拆穿静态模型")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/christoffersen-power-curve.jpg", dpi=110)
plt.close()

print("done, images at", OUT)
