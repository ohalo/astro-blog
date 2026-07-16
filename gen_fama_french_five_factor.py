#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「Fama-French 五因子实战拆解：把规模价值再加盈利与投资」生成真实配图与统计数字。

核心逻辑(Fama & French 2015, "A Five-Factor Asset Pricing Model"):
  - FF3 (MKT/SMB/HML) 在 1980s-2000s 暴露两大漏洞:
      ① "盈利能力异常": 高盈利公司未来回报系统性更高, HML 解释不了
      ② "投资异常": 保守投资公司未来回报更高, HML 也解释不了
  - 2015 版加两个因子:
      RMW = 高盈利减低盈利 (Robust-minus-Weak, 按 OP=营业盈利/账面权益排序)
      CMA = 保守减激进投资 (Conservative-minus-Aggressive, 按投资=总资产增长排序)
    变成 5 因子: MKT, SMB, HML, RMW, CMA。
  - 验证(同 q 文章的合成面板, 保证两文一致可比):
      ① RMW / CMA 十分位单调(盈利高→高收益, 投资保守→高收益)
      ② RMW L-S / CMA L-S 多空累计净值显著为正(且 CMA 与 q 文章的 IA 反向——印证两文同构)
      ③ 18 个测试组合上: FF5 平均 R² 高于 FF3, 平均 |alpha| 更低; GRS 检验 F 统计量下降
        —— 说明 RMW/CMA 确实"补上了 FF3 的洞"
  - GRS 检验(H0: 所有 alpha=0): F = (T-N-K)/N · (1+μ_f'Σ_f⁻¹μ_f)⁻¹ · α'Σ_res⁻¹α

全部数字由文中 Python 真实计算(自包含, 仅依赖 numpy/scipy)。
图片:
  ff5_decile_rmw_cma.png  —— RMW 与 CMA 十分位下月平均收益(双栏)
  ff5_factor_curves.png   —— RMW / CMA / HML 多空累计净值(三因子同图)
  ff5_r2_compare.png      —— FF3 vs FF5: 平均 R² 与平均 |alpha| 对比
  ff5_grs_alpha.png       —— 18 组合 alpha 对比: FF3(灰) vs FF5(红), 加 GRS F 统计量
"""
import os
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "fama-french-five-factor")
os.makedirs(D, exist_ok=True)

rng = np.random.default_rng(20260717)
N, T = 300, 240
n_dec = 10

# ---------- 同 q 文章的面板(保证一致可比) ----------
size = np.zeros((N, T)); q = np.zeros((N, T)); prof = np.zeros((N, T))
size[:, 0] = rng.normal(0, 1, N); q[:, 0] = rng.normal(0, 0.6, N); prof[:, 0] = rng.normal(0, 1, N)
for t in range(1, T):
    size[:, t] = 0.95 * size[:, t-1] + 0.12 * rng.normal(0, 1, N)
    q[:, t]    = 0.88 * q[:, t-1]    + 0.20 * rng.normal(0, 1, N)
    prof[:, t] = 0.90 * prof[:, t-1] + 0.18 * rng.normal(0, 1, N)
inv = 0.55 * q + 0.30 * rng.normal(0, 1, (N, T)) + 0.10 * rng.normal(0, 1, (N, T))
bm  = -0.50 * inv + 0.20 * size + 0.45 * rng.normal(0, 1, (N, T))

def zcol(X):
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-9)
INV_z, PROF_z, SIZE_z, BM_z = zcol(inv), zcol(prof), zcol(size), zcol(bm)

mkt = rng.normal(0, 0.04, T)
beta = 0.8 + 0.3 * rng.normal(0, 1, N)
ret = np.zeros((N, T + 1))
for t in range(T):
    ret[:, t + 1] = (beta * mkt[t]
                     + 0.0050 * PROF_z[:, t]
                     - 0.0050 * INV_z[:, t]
                     + 0.0003 * SIZE_z[:, t]
                     + rng.normal(0, 0.045, N))

# ---------- 2x3 组合(规模 × 排序变量) ----------
def factor_2x3(sort_a, sort_b):
    out = {}
    for t in range(T):
        oa = np.argsort(sort_a[:, t]); ob = np.argsort(sort_b[:, t])
        ra = np.empty(N); ra[oa] = np.arange(N)
        rb = np.empty(N); rb[ob] = np.arange(N)
        na = np.clip(ra // (N // 2), 0, 1)            # 0=small,1=big
        nb = np.clip(rb // (N // 3), 0, 2)            # 0/1/2 = L/M/H
        for ia in (0, 1):
            for ib in (0, 1, 2):
                sel = (na == ia) & (nb == ib)
                key = f"{'S' if ia==0 else 'B'}{['L','M','H'][ib]}"
                out.setdefault(key, np.zeros(T))[t] = ret[sel, t + 1].mean()
    return out

p_smb = factor_2x3(SIZE_z, BM_z)
p_ia  = factor_2x3(SIZE_z, INV_z)
p_op  = factor_2x3(SIZE_z, PROF_z)

smb = (p_smb["SL"] + p_smb["SM"] + p_smb["SH"]) / 3 \
    - (p_smb["BL"] + p_smb["BM"] + p_smb["BH"]) / 3
hml = (p_smb["SH"] + p_smb["BH"]) / 2 - (p_smb["SL"] + p_smb["BL"]) / 2
MKT = mkt.copy()

# RMW (高 OP - 低 OP): OP = 盈利 prof
rmw = (p_op["SH"] + p_op["BH"]) / 2 - (p_op["SL"] + p_op["BL"]) / 2
# CMA (保守投资 - 激进投资): 低 inv 减高 inv = 与 q 文章 IA 反向
cma = (p_ia["SL"] + p_ia["BL"]) / 2 - (p_ia["SH"] + p_ia["BH"]) / 2

# ---------- 十分位(ROE=prof, CMA 排序=inv) ----------
def decile_ls(sig):
    dec_ret = np.zeros((n_dec, T))
    for t in range(T):
        order = np.argsort(sig[:, t]); ranks = np.empty(N); ranks[order] = np.arange(N)
        d = np.clip(ranks // (N // n_dec), 0, n_dec - 1)
        for k in range(n_dec):
            sel = (d == k); dec_ret[k, t] = ret[sel, t + 1].mean()
    return dec_ret, dec_ret[-1, :] - dec_ret[0, :]
dec_op, ls_op = decile_ls(PROF_z)       # RMW 代理: 高盈利→高收益
dec_cma, ls_cma = decile_ls(-INV_z)     # CMA 代理: 低投资(保守)→高收益

def perf(ls):
    a = 12.0 * ls.mean(); vol = ls.std() * np.sqrt(12); return a, vol, a / vol
aop, vop, sop = perf(ls_op)
acm, vcm, scm = perf(ls_cma)
ahm, vhm, shm = perf(hml)

# ---------- 测试组合: 18 个(2x3 size×bm, size×prof, size×inv) ----------
test_ports = {}
test_ports.update({f"bm_{k}": v for k, v in p_smb.items()})
test_ports.update({f"op_{k}": v for k, v in p_op.items()})
test_ports.update({f"inv_{k}": v for k, v in p_ia.items()})
port_names = list(test_ports.keys()); P = len(port_names)
Y = np.array([test_ports[k] for k in port_names])   # P x T

X_ff3 = np.vstack([np.ones(T), MKT, smb, hml]).T
X_ff5 = np.vstack([np.ones(T), MKT, smb, hml, rmw, cma]).T

def ols_full(Y, X):
    # Y: P x T (组合), X: T x K (因子+截距)
    coef, *_ = np.linalg.lstsq(X, Y.T, rcond=None)   # coef: K x P
    resid = Y.T - X @ coef                            # T x P
    return coef.T, resid.T                            # P x K, P x T

coef3, resid3 = ols_full(Y, X_ff3)
coef5, resid5 = ols_full(Y, X_ff5)
alphas3 = coef3[:, 0]; alphas5 = coef5[:, 0]
r2_ff3 = 1 - (resid3**2).sum(1) / ((Y - Y.mean(1, keepdims=True))**2).sum(1)
r2_ff5 = 1 - (resid5**2).sum(1) / ((Y - Y.mean(1, keepdims=True))**2).sum(1)
r2_ff3_m, r2_ff5_m = r2_ff3.mean(), r2_ff5.mean()
absa_ff3_m = np.abs(alphas3).mean(); absa_ff5_m = np.abs(alphas5).mean()

# ---------- GRS 检验 ----------
def grs_test(alphas, resid, factors, K):
    Np, Tt = resid.shape
    fcov = np.cov(factors, rowvar=False) + 1e-10 * np.eye(K)
    fmu = factors.mean(0)
    inv_f = np.linalg.inv(fcov)
    g = 1 + fmu @ inv_f @ fmu
    rcov = np.cov(resid) + 1e-10 * np.eye(Np)
    rinv = np.linalg.inv(rcov)
    aSa = alphas @ rinv @ alphas
    F = (Tt - Np - K) / Np * (1 / g) * aSa
    return F, g
grs_ff3, _ = grs_test(alphas3, resid3, X_ff3[:, 1:], 3)
grs_ff5, _ = grs_test(alphas5, resid5, X_ff5[:, 1:], 5)

metrics = {
    "N": N, "T_months": T, "P_test_ports": P,
    "rmw_ls_ann_ret": round(float(aop), 4), "rmw_ls_sharpe": round(float(sop), 3),
    "cma_ls_ann_ret": round(float(acm), 4), "cma_ls_sharpe": round(float(scm), 3),
    "hml_ls_ann_ret": round(float(ahm), 4), "hml_ls_sharpe": round(float(shm), 3),
    "r2_ff3_mean": round(float(r2_ff3_m), 3), "r2_ff5_mean": round(float(r2_ff5_m), 3),
    "absalpha_ff3_mean": round(float(absa_ff3_m), 4), "absalpha_ff5_mean": round(float(absa_ff5_m), 4),
    "grs_ff3": round(float(grs_ff3), 2), "grs_ff5": round(float(grs_ff5), 2),
}
with open(os.path.join(D, "_metrics.txt"), "w") as f:
    for k, vv in metrics.items():
        f.write(f"{k}={vv}\n")
print("METRICS", metrics)

# ============ 图1: RMW 与 CMA 十分位 ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
x = np.arange(n_dec); w = 0.8
mp = dec_op.mean(1); mc = dec_cma.mean(1)
ax1.bar(x, mp, w, color="#2F4B7C")
ax1.axhline(0, color="black", lw=0.8)
ax1.set_xticks(x); ax1.set_xticklabels([f"D{k+1}" for k in range(n_dec)])
ax1.set_title("盈利能力 RMW 十分位（D1=最弱 → D10=最强）", fontsize=12, fontweight="bold")
ax1.set_xlabel("OP 十分位"); ax1.set_ylabel("下月平均收益"); ax1.grid(True, alpha=0.3, axis="y")
ax1.annotate(f"L-S 年化 {aop:.1%} / Sharpe {sop:.2f}", xy=(9, mp[9]),
             xytext=(3.3, mp[9]*1.4), fontsize=10, color="#2F4B7C", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#2F4B7C"))
ax2.bar(x, mc, w, color="#55A868")
ax2.axhline(0, color="black", lw=0.8)
ax2.set_xticks(x); ax2.set_xticklabels([f"D{k+1}" for k in range(n_dec)])
ax2.set_title("投资 CMA 十分位（D1=最激进 → D10=最保守）", fontsize=12, fontweight="bold")
ax2.set_xlabel("投资 INV 十分位"); ax2.set_ylabel("下月平均收益"); ax2.grid(True, alpha=0.3, axis="y")
ax2.annotate(f"L-S 年化 {acm:.1%} / Sharpe {scm:.2f}", xy=(9, mc[9]),
             xytext=(2.8, mc[9]*1.25), fontsize=10, color="#55A868", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#55A868"))
plt.tight_layout()
plt.savefig(os.path.join(D, "ff5_decile_rmw_cma.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============ 图2: RMW / CMA / HML 多空累计净值 ============
cum_op = np.cumprod(1 + ls_op); cum_cm = np.cumprod(1 + ls_cma); cum_hm = np.cumprod(1 + hml)
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.plot(range(T), cum_op, color="#2F4B7C", lw=1.9, label=f"RMW L-S Sharpe {sop:.2f}")
ax.plot(range(T), cum_cm, color="#55A868", lw=1.9, label=f"CMA L-S Sharpe {scm:.2f}")
ax.plot(range(T), cum_hm, color="#9970AB", lw=1.9, label=f"HML L-S Sharpe {shm:.2f}")
ax.set_title("RMW / CMA / HML 多空累计净值（2015 新增两因子显著为正）", fontsize=11.5, fontweight="bold")
ax.set_xlabel("月份"); ax.set_ylabel("净值（起始=1）"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "ff5_factor_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============ 图3: FF3 vs FF5 R² / |alpha| ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0))
b1 = ax1.bar(["FF3", "FF5"], [r2_ff3_m, r2_ff5_m], color=["#999999", "#2F4B7C"], width=0.5)
ax1.set_title("18 测试组合平均 R²（越高越好）", fontsize=12, fontweight="bold")
ax1.set_ylabel("平均 R²"); ax1.set_ylim(0, 1); ax1.grid(True, alpha=0.3, axis="y")
for bb, vv in zip(b1, [r2_ff3_m, r2_ff5_m]):
    ax1.text(bb.get_x()+bb.get_width()/2, vv+0.02, f"{vv:.2f}", ha="center", fontsize=11, fontweight="bold")
b2 = ax2.bar(["FF3", "FF5"], [absa_ff3_m, absa_ff5_m], color=["#999999", "#C44E52"], width=0.5)
ax2.set_title("18 测试组合平均 |alpha|（越低越好）", fontsize=12, fontweight="bold")
ax2.set_ylabel("平均 |alpha| (月)"); ax2.grid(True, alpha=0.3, axis="y")
for bb, vv in zip(b2, [absa_ff3_m, absa_ff5_m]):
    ax2.text(bb.get_x()+bb.get_width()/2, vv+0.0005, f"{vv:.3f}", ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(D, "ff5_r2_compare.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============ 图4: 18 组合 alpha 对比 + GRS ============
y = np.arange(P)
fig, ax = plt.subplots(figsize=(9.5, 7.0))
ax.hlines(y, alphas3, alphas5, color="#CCCCCC", lw=1.2, zorder=1)
ax.scatter(alphas3, y, color="#999999", s=45, label=f"FF3 alpha (GRS F={grs_ff3:.0f})", zorder=2)
ax.scatter(alphas5, y, color="#C44E52", s=45, label=f"FF5 alpha (GRS F={grs_ff5:.0f})", zorder=2)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels([f"{k}" for k in port_names], fontsize=7)
ax.set_title("18 测试组合 alpha：FF3(灰) → FF5(红) 普遍收缩", fontsize=12, fontweight="bold")
ax.set_xlabel("alpha (月)"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="x")
ax.annotate(f"GRS F 统计量: FF3={grs_ff3:.0f} → FF5={grs_ff5:.0f}\n(下降说明更多组合被模型解释)",
            xy=(alphas5.max(), P-3), xytext=(alphas5.min(), 2), fontsize=9, color="#C44E52",
            arrowprops=dict(arrowstyle="->", color="#C44E52"))
plt.tight_layout()
plt.savefig(os.path.join(D, "ff5_grs_alpha.png"), dpi=150, bbox_inches="tight")
plt.close()
print("DONE fama-french-five-factor images")
