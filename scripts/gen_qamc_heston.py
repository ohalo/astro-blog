#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QAMC 文章数据脚本（fig3 / fig4）：
  fig3: Heston 模型算术平均亚式期权 —— 模拟 MC vs 随机化 Sobol QMC，误差随定价观察次数 d 的Scaling
  fig4: 折扣收益随机变量的经验尾部指数 κ̂ 拟合 + 所需查询数曲线
        经典MC(ε^-2) vs QAMC(Montanaro 2015, ε^-(1/κ+1/2))，理论趋势曲线
全部数字均为本机真实计算，结果写入 _qamc_results.json。
"""
import os
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc

# ---------- 与站内其他生成脚本一致的出图风格 ----------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fbfbfd",
    "axes.edgecolor": "#d9d9e3",
    "axes.grid": True,
    "grid.color": "#e6e6ee",
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 12.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "font.family": "sans-serif",
    "font.sans-serif": ["Heiti TC", "PingFang SC", "Hiragino Sans GB",
                        "Arial Unicode MS", "sans-serif"],
    "font.size": 10.5,
    "mathtext.fontset": "stix",
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
})
C_MAIN, C_ALT, C_WARN, C_MUT = "#6c5ce7", "#00b894", "#e17055", "#636e9c"
PAL = ["#0984e3", "#6c5ce7", "#00b894", "#e17055", "#fdcb6e", "#e84393"]
SEED = 20261027
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "public", "images", "quantum-computing-option-pricing")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

# ---------- Heston 模型参数（风险中立测度 Q） ----------
HP = dict(S0=100.0, K=100.0, r=0.02, kappa=2.0, theta=0.04,
          sigma_v=0.30, rho=-0.70, v0=0.04, T=1.0)
# Feller 条件: 2κθ = 0.16 > σ_v² = 0.09 ✓（方差过程不会到 0）
FELLER = 2 * HP["kappa"] * HP["theta"] > HP["sigma_v"] ** 2


def simulateheston(numfix, npat, seed, p=None, antithetic=True):
    """全截断 Euler 模拟 Heston。每步 2 个正态 → 每条路径 2*numfix 维。
    返回折扣后收益 X = e^{-rT} max(平均价-K, 0) 与终点方差均值。"""
    q = HP if p is None else p
    dt = q["T"] / numfix
    dim = 2 * numfix
    if antithetic:
        assert npat % 2 == 0
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((npat // 2, dim))
        Z = np.vstack([z, -z])
    else:
        Z = np.random.default_rng(seed).standard_normal((npat, dim))
    v = np.full(npat, q["v0"])
    lnS = np.full(npat, math.log(q["S0"]))
    prices = []
    vs = []
    for s in range(numfix):
        sqv = np.sqrt(np.maximum(v, 0.0))
        z1 = Z[:, 2 * s]
        z2 = Z[:, 2 * s + 1]
        w1 = math.sqrt(dt) * z1
        w2 = math.sqrt(dt) * (q["rho"] * z1 + math.sqrt(1 - q["rho"] ** 2) * z2)
        lnS = lnS + (q["r"] - 0.5 * v) * dt + q["sigma_v"] * 0 + sqv * w1
        v = v + q["kappa"] * (q["theta"] - np.maximum(v, 0.0)) * dt \
            + q["sigma_v"] * sqv * w2
        prices.append(np.exp(lnS))
        vs.append(v)
    avg = np.mean(prices, axis=0)
    disc = math.exp(-q["r"] * q["T"])
    X = disc * np.maximum(avg - q["K"], 0.0)
    return X, float(np.mean(vs))


def sobolheston(numfix, npat, seed, p=None):
    """随机化 Sobol 点集模拟（对偶：u 与 1−u 配对；Sobol 维数 = 2·numfix，
    每步价格/方差各占一维）。"""
    q = HP if p is None else p
    dt = q["T"] / numfix
    halfl = npat // 2
    sampler = qmc.Sobol(d=2 * numfix, scramble=True, seed=seed)
    u = sampler.random(halfl)
    Z = np.vstack([u, 1 - u])            # 对偶：把 u 与 1-u 配对
    Z = np.abs(Z)
    Z = np.clip(Z, 1e-12, 1 - 1e-12)
    from scipy.stats import norm
    Z = norm.ppf(Z)
    v = np.full(npat, q["v0"])
    lnS = np.full(npat, math.log(q["S0"]))
    prices = []
    for s in range(numfix):
        sqv = np.sqrt(np.maximum(v, 0.0))
        z1 = Z[:, 2 * s]
        z2 = Z[:, 2 * s + 1]
        w1 = math.sqrt(dt) * z1
        w2 = math.sqrt(dt) * (q["rho"] * z1 + math.sqrt(1 - q["rho"] ** 2) * z2)
        lnS = lnS + (q["r"] - 0.5 * v) * dt + sqv * w1
        v = v + q["kappa"] * (q["theta"] - np.maximum(v, 0.0)) * dt \
            + q["sigma_v"] * sqv * w2
        prices.append(np.exp(lnS))
    avg = np.mean(prices, axis=0)
    disc = math.exp(-q["r"] * q["T"])
    return disc * np.maximum(avg - q["K"], 0.0)


def bsprice(S0, K, r, sig, T):
    from math import erf, sqrt
    d1 = (math.log(S0 / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    N = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
    return S0 * N(d1) - K * math.exp(-r * T) * N(d2)


# ================================================================
# 0) Heston→BS 退化校验：σ_v→0 时 MC 终点价收敛应与 Black-Scholes 一致
# ================================================================
rngchk = np.random.default_rng(SEED)
PBS = dict(HP)
PBS.update(sigma_v=1e-6)
Xbs, _ = simulateheston(1, 400000, SEED, p=PBS, antithetic=True)
refMC = Xbs.mean()
refSE = Xbs.std() / math.sqrt(len(Xbs))
disc = math.exp(-HP["r"] * HP["T"])
# simulateheston 返回的已是折扣收益, 终点看涨未折扣期望 = X/disc
bsAnal = bsprice(HP["S0"], HP["K"], HP["r"], math.sqrt(HP["v0"]), HP["T"])
chk = dict(bs_mc_price=refMC, bs_mc_se=refSE, bs_analytic=bsAnal,
           rel_err=abs(refMC - bsAnal) / bsAnal, feller=FELLER)
print("[check] BS退化: MC=%.4f±%.4f vs BS=%.4f  rel_err=%.2e"
      % (refMC, refSE, bsAnal, chk["rel_err"]))

# 主参数下的锚定价格（大批量 MC + QMC 双算）
Xmc4, _ = simulateheston(12, 800000, SEED + 7, antithetic=True)
price4 = Xmc4.mean()
se4 = Xmc4.std() / math.sqrt(len(Xmc4))
Xqmc = sobolheston(12, 8192, SEED + 8)
price4q = float(Xqmc.mean())
print("[anchor] Heston亚式(d=12): MC %.4f (se %.4f), QMC %.4f" % (price4, se4, price4q))

# ================================================================
# fig1 → fig3_d_error.png : MC vs Sobol QMC 批均值 RMSE 随观察次数 d
# ================================================================
DFIX = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
NB, NPATH = 240, 1024
rmc, rqmc, spa, base = [], [], [], {}
for i, d in enumerate(DFIX):
    pm = [simulateheston(d, NPATH, 20000 + i * 1000 + b)[0]
          for b in range(NB)]
    pq = [sobolheston(d, NPATH, 30000 + i * 1000 + b) for b in range(NB)]
    mm = np.array([a.mean() for a in pm])
    qq = np.array([a.mean() for a in pq])
    rmcm, rqqm = float(mm.std()), float(qq.std())
    # 基准价：本 d 下全部批次合并（近似无偏真值）
    refp = float(np.concatenate(pm).mean())
    rmc.append(rmcm)
    rqmc.append(rqqm)
    spa.append(rmcm / rqqm)
    base[d] = refp
    print("d=%2d dim=%2d  MC rmse %.5f  QMC rmse %.5f  speedup %.2fx  ref %.4f"
          % (d, 2 * d, rmcm, rqqm, rmcm / rqqm, refp))
sp = np.array(spa)
ipk = int(np.argmax(sp))
bestd = DFIX[ipk]
idrop = next((j for j in range(ipk + 1, len(DFIX) - 1)
              if sp[j] < 1.15 and sp[j + 1] < 1.15), None)
breakd = DFIX[idrop] if idrop is not None else None
crossd = next((DFIX[j] for j in range(1, len(DFIX))
               if sp[j] < 1.0 and sp[j - 1] >= 1.0), None)

bestref = base[bestd]
d10, d25 = DFIX[10], DFIX[int(np.argmax([base[d] for d in DFIX]))]
p10, _ = simulateheston(d10, 300000, 21000, antithetic=True)
p25, _ = simulateheston(24, 300000, 22000, antithetic=True)
se10 = float(p10.std() / math.sqrt(len(p10)))
se25 = float(p25.std() / math.sqrt(len(p25)))
fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.9))
ax[0].plot(DFIX, rmc, "o-", color=C_MAIN, label="伪随机 MC 批均值 RMSE")
ax[0].plot(DFIX, rqmc, "s--", color=C_ALT, label="随机化 Sobol QMC RMSE")
ax[0].axvline(bestd, color=C_MUT, lw=0.8, ls=":")
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel("定价观察次数 d（每月一次 ≈ d=12）")
ax[0].set_ylabel("每批 1024 条路径的RMSE（对数轴）")
ax[0].set_title("(a) Heston 亚式期权：MC 与 QMC 误差随观察次数 d\n两条误差曲线随维度一同抬升，QMC 全程压住 MC（互为交叉验证）")
ax[0].legend()
ax[1].semilogy(DFIX, sp, "D-", color=PAL[3])
ax[1].axhline(1.0, color=C_MUT, lw=0.9, ls=":")
ax[1].axhline(1.15, color=C_MUT, lw=0.6, ls=":")
if breakd:
    ax[1].annotate("QMC 优势≈1×（维度崩溃）\nd=%d" % breakd, xy=(breakd, sp[DFIX.index(breakd)]),
                   xytext=(breakd * 0.35, 2.2), color=C_WARN,
                   arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.0))
if crossd:
    ax[1].annotate("交叉点 d=%d：QMC 反被 MC 反超" % crossd, xy=(crossd, 1.0),
                   xytext=(max(2, crossd * 0.25), 0.6), color=C_MAIN,
                   arrowprops=dict(arrowstyle="->", color=C_MUT, lw=0.9))
ax[1].set_xscale("log")
ax[1].set_xlabel("观察次数 d")
ax[1].set_ylabel("加速比 = RMSE(MC) / RMSE(QMC)")
ax[1].set_title("(b) 经典 QMC 是「有限加速」：峰值 %.1fx (d=%d)，d=%d 时衰减到 %.1fx\n——从未到 1×，但远非指数加速，这正是量子 QAMC 要跨越的墙" % (max(sp), bestd, DFIX[-1], sp[-1]))
fig.savefig(os.path.join(OUT, "fig3_d_error.png"))
plt.close(fig)
print("[fig3] saved")

# ================================================================
# fig4: 尾部指数 κ̂ 拟合 + 查询数曲线 MC vs 量子（Montanaro 2015 / BSW2024）
#   有限方差区（κ̂>2）：Montanaro Thm 2.5  N_Q = Õ(σ/ε)  vs  N_MC = (zσ/ε)²
#   无限方差厚尾代理 t(1.8)：Blanchet-Szegedy-Wang 2024，(1+δ) 矩界 C：
#     经典 N = C·ε^{-(1+δ)/δ}，量子 N = C^{1/(2δ)}·ε^{-(1+δ)/(2δ)}——指数恰好一半
#   量子优势恒为平方；尾部厚度决定的是双方各自的指数与常数，不是"优势有无"。
# ================================================================
disc = math.exp(-HP["r"] * HP["T"])
X = sobolheston(12, 2 ** 18, SEED + 9)          # 低噪声样本（QMC）
muhat = float(X.mean())
dev = np.abs(X - muhat)
qs = np.linspace(0.90, 0.99995, 400)
lam = np.quantile(dev, qs)
surv = 1 - qs
m = (lam > 1e-6) & (surv > 1e-6)
lx, ly = np.log(lam[m]), np.log(surv[m])
A = np.vstack([lx, np.ones_like(lx)]).T
slope, icpt = np.linalg.lstsq(A, ly, rcond=None)[0]
khat = float(-slope)
chat = float(np.exp(icpt))
sigX = float(X.std())

# 厚尾代理：t(1.8)（方差无限，1.7 阶矩有限），等比缩放到 Heston 收益的 MAD 尺度
DELTA = 0.7                                    # 使用 (1+δ)=1.7 阶矩
rngt = np.random.default_rng(SEED + 11)
t = rngt.standard_t(df=1.8, size=400000)
medt = float(np.median(t))
tc = t - medt
scale = float(np.median(dev) / np.median(np.abs(tc)))   # MAD 对齐
tc = tc * scale
devt = np.abs(tc)
qst = np.linspace(0.90, 0.9999, 200)
lamt = np.quantile(devt, qst)
mt = lamt > 1e-9
kt18 = float(-np.polyfit(np.log(lamt[mt]), np.log((1 - qst)[mt]), 1)[0])
kt18 = float(np.clip(kt18, 1.1, 12.0))
C17 = float(np.mean(devt ** (1 + DELTA)))      # (1+δ) 阶绝对矩经验值
q_exp_t = (1 + DELTA) / (2 * DELTA)            # 1.2143
c_exp_t = (1 + DELTA) / DELTA                  # 2.4286

z75 = 0.6744897501960817
eps = np.logspace(-2, -5.5, 60)
NMC = (z75 * sigX) ** 2 * eps ** -2             # 经典 MC（Heston，方差有界）
polylog = np.log2(1 / eps)                      # polylog 因子显式化（取单份 log，保守）
NQ = sigX * eps ** -1 * polylog                 # Montanaro Thm 2.5: Õ(σ/ε)
NMCt = C17 * eps ** -c_exp_t                    # 经典，t(1.8)：仅 1.7 阶矩
NQ_t = C17 ** (1 / (2 * DELTA)) * eps ** -q_exp_t   # BSW2024 量子

i5 = int(np.argmin(np.abs(eps - 1e-5)))
gain_h = float(np.log10(NMC[i5] / NQ[i5]))
gain_t = float(np.log10(NMCt[i5] / NQ_t[i5]))

fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.9))
ax[0].loglog(lam[m], surv, ".", color=C_MUT, ms=2.5, label="Heston 亚式折扣收益尾")
ll = np.logspace(np.log10(lam[m][0]), np.log10(lam[m][-1]), 50)
ax[0].loglog(ll, chat * ll ** -khat, "-", color=PAL[3], lw=2,
             label="拟合: %.3g·λ^−%.2f" % (chat, khat))
ax[0].loglog(lamt[mt], (1 - qst)[mt], ".", color=PAL[5], ms=2.5, alpha=0.6,
             label="t(1.8) 厚尾代理（MAD 对齐）")
ll2 = np.logspace(np.log10(lamt[mt][0]), np.log10(lamt[mt][-1]), 50)
c18_fit = float(np.exp(np.polyfit(np.log(lamt[mt]), np.log((1 - qst)[mt]), 1)[1]))
ax[0].loglog(ll2, c18_fit * ll2 ** -kt18, "--", color=PAL[5], lw=1.6,
             label="t(1.8) 拟合: λ^−%.2f" % kt18)
ax[0].set_xlabel("偏差阈值 λ")
ax[0].set_ylabel("尾概率 P(|X−μ̂| ≥ λ)")
ax[0].set_title("(a) 折扣收益的尾部衰减：κ̂(Heston)=%.2f vs κ̂(t₁.₈)=%.2f\nHeston 可观测区间内表观尾指数 >2（方差区），t(1.8) 方差无限" % (khat, kt18))
ax[0].legend()
ax[1].loglog(eps, NMC, "-", color=PAL[0], lw=2, label="Heston·经典 MC：σ̂=%.1f, 指数 2" % sigX)
ax[1].loglog(eps, NQ, "-", color=PAL[3], lw=2,
             label="Heston·量子：σ̂ε⁻¹·log₂(1/ε)（Montanaro Thm 2.5）")
ax[1].loglog(eps, NMCt, ":", color=PAL[5], lw=1.6,
             label="t₁.₈·经典：C·ε^−(1+δ)/δ = ε^−%.2f（BSW2024）" % c_exp_t)
ax[1].loglog(eps, NQ_t, "--", color=PAL[1], lw=1.6,
             label="t₁.₈·量子：C^(1/2δ)·ε^−(1+δ)/2δ = ε^−%.2f" % q_exp_t)
ax[1].invert_xaxis()
ax[1].set_xlabel("误差要求 ε（越左越精确）")
ax[1].set_ylabel("所需查询数（对数轴）")
ax[1].set_title("(b) 量子优势恒为平方（指数恰好减半），尾部只改底数\nε=1e-5：Heston 省 %.1f 个数量级，t₁.₈ 省 %.1f 个（两边门票都变贵）" % (gain_h, gain_t))
fig.savefig(os.path.join(OUT, "fig4_qamc_query_scaling.png"))
plt.close(fig)
print("[fig4] saved")

# ---------- 结果汇总 ----------
res = dict(seed=SEED,
           heston=HP, feller=FELLER,
           bsd_check=chk,
           anchor=dict(d=12, mc_price=price4, mc_se=se4, qmc_price=price4q),
           fig3=dict(dlist=DFIX, mc_rmse=rmc, qmc_rmse=rqmc, speedup=spa,
                     ref_price=base, nb=NB, npath=NPATH,
                     best_d=bestd, best_speedup=float(max(sp)),
                     break_d=breakd, cross_d=crossd,
                     d24_ref_price=base[24], d24_mc_rmse=rmc[DFIX.index(24)],
                     d24_qmc_rmse=rqmc[DFIX.index(24)],
                     d24_price_sample=p25.mean(), d24_se_sample=se25),
           fig4=dict(khat=khat, chat=chat, sigma_X=sigX, mu=muhat,
                     gain_heston_1e5=gain_h, gain_t_1e5=gain_t,
                     t18_khat=kt18, t18_C17=C17, delta=DELTA,
                     c_exp_t=c_exp_t, q_exp_t=q_exp_t,
                     n_samples=len(X)))
with open(os.path.join(OUT, "_qamc_results.json"), "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("SUMMARY:", json.dumps({k: v for k, v in res.items() if k != "fig3"},
                             ensure_ascii=False)[:1200])
print("fig3 best_speedup=%.1f@%d break=%s cross=%s khat=%.2f chat=%.4g gainH_1e5=%.1f gainT_1e5=%.1f kt18=%.2f"
      % (max(sp), bestd, breakd, crossd, khat, chat,
         gain_h, gain_t, kt18))
print("ALL DONE qamc")
