#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成「Merton 跳跃扩散资产定价：把跳变成分从连续波动里剥出来」文章配图 + 核心数值。
数据：numpy 合成 Merton(1976) 跳跃扩散，从零实现闭式期权定价(BS 无穷级数)并 Monte Carlo 校验，
把期权价格分解为连续扩散(n=0)与跳变(n>=1)两部分。
配图保存到 public/images/merton-jump-diffusion-asset/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import norm

for f in ["PingFang SC", "Heiti SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130

OUT = "public/images/merton-jump-diffusion-asset"
os.makedirs(OUT, exist_ok=True)

# ===== 参数 =====
S0, T, r, q = 100.0, 1.0, 0.03, 0.0
SIG, LAM, M, DELTA = 0.20, 3.0, -0.10, 0.15     # 扩散波动 / 年跳跃强度 / 跳 size 对数均值 / 对数 std
KAPPA = np.exp(M + 0.5 * DELTA ** 2) - 1.0       # 价格跳平均超额 = E[J]-1


def bs_call(S, K, T_, r_, q_, sig):
    if sig <= 0 or T_ <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r_ - q_ + 0.5 * sig ** 2) * T_) / (sig * np.sqrt(T_))
    d2 = d1 - sig * np.sqrt(T_)
    return S * np.exp(-q_ * T_) * norm.cdf(d1) - K * np.exp(-r_ * T_) * norm.cdf(d2)


def merton_call(S, K, sig, lam, m, delta, Nmax=60):
    """Merton(1976) 闭式：对 Poisson 跳数次 n 求和，每项是一个调整的 BS 价格。"""
    tot = 0.0
    pn = np.exp(-lam * T)                      # p_0 = e^{-λT}
    for n in range(Nmax + 1):
        sn = np.sqrt(sig ** 2 + n * delta ** 2 / T)
        Sn = S * np.exp(-lam * KAPPA * T + n * (m + 0.5 * delta ** 2))
        tot += pn * bs_call(Sn, K, T, r, q, sn)
        pn *= (lam * T) / (n + 1)              # p_{n+1} = p_n * (λT)/(n+1)
    return tot


def mc_merton_call(S, K, sig, lam, m, delta, M_=120000, seed=7):
    rng = np.random.default_rng(seed)
    Nr = rng.poisson(lam * T, M_)
    Z = rng.standard_normal(M_)
    total = int(Nr.sum())
    allj = rng.normal(m, delta, total)
    idx = np.cumsum(Nr)
    jsum = np.zeros(M_)
    if idx[0] > 0:
        jsum[0] = allj[:idx[0]].sum()
    for i in range(1, M_):
        if idx[i] > idx[i - 1]:
            jsum[i] = allj[idx[i - 1]:idx[i]].sum()
    drift = (r - q - lam * KAPPA - 0.5 * sig ** 2) * T
    logret = drift + sig * np.sqrt(T) * Z + jsum
    ST = S * np.exp(logret)
    return np.exp(-r * T) * np.maximum(ST - K, 0).mean()


# 校验闭式 vs MC（在若干行权价上）
Ks = np.linspace(70, 130, 13)
cf = [merton_call(S0, k, SIG, LAM, M, DELTA) for k in Ks]
mc = [mc_merton_call(S0, k, SIG, LAM, M, DELTA) for k in Ks]
max_err = max(abs(a - b) for a, b in zip(cf, mc))
bs_only = [bs_call(S0, k, T, r, q, SIG) for k in Ks]

print("=== 核心统计（用于正文）===")
print(f"参数: S={S0}, T={T}, r={r}, σ={SIG}, λ={LAM}(年), 跳size~LogN(m={M},δ={DELTA}), KAPPA={KAPPA:.4f}")
print(f"闭式 vs Monte Carlo(12万路径) 最大误差 = {max_err:.4f}（应 < 0.05，确认公式正确）")
print(f"ATM(K=100) 期权价: Merton闭式={merton_call(S0,100,SIG,LAM,M,DELTA):.3f}  "
      f"MC={mc_merton_call(S0,100,SIG,LAM,M,DELTA):.3f}  BS(无跳)={bs_call(S0,100,T,r,q,SIG):.3f}")

# ===== 图1：GBM vs 跳跃扩散 价格路径 =====
SEED = 20260830
rng = np.random.default_rng(SEED)
steps = 252
dt = T / steps
tgrid = np.linspace(0, T, steps + 1)
# 纯扩散
gbm = S0 * np.exp(np.cumsum((r - 0.5 * SIG ** 2) * dt + SIG * np.sqrt(dt) * rng.standard_normal(steps)))
gbm = np.concatenate([[S0], gbm])
# 跳跃扩散
rng2 = np.random.default_rng(SEED + 1)
Nj = rng2.poisson(LAM * dt, steps)
jd = np.empty(steps + 1); jd[0] = S0
cont = (r - LAM * KAPPA - 0.5 * SIG ** 2) * dt
for t in range(1, steps + 1):
    z = rng2.standard_normal()
    jumps = Nj[t - 1]
    jret = np.sum(rng2.normal(M, DELTA, jumps)) if jumps > 0 else 0.0
    jd[t] = jd[t - 1] * np.exp(cont + SIG * np.sqrt(dt) * z + jret)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(tgrid, gbm, color="#2a9d8f", lw=1.6, label="纯几何布朗运动 (无跳)")
ax.plot(tgrid, jd, color="#1b6ca8", lw=1.6, label="Merton 跳跃扩散")
# 标记跳跃点
jump_times = tgrid[1:][Nj > 0]
jump_vals = jd[1:][Nj > 0]
ax.scatter(jump_times, jump_vals, color="#d1495b", s=40, zorder=5, label=f"跳跃 ({int(Nj.sum())} 次)")
ax.set_title("同一起点、同一漂移：跳跃扩散路径出现不连续的跳", fontsize=12.5)
ax.set_xlabel("时间(年)"); ax.set_ylabel("价格")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/mjd_paths.png"); plt.close(fig)

# ===== 图2：1年对数收益分布 + 厚尾 =====
rng3 = np.random.default_rng(SEED + 2)
Nr = rng3.poisson(LAM * T, 200000)
Z = rng3.standard_normal(200000)
allj = rng3.normal(M, DELTA, int(Nr.sum()))
idx = np.cumsum(Nr)
jsum = np.zeros(200000)
if idx[0] > 0:
    jsum[0] = allj[:idx[0]].sum()
for i in range(1, 200000):
    if idx[i] > idx[i - 1]:
        jsum[i] = allj[idx[i - 1]:idx[i]].sum()
drift = (r - LAM * KAPPA - 0.5 * SIG ** 2) * T
logret = drift + SIG * np.sqrt(T) * Z + jsum
mu = logret.mean(); vv = logret.var()
exk = np.mean((logret - mu) ** 4) / vv ** 2 - 3.0
gauss = rng3.normal(mu, np.sqrt(vv), 200000)
fig, ax = plt.subplots(figsize=(9.5, 5))
ax.hist(logret, bins=120, density=True, color="#1b6ca8", alpha=0.6, label="Merton 跳跃扩散收益")
ax.hist(gauss, bins=120, density=True, histtype="step", color="#d1495b", lw=2, label="同方差高斯(无跳)")
ax.set_yscale("log")
ax.set_title(f"厚尾：Merton 超额峰度 {exk:.1f} vs 高斯 0（跳变把尾部加肥）", fontsize=12)
ax.set_xlabel("1年对数收益"); ax.set_ylabel("概率密度(log)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/mjd_return_dist.png"); plt.close(fig)

# ===== 图3：期权价格 vs 行权价 + 跳变溢价 =====
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(Ks, cf, color="#1b6ca8", lw=2.2, marker="o", ms=4, label="Merton 跳跃扩散")
ax.plot(Ks, bs_only, color="#d1495b", lw=2.0, ls="--", marker="s", ms=4, label="BS(无跳, 连续波动)")
ax.plot(Ks, np.array(cf) - np.array(bs_only), color="#6a4c93", lw=1.8, label="跳变溢价 = 两者之差")
ax.set_title("期权价格曲线与『跳变溢价』：低行权价处跳变贡献最大", fontsize=12)
ax.set_xlabel("行权价 K"); ax.set_ylabel("看涨期权价格")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/mjd_option_curve.png"); plt.close(fig)

# ===== 图4：BS 级数收敛 + 跳变成分占比 =====
Nmax = 40
prices_conv = [merton_call(S0, 100, SIG, LAM, M, DELTA, Nmax=n) for n in range(1, Nmax + 1)]
final = merton_call(S0, 100, SIG, LAM, M, DELTA, Nmax=60)
# 分解：n=0(扩散) vs n>=1(跳变)
pn = np.exp(-LAM * T)
diff_part = 0.0
for n in range(61):
    sn = np.sqrt(SIG ** 2 + n * DELTA ** 2 / T)
    Sn = S0 * np.exp(-LAM * KAPPA * T + n * (M + 0.5 * DELTA ** 2))
    contrib = pn * bs_call(Sn, 100, T, r, q, sn)
    if n == 0:
        diff_share = contrib
    else:
        diff_part += contrib
    pn *= (LAM * T) / (n + 1)
jump_share = final - diff_share
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, Nmax + 1), prices_conv, color="#1b6ca8", lw=2, marker=".", label="截断到 n 项的 BS 级数和")
ax.axhline(final, color="gray", ls=":", lw=1.5, label=f"收敛价 {final:.3f}")
ax.set_title(f"闭式级数 {final:.2f} 中：连续扩散占 {diff_share/final:.1%}，跳变成分占 {jump_share/final:.1%}",
             fontsize=11.5)
ax.set_xlabel("级数截断项数 n"); ax.set_ylabel("看涨期权价格")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/mjd_convergence.png"); plt.close(fig)

print(f"ATM 价格分解: 连续扩散 n=0 贡献 {diff_share:.3f} ({diff_share/final:.1%}), "
      f"跳变 n>=1 贡献 {jump_share:.3f} ({jump_share/final:.1%})")
print(f"图片已保存到 {OUT}")
