#!/usr/bin/env python3
"""
为文章「跳跃扩散与 Bates 模型：给波动率加跳，给期权定价加尾」(bates-jump-diffusion)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

图表：
  1. bates_paths.png        三类过程的样本路径：GBM / Merton 跳跃扩散 / Bates(SV+跳)
  2. bates_terminal_hist.png 到期价格分布：对数正态 vs 跳扩散（肥尾）
  3. bates_iv_smile.png      隐含波动率微笑/偏斜：BS 平 vs Merton 微笑 vs Bates 偏斜
  4. bates_price_compare.png 看涨期权价格-行权价曲线：BS vs Merton vs Bates

数值校验：用特征函数(FFT)定价，与蒙特卡洛对照，确认一致。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import norm

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "bates-jump-diffusion")
os.makedirs(D, exist_ok=True)

C = {"gbm": "#4C72B0", "merton": "#C44E52", "bates": "#55A868",
     "grid": "#DDDDDD", "jump": "#d62728"}
np.set_printoptions(suppress=True, precision=4)

# ============================================================
# 模型参数（一只指数期权的半年期参数，贴近真实量级）
# ============================================================
S0, r, q, T = 100.0, 0.02, 0.01, 0.5
v0, kappa, theta, sigma_v, rho = 0.04, 2.0, 0.04, 0.3, -0.7   # Heston(SV) 部分
lam, muJ, sigJ = 0.5, -0.05, 0.10                            # 跳跃部分：负跳→左偏
m = np.exp(muJ + 0.5 * sigJ ** 2) - 1.0                      # 单跳均值 E[e^J-1]


# ============================================================
# Bates 特征函数（Gatheral 约定 + 独立对数正态跳）
# ============================================================
def bates_cf(u, v0=v0):
    """phi(u) = E[exp(i u ln S_T)]，u 可为复数数组。"""
    iu = 1j * u
    d = np.sqrt((kappa - rho * sigma_v * iu) ** 2 + sigma_v ** 2 * (iu + u ** 2))
    g = (kappa - rho * sigma_v * iu + d) / (kappa - rho * sigma_v * iu - d)
    exp_dT = np.exp(d * T)
    Ccoef = (kappa * theta / sigma_v ** 2) * (
        (kappa - rho * sigma_v * iu + d) * T - 2 * np.log((1 - g * exp_dT) / (1 - g))
    )
    Dcoef = (kappa - rho * sigma_v * iu + d) / sigma_v ** 2 * (1 - exp_dT) / (1 - g * exp_dT)
    # Heston 扩散部分（含 (r-q) 漂移项）
    phi_heston = np.exp(iu * (np.log(S0) + (r - q) * T) + Ccoef + Dcoef * v0)
    # 跳跃部分 + 漂移项补偿（保证贴现股价为鞅）
    phi_jump = np.exp(lam * T * (np.exp(iu * muJ - 0.5 * u ** 2 * sigJ ** 2) - 1))
    return np.exp(iu * (-lam * m) * T) * phi_heston * phi_jump


def merton_cf(u):
    """纯 Merton（常数扩散波动率 v0 + 跳）。"""
    iu = 1j * u
    phi_diff = np.exp(iu * (np.log(S0) + (r - q - lam * m) * T)
                      + 0.5 * v0 * T * (iu ** 2 - iu)
                      + lam * T * (np.exp(iu * muJ - 0.5 * u ** 2 * sigJ ** 2) - 1))
    return phi_diff


def bs_call(S, K, T, r, q, vol):
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price, K, S=S0, T=T, r=r, q=q, lo=1e-4, hi=5.0):
    if price <= 0:
        return np.nan
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        p = bs_call(S, K, T, r, q, mid)
        if p > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ============================================================
# Carr-Madan FFT 定价（用特征函数一次性给出整条行权价网格的看涨价）
# ============================================================
def fft_call_prices(cf, alpha=1.5, N=2 ** 15, eta=0.05):
    dk = 2.0 * np.pi / (N * eta)
    b_sym = np.pi / eta
    k = -b_sym + np.arange(N) * dk                  # ln(K)
    v = eta * np.arange(N)
    w = np.zeros(N)
    for j in range(N):
        if j == 0 or j == N - 1:
            w[j] = 1.0
        else:
            w[j] = 4.0 if j % 2 == 1 else 2.0
    w *= eta / 3.0
    psi = np.exp(-r * T) * cf(v - (alpha + 1) * 1j) / (
        (alpha + 1j * v) * (alpha + 1 + 1j * v)
    )
    x = np.exp(-1j * v * b_sym) * psi * w
    y = np.fft.fft(x)
    C = np.real(y) * np.exp(-alpha * k) / np.pi
    K = np.exp(k)
    return K, C


# ============================================================
# 蒙特卡洛模拟三类过程（向量化），用于可视化与校验
# ============================================================
def sim_terminal(n_paths, steps, sv=False, jump=False, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / steps
    lnS = np.full(n_paths, np.log(S0))
    v = np.full(n_paths, v0)
    Z1 = rng.standard_normal((steps, n_paths))
    Z2 = rng.standard_normal((steps, n_paths))
    e = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2
    if jump:
        Nj = rng.poisson(lam * dt, (steps, n_paths))
        Jumps = Nj * muJ + np.sqrt(np.maximum(Nj, 0)) * sigJ * rng.standard_normal((steps, n_paths))
    else:
        Jumps = np.zeros((steps, n_paths))
    for t in range(steps):
        vs = np.maximum(v, 0)
        if sv:
            v = v + kappa * (theta - vs) * dt + sigma_v * np.sqrt(vs) * np.sqrt(dt) * e[t]
            v = np.maximum(v, 0)
        lnS = lnS + (r - q - (lam * m if jump else 0) - 0.5 * vs) * dt \
              + np.sqrt(vs) * np.sqrt(dt) * Z1[t] + Jumps[t]
    return np.exp(lnS)


# ============================================================
# 校验：FFT 价格 vs 蒙特卡洛价格（取若干行权价）
# ============================================================
K_grid, C_grid = fft_call_prices(bates_cf)
mc_paths = sim_terminal(200_000, 63, sv=True, jump=True, seed=7)
print("=== 校验：FFT(特征函数) vs 蒙特卡洛(20万路径) ===")
for KK in [90, 100, 110]:
    idx = np.argmin(np.abs(K_grid - KK))
    fft_p = C_grid[idx]
    mc_p = np.mean(np.maximum(mc_paths - KK, 0.0)) * np.exp(-r * T)
    print(f"K={KK:6.1f}  FFT={fft_p:8.4f}  MC={mc_p:8.4f}  diff={fft_p-mc_p:+.4f}")
print(f"E[S_T]={mc_paths.mean():.4f}  目标 S0*exp((r-q)T)={S0*np.exp((r-q)*T):.4f}")


# ============================================================
# 图 1：三类过程样本路径（1 条代表路径 + 跳点标记）
# ============================================================
def path_with_jumps(sv, jump, seed):
    rng = np.random.default_rng(seed)
    steps = 252
    dt = T / steps
    lnS = np.log(S0); v = v0
    ts = [lnS]; jx = []
    Z1 = rng.standard_normal(steps); Z2 = rng.standard_normal(steps)
    e = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2
    for t in range(steps):
        vs = max(v, 0)
        if sv:
            v = max(v + kappa * (theta - vs) * dt + sigma_v * np.sqrt(vs) * np.sqrt(dt) * e[t], 0)
        j = 0.0
        if jump:
            Nj = rng.poisson(lam * dt)
            if Nj > 0:
                j = Nj * muJ + np.sqrt(Nj) * sigJ * rng.standard_normal()
                jx.append((t + 1, lnS + j))
        lnS = lnS + (r - q - (lam * m if jump else 0) - 0.5 * vs) * dt \
              + np.sqrt(vs) * np.sqrt(dt) * Z1[t] + j
        ts.append(lnS)
    return np.array(ts), jx

fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
p_gbm, _ = path_with_jumps(False, False, 11)
p_mer, j_mer = path_with_jumps(False, True, 22)
p_bat, j_bat = path_with_jumps(True, True, 33)
x = np.arange(len(p_gbm))
ax[0].plot(x, np.exp(p_gbm), color=C["gbm"], lw=1.1); ax[0].set_ylabel("GBM\n(纯 BS)")
ax[1].plot(x, np.exp(p_mer), color=C["merton"], lw=1.1)
for jx, _ in j_mer:
    ax[1].plot(jx, np.exp(p_mer)[jx], "o", color=C["jump"], ms=3)
ax[1].set_ylabel("Merton\n(扩散+跳)")
ax[2].plot(x, np.exp(p_bat), color=C["bates"], lw=1.1)
for jx, _ in j_bat:
    ax[2].plot(jx, np.exp(p_bat)[jx], "o", color=C["jump"], ms=3)
ax[2].set_ylabel("Bates\n(SV+跳)")
for a in ax:
    a.grid(True, color=C["grid"], lw=0.6); a.set_xlim(0, len(p_gbm))
ax[2].set_xlabel("交易日")
fig.suptitle("三类价格过程：只有 Bates 同时有随机波动(曲曲折折)与跳(红点)", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(D, "bates_paths.png"), dpi=130)
plt.close()

# ============================================================
# 图 2：到期价格分布（肥尾对比）
# ============================================================
gbm = sim_terminal(120_000, 63, sv=False, jump=False, seed=1)
mer = sim_terminal(120_000, 63, sv=False, jump=True, seed=2)
bat = sim_terminal(120_000, 63, sv=True, jump=True, seed=3)
fig, ax = plt.subplots(figsize=(10, 5.2))
bins = np.linspace(50, 170, 80)
ax.hist(gbm, bins=bins, density=True, alpha=0.5, color=C["gbm"], label="GBM（对数正态，无跳）")
ax.hist(mer, bins=bins, density=True, alpha=0.5, color=C["merton"], label="Merton（扩散+跳）")
ax.hist(bat, bins=bins, density=True, alpha=0.55, color=C["bates"], label="Bates（SV+跳）")
ax.axvline(S0, color="black", ls=":", lw=1)
ax.set_xlabel("到期价格 S_T")
ax.set_ylabel("概率密度")
ax.set_title("到期价格分布：跳把左尾(暴跌)显著加肥，SV 再额外加宽")
ax.legend()
ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "bates_terminal_hist.png"), dpi=130)
plt.close()

# ============================================================
# 图 3：隐含波动率微笑/偏斜（FFT 价格 → BS 反解 IV）
# ============================================================
KK = np.linspace(70, 130, 61)
idxs = [np.argmin(np.abs(K_grid - k)) for k in KK]
iv_bates = np.array([implied_vol(C_grid[i], KK[j]) for j, i in enumerate(idxs)])
Km, Cm = fft_call_prices(merton_cf)
iv_merton = np.array([implied_vol(Cm[np.argmin(np.abs(Km - k))], k) for k in KK])
iv_bs = np.full_like(KK, np.sqrt(v0) * 100)

fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(KK, iv_bs, "--", color="gray", lw=1.5, label="BS（常数波动率 20%）")
ax.plot(KK, iv_merton * 100, color=C["merton"], lw=1.6, label="Merton（对称微笑）")
ax.plot(KK, iv_bates * 100, color=C["bates"], lw=1.8, label="Bates（左偏斜 smirk）")
ax.axvline(S0, color="black", ls=":", lw=1)
ax.set_xlabel("行权价 K")
ax.set_ylabel("隐含波动率 (%)")
ax.set_title("隐含波动率形态：跳产生偏斜，SV 与跳共同塑造真实的左侧肥尾")
ax.legend(); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "bates_iv_smile.png"), dpi=130)
plt.close()

# ============================================================
# 图 4：看涨期权价格-行权价曲线
# ============================================================
p_bs = np.array([bs_call(S0, k, T, r, q, np.sqrt(v0)) for k in KK])
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(KK, p_bs, "--", color="gray", lw=1.5, label="BS（v=20%）")
ax.plot(KK, [Cm[np.argmin(np.abs(Km - k))] for k in KK], color=C["merton"], lw=1.6, label="Merton")
ax.plot(KK, [C_grid[np.argmin(np.abs(K_grid - k))] for k in KK], color=C["bates"], lw=1.8, label="Bates")
ax.axvline(S0, color="black", ls=":", lw=1)
ax.set_xlabel("行权价 K"); ax.set_ylabel("看涨期权价格")
ax.set_title("看涨期权价格曲线：深度虚值看跌保护(低 K)被跳显著抬价")
ax.legend(); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "bates_price_compare.png"), dpi=130)
plt.close()

# ============================================================
# 输出关键数字（供文章引用）
# ============================================================
print("\n=== 关键数字 ===")
print(f"ATM 看涨(BS, v=20%) = {bs_call(S0,S0,T,r,q,np.sqrt(v0)):.4f}")
print(f"ATM 看涨(Bates)      = {C_grid[np.argmin(np.abs(K_grid-S0))]:.4f}")
iv_90 = implied_vol(C_grid[np.argmin(np.abs(K_grid-90))], 90)
iv_100 = implied_vol(C_grid[np.argmin(np.abs(K_grid-100))], 100)
iv_110 = implied_vol(C_grid[np.argmin(np.abs(K_grid-110))], 110)
print(f"IV@K=90  = {iv_90*100:.2f}%   IV@K=100 = {iv_100*100:.2f}%   IV@K=110 = {iv_110*100:.2f}%")
print(f"左侧偏斜(IV90-IV110) = {(iv_90-iv_110)*100:.2f} vol 点")
print("图片已保存到:", D)
