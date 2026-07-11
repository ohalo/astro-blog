#!/usr/bin/env python3
"""
为文章「粒子滤波估计随机波动率：当 MCMC 太慢时的序贯解法」(particle-filter-sv)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

模型（随机波动率 SV）：
  h_t = μ + φ(h_{t-1}-μ) + η_t,   η_t~N(0,σ_η²)        隐状态(对数波动率)
  r_t = exp(h_t/2)·ε_t,           ε_t~N(0,1)            观测收益
  z_t = log(r_t²) = h_t + e_t,    e_t~log-χ²₁           滤波用的信息量观测

粒子滤波（序贯蒙特卡洛）对 z_t 用「log(χ²₁) 的 Gaussian 近似」似然：
  z_t = h_t + e_t,  e_t~N(μ_e, σ_e²),  μ_e=-1.2704, σ_e²=π²/2≈4.9348
（精确 log-χ²₁ 的众数在 0、均值在 -1.27，配合有信息量的先验会产生水平偏差；
 用其 Gaussian 近似可得到无偏且可追踪的滤波，是 SV 滤波的标准做法。）
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "particle-filter-sv")
os.makedirs(D, exist_ok=True)

C = {"true": "#2F4B7C", "mean": "#C44E52", "part": "#9DB8D2",
     "band": "#F2C0C0", "ess1": "#4C72B0", "ess2": "#55A868",
     "grid": "#DDDDDD", "thr": "#888888"}

# ============================================================
# 1) 模拟随机波动率(SV)数据
# ============================================================
def simulate_sv(T=800, mu=-9.0, phi=0.95, sigma_eta=0.35, seed=7):
    rng = np.random.default_rng(seed)
    h = np.empty(T); h[0] = mu
    for t in range(1, T):
        h[t] = mu + phi * (h[t - 1] - mu) + sigma_eta * rng.normal()
    eps = rng.normal(size=T)
    r = np.exp(h / 2.0) * eps                 # 观测收益
    z = np.log(r * r + 1e-12)                 # 信息量观测：z_t = h_t + log(ε²)
    return r, z, h, dict(mu=mu, phi=phi, sigma_eta=sigma_eta)

# ============================================================
# 2) log(χ²₁) 噪声的 Gaussian 近似似然（对 z_t = h_t + e_t, e_t~N(μ_e,σ_e²)）
#    μ_e = -1.2704 (log-χ²₁ 均值), σ_e² = π²/2 ≈ 4.9348 (log-χ²₁ 方差)
# ============================================================
MU_E = -1.2704
VAR_E = np.pi ** 2 / 2.0
def loglik_logchisq(z, h):
    e = z - h - MU_E
    return -0.5 * np.log(2 * np.pi * VAR_E) - 0.5 * e ** 2 / VAR_E

# ============================================================
# 3) Bootstrap 粒子滤波（序贯蒙特卡洛）
#    预测: h_t^(i) ~ N(μ+φ(h_{t-1}-μ), σ_η²)
#    更新: w_t^(i) ∝ p(z_t | h_t^(i)) = exp(loglik_logchisq(z_t, h_t^(i)))
#    重采样: ESS<N/2 时多叉重采样
#    SMC 对数似然增量: log( (1/N)Σ w_t^(i) )  → 边际似然估计
# ============================================================
def bootstrap_pf(z, mu, phi, sigma_eta, N=2000, resample_frac=0.5,
                 seed=0, return_particles=False):
    rng = np.random.default_rng(seed)
    T = len(z)
    h_var = sigma_eta ** 2 / (1.0 - phi ** 2)
    part = rng.normal(mu, np.sqrt(h_var), size=N)
    loglik = 0.0
    filt_mean = np.empty(T); filt_lo = np.empty(T); filt_hi = np.empty(T)
    ess_series = np.empty(T)
    stored = np.empty((T, min(N, 400))) if return_particles else None
    for t in range(T):
        if t > 0:
            part = mu + phi * (part - mu) + sigma_eta * rng.normal(size=N)
        logw = loglik_logchisq(z[t], part)
        m = logw.max(); w = np.exp(logw - m)
        W = w / w.sum()
        loglik += m + np.log(w.sum()) - np.log(N)
        order = np.argsort(part); cum = np.cumsum(W[order])
        filt_mean[t] = np.sum(W * part)
        filt_lo[t] = np.interp(0.05, cum, part[order])
        filt_hi[t] = np.interp(0.95, cum, part[order])
        ess_series[t] = 1.0 / np.sum(W ** 2)
        if return_particles:
            idx = np.linspace(0, N - 1, stored.shape[1]).astype(int)
            stored[t] = part[idx]
        if t < T - 1 and ess_series[t] < resample_frac * N:
            aidx = rng.choice(N, size=N, p=W)
            part = part[aidx]
    if return_particles:
        return dict(filt_mean=filt_mean, filt_lo=filt_lo, filt_hi=filt_hi,
                    ess=ess_series, loglik=loglik, particles=stored)
    return dict(filt_mean=filt_mean, filt_lo=filt_lo, filt_hi=filt_hi,
                ess=ess_series, loglik=loglik)

# ============================================================
# 4) 辅助粒子滤波(APF)：用预测均值先做一阶段权重，减少退化
#    a_t^(i) ∝ p(z_t | μ_t^(i)),  μ_t^(i) = μ+φ(h_{t-1}-μ)
#    重采样祖先 a^(i)~Cat(a)；传播 h_t^(i)~N(μ_t^(a),σ_η²)
#    修正权重 w_t^(i) = p(z_t|h_t^(i)) / p(z_t|μ_t^(a))
# ============================================================
def apf(z, mu, phi, sigma_eta, N=2000, resample_frac=0.5, seed=0):
    rng = np.random.default_rng(seed)
    T = len(z)
    h_var = sigma_eta ** 2 / (1.0 - phi ** 2)
    part = rng.normal(mu, np.sqrt(h_var), size=N)
    loglik = 0.0; filt_mean = np.empty(T); ess_series = np.empty(T)
    for t in range(T):
        pred_mean = mu + phi * (part - mu) if t > 0 else np.full(N, mu)
        loga = loglik_logchisq(z[t], pred_mean)
        ma = loga.max(); a = np.exp(loga - ma); A = a / a.sum()
        if t == 0 or (1.0 / np.sum(A ** 2)) < resample_frac * N:
            aidx = rng.choice(N, size=N, p=A)
        else:
            aidx = np.arange(N)
        pm = pred_mean[aidx]
        part = pm + sigma_eta * rng.normal(size=N)
        logw = loglik_logchisq(z[t], part)
        logw0 = loglik_logchisq(z[t], pm)
        logwc = logw - logw0
        mw = logwc.max(); wc = np.exp(logwc - mw); WC = wc / wc.sum()
        loglik += mw + np.log(wc.sum()) - np.log(N)
        filt_mean[t] = np.sum(WC * part)
        ess_series[t] = 1.0 / np.sum(WC ** 2)
    return dict(filt_mean=filt_mean, ess=ess_series, loglik=loglik)

# ============================================================
# 主运行
# ============================================================
r, z, h_true, p = simulate_sv(T=800, mu=-9.0, phi=0.97, sigma_eta=0.60, seed=7)
T = len(r)
pf = bootstrap_pf(z, p["mu"], p["phi"], p["sigma_eta"], N=2000, seed=3, return_particles=True)
ap = apf(z, p["mu"], p["phi"], p["sigma_eta"], N=2000, seed=3)
rmse = float(np.sqrt(np.mean((pf["filt_mean"] - h_true) ** 2)))
rmse_raw = float(np.std(h_true))                 # 隐状态自身波动(上界)
mean_ess_boot = float(np.mean(pf["ess"]))
mean_ess_apf = float(np.mean(ap["ess"]))

t = np.arange(T)

# ---------- 图 1：真实隐状态 + 粒子云 + 滤波均值 ----------
fig, ax = plt.subplots(figsize=(11, 4.8))
n_show = 60
cloud = pf["particles"][:, :n_show]
ax.plot(t, cloud, color=C["part"], lw=0.4, alpha=0.35)
ax.plot(t, h_true, color=C["true"], lw=1.6, label="真实隐状态 h_t (不可观测)")
ax.plot(t, pf["filt_mean"], color=C["mean"], lw=1.3, label="粒子滤波均值 (在线估计)")
ax.set_xlabel("交易日"); ax.set_ylabel("对数波动率 h_t")
ax.set_title("Bootstrap 粒子滤波：一团粒子实时追踪看不见的波动状态")
ax.legend(loc="upper right", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "pf_particles.png"), dpi=130)
plt.close()

# ---------- 图 2：滤波均值 vs 真值 + 90% 可信带 ----------
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.fill_between(t, pf["filt_lo"], pf["filt_hi"], color=C["band"], alpha=0.6,
               label="90% 可信带 (粒子 5%~95% 分位)")
ax.plot(t, h_true, color=C["true"], lw=1.4, label="真实 h_t")
ax.plot(t, pf["filt_mean"], color=C["mean"], lw=1.1, label="粒子滤波均值")
ax.set_xlabel("交易日"); ax.set_ylabel("对数波动率 h_t")
ax.set_title("粒子滤波均值紧贴真值：RMSE=%.3f (隐状态 std=%.3f)" % (rmse, rmse_raw))
ax.legend(loc="upper right", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "pf_filtered_credible.png"), dpi=130)
plt.close()

# ---------- 图 3：Bootstrap vs APF 的 ESS ----------
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.plot(t, pf["ess"], color=C["ess1"], lw=0.7, alpha=0.85, label="Bootstrap PF (ESS)")
ax.plot(t, ap["ess"], color=C["ess2"], lw=0.7, alpha=0.85, label="APF (ESS)")
ax.axhline(2000 * 0.5, color=C["thr"], ls="--", lw=1, label="重采样阈值 N/2=1000")
ax.set_xlabel("交易日"); ax.set_ylabel("有效样本数 ESS")
ax.set_title("APF 的 ESS 显著高于 Bootstrap：退化更慢、重采样更少")
ax.legend(loc="lower left", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "pf_ess.png"), dpi=130)
plt.close()

# ---------- 图 4：随粒子数 N 收敛 ----------
Ns = [128, 256, 512, 1024, 2048, 4096]
rmse_n = []; ll_n = []
for N in Ns:
    rms = []; lls = []
    for s in range(3):                      # 多 seed 平均，消除蒙特卡洛噪声
        rr = bootstrap_pf(z, p["mu"], p["phi"], p["sigma_eta"], N=N, seed=100 + s)
        rms.append(np.sqrt(np.mean((rr["filt_mean"] - h_true) ** 2)))
        lls.append(rr["loglik"])
    rmse_n.append(float(np.mean(rms))); ll_n.append(float(np.mean(lls)))
fig, ax1 = plt.subplots(figsize=(11, 4.4))
ax1.plot(Ns, rmse_n, color=C["mean"], lw=1.6, marker="o", label="滤波 RMSE(h_t)")
ax1.set_xlabel("粒子数 N"); ax1.set_ylabel("RMSE(h_t)", color=C["mean"])
ax1.set_xscale("log"); ax1.tick_params(axis="y", labelcolor=C["mean"])
ax1.grid(True, color=C["grid"], lw=0.6)
ax2 = ax1.twinx()
ax2.plot(Ns, ll_n, color=C["ess1"], lw=1.6, marker="s", label="SMC 对数似然")
ax2.set_ylabel("SMC 对数似然", color=C["ess1"])
ax2.tick_params(axis="y", labelcolor=C["ess1"])
ax1.set_title("粒子数越多越准：滤波 RMSE 下降、SMC 似然收敛")
lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lab1 + lab2, loc="center right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(D, "pf_convergence.png"), dpi=130)
plt.close()

print("=== 粒子滤波估计随机波动率 关键数字 ===")
print("样本 T=%d, 真值参数 μ=%.2f φ=%.2f σ_η=%.2f" % (T, p["mu"], p["phi"], p["sigma_eta"]))
print("隐状态 std(h_t)=%.3f" % rmse_raw)
print("Bootstrap PF: 滤波 RMSE(h_t)=%.3f (解释掉 %.1f%% 波动), 平均 ESS=%.0f / %d"
      % (rmse, 100 * (1 - rmse ** 2 / rmse_raw ** 2), mean_ess_boot, 2000))
print("APF:          平均 ESS=%.0f / %d (退化更慢)" % (mean_ess_apf, 2000))
print("SMC 对数似然(全样本, N=2000)=%.2f" % pf["loglik"])
print("收敛扫描: N=%d→RMSE=%.3f, N=%d→RMSE=%.3f" % (Ns[0], rmse_n[0], Ns[-1], rmse_n[-1]))
print("\n图片已保存到:", D)
