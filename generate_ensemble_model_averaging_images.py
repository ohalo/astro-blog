#!/usr/bin/env python3
"""
为文章「集成模型平均：把多个弱预测融成稳健 Alpha」(ensemble-model-averaging)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（自洽合成，仅用于演示方法；落地见文末路径）：
  * 设真实截面预期收益 μ*（每只股票一个「真 alpha」，已对截面去均值），日收益
        r_{t,i} = μ*_i + 特质噪声
    用横截面多空（dollar-neutral）组合 w ∝ (S − 均值) 来隔离 alpha、剔除市场暴露。
  * 有 M 个「弱模型」给出带噪信号 s_k = μ* + η_k，η_k 之间相关系数 ρ（共享误差占比）。
    集成信号 S_M = (1/M) Σ s_k，用它选股/加权。
  * 核心结论（有解析保证）：集成信号噪声方差
        Var(S_M - μ*) = (σ_η²/M)·(1 + (M-1)·ρ)
    所以只有模型误差「相互独立」(ρ 小) 时加模型才持续降噪；若共享同一份误差(ρ=1)，
    平均也无济于事——这是「有效分散化」的量化表达。

注意：本模拟嵌入「真 alpha + 共享/独立噪声」以演示机制（与全库高阶文一致），真实
因子模型中模型误差相关性来自共同数据来源与因子暴露，文末已说明。
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
D = os.path.join(BASE, "ensemble-model-averaging")
os.makedirs(D, exist_ok=True)

C = {"eq": "#4C72B0", "bd": "#55A868", "gd": "#DD8452", "ens": "#C44E52",
     "single": "#999999", "oracle": "#8172B3", "grid": "#DDDDDD", "thr": "#888888",
     "accent": "#8172B3", "warn": "#C44E52", "calm": "#55A868"}

rng = np.random.default_rng(20260713)
N = 10
T = 252
SIG_MU = 0.004          # 真 alpha 截面波动（日频，已去均值）
SIG_R = 0.020           # 特质噪声
SIG_ETA = 0.030         # 单模型信号噪声
RHO = 0.30              # 模型误差相关


def make_models(M, rho, seed=20260713):
    r = np.random.default_rng(seed)
    mu = r.normal(0, SIG_MU, N)
    mu = mu - mu.mean()                       # 截面去均值
    R = mu[None, :] + r.normal(0, SIG_R, (T, N))
    zc = r.normal(0, SIG_ETA, (T, N))
    signals = []
    for k in range(M):
        zk = r.normal(0, SIG_ETA, (T, N))
        eta = np.sqrt(rho) * zc + np.sqrt(1 - rho) * zk
        signals.append(mu[None, :] + eta)
    S = np.stack(signals, 0).mean(0)
    return mu, R, S, signals


def ls_sharpe(S, R):
    w = S - S.mean(1, keepdims=True)
    w = w / np.abs(w).sum(1, keepdims=True)
    ret = (w * R).sum(1)
    ann = ret.mean() * 252
    vol = ret.std(ddof=1) * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else 0.0
    return sharpe, ann, vol, ret, w


def ic(sig, target):
    if target.ndim == 1:
        target = target[None, :].repeat(sig.shape[0], 0)

    def rank(x):
        return x.argsort().argsort().astype(float)

    out = []
    for t in range(sig.shape[0]):
        rs = rank(sig[t]) - rank(sig[t]).mean()
        rt = rank(target[t]) - rank(target[t]).mean()
        out.append((rs * rt).sum() / np.sqrt((rs ** 2).sum() * (rt ** 2).sum()))
    return float(np.mean(out))


def theoretical_ic(M, rho, R=SIG_ETA ** 2 / SIG_MU ** 2):
    return 1.0 / np.sqrt(1.0 + R * (1.0 + (M - 1) * rho) / M)


# ----------------------------------------------------------------------------
# 图 1：集成 IC 随模型数 M 的理论曲线（不同 ρ）
# ----------------------------------------------------------------------------
M_grid = [1, 2, 3, 5, 8, 12, 20, 30, 50]
fig, ax = plt.subplots(figsize=(10, 5.2))
R = SIG_ETA ** 2 / SIG_MU ** 2
for rho, col in [(0.0, C["bd"]), (0.3, C["eq"]), (0.6, C["gd"]), (1.0, C["warn"])]:
    ic_th = [theoretical_ic(m, rho, R) for m in M_grid]
    ax.plot(M_grid, ic_th, "o-", color=col, lw=2.0, label=f"ρ={rho:.1f}（模型误差相关）")
ax.set_xlabel("模型数 M"); ax.set_ylabel("集成信号 IC（与真 alpha 的秩相关）")
ax.set_title("集成 IC 随 M 变化：只有模型误差『相互独立』(ρ 小) 才持续降噪")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "ensemble_ic.png"), dpi=130)
plt.close()

# ----------------------------------------------------------------------------
# 主数值（canonical seed, ρ=0.3）
# ----------------------------------------------------------------------------
mu, R, S, signals = make_models(40, RHO, seed=20260713)
sp_ens, ann_ens, vol_ens, ret_ens, w_ens = ls_sharpe(S, R)
sp_oracle, ann_or, vol_or, ret_or, w_or = ls_sharpe(mu[None, :].repeat(T, 0), R)
# 随机信号基准（IC≈0）：多种子均值以稳定小样本噪声
_rnd_sharpe = []
for rsd in range(1, 6):
    rnd = np.random.default_rng(rsd).normal(0, 1, (T, N))
    _rnd_sharpe.append(ls_sharpe(rnd, R)[0])
sp_rnd = float(np.mean(_rnd_sharpe))
# 单模型平均
res_single = [ls_sharpe(signals[k], R) for k in range(40)]
sp_single_mean = float(np.mean([x[0] for x in res_single]))
ann_single_mean = float(np.mean([x[1] for x in res_single]))
vol_single_mean = float(np.mean([x[2] for x in res_single]))

# 不同 M 的 Sharpe 曲线
M_curve = [1, 2, 3, 5, 8, 12, 20, 30, 40]
sp_vs_M = []
ann_vs_M = []
for m in M_curve:
    Sm = np.stack(signals[:m], 0).mean(0)
    s, a, _, _, _ = ls_sharpe(Sm, R)
    sp_vs_M.append(s); ann_vs_M.append(a)

ic_single = ic(signals[0], mu)
ic_ens = ic(S, mu)

# ----------------------------------------------------------------------------
# 图 2：组合 Sharpe 随 M 收敛到 oracle
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(M_curve, sp_vs_M, "o-", color=C["ens"], lw=2.2,
        label=f"集成(M 个弱模型) Sharpe={sp_ens:.2f} @M=40")
ax.axhline(sp_oracle, color=C["oracle"], ls="--", lw=2.0,
           label=f"Oracle（用真 alpha，上界）Sharpe={sp_oracle:.2f}")
ax.axhline(sp_single_mean, color=C["single"], ls=":", lw=2.0,
           label=f"单模型平均 Sharpe={sp_single_mean:.2f}")
ax.axhline(sp_rnd, color=C["calm"], ls="-.", lw=1.6,
           label=f"纯噪声信号基准 Sharpe≈{sp_rnd:.2f}（IC≈0）")
ax.set_xlabel("模型数 M"); ax.set_ylabel("横截面多空组合 年化 Sharpe")
ax.set_title("集成模型平均：Sharpe 随 M 上升，收敛向 Oracle，显著跑赢单模型")
ax.legend(fontsize=8, loc="upper left"); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "ensemble_sharpe.png"), dpi=130)
plt.close()

# ----------------------------------------------------------------------------
# 图 3：组合日收益分布（单模型 vs 集成 vs oracle）
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.hist(ret_ens * 100, bins=40, color=C["ens"], alpha=0.55, density=True,
        label=f"集成(M=40) 日收益 μ={ret_ens.mean()*100:.3f}% σ={ret_ens.std()*100:.3f}%")
ax.hist(ret_or * 100, bins=40, color=C["oracle"], alpha=0.45, density=True,
        label=f"Oracle 日收益 μ={ret_or.mean()*100:.3f}% σ={ret_or.std()*100:.3f}%")
ret_s0 = res_single[0][3]
ax.hist(ret_s0 * 100, bins=40, color=C["single"], alpha=0.40, density=True,
        label=f"单模型 日收益 μ={ret_s0.mean()*100:.3f}% σ={ret_s0.std()*100:.3f}%")
ax.set_xlabel("日收益 (%)"); ax.set_ylabel("密度")
ax.set_title("横截面多空日收益分布：集成把噪声『削平』，均值更接近 Oracle")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "ensemble_dist.png"), dpi=130)
plt.close()

# ----------------------------------------------------------------------------
# 图 4：集成权重与 Oracle 权重的相关性随 M 收敛
# ----------------------------------------------------------------------------
corr_vs_M = []
for m in M_curve:
    Sm = np.stack(signals[:m], 0).mean(0)
    wm = (Sm - Sm.mean(1, keepdims=True)); wm = wm / np.abs(wm).sum(1, keepdims=True)
    c = np.corrcoef(wm.ravel(), w_or.ravel())[0, 1]
    corr_vs_M.append(c)
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(M_curve, corr_vs_M, "s-", color=C["accent"], lw=2.2,
        label="集成权重 与 Oracle 权重的相关性")
ax.axhline(1.0, color=C["oracle"], ls="--", lw=1.5)
ax.set_xlabel("模型数 M"); ax.set_ylabel("与 Oracle 权重的相关系数")
ax.set_title(f"集成权重随 M 收敛到 Oracle：M=40 时相关 {corr_vs_M[-1]:.2f}")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "ensemble_weights.png"), dpi=130)
plt.close()

# ----------------------------------------------------------------------------
# 鲁棒性：10 个种子
# ----------------------------------------------------------------------------
sp_ens_seeds, sp_single_seeds, sp_oracle_seeds, sp_rnd_seeds = [], [], [], []
for sd in range(10):
    mu2, R2, S2, sig2 = make_models(40, RHO, seed=1000 + sd)
    e, _, _, _, _ = ls_sharpe(S2, R2)
    o, _, _, _, _ = ls_sharpe(mu2[None, :].repeat(T, 0), R2)
    ss = np.mean([ls_sharpe(sig2[k], R2)[0] for k in range(40)])
    rr, _, _, _, _ = ls_sharpe(np.random.default_rng(sd).normal(0, 1, (T, N)), R2)
    sp_ens_seeds.append(e); sp_oracle_seeds.append(o); sp_single_seeds.append(ss); sp_rnd_seeds.append(rr)
sp_ens_mean = float(np.mean(sp_ens_seeds)); sp_ens_std = float(np.std(sp_ens_seeds))
sp_single_mean_r = float(np.mean(sp_single_seeds))
sp_oracle_mean = float(np.mean(sp_oracle_seeds))
sp_rnd_mean = float(np.mean(sp_rnd_seeds))
beats = int(np.sum(np.array(sp_ens_seeds) > np.array(sp_single_seeds)))
beats_rnd = int(np.sum(np.array(sp_ens_seeds) > np.array(sp_rnd_seeds)))

print("===== ENSEMBLE KEY NUMBERS =====")
print(f"N={N} T={T} SIG_MU={SIG_MU} SIG_R={SIG_R} SIG_ETA={SIG_ETA} RHO={RHO}")
print(f"IC: single={ic_single:.3f}  ensemble(M=40)={ic_ens:.3f}")
print(f"Sharpe(canonical): random={sp_rnd:.2f}  single_mean={sp_single_mean:.2f}  ensemble(M=40)={sp_ens:.2f}  oracle={sp_oracle:.2f}")
print(f"annRet: single={ann_single_mean:.3f}  ensemble={ann_ens:.3f}  oracle={ann_or:.3f}  vol~{vol_ens:.3f}")
print(f"ens/single Sharpe = {sp_ens/sp_single_mean:.2f}x")
print(f"robustness(10 seeds): ens={sp_ens_mean:.2f}±{sp_ens_std:.2f}  single={sp_single_mean_r:.2f}  oracle={sp_oracle_mean:.2f}  random={sp_rnd_mean:.2f}")
print(f"ensemble>single on {beats}/10 ; ensemble>random on {beats_rnd}/10")
print(f"weight corr with oracle @M=40 = {corr_vs_M[-1]:.2f}")
print("IMAGES WRITTEN:", sorted(os.listdir(D)))
