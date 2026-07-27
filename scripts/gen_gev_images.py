#!/usr/bin/env python3
"""区组极大值 GEV 配图: 4 张"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
OUT = Path(__file__).resolve().parent.parent / "public/images/block-maxima-gev-risk"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)

# ---------- DGP: GARCH(1,1)-t(5) 日损失序列, 40 年 ----------
YEARS = 40
DAYS_M = 21
T = YEARS * 12 * DAYS_M          # 月块 = 21 天
BURN = 500
om, al, be = 0.02, 0.09, 0.89
NU = 5.0
N = T + BURN
eps = stats.t.rvs(NU, size=N, random_state=rng) / np.sqrt(NU / (NU - 2))
r = np.zeros(N); s2 = np.zeros(N); s2[0] = om / (1 - al - be)
for t in range(1, N):
    s2[t] = om + al * r[t-1]**2 + be * s2[t-1]
    r[t] = np.sqrt(s2[t]) * eps[t]
loss = -r[BURN:]                  # 日损失(%)

# ---------- 区组极大值 ----------
n_blocks = T // DAYS_M
blocks = loss[:n_blocks * DAYS_M].reshape(n_blocks, DAYS_M)
M = blocks.max(axis=1)            # 月最大日损失

# GEV 拟合 (scipy: genextreme, c = -xi)
c_hat, loc_hat, sc_hat = stats.genextreme.fit(M)
xi_hat = -c_hat

# Gumbel (xi=0) 拟合作对照
locg, scg = stats.gumbel_r.fit(M)

# 对照: 正态假设下月最大值的分布（错误模型）
mu_n, sd_n = loss.mean(), loss.std()

# ---------- 图1: 日损失 vs 月块最大值 ----------
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1.4]})
show_days = 5 * 252
axes[0].plot(loss[:show_days], lw=0.5, color="0.6")
# 标出该窗口的月最大值
mm = loss[:show_days].reshape(-1, DAYS_M) if show_days % DAYS_M == 0 else None
nb_show = show_days // DAYS_M
for b in range(nb_show):
    seg = loss[b*DAYS_M:(b+1)*DAYS_M]
    i = np.argmax(seg)
    axes[0].plot(b*DAYS_M + i, seg[i], "v", ms=4, color="#c1121f")
axes[0].set_title("日损失序列（前 5 年）与每个月块的最大值（红三角）", fontsize=11)
axes[0].set_ylabel("日损失 (%)")
axes[1].plot(M, lw=0.7, color="#345995")
axes[1].set_title(f"月块最大值序列（{n_blocks} 个月，均值 {M.mean():.2f}%，最大 {M.max():.2f}%）", fontsize=11)
axes[1].set_xlabel("月块序号"); axes[1].set_ylabel("块最大 (%)")
for ax in axes: ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT / "gev-block-maxima.jpg", dpi=110, bbox_inches="tight"); plt.close(fig)

# ---------- 图2: 三个分布对块最大值的拟合 (直方图+PDF, 右尾QQ) ----------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
xs = np.linspace(0, M.max() * 1.15, 500)
axes[0].hist(M, bins=45, density=True, color="0.85", edgecolor="0.6", label="月最大值")
axes[0].plot(xs, stats.genextreme.pdf(xs, c_hat, loc_hat, sc_hat), lw=1.8, color="#c1121f",
             label=f"GEV (ξ={xi_hat:.2f})")
axes[0].plot(xs, stats.gumbel_r.pdf(xs, locg, scg), lw=1.6, color="#345995", ls="--",
             label="Gumbel (ξ=0)")
# 正态模型隐含的月最大值分布: F(x)^21
pdf_norm_max = DAYS_M * stats.norm.pdf(xs, mu_n, sd_n) * stats.norm.cdf(xs, mu_n, sd_n)**(DAYS_M - 1)
axes[0].plot(xs, pdf_norm_max, lw=1.6, color="#8d99ae", ls=":", label="正态日损失隐含")
axes[0].set_xlabel("月最大日损失 (%)"); axes[0].set_ylabel("密度")
axes[0].set_title("块最大值分布拟合", fontsize=11); axes[0].legend(fontsize=9)
# QQ
q_emp = np.sort(M)
pp = (np.arange(1, n_blocks + 1) - 0.5) / n_blocks
axes[1].plot(stats.genextreme.ppf(pp, c_hat, loc_hat, sc_hat), q_emp, ".", ms=3.5,
             color="#c1121f", label="GEV")
axes[1].plot(stats.gumbel_r.ppf(pp, locg, scg), q_emp, ".", ms=3.5, color="#345995", label="Gumbel")
lim = [0, q_emp.max() * 1.05]
axes[1].plot(lim, lim, "k--", lw=0.8)
axes[1].set_xlim(lim); axes[1].set_ylim(lim)
axes[1].set_xlabel("模型分位数 (%)"); axes[1].set_ylabel("经验分位数 (%)")
axes[1].set_title("QQ 图：Gumbel 在深尾整体低估", fontsize=11); axes[1].legend(fontsize=9)
for ax in axes: ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT / "gev-fit-qq.jpg", dpi=110, bbox_inches="tight"); plt.close(fig)

# ---------- 图3: 回报水平曲线 + Profile 不确定性(参数化bootstrap) ----------
def ret_level(c, loc, sc, m):
    return stats.genextreme.ppf(1 - 1.0 / m, c, loc, sc)
ms = np.array([2, 5, 10, 20, 60, 120, 240, 600, 1200])  # 以月为单位
rl_gev = ret_level(c_hat, loc_hat, sc_hat, ms)
rl_gum = stats.gumbel_r.ppf(1 - 1.0 / ms, locg, scg)
# bootstrap
B = 400
boot = np.zeros((B, len(ms)))
for b in range(B):
    samp = stats.genextreme.rvs(c_hat, loc_hat, sc_hat, size=n_blocks, random_state=rng)
    cb, lb, sb = stats.genextreme.fit(samp)
    boot[b] = ret_level(cb, lb, sb, ms)
lo, hi = np.percentile(boot, [5, 95], axis=0)
# 经验对照
emp_ms = np.array([2, 5, 10, 20, 60, 120, 240])
emp_rl = np.quantile(M, 1 - 1.0 / emp_ms)
fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.fill_between(ms, lo, hi, color="#c1121f", alpha=0.15, label="GEV 90% bootstrap 区间")
ax.plot(ms, rl_gev, "o-", ms=4, color="#c1121f", label=f"GEV (ξ={xi_hat:.2f})")
ax.plot(ms, rl_gum, "s--", ms=4, color="#345995", label="Gumbel (ξ=0)")
ax.plot(emp_ms, emp_rl, "^", ms=7, color="k", label="经验分位数（样本内）")
ax.axvline(n_blocks, color="0.5", ls=":", lw=1)
ax.text(n_blocks * 1.05, rl_gev[0], f"样本长度\n{n_blocks} 个月", fontsize=8, color="0.4")
ax.set_xscale("log")
ax.set_xlabel("回报周期 m（月）"); ax.set_ylabel("m-月回报水平（日损失 %）")
ax.set_title("回报水平曲线：100 年一遇（1200 月）的最坏单日损失", fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.25, which="both")
fig.tight_layout(); fig.savefig(OUT / "gev-return-levels.jpg", dpi=110, bbox_inches="tight"); plt.close(fig)

# ---------- 图4: 块长权衡 (bias-variance) ----------
block_lens = [5, 10, 21, 42, 63, 126]
xi_by_len, se_by_len, rl20y = [], [], []
for bl in block_lens:
    nb = T // bl
    Mb = loss[:nb * bl].reshape(nb, bl).max(axis=1)
    cb, lb, sb = stats.genextreme.fit(Mb)
    xi_by_len.append(-cb)
    # 20年一遇日损失 (回报周期换算到块数: 20年 = 20*252/bl 块)
    m20 = 20 * 252 / bl
    rl20y.append(stats.genextreme.ppf(1 - 1 / m20, cb, lb, sb))
    # bootstrap se of xi
    bs = []
    for b in range(150):
        samp = stats.genextreme.rvs(cb, lb, sb, size=nb, random_state=rng)
        bs.append(-stats.genextreme.fit(samp)[0])
    se_by_len.append(np.std(bs))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
axes[0].errorbar(block_lens, xi_by_len, yerr=np.array(se_by_len) * 1.645, fmt="o-",
                 color="#345995", capsize=4, ms=5)
axes[0].axhline(0, color="0.5", ls=":")
axes[0].set_xlabel("块长度（天）"); axes[0].set_ylabel("形状参数 ξ 估计 (±90% CI)")
axes[0].set_title("块太短→GEV 近似差（偏差）；块太长→块数少（方差）", fontsize=10)
axes[1].plot(block_lens, rl20y, "s-", color="#c1121f", ms=5)
axes[1].set_xlabel("块长度（天）"); axes[1].set_ylabel("20 年一遇日损失 (%)")
axes[1].set_title("同一目标（20 年回报水平）随块长的漂移", fontsize=10)
for ax in axes: ax.grid(alpha=0.25)
fig.suptitle(f"块长权衡（总样本固定 {T} 天）", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "gev-block-length-tradeoff.jpg", dpi=110, bbox_inches="tight"); plt.close(fig)

# 正文数字
ks = stats.kstest(M, lambda x: stats.genextreme.cdf(x, c_hat, loc_hat, sc_hat))
ks_g = stats.kstest(M, lambda x: stats.gumbel_r.cdf(x, locg, scg))
print(f"n_blocks={n_blocks}, xi={xi_hat:.3f}, loc={loc_hat:.3f}, scale={sc_hat:.3f}")
print(f"gumbel loc={locg:.3f} scale={scg:.3f}")
print(f"KS GEV p={ks.pvalue:.3f}, KS Gumbel p={ks_g.pvalue:.4f}")
print("return levels (2,5,10,20,60,120,240,600,1200 mo):", np.round(rl_gev, 2))
print("gumbel same:", np.round(rl_gum, 2))
print("empirical:", np.round(emp_rl, 2), "at", emp_ms)
print("CI at 1200mo:", lo[-1].round(2), hi[-1].round(2))
print("xi by block len:", dict(zip(block_lens, np.round(xi_by_len, 3))))
print("rl20y by block len:", dict(zip(block_lens, np.round(rl20y, 2))))
print("M stats: mean", M.mean().round(2), "max", M.max().round(2))
print("theory xi = 1/nu =", 1 / NU)
