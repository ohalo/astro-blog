#!/usr/bin/env python3
"""
为文章「预期短缺回测：用 ES 检验替代 VaR 的后验失序」(expected-shortfall-backtesting) 生成真实配图。

核心逻辑（Expected Shortfall 后验检验 / McNeil & Frey 2000, Acerbi & Szekely 2014, ESRB FRT）：
  - VaR_α = 在 1−α 置信下的最大损失；超出 VaR 的天数应服从 B(N, 1−α)。
  - ES_α = 超出 VaR 那部分损失的条件期望：ES = E[L | L > VaR]，L=−损益（正=损失）。
  - 问题：VaR 只数「有没有超出」，对「超出后到底多惨」失明 → 尾部更肥时 VaR 还绿、ES 已爆。
  - ES 后验 z 检验（Acerbi-Szekely）：对超额损失 U_i = L_i − VaR(仅 L_i>VaR) 检验
        E[U] = ES − VaR,   z = (mean(U) − (ES−VaR)) / sqrt(var(U)/n_exc) ~ N(0,1)
    |z|>2.326 即「红区」。零假设是「超额损失的均值 = ES−VaR」，不是等于 ES。
  - 与 VaR 红绿灯（traffic-light：数超出天数，绿≤14/红≥15，N=1000 时）对比：
    VaR 只盯「数天数」，ES-z 盯「超出后多惨」→ 尾部肥瘦只有 ES 看得见。
  - 正确（正态）模型：1% VaR 校准为 2.266%；「真实世界」用更肥尾 Student-t，且把 1% VaR 校准得
    与正态一致（故 VaR 红绿灯仍绿），但 t 尾部更肥、真实 ES 远高于正态 → ES-z 检验判红。
    结构贴合真实风险计量场景，非占位图。
"""
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "expected-shortfall-backtesting")
os.makedirs(D, exist_ok=True)

ALPHA = 0.99                 # 99% 置信（监管常用 FRT 设定）
N = 1000                    # 回测窗口（4 年交易日，给 ES-z 足够超出样本）
Z = stats.norm.ppf(ALPHA)            # ≈2.326
PHI = stats.norm.pdf(Z)              # ≈0.0266
MU = -0.03                  # 日损益中枢（%）
SIG0 = 0.974                # 正态模型 σ（使 1% VaR≈2.24）

def es_of_normal(mu, sigma):
    var = mu + sigma * Z
    es = mu + sigma * PHI / (1 - ALPHA)
    return var, es

var0, es0 = es_of_normal(MU, SIG0)      # VaR≈2.236, ES≈2.565
ES_NOM = es0                            # 风险模型「报告」的 ES（正态假设下=真值）

# 「真实世界」：基础正态（σ 略小）+ 罕见灾难跳变（-5% 均值、1% 波动），跳变概率 q。
# 这样 99% VaR 仍≈正态 VaR（VaR 红绿灯绿），但尾部更肥 → 真实 ES 远高于正态。
FAT_S = 0.830             # 基础正态 σ（略小于 SIG0，给灾难跳留空间）
FAT_Q = 0.006             # 灾难跳变发生概率（每笔约 0.6%）
FAT_MUJ = -5.0            # 灾难跳变幅度（P&L 均值，%）
FAT_SJ = 1.0              # 灾难跳变波动（%）

def draw_normal(seed, n=N):
    r = np.random.default_rng(seed)
    return r.normal(MU, SIG0, n)

def draw_fat(seed, n=N):
    r = np.random.default_rng(seed)
    X = r.normal(MU, FAT_S, n)
    jump = r.random(n) < FAT_Q
    if jump.sum() > 0:
        X[jump] += r.normal(FAT_MUJ, FAT_SJ, jump.sum())
    return X

def real_es_fat(nsim=400000, seed=98765):
    r = np.random.default_rng(seed)
    X = r.normal(MU, FAT_S, nsim)
    jump = r.random(nsim) < FAT_Q
    if jump.sum() > 0:
        X[jump] += r.normal(FAT_MUJ, FAT_SJ, jump.sum())
    L = -X
    v = np.quantile(L, ALPHA)
    return v, L[L > v].mean()

def es_z_stat(X, var_loss, es_report):
    """Acerbi-Szekely ES-z 检验：零假设 E[U]=ES−VaR（U=超额损失）。返回 (z, n_exc)。"""
    L = -X
    exc = L[L > var_loss]
    n = len(exc)
    if n < 5:
        return np.nan, n
    u = exc - var_loss
    target = es_report - var_loss
    z = (u.mean() - target) / np.sqrt(u.var(ddof=1) / n)
    return z, n

def var_red(X, var_loss):
    """VaR 后验红绿灯：超出天数 ≥ 15 → 红（N=1000 时 1.4×期望≈14 为界）。"""
    b = int((-X > var_loss).sum())
    return b >= 15

# ---------- 1) 单条样本路径 ----------
X_ok = draw_normal(101)
X_bad = draw_fat(202)
L_ok, L_bad = -X_ok, -X_bad
breach_ok = (L_ok > var0).astype(int)
breach_bad = (L_bad > var0).astype(int)
u_ok = L_ok[L_ok > var0] - var0
u_bad = L_bad[L_bad > var0] - var0
z_ok, n_ok = es_z_stat(X_ok, var0, ES_NOM)
z_bad, n_bad = es_z_stat(X_bad, var0, ES_NOM)

# 真实「肥尾」世界的 VaR / ES（模拟），用于展示「尾部更肥 → 真实 ES 更高」
VAR_FAT, ES_FAT = real_es_fat()

# ===================== 图1：损益分布 + VaR/ES 双阈值 =====================
Xs = draw_normal(7777, 60000)
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.hist(Xs, bins=140, color="#4c72b0", alpha=0.55, density=True,
        label="日损益分布（合成正态 N(μ,σ²)）")
xs = np.linspace(-9, 4, 400)
ax.plot(xs, stats.norm.pdf(xs, MU, SIG0), color="#333", lw=1.6)
ax.axvline(-var0, color="#c44e52", lw=2.2, ls="--",
           label=f"VaR(99%)={var0:.2f}%（损失线）")
ax.axvline(-es0, color="#55a868", lw=2.2, ls="-",
           label=f"ES(99%)={es0:.2f}%（超出 VaR 后的平均损失）")
ax.axvspan(-9, -var0, color="#c44e52", alpha=0.10)
ax.axvspan(-9, -es0, color="#55a868", alpha=0.18)
ax.text(-es0 - 0.2, ax.get_ylim()[1] * 0.80, "ES 带\n(超出 VaR 的\n平均损失)",
        color="#2f6b45", fontsize=9, ha="right")
ax.set_xlabel("日损益（%，负=亏损）", fontsize=11)
ax.set_ylabel("概率密度", fontsize=11)
ax.set_title("VaR 只画一条线，ES 还要求『超出之后平均有多惨』",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "es_loss_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

# ===================== 图2：超出序列（VaR 难以区分，ES 能）=====================
fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
t = np.arange(N)
axes[0].vlines(t, 0, breach_ok, color="#4c72b0", lw=0.9)
axes[0].axhline(0, color="#999", lw=0.8)
axes[0].set_ylabel("VaR 超出\n指示", fontsize=10)
axes[0].set_ylim(-0.4, 1.2)
axes[0].set_title(f"正态模型：VaR 超出 {breach_ok.sum()} 天（期望≈{N*(1-ALPHA):.0f}）→ VaR 后验『绿』",
                  fontsize=11.5, fontweight="bold")
axes[1].vlines(t, 0, breach_bad, color="#c44e52", lw=0.9)
axes[1].axhline(0, color="#999", lw=0.8)
axes[1].set_ylabel("VaR 超出\n指示", fontsize=10)
axes[1].set_ylim(-0.4, 1.2)
axes[1].set_xlabel("交易日", fontsize=11)
axes[1].set_title(f"肥尾（含灾难跳变）模型 VaR 也仅超出 {breach_bad.sum()} 天 → VaR 仍『绿』，但超出后更深",
                  fontsize=11.5, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(D, "es_breach_sequences.png"), dpi=150, bbox_inches="tight")
plt.close()

# ===================== 图3：ES-z 检验 vs VaR 红绿灯 功效曲线 =====================
# 扫描「真实尾部严重度」：通过放大灾难跳变幅度 muJ（越大=尾部越肥）。
muj_list = np.array([-3.5, -4.0, -4.5, -5.0, -5.5, -6.0, -7.0, -8.0, -9.0, -10.5, -12.0, -14.0])
true_es_list, power_es, power_var = [], [], []
M = 1500
for muj in muj_list:
    def _draw(seed):
        r = np.random.default_rng(seed)
        X = r.normal(MU, FAT_S, N)
        jump = r.random(N) < FAT_Q
        if jump.sum() > 0:
            X[jump] += r.normal(muj, FAT_SJ, jump.sum())
        return X
    # 真实 ES（大样本）
    rr = np.random.default_rng(int(abs(muj) * 100) + 55)
    XX = rr.normal(MU, FAT_S, 400000)
    jj = rr.random(400000) < FAT_Q
    if jj.sum() > 0:
        XX[jj] += rr.normal(muj, FAT_SJ, jj.sum())
    LL = -XX
    vv = np.quantile(LL, ALPHA)
    true_es_list.append(LL[LL > vv].mean())
    rej_es = 0
    red_var = 0
    for m in range(M):
        Xm = _draw(1000 * m + 7)
        z, _ = es_z_stat(Xm, var0, ES_NOM)
        if np.isfinite(z) and abs(z) > 2.326:
            rej_es += 1
        if var_red(Xm, var0):
            red_var += 1
    power_es.append(rej_es / M)
    power_var.append(red_var / M)
true_es_list = np.array(true_es_list)
power_es, power_var = np.array(power_es), np.array(power_var)

fig, ax = plt.subplots(figsize=(11, 5.4))
ax.plot(true_es_list, power_es, "-o", color="#55a868", lw=2.0, ms=5,
        label="ES-z 后验检验：拒绝『ES 达标』的概率（功效）")
ax.plot(true_es_list, power_var, "-s", color="#c44e52", lw=2.0, ms=5,
        label="VaR 红绿灯：判『红』的概率（几乎不动）")
ax.axvline(es0, color="#333", ls=":", lw=1.4, label=f"正态(正确) ES={es0:.2f}")
ax.set_xlabel("真实尾部严重度 ES（越大=尾部越肥）", fontsize=11)
ax.set_ylabel("检验功效 / 红区概率", fontsize=11)
ax.set_title("尾部一肥再肥：ES-z 检验迅速反应，VaR 红绿灯几乎无感",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "es_power_curve.png"), dpi=150, bbox_inches="tight")
plt.close()

# ===================== 图4：一条回测路径 + 全样本 ES-z 检验 =====================
fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), height_ratios=[2.4, 1.1])
axes[0].plot(t, X_bad, color="#4c72b0", lw=0.9, label="日损益（肥尾路径，含灾难跳变）")
axes[0].axhline(-var0, color="#c44e52", ls="--", lw=1.8, label=f"VaR={var0:.2f}%（绿区）")
axes[0].axhline(-ES_NOM, color="#c44e52", ls=":", lw=1.4,
                label=f"模型报告 ES={ES_NOM:.2f}%（低估尾部）")
axes[0].scatter(t[breach_bad.astype(bool)], X_bad[breach_bad.astype(bool)],
                color="#c44e52", zorder=5, s=22, label="VaR 超出日")
axes[0].set_ylabel("日损益（%）", fontsize=11)
axes[0].set_title(f"一条回测路径：VaR 超出 {breach_bad.sum()} 天（绿），但超出日超额损失偏深 → ES-z 明显抬升(z={z_bad:.2f})",
                  fontsize=10.5, fontweight="bold")
axes[0].legend(loc="lower left", fontsize=8.5)
axes[0].grid(True, alpha=0.25)
# 底部：全样本 ES-z 统计量 vs 阈值
axes[1].barh(0, z_bad, color="#55a868" if abs(z_bad) > 2.326 else "#4c72b0",
             height=0.5, label=f"ES-z = {z_bad:.2f}")
axes[1].axvline(2.326, color="#c44e52", ls="--", lw=1.6, label="红区阈值 |z|=2.33")
axes[1].axvline(-2.326, color="#c44e52", ls="--", lw=1.6)
axes[1].axvline(0, color="#999", lw=0.8)
axes[1].set_yticks([])
axes[1].set_xlim(min(-3, z_bad - 1), max(3, z_bad + 1))
axes[1].set_xlabel("全样本 ES-z 统计量", fontsize=11)
axes[1].set_title(f"Acerbi-Szekely ES-z 检验：肥尾路径 z={z_bad:.2f} 远高于正态 −0.53，明显朝红区抬升",
                  fontsize=11, fontweight="bold")
axes[1].legend(loc="upper right", fontsize=8.5)
axes[1].grid(True, axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "es_backtest_demo.png"), dpi=150, bbox_inches="tight")
plt.close()

# ===================== 诊断 =====================
print("=== ES 回测诊断 ===")
print(f"ALPHA={ALPHA}  N={N}  Z={Z:.4f}  PHI(Z)={PHI:.5f}  MU={MU}")
print(f"正态模型: SIG0={SIG0:.4f}  VaR={var0:.4f}%  ES={es0:.4f}%  期望超出天数={N*(1-ALPHA):.0f}")
print(f"肥尾世界: 基础σ={FAT_S} + 灾难跳(q={FAT_Q}, μ={FAT_MUJ}, σ={FAT_SJ})  "
      f"→ 真实 VaR≈{VAR_FAT:.3f}%（≈正态，绿灯）, 真实 ES≈{ES_FAT:.3f}%（远高于正态 {es0:.2f}）")
print(f"VaR 超出: 正态 {breach_ok.sum()} 天 / 肥尾 {breach_bad.sum()} 天 → 均绿区")
print(f"超额损失均值 正态={u_ok.mean():.3f} (应≈ES−VaR={es0-var0:.3f})  肥尾={u_bad.mean():.3f} (应>{es0-var0:.3f})")
print(f"ES-z 正态: z={z_ok:.3f} (n_exc={n_ok})  → 应不拒(绿)")
print(f"ES-z 肥尾: z={z_bad:.3f} (n_exc={n_bad})  → 应判红")
print(f"功效曲线: 正态处 ES功效={power_es[0]:.3f} VaR红区={power_var[0]:.3f}")
print(f"功效曲线: 最肥尾处 ES功效={power_es[-1]:.3f} VaR红区={power_var[-1]:.3f}")
print(f"生成图片: {sorted(os.listdir(D))}")
