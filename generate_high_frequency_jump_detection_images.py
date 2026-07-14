#!/usr/bin/env python3
"""
为文章「高频跳跃检测(Lee-Mykland)：从噪声里揪出真实的瞬时跳变」
生成真实配图 + 计算正文引用的所有关键数字。

机制（自洽合成，仅用于演示；落地见文末路径）：
  * 一根分钟级对数收益序列(一天 390 分钟)，波动率聚类：平静/正常/动荡三段 regime。
  * 人工注入 5 个真实跳变(±)，散布在三段 regime 中。
  * Lee-Mykland(2008) 检验：L(r_t)= r_t / ŝ_{t-1}，其中 ŝ_{t-1} 是过去 K=30 分钟
    平方收益的滚动标准差（局部波动率估计）。在「无跳变、局部波动恒定」下 L ≈ N(0,1)。
  * 多重检验校正临界值 c = Φ⁻¹(1 − α/(2n))；|L|>c 判为跳变。
  * 对比「朴素固定阈值」(用平静市 σ 标定)：能抓跳变，却在动荡市疯狂误报；
    LM 因用局部波动率缩放，动荡市自动抬高门槛，误报近零。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams


def norm_ppf(p):
    """标准正态分位函数（Acklam 有理逼近），不依赖 scipy。"""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "high-frequency-jump-detection")
os.makedirs(D, exist_ok=True)

C = {"jump": "#C44E52", "stat": "#4C72B0", "vol": "#55A868",
     "grid": "#DDDDDD", "warn": "#DD8452", "calm": "#8172B3", "true": "#2F2F2F"}

rng = np.random.default_rng(20260714)
N = 390                                  # 一天交易分钟数(6.5h×60)
# 波动率聚类：平静 / 正常 / 动荡 三段
sigma_true = np.empty(N)
sigma_true[:130] = 0.0007                # 平静
sigma_true[130:260] = 0.0014             # 正常
sigma_true[260:] = 0.0030                # 动荡
ret = rng.normal(0, sigma_true)
# 注入 5 个真实跳变（散布在三段 regime）
jumps = {45: 0.012, 150: -0.016, 250: 0.020, 330: -0.014, 370: 0.013}
true_jump_min = sorted(jumps.keys())
for m, s in jumps.items():
    ret[m] += s

# ----------------------------------------------------------------------------
# Lee-Mykland 检验
# ----------------------------------------------------------------------------
K = 30
locvol = np.full(N, np.nan)              # ŝ_{t-1}：过去 K 分钟平方收益滚动 std
for t in range(K, N):
    locvol[t] = np.sqrt(np.mean(ret[t - K:t] ** 2))
L = np.full(N, np.nan)                   # 检验统计量
for t in range(K, N):
    L[t] = ret[t] / locvol[t]

n_eff = N - K
alpha = 0.05
c = norm_ppf(1 - alpha / (2 * n_eff))    # Bonferroni 多重检验校正
detected = [t for t in range(K, N) if abs(L[t]) > c]

# ----------------------------------------------------------------------------
# 朴素固定阈值（用平静市 σ 标定）—— 对比组
# ----------------------------------------------------------------------------
calm_std = float(np.std(ret[:130]))
naive_thr = 4.0 * calm_std
naive_flags = [t for t in range(N) if abs(ret[t]) > naive_thr]

def match(detected_list, true_list, tol=2):
    tp = 0; used = set()
    for tm in true_list:
        for d in detected_list:
            if abs(d - tm) <= tol and d not in used:
                tp += 1; used.add(d); break
    fp = len(detected_list) - tp
    return tp, fp

tp_lm, fp_lm = match(detected, true_jump_min)
tp_naive, fp_naive = match(naive_flags, true_jump_min)
naive_fp_turb = [t for t in naive_flags
                 if t >= 260 and not any(abs(t - tm) <= 2 for tm in true_jump_min)]

# 各真实跳变的 L 值与是否被 LM 抓到
jump_L = {m: L[m] for m in true_jump_min}
LM_caught = {m: (m in detected) for m in true_jump_min}

print("=" * 70)
print("高频跳跃检测 (Lee-Mykland) 关键数字 (seed 20260714)")
print("=" * 70)
print(f"分钟数 N={N}  局部波动率窗 K={K}  有效样本 n_eff={n_eff}")
print(f"Bonferroni 临界值 c = Φ⁻¹(1−α/(2n)) = {c:.3f}  (α={alpha})")
print(f"真实跳变分钟: {true_jump_min}  幅度: {[jumps[m] for m in true_jump_min]}")
print(f"各真实跳变 L 值: {[round(jump_L[m],2) for m in true_jump_min]}")
print(f"      LM 是否抓到: {[LM_caught[m] for m in true_jump_min]}")
print(f"\nLM 检测: 命中 {tp_lm}/{len(true_jump_min)}  误报 {fp_lm}  总标记 {len(detected)}")
print(f"朴素阈值(={naive_thr:.5f}, 平静市4σ): 命中 {tp_naive}/{len(true_jump_min)}  误报 {fp_naive}  总标记 {len(naive_flags)}")
print(f"  其中动荡市(分钟≥260)误报: {len(naive_fp_turb)} 个")
print(f"平静市 σ={sigma_true[0]:.4f} 正常市 σ={sigma_true[130]:.4f} 动荡市 σ={sigma_true[260]:.4f}")

# ----------------------------------------------------------------------------
# 图 1：分钟级对数收益 + 真实跳变
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(range(N), ret * 100, color=C["calm"], lw=0.8)
for m in true_jump_min:
    ax.scatter([m], [ret[m] * 100], color=C["jump"], s=45, zorder=5)
ax.axvspan(0, 130, color=C["vol"], alpha=0.06)
ax.axvspan(260, N, color=C["warn"], alpha=0.08)
ax.set_xlabel("交易分钟 (9:30→16:00)"); ax.set_ylabel("分钟对数收益 (%)")
ax.set_title("注入跳变的分钟级对数收益：红点=5 个真实跳变（平静/正常/动荡三段波动）")
ax.grid(True, color=C["grid"], lw=0.6)
ax.legend(["分钟收益", "真实跳变"], fontsize=9, loc="upper right")
plt.tight_layout(); plt.savefig(os.path.join(D, "returns_with_jumps.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 图 2：LM 检验统计量 + 临界带
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(range(N), np.where(np.isnan(L), np.nan, L), color=C["stat"], lw=0.9)
ax.axhline(c, color=C["jump"], ls="--", lw=1.2, label=f"+临界 c={c:.2f}")
ax.axhline(-c, color=C["jump"], ls="--", lw=1.2)
for m in detected:
    ax.scatter([m], [L[m]], color=C["jump"], s=30, zorder=5)
ax.set_xlabel("交易分钟"); ax.set_ylabel("LM 统计量 L(r_t)")
ax.set_title("Lee-Mykland 统计量：|L|>c 判为跳变（红点=被标记时刻）")
ax.grid(True, color=C["grid"], lw=0.6)
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout(); plt.savefig(os.path.join(D, "lm_statistic.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 图 3：局部波动率估计 vs 真实波动率
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(range(N), sigma_true * 100, color=C["true"], lw=1.4, label="真实波动率 σ_t")
ax.plot(range(N), np.where(np.isnan(locvol), np.nan, locvol) * 100,
        color=C["vol"], lw=1.0, alpha=0.85, label="LM 局部估计 ŝ_{t-1}")
ax.set_xlabel("交易分钟"); ax.set_ylabel("波动率 (%)")
ax.set_title("局部波动率估计：LM 靠它自适应缩放门槛（动荡段自动抬高）")
ax.grid(True, color=C["grid"], lw=0.6)
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout(); plt.savefig(os.path.join(D, "local_vol_estimate.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 图 4：LM vs 朴素阈值 标记对比（动荡段误报一目了然）
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(range(N), ret * 100, color=C["calm"], lw=0.7, alpha=0.7, label="分钟收益")
for m in detected:
    ax.scatter([m], [ret[m] * 100], color=C["stat"], s=40, marker="o",
               zorder=6, label=None if m != detected[0] else "LM 标记")
for t in naive_flags:
    ax.scatter([t], [ret[t] * 100], color=C["warn"], s=22, marker="x",
               zorder=5, label=None if t != naive_flags[0] else "朴素阈值标记")
ax.axvspan(260, N, color=C["warn"], alpha=0.06)
ax.set_xlabel("交易分钟"); ax.set_ylabel("分钟对数收益 (%)")
ax.set_title("LM vs 朴素固定阈值：动荡段朴素检测器(×)疯狂误报，LM(○)几乎不误报")
ax.grid(True, color=C["grid"], lw=0.6)
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout(); plt.savefig(os.path.join(D, "lm_vs_naive.png"), dpi=130); plt.close()

print("done ->", D)
