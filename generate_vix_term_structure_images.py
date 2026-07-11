#!/usr/bin/env python3
"""
为文章「VIX 期限结构与波动率风险溢价：跨期结构」(vix-term-structure)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

模型（合成但自洽的 VIX 期限结构）：
  V0(t)   = V0_MEAN + shock_t      近端瞬时波动(年化, %)，均值低于 LV → contango 为常态
  shock_t = φ·shock_{t-1} + η_t    均值回复的冲击(均值 0)
  IV(M,t) = LV + (V0_t-LV)·e^{-M/τ} + VRP0·e^{-M/τ_vrp}  期限 M(交易日) 的期货隐含波动
  E[RV](M,t) = LV + (V0_t-LV)·e^{-M/τ}                  DGP 隐含的预期未来实现波动
  VRP(M,t) = IV - E[RV] = VRP0·e^{-M/τ_vrp}            波动率风险溢价(期限结构)
  slope(t) = (IV_2M - IV_1M)/IV_1M                      期限结构斜率

carry 策略(卖近买远)：持仓日 t 的滚动收益
  PnL_t = [IV_21(t)-IV_20(t+1)] + [IV_125(t+1)-IV_126(t)]   (短腿随到期下滑赚价差, 长腿微损)
  contango(slope>0) 持有, backwardation 平仓。
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
D = os.path.join(BASE, "vix-term-structure")
os.makedirs(D, exist_ok=True)

C = {"cont": "#55A868", "back": "#C44E52", "iv": "#4C72B0", "rv": "#DD8452",
     "grid": "#DDDDDD", "pnl": "#55A868", "thr": "#888888"}

# ============================================================
# 1) 生成 shock 序列 + 期限结构函数
# ============================================================
def gen_shock(T=2500, phi=0.97, sigma=2.2, p_jump=0.010, seed=21):
    rng = np.random.default_rng(seed)
    s = np.empty(T); s[0] = 0.0
    for t in range(1, T):
        s[t] = phi * s[t - 1] + sigma * rng.normal()
        if rng.random() < p_jump:                 # 向上跳：波动率危机
            s[t] += max(0.0, rng.normal(22.0, 8.0))
    s = s - s.mean()                      # 去均值，保证 contango 占比稳定
    return s

LV = 20.0          # 长端中枢(隐含波动向此均值回复)
V0_MEAN = 5.0      # 近端瞬时波动均值(低于 LV → contango 为常态)
s = gen_shock(T=2500, seed=21)
V0 = V0_MEAN + s
V0 = np.clip(V0, 3.0, 200.0)              # 波动不能为负、不可爆炸
mats = np.array([21, 42, 63, 84, 105, 126])
IV = {M: (LV + (V0 - LV) * np.exp(-M / 42.0) + 3.0 * np.exp(-M / 30.0)) for M in mats}
# 任意期限上的 IV（用于滚动收益）
def IV_at(M, t):
    return LV + (V0[t] - LV) * np.exp(-M / 42.0) + 3.0 * np.exp(-M / 30.0)

slope = (IV[42] - IV[21]) / IV[21]
E_RV = {M: LV + (V0 - LV) * np.exp(-M / 42.0) for M in mats}
VRP = {M: IV[M] - E_RV[M] for M in mats}     # = 3.0·e^{-M/30}，随期限递减

# 快照索引
cand = np.where(slope > 0.04)[0]
cont_idx = int(cand[len(cand) // 3])          # 一段稳定 contango
back_idx = int(np.argmax(s))                  # 最深的恐慌(贴水)

# ============================================================
# 图 1：两种状态的期限结构
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
xlab = ["1M", "2M", "3M", "4M", "5M", "6M"]; x = np.arange(len(mats))
for ax, (idx, tag, col) in zip(axes, [(cont_idx, "Contango 升水 (常态)", C["cont"]),
                                       (back_idx, "Backwardation 贴水 (恐慌)", C["back"])]):
    ys = [IV[M][idx] for M in mats]
    ax.plot(x, ys, color=col, lw=2, marker="o", label="VIX 期货 IV")
    ax.axhline(LV, color=C["thr"], ls="--", lw=1, label="长端中枢 LV=17")
    ax.set_xticks(x); ax.set_xticklabels(xlab)
    ax.set_xlabel("到期期限"); ax.set_ylabel("隐含波动率 (%)")
    ax.set_title("%s\n近端 V0=%.1f" % (tag, V0[idx]))
    ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
fig.suptitle("同一标的、不同时点的 VIX 期货期限结构会整体翻转", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(D, "vix_term_structure.png"), dpi=130)
plt.close()

# ============================================================
# 图 2：期限结构斜率时序
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.plot(np.arange(len(slope)), slope, color=C["iv"], lw=0.8)
ax.axhline(0, color="black", lw=0.8)
ax.fill_between(np.arange(len(slope)), 0, slope, where=slope > 0,
               color=C["cont"], alpha=0.25, label="Contango (斜率>0)")
ax.fill_between(np.arange(len(slope)), slope, 0, where=slope < 0,
               color=C["back"], alpha=0.30, label="Backwardation (斜率<0)")
ax.set_xlabel("交易日"); ax.set_ylabel("期限结构斜率 (IV_2M-IV_1M)/IV_1M")
ax.set_title("期限结构斜率在 contango 与 backwardation 之间反复切换")
ax.legend(loc="upper right", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "vix_term_slope.png"), dpi=130)
plt.close()

# ============================================================
# 图 3：VRP 期限结构（模型隐含，短端最大）
# ============================================================
vrp_mean = np.array([np.mean(VRP[M]) for M in mats])
vrp_std = np.array([np.std(VRP[M]) for M in mats])
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.bar(x, vrp_mean, color=C["rv"], yerr=vrp_std, capsize=3, alpha=0.85,
       label="VRP 期限结构均值")
ax.plot(x, vrp_mean, color="black", lw=1.2, marker="s")
ax.set_xticks(x); ax.set_xticklabels(xlab)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("到期期限"); ax.set_ylabel("波动率风险溢价 VRP = IV - E[RV] (%)")
ax.set_title("VRP 期限结构：短端溢价最大、随期限递减(均值 %.2f%%→%.2f%%)"
             % (vrp_mean[0], vrp_mean[-1]))
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(D, "vix_vrp_term.png"), dpi=130)
plt.close()

# ============================================================
# 图 4：做多曲线 carry 策略（卖近买远，滚动下滑赚价差）
# ============================================================
pos = (slope > 0).astype(float)
daily = np.zeros(len(slope))
for t in range(1, len(slope) - 1):
    if pos[t] > 0:                 # 仅 contango 持有；次日按"下滑一个期限"重新估值
        short_leg = IV_at(21, t) - IV_at(20, t + 1)  # 短近月：随到期下滑赚价差
        long_leg = IV_at(125, t + 1) - IV_at(126, t) # 长远月：微损
        daily[t] = short_leg + long_leg
eq = np.cumsum(daily)
mdd = float(np.min(eq - np.maximum.accumulate(eq)))
n_cont = int(np.sum(slope > 0)); n_back = int(np.sum(slope <= 0))
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.plot(np.arange(len(eq)), eq, color=C["pnl"], lw=1.2, label="做多曲线 carry 权益")
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("交易日"); ax.set_ylabel("累计收益 (vol 点)")
ax.set_title("卖近买远 carry：contango 稳赚，backwardation 翻转日吃掉利润 (累计 %.1f, 回撤 %.1f)"
             % (eq[-1], mdd))
ax.legend(loc="upper left", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "vix_strategy.png"), dpi=130)
plt.close()

print("=== VIX 期限结构与波动率风险溢价 关键数字 ===")
print("样本 T=%d, 中枢 LV=%.0f, 近端均值 V0=%.0f, 期限 M=%s(交易日)" % (len(s), LV, V0_MEAN, mats.tolist()))
print("Contango 快照(第%d日): V0=%.1f IV_1M=%.1f IV_6M=%.1f 斜率=%.3f"
      % (cont_idx, V0[cont_idx], IV[21][cont_idx], IV[126][cont_idx], slope[cont_idx]))
print("Backwardation 快照(第%d日): V0=%.1f IV_1M=%.1f IV_6M=%.1f 斜率=%.3f"
      % (back_idx, V0[back_idx], IV[21][back_idx], IV[126][back_idx], slope[back_idx]))
print("VRP 期限结构均值(1M→6M): " + ", ".join("%.2f" % v for v in vrp_mean) + " %")
print("contango 占比=%.1f%%(%d日), backwardation 占比=%.1f%%(%d日)"
      % (100 * n_cont / len(slope), n_cont, 100 * n_back / len(slope), n_back))
print("做多曲线 carry: 累计=%.1f vol点, 最大回撤=%.1f" % (eq[-1], mdd))
print("\n图片已保存到:", D)
