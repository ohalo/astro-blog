#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 Diebold-Mariano 预测精度检验文章配图（合成数据）。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for cand in ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "Heiti SC", "STHeiti"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [cand]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/diebold-mariano-test"
os.makedirs(OUT, exist_ok=True)

BLUE = "#2c6fbb"
RED = "#d1495b"
GREEN = "#2a9d8f"
GRAY = "#6c757d"
ORANGE = "#e09f3e"
PURPLE = "#8e6bbf"

rng = np.random.default_rng(2026)

# ============================================================
# 公共：生成真实波动率序列，两个模型给出预测
# ============================================================
T = 500
# 真实已实现波动率：均值回复 + 波动聚集
true_vol = np.zeros(T)
true_vol[0] = 0.20
for t in range(1, T):
    true_vol[t] = 0.02 + 0.9 * true_vol[t - 1] + rng.normal(0, 0.015)
true_vol = np.abs(true_vol)

# 模型A（GARCH型）：偏差小、噪声小
predA = true_vol + rng.normal(0, 0.020, T) + 0.002
# 模型B（简单移动平均）：滞后导致系统偏差 + 更大噪声
predB = np.roll(true_vol, 1) + rng.normal(0, 0.032, T) + 0.004
predB[0] = predA[0]

# 预测误差
eA = true_vol - predA
eB = true_vol - predB

# ============================================================
# 图1：两个模型的预测 vs 真实
# ============================================================
fig, ax = plt.subplots(figsize=(10.5, 4.4))
s = slice(100, 220)
idx = np.arange(100, 220)
ax.plot(idx, true_vol[s] * 100, color="black", lw=2.0, label="真实已实现波动率")
ax.plot(idx, predA[s] * 100, color=BLUE, lw=1.3, alpha=0.85, label="模型A（GARCH）预测")
ax.plot(idx, predB[s] * 100, color=RED, lw=1.3, alpha=0.75, label="模型B（移动平均）预测")
ax.set_title("两个波动率模型的预测轨迹（局部放大）", fontsize=11)
ax.set_xlabel("时间")
ax.set_ylabel("波动率 (%)")
ax.legend(fontsize=9.5)
plt.tight_layout()
plt.savefig(f"{OUT}/forecast_vs_actual.png", dpi=130, bbox_inches="tight")
plt.close()
print("saved forecast_vs_actual.png")

# ============================================================
# 图2：损失差序列 d_t = L(eA) - L(eB)
# ============================================================
lossA = eA ** 2  # MSE 损失
lossB = eB ** 2
d = lossA - lossB  # 负值=A更好

fig, ax = plt.subplots(figsize=(10.5, 4.2))
ax.bar(np.arange(T), d * 1e4, color=np.where(d <= 0, BLUE, RED), width=1.0, alpha=0.7)
ax.axhline(0, color="black", lw=0.8)
ax.axhline(d.mean() * 1e4, color=GREEN, lw=1.8, ls="--",
           label=f"平均损失差 = {d.mean()*1e4:.2f}（<0 表示A整体更优）")
ax.set_title("损失差序列 d(t) = MSE_A - MSE_B（负=模型A当期更准）", fontsize=11)
ax.set_xlabel("时间")
ax.set_ylabel("损失差 (x10^-4)")
ax.legend(fontsize=9.5)
plt.tight_layout()
plt.savefig(f"{OUT}/loss_differential.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"saved loss_differential.png  mean_d={d.mean():.6e}")

# ============================================================
# DM 统计量计算
# ============================================================
def dm_test(d, h=1):
    """Diebold-Mariano 统计量（含 HAC 长期方差 + Harvey 小样本修正）。"""
    n = len(d)
    dbar = d.mean()
    # 长期方差：Newey-West，带宽 h-1
    u = d - dbar
    gamma0 = np.sum(u * u) / n
    lrv = gamma0
    for k in range(1, h):
        gk = np.sum(u[k:] * u[:n - k]) / n
        lrv += 2 * gk
    dm = dbar / np.sqrt(lrv / n)
    # Harvey, Leybourne, Newbold (1997) 小样本修正
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * corr
    return dm, dm_hln

dm1, dm1_hln = dm_test(d, h=1)
print(f"DM(h=1)={dm1:.3f}  HLN修正={dm1_hln:.3f}")

# ============================================================
# 图3：DM 统计量 vs 天真的配对t检验 —— 预测期 h 的影响
# 展示多步预测下损失差自相关必须用 HAC
# ============================================================
# 构造多步预测（h=5）导致损失差有自相关
h_multi = 5
eA_h = np.convolve(eA, np.ones(h_multi) / h_multi, mode="same")
eB_h = np.convolve(eB, np.ones(h_multi) / h_multi, mode="same")
d_h = eA_h ** 2 - eB_h ** 2

def naive_t(d):
    n = len(d)
    return d.mean() / (np.std(d, ddof=1) / np.sqrt(n))

hs = np.arange(1, 11)
dm_vals, naive_vals = [], []
for h in hs:
    eA_h = np.convolve(eA, np.ones(h) / h, mode="same")
    eB_h = np.convolve(eB, np.ones(h) / h, mode="same")
    dh = eA_h ** 2 - eB_h ** 2
    _, dm_c = dm_test(dh, h=h)
    dm_vals.append(dm_c)
    naive_vals.append(naive_t(dh))

fig, ax = plt.subplots(figsize=(9.2, 4.5))
ax.plot(hs, naive_vals, "-o", color=RED, ms=5, lw=1.6,
        label="天真配对t（忽略自相关，绝对值虚高）")
ax.plot(hs, dm_vals, "-s", color=BLUE, ms=5, lw=1.6,
        label="DM 统计量（HAC + HLN 修正）")
ax.axhline(-1.96, color="black", ls="--", lw=1, label="±1.96 临界值")
ax.axhline(1.96, color="black", ls="--", lw=1)
ax.set_title("多步预测 h 越大，损失差自相关越强，天真t越离谱", fontsize=11)
ax.set_xlabel("预测步长 h")
ax.set_ylabel("检验统计量")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/dm_vs_naive.png", dpi=130, bbox_inches="tight")
plt.close()
print("saved dm_vs_naive.png")

# ============================================================
# 图4：蒙特卡洛——两模型真实等优时，DM 的经验分布应≈N(0,1)
# ============================================================
n_sim = 4000
dm_null = []
naive_null = []
for _ in range(n_sim):
    # 两模型损失完全等优：损失差纯噪声（但注入 h=3 步的自相关）
    raw = rng.normal(0, 1, T + 4)
    dd = np.convolve(raw, np.ones(3) / 3, mode="valid")[:T]
    _, dmc = dm_test(dd, h=3)
    dm_null.append(dmc)
    naive_null.append(naive_t(dd))

rej_dm = np.mean(np.abs(dm_null) > 1.96)
rej_naive = np.mean(np.abs(naive_null) > 1.96)

fig, ax = plt.subplots(figsize=(9.2, 4.6))
bins = np.linspace(-5, 5, 60)
ax.hist(naive_null, bins=bins, color=RED, alpha=0.5,
        label=f"天真配对t（错误拒绝率={rej_naive:.1%}）")
ax.hist(dm_null, bins=bins, color=BLUE, alpha=0.55,
        label=f"DM统计量（错误拒绝率={rej_dm:.1%}）")
xs = np.linspace(-5, 5, 200)
ax.plot(xs, len(dm_null) * (bins[1] - bins[0]) *
        np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi),
        color="black", lw=1.6, label="标准正态 N(0,1)")
ax.axvline(1.96, color=GRAY, ls="--", lw=1)
ax.axvline(-1.96, color=GRAY, ls="--", lw=1)
ax.set_title(f"两模型等优时的统计量分布（h=3多步预测，名义5%）", fontsize=11)
ax.set_xlabel("统计量")
ax.set_ylabel("频数")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/monte_carlo_size.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"saved monte_carlo_size.png  rej_dm={rej_dm:.3f} rej_naive={rej_naive:.3f}")
print("ALL DONE")
