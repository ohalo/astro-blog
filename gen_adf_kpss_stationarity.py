#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「ADF 与 KPSS 联合检验：把是否平稳从猜变成判据」
(adf-kpss-stationarity) 生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（自洽合成，仅用于演示方法；真实落地见文末路径）：
  * 用 Davies-Harte 生成三类已知状态的序列：
      A. 随机游走 I(1)（非平稳，d=1）
      B. AR(1) 平稳过程（φ=0.85，平稳，d=0）
      C. 分数差分长记忆 FI(d=0.45)（均值回复但非平稳，0<d<1）
  * 用 statsmodels 的 adfuller (ADF) 与 kpss 做联合检验：
      - ADF 原假设=有单位根(非平稳)
      - KPSS 原假设=平稳
  * 图1：三序列轨迹对照（肉眼几乎分不清 A 与 C，判据才能分）
  * 图2：ADF 统计量 vs 临界值 + 是否拒绝（三类结果）
  * 图3：KPSS 统计量 vs 临界值 + 是否拒绝
  * 图4：联合判定矩阵（4 区：平稳 / 趋势平稳 / 差分平稳(I1) / 长记忆）
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from statsmodels.tsa.stattools import adfuller, kpss

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "adf-kpss-stationarity"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"rw": "#C44E52", "ar": "#4C72B0", "frac": "#55A868", "grid": "#DDDDDD",
     "rej": "#C44E52", "norej": "#55A868", "mk": "#8172B3", "dark": "#333333"}

rng = np.random.default_rng(20260719)
N = 1500

# ---------------- 三类序列 ----------------
# A. 随机游走（单位根，非平稳）
rw = np.cumsum(rng.standard_normal(N))
# B. AR(1) 平稳 φ=0.85
ar = np.zeros(N)
for t in range(1, N):
    ar[t] = 0.85 * ar[t-1] + rng.standard_normal()
# C. 分数差分长记忆 FI(d=0.45)：用 fracdiff 权重对白噪声卷积
d = 0.45
w = np.zeros(N + 50)
for k in range(len(w)):
    w[k] = np.prod([(d + (k - j) - 1) / (k - j) for j in range(k)]) if k > 0 else 1.0
white = rng.standard_normal(N + 50) * 0.25
frac = np.convolve(white, w, mode="full")[:N]
frac = frac - frac.mean()

labels = ["随机游走 I(1)\n(非平稳)", "AR(1) φ=0.85\n(平稳)", "FI(d=0.45)\n(长记忆, 0<d<1)"]
series = [rw, ar, frac]
colors = [C["rw"], C["ar"], C["frac"]]

# ---------------- 联合检验 ----------------
def test(seq):
    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(seq, autolag="AIC")
    # KPSS 默认检验 level 平稳；用 regression='c'
    try:
        kpss_stat, kpss_p, kpss_lag, kpss_crit = kpss(seq, regression="c", nlags="auto")
    except Exception:
        kpss_stat, kpss_p, kpss_lag, kpss_crit = kpss(seq)
    return {
        "adf_stat": adf_stat, "adf_p": adf_p, "adf_crit": adf_crit,
        "kpss_stat": kpss_stat, "kpss_p": kpss_p, "kpss_crit": kpss_crit,
    }

results = [test(s) for s in series]

def classify(r):
    # 4 区判定
    adf_rej = r["adf_p"] < 0.05       # 拒绝单位根 -> 平稳证据
    kpss_rej = r["kpss_p"] < 0.05    # 拒绝平稳 -> 非平稳证据
    if (not adf_rej) and (not kpss_rej):
        return "平稳 / 趋势平稳"
    if adf_rej and kpss_rej:
        return "趋势平稳（需去趋势）"
    if adf_rej and (not kpss_rej):
        return "平稳"
    if (not adf_rej) and kpss_rej:
        return "差分平稳 I(1) / 长记忆"
    return "不确定"

verdicts = [classify(r) for r in results]
for nm, r, v in zip(labels, results, verdicts):
    print("  %s" % nm.replace("\n", " "))
    print("    ADF stat=%.3f p=%.4f | KPSS stat=%.3f p=%.4f -> %s"
          % (r["adf_stat"], r["adf_p"], r["kpss_stat"], r["kpss_p"], v))

# =========================== 图 1：三序列轨迹 ===========================
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for ax, s, col, lab, v in zip(axes, series, colors, labels, verdicts):
    ax.plot(s, color=col, lw=0.9)
    ax.set_ylabel("取值", fontsize=9)
    ax.set_title("%s  →  联合判定: %s" % (lab, v), fontsize=10.5)
    ax.grid(True, color=C["grid"], lw=0.5)
axes[-1].set_xlabel("时间")
fig.suptitle("三序列「肉眼难分」：A 与 C 都像在乱走，但判据能区分 (N=%d)" % N, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, "three_series.png"))
plt.close(fig)

# =========================== 图 2：ADF 检验 ===========================
fig, ax = plt.subplots(figsize=(10, 5.0))
names = [l.replace("\n", " ") for l in labels]
xpos = np.arange(len(names))
adf_stats = [r["adf_stat"] for r in results]
# 取 1% / 5% / 10% 临界值（取 5% 作主参考线）
crit5 = [r["adf_crit"]["5%"] for r in results]
colors_bar = []
for r in results:
    colors_bar.append(C["rej"] if r["adf_p"] < 0.05 else C["norej"])
bars = ax.bar(xpos, adf_stats, color=colors_bar, width=0.55)
for b, s in zip(bars, adf_stats):
    ax.text(b.get_x() + b.get_width() / 2, s - 0.4, "%.2f" % s, ha="center", fontsize=10, color="#222")
for xp, c in zip(xpos, crit5):
    ax.axhline(c, xmin=(xp - 0.3) / len(names), xmax=(xp + 0.3) / len(names),
               color=C["dark"], ls="--", lw=1.0)
ax.axhline(0, color=C["dark"], lw=0.8)
ax.set_xticks(xpos); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("ADF 检验统计量")
ax.set_title("ADF 检验：统计量 < 临界值(虚线, 5%) 则拒绝单位根 → 平稳证据", fontsize=11)
ax.grid(True, color=C["grid"], axis="y", lw=0.5)
ax.text(0.02, 0.04, "红=拒绝单位根(平稳)  灰=不拒绝(非平稳)", transform=ax.transAxes, fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "adf_test.png"))
plt.close(fig)

# =========================== 图 3：KPSS 检验 ===========================
fig, ax = plt.subplots(figsize=(10, 5.0))
kpss_stats = [r["kpss_stat"] for r in results]
kpss_crit5 = [r["kpss_crit"]["5%"] for r in results]
colors_bar = []
for r in results:
    colors_bar.append(C["rej"] if r["kpss_p"] < 0.05 else C["norej"])
bars = ax.bar(xpos, kpss_stats, color=colors_bar, width=0.55)
for b, s in zip(bars, kpss_stats):
    ax.text(b.get_x() + b.get_width() / 2, s + 0.02, "%.2f" % s, ha="center", fontsize=10, color="#222")
for xp, c in zip(xpos, kpss_crit5):
    ax.axhline(c, xmin=(xp - 0.3) / len(names), xmax=(xp + 0.3) / len(names),
               color=C["dark"], ls="--", lw=1.0)
ax.set_xticks(xpos); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("KPSS 检验统计量")
ax.set_title("KPSS 检验：统计量 > 临界值(虚线, 5%) 则拒绝平稳 → 非平稳证据", fontsize=11)
ax.grid(True, color=C["grid"], axis="y", lw=0.5)
ax.text(0.02, 0.90, "红=拒绝平稳(非平稳)  灰=不拒绝(平稳)", transform=ax.transAxes, fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "kpss_test.png"))
plt.close(fig)

# =========================== 图 4：联合判定矩阵 ===========================
# 四象限：x=ADF 拒绝?  y=KPSS 拒绝?
# 区域定义（左下=都平稳, 右上=都非平稳, 右下=ADF平稳KPSS非平稳, 左上=ADF非平稳KPSS平稳）
fig, ax = plt.subplots(figsize=(8.5, 7))
# 背景区
ax.axvspan(0.5, 1.5, ymin=0.5, ymax=1.5, color="#F2C4C4", alpha=0.6)  # 都拒绝(趋势平稳)
ax.axvspan(-0.5, 0.5, ymin=-0.5, ymax=0.5, color="#D9EAD3", alpha=0.6)  # 都接受(平稳)
ax.axvspan(0.5, 1.5, ymin=-0.5, ymax=0.5, color="#FCE5C3", alpha=0.6)   # ADF平稳 KPSS非平稳(长记忆/I1)
ax.axvspan(-0.5, 0.5, ymin=0.5, ymax=1.5, color="#CFE2F3", alpha=0.6)    # ADF非平稳 KPSS平稳(矛盾/倾向I1)

zones = [
    (0.0, 0.0, "平稳\n(两者一致)"),
    (1.0, 1.0, "趋势平稳\n(两者都拒)"),
    (1.0, 0.0, "差分平稳 I(1) /\n长记忆 (KPSS拒)"),
    (0.0, 1.0, "矛盾信号\n(倾向 I(1))"),
]
for zx, zy, txt in zones:
    ax.text(zx, zy, txt, ha="center", va="center", fontsize=9.5, color="#444")

# 打点：x=ADF 是否拒绝(1=是), y=KPSS 是否拒绝(1=是)
pts = [(1 if r["adf_p"] < 0.05 else 0, 1 if r["kpss_p"] < 0.05 else 0) for r in results]
for (px, py), col, lab in zip(pts, colors, labels):
    ax.scatter(px, py, s=220, color=col, edgecolor="#222", zorder=5, linewidth=1.2)
    ax.text(px + 0.06, py + 0.06, lab.replace("\n", " "), fontsize=8.5, color="#222")

ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.4, 1.4)
ax.set_xticks([0, 1]); ax.set_xticklabels(["ADF 不拒\n(有单位根)", "ADF 拒绝\n(平稳)"], fontsize=9)
ax.set_yticks([0, 1]); ax.set_yticklabels(["KPSS 不拒\n(平稳)", "KPSS 拒绝\n(非平稳)"], fontsize=9)
ax.set_xlabel("ADF 方向", fontsize=10)
ax.set_ylabel("KPSS 方向", fontsize=10)
ax.set_title("联合判定矩阵：两个检验互补，避免单向误判", fontsize=11.5)
ax.grid(True, color=C["grid"], lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "joint_matrix.png"))
plt.close(fig)

print("DONE ->", OUT, os.listdir(OUT))
