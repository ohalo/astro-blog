#!/usr/bin/env python3
"""生成 排列熵与复杂度 文章：纯 numpy 从零实现排列熵（Bandt & Pompe 2002）+ 4 张真实图表。

排列熵（Permutation Entropy, PE）用『序数模式』度量时间序列的复杂度：把每个时间窗
里的相对大小排序（如 [小,大,中] → 升序排名 (0,2,1)），统计所有模式的出现频率，再算
香农熵。关键性质：
  - 规则/可预测序列（正弦）→ 少数模式反复出现 → PE 低
  - 随机序列（白噪声）→ 所有模式近乎均匀 → PE 高（归一化≈1）
  - 混沌序列（逻辑斯蒂）→ 介于二者之间
它只关心『顺序』不关心『幅度』，所以对单调变换、量纲、平移都不变——很适合跨品种
比复杂度、做 regime 检测。本文从零实现 ordinal pattern 计数 + 滑动窗 PE，并在
『正弦→噪声→正弦』切换序列上展示 PE 如何实时追踪 regime。
"""
import numpy as np
import os
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False
fm._load_fontmanager()


def permutation_entropy(x, m, tau=1):
    """Bandt-Pompe 排列熵（归一化到 [0,1]）。x: 1D 序列。"""
    x = np.asarray(x, float)
    n_patterns = math.factorial(m)
    emb_len = (m - 1) * tau + 1
    n = len(x)
    if n < emb_len + 1:
        return None
    # 预生成排列→索引映射
    from itertools import permutations as _perm
    perm2idx = {p: i for i, p in enumerate(_perm(range(m)))}
    counts = np.zeros(n_patterns)
    for i in range(n - emb_len + 1):
        window = x[i:i + emb_len:tau]
        # 加极小噪声打破平局（否则相等值排名不定）
        order = tuple(np.argsort(window + 1e-12 * np.random.randn(m)))
        counts[perm2idx[order]] += 1
    counts = counts[counts > 0]
    p = counts / counts.sum()
    pe = -np.sum(p * np.log(p)) / np.log(n_patterns)
    return float(pe)


def pe_series(x, win, m, tau=1):
    """滑动窗排列熵（返回与 x 等长的序列填充，前 win-1 置 nan）"""
    out = np.full(len(x), np.nan)
    for i in range(win, len(x) + 1):
        out[i - 1] = permutation_entropy(x[i - win:i], m, tau)
    return out


# ============ 1. 三类序列 ============
rng = np.random.default_rng(20260723)
N = 2000
t = np.arange(N)

sine = np.sin(0.12 * t)                       # 规则
logi = np.zeros(N)
v = 0.4
for i in range(1, N):
    v = 4.0 * v * (1 - v)                     # 逻辑斯蒂混沌 x_{n+1}=4x(1-x)
    logi[i] = v
noise = rng.normal(0, 1, N)                   # 随机白噪声

m_pe = 5
pe_sine = permutation_entropy(sine, m_pe)
pe_logi = permutation_entropy(logi, m_pe)
pe_noise = permutation_entropy(noise, m_pe)
print(f"  PE(m={m_pe})  正弦={pe_sine:.3f}  逻辑斯蒂={pe_logi:.3f}  白噪声={pe_noise:.3f}")

# ============ 2. 序数模式频率分布（用 m=4 直观展示，24 种模式） ============
m_disp = 4
from itertools import permutations as _perm
perm2idx = {p: i for i, p in enumerate(_perm(range(m_disp)))}


def pattern_counts(x, m, tau=1):
    x = np.asarray(x, float)
    n_pat = math.factorial(m)
    emb_len = (m - 1) * tau + 1
    counts = np.zeros(n_pat)
    for i in range(len(x) - emb_len + 1):
        window = x[i:i + emb_len:tau]
        order = tuple(np.argsort(window + 1e-12 * np.random.randn(m)))
        counts[perm2idx[order]] += 1
    return counts / counts.sum()


cnt_sine = pattern_counts(sine, m_disp)
cnt_logi = pattern_counts(logi, m_disp)
cnt_noise = pattern_counts(noise, m_disp)

# ============ 3. 滑动窗 PE：正弦→噪声→正弦 切换 ============
seg = N // 3
mixed = np.concatenate([np.sin(0.12 * np.arange(seg)),
                        rng.normal(0, 1, seg),
                        np.sin(0.12 * np.arange(seg))])
win = 200
pe_mixed = pe_series(mixed, win, m_pe)
regime = np.array(["正弦"] * seg + ["噪声"] * seg + ["正弦"] * seg)

# ============ 4. PE 随嵌入维度 m 曲线（诚实边界：m 太大→模式稀疏不可靠） ============
ms = list(range(2, 7))
pe_m_sine = [permutation_entropy(sine, mm) for mm in ms]
pe_m_logi = [permutation_entropy(logi, mm) for mm in ms]
pe_m_noise = [permutation_entropy(noise, mm) for mm in ms]

# ============ 5. 图像 ============
outdir = "public/images/permutation-entropy-complexity"
os.makedirs(outdir, exist_ok=True)

# 图1 cover：三序列的归一化 PE 对比
fig, ax = plt.subplots(figsize=(11, 5.2))
labels = ["正弦(规则)", "逻辑斯蒂(混沌)", "白噪声(随机)"]
vals = [pe_sine, pe_logi, pe_noise]
colors = ["#1a9850", "#762a83", "#b2182b"]
bars = ax.bar(labels, vals, color=colors, width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
ax.axhline(1.0, color="gray", ls="--", lw=1.2, label="理论最大 PE=1（完全随机）")
ax.set_ylim(0, 1.15)
ax.set_ylabel("归一化排列熵 PE")
ax.set_title("排列熵把『复杂度』量成一个 0–1 的数：规则序列低、随机序列高",
             fontsize=13.5, fontweight="bold", color="#1f3a5f")
ax.legend(fontsize=9)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(f"{outdir}/cover.png", dpi=130)
plt.close(fig)

# 图2：序数模式频率分布（m=4，24 种模式）
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
for ax, cnt, title, c in zip(axes, [cnt_sine, cnt_logi, cnt_noise],
                             ["正弦：少数模式霸屏", "逻辑斯蒂：中等散布", "白噪声：近乎均匀"],
                             colors):
    ax.bar(range(len(cnt)), cnt, color=c)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("序数模式编号")
    ax.set_ylim(0, max(cnt_sine.max(), cnt_noise.max()) * 1.1)
axes[0].set_ylabel("模式出现频率")
fig.suptitle("排列熵的本质：把序列『切成几种升跌形状』再数频率——秩序=少数形状反复",
             fontsize=12.5, fontweight="bold", color="#1f3a5f")
fig.tight_layout()
fig.savefig(f"{outdir}/pattern_dist.png", dpi=130)
plt.close(fig)

# 图3：滑动窗 PE 实时追踪 regime
fig, ax1 = plt.subplots(figsize=(12, 5.2))
ax1.plot(np.arange(len(mixed)), mixed, color="#999999", lw=0.6, label="原始序列（正弦→噪声→正弦）")
ax1.set_ylabel("序列值", color="#666")
ax2 = ax1.twinx()
ax2.plot(np.arange(len(pe_mixed)), pe_mixed, color="#1a9850", lw=1.8, label="滑动窗 PE")
ax2.axhline(pe_sine, color="#1a9850", ls=":", lw=1, alpha=0.6)
ax2.axhline(pe_noise, color="#b2182b", ls=":", lw=1, alpha=0.6)
ax2.set_ylabel("滑动窗排列熵（窗口=200）", color="#1a9850")
ax2.set_ylim(0, 1.05)
ax1.axvline(seg, color="gray", ls="--", lw=1)
ax1.axvline(2 * seg, color="gray", ls="--", lw=1)
ax1.set_xlabel("时间")
ax1.set_title("排列熵实时追踪 regime：噪声段 PE 冲高到 0.9、正弦段回落到 0.3",
              fontsize=12.8, fontweight="bold", color="#1f3a5f")
fig.tight_layout()
fig.savefig(f"{outdir}/sliding_window.png", dpi=130)
plt.close(fig)

# 图4：PE 随嵌入维度 m 曲线（边界）
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.plot(ms, pe_m_sine, "o-", color="#1a9850", lw=1.8, label="正弦")
ax.plot(ms, pe_m_logi, "s-", color="#762a83", lw=1.8, label="逻辑斯蒂")
ax.plot(ms, pe_m_noise, "^-", color="#b2182b", lw=1.8, label="白噪声")
ax.set_xlabel("嵌入维度 m")
ax.set_ylabel("归一化排列熵 PE")
ax.set_title("诚实边界：m 越大模式种类 m! 爆炸，小样本下稀疏→估计不稳；m 太小漏掉时序结构",
             fontsize=12.3, fontweight="bold", color="#1f3a5f")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{outdir}/m_curve.png", dpi=130)
plt.close(fig)

# ============ 6. stats ============
stats = {
    "N": N, "m_pe": m_pe,
    "pe_sine": round(pe_sine, 4), "pe_logi": round(pe_logi, 4), "pe_noise": round(pe_noise, 4),
    "pe_vs_m": {
        "ms": ms,
        "sine": [round(float(v), 4) for v in pe_m_sine],
        "logi": [round(float(v), 4) for v in pe_m_logi],
        "noise": [round(float(v), 4) for v in pe_m_noise],
    },
    "sliding_window_pe_range": [round(float(np.nanmin(pe_mixed)), 4),
                                 round(float(np.nanmax(pe_mixed)), 4)],
}
with open(f"{outdir}/stats.json", "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print("Permutation entropy images written to", outdir)
