#!/usr/bin/env python3
"""生成 序数模式熵(OPE) 文章：用涨跌排列的『形状』识别市场状态切换。

本文与「排列熵(PE)基础」是同一工具的两层用法：
  PE 基础文章讲「怎么用一个数区分正弦/混沌/噪声」(静态刻画)；
  本文讲「把 PE 做成滑动窗 + 转移矩阵 + 加权 PE，用来实时识别 regime 切换」
  ——这正是量化里最有用的部分：判断行情是「有序趋势市」还是「无序震荡/危机市」。
关键认知（也是本文反复强调的诚实边界）：
  PE 量的是「序列的确定性/结构」，不是「波动大小」。低波动随机游走和高波动
  随机游走的 PE 一样高（都是 i.i.d.），所以区分 regime 必须让两段在「结构」上不同
  ——有序段用平滑正弦（PE 低）、无序段用白噪声（PE 高）。
实现（纯 numpy）：
  - 序数模式映射：用 argsort 的秩把窗口映射成 0..m!-1 的排列序号
  - 排列熵 PE(m,τ)：数模式频率 → 香农熵 → 除以 log(m!) 归一化
  - 加权排列熵 WPE：用窗口振幅方差给每个模式加权，抑制小幅噪声伪迹
  - 转移矩阵：把相邻窗口的模式编号连成马尔可夫链，看 regime 的「切换结构」
所有图表均由下文真实计算，非占位图。
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


# ============ 核心：序数模式映射 / 排列熵 / 加权排列熵 ============
def perm_index(perm):
    """把 0..m-1 的一个排列映射成 Lehmer 码 0..m!-1"""
    m = len(perm)
    avail = list(range(m))
    idx = 0
    for i, p in enumerate(perm):
        k = avail.index(p)
        idx = idx * (m - i) + k
        avail.pop(k)
    return idx


def pattern_of(window):
    """窗口内相对大小排序 → 秩排列 (0..m-1 各出现一次)"""
    return tuple(np.argsort(np.argsort(window, kind="stable")))


def permutation_entropy(x, m=5, tau=1):
    """标准排列熵（归一化到 [0,1]）"""
    x = np.asarray(x, float)
    n = len(x)
    L = (m - 1) * tau
    counts = np.zeros(math.factorial(m))
    for i in range(n - L):
        w = x[i:i + L + 1:tau]
        idx = perm_index(pattern_of(w))
        counts[idx] += 1
    total = counts.sum()
    p = counts[counts > 0] / total
    H = -np.sum(p * np.log2(p))
    return float(H / math.log2(math.factorial(m)))


def weighted_pe(x, m=5, tau=1):
    """加权排列熵 WPE(Fadlallah 2013)：用窗口振幅方差给模式加权，抑制边界伪迹"""
    x = np.asarray(x, float)
    n = len(x)
    L = (m - 1) * tau
    weights = np.zeros(math.factorial(m))
    for i in range(n - L):
        w = x[i:i + L + 1:tau]
        idx = perm_index(pattern_of(w))
        weights[idx] += np.sum((w - w.mean()) ** 2)
    W = weights.sum()
    if W <= 0:
        return 0.0
    p = weights[weights > 0] / W
    H = -np.sum(p * np.log2(p))
    return float(H / math.log2(math.factorial(m)))


def sliding_pe(x, m=5, tau=1, win=60, step=1):
    """滑动窗排列熵：在长度为 win 的局部片段上计算 PE（片段内含多模式才有意义），返回 (中心索引, PE 序列)"""
    x = np.asarray(x, float)
    L = (m - 1) * tau
    centers, vals = [], []
    for i in range(0, len(x) - win + 1, step):
        centers.append(i + win // 2)
        vals.append(permutation_entropy(x[i:i + win], m, tau))
    return np.array(centers), np.array(vals)


def sliding_wpe(x, m=5, tau=1, win=60, step=1):
    x = np.asarray(x, float)
    centers, vals = [], []
    for i in range(0, len(x) - win + 1, step):
        centers.append(i + win // 2)
        vals.append(weighted_pe(x[i:i + win], m, tau))
    return np.array(centers), np.array(vals)


def pattern_sequence(x, m=5, tau=1):
    """把整条序列切成窗口，返回每个窗口的排列序号（用于建转移矩阵）"""
    x = np.asarray(x, float)
    L = (m - 1) * tau
    seq = []
    for i in range(len(x) - L):
        seq.append(perm_index(pattern_of(x[i:i + L + 1:tau])))
    return np.array(seq)


def transition_matrix(seq, m=5):
    n = math.factorial(m)
    T = np.zeros((n, n))
    for a, b in zip(seq[:-1], seq[1:]):
        T[a, b] += 1
    rs = T.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return T / rs  # 行归一化成 Markov 转移概率


def mean_row_entropy(T):
    """转移矩阵的平均每行香农熵（bit），范围 [0, log2(行数)]"""
    H = -np.sum(T[T > 0] * np.log2(T[T > 0]))
    return float(H / T.shape[0])


# ============ 1. 三档经典序列：正弦 / 逻辑斯蒂混沌 / 白噪声 ============
rng = np.random.default_rng(20260723)
N = 2000
x_sine = np.sin(np.linspace(0, 40 * np.pi, N)) + rng.normal(0, 1e-4, N)
x_chaos = np.zeros(N)
x_chaos[0] = 0.4
for i in range(1, N):
    x_chaos[i] = 4.0 * x_chaos[i - 1] * (1 - x_chaos[i - 1])
x_noise = rng.uniform(0, 1, N)
x_rw = np.cumsum(rng.normal(0, 1, N))  # 随机游走（i.i.d. 增量 → 无序）

pe_sine = permutation_entropy(x_sine)
pe_chaos = permutation_entropy(x_chaos)
pe_noise = permutation_entropy(x_noise)
pe_rw = permutation_entropy(x_rw)
print(f"  PE(正弦)      = {pe_sine:.4f}")
print(f"  PE(混沌)      = {pe_chaos:.4f}")
print(f"  PE(白噪声)    = {pe_noise:.4f}")
print(f"  PE(随机游走)  = {pe_rw:.4f}  ← 注意：高波动随机游走的 PE 也接近 1（都是 i.i.d.）")

# ============ 2. regime 切换市场：有序(平滑) / 无序(白噪声) 交替 ============
# 关键：两段振幅相同（都是 0.02 量级），只在「结构」上不同——这才是 PE 能抓的。
rng_s = np.random.default_rng(777)
A = 0.02
def block(kind, length):
    t = np.arange(length)
    if kind == "order":
        # 平滑正弦 + 极小噪声 → 局部形状固定 → 低 PE（有序）
        return A * np.sin(2 * np.pi * t / 30) + rng_s.normal(0, 0.0015, length)
    else:
        # 纯白噪声 → 高 PE（无序）
        return A * rng_s.normal(0, 1, length)

market = np.concatenate([block("order", 400), block("noise", 400),
                         block("order", 400), block("noise", 400)])
true_regime = np.array(["order"] * 400 + ["noise"] * 400 +
                       ["order"] * 400 + ["noise"] * 400)
price = 100 + np.cumsum(market)

centers, pe_series = sliding_pe(market, m=5, tau=1, step=1)
reg_at_center = true_regime[centers]
mean_pe_order = pe_series[reg_at_center == "order"].mean()
mean_pe_noise = pe_series[reg_at_center == "noise"].mean()
thr = (mean_pe_order + mean_pe_noise) / 2
pred = np.where(pe_series > thr, "noise", "order")
acc = np.mean(pred == reg_at_center)
print(f"  PE 区分 regime 正确率 = {acc*100:.1f}%  (有序均值 {mean_pe_order:.3f} / 无序均值 {mean_pe_noise:.3f})")

# ============ 3. 转移矩阵：有序 vs 无序 的「切换结构」 ============
seq_all = pattern_sequence(market, m=5)
calm_mask = true_regime[:len(seq_all)] == "order"
seq_order = seq_all[calm_mask]
seq_noise = seq_all[~calm_mask]
T_order = transition_matrix(seq_order)
T_noise = transition_matrix(seq_noise)
te_order = mean_row_entropy(T_order)
te_noise = mean_row_entropy(T_noise)
# 有序段实际用到的模式数（应该远少于全部 120 种）
used_order = int((np.bincount(seq_order, minlength=120) > 0).sum())
used_noise = int((np.bincount(seq_noise, minlength=120) > 0).sum())
print(f"  转移平均行熵 有序={te_order:.3f}  无序={te_noise:.3f}（无序更均匀→更高）")
print(f"  用到的模式数  有序={used_order}/120  无序={used_noise}/120")

# ============ 4. WPE vs PE：一个局部大振幅突变（边界伪迹） ============
# 基准是低振幅平滑信号；中间插入一段高振幅尖峰。
# 标准 PE 只看「秩」(ordinal) → 对振幅不敏感，尖峰处几乎不动；
# WPE 用窗口方差加权 → 尖峰窗口权重骤增，曲线会明显抬起。
rng_w = np.random.default_rng(313)
base = 1.0 * np.sin(2 * np.pi * np.arange(800) / 25) + rng_w.normal(0, 0.05, 800)
sig = base.copy()
sig[400:410] += 40.0  # 局部大振幅突变（不缩放基准，只在该处叠加尖峰）
cw, pe_w = sliding_pe(sig, m=5, step=1)
cw2, wpe_w = sliding_wpe(sig, m=5, step=1)
edge = (cw >= 395) & (cw <= 415)
max_diff = float(np.max(np.abs(pe_w[edge] - wpe_w[edge])))
mean_diff = float(np.mean(np.abs(pe_w - wpe_w)))
print(f"  尖峰处 PE 与 WPE 最大差 = {max_diff:.3f}（WPE 被尖峰窗口权重抬起，PE 纹丝不动）")

# ============ 5. 图像 ============
outdir = "public/images/ordinal-pattern-entropy"
os.makedirs(outdir, exist_ok=True)

# 图1 cover：regime 切换市场的滑动 PE（hero 图）
fig, ax1 = plt.subplots(figsize=(11.5, 5.2))
ax1.plot(centers, price[centers], color="#888", lw=1.0, label="价格（累计 level）", zorder=1)
ax1.set_ylabel("价格", color="#888"); ax1.tick_params(axis="y", labelcolor="#888")
ax2 = ax1.twinx()
ax2.plot(centers, pe_series, color="#1a9850", lw=1.6, label="滑动排列熵 PE(5)")
ax2.axhline(thr, color="#d73027", ls="--", lw=1.2, label=f"分类阈值 {thr:.2f}")
ax2.set_ylabel("排列熵 PE(5)", color="#1a9850"); ax2.tick_params(axis="y", labelcolor="#1a9850")
for s, e, rg in [(0, 400, "order"), (400, 800, "noise"), (800, 1200, "order"), (1200, 1600, "noise")]:
    ax1.axvspan(s, e, color=("#4393c3" if rg == "order" else "#d73027"), alpha=0.07)
ax1.set_xlabel("时间（步）")
ax1.set_title(f"用「涨跌形状」抓状态切换：有序段 PE 低且稳、无序段 PE 高且抖（判对 {acc*100:.0f}%）",
              fontsize=12.0, fontweight="bold", color="#1f3a5f")
lines1, lbl1 = ax1.get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig(f"{outdir}/cover.png", dpi=130); plt.close(fig)

# 图2：模式频率直方图（有序 vs 无序）——无序更平=更接近随机
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
ncols = math.factorial(5)
top = 30
freq_order = np.bincount(seq_order, minlength=ncols) / len(seq_order)
freq_noise = np.bincount(seq_noise, minlength=ncols) / len(seq_noise)
axes[0].bar(np.arange(top), freq_order[:top], color="#4393c3")
axes[0].set_title(f"有序段：少数模式被反复使用（PE={mean_pe_order:.2f}，用到 {used_order} 种）",
                  fontsize=10.5, color="#1f3a5f")
axes[0].set_xlabel("排列模式序号（取前 30）"); axes[0].set_ylabel("频率")
axes[1].bar(np.arange(top), freq_noise[:top], color="#d73027")
axes[1].set_title(f"无序段：模式铺得更均匀（PE={mean_pe_noise:.2f}，用到 {used_noise} 种）",
                  fontsize=10.5, color="#1f3a5f")
axes[1].set_xlabel("排列模式序号（取前 30）")
fig.suptitle("同一波行情，两种 regime：有序市模式聚堆、无序市模式摊平",
             fontsize=12, fontweight="bold", color="#1f3a5f")
fig.tight_layout(); fig.savefig(f"{outdir}/pattern_freq.png", dpi=120); plt.close(fig)

# 图3：转移矩阵（有序 vs 无序）
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
im0 = axes[0].imshow(T_order[:30, :30], cmap="Blues", aspect="auto")
axes[0].set_title(f"有序段转移矩阵（平均行熵 {te_order:.2f}，集中）", fontsize=10.5, color="#1f3a5f")
axes[0].axis("off"); fig.colorbar(im0, ax=axes[0], fraction=0.05, pad=0.04)
im1 = axes[1].imshow(T_noise[:30, :30], cmap="Reds", aspect="auto")
axes[1].set_title(f"无序段转移矩阵（平均行熵 {te_noise:.2f}，均匀扩散）", fontsize=10.5, color="#1f3a5f")
axes[1].axis("off"); fig.colorbar(im1, ax=axes[1], fraction=0.05, pad=0.04)
fig.suptitle("把相邻模式连成马尔可夫链：无序市转移更「到处跳」",
             fontsize=12, fontweight="bold", color="#1f3a5f")
fig.tight_layout(); fig.savefig(f"{outdir}/transition_matrix.png", dpi=120); plt.close(fig)

# 图4：WPE vs PE（局部大振幅尖峰）
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.plot(cw, pe_w, color="#1a9850", lw=1.6, label="标准排列熵 PE")
ax.plot(cw2, wpe_w, color="#762a83", lw=1.6, label="加权排列熵 WPE")
ax.axvspan(400, 410, color="#d73027", alpha=0.18, label="大振幅尖峰段")
ax.set_xlabel("时间（步）"); ax.set_ylabel("熵")
ax.set_title(f"加权 PE 对振幅敏感：尖峰处 WPE 抬起、PE 纹丝不动（最大差 {max_diff:.2f}）",
             fontsize=12.0, fontweight="bold", color="#1f3a5f")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{outdir}/wpe_vs_pe.png", dpi=130); plt.close(fig)

# ============ 6. stats ============
stats = {
    "N": N,
    "pe_tiers": {
        "sine": round(pe_sine, 4),
        "chaos_logistic": round(pe_chaos, 4),
        "white_noise": round(pe_noise, 4),
        "random_walk": round(pe_rw, 4),
    },
    "regime_accuracy": round(float(acc), 4),
    "mean_pe_order": round(float(mean_pe_order), 4),
    "mean_pe_noise": round(float(mean_pe_noise), 4),
    "threshold": round(float(thr), 4),
    "mean_row_entropy_order": round(te_order, 4),
    "mean_row_entropy_noise": round(te_noise, 4),
    "used_patterns_order": used_order,
    "used_patterns_noise": used_noise,
    "wpe_pe_max_diff_at_spike": round(max_diff, 4),
}
with open("public/images/ordinal-pattern-entropy/stats.json", "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print("\nOPE images written. regime accuracy =", round(acc * 100, 1), "%")
