#!/usr/bin/env python3
"""
为文章「多重分形去趋势波动分析(MF-DFA)：用分形谱识别市场的多重尺度异象」(multifractal-dfa)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（自洽合成，仅用于演示方法；落地见文末路径）：
  * 用 MF-DFA(multifractal detrended fluctuation analysis, Kantelhardt et al. 2002)
    估计广义 Hurst 指数 h(q) 与多重分形谱 f(alpha)，把「单分形 vs 多分形」量化。
  * 构造两条对照序列：
      - 单分形基准：iid 高斯白噪声（H≈0.5），理论上 h(q) 应为常数、谱宽 Δalpha≈0；
      - 多分形序列：p 模型二项乘性级联(p-model binomial cascade)，天然带多重分形谱。
  * 用一段「平稳/动荡」区制切换波动序列演示多重分形宽度如何随波动率聚集增强，
    并以滚动窗口 Δalpha 作为危机强度的无分布探针。

注意：本模拟嵌入多分形结构以演示机制（与全库高阶文一致），真实市场数据的多重分形
来自波动率聚集 + 跳跃 + 微观结构噪声，文末已说明。
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
D = os.path.join(BASE, "multifractal-dfa")
os.makedirs(D, exist_ok=True)

C = {"eq": "#4C72B0", "bd": "#55A868", "gd": "#DD8452", "multi": "#C44E52",
     "mono": "#999999", "mkt": "#999999", "grid": "#DDDDDD", "thr": "#888888",
     "accent": "#8172B3", "warn": "#C44E52", "calm": "#55A868"}

rng = np.random.default_rng(20260713)


# ----------------------------------------------------------------------------
# 1) 序列生成
# ----------------------------------------------------------------------------
def pmodel_cascade(L, p):
    """二项乘性级联：从 2 个盒子开始递归分裂 L 次，返回长度 2**(L+1) 的（正）测度，天然多重分形。"""
    mu = np.ones(2)
    for _ in range(L):
        choose = rng.random(len(mu)) < 0.5
        left_w = np.where(choose, p, 1.0 - p)
        new = np.empty(len(mu) * 2)
        new[0::2] = mu * left_w
        new[1::2] = mu * (1.0 - left_w)
        mu = new
    return mu


# 单分形基准：白噪声（H=0.5）
N = 2 ** 16
x_mono = rng.standard_normal(N)
x_mono -= x_mono.mean()

# 多分形序列：p 模型级联（取增量作为收益型序列）
cascade = pmodel_cascade(15, 0.65)
x_multi = cascade.copy()
x_multi -= x_multi.mean()

# 区制切换波动序列（用于滚动危机探测演示）
T = 4000
vol_low, vol_high = 0.008, 0.028
state = np.zeros(T, dtype=int)
state[0] = 0
for t in range(1, T):
    # 低波动态黏性强、高波动态也容易持续（波动率聚集）
    p_same = 0.97 if state[t - 1] == 0 else 0.90
    state[t] = state[t - 1] if rng.random() < p_same else 1 - state[t - 1]
sig = np.where(state == 0, vol_low, vol_high)
x_regime = rng.standard_normal(T) * sig
x_regime -= x_regime.mean()

# ----------------------------------------------------------------------------
# 2) MF-DFA 核心
# ----------------------------------------------------------------------------
def mdfa(x, scales, q_list, order=1):
    """返回 {scale: {q: F_q(s)}} 与每个 q 的标度指数 h(q)。
    一阶去趋势用向量化 pinv 实现（避免逐段 polyfit 的循环开销）。"""
    x = np.asarray(x, float)
    N = len(x)
    y = np.cumsum(x - x.mean())           # 累积离差（去趋势前先减均值）
    rev = y[::-1]
    F = {}
    t = np.arange(max(scales))
    for s in scales:
        if s >= N:
            continue
        n = N // s
        segs = np.concatenate(
            [y[v * s:(v + 1) * s] for v in range(n)] +
            [rev[v * s:(v + 1) * s] for v in range(n)]
        ).reshape(2 * n, s)
        A = np.vstack([np.ones(s), np.arange(s)]).T          # (s, 2)
        beta = np.linalg.lstsq(A, segs.T, rcond=None)[0].T   # (2n, 2)
        fit = beta @ A.T                                     # (2n, s)
        var_arr = ((segs - fit) ** 2).mean(1) + 1e-12        # (2n,)
        fq = {}
        for q in q_list:
            if abs(q) < 1e-9:
                fq[q] = np.exp(0.5 * np.mean(np.log(var_arr)))
            else:
                fq[q] = (np.mean(var_arr ** (q / 2.0))) ** (1.0 / q)
        F[s] = fq
    # 对每个 q 做 log-log 回归得到 h(q)
    scales_arr = np.array(sorted(F.keys()), float)
    ls = np.log10(scales_arr)
    h = {}
    for q in q_list:
        fv = np.array([F[s][q] for s in scales_arr], float)
        lf = np.log10(fv)
        A = np.vstack([ls, np.ones_like(ls)]).T
        coef = np.linalg.lstsq(A, lf, rcond=None)[0]
        h[q] = coef[0]
    return F, h


SCALES = [32, 64, 128, 256, 512, 1024, 2048, 4096]
Q_LIST = list(range(-8, 9))          # -8..8
q_show = [-8, -4, -2, 0, 2, 4, 8]

F_multi, h_multi = mdfa(x_multi, SCALES, Q_LIST, order=1)
F_mono, h_mono = mdfa(x_mono, SCALES, Q_LIST, order=1)

# 多重分谱 f(alpha)
def spectrum(h, q_list):
    qs = np.array(sorted(q_list), float)
    hs = np.array([h[q] for q in qs])
    tau = qs * hs - 1.0
    # dtau/dq 用中心差分
    alpha = np.gradient(tau, qs)
    f = qs * alpha - tau
    return qs, alpha, f, tau


qs_m, alpha_m, f_m, tau_m = spectrum(h_multi, Q_LIST)
qs_o, alpha_o, f_o, tau_o = spectrum(h_mono, Q_LIST)

# 谱宽
width_multi = float(alpha_m.max() - alpha_m.min())
width_mono = float(alpha_o.max() - alpha_o.min())
H0_multi = h_multi[0]
H0_mono = h_mono[0]
h_span_multi = h_multi[8] - h_multi[-8]      # h(+8)-h(-8)：多分形越强差值越大
h_span_mono = h_mono[8] - h_mono[-8]

# ----------------------------------------------------------------------------
# 3) 滚动窗口 Δalpha（区制切换序列）
# ----------------------------------------------------------------------------
def rolling_width(x, win, step, scales, q_list):
    out = []
    centers = []
    L = len(x)
    s = max(scales)
    for start in range(0, L - win + 1, step):
        seg = x[start:start + win]
        if len(seg) <= s * 4:
            continue
        _, h = mdfa(seg, scales, q_list, order=1)
        qs = np.array(sorted(q_list), float)
        hs = np.array([h[q] for q in qs])
        tau = qs * hs - 1.0
        alpha = np.gradient(tau, qs)
        w = alpha.max() - alpha.min()
        out.append(w)
        centers.append(start + win // 2)
    return np.array(centers), np.array(out)


ROLL_SCALES = [25, 50, 100, 200, 400]
ROLL_Q = list(range(-4, 5))
centers, widths = rolling_width(x_regime, win=2000, step=500,
                                scales=ROLL_SCALES, q_list=ROLL_Q)
# 把窗口中心映射到「该窗口内高波动占比」作为对照
win = 1000
high_frac = np.array([state[c - win // 2:c + win // 2].mean() for c in centers])

# ----------------------------------------------------------------------------
# 打印关键数字（供正文嵌入）
# ----------------------------------------------------------------------------
print("===== MF-DFA KEY NUMBERS =====")
print(f"N={N}  cascade p=0.65")
print(f"[mono white noise] H(q=0)={H0_mono:.3f}  h(+8)-h(-8)={h_span_mono:.3f}  spectrum width Δα={width_mono:.3f}")
print(f"[multifractal cascade] H(q=0)={H0_multi:.3f}  h(+8)-h(-8)={h_span_multi:.3f}  spectrum width Δα={width_multi:.3f}")
print(f"cascade alpha range = [{alpha_m.min():.3f}, {alpha_m.max():.3f}]  peak f(alpha)={f_m.max():.3f}")
print(f"rolling Δα mean={widths.mean():.3f}  min={widths.min():.3f}  max={widths.max():.3f}")
print(f"rolling Δα vs high-vol fraction corr = {np.corrcoef(high_frac, widths)[0,1]:.3f}")
print(f"rolling Δα at calmest window={widths.min():.3f}  at most-turbulent window={widths.max():.3f}")

# ============================================================================
# 图 1：F_q(s) 标度曲线（多分形 vs 单分形）
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
scales_arr = np.array(sorted(F_multi.keys()), float)
for ax, Fh, title, col in [
    (axes[0], (F_multi, h_multi), "多分形序列（p 模型级联）", C["multi"]),
    (axes[1], (F_mono, h_mono), "单分形基准（白噪声 H≈0.5）", C["mono"])]:
    F, h = Fh
    for q in q_show:
        fv = np.array([F[s][q] for s in scales_arr])
        ax.plot(np.log10(scales_arr), np.log10(fv),
                color=col if q == 0 else plt.cm.coolwarm((q + 8) / 16.0),
                lw=2.0, marker="o", ms=3,
                label=f"q={q}")
    ax.set_xlabel("log₁₀(尺度 s)"); ax.set_ylabel("log₁₀ F_q(s)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, color=C["grid"], lw=0.6)
    if q == 0:
        ax.legend(fontsize=7, ncol=2, loc="upper left")
fig.suptitle("MF-DFA 标度曲线：多分形序列的 F_q(s) 斜率随 q 明显变化，单分形序列近乎平行",
             fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(D, "mdfa_scaling.png"), dpi=130)
plt.close()

# ============================================================================
# 图 2：广义 Hurst 指数 h(q)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5.0))
qarr = np.array(Q_LIST)
ax.plot(qarr, [h_mono[q] for q in qarr], "o-", color=C["mono"], lw=2.0,
        label=f"白噪声（单分形）：h(q)≈{H0_mono:.2f} 近平")
ax.plot(qarr, [h_multi[q] for q in qarr], "s-", color=C["multi"], lw=2.0,
        label=f"p 模型级联（多分形）：h(q) 随 q 弯折，跨度 {h_span_multi:+.2f}")
ax.axhline(0.5, color=C["thr"], ls="--", lw=1.2)
ax.set_xlabel("q（矩阶）"); ax.set_ylabel("广义 Hurst 指数 h(q)")
ax.set_title(f"广义 Hurst 指数 h(q)：多分形越强，h(q) 对 q 的依赖越明显（跨度 {h_span_multi:+.2f}）")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "mdfa_hq.png"), dpi=130)
plt.close()

# ============================================================================
# 图 3：多重分形谱 f(alpha)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5.0))
ax.plot(alpha_o, f_o, "o-", color=C["mono"], lw=2.0,
        label=f"白噪声：Δα={width_mono:.3f}（退化成一点）")
ax.plot(alpha_m, f_m, "s-", color=C["multi"], lw=2.2,
        label=f"p 模型级联：Δα={width_multi:.3f}（宽谱 = 多分形）")
ax.scatter([alpha_m.min()], [f_m[np.argmin(alpha_m)]], color=C["warn"], zorder=5)
ax.scatter([alpha_m.max()], [f_m[np.argmax(alpha_m)]], color=C["accent"], zorder=5)
ax.annotate(f"Δα={width_multi:.2f}", xy=(alpha_m.mean(), f_m.max()),
            xytext=(alpha_m.mean(), f_m.max() + 0.05), ha="center", fontsize=9, color=C["multi"])
ax.set_xlabel("奇异指数 α"); ax.set_ylabel("分形维数 f(α)")
ax.set_title("多重分形谱 f(α)：谱越宽，序列的奇异结构越丰富（危机/跳跃越多）")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "mdfa_spectrum.png"), dpi=130)
plt.close()

# ============================================================================
# 图 4：滚动 Δα 作为危机/波动聚集探针
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5.0))
ax.plot(centers, widths, "o-", color=C["multi"], lw=1.8, label="滚动窗口多重分形宽度 Δα")
ax.fill_between(centers, 0, widths, color=C["multi"], alpha=0.12)
ax2 = ax.twinx()
ax2.plot(centers, high_frac, color=C["warn"], lw=1.4, ls="--",
         label="窗口内高波动态占比")
ax.set_xlabel("时间（日，窗口中心）"); ax.set_ylabel("Δα（谱宽）", color=C["multi"])
ax2.set_ylabel("高波动态占比", color=C["warn"])
ax.set_title(f"滚动 Δα 随波动率聚集增强：与高波动占比相关 {np.corrcoef(high_frac, widths)[0,1]:.2f}")
ax.tick_params(axis='y', labelcolor=C["multi"])
ax2.tick_params(axis='y', labelcolor=C["warn"])
ax.grid(True, color=C["grid"], lw=0.6)
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(D, "mdfa_rolling.png"), dpi=130)
plt.close()

print("IMAGES WRITTEN:", sorted(os.listdir(D)))
