# -*- coding: utf-8 -*-
"""多重分形 DFA 实验：MF-DFA 从零实现 + 单分形/多重分形对照 + regime 应用"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)
OUT = "/Users/halo/workspace/astro-blog/public/images/multifractal-dfa-analysis"
os.makedirs(OUT, exist_ok=True)

# ---------- MF-DFA 核心 ----------
def mfdfa(x, scales, qs, m=1):
    """x: 序列(收益)，scales: 窗口长度数组，qs: 阶数数组，m: 去趋势多项式阶
    返回 Fq: (len(qs), len(scales))"""
    y = np.cumsum(x - x.mean())  # profile
    N = len(y)
    Fq = np.zeros((len(qs), len(scales)))
    for si, s in enumerate(scales):
        ns = N // s
        # 正向 + 反向切段，共 2ns 段
        F2 = []
        t = np.arange(s)
        for seg in range(ns):
            for yy in (y[seg * s:(seg + 1) * s], y[N - (seg + 1) * s:N - seg * s]):
                coef = np.polyfit(t, yy, m)
                res = yy - np.polyval(coef, t)
                F2.append((res ** 2).mean())
        F2 = np.array(F2)
        for qi, q in enumerate(qs):
            if abs(q) < 1e-9:
                Fq[qi, si] = np.exp(0.5 * np.log(F2 + 1e-30).mean())
            else:
                Fq[qi, si] = (np.mean(F2 ** (q / 2))) ** (1.0 / q)
    return Fq

def hurst_q(Fq, scales, qs):
    """对每个 q 拟合 log F_q(s) ~ h(q) log s"""
    ls = np.log(scales)
    hq = np.zeros(len(qs))
    for qi in range(len(qs)):
        hq[qi] = np.polyfit(ls, np.log(Fq[qi] + 1e-30), 1)[0]
    return hq

def spectrum(hq, qs):
    """Legendre 变换: tau(q)=q*h(q)-1; alpha=dtau/dq; f=q*alpha-tau"""
    tau = qs * hq - 1
    alpha = np.gradient(tau, qs)
    f = qs * alpha - tau
    return alpha, f

# ---------- 三类合成序列 ----------
N = 16384

def fgn(H, n, seed=0):
    """分数高斯噪声：谱方法近似"""
    r = np.random.default_rng(seed)
    # Davies-Harte 简化：用 FFT 谱形状 (功率 ~ f^{-(2H-1)})
    f = np.fft.rfftfreq(n, 1.0)
    f[0] = f[1]
    amp = f ** (-(2 * H - 1) / 2)
    phase = r.uniform(0, 2 * np.pi, len(f))
    spec = amp * np.exp(1j * phase)
    x = np.fft.irfft(spec, n)
    return (x - x.mean()) / x.std()

# 1) 白噪声 (单分形, h=0.5)
x_wn = rng.standard_normal(N)
# 2) 长记忆单分形 fGn H=0.7
x_fgn = fgn(0.7, N, seed=3)
# 3) 多重分形：二项级联波动 * 高斯
def binomial_cascade(n_levels, m0=0.6, seed=5):
    r = np.random.default_rng(seed)
    w = np.array([1.0])
    for _ in range(n_levels):
        left = np.where(r.random(len(w)) < 0.5, m0, 1 - m0)
        w = np.repeat(w, 2) * np.column_stack([left, 1 - left]).ravel() * 2
    return w
vol = binomial_cascade(14, m0=0.75, seed=5)  # 2^14 = 16384
x_mf = np.sqrt(vol) * rng.standard_normal(N)
x_mf = (x_mf - x_mf.mean()) / x_mf.std()

scales = np.unique(np.logspace(np.log10(16), np.log10(N // 8), 20).astype(int))
qs = np.linspace(-5, 5, 21)

series = {"白噪声": x_wn, "长记忆 fGn (H=0.7)": x_fgn, "二项级联多重分形": x_mf}
results = {}
for name, x in series.items():
    Fq = mfdfa(x, scales, qs)
    hq = hurst_q(Fq, scales, qs)
    alpha, f = spectrum(hq, qs)
    width = alpha.max() - alpha.min()
    results[name] = dict(Fq=Fq, hq=hq, alpha=alpha, f=f, width=width)
    print(f"{name}: h(2)={hq[np.argmin(np.abs(qs-2))]:.3f}  h(-4)={hq[np.argmin(np.abs(qs+4))]:.3f}  h(4)={hq[np.argmin(np.abs(qs-4))]:.3f}  谱宽Δα={width:.3f}")

# ---------- 图1：三类序列 ----------
fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
for ax, (name, x) in zip(axes, series.items()):
    ax.plot(x[:4000], lw=0.5)
    ax.set_ylabel(name, fontsize=9)
    ax.grid(alpha=0.3)
axes[0].set_title("三类合成序列（前 4000 点）：肉眼看相似，记忆结构完全不同")
axes[-1].set_xlabel("时间步")
plt.tight_layout()
plt.savefig(f"{OUT}/three_series.png", dpi=130)
plt.close()

# ---------- 图2：h(q) 曲线 ----------
fig, ax = plt.subplots(figsize=(8, 4.8))
for name, res in results.items():
    ax.plot(qs, res["hq"], "o-", ms=4, label=f"{name} (Δα={res['width']:.2f})")
ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6)
ax.set_xlabel("阶数 q")
ax.set_ylabel("广义 Hurst 指数 h(q)")
ax.set_title("h(q) 曲线：单分形是水平线，多重分形随 q 下弯")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/hq_curves.png", dpi=130)
plt.close()

# ---------- 图3：多重分形谱 f(alpha) ----------
fig, ax = plt.subplots(figsize=(8, 4.8))
for name, res in results.items():
    ax.plot(res["alpha"], res["f"], "o-", ms=4, label=f"{name} (Δα={res['width']:.2f})")
ax.set_xlabel("奇异性强度 α")
ax.set_ylabel("f(α)")
ax.set_title("多重分形谱：谱宽 Δα 度量『不同尺度记忆结构的丰富度』")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/spectrum.png", dpi=130)
plt.close()

# ---------- 应用实验：滚动谱宽 vs 波动 regime ----------
# 构造两段 regime 序列：前半段平静(白噪声)，后半段级联波动聚集
Nr = 8192
x1 = rng.standard_normal(Nr) * 0.8
vol2 = binomial_cascade(13, m0=0.7, seed=11)
x2 = np.sqrt(vol2) * rng.standard_normal(Nr)
x2 = x2 / x2.std() * 1.0
x_regime = np.concatenate([x1, x2])

win = 2048
step = 256
sc2 = np.unique(np.logspace(np.log10(16), np.log10(win // 8), 12).astype(int))
qs2 = np.linspace(-4, 4, 9)
centers, widths = [], []
for start in range(0, len(x_regime) - win + 1, step):
    seg = x_regime[start:start + win]
    Fq = mfdfa(seg, sc2, qs2)
    hq = hurst_q(Fq, sc2, qs2)
    a, f = spectrum(hq, qs2)
    centers.append(start + win // 2)
    widths.append(a.max() - a.min())
centers = np.array(centers); widths = np.array(widths)

w1 = widths[centers < Nr].mean()
w2 = widths[centers >= Nr].mean()
print(f"滚动谱宽: 平静段均值={w1:.3f}  级联段均值={w2:.3f}")

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
axes[0].plot(x_regime, lw=0.4)
axes[0].axvline(Nr, color="red", ls="--", lw=1.5)
axes[0].set_ylabel("收益")
axes[0].set_title("regime 切换实验：前半段白噪声，后半段多重分形级联（红线为切换点）")
axes[0].grid(alpha=0.3)
axes[1].plot(centers, widths, "o-", color="#d62828", ms=4)
axes[1].axvline(Nr, color="red", ls="--", lw=1.5)
axes[1].axhline(w1, color="gray", ls=":", lw=1)
axes[1].set_ylabel("滚动谱宽 Δα")
axes[1].set_xlabel("时间步")
axes[1].set_title(f"滚动窗口(2048)谱宽：平静段均值 {w1:.2f} → 级联段均值 {w2:.2f}")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/rolling_width.png", dpi=130)
plt.close()

# ---------- 陷阱实验：打乱序列后的谱宽（区分“分布厚尾”与“时序相关”来源） ----------
x_shuf = x_mf.copy()
rng.shuffle(x_shuf)
Fq_s = mfdfa(x_shuf, scales, qs)
hq_s = hurst_q(Fq_s, scales, qs)
a_s, f_s = spectrum(hq_s, qs)
w_orig = results["二项级联多重分形"]["width"]
w_shuf = a_s.max() - a_s.min()
print(f"级联序列谱宽: 原始={w_orig:.3f}  打乱后={w_shuf:.3f}")

fig, ax = plt.subplots(figsize=(8, 4.5))
res = results["二项级联多重分形"]
ax.plot(res["alpha"], res["f"], "o-", label=f"原始级联序列 (Δα={w_orig:.2f})")
ax.plot(a_s, f_s, "s--", label=f"随机打乱后 (Δα={w_shuf:.2f})")
ax.set_xlabel("奇异性强度 α")
ax.set_ylabel("f(α)")
ax.set_title("打乱检验：打乱摧毁时序相关，剩余谱宽来自收益分布厚尾本身")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/shuffle_test.png", dpi=130)
plt.close()

print("charts saved:", os.listdir(OUT))
