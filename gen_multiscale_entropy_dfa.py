#!/usr/bin/env python3
"""生成「多尺度熵与 DFA」文章配图与 stats.json。

核心思想：单尺度样本熵(SampEn) 只看一个时间尺度，会把「白噪声」误判成
最复杂的序列——其实白噪声的复杂度只存在于最细尺度。多尺度熵(MSE) 用
粗粒化(coarse-graining) 把序列逐尺度「平均掉」，分别在每个尺度上算 SampEn，
把不同时间尺度的不规则性分开。

DFA(去趋势波动分析) 则给出「长程相关性」的标度指数 α：
  α≈0.5 近似白噪声(无记忆) / α>0.5 持续相关(趋势/分形) / α<0.5 反持续。

所有图与数字均由本脚本真实计算(numpy + matplotlib)，非占位图。
"""
import numpy as np
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False
fm._load_fontmanager()

SLUG = "multiscale-entropy-dfa"
OUT = os.path.join("public", "images", SLUG)
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260723)


# ============ 1. 样本熵 SampEn（纯 numpy） ============
def sampen(x, m=2, r=None):
    """Sample Entropy (Richman & Moorman 2000)：m 维模板向量里，
    「再延长 1 步仍匹配」的比例的对数负号。值越大越不规则/越复杂。
    r 为容忍半径，默认 0.15·std（抗噪、稳定）。"""
    x = np.asarray(x, float)
    if r is None:
        r = 0.15 * np.std(x)
    if np.std(x) == 0:
        return 0.0
    N = len(x)

    def count_m(mm):
        X = np.array([x[i:i + mm] for i in range(N - mm + 1)])
        cm = 0
        for i in range(len(X)):
            diff = np.max(np.abs(X - X[i]), axis=1)
            cm += np.sum(diff < r) - 1  # 排除与自身比较
        return cm

    B = count_m(m)        # 长度为 m 的匹配对数
    A = count_m(m + 1)    # 长度为 m+1 的匹配对数
    if B <= 0 or A <= 0:
        return 0.0
    return float(-np.log(A / B))


# ============ 2. 粗粒化 + 多尺度熵 MSE ============
def coarse_grain(x, tau):
    """粗粒化：把序列按窗口 τ 不重叠平均，序列长度变为 N//τ。
    τ 越大，越「抹掉」细尺度波动，暴露粗尺度结构。"""
    x = np.asarray(x, float)
    n = len(x) // tau
    if n == 0:
        return np.array([])
    y = x[: n * tau].reshape(n, tau)
    return y.mean(axis=1)


def mse(x, scales, m=2, r=None):
    """多尺度熵 MSE (Costa et al. 2002)：关键是用**固定**容忍半径
    r = 0.15·std(原始序列)，跨所有尺度不变。否则粗粒化后序列方差变小、
    r 跟着变小，会把样本熵人为推高，掩盖白噪声「粗尺度塌缩」的真特征。"""
    x = np.asarray(x, float)
    if r is None:
        r = 0.15 * np.std(x)
    out = []
    for tau in scales:
        cg = coarse_grain(x, tau)
        if len(cg) < m + 2:
            out.append(np.nan)
        else:
            out.append(sampen(cg, m=m, r=r))
    return np.array(out)


# ============ 3. DFA 去趋势波动分析（纯 numpy） ============
def dfa(x, scales=None):
    """Detrended Fluctuation Analysis (Peng 1994)：
    累计和 → 划窗线性去趋势 → 残差 RMS 得 F(s)，
    对 log F(s) ~ α·log s 做回归得标度指数 α。"""
    y = np.cumsum(np.asarray(x, float) - np.mean(x))
    N = len(y)
    if scales is None:
        scales = np.unique(np.round(np.logspace(np.log10(10),
                                                 np.log10(N // 4), 18))).astype(int)
    Fs = []
    used = []
    for s in scales:
        n_parts = N // s
        if n_parts < 2:
            continue
        fv = []
        for p in range(n_parts):
            seg = y[p * s:(p + 1) * s]
            t = np.arange(s)
            coef = np.polyfit(t, seg, 1)
            resid = seg - np.polyval(coef, t)
            fv.append(np.sqrt(np.mean(resid ** 2)))
        Fs.append(np.mean(fv))
        used.append(s)
    Fs = np.array(Fs)
    used = np.array(used)
    alpha = float(np.polyfit(np.log(used), np.log(Fs), 1)[0])
    return used, Fs, alpha


# ============ 合成三类序列 ============
N = 2400

# 白噪声：最细尺度最复杂，粗尺度迅速塌成规则
white = rng.normal(0, 1, N)

# 1/f 粉红噪声：多尺度都复杂（分形/长程相关）——用谱滤波生成
def pink_noise(n, exp=1.0):
    f = np.fft.rfft(rng.normal(size=n))
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1e-6
    f = f * freqs ** (-exp / 2.0)
    s = np.fft.irfft(f, n=n)
    return (s - s.mean()) / s.std()

pink = pink_noise(N, exp=1.0)

# 周期 + 噪声混合：单一主导频率，粗粒化后规则性暴露
t = np.arange(N)
periodic = 0.7 * np.sin(2 * np.pi * t / 47.0) + 0.3 * rng.normal(0, 1, N)
periodic = (periodic - periodic.mean()) / periodic.std()

series = {"白噪声": white, "1/f 粉红噪声": pink, "周期+噪声": periodic}
colors = {"白噪声": "#4C72B0", "1/f 粉红噪声": "#DD8452", "周期+噪声": "#55A868"}

# ============ 图1：三类序列原貌 ============
fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True)
for ax, (name, s) in zip(axes, series.items()):
    ax.plot(s[:600], color=colors[name], lw=0.9)
    ax.set_ylabel(name, fontsize=11)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("时间（前 600 点）")
fig.suptitle("三类序列的形态：白噪声无结构 vs 粉红噪声分形 vs 周期主导", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "cover.png"), dpi=120)
plt.close(fig)

# ============ 计算 MSE & DFA ============
scales = np.arange(1, 21)
mse_res = {}
for name, s in series.items():
    mse_res[name] = mse(s, scales, m=2)

dfa_alpha = {}
dfa_data = {}
for name, s in series.items():
    sc, F, a = dfa(s)
    dfa_alpha[name] = a
    dfa_data[name] = (sc, F)

# ============ 图2：MSE 曲线 ============
fig, ax = plt.subplots(figsize=(10, 6))
for name, vals in mse_res.items():
    ax.plot(scales, vals, marker="o", ms=4, lw=1.8, color=colors[name], label=name)
ax.set_xlabel("尺度因子 τ（粗粒化窗口）")
ax.set_ylabel("样本熵 SampEn")
ax.set_title("多尺度熵 MSE：白噪声在粗尺度迅速塌缩，粉红噪声持久复杂", fontsize=13)
ax.grid(alpha=0.3)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "mse_curves.png"), dpi=120)
plt.close(fig)

# ============ 图3：DFA 标度 ============
fig, ax = plt.subplots(figsize=(10, 6))
for name, (sc, F) in dfa_data.items():
    ax.loglog(sc, F, marker="s", ms=5, lw=1.6, color=colors[name], label=f"{name} (α={dfa_alpha[name]:.2f})")
ax.set_xlabel("窗口尺度 s")
ax.set_ylabel("波动函数 F(s)")
ax.set_title("DFA：log F(s) ~ α·log s，斜率 α 区分噪声/长程相关", fontsize=13)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "dfa_scaling.png"), dpi=120)
plt.close(fig)

# ============ 图4：单尺度熵的「欺骗性」——用金融收益对照 ============
# 构造一段「平静-危机」切换的收益率，展示单尺度熵不好用、MSE 更稳
def regime_ret(n_calm, n_crisis, seed):
    rg = np.random.default_rng(seed)
    calm = rg.normal(0, 0.008, n_calm)
    crisis = rg.normal(-0.0003, 0.028, n_crisis)
    return np.concatenate([calm, crisis, calm, crisis])

ret = regime_ret(300, 300, 7)
# 单尺度 SampEn(τ=1) 滑动窗
win = 120
single = np.array([sampen(ret[i:i + win]) for i in range(0, len(ret) - win, 20)])
xs = np.arange(0, len(ret) - win, 20)
true_regime = np.where(((xs < 300) | ((xs >= 600) & (xs < 900))), 0, 1)  # 0 平静 1 危机

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(xs, single, color="#C44E52", lw=1.8, label="滑动窗 SampEn(τ=1)")
for seg_start, seg_end, lab in [(0, 300, "平静"), (300, 600, "危机"),
                                  (600, 900, "平静"), (900, 1200, "危机")]:
    ax.axvspan(seg_start, seg_end, alpha=0.08,
               color="#DD8452" if lab == "危机" else "#4C72B0")
ax.set_xlabel("时间（点）")
ax.set_ylabel("SampEn(τ=1)")
ax.set_title("单尺度熵在 regime 切换里很吵；MSE 跨尺度结构才稳", fontsize=13)
ax.grid(alpha=0.3)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "regime_single_scale.png"), dpi=120)
plt.close(fig)

# ============ stats.json ============
stats = {
    "N": N,
    "mse_tau1": {k: float(v[0]) for k, v in mse_res.items()},
    "mse_tau5": {k: float(v[4]) for k, v in mse_res.items()},
    "mse_tau10": {k: float(v[9]) for k, v in mse_res.items()},
    "mse_tau20": {k: float(v[19]) for k, v in mse_res.items()},
    "dfa_alpha": {k: float(v) for k, v in dfa_alpha.items()},
    "regime_single_scale_mean_calm": float(np.mean(single[true_regime == 0])),
    "regime_single_scale_mean_crisis": float(np.mean(single[true_regime == 1])),
}
with open(os.path.join(OUT, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print("=== MSE (τ=1,5,10,20) ===")
for k in mse_res:
    v = mse_res[k]
    print(f"  {k:12s}: τ1={v[0]:.3f}  τ5={v[4]:.3f}  τ10={v[9]:.3f}  τ20={v[19]:.3f}")
print("=== DFA α ===")
for k, a in dfa_alpha.items():
    print(f"  {k:12s}: α={a:.3f}")
print("=== regime 单尺度熵 均值 ===")
print(f"  平静={stats['regime_single_scale_mean_calm']:.3f}  危机={stats['regime_single_scale_mean_crisis']:.3f}")
print("DONE", OUT)
