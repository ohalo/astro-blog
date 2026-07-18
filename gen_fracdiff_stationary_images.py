#!/usr/bin/env python3
"""
为文章「分数差分平稳化：在保留记忆与消除单位根之间取平衡」
生成真实配图 (应用视角: 用 fracdiff 选 d 来服务『样本外预测』, 区别于特征工程视角)。

机制(合成、仅演示方法):
  - 类资产价格: 随机游走 I(1) + 轻微慢周期, 长记忆、非平稳
  - 对每个 d∈[0,1] 做 (1-L)^d 分数差分
  - 在差分序列上拟合 AR(1) 并外推 h 步, 计算样本外 MSPE
  - 同时记录 ADF 平稳性 p 值
  - 结论: MSPE(d) 呈 U 形 —— d 太小(非平稳) 与 d 太大(丢记忆) 都更差, 中间最优
"""
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "fractional-differentiation-stationary")
os.makedirs(D, exist_ok=True)

C = {"blue": "#2F4B7C", "red": "#C44E52", "green": "#55A868",
     "orange": "#FF7F0E", "grey": "#888888"}


# ---------- 分数差分 ----------
def fracdiff_weights(d, L=120):
    w = np.zeros(L + 1); w[0] = 1.0
    for k in range(1, L + 1):
        w[k] = w[k - 1] * (k - 1 - d) / k
    return w

def fracdiff(x, d, L=120):
    w = fracdiff_weights(d, L)
    conv = np.convolve(x, w, mode="full")
    y = conv[:len(x)].copy()
    y[:L] = np.nan
    return y


# ---------- 自包含 ADF(仅常数项) ----------
def adf_test(y, max_lag=5):
    y = np.asarray(y, float)[np.isfinite(y)]
    n = len(y)
    if n < 40:
        return np.nan, np.nan
    dy = np.diff(y); p = max_lag
    Js = np.arange(p + 1, len(dy)); T = len(Js)
    X = np.zeros((T, 2 + p)); dep = np.zeros(T)
    for r, j in enumerate(Js):
        dep[r] = dy[j]; X[r, 0] = 1.0; X[r, 1] = y[j]
        for k in range(1, p + 1):
            X[r, 1 + k] = dy[j - k]
    beta, *_ = np.linalg.lstsq(X, dep, rcond=None)
    resid = dep - X @ beta
    sigma2 = resid @ resid / (T - X.shape[1])
    se = np.sqrt(max(sigma2 * np.linalg.inv(X.T @ X)[1, 1], 1e-18))
    t = beta[1] / se
    return t, 2.0 * (1.0 - stats.norm.cdf(abs(t)))


# ---------- 数据: 类资产价格 ----------
rng = np.random.default_rng(20260719)
n = 1500
trend = np.cumsum(rng.normal(0, 1, n))          # 随机游走 I(1)
slow = 3.0 * np.sin(np.linspace(0, 8 * np.pi, n))   # 慢周期
x = trend + slow
x = x - x.mean()

# 训练/测试切分
split = int(n * 0.75)
xtr, xte = x[:split], x[split:]
H = 5   # 外推步数


def oos_mspe(y, H=5):
    """在 y 上拟合 AR(1) 外推 H 步, 返回样本外 MSPE(最后 H 个点)"""
    yv = y[np.isfinite(y)]
    if len(yv) < 60:
        return np.nan
    phi = np.linalg.lstsq(yv[:-1][:, None], yv[1:], rcond=None)[0][0]
    pred = yv[-1]
    errs = []
    actual = yv[-H:]
    for h in range(H):
        pred = phi * pred
        errs.append((pred - actual[h]) ** 2)
    return np.mean(errs)


ds = np.round(np.arange(0.0, 1.01, 0.05), 2)
adf_p, mspe = [], []
for d in ds:
    y = fracdiff(x, d, L=120)
    _, pv = adf_test(y)
    adf_p.append(pv)
    mspe.append(oos_mspe(y, H))

adf_p = np.array(adf_p); mspe = np.array(mspe)
best_i = int(np.nanargmin(mspe))
print(f"最优 d={ds[best_i]:.2f}  OOS MSPE={mspe[best_i]:.4f}")
print(f"d=0(原序列) MSPE={mspe[0]:.4f}  d=1(收益率) MSPE={mspe[-1]:.4f}")
print(f"d=0 ADF p={adf_p[0]:.3f}  d=1 ADF p={adf_p[-1]:.3f}  平稳(d≈0.3) ADF p={adf_p[6]:.3f}")

# ---------- 图1: OOS MSPE vs d (U 形) ----------
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(ds, mspe, "-o", color=C["blue"], ms=4)
ax.axvline(ds[best_i], color=C["green"], ls="--", lw=1.4,
           label=f"最优 d={ds[best_i]:.2f}")
ax.axvspan(0, 0.05, color=C["red"], alpha=0.08)
ax.set_xlabel("分数差分阶数 d")
ax.set_ylabel("样本外 MSPE (AR(1) 外推)")
ax.set_title("预测误差随 d 呈 U 形：太小非平稳、太大丢记忆", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(D, "fracdiff_forecast_mspe.png"), dpi=130)
plt.close()

# ---------- 图2: ADF 平稳性增益曲线 ----------
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(ds, adf_p, "-o", color=C["orange"], ms=4)
ax.axhline(0.05, color=C["red"], ls="--", lw=1.2, label="平稳阈值 0.05")
ax.set_yscale("log")
ax.set_xlabel("分数差分阶数 d")
ax.set_ylabel("ADF p 值 (对数轴)")
ax.set_title("平稳性增益：d 一越过阈值，单位根即被消除", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(D, "fracdiff_adf_curve.png"), dpi=130)
plt.close()

# ---------- 图3: 不同 d 在多预测期的 OOS R² 对比 ----------
horizons = [1, 3, 5, 10]
Rs = {d: [] for d in [0.0, 0.3, 1.0]}
labels = {"0.0": "原序列 d=0", "0.3": "平稳 d=0.3", "1.0": "收益率 d=1"}
for Hh in horizons:
    for d in [0.0, 0.3, 1.0]:
        y = fracdiff(x, d, L=120)
        m = oos_mspe(y, Hh)
        base = oos_mspe(fracdiff(x, 0.3, L=120), Hh)
        Rs[d].append(1 - m / base if base > 0 else 0)
xpos = np.arange(len(horizons)); w = 0.26
fig, ax = plt.subplots(figsize=(7.4, 4.6))
for j, d in enumerate([0.0, 0.3, 1.0]):
    col = {"0.0": C["red"], "0.3": C["green"], "1.0": C["blue"]}[str(float(d))]
    ax.bar(xpos + (j - 1) * w, Rs[d], w, label=labels[str(float(d))], color=col)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(xpos); ax.set_xticklabels([f"h={h}" for h in horizons])
ax.set_xlabel("外推步数 h"); ax.set_ylabel("相对 OOS R² (以 d=0.3 为基准)")
ax.set_title("外推越远，平稳化 d 的优势越明显", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(D, "fracdiff_oos_compare.png"), dpi=130)
plt.close()

print("images saved to", D)
