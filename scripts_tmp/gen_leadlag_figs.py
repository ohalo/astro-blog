# -*- coding: utf-8 -*-
"""高频领先滞后估计配图生成：受控模拟，A 领先 B"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/lead-lag-highfreq-estimation"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

# ---------- 模拟设置 ----------
# 有效价格：X 为领先者（布朗运动），Y 跟随 X 滞后 theta 秒 + 自身独立成分
DT = 0.1          # 生成步长 0.1s
T = 6.5 * 3600    # 一个交易日
N = int(T / DT)
THETA = 2.0       # 真实领先滞后：2 秒
LAG_STEPS = int(THETA / DT)
SIGMA = 0.0002    # 每 0.1s 波动
BETA = 0.8        # Y 对 X 的载荷

def simulate_day(rng):
    dWx = rng.normal(0, SIGMA, N)
    dWe = rng.normal(0, SIGMA * 0.6, N)
    x = np.cumsum(dWx)
    # y 的增量 = beta * x 滞后增量 + 独立噪声
    dWx_lag = np.concatenate([np.zeros(LAG_STEPS), dWx[:-LAG_STEPS]])
    y = np.cumsum(BETA * dWx_lag + dWe)
    return x, y

def poisson_obs(path, mean_gap, rng, noise=1e-4):
    """泊松到达观测 + 微观结构噪声"""
    t = []
    cur = 0.0
    while True:
        cur += rng.exponential(mean_gap)
        if cur >= T:
            break
        t.append(cur)
    t = np.array(t)
    idx = np.minimum((t / DT).astype(int), N - 1)
    p = path[idx] + rng.normal(0, noise, len(t))
    return t, p

def prev_tick(t_obs, p_obs, grid):
    idx = np.searchsorted(t_obs, grid, side="right") - 1
    return p_obs[np.maximum(idx, 0)]

# ---------- 实验1：滞后互相关函数（网格法） ----------
def lagged_corr_curve(n_days=20, h=1.0, max_lag=8.0):
    lags = np.arange(-max_lag, max_lag + h / 2, h)
    acc = np.zeros(len(lags))
    for d in range(n_days):
        x, y = simulate_day(rng)
        tx, px = poisson_obs(x, 3.0, rng)
        ty, py = poisson_obs(y, 3.0, rng)
        grid = np.arange(h, T, h)
        rx = np.diff(prev_tick(tx, px, grid))
        ry = np.diff(prev_tick(ty, py, grid))
        for i, L in enumerate(lags):
            k = int(round(L / h))
            if k > 0:      # Y 滞后 k：corr(rx[t], ry[t+k])
                a, b = rx[:-k], ry[k:]
            elif k < 0:
                a, b = rx[-k:], ry[:k]
            else:
                a, b = rx, ry
            acc[i] += np.corrcoef(a, b)[0, 1]
    return lags, acc / n_days

print("实验1：滞后互相关 ...")
lags, cc = lagged_corr_curve()
peak_lag = lags[np.argmax(cc)]
print(f"  峰值滞后 = {peak_lag:.1f}s, 峰值相关 = {cc.max():.3f}, 同期相关 = {cc[len(lags)//2]:.3f}")

plt.figure(figsize=(9, 5))
plt.plot(lags, cc, "o-", color="#1f77b4", lw=2, ms=5)
plt.axvline(0, color="gray", ls="--", lw=1)
plt.axvline(THETA, color="#d62728", ls=":", lw=2, label=f"真实滞后 θ = {THETA:.0f}s")
plt.axvline(peak_lag, color="#2ca02c", ls="-.", lw=1.5, label=f"估计峰值 = {peak_lag:.0f}s")
plt.xlabel("滞后 ℓ（秒），ℓ>0 表示 Y 滞后于 X")
plt.ylabel("Corr( rX(t), rY(t+ℓ) )")
plt.title("滞后互相关函数：峰值偏离零点 = 领先滞后关系（1s 网格，20 日均值）")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/leadlag-ccf.png", dpi=130)
plt.close()

# ---------- 实验2：HY 互相关 vs 偏移量（HRY 估计器思想） ----------
def hy_cov_shift(tx, px, ty, py, shift):
    """Hayashi-Yoshida 协方差，Y 的时间轴平移 -shift 后区间重叠配对"""
    rx = np.diff(px)
    ry = np.diff(py)
    ax0, ax1 = tx[:-1], tx[1:]
    ay0, ay1 = ty[:-1] - shift, ty[1:] - shift
    total = 0.0
    j = 0
    for i in range(len(rx)):
        while j < len(ry) and ay1[j] <= ax0[i]:
            j += 1
        k = j
        while k < len(ry) and ay0[k] < ax1[i]:
            total += rx[i] * ry[k]
            k += 1
    return total

def hry_curve(n_days=6, shifts=None):
    if shifts is None:
        shifts = np.arange(-8, 8.5, 1.0)
    acc = np.zeros(len(shifts))
    for d in range(n_days):
        x, y = simulate_day(rng)
        tx, px = poisson_obs(x, 3.0, rng, noise=0)
        ty, py = poisson_obs(y, 3.0, rng, noise=0)
        vx = np.sum(np.diff(px) ** 2)
        vy = np.sum(np.diff(py) ** 2)
        for i, s in enumerate(shifts):
            acc[i] += hy_cov_shift(tx, px, ty, py, s) / np.sqrt(vx * vy)
    return shifts, acc / n_days

print("实验2：HRY 对比曲线 ...")
shifts, hy = hry_curve()
hy_peak = shifts[np.argmax(hy)]
print(f"  HRY 峰值偏移 = {hy_peak:.1f}s, 峰值 = {hy.max():.3f}")

plt.figure(figsize=(9, 5))
plt.plot(shifts, hy, "s-", color="#9467bd", lw=2, ms=5)
plt.axvline(0, color="gray", ls="--", lw=1)
plt.axvline(THETA, color="#d62728", ls=":", lw=2, label=f"真实滞后 θ = {THETA:.0f}s")
plt.axvline(hy_peak, color="#2ca02c", ls="-.", lw=1.5, label=f"HRY 峰值 = {hy_peak:.0f}s")
plt.xlabel("时间平移 δ（秒）：把 Y 的时钟拨快 δ 再做 HY 配对")
plt.ylabel("HY 相关（区间重叠配对）")
plt.title("HRY 估计器：对平移量扫描，无需网格对齐（6 日均值，无噪声）")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/leadlag-hry.png", dpi=130)
plt.close()

# ---------- 实验3：采样频率的破坏作用 ----------
print("实验3：粗网格如何抹掉领先滞后 ...")
hs = [1, 2, 5, 10, 30, 60]
asym = []
for h in hs:
    lags_h, cc_h = lagged_corr_curve(n_days=8, h=float(h), max_lag=max(8.0, 2 * h))
    # 不对称度：max(正滞后) - max(负滞后)
    pos = cc_h[lags_h > 0].max()
    neg = cc_h[lags_h < 0].max()
    asym.append(pos - neg)
    print(f"  h={h}s: 正向峰 {pos:.3f}, 反向峰 {neg:.3f}, 不对称度 {pos-neg:.3f}")

plt.figure(figsize=(9, 5))
plt.plot(hs, asym, "o-", color="#d62728", lw=2, ms=7)
plt.xscale("log")
plt.xlabel("采样间隔 h（秒，对数轴）")
plt.ylabel("互相关不对称度（正向峰 − 反向峰）")
plt.title("领先滞后信号随采样间隔增大而消失：h 远大于 θ 时两边趋于对称")
plt.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(f"{OUT}/leadlag-decay.png", dpi=130)
plt.close()

# ---------- 实验4：领先滞后策略的信息含量 ----------
print("实验4：预测性检验 ...")
n_days = 15
ics = []
for d in range(n_days):
    x, y = simulate_day(rng)
    tx, px = poisson_obs(x, 3.0, rng)
    ty, py = poisson_obs(y, 3.0, rng)
    grid = np.arange(1.0, T, 1.0)
    rx = np.diff(prev_tick(tx, px, grid))
    ry = np.diff(prev_tick(ty, py, grid))
    # 用 X 过去 2 秒收益预测 Y 未来 2 秒收益
    w = 2
    fx = np.convolve(rx, np.ones(w), "valid")[:-w]
    fy = np.convolve(ry, np.ones(w), "valid")[w:]
    n = min(len(fx), len(fy))
    ics.append(np.corrcoef(fx[:n], fy[:n])[0, 1])
ics = np.array(ics)
print(f"  日频 IC 均值 = {ics.mean():.3f}, std = {ics.std():.3f}, t = {ics.mean()/ics.std()*np.sqrt(len(ics)):.1f}")

# 反向对照
ics_rev = []
for d in range(n_days):
    x, y = simulate_day(rng)
    tx, px = poisson_obs(x, 3.0, rng)
    ty, py = poisson_obs(y, 3.0, rng)
    grid = np.arange(1.0, T, 1.0)
    rx = np.diff(prev_tick(tx, px, grid))
    ry = np.diff(prev_tick(ty, py, grid))
    w = 2
    fy_past = np.convolve(ry, np.ones(w), "valid")[:-w]
    fx_fut = np.convolve(rx, np.ones(w), "valid")[w:]
    n = min(len(fy_past), len(fx_fut))
    ics_rev.append(np.corrcoef(fy_past[:n], fx_fut[:n])[0, 1])
ics_rev = np.array(ics_rev)
print(f"  反向 IC 均值 = {ics_rev.mean():.3f}, std = {ics_rev.std():.3f}")

plt.figure(figsize=(9, 5))
pos_x = np.arange(len(ics))
plt.bar(pos_x - 0.2, ics, 0.38, label=f"X→Y 正向（均值 {ics.mean():.3f}）", color="#1f77b4")
plt.bar(pos_x + 0.2, ics_rev, 0.38, label=f"Y→X 反向对照（均值 {ics_rev.mean():.3f}）", color="#bbbbbb")
plt.axhline(0, color="black", lw=0.8)
plt.xlabel("交易日")
plt.ylabel("IC：领先者过去 2s 收益 vs 跟随者未来 2s 收益")
plt.title("方向性检验：只有「领先者预测跟随者」方向有稳定正 IC")
plt.legend()
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/leadlag-ic.png", dpi=130)
plt.close()

print("完成，输出目录：", OUT)
