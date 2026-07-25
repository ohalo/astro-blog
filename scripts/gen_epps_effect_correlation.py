import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(2026)
OUT = "public/images/epps-effect-correlation"
os.makedirs(OUT, exist_ok=True)

BLUE = "#3b6ea5"; RED = "#c0392b"; GREEN = "#27ae60"; ORANGE = "#e67e22"; PURPLE = "#8e44ad"; GRAY = "#7f8c8d"

# ============================================================
# 合成两条相关的高频价格路径 + 异步稀疏成交
# 真实 latent 相关 rho_true，采样越细相关性越假（Epps 效应）
# ============================================================
T = 6.5 * 3600            # 一个交易日 6.5 小时（秒）
dt = 1.0                  # latent 过程以 1 秒推进
n = int(T / dt)
rho_true = 0.6
sigma1 = 0.20 / np.sqrt(n)   # 日波动 ~20%
sigma2 = 0.25 / np.sqrt(n)

# 相关的连续 latent 对数价格（Brownian）
z = rng.standard_normal((n, 2))
z[:, 1] = rho_true * z[:, 0] + np.sqrt(1 - rho_true**2) * z[:, 1]
r1 = sigma1 * z[:, 0]
r2 = sigma2 * z[:, 1]
p1 = 100 * np.exp(np.cumsum(r1))
p2 = 50 * np.exp(np.cumsum(r2))

# ============================================================
# 图 1：Epps 效应主图 —— 采样频率 vs 观测相关性
# 关键：真实市场的成交是异步、离散的。这里用泊松成交 + last-tick 填充
# 复现 Epps 效应：越往细采样，相关性越向 0 塌陷
# ============================================================
def ffill(x):
    out = x.copy(); last = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out

def observed_corr(la, lb, intensity, step):
    # 泊松稀疏成交 + last-tick 前向填充 + 固定网格采样
    obs_a = np.where(rng.random(n) < intensity, la, np.nan)
    obs_b = np.where(rng.random(n) < intensity, lb, np.nan)
    fa = ffill(obs_a); fb = ffill(obs_b)
    idx = np.arange(0, n, step)
    ra = np.diff(fa[idx]); rb = np.diff(fb[idx])
    mask = ~(np.isnan(ra) | np.isnan(rb))
    ra = ra[mask]; rb = rb[mask]
    if len(ra) < 5 or ra.std() == 0 or rb.std() == 0:
        return np.nan
    return np.corrcoef(ra, rb)[0, 1]

steps = [1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, 1800]
labels = ["1s", "2s", "5s", "10s", "30s", "1m", "2m", "5m", "10m", "20m", "30m"]
# 蒙特卡洛平均降噪
M = 30
curve = np.zeros(len(steps))
for m in range(M):
    zz = rng.standard_normal((n, 2))
    zz[:, 1] = rho_true * zz[:, 0] + np.sqrt(1 - rho_true**2) * zz[:, 1]
    la = np.cumsum(sigma1 * zz[:, 0]); lb = np.cumsum(sigma2 * zz[:, 1])
    for i, s in enumerate(steps):
        curve[i] += observed_corr(la, lb, 0.1, s)
curve /= M

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(range(len(steps)), curve, "o-", color=BLUE, lw=2, ms=6, label="观测相关性（异步成交）")
ax.axhline(rho_true, color=RED, ls="--", lw=1.8, label=f"真实 latent 相关 = {rho_true}")
ax.set_xticks(range(len(steps))); ax.set_xticklabels(labels)
ax.set_xlabel("采样间隔（越左越细）"); ax.set_ylabel("估计相关系数")
ax.set_title("Epps 效应：采样越细，观测相关性越向 0 塌陷", fontsize=13)
ax.legend(fontsize=10); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/epps_main.png", dpi=110); plt.close()

# ============================================================
# 图 2：异步成交示意 —— 两只股票的 tick 不在同一时刻
# ============================================================
fig, ax = plt.subplots(figsize=(10, 3.6))
# 泊松稀疏成交
t_a = np.sort(rng.uniform(0, 120, 18))
t_b = np.sort(rng.uniform(0, 120, 12))
ax.vlines(t_a, 0.55, 0.95, color=BLUE, lw=1.6)
ax.vlines(t_b, 0.05, 0.45, color=RED, lw=1.6)
ax.scatter(t_a, np.full_like(t_a, 0.95), color=BLUE, s=22, zorder=3)
ax.scatter(t_b, np.full_like(t_b, 0.45), color=RED, s=22, zorder=3)
# 展示一个对齐窗口内可能只有一边有成交
for x in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]:
    ax.axvline(x, color=GRAY, ls=":", lw=0.7, alpha=0.6)
ax.text(2, 0.98, "股票 A 成交", color=BLUE, fontsize=10, va="bottom")
ax.text(2, 0.02, "股票 B 成交", color=RED, fontsize=10, va="top")
ax.set_ylim(-0.05, 1.15); ax.set_xlim(0, 120)
ax.set_yticks([]); ax.set_xlabel("时间（秒），虚线为 10 秒对齐格点")
ax.set_title("异步成交：细格点内常常只有一边有 tick → 用旧价填充制造零收益", fontsize=12)
plt.tight_layout(); plt.savefig(f"{OUT}/async_ticks.png", dpi=110); plt.close()

# ============================================================
# 图 3：成交强度（流动性）对 Epps 衰减速度的影响
# 成交越稀疏，相关性在越粗的尺度上才恢复
# ============================================================
def epps_with_trading(intensity, step):
    # intensity: 平均每秒成交概率
    zz = rng.standard_normal((n, 2))
    zz[:, 1] = rho_true * zz[:, 0] + np.sqrt(1 - rho_true**2) * zz[:, 1]
    la = np.cumsum(sigma1 * zz[:, 0])
    lb = np.cumsum(sigma2 * zz[:, 1])
    # 成交时刻
    trade_a = rng.random(n) < intensity
    trade_b = rng.random(n) < intensity
    obs_a = np.where(trade_a, la, np.nan)
    obs_b = np.where(trade_b, lb, np.nan)
    # 前向填充（last-tick）
    def ffill(x):
        out = x.copy()
        last = np.nan
        for i in range(len(out)):
            if np.isnan(out[i]):
                out[i] = last
            else:
                last = out[i]
        return out
    fa = ffill(obs_a); fb = ffill(obs_b)
    idx = np.arange(0, n, step)
    ra = np.diff(fa[idx]); rb = np.diff(fb[idx])
    mask = ~(np.isnan(ra) | np.isnan(rb))
    ra = ra[mask]; rb = rb[mask]
    if len(ra) < 5 or ra.std() == 0 or rb.std() == 0:
        return np.nan
    return np.corrcoef(ra, rb)[0, 1]

steps2 = [1, 5, 10, 30, 60, 120, 300, 600, 1200]
lab2 = ["1s", "5s", "10s", "30s", "1m", "2m", "5m", "10m", "20m"]
fig, ax = plt.subplots(figsize=(9, 4.6))
for inten, col, name in [(0.5, GREEN, "高流动（0.5 笔/秒）"),
                          (0.1, ORANGE, "中流动（0.1 笔/秒）"),
                          (0.02, PURPLE, "低流动（0.02 笔/秒）")]:
    c = []
    for s in steps2:
        vals = [epps_with_trading(inten, s) for _ in range(15)]
        c.append(np.nanmean(vals))
    ax.plot(range(len(steps2)), c, "o-", color=col, lw=2, ms=5, label=name)
ax.axhline(rho_true, color=RED, ls="--", lw=1.6, label=f"真实相关 = {rho_true}")
ax.set_xticks(range(len(steps2))); ax.set_xticklabels(lab2)
ax.set_xlabel("采样间隔"); ax.set_ylabel("估计相关系数")
ax.set_title("流动性越低，Epps 衰减越狠、恢复越晚", fontsize=13)
ax.legend(fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/liquidity_effect.png", dpi=110); plt.close()

# ============================================================
# 图 4：Hayashi-Yoshida 估计量 vs 固定网格采样
# HY 用重叠区间累加，不需要同步、不丢数据
# ============================================================
def hy_covariance(times_a, ret_a, times_b, ret_b):
    # 每个 a 的收益区间 [t_{i-1}, t_i]，与所有重叠的 b 收益区间相乘
    cov = 0.0
    ta0 = times_a[:-1]; ta1 = times_a[1:]
    tb0 = times_b[:-1]; tb1 = times_b[1:]
    for i in range(len(ret_a)):
        overlap = (tb0 < ta1[i]) & (tb1 > ta0[i])
        cov += ret_a[i] * ret_b[overlap].sum()
    return cov

def hy_corr_once(intensity):
    zz = rng.standard_normal((n, 2))
    zz[:, 1] = rho_true * zz[:, 0] + np.sqrt(1 - rho_true**2) * zz[:, 1]
    la = np.cumsum(sigma1 * zz[:, 0]); lb = np.cumsum(sigma2 * zz[:, 1])
    ia = np.where(rng.random(n) < intensity)[0]
    ib = np.where(rng.random(n) < intensity)[0]
    if len(ia) < 5 or len(ib) < 5:
        return np.nan
    ta = ia.astype(float); tb = ib.astype(float)
    ra = np.diff(la[ia]); rb = np.diff(lb[ib])
    cov = hy_covariance(ta, ra, tb, rb)
    va = (ra**2).sum(); vb = (rb**2).sum()
    if va == 0 or vb == 0:
        return np.nan
    return cov / np.sqrt(va * vb)

intens = [0.5, 0.1, 0.02]
inames = ["高", "中", "低"]
hy_vals = []; grid1s = []
for inten in intens:
    hy_vals.append(np.nanmean([hy_corr_once(inten) for _ in range(30)]))
    grid1s.append(np.nanmean([epps_with_trading(inten, 1) for _ in range(30)]))

x = np.arange(len(intens)); w = 0.35
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.bar(x - w/2, grid1s, w, color=GRAY, label="1 秒固定网格（被 Epps 咬伤）")
ax.bar(x + w/2, hy_vals, w, color=GREEN, label="Hayashi-Yoshida 估计量")
ax.axhline(rho_true, color=RED, ls="--", lw=1.6, label=f"真实相关 = {rho_true}")
ax.set_xticks(x); ax.set_xticklabels([f"{s}流动性" for s in inames])
ax.set_ylabel("估计相关系数")
ax.set_title("解药：HY 估计量在异步稀疏数据下几乎无偏", fontsize=13)
ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y")
plt.tight_layout(); plt.savefig(f"{OUT}/hy_solution.png", dpi=110); plt.close()

print("epps charts done ->", OUT)
for f in os.listdir(OUT):
    print(" ", f)
