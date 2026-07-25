import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)
OUT = "public/images/lead-lag-hayashi-yoshida"
os.makedirs(OUT, exist_ok=True)

BLUE = "#3b6ea5"; RED = "#c0392b"; GREEN = "#27ae60"; ORANGE = "#e67e22"; PURPLE = "#8e44ad"; GRAY = "#7f8c8d"

# ============================================================
# 造一个 leader / follower 结构：follower = leader 滞后 L 秒 + 噪声
# ============================================================
n = 20000          # 秒
true_lag = 8       # follower 落后 leader 8 秒
rho = 0.7
sigma = 0.15 / np.sqrt(n)

# leader 的 latent 收益
zl = rng.standard_normal(n)
r_leader = sigma * zl
# follower：把 leader 收益平移 true_lag 后混入自身噪声
r_follower = np.zeros(n)
noise = rng.standard_normal(n) * sigma
r_follower[true_lag:] = rho * r_leader[:-true_lag] + np.sqrt(1 - rho**2) * noise[true_lag:]
r_follower[:true_lag] = noise[:true_lag]

p_leader = 100 * np.exp(np.cumsum(r_leader))
p_follower = 80 * np.exp(np.cumsum(r_follower))

# ============================================================
# 图 1：领先-滞后示意，价格曲线错位
# ============================================================
seg = slice(500, 800)
fig, ax = plt.subplots(figsize=(10, 4.2))
ax2 = ax.twinx()
tt = np.arange(500, 800)
ax.plot(tt, p_leader[seg], color=BLUE, lw=1.8, label="Leader（领先方）")
ax2.plot(tt, p_follower[seg], color=RED, lw=1.8, label="Follower（滞后方）")
ax.set_xlabel("时间（秒）"); ax.set_ylabel("Leader 价格", color=BLUE)
ax2.set_ylabel("Follower 价格", color=RED)
ax.tick_params(axis="y", labelcolor=BLUE); ax2.tick_params(axis="y", labelcolor=RED)
ax.set_title(f"Follower 的走势像是 Leader 平移 {true_lag} 秒后的回声", fontsize=13)
l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, la1 + la2, fontsize=9, loc="upper left")
ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig(f"{OUT}/lead_lag_price.png", dpi=110); plt.close()

# ============================================================
# 图 2：HY 交叉相关随位移 L 变化的曲线（同步数据）
# HY 协方差在把一条序列平移 L 后计算，峰值处即领先滞后时间
# ============================================================
def shifted_hy_corr(ra, rb, shift):
    # 正 shift: 把 b 相对 a 往后挪（检验 a 领先 b）
    if shift > 0:
        a = ra[:-shift]; b = rb[shift:]
    elif shift < 0:
        a = ra[-shift:]; b = rb[:shift]
    else:
        a = ra; b = rb
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return np.corrcoef(a, b)[0, 1]

lags = np.arange(-25, 26)
cc = [shifted_hy_corr(r_leader, r_follower, s) for s in lags]
peak = lags[int(np.argmax(cc))]
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(lags, cc, "o-", color=PURPLE, lw=2, ms=4)
ax.axvline(peak, color=RED, ls="--", lw=1.8, label=f"峰值位移 = {peak} 秒（估计领先滞后）")
ax.axvline(true_lag, color=GREEN, ls=":", lw=1.8, label=f"真实滞后 = {true_lag} 秒")
ax.axhline(0, color=GRAY, lw=0.8)
ax.set_xlabel("位移 L（正=Leader 领先 Follower）"); ax.set_ylabel("交叉相关")
ax.set_title("交叉相关峰值定位领先滞后时间", fontsize=13)
ax.legend(fontsize=10); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/cross_corr_peak.png", dpi=110); plt.close()

# ============================================================
# 图 3：异步 tick 下，固定网格 vs HY 谁能找回 lag
# ============================================================
def hy_cross(times_a, ret_a, times_b, ret_b, shift):
    # 把 b 的时间轴整体平移 shift 秒后做 HY 协方差
    tb0 = times_b[:-1] + shift; tb1 = times_b[1:] + shift
    ta0 = times_a[:-1]; ta1 = times_a[1:]
    cov = 0.0
    for i in range(len(ret_a)):
        overlap = (tb0 < ta1[i]) & (tb1 > ta0[i])
        cov += ret_a[i] * ret_b[overlap].sum()
    return cov

def build_async(intensity_a, intensity_b):
    ia = np.where(rng.random(n) < intensity_a)[0]
    ib = np.where(rng.random(n) < intensity_b)[0]
    ta = ia.astype(float); tb = ib.astype(float)
    ra = np.diff(np.log(p_leader[ia])); rb = np.diff(np.log(p_follower[ib]))
    return ta, ra, tb, rb

ta, ra, tb, rb = build_async(0.15, 0.08)
shifts = np.arange(-25, 26)
hy_curve = []
for s in shifts:
    cov = hy_cross(ta, ra, tb, rb, s)
    hy_curve.append(cov)
hy_curve = np.array(hy_curve)
hy_curve = hy_curve / np.max(np.abs(hy_curve))
hy_peak = shifts[int(np.argmax(hy_curve))]

# 固定网格 last-tick 前向填充
def ffill(idx, logp):
    out = np.full(n, np.nan)
    out[idx] = logp[idx]
    last = np.nan
    for i in range(n):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out
fa = ffill(np.where(rng.random(n) < 0.15)[0], np.log(p_leader))
fb = ffill(np.where(rng.random(n) < 0.08)[0], np.log(p_follower))
step = 10
idx = np.arange(0, n, step)
gra = np.diff(fa[idx]); grb = np.diff(fb[idx])
grid_shifts = np.arange(-25, 26)
grid_curve = []
for s in grid_shifts:
    ss = int(round(s / step))
    grid_curve.append(shifted_hy_corr(np.nan_to_num(gra), np.nan_to_num(grb), ss if ss != 0 else 1) if ss != 0 else shifted_hy_corr(np.nan_to_num(gra), np.nan_to_num(grb), 0))
grid_curve = np.array(grid_curve)
grid_curve = grid_curve / (np.max(np.abs(grid_curve)) + 1e-12)

fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.plot(shifts, hy_curve, "o-", color=GREEN, lw=2, ms=4, label="HY 交叉相关（异步原生）")
ax.plot(grid_shifts, grid_curve, "s--", color=GRAY, lw=1.6, ms=3, label="10 秒固定网格 last-tick")
ax.axvline(true_lag, color=RED, ls=":", lw=1.8, label=f"真实滞后 = {true_lag} 秒")
ax.axvline(hy_peak, color=GREEN, ls="--", lw=1.2, alpha=0.7, label=f"HY 峰值 = {hy_peak} 秒")
ax.axhline(0, color=GRAY, lw=0.7)
ax.set_xlabel("位移 L（秒）"); ax.set_ylabel("归一化交叉相关")
ax.set_title("异步 tick：HY 精准锁定 lag，固定网格被粗粒度模糊", fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/async_leadlag.png", dpi=110); plt.close()

# ============================================================
# 图 4：lag 估计的稳健性 —— 样本量 / 噪声对峰值定位的影响
# ============================================================
def estimate_lag(nn, rho_local):
    zl = rng.standard_normal(nn)
    rl = sigma * zl
    noise = rng.standard_normal(nn) * sigma
    rf = np.zeros(nn)
    rf[true_lag:] = rho_local * rl[:-true_lag] + np.sqrt(1 - rho_local**2) * noise[true_lag:]
    lg = np.arange(-25, 26)
    c = [shifted_hy_corr(rl, rf, s) for s in lg]
    return lg[int(np.argmax(c))]

sample_sizes = [500, 1000, 2000, 5000, 10000, 20000]
est_by_n = []
err_by_n = []
for nn in sample_sizes:
    vals = [estimate_lag(nn, 0.5) for _ in range(60)]
    est_by_n.append(np.mean(vals)); err_by_n.append(np.std(vals))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].errorbar(range(len(sample_sizes)), est_by_n, yerr=err_by_n, fmt="o-",
               color=BLUE, lw=2, ms=6, capsize=4, label="估计滞后 ± 1σ")
ax[0].axhline(true_lag, color=RED, ls="--", lw=1.6, label=f"真实 = {true_lag} 秒")
ax[0].set_xticks(range(len(sample_sizes))); ax[0].set_xticklabels([str(s) for s in sample_sizes])
ax[0].set_xlabel("样本量（tick 数）"); ax[0].set_ylabel("估计滞后（秒）")
ax[0].set_title("样本越多，lag 估计越收敛", fontsize=12)
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.25)

# 相关强度对峰值清晰度的影响
rhos = [0.2, 0.4, 0.6, 0.8]
for rr, col in zip(rhos, [GRAY, ORANGE, GREEN, PURPLE]):
    zl = rng.standard_normal(n); rl = sigma * zl
    noise = rng.standard_normal(n) * sigma
    rf = np.zeros(n)
    rf[true_lag:] = rr * rl[:-true_lag] + np.sqrt(1 - rr**2) * noise[true_lag:]
    lg = np.arange(-25, 26)
    c = [shifted_hy_corr(rl, rf, s) for s in lg]
    ax[1].plot(lg, c, lw=1.8, color=col, label=f"ρ={rr}")
ax[1].axvline(true_lag, color=RED, ls=":", lw=1.6)
ax[1].set_xlabel("位移 L（秒）"); ax[1].set_ylabel("交叉相关")
ax[1].set_title("信号越弱，峰值越平、越难辨认", fontsize=12)
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/robustness.png", dpi=110); plt.close()

print("lead-lag charts done ->", OUT)
for f in os.listdir(OUT):
    print(" ", f)
