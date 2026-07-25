import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)
OUT = "public/images/cusum-structural-break"
os.makedirs(OUT, exist_ok=True)

BLUE = "#3b6ea5"; RED = "#c0392b"; GREEN = "#27ae60"; ORANGE = "#e67e22"; GRAY = "#7f8c8d"

# ---------- synthetic return series with two true regime shifts in mean ----------
T = 900
true_bp = [300, 600]
mu = np.zeros(T)
mu[:300] = 0.0010
mu[300:600] = -0.0025
mu[600:] = 0.0012
sigma = 0.010
rets = mu + sigma * rng.standard_normal(T)
price = 100 * np.cumprod(1 + rets)
t = np.arange(T)

# 1. price + true regimes
fig, ax = plt.subplots(figsize=(9, 4.0))
ax.plot(t, price, color=BLUE, lw=1.2)
for bp in true_bp:
    ax.axvline(bp, color=RED, ls="--", lw=1.2)
ax.axvspan(0, 300, color=GREEN, alpha=0.07)
ax.axvspan(300, 600, color=RED, alpha=0.07)
ax.axvspan(600, T, color=GREEN, alpha=0.07)
ymax = price.max()
ax.text(150, ymax*0.99, "上行\nμ=+10bp", ha="center", fontsize=9, color=GREEN)
ax.text(450, ymax*0.99, "下行\nμ=-25bp", ha="center", fontsize=9, color=RED)
ax.text(750, ymax*0.99, "回升\nμ=+12bp", ha="center", fontsize=9, color=GREEN)
ax.set_title("合成价格序列：三段均值状态，两个真实变点（红虚线）")
ax.set_xlabel("交易日"); ax.set_ylabel("价格")
plt.tight_layout(); plt.savefig(f"{OUT}/price_regimes.png", dpi=120); plt.close()

# ---------- CUSUM statistic ----------
mu_hat = rets.mean()
S = np.cumsum(rets - mu_hat)

fig, ax = plt.subplots(figsize=(9, 4.0))
ax.plot(t, S, color=ORANGE, lw=1.4, label="CUSUM 累积和 $S_t$")
ax.axhline(0, color=GRAY, lw=0.8)
for bp in true_bp:
    ax.axvline(bp, color=RED, ls="--", lw=1.0, alpha=0.7)
ax.set_title("CUSUM 累积和：均值漂移把曲线折成不同斜率的段")
ax.set_xlabel("交易日"); ax.set_ylabel("累积和")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/cusum_curve.png", dpi=120); plt.close()

# ---------- binary segmentation ----------
def cusum_stat(x):
    n = len(x)
    if n < 30:
        return None, 0.0
    xm = x - x.mean()
    S = np.cumsum(xm)
    s = x.std(ddof=1)
    if s == 0:
        return None, 0.0
    stat = np.abs(S) / (s * np.sqrt(n))
    k = int(np.argmax(stat))
    return k, stat[k]

def binseg(x, offset, threshold, found):
    k, stat = cusum_stat(x)
    if k is None or stat < threshold:
        return
    bp = offset + k
    found.append((bp, stat))
    binseg(x[:k], offset, threshold, found)
    binseg(x[k:], bp, threshold, found)

found = []
binseg(rets, 0, 1.10, found)
found_bp = sorted([bp for bp, _ in found])

fig, ax = plt.subplots(figsize=(9, 4.0))
ax.plot(t, price, color=BLUE, lw=1.0, alpha=0.6, label="价格")
for bp in true_bp:
    ax.axvline(bp, color=RED, ls="--", lw=1.4, label="真实变点" if bp == true_bp[0] else None)
for bp in found_bp:
    ax.axvline(bp, color=GREEN, ls="-", lw=1.2, alpha=0.8, label="检测变点" if bp == found_bp[0] else None)
ax.set_title(f"二分分割检测：真实 {true_bp} → 检测 {found_bp}")
ax.set_xlabel("交易日"); ax.set_ylabel("价格")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/binseg_detection.png", dpi=120); plt.close()

# ---------- threshold sensitivity ----------
thr_grid = np.linspace(0.6, 2.2, 25)
n_detected = []
for thr in thr_grid:
    f = []
    binseg(rets, 0, thr, f)
    n_detected.append(len(f))

fig, ax = plt.subplots(figsize=(9, 4.0))
ax.plot(thr_grid, n_detected, marker="o", color=BLUE, lw=1.4, ms=4)
ax.axhline(2, color=GREEN, ls="--", lw=1.0, label="真实变点数=2")
ax.axvspan(1.0, 1.3, color=GREEN, alpha=0.12, label="稳健阈值区")
ax.set_title("阈值敏感性：太低过度分割，太高漏检")
ax.set_xlabel("CUSUM 检测阈值"); ax.set_ylabel("检测出的变点数")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/threshold_sensitivity.png", dpi=120); plt.close()

# ---------- online CUSUM (Page's test): detection delay for negative drift ----------
k_ref = 0.5 * abs(-0.0025)
g = np.zeros(T)
h = 0.03
alarm = None
for i in range(1, T):
    g[i] = max(0, g[i-1] - (rets[i] - mu_hat) - k_ref)
    if g[i] > h and alarm is None and i > 300:
        alarm = i

fig, ax = plt.subplots(figsize=(9, 4.0))
ax.plot(t, g, color=ORANGE, lw=1.3, label="Page 单侧 CUSUM 统计量 $g_t$")
ax.axhline(h, color=RED, ls="--", lw=1.0, label=f"决策界 h={h}")
ax.axvline(300, color=RED, ls=":", lw=1.2, label="真实下行起点 t=300")
if alarm:
    ax.axvline(alarm, color=GREEN, ls="-", lw=1.2, label=f"报警 t={alarm}（延迟 {alarm-300} 日）")
ax.set_title("在线 Page-CUSUM：负向漂移的检测延迟")
ax.set_xlabel("交易日"); ax.set_ylabel("累积统计量")
ax.legend(loc="upper left", fontsize=9)
ax.set_xlim(250, 500)
plt.tight_layout(); plt.savefig(f"{OUT}/online_cusum_delay.png", dpi=120); plt.close()

print("done cusum:", found_bp, "alarm delay:", (alarm-300) if alarm else None)
