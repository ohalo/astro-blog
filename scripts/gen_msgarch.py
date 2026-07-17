#!/usr/bin/env python3
"""Generate real charts + live stats for the Markov-switching GARCH article.
All numbers printed are real outputs used in the blog text. No placeholders."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

CJK = "/System/Library/Fonts/Hiragino Sans GB.ttc"
fm.fontManager.addfont(CJK)
CJK_NAME = fm.FontProperties(fname=CJK).get_name()

BASE = "/Users/halo/workspace/astro-blog/public/images/markov-switching-garch"
os.makedirs(BASE, exist_ok=True)

plt.rcParams.update({
    "font.family": CJK_NAME,
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
})

C1, C2, C3, C4, GREY = "#2c7fb8", "#d95f0e", "#1b9e77", "#7570b3", "#666666"


# ---------------------------------------------------------------------------
# Self-contained 2-regime Markov-switching GARCH(1,1)
#   state s_t in {0 (low-vol), 1 (high-vol)}
#   omega[s]  +  alpha*eps^2_{t-1}  +  beta*h_{t-1}   =  h_t
#   P(s_t=j | s_{t-1}=i) = p[i,j]  (fixed transition matrix)
#   filtered/smoothed probs via Hamilton filter
# ---------------------------------------------------------------------------
def simulate_msgarch(n=4000, seed=42, p=np.array([[0.98, 0.02],
                                                   [0.05, 0.95]]),
                    omega=np.array([0.03, 0.30]),
                    alpha=0.08, beta=0.90, mu=0.0):
    rng = np.random.default_rng(seed)
    s = np.zeros(n, dtype=int)
    h = np.zeros(n)
    e = np.zeros(n)
    r = np.zeros(n)
    s[0] = 0
    h[0] = omega[s[0]]
    for t in range(1, n):
        # draw next state from transition row of current state
        s[t] = rng.choice([0, 1], p=p[s[t - 1]])
        h[t] = omega[s[t]] + alpha * e[t - 1] ** 2 + beta * h[t - 1]
        z = rng.standard_normal()
        e[t] = np.sqrt(h[t]) * z
        r[t] = mu + e[t]
    return r, h, s, p, omega, alpha, beta


def hamilton_filter(r, p, omega, alpha, beta, mu=0.0, init=(0.5, 0.5)):
    n = len(r)
    # regime conditional variances
    nreg = 2
    hcond = np.zeros((nreg, n))
    eta = np.zeros((nreg, n))          # conditional densities
    pred_prob = np.zeros((nreg, n))    # P(s_t | info_{t-1})
    filt_prob = np.zeros((nreg, n))    # P(s_t | info_t)
    joint = np.zeros((nreg, n))
    # init
    pi = np.array(init)
    e0 = r[0] - mu
    for i in range(nreg):
        hcond[i, 0] = omega[i] + beta * (e0 ** 2 if False else 1.0)  # rough init
        hcond[i, 0] = omega[i] + 1.0
        eta[i, 0] = (1 / np.sqrt(2 * np.pi * hcond[i, 0])) * \
                    np.exp(-0.5 * e0 ** 2 / hcond[i, 0])
    pred_prob[:, 0] = pi
    joint[:, 0] = pred_prob[:, 0] * eta[:, 0]
    filt_prob[:, 0] = joint[:, 0] / joint[:, 0].sum()
    # initialize each regime's conditional variance to its unconditional level
    for i in range(nreg):
        hcond[i, 0] = omega[i] / (1.0 - alpha - beta)

    for t in range(1, n):
        e = r[t] - mu
        for i in range(nreg):
            hcond[i, t] = omega[i] + alpha * (r[t - 1] - mu) ** 2 + beta * hcond[i, t - 1]
            eta[i, t] = (1 / np.sqrt(2 * np.pi * hcond[i, t])) * \
                        np.exp(-0.5 * e ** 2 / hcond[i, t])
        pred_prob[:, t] = p.T @ filt_prob[:, t - 1]
        joint[:, t] = pred_prob[:, t] * eta[:, t]
        filt_prob[:, t] = joint[:, t] / joint[:, t].sum()
    return filt_prob, hcond


def smoothing_prob(filt_prob, p, n):
    # Hamilton backward smoother (2-regime), correct formula:
    #   P(s_t=i | s_{t+1}=j, I_t) = p[i,j]*filt[i,t] / sum_k p[k,j]*filt[k,t]
    #   smooth[i,t] = sum_j P(s_t=i | s_{t+1}=j, I_t) * smooth[j,t+1]
    smooth = filt_prob.copy()
    for t in range(n - 2, -1, -1):
        denom = p.T @ filt_prob[:, t]                 # P(s_{t+1}=j | I_t)
        trans = (p * filt_prob[:, t][:, None]) / denom[None, :]   # [i, j]
        smooth[:, t] = trans @ smooth[:, t + 1]
        ssum = smooth[:, t].sum()
        if ssum > 0:
            smooth[:, t] /= ssum
    return smooth


# ---------------------------------------------------------------------------
rng_seed = 20260717
r, h_true, s_true, p_true, omega, alpha, beta = simulate_msgarch(
    n=4000, seed=rng_seed,
    p=np.array([[0.97, 0.03],
                [0.07, 0.93]]),
    omega=np.array([0.02, 0.55]),
    alpha=0.10, beta=0.85)

print("=== MARKOV SWITCHING GARCH STATS ===")
print(f"n={len(r)}  true omega=[{omega[0]:.2f},{omega[1]:.2f}]  alpha={alpha}  beta={beta}")
print(f"true transition P=\n{p_true}")
low_vol = h_true[s_true == 0].mean()
high_vol = h_true[s_true == 1].mean()
print(f"true low-vol regime mean h={low_vol:.3f}  high-vol regime mean h={high_vol:.3f}  ratio={high_vol/low_vol:.1f}x")
print(f"true regime share: low={np.mean(s_true==0):.1%}  high={np.mean(s_true==1):.1%}")

# Hamilton filter + smoother
filt_prob, hcond = hamilton_filter(r, p_true, omega, alpha, beta, mu=0.0)
smooth_prob = smoothing_prob(filt_prob, p_true, len(r))
# regime classification = argmax smoothed prob
s_est = smooth_prob[1] > 0.5   # 1 = high vol

# accuracy of regime detection
acc = np.mean(s_est == s_true)
print(f"smoothed high-vol prob range=({smooth_prob[1].min():.3f},{smooth_prob[1].max():.3f})")
print(f"regime classification accuracy = {acc:.3f}")

# compare: MSGARCH conditional vol vs plain GARCH(1,1) (single regime) on same series
# single-regime GARCH via simple recursion with fixed params matched to unconditional
omega_s = np.mean([omega[0], omega[1]]) * np.mean(s_true == 0) + np.mean([omega[0], omega[1]]) * np.mean(s_true == 1)
# emulate a naive GARCH fit: use average omega
omega_single = np.mean([omega[0], omega[1]])
h_single = np.zeros(len(r))
h_single[0] = omega_single + 1.0
for t in range(1, len(r)):
    h_single[t] = omega_single + alpha * r[t - 1] ** 2 + beta * h_single[t - 1]

# RMS scaling error of conditional vol vs TRUE (both standardized by true)
err_msg = np.sqrt(np.mean(((np.sqrt(hcond[0]) * (1 - smooth_prob[1]) + np.sqrt(hcond[1]) * smooth_prob[1]) - np.sqrt(h_true)) ** 2))
err_single = np.sqrt(np.mean((np.sqrt(h_single) - np.sqrt(h_true)) ** 2))
print(f"cond-vol RMSE vs true:  MSGARCH={err_msg:.3f}   single-regime GARCH={err_single:.3f}")

# Value at Risk backtest: 1-day 95% VaR
# MSGARCH VaR: weighted by smoothed regime prob
z95 = 1.645
var_msg = z95 * (np.sqrt(hcond[0]) * (1 - smooth_prob[1]) + np.sqrt(hcond[1]) * smooth_prob[1])
var_single = z95 * np.sqrt(h_single)
breach_msg = np.mean(r[1:] < -var_msg[:-1])
breach_single = np.mean(r[1:] < -var_single[:-1])
print(f"95% VaR breach rate:  MSGARCH={breach_msg:.3f}  single-regime={breach_single:.3f}  (target 0.05)")

# ---------------------------------------------------------------------------
# FIG 1: returns with regime coloring + true vol
# ---------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
seg = 1500
idx = np.arange(seg)
colors = np.where(s_true[:seg] == 1, C2, C1)
a1.bar(idx, r[:seg], color=colors, width=1.0)
a1.set_title("模拟收益序列：红=高波动区制，蓝=低波动区制")
a1.set_ylabel("收益")
a2.plot(idx, np.sqrt(h_true[:seg]), color=C4, lw=1.4)
a2.set_title("真实条件波动率 h_t^0.5（两区制切换）")
a2.set_ylabel("条件波动率")
a2.set_xlabel("交易日")
plt.tight_layout()
plt.savefig(f"{BASE}/msgarch_returns_regime.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------------
# FIG 2: smoothed regime probability
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 4.2))
ax.plot(idx, smooth_prob[1, :seg], color=C2, lw=1.4, label="平滑后高波动概率 P(s_t=1)")
ax.axhline(0.5, color="k", ls="--", lw=1, label="0.5 分类阈值")
ax.fill_between(idx, 0, smooth_prob[1, :seg], color=C2, alpha=0.15)
ax.set_title("马尔可夫平滑概率：区制何时切到高波动？")
ax.set_xlabel("交易日")
ax.set_ylabel("高波动概率")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{BASE}/msgarch_smoothed_prob.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------------
# FIG 3: conditional vol MSGARCH vs single-regime vs true
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 4.5))
h_msg = np.sqrt(hcond[0, :seg]) * (1 - smooth_prob[1, :seg]) + np.sqrt(hcond[1, :seg]) * smooth_prob[1, :seg]
ax.plot(idx, np.sqrt(h_true[:seg]), color=GREY, lw=1.6, label="真实条件波动率")
ax.plot(idx, h_msg, color=C1, lw=1.2, label="MSGARCH 估计")
ax.plot(idx, np.sqrt(h_single[:seg]), color=C2, lw=1.2, label="单一区制 GARCH")
ax.set_title("条件波动率估计：MSGARCH 贴着真实两区制，单区制被平均化")
ax.set_xlabel("交易日")
ax.set_ylabel("条件波动率")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{BASE}/msgarch_vol_compare.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------------
# FIG 4: VaR breach illustration
# ---------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
seg2 = 800
i2 = np.arange(seg2)
a1.plot(i2, r[:seg2], color=C4, lw=0.8, label="收益")
a1.plot(i2, -var_msg[:seg2], color=C1, lw=1.4, label="MSGARCH 95% VaR(下界)")
a1.plot(i2, -var_single[:seg2], color=C2, ls="--", lw=1.4, label="单区制 95% VaR(下界)")
a1.set_title("95% 单日 VaR：MSGARCH 在高波动段更紧、低波动段更宽")
a1.legend(loc="lower left", fontsize=9)
# breach counts
br_msg = (r[1:seg2+1] < -var_msg[:seg2])
br_sg = (r[1:seg2+1] < -var_single[:seg2])
a2.plot(i2, np.where(br_msg[:seg2], -1, 0), color=C1, lw=0.6, drawstyle="steps", label=f"MSGARCH 击穿 {br_msg[:seg2].sum()} 次")
a2.plot(i2, np.where(br_sg[:seg2], 1, 0), color=C2, lw=0.6, drawstyle="steps", label=f"单区制 击穿 {br_sg[:seg2].sum()} 次")
a2.set_title("VaR 击穿事件计数（目标: 0.05×样本）")
a2.set_xlabel("交易日")
a2.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{BASE}/msgarch_var_backtest.png", dpi=130)
plt.close()

print("SAVED figures to", BASE)
