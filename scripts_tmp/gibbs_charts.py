#!/usr/bin/env python3
"""Charts for Gibbs sampler spread estimation article."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/gibbs-sampler-spread"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(42)

# ---------- Roll model simulation ----------
def simulate_roll(T, c, sigma_u, rng):
    q = rng.choice([-1, 1], size=T)
    u = rng.normal(0, sigma_u, size=T)
    m = np.cumsum(u) + 10.0
    p = m + c * q
    return p, q, m

# ---------- Gibbs sampler ----------
def gibbs_spread(p, n_iter=1000, burn=200, rng=None):
    rng = rng or np.random.default_rng(0)
    T = len(p)
    dp = np.diff(p)
    # init
    q = rng.choice([-1, 1], size=T)
    c = 0.01
    sig2 = np.var(dp) / 2
    c_draws = np.empty(n_iter)
    sig_draws = np.empty(n_iter)
    for it in range(n_iter):
        # 1) sample q_t | c, sig2  (t = 0..T-1), Bernoulli from likelihood of adjacent dp
        for t in range(T):
            ll = np.zeros(2)  # q_t = +1, -1
            for k, qt in enumerate((1.0, -1.0)):
                s = 0.0
                if t >= 1:
                    e = dp[t-1] - c * (qt - q[t-1])
                    s += -0.5 * e * e / sig2
                if t < T - 1:
                    e = dp[t] - c * (q[t+1] - qt)
                    s += -0.5 * e * e / sig2
                ll[k] = s
            pr = np.exp(ll - ll.max())
            pr /= pr.sum()
            q[t] = 1 if rng.random() < pr[0] else -1
        # 2) sample c | q, sig2 : regression dp = c * dq + u, truncated at 0
        dq = np.diff(q.astype(float))
        xx = np.dot(dq, dq)
        if xx > 0:
            mu_c = np.dot(dq, dp) / xx
            var_c = sig2 / xx
            # truncated normal >= 0 via rejection
            draw = rng.normal(mu_c, np.sqrt(var_c))
            tries = 0
            while draw < 0 and tries < 100:
                draw = rng.normal(mu_c, np.sqrt(var_c))
                tries += 1
            c = max(draw, 1e-6)
        # 3) sample sig2 | c, q : inverse gamma
        resid = dp - c * dq
        a = 2.0 + len(resid) / 2
        b = 1e-6 + 0.5 * np.dot(resid, resid)
        sig2 = b / rng.gamma(a)
        c_draws[it] = c
        sig_draws[it] = np.sqrt(sig2)
    return c_draws[burn:], sig_draws[burn:], c_draws

def roll_estimator(p):
    dp = np.diff(p)
    cov = np.cov(dp[1:], dp[:-1])[0, 1]
    if cov < 0:
        return np.sqrt(-cov)
    return np.nan

# ================= Chart 1: trace + posterior =================
true_c, sigma_u = 0.05, 0.02
p, q_true, m = simulate_roll(500, true_c, sigma_u, rng)
post_c, post_sig, full_trace = gibbs_spread(p, n_iter=1200, burn=200, rng=np.random.default_rng(1))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(full_trace, lw=0.7, color="#1f77b4")
axes[0].axhline(true_c, color="crimson", ls="--", label=f"真实半价差 c={true_c}")
axes[0].axvspan(0, 200, color="gray", alpha=0.2, label="burn-in（丢弃）")
axes[0].set_title("Gibbs 抽样轨迹：半价差 c")
axes[0].set_xlabel("迭代次数")
axes[0].set_ylabel("c 抽样值")
axes[0].legend(fontsize=9)

axes[1].hist(post_c, bins=40, color="#1f77b4", alpha=0.75, density=True)
axes[1].axvline(true_c, color="crimson", ls="--", label=f"真实值 {true_c}")
axes[1].axvline(post_c.mean(), color="darkorange", ls="-", label=f"后验均值 {post_c.mean():.4f}")
lo, hi = np.percentile(post_c, [2.5, 97.5])
axes[1].axvspan(lo, hi, color="orange", alpha=0.15, label=f"95% 可信区间 [{lo:.3f}, {hi:.3f}]")
axes[1].set_title("半价差 c 的后验分布")
axes[1].set_xlabel("c")
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/gs-trace-posterior.png", dpi=130)
plt.close()
print("chart1 done", post_c.mean(), (lo, hi))

# ================= Chart 2: Gibbs vs Roll across sims =================
n_sims = 30
cs = [0.01, 0.03, 0.05, 0.08, 0.12]
gibbs_est, roll_est, true_list = [], [], []
roll_fail = 0
for c0 in cs:
    for s in range(n_sims // 5 + 2):
        r2 = np.random.default_rng(1000 + int(c0 * 1000) + s)
        p2, _, _ = simulate_roll(300, c0, 0.02, r2)
        pc, _, _ = gibbs_spread(p2, n_iter=600, burn=150, rng=np.random.default_rng(s + int(c0*1e4)))
        gibbs_est.append(pc.mean())
        re = roll_estimator(p2)
        if np.isnan(re):
            roll_fail += 1
        roll_est.append(re)
        true_list.append(c0)

true_arr = np.array(true_list); g_arr = np.array(gibbs_est); r_arr = np.array(roll_est)
fig, ax = plt.subplots(figsize=(7.5, 6))
jit = rng.normal(0, 0.0012, size=len(true_arr))
ax.scatter(true_arr + jit, g_arr, s=45, alpha=0.75, label="Gibbs 后验均值", color="#1f77b4")
mask = ~np.isnan(r_arr)
ax.scatter(true_arr[mask] + jit[mask] + 0.003, r_arr[mask], s=45, alpha=0.6,
           marker="^", label="Roll 估计量（可算出的样本）", color="#2ca02c")
lims = [0, 0.15]
ax.plot(lims, lims, "k--", lw=1, label="45° 线（无偏）")
ax.set_xlim(lims); ax.set_ylim(-0.005, 0.16)
ax.set_xlabel("真实半价差 c")
ax.set_ylabel("估计值")
ax.set_title(f"Gibbs vs Roll：{len(true_arr)} 次模拟\nRoll 失效（正协方差算不出）{roll_fail} 次，Gibbs 全部给出估计")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/gs-vs-roll.png", dpi=130)
plt.close()
print("chart2 done, roll fails:", roll_fail, "/", len(true_arr))

# ================= Chart 3: q_t recovery accuracy =================
p3, q3, _ = simulate_roll(400, 0.05, 0.02, np.random.default_rng(7))
# run gibbs and track q agreement
def gibbs_track_q(p, q_true, n_iter=800, burn=200, rng=None):
    rng = rng or np.random.default_rng(0)
    T = len(p); dp = np.diff(p)
    q = rng.choice([-1, 1], size=T)
    c = 0.01; sig2 = np.var(dp) / 2
    q_sum = np.zeros(T)
    cnt = 0
    for it in range(n_iter):
        for t in range(T):
            ll = np.zeros(2)
            for k, qt in enumerate((1.0, -1.0)):
                s = 0.0
                if t >= 1:
                    e = dp[t-1] - c * (qt - q[t-1]); s += -0.5*e*e/sig2
                if t < T-1:
                    e = dp[t] - c * (q[t+1] - qt); s += -0.5*e*e/sig2
                ll[k] = s
            pr = np.exp(ll - ll.max()); pr /= pr.sum()
            q[t] = 1 if rng.random() < pr[0] else -1
        dq = np.diff(q.astype(float))
        xx = np.dot(dq, dq)
        if xx > 0:
            mu_c = np.dot(dq, dp)/xx; var_c = sig2/xx
            d = rng.normal(mu_c, np.sqrt(var_c)); t2 = 0
            while d < 0 and t2 < 100:
                d = rng.normal(mu_c, np.sqrt(var_c)); t2 += 1
            c = max(d, 1e-6)
        resid = dp - c*dq
        a = 2.0 + len(resid)/2; b = 1e-6 + 0.5*np.dot(resid, resid)
        sig2 = b/rng.gamma(a)
        if it >= burn:
            q_sum += q; cnt += 1
    q_prob = (q_sum/cnt + 1)/2  # prob of buy
    return q_prob

qp = gibbs_track_q(p3, q3, rng=np.random.default_rng(11))
pred = np.where(qp > 0.5, 1, -1)
acc = (pred == q3).mean()

# accuracy vs c/sigma ratio
ratios, accs = [], []
for c0 in [0.01, 0.02, 0.04, 0.06, 0.1, 0.15]:
    r3 = np.random.default_rng(int(c0*1e4)+3)
    pp, qq, _ = simulate_roll(300, c0, 0.02, r3)
    qprob = gibbs_track_q(pp, qq, n_iter=500, burn=150, rng=np.random.default_rng(int(c0*1e4)))
    pr = np.where(qprob > 0.5, 1, -1)
    ratios.append(c0/0.02); accs.append((pr == qq).mean())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
n_show = 60
x = np.arange(n_show)
axes[0].bar(x, np.where(q3[:n_show] > 0, 1, -1), color=["#2ca02c" if v > 0 else "#d62728" for v in q3[:n_show]],
            alpha=0.35, label="真实方向")
axes[0].plot(x, qp[:n_show]*2 - 1, "o-", ms=3.5, lw=0.8, color="#1f77b4", label="后验 P(买)×2−1")
axes[0].set_title(f"隐藏交易方向的后验恢复（前 60 笔，整段准确率 {acc:.1%}）")
axes[0].set_xlabel("交易序号"); axes[0].set_ylim(-1.3, 1.3)
axes[0].legend(fontsize=9)

axes[1].plot(ratios, accs, "o-", color="#1f77b4")
axes[1].axhline(0.5, color="gray", ls=":", label="随机猜（50%）")
axes[1].set_xlabel("信噪比 c / σ_u")
axes[1].set_ylabel("方向恢复准确率")
axes[1].set_title("价差越大（相对基本面噪声），方向越容易被认出来")
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/gs-direction-recovery.png", dpi=130)
plt.close()
print("chart3 done, acc:", acc, list(zip(ratios, accs)))
