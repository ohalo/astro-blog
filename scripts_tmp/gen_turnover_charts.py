#!/usr/bin/env python3
"""换手率约束组合优化 配图生成"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

rng = np.random.default_rng(42)
OUT = "/Users/halo/workspace/astro-blog/public/images/turnover-constrained-optimization"
os.makedirs(OUT, exist_ok=True)

# ---------- 合成数据：5 资产，252*5 天 ----------
N, T = 5, 252 * 5
true_mu = np.array([0.08, 0.06, 0.10, 0.04, 0.07]) / 252
vols = np.array([0.18, 0.14, 0.25, 0.08, 0.16]) / np.sqrt(252)
C = np.array([
    [1.0, 0.5, 0.6, 0.1, 0.4],
    [0.5, 1.0, 0.4, 0.2, 0.3],
    [0.6, 0.4, 1.0, 0.0, 0.5],
    [0.1, 0.2, 0.0, 1.0, 0.1],
    [0.4, 0.3, 0.5, 0.1, 1.0],
])
cov = np.outer(vols, vols) * C
L = np.linalg.cholesky(cov)
def gen_rets(seed):
    r = np.random.default_rng(seed)
    return true_mu + r.standard_normal((T, N)) @ L.T

rets = gen_rets(42)

# ---------- 滚动估计 + 三种调仓策略 ----------
LOOKBACK = 126
REB = 21          # 每月调仓
COST = 0.003      # 单边 30bp（含冲击）

def mv_weights(mu, S, gamma=8.0):
    """无约束均值方差（多头归一化）"""
    w = np.linalg.solve(gamma * S, mu)
    w = np.clip(w, 0, None)
    s = w.sum()
    return w / s if s > 1e-12 else np.ones(len(mu)) / len(mu)

def mv_weights_turnover(mu, S, w_prev, lam, gamma=8.0, iters=2000):
    """带 L1 换手惩罚：max mu'w - gamma/2 w'Sw - lam*|w-w_prev|_1，近端投影梯度"""
    w = mv_weights(mu, S, gamma) if lam == 0 else w_prev.copy()
    lr = 0.8
    for _ in range(iters):
        grad = mu - gamma * (S @ w)
        w_new = w + lr * grad
        # soft-threshold 朝 w_prev 收缩（近端算子）
        d = w_new - w_prev
        d = np.sign(d) * np.maximum(np.abs(d) - lr * lam, 0.0)
        w_new = w_prev + d
        w_new = np.clip(w_new, 0, None)
        s = w_new.sum()
        w_new = w_new / s if s > 1e-12 else np.ones(len(mu)) / len(mu)
        w = w_new
    return w

def run(policy, lam=0.0, band=0.0, cost=None, data=None):
    cost_rate = COST if cost is None else cost
    R = rets if data is None else data
    w = np.ones(N) / N
    eq_g, eq_n = [1.0], [1.0]
    turns = []
    for t in range(LOOKBACK, T):
        if (t - LOOKBACK) % REB == 0:
            mu_h = R[t - LOOKBACK:t].mean(axis=0)
            S_h = np.cov(R[t - LOOKBACK:t].T) + np.eye(N) * 1e-8
            if policy == "naive":
                w_tgt = mv_weights(mu_h, S_h)
            elif policy == "penalty":
                w_tgt = mv_weights_turnover(mu_h, S_h, w, lam)
            elif policy == "band":
                w_star = mv_weights(mu_h, S_h)
                w_tgt = w.copy()
                move = np.abs(w_star - w) > band
                w_tgt[move] = w_star[move]
                s = w_tgt.sum()
                w_tgt = w_tgt / s
            turn = np.abs(w_tgt - w).sum()
            turns.append(turn)
            cost_t = cost_rate * turn
            w = w_tgt
        else:
            cost_t = 0.0
        r = float(w @ R[t])
        eq_g.append(eq_g[-1] * (1 + r))
        eq_n.append(eq_n[-1] * (1 + r - cost_t))
        # 权重漂移
        w = w * (1 + R[t])
        w = w / w.sum()
    return np.array(eq_g), np.array(eq_n), np.array(turns)

def stats(eq):
    r = np.diff(eq) / eq[:-1]
    ann = (eq[-1] / eq[0]) ** (252 / len(r)) - 1
    sharpe = r.mean() / (r.std() + 1e-12) * np.sqrt(252)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return ann, sharpe, dd

eqg_naive, eqn_naive, turn_naive = run("naive")
eqg_pen, eqn_pen, turn_pen = run("penalty", lam=0.0005)
eqg_band, eqn_band, turn_band = run("band", band=0.10)

print("=== 年均换手（双边, 单次调仓均值 x 12） ===")
for name, tu in [("朴素 MV", turn_naive), ("λ=0.0005 惩罚", turn_pen), ("10% 无交易带", turn_band)]:
    print(f"{name}: 单次均值 {tu.mean():.3f}, 年化 {tu.mean()*12:.2f}")

print("=== 净值统计（净费后） ===")
for name, eq in [("朴素 MV", eqn_naive), ("惩罚", eqn_pen), ("无交易带", eqn_band)]:
    a, s, d = stats(eq)
    print(f"{name}: 年化 {a:.2%}, Sharpe {s:.2f}, MDD {d:.2%}")
print("=== 毛收益统计 ===")
for name, eq in [("朴素 MV", eqg_naive), ("惩罚", eqg_pen), ("无交易带", eqg_band)]:
    a, s, d = stats(eq)
    print(f"{name}: 年化 {a:.2%}, Sharpe {s:.2f}, MDD {d:.2%}")

# ---------- 图1：净值对比 ----------
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})
x = np.arange(len(eqn_naive))
axes[0].plot(x, eqn_naive, label="朴素 MV（月度全量调仓）", color="#d62728", lw=1.4)
axes[0].plot(x, eqn_pen, label="L1 换手惩罚 λ=0.0005", color="#1f77b4", lw=1.4)
axes[0].plot(x, eqn_band, label="10% 无交易带", color="#2ca02c", lw=1.4)
axes[0].set_title("净费后净值对比（单边成本 30bp）")
axes[0].set_ylabel("净值")
axes[0].legend()
axes[0].grid(alpha=0.3)
gap_pen = eqn_pen / eqn_naive - 1
gap_band = eqn_band / eqn_naive - 1
axes[1].plot(x, gap_pen * 100, color="#1f77b4", lw=1.2, label="惩罚 vs 朴素")
axes[1].plot(x, gap_band * 100, color="#2ca02c", lw=1.2, label="无交易带 vs 朴素")
axes[1].axhline(0, color="gray", lw=0.8)
axes[1].set_ylabel("相对朴素 MV (%)")
axes[1].set_xlabel("交易日")
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/turnover_equity_net.png", dpi=110)
plt.close()

# ---------- 图2：换手率分布对比 ----------
fig, ax = plt.subplots(figsize=(10, 5))
idx = np.arange(len(turn_naive))
ax.plot(idx, np.cumsum(turn_naive), label=f"朴素 MV（累计 {turn_naive.sum():.1f}）", color="#d62728", lw=1.5)
ax.plot(idx, np.cumsum(turn_pen), label=f"L1 惩罚（累计 {turn_pen.sum():.1f}）", color="#1f77b4", lw=1.5)
ax.plot(idx, np.cumsum(turn_band), label=f"无交易带（累计 {turn_band.sum():.1f}）", color="#2ca02c", lw=1.5)
ax.set_title("累计换手对比：估计噪声让朴素 MV 每月大幅倒仓")
ax.set_xlabel("调仓次数（月）")
ax.set_ylabel("累计换手（∑|Δw|）")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/turnover_cumsum.png", dpi=110)
plt.close()

# ---------- 图3：λ 扫描（换手-净收益前沿，5 种子平均） ----------
lams = [0.0, 0.0002, 0.0005, 0.001, 0.002, 0.004]
seeds = [42, 43, 44, 45, 46]
datasets = [gen_rets(s) for s in seeds]
res = []
for lam in lams:
    anns, shs, tus = [], [], []
    for D in datasets:
        _, eqn, tu = run("penalty", lam=lam, data=D)
        a, s, d = stats(eqn)
        anns.append(a); shs.append(s); tus.append(tu.mean() * 12)
    res.append((lam, np.mean(tus), np.mean(anns), np.mean(shs)))
    print(f"lam={lam}: 年化换手 {np.mean(tus):.2f}, 净年化 {np.mean(anns):.2%}, Sharpe {np.mean(shs):.2f}")

res = np.array(res)
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(res[:, 1], res[:, 2] * 100, "o-", color="#1f77b4", lw=1.5)
for i, lam in enumerate(lams):
    ax1.annotate(f"λ={lam}", (res[i, 1], res[i, 2] * 100),
                 textcoords="offset points", xytext=(6, 6), fontsize=9)
ax1.set_xlabel("年化换手（∑|Δw|）")
ax1.set_ylabel("净费后年化收益 (%)")
ax1.set_title("换手-净收益前沿（5 组独立模拟平均）：λ 加大 → 换手骤降")
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/turnover_lambda_frontier.png", dpi=110)
plt.close()

# ---------- 图4：成本敏感性 ----------
costs = [0.0005, 0.0015, 0.003, 0.005]
width = 0.25
fig, ax = plt.subplots(figsize=(10, 5))
labels = ["朴素 MV", "L1 惩罚 λ=0.0005", "10% 无交易带"]
colors = ["#d62728", "#1f77b4", "#2ca02c"]
annret = {k: [] for k in labels}
for c in costs:
    for name, args in zip(labels, [("naive", {}), ("penalty", {"lam": 0.0005}), ("band", {"band": 0.10})]):
        _, eqn, _ = run(args[0], cost=c, **args[1])
        a, _, _ = stats(eqn)
        annret[name].append(a * 100)
xpos = np.arange(len(costs))
for i, name in enumerate(labels):
    ax.bar(xpos + (i - 1) * width, annret[name], width, label=name, color=colors[i], alpha=0.85)
ax.set_xticks(xpos)
ax.set_xticklabels([f"{c*1e4:.0f}bp" for c in costs])
ax.set_xlabel("单边交易成本")
ax.set_ylabel("净费后年化收益 (%)")
ax.set_title("成本敏感性：成本越高，换手控制的相对优势越大")
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/turnover_cost_sensitivity.png", dpi=110)
plt.close()
print("saved:", os.listdir(OUT))
