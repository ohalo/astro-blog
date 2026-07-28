#!/usr/bin/env python3
"""Glosten-Milgrom 序贯交易模型配图生成"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = ["PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "..", "public", "images", "glosten-milgrom-spread")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(11)

# ---------- GM 模型核心 ----------
# 资产真值 V ∈ {V_L, V_H}，做市商先验 p = P(V=V_H)
# 知情者占比 mu：知情者永远朝真值方向交易；不知情者 50/50 买卖
V_L, V_H = 98.0, 102.0

def gm_quotes(p, mu):
    """贝叶斯报价：ask = E[V|买单], bid = E[V|卖单]"""
    # P(buy) = mu*p + (1-mu)/2   （知情者只在 V=H 时买）
    p_buy = mu * p + (1 - mu) / 2
    p_sell = mu * (1 - p) + (1 - mu) / 2
    # 后验
    p_up_buy = (mu * p + (1 - mu) / 2 * p) / p_buy
    p_up_sell = ((1 - mu) / 2 * p) / p_sell
    ask = p_up_buy * V_H + (1 - p_up_buy) * V_L
    bid = p_up_sell * V_H + (1 - p_up_sell) * V_L
    return bid, ask, p_up_buy, p_up_sell

# ============ 图1：价差 vs 知情者比例 mu ============
mus = np.linspace(0, 0.95, 60)
spreads_half = []
for mu in mus:
    b, a, _, _ = gm_quotes(0.5, mu)
    spreads_half.append(a - b)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(mus, spreads_half, color="#2563eb", lw=2.5)
ax.fill_between(mus, 0, spreads_half, alpha=0.12, color="#2563eb")
for mu0 in [0.1, 0.3, 0.6]:
    b, a, _, _ = gm_quotes(0.5, mu0)
    ax.plot(mu0, a - b, "o", color="#dc2626", ms=7)
    ax.annotate(f"μ={mu0}: 价差 {a-b:.2f}", (mu0, a - b),
                textcoords="offset points", xytext=(10, -14), fontsize=10)
ax.set_xlabel("知情交易者比例 μ")
ax.set_ylabel("买卖价差 (ask − bid)")
ax.set_title("GM 模型：价差随知情者比例单调放大（p=0.5, V∈{98,102}）", fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "spread-vs-mu.png"), dpi=150)
plt.close()

# ============ 图2：单路径模拟——报价收敛到真值 ============
def simulate_path(mu, V_true, n_trades, seed):
    r = np.random.default_rng(seed)
    p = 0.5
    mids, bids, asks, ps = [], [], [], []
    for k in range(n_trades):
        bid, ask, p_up_buy, p_up_sell = gm_quotes(p, mu)
        informed = r.random() < mu
        if informed:
            side = 1 if V_true == V_H else -1
        else:
            side = 1 if r.random() < 0.5 else -1
        p = p_up_buy if side == 1 else p_up_sell
        bids.append(bid); asks.append(ask); ps.append(p)
        mids.append((bid + ask) / 2)
    return np.array(bids), np.array(asks), np.array(mids), np.array(ps)

n_tr = 200
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
for ax, mu0, ttl in zip(axes, [0.15, 0.5], ["μ=0.15（知情者少）", "μ=0.5（知情者多）"]):
    b, a, m, ps = simulate_path(mu0, V_H, n_tr, seed=3)
    ax.fill_between(np.arange(n_tr), b, a, alpha=0.25, color="#2563eb", label="买卖报价区间")
    ax.plot(m, color="#1e293b", lw=1.4, label="中间价")
    ax.axhline(V_H, color="#16a34a", ls="--", lw=1.3, label="真值 V=102")
    ax.set_xlabel("成交笔数")
    ax.set_title(ttl, fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("价格")
fig.suptitle("知情者越多，价格发现越快、但每一步价差越宽", fontsize=12.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "price-discovery-paths.png"), dpi=150)
plt.close()

# ============ 图3：做市商损益分解（对知情 vs 不知情） ============
def pnl_decompose(mu, V_true, n_trades, n_sims):
    pnl_inf, pnl_uninf = [], []
    for s in range(n_sims):
        r = np.random.default_rng(10_000 + s)
        p = 0.5
        pi, pu = 0.0, 0.0
        for k in range(n_trades):
            bid, ask, p_up_buy, p_up_sell = gm_quotes(p, mu)
            informed = r.random() < mu
            if informed:
                side = 1 if V_true == V_H else -1
            else:
                side = 1 if r.random() < 0.5 else -1
            px = ask if side == 1 else bid
            # 做市商站在对手方：客户买 -> 做市商卖出，损益 = px - V_true
            mm_pnl = (px - V_true) if side == 1 else (V_true - px)
            if informed:
                pi += mm_pnl
            else:
                pu += mm_pnl
            p = p_up_buy if side == 1 else p_up_sell
        pnl_inf.append(pi); pnl_uninf.append(pu)
    return np.mean(pnl_inf), np.mean(pnl_uninf)

mus3 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
inf_l, uninf_l, tot_l = [], [], []
for mu0 in mus3:
    a_, b_ = pnl_decompose(mu0, V_H, 100, 400)
    inf_l.append(a_); uninf_l.append(b_); tot_l.append(a_ + b_)

x = np.arange(len(mus3))
w = 0.28
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w, inf_l, w, color="#dc2626", label="对知情者损益（逆向选择成本）")
ax.bar(x, uninf_l, w, color="#16a34a", label="对不知情者损益（价差收入）")
ax.bar(x + w, tot_l, w, color="#64748b", label="合计")
ax.axhline(0, color="k", lw=1)
ax.set_xticks(x); ax.set_xticklabels([f"μ={m}" for m in mus3])
ax.set_ylabel("做市商累计损益（100 笔，400 次平均）")
ax.set_title("做市商的生意经：赚不知情者的价差，赔知情者的信息", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "mm-pnl-decomposition.png"), dpi=150)
plt.close()

# ============ 图4：价差衰减——信息逐渐进价 ============
fig, ax = plt.subplots(figsize=(9, 5))
for mu0, c in zip([0.15, 0.3, 0.5], ["#94a3b8", "#2563eb", "#dc2626"]):
    sp = np.zeros(n_tr)
    for s in range(300):
        b, a, m, ps = simulate_path(mu0, V_H if s % 2 == 0 else V_L, n_tr, seed=20_000 + s)
        sp += (a - b)
    sp /= 300
    ax.plot(sp, color=c, lw=2, label=f"μ={mu0}")
ax.set_xlabel("成交笔数")
ax.set_ylabel("平均价差")
ax.set_title("价差随成交衰减：信息进价后，逆向选择风险消失", fontsize=12)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "spread-decay.png"), dpi=150)
plt.close()

print("GM images done:", os.listdir(OUT))
for mu0 in [0.1, 0.3, 0.6]:
    b, a, _, _ = gm_quotes(0.5, mu0)
    print(f"mu={mu0}: bid={b:.3f} ask={a:.3f} spread={a-b:.3f}")
print("pnl decompose:", list(zip(mus3, [f"{v:.1f}" for v in inf_l], [f"{v:.1f}" for v in uninf_l], [f"{v:.1f}" for v in tot_l])))
