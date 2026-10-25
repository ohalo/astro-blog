#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 2026-10-25（第三弹）两篇量化文章的核心数值 + 配图（固定 seed 可复现）。
  A. market-maker-deep-rl-inventory  做市商深度强化学习库存控制：用 POMDP 平衡存货与价差
  B. vol-surface-butterfly-risk      波动率曲面 Butterfly 风险：中段凸性的均值回归交易
所有图表均为真实计算图，数值固定随机种子可复现。
记账口径：做市商是被动方——主动买单打穿 ask（做市商卖出），主动卖单打穿 bid（做市商买入）。
A：tick=0.01，基本面每步波动 1.5-4.5 tick（日波动约 0.6%），知情单毒性流出 20%。
B：Heston 标的状态 + 风格化微笑（sk*k + c*k^2 乘性）；曲率缺口 z-score 信号；
   头寸 = 蝴蝶 + 每日再平衡的 ATM 跨式 vega 对冲（隔离凸性暴露）。
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

for f in ["Heiti TC", "PingFang SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130
rcParams["figure.facecolor"] = "white"

BASE = "/Users/halo/workspace/astro-blog/public/images"

# ============================================================
# 文章 A：做市商深度强化学习库存控制（DQN，numpy 从零实现）
# ============================================================
def gen_mm_dqn():
    slug = "market-maker-deep-rl-inventory"
    OUT = os.path.join(BASE, slug)
    os.makedirs(OUT, exist_ok=True)

    TICK = 0.01
    T = 390                       # 一个交易日（分钟）
    QMAX = 30                     # 库存硬闸（股）
    SIG = [0.015, 0.045]          # 低波 / 高波 regime 每步波动（1.5 / 4.5 tick）
    P_STAY = 0.995
    JUMP_P = 0.002                # 跳变概率（每步）
    JUMP_TICKS = 12               # 跳变幅度（tick）
    A_HIT = 0.65                  # 噪声单零价差到达强度（每侧每步）
    KAPPA = 0.5                   # 价差弹性（每 tick 衰减）
    P_INF = 0.20                  # 知情者优势 >2.5 tick 时的下单概率
    INF_EDGE = 2.5                # 知情者门槛（tick）
    INV_PEN = 1.0                 # 收盘库存惩罚（tick/股）

    SPREADS = [1.5, 2.0, 2.5, 3.0]
    SKEWS = [0.0, 0.10, 0.25]
    ACTIONS = [(s, k) for s in SPREADS for k in SKEWS]
    N_ACT = len(ACTIONS)

    AS_GAMMA, AS_KAPPA = 4e-4, 0.5
    REWARD_SCALE = 5.0

    def simulate_day(seed, policy):
        rng = np.random.default_rng(seed)
        v = 100.0
        vol_state = 0
        cash, q = 0.0, 0
        mid = v
        spread_income = 0.0
        hold_pnl = 0.0
        flow_imb = 0.0
        last_ret = 0.0
        n_fill = 0
        max_abs_q = 0
        vol_traded = 0.0
        v_path = np.empty(T)
        q_path = np.zeros(T)

        for t in range(T):
            v_path[t] = v
            q_path[t] = q
            vol_ratio = SIG[vol_state] / SIG[0]
            a = policy(t, q, vol_ratio, flow_imb, last_ret)
            hs, sk = ACTIONS[a]
            shift = -sk * q * TICK
            bid = mid - hs * TICK + shift
            ask = mid + hs * TICK + shift
            lam_buy = A_HIT * math.exp(-KAPPA * max(0.05, hs - sk * q))
            lam_sell = A_HIT * math.exp(-KAPPA * max(0.05, hs + sk * q))

            if rng.random() < P_STAY:
                pass
            else:
                vol_state = 1 - vol_state
            dv = SIG[vol_state] * rng.normal()
            if rng.random() < JUMP_P:
                dv += (JUMP_TICKS * TICK) * (1 if rng.random() < 0.5 else -1)
            v_new = v + dv
            hit = False
            d_cash = 0.0

            if rng.random() < lam_buy:              # 主动买单 -> 做市商卖
                qty = int(rng.integers(1, 4))
                d_cash += ask * qty; q -= qty
                spread_income += (ask - v) * qty
                flow_imb = 0.9 * flow_imb + 0.1
                n_fill += 1; vol_traded += qty; hit = True
            if rng.random() < lam_sell:             # 主动卖单 -> 做市商买
                qty = int(rng.integers(1, 4))
                d_cash -= bid * qty; q += qty
                spread_income += (v - bid) * qty
                flow_imb = 0.9 * flow_imb - 0.1
                n_fill += 1; vol_traded += qty; hit = True
            if v_new > ask + INF_EDGE * TICK and rng.random() < P_INF:   # 知情买单
                qty = int(rng.integers(1, 4))
                d_cash += ask * qty; q -= qty
                spread_income += (ask - v) * qty
                flow_imb = 0.9 * flow_imb + 0.1
                n_fill += 1; vol_traded += qty; hit = True
            elif v_new < bid - INF_EDGE * TICK and rng.random() < P_INF:  # 知情卖单
                qty = int(rng.integers(1, 4))
                d_cash -= bid * qty; q += qty
                spread_income += (v - bid) * qty
                flow_imb = 0.9 * flow_imb - 0.1
                n_fill += 1; vol_traded += qty; hit = True

            if q > QMAX:
                d_cash += (mid - 2 * TICK) * (q - QMAX); q = QMAX
            elif q < -QMAX:
                d_cash += (mid + 2 * TICK) * (q + QMAX); q = -QMAX

            cash += d_cash
            hold_pnl += q * (v_new - v)
            max_abs_q = max(max_abs_q, abs(q))
            mid = v_new if hit else 0.98 * mid + 0.02 * v_new
            v = v_new

        liq_cost = INV_PEN * TICK * abs(q)
        return dict(total_pnl=cash + q * v - liq_cost, spread_income=spread_income,
                    hold_pnl=hold_pnl, n_fill=n_fill, max_abs_q=max_abs_q,
                    vol_traded=vol_traded, final_q=q, v_path=v_path, q_path=q_path)

    def pol_symmetric(t, q, vr, fi, lr):
        return ACTIONS.index((2.0, 0.0))

    def pol_as(t, q, vr, fi, lr):
        """Avellaneda-Stolz 闭式解（tick 单位）。"""
        tau = T - t
        sig_ticks = (SIG[1] if vr > 1.5 else SIG[0]) / TICK
        g = AS_GAMMA
        spread = g * sig_ticks ** 2 * tau + (2.0 / g) * math.log(1 + g / AS_KAPPA)
        hs = max(SPREADS[0], min(SPREADS[-1], spread / 2.0))
        sk = min(SKEWS[-1], g * sig_ticks ** 2 * tau)
        best = min(ACTIONS, key=lambda a: abs(a[0] - hs) + abs(a[1] - sk) * 3)
        return ACTIONS.index(best)

    class DQN:
        def __init__(self, n_feat, n_act, hidden=32, lr=2e-4, seed=7):
            rng = np.random.default_rng(seed)
            self.W1 = rng.normal(0, 1 / math.sqrt(n_feat), (n_feat, hidden))
            self.b1 = np.zeros(hidden)
            self.W2 = rng.normal(0, 1 / math.sqrt(hidden), (hidden, n_act))
            self.b2 = np.zeros(n_act)
            self.lr = lr
            self._sync_target()

        def _sync_target(self):
            self.tW1 = self.W1.copy(); self.tb1 = self.b1.copy()
            self.tW2 = self.W2.copy(); self.tb2 = self.b2.copy()

        def qvals(self, x, target=False):
            W1, b1, W2, b2 = ((self.tW1, self.tb1, self.tW2, self.tb2)
                              if target else (self.W1, self.b1, self.W2, self.b2))
            h = np.maximum(0.0, x @ W1 + b1)
            return h @ W2 + b2

        def update(self, x, a, r, x2, done):
            x = np.atleast_2d(x); x2 = np.atleast_2d(x2)
            a = np.atleast_1d(a); r = np.atleast_1d(r); done = np.atleast_1d(done)
            y = r + 0.99 * self.qvals(x2, target=True).max(axis=1) * (1 - done)
            q_now = self.qvals(x).copy()
            B = x.shape[0]
            q_now[np.arange(B), a] = y
            h = np.maximum(0.0, x @ self.W1 + self.b1)
            dq = (self.qvals(x) - q_now) / B
            gW2 = h.T @ dq; gb2 = dq.sum(0)
            dh = (dq @ self.W2.T) * (h > 0)
            gW1 = x.T @ dh; gb1 = dh.sum(0)
            for g in (gW1, gW2):
                np.clip(g, -1.0, 1.0, out=g)
            self.W1 -= self.lr * gW1; self.b1 -= self.lr * gb1
            self.W2 -= self.lr * gW2; self.b2 -= self.lr * gb2

    NF = 5

    def feat(q, t, vr, fi, lr):
        return np.array([q / QMAX, (T - t) / T, vr - 1.0, fi, lr])

    # ---------- 训练 ----------
    net = DQN(NF, N_ACT)
    buf = []
    BUF_MAX = 8000
    rng = np.random.default_rng(20261025)
    eval_seeds = [20260101 + i * 7919 for i in range(8)]
    EPS0, EPS_END, EPS_DECAY = 1.0, 0.05, 0.99
    N_TRAIN = 350
    BATCH = 64
    eps = EPS0
    train_curve = []

    for ep in range(N_TRAIN):
        day_rng = np.random.default_rng(70000 + ep)
        v = 100.0; vol_state = 0; cash = 0.0; q = 0; mid = v
        flow_imb = 0.0; last_ret = 0.0
        ep_pnl = 0.0
        for t in range(T):
            vr = SIG[vol_state] / SIG[0]
            x = feat(q, t, vr, flow_imb, last_ret)
            if rng.random() < eps:
                a = int(rng.integers(N_ACT))
            else:
                a = int(np.argmax(net.qvals(x)))
            hs, sk = ACTIONS[a]
            shift = -sk * q * TICK
            bid = mid - hs * TICK + shift
            ask = mid + hs * TICK + shift
            lam_buy = A_HIT * math.exp(-KAPPA * max(0.05, hs - sk * q))
            lam_sell = A_HIT * math.exp(-KAPPA * max(0.05, hs + sk * q))
            if day_rng.random() < P_STAY:
                pass
            else:
                vol_state = 1 - vol_state
            dv = SIG[vol_state] * day_rng.normal()
            if day_rng.random() < JUMP_P:
                dv += (JUMP_TICKS * TICK) * (1 if day_rng.random() < 0.5 else -1)
            v_new = v + dv
            d_cash = 0.0
            if day_rng.random() < lam_buy:
                qty = int(day_rng.integers(1, 4)); d_cash += ask * qty; q -= qty
                flow_imb = 0.9 * flow_imb + 0.1
            if day_rng.random() < lam_sell:
                qty = int(day_rng.integers(1, 4)); d_cash -= bid * qty; q += qty
                flow_imb = 0.9 * flow_imb - 0.1
            if v_new > ask + INF_EDGE * TICK and day_rng.random() < P_INF:
                qty = int(day_rng.integers(1, 4)); d_cash += ask * qty; q -= qty
                flow_imb = 0.9 * flow_imb + 0.1
            elif v_new < bid - INF_EDGE * TICK and day_rng.random() < P_INF:
                qty = int(day_rng.integers(1, 4)); d_cash -= bid * qty; q += qty
                flow_imb = 0.9 * flow_imb - 0.1
            if q > QMAX:
                d_cash += (mid - 2 * TICK) * (q - QMAX); q = QMAX
            elif q < -QMAX:
                d_cash += (mid + 2 * TICK) * (q + QMAX); q = -QMAX
            r = (d_cash + q * (v_new - v)) / REWARD_SCALE
            ep_pnl += r * REWARD_SCALE
            x2 = feat(q, t + 1, vr, flow_imb, dv / (5 * TICK))
            done = (t == T - 1)
            if done:
                r -= INV_PEN * TICK * abs(q) / REWARD_SCALE
            buf.append((x, a, r, x2, done))
            if len(buf) > BUF_MAX:
                buf.pop(0)
            cash += d_cash
            mid = v_new if (d_cash != 0) else 0.98 * mid + 0.02 * v_new
            v = v_new
        for _ in range(20):
            if len(buf) < BATCH:
                break
            idx = rng.integers(0, len(buf), BATCH)
            net.update(np.array([buf[i][0] for i in idx]),
                       np.array([buf[i][1] for i in idx]),
                       np.array([buf[i][2] for i in idx]),
                       np.array([buf[i][3] for i in idx]),
                       np.array([buf[i][4] for i in idx]))
        if ep % 5 == 0:
            net._sync_target()
        eps = max(EPS_END, eps * EPS_DECAY)
        if ep % 25 == 0 or ep == N_TRAIN - 1:
            pnl_d, pnl_s, pnl_a = [], [], []
            for sd in eval_seeds:
                pnl_d.append(simulate_day(sd, lambda t, q, vr, fi, lr: int(np.argmax(net.qvals(feat(q, t, vr, fi, lr)))))["total_pnl"])
                pnl_s.append(simulate_day(sd, pol_symmetric)["total_pnl"])
                pnl_a.append(simulate_day(sd, pol_as)["total_pnl"])
            train_curve.append((ep, float(np.mean(pnl_d)), float(np.mean(pnl_s)),
                                float(np.mean(pnl_a)), float(ep_pnl)))
            print(f"[A] ep {ep:3d} eps {eps:.3f} | eval DQN {np.mean(pnl_d):7.3f} vs SYM {np.mean(pnl_s):7.3f} vs AS {np.mean(pnl_a):7.3f} | 训练日PnL(含探索) {ep_pnl:7.3f}")

    # ---------- 样本外评估 ----------
    N_EVAL = 60
    res = {"dqn": [], "sym": [], "as": []}
    for i in range(N_EVAL):
        sd = 555000 + i * 104729
        res["dqn"].append(simulate_day(sd, lambda t, q, vr, fi, lr: int(np.argmax(net.qvals(feat(q, t, vr, fi, lr))))))
        res["sym"].append(simulate_day(sd, pol_symmetric))
        res["as"].append(simulate_day(sd, pol_as))

    def agg(key, field):
        return np.array([r[field] for r in res[key]])

    print("\n===== 文章 A 样本外统计（60 个独立交易日）=====")
    stats = {}
    for name, cn in [("sym", "对称做市"), ("as", "AS 闭式"), ("dqn", "DQN")]:
        pnl, si, hp = agg(name, "total_pnl"), agg(name, "spread_income"), agg(name, "hold_pnl")
        nf, mq, vt = agg(name, "n_fill"), agg(name, "max_abs_q"), agg(name, "vol_traded")
        stats[name] = dict(pnl=float(pnl.mean()), std=float(pnl.std()),
                           sharpe=float(pnl.mean() / max(pnl.std(), 1e-9)),
                           si=float(si.mean()), hp=float(hp.mean()), fills=float(nf.mean()),
                           maxq=float(mq.mean()), vol=float(vt.mean()), winrate=float((pnl > 0).mean()))
        print(f"{cn}: 日均PnL {pnl.mean():7.3f} | std {pnl.std():6.3f} | Sharpe {pnl.mean()/max(pnl.std(),1e-9):6.2f} "
              f"| 价差收入 {si.mean():7.3f} | 持仓盈亏 {hp.mean():7.3f} | 成交 {nf.mean():6.1f} 笔 "
              f"| 最大|库存| {mq.mean():5.1f} | 成交量 {vt.mean():6.0f} | 盈利日 {(pnl>0).mean()*100:4.0f}%")
    pnl_d, pnl_s, pnl_a = agg("dqn", "total_pnl"), agg("sym", "total_pnl"), agg("as", "total_pnl")
    print(f"DQN-SYM: {pnl_d.mean()-pnl_s.mean():+.3f} ({100*(pnl_d.mean()-pnl_s.mean())/max(abs(pnl_s.mean()),1e-9):+.0f}%) | DQN-AS: {pnl_d.mean()-pnl_a.mean():+.3f} | 胜率vs SYM {(pnl_d>pnl_s).mean()*100:.0f}% | 胜率vs AS {(pnl_d>pnl_a).mean()*100:.0f}%")

    print("\n===== 学到的策略形态（贪心，半日扫描）=====")
    for vr, tag in [(1.0, "低波"), (3.0, "高波")]:
        acts = [ACTIONS[int(np.argmax(net.qvals(feat(q, T // 2, vr, 0.0, 0.0))))] for q in range(-QMAX, QMAX + 1, 5)]
        print(f"{tag}: 半价差 {[a[0] for a in acts]}")
        print(f"{tag}: 偏移   {[a[1] for a in acts]}")

    # ---------- 图 1：训练曲线 ----------
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    xs = [c[0] for c in train_curve]
    ax.plot(xs, [c[4] for c in train_curve], "-", color="#bdc3c7", lw=1.0,
            label="训练中日收益（含 ε 探索）")
    ax.plot(xs, [c[1] for c in train_curve], "o-", color="#c0392b", lw=1.8, label="DQN 做市商（贪心评估）")
    ax.plot(xs, [c[2] for c in train_curve], "s--", color="#7f8c8d", lw=1.5, label="对称做市（同路径）")
    ax.plot(xs, [c[3] for c in train_curve], "^:", color="#2980b9", lw=1.5, label="AS 闭式（同路径）")
    ax.set_xlabel("训练回合（模拟交易日）"); ax.set_ylabel("日均 PnL（元）")
    ax.set_title("训练过程：从随机报价到追平闭式解")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "training_curve.png")); plt.close(fig)

    # ---------- 图 2：库存路径 ----------
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
    for ax, key, title, color in [
        (axes[0], "sym", "对称做市：报价不理会库存", "#7f8c8d"),
        (axes[1], "dqn", "DQN：库存偏离即偏移报价", "#c0392b"),
    ]:
        ax.plot(res[key][0]["q_path"], color=color, lw=1.1)
        ax.axhline(0, color="black", lw=0.6, alpha=0.4)
        ax.set_title(title); ax.set_xlabel("交易日内的步数"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("库存（股）")
    fig.suptitle("同一订单流下的库存路径（样本外第 1 日）")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "inventory_paths.png")); plt.close(fig)

    # ---------- 图 3：学到的策略形态 ----------
    qs = np.arange(-QMAX, QMAX + 1, 2)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for vr, tag, color in [(1.0, "低波 regime", "#2980b9"), (3.0, "高波 regime", "#c0392b")]:
        acts = [ACTIONS[int(np.argmax(net.qvals(feat(q, T // 2, vr, 0.0, 0.0))))] for q in qs]
        axes[0].plot(qs, [a[0] for a in acts], "o-", ms=3, color=color, label=tag)
        axes[1].plot(qs, [a[1] for a in acts], "s-", ms=3, color=color, label=tag)
    axes[0].set_ylabel("半价差（tick）"); axes[0].set_title("库存 → 半价差")
    axes[1].set_ylabel("库存偏移（tick/股）"); axes[1].set_title("库存 → 报价偏移")
    for ax in axes:
        ax.set_xlabel("当前库存（股）"); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("贪心策略扫描：高波挂宽、多头往下偏")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "learned_policy.png")); plt.close(fig)

    # ---------- 图 4：三策略对比 ----------
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    data = [pnl_s, pnl_a, pnl_d]
    labels = ["对称做市", "AS 闭式", "DQN"]
    colors = ["#7f8c8d", "#2980b9", "#c0392b"]
    bp = axes[0].boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    axes[0].axhline(0, color="black", lw=0.7, ls="--", alpha=0.5)
    axes[0].set_ylabel("单日 PnL（元）"); axes[0].set_title("60 个样本外交易日 PnL 分布")
    xpos = np.arange(3)
    means = [d.mean() for d in data]; stds = [d.std() for d in data]
    axes[1].bar(xpos, means, yerr=stds, color=colors, alpha=0.75, capsize=5)
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[1].text(i, m + s + 0.15, f"{m:.2f}", ha="center", fontsize=9)
    axes[1].set_xticks(xpos); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("日均 PnL ± 1 std（元）"); axes[1].set_title("均值与波动")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "strategy_comparison.png")); plt.close(fig)

    # ---------- 图 5：盈亏分解 ----------
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    labels = ["对称做市", "AS 闭式", "DQN"]
    si = [stats[k]["si"] for k in ("sym", "as", "dqn")]
    hp = [stats[k]["hp"] for k in ("sym", "as", "dqn")]
    x = np.arange(3)
    ax.bar(x, si, 0.5, label="价差收入", color="#27ae60", alpha=0.85)
    ax.bar(x, hp, 0.5, bottom=si, label="持仓盈亏（含逆向选择）", color="#e67e22", alpha=0.85)
    for i in range(3):
        ax.text(i, si[i] + hp[i] + 0.15, f"合计 {si[i]+hp[i]:.2f}", ha="center", fontsize=9)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("日均金额（元）")
    ax.set_title("盈亏分解：价差收入人人都赚，差距全在持仓盈亏")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pnl_decomposition.png")); plt.close(fig)

    return {"train_curve": train_curve, "stats": stats,
            "winrate_vs_sym": float((pnl_d > pnl_s).mean()),
            "winrate_vs_as": float((pnl_d > pnl_a).mean())}


# ============================================================
# 文章 B：波动率曲面 Butterfly 风险（中段凸性均值回归）
# ============================================================
def gen_butterfly():
    slug = "vol-surface-butterfly-risk"
    OUT = os.path.join(BASE, slug)
    os.makedirs(OUT, exist_ok=True)

    from math import log, sqrt, exp, erf

    def norm_cdf(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def norm_pdf(x):
        return exp(-0.5 * x * x) / sqrt(2 * math.pi)

    def bs_call(S, K, T, r, sig):
        if T <= 0:
            return max(S - K, 0.0)
        if sig <= 1e-9:
            return max(S - K * exp(-r * T), 0.0)
        d1 = (log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * sqrt(T))
        d2 = d1 - sig * sqrt(T)
        return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)

    def bs_put(S, K, T, r, sig):
        return bs_call(S, K, T, r, sig) - S + K * exp(-r * T)

    def bs_vega(S, K, T, r, sig):
        if T <= 0 or sig <= 1e-9:
            return 0.0
        d1 = (log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * sqrt(T))
        return S * norm_pdf(d1) * sqrt(T)

    # ---------- 市场动力学：Heston 标的状态 ----------
    N_DAYS = 500
    DT = 1.0 / 252
    rng = np.random.default_rng(424242)
    S0, V0 = 100.0, 0.040
    KAPPA, THETA, XI, RHO = 2.0, 0.040, 0.40, -0.7
    R = 0.02
    S = np.empty(N_DAYS); V = np.empty(N_DAYS)
    S[0], V[0] = S0, V0
    for t in range(1, N_DAYS):
        z1 = rng.normal(); z2 = RHO * z1 + sqrt(1 - RHO ** 2) * rng.normal()
        V[t] = max(V[t - 1] + KAPPA * (THETA - V[t - 1]) * DT
                   + XI * sqrt(max(V[t - 1], 1e-6)) * sqrt(DT) * z2, 0.005)
        S[t] = S[t - 1] * exp((R - 0.5 * V[t - 1]) * DT + sqrt(max(V[t - 1], 1e-6) * DT) * z1)

    # ---------- 微笑参数状态：偏斜 sk 与凸性 c（OU，c 是可交易对象） ----------
    SK_BAR, PHI_S, SIG_S = -0.60, 0.97, 0.05
    C_BAR, PHI, SIG_C = 14.0, 0.96, 0.80
    C = np.empty(N_DAYS); SK = np.empty(N_DAYS)
    C[0], SK[0] = C_BAR, SK_BAR
    n_regime_jumps = 0
    for t in range(1, N_DAYS):
        C[t] = C_BAR + PHI * (C[t - 1] - C_BAR) + SIG_C * rng.normal()
        SK[t] = SK_BAR + PHI_S * (SK[t - 1] - SK_BAR) + SIG_S * rng.normal()
        if rng.random() < 0.008:      # 微笑机制切换（凸性状态跳变）
            C[t] += 3.0 * (1 if rng.random() < 0.5 else -1)
            n_regime_jumps += 1

    TAU = 30 / 252
    DK = 0.10
    COST = 0.0002   # 蝴蝶单边成本 2bp（占 S）
    HEDGE_COST = 0.00005  # 每日对冲再平衡成本 0.5bp（占 S）

    def surface_sig(v_t, c_t, sk_t, k):
        return sqrt(max(v_t, 1e-6)) * (1 + sk_t * k + c_t * k * k)

    def fly_legs(S_t, v_t, c_t, sk_t):
        K1, K2, K3 = S_t * exp(-DK), S_t, S_t * exp(DK)
        s1 = surface_sig(v_t, c_t, sk_t, -DK)
        s2 = surface_sig(v_t, c_t, sk_t, 0.0)
        s3 = surface_sig(v_t, c_t, sk_t, DK)
        return (K1, s1), (K2, s2), (K3, s3)

    def fly_premium(S_t, v_t, c_t, sk_t):
        (K1, s1), (K2, s2), (K3, s3) = fly_legs(S_t, v_t, c_t, sk_t)
        return (bs_call(S_t, K1, TAU, R, s1) - 2 * bs_call(S_t, K2, TAU, R, s2)
                + bs_call(S_t, K3, TAU, R, s3))

    def fly_vega(S_t, v_t, c_t, sk_t):
        (K1, s1), (K2, s2), (K3, s3) = fly_legs(S_t, v_t, c_t, sk_t)
        return (bs_vega(S_t, K1, TAU, R, s1) - 2 * bs_vega(S_t, K2, TAU, R, s2)
                + bs_vega(S_t, K3, TAU, R, s3))

    def straddle_premium(S_t, v_t, c_t, sk_t):
        (K2, s2) = fly_legs(S_t, v_t, c_t, sk_t)[1]
        return bs_call(S_t, K2, TAU, R, s2) + bs_put(S_t, K2, TAU, R, s2)

    def curvature(v_t, c_t, sk_t):
        """中段凸性缺口：两翼平均高出 ATM 的部分（vol 点）。"""
        return (0.5 * (surface_sig(v_t, c_t, sk_t, -DK) + surface_sig(v_t, c_t, sk_t, DK))
                - surface_sig(v_t, c_t, sk_t, 0.0))

    prem = np.array([fly_premium(S[t], V[t], C[t], SK[t]) for t in range(N_DAYS)])
    vega_fly = np.array([fly_vega(S[t], V[t], C[t], SK[t]) for t in range(N_DAYS)])
    strad = np.array([straddle_premium(S[t], V[t], C[t], SK[t]) for t in range(N_DAYS)])
    curv = np.array([curvature(V[t], C[t], SK[t]) for t in range(N_DAYS)])
    # 相对凸性 = 凸性缺口 / σ_atm = 0.01*c（sk 项在双翼平均中抵消）——纯净的凸性状态代理
    rel_curv = curv / np.sqrt(np.maximum(V, 1e-6))
    prem_pct = prem / S * 100

    print("\n===== 文章 B 诊断 =====")
    print(f"σ_atm 范围: {np.sqrt(V).min():.3f} - {np.sqrt(V).max():.3f}（均值 {np.sqrt(V).mean():.3f}）")
    print(f"凸性状态 c: 均值 {C.mean():.2f} std {C.std():.2f} min {C.min():.2f} max {C.max():.2f}")
    print(f"曲率缺口（vol 点）: 均值 {curv.mean():.4f} std {curv.std():.4f} min {curv.min():.4f} max {curv.max():.4f}")
    print(f"相对凸性（=0.01*c）: 均值 {rel_curv.mean():.4f} std {rel_curv.std():.4f}")
    print(f"蝴蝶溢价（占 S %）: 均值 {prem_pct.mean():.3f} std {prem_pct.std():.3f}")
    print(f"裸蝴蝶 vega: 均值 {vega_fly.mean():.3f}（每股每 1.00 vol）")
    print(f"corr(溢价, sqrt(v)) = {np.corrcoef(prem_pct, np.sqrt(V))[0,1]:.3f} | corr(溢价, S) = {np.corrcoef(prem_pct, S)[0,1]:.3f} | corr(溢价, 相对凸性) = {np.corrcoef(prem_pct, rel_curv)[0,1]:.3f}")
    print(f"corr(曲率缺口, c) = {np.corrcoef(curv, C)[0,1]:.3f} | corr(曲率缺口, sqrt(v)) = {np.corrcoef(curv, np.sqrt(V))[0,1]:.3f} | corr(相对凸性, c) = {np.corrcoef(rel_curv, C)[0,1]:.3f}")
    print(f"微笑机制跳变次数: {n_regime_jumps}")

    ar1 = np.corrcoef(rel_curv[:-1], rel_curv[1:])[0, 1]
    half_life = -np.log(2) / np.log(ar1)
    print(f"相对凸性 AR(1): {ar1:.3f} | 半衰期 {half_life:.1f} 天")

    # ---------- z-score 交易（信号 = 相对凸性；头寸 = 蝴蝶 + 每日再平衡跨式 vega 对冲） ----------
    WIN = 40
    z = np.full(N_DAYS, np.nan)
    for t in range(WIN, N_DAYS):
        mu, sd = rel_curv[t - WIN:t].mean(), rel_curv[t - WIN:t].std()
        z[t] = (rel_curv[t] - mu) / max(sd, 1e-9)

    # 预计算每日对冲比率 n_t = -V_fly / vega_straddle
    vega_strad = np.array([2 * bs_vega(S[t], S[t], TAU, R, surface_sig(V[t], C[t], SK[t], 0.0))
                           for t in range(N_DAYS)])
    n_hedge = -vega_fly / vega_strad

    pos, entry_t = 0, 0
    trades = []          # (entry, exit, side, pnl, n_avg)
    pnl_curve = np.zeros(N_DAYS)
    hedge_cost_total = 0.0
    for t in range(1, N_DAYS):
        if pos != 0:
            # 持仓期间：裸蝶 + 上日对冲比率的跨式
            dp_fly = (prem[t] - prem[t - 1]) / S[t] * 100
            dp_strad = (strad[t] - strad[t - 1]) / S[t] * 100
            pnl_curve[t] = pos * dp_fly + pos * n_hedge[t - 1] * dp_strad - HEDGE_COST * 100
            hedge_cost_total += HEDGE_COST * 100
        if pos == 0 and t >= WIN:
            if z[t] > 1.5:
                pos, entry_t = -1, t
            elif z[t] < -1.5:
                pos, entry_t = 1, t
        elif pos != 0 and (abs(z[t]) < 0.5 or t - entry_t >= 12 or t == N_DAYS - 1):
            # 出场：对冲成本已在持仓日逐日扣除；这里收掉双边点差，按持仓全程累计记账
            pnl_curve[t] -= COST * 100 * 2
            trades.append((entry_t, t, pos, pnl_curve[entry_t:t + 1].sum()))
            pos = 0
    cum = np.cumsum(pnl_curve)
    tr_pnl = np.array([tr[3] for tr in trades])
    sides = np.array([tr[2] for tr in trades])
    holds = np.array([tr[1] - tr[0] for tr in trades])

    # ---------- 对照组：裸蝴蝶（同一信号，无 vega 对冲）----------
    pos_n, entry_tn = 0, 0
    trades_n = []
    pnl_curve_n = np.zeros(N_DAYS)
    for t in range(1, N_DAYS):
        if pos_n != 0:
            dp_fly_n = (prem[t] - prem[t - 1]) / S[t] * 100
            pnl_curve_n[t] = pos_n * dp_fly_n
        if pos_n == 0 and t >= WIN:
            if z[t] > 1.5:
                pos_n, entry_tn = -1, t
            elif z[t] < -1.5:
                pos_n, entry_tn = 1, t
        elif pos_n != 0 and (abs(z[t]) < 0.5 or t - entry_tn >= 12 or t == N_DAYS - 1):
            pnl_curve_n[t] -= COST * 100 * 2
            trades_n.append((entry_tn, t, pos_n, pnl_curve_n[entry_tn:t + 1].sum()))
            pos_n = 0
    cum_n = np.cumsum(pnl_curve_n)
    tr_pnl_n = np.array([tr[3] for tr in trades_n])
    dd_n = cum_n - np.maximum.accumulate(cum_n)

    print(f"交易: {len(trades)} 笔（买 {int((sides==1).sum())} / 卖 {int((sides==-1).sum())}），平均持仓 {holds.mean():.1f} 天")
    print(f"对冲后 | 总PnL {tr_pnl.sum():+.3f}% | 平均 {tr_pnl.mean():+.4f}% | 胜率 {(tr_pnl>0).mean()*100:.0f}% | std {tr_pnl.std():.4f}% | IR {tr_pnl.mean()/max(tr_pnl.std(),1e-9):.2f}")
    print(f"裸蝴蝶 | 交易 {len(trades_n)} 笔 | 总PnL {tr_pnl_n.sum():+.3f}% | 平均 {tr_pnl_n.mean():+.4f}% "
          f"| 胜率 {(tr_pnl_n>0).mean()*100:.0f}% | std {tr_pnl_n.std():.4f}% | IR {tr_pnl_n.mean()/max(tr_pnl_n.std(),1e-9):.2f}")
    print(f"最大单笔亏损 {tr_pnl.min():.4f}% | 最大单笔盈利 {tr_pnl.max():.4f}%")
    dd = cum - np.maximum.accumulate(cum)
    print(f"累计 PnL {cum[-1]:.3f}% | 最大回撤 {dd.min():.3f}% | 对冲成本合计 {hedge_cost_total:.3f}%")
    hi = rel_curv > np.quantile(rel_curv, 0.9); lo = rel_curv < np.quantile(rel_curv, 0.1)
    print(f"相对凸性前10%分位后5日平均变化: {np.mean(rel_curv[5:][hi[:-5]])-np.mean(rel_curv[:-5][hi[:-5]]):+.4f}")
    print(f"相对凸性后10%分位后5日平均变化: {np.mean(rel_curv[5:][lo[:-5]])-np.mean(rel_curv[:-5][lo[:-5]]):+.4f}")

    # ---------- 归因（含对冲） ----------
    attr_curv, attr_vol, attr_spot, attr_res = [], [], [], []
    for (t0, t1, side, pnl) in trades:
        # 曲率 leg：只改 c
        d_curv = (fly_premium(S[t0], V[t0], C[t1], SK[t0]) - fly_premium(S[t0], V[t0], C[t0], SK[t0])) / S[t0] * 100
        # ATM leg：只改 v（裸蝶 + 对冲跨式）
        d_vol_fly = (fly_premium(S[t0], V[t1], C[t0], SK[t0]) - fly_premium(S[t0], V[t0], C[t0], SK[t0])) / S[t0] * 100
        d_vol_strad = (straddle_premium(S[t0], V[t1], C[t0], SK[t0]) - straddle_premium(S[t0], V[t0], C[t0], SK[t0])) / S[t0] * 100
        n0 = -fly_vega(S[t0], V[t0], C[t0], SK[t0]) / (2 * bs_vega(S[t0], S[t0], TAU, R, surface_sig(V[t0], C[t0], SK[t0], 0.0)))
        # spot leg：只改 S
        d_spot_fly = (fly_premium(S[t1], V[t0], C[t0], SK[t0]) - fly_premium(S[t0], V[t0], C[t0], SK[t0])) / S[t0] * 100
        d_spot_strad = (straddle_premium(S[t1], V[t0], C[t0], SK[t0]) - straddle_premium(S[t0], V[t0], C[t0], SK[t0])) / S[t0] * 100
        attr_curv.append(side * d_curv)
        attr_vol.append(side * (d_vol_fly + n0 * d_vol_strad))
        attr_spot.append(side * (d_spot_fly + n0 * d_spot_strad))
        attr_res.append(pnl - side * (d_curv + d_vol_fly + n0 * d_vol_strad + d_spot_fly + n0 * d_spot_strad))
    print("归因（单边平均 %）: "
          f"凸性 {np.mean(attr_curv):+.4f} | ATM 波动 {np.mean(attr_vol):+.4f} | 标的价格 {np.mean(attr_spot):+.4f} | 残差 {np.mean(attr_res):+.4f}")
    print(f"归因绝对值: 凸性 {np.mean(np.abs(attr_curv)):.4f} | ATM {np.mean(np.abs(attr_vol)):.4f} | 价格 {np.mean(np.abs(attr_spot)):.4f} | 残差 {np.mean(np.abs(attr_res)):.4f}")

    # ---------- 图 1：波动率曲面热力图 ----------
    ks = np.linspace(-0.25, 0.25, 41)
    surf = np.empty((N_DAYS, len(ks)))
    for t in range(N_DAYS):
        for j, k in enumerate(ks):
            surf[t, j] = surface_sig(V[t], C[t], SK[t], k)
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    im = ax.imshow(surf.T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, N_DAYS, ks[0] * 100, ks[-1] * 100])
    ax.set_xlabel("交易日"); ax.set_ylabel("log-moneyness k（%）")
    ax.set_title("合成隐含波动率曲面（Heston 状态 + 风格化微笑，500 天）")
    cb = fig.colorbar(im, ax=ax); cb.set_label("隐含波动率")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "vol_surface.png")); plt.close(fig)

    # ---------- 图 2：曲率时间序列 + 交易信号 ----------
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(rel_curv * 100, color="#34495e", lw=1.2, label="相对凸性（凸性缺口 / σ_atm = 0.01×c）")
    axes[0].axhline(rel_curv[:WIN].mean() * 100, color="#95a5a6", ls="--", lw=1, label="前 40 日均值")
    axes[0].set_ylabel("相对凸性（×100）"); axes[0].set_title("30DTE ±10% 蝴蝶的相对凸性与 z-score 交易信号")
    axes[0].legend(loc="upper left"); axes[0].grid(alpha=0.3)
    axes[1].plot(z, color="#8e44ad", lw=1.0)
    axes[1].axhline(1.5, color="#c0392b", ls="--", lw=1)
    axes[1].axhline(-1.5, color="#27ae60", ls="--", lw=1)
    axes[1].axhline(0, color="black", lw=0.5, alpha=0.4)
    axes[1].set_ylabel("z-score"); axes[1].set_xlabel("交易日"); axes[1].grid(alpha=0.3)
    for (t0, t1, side, pnl) in trades:
        axes[1].annotate("", xy=(t0, z[t0]), xytext=(t0, 0),
                         arrowprops=dict(arrowstyle="->",
                                         color="#c0392b" if side == -1 else "#27ae60", lw=1.2))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "curvature_signals.png")); plt.close(fig)

    # ---------- 图 3：均值回归 ----------
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].scatter(rel_curv[:-1], rel_curv[1:], s=8, alpha=0.45, color="#2980b9")
    lims = [rel_curv.min() - 0.002, rel_curv.max() + 0.002]
    axes[0].plot(lims, lims, color="#c0392b", lw=1.2, ls="--", label="45° 线（无回归）")
    b, a = np.polyfit(rel_curv[:-1], rel_curv[1:], 1)
    xs_ = np.linspace(*lims, 50)
    axes[0].plot(xs_, a + b * xs_, color="#e67e22", lw=1.6, label=f"AR(1) 拟合 斜率={b:.2f}")
    axes[0].set_xlabel("当日相对凸性"); axes[0].set_ylabel("次日相对凸性")
    axes[0].set_title(f"凸性均值回归（lag-1 相关 {ar1:.2f}，半衰期 {half_life:.0f} 天）")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].hist(tr_pnl, bins=18, color="#16a085", alpha=0.85, edgecolor="white")
    axes[1].axvline(0, color="#c0392b", ls="--", lw=1.2)
    axes[1].set_xlabel("单笔 PnL（%）"); axes[1].set_ylabel("笔数")
    axes[1].set_title(f"{len(trades)} 笔交易盈亏分布（胜率 {(tr_pnl>0).mean()*100:.0f}%）")
    axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "mean_reversion.png")); plt.close(fig)

    # ---------- 图 4：策略 PnL ----------
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.plot(cum, color="#c0392b", lw=1.7, label="凸性 z-score + 每日跨式 vega 对冲")
    ax.plot(cum_n, color="#7f8c8d", lw=1.3, ls="--", label="裸蝴蝶（同一信号，无对冲）")
    ax.fill_between(range(N_DAYS), cum, 0, where=(cum >= 0), color="#c0392b", alpha=0.12)
    ax.fill_between(range(N_DAYS), cum, 0, where=(cum < 0), color="#27ae60", alpha=0.12)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("交易日"); ax.set_ylabel("累计 PnL（%）")
    ax.set_title("凸性 z-score 策略累计盈亏（含对冲成本与 4bp 双边点差）")
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "strategy_pnl.png")); plt.close(fig)

    # ---------- 图 5：归因 ----------
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    comp = [np.mean(attr_curv), np.mean(attr_vol), np.mean(attr_spot), np.mean(attr_res)]
    labels = ["凸性变动\n(目标暴露)", "ATM 波动\n(对冲后残余)", "标的价格\n(γ)", "残差"]
    colors = ["#27ae60", "#e67e22", "#7f8c8d", "#95a5a6"]
    bars = ax.bar(labels, comp, color=colors, alpha=0.85)
    for b_, v_ in zip(bars, comp):
        ax.text(b_.get_x() + b_.get_width() / 2, v_ + 2e-4 * (1 if v_ >= 0 else -1),
                f"{v_:+.4f}", ha="center", fontsize=9)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel("对单笔 PnL 的平均贡献（%）")
    ax.set_title("盈亏归因：对冲掉 ATM 波动后，赚的才是凸性回归")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "attribution.png")); plt.close(fig)

    return dict(ar1=float(ar1), half_life=float(half_life),
                c_mean=float(C.mean()), c_std=float(C.std()),
                curv_mean=float(curv.mean()), curv_std=float(curv.std()),
                rel_curv_mean=float(rel_curv.mean()), rel_curv_std=float(rel_curv.std()),
                prem_mean=float(prem_pct.mean()), prem_std=float(prem_pct.std()),
                vega_fly=float(vega_fly.mean()),
                corr_v=float(np.corrcoef(prem_pct, np.sqrt(V))[0, 1]),
                corr_rel=float(np.corrcoef(prem_pct, rel_curv)[0, 1]),
                corr_gap_c=float(np.corrcoef(curv, C)[0, 1]),
                corr_gap_v=float(np.corrcoef(curv, np.sqrt(V))[0, 1]),
                n_regime_jumps=int(n_regime_jumps),
                n_trades=len(trades), n_buy=int((sides == 1).sum()), n_sell=int((sides == -1).sum()),
                avg_hold=float(holds.mean()), total_pnl=float(tr_pnl.sum()),
                avg_pnl=float(tr_pnl.mean()), winrate=float((tr_pnl > 0).mean()),
                pnl_std=float(tr_pnl.std()), ir=float(tr_pnl.mean() / max(tr_pnl.std(), 1e-9)),
                max_loss=float(tr_pnl.min()), max_win=float(tr_pnl.max()),
                max_dd=float(dd.min()), attr=[float(c) for c in comp],
                naked_n=len(trades_n), naked_total=float(tr_pnl_n.sum()),
                naked_avg=float(tr_pnl_n.mean()), naked_win=float((tr_pnl_n > 0).mean()),
                naked_ir=float(tr_pnl_n.mean() / max(tr_pnl_n.std(), 1e-9)),
                naked_dd=float(dd_n.min()),
                attr_abs=[float(c) for c in (np.mean(np.abs(attr_curv)), np.mean(np.abs(attr_vol)),
                                             np.mean(np.abs(attr_spot)), np.mean(np.abs(attr_res)))])


if __name__ == "__main__":
    print("########## 文章 A：做市商 DQN ##########")
    gen_mm_dqn()
    print("\n########## 文章 B：Butterfly ##########")
    gen_butterfly()
    print("\nDONE")
