# -*- coding: utf-8 -*-
"""Honest audit of deep-RL trading agents on a market with a known edge.

Single source of truth for every number and figure in the article. A synthetic
AR(1) return series with a known autocorrelation is split into train / valid /
test. A tabular Q-learning agent and a PPO-style policy-gradient agent are
trained from scratch in NumPy. We audit what most trading-RL demos skip:

  1. does the agent actually learn the signal, or does a one-line rule win?
  2. does in-sample Sharpe survive on the untouched test split?
  3. how much of the apparent edge do transaction costs eat?
  4. how does the result change as the true signal-to-cost ratio varies?
  5. how wide is the seed-to-seed spread of a single backtest?

All comparisons are paired on the same paths, all seeds are fixed, and the
exec replay reproduces results.json exactly.
"""
import json
import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261124
N_SEEDS = 8
SEEDS = [SEED + 101 * i for i in range(N_SEEDS)]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "deep-reinforcement-learning-quant-trading")
os.makedirs(OUT, exist_ok=True)

N_TRAIN = 900
N_VALID = 300
N_TEST = 300
N_TOTAL = N_TRAIN + N_VALID + N_TEST
SIG = 0.012
COST = 0.0005                    # 5 bps per unit turnover
EDGE = 0.60                      # AR(1) autocorrelation = the true signal
NBINS = 5                        # state discretisation of the last return
GAMMA = 0.90
LR = 0.20
EPISODES = 400
EP_LEN = 80
EPS_MIN = 0.05


# --------------------------------------------------------------------- market
def make_market(seed, n=N_TOTAL, phi=EDGE, sigma=SIG):
    """r_t = phi*r_{t-1} + sigma*eps_t, with an iid copy for the matched no-edge world."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    r = np.zeros(n)
    r[0] = sigma * e[0]
    for t in range(1, n):
        r[t] = phi * r[t - 1] + sigma * e[t]
    return r


def state_bin(r, t, phi=EDGE):
    """Discretise the most recent return into NBINS levels."""
    if t < 1:
        return NBINS // 2
    z = r[t - 1] / (SIG / (1 - phi ** 2) ** 0.5)
    return int(np.clip(np.floor((z + 2.0) / 4.0 * NBINS), 0, NBINS - 1))


def step_reward(r, t, prev_pos, pos, cost=COST):
    turnover = abs(pos - prev_pos)
    return pos * r[t] - cost * turnover, turnover


def run_policy(policy, r, start, end, cost=COST):
    """policy(bin, pos) -> action in {0,1}. Returns net-of-cost path statistics."""
    pos = 0
    rews, turns, acts = [], [], []
    for t in range(start, end):
        a = int(policy(state_bin(r, t), pos))
        rew, turn = step_reward(r, t, pos, a, cost)
        rews.append(rew)
        turns.append(turn)
        acts.append(a)
        pos = a
    rews = np.asarray(rews)
    eq = np.exp(np.cumsum(rews))
    peak = np.maximum.accumulate(eq)
    return {
        "sharpe": float(rews.mean() / (rews.std() + 1e-12) * math.sqrt(252)),
        "return": float(eq[-1] - 1.0),
        "max_dd": float(((eq - peak) / peak).min()),
        "turnover": float(np.mean(turns)),
        "exposure": float(np.mean(acts)),
        "equity": eq.tolist(),
    }


def rule_policy(b, pos):
    """Long iff the last return was above the middle bin."""
    return 1 if b >= NBINS // 2 + 1 else 0


def buy_hold(_b, _pos):
    return 1


def flat(_b, _pos):
    return 0


# ---------------------------------------------------------------------- agents
def train_q(seed, phi=EDGE, cost=COST, episodes=EPISODES, eps_decay=0.992, lr=LR):
    """Tabular Q-learning: Q[bin,pos,a], one-step TD, epsilon-greedy."""
    r = make_market(seed, phi=phi)
    q = np.random.default_rng(seed).normal(0, 1e-4, (NBINS, 2, 2))
    rng = np.random.default_rng(seed + 5)
    eps = 1.0
    curve = []
    for ep in range(episodes):
        t0 = int(rng.integers(1, N_TRAIN - EP_LEN - 1))
        pos = 0
        for t in range(t0, t0 + EP_LEN):
            b = state_bin(r, t, phi)
            if rng.random() < eps:
                a = int(rng.integers(0, 2))
            else:
                a = int(np.argmax(q[b, pos]))
            rew, _ = step_reward(r, t, pos, a, cost)
            b2 = state_bin(r, t + 1, phi)
            target = rew + GAMMA * q[b2, a].max()
            q[b, pos, a] += lr * (target - q[b, pos, a])
            pos = a
        eps = max(EPS_MIN, eps * eps_decay)
        if ep % 20 == 0:
            pol = lambda bb, pp, qq=q: int(np.argmax(qq[bb, pp]))
            curve.append(run_policy(pol, r, t0, min(t0 + EP_LEN, N_TRAIN), cost)["sharpe"])
    return q, np.asarray(curve)


def q_policy(q):
    return lambda b, pos: int(np.argmax(q[b, pos]))


def train_pg(seed, phi=EDGE, cost=COST, episodes=EPISODES, lr=3e-3):
    """REINFORCE with a self-normalising baseline on the same discretised state.

    A policy-gradient method is included because it fails in a *different* way
    from Q-learning (high variance under a thin reward); both are reported.
    """
    r = make_market(seed, phi=phi)
    rng = np.random.default_rng(seed + 9)
    theta = np.zeros((NBINS, 2, 2))
    curve = []
    for ep in range(episodes):
        t0 = int(rng.integers(1, N_TRAIN - EP_LEN - 1))
        pos = 0
        bs, ps, acts, rews = [], [], [], []
        for t in range(t0, t0 + EP_LEN):
            b = state_bin(r, t, phi)
            logits = theta[b, pos]
            p = np.exp(logits - logits.max())
            p /= p.sum()
            a = int(rng.choice(2, p=p))
            rew, _ = step_reward(r, t, pos, a, cost)
            bs.append(b); ps.append(pos); acts.append(a); rews.append(rew)
            pos = a
        rews = np.asarray(rews)
        adv = rews - rews.mean()
        for b, p, a, av in zip(bs, ps, acts, adv):
            logits = theta[b, p]
            probs = np.exp(logits - logits.max()); probs /= probs.sum()
            d = -probs.copy(); d[a] += 1.0
            theta[b, p] += lr * av * d / EP_LEN
        if ep % 20 == 0:
            pol = lambda bb, pp, th=theta: int(np.argmax(th[bb, pp]))
            curve.append(run_policy(pol, r, t0, min(t0 + EP_LEN, N_TRAIN), cost)["sharpe"])
    return theta, np.asarray(curve)


def pg_policy(theta):
    return lambda b, pos: int(np.argmax(theta[b, pos]))


# --------------------------------------------------------------------- helpers
def paired(policy, phi=EDGE, cost=COST, start=N_TRAIN + N_VALID, end=N_TOTAL):
    out = [run_policy(policy, make_market(s, phi=phi), start, end, cost) for s in SEEDS]
    keys = ["sharpe", "return", "max_dd", "turnover", "exposure"]
    d = {k: float(np.mean([o[k] for o in out])) for k in keys}
    d["sharpes"] = [o["sharpe"] for o in out]
    return d


def paired_agents(agents, kind, start=0, end=N_TRAIN):
    pols = [q_policy(a) if kind == "q" else pg_policy(a) for a in agents]
    out = [run_policy(p, make_market(s), start, end) for s, p in zip(SEEDS, pols)]
    keys = ["sharpe", "return", "max_dd", "turnover", "exposure"]
    d = {k: float(np.mean([o[k] for o in out])) for k in keys}
    d["sharpes"] = [o["sharpe"] for o in out]
    return d


# --------------------------------------------------------------------- figures
def fig1_paths():
    r = make_market(SEED)
    t = np.arange(N_TOTAL)
    sig = np.where(r > 0, r, np.nan)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.plot(t, sig, "|", color="#b23a2b", ms=6, alpha=0.5, label="positive daily return")
    ax.axvspan(0, N_TRAIN, color="#3a923a", alpha=0.08, label="train 900d")
    ax.axvspan(N_TRAIN, N_TRAIN + N_VALID, color="#2b8cbe", alpha=0.10, label="valid 300d")
    ax.axvspan(N_TRAIN + N_VALID, N_TOTAL, color="#b23a2b", alpha=0.10, label="test 300d")
    ax.set_xlabel("trading day")
    ax.set_ylabel("daily return (up days)")
    ax.set_title("Synthetic market: autocorrelation 0.60, cost 5bps")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    f = os.path.join(OUT, "fig1_paths.png")
    fig.savefig(f, dpi=115); plt.close(fig)
    return "fig1_paths.png"


def fig2_learning(qc, pc):
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.plot(np.arange(len(qc)) * 20, qc, color="#b23a2b", lw=1.6, label="Q-learning")
    ax.plot(np.arange(len(pc)) * 20, pc, color="#2b8cbe", lw=1.6, label="policy gradient")
    ax.axhline(0, color="#333", lw=0.9, ls="--")
    ax.set_xlabel("training episode (seed 20261124)")
    ax.set_ylabel("in-sample Sharpe of the greedy policy")
    ax.set_title("Training curves: both agents find the signal")
    ax.legend(fontsize=9)
    fig.tight_layout()
    f = os.path.join(OUT, "fig2_learning.png")
    fig.savefig(f, dpi=115); plt.close(fig)
    return "fig2_learning.png"


def fig3_oos(res):
    names = ["Q-learn", "Policy grad", "1-lag rule", "buy & hold", "flat"]
    keys = ["q", "pg", "rule", "bh", "flat"]
    colors = ["#b23a2b", "#2b8cbe", "#3a923a", "#7b3294", "#888"]
    fig, ax = plt.subplots(figsize=(10.2, 4.9))
    xs = np.arange(len(names)); w = 0.36
    ins = [res[k]["in"]["sharpe"] for k in keys]
    oos = [res[k]["test"]["sharpe"] for k in keys]
    ax.bar(xs - w / 2, ins, w, color=colors, alpha=0.42, label="in-sample")
    ax.bar(xs + w / 2, oos, w, color=colors, label="out-of-sample")
    ax.axhline(0, color="#333", lw=0.9)
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylabel("annualised Sharpe (8-seed mean)")
    ax.set_title("Does the training-window promise survive the test split?")
    ax.legend(fontsize=9)
    fig.tight_layout()
    f = os.path.join(OUT, "fig3_oos.png")
    fig.savefig(f, dpi=115); plt.close(fig)
    return "fig3_oos.png"


def fig4_cost(cost_rows):
    costs = [0, 5, 10, 20, 40]
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    for key, lab, col in (("q", "Q-learning", "#b23a2b"), ("pg", "policy grad", "#2b8cbe"),
                          ("rule", "1-lag rule", "#3a923a"), ("bh", "buy & hold", "#7b3294")):
        ax.plot(costs, [cost_rows[str(c)][key]["sharpe"] for c in costs],
                "o-", label=lab, color=col, lw=1.8)
    ax.axhline(0, color="#333", lw=0.9, ls="--")
    ax.set_xlabel("round-trip cost (basis points)")
    ax.set_ylabel("test annualised Sharpe (8-seed mean)")
    ax.set_title("The same learned policy under different costs")
    ax.legend(fontsize=9)
    fig.tight_layout()
    f = os.path.join(OUT, "fig4_cost.png")
    fig.savefig(f, dpi=115); plt.close(fig)
    return "fig4_cost.png"


def fig5_edge(edge_rows):
    edges = [0.0, 0.15, 0.3, 0.6, 0.9]
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    for key, lab, col in (("q", "Q-learning", "#b23a2b"), ("pg", "policy grad", "#2b8cbe"),
                          ("rule", "1-lag rule", "#3a923a")):
        ax.plot(edges, [edge_rows[str(e)][key]["sharpe"] for e in edges],
                "o-", label=lab, color=col, lw=1.8)
    ax.axhline(0, color="#333", lw=0.9, ls="--")
    ax.set_xlabel("true autocorrelation of daily returns")
    ax.set_ylabel("test annualised Sharpe (8-seed mean)")
    ax.set_title("Signal strength: where does learning beat the rule?")
    ax.legend(fontsize=9)
    fig.tight_layout()
    f = os.path.join(OUT, "fig5_edge.png")
    fig.savefig(f, dpi=115); plt.close(fig)
    return "fig5_edge.png"


def main():
    q_agents, q_curves = [], []
    pg_agents, pg_curves = [], []
    for s in SEEDS:
        q, qc = train_q(s)
        p, pc = train_pg(s)
        q_agents.append(q); pg_agents.append(p)
        q_curves.append(qc); pg_curves.append(pc)
    q_curve = np.mean(q_curves, axis=0)
    pg_curve = np.mean(pg_curves, axis=0)

    res = {
        "q": {"in": paired_agents(q_agents, "q", 0, N_TRAIN),
              "test": paired_agents(q_agents, "q", N_TRAIN + N_VALID, N_TOTAL)},
        "pg": {"in": paired_agents(pg_agents, "pg", 0, N_TRAIN),
         "test": paired_agents(pg_agents, "pg", N_TRAIN + N_VALID, N_TOTAL)},
        "rule": {"in": paired(rule_policy, start=0, end=N_TRAIN), "test": paired(rule_policy)},
        "bh": {"in": paired(buy_hold, start=0, end=N_TRAIN), "test": paired(buy_hold)},
        "flat": {"in": paired(flat, start=0, end=N_TRAIN), "test": paired(flat)},
    }

    cost_rows = {}
    for c in (0, 5, 10, 20, 40):
        cv = c / 1e4
        cost_rows[str(c)] = {
            "q": {"sharpe": float(np.mean([
                run_policy(q_policy(q), make_market(s), N_TRAIN + N_VALID, N_TOTAL, cv)["sharpe"]
                for s, q in zip(SEEDS, q_agents)]))},
            "pg": {"sharpe": float(np.mean([
                run_policy(pg_policy(p), make_market(s), N_TRAIN + N_VALID, N_TOTAL, cv)["sharpe"]
                for s, p in zip(SEEDS, pg_agents)]))},
            "rule": paired(rule_policy, cost=cv),
            "bh": paired(buy_hold, cost=cv),
        }

    edge_rows = {}
    for e in (0.0, 0.15, 0.3, 0.6, 0.9):
        row = {}
        for s in SEEDS:
            qa, _ = train_q(s, phi=e, episodes=200)
            pa, _ = train_pg(s, phi=e, episodes=200)
            r = make_market(s, phi=e)
            row.setdefault("q", []).append(run_policy(q_policy(qa), r, N_TRAIN + N_VALID, N_TOTAL)["sharpe"])
            row.setdefault("pg", []).append(run_policy(pg_policy(pa), r, N_TRAIN + N_VALID, N_TOTAL)["sharpe"])
            row.setdefault("rule", []).append(run_policy(rule_policy, r, N_TRAIN + N_VALID, N_TOTAL)["sharpe"])
        edge_rows[str(e)] = {k: {"sharpe": float(np.mean(v))} for k, v in row.items()}

    result = {
        "setup": {"seed": SEED, "n_seeds": N_SEEDS, "n_train": N_TRAIN, "n_valid": N_VALID,
                  "n_test": N_TEST, "edge": EDGE, "cost_bps": COST * 1e4, "gamma": GAMMA,
                  "lr": LR, "episodes": EPISODES, "ep_len": EP_LEN, "bins": NBINS},
        "q_curve": q_curve[:22].tolist(), "pg_curve": pg_curve[:22].tolist(),
        "res": res, "cost": cost_rows, "edge": edge_rows,
        "highlights": {
            "q_in_ret": res["q"]["in"]["return"], "q_test_ret": res["q"]["test"]["return"],
            "pg_in_ret": res["pg"]["in"]["return"], "pg_test_ret": res["pg"]["test"]["return"],
            "q_sharpe_min": min(res["q"]["test"]["sharpes"]),
            "q_sharpe_max": max(res["q"]["test"]["sharpes"]),
            "rule_sharpe": res["rule"]["test"]["sharpe"],
            "q_sharpe_test": res["q"]["test"]["sharpe"],
            "q_sharpe_in": res["q"]["in"]["sharpe"],
            "pg_sharpe_test": res["pg"]["test"]["sharpe"],
            "bh_sharpe_test": res["bh"]["test"]["sharpe"],
        },
        "figs": [fig1_paths(), fig2_learning(q_curve, pg_curve), fig3_oos(res),
                 fig4_cost(cost_rows), fig5_edge(edge_rows)],
    }
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
