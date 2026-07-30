#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分层抽样指数复制：跟踪误差 vs 持仓数量的工程权衡"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)
OUT = "/Users/halo/workspace/astro-blog/public/images/stratified-sampling-index-replication"
os.makedirs(OUT, exist_ok=True)

# ---------- 1. 构造 300 只成分股的因子结构市场 ----------
N = 300           # 成分股数量
T = 5 * 252       # 5 年日频
n_sectors = 10

# 行业标签（不均匀分布，模拟真实指数：金融/消费占比高）
sector_weights = np.array([0.20, 0.16, 0.13, 0.11, 0.10, 0.08, 0.08, 0.06, 0.05, 0.03])
sector = rng.choice(n_sectors, size=N, p=sector_weights)

# 市值：幂律分布（少数巨头 + 长尾）
size_raw = rng.pareto(1.4, N) + 1
mcap = size_raw / size_raw.sum()
# 指数权重 = 市值加权
w_idx = mcap / mcap.sum()

# 因子模型: r_i = beta_mkt*f_mkt + beta_sec*f_sec + beta_size*f_size + eps
beta_mkt = 1.0 + 0.25 * rng.standard_normal(N)
beta_size = -np.log(mcap / mcap.mean())  # 小市值暴露高
beta_size = (beta_size - beta_size.mean()) / beta_size.std()

vol_mkt, vol_sec, vol_size, vol_idio = 0.16, 0.10, 0.06, 0.22
f_mkt = vol_mkt/np.sqrt(252) * rng.standard_normal(T) + 0.06/252
f_sec = vol_sec/np.sqrt(252) * rng.standard_normal((T, n_sectors))
f_size = vol_size/np.sqrt(252) * rng.standard_normal(T)
eps = vol_idio/np.sqrt(252) * rng.standard_normal((T, N))

R = (np.outer(f_mkt, beta_mkt) + f_sec[:, sector] +
     np.outer(f_size, beta_size) + eps)          # T x N 日收益
idx_ret = R @ w_idx                               # 指数日收益

dates = pd.bdate_range("2021-01-04", periods=T)

def tracking_error(port_ret, bench_ret):
    return float(np.std(port_ret - bench_ret, ddof=1) * np.sqrt(252))

# ---------- 2. 三种复制方法 ----------
def full_replication():
    return w_idx.copy()

def naive_topk(k):
    """朴素法：选市值最大的 k 只，按指数权重归一化"""
    top = np.argsort(w_idx)[::-1][:k]
    w = np.zeros(N); w[top] = w_idx[top]; w /= w.sum()
    return w

def stratified(k):
    """分层抽样：按行业分层，层内配额 ∝ 层权重，层内选最大市值 + 权重按层校准"""
    w = np.zeros(N)
    sec_w = np.array([w_idx[sector == s].sum() for s in range(n_sectors)])
    # 每层至少 1 只，配额按层权重分配（最大余数法）
    quota_f = sec_w / sec_w.sum() * k
    quota = np.maximum(np.floor(quota_f).astype(int), 1)
    # 最大余数法补齐/削减，带安全计数器防死循环
    guard = 0
    while quota.sum() < k and guard < 1000:
        i = int(np.argmax(quota_f - quota)); quota[i] += 1; guard += 1
    while quota.sum() > k and guard < 2000:
        cand = np.where(quota > 1)[0]
        if len(cand) == 0: break
        i = cand[int(np.argmax((quota - quota_f)[cand]))]; quota[i] -= 1; guard += 1
    for s in range(n_sectors):
        members = np.where(sector == s)[0]
        pick = members[np.argsort(w_idx[members])[::-1][:quota[s]]]
        # 层内按市值权重分配，层总权重 = 指数中该层权重（关键：校准层暴露）
        wi = w_idx[pick] / w_idx[pick].sum() * sec_w[s]
        w[pick] = wi
    return w / w.sum()

def stratified_2d(k):
    """二维分层：行业 x 市值(大/小两档)"""
    w = np.zeros(N)
    med = np.median(np.log(mcap))
    size_bin = (np.log(mcap) > med).astype(int)
    cells = sector * 2 + size_bin
    cell_ids = np.unique(cells)
    cell_w = np.array([w_idx[cells == c].sum() for c in cell_ids])
    quota_f = cell_w / cell_w.sum() * k
    quota = np.maximum(np.round(quota_f).astype(int), 1)
    for c, q, cw in zip(cell_ids, quota, cell_w):
        members = np.where(cells == c)[0]
        pick = members[np.argsort(w_idx[members])[::-1][:q]]
        wi = w_idx[pick] / w_idx[pick].sum() * cw
        w[pick] = wi
    return w / w.sum()

# ---------- 3. TE vs k 扫描 ----------
ks = [10, 20, 30, 50, 75, 100, 150, 200, 250]
te_naive, te_strat, te_2d = [], [], []
for k in ks:
    te_naive.append(tracking_error(R @ naive_topk(k), idx_ret))
    te_strat.append(tracking_error(R @ stratified(k), idx_ret))
    te_2d.append(tracking_error(R @ stratified_2d(k), idx_ret))

# ---------- 4. 含成本的净跟踪差：漂移口径 + 季度校准 ----------
# 真实市值加权指数是自漂移的：先生成指数的漂移权重路径
W_drift = np.zeros((T, N))
w = w_idx.copy()
idx_ret_d = np.zeros(T)
for t in range(T):
    W_drift[t] = w
    idx_ret_d[t] = w @ R[t]
    w = w * (1 + R[t]); w /= w.sum()

# 成本模型：单边 20bp（冲击+佣金）
cost_bps = 0.0020

def build_weights(method, k, w_base):
    """基于给定基准权重重新计算抽样组合"""
    global w_idx
    w_save = w_idx
    w_idx = w_base  # 临时替换基准
    try:
        if method == "full": out = w_base.copy()
        elif method == "naive": out = naive_topk(k)
        elif method == "strat": out = stratified(k)
        else: out = stratified_2d(k)
    finally:
        w_idx = w_save
    return out

def run_with_rebalance(method, k, drift_reset=63):
    """组合 buy-and-hold 漂移，每季度校准到基于当日指数权重的目标；指数自漂移"""
    w_cur = build_weights(method, k, w_idx)
    port_ret = np.zeros(T); turnover_acc = 0.0
    for t in range(T):
        if t > 0 and t % drift_reset == 0:
            w_target = build_weights(method, k, W_drift[t])
            tw = np.abs(w_cur - w_target).sum() / 2
            turnover_acc += tw
            port_ret[t] = (w_cur * R[t]).sum() - tw * 2 * cost_bps
            w_cur = w_target.copy()
        else:
            port_ret[t] = (w_cur * R[t]).sum()
        w_cur = w_cur * (1 + R[t]); w_cur /= w_cur.sum()
    return port_ret, turnover_acc

k_demo = 50
pr_full, to_full = run_with_rebalance("full", None)
pr_naive, to_naive = run_with_rebalance("naive", k_demo)
pr_strat, to_strat = run_with_rebalance("strat", k_demo)
pr_2d, to_2d = run_with_rebalance("s2d", k_demo)

def summarize(pr):
    cum_p = np.prod(1 + pr); cum_b = np.prod(1 + idx_ret_d)
    te = tracking_error(pr, idx_ret_d)
    ann_gap = (cum_p ** (252/T) - 1) - (cum_b ** (252/T) - 1)
    return te, ann_gap

res = {name: summarize(pr) for name, pr in
       [("full", pr_full), ("naive", pr_naive), ("strat", pr_strat), ("s2d", pr_2d)]}
turnovers = {"full": to_full, "naive": to_naive, "strat": to_strat, "s2d": to_2d}

# ---------- 5. 蒙特卡洛：抽样方案的稳定性（重抽 idio 噪音 50 次） ----------
n_mc = 50
mc_naive, mc_strat = [], []
w_n50, w_s50 = naive_topk(50), stratified(50)
for m in range(n_mc):
    eps_m = vol_idio/np.sqrt(252) * rng.standard_normal((T, N))
    R_m = (np.outer(f_mkt, beta_mkt) + f_sec[:, sector] + np.outer(f_size, beta_size) + eps_m)
    idx_m = R_m @ w_idx
    mc_naive.append(tracking_error(R_m @ w_n50, idx_m))
    mc_strat.append(tracking_error(R_m @ w_s50, idx_m))
mc_naive, mc_strat = np.array(mc_naive), np.array(mc_strat)

# ---------- 6. 行业暴露偏差 ----------
def sector_gap(w):
    return np.array([w[sector==s].sum() - w_idx[sector==s].sum() for s in range(n_sectors)])
gap_naive = sector_gap(naive_topk(50)) * 100
gap_strat = sector_gap(stratified(50)) * 100

# ---------- 图 ----------
# cover: TE vs k
fig, ax = plt.subplots(figsize=(10.5, 5.2))
ax.plot(ks, np.array(te_naive)*100, "o-", label="朴素 Top-K 市值法", color="#d62728")
ax.plot(ks, np.array(te_strat)*100, "s-", label="行业分层抽样", color="#1f77b4")
ax.plot(ks, np.array(te_2d)*100, "^-", label="行业×市值二维分层", color="#2ca02c")
ax.set_xlabel("持仓数量 K"); ax.set_ylabel("年化跟踪误差 (%)")
ax.set_title(f"跟踪误差 vs 持仓数量（{N} 只成分股指数，5 年日频）")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/cover.png", dpi=110); plt.close()

# 行业暴露
fig, ax = plt.subplots(figsize=(10.5, 4.6))
x = np.arange(n_sectors)
ax.bar(x-0.2, gap_naive, width=0.4, label="朴素 Top-50", color="#d62728", alpha=0.85)
ax.bar(x+0.2, gap_strat, width=0.4, label="分层抽样 50 只", color="#1f77b4", alpha=0.85)
ax.axhline(0, color="k", lw=0.7)
ax.set_xlabel("行业"); ax.set_ylabel("权重偏差 (百分点)")
ax.set_xticks(x); ax.set_xticklabels([f"S{i+1}" for i in x])
ax.set_title("相对指数的行业权重偏差：分层抽样把行业暴露钉死在零附近")
ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/sector_gap.png", dpi=110); plt.close()

# 累计跟踪差路径
fig, ax = plt.subplots(figsize=(10.5, 4.8))
for pr, name, c in [(pr_naive, f"朴素 Top-{k_demo}", "#d62728"),
                    (pr_strat, f"行业分层 {k_demo} 只", "#1f77b4"),
                    (pr_2d, f"二维分层 {k_demo} 只", "#2ca02c"),
                    (pr_full, "全复制(含成本)", "#7f7f7f")]:
    cum_gap = np.cumprod(1+pr) / np.cumprod(1+idx_ret_d) - 1
    ax.plot(dates, cum_gap*100, lw=1.1, label=name, color=c)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("累计相对指数收益差 (%)"); ax.legend(loc="best")
ax.set_title("累计跟踪偏差路径（季度再平衡，单边成本 20bp）")
plt.tight_layout(); plt.savefig(f"{OUT}/cum_gap.png", dpi=110); plt.close()

# 蒙特卡洛
fig, ax = plt.subplots(figsize=(10.5, 4.4))
ax.hist(mc_naive*100, bins=18, alpha=0.65, label=f"朴素 Top-50: {mc_naive.mean()*100:.2f}% ± {mc_naive.std()*100:.2f}%", color="#d62728")
ax.hist(mc_strat*100, bins=18, alpha=0.65, label=f"分层 50 只: {mc_strat.mean()*100:.2f}% ± {mc_strat.std()*100:.2f}%", color="#1f77b4")
ax.set_xlabel("年化跟踪误差 (%)"); ax.set_ylabel("频数")
ax.set_title(f"{n_mc} 次特质噪音重抽下的 TE 分布：分层优势是结构性的")
ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/mc_te.png", dpi=110); plt.close()

stats = {
    "N": N, "T": T,
    "te_curve": {"ks": ks,
                 "naive": [round(x*100, 2) for x in te_naive],
                 "strat": [round(x*100, 2) for x in te_strat],
                 "s2d": [round(x*100, 2) for x in te_2d]},
    "demo_k": k_demo,
    "net": {k: {"te_pct": round(v[0]*100, 2), "ann_gap_bps": round(v[1]*1e4, 1)}
            for k, v in res.items()},
    "turnover_5y": {k: round(v, 2) for k, v in turnovers.items()},
    "mc": {"naive_mean": round(float(mc_naive.mean()*100), 2), "naive_std": round(float(mc_naive.std()*100), 2),
           "strat_mean": round(float(mc_strat.mean()*100), 2), "strat_std": round(float(mc_strat.std()*100), 2)},
    "sector_gap_max": {"naive": round(float(np.abs(gap_naive).max()), 2),
                       "strat": round(float(np.abs(gap_strat).max()), 2)},
    "top50_weight_cover": round(float(np.sort(w_idx)[::-1][:50].sum()*100), 1),
}
with open(f"{OUT}/stats.json", "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(json.dumps(stats, ensure_ascii=False, indent=2))
