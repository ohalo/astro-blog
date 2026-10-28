#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""蒙特卡洛树搜索(MCTS)组合优化：用 UCB1 在指数级组合空间里剪枝选股
E1 合成 15 只资产、选 5 只最大化等权组合 Sharpe（注入 5 只高 Sharpe "好资产" 做真最优）
E2 穷举 C(15,5)=3003 求真实最优 Sharpe* 与最优组合（ground truth）
E3 手写 UCB1 MCTS（选择/扩展/模拟/回传）2000 次迭代，记录收敛与访问
E4 MCTS vs 随机搜索同预算对比 + 剪枝比（访问的不同子集 << 穷举）
E5 资产入选频率 / 目标分布对比
输出 5 张图 + _results.json 到 public/images/{slug}/
"""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for f in ["Heiti TC", "PingFang SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

SLUG = "mcts-portfolio-optimization"
OUT = f"/Users/halo123456/workspace/astro-blog/public/images/{SLUG}"
os.makedirs(OUT, exist_ok=True)
RED, GREEN, BLUE, GRAY, ORANGE, BLACK, PURPLE = "#d63031", "#00b894", "#0984e3", "#636e72", "#e17055", "#2d3436", "#6c5ce7"
N, K, T, SEED = 15, 5, 250, 20261028
rng = np.random.default_rng(SEED)
R = {}

# ---------------- 合成收益矩阵：5 只高 Sharpe 好资产 + 10 只差资产 ----------------
market = rng.normal(0, 0.005, T)
alpha = np.array([0.0010] * 5 + [0.0001] * 10)      # 好资产日 alpha 10bp，差资产 1bp
beta = np.ones(N)
idio = np.array([0.008] * 5 + [0.010] * 10)
eps = rng.normal(0, 1, (N, T))
Rmat = alpha[:, None] + beta[:, None] * market[None, :] + idio[:, None] * eps

def obj(sel):
    p = Rmat[list(sel)].mean(0)
    m, s = p.mean(), p.std(ddof=1)
    return m / s if s > 0 else -1e9

ind_sharpe = np.array([obj({i}) for i in range(N)])
R["asset_individual_sharpe"] = [round(float(v), 4) for v in ind_sharpe]
R["good_assets"] = [0, 1, 2, 3, 4]

# ---------------- 穷举 ground truth ----------------
from itertools import combinations
all_objs = []
best_e, best_sel_e = -1e18, None
for comb in combinations(range(N), K):
    v = obj(comb); all_objs.append(v)
    if v > best_e:
        best_e, best_sel_e = v, comb
S_star = best_e
R["exhaustive_best_sharpe"] = round(float(S_star), 4)
R["exhaustive_best_sel"] = list(best_sel_e)
R["n_exhaustive"] = len(all_objs)
R["exhaustive_mean_obj"] = round(float(np.mean(all_objs)), 4)
R["exhaustive_p90_obj"] = round(float(np.percentile(all_objs, 90)), 4)

# ---------------- 手写 UCB1 MCTS ----------------
C = math.sqrt(2.0)
terminal_set = set()
def make_node(selected):
    avail = [a for a in range(N) if a not in selected]
    node = {'sel': tuple(sorted(selected)), 'avail': avail, 'children': {},
            'visits': 0, 'val': 0.0, 'terminal': len(selected) == K}
    if node['terminal']:
        terminal_set.add(node['sel'])
    return node

def rollout(node):
    if node['terminal']:
        return obj(set(node['sel'])), set(node['sel'])
    sel = set(node['sel']); avail = list(node['avail'])
    rng.shuffle(avail); need = K - len(sel)
    for a in avail[:need]:
        sel.add(a)
    return obj(sel), sel

def select_expand(root):
    path = [root]; cur = root
    while not cur['terminal']:
        if len(cur['children']) < len(cur['avail']):
            taken = set(cur['children']); cand = [a for a in cur['avail'] if a not in taken]
            a = rng.choice(cand); child = make_node(set(cur['sel']) | {a})
            cur['children'][a] = child; path.append(child); return path
        best, bs = None, -1e18
        for a, ch in cur['children'].items():
            score = ch['val'] / ch['visits'] + C * math.sqrt(math.log(cur['visits']) / ch['visits'])
            if score > bs:
                bs = score; best = ch
        cur = best; path.append(cur)
    return path

ITERS = 2000
root = make_node(set())
best_m, best_sel_m, conv_m, terminal_count = -1e18, None, [], 0
evaluated = []          # 所有被评估的完整 K 子集（树终节点 + 模拟补全），用于入选频率与剪枝比
for it in range(ITERS):
    path = select_expand(root)
    val, completed = rollout(path[-1])
    for n in path:
        n['visits'] += 1; n['val'] += val
    evaluated.append(tuple(sorted(completed)))
    if path[-1]['terminal']:
        terminal_count += 1
        tv = obj(set(path[-1]['sel']))
        if tv > best_m:
            best_m, best_sel_m = tv, tuple(sorted(path[-1]['sel']))
    else:
        if val > best_m:
            best_m, best_sel_m = val, tuple(sorted(completed))
    conv_m.append(best_m)

# 随机搜索（同预算 = 同样 2000 次 obj 评估）
best_r, conv_r = -1e18, []
for it in range(ITERS):
    comb = tuple(sorted(rng.choice(N, K, replace=False)))
    v = obj(set(comb))
    if v > best_r:
        best_r = v
    conv_r.append(best_r)

R["mcts_iters"] = ITERS
R["mcts_best_sharpe"] = round(float(best_m), 4)
R["mcts_best_sel"] = list(best_sel_m)
R["mcts_gap_pct"] = round((S_star - best_m) / S_star * 100, 2)
R["mcts_distinct_evaluated"] = len(set(evaluated))   # 实际评估的不同 K 子集数
R["mcts_leaf_visits"] = terminal_count
R["random_best_sharpe"] = round(float(best_r), 4)
R["random_gap_pct"] = round((S_star - best_r) / S_star * 100, 2)
R["pruning_ratio"] = round((1 - len(set(evaluated)) / R["n_exhaustive"]) * 100, 1)
R["mcts_hit_optimal"] = bool(set(best_sel_m) == set(best_sel_e))

# 入选频率（在所有被评估的完整子集里统计）
incl = np.zeros(N)
obj_by_asset = np.zeros(N)
cnt_by_asset = np.zeros(N)
for sel in set(evaluated):
    v = obj(set(sel))
    for a in sel:
        incl[a] += 1
        obj_by_asset[a] += v
        cnt_by_asset[a] += 1
incl_rate = incl / max(len(set(evaluated)), 1)
avg_obj_by_asset = obj_by_asset / np.maximum(cnt_by_asset, 1)
R["inclusion_rate"] = [round(float(v), 3) for v in incl_rate]
R["good_inclusion_avg"] = round(float(incl_rate[:5].mean()), 3)
R["bad_inclusion_avg"] = round(float(incl_rate[5:].mean()), 3)
R["avg_obj_by_asset"] = [round(float(v), 4) for v in avg_obj_by_asset]
R["good_avg_obj"] = round(float(avg_obj_by_asset[:5].mean()), 4)
R["bad_avg_obj"] = round(float(avg_obj_by_asset[5:].mean()), 4)

# ---------------- 图 1：资产散点 + 入选频率 ----------------
fig, ax = plt.subplots(figsize=(12, 4.8))
vol = Rmat.std(1)
colors = [GREEN if i < 5 else GRAY for i in range(N)]
sizes = 40 + 260 * incl_rate
for i in range(N):
    ax.scatter(vol[i] * 100, alpha[i] * 1e4, s=sizes[i], c=colors[i], alpha=0.8,
               edgecolor=BLACK, zorder=3)
    ax.annotate(f"{i}", (vol[i] * 100, alpha[i] * 1e4), fontsize=7, ha="center", va="center")
sel_set = set(best_sel_m)
for i in sel_set:
    ax.scatter(vol[i] * 100, alpha[i] * 1e4, s=sizes[i] + 20, facecolors="none",
               edgecolor=RED, lw=2.0, zorder=4)
ax.set_xlabel("个股日波动 σ (%)"); ax.set_ylabel("个股日 alpha (×1e4)")
ax.set_title(f"15 只资产（绿=注入的 5 只好资产）；红圈=MCTS 选出组合；点大小=MCTS 该资产入选频率")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_asset_scatter.png"); plt.close(fig)

# ---------------- 图 2：收敛曲线 ----------------
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.plot(conv_m, color=BLUE, lw=1.3, label=f"MCTS 最优 Sharpe（终 {best_m:.3f}，差距 {R['mcts_gap_pct']}%）")
ax.plot(conv_r, color=GRAY, lw=1.0, alpha=0.8, label=f"随机搜索（终 {best_r:.3f}，差距 {R['random_gap_pct']}%）")
ax.axhline(S_star, color=RED, ls="--", lw=1.4, label=f"穷举真实最优 Sharpe*={S_star:.3f}")
ax.set_xlabel("MCTS / 随机搜索 迭代（每次 1 次目标评估）"); ax.set_ylabel("迄今找到的最优 Sharpe")
ax.set_title("收敛对比：同 2000 次评估预算下 MCTS 用 UCB1 剪枝远快于随机搜索")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_convergence.png"); plt.close(fig)

# ---------------- 图 3：剪枝比 ----------------
fig, ax = plt.subplots(figsize=(12, 4.8))
vals = [R["mcts_distinct_evaluated"], R["n_exhaustive"]]
bars = ax.bar(["MCTS 实际评估的不同子集", "穷举总数 C(15,5)"], vals, color=[BLUE, GRAY], alpha=0.85)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 30, str(v), ha="center", fontsize=10)
ax.set_ylabel("评估的子集数量")
ax.set_title(f"剪枝：MCTS 只评估 {R['mcts_distinct_evaluated']} 个不同子集就逼近最优（省 {R['pruning_ratio']}% 穷举量）")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_pruning.png"); plt.close(fig)

# ---------------- 图 4：每个资产在 MCTS 评估子集里的平均组合 Sharpe ----------------
fig, ax = plt.subplots(figsize=(12, 4.8))
order = np.argsort(-avg_obj_by_asset)
xs = np.arange(N)
cols = [GREEN if i < 5 else GRAY for i in range(N)]
ax.bar(xs, [avg_obj_by_asset[i] for i in order], color=[cols[i] for i in order], alpha=0.85, edgecolor=BLACK, lw=0.4)
ax.set_xticks(xs); ax.set_xticklabels([str(i) for i in order], fontsize=8)
ax.set_xlabel("资产编号（按平均组合 Sharpe 降序）"); ax.set_ylabel("含该资产的 MCTS 评估子集之平均 Sharpe")
ax.set_title(f"搜索偏好：被 MCTS 反复放进高 Sharpe 子集的资产（绿=注入的好资产，平均 {R['good_avg_obj']:.3f} vs 差资产 {R['bad_avg_obj']:.3f}）")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_inclusion.png"); plt.close(fig)

# ---------------- 图 5：目标分布对比 ----------------
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.hist(all_objs, bins=40, density=True, color=GRAY, alpha=0.5, label=f"穷举全部 {R['n_exhaustive']} 个组合目标值")
mcts_terms = [obj(set(s)) for s in set(evaluated)]
ax.hist(mcts_terms, bins=40, density=True, histtype="step", lw=2.0, color=BLUE,
        label=f"MCTS 到达的子集（{len(terminal_set)} 个）")
ax.axvline(S_star, color=RED, ls="--", lw=1.4, label=f"最优 Sharpe*={S_star:.3f}")
ax.set_xlabel("等权组合 Sharpe"); ax.set_ylabel("密度")
ax.set_title("目标分布：MCTS 把搜索集中到高 Sharpe 右尾，而非均匀撒网")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_objective_dist.png"); plt.close(fig)

def _ser(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)
with open(f"{OUT}/_results.json", "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=1, default=_ser)
print(json.dumps(R, ensure_ascii=False, indent=1, default=_ser))
