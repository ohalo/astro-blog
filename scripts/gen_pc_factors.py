# -*- coding: utf-8 -*-
"""PC 算法因果因子发现：5 张真实计算配图。"""
import json, os
import numpy as np
from scipy.stats import norm
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Heiti TC"
plt.rcParams["axes.unicode_minus"] = False
ROOT = "/Users/halo123456/workspace/astro-blog"
OUT = ROOT + "/public/images/pc-algorithm-causal-factor-discovery"
os.makedirs(OUT, exist_ok=True)

RED, GREEN, BLUE, GRAY = "#d63031", "#00b894", "#0984e3", "#636e72"
ORANGE, BLACK, PURPLE, TEAL = "#e17055", "#2d3436", "#6c5ce7", "#00cec9"

NODES = ["S", "Mo", "V", "Vo", "To", "Y"]
NAMES = {"S": "市值 S", "Mo": "动量 Mo", "V": "价值 V",
         "Vo": "波动 Vo", "To": "换手 To", "Y": "未来收益 Y"}
POS = {"S": (0.10, 0.88), "Mo": (0.90, 0.88), "V": (0.03, 0.50),
       "Vo": (0.36, 0.50), "To": (0.72, 0.50), "Y": (0.42, 0.06)}
# 真值 DAG：(父母, 孩子, 系数)
TRUE_PARENTS = {"V": [("S", .62)], "Vo": [("S", .50)],
                "To": [("S", .55), ("Mo", .50)],
                "Y": [("Mo", .30), ("V", .22), ("Vo", .18)]}
TRUE_E = {frozenset((a, b)) for a in TRUE_PARENTS for b, _ in TRUE_PARENTS[a]}
ARCS_OK = {(0, 4), (1, 4), (1, 5), (2, 5), (3, 5)}  # S→To, Mo→To, Mo→Y, V→Y, Vo→Y（索引序）
IDX = {n: i for i, n in enumerate(NODES)}
TRUE_E_IDX = {frozenset((IDX[a], IDX[b])) for a in TRUE_PARENTS for b, _ in TRUE_PARENTS[a]}


def gen(n, g):
    """按真值 DAG 生成标准化因子面板 (n,6)，列序 = NODES。"""
    S = g.normal(size=n); Mo = g.normal(size=n)
    V = .62 * S + np.sqrt(1 - .62 ** 2) * g.normal(size=n)
    Vo = .50 * S + np.sqrt(1 - .50 ** 2) * g.normal(size=n)
    To = .55 * S + .50 * Mo + np.sqrt(1 - .55 ** 2 - .50 ** 2) * g.normal(size=n)
    # V,Vo 通过 S 相关 .62*.5=.31，Y 的解释方差含交叉项
    r2 = .30 ** 2 + .22 ** 2 + .18 ** 2 + 2 * .22 * .18 * (.62 * .50)
    Y = .30 * Mo + .22 * V + .18 * Vo + np.sqrt(max(1e-3, 1 - r2)) * g.normal(size=n)
    M = np.column_stack([S, Mo, V, Vo, To, Y])
    return (M - M.mean(0)) / M.std(0)


def pcorr(C, i, j, S):
    if len(S) == 0:
        return C[i, j]
    idx = [i, j] + list(S)
    P = np.linalg.inv(C[np.ix_(idx, idx)])
    return -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])


def fisher_p(r, n, k):
    r = np.clip(r, -0.999999, 0.999999)
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(n - k - 3)
    return 2 * (1 - norm.cdf(abs(z)))


def run_pc(C, n, alpha=0.01, maxdep=4):
    """PC 算法：骨架（Fisher-z 消边）→ v-结构 → Meek R1-R3。返回 CPDAG。"""
    N = C.shape[0]
    adj = {i: set(range(N)) - {i} for i in range(N)}
    sep = {}
    ntest = 0
    for depth in range(maxdep + 1):
        for i in range(N):
            for j in sorted(adj[i]):
                if j < i or j not in adj[i]:
                    continue
                nb = sorted(adj[i] - {j})
                if len(nb) < depth:
                    continue
                for S in combinations(nb, depth):
                    ntest += 1
                    if fisher_p(pcorr(C, i, j, S), n, depth) >= alpha:
                        adj[i].discard(j); adj[j].discard(i)
                        sep[(i, j)] = set(S); sep[(j, i)] = set(S)
                        break
    directed = set()
    edge = {frozenset((i, j)) for i in range(N) for j in adj[i]}

    def is_dir(a, b): return (a, b) in directed

    def is_und(a, b):
        return frozenset((a, b)) in edge and not is_dir(a, b) and not is_dir(b, a)

    def orient(a, b):
        if (b, a) in directed:
            return False
        directed.add((a, b))
        return True
    # v-结构：i-k-j 且 k 不在 sepset(i,j)
    for k in range(N):
        for i, j in combinations(sorted(adj[k]), 2):
            if j in adj[i]:
                continue
            s = sep.get((i, j))
            if s is not None and k not in s:
                orient(i, k); orient(j, k)
    # Meek R1-R3，迭代至不动点
    changed = True
    while changed:
        changed = False
        for (a, b) in list(directed):                      # R1
            for c in range(N):
                if is_und(b, c) and c != a and c not in adj[a]:
                    if orient(b, c):
                        changed = True
        for (c, b) in list(directed):                      # R2
            for a in range(N):
                if is_dir(a, c) and is_und(a, b):
                    if orient(a, b):
                        changed = True
        for b in range(N):                                 # R3
            for a in list(adj[b]):
                if not is_und(a, b):
                    continue
                cs = [c for c in range(N) if is_dir(c, b) and is_und(a, c)]
                for c1, c2 in combinations(cs, 2):
                    if c2 not in adj[c1]:
                        if orient(a, b):
                            changed = True
                        break
    return adj, directed, edge, sep, ntest


def score_cpdag(edge, directed):
    """返回 (骨架召回, 假边数, 正确弧数, 错误弧列表, 无向边集合)。"""
    e_ok = sum(1 for e in TRUE_E_IDX if e in edge)
    fp = len([e for e in edge if e not in TRUE_E_IDX])
    arcs = {a for a in directed if frozenset(a) in edge}
    ok = arcs & ARCS_OK
    bad = [(NODES[a], NODES[b]) for (a, b) in arcs - ARCS_OK]
    und = set()
    for e in edge:
        a, b = sorted(e)
        if (a, b) not in directed and (b, a) not in directed:
            und.add(e)
    return e_ok, fp, len(ok), bad, und


def draw_graph(ax, arcs, undirected, title, coef_map=None, note=None):
    ax.set_xlim(-0.14, 1.14); ax.set_ylim(-0.10, 1.06); ax.axis("off")
    drawn = set()
    for (a, b) in arcs:
        ka, kb = NODES[a], NODES[b]
        (xa, ya), (xb, yb) = POS[ka], POS[kb]
        ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.0,
                                    shrinkA=24, shrinkB=24, mutation_scale=16))
        drawn.add(frozenset((a, b)))
        if coef_map and (ka, kb) in coef_map:
            ax.text((xa + xb) / 2 + 0.035, (ya + yb) / 2, "%.2f" % coef_map[(ka, kb)],
                    fontsize=8.5, color=BLUE, ha="center")
    for e in undirected:
        a, b = sorted(e)
        ka, kb = NODES[a], NODES[b]
        (xa, ya), (xb, yb) = POS[ka], POS[kb]
        ax.plot([xa, xb], [ya, yb], color=GRAY, lw=2.0,
                solid_capstyle="round",
                transform=ax.transData,
                )
        # 截短避免压住节点框
        ln = np.hypot(xb - xa, yb - ya)
        t0, t1 = 26 / 700 / ln, 1 - 26 / 700 / ln
        ax.plot([xa + (xb - xa) * t0, xa + (xb - xa) * t1],
                [ya + (yb - ya) * t0, ya + (yb - ya) * t1], color=GRAY, lw=2.2)
    for nm, (x, y) in POS.items():
        ax.text(x, y, NAMES[nm], ha="center", va="center", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.34", fc="white", ec=BLACK, lw=1.1))
    ax.set_title(title, fontsize=10.5)
    if note:
        ax.text(0.5, -0.06, note, transform=ax.transAxes, ha="center",
                fontsize=8.2, color=GRAY)


STATS = {}

# ---------- 图1 真值 DAG ----------
fig, ax = plt.subplots(figsize=(7.6, 5.4))
arcs_t = {(IDX[p], IDX[c]) for c in TRUE_PARENTS for p, _ in TRUE_PARENTS[c]}
cm = {(p, c): b for c in TRUE_PARENTS for p, b in TRUE_PARENTS[c]}
draw_graph(ax, arcs_t, set(),
           "数据生成过程（模拟的 A 股因子面板真值）\nS/Mo 外生；Y 只被 Mo、V、Vo 直接驱动",
           coef_map=cm)
plt.tight_layout(); plt.savefig(OUT + "/true-dag.png", dpi=130); plt.close(fig)
print("fig1 ok")

# ---------- 主实验设定 ----------
N_MAIN, SEED_MAIN, ALPHA = 6000, 1, 0.01
g0 = np.random.default_rng(SEED_MAIN)
M = gen(N_MAIN, g0)
C = np.corrcoef(M.T)
adj, directed, edge, sep, ntest = run_pc(C, N_MAIN, ALPHA)
e_ok, fp, arcs_okn, bad_arcs, und_set = score_cpdag(edge, directed)
STATS["main"] = dict(n=N_MAIN, alpha=ALPHA, seed=SEED_MAIN, edges="%d/7" % e_ok,
                     false_edges=fp, arcs_ok="%d/5" % arcs_okn, wrong_arcs=bad_arcs,
                     undirected=[sorted(NODES[i] for i in e) for e in und_set],
                     ci_tests=ntest)
coll_r = pcorr(C, IDX["Mo"], IDX["S"], [IDX["To"]])
STATS["collider"] = dict(corr_MoS=float(C[1, 0]), p0=float(fisher_p(C[1, 0], N_MAIN, 0)),
                         corr_MoS_given_To=float(coll_r),
                         p1=float(fisher_p(coll_r, N_MAIN, 1)))

# ---------- 图2 边际相关 vs 条件相关（碰撞器） ----------
fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.3))
for ax_, Mx, ttl in ((axes[0], C, "边际相关矩阵：碰撞器 S、Mo 之间一片清白"),
                     (axes[1], np.array([[1.0 if i == j else
                                         (C[i, j] if IDX["To"] in (i, j)
                                          else pcorr(C, i, j, [IDX["To"]]))
                                          for j in range(6)] for i in range(6)]),
                      "给定碰撞器 To 之后：S–Mo 被“连线”，r = %.2f" % coll_r)):
    im = ax_.imshow(Mx, cmap="RdBu_r", vmin=-1, vmax=1)
    ax_.set_xticks(range(6)); ax_.set_xticklabels(NODES)
    ax_.set_yticks(range(6)); ax_.set_yticklabels(NODES)
    for i in range(6):
        for j in range(6):
            ax_.text(j, i, "%.2f" % Mx[i, j], ha="center", va="center", fontsize=8.5,
                     color="white" if abs(Mx[i, j]) > 0.55 else BLACK)
    ax_.set_title(ttl, fontsize=9.8)
    plt.colorbar(im, ax=ax_, fraction=0.046)
i_s, i_mo, i_to = IDX["S"], IDX["Mo"], IDX["To"]
axes[0].add_patch(plt.Rectangle((i_mo - .5 - 0.10, i_s - .5), 1.2, 1.0, fill=False,
                                ec=RED, lw=2.2))
axes[1].add_patch(plt.Rectangle((i_mo - .5 - 0.10, i_s - .5), 1.2, 1.0, fill=False,
                                ec=RED, lw=2.2))
fig.suptitle("图2 相关矩阵会撒谎：corr(S,Mo)=%.3f（p=%.2f）→ corr(S,Mo|To)=%.3f（p<1e-12）"
             % (C[1, 0], fisher_p(C[1, 0], N_MAIN, 0), coll_r), fontsize=11)
fig.text(0.5, 0.015, "红框：如果你只查“边际相关>0.1 就算有关”，筛因子时永远不会想到 S 和 Mo 有牵连；"
         "一旦模型里放进 To，两者的偏相关立刻跳到 ±0.4 附近", ha="center", fontsize=8.6, color=GRAY)
fig.tight_layout(rect=[0, 0.035, 1, 0.93]); fig.savefig(OUT + "/corr-vs-partial.png", dpi=130); plt.close(fig)
print("fig2 ok", STATS["collider"])

# ---------- 图3 恢复的 CPDAG ----------
und_list = sorted(und_set, key=lambda e: sorted(NODES[i] for i in e))
draw_note = ("蓝色实心箭头 = 被条件独立证据定向的因果边；灰色线段 = Markov 等价类内无法定向（S 与 V、Vo 的因果方向数据说了不算）"
             if len(und_list) == 2 and e_ok == 7 else
             "本次运行的骨架/定向结果，见正文诚实边界")
fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6))
draw_graph(axes[0], arcs_t, set(), "真值 DAG（7 条边，全部有向）", coef_map=cm)
draw_graph(axes[1], {(a, b) for (a, b) in directed}, und_list,
           "PC 算法输出：CPDAG（%d/7 边找回，假边 %d，错误定向 %d）" % (e_ok, fp, len(bad_arcs)),
           note=draw_note)
fig.suptitle("图3 n=%d、α=%.2g 下 PC 算法的还原结果：%s"
             % (N_MAIN, ALPHA, "定向 5 条全部正确，S–V、S–Vo 诚实地留白"
                if len(bad_arcs) == 0 else "存在定错误差，见正文"), fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(OUT + "/cpdag-recovered.png", dpi=130); plt.close(fig)
print("fig3 ok", STATS["main"])

# ---------- 图4 α 敏感性 + 样本量脆弱性 ----------
alphas = [0.001, 0.005, 0.01, 0.05, 0.10, 0.20]
rec, fps_, nts = [], [], []
for al in alphas:
    a2, d2, e2, s2, t2 = run_pc(C, N_MAIN, al)
    eo, fpn, _, _, _ = score_cpdag(e2, d2)
    rec.append(eo / 7); fps_.append(fpn); nts.append(t2)
STATS["alpha_scan"] = [dict(alpha=a, recall=r, false=f, tests=t)
                       for a, r, f, t in zip(alphas, rec, fps_, nts)]
ns = [500, 1000, 2000, 4000, 6000, 10000]
perfect, badcnt = [], []
for nn in ns:
    p_hit, b_hit = [], []
    for sd in range(12):
        gg = np.random.default_rng(500 + sd)
        Cn = np.corrcoef(gen(nn, gg).T)
        a2, d2, e2, s2, t2 = run_pc(Cn, nn, 0.01)
        eo, fpn, ao, bad, und_ = score_cpdag(e2, d2)
        p_hit.append(1.0 if (eo == 7 and fpn == 0 and ao == 5 and len(bad) == 0
                             and sorted(und_) == [frozenset((0, 2)), frozenset((0, 3))]) else 0.0)
        b_hit.append(len(bad))
    perfect.append(float(np.mean(p_hit))); badcnt.append(float(np.mean(b_hit)))
STATS["n_scan"] = [dict(n=a, perfect_cpdag=b, avg_wrong_arcs=c)
                   for a, b, c in zip(ns, perfect, badcnt)]
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
ax = axes[0]
ax.plot(range(len(alphas)), rec, "-o", color=GREEN, lw=2, label="真边召回率")
ax.plot(range(len(alphas)), [f / 7 for f in fps_], "-s", color=RED, lw=2, label="假边数/真边数")
for x, (f, t) in enumerate(zip(fps_, nts)):
    if f:
        ax.text(x, f / 7 + 0.03, "假边%d" % f, ha="center", fontsize=8, color=RED)
ax.set_xticks(range(len(alphas))); ax.set_xticklabels(["%.3g" % a for a in alphas])
ax.set_xlabel("显著性水平 α"); ax.set_ylim(-0.05, 1.15)
ax.axhline(1.0, color=GRAY, lw=0.8, ls=":")
ax.legend(fontsize=8.5, loc="center right"); ax.grid(alpha=0.25)
ax.set_title("α 扫描（n=%d 固定）：α 越小越保守，假边越少但真边召回受检验噪声限制" % N_MAIN,
             fontsize=9.6)
ax2 = axes[1]
ax2.bar([i - 0.18 for i in range(len(ns))], [p * 100 for p in perfect], 0.36,
        color=PURPLE, label="完整 CPDAG 还原率 %")
ax2.set_ylim(0, 108)
for i, p in enumerate(perfect):
    ax2.text(i - 0.18, p * 100 + 2, "%.0f%%" % (p * 100), ha="center", fontsize=8, color=PURPLE)
ax2b = ax2.twinx()
ax2b.plot([i + 0.22 for i in range(len(ns))], badcnt, "-o", color=ORANGE, lw=2,
          label="平均错误定向弧数")
ax2b.set_ylim(0, max(0.6, max(badcnt) + 0.35)); ax2b.set_ylabel("平均错误定向弧数", color=ORANGE)
ax2b.tick_params(colors=ORANGE)
ax2.set_xticks(range(len(ns))); ax2.set_xticklabels(ns)
ax2.set_xlabel("样本量 n（横轴对数感，等距摆放）")
ax2.axhline(100, color=GRAY, lw=0.8, ls=":")
h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
ax2.set_title("样本量扫描（α=0.01）：弱旁路 S→Vo→Y 的偏相关≈0.08，\nn 小的时候 Fisher-z 把它当成噪声",
              fontsize=9.6)
fig.suptitle("图4 PC 算法的两个命门：α 与 n", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(OUT + "/alpha-scan.png", dpi=130); plt.close(fig)
print("fig4 ok")

# ---------- 图5 下游因子选择：naive vs 因果 ----------
def ols_mse(Mtr, Mte, cols):
    def dm(X):
        return np.column_stack([np.ones(len(X))] + [X[:, c] for c in cols])
    beta, *_ = np.linalg.lstsq(dm(Mtr), Mtr[:, IDX["Y"]], rcond=None)
    return float(np.mean((Mte[:, IDX["Y"]] - dm(Mte) @ beta) ** 2))

def marginal_top3(Mtr):
    cs = [abs(np.corrcoef(Mtr[:, j], Mtr[:, IDX["Y"]])[0, 1]) for j in range(5)]
    return sorted(range(5), key=lambda j: -cs[j])[:3], cs

PARENT_SET = [IDX["Mo"], IDX["V"], IDX["Vo"]]
SEEDS5 = list(range(12))
res = {"Mo2": [], "naive": [], "causal": [], "all5": []}
to_in, vo_out, memb = 0, 0, {}
for sd in SEEDS5:
    gg = np.random.default_rng(900 + sd)
    A = gen(6000, gg); B = gen(6000, gg)
    top, cs = marginal_top3(A)
    res["naive"].append(ols_mse(A, B, top))
    names = tuple(NODES[j] for j in top)
    memb[names] = memb.get(names, 0) + 1
    "To" in names and None
    if "To" in names: to_in += 1
    if "Vo" not in names: vo_out += 1
    res["causal"].append(ols_mse(A, B, PARENT_SET))
    res["all5"].append(ols_mse(A, B, [IDX["S"], IDX["Mo"], IDX["V"], IDX["Vo"], IDX["To"]]))
    res["Mo2"].append(ols_mse(A, B, [IDX["Mo"]]))
mean5 = {k: float(np.mean(v)) for k, v in res.items()}
marginal_mean = float(np.mean([np.mean(C[:, j] ** 2) for j in range(5)]))
STATS["oos"] = dict(means=mean5, naive_members={"+".join(k): v for k, v in memb.items()},
                    pct_seeds_To_selected=100.0 * to_in / len(SEEDS5),
                    pct_seeds_Vo_dropped=100.0 * vo_out / len(SEEDS5),
                    loss_naive_vs_causal_pct=100.0 * (mean5["naive"] / mean5["causal"] - 1))
fig, ax = plt.subplots(figsize=(9.2, 5.4))
labels = ["只用 Mo\n(边际第一)", "朴素 top-3\n(边际相关)", "CPDAG 父集\n{Mo,V,Vo}", "全部 5 因子\n(无脑全都要)"]
colors = [GRAY, RED, GREEN, BLUE]
keys = ["Mo2", "naive", "causal", "all5"]
xx = np.arange(4)
for xi, k, col in zip(xx, keys, colors):
    vals = res[k]
    ax.scatter(np.full(len(vals), xi) + 0.02, vals, s=16, color=col, alpha=0.45, zorder=3)
    m = mean5[k]
    ax.scatter([xi], [m], s=140, marker="D", color=col, zorder=4)
    ax.text(xi, m + 0.011, "%.4f" % m, ha="center", fontsize=10, color=col, weight="bold")
ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("样本外 MSE（n_train=n_test=6000）")
ax.grid(alpha=0.25, axis="y")
lost = (mean5["naive"] / mean5["causal"] - 1) * 100
ax.set_title("图5 排序依据从“边际相关”换成“CPDAG 父集”后的样本外成绩单\n"
             "12 个种子上：朴素 top-3 比因果父集 MSE 高 %.1f%%；To 在 %.0f%% 的种子里混进前 3，"
             "真父集成员 Vo 在 %.0f%% 的种子里被踢掉"
             % (lost, STATS["oos"]["pct_seeds_To_selected"], STATS["oos"]["pct_seeds_Vo_dropped"]),
             fontsize=10)
fig.text(0.5, 0.012, "菱形 = 12 种子均值，圆点 = 单种子。全部 5 因子 ≈ 全知基准：S/To 不增信息，"
         "只带来共线性方差；因果父集以 3 个因子逼近全知基准",
         ha="center", fontsize=8.4, color=GRAY)
fig.tight_layout(rect=[0, 0.03, 1, 0.90]); fig.savefig(OUT + "/selection-oos.png", dpi=130); plt.close(fig)
print("fig5 ok", STATS["oos"])

with open(ROOT + "/_pc_stats.json", "w") as f:
    json.dump(STATS, f, ensure_ascii=False, indent=1)
print("ALL DONE")
