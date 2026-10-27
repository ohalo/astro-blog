# -*- coding: utf-8 -*-
"""
TDA 持续同调实验：手写 Rips 持续同调（beta0 union-find + beta1 相对约化）
输出 5 张图到 public/images/topological-data-analysis-persistence/
结果 json 到 scripts/_tda_ot_out/_tda_results.json
seed=20261027
"""
import json, math, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

SEED = 20261027
RNG = np.random.default_rng(SEED)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "public/images/topological-data-analysis-persistence")
JDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tda_ot_out")
os.makedirs(OUT, exist_ok=True); os.makedirs(JDIR, exist_ok=True)
R = {}  # results

# ---------------- Rips persistence (beta0 + beta1) ----------------

def rips_persistence(P, max_r):
    """P: (n,D) 点集。返回 (b0, b1, essential, n_components)。
    b0/b1: [(birth, death)] 有限持续对；essential: 在 max_r 内未杀的 β1 条 [(birth, None)]。
    算法（Ripser 风格）：β0 用 union-find；β1 以生成森林(union-find 树边)为零边、
    余树边为 H1 生成元基，三角形按最大边长排序后在基上做 Z2 列约化（整数 bitmask）。"""
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    n = len(P)
    iu = np.triu_indices(n, 1)
    order = np.argsort(D[iu], kind="stable")
    E = list(zip(iu[0][order], iu[1][order], D[iu][order]))

    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    birth = [0.0] * n
    b0 = []
    co_tree = []           # (a,b,v) 余树边（环边）
    edge_id = {}           # 余树边 -> bit 位
    for a, b, v in E:
        if v > max_r:
            break
        ra, rb = find(a), find(b)
        if ra != rb:
            # 树边：杀死较晚出生的分量
            if birth[ra] > birth[rb]:
                b0.append((birth[ra], v))
                parent[ra] = rb
            else:
                b0.append((birth[rb], v))
                parent[rb] = ra
        else:
            edge_id[(a, b)] = len(co_tree)
            co_tree.append((a, b, v))
    # 无限 beta0 条（连通分量数），最后一条记 (0, inf)
    roots = {find(i) for i in range(n)}
    finites0 = b0
    # beta1: 三角形列在余树边基上 Z2 约化（树边在商复形中为 0）
    ei = dict(edge_id)
    def ekey(x, y):
        return (x, y) if x < y else (y, x)
    # 全边集邻接表（树边 + 余树边，三边均 <= max_r 才算入复形）
    adj = [set() for _ in range(n)]
    all_edges = set()
    for a, b, v in E:
        if v > max_r:
            break
        adj[a].add(b); adj[b].add(a); all_edges.add(ekey(a, b))
    tri_vals = []
    for (a, b) in all_edges:
        common = adj[a] & adj[b]
        for c in common:
            if not (a < b < c):
                continue
            # 三边中余树边的 bit 异或；树边贡献 0（商掉）
            bits = 0
            for (u, w) in ((a, b), (a, c), (b, c)):
                k = ei.get(ekey(u, w))
                if k is not None:
                    bits ^= (1 << k)
            vtri = max(D[a, b], D[a, c], D[b, c])
            if vtri <= max_r:
                tri_vals.append((vtri, bits))
    tri_vals.sort(key=lambda x: x[0])
    pivots = {}            # pivot bit -> reduced column
    birth_edge = {k: co_tree[k][2] for k in range(len(co_tree))}
    b1 = []; killed = set(); n_sphere = 0
    for vtri, col in tri_vals:
        c = col
        while c:
            p = c.bit_length() - 1
            if p in pivots:
                c ^= pivots[p]
            else:
                pivots[p] = c
                b1.append((birth_edge[p], vtri))  # 边 p 的环在 vtri 处被杀死
                killed.add(p)
                break
        # c==0: 该三角形约化为0（2-闭链/球面），忽略
    # 未配对余树边 = 长寿命特征（在 max_r 内未杀）
    essential = [k for k in range(len(co_tree)) if k not in killed]
    ess_bars = [(birth_edge[k], None) for k in essential]
    conn = 0.0
    for a, b, v in E:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            conn = max(conn, v)
    return finites0, b1, ess_bars, len(roots), conn

def pers_entropy(bars, norm_r=None):
    """有限条的持续熵 (0-1 归一化)"""
    ds = [(d - b) for b, d in bars if d is not None and d > b]
    if not ds:
        return 0.0
    p = np.array(ds); p = p / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0

def max_bar(bars, cap=None):
    """最长条长度；未杀条（death=None）以 cap 截断计长。"""
    if not bars:
        return 0.0
    C = 1e9 if cap is None else cap
    return max(((C if d is None else min(d, C)) - b) for b, d in bars)

# ---------------- 数据生成 ----------------

def delay_embed(x, e):
    return np.array([x[i:i + e] for i in range(len(x) - e + 1)])

def pca2(X):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T

def gen_ou(n, p, snr, rng, burn=4000):
    """均值回归“震荡信号”：AR(2) 平稳振荡 x_t=a1 x_{t-1}+a2 x_{t-2}+eps。
    用最优一步预测 f_t=E[x_t|x_{<t}] 定义“可预测振荡分量”，残差 d_t=x_t-f_t 定义噪声，
    输出 y_t = 标准化(信号) + snr·标准化(残差)，使 snr = 噪声/振幅比（残差尺度上）。"""
    T = burn + n + 1
    ang = 2 * math.pi / p
    r = 0.995
    a1, a2 = 2 * r * math.cos(ang), -r * r
    eps = rng.standard_normal(T)
    x = np.zeros(T)
    for t in range(2, T):
        x[t] = a1 * x[t - 1] + a2 * x[t - 2] + eps[t]
    f = np.zeros(T)
    for t in range(2, T):
        f[t] = a1 * x[t - 1] + a2 * x[t - 2]
    d = x - f
    sl = slice(burn, burn + n)
    sig = f[sl] - f[sl].mean()
    noi = d[sl]
    sig = sig / sig.std()
    noi = noi / noi.std() * snr
    return sig + noi

def gen_rw(n, rng):
    z = rng.standard_normal(n + 500); z = z - z.mean()
    return np.cumsum(z)[-n:]

def embed_points(y, n, e, rng, m=100):
    X = delay_embed(y, e)
    if len(X) > m:
        idx = rng.choice(len(X) - 2, size=m, replace=False)
        X = X[idx]
    return pca2(X)

# ---------------- 教学示例数据：带噪圆 vs 高斯云 ----------------
rng0 = np.random.default_rng(SEED)
nc = 110
th = np.sort(rng0.uniform(0, 2 * math.pi, nc))
circ = np.stack([np.cos(th), np.sin(th)], 1) + rng0.standard_normal((nc, 2)) * 0.07
gauss = rng0.standard_normal((nc, 2))
CAP = 2.6
t0 = time.time()
b0c, b1c, esc, ncc, connc = rips_persistence(circ, CAP)
b0g, b1g, esg, ncg, conng = rips_persistence(gauss, CAP)
t_compute = time.time() - t0
R["demo"] = {
    "n": nc, "cap": CAP,
    "circle_b1": [(round(b, 3), round(d, 3)) for b, d in sorted(b1c, key=lambda x: -(x[1] - x[0]))[:6]],
    "circle_b1_top_len": round(max_bar(b1c), 3),
    "circle_essential": len(esc),
    "gauss_b1": [(round(b, 3), round(d, 3)) for b, d in sorted(b1g, key=lambda x: -(x[1] - x[0]))[:6]],
    "gauss_b1_top_len": round(max_bar(b1g), 3),
    "gauss_essential": len(esg),
    "compute_sec": round(t_compute, 1),
}

# fig1 数据
fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.4))
for a, dat, ttl in [(ax[0], circ, "带噪圆周（n=110，噪声 σ=0.07）"),
                    (ax[1], gauss, "标准高斯云（n=110）")]:
    a.scatter(dat[:, 0], dat[:, 1], s=18, c="#3b6ea5", alpha=0.85, ec="none")
    a.set_title(ttl, fontsize=11); a.set_aspect("equal")
    a.axhline(0, c="#ddd", lw=0.6); a.axvline(0, c="#ddd", lw=0.6)
fig.suptitle("图1  两种点云：一个有“洞”，一个没有", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_point_clouds.png", dpi=140); plt.close(fig)

# fig2 持续图
fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.6))
for a, (b0, b1, ttl) in [(ax[0], (b0c, b1c, "带噪圆周")), (ax[1], (b0g, b1g, "高斯云"))]:
    for b, d in b0: a.scatter(b, d, s=14, c="#c0504d", alpha=0.5, label="β0" if (b, d) == b0[0] else None)
    labs = sorted(b1, key=lambda x: -(x[1] - x[0]))
    for k, (b, d) in enumerate(labs):
        a.scatter(b, d, s=34, c="#4472c4", marker="D",
                  label="β1" if k == 0 else None)
        if d - b > 0.8:
            a.annotate(f"{d-b:.2f}", (b, d), fontsize=9, xytext=(4, 6),
                       textcoords="offset points", c="#1f4e9c")
    a.plot([0, CAP], [0, CAP], "k--", lw=0.7)
    a.set_xlim(0, CAP); a.set_ylim(0, CAP)
    a.set_xlabel("出生 ε"); a.set_ylabel("死亡 ε"); a.set_title(ttl, fontsize=11); a.legend(fontsize=8)
fig.suptitle("图2  Rips 持续图：圆周有一个显著的长 β1 条，高斯云没有", fontsize=12.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_persistence_diagrams.png", dpi=140); plt.close(fig)

# ---------------- 金融主实验：OU 震荡 vs 随机游走的延迟嵌入 ----------------
E = 10
CAPT = 1.8
def statistic(y, rng, m=80):
    """统计量：延迟嵌入点云 Rips 图中，死亡不超过 CAPT 的最长有限 β1 条长度。
    数据先按整体 std 归一，CAPT 是“标准差单位”下的固定尺度。"""
    Pts = embed_points(y, 900, E, rng, m=m)
    _, b1, es, nc, conn = rips_persistence(Pts, CAPT)
    return max([d - a for a, d in b1 if d is not None] + [0.0]), conn

# 功效/虚警：snr 从 0.2 到 3.0，每个 12 次模拟；RW 零假设 60 次
snrs = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
rng1 = np.random.default_rng(SEED + 1)
null_stats = []
for i in range(60):
    y = gen_rw(900, rng1)
    s, _ = statistic(y, rng1)
    null_stats.append(s)
null_stats = np.array(null_stats)
thr = float(np.quantile(null_stats, 0.95))   # 5% 虚警阈值（由零假设样本估）
power = {}
ou_samples = {}
for snr in snrs:
    ss = []
    for i in range(12):
        y = gen_ou(900, 24, snr, rng1)
        s, _ = statistic(y, rng1)
        ss.append(s)
    power[snr] = float(np.mean(np.array(ss) > thr))
    ou_samples[snr] = ss
R["power"] = {"E": E, "period": 24, "null_n": 60, "per_snr_n": 12,
              "cap": CAPT, "thr_95": round(thr, 4),
              "null_mean": round(float(null_stats.mean()), 4),
              "power": {str(k): round(v, 3) for k, v in power.items()},
              "ou_median": {str(k): round(float(np.median(v)), 3) for k, v in ou_samples.items()}}

fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.4))
ax[0].hist(null_stats, bins=18, color="#8faadc", alpha=0.9, label="RW 零假设 (60)")
for snr, col in zip([0.5, 1.0, 2.0], ["#e8a33d", "#c0504d", "#4f8a3b"]):
    ax[0].hist(ou_samples[snr], bins=12, alpha=0.45, color=col,
               label=f"OU 噪声/振幅={snr}")
ax[0].axvline(thr, c="k", ls="--", lw=1.1)
ax[0].text(thr * 1.03, ax[0].get_ylim()[1] * 0.86, f"5% 虚警阈值\n{thr:.2f}", fontsize=8.5)
ax[0].set_xlabel("最长 β1 条长度 (ε)"); ax[0].set_ylabel("次数"); ax[0].legend(fontsize=7.5)
ax[0].set_title("(a) 统计量分布：RW vs OU", fontsize=11)
sks = list(power.keys()); pws = [power[k] for k in sks]
ax[1].plot(sks, pws, "o-", c="#c0504d", lw=1.8, ms=6)
ax[1].axhline(0.05, c="grey", ls="--", lw=1)
ax[1].set_xlabel("噪声/振幅比"); ax[1].set_ylabel("检出率 (power)")
ax[1].set_ylim(-0.03, 1.05); ax[1].set_title("(b) 检出功效曲线（周期 p=24）", fontsize=11)
fig.suptitle("图3  用拓扑统计量区分“真震荡”与随机游走：噪声压过振幅就失效", fontsize=12.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_power_vs_noise.png", dpi=140); plt.close(fig)

# fig4 周期敏感性：固定 snr=1.0，嵌入维 e 与周期 p 的组合功效
ps = [10, 16, 24, 36, 60]; es_list = [6, 10, 14, 20]
sens = {}
rng2 = np.random.default_rng(SEED + 2)
for e in es_list:
    row = []
    for p in ps:
        cnt = 0; K = 8
        for i in range(K):
            y = gen_ou(900, p, 1.0, rng2)
            Pts = embed_points(y, 900, e, rng2, m=70)
            _, b1, elist, _, conn = rips_persistence(Pts, CAPT)
            row_val = max([d - a for a, d in b1 if d is not None] + [0.0])
            cnt += row_val > thr
        row.append(cnt / K)
    sens[e] = row
R["sensitivity"] = {"ps": ps, "E": es_list, "snr": 1.0, "per_cell": 8,
                    "power": {str(k): v for k, v in sens.items()}}
fig, ax = plt.subplots(figsize=(7.6, 4.4))
for e in es_list:
    ax.plot(ps, sens[e], "o-", label=f"嵌入维 e={e}", lw=1.6)
ax.axhline(0.05, c="grey", ls="--", lw=1)
ax.set_xscale("log"); ax.set_xticks(ps); ax.set_xticklabels(ps)
ax.set_xlabel("震荡周期 p（交易日）"); ax.set_ylabel("检出率（噪声/振幅=1.0）")
ax.set_ylim(-0.05, 1.08); ax.legend(fontsize=8.5)
ax.set_title("图4  反直觉：低嵌入维 e=6 反而最强，高维嵌入把噪声维数放大", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_period_sensitivity.png", dpi=140); plt.close(fig)

# fig5 TDA 统计量 vs 传统方差比统计量（同为 5% 校准阈值）的功效对比
def var_ratio(y, k=5):
    r = np.diff(y)
    v1 = r.var(); n = len(r)
    rk = np.array([np.corrcoef(r[:-j], r[j:])[0, 1] for j in range(1, k + 1)])
    vr = (v1 + 2 * (rk * v1 * (1 - np.arange(1, k + 1) / n)).sum()) / v1
    return vr
rng3 = np.random.default_rng(SEED + 3)
K_ORACLE = 12   # 诚实对比：VR 的 k 取半个信号周期 p/2=12（先验已知周期）
vr_null = []
for i in range(60):
    y = gen_rw(900, rng3)
    vr_null.append(var_ratio(y, K_ORACLE))
vr_null = np.array(vr_null); vr_thr = float(np.quantile(vr_null, 0.95))
cmp_snr = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
tda_p = [power[s] for s in cmp_snr]
vr_p = []
for s in cmp_snr:
    cnt = 0; K = 12
    for i in range(K):
        y = gen_ou(900, 24, s, rng3)
        cnt += var_ratio(y, K_ORACLE) < vr_thr   # 均值回归 -> VR<1
    vr_p.append(cnt / K)
R["compare"] = {"vr_lag_oracle": K_ORACLE, "vr_thr_95": round(vr_thr, 4),
                "vr_null_mean": round(float(vr_null.mean()), 4),
                "snrs": cmp_snr, "tda_power": tda_p, "vr_power": vr_p}
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.plot(cmp_snr, tda_p, "o-", c="#c0504d", lw=1.9, label="TDA 最长 β1 条")
ax.plot(cmp_snr, vr_p, "s--", c="#3b6ea5", lw=1.9, label="方差比 VR(12)（传统，周期先验）")
ax.axhline(0.05, c="grey", ls="--", lw=1)
ax.set_xlabel("噪声/振幅比"); ax.set_ylabel("检出率（p=24 震荡，5% 校准）")
ax.set_ylim(-0.03, 1.06); ax.legend(fontsize=9)
ax.set_title("图5  单变量均值回归任务上，传统 VR 统计量功效不输 TDA", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_vs_vr.png", dpi=140); plt.close(fig)

with open(f"{JDIR}/_tda_results.json", "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False)[:1200])
print("OK figs:", sorted(os.listdir(OUT)))
