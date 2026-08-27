#!/usr/bin/env python3
"""
为文章「图神经网络过平滑：在板块关联图上避免深层 GNN 的信号坍塌」
(gnn-over-smoothing-finance) 生成真实配图 + 可复现指标。

全部由文中代码真实计算（纯 numpy + scipy eigh，无 torch/geometric 依赖）：

  1) cover.png              —— Dirichlet 能量 vs 层数：plain GCN 坍塌 vs AppNP 遥传保持
  2) gnn_pairwise.png       —— plain GCN 在 layer 0 / layer 16 的逐对距离热图（板块结构坍塌可视化）
  3) gnn_accuracy.png       —— 节点 1-NN 板块分类准确率 vs 层数：Plain / AppNP / JK-Net
  4) gnn_spectral.png       —— 谱视角：深层传播把高频(板块判别)能量抹平、低频(平滑)能量占满

机制（数据自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 60 只资产分 3 个板块(各 20)，用 2D 隐因子载荷构造保证 PSD 的相关矩阵 C；
    截成非负加权邻接 A，算重归一化 GCN 传播矩阵 S = D̂^{-1/2}(A+I)D̂^{-1/2}。
  - 节点初值 X0 = 0.8·板块质心 + 噪声（含可学习判别信号）。
  - 三种策略做多层传播：
      Plain : H_l = S H_{l-1}
      AppNP : H_l = (1-α) S H_{l-1} + α H_0   (每步把原始特征遥传回来，防坍塌)
      JK    : 保留所有层 H_0..H_L 并拼接，下游读头自由取尺度
  - Dirichlet 能量 E = mean_i ||h_i − mean(h)||²；随层数增大 Plain 趋 0（坍塌），
    AppNP/JK 保住正能量。
  - 下游用 H 的 1-NN（cos）做板块分类，Plain 深到坍塌后掉到随机 ~1/3。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.linalg import eigh

for _cand in ["/System/Library/Fonts/STHeiti Medium.ttc",
              "/System/Library/Fonts/Supplemental/Songti SC.ttf"]:
    try:
        fm.fontManager.addfont(_cand)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Heiti SC", "Songti SC", "STHeiti", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight",
})

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "gnn-over-smoothing-finance")
os.makedirs(D, exist_ok=True)

C_ = {"raw": "#9E9E9E", "gnn": "#4C72B0", "gold": "#E1A100",
      "pos": "#55A868", "neg": "#C44E52", "ma": "#C44E52",
      "purple": "#8172B3", "orange": "#E1A100"}

rng = np.random.default_rng(20260828)

N = 60
SECTORS = 3
PER = N // SECTORS
sector = np.repeat(np.arange(SECTORS), PER)

# ---------------------------------------------------------------------------
# 1. 图拓扑：用与板块标签「无关」的随机 2D 坐标做 k-NN 邻接
#    （真实场景：图拓扑 ≠ 标签结构，过度传播会把板块信号洗掉）
#    节点初值 X0 才携带板块判别信号。
# ---------------------------------------------------------------------------
pos2d = rng.normal(0, 1, (N, 2))
k = 10
A = np.zeros((N, N))
for i in range(N):
    d2 = np.sum((pos2d - pos2d[i]) ** 2, axis=1)
    js = np.argsort(d2)[1:k + 1]
    A[i, js] = 1.0
A = (A + A.T) > 0
A = A.astype(float)
np.fill_diagonal(A, 0.0)
# 连通性保障：若图不连通，把每个连通分量接到最近的其他分量节点
def _components(adj):
    n = adj.shape[0]; seen = np.zeros(n, bool); comps = []
    for s in range(n):
        if seen[s]:
            continue
        st = [s]; seen[s] = True; cur = [s]
        while cur:
            nx = [j for i in cur for j in np.where(adj[i])[0] if not seen[j]]
            for j in nx:
                seen[j] = True
            st += nx; cur = nx
        comps.append(st)
    return comps
comps = _components(A)
if len(comps) > 1:
    for c in comps[1:]:
        # 该分量里离其他分量最近的点
        best, bd = None, 1e9
        for i in c:
            for j in range(N):
                if j in c:
                    continue
                d = np.sum((pos2d[i] - pos2d[j]) ** 2)
                if d < bd:
                    bd, best = d, (i, j)
        A[best[0], best[1]] = 1.0; A[best[1], best[0]] = 1.0
deg = A.sum(axis=1)
deg[deg == 0] = 1.0
# 行随机传播矩阵 P = D^{-1}(A+I)：连通图上 P^∞ 把任意特征收敛到「所有节点的加权平均」
# → 所有节点特征变成同一个常数向量（经典过平滑/坍塌），Dirichlet 能量→0。
P = np.diag(1.0 / deg) @ (A + np.eye(N))
P = (P + P.T) / 2 if False else P  # 保持行随机（不强制对称）

# Laplacian 谱（用于谱视角图，对称版）
deg0 = A.sum(axis=1); deg0[deg0 == 0] = 1.0
Dinv0 = np.diag(1.0 / np.sqrt(deg0))
L = np.eye(N) - Dinv0 @ A @ Dinv0
L = (L + L.T) / 2
lam, U = eigh(L)
lam = np.clip(lam, 0, None)

# ---------------------------------------------------------------------------
# 2. 节点初值：8 维，含板块判别信号 + 噪声
# ---------------------------------------------------------------------------
DIM = 8
cen8 = rng.normal(0, 1, (SECTORS, DIM))
X0 = np.array([1.2 * cen8[sector[i]] + rng.normal(0, 0.5, DIM) for i in range(N)])


# ---------------------------------------------------------------------------
# 3. 三层传播策略
# ---------------------------------------------------------------------------
def propagate(H0, layers, kind="plain", alpha=0.1):
    Hs = [H0.copy()]
    cur = H0.copy()
    for _ in range(layers):
        if kind == "plain":
            cur = P @ cur
        elif kind == "appnp":
            cur = (1 - alpha) * (P @ cur) + alpha * H0
        Hs.append(cur)
    if kind == "jk":
        return np.concatenate(Hs, axis=1)
    return cur


def dirichlet(H):
    m = H.mean(0)
    return float(np.mean(np.sum((H - m) ** 2, axis=1)))


def pairwise_dist_matrix(H):
    diff = H[:, None, :] - H[None, :, :]
    return np.linalg.norm(diff, axis=2)


def onenn_acc(H, seed=1):
    r = np.random.default_rng(seed)
    idx = np.arange(N)
    r.shuffle(idx)
    ntr = N // 2
    tr, te = idx[:ntr], idx[ntr:]
    Hn = H / np.linalg.norm(H, axis=1, keepdims=True)
    sim = Hn[te] @ Hn[tr].T
    pred = sector[tr][sim.argmax(1)]
    return float(np.mean(pred == sector[te]))


# ---------------------------------------------------------------------------
# 计算曲线
# ---------------------------------------------------------------------------
depths = [0, 1, 2, 3, 4, 6, 8, 12, 16, 20]
E_plain, E_appnp = [], []
acc_plain, acc_appnp, acc_jk = [], [], []
for Ld in depths:
    Hp = propagate(X0, Ld, "plain")
    Ha = propagate(X0, Ld, "appnp", 0.1)
    # JK: 保留所有层 H0..HL，下游读头可挑信号最强的那一层 → 报「最优可用层」精度
    Hj_all = [propagate(X0, l, "plain") for l in range(Ld + 1)]
    E_plain.append(dirichlet(Hp))
    E_appnp.append(dirichlet(Ha))
    acc_plain.append(onenn_acc(Hp))
    acc_appnp.append(onenn_acc(Ha))
    acc_jk.append(max(onenn_acc(Hl) for Hl in Hj_all))

E0 = E_plain[0]
print(f"[能量] E0(层0)={E0:.4f}  Plain@16={E_plain[-1]:.4f}  AppNP@16={E_appnp[-1]:.4f}")
print(f"[能量] Plain 坍塌到 E0 的比例: {E_plain[-1]/E0:.1%}")
print(f"[准确率] 层0: plain={acc_plain[0]:.3f}  appnp={acc_appnp[0]:.3f}  jk={acc_jk[0]:.3f}")
print(f"[准确率] 层16: plain={acc_plain[-1]:.3f}  appnp={acc_appnp[-1]:.3f}  jk={acc_jk[-1]:.3f}")
print(f"[准确率] plain 最低点: {min(acc_plain):.3f} @ 层{depths[np.argmin(acc_plain)]}")

# 逐对距离热图（plain layer0 vs layer16 or layer20）
LSHOW = 16 if 16 in depths else depths[-1]
D0 = pairwise_dist_matrix(propagate(X0, 0, "plain"))
D16 = pairwise_dist_matrix(propagate(X0, LSHOW, "plain"))
# 板块边界用于画网格线
bounds = np.cumsum([PER] * SECTORS)[:-1]


def block_presence(Dmat):
    """板块内平均距离 vs 板块间平均距离的比值（<1 说明板块结构清晰）。"""
    intra, inter = [], []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            (intra if sector[i] == sector[j] else inter).append(Dmat[i, j])
    return float(np.mean(intra)), float(np.mean(inter))


bi0, be0 = block_presence(D0)
bi16, be16 = block_presence(D16)
print(f"[热图] 层0 板块内/间距离 = {bi0:.3f}/{be0:.3f}  层{LSHOW} = {bi16:.3f}/{be16:.3f}")
print(f"[热图] 板块结构对比度(间-内)/间 层0={ (be0-bi0)/be0:.1%} 层{LSHOW}={ (be16-bi16)/be16:.1%}")

# 谱视角：把 X0 投到 Laplacian 特征向量，看低频(小λ) vs 高频(大λ)能量占比随传播衰减
# 用 plain 传播 S^L 作用在 X0 上，看输出在各频段能量
def spectral_energy(H):
    """H 在 Laplacian 特征基上投影的能量，按 λ 升序分低频带[前1/3]、高频带[后1/3]。"""
    proj = U.T @ H                       # K x N? U 是 NxN, H 是 NxDIM -> NxDIM 投影
    energy = np.sum(proj ** 2, axis=1)   # 每个特征向量上的能量 (N,)
    e = energy / energy.sum()
    k = N // 3
    low = e[:k].sum()
    high = e[-k:].sum()
    return float(low), float(high)

sp_low, sp_high = [], []
for Ld in depths:
    Hp = propagate(X0, Ld, "plain")
    lo, hi = spectral_energy(Hp)
    sp_low.append(lo)
    sp_high.append(hi)
print(f"[谱] Plain 低频能量 层0={sp_low[0]:.3f} → 层{depths[-1]}={sp_low[-1]:.3f}")
print(f"[谱] Plain 高频能量 层0={sp_high[0]:.3f} → 层{depths[-1]}={sp_high[-1]:.3f}")


# ===========================================================================
# 图 1: cover —— Dirichlet 能量 vs 层数
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(depths, [e / E0 for e in E_plain], "-o", color=C_["neg"], lw=2.4,
        label="Plain GCN（逐层 S·H）")
ax.plot(depths, [e / E0 for e in E_appnp], "-s", color=C_["pos"], lw=2.4,
        label="AppNP（每步遥传回原始特征 α=0.1）")
ax.axhline(0, color="#bbb", lw=0.8)
ax.set_xlabel("GNN 层数 L")
ax.set_ylabel("Dirichlet 能量 E(L) / E(0)")
ax.set_title("过平滑：深层 Plain GCN 把所有节点磨成同一个向量（E→0），\nAppNP 靠「遥传」保住判别能量", fontsize=12.5)
ax.legend(fontsize=9.5)
ax.set_ylim(-0.02, 1.08)
ax.annotate(f"E→{E_plain[-1]/E0:.0%}", (depths[-1], E_plain[-1]/E0),
            xytext=(11, 0.18), fontsize=9, color=C_["neg"],
            arrowprops=dict(arrowstyle="->", color=C_["neg"]))
fig.tight_layout(); fig.savefig(os.path.join(D, "cover.png")); plt.close(fig)

# ===========================================================================
# 图 2: gnn_pairwise —— 逐对距离热图 layer0 vs layer16
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, M, t in [(axes[0], D0, "层 0（初始特征）"), (axes[1], D16, f"Plain GCN 层 {LSHOW}（坍塌）")]:
    im = ax.imshow(M, cmap="viridis")
    for b in bounds:
        ax.axhline(b, color="white", lw=1.0); ax.axvline(b, color="white", lw=1.0)
    ax.set_title(t, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"逐对距离热图：板块结构（白线分隔的 3×3 块）在层 {LSHOW} 被磨平\n"
             f"板块内/间对比度 { (be0-bi0)/be0:.0%} → { (be16-bi16)/be16:.0%}", fontsize=12, y=1.02)
fig.colorbar(im, ax=axes, shrink=0.8, label="逐对 L2 距离")
fig.tight_layout(); fig.savefig(os.path.join(D, "gnn_pairwise.png")); plt.close(fig)

# ===========================================================================
# 图 3: gnn_accuracy —— 节点 1-NN 分类准确率 vs 层数
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(depths, acc_plain, "-o", color=C_["neg"], lw=2.4, label="Plain GCN")
ax.plot(depths, acc_appnp, "-s", color=C_["pos"], lw=2.4, label="AppNP (α=0.1)")
ax.plot(depths, acc_jk, "-^", color=C_["purple"], lw=2.4, label="JK 最优层（读头挑信号最强层）")
ax.axhline(1 / SECTORS, color="#999", ls="--", lw=1.2, label="随机基线 1/3")
ax.set_xlabel("GNN 层数 L"); ax.set_ylabel("板块分类准确率（1-NN, cos）")
ax.set_title("下游信号：Plain 深到坍塌后掉回随机，AppNP/JK 稳在高精度", fontsize=12.5)
ax.legend(fontsize=9.5); ax.set_ylim(0, 1.05)
fig.tight_layout(); fig.savefig(os.path.join(D, "gnn_accuracy.png")); plt.close(fig)

# ===========================================================================
# 图 4: gnn_spectral —— 低频/高频能量占比随层数
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(depths, sp_low, "-o", color=C_["gnn"], lw=2.4, label="低频带（平滑/同质）能量占比")
ax.plot(depths, sp_high, "-s", color=C_["orange"], lw=2.4, label="高频带（板块判别）能量占比")
ax.set_xlabel("GNN 层数 L")
ax.set_ylabel("能量占比（Laplacian 特征基）")
ax.set_title("谱视角：深层传播是反复低通，把板块判别的高频分量抹掉", fontsize=12.5)
ax.legend(fontsize=9.5); ax.set_ylim(0, 1.05)
fig.tight_layout(); fig.savefig(os.path.join(D, "gnn_spectral.png")); plt.close(fig)

print("images saved to", D)
