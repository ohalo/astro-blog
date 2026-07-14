#!/usr/bin/env python3
"""
为文章「智能订单路由(SOR)：在多个交易场所之间把大单拆成最优流」生成真实配图 + 计算正文引用的关键数字。

机制（自洽合成，仅用于演示；落地见文末路径）：
  * 5 个交易场所（venue），每个有不同：每期可用深度 D_v、半价差 spread/2、taker 费用 fee、
    以及临时冲击系数 gamma_v。单笔 child 切片 q 在某场所的成本（相对到达中间价的 bps）：
        cost_bps(q) = spread/2 + fee + gamma_v * (q/D_v)^2 * 10000
    临时冲击是切片占深度比例的凸函数（二次），深度越浅、系数越大，塞大单越贵。
  * 父单 Q=200,000 股，在 T=25 期内均匀拆成 child（每期 S=8,000），每期把 S 分配到 5 个场所。
  * 三种路由：
        - 等权拆分（naive）：每期 S 均分 5 份，无视成本与深度；
        - 静态成本分配（semi-smart）：用 t=0 的静态成本做一次凸优化（water-filling），之后每期照旧；
        - 智能自适应 SOR（smart）：每期用「当前」各场所深度重解凸优化，随流动性事件转移仓位。
  * 各场所深度逐期随机波动（对数正态），并不时触发「流动性枯竭」事件（某场所深度骤降 3 期），
    用以体现自适应 SOR 相对静态/等权的优势。
  * 衡量：实现短缺（implementation shortfall, bps，相对到达中间价）。越低越好。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "smart-order-routing-venues")
os.makedirs(D, exist_ok=True)

C = {"ot": "#4C72B0", "mean": "#C44E52", "equal": "#999999",
     "target": "#55A868", "grid": "#DDDDDD", "warn": "#DD8452", "calm": "#8172B3",
     "smart": "#2A6F97", "static": "#E9C46A", "equal2": "#B0B0B0"}

# ---------------- 场所参数（基础）----------------
# name, depth(股/期), spread_half(bps), fee(bps), gamma(冲击系数)
VEN = [
    ("场所A·深流动性", 12000, 1.0, 0.10, 0.0040),
    ("场所B·均衡",      8000,  2.0, 0.30, 0.0060),
    ("场所C·极窄价差",  3000,  0.5, 0.00, 0.0025),
    ("场所D·贵且浅",    6000,  5.0, 1.00, 0.0100),
    ("场所E·中等",      5000,  3.0, 0.50, 0.0070),
]
names = [v[0] for v in VEN]
DEPTH0 = np.array([v[1] for v in VEN], float)
SH     = np.array([v[2] for v in VEN], float)
FEE    = np.array([v[3] for v in VEN], float)
GAM    = np.array([v[4] for v in VEN], float)
N = len(VEN)

Q = 200_000          # 父单总股数
T = 25               # 拆单期数
S = Q / T            # 每期切片 = 8000


def slice_cost(q, Dv, v):
    """单笔切片在某场所的成本（bps，相对到达中间价）。"""
    q = max(q, 0.0)
    impact = GAM[v] * (q / Dv) ** 2 * 10000.0
    return SH[v] + FEE[v] + impact


def waterfill(S_total, Dvec):
    """凸优化：给定每期总切片 S_total 与各场所当前可用深度 Dvec，
    最小化 sum cost(q_v) s.t. sum q_v = S_total, 0<=q_v<=Dvec。
    逐场所边际成本斜率 m_v = 2*GAM_v*10000/D_v^2，最优 q_v ∝ 1/m_v = D_v^2/GAM_v。
    用 water-filling：先按 1/m_v 比例分，再处理深度上限封顶。"""
    m = 2.0 * GAM * 10000.0 / (Dvec ** 2 + 1e-12)
    # 无限斜率（深度 0）直接排除
    active = Dvec > 1.0
    if not active.any():
        return np.zeros(N)
    w = 1.0 / m[active]
    w = w / w.sum()
    q = np.zeros(N)
    q[active] = S_total * w
    # 封顶：超过当前深度的部分，回收后按剩余比例再分配（迭代几次即可）
    for _ in range(20):
        over = q > Dvec
        if not over.any():
            break
        excess = (q[over] - Dvec[over]).sum()
        q[over] = Dvec[over]
        still = (~over) & (Dvec > 0)
        if not still.any():
            break
        rem = S_total - q.sum()
        if rem <= 0:
            break
        m2 = 2.0 * GAM[still] * 10000.0 / (Dvec[still] ** 2 + 1e-12)
        w2 = (1.0 / m2); w2 = w2 / w2.sum()
        q[still] = q[still] + rem * w2
    # 兜底：若仍差一点（数值），平摊到 active
    diff = S_total - q.sum()
    if abs(diff) > 1e-6 and (Dvec > 0).any():
        act = Dvec > 0
        q[act] += diff / act.sum()
    return np.clip(q, 0, Dvec)


def depth_path(rng):
    """生成 T 期内各场所可用深度（含随机波动 + 偶发枯竭事件）。"""
    Dmat = np.zeros((T, N))
    mult = np.ones(N)
    for t in range(T):
        # 对数正态波动
        mult = mult * rng.lognormal(0, 0.18, N)
        mult = np.clip(mult, 0.35, 3.0)
        # 偶发流动性枯竭：随机选一个场所深度骤降，持续 3 期
        if rng.random() < 0.08:
            v = rng.integers(N)
            mult[v] *= 0.25
        Dmat[t] = DEPTH0 * mult
    return Dmat


def route_equal(Dmat):
    q = np.full((T, N), S / N)
    return q


def route_static(Dmat):
    """用 t=0 深度做一次 water-filling，之后每期照旧（忽略动态）。"""
    q0 = waterfill(S, Dmat[0])
    return np.tile(q0, (T, 1))


def route_smart(Dmat):
    return np.array([waterfill(S, Dmat[t]) for t in range(T)])


def shortfall(qmat, Dmat):
    """实现短缺（bps，相对到达中间价）：按切片股数加权的平均单切片成本。
    总执行成本(货币) = Σ_v,t cost_bps(q)*q/10000 * mid；总名义 = Q*mid。
    故总短缺 bps = Σ_v,t [cost_bps(q)*q] / Q —— 即按股数加权的平均单切片成本。"""
    total = 0.0
    for t in range(T):
        for v in range(N):
            total += slice_cost(qmat[t, v], Dmat[t, v], v) * qmat[t, v]
    return total / Q  # 数量加权平均，单位 bps


# ---------------- 蒙特卡洛评估 ----------------
rng0 = np.random.default_rng(20260714)
NSIM = 300
res_equal, res_static, res_smart = [], [], []
rep_qsmart, rep_qequal, rep_D = None, None, None
for s in range(NSIM):
    rng = np.random.default_rng(1000 + s)
    Dmat = depth_path(rng)
    se = shortfall(route_equal(Dmat), Dmat)
    ss = shortfall(route_static(Dmat), Dmat)
    sm = shortfall(route_smart(Dmat), Dmat)
    res_equal.append(se); res_static.append(ss); res_smart.append(sm)
    if s == 7:  # 存一条代表性路径用于配图
        rng = np.random.default_rng(1007)
        rep_D = depth_path(rng)
        rep_qsmart = route_smart(rep_D)
        rep_qequal = route_equal(rep_D)

res_equal = np.array(res_equal)
res_static = np.array(res_static)
res_smart = np.array(res_smart)

print("=" * 70)
print("智能订单路由(SOR) 关键数字 (seed 20260714, %d sims)" % NSIM)
print("=" * 70)
print(f"父单 Q={Q:,} 股, 拆 {T} 期, 每期切片 S={S:.0f} 股")
print("各场所基础参数:")
for i, n in enumerate(names):
    print(f"  {n:14s} 深度={DEPTH0[i]:.0f} 半价差={SH[i]:.1f}bps 费={FEE[i]:.2f}bps γ={GAM[i]}")
print("\n实现短缺 (bps, 越低越好)  — 均值 / 中位数 / 标准差:")
print(f"  等权拆分(naive)   : {res_equal.mean():.2f} / {np.median(res_equal):.2f} / {res_equal.std():.2f}")
print(f"  静态成本分配       : {res_static.mean():.2f} / {np.median(res_static):.2f} / {res_static.std():.2f}")
print(f"  智能自适应 SOR     : {res_smart.mean():.2f} / {np.median(res_smart):.2f} / {res_smart.std():.2f}")
print(f"\nSOR 相对等权节省: {(1-res_smart.mean()/res_equal.mean())*100:.1f}%")
print(f"SOR 相对静态节省: {(1-res_smart.mean()/res_static.mean())*100:.1f}%")
print(f"静态相对等权节省: {(1-res_static.mean()/res_equal.mean())*100:.1f}%")

# ============================================================================
# 图 1：场所全景 —— 参考切片下的单场所成本 + 深度
# ============================================================================
ref_q = S / N  # 等权参考切片
standalone = np.array([slice_cost(ref_q, DEPTH0[i], i) for i in range(N)])
fig, ax = plt.subplots(figsize=(10, 5.4))
x = np.arange(N)
bars = ax.bar(x, standalone, color=[C["smart"], C["calm"], C["target"], C["warn"], C["ot"]], alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels([n.split("·")[0] for n in names], fontsize=9)
ax.set_ylabel("参考切片成本 (bps)")
ax.set_title("五个场所的单切片成本全景（等权参考切片 %.0f 股，基础深度）" % ref_q)
for i, b in enumerate(bars):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.4, f"{standalone[i]:.1f}",
            ha="center", va="bottom", fontsize=9)
ax2 = ax.twinx()
ax2.plot(x, DEPTH0, "o--", color="#333333", lw=1.3, ms=7, label="基础深度(股)")
ax2.set_ylabel("基础深度 (股/期)"); ax2.set_ylim(0, DEPTH0.max()*1.3)
ax2.legend(fontsize=8, loc="upper right")
ax.grid(True, color=C["grid"], lw=0.6, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(D, "venue_landscape.png"), dpi=130); plt.close()

# ============================================================================
# 图 2：单场所成本曲线（凸性）—— 展示「塞大单非线性变贵」
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5.2))
for i, col in [(0, C["smart"]), (2, C["target"]), (3, C["warn"])]:
    qv = np.linspace(0, DEPTH0[i], 120)
    cv = np.array([slice_cost(q, DEPTH0[i], i) for q in qv])
    ax.plot(qv, cv, color=col, lw=2.0, label=f"{names[i].split('·')[0]} (γ={GAM[i]})")
ax.set_xlabel("该场所切片股数 q (深度=%d)" % 1)  # 占位，下一行修正
ax.set_xlabel("该场所切片股数 q")
ax.set_ylabel("切片成本 (bps)")
ax.set_title("临时冲击的凸性：单场所成本随切片非线性上升（深度=基础值）")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "cost_curves.png"), dpi=130); plt.close()

# ============================================================================
# 图 3：路由热力图 —— 代表性路径下，每期切片在各场所的占比（SOR vs 等权）
# ============================================================================
share_smart = rep_qsmart / rep_qsmart.sum(1, keepdims=True)
share_equal = rep_qequal / rep_qequal.sum(1, keepdims=True)
fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
for ax, share, title, cmap in [
    (axes[0], share_smart, "智能自适应 SOR：每期切片在各场所的占比（随流动性转移）", "YlGnBu"),
    (axes[1], share_equal, "等权拆分：每期固定均分 20%", "YlGnBu"),
]:
    im = ax.imshow(share.T, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(range(N)); ax.set_yticklabels([n.split("·")[0] for n in names], fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("场所")
fig.colorbar(im, ax=axes, label="切片占比")
axes[1].set_xlabel("拆单期 (t)")
plt.tight_layout(); plt.savefig(os.path.join(D, "routing_heatmap.png"), dpi=130); plt.close()

# ============================================================================
# 图 4：实现短缺分布（蒙特卡洛 300 次）—— SOR 把成本整体左移
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5.4))
bins = np.linspace(min(res_equal.min(), res_smart.min()) - 2,
                   max(res_equal.max(), res_smart.max()) + 2, 40)
ax.hist(res_equal, bins=bins, alpha=0.45, color=C["equal2"], label=f"等权拆分 (均值 {res_equal.mean():.1f})")
ax.hist(res_static, bins=bins, alpha=0.45, color=C["static"], label=f"静态成本分配 (均值 {res_static.mean():.1f})")
ax.hist(res_smart, bins=bins, alpha=0.55, color=C["smart"], label=f"智能 SOR (均值 {res_smart.mean():.1f})")
ax.set_xlabel("实现短缺 (bps, 越低越好)")
ax.set_ylabel("模拟次数")
ax.set_title("实现短缺分布（300 次蒙特卡洛）：SOR 把成本整体左移")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "shortfall_distribution.png"), dpi=130); plt.close()

print("done ->", D)
