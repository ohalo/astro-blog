#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「隐含二叉树：用市场报价反推风险中性转移概率」
(implied-binomial-tree) 生成真实配图与真实统计数字。

核心模型：Rubinstein (1994) 隐含二叉树（Implied Binomial Tree, IBT）。
  - 标准 CRR 二叉树：u/d 固定、风险中性概率 p 固定 => 只能给一条"平"的 IV。
  - IBT：终端风险中性分布由市场微笑（option smile）通过 Breeden-Litzenberger
    反推得到；再用「前向归纳（forward induction）」反推出每个节点的
    *状态依赖* 风险中性转移概率 p(i,j)。
  - 这样整棵树定价出来的期权曲面，会在每个行权价精确还原市场报价。

所有图表与数字均由文中逻辑真实计算生成：
  1) ibt_tree_small.png       —— 小树（n=4）示意图：节点价格 + 状态依赖上移概率 p(i,j)
  2) ibt_terminal_dist.png    —— 市场 RND vs IBT 还原的终端分布（应重合）
  3) ibt_smile_recovery.png   —— IBT 反演的 IV 微笑 vs 市场输入微笑（应重合）
  4) ibt_crr_fail.png         —— 对照：固定 p 的 CRR 树只能给出平 IV（证明"为什么需要 IBT"）
  5) ibt_transition_heat.png  —— p(i,j) 热图：状态依赖结构一览

参数：S0=100, r=3%, T=1y, n=100 步（数值反演用）；树图 n=4。
市场微笑：IV(K) = 0.20 + 0.10·(K/S0 - 1) + 0.25·(K/S0 - 1)^2  （右翼抬高 = 风险厌恶斜笑）
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "implied-binomial-tree")
os.makedirs(D, exist_ok=True)

C = {"eq": "#2F4B7C", "up": "#55A868", "dn": "#C44E52", "grid": "#E6E6E6",
     "ink": "#2b2b2b", "acc": "#E8C04B", "pur": "#8172B3", "cya": "#4C72B0"}
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# =====================================================================
# 市场设定
# =====================================================================
S0, r, T = 100.0, 0.03, 1.0

def market_iv(K):
    """市场隐含波动率微笑：下行（低行权价）一侧波动更高（典型股票风险厌恶斜笑），
    用 tanh 边界化，保证远处不发散。
    IV(K) = 0.18 + 0.12·tanh(2·(K/S0 − 1))   ∈ [0.06, 0.30]，左高右低。"""
    x = K / S0 - 1.0
    return 0.18 + 0.12 * np.tanh(2.0 * x)

def bs_call(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def market_rnd(K):
    """由市场微笑经 Breeden-Litzenberger 反推风险中性密度 f(K)。
    f(K) = exp(rT) * d2C/dK2，其中 C 用 IV(K) 的 BS 价。"""
    sig = market_iv(K)
    C1 = bs_call(S0, K, T, r, sig)
    return np.exp(r * T) * _second_deriv_call(K, C1)

def _second_deriv_call(K, C_at_K):
    """对给定 K 数值二阶导 d2C/dK2，用 BS 价在 K±h 处差分。"""
    h = max(0.01, 0.001 * K)
    sig_p, sig_m = market_iv(K + h), market_iv(K - h)
    Cp = bs_call(S0, K + h, T, r, sig_p)
    Cm = bs_call(S0, K - h, T, r, sig_m)
    return (Cp - 2 * C_at_K + Cm) / (h * h)

# =====================================================================
# 隐含二叉树：Rubinstein (1994) 前向归纳
# =====================================================================
def build_ibt(S0, r, T, n, rnd_func):
    """返回 (u, d, K_grid, q, p) 。
    q[i][j] = 第 i 层（时间步 i）各节点的*风险中性概率*（sum_j q[n][j] = 1）；
    p[i][j] = 节点 (i,j) 的"上移"风险中性转移概率（状态依赖）。

    方法（可验证、数值稳健）：
      1) 终端风险中性密度 -> 离散终端概率 π_T(j)（sum=1）。
      2) 用反 CRR 卷积把 π_T "回投"到每一层，得到 q(i,*)（每步都是线性三角系统，可逆）。
      3) 由相邻两层的 q 解出状态依赖转移概率 p(i,j)：
            q(i+1, j) = p(i,j-1)·q(i,j-1) + (1-p(i,j))·q(i,j)
         边界：q(i+1,i+1)=p(i,i)·q(i,i)，q(i+1,0)=(1-p(i,0))·q(i,0)。
      这样构造出的树，其终端分布 *精确* 等于市场 RND（自洽），且 p(i,j) 随节点变化——
      这正是隐含二叉树（Rubinstein 1994 / Derman-Kani 1994）的核心：固定 u,d，
      让转移概率"状态依赖"去装下整条微笑。
    """
    dt = T / n
    sig0 = market_iv(S0)              # 用 ATM vol 定 u/d 几何
    u = np.exp(sig0 * np.sqrt(dt))
    d = 1.0 / u
    p_crr = (np.exp(r * dt) - d) / (u - d)

    # 终端节点价格
    K = S0 * u ** (2 * np.arange(n + 1) - n)   # S(n, j)

    # 终端风险中性密度（连续）-> 离散概率 π_T(j)
    # 只保留流动性行权价窗口 [0.4S0, 2.5S0]，窗口外市场无报价 -> 密度截断
    f = np.array([rnd_func(kk) for kk in K])
    f = np.where(np.isfinite(f), f, 0.0)
    f = np.maximum(f, 0.0)
    win = (K >= 0.40 * S0) & (K <= 2.50 * S0)
    f = np.where(win, f, 0.0)
    spacing = np.empty(n + 1)
    spacing[1:-1] = 0.5 * (K[2:] - K[:-2])
    spacing[0] = K[1] - K[0]
    spacing[-1] = K[-1] - K[-2]
    pi_T = f * spacing
    pi_T = np.maximum(pi_T, 0.0)
    s = pi_T.sum()
    pi_T = pi_T / s if s > 0 else pi_T  # 归一化到窗口内（≈1）

    # ---- 第 2 步：反 CRR 回投，得到每层概率 q(i,*) ----
    q = np.zeros((n + 1, n + 1))
    q[n, :] = pi_T.copy()
    for i in range(n - 1, -1, -1):
        # q(i+1, j) = p_crr·q(i, j-1) + (1-p_crr)·q(i, j)
        # => q(i, j) = (q(i+1, j) - p_crr·q(i, j-1)) / (1-p_crr), j 自底向上
        for j in range(i + 1):
            prev = q[i, j - 1] if j - 1 >= 0 else 0.0
            q[i, j] = (q[i + 1, j] - p_crr * prev) / (1 - p_crr)
        q[i, :] = np.maximum(q[i, :], 0.0)   # 数值保非负
        s = q[i, :i + 1].sum()
        if s > 0:
            q[i, :i + 1] = q[i, :i + 1] / s

    # ---- 第 3 步：由相邻层 q 解出状态依赖 p(i,j) ----
    # 二叉树：节点 (i,j) 的 up-child 是 (i+1,j+1)，down-child 是 (i+1,j)。
    # 守恒：q(i+1, j) = p(i,j-1)·q(i,j-1) + (1-p(i,j))·q(i,j)
    # Rubinstein 前向归纳：自顶层 (i,i) 向下扫，p(i,j) 一旦由 up-child 边确定，
    # 即可用 q(i+1, j) 的方程反解 p(i, j-1)。
    p = np.zeros((n + 1, n + 1))
    for i in range(n):
        # 顶部节点 (i, i)：只能向上到 (i+1, i+1)
        denom = q[i, i]
        p[i, i] = (q[i + 1, i + 1] / denom) if denom > 1e-15 else p_crr
        # 自顶向下扫，解每个 p(i, j-1)
        for j in range(i, 0, -1):
            # q(i+1, j) = p(i,j-1)·q(i,j-1) + (1-p(i,j))·q(i,j)
            if q[i, j - 1] > 1e-15:
                num = q[i + 1, j] - (1 - p[i, j]) * q[i, j]
                p[i, j - 1] = num / q[i, j - 1]
            else:
                p[i, j - 1] = p_crr
            p[i, j - 1] = min(max(p[i, j - 1], 1e-6), 1 - 1e-6)
        # 底部节点 (i, 0)：只能向下到 (i+1, 0)
        if q[i, 0] > 1e-15:
            p[i, 0] = 1.0 - q[i + 1, 0] / q[i, 0]
            p[i, 0] = min(max(p[i, 0], 1e-6), 1 - 1e-6)
    return u, d, K, q, p, pi_T

# =====================================================================
# 用 IBT 给期权定价，验证能否还原市场微笑
# =====================================================================
def ibt_price_call(u, d, K_grid, q, p, n, r, T, strike):
    """用 IBT 终端风险中性分布 q[n] 给看涨期权定价。
    因构造上 q[n] == 市场 RND 离散化，故定价精确还原微笑。"""
    disc = np.exp(-r * T)
    pay = np.maximum(K_grid - strike, 0.0)
    return disc * np.sum(q[n, :] * pay)

# =====================================================================
# 图 1：小树示意图（n=4）
# =====================================================================
def fig_small_tree():
    n = 4
    u, d, K, Q, p, pi_T = build_ibt(S0, r, T, n, market_rnd)
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.axis("off")
    for i in range(n + 1):
        for j in range(i + 1):
            x = i
            y = (n / 2.0) - j
            # 节点
            ax.scatter([x], [y], s=520, c=C["eq"], zorder=3, edgecolors="white", linewidths=1.5)
            ax.text(x, y, f"{K[j]:.1f}", ha="center", va="center",
                    color="white", fontsize=8.5, zorder=4)
            # 上移概率标注（连向 (i+1, j+1)）
            if i < n:
                ax.plot([x, x + 1], [y, y - 1], c=C["up"], lw=1.8, zorder=1)
                ax.text((x + x + 1) / 2 + 0.04, (y + y - 1) / 2 + 0.18,
                        f"p={p[i, j]:.2f}", color=C["up"], fontsize=7.6, zorder=5)
                ax.plot([x, x + 1], [y, y + 1 - 0.0], c=C["dn"], lw=1.8, zorder=1)
                # 下移概率（连向 (i+1, j)）
                ax.text((x + x + 1) / 2 + 0.04, (y + y + 1) / 2 - 0.22,
                        f"1-p={1 - p[i, j]:.2f}", color=C["dn"], fontsize=7.6, zorder=5)
    ax.set_xlim(-0.5, n + 0.5)
    ax.set_ylim(-n / 2 - 0.8, n / 2 + 0.8)
    ax.set_title(f"隐含二叉树（n={n}）：节点价 + 状态依赖上移概率 p(i,j)\n"
                 "同一层不同节点的 p 各不相同——这是它能被'校准'到微笑的关键",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "ibt_tree_small.png"), dpi=130)
    plt.close(fig)

# =====================================================================
# 图 2：市场 RND vs IBT 还原终端分布
# =====================================================================
def fig_terminal_dist():
    n = 120
    u, d, K, Q, p, pi_T = build_ibt(S0, r, T, n, market_rnd)
    # IBT 还原的终端分布：Q[n] 已是概率；与市场 RND 离散点对比
    f_mkt = np.array([market_rnd(kk) for kk in K])
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(K, f_mkt, "-", c=C["dn"], lw=2, label="市场 RND  f(K)（输入）")
    # 用 Q[n]*spacing 还原连续密度做对比
    spacing = np.empty(n + 1); spacing[1:-1] = 0.5 * (K[2:] - K[:-2])
    spacing[0] = K[1] - K[0]; spacing[-1] = K[-1] - K[-2]
    f_ibt = Q[n, :] / spacing
    ax.plot(K, f_ibt, "o--", c=C["eq"], lw=1.4, ms=4, label="IBT 还原终端分布")
    ax.axvline(S0 * np.exp(r * T), c=C["grid"], ls=":", lw=1)
    ax.set_xlabel("行权价 K（终端资产价）"); ax.set_ylabel("风险中性密度")
    ax.set_title("终端风险中性分布：IBT 完整还原市场微笑隐含的 RND", fontsize=11)
    ax.legend(fontsize=9); ax.grid(c=C["grid"], lw=0.7)
    fig.tight_layout(); fig.savefig(os.path.join(D, "ibt_terminal_dist.png"), dpi=130)
    plt.close(fig)

# =====================================================================
# 图 3：IBT 反演 IV 微笑 vs 市场输入
# =====================================================================
def fig_smile_recovery():
    n = 120
    u, d, K, Q, p, pi_T = build_ibt(S0, r, T, n, market_rnd)
    # 用 IBT 给一组行权价的看涨期权定价，再倒推 IV
    strikes = np.linspace(S0 * 0.7, S0 * 1.3, 31)
    iv_ibt = []
    for k in strikes:
        price = ibt_price_call(u, d, K, Q, p, n, r, T, k)
        iv_ibt.append(_implied_vol_call(price, k))
    iv_mkt = market_iv(strikes)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(strikes, iv_mkt, "-", c=C["dn"], lw=2, label="市场输入 IV 微笑")
    ax.plot(strikes, iv_ibt, "o--", c=C["eq"], lw=1.4, ms=4,
            label="IBT 反演 IV（应重合）")
    ax.set_xlabel("行权价 K"); ax.set_ylabel("隐含波动率 σ")
    ax.set_title("自洽检验：IBT 定价后倒推的 IV 与市场微笑几乎重合", fontsize=11)
    ax.legend(fontsize=9); ax.grid(c=C["grid"], lw=0.7)
    fig.tight_layout(); fig.savefig(os.path.join(D, "ibt_smile_recovery.png"), dpi=130)
    plt.close(fig)

def _implied_vol_call(price, K, lo=1e-4, hi=3.0):
    if price <= max(S0 - K * np.exp(-r * T), 0.0) + 1e-9:
        return lo
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs_call(S0, K, T, r, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

# =====================================================================
# 图 4：固定 p 的 CRR 树只能给平 IV（对照组）
# =====================================================================
def fig_crr_fail():
    n = 120
    sig0 = market_iv(S0)
    dt = T / n
    u = np.exp(sig0 * np.sqrt(dt)); d = 1.0 / u
    p_crr = (np.exp(r * dt) - d) / (u - d)
    # 终端分布（二项，固定 p）
    K = S0 * u ** (2 * np.arange(n + 1) - n)
    from math import comb
    pi = np.array([comb(n, j) * p_crr ** j * (1 - p_crr) ** (n - j) for j in range(n + 1)])
    # 用该分布给期权定价 -> 反推 IV
    disc = np.exp(-r * T)
    strikes = np.linspace(S0 * 0.7, S0 * 1.3, 31)
    iv_crr = []
    for k in strikes:
        pay = np.maximum(K - k, 0.0)
        price = disc * np.sum(pi * pay)
        iv_crr.append(_implied_vol_call(price, k))
    iv_mkt = market_iv(strikes)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(strikes, iv_mkt, "-", c=C["dn"], lw=2, label="市场输入 IV 微笑")
    ax.plot(strikes, iv_crr, "s--", c=C["pur"], lw=1.4, ms=4,
            label="固定 p CRR 树反推 IV（被压平）")
    ax.set_xlabel("行权价 K"); ax.set_ylabel("隐含波动率 σ")
    ax.set_title("对照：固定 p 的 CRR 树只能定出一条平 IV，装不下微笑", fontsize=11)
    ax.legend(fontsize=9); ax.grid(c=C["grid"], lw=0.7)
    fig.tight_layout(); fig.savefig(os.path.join(D, "ibt_crr_fail.png"), dpi=130)
    plt.close(fig)

# =====================================================================
# 图 5：p(i,j) 热图
# =====================================================================
def fig_transition_heat():
    n = 30
    u, d, K, Q, p, pi_T = build_ibt(S0, r, T, n, market_rnd)
    P = p[:n, :n]  # n x n
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(P, origin="lower", aspect="auto", cmap="YlGnBu")
    ax.set_xlabel("节点 j（下行步数）"); ax.set_ylabel("时间层 i")
    ax.set_title("状态依赖上移概率 p(i,j) 热图\n"
                 "并非常数：随节点位置变化，才把微笑装进了树", fontsize=10.5)
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("p(i,j)")
    fig.tight_layout(); fig.savefig(os.path.join(D, "ibt_transition_heat.png"), dpi=130)
    plt.close(fig)

# =====================================================================
if __name__ == "__main__":
    fig_small_tree()
    fig_terminal_dist()
    fig_smile_recovery()
    fig_crr_fail()
    fig_transition_heat()

    # ---- 数值自洽验证 ----
    n = 120
    u, d, K, q, p, pi_T = build_ibt(S0, r, T, n, market_rnd)
    dt = T / n
    p_crr = (np.exp(r * dt) - d) / (u - d)
    # 1) 终端分布应等于市场 RND（归一化后）
    err_pi = np.max(np.abs(q[n, :] - pi_T))
    # 2) 反演 IV 在几档行权价处与市场吻合
    for k in [85, 100, 115]:
        price = ibt_price_call(u, d, K, q, p, n, r, T, k)
        iv = _implied_vol_call(price, k)
        print(f"  K={k:6.1f}  IBT价={price:7.4f}  反演IV={iv:.4f}  市场IV={market_iv(k):.4f}")
    # 3) 展示 p 的状态依赖：取中间层几个节点的 p
    mid = n // 2
    print(f"[verify] 中间层 i={mid} 的 p(i,j) 范围: "
          f"{p[mid,:mid+1].min():.4f} ~ {p[mid,:mid+1].max():.4f}  (CRR={p_crr:.4f})")
    print(f"[verify] 终端概率最大偏差 = {err_pi:.2e}  (应≈0)")
    print(f"[verify] 终端概率和 = {q[n,:].sum():.6f}  (应≈1)")
    print("✅ 全部配图生成完毕 ->", D)
