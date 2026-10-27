#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 2026-10-25（第二弹）两篇量化文章配图 + 核心数值（固定 seed 可复现）。
  A. multi-agent-market-simulation-lob  多智能体市场仿真：用 Agent 重演限价簿价格发现
  B. kyle-continuous-auction-flow       订单流回归 Kyle 连续拍卖：把知情者最优下单速度解出来
所有图表均为真实计算图，数值固定随机种子可复现。
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

BASE = "/Users/halo123456/workspace/astro-blog/public/images"


# ============================================================
# 文章 A：多智能体市场仿真（事件驱动限价簿）
# ============================================================
def gen_multi_agent(mom_thr=6.0, mom_qty=5, mom_cd=10, skew_k=0.1, cap=150,
                     inf_cap=25, inf_coef=4.0, jump_p=0.008):
    slug = "multi-agent-market-simulation-lob"
    OUT = os.path.join(BASE, slug)
    os.makedirs(OUT, exist_ok=True)

    TICK = 0.01
    N_STEPS = 3000
    WARMUP = 500
    SEED = 20261025
    AGENTS = ["mm", "noise", "informed", "momentum"]

    def simulate(seed=SEED, informed=True, momentum=True, mm_depth=10,
                 mom_thr=6.0, mom_qty=5, mom_cd=10, skew_k=0.1, cap=150,
                 inf_cap=25, inf_coef=4.0, jump_p=0.008):
        rng = np.random.default_rng(seed)
        # 预先抽取全部随机数（共同随机数，保证各变体可比）
        U = {
            "sw": rng.random(N_STEPS),                     # 波动 regime 切换
            "jp": rng.random(N_STEPS),                     # 跳变
            "jz": rng.normal(0, 1, N_STEPS),
            "dz": rng.normal(0, 1, N_STEPS),
            "no": rng.random((N_STEPS, 8)),                # 噪声交易者是否下单
            "ns": rng.random((N_STEPS, 8)),                # 买卖方向
            "nq": rng.integers(1, 6, (N_STEPS, 8)),        # 市价单量
        }
        SIG = [0.015, 0.05]          # 低波 / 高波 regime 的基本面波动
        P_STAY = 0.99                # regime 持续概率
        INV_CAP = 100                # 做市商库存上限（到限抽掉被攻击的一侧）
        vol_state = 0
        v = 100.0
        book = {}                    # price -> [[agent, qty, side], ...] FIFO
        last_trade, mid = 100.0, 100.0
        mid_valid = False           # 双边报价存在时才有中间价，否则沿用上一个
        mids = np.empty(N_STEPS); vs = np.empty(N_STEPS)
        spreads = np.full(N_STEPS, np.nan)
        volumes = np.zeros(N_STEPS)
        volr = np.empty(N_STEPS)
        cash = {k: 0.0 for k in AGENTS}
        pos = {k: 0.0 for k in AGENTS}
        vol_by = {k: 0.0 for k in AGENTS}
        n_trade = {k: 0 for k in AGENTS}
        mm_inv = np.empty(N_STEPS)
        mid_hist = []
        vol_this = [0.0]
        last_mom_t = -10 ** 9

        def side_levels(side):
            ps = [p for p, q in book.items() if q and q[0][2] == side]
            return sorted(ps)

        def do_market(agent, side, qty):
            nonlocal mid, last_trade
            filled, notional = 0.0, 0.0
            rest_side = "ask" if side == "buy" else "bid"
            prices = side_levels(rest_side)
            if side == "sell":
                prices = prices[::-1]
            for p in prices:
                if filled >= qty:
                    break
                queue = book[p]
                while queue and filled < qty:
                    o = queue[0]
                    take = min(qty - filled, o[1])
                    o[1] -= take
                    owner = o[0]
                    if side == "buy":
                        cash[agent] -= p * take; pos[agent] += take
                        cash[owner] += p * take; pos[owner] -= take
                    else:
                        cash[agent] += p * take; pos[agent] -= take
                        cash[owner] -= p * take; pos[owner] += take
                    vol_by[agent] += take; vol_by[owner] += take
                    filled += take; notional += p * take
                    if o[1] <= 1e-9:
                        queue.pop(0)
                if not queue:
                    del book[p]
            if filled > 0:
                vol_this[0] += filled
                n_trade[agent] += 1
                last_trade = notional / filled
                bids, asks = side_levels("bid"), side_levels("ask")
                bb = bids[-1] if bids else None
                ba = asks[0] if asks else None
                mid = 0.5 * (bb + ba) if (bb is not None and ba is not None) else last_trade

        for t in range(N_STEPS):
            # 1. 基本面：regime 切换随机游走 + 罕见跳变
            if U["sw"][t] > P_STAY:
                vol_state = 1 - vol_state
            sig = SIG[vol_state]
            if U["jp"][t] < jump_p:
                v += 8 * sig * U["jz"][t]          # 跳变 ~8 倍 regime 波动
            else:
                v += sig * U["dz"][t]
            vr = sig / 0.02
            volr[t] = vr

            # 2. 做市商撤单重报：围绕 mid 挂双边，价差随波动与库存走，
            #    库存到限后抽掉会进一步加重库存的一侧（流动性真空）
            for p in list(book.keys()):
                book[p] = [o for o in book[p] if o[0] != "mm"]
                if not book[p]:
                    del book[p]
            inv = pos["mm"]
            half_ticks = 1.5 + 2.0 * vr + 0.1 * abs(inv)
            skew = np.clip(skew_k * inv * TICK,
                           -(half_ticks - 1) * TICK, (half_ticks - 1) * TICK)  # 偏移不超半价差
            if inv > -cap:                             # 没短过头：可以挂卖单
                for lvl in range(2):
                    ap = round((mid + (half_ticks + 2 * lvl) * TICK - skew) / TICK) * TICK
                    book.setdefault(ap, []).append(["mm", mm_depth, "ask"])
            if inv < cap:                              # 没长过头：可以挂买单
                for lvl in range(2):
                    bp = round((mid - (half_ticks + 2 * lvl) * TICK - skew) / TICK) * TICK
                    book.setdefault(bp, []).append(["mm", mm_depth, "bid"])

            # 3. 噪声交易者：泊松式市价单，高波动时更活跃
            p_act = min(1.0, 0.30 + 0.45 * vr)
            for i in range(8):
                if U["no"][t, i] < p_act:
                    side = "buy" if U["ns"][t, i] < 0.5 else "sell"
                    do_market("noise", side, int(U["nq"][t, i]))

            # 4. 知情交易者：看得见 v 与盘口，只在覆盖价差后仍有赚头时下单
            if informed:
                bids, asks = side_levels("bid"), side_levels("ask")
                ba = asks[0] if asks else None
                bb = bids[-1] if bids else None
                if ba is not None and bb is not None:
                    edge_buy = v - ba
                    edge_sell = bb - v
                    if edge_buy > 1 * TICK:
                        qty = int(min(inf_cap, max(1, inf_coef * edge_buy / TICK)))
                        do_market("informed", "buy", qty)
                    elif edge_sell > 1 * TICK:
                        qty = int(min(inf_cap, max(1, inf_coef * edge_sell / TICK)))
                        do_market("informed", "sell", qty)

            # 5. 动量交易者：追 20 步价格变动（价格单位阈值，带冷却）
            if momentum and len(mid_hist) >= 20 and t - last_mom_t >= mom_cd:
                move = mid - mid_hist[-20]
                if move > mom_thr * TICK:
                    do_market("momentum", "buy", mom_qty)
                    last_mom_t = t
                elif move < -mom_thr * TICK:
                    do_market("momentum", "sell", mom_qty)
                    last_mom_t = t

            mids[t] = mid; vs[t] = v; volumes[t] = vol_this[0]; vol_this[0] = 0.0
            mm_inv[t] = pos["mm"]
            bids, asks = side_levels("bid"), side_levels("ask")
            if bids and asks:
                spreads[t] = (asks[0] - bids[-1]) / TICK
            mid_hist.append(mid)

        pnl = {k: cash[k] + pos[k] * mids[-1] for k in AGENTS}  # 按最终中间价平仓
        pnl_share = {k: (pnl[k] / vol_by[k] if vol_by[k] > 0 else 0.0) for k in AGENTS}
        return dict(mids=mids, vs=vs, spreads=spreads, volumes=volumes,
                    volr=volr, pnl=pnl, pnl_share=pnl_share, vol_by=vol_by,
                    n_trade=n_trade, pos=pos, mm_inv=mm_inv)

    def stylized(r):
        ret = np.diff(r["mids"][WARMUP:])
        vol = r["volumes"][WARMUP + 1:]
        kurt = float(np.mean((ret - ret.mean()) ** 4) / ret.var() ** 2 - 3)
        ret20 = ret[:len(ret) // 20 * 20].reshape(-1, 20).sum(1)
        kurt20 = float(np.mean((ret20 - ret20.mean()) ** 4) / ret20.var() ** 2 - 3)
        a = np.abs(ret - ret.mean())
        ac1 = float(np.corrcoef(a[:-1], a[1:])[0, 1])
        ac5 = float(np.corrcoef(a[:-5], a[5:])[0, 1])
        n20 = len(ret) // 20
        rv20 = np.abs(ret[:n20 * 20]).reshape(-1, 20).mean(1)
        v20 = vol[:n20 * 20].reshape(-1, 20).mean(1)
        cvv = float(np.corrcoef(np.abs(ret), vol)[0, 1])
        cvv20 = float(np.corrcoef(rv20, v20)[0, 1])
        sp = r["spreads"][WARMUP:]
        lo = r["volr"][WARMUP:] < 1.5
        rmse = float(np.sqrt(np.mean((r["mids"][WARMUP:] - r["vs"][WARMUP:]) ** 2)))
        return dict(kurt=kurt, kurt20=kurt20, ac1=ac1, ac5=ac5, cvv=cvv, cvv20=cvv20,
                    spread=float(np.nanmean(sp)),
                    spread_lo=float(np.nanmean(sp[lo])),
                    spread_hi=float(np.nanmean(sp[~lo])),
                    vol_lo=float(np.mean(r["volumes"][WARMUP:][lo])),
                    vol_hi=float(np.mean(r["volumes"][WARMUP:][~lo])),
                    rmse=rmse)

    kw = dict(mom_thr=mom_thr, mom_qty=mom_qty, mom_cd=mom_cd, skew_k=skew_k, cap=cap,
              inf_cap=inf_cap, inf_coef=inf_coef, jump_p=jump_p)
    full = simulate(**kw)
    no_inf = simulate(informed=False, **kw)
    no_mom = simulate(momentum=False, **kw)
    shallow = simulate(mm_depth=5, **kw)

    s_full, s_noinf, s_nomom, s_shallow = map(stylized, [full, no_inf, no_mom, shallow])

    # ---------- 图1：价格发现（含跳变窗口） ----------
    dv = np.diff(full["vs"])
    j = int(np.argmax(np.abs(dv))) + 1
    w0, w1 = max(0, j - 200), min(N_STEPS, j + 200)
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    tt = np.arange(w0, w1)
    ax.plot(tt, full["vs"][w0:w1], color="#333", lw=1.5, label="基本面价值 v（仅知情者可见）")
    ax.plot(tt, full["mids"][w0:w1], color="#E8684A", lw=1.2, label="限价簿中间价 mid")
    ax.axvline(j, color="#5B8FF9", ls="--", lw=1.2)
    ax.annotate(f"跳变 Δv={dv[j-1]:+.2f}", xy=(j, full["vs"][j]),
                xytext=(j + 25, full["vs"][j] + 0.06), fontsize=10, color="#5B8FF9",
                arrowprops=dict(arrowstyle="->", color="#5B8FF9"))
    ax.set_xlabel("仿真步"); ax.set_ylabel("价格")
    ax.set_title("价格发现：知情交易者的订单流把 mid 拉回跳变后的基本面")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "price_discovery.png")); plt.close(fig)

    # ---------- 图2：收益厚尾（完整模型 vs 无知情者） ----------
    ret_f = np.diff(full["mids"][WARMUP:])
    ret_n = np.diff(no_inf["mids"][WARMUP:])
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    bins = np.linspace(min(ret_f.min(), ret_n.min()), max(ret_f.max(), ret_n.max()), 150)
    ax.hist(ret_n, bins=bins, density=True, color="#9B6DD1", alpha=0.55,
            label=f"无知情者（峰度 {s_noinf['kurt']:.1f}）")
    ax.hist(ret_f, bins=bins, density=True, color="#9DB9F3", alpha=0.75,
            label=f"完整模型（峰度 {s_full['kurt']:.1f}）")
    xs = np.linspace(bins[0], bins[-1], 400)
    m_, s_ = ret_f.mean(), ret_f.std()
    ax.plot(xs, np.exp(-(xs - m_) ** 2 / (2 * s_ ** 2)) / (s_ * math.sqrt(2 * math.pi)),
            color="#333", lw=1.6, label="正态分布（同方差）")
    ax.set_yscale("log")
    ax.set_xlabel("mid 收益率"); ax.set_ylabel("密度（对数轴）")
    ax.set_title("厚尾从哪来：跳变被知情者连续扫单 impound，单步大变动在时间上聚集")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fat_tails.png")); plt.close(fig)

    # ---------- 图3：波动率聚集 ----------
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 5.2), sharex=True)
    ttx = np.arange(WARMUP, WARMUP + 1500)
    axes[0].plot(ttx, ret_f[:1500], color="#888", lw=0.7)
    axes[0].set_ylabel("收益率"); axes[0].grid(alpha=0.3)
    axes[0].set_title("波动率聚集：平静段与爆发段交替（|r| 滞后1阶自相关 "
                      f"{s_full['ac1']:.2f}、滞后5阶 {s_full['ac5']:.2f}）")
    roll = pd_rolling_std(ret_f, 20)[:1500]
    axes[1].plot(ttx, roll, color="#E8684A", lw=1.1)
    axes[1].fill_between(ttx, 0, roll, color="#E8684A", alpha=0.25)
    axes[1].set_xlabel("仿真步"); axes[1].set_ylabel("20 步滚动波动"); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "vol_clustering.png")); plt.close(fig)

    # ---------- 图4：价差与成交量的波动联动 ----------
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.plot(ttx, full["spreads"][WARMUP:WARMUP + 1500], color="#5B8FF9", lw=0.9,
            label="买卖价差（tick）")
    ax.set_xlabel("仿真步"); ax.set_ylabel("价差（tick）", color="#5B8FF9")
    ax.tick_params(axis="y", labelcolor="#5B8FF9")
    ax2 = ax.twinx()
    ax2.plot(ttx, full["volumes"][WARMUP:WARMUP + 1500], color="#E8684A", lw=0.7, alpha=0.8,
             label="每步成交量（股）")
    ax2.set_ylabel("成交量（股）", color="#E8684A")
    ax2.tick_params(axis="y", labelcolor="#E8684A")
    ax.set_title(f"价差与成交量的波动联动：低波段价差 {s_full['spread_lo']:.1f} tick / 高波段 "
                 f"{s_full['spread_hi']:.1f} tick，corr(|r|,vol)={s_full['cvv']:.2f}")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "spread_volume.png")); plt.close(fig)

    # ---------- 图5：消融 + 盈亏分解 ----------
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4))
    names = ["完整模型", "无知情者", "无动量", "浅簿(深度5)"]
    vals = [s_full["rmse"], s_noinf["rmse"], s_nomom["rmse"], s_shallow["rmse"]]
    cols = ["#E8684A", "#9B6DD1", "#52A973", "#5B8FF9"]
    bars = axes[0].bar(names, vals, color=cols, alpha=0.85, width=0.55)
    for b, v_ in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v_ + 0.02, f"{v_:.3f}",
                     ha="center", va="bottom", fontsize=10)
    axes[0].set_ylabel("价格发现误差 RMSE(mid−v)")
    axes[0].set_title("消融：拿掉知情者，价格发现误差放大一个量级")
    axes[0].grid(alpha=0.3, axis="y")
    ps = full["pnl_share"]
    an = ["做市商", "噪声", "知情者", "动量"]
    pv = [ps["mm"], ps["noise"], ps["informed"], ps["momentum"]]
    bars2 = axes[1].bar(an, pv, color=cols, alpha=0.85, width=0.55)
    for b, v_ in zip(bars2, pv):
        axes[1].text(b.get_x() + b.get_width() / 2, v_ + (0.0002 if v_ >= 0 else -0.0006),
                     f"{v_*1e4:.1f}", ha="center",
                     va="bottom" if v_ >= 0 else "top", fontsize=10)
    axes[1].axhline(0, color="#333", lw=0.8)
    axes[1].set_ylabel("每股盈亏（×10⁻⁴ 元）")
    axes[1].set_title("盈亏分解：做市商赚价差、知情者赚信息租、噪声与动量输血")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "ablation_pnl.png")); plt.close(fig)

    print("=" * 70)
    print("ARTICLE_A_MULTI_AGENT_METRICS")
    for nm, s in [("full", s_full), ("no_informed", s_noinf),
                  ("no_momentum", s_nomom), ("shallow", s_shallow)]:
        print(f"[{nm:12s}] kurt={s['kurt']:6.2f} kurt20={s['kurt20']:6.2f} "
              f"ac1={s['ac1']:+.3f} ac5={s['ac5']:+.3f} "
              f"corr|r|,v={s['cvv']:+.3f} corr20={s['cvv20']:+.3f} spread={s['spread']:.2f}t "
              f"(lo {s['spread_lo']:.1f}/hi {s['spread_hi']:.1f}) "
              f"vol(lo {s['vol_lo']:.1f}/hi {s['vol_hi']:.1f}) rmse={s['rmse']:.4f}")
    print(f"PnL/share(full): mm={full['pnl_share']['mm']:+.6f} "
          f"noise={full['pnl_share']['noise']:+.6f} "
          f"informed={full['pnl_share']['informed']:+.6f} "
          f"momentum={full['pnl_share']['momentum']:+.6f}")
    print(f"PnL/share(no_informed): mm={no_inf['pnl_share']['mm']:+.6f} "
          f"noise={no_inf['pnl_share']['noise']:+.6f} "
          f"momentum={no_inf['pnl_share']['momentum']:+.6f}")
    print(f"PnL total(full): mm={full['pnl']['mm']:+.1f} noise={full['pnl']['noise']:+.1f} "
          f"informed={full['pnl']['informed']:+.1f} momentum={full['pnl']['momentum']:+.1f}")
    print(f"volume by agent: {full['vol_by']}  trades: {full['n_trade']}")
    print(f"MM inventory: final={full['pos']['mm']:.0f} "
          f"max|inv|={np.abs(full['mm_inv']).max():.0f} "
          f"mean|inv|={np.abs(full['mm_inv']).mean():.1f}")
    print(f"jump at t={j}, dv={dv[j-1]:+.3f}, v range [{full['vs'].min():.2f},{full['vs'].max():.2f}]")
    print("=" * 70)


def pd_rolling_std(x, w):
    if len(x) <= w:
        return np.full(len(x), np.nan)
    c = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x ** 2, 0, 0.0))
    s1 = c[w:] - c[:-w]
    s2 = c2[w:] - c2[:-w]
    mean = s1 / w
    var = np.maximum(s2 / w - mean ** 2, 0.0)
    out = np.sqrt(var)
    return np.concatenate([np.full(w - 1, np.nan), out])


# ============================================================
# 文章 B：Kyle 连续拍卖（单次拍卖均衡 + N 期动态规划）
# ============================================================
def gen_kyle():
    slug = "kyle-continuous-auction-flow"
    OUT = os.path.join(BASE, slug)
    os.makedirs(OUT, exist_ok=True)

    P0, SIG_V, SIG_U = 100.0, 1.0, 1.0
    M = SIG_U ** 2
    rng = np.random.default_rng(7)

    # ---------- 1. 单次拍卖：alpha 网格 + 蒙特卡洛 ----------
    alphas = np.linspace(0.05, 2.5, 50)
    n_mc = 200000
    v_mc = P0 + SIG_V * rng.standard_normal(n_mc)
    u_mc = SIG_U * rng.standard_normal(n_mc)
    d = v_mc - P0
    pi_ins_mc, pi_mm_mc, pi_noise_mc = [], [], []
    for a in alphas:
        x = a * d
        X = x + u_mc
        lam = a * SIG_V ** 2 / (a ** 2 * SIG_V ** 2 + M)   # 市场 maker 的线性定价
        p = P0 + lam * X
        pi_ins_mc.append(float(np.mean((v_mc - p) * x)))
        pi_mm_mc.append(float(np.mean((p - v_mc) * X)))
        pi_noise_mc.append(float(np.mean((v_mc - p) * u_mc)))
    pi_ins_mc = np.array(pi_ins_mc); pi_mm_mc = np.array(pi_mm_mc)
    pi_noise_mc = np.array(pi_noise_mc)
    a_star = SIG_U / SIG_V
    i_star = int(np.argmin(np.abs(alphas - a_star)))
    lam_star = SIG_V / (2 * SIG_U)
    pi_ins_theo = a_star * SIG_V ** 2 * (1 - lam_star * a_star)

    # 回归验证：v-p0 ~ X 的斜率应等于 lambda*
    n_sc = 6000
    v_s = P0 + SIG_V * rng.standard_normal(n_sc)
    u_s = SIG_U * rng.standard_normal(n_sc)
    X_s = a_star * (v_s - P0) + u_s
    slope = float(np.polyfit(X_s, v_s - P0, 1)[0])

    # ---------- 2. N 期 DP：向后归纳解出每期最优下单速度 ----------
    N = 50
    gammas = np.linspace(1e-3, 30.0, 60000)
    g2 = gammas ** 2
    sm = math.sqrt(M)
    c_cur = 0.0                       # c[剩余1期] 由 FOC 给出（=sqrt(M)/2）
    gam_by_r = [None]                 # gam_by_r[r] = 还剩 r 期时的 gamma
    for r in range(1, N + 1):
        obj = gammas * sm - gammas ** 3 * sm / (g2 + 1) + c_cur / np.sqrt(g2 + 1)
        i = int(np.argmax(obj))
        gam_by_r.append(gammas[i])
        c_cur = float(obj[i])
    dp_value = c_cur * math.sqrt(SIG_V ** 2)   # 均衡策略下的期望总利润

    Sig = np.empty(N + 1); Sig[0] = SIG_V ** 2
    beta = np.empty(N); lam = np.empty(N)
    for t in range(1, N + 1):
        r = N - t + 1
        g = gam_by_r[r]
        beta[t - 1] = g * math.sqrt(M / Sig[t - 1])
        lam[t - 1] = beta[t - 1] * Sig[t - 1] / (beta[t - 1] ** 2 * Sig[t - 1] + M)
        Sig[t] = Sig[t - 1] * M / (beta[t - 1] ** 2 * Sig[t - 1] + M)

    # ---------- 3. N 期仿真 ----------
    npath = 20000
    vp = P0 + SIG_V * rng.standard_normal(npath)
    p = np.full(npath, P0)
    pmat = np.empty((npath, N + 1)); pmat[:, 0] = P0
    xmat = np.empty((npath, N))
    Xmat = np.empty((npath, N))
    umat = np.empty((npath, N))
    for t in range(N):
        x = beta[t] * (vp - p)
        u = SIG_U * rng.standard_normal(npath)
        X = x + u
        p = p + lam[t] * X
        pmat[:, t + 1] = p; xmat[:, t] = x; Xmat[:, t] = X; umat[:, t] = u
    pi_ins_per = (vp[:, None] - pmat[:, 1:]) * xmat          # 每期利润
    pi_mm_per = (pmat[:, 1:] - vp[:, None]) * Xmat
    pi_noise_per = (vp[:, None] - pmat[:, 1:]) * umat
    ins_total = pi_ins_per.sum(1)
    mm_total = pi_mm_per.sum(1)
    noise_total = pi_noise_per.sum(1)
    slope_N = float(np.polyfit(pmat[:, N], vp, 1)[0])
    sig_real = np.var(vp[:, None] - pmat[:, 1:], axis=0)

    # ---------- 4. 策略对比（固定 maker 均衡定价规则） ----------
    def strat_profit(kind, rng_):
        p = np.full(npath, P0)
        tot = np.zeros(npath)
        for t in range(N):
            if kind == "eq":
                x = beta[t] * (vp - p)
            elif kind == "pace_rev":
                x = beta[N - 1 - t] * (vp - p)     # 均衡节奏倒序
            elif kind == "twap":
                x = (vp - P0) / N
            elif kind == "front":
                x = 2 * (vp - P0) / N if t < N // 2 else 0.0
            elif kind == "back":
                x = 0.0 if t < N // 2 else 2 * (vp - P0) / N
            elif kind == "once":
                x = (vp - P0) if t == 0 else 0.0
            u = SIG_U * rng_.standard_normal(npath)
            p = p + lam[t] * (x + u)
            tot += (vp - p) * x
        return float(tot.mean())

    rng2 = np.random.default_rng(23)
    sp = {k: strat_profit(k, rng2) for k in
          ["eq", "pace_rev", "twap", "front", "back", "once"]}

    # ---------- 图1：单次拍卖 alpha 网格 ----------
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.plot(alphas, pi_ins_mc, color="#E8684A", lw=1.8, label="知情者期望利润（MC）")
    ath = alphas * SIG_V ** 2 * (1 - (alphas * SIG_V ** 2 / (alphas ** 2 * SIG_V ** 2 + M)) * alphas)
    ax.plot(alphas, ath, color="#E8684A", lw=1.0, ls="--", label="知情者理论值")
    ax.plot(alphas, pi_mm_mc, color="#5B8FF9", lw=1.6, label="做市商期望利润（MC）")
    ax.plot(alphas, pi_noise_mc, color="#52A973", lw=1.6, label="噪声交易者期望利润（MC）")
    ax.axvline(a_star, color="#333", ls=":", lw=1.2)
    ax.annotate(f"均衡 α* = σ_u/σ_v = {a_star:.2f}", xy=(a_star, pi_ins_theo),
                xytext=(a_star + 0.35, pi_ins_theo + 0.12), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#333"))
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel("知情者下单强度 α（x = α·(v−p0)）"); ax.set_ylabel("期望利润")
    ax.set_title("单次拍卖：知情者的最优点恰好是做市商的盈亏平衡点——这就是 Kyle 均衡")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "single_auction.png")); plt.close(fig)

    # ---------- 图2：MC 散点回归 ----------
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.scatter(X_s[:3000], v_s[:3000] - P0, s=7, alpha=0.35, color="#5B8FF9", label="蒙特卡洛样本")
    xs = np.linspace(X_s.min(), X_s.max(), 50)
    ax.plot(xs, slope * xs, color="#E8684A", lw=1.8,
            label=f"OLS 拟合斜率 = {slope:.3f}")
    ax.plot(xs, lam_star * xs, color="#333", lw=1.2, ls="--",
            label=f"理论 λ* = σ_v/(2σ_u) = {lam_star:.3f}")
    ax.set_xlabel("市场 maker 观察到的订单流 X = x + u")
    ax.set_ylabel("真实价值偏离 v − p0")
    ax.set_title("定价规则识别：订单流每多 1 单位，价格上调 λ*——回归斜率还原理论值")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "mc_scatter.png")); plt.close(fig)

    # ---------- 图3：N 期价格路径 + Sigma 衰减 ----------
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    vfix = P0 + 1.0        # 固定真实价值，路径差异全部来自噪声
    pfix = np.full(npath, P0)
    paths = np.empty((npath, N + 1)); paths[:, 0] = P0
    rng3 = np.random.default_rng(99)
    for t in range(N):
        u = SIG_U * rng3.standard_normal(npath)
        pfix = pfix + lam[t] * (beta[t] * (vfix - pfix) + u)
        paths[:, t + 1] = pfix
    for k in range(30):
        axes[0].plot(paths[k], color="#9DB9F3", lw=0.7, alpha=0.8)
    axes[0].axhline(vfix, color="#E8684A", lw=1.6, label=f"真实价值 v = {vfix:.0f}")
    axes[0].plot(paths.mean(0), color="#333", lw=1.4, ls="--", label="路径均值")
    axes[0].set_xlabel("交易期 t"); axes[0].set_ylabel("价格")
    axes[0].set_title(f"{N} 期连续拍卖：价格从 p0 收敛到 v")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].plot(np.arange(N + 1), Sig / Sig[0], color="#5B8FF9", lw=1.8, label="理论 Σ_t/Σ0")
    axes[1].plot(np.arange(1, N + 1), sig_real / Sig[0], color="#E8684A", lw=1.2,
                 ls="--", label="仿真实现 var(v−p_t)/Σ0")
    axes[1].axhline(0.5, color="#999", ls=":", lw=1)
    axes[1].annotate("半程信息线", xy=(N / 2, 0.5), xytext=(N / 2 + 4, 0.56), fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="#999"))
    axes[1].set_xlabel("交易期 t"); axes[1].set_ylabel("剩余不确定性")
    axes[1].set_title("信息逐期 impounded：不确定性的衰减路径")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "price_paths.png")); plt.close(fig)

    # ---------- 图4：beta_t 与 lambda_t 的均衡动态 ----------
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.plot(np.arange(1, N + 1), beta, color="#E8684A", lw=1.8, marker="o", ms=3.5,
            label="下单速度 β_t（左轴）")
    ax.set_xlabel("交易期 t"); ax.set_ylabel("β_t", color="#E8684A")
    ax.tick_params(axis="y", labelcolor="#E8684A")
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, N + 1), lam, color="#5B8FF9", lw=1.8, marker="s", ms=3.5,
             label="价格冲击 λ_t（右轴）")
    ax2.set_ylabel("λ_t", color="#5B8FF9")
    ax2.tick_params(axis="y", labelcolor="#5B8FF9")
    ax.set_title("均衡动态：越到收盘知情者下单越快，价格冲击随剩余信息衰减")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "beta_lambda.png")); plt.close(fig)

    # ---------- 图5：策略对比 ----------
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    order = ["eq", "pace_rev", "twap", "front", "back", "once"]
    labels = ["均衡策略\n(DP解)", "均衡节奏\n倒序", "匀速倾销\nTWAP", "前重后轻", "前轻后重", "一次性\n全下"]
    vals = [sp[k] for k in order]
    cols = ["#E8684A", "#F5B93F", "#5B8FF9", "#9B6DD1", "#52A973", "#888888"]
    bars = ax.bar(labels, vals, color=cols, alpha=0.88, width=0.55)
    for b, v_ in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v_ + 0.1, f"{v_:.2f}",
                ha="center", va="bottom", fontsize=10)
    ax.axhline(dp_value, color="#333", ls="--", lw=1.2,
               label=f"DP 理论值 {dp_value:.2f}")
    ax.set_ylabel("知情者期望总利润")
    ax.set_title("下单路径的时机价值：同一条信息，均衡节奏把利润放大约 7 倍")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "strategy_profit.png")); plt.close(fig)

    print("=" * 70)
    print("ARTICLE_B_KYLE_METRICS")
    print(f"single auction: a*={a_star:.3f} lam*={lam_star:.3f} "
          f"insider MC={pi_ins_mc[i_star]:.4f} theo={pi_ins_theo:.4f} "
          f"MM MC={pi_mm_mc[i_star]:+.5f} noise MC={pi_noise_mc[i_star]:+.4f} "
          f"reg slope={slope:.4f}")
    print(f"DP: N={N} dp_value={dp_value:.4f} beta_1={beta[0]:.3f} beta_N={beta[-1]:.3f} "
          f"lam_1={lam[0]:.4f} lam_N={lam[-1]:.4f}")
    print(f"Sigma: S0={Sig[0]:.3f} S_mid={Sig[N//2]:.4f} ({Sig[N//2]/Sig[0]:.1%}) "
          f"S_N={Sig[-1]:.5f} ({Sig[-1]/Sig[0]:.2%})")
    print(f"MC N-period: insider={ins_total.mean():.4f} MM={mm_total.mean():+.5f} "
          f"noise={noise_total.mean():+.4f} slope(v~pN)={slope_N:.4f} "
          f"mean|pN-v|={np.mean(np.abs(pmat[:,N]-vp)):.4f}")
    ppm = pi_ins_per.mean(0)
    print(f"per-period insider profit: mean={ppm.mean():.4f} std={ppm.std():.4f} "
          f"first={ppm[0]:.4f} mid={ppm[N//2]:.4f} last={ppm[-1]:.4f}")
    print("strategies: " + " ".join(f"{k}={sp[k]:.3f}" for k in
                                    ["eq", "pace_rev", "twap", "front", "back", "once"]))
    print(f"MC vs DP: {ins_total.mean():.4f} vs {dp_value:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    import os as _os
    _kw = dict(
        mom_thr=float(_os.environ.get("MOM_THR", 6)),
        mom_qty=int(_os.environ.get("MOM_QTY", 8)),
        mom_cd=int(_os.environ.get("MOM_CD", 10)),
        skew_k=float(_os.environ.get("SKEW_K", 0.1)),
        cap=float(_os.environ.get("CAP", 150)),
        inf_cap=int(_os.environ.get("INF_CAP", 25)),
        inf_coef=float(_os.environ.get("INF_COEF", 4.0)),
        jump_p=float(_os.environ.get("JUMP_P", 0.008)),
    )
    gen_multi_agent(**_kw)
    gen_kyle()
    print("ALL DONE")
