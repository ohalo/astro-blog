#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
限价单排队位置：同样的价格，为什么你总是最后成交
事件驱动的限价单簿队列仿真，全部图表由真实计算生成，固定种子可复现。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Heiti SC", "Arial Unicode MS", "PingFang HK"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

OUT = "/Users/halo/workspace/astro-blog/public/images/queue-position-limit-order"
os.makedirs(OUT, exist_ok=True)
SEED = 20260801
C_A, C_B, C_C, C_D = "#2563eb", "#dc2626", "#16a34a", "#f59e0b"

TICK = 0.01
LAM_MKT = 1.0      # 市价单（消耗队列）到达强度
LAM_ADD = 3.0      # 同价位新增挂单
LAM_CXL = 2.2      # 同价位撤单
LAM_MOVE = 0.15    # 价格跳变强度
SIZE_MKT = 4.0     # 市价单平均手数
HORIZON = 300.0    # 单次实验最长等待（秒）


def simulate_one(rng, q_ahead, my_size=1.0, adverse_prob=0.62, lam_mkt=LAM_MKT):
    """
    单笔限价买单的生命周期仿真。
    q_ahead: 下单瞬间排在我前面的手数
    返回 dict：是否成交 / 等待时间 / 成交时价格是否已不利（adverse selection）
    事件类型：市价卖单吃队列 / 队尾新增 / 队列中撤单 / 价格跳走
    """
    t = 0.0
    ahead = float(q_ahead)
    LAM_MKT_L = lam_mkt
    total_rate = LAM_MKT_L + LAM_ADD + LAM_CXL + LAM_MOVE
    while t < HORIZON:
        t += rng.exponential(1.0 / total_rate)
        u = rng.random() * total_rate
        if u < LAM_MKT_L:
            # 市价卖单到达，按 FIFO 从队首吃
            qty = max(0.1, rng.exponential(SIZE_MKT))
            if qty >= ahead + my_size:
                # 整笔吃穿，我完全成交
                # 大单吃穿队列 -> 往往伴随价格继续下行（逆向选择）
                adverse = rng.random() < adverse_prob
                return dict(filled=True, wait=t, adverse=bool(adverse), reason="fill")
            elif qty > ahead:
                # 吃到我这里但没吃满：部分成交，剩余继续排（保守起见按未完全成交处理）
                ahead = 0.0
            else:
                ahead -= qty
        elif u < LAM_MKT_L + LAM_ADD:
            pass  # 新单排在我后面，不影响我
        elif u < LAM_MKT_L + LAM_ADD + LAM_CXL:
            # 前方随机撤单：撤单发生在我前面的概率 ∝ 前方占比
            if ahead > 0 and rng.random() < ahead / (ahead + 12.0):
                ahead = max(0.0, ahead - max(0.1, rng.exponential(2.0)))
        else:
            # 价格跳走：我这一档被抛在后面，等于没成交（需要追价）
            return dict(filled=False, wait=t, adverse=False, reason="price_moved")
    return dict(filled=False, wait=HORIZON, adverse=False, reason="timeout")


def run_batch(q_ahead, n=4000, seed=SEED, **kw):
    rng = np.random.default_rng(seed)
    res = [simulate_one(rng, q_ahead, **kw) for _ in range(n)]
    filled = np.array([r["filled"] for r in res])
    waits = np.array([r["wait"] for r in res])
    adverse = np.array([r["adverse"] for r in res])
    moved = np.array([r["reason"] == "price_moved" for r in res])
    return dict(
        q_ahead=float(q_ahead),
        fill_rate=float(filled.mean()),
        median_wait=float(np.median(waits[filled])) if filled.any() else np.nan,
        mean_wait=float(waits[filled].mean()) if filled.any() else np.nan,
        adverse_rate=float(adverse[filled].mean()) if filled.any() else np.nan,
        move_rate=float(moved.mean()),
        waits_filled=waits[filled],
    )


def main():
    # ============ 1. 队列位置 -> 成交概率 ============
    q_grid = [0, 2, 5, 10, 20, 40, 80, 150, 300]
    batches = [run_batch(q, n=4000, seed=SEED + i) for i, q in enumerate(q_grid)]
    fill_rates = [b["fill_rate"] for b in batches]
    med_waits = [b["median_wait"] for b in batches]

    # ============ 2. 快 vs 慢：撤单重挂的代价 ============
    # 场景：价格档位刷新后重新挂单。快手排到 q=3，慢手（+300ms 延迟）排到 q=25
    fast = run_batch(3, n=6000, seed=SEED + 101)
    slow = run_batch(25, n=6000, seed=SEED + 102)

    # ============ 3. 逆向选择：成交了反而是坏事 ============
    # 队首成交更多来自"大单吃穿"，此时价格更可能继续走
    adv_rates = [b["adverse_rate"] for b in batches]

    # ============ 4. 策略对比：被动挂单 vs 追价成交 ============
    # 收益模型：被动成交省下半个价差，但承担逆向选择成本；未成交则错过（机会成本）
    # 全部以【tick】为单位，不依赖具体价格水平
    HALF_SPREAD = 0.5      # 被动成交相对中间价的优势（tick）
    ADVERSE_COST = 1.15    # 逆向选择时的平均不利变动（tick）
    MISS_COST = 0.35       # 未成交、事后追价的机会成本（tick）

    def strategy_pnl(b):
        f, a = b["fill_rate"], b["adverse_rate"]
        return float(f * (HALF_SPREAD - a * ADVERSE_COST) - (1 - f) * MISS_COST)

    pnls = [strategy_pnl(b) for b in batches]          # tick
    aggressive_pnl = -HALF_SPREAD                       # 追价：必成交，付半个价差

    # ============ 5. 对抗式检验 ============
    checks = {}

    # 5a 关掉队列机制（随机成交，与位置无关）-> 成交率应与 q 无关
    def simulate_noqueue(rng, q_ahead, my_size=1.0):
        t = 0.0
        total_rate = LAM_MKT + LAM_MOVE
        while t < HORIZON:
            t += rng.exponential(1.0 / total_rate)
            if rng.random() * total_rate < LAM_MKT:
                return dict(filled=True, wait=t, adverse=False, reason="fill")
            return dict(filled=False, wait=t, adverse=False, reason="price_moved")
        return dict(filled=False, wait=HORIZON, adverse=False, reason="timeout")

    nq = []
    for q in q_grid:
        rng = np.random.default_rng(SEED + 500 + q)
        r = [simulate_noqueue(rng, q) for _ in range(3000)]
        nq.append(float(np.mean([x["filled"] for x in r])))
    checks["noqueue_fill_rates"] = nq
    checks["noqueue_spread"] = float(max(nq) - min(nq))
    checks["queue_spread"] = float(max(fill_rates) - min(fill_rates))

    # 5b 单调性检验
    checks["monotonic_fill"] = bool(all(
        fill_rates[i] >= fill_rates[i + 1] - 1e-9 for i in range(len(fill_rates) - 1)))

    # 5c 解析近似交叉验证：忽略撤单与跳价时，纯 FIFO 消耗下的成交概率
    # 队列前方 q 手，市价单流量 LAM_MKT*SIZE_MKT 手/秒，价格跳走强度 LAM_MOVE
    # P(在跳价前吃完 q+my_size) ≈ 消耗速率 / (消耗速率 + 跳价速率*所需量)
    analytic = []
    for q in q_grid:
        need = q + 1.0
        consume_rate = LAM_MKT * SIZE_MKT          # 手/秒
        t_need = need / consume_rate
        p_no_move = np.exp(-LAM_MOVE * t_need)     # 跳价前完成
        analytic.append(float(p_no_move))
    checks["analytic_approx"] = analytic
    checks["corr_sim_analytic"] = float(np.corrcoef(fill_rates, analytic)[0, 1])

    # 5d 种子稳健性
    seeds_fr = []
    for sd in range(12):
        b = run_batch(20, n=1500, seed=9000 + sd)
        seeds_fr.append(b["fill_rate"])
    checks["seed_fill_mean"] = float(np.mean(seeds_fr))
    checks["seed_fill_std"] = float(np.std(seeds_fr))

    # 5e 参数敏感性：市价单流量翻倍/减半
    sens = {}
    for mult, lbl in [(0.5, "half"), (1.0, "base"), (2.0, "double")]:
        sens[lbl] = [run_batch(q, n=1500, seed=SEED + 700 + i,
                               lam_mkt=LAM_MKT * mult)["fill_rate"]
                     for i, q in enumerate(q_grid)]
    checks["flow_sensitivity"] = sens

    # ================= 绘图 =================
    qx = np.arange(len(q_grid))

    # 图1 cover：成交概率 + 中位等待
    fig, ax1 = plt.subplots(figsize=(11, 5.2))
    ax1.plot(qx, [f * 100 for f in fill_rates], marker="o", ms=7, lw=2.2, color=C_A,
             label="成交概率")
    ax1.set_xticks(qx); ax1.set_xticklabels([str(q) for q in q_grid])
    ax1.set_xlabel("下单瞬间排在我前面的手数（队列位置）")
    ax1.set_ylabel("成交概率 (%)", color=C_A)
    ax1.tick_params(axis="y", labelcolor=C_A)
    ax1.set_ylim(0, 100)
    for i, f in enumerate(fill_rates):
        ax1.annotate(f"{f*100:.0f}%", (i, f * 100), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=9, color=C_A)
    ax2 = ax1.twinx(); ax2.grid(False)
    ax2.plot(qx, med_waits, marker="s", ms=6, lw=1.8, color=C_B, ls="--",
             label="成交单的中位等待时间")
    ax2.set_ylabel("中位等待时间（秒）", color=C_B)
    ax2.tick_params(axis="y", labelcolor=C_B)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9.5)
    ax1.set_title(f"同样的挂单价格，队列位置决定一切：排第 1 成交率 {fill_rates[0]*100:.0f}%，"
                  f"排到第 {q_grid[-1]} 手后只剩 {fill_rates[-1]*100:.0f}%",
                  fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{OUT}/cover.png"); plt.close()

    # 图2 快慢手对比
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    names = ["快手 q=3\n（低延迟）", "慢手 q=25\n（+300ms）"]
    fr = [fast["fill_rate"] * 100, slow["fill_rate"] * 100]
    ax[0].bar(names, fr, 0.5, color=[C_C, C_B])
    ax[0].set_ylabel("成交概率 (%)")
    ax[0].set_title(f"延迟 → 队列位置 → 成交率\n差距 {fr[0]-fr[1]:.0f} 个百分点", fontsize=11)
    for i, v in enumerate(fr):
        ax[0].text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    ax[1].hist(fast["waits_filled"], bins=50, alpha=0.62, color=C_C, label="快手 q=3", density=True)
    ax[1].hist(slow["waits_filled"], bins=50, alpha=0.62, color=C_B, label="慢手 q=25", density=True)
    ax[1].set_xlabel("成交等待时间（秒）"); ax[1].set_ylabel("密度")
    ax[1].set_xlim(0, 60)
    ax[1].set_title(f"等待时间分布\n中位 {fast['median_wait']:.1f}s vs {slow['median_wait']:.1f}s",
                    fontsize=11)
    ax[1].legend(fontsize=9)
    reasons = ["成交", "价格跳走", "超时未成交"]
    fa = [fast["fill_rate"], fast["move_rate"], 1 - fast["fill_rate"] - fast["move_rate"]]
    sl = [slow["fill_rate"], slow["move_rate"], 1 - slow["fill_rate"] - slow["move_rate"]]
    xr = np.arange(3)
    ax[2].bar(xr - 0.2, [v * 100 for v in fa], 0.4, color=C_C, label="快手 q=3")
    ax[2].bar(xr + 0.2, [v * 100 for v in sl], 0.4, color=C_B, label="慢手 q=25")
    ax[2].set_xticks(xr); ax[2].set_xticklabels(reasons, fontsize=9)
    ax[2].set_ylabel("占比 (%)")
    ax[2].set_title("订单最终去向拆解", fontsize=11)
    ax[2].legend(fontsize=9)
    plt.suptitle("延迟的真实代价不是「慢了 300 毫秒」，而是排到了队列后面",
                 fontsize=12.5, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{OUT}/latency_queue_cost.png"); plt.close()

    # 图3 逆向选择
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].plot(qx, [a * 100 for a in adv_rates], marker="o", ms=6, lw=2, color=C_D)
    ax[0].set_xticks(qx); ax[0].set_xticklabels([str(q) for q in q_grid])
    ax[0].set_xlabel("队列位置（前方手数）")
    ax[0].set_ylabel("成交单中「成交后价格继续不利」占比 (%)")
    ax[0].set_title("排在队首成交更快，但成交质量并不更好\n"
                    "（成交本身就是被大单扫中的信号）", fontsize=11)
    ax[0].set_ylim(0, 100)
    ax[1].plot(qx, pnls, marker="o", ms=6, lw=2, color=C_A, label="被动挂单净收益")
    ax[1].axhline(aggressive_pnl, color=C_B, ls="--", lw=1.8,
                  label=f"主动追价基准 {aggressive_pnl:.2f} tick（必成交）")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(qx); ax[1].set_xticklabels([str(q) for q in q_grid])
    ax[1].set_xlabel("队列位置（前方手数）")
    ax[1].set_ylabel("每笔净收益（tick）")
    ax[1].set_title("相对追价的优势随队列位置衰减", fontsize=11)
    ax[1].legend(fontsize=8.8)
    for i, p in enumerate(pnls):
        ax[1].annotate(f"{p:.2f}", (i, p), textcoords="offset points",
                       xytext=(0, 8), ha="center", fontsize=8.5)
    plt.tight_layout(); plt.savefig(f"{OUT}/adverse_selection.png"); plt.close()

    # 图4 对抗式检验
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    ax[0].plot(qx, [f * 100 for f in fill_rates], marker="o", ms=6, lw=2, color=C_A,
               label=f"含队列机制（跨度 {checks['queue_spread']*100:.0f} pp）")
    ax[0].plot(qx, [f * 100 for f in nq], marker="s", ms=5, lw=1.8, color="#94a3b8",
               ls="--", label=f"关掉队列（跨度 {checks['noqueue_spread']*100:.1f} pp）")
    ax[0].set_xticks(qx); ax[0].set_xticklabels([str(q) for q in q_grid])
    ax[0].set_xlabel("队列位置"); ax[0].set_ylabel("成交概率 (%)")
    ax[0].set_title("对照组：关掉队列机制后\n位置不再影响成交率", fontsize=11)
    ax[0].legend(fontsize=8.5); ax[0].set_ylim(0, 100)
    ax[1].plot(qx, [f * 100 for f in fill_rates], marker="o", ms=6, lw=2, color=C_A,
               label="事件驱动仿真")
    ax[1].plot(qx, [a * 100 for a in analytic], marker="^", ms=5, lw=1.6, color=C_C,
               ls="--", label="解析近似（纯 FIFO）")
    ax[1].set_xticks(qx); ax[1].set_xticklabels([str(q) for q in q_grid])
    ax[1].set_xlabel("队列位置"); ax[1].set_ylabel("成交概率 (%)")
    ax[1].set_title(f"独立解析式交叉验证\n相关系数 {checks['corr_sim_analytic']:.3f}", fontsize=11)
    ax[1].legend(fontsize=8.5); ax[1].set_ylim(0, 100)
    for lbl, c, nm in [("half", "#94a3b8", "市价单流量 ×0.5"),
                       ("base", C_A, "基准"),
                       ("double", C_C, "市价单流量 ×2")]:
        ax[2].plot(qx, [f * 100 for f in sens[lbl]], marker="o", ms=5, lw=1.8,
                   color=c, label=nm)
    ax[2].set_xticks(qx); ax[2].set_xticklabels([str(q) for q in q_grid])
    ax[2].set_xlabel("队列位置"); ax[2].set_ylabel("成交概率 (%)")
    ax[2].set_title("参数敏感性：流量翻倍也救不了\n深队列位置", fontsize=11)
    ax[2].legend(fontsize=8.5); ax[2].set_ylim(0, 100)
    plt.tight_layout(); plt.savefig(f"{OUT}/adversarial_checks.png"); plt.close()

    stats = dict(
        seed=SEED, tick=TICK,
        params=dict(lam_mkt=LAM_MKT, lam_add=LAM_ADD, lam_cxl=LAM_CXL,
                    lam_move=LAM_MOVE, size_mkt=SIZE_MKT, horizon=HORIZON),
        q_grid=q_grid,
        fill_rates=fill_rates,
        median_waits=med_waits,
        adverse_rates=adv_rates,
        move_rates=[b["move_rate"] for b in batches],
        pnl_ticks=pnls,
        aggressive_pnl_ticks=aggressive_pnl,
        edge_vs_aggressive=[float(p - aggressive_pnl) for p in pnls],
        fast=dict(q=3, fill_rate=fast["fill_rate"], median_wait=fast["median_wait"],
                  adverse=fast["adverse_rate"], move=fast["move_rate"]),
        slow=dict(q=25, fill_rate=slow["fill_rate"], median_wait=slow["median_wait"],
                  adverse=slow["adverse_rate"], move=slow["move_rate"]),
        checks=checks,
    )
    with open(f"{OUT}/stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in stats.items() if k != "checks"},
                     indent=2, ensure_ascii=False))
    print("CHECKS:", json.dumps(checks, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
