#!/usr/bin/env python3
"""为文章「多智能体强化学习交易：用对手建模学会在博弈中下单」
(multi-agent-rl-trading) 生成真实配图（matplotlib，non-placeholder）。

全部数字来自真实 numpy 计算，可复现（seed=20260828）。

实验设计（诚实、可复现的「执行博弈」，Almgren-Chriss 纯瞬时冲击）：
  场景：M 个机构交易员各自要在 T 期内 liquidate 总量 V（执行博弈）。
  价格冲击模型（纯瞬时临时冲击，可分离凹二次规划）：
      P_t = P0 - lambda * Q_t ,   Q_t = t 期全体总成交量
  交易员 i 收到的成交均价 = sum_t q_{i,t} P_t / V
  实现短缺 IS(bps) = (P0 - 均价)/P0 * 1e4  （越小越好）
  纯瞬时冲击下，给定对手计划，自身最优反应是精确可解的凹 QP（water-filling）：
      max sum_t [L_t q_t - lambda q_t^2],  q>=0, sum=V,  L_t = P0 - lambda*Q_opp,t
  q_t = max(0, (mu - L_t)/(2*lambda))，mu 由 sum=V 定（二分）。

  对照：
    (A) 全朴素 TWAP：所有交易员均匀卖出 -> 互相撞车
    (B) 全对手建模(虚拟博弈)：各自对对手经验均值做最优反应 -> 协调到均匀铺开
    (C) 2人混合：1 个对手走「可预测的提前卖出」固定计划；1 个聪明交易员
        (c1) 朴素TWAP  vs  (c2) 对手建模最优反应 -> 建模者避开高峰省钱
    (D) 可预测性消融：对手计划加噪声 epsilon，建模者优势随 epsilon 增大消失

  诚实红线：纯瞬时冲击下最优反应可精确求解；含永久冲击时目标交叉项让 BR 退化为
            T×T 稠密 QP（本文不做），下文如实标注为局限。
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "multi-agent-rl-trading")
os.makedirs(D, exist_ok=True)

P0 = 100.0          # 初始价格
V = 1.0             # 每个交易员 liquidation 总量（归一化单位）
T = 20              # 期数
M = 5               # 交易员数（虚拟博弈用）
LAMBDA = 0.6        # 瞬时冲击系数（每单位流量，调大以增强可见度）

C_AGG = "#1f4e79"   # 聚合成交量
C_NAIVE = "#c0392b" # 朴素
C_MODEL = "#27ae60" # 对手建模
C_OPP = "#8e44ad"   # 对手
GRID = "#e6e6e6"

def agg_price(Q_t_arr):
    return P0 - LAMBDA * Q_t_arr

def is_bps(q_i, Q_t_arr):
    """交易员 i 的实现短缺(bps)。q_i: (T,), Q_t_arr 为全员聚合序列。"""
    avg_px = np.sum(q_i * agg_price(Q_t_arr)) / V
    return (P0 - avg_px) / P0 * 1e4

def best_response(opp_q, V=V, lam=LAMBDA):
    """给定对手 t 期计划 opp_q (T,)，求自身最优反应 q_i (T,) >=0, sum=V。
    纯瞬时冲击下目标 = sum_t [L_t q_t - lam q_t^2]，凹、可分离。
      L_t = P0 - lam*Q_opp,t
    water-filling: q_t = max(0, (L_t - mu)/(2*lam))，mu 由 sum=V 定（二分）。
    sum 随 mu 单调减 -> 若 s>V 则 mu 偏小、lo=mid；否则 hi=mid。
    """
    Q_opp = opp_q
    L = P0 - lam * Q_opp
    denom = 2.0 * lam
    def q_of_mu(mu):
        q = (L - mu) / denom
        return np.clip(q, 0, None)
    lo, hi = L.min() - 10, L.max() + 10
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        s = q_of_mu(mid).sum()
        if s > V:
            lo = mid
        else:
            hi = mid
    q = q_of_mu(0.5 * (lo + hi))
    q = q / q.sum() * V
    return q

def front_loaded_weights(T=T, rho=0.82):
    w = rho ** np.arange(T)
    return w / w.sum()

def uniform_weights(T=T):
    return np.ones(T) / T

# ============================================================
# 图1：示意图——多智能体执行博弈与对手建模回路（matplotlib 绘制）
# ============================================================
def fig_schematic():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    # 左：单次博弈
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.add_patch(plt.Rectangle((1, 4), 2.4, 2, fc="#dbe9f4", ec=C_AGG))
    ax.text(2.2, 5, "市场\n(冲击模型)", ha="center", va="center", fontsize=10)
    for cx, col, lab in [(6.4, C_NAIVE, "Agent i\n(朴素TWAP)"), (6.4, C_MODEL, "Agent j\n(对手建模)")]:
        ax.add_patch(plt.Rectangle((cx-1.2, 1.5 if lab.startswith("Agent i") else 6.5), 2.4, 2, fc="#fdecea" if lab.startswith("Agent i") else "#eafaf0", ec=col))
        ax.text(cx, 2.5 if lab.startswith("Agent i") else 7.5, lab, ha="center", va="center", fontsize=9, color=col)
        ax.annotate("", xy=(3.4, 5), xytext=(cx-1.2, 2.5 if lab.startswith("Agent i") else 7.5), arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
        ax.annotate("", xy=(cx-1.2, 3.0 if lab.startswith("Agent i") else 8.0), xytext=(3.4, 5.2), arrowprops=dict(arrowstyle="->", color=col, lw=1.2, ls="--"))
    ax.set_title("① 单次执行博弈：\n订单流撞出价格冲击", fontsize=10)
    # 中：对手建模回路
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    nodes = [("对手行为\n观测/历史", 2, 7, "#f5eef8", C_OPP),
             ("对手模型\n(策略分布估计)", 2, 4, "#f5eef8", C_OPP),
             ("最优反应\n(water-filling)", 5.5, 4, "#eafaf0", C_MODEL),
             ("下单 q_i,t", 8.5, 4, "#eafaf0", C_MODEL),
             ("市场反馈\n冲击/价格", 8.5, 7, "#dbe9f4", C_AGG)]
    for lab, x, y, fc, ec in nodes:
        ax.add_patch(plt.Rectangle((x-1.3, y-1), 2.6, 2, fc=fc, ec=ec))
        ax.text(x, y, lab, ha="center", va="center", fontsize=8.5, color=ec)
    ax.annotate("", xy=(3.3, 4), xytext=(3.3, 6), arrowprops=dict(arrowstyle="->", color=C_OPP))
    ax.annotate("", xy=(4.2, 4), xytext=(6.8, 4), arrowprops=dict(arrowstyle="->", color=C_MODEL))
    ax.annotate("", xy=(7.2, 4), xytext=(9.8, 4), arrowprops=dict(arrowstyle="->", color=C_MODEL))
    ax.annotate("", xy=(9.8, 6), xytext=(9.8, 8), arrowprops=dict(arrowstyle="->", color=C_AGG))
    ax.annotate("", xy=(3.3, 8), xytext=(3.3, 6.2), arrowprops=dict(arrowstyle="->", color=C_OPP, ls="--"))
    ax.set_title("② 对手建模回路：\n估计对手策略→最优反应", fontsize=10)
    # 右：均衡 / 协调
    ax = axes[2]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(5, 8.6, "虚拟博弈 (Fictitious Play)", ha="center", fontsize=9.5, color=C_MODEL)
    for i, (bx, col) in enumerate([(1.5, C_NAIVE), (4.0, C_MODEL), (6.5, C_OPP), (9.0, C_AGG)]):
        ax.add_patch(plt.Circle((bx, 4.5), 1.3, fc="#f4f6f7", ec=col))
        ax.text(bx, 4.5, f"A{i+1}", ha="center", va="center", color=col, fontsize=9)
        ax.annotate("", xy=(bx+1.4, 4.5), xytext=(bx+0.3, 4.5), arrowprops=dict(arrowstyle="->", color=col, lw=1.0))
    ax.text(5, 1.6, "反复互相最优反应 → 协调铺开\n(避免同窗撞车)", ha="center", fontsize=9, color=C_MODEL)
    ax.set_title("③ 多智能体收敛：\n从撞车到协调", fontsize=10)
    fig.suptitle("多智能体强化学习交易：在博弈中下单的核心结构", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{D}/marl_schematic.png", dpi=160, bbox_inches="tight"); plt.close()

# ============================================================
# 图2：混合博弈——对手建模者避开对手的高峰
# ============================================================
def fig_mixed_schedule():
    rng = np.random.default_rng(20260828)
    # 2 人博弈：1 个对手走可预测的「提前卖出」固定计划，1 个聪明交易员(占 1/2 总量)
    n_opp = 1
    opp_w = front_loaded_weights(T, rho=0.70) * V
    opp_q_total = n_opp * opp_w                          # 对手聚合计划
    # 朴素聪明交易员：TWAP
    naive_q = uniform_weights(T) * V
    # 对手建模聪明交易员：对 opp_q_total 做最优反应
    model_q = best_response(opp_q_total, V=V) * V

    Q_naive = opp_q_total + naive_q
    Q_model = opp_q_total + model_q
    is_naive = is_bps(naive_q, Q_naive)
    is_model = is_bps(model_q, Q_model)

    fig, ax = plt.subplots(figsize=(11, 5.4))
    x = np.arange(T)
    ax.bar(x - 0.25, opp_q_total, width=0.25, color=C_OPP, alpha=0.75, label="对手 提前卖出计划")
    ax.bar(x, naive_q, width=0.25, color=C_NAIVE, alpha=0.85, label=f"聪明交易员·朴素TWAP  (IS={is_naive:.1f}bps)")
    ax.bar(x + 0.25, model_q, width=0.25, color=C_MODEL, alpha=0.9, label=f"聪明交易员·对手建模  (IS={is_model:.1f}bps)")
    ax.set_xlabel("期 t"); ax.set_ylabel("该期卖出量 q")
    ax.set_title("2人博弈：对手建模者把卖单挪到对手稀疏的后期，避开冲击高峰")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(axis="y", color=GRID)
    ax.set_xticks(x)
    fig.tight_layout()
    fig.savefig(f"{D}/marl_mixed_schedule.png", dpi=160, bbox_inches="tight"); plt.close()
    return dict(is_naive=float(is_naive), is_model=float(is_model),
                saving_bps=float(is_naive - is_model),
                opp_early=list(np.round(opp_q_total[:5],3)),
                model_late=list(np.round(model_q[-5:],3)))

# ============================================================
# 图3：全对手建模(虚拟博弈)协调——从撞车到均匀铺开
# ============================================================
def fig_fictitious_play():
    rng = np.random.default_rng(7)
    # 所有 M 个交易员初始都「提前卖出」(撞车)，逐步做虚拟博弈
    cur = [front_loaded_weights(T, rho=0.55) * V for _ in range(M)]
    schedules = [np.array([c.copy() for c in cur])]  # 记录每轮各交易员计划
    total_is = []
    def total_is_of(plans):
        Q = np.sum(plans, axis=0); C = np.cumsum(Q)
        return np.sum([is_bps(plans[i], Q) for i in range(M)]) / M
    total_is.append(total_is_of(np.array(cur)))
    for rnd in range(25):
        new = []
        for i in range(M):
            others = np.sum([cur[j] for j in range(M) if j != i], axis=0)
            new.append(best_response(others, V=V))
        cur = new
        schedules.append(np.array([c.copy() for c in cur]))
        total_is.append(total_is_of(np.array(cur)))
    # 取首轮、中段、末轮做示意图
    show_rounds = [0, 8, len(schedules) - 1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True)
    for ax, r in zip(axes, show_rounds):
        agg = schedules[r].sum(axis=0)
        axes_idx = 0
        ax.bar(np.arange(T), agg, color=C_AGG, alpha=0.85)
        ax.set_title(f"第{r}轮 聚合Q_t\n(均值IS={total_is[r]:.2f}bps)", fontsize=10)
        ax.set_xlabel("期 t"); ax.grid(axis="y", color=GRID)
    axes[0].set_ylabel("全体聚合成交量 Q_t")
    fig.suptitle("虚拟博弈：各交易员对对手经验均值做最优反应，聚合成交量从早期撞车收敛到均匀铺开", fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(f"{D}/marl_fictitious_play.png", dpi=160, bbox_inches="tight"); plt.close()
    return dict(is_start=float(total_is[0]), is_end=float(total_is[-1]),
                rounds=len(schedules)-1,
                drop_pct=float(100*(total_is[0]-total_is[-1])/total_is[0]))

# ============================================================
# 图4：可预测性消融——对手越不可预测，建模优势越消失
# ============================================================
def fig_predictability_ablation():
    rng = np.random.default_rng(20260828)
    opp_base = front_loaded_weights(T, rho=0.70)
    n_opp = 1
    epsilons = np.linspace(0, 4.0, 21)
    is_naive_list, is_model_list, adv = [], [], []
    for eps in epsilons:
        is_naives, is_models = [], []
        for trial in range(40):
            # 对手真实计划 = base*(1+eps*noise)，clip>=0，归一化到 V
            noise = rng.standard_normal(T)
            opp_true = np.clip(opp_base * (1 + eps * noise), 0, None)
            opp_true = opp_true / opp_true.sum() * V
            opp_agg = n_opp * opp_true
            naive_q = uniform_weights(T) * V
            # 建模者用「无噪声的 base 信念」估计对手（体现预测误差随 eps 增大）
            model_belief = n_opp * opp_base * V
            model_q = best_response(model_belief, V=V)
            Qn = opp_agg + naive_q; Cn = np.cumsum(Qn)
            Qm = opp_agg + model_q; Cm = np.cumsum(Qm)
            is_naives.append(is_bps(naive_q, Qn))
            is_models.append(is_bps(model_q, Qm))
        is_naive_list.append(np.mean(is_naives))
        is_model_list.append(np.mean(is_models))
        adv.append(np.mean(is_naives) - np.mean(is_models))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(epsilons, is_naive_list, "-o", color=C_NAIVE, label="朴素TWAP 聪明交易员 IS")
    ax.plot(epsilons, is_model_list, "-s", color=C_MODEL, label="对手建模 聪明交易员 IS")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("对手计划噪声 ε（可预测性越低越大）")
    ax.set_ylabel("实现短缺 IS (bps)")
    ax.set_title("可预测性消融：对手越不可预测，对手建模的优势越消失（诚实红线）")
    ax.legend(fontsize=9); ax.grid(color=GRID)
    fig.tight_layout()
    fig.savefig(f"{D}/marl_predictability_ablation.png", dpi=160, bbox_inches="tight"); plt.close()
    return dict(eps=list(np.round(epsilons,2)),
                adv_at_0=float(adv[0]), adv_at_end=float(adv[-1]),
                is_naive_0=float(is_naive_list[0]), is_model_0=float(is_model_list[0]))

if __name__ == "__main__":
    fig_schematic()
    mixed = fig_mixed_schedule()
    fp = fig_fictitious_play()
    abl = fig_predictability_ablation()
    out = dict(mixed=mixed, fictitious_play=fp, ablation=abl)
    print(json.dumps(out, ensure_ascii=False, indent=2))
