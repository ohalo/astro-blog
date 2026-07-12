#!/usr/env python3
"""
为文章「Maker-Taker 费率结构与订单类型博弈：谁在补贴谁」(maker-taker-pricing)
生成真实配图。所有图表均由脚本内自洽合成数据 + 文中方法真实计算生成。

机制：
  - maker-taker 费率：挂单(maker,提供流动性)拿返佣 rebate>0；吃单(taker,拿走流动性)付费 fee>0。
    每笔成交 exchange 净收 = fee - rebate，由 taker 付费中切出 rebate 补贴 maker。
  - 订单类型博弈：同一笔「想买 100 股」的经济决策，市价单(taker)立即成交但付 spread+fee；
    限价单(maker)挂买一等待，拿到 rebate+半价差，但要承担「被知情单挑中」的逆选择成本。
  - 谁补贴谁：taker 交的费用里，rebate 部分流入 maker 口袋，fee-rebate 部分归交易所。
    maker 的净盈利 = 半价差 + rebate - 逆选择；当 p_info 足够大时 maker 转亏，补贴关系逆转。
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
D = os.path.join(BASE, "maker-taker-pricing")
os.makedirs(D, exist_ok=True)

C = {"maker": "#2F7ED8", "taker": "#C44E52", "exch": "#55A868",
     "grid": "#DDDDDD", "info": "#8C564B", "line": "#333333"}

# ===================== 参数 =====================
P0 = 50.0          # 标的初始价(美元)
s = 0.01           # 半价差(美元/股) = 1 分
R = 0.002          # maker 返佣(美元/股)
F = 0.003          # taker 费用(美元/股)
q = 1000           # 每笔成交股数(整手)
T = 2520           # 交易日(约 10 年)
p_take = 0.50      # 每个交易日出现吃单的概率
p_info = 0.30      # 吃单中「知情单」比例(逆选择来源)
info_mag = 0.02    # 知情单触发后下一期不利移动(美元/股)
mu = 0.0002        # 日度漂移项
sigma = 0.012      # 日度波动

rng = np.random.default_rng(20260712)

# ===================== 模拟 =====================
mid = np.empty(T)
mid[0] = P0
maker_rebate = 0.0       # maker 累计返佣(现金+)
maker_spread = 0.0       # maker 通过「挂买一、次日按中间价平仓」捕获的半价差
maker_adv = 0.0          # maker 累计逆选择损失
taker_fee = 0.0          # taker 累计付费
taker_spread = 0.0       # taker 累计付出的价差(买在 ask)
taker_avoid = 0.0        # taker 知情单「提前离场避免的下跌」
fills = 0
informed_fills = 0

maker_pnl_series = [0.0]
taker_cost_series = [0.0]
exch_series = [0.0]

for t in range(T - 1):
    # 当日中间价随机游走
    ret = mu + sigma * rng.standard_normal()
    mid[t + 1] = mid[t] * (1.0 + ret)
    bid = mid[t] - s
    ask = mid[t] + s

    if rng.random() < p_take:
        fills += 1
        informed = rng.random() < p_info
        # ---- maker 挂买一被吃 ----
        maker_rebate += q * R
        maker_spread += q * s                      # 以 bid 买入, 次日按 mid 平, 赚 half-spread
        # ---- taker 市价卖(吃买一) ----
        taker_fee += q * F
        taker_spread += q * s                      # 卖在 bid, 比 mid 少 s

        if informed:
            informed_fills += 1
            adv = info_mag                          # 下一期中间价不利移动(对刚买入的 maker 是损失)
            maker_adv += q * adv
            taker_avoid += q * adv                  # 知情 taker 提前离场, 避开这笔下跌

    maker_net = maker_rebate + maker_spread - maker_adv
    taker_net = -(taker_fee + taker_spread - taker_avoid)
    exch_net = taker_fee - maker_rebate
    maker_pnl_series.append(maker_net)
    taker_cost_series.append(taker_net)
    exch_series.append(exch_net)

# ===================== 统计 =====================
n_fill = max(fills, 1)
rebate_per_fill = maker_rebate / n_fill
fee_per_fill = taker_fee / n_fill
print("==== MAKER-TAKER 关键统计 ====")
print(f"交易日 T={T}, 成交笔数 fills={fills}, 知情成交 informed={informed_fills} ({informed_fills/n_fill:.1%})")
print(f"每笔 maker 返佣 = {R:.4f} 美元/股 = {R/P0*1e4:.2f} bps; 每笔 taker 费用 = {F:.4f} 美元/股 = {F/P0*1e4:.2f} bps")
print(f"交易所每笔净收 = {F-R:.4f} 美元/股 (占 taker 费用 {(F-R)/F:.1%})")
print(f"maker 总返佣 = ${maker_rebate:,.0f}; taker 总费用 = ${taker_fee:,.0f}; 交易所总收入 = ${taker_fee-maker_rebate:,.0f}")
print(f"maker 半价差捕获 = ${maker_spread:,.0f}; maker 逆选择损失 = ${maker_adv:,.0f}")
print(f"maker 净盈利 = ${maker_rebate+maker_spread-maker_adv:,.0f}")
print(f"taker 净成本(付费+价差-避损) = ${taker_fee+taker_spread-taker_avoid:,.0f}; 知情避损 = ${taker_avoid:,.0f}")
print(f"补贴占比: taker 每 1 美元费用中 {R/F:.1%} 流入 maker 返佣, {(F-R)/F:.1%} 归交易所")

# 盈亏平衡: maker 每笔净 = s + R - p_info*info_mag ; taker 每笔净成本 = s + F - p_info*info_mag
def maker_net_per_fill(pinfo):
    return s + R - pinfo * info_mag
def taker_net_per_fill(pinfo):
    return s + F - pinfo * info_mag
p_break = (s + R) / info_mag
print(f"maker 盈亏平衡信息比例 p_info* = {p_break:.3f} (本校准 p_info={p_info})")

# ===================== 图 1: 每笔交易费用去向 =====================
fig, ax = plt.subplots(figsize=(8, 4.6))
labels = ["taker 付费 (0.003)\n全部来源", "maker 返佣 (0.002)\n流向 maker", "交易所净收 (0.001)\n平台抽水"]
vals = [F, R, F - R]
colors = [C["taker"], C["maker"], C["exch"]]
y = np.arange(len(labels))
ax.barh(y, vals, color=colors, edgecolor="white")
for i, v in enumerate(vals):
    ax.text(v + 0.00008, i, f"{v:.4f}  ({v/F:.1%})", va="center", fontsize=12)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlim(0, F * 1.35)
ax.set_xlabel("美元 / 股", fontsize=11)
ax.set_title("每笔成交的费用分配：taker 付费被拆成 maker 返佣 + 交易所净收", fontsize=13, fontweight="bold")
ax.grid(axis="x", color=C["grid"])
plt.tight_layout()
plt.savefig(os.path.join(D, "mt_fee_structure.png"), dpi=130)
plt.close()

# ===================== 图 2: 累计 PnL =====================
fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(len(maker_pnl_series))
ax.plot(x, maker_pnl_series, color=C["maker"], lw=1.8, label="maker 策略累计净盈利")
ax.plot(x, exch_series, color=C["exch"], lw=1.8, label="交易所累计净收入")
ax.plot(x, taker_cost_series, color=C["taker"], lw=1.8, label="taker 累计净成本(取负)")
ax.axhline(0, color=C["line"], lw=0.8, ls="--")
ax.set_xlabel("交易日", fontsize=11)
ax.set_ylabel("累计美元", fontsize=11)
ax.set_title("资金流向：maker 稳赚、交易所抽水、taker 持续失血", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
ax.grid(color=C["grid"], alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "mt_cum_pnl.png"), dpi=130)
plt.close()

# ===================== 图 3: 盈亏平衡(逆选择敏感度) =====================
pinfo_grid = np.linspace(0, 0.8, 200)
mk = [maker_net_per_fill(p) for p in pinfo_grid]
tk = [taker_net_per_fill(p) for p in pinfo_grid]
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.plot(pinfo_grid, mk, color=C["maker"], lw=2.2, label="maker 每笔净盈利")
ax.plot(pinfo_grid, tk, color=C["taker"], lw=2.2, label="taker 每笔净成本")
ax.axhline(0, color=C["line"], lw=0.9, ls="--")
ax.axvline(p_break, color=C["info"], lw=1.5, ls=":", label=f"maker 盈亏平衡 p_info*={p_break:.2f}")
ax.scatter([p_info], [maker_net_per_fill(p_info)], color=C["maker"], zorder=5, s=60)
ax.annotate(f"本校准 p_info={p_info}\nmaker 净={maker_net_per_fill(p_info):.4f}/股",
            (p_info, maker_net_per_fill(p_info)), xytext=(p_info + 0.05, maker_net_per_fill(p_info) + 0.002),
            fontsize=10, arrowprops=dict(arrowstyle="->", color=C["info"]))
ax.set_xlabel("知情单比例 p_info（逆选择强度）", fontsize=11)
ax.set_ylabel("每笔净收益 / 美元每手", fontsize=11)
ax.set_title("逆选择强度决定补贴方向：越过盈亏平衡, maker 转亏、taker 反补", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(color=C["grid"], alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "mt_breakeven.png"), dpi=130)
plt.close()

# ===================== 图 4: 资金总流量 =====================
fig, ax = plt.subplots(figsize=(8.5, 4.8))
cats = ["taker 总付费", "maker 总返佣", "交易所总收入", "maker 净盈利", "taker 净成本"]
# 用绝对值展示流向
v_taker_fee = taker_fee
v_maker_reb = maker_rebate
v_exch = taker_fee - maker_rebate
v_maker_net = maker_rebate + maker_spread - maker_adv
v_taker_net = taker_fee + taker_spread - taker_avoid
vals4 = [v_taker_fee, v_maker_reb, v_exch, v_maker_net, v_taker_net]
cols4 = [C["taker"], C["maker"], C["exch"], C["maker"], C["taker"]]
y = np.arange(len(cats))
ax.barh(y, vals4, color=cols4, edgecolor="white")
for i, v in enumerate(vals4):
    ax.text(v + v_taker_fee * 0.01, i, f"${v:,.0f}", va="center", fontsize=11)
ax.set_yticks(y)
ax.set_yticklabels(cats, fontsize=11)
ax.set_xlim(0, v_taker_fee * 1.18)
ax.set_xlabel("美元（全样本累计）", fontsize=11)
ax.set_title("10 年累计资金流向：taker 付费养活 maker 与交易所", fontsize=13, fontweight="bold")
ax.grid(axis="x", color=C["grid"])
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(D, "mt_subsidy_flow.png"), dpi=130)
plt.close()

print("charts saved to", D)
