import numpy as np, os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(20260730)
OUT = "public/images/pfof-order-flow-payment"
os.makedirs(OUT, exist_ok=True)

BLUE = "#3b6ea5"; RED = "#c0392b"; GREEN = "#27ae60"; ORANGE = "#e67e22"; GRAY = "#7f8c8d"; PURPLE = "#8e44ad"

R = {}

# ============================================================
# 1. 订单簿与执行价格模型
#    交易所(lit): 在 NBBO 买卖价成交, 付交易所费
#    内化商(wholesaler): 在 NBBO 内部改善 x bp, 但把 PFOF 返给券商
# ============================================================
N = 200_000                      # 散户订单数
S0 = 50.0                        # 参考中间价
spread_bp = rng.gamma(2.0, 2.5, N) + 1.0     # 半价差(bp), 右偏
spread_bp = np.clip(spread_bp, 1.0, 40.0)
half_spread = S0 * spread_bp / 1e4           # 元/股
qty = rng.choice([100, 200, 300, 500, 1000], N, p=[.42, .23, .15, .13, .07])
side = rng.choice([1, -1], N)                # +1 买, -1 卖

# 路由1 交易所 taker: 全额付半价差 + taker fee 0.30 美分/股
EX_TAKER_FEE = 0.0030
cost_lit = half_spread * qty + EX_TAKER_FEE * qty

# 路由2 内化商: 价格改善率随价差宽度上升(宽价差里油水多), 但有上限
pi_rate = np.clip(0.12 + 0.016 * spread_bp, 0.10, 0.42)   # 改善占半价差比例
price_improve = half_spread * pi_rate
cost_wholesale = (half_spread - price_improve) * qty       # 内化无交易所费

# 路由3 暗池中点: 成交则零价差，但只有部分概率成交；
#   未成交需后回交易所，等待期间中点漂移产生延迟成本
fill_p = np.clip(0.45 - 0.010 * spread_bp, 0.12, 0.45)      # 价差越宽中点流动性越差
filled_mid = rng.random(N) < fill_p
delay_cost = np.abs(rng.normal(0, S0 * 3.5 / 1e4, N)) * qty  # 等待期间中点漂移
cost_midpoint = np.where(filled_mid, 0.0, cost_lit + delay_cost)

best = np.argmin(np.vstack([cost_lit, cost_wholesale, cost_midpoint]), axis=0)

R["n_orders"] = N
R["avg_half_spread_bp"] = float(spread_bp.mean())
R["avg_pi_rate"] = float(pi_rate.mean())
R["pi_per_share_cent"] = float(price_improve.mean() * 100)
R["lit_cost_total"] = float(cost_lit.sum())
R["wholesale_cost_total"] = float(cost_wholesale.sum())
R["midpoint_cost_total"] = float(cost_midpoint.sum())
R["saving_total"] = float(cost_lit.sum() - cost_wholesale.sum())
R["saving_per_order"] = float((cost_lit - cost_wholesale).mean())
R["pct_orders_better"] = float((cost_wholesale < cost_lit).mean() * 100)
R["mid_fill_rate"] = float(filled_mid.mean() * 100)
R["best_lit_pct"] = float((best == 0).mean() * 100)
R["best_wholesale_pct"] = float((best == 1).mean() * 100)
R["best_mid_pct"] = float((best == 2).mean() * 100)
R["oracle_cost"] = float(np.minimum(np.minimum(cost_lit, cost_wholesale), cost_midpoint).sum())
R["wholesale_vs_oracle_gap"] = float(R["wholesale_cost_total"] - R["oracle_cost"])

# ============================================================
# 2. PFOF 分配: 内化商赚到的价差留存, 按比例返给券商
# ============================================================
captured = (half_spread - price_improve) * qty            # 内化商毛收入
pfof_rate = 0.30                                          # 返给券商 30%
pfof = captured * pfof_rate
wholesaler_net = captured - pfof

R["wholesaler_gross"] = float(captured.sum())
R["pfof_to_broker"] = float(pfof.sum())
R["wholesaler_net"] = float(wholesaler_net.sum())
R["pfof_per_100sh_cent"] = float((pfof / qty * 100).mean() * 100)
R["client_saving_vs_pfof"] = float(R["saving_total"] / R["pfof_to_broker"])

# ---- 图1: 三方蛋糕分配 ----
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax = axes[0]
labels = ["客户价格改善", "券商 PFOF 收入", "内化商净留存"]
vals = [R["saving_total"], R["pfof_to_broker"], R["wholesaler_net"]]
cols = [GREEN, ORANGE, BLUE]
bars = ax.bar(labels, vals, color=cols, width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v, f"${v/1e3:,.0f}k",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("总额（美元，20 万笔订单）")
ax.set_title("零佣金蛋糕：价差被三方瓜分")
ax.grid(axis="y", alpha=0.25)

ax = axes[1]
names = ["交易所 taker", "内化商", "暗池中点"]
totals = [R["lit_cost_total"], R["wholesale_cost_total"], R["midpoint_cost_total"]]
shares = [R["best_lit_pct"], R["best_wholesale_pct"], R["best_mid_pct"]]
b2 = ax.bar(names, totals, color=[RED, GREEN, PURPLE], width=0.55, alpha=0.85)
for b, v, s in zip(b2, totals, shares):
    ax.text(b.get_x()+b.get_width()/2, v, f"${v/1e3:,.0f}k\n事后最优占 {s:.0f}%",
            ha="center", va="bottom", fontsize=9)
ax.axhline(R["oracle_cost"], color=GRAY, ls="--", lw=1.5)
ax.text(2.42, R["oracle_cost"], "先知最优", color=GRAY, fontsize=9, va="bottom", ha="right")
ax.set_ylabel("客户总交易成本（美元）")
ax.set_ylim(0, max(totals)*1.30)
ax.set_title("三条路由的总成本：内化胜在均值，不胜在每一笔")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/pfof-pie-split.png", dpi=120); plt.close()

# ============================================================
# 3. 逆向选择: 内化商挑单(cream-skimming)
#    知情订单 -> 后续价格朝订单方向漂移 -> 内化商拒绝内化, 甩回交易所
# ============================================================
informed = rng.random(N) < 0.08                          # 8% 知情单
# 订单后 5 分钟的中间价漂移(bp), 知情单有系统性方向漂移
#   知情单漂移幅度必须显著超过半价差，否则内化商根本不在乎逆向选择
drift = np.where(informed,
                 side * rng.gamma(3.0, 9.0, N),          # 知情: 顺向漂移均值 ~27bp
                 side * rng.normal(0, 2.0, N))           # 噪音: 零均值
# 内化商对知情单的识别能力(不完美)
detect = informed & (rng.random(N) < 0.62)
routed_lit = detect                                       # 被甩回交易所

R["informed_share"] = float(informed.mean() * 100)
R["detect_rate"] = float(detect.sum() / informed.sum() * 100)
R["drift_informed_bp"] = float((drift[informed] * side[informed]).mean())
R["drift_noise_bp"] = float((drift[~informed] * side[~informed]).mean())
R["internalized_share"] = float((~routed_lit).mean() * 100)

# 内化商在保留订单上的实际盈亏 = 留存价差 - 逆向选择损失
# 注意：必须把漂移投影到订单自身方向上（drift 已含 side，再乘 side 得顺向幅度），
# 否则求和时买单卖单的符号互相抵消，逆向选择损失会被算成接近零
signed_drift = drift * side                               # 顺向漂移（bp）
adverse = S0 * signed_drift / 1e4 * qty                   # 中间价顺向漂移 = 内化商亏损
pnl_keep = captured[~routed_lit] - adverse[~routed_lit]
pnl_all = captured - adverse                              # 若全盘接单
R["pnl_selective"] = float(pnl_keep.sum())
R["pnl_indiscriminate"] = float(pnl_all.sum())
R["adverse_avoided"] = float(adverse[routed_lit].sum())
R["pnl_informed_kept"] = float((captured - adverse)[informed & ~routed_lit].sum())
R["pnl_noise"] = float((captured - adverse)[~informed].sum())
R["adverse_per_informed"] = float(adverse[informed].mean())
R["capture_per_order"] = float(captured.mean())

# ---- 图2: 逆向选择与挑单 ----
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax = axes[0]
sd_inf = signed_drift[informed]
sd_noi = signed_drift[~informed]
bins = np.linspace(-15, 30, 70)
ax.hist(sd_noi, bins=bins, color=BLUE, alpha=0.6, density=True, label=f"噪音单 均值 {R['drift_noise_bp']:.2f}bp")
ax.hist(sd_inf, bins=bins, color=RED, alpha=0.65, density=True, label=f"知情单 均值 {R['drift_informed_bp']:.2f}bp")
ax.axvline(0, color=GRAY, lw=1)
ax.set_xlabel("成交后 5 分钟中间价顺向漂移（bp）"); ax.set_ylabel("密度")
ax.set_title("知情订单的『毒性』：成交后价格继续朝你方向走")
ax.legend(fontsize=9); ax.grid(alpha=0.25)

ax = axes[1]
labs = ["无差别全接", "识别后挑单"]
vs = [R["pnl_indiscriminate"], R["pnl_selective"]]
bars = ax.bar(labs, vs, color=[RED, GREEN], width=0.5)
for b, v in zip(bars, vs):
    ax.text(b.get_x()+b.get_width()/2, v, f"${v/1e3:,.1f}k", ha="center",
            va="bottom" if v > 0 else "top", fontsize=11, fontweight="bold")
ax.axhline(0, color=GRAY, lw=1)
ax.set_ylabel("内化商毛利（美元）")
ax.set_title("挑单是内化商的生存条件，不是可选项")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/adverse-selection.png", dpi=120); plt.close()

# ============================================================
# 4. 券商目标函数冲突: 最大化 PFOF vs 最大化执行质量
#    内化商竞价: 出价高的拿更多单流, 但出价高意味着留给客户的改善少
# ============================================================
grid = np.linspace(0.05, 0.55, 26)      # 券商要求的 PFOF 分成比例
client_pi, broker_rev = [], []
base_capture = captured.sum()
for g in grid:
    # 内化商在总留存里先付 PFOF, 剩下的部分与客户分享(竞争压力下按剩余的固定比例让利)
    pfof_g = base_capture * g
    left = base_capture * (1 - g)
    improve_g = left * 0.35                        # 竞争让利系数
    client_pi.append(improve_g)
    broker_rev.append(pfof_g)
client_pi = np.array(client_pi); broker_rev = np.array(broker_rev)
total_welfare = client_pi + broker_rev

R["pfof_grid_lo"] = float(grid[0]); R["pfof_grid_hi"] = float(grid[-1])
R["client_pi_at_lo"] = float(client_pi[0]); R["client_pi_at_hi"] = float(client_pi[-1])
R["broker_rev_at_lo"] = float(broker_rev[0]); R["broker_rev_at_hi"] = float(broker_rev[-1])
R["client_pi_drop_pct"] = float((1 - client_pi[-1]/client_pi[0]) * 100)
R["broker_rev_gain_x"] = float(broker_rev[-1]/broker_rev[0])

fig, ax = plt.subplots(figsize=(9.2, 4.6))
ax.plot(grid*100, client_pi/1e3, color=GREEN, lw=2.2, marker="o", ms=3.5, label="客户获得的价格改善")
ax.plot(grid*100, broker_rev/1e3, color=ORANGE, lw=2.2, marker="s", ms=3.5, label="券商 PFOF 收入")
ax.plot(grid*100, total_welfare/1e3, color=GRAY, lw=1.5, ls="--", label="两者之和")
ix = int(np.argmin(np.abs(grid - 0.30)))
ax.axvline(30, color=PURPLE, ls=":", lw=1.6)
ax.text(30.6, ax.get_ylim()[1]*0.72, "行业典型\n分成 ~30%", color=PURPLE, fontsize=9)
ax.set_xlabel("券商索取的 PFOF 分成比例（%）"); ax.set_ylabel("金额（千美元）")
ax.set_title("券商的利益冲突：同一块价差，给客户的和给自己的此消彼长")
ax.legend(); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/broker-conflict.png", dpi=120); plt.close()

# ============================================================
# 5. 执行质量对不同交易者的分层影响 (Rule 605 风格分组)
# ============================================================
groups = [("100-499股", (qty >= 100) & (qty < 500)),
          ("500-1999股", (qty >= 500) & (qty < 2000)),
          ("窄价差(<5bp)", spread_bp < 5),
          ("宽价差(>15bp)", spread_bp > 15)]
tbl = []
for name, m in groups:
    eff_lit = (half_spread[m] * 2 / S0 * 1e4).mean()                 # 有效价差 bp
    eff_wh = ((half_spread[m]-price_improve[m]) * 2 / S0 * 1e4).mean()
    tbl.append((name, int(m.sum()), eff_lit, eff_wh, (1-eff_wh/eff_lit)*100))
R["quality_table"] = [[t[0], t[1], round(t[2],2), round(t[3],2), round(t[4],1)] for t in tbl]

fig, ax = plt.subplots(figsize=(9.2, 4.4))
x = np.arange(len(tbl)); w = 0.36
ax.bar(x-w/2, [t[2] for t in tbl], w, color=RED, alpha=0.8, label="交易所有效价差")
ax.bar(x+w/2, [t[3] for t in tbl], w, color=GREEN, alpha=0.85, label="内化后有效价差")
for i, t in enumerate(tbl):
    ax.text(i, max(t[2], t[3])*1.02, f"-{t[4]:.1f}%", ha="center", fontsize=9.5, fontweight="bold", color=BLUE)
ax.set_xticks(x); ax.set_xticklabels([t[0] for t in tbl])
ax.set_ylabel("有效价差（bp）")
ax.set_title("价格改善的分布极不均匀：宽价差订单获益最多")
ax.legend(); ax.grid(axis="y", alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/exec-quality-groups.png", dpi=120); plt.close()

with open("scripts/_pfof_results.json", "w") as f:
    json.dump(R, f, indent=2, ensure_ascii=False)
for k, v in R.items():
    print(k, "=", v)
