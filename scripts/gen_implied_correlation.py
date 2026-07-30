import numpy as np, os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(31415)
OUT = "public/images/implied-correlation-dispersion"
os.makedirs(OUT, exist_ok=True)
BLUE="#3b6ea5"; RED="#c0392b"; GREEN="#27ae60"; ORANGE="#e67e22"; GRAY="#7f8c8d"; PURPLE="#8e44ad"
R = {}

# =====================================================================
# 0. 恒等式演示: sigma_idx^2 = sum wi^2 si^2 + sum_{i!=j} wi wj si sj rho
#    -> 隐含相关性 rho_imp 由指数波动率与成分股波动率反解
# =====================================================================
def implied_corr(w, sig, sig_idx):
    var_idx = sig_idx**2
    diag = np.sum(w**2 * sig**2)
    cross = np.sum(np.outer(w, sig*w) * np.outer(sig, np.ones_like(sig))) - diag
    # cross = sum_{i!=j} wi wj si sj
    return (var_idx - diag) / cross

n = 30
w = np.ones(n)/n
sig = np.clip(rng.normal(0.30, 0.06, n), 0.15, 0.50)

# 校验: 给定真实 rho 造出指数波动率, 再反解, 应完全还原
rho_true = 0.35
diag = np.sum(w**2 * sig**2)
cross = (np.sum(w*sig))**2 - diag
sig_idx_true = np.sqrt(diag + rho_true*cross)
R["roundtrip_rho"] = float(implied_corr(w, sig, sig_idx_true))
R["roundtrip_err"] = float(abs(R["roundtrip_rho"] - rho_true))
R["sig_idx_true"] = float(sig_idx_true)
R["avg_single_vol"] = float(sig.mean())
R["n_names"] = n

# 图1: 指数波动率 vs 隐含相关性的映射关系
rhos = np.linspace(0.05, 0.95, 100)
idxvols = np.sqrt(diag + rhos*cross)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
ax = axes[0]
ax.plot(rhos, idxvols*100, color=BLUE, lw=2.2)
ax.axhline(sig.mean()*100, color=GRAY, ls="--", lw=1.2)
ax.text(0.06, sig.mean()*100+0.4, f"成分股平均波动 {sig.mean()*100:.1f}%", fontsize=9, color=GRAY)
ax.scatter([rho_true],[sig_idx_true*100], color=RED, zorder=5, s=55)
ax.annotate(f"ρ={rho_true}\n指数波动 {sig_idx_true*100:.1f}%",
            (rho_true, sig_idx_true*100), textcoords="offset points", xytext=(18,-28),
            fontsize=9, color=RED)
ax.set_xlabel("平均相关系数 ρ"); ax.set_ylabel("指数年化波动率 (%)")
ax.set_title("指数波动率完全由 ρ 决定（成分股波动固定）")
ax.grid(alpha=0.25)

# 分散化收益
ax = axes[1]
div_benefit = (sig.mean() - idxvols)*100
ax.plot(rhos, div_benefit, color=GREEN, lw=2.2)
ax.fill_between(rhos, 0, div_benefit, color=GREEN, alpha=0.15)
ax.axhline(0, color=GRAY, lw=1)
ax.set_xlabel("平均相关系数 ρ"); ax.set_ylabel("分散化收益 (波动率百分点)")
ax.set_title("离散度交易做多的就是这块『分散化收益』")
ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/implied-corr-identity.png", dpi=120); plt.close()

# =====================================================================
# 1. 相关性风险溢价: 隐含 rho 系统性高于已实现 rho
#    模拟 1000 天, 隐含 rho 报价 = 真实 rho + 溢价 + 噪声
# =====================================================================
T = 1000
# 真实相关性: 均值回复 + 危机跳升
rho_real = np.zeros(T); rho_real[0] = 0.30
crisis = np.zeros(T, dtype=bool)
c_start = [230, 640]; c_len = [55, 40]
for s, L in zip(c_start, c_len): crisis[s:s+L] = True
for t in range(1, T):
    target = 0.72 if crisis[t] else 0.28
    rho_real[t] = rho_real[t-1] + 0.06*(target - rho_real[t-1]) + 0.022*rng.standard_normal()
rho_real = np.clip(rho_real, 0.02, 0.95)

# 关键：隐含相关性只能用 t 时刻的信息（近期已实现 + 溢价），不能偷看未来危机。
# 第一版直接用 rho_real[t] + 溢价报价，等于让期权市场提前知道危机，Sharpe 算出 46。
H = 21                                            # 期权到期（交易日）
premium = 0.085                                   # 相关性风险溢价
rho_bwd = np.array([rho_real[max(0,i-H):i+1].mean() for i in range(T)])   # 近期已实现
rho_imp = np.clip(rho_bwd + premium + 0.030*rng.standard_normal(T), 0.03, 0.97)

# 前瞻已实现相关性：持有期内真正兑现的那个 rho
rho_fwd = np.array([rho_real[i:min(T,i+H)].mean() for i in range(T)])
valid = np.arange(T) < T - H                      # 最后 H 天未到期，不计入

R["horizon_days"] = H
R["mean_rho_imp"] = float(rho_imp.mean())
R["mean_rho_real"] = float(rho_real.mean())
R["mean_spread"] = float((rho_imp - rho_fwd)[valid].mean())
R["pct_imp_above"] = float((rho_imp > rho_fwd)[valid].mean()*100)
R["crisis_days"] = int(crisis.sum())
R["spread_normal"] = float((rho_imp-rho_fwd)[valid & ~crisis].mean())
R["spread_crisis"] = float((rho_imp-rho_fwd)[valid & crisis].mean())

fig, ax = plt.subplots(figsize=(10.5, 4.5))
t = np.arange(T)
ax.plot(t, rho_imp, color=RED, lw=1.3, label=f"隐含相关性（均值 {rho_imp.mean():.3f}）")
ax.plot(t, rho_fwd, color=BLUE, lw=1.3, label=f"前瞻已实现相关性（均值 {rho_fwd.mean():.3f}）")
for s, L in zip(c_start, c_len):
    ax.axvspan(s, s+L, color=RED, alpha=0.10)
ax.text(c_start[0]+L/2, 0.93, "危机期", color=RED, fontsize=9, ha="center")
ax.set_xlabel("交易日"); ax.set_ylabel("平均相关系数")
ax.set_title(f"相关性风险溢价：隐含高于前瞻已实现（{R['pct_imp_above']:.1f}% 的交易日）")
ax.legend(loc="lower right"); ax.grid(alpha=0.25); ax.set_ylim(0, 1.0)
plt.tight_layout(); plt.savefig(f"{OUT}/correlation-risk-premium.png", dpi=120); plt.close()

# =====================================================================
# 2. 离散度策略损益: 卖指数方差 / 买成分股方差
#    近似: PnL ~ (rho_imp - rho_real) * cross_term  (vega 归一)
# =====================================================================
notional = 1.0
# 每 H 天开一笔不重叠的新交易（重叠持仓会让 sqrt(252) 年化严重高估 Sharpe）
entries = np.arange(0, T-H, H)
# 篮子复制误差：个股期权无法完美复制指数方差（离散对冲/跳空/vol-of-vol）
replication_noise = 0.022 * rng.standard_normal(len(entries))
trade_pnl = ((rho_imp[entries] - rho_fwd[entries]) + replication_noise) * cross * notional * 100

per_year = 252 / H
sharpe = trade_pnl.mean()/trade_pnl.std()*np.sqrt(per_year)
equity = np.concatenate([[0.0], np.cumsum(trade_pnl)])
dd = equity - np.maximum.accumulate(equity)
crisis_tr = crisis[entries]

R["n_trades"] = int(len(entries))
R["disp_sharpe"] = float(sharpe)
R["disp_total"] = float(equity[-1])
R["disp_maxdd"] = float(dd.min())
R["disp_winrate"] = float((trade_pnl>0).mean()*100)
R["worst_trade"] = float(trade_pnl.min())
R["best_trade"] = float(trade_pnl.max())
R["avg_win"] = float(trade_pnl[trade_pnl>0].mean())
R["avg_loss"] = float(trade_pnl[trade_pnl<0].mean())
R["win_loss_ratio"] = float(abs(trade_pnl[trade_pnl>0].mean()/trade_pnl[trade_pnl<0].mean()))
R["crisis_pnl"] = float(trade_pnl[crisis_tr].sum())
R["normal_pnl"] = float(trade_pnl[~crisis_tr].sum())
R["n_crisis_trades"] = int(crisis_tr.sum())
R["crisis_share_days"] = float(crisis.mean()*100)
R["skew"] = float(((trade_pnl-trade_pnl.mean())**3).mean()/trade_pnl.std()**3)
# 扣除执行成本后的净损益：n=30 篮子，每笔交易开平仓各一次
#   成本口径与第 3 节一致，换算到同一 vega 归一单位
cost_per_trade_n30 = (8.0 + 30*55.0/30.0) / 1e4 * cross * 100 * 2
net_pnl_30 = trade_pnl - cost_per_trade_n30
R["cost_per_trade_n30"] = float(cost_per_trade_n30)
R["net_sharpe_n30"] = float(net_pnl_30.mean()/net_pnl_30.std()*np.sqrt(per_year))
R["net_total_n30"] = float(net_pnl_30.sum())
R["net_winrate_n30"] = float((net_pnl_30>0).mean()*100)
cost_per_trade_n100 = (8.0 + 100*55.0/30.0) / 1e4 * cross * 100 * 2
net_pnl_100 = trade_pnl - cost_per_trade_n100
R["net_sharpe_n100"] = float(net_pnl_100.mean()/net_pnl_100.std()*np.sqrt(per_year))
R["net_total_n100"] = float(net_pnl_100.sum())
R["net_winrate_n100"] = float((net_pnl_100>0).mean()*100)

srt = np.sort(trade_pnl); k = max(1, int(0.10*len(trade_pnl)))
R["worst10pct_sum"] = float(srt[:k].sum())
R["worst10pct_vs_gains"] = float(abs(srt[:k].sum())/srt[k:].sum()*100)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
ax = axes[0]
ax.plot(np.concatenate([[0], entries+H]), equity, color=GREEN, lw=1.9, marker="o", ms=3)
for s_, L_ in zip(c_start, c_len):
    ax.axvspan(s_, s_+L_, color=RED, alpha=0.12)
ax.set_xlabel("交易日"); ax.set_ylabel("累计损益（vega 归一单位）")
ax.set_title(f"离散度策略净值：Sharpe {sharpe:.2f}，最大回撤 {dd.min():.2f}")
ax.grid(alpha=0.25)

ax = axes[1]
bins = np.linspace(trade_pnl.min()*1.05, trade_pnl.max()*1.05, 26)
ax.hist(trade_pnl[~crisis_tr], bins=bins, color=BLUE, alpha=0.7, label="平常期开仓")
ax.hist(trade_pnl[crisis_tr], bins=bins, color=RED, alpha=0.8, label="危机期开仓")
ax.axvline(0, color=GRAY, lw=1.2)
ax.set_xlabel(f"单笔交易损益（{H} 交易日持有）"); ax.set_ylabel("笔数")
ax.set_title(f"捡钢镚分布：胜率 {R['disp_winrate']:.0f}%，偏度 {R['skew']:.2f}")
ax.legend(); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/dispersion-pnl.png", dpi=120); plt.close()

# =====================================================================
# 3. 成分股数量与执行成本: 卖1个指数期权 vs 买 N 个成分股期权
# =====================================================================
# 单位统一为「每单位 vega 名义的年化收益（bp）」
ns = np.array([5, 10, 20, 30, 50, 100, 200, 500])
# 毛边际: 溢价换算成指数方差口径后的年化 bp，与 n 无关
gross_edge = R["mean_spread"] * cross / sig_idx_true**2 * 1e4 / 2
# 成本: 指数腿 1 条 + 个股腿 n 条；单腿成本占 vega 名义的 bp，个股期权价差远宽于指数
#   关键: n 条腿共同构成一份篮子 vega，单腿 vega 随 1/n 下降，但每腿的固定成本不降，
#   且小额单在宽价差市场上滑点占比更高 -> 成本随 n 近似线性上升
leg_bp_index, leg_bp_single = 8.0, 55.0
rebal_per_year = 12
total_cost = (leg_bp_index + ns * leg_bp_single / 30.0) * rebal_per_year / 12
net_edge = gross_edge - total_cost

R["gross_edge_annual"] = float(gross_edge)
for q in (10, 30, 100, 500):
    R[f"cost_n{q}"] = float(total_cost[ns==q][0])
    R[f"net_n{q}"] = float(net_edge[ns==q][0])
R["breakeven_n"] = int(ns[net_edge>0].max()) if (net_edge>0).any() else -1

fig, ax = plt.subplots(figsize=(9.5, 4.5))
ax.plot(ns, np.full_like(ns, gross_edge, dtype=float), color=GREEN, lw=2, ls="--", label="毛边际（与 n 无关）")
ax.plot(ns, total_cost, color=RED, lw=2.2, marker="o", ms=4, label="执行成本（腿数驱动）")
ax.plot(ns, net_edge, color=BLUE, lw=2.4, marker="s", ms=4, label="净边际")
ax.axhline(0, color=GRAY, lw=1)
ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns)
ax.set_xlabel("复制篮子的成分股数量 n"); ax.set_ylabel("年化边际（bp of vega 名义）")
ax.set_title("离散度交易的规模诅咒：腿数越多，边际被成本吃光")
ax.legend(); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/dispersion-cost-scaling.png", dpi=120); plt.close()

with open("scripts/_impcorr_results.json","w") as f: json.dump(R, f, indent=2, ensure_ascii=False)
for k,v in R.items(): print(k,"=",v)
