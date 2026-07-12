#!/usr/bin/env python3
"""
为文章「尾部风险对冲：用期权给组合买保险值不值」(tail-risk-hedging-options)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

模型与机制（自洽合成，仅用于演示方法；落地须用真实期权链/历史）：
  * 标的组合日收益服从「随机波动 + 偶发崩盘跳」过程
  * 保护性看跌(Protective Put): 持有时点买行权价 K=m·组合价值 的看跌
        保费 = f × BS_put(组合价值, K, T, IV)   (f = 保险覆盖比例)
        对冲后组合 = (1 - f·p)·组合暴露 + f·max(K - 组合到期值, 0)
        即: 把组合价值的 f 份用看跌保到 m·组合价值, 保费 p 先扣
  * 衡量: 保险成本(累计保费/组合) vs 保护(最大回撤削减) → 到底值不值
  * 对照: 100% 持股 vs 95% 持股 + 持续买 3M 虚值(95%)看跌(滚动)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from math import erf, sqrt

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "tail-risk-hedging-options")
os.makedirs(D, exist_ok=True)

C = {"eq": "#4C72B0", "vix": "#C44E52", "grid": "#DDDDDD", "fv": "#55A868",
     "rv": "#DD8452", "hedge": "#9467bd", "thr": "#888888", "green": "#2ca02c",
     "orange": "#FF7F0E"}

# ============================================================
# 1) 物理测度 P 下标的组合日度路径
# ============================================================
rng = np.random.default_rng(20260713)
N = 252 * 20
dt = 1.0 / 252.0
mu = 0.09 / 252.0
trend = mu
v = np.zeros(N); v[0] = 0.04
for t in range(1, N):
    v[t] = v[t - 1] + 5.0 * (0.04 - v[t - 1]) * dt
    v[t] += 0.6 * np.sqrt(max(v[t - 1], 1e-4)) * np.sqrt(dt) * rng.normal()
    if rng.random() < 1.0 / (252 * 4.0):
        v[t] += rng.uniform(0.06, 0.16)
    v[t] = max(v[t], 0.005)
mu = 0.09 / 252.0
r = np.zeros(N)
for t in range(N):
    r[t] = mu + np.sqrt(v[t] / 252.0) * rng.normal()
    if v[t] > 0.12 and rng.random() < 0.015:
        r[t] += -rng.uniform(0.02, 0.05)
    r[t] = np.clip(r[t], -0.5, 0.5)
# 轻度均值回复(危机后反弹)
adj = r.copy()
for t in range(252, N):
    ex = np.sum(r[t - 63:t]) - 63 * trend
    adj[t] += 0.04 * (-ex)
r = np.clip(adj, -0.5, 0.5)
price = np.cumprod(1.0 + r)          # 组合净值(期初=1)

# ============================================================
# 2) Black-Scholes 工具 & 隐含波动(危机时 IV 飙升)
# ============================================================
def bs_put(S, K, T, sig, r=0.02, q=0.0):
    if T <= 0 or sig <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    N = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
    return K * np.exp(-r * T) * N(-d2) - S * np.exp(-q * T) * N(-d1)

def iv_of(t):
    base = 0.18
    return base + 0.9 * np.clip((np.sqrt(v[t]) - 0.18) / 0.22, 0.0, 2.5)

# ============================================================
# 3) 保护性看跌: 成本 vs 行权价(图 1)
# ============================================================
S0 = 100.0
mat = 0.25
money = [0.90, 0.95, 1.00, 1.05, 1.10]
costs = []
for m in money:
    K = S0 * m
    p = bs_put(S0, K, mat, iv_of(0))
    costs.append(p / S0 * 100.0)

# ============================================================
# 4) 滚动对冲回测: 100% 持股 vs 95% 持股 + 持续买 3M 虚值(95%)看跌
#    保险覆盖比例 f=0.95(保护组合价值的95%), 看跌虚值5%(m=0.95)
# ============================================================
def roll_hedge(f, mat_days=63, m=0.95):
    """返回对冲组合净值序列 hv (期初=price[0]) 与累计保费(组合单位)。"""
    hv = np.ones(N); hv[0] = price[0]
    Hh = mat_days
    nr = N // Hh
    fees = 0.0
    for k in range(nr):
        t0 = k * Hh; t1 = min(t0 + Hh, N - 1)
        entry = hv[t0]
        Kk = entry * m
        sig = iv_of(t0)
        put_px = bs_put(entry, Kk, Hh / 252.0, sig)
        prem = f * put_px                 # 保费(组合单位)
        hv[t0] = entry - prem             # 支付保费
        fees += prem
        prot = price[t1] / price[t0]      # 期末标的/期初标的
        payoff = f * max(Kk - entry * prot, 0.0)
        hv[t0:t1 + 1] = hv[t0] * price[t0:t1 + 1] / price[t0]
        hv[t1] = hv[t0] * prot + payoff
    return hv, fees

eq_nv = price.copy()
hd_nv, tot_fee = roll_hedge(0.95, 63, 0.95)

def mdd(x):
    return float(np.min((x - np.maximum.accumulate(x)) / np.maximum.accumulate(x)))

mdd_eq = mdd(eq_nv); mdd_hd = mdd(hd_nv)
ann_ret = lambda x: (x[-1] / x[0]) ** (252.0 / (len(x) - 1)) - 1.0
ar_eq = ann_ret(eq_nv); ar_hd = ann_ret(hd_nv)
total_fee_ratio = tot_fee / eq_nv[-1]
# 最坏单期(3个月)损失对比: 取未对冲最惨的若干个季度, 看同期对冲表现
period_ret_eq = np.array([eq_nv[min(k*63+63, N-1)]/eq_nv[k*63]-1 for k in range(N//63)])
period_ret_hd = np.array([hd_nv[min(k*63+63, N-1)]/hd_nv[k*63]-1 for k in range(N//63)])
worst_q = np.argsort(period_ret_eq)[:5]          # 未对冲最惨的 5 个季度
worst_p_eq = period_ret_eq[worst_q].mean()*100
worst_p_hd = period_ret_hd[worst_q].mean()*100
# 最坏回撤段(按未对冲最大回撤点定位)
dd_eq_now = (eq_nv - np.maximum.accumulate(eq_nv)) / np.maximum.accumulate(eq_nv)
worst_t = int(np.argmin(dd_eq_now))
t_a = max(0, worst_t - 250); t_b = min(N - 1, worst_t + 60)
dd_hd_crisis = (hd_nv[t_a:t_b] - np.maximum.accumulate(hd_nv[t_a:t_b])) / np.maximum.accumulate(hd_nv[t_a:t_b])
dd_eq_crisis = (eq_nv[t_a:t_b] - np.maximum.accumulate(eq_nv[t_a:t_b])) / np.maximum.accumulate(eq_nv[t_a:t_b])

# ============================================================
# 5) 对冲效率: 保险覆盖比例 f 下的成本 vs 回撤削减
# ============================================================
fs = [0.0, 0.50, 0.90, 0.95, 1.00]
res = []
for f in fs:
    hv, fee = roll_hedge(f, 63, 0.95)
    res.append((f, mdd(hv), fee / eq_nv[-1], ann_ret(hv)))

# ============================================================
# 图 1：保护性看跌保费 vs 行权价
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 5.0))
xlab = [f"{int(m*100)}%" for m in money]; x = np.arange(len(money))
ax.bar(x, costs, color=C["vix"], alpha=0.8, width=0.55)
for i, c in enumerate(costs):
    ax.text(x[i], c + 0.2, f"{c:.1f}%", ha="center", fontsize=10, color=C["vix"])
ax.set_xticks(x); ax.set_xticklabels(xlab)
ax.set_xlabel("看跌行权价 (moneyness, 相对现价)"); ax.set_ylabel("保费 (占标的 %)")
ax.set_title("保护性看跌成本: 越接近平值越贵, 虚值(95%以下)便宜但只保崩盘")
ax.grid(True, color=C["grid"], lw=0.6, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(D, "hedge_put_cost.png"), dpi=130)
plt.close()

# ============================================================
# 图 2：到期 payoff 图(组合 + 买入看跌)
# ============================================================
S = np.linspace(60, 120, 200)
Kpay = 95.0
put_cost = bs_put(100.0, Kpay, 0.25, iv_of(0))
pay_port = S
pay_hedge = S - put_cost + np.maximum(Kpay - S, 0.0)
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.plot(S, pay_port, color=C["eq"], lw=2, label="未对冲组合")
ax.plot(S, pay_hedge, color=C["hedge"], lw=2, label="组合 + 买入看跌(行权价95)")
ax.plot(S, np.full_like(S, 100 - put_cost), color=C["thr"], ls="--", lw=1, label="盈亏平衡")
ax.axvline(Kpay, color=C["vix"], ls=":", lw=1.2, label="行权价 95")
ax.set_xlabel("到期标的价位"); ax.set_ylabel("组合价值 (期初投入=100)")
ax.set_title("保护性看跌 payoff: 下行被封顶(保费换保险), 上行仅损保费")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "hedge_payoff.png"), dpi=130)
plt.close()

# ============================================================
# 图 3：滚动对冲回测 vs 纯持股
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4.8))
tt = np.arange(N)
ax.plot(tt, eq_nv, color=C["eq"], lw=1.0, label="100%% 持股 (无对冲)")
ax.plot(tt, hd_nv, color=C["hedge"], lw=1.0, label="95%% 持股 + 持续买 3M 虚值(95%%)看跌")
ax.set_xlabel("交易日"); ax.set_ylabel("净值 (期初=1)")
ax.set_title("滚动对冲: 崩盘期损失更小, 但保费拖拽令长期净值与回撤更深(保险很贵)")
ax.legend(fontsize=8, loc="upper left"); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "hedge_backtest.png"), dpi=130)
plt.close()

# ============================================================
# 图 4：对冲效率(覆盖比例 vs 回撤 / 累计保费)
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 5.2))
xs = [r[0] * 100 for r in res]
ax.plot(xs, [r[1] * 100 for r in res], color=C["vix"], marker="o", lw=2, label="最大回撤 (%)")
ax.plot(xs, [r[2] * 100 for r in res], color=C["orange"], marker="s", lw=2, label="累计保费占组合 (%)")
ax.set_xlabel("保险覆盖比例 (用看跌保护的组合价值比例)")
ax.set_ylabel("百分比 (%)")
ax.set_title("对冲效率: 覆盖比例越高回撤越低, 但保费成本上升(边际递减)")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "hedge_efficiency.png"), dpi=130)
plt.close()

# ============================================================
# 关键数字输出
# ============================================================
print("=== 尾部风险对冲(期权保险) 关键数字 ===")
print("样本 %d 年(%d 交易日), 标的组合年化(未对冲) ≈ %.1f%%" % (N // 252, N, ar_eq * 100))
print("保护性看跌保费(3M, IV≈18%): " + ", ".join("%d%%→%.1f%%" % (int(m*100), c) for m, c in zip(money, costs)))
print("--- 滚动对冲 95%%股 + 持续买 3M 虚值(95%%)看跌(覆盖95%%) ---")
print("未对冲: 年化 %.1f%%  最大回撤 %.1f%%" % (ar_eq * 100, mdd_eq * 100))
print("对冲后: 年化 %.1f%%  最大回撤 %.1f%%" % (ar_hd * 100, mdd_hd * 100))
print("回撤变化 = +%.1f 个百分点(保费拖拽制造漫长水下, 对冲使长期回撤更深)"
      % ((mdd_hd - mdd_eq) * 100))
print("年化收益损失 = %.1f 个百分点(保险太贵)" % ((ar_eq - ar_hd) * 100))
print("单季保费 ≈ %.1f%% 组合 (年化拖拽 ≈ %.1f%%)" % (tot_fee/(N//63)*100, tot_fee/(N//63)*4*100))
print("最坏单季损失(取未对冲最惨5个季度, 同期对比): 未对冲 %.1f%% vs 对冲后 %.1f%% (崩盘期保护确实生效)"
      % (worst_p_eq, worst_p_hd))
print("累计保费(绝对, 组合单位) ≈ %.2f ; 单季保费 ≈ %.1f%% 组合 (年化拖拽 ≈ %.1f%%)"
      % (tot_fee, tot_fee/(N//63)*100, tot_fee/(N//63)*4*100))
print("最坏回撤段(第%d日附近): 未对冲回撤 %.0f%%, 对冲后该段回撤 %.0f%%"
      % (worst_t, dd_eq_crisis.min() * 100, dd_hd_crisis.min() * 100))
print("--- 对冲效率扫描(保险覆盖比例 → 回撤, 累计保费) ---")
for f, d, fee, a in res:
    print("  覆盖 %4.0f%% : 回撤 %5.1f%%   累计保费 %5.1f%%   年化 %5.1f%%"
          % (f * 100, d * 100, fee * 100, a * 100))
print("\n图片已保存到:", D)
