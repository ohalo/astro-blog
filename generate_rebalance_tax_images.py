#!/usr/bin/env python3
"""
为文章「组合再平衡的税制与换手率摩擦的隐式成本」(rebalance-tax-friction) 生成真实配图。

核心：带「税务 lot 记账 + 换手率摩擦」的多资产组合再平衡模拟。
  - 5 只资产、120 个月（10 年）相关月度收益路径
  - 每种持仓按「批次(lot)」记账成本基础，卖出用指定辨认(specific identification)
  - 短期(持有<12月)资本利得税 35%，长期 15%；亏损可抵减并结转
  - 交易成本 = COST_RATE × 成交名义额（佣金+买卖价差+冲击，合并建模）
对比三种再平衡：
  A. 月度朴素再平衡（FIFO、无税务感知）—— 换手最高、税最重
  B. 5% 阈值带再平衡（FIFO）—— 降低换手
  C. 税务感知再平衡（5% 带 + 亏损 harvesting + 长期优先）—— 最优税后
另加无摩擦月度再平衡作为上界基准。

图表（全部真实数值，非占位）：
  1. equity_curves.png   各类策略净值对比（含无摩擦上界 + 买入持有）
  2. turnover.png        滚动年化换手率：月度 vs 阈值 vs 税务感知
  3. tax_decomposition.png  累计隐性成本拆解：资本利得税 vs 交易成本（按策略）
  4. band_sweep.png      阈值带宽扫描：税后 CAGR 与年化换手（双轴）
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "rebalance-tax-friction")
os.makedirs(D, exist_ok=True)

# ============================================================
# 0) 参数与价格路径
# ============================================================
rng = np.random.default_rng(20260711)
N = 5
M = 120                      # 月份数（10 年）
mu = np.array([0.0090, 0.0110, 0.0075, 0.0130, 0.0085])
vol = np.array([0.045, 0.055, 0.040, 0.070, 0.050])
corr = np.array([
    [1.00, 0.60, 0.50, 0.30, 0.40],
    [0.60, 1.00, 0.55, 0.35, 0.45],
    [0.50, 0.55, 1.00, 0.30, 0.40],
    [0.30, 0.35, 0.30, 1.00, 0.35],
    [0.40, 0.45, 0.40, 0.35, 1.00],
])
cov = (vol[:, None] * vol[None, :]) * corr
L = np.linalg.cholesky(cov)
z = rng.standard_normal((M, N))
ret = mu[None, :] + z @ L.T          # (M,N) 月度收益
price = 100.0 * np.cumprod(1.0 + ret, axis=0)   # 价格路径 (M,N)

SHORT_RATE = 0.35        # 短期资本利得税（近似普通所得税）
LONG_RATE = 0.15         # 长期资本利得税
COST_RATE = 0.0015       # 单边 15bps：含佣金+价差+冲击的合并摩擦
LT_MONTHS = 12           # 持有 >=12 月视为长期


# ============================================================
# 1) 组合类：lot 记账 + 税务感知卖出 + 再平衡
# ============================================================
class Port:
    def __init__(self, p0, V0=1_000_000.0):
        self.shares = np.zeros(N)
        self.lots = [[] for _ in range(N)]     # 每个 lot: [shares, cost, month_bought]
        w = V0 / N
        for i in range(N):
            s = w / p0[i]
            self.shares[i] = s
            self.lots[i].append([s, p0[i], 0])
        self.cum_tax = 0.0
        self.cum_cost = 0.0
        self.loss_carry = 0.0

    def value(self, p):
        return float(np.sum(self.shares * p))

    def sell_lots(self, i, q, p, month, tax_aware=True):
        """卖出资产 i 的 q 股，返回 (短盈,长盈,短亏,长亏)。"""
        lots = self.lots[i]
        if tax_aware:
            def key(k):
                ls, cst, mb = lots[k]
                g = (p - cst) * ls
                if g < 0:
                    return (0, g)                 # 亏损组：先 harvest 最大亏损
                return (1, 1 if (month - mb) >= LT_MONTHS else 0, g)  # 盈利组：长期优先、小盈利优先
            order = sorted(range(len(lots)), key=key)
        else:
            order = list(range(len(lots)))        # FIFO：按建仓顺序
        rem = q
        gs = gl = ls = ll = 0.0
        for k in order:
            if rem <= 1e-9:
                break
            take = min(lots[k][0], rem)
            g = (p - lots[k][1]) * take
            long_h = (month - lots[k][2]) >= LT_MONTHS
            if g >= 0:
                if long_h:
                    gl += g
                else:
                    gs += g
            else:
                if long_h:
                    ll += -g
                else:
                    ls += -g
            lots[k][0] -= take
            rem -= take
        self.lots[i] = [x for x in lots if x[0] > 1e-9]
        return gs, gl, ls, ll

    def pay_tax(self, gs, gl, ls, ll):
        short_net = gs - ls
        long_net = gl - ll
        tax = 0.0
        if short_net > 0:
            tax += short_net * SHORT_RATE
        if long_net > 0:
            tax += long_net * LONG_RATE
        # 超出的亏损结转，未来抵减（简化：统一池）
        if short_net < 0:
            self.loss_carry += -short_net
        if long_net < 0:
            self.loss_carry += -long_net
        if self.loss_carry > 0 and tax > 0:
            use = min(self.loss_carry, tax)
            tax -= use
            self.loss_carry -= use
        self.cum_tax += tax
        return tax

    def rebalance(self, p, month, tax_aware=True):
        V = self.value(p)
        target = V / N
        total_tax = total_cost = traded = 0.0
        for i in range(N):
            cur = self.shares[i] * p[i]
            trade = target - cur                  # <0 卖, >0 买
            if abs(trade) < 1e-6:
                continue
            traded += abs(trade)
            if trade < 0:
                gs, gl, ls, ll = self.sell_lots(i, -trade / p[i], p[i], month, tax_aware)
                tax = self.pay_tax(gs, gl, ls, ll)
                cost = COST_RATE * abs(trade)
                total_tax += tax
                total_cost += cost
                self.shares[i] += trade / p[i]
            else:
                q = trade / p[i]
                self.shares[i] += q
                self.lots[i].append([q, p[i], month])
                total_cost += COST_RATE * trade
        # 隐性成本拖累：把持仓按 (V - 税 - 费)/V 缩放，保持目标权重
        f = (V - total_tax - total_cost) / V
        self.shares *= f
        for i in range(N):
            for lot in self.lots[i]:
                lot[0] *= f
        self.cum_cost += total_cost
        return total_tax, total_cost, traded


def run(mode, band=None, tax_aware=True):
    p = Port(price[0])
    navs = []
    for t in range(M):
        Vcur = p.value(price[t])
        navs.append(Vcur)
        do = False
        if mode == "monthly":
            do = (t > 0)
        elif mode == "band":
            do = (t > 0)
            if do and band is not None:
                w = (p.shares * price[t]) / Vcur
                if np.max(np.abs(w - 1.0 / N)) <= band:
                    do = False
        if do:
            p.rebalance(price[t], t, tax_aware)
    return np.array(navs), p.cum_tax, p.cum_cost


def metrics(nav):
    nav = nav / nav[0]
    rets = nav[1:] / nav[:-1] - 1.0
    cagr = nav[-1] ** (12.0 / len(nav)) - 1.0
    sharpe = rets.mean() / (rets.std() + 1e-12) * np.sqrt(12)
    mdd = (nav / np.maximum.accumulate(nav) - 1.0).min()
    return cagr, sharpe, mdd


# ============================================================
# 2) 跑全部策略
# ============================================================
nav_gross, _, _ = run("monthly", tax_aware=True)   # 无摩擦（税/费置 0 见下）
# 无摩擦上界：临时把税率/成本设 0
SHORT_RATE_S, LONG_RATE_S, COST_RATE_S = SHORT_RATE, LONG_RATE, COST_RATE
SHORT_RATE = LONG_RATE = COST_RATE = 0.0
nav_gross, _, _ = run("monthly")
SHORT_RATE, LONG_RATE, COST_RATE = SHORT_RATE_S, LONG_RATE_S, COST_RATE_S

nav_bh = np.cumprod(1.0 + ret.mean(1)) * 1_000_000.0   # 等权买入持有（无再平衡、无摩擦用作参考）
nav_naive, tax_naive, cost_naive = run("monthly", tax_aware=False)
nav_band, tax_band, cost_band = run("band", band=0.05, tax_aware=False)
nav_tax, tax_tax, cost_tax = run("band", band=0.05, tax_aware=True)

mg, mb, mh = metrics(nav_gross)
na, nb, nh = metrics(nav_naive)
ba, bb, bh = metrics(nav_band)
ta, tb, th = metrics(nav_tax)
print("=== 策略对比（起始 100 万，10 年）===")
print(f"{'策略':<22}{'税后CAGR':>10}{'Sharpe':>9}{'MDD':>9}{'累计税':>12}{'累计费':>12}")
print(f"{'无摩擦月度再平衡':<22}{mg:>9.2%}{mg and mb:>9.2f}{mh:>8.1%}{0.0:>12.0f}{0.0:>12.0f}")
print(f"{'月度朴素(FIFO)':<22}{na:>9.2%}{nb:>9.2f}{nh:>8.1%}{tax_naive:>12,.0f}{cost_naive:>12,.0f}")
print(f"{'5%阈值(FIFO)':<22}{ba:>9.2%}{bb:>9.2f}{bh:>8.1%}{tax_band:>12,.0f}{cost_band:>12,.0f}")
print(f"{'税务感知(5%带)':<22}{ta:>9.2%}{tb:>9.2f}{th:>8.1%}{tax_tax:>12,.0f}{cost_tax:>12,.0f}")
drag_naive = mg - na
drag_band = mg - ba
drag_tax = mg - ta
print(f"\n隐性成本拖累(相对无摩擦上界 CAGR): 月度朴素={drag_naive:.2%}  阈值={drag_band:.2%}  税务感知={drag_tax:.2%}")
print(f"税务感知 vs 月度朴素 的 CAGR 改善: {ta-na:.2%}")
print(f"阈值 vs 月度朴素的换手节省(累计费): {cost_naive-cost_band:,.0f}")

# ============================================================
# 3) 图1：净值曲线
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.8))
x = np.arange(M)
ax.plot(x, nav_gross / 1e6, color="#888888", lw=2.0, ls="--", label=f"无摩擦月度再平衡 (CAGR={mg:.1%})")
ax.plot(x, nav_bh / 1e6, color="#9467bd", lw=1.4, ls=":", label=f"等权买入持有 (CAGR={metrics(nav_bh)[0]:.1%})")
ax.plot(x, nav_naive / 1e6, color="#d62728", lw=1.8, label=f"月度朴素 FIFO (CAGR={na:.1%})")
ax.plot(x, nav_band / 1e6, color="#ff7f0e", lw=1.8, label=f"5% 阈值 FIFO (CAGR={ba:.1%})")
ax.plot(x, nav_tax / 1e6, color="#2ca02c", lw=2.0, label=f"税务感知 5% 带 (CAGR={ta:.1%})")
ax.set_xlabel("月份", fontsize=11)
ax.set_ylabel("净值（百万，起始=1）", fontsize=11)
ax.set_title("再平衡的隐性成本：税 + 换手摩擦如何吞噬复利", fontsize=13, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "equity_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 4) 图2：滚动年化换手率（用再平衡触发密度近似）
# ============================================================
def rebal_flags(mode, band=None):
    p = Port(price[0])
    flags = []
    for t in range(M):
        Vcur = p.value(price[t])
        do = False
        if mode == "monthly":
            do = (t > 0)
        else:
            do = (t > 0)
            if do and band is not None:
                w = (p.shares * price[t]) / Vcur
                if np.max(np.abs(w - 1.0 / N)) <= band:
                    do = False
        flags.append(do)
        if do:
            p.rebalance(price[t], t, tax_aware=(mode == "tax"))
    return np.array(flags)

fl_naive = rebal_flags("monthly")
fl_band = rebal_flags("band", 0.05)
fl_tax = rebal_flags("tax", 0.05)
win = 24
def roll_annual_turn(flags):
    out = np.full(M, np.nan)
    for t in range(win, M):
        out[t] = flags[t - win:t].sum() / (win / 12.0)   # 12月=1年
    return out
ra_n = roll_annual_turn(fl_naive)
ra_b = roll_annual_turn(fl_band)
ra_t = roll_annual_turn(fl_tax)
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.plot(x, ra_n, color="#d62728", lw=1.8, label=f"月度朴素 (年均再平衡 {fl_naive.mean()*12:.1f} 次)")
ax.plot(x, ra_b, color="#ff7f0e", lw=1.8, label=f"5% 阈值 (年均 {fl_band.mean()*12:.1f} 次)")
ax.plot(x, ra_t, color="#2ca02c", lw=1.8, label=f"税务感知 (年均 {fl_tax.mean()*12:.1f} 次)")
ax.set_xlabel("月份", fontsize=11)
ax.set_ylabel("滚动 24 月年化再平衡次数", fontsize=11)
ax.set_title("换手频率：阈值带把再平衡次数砍掉一大半", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "turnover.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 5) 图3：隐性成本拆解（资本利得税 vs 交易成本）
# ============================================================
fig, ax = plt.subplots(figsize=(10.5, 5.6))
strat = ["月度朴素\n(FIFO)", "5%阈值\n(FIFO)", "税务感知\n(5%带)"]
tax_vals = [tax_naive, tax_band, tax_tax]
cost_vals = [cost_naive, cost_band, cost_tax]
xpos = np.arange(len(strat))
ax.bar(xpos, tax_vals, 0.55, label="资本利得税", color="#d62728")
ax.bar(xpos, cost_vals, 0.55, bottom=tax_vals, label="交易成本", color="#ff7f0e")
for i in range(len(strat)):
    tot = tax_vals[i] + cost_vals[i]
    ax.text(i, tot + max(tax_vals) * 0.02, f"{tot/1e6:.2f}M", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(xpos)
ax.set_xticklabels(strat, fontsize=10)
ax.set_ylabel("累计隐性成本（元）", fontsize=11)
ax.set_title("隐性成本拆解：税是主力，阈值带 + 税务感知显著压缩", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9.5)
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "tax_decomposition.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 6) 图4：阈值带宽扫描
# ============================================================
bands = [0.01, 0.03, 0.05, 0.10, 0.20]
cagr_sweep, turn_sweep = [], []
for b in bands:
    nv, tx, cs = run("band", band=b, tax_aware=True)
    cagr_sweep.append(metrics(nv)[0])
    # 年化换手：用再平衡次数近似
    p = Port(price[0]); cnt = 0
    for t in range(1, M):
        Vcur = p.value(price[t])
        w = (p.shares * price[t]) / Vcur
        if np.max(np.abs(w - 1.0 / N)) > b:
            p.rebalance(price[t], t, tax_aware=True); cnt += 1
    turn_sweep.append(cnt / (M / 12.0))
fig, ax1 = plt.subplots(figsize=(10.5, 5.6))
ax1.plot([b * 100 for b in bands], [c * 100 for c in cagr_sweep], "o-", color="#2ca02c", lw=2, label="税后 CAGR")
ax1.set_xlabel("再平衡阈值带宽 (%)", fontsize=11)
ax1.set_ylabel("税后年化 CAGR (%)", fontsize=11, color="#2ca02c")
ax1.tick_params(axis="y", labelcolor="#2ca02c")
ax2 = ax1.twinx()
ax2.plot([b * 100 for b in bands], turn_sweep, "s--", color="#1f77b4", lw=2, label="年化再平衡次数")
ax2.set_ylabel("年化再平衡次数", fontsize=11, color="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#1f77b4")
ax1.set_title("带宽扫描：阈值越宽，换手越低、税后收益越高（存在拐点）", fontsize=12.5, fontweight="bold")
ax1.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "band_sweep.png"), dpi=150, bbox_inches="tight")
plt.close()

print("\n✅ 配图生成完成：", sorted(os.listdir(D)))
