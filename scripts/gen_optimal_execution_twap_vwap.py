import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)
OUT = "public/images/optimal-execution-twap-vwap"
os.makedirs(OUT, exist_ok=True)

BLUE = "#3b6ea5"; RED = "#c0392b"; GREEN = "#27ae60"; ORANGE = "#e67e22"; PURPLE = "#8e44ad"; GRAY = "#7f8c8d"

# ============================================================
# 交易日日内 U 型成交量曲线（分钟级），后面 VWAP/TWAP 都基于它
# ============================================================
M = 390  # 一个交易日的分钟数（美股 6.5 小时）
t = np.arange(M)
# U 型：开盘、收盘放量，午间清淡
u_shape = 1.0 + 1.6 * np.exp(-t / 40.0) + 1.9 * np.exp(-(M - 1 - t) / 45.0)
u_shape += 0.15 * rng.standard_normal(M)  # 微噪声
u_shape = np.clip(u_shape, 0.2, None)
volume_profile = u_shape / u_shape.sum()  # 归一化成占比

# ---------- 图 1：日内成交量 U 型曲线 + TWAP vs VWAP 拆单 ----------
Q = 100000.0  # 总下单量（股）
twap_slices = np.full(M, Q / M)          # TWAP：均匀切
vwap_slices = Q * volume_profile         # VWAP：按成交量占比切

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].fill_between(t, volume_profile * 100, color=BLUE, alpha=0.30)
ax[0].plot(t, volume_profile * 100, color=BLUE, lw=1.6)
ax[0].set_title("日内成交量分布：典型 U 型", fontsize=12)
ax[0].set_xlabel("交易分钟（开盘→收盘）"); ax[0].set_ylabel("成交量占比 (%)")
ax[0].grid(alpha=0.25)

ax[1].plot(t, twap_slices, color=ORANGE, lw=1.8, label="TWAP 拆单（均匀）")
ax[1].plot(t, vwap_slices, color=GREEN, lw=1.8, label="VWAP 拆单（跟量）")
ax[1].set_title("同样 10 万股，两种拆法", fontsize=12)
ax[1].set_xlabel("交易分钟"); ax[1].set_ylabel("单分钟下单量（股）")
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/volume_profile_slicing.png", dpi=110); plt.close()

# ============================================================
# 蒙特卡洛：模拟大单执行，比较 TWAP / VWAP / 一次性冲击
# 价格 = 随机游走 + 永久冲击 + 临时冲击
# ============================================================
MKT_DAY_VOL = Q * 6.0                     # 假设我们的单约占市场日成交量的 1/6
MKT_VOL = volume_profile * MKT_DAY_VOL    # 每分钟市场成交量（股）
ETA = 55.0    # 临时冲击系数(bp)：本笔成交价 ∝ 参与率（线性冲击，成交后消散）
GAMMA = 0.9   # 永久冲击系数(bp)：推动中间价，不消散
SIG = 2.0     # 每分钟随机游走波动(bp)

def simulate_execution(slices, n_paths=4000, seed=0):
    """给定拆单向量 slices（每分钟下单量），返回每条路径相对到达价的实现滑点(bp)。
    临时冲击：本笔成交价偏移 = ETA * 参与率(q_i/v_i)，线性冲击→成本对切片量凸，
    所以把单拆细、并按流动性分配（VWAP）能显著降低总冲击（Cauchy-Schwarz）。"""
    r = np.random.default_rng(seed)
    total = slices.sum()
    partic = slices / MKT_VOL          # 每分钟参与率
    slip = np.zeros(n_paths)
    for p in range(n_paths):
        mid = 0.0                      # 中间价相对到达价的偏移(bp)
        cost = 0.0
        for i in range(M):
            if slices[i] > 0:
                perm = GAMMA * partic[i]          # 永久冲击
                temp = ETA * partic[i]            # 临时冲击（线性于参与率）
                fill_price = mid + perm * 0.5 + temp
                cost += fill_price * slices[i]
                mid += perm                       # 永久冲击留存推动中间价
            mid += SIG * r.standard_normal()      # 随机游走
        slip[p] = cost / total
    return slip

slip_twap = simulate_execution(twap_slices, seed=11)
slip_vwap = simulate_execution(vwap_slices, seed=11)
# 激进执行：把整单集中在开盘前 30 分钟摧完（接近实盘的“抢流动性”打法）
agg = np.zeros(M); agg[:30] = Q / 30.0
slip_agg = simulate_execution(agg, seed=11)

# ---------- 图 2：三种执行方式的滑点分布 ----------
fig, ax = plt.subplots(figsize=(9.5, 4.6))
bins = np.linspace(min(slip_vwap.min(), slip_twap.min()) - 2,
                   np.percentile(slip_agg, 99) + 5, 70)
ax.hist(slip_agg, bins=bins, color=RED, alpha=0.55, label=f"一次性冲击  均值 {slip_agg.mean():.1f}bp")
ax.hist(slip_twap, bins=bins, color=ORANGE, alpha=0.60, label=f"TWAP  均值 {slip_twap.mean():.1f}bp")
ax.hist(slip_vwap, bins=bins, color=GREEN, alpha=0.60, label=f"VWAP  均值 {slip_vwap.mean():.1f}bp")
ax.axvline(slip_agg.mean(), color=RED, ls="--", lw=1.2)
ax.axvline(slip_twap.mean(), color=ORANGE, ls="--", lw=1.2)
ax.axvline(slip_vwap.mean(), color=GREEN, ls="--", lw=1.2)
ax.set_title("4000 条路径：执行滑点分布（相对到达价，越小越好）", fontsize=12)
ax.set_xlabel("实现滑点 (bp)"); ax.set_ylabel("路径数")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/slippage_distribution.png", dpi=110); plt.close()

# ============================================================
# Almgren-Chriss 有效前沿：执行成本 vs 时序风险
# ============================================================
# 简化 AC 模型：线性永久+临时冲击，风险来自持仓暴露于波动
X = Q                       # 总量
T_days = 1.0
Nsteps = 20
tau = T_days / Nsteps
sigma = 0.02               # 日波动
eta_ac = 2.5e-6            # 临时冲击系数
gamma_ac = 2.5e-7         # 永久冲击系数

def ac_trajectory(kappa):
    """给定风险厌恶参数隐含的 kappa，返回持仓轨迹"""
    if kappa < 1e-9:
        # 线性清仓（TWAP）
        hold = X * (1 - np.arange(Nsteps + 1) / Nsteps)
    else:
        j = np.arange(Nsteps + 1)
        hold = X * np.sinh(kappa * (Nsteps - j) * tau) / np.sinh(kappa * Nsteps * tau)
    return hold

def cost_and_risk(hold):
    trades = -np.diff(hold)                 # 每步卖出量
    # 临时冲击成本
    temp_cost = np.sum(eta_ac / tau * trades**2)
    # 永久冲击成本
    perm_cost = 0.5 * gamma_ac * X**2
    exp_cost = temp_cost + perm_cost
    # 风险：持仓方差
    var = sigma**2 * tau * np.sum(hold[1:]**2)
    return exp_cost, np.sqrt(var)

kappas = np.concatenate([[0.0], np.linspace(0.5, 12, 40)])
costs, risks = [], []
for k in kappas:
    c, r_ = cost_and_risk(ac_trajectory(k))
    costs.append(c); risks.append(r_)
costs = np.array(costs); risks = np.array(risks)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
# 左：几条代表性持仓轨迹
for k, col, lab in [(0.0, ORANGE, "κ=0  匀速(TWAP)"),
                    (3.0, BLUE, "κ=3  中等风险厌恶"),
                    (9.0, RED, "κ=9  高风险厌恶(快清)")]:
    hold = ac_trajectory(k)
    ax[0].plot(np.arange(Nsteps + 1) * tau, hold / X * 100, color=col, lw=2, marker="o", ms=3, label=lab)
ax[0].set_title("Almgren-Chriss 最优清仓轨迹", fontsize=12)
ax[0].set_xlabel("时间（交易日）"); ax[0].set_ylabel("剩余持仓 (%)")
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.25)

# 右：有效前沿
ax[1].plot(risks * 1e-3, costs * 1e4, color=PURPLE, lw=2)
ax[1].scatter([risks[0] * 1e-3], [costs[0] * 1e4], color=ORANGE, zorder=5, s=60, label="TWAP（成本低/风险高）")
ax[1].scatter([risks[-1] * 1e-3], [costs[-1] * 1e4], color=RED, zorder=5, s=60, label="快清（成本高/风险低）")
ax[1].set_title("执行的有效前沿：成本 vs 时序风险", fontsize=12)
ax[1].set_xlabel("时序风险（持仓暴露，任意单位）"); ax[1].set_ylabel("预期执行成本（任意单位）")
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/almgren_chriss_frontier.png", dpi=110); plt.close()

# ============================================================
# VWAP 跟踪误差：预测成交量 vs 实际成交量偏差导致跑输 VWAP 基准
# ============================================================
n_days = 500
# 每天：真实成交量曲线 = 历史平均 U 型 + 当日随机扰动
track_err_static = np.zeros(n_days)  # 用静态历史曲线拆单
track_err_dynamic = np.zeros(n_days) # 用当日实时更新（理想化）
for d in range(n_days):
    shock = 1.0 + 0.35 * rng.standard_normal(M)   # 当日成交量整体偏移
    intraday_noise = 1.0 + 0.25 * rng.standard_normal(M)
    actual_vol = np.clip(volume_profile * shock * intraday_noise, 1e-6, None)
    actual_vol /= actual_vol.sum()
    # 市场 VWAP（基准）
    price_path = np.cumsum(0.02 * rng.standard_normal(M))
    mkt_vwap = np.sum(price_path * actual_vol)
    # 静态拆单：用历史 volume_profile
    my_vwap_static = np.sum(price_path * volume_profile)
    track_err_static[d] = (my_vwap_static - mkt_vwap) * 100  # bp 量级
    # 动态：假设能观测到实际（下界），误差更小
    track_err_dynamic[d] = (np.sum(price_path * (0.7*volume_profile+0.3*actual_vol)) - mkt_vwap) * 100

fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.hist(track_err_static, bins=45, color=RED, alpha=0.55,
        label=f"静态历史曲线  std={track_err_static.std():.2f}")
ax.hist(track_err_dynamic, bins=45, color=GREEN, alpha=0.55,
        label=f"部分动态更新  std={track_err_dynamic.std():.2f}")
ax.axvline(0, color=GRAY, ls="--", lw=1)
ax.set_title("VWAP 跟踪误差：预测成交量的偏差直接变成跑赢/跑输基准", fontsize=12)
ax.set_xlabel("相对市场 VWAP 的偏离"); ax.set_ylabel("交易日数")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/vwap_tracking_error.png", dpi=110); plt.close()

print("done. mean slippage bp -> agg:%.2f twap:%.2f vwap:%.2f" %
      (slip_agg.mean(), slip_twap.mean(), slip_vwap.mean()))
print("track err std static:%.3f dynamic:%.3f" % (track_err_static.std(), track_err_dynamic.std()))
