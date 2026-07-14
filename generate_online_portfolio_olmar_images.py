#!/usr/bin/env python3
"""
为文章「在线投资组合 OLMAR：跟随均值回复的在线再平衡算法」(online-portfolio-olmar)
生成真实配图 + 计算正文引用的所有关键数字。

机制（自洽合成，仅用于演示算法；落地见文末路径）：
  * OLMAR (On-Line Moving Average Reversion, Li & Hoi 2012) 是一类
    在线投资组合选择(OLPS)算法：每天用一个数据驱动的价格相对预测 x̃_{t+1}，
    再把当前组合 b_t 沿「让预测收益 >= ε」的方向做被动-主动更新，投影回单纯形。
  * 预测用移动平均反转(MAR)：x̃_{t+1,i} = MA_w(p)_i / p_{t,i}
    —— 价格低于均线 -> 预测反弹(相对>1) -> 加仓；反之减仓。这就是均值回复。
  * 更新规则(PA 风格闭式解):
        b_{t+1} = b_t + λ_t (x̃ - x̄·1),  λ_t = max(0, (ε - b_t·x̃)/‖x̃-x̄1‖²)
    再投影到单纯形 Δ_m。
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
D = os.path.join(BASE, "online-portfolio-olmar")
os.makedirs(D, exist_ok=True)

C = {"olmar": "#C44E52", "ucrp": "#4C72B0", "bh": "#999999", "best": "#8172B3",
     "grid": "#DDDDDD", "warn": "#DD8452", "calm": "#55A868", "eps": "#8172B3"}

# ----------------------------------------------------------------------------
# 合成市场：均值回复 + 温和趋势 + 一个持续赢家 + 一次崩盘
# ----------------------------------------------------------------------------
rng = np.random.default_rng(20260714)
M = 6            # 资产数
T = 756          # 3 年交易日
WIN = 5          # 移动平均窗口
EPS = 10.0       # 反转强度阈值

def make_market(seed=20260714):
    r = np.random.default_rng(seed)
    # 每只资产：围绕均值回复的 log 价格 (OU) + 一个整体温和漂移
    prices = np.ones((T, M))
    theta = np.array([0.05, 0.05, 0.04, 0.03, 0.02, 0.06])   # 回复速度
    sig = np.array([0.018, 0.020, 0.016, 0.022, 0.014, 0.024])
    drift = np.array([0.0002, 0.0000, 0.0004, -0.0001, 0.0006, 0.0001])
    logp = np.zeros(M)
    center = np.zeros(M)
    for t in range(1, T):
        # 均值回复：偏离 center 越大，回拉越强
        mr = -theta * (logp - center)
        shock = r.normal(0, sig)
        logp = logp + drift + mr + shock
        center = center + drift  # center 缓慢随 drift 漂移
        prices[t] = np.exp(logp)
    # 注入一次系统性崩盘（第 500-515 天），之后反弹 —— 制造反转机会
    prices[500:516] *= np.linspace(1.0, 0.80, 16)[:, None]
    prices[516:540] *= np.linspace(0.80, 0.97, 24)[:, None] / 0.80
    return prices

prices = make_market()
# 价格相对 x_t = p_t / p_{t-1}
X = np.ones((T, M))
X[1:] = prices[1:] / prices[:-1]


def project_simplex(v):
    """把向量投影到单纯形 {b>=0, sum b =1}（Duchi 2008）。"""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    tau = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - tau, 0)


def olmar(X, prices, window=WIN, eps=EPS):
    """OLMAR-1（移动平均反转）。返回每日组合收益与权重历史。"""
    T, M = X.shape
    b = np.ones(M) / M
    rets = np.zeros(T)
    W = np.zeros((T, M))
    for t in range(T):
        W[t] = b
        # 用 b_t 面对当日相对 X[t]（b_t 在昨收后定，今日实现）
        rets[t] = float(b @ X[t]) - 1.0 if t > 0 else 0.0
        # 预测 t+1 的价格相对：MA_w(p_t)/p_t
        if t >= window:
            ma = prices[t - window + 1:t + 1].mean(0)
            x_pred = ma / prices[t]
        else:
            x_pred = np.ones(M)
        # PA 更新
        x_bar = x_pred.mean()
        denom = float(((x_pred - x_bar) ** 2).sum())
        if denom > 1e-12:
            lam = max(0.0, (eps - float(b @ x_pred)) / denom)
        else:
            lam = 0.0
        b = b + lam * (x_pred - x_bar)
        b = project_simplex(b)
    return rets, W


def ucrp(X):
    """均匀恒定再平衡组合（每天再平衡回等权）。"""
    T, M = X.shape
    b = np.ones(M) / M
    rets = (X @ b) - 1.0
    rets[0] = 0.0
    return rets


def buy_hold(prices):
    """等权买入持有。"""
    w0 = np.ones(M) / M
    val = (prices / prices[0]) @ w0
    rets = np.zeros(T)
    rets[1:] = val[1:] / val[:-1] - 1.0
    return rets, val


def stats(rets):
    eq = np.cumprod(1 + rets)
    total = eq[-1] - 1
    ann = eq[-1] ** (252 / len(rets)) - 1
    vol = rets.std(ddof=1) * np.sqrt(252)
    sharpe = (rets.mean() * 252) / vol if vol > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min()
    return dict(total=total, ann=ann, vol=vol, sharpe=sharpe, mdd=mdd, eq=eq)


rets_ol, W_ol = olmar(X, prices)
rets_cr = ucrp(X)
rets_bh, val_bh = buy_hold(prices)
# 事后最优单一资产（best stock, hindsight）
final_rel = prices[-1] / prices[0]
best_idx = int(final_rel.argmax())
rets_best = np.zeros(T)
rets_best[1:] = prices[1:, best_idx] / prices[:-1, best_idx] - 1

s_ol = stats(rets_ol)
s_cr = stats(rets_cr)
s_bh = stats(rets_bh)
s_best = stats(rets_best)

# 换手率（OLMAR）
turn = np.abs(np.diff(W_ol, axis=0)).sum(1)
turn_mean = float(turn.mean())

print("=" * 60)
print("OLMAR 关键数字（canonical seed 20260714）")
print("=" * 60)
for name, s in [("OLMAR", s_ol), ("UCRP等权再平衡", s_cr),
                ("等权买入持有", s_bh), (f"事后最优单资产#{best_idx}", s_best)]:
    print(f"{name:16s} 总收益 {s['total']*100:7.1f}%  年化 {s['ann']*100:6.2f}%  "
          f"Sharpe {s['sharpe']:5.2f}  波动 {s['vol']*100:5.1f}%  MDD {s['mdd']*100:6.1f}%")
print(f"OLMAR 日均换手率: {turn_mean*100:.1f}%")

# ----------------------------------------------------------------------------
# 图 1：净值曲线对比
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.4))
ax.plot(s_ol["eq"], color=C["olmar"], lw=2.2, label=f"OLMAR（总收益 {s_ol['total']*100:.0f}%）")
ax.plot(s_cr["eq"], color=C["ucrp"], lw=1.8, label=f"UCRP 等权再平衡（{s_cr['total']*100:.0f}%）")
ax.plot(s_bh["eq"], color=C["bh"], lw=1.6, label=f"等权买入持有（{s_bh['total']*100:.0f}%）")
ax.plot(s_best["eq"], color=C["best"], lw=1.4, ls="--", label=f"事后最优单资产（{s_best['total']*100:.0f}%）")
ax.axvspan(500, 540, color=C["warn"], alpha=0.12, label="系统性崩盘窗口")
ax.set_xlabel("交易日"); ax.set_ylabel("净值（初始=1）")
ax.set_title("OLMAR vs 基准：均值回复市场里，反转再平衡跑赢等权")
ax.legend(fontsize=8.5); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "olmar_equity.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 图 2：ε 敏感性（反转强度阈值）
# ----------------------------------------------------------------------------
eps_grid = [1, 2, 3, 5, 8, 10, 15, 20, 30]
sp_eps, tot_eps, mdd_eps = [], [], []
for e in eps_grid:
    r, _ = olmar(X, prices, eps=e)
    s = stats(r)
    sp_eps.append(s["sharpe"]); tot_eps.append(s["total"] * 100); mdd_eps.append(s["mdd"] * 100)
fig, ax1 = plt.subplots(figsize=(10, 5.2))
ax1.plot(eps_grid, sp_eps, "o-", color=C["olmar"], lw=2.0, label="Sharpe")
ax1.set_xlabel("反转强度阈值 ε"); ax1.set_ylabel("年化 Sharpe", color=C["olmar"])
ax1.tick_params(axis="y", labelcolor=C["olmar"]); ax1.grid(True, color=C["grid"], lw=0.6)
ax2 = ax1.twinx()
ax2.plot(eps_grid, tot_eps, "s--", color=C["ucrp"], lw=1.8, label="总收益(%)")
ax2.set_ylabel("总收益 (%)", color=C["ucrp"]); ax2.tick_params(axis="y", labelcolor=C["ucrp"])
ax1.set_title("ε 敏感性：阈值太小不动、太大过度激进，中段最优")
fig.tight_layout(); plt.savefig(os.path.join(D, "olmar_eps.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 图 3：窗口 w 敏感性
# ----------------------------------------------------------------------------
win_grid = [2, 3, 5, 8, 10, 15, 20, 30]
sp_w, tot_w = [], []
for w in win_grid:
    r, _ = olmar(X, prices, window=w)
    s = stats(r)
    sp_w.append(s["sharpe"]); tot_w.append(s["total"] * 100)
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(win_grid, sp_w, "o-", color=C["calm"], lw=2.0, label="Sharpe")
ax.set_xlabel("移动平均窗口 w（天）"); ax.set_ylabel("年化 Sharpe")
ax.set_title("窗口敏感性：窗口过短追噪声、过长反转信号钝化")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "olmar_window.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 图 4：权重演化热力图（OLMAR 主动再平衡）
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.0))
im = ax.imshow(W_ol.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
               extent=[0, T, M - 0.5, -0.5])
ax.set_yticks(range(M)); ax.set_yticklabels([f"资产{i+1}" for i in range(M)])
ax.set_xlabel("交易日"); ax.set_title("OLMAR 组合权重演化：崩盘后主动加仓被错杀资产")
ax.axvline(500, color="#333333", lw=1.0, ls="--"); ax.axvline(540, color="#333333", lw=1.0, ls="--")
cbar = fig.colorbar(im, ax=ax); cbar.set_label("权重")
plt.tight_layout(); plt.savefig(os.path.join(D, "olmar_weights.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 鲁棒性：多种子
# ----------------------------------------------------------------------------
ol_sh, cr_sh, win_cnt = [], [], 0
for sd in range(1, 21):
    p = make_market(seed=sd)
    x = np.ones((T, M)); x[1:] = p[1:] / p[:-1]
    r_o, _ = olmar(x, p); r_c = ucrp(x)
    so, sc = stats(r_o)["sharpe"], stats(r_c)["sharpe"]
    ol_sh.append(so); cr_sh.append(sc)
    if so > sc:
        win_cnt += 1
print("-" * 60)
print(f"20 种子: OLMAR Sharpe 均值 {np.mean(ol_sh):.2f}±{np.std(ol_sh):.2f}  "
      f"UCRP {np.mean(cr_sh):.2f}±{np.std(cr_sh):.2f}  OLMAR>UCRP: {win_cnt}/20")
print("done ->", D)
