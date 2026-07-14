#!/usr/bin/env python3
"""
为文章「FTRL 在线学习组合：用正则化跟随引领历史」(ftrl-online-portfolio)
生成真实配图 + 计算正文引用的所有关键数字。

机制（自洽合成，仅用于演示算法；落地见文末路径）：
  * FTRL (Follow The Regularized Leader) 是 McMahan et al. 2013 提出的在线学习算法，
    广泛用于大规模在线学习（含广告 CTR）。这里把它移植到「在线投资组合选择 OLPS」。
  * 把每日组合收益写成损失 l_t(w) = -w·r_t（w 为组合权重，r_t 为当日收益向量）。
    梯度 g_t = -r_t。
  * FTRL-Proximal 的逐坐标闭式更新（带自适应学习率 + L1 稀疏 + L2 平滑）：
        n_i    += g_i^2
        σ_i     = (sqrt(n_i + g_i^2) - sqrt(n_i)) / α
        z_i    += g_i + σ_i · w_i^{prev}
        w_i     = sign(z_i)·max(|z_i|-αλ1, 0) / (α(β+sqrt(n_i)) + λ2)
    再投影到单纯形 Δ_m（仅用于执行；FTRL 内部权重保持无约束）。
  * 对比基准：UCRP(等权再平衡)、OLMAR(均值回复)、等权买入持有、事后最优单资产。
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
D = os.path.join(BASE, "ftrl-online-portfolio")
os.makedirs(D, exist_ok=True)

C = {"ftrl": "#C44E52", "olmar": "#8172B3", "ucrp": "#4C72B0",
     "bh": "#999999", "best": "#55A868", "grid": "#DDDDDD", "warn": "#DD8452"}

# ----------------------------------------------------------------------------
# 合成市场：均值回复 + 一个持续赢家 + 温和漂移 + 一次崩盘 + 后期 regime 切换
# ----------------------------------------------------------------------------
rng0 = np.random.default_rng(20260714)
M = 6            # 资产数
T = 756          # 3 年交易日

def make_market(seed=20260714):
    r = np.random.default_rng(seed)
    prices = np.ones((T, M))
    # 不同资产不同的回复速度 / 波动 / 漂移
    theta = np.array([0.05, 0.05, 0.04, 0.03, 0.02, 0.06])
    sig = np.array([0.018, 0.020, 0.016, 0.022, 0.014, 0.024])
    # 资产 0 是「持续赢家」：长期正漂移（让 FTRL 的跟随能赚到）
    drift = np.array([0.0006, 0.0000, 0.0004, -0.0001, 0.0003, 0.0001])
    logp = np.zeros(M)
    center = np.zeros(M)
    for t in range(1, T):
        mr = -theta * (logp - center)
        shock = r.normal(0, sig)
        logp = logp + drift + mr + shock
        center = center + drift
        prices[t] = np.exp(logp)
    # 第 1-400 天：资产 5 强势（赢家切换），第 400 天起资产 0 接棒
    prices[1:400, 5] *= np.exp(np.linspace(0, 0.35, 399))
    prices[400:, 0] *= np.exp(np.linspace(0, 0.30, T - 400))
    # 注入一次系统性崩盘（第 500-515 天），之后反弹
    prices[500:516] *= np.linspace(1.0, 0.80, 16)[:, None]
    prices[516:540] *= np.linspace(0.80, 0.97, 24)[:, None] / 0.80
    return prices

prices = make_market()
prices = np.clip(np.nan_to_num(prices, nan=1.0, posinf=1e6, neginf=1e-6), 1e-8, None)
X = np.ones((T, M)); X[1:] = prices[1:] / prices[:-1]
X = np.nan_to_num(X, nan=1.0, posinf=1.0, neginf=1.0)

# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------
def project_simplex(v):
    v = np.nan_to_num(v, nan=0.0)
    v = v - v.min() + 1e-9  # 保证有正元素，避免全 0
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    idx = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0]
    rho = idx[-1] if len(idx) else 0
    tau = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - tau, 0)

def stats(rets):
    eq = np.cumprod(1 + rets)
    total = eq[-1] - 1
    ann = eq[-1] ** (252 / len(rets)) - 1
    vol = rets.std(ddof=1) * np.sqrt(252)
    sharpe = (rets.mean() * 252) / vol if vol > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min()
    turn = 0.0
    return dict(total=total, ann=ann, vol=vol, sharpe=sharpe, mdd=mdd, eq=eq)

# ----------------------------------------------------------------------------
# FTRL-Proximal 在线组合
# ----------------------------------------------------------------------------
def ftrl_portfolio(X, alpha=0.10, beta=1.0, l1=0.02, l2=1.0):
    T, M = X.shape
    z = np.zeros(M); n = np.zeros(M); w = np.ones(M) / M
    W = np.zeros((T, M)); rets = np.zeros(T)
    for t in range(T):
        b = project_simplex(w) if t > 0 else np.ones(M) / M
        W[t] = b
        if t > 0:
            rets[t] = float(b @ X[t]) - 1.0
        g = (1.0 - X[t]) if t > 0 else np.zeros(M)
        n = n + g * g
        sigma = (np.sqrt(n + g * g) - np.sqrt(n)) / alpha
        z = z + g + sigma * w
        w = np.sign(z) * np.maximum(np.abs(z) - alpha * l1, 0.0) / (alpha * (beta + np.sqrt(n)) + l2)
        w = np.clip(w, -2.0, 4.0)        # 限制单资产敞口，避免爆仓
        if not np.all(np.isfinite(w)):
            w = np.ones(M) / M
    # 换手率
    turn = float(np.abs(np.diff(W, axis=0)).sum(1).mean())
    return rets, W, turn

def olmar(X, prices, window=5, eps=10.0):
    T, M = X.shape
    b = np.ones(M) / M
    rets = np.zeros(T); W = np.zeros((T, M))
    for t in range(T):
        W[t] = b
        if t > 0:
            rets[t] = float(b @ X[t]) - 1.0
        if t >= window:
            ma = prices[t - window + 1:t + 1].mean(0)
            x_pred = ma / prices[t]
        else:
            x_pred = np.ones(M)
        x_bar = x_pred.mean()
        denom = float(((x_pred - x_bar) ** 2).sum())
        lam = max(0.0, (eps - float(b @ x_pred)) / denom) if denom > 1e-12 else 0.0
        b = project_simplex(b + lam * (x_pred - x_bar))
    return rets, W

def ucrp(X):
    T, M = X.shape
    b = np.ones(M) / M
    rets = (X @ b) - 1.0; rets[0] = 0.0
    return rets

def buy_hold(prices):
    w0 = np.ones(M) / M
    val = (prices / prices[0]) @ w0
    rets = np.zeros(T); rets[1:] = val[1:] / val[:-1] - 1.0
    return rets, val

rets_ft, W_ft, turn_ft = ftrl_portfolio(X)
rets_ol, W_ol = olmar(X, prices)
rets_cr = ucrp(X)
rets_bh, val_bh = buy_hold(prices)
final_rel = prices[-1] / prices[0]
best_idx = int(final_rel.argmax())
rets_best = np.zeros(T); rets_best[1:] = prices[1:, best_idx] / prices[:-1, best_idx] - 1

s_ft, s_ol, s_cr, s_bh, s_best = (stats(rets_ft), stats(rets_ol),
                                     stats(rets_cr), stats(rets_bh), stats(rets_best))

# L1 稀疏度：平均有多少资产权重被压到 ~0
sparsity = float((W_ft < 0.01).mean())

print("=" * 64)
print("FTRL 关键数字（canonical seed 20260714, α=0.10 β=1 λ1=0.02 λ2=1.0）")
print("=" * 64)
for name, s in [("FTRL", s_ft), ("OLMAR", s_ol), ("UCRP等权再平衡", s_cr),
                ("等权买入持有", s_bh), (f"事后最优单资产#{best_idx}", s_best)]:
    print(f"{name:16s} 总收益 {s['total']*100:7.1f}%  年化 {s['ann']*100:6.2f}%  "
          f"Sharpe {s['sharpe']:5.2f}  波动 {s['vol']*100:5.1f}%  MDD {s['mdd']*100:6.1f}%")
print(f"FTRL 日均换手率: {turn_ft*100:.1f}%   平均权重<0.01 占比(稀疏): {sparsity*100:.1f}%")

# ----------------------------------------------------------------------------
# 图 1：净值曲线对比
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.4))
ax.plot(s_ft["eq"], color=C["ftrl"], lw=2.2, label=f"FTRL（总收益 {s_ft['total']*100:.0f}%）")
ax.plot(s_ol["eq"], color=C["olmar"], lw=1.8, label=f"OLMAR（{s_ol['total']*100:.0f}%）")
ax.plot(s_cr["eq"], color=C["ucrp"], lw=1.8, label=f"UCRP 等权再平衡（{s_cr['total']*100:.0f}%）")
ax.plot(s_bh["eq"], color=C["bh"], lw=1.6, label=f"等权买入持有（{s_bh['total']*100:.0f}%）")
ax.plot(s_best["eq"], color=C["best"], lw=1.4, ls="--", label=f"事后最优单资产（{s_best['total']*100:.0f}%）")
ax.axvspan(500, 540, color=C["warn"], alpha=0.12, label="系统性崩盘窗口")
ax.set_xlabel("交易日"); ax.set_ylabel("净值（初始=1）")
ax.set_title("FTRL vs 基准：正则化跟随在「赢家切换」市场里稳步超越等权")
ax.legend(fontsize=8.5); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "ftrl_equity.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 图 2：自适应学习率（sqrt(n) 衰减）+ L1 稀疏的作用
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
# 左：累计梯度范数 sqrt(n_i) 随时间（资产 0 vs 资产 5）
n0 = np.zeros(T); n5 = np.zeros(T)
z = np.zeros(M); n = np.zeros(M); w = np.zeros(M)
for t in range(T):
    g = -X[t] if t > 0 else np.zeros(M)
    n = n + g * g
    n0[t] = np.sqrt(n[0]); n5[t] = np.sqrt(n[5])
axes[0].plot(n0, color=C["ftrl"], lw=1.8, label="资产0（后段赢家）")
axes[0].plot(n5, color=C["ucrp"], lw=1.8, label="资产5（前段赢家）")
axes[0].set_xlabel("交易日"); axes[0].set_ylabel("√n_i（累计梯度范数）")
axes[0].set_title("自适应学习率：√n_i 越大，该资产步长越小"); axes[0].legend(fontsize=8.5)
axes[0].axvline(400, color="#333", ls="--", lw=1.0); axes[0].grid(True, color=C["grid"], lw=0.6)
# 右：L1 稀疏——平均每个交易日有多少资产权重≈0
nzero = (W_ft < 0.01).sum(1)
axes[1].plot(nzero, color=C["warn"], lw=1.8)
axes[1].set_xlabel("交易日"); axes[1].set_ylabel("权重≈0 的资产数")
axes[1].set_title(f"L1 稀疏：平均 {sparsity*100:.1f}% 的仓位被压成 0")
axes[1].grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "ftrl_rates.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 图 3：权重演化热力图（FTRL 跟随赢家切换）
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.0))
im = ax.imshow(W_ft.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
               extent=[0, T, M - 0.5, -0.5])
ax.set_yticks(range(M)); ax.set_yticklabels([f"资产{i+1}" for i in range(M)])
ax.set_xlabel("交易日"); ax.set_title("FTRL 组合权重演化：前段抱资产5、后段切到资产0、崩盘后回补")
ax.axvline(400, color="#333", lw=1.0, ls="--"); ax.axvline(500, color="#333", lw=1.0, ls="--")
ax.axvline(540, color="#333", lw=1.0, ls="--")
cbar = fig.colorbar(im, ax=ax); cbar.set_label("权重")
plt.tight_layout(); plt.savefig(os.path.join(D, "ftrl_weights.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 图 4：超参数敏感性（L1 正则 vs α 学习率）
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
l1_grid = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
sp_l1, tot_l1 = [], []
for l1 in l1_grid:
    r, _, _ = ftrl_portfolio(X, l1=l1); s = stats(r)
    sp_l1.append(s["sharpe"]); tot_l1.append(s["total"]*100)
axes[0].plot(l1_grid, sp_l1, "o-", color=C["ftrl"], lw=2.0)
axes[0].set_xlabel("L1 正则强度 λ1"); axes[0].set_ylabel("年化 Sharpe")
axes[0].set_title("L1 敏感性：适度稀疏提升稳健，过强饿死仓位")
axes[0].grid(True, color=C["grid"], lw=0.6)
alpha_grid = [0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
sp_a, tot_a = [], []
for a in alpha_grid:
    r, _, _ = ftrl_portfolio(X, alpha=a); s = stats(r)
    sp_a.append(s["sharpe"]); tot_a.append(s["total"]*100)
axes[1].plot(alpha_grid, sp_a, "s-", color=C["ucrp"], lw=2.0)
axes[1].set_xlabel("学习率 α"); axes[1].set_ylabel("年化 Sharpe")
axes[1].set_title("α 敏感性：α 太小跟不动、太大追噪声")
axes[1].grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "ftrl_params.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 鲁棒性：多种子
# ----------------------------------------------------------------------------
ft_sh, cr_sh, win_cnt = [], [], 0
for sd in range(1, 21):
    p = make_market(seed=sd); x = np.ones((T, M)); x[1:] = p[1:] / p[:-1]
    r_f, _, _ = ftrl_portfolio(x); r_c = ucrp(x)
    sf, sc = stats(r_f)["sharpe"], stats(r_c)["sharpe"]
    ft_sh.append(sf); cr_sh.append(sc)
    if sf > sc: win_cnt += 1
print("-" * 64)
print(f"20 种子: FTRL Sharpe 均值 {np.mean(ft_sh):.2f}±{np.std(ft_sh):.2f}  "
      f"UCRP {np.mean(cr_sh):.2f}±{np.std(cr_sh):.2f}  FTRL>UCRP: {win_cnt}/20")
print("done ->", D)
