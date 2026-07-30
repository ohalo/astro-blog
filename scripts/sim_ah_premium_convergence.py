#!/usr/bin/env python3
"""AH溢价收敛交易模拟：双重上市股票 A/H 价差的均值回归与可交易性
生成 4 张配图到 public/images/ah-premium-convergence/
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

OUT = Path(__file__).resolve().parent.parent / "public/images/ah-premium-convergence"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- 市场构造 ----------------
N = 60          # 双重上市对数
T = 3000        # ~12年日频
DT = 1/252

# 共同溢价因子（市场层面的 AH 溢价水位，缓慢均值回归 + 大幅波动）
kappa_c = 0.004          # 极慢回归（半衰期 ~173 天）
theta_c = np.log(1.35)   # 长期溢价水位 35%
sig_c = 0.006
c = np.zeros(T)
c[0] = np.log(1.25)
for t in range(1, T):
    c[t] = c[t-1] + kappa_c*(theta_c - c[t-1]) + sig_c*rng.standard_normal()

# 每对的个体长期溢价 mu_i：横截面差异大，且缓慢漂移（随机游走）
mu0 = rng.normal(0.0, 0.20, N)          # 对数溢价的个体成分
mu_drift = np.zeros((T, N))
mu_drift[0] = mu0
sig_mu = 0.0025                          # 长期均值本身在漂移！
for t in range(1, T):
    mu_drift[t] = mu_drift[t-1] + sig_mu*rng.standard_normal(N)

# 个体 AR(1) 偏离（真正可交易的均值回归成分）
kappa_i = 0.035                          # 半衰期 ~20 天
sig_i = 0.010
x = np.zeros((T, N))
for t in range(1, T):
    x[t] = x[t-1]*(1-kappa_i) + sig_i*rng.standard_normal(N)

log_prem = c[:, None] + mu_drift + x     # 对数溢价 log(P_A / (P_H * fx))
prem = np.exp(log_prem) - 1

# 用 H 股与 A 股各自的价格路径实现该溢价：
# 共同基本面 v_t + 溢价在两条腿之间的分摊
mkt = np.cumsum(rng.normal(0.0002, 0.011, T))            # 基本面/市场
idio = np.cumsum(rng.normal(0, 0.014, (T, N)), axis=0)   # 个股基本面
# 溢价收敛的分摊：alpha_A 的比例由 A 腿完成（A 下跌），其余由 H 腿完成（H 上涨）
share_A = 0.65
logH = mkt[:, None] + idio - (1-share_A)*log_prem
logA = logH + log_prem
retA = np.diff(logA, axis=0)
retH = np.diff(logH, axis=0)

# ---------------- 策略 ----------------
LOOKBACK = 250
COST = 0.0015    # 单边 15bp（含港股印花税）

def zscore_signal(lp, lookback):
    """滚动 z 分数（只用历史）"""
    z = np.full(lp.shape, np.nan)
    for t in range(lookback, lp.shape[0]):
        w = lp[t-lookback:t]
        z[t] = (lp[t] - w.mean(axis=0)) / (w.std(axis=0) + 1e-12)
    return z

z_roll = zscore_signal(log_prem, LOOKBACK)

# 对照：全样本均值 z（look-ahead 陷阱）
z_full = (log_prem - log_prem.mean(axis=0)) / log_prem.std(axis=0)

def run_strategy(z, mode, threshold=1.5):
    """mode: LS=多H空A(理论), LH=只多H, HF=多H+空指数期货"""
    pos = np.zeros(N)
    pnl = np.zeros(T-1)
    turnover = 0.0
    for t in range(LOOKBACK, T-1):
        zt = z[t]
        tgt = np.where(zt > threshold, 1.0, np.where(zt < 0.3, 0.0, np.nan))
        new = np.where(np.isnan(tgt), pos, tgt)
        n_active = max(new.sum(), 1)
        w = new / n_active
        w_old = pos / max(pos.sum(), 1)
        turnover += np.abs(w - w_old).sum()
        if mode == "LS":
            step = (w * (retH[t] - retA[t])).sum()
        elif mode == "LH":
            step = (w * retH[t]).sum()
        else:  # HF
            step = (w * (retH[t] - (mkt[t+1]-mkt[t]))).sum()
        cost = np.abs(w - w_old).sum() * COST
        pnl[t] = step - cost
        pos = new
    return pnl

pnl_ls = run_strategy(z_roll, "LS")
pnl_lh = run_strategy(z_roll, "LH")
pnl_hf = run_strategy(z_roll, "HF")
pnl_la = run_strategy(z_full, "LS")   # look-ahead 版本

def sharpe(p):
    p = p[LOOKBACK:]
    return p.mean()/ (p.std()+1e-12) * np.sqrt(252)

def ann(p):
    p = p[LOOKBACK:]
    return p.mean()*252

print(f"理论多空 LS:   Sharpe {sharpe(pnl_ls):.2f}  年化 {ann(pnl_ls)*100:.2f}%")
print(f"只多H  LH:    Sharpe {sharpe(pnl_lh):.2f}  年化 {ann(pnl_lh)*100:.2f}%")
print(f"多H空期货 HF: Sharpe {sharpe(pnl_hf):.2f}  年化 {ann(pnl_hf)*100:.2f}%")
print(f"Look-ahead LS: Sharpe {sharpe(pnl_la):.2f}  年化 {ann(pnl_la)*100:.2f}%")

# 收敛分解：进场后 60 天，溢价收敛多少来自 A 腿 vs H 腿
horizon = 60
entries = []
for t in range(LOOKBACK, T-1-horizon):
    hit = np.where((z_roll[t] > 1.5) & (z_roll[t-1] <= 1.5))[0]
    for i in hit:
        dA = logA[t+horizon, i] - logA[t, i]
        dH = logH[t+horizon, i] - logH[t, i]
        dP = log_prem[t+horizon, i] - log_prem[t, i]
        entries.append((dP, dA, dH))
entries = np.array(entries)
conv = entries[entries[:, 0] < 0]  # 实际收敛的事件
mean_dP = conv[:, 0].mean()
# 去掉市场共同项后各腿贡献
contribA = (-conv[:, 1] + conv[:, 2] + conv[:, 0]).mean()  # 恒等式辅助
print(f"\n进场事件 {len(entries)}，其中收敛 {len(conv)} ({len(conv)/len(entries)*100:.1f}%)")
print(f"60日平均溢价变化(收敛事件): {mean_dP*100:.2f}%")
# 直接分解: dP = dA - dH → A 腿贡献 = -dA / (-dP) 中 dA<0 部分
a_contrib = (-conv[:, 1]).mean() / (-mean_dP)
h_contrib = (conv[:, 2]).mean() / (-mean_dP)
print(f"A腿贡献占比(A下跌): {a_contrib*100:.1f}%, H腿贡献占比(H上涨): {h_contrib*100:.1f}%")

# 置换检验：打乱 z 信号时间轴
perm_sh = []
for k in range(200):
    shift = rng.integers(100, T-100)
    z_perm = np.roll(z_roll, shift, axis=0)
    perm_sh.append(sharpe(run_strategy(z_perm, "LS")))
perm_sh = np.array(perm_sh)
print(f"\n置换检验: 真实 LS Sharpe {sharpe(pnl_ls):.2f}, 置换95分位 {np.percentile(perm_sh,95):.2f}, 超过真实值比例 {(perm_sh>=sharpe(pnl_ls)).mean()*100:.1f}%")

# ---------------- 画图 ----------------
C1, C2, C3, C4 = "#2563eb", "#dc2626", "#16a34a", "#9333ea"

# 图1: 溢价指数 + 横截面分布
fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
years = np.arange(T)/252
prem_idx = (np.exp(c + mu_drift.mean(axis=1))-1)*100
ax[0].plot(years, prem_idx, color=C1, lw=1.2)
ax[0].axhline(35, color="gray", ls="--", lw=0.8, label="长期水位 35%")
ax[0].set_ylabel("整体溢价水位 (%)")
ax[0].set_title("模拟 AH 溢价：市场整体水位 12 年不回零")
ax[0].legend()
p10, p50, p90 = [np.percentile(prem*100, q, axis=1) for q in (10, 50, 90)]
ax[1].fill_between(years, p10, p90, color=C1, alpha=0.2, label="横截面 10-90 分位")
ax[1].plot(years, p50, color=C1, lw=1)
ax[1].set_xlabel("年")
ax[1].set_ylabel("个股溢价 (%)")
ax[1].set_title("个股溢价横截面：分化巨大且长期均值各不相同")
ax[1].legend()
fig.tight_layout()
fig.savefig(OUT/"premium-level-dispersion.png", dpi=110)
plt.close(fig)

# 图2: 三种实现方式净值
fig, ax = plt.subplots(figsize=(9, 5))
for p, label, color in [(pnl_ls, f"理论多H空A (Sharpe {sharpe(pnl_ls):.2f})", C3),
                        (pnl_hf, f"多H+空指数期货 (Sharpe {sharpe(pnl_hf):.2f})", C1),
                        (pnl_lh, f"只多H (Sharpe {sharpe(pnl_lh):.2f})", C2)]:
    ax.plot(np.arange(T-1)/252, np.cumsum(p)*100, label=label, color=color, lw=1.2)
ax.set_xlabel("年")
ax.set_ylabel("累计收益 (%)")
ax.set_title("同一信号的三种实现：A股不能做空是这笔交易的分水岭")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT/"three-implementations.png", dpi=110)
plt.close(fig)

# 图3: 收敛分解
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(["A腿贡献\n(A股下跌)", "H腿贡献\n(H股上涨)"],
              [a_contrib*100, h_contrib*100], color=[C2, C3], width=0.5)
for b, v in zip(bars, [a_contrib*100, h_contrib*100]):
    ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.1f}%", ha="center", fontsize=13)
ax.set_ylabel("收敛贡献占比 (%)")
ax.set_title(f"溢价收敛的分解（{len(conv)} 个收敛事件，60 日窗口）")
ax.axhline(0, color="black", lw=0.8)
fig.tight_layout()
fig.savefig(OUT/"convergence-decomposition.png", dpi=110)
plt.close(fig)

# 图4: look-ahead 陷阱 + 置换检验
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
ax[0].plot(np.arange(T-1)/252, np.cumsum(pnl_la)*100, color=C4,
           label=f"全样本均值z (look-ahead) Sharpe {sharpe(pnl_la):.2f}", lw=1.2)
ax[0].plot(np.arange(T-1)/252, np.cumsum(pnl_ls)*100, color=C3,
           label=f"滚动均值z (合法) Sharpe {sharpe(pnl_ls):.2f}", lw=1.2)
ax[0].set_xlabel("年"); ax[0].set_ylabel("累计收益 (%)")
ax[0].set_title("用全样本均值算 z 分数 = 偷看未来")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].hist(perm_sh, bins=30, color="gray", alpha=0.6, label="200次置换")
ax[1].axvline(sharpe(pnl_ls), color=C3, lw=2, label=f"真实 {sharpe(pnl_ls):.2f}")
ax[1].axvline(np.percentile(perm_sh, 95), color=C2, ls="--", lw=1.5,
              label=f"95分位 {np.percentile(perm_sh,95):.2f}")
ax[1].set_xlabel("Sharpe"); ax[1].set_title("置换检验：信号时序被打乱后优势消失")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT/"lookahead-permutation.png", dpi=110)
plt.close(fig)

print("\n图片已生成:", sorted(p.name for p in OUT.glob("*.png")))
