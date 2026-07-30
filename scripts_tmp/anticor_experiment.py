#!/usr/bin/env python3
"""Anticor 反相关组合实验：合成市场 + 窗口敏感性 + 趋势市对照"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)
OUT = "/Users/halo/workspace/astro-blog/public/images/anticor-portfolio"
os.makedirs(OUT, exist_ok=True)

# ---------- 市场生成 ----------
ROT = 60  # 轮动块长（天）

def make_anticorr_market(T=1500, seed=42, block=ROT, amp=0.004):
    """两资产反相关轮动市场：领涨权每 block 天交接一次（方波轮动）"""
    r = np.random.default_rng(seed)
    t = np.arange(T)
    phase = np.where((t // block) % 2 == 0, 1.0, -1.0)  # 方波
    mu1 = 0.0004 + amp * phase
    mu2 = 0.0004 - amp * phase
    x1 = 1 + mu1 + r.normal(0, 0.012, T)
    x2 = 1 + mu2 + r.normal(0, 0.012, T)
    return np.column_stack([x1, x2]), phase  # price relatives

def make_trend_market(T=1500, seed=7):
    """趋势市对照：资产1持续跑赢，无轮动"""
    r = np.random.default_rng(seed)
    x1 = 1 + 0.0009 + r.normal(0, 0.012, T)
    x2 = 1 + 0.0001 + r.normal(0, 0.012, T)
    return np.column_stack([x1, x2])

# ---------- Anticor 算法 ----------
def anticor_weights(X, w):
    """Borodin-El-Yaniv-Gogan 2004. X: (T, m) price relatives. 返回每日权重 (T, m)"""
    T, m = X.shape
    b = np.ones(m) / m
    B = np.zeros((T, m))
    logX = np.log(X)
    for t in range(T):
        B[t] = b
        if t >= 2 * w:
            LX1 = logX[t - 2 * w: t - w]   # 前窗口
            LX2 = logX[t - w: t]           # 后窗口
            mu1, mu2 = LX1.mean(0), LX2.mean(0)
            s1, s2 = LX1.std(0, ddof=1), LX2.std(0, ddof=1)
            Mcov = (LX1 - mu1).T @ (LX2 - mu2) / (w - 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                Mcor = Mcov / np.outer(s1, s2)
            Mcor = np.nan_to_num(Mcor)
            claim = np.zeros((m, m))
            for i in range(m):
                for j in range(m):
                    if i != j and mu2[i] >= mu2[j] and Mcor[i, j] > 0:
                        claim[i, j] = Mcor[i, j] + max(0, -Mcor[i, i]) + max(0, -Mcor[j, j])
            transfer = np.zeros((m, m))
            for i in range(m):
                tot = claim[i].sum()
                if tot > 0:
                    for j in range(m):
                        transfer[i, j] = b[i] * claim[i, j] / tot
            b = b - transfer.sum(1) + transfer.sum(0)
            b = np.maximum(b, 0)
            b = b / b.sum()
        # 权重随市场漂移再于次日调回目标（简化：直接持有目标权重）
    return B

def wealth_curve(X, B):
    return np.cumprod((X * B).sum(1))

def crp_wealth(X, b):
    return np.cumprod((X * np.asarray(b)).sum(1))

def best_crp(X):
    grid = np.linspace(0, 1, 201)
    best, bw = None, None
    for g in grid:
        w = crp_wealth(X, [g, 1 - g])[-1]
        if best is None or w > best:
            best, bw = w, g
    return bw, best

# ---------- 主实验 ----------
X, phase = make_anticorr_market()
T = len(X)
w_main = 90
B_ac = anticor_weights(X, w_main)
W_ac = wealth_curve(X, B_ac)
bstar, wstar = best_crp(X)
W_bcrp = crp_wealth(X, [bstar, 1 - bstar])
W_5050 = crp_wealth(X, [0.5, 0.5])
W_bh1 = np.cumprod(X[:, 0])
W_bh2 = np.cumprod(X[:, 1])

print(f"[轮动市] Anticor 终值={W_ac[-1]:.2f}  BCRP(b*={bstar:.2f})={W_bcrp[-1]:.2f}  "
      f"50/50={W_5050[-1]:.2f}  BH1={W_bh1[-1]:.2f}  BH2={W_bh2[-1]:.2f}")

# 图1：财富曲线
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(W_ac, label=f"Anticor (w={w_main}) 终值 {W_ac[-1]:.2f}", lw=2, color="#d62728")
ax.plot(W_bcrp, label=f"事后最优 CRP b*={bstar:.2f} 终值 {W_bcrp[-1]:.2f}", lw=1.6, color="#1f77b4")
ax.plot(W_5050, label=f"50/50 每日再平衡 终值 {W_5050[-1]:.2f}", lw=1.4, color="#2ca02c")
ax.plot(W_bh1, label=f"买入持有资产1 终值 {W_bh1[-1]:.2f}", lw=1, ls="--", color="gray")
ax.plot(W_bh2, label=f"买入持有资产2 终值 {W_bh2[-1]:.2f}", lw=1, ls=":", color="gray")
ax.set_yscale("log")
ax.set_xlabel("交易日"); ax.set_ylabel("财富（对数轴）")
ax.set_title("反相关轮动市场（1500日，120日轮动周期）：Anticor vs 基准")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/anticor-wealth.png", dpi=130); plt.close(fig)

# 图2：权重轨迹 vs 隐藏相位
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(B_ac[:, 0], lw=1.2, color="#d62728", label="Anticor 资产1权重")
ax2 = ax.twinx()
ax2.plot(-phase, lw=1, color="#1f77b4", alpha=0.5, label="资产1相对劣势相位（-phase）")
ax.axhline(0.5, color="gray", ls="--", lw=0.8)
ax.set_xlabel("交易日"); ax.set_ylabel("资产1权重"); ax2.set_ylabel("-phase")
ax.set_title("Anticor 权重轨迹：资金被搬向『刚跑输但即将轮动到』的资产")
ax.set_xlim(200, 800)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/anticor-weights.png", dpi=130); plt.close(fig)

# 图3：窗口敏感性
ws = [3, 5, 8, 10, 15, 20, 30, 45, 60, 90, 120]
finals = []
for wi in ws:
    Bw = anticor_weights(X, wi)
    finals.append(wealth_curve(X, Bw)[-1])
    print(f"  w={wi:3d}  终值={finals[-1]:.2f}")
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(ws, finals, "o-", color="#d62728", lw=1.8)
ax.axhline(W_bcrp[-1], color="#1f77b4", ls="--", lw=1.2, label=f"事后最优 CRP {W_bcrp[-1]:.2f}")
ax.axhline(W_5050[-1], color="#2ca02c", ls=":", lw=1.2, label=f"50/50 {W_5050[-1]:.2f}")
ax.set_xlabel("窗口长度 w（天）"); ax.set_ylabel("终值财富")
ax.set_title("窗口敏感性：w 与轮动周期（120日）的匹配决定成败")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/anticor-window-scan.png", dpi=130); plt.close(fig)

# 图4：趋势市对照
Xt = make_trend_market()
np.random.seed(0)
B_act = anticor_weights(Xt, w_main)
W_act = wealth_curve(Xt, B_act)
Wt_bh1 = np.cumprod(Xt[:, 0])
Wt_5050 = crp_wealth(Xt, [0.5, 0.5])
bst, _ = best_crp(Xt)
Wt_bcrp = crp_wealth(Xt, [bst, 1 - bst])
print(f"[趋势市] Anticor={W_act[-1]:.2f}  BH强者={Wt_bh1[-1]:.2f}  50/50={Wt_5050[-1]:.2f}  BCRP(b*={bst:.2f})={Wt_bcrp[-1]:.2f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(W_act, label=f"Anticor (w={w_main}) 终值 {W_act[-1]:.2f}", lw=2, color="#d62728")
ax.plot(Wt_bh1, label=f"买入持有强势资产 终值 {Wt_bh1[-1]:.2f}", lw=1.6, color="#1f77b4")
ax.plot(Wt_5050, label=f"50/50 终值 {Wt_5050[-1]:.2f}", lw=1.4, color="#2ca02c")
ax.set_yscale("log")
ax.set_xlabel("交易日"); ax.set_ylabel("财富（对数轴）")
ax.set_title("趋势市对照：没有轮动时，Anticor 反复给落后者输血")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/anticor-trend-control.png", dpi=130); plt.close(fig)

# 换手率
turnover = np.abs(np.diff(B_ac, axis=0)).sum(1).mean()
print(f"[换手] Anticor 日均单边换手 = {turnover*100:.1f}%")
print("done")
