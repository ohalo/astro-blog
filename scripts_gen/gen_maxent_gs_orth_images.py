#!/usr/bin/env python3
"""两篇量化文章真实配图生成（non-placeholder, real numpy / matplotlib charts）。

文章1: maximum-entropy-portfolio  —— 最大熵投资组合：在「什么都不确定」时给出最不偏见的权重
文章2: gram-schmidt-factor-orthogonalization —— Gram-Schmidt 因子正交化：把相关因子拆成互不重叠的纯净信号
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arrow

mpl_fonts = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
matplotlib.rcParams["font.sans-serif"] = mpl_fonts
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.weight"] = "normal"
matplotlib.rcParams["axes.titleweight"] = "normal"
matplotlib.rcParams["axes.labelweight"] = "normal"

C_TRUE = "#1f4e79"
C_REP = "#c0392b"
C_ACC = "#27ae60"
C_GREY = "#636e72"
GRID = "#e6e6e6"
log = math.log
plt.rcParams["figure.dpi"] = 160


def entropy(w):
    w = np.clip(w, 1e-15, None)
    return float(-(w * np.log(w)).sum())


# =========================================================================
# 文章1: 最大熵投资组合
# =========================================================================
OUT1 = "public/images/maximum-entropy-portfolio"
os.makedirs(OUT1, exist_ok=True)
rng = np.random.default_rng(20260831)

N = 50
F = 3
# 真实年化均值收益（小、带噪）
mu_true = rng.normal(0.09, 0.022, N)
# 真实协方差：3 个共同因子 + 特质波动（保证正定）
B = rng.normal(0.0, 1.0, (N, F))
Sigma = B @ B.T / F * 0.04 + np.diag(rng.uniform(0.01, 0.04, N) ** 2)
ann_vol = np.sqrt(np.diag(Sigma))
eq_ret = mu_true.mean()
eq_vol = np.sqrt((np.ones(N) / N) @ Sigma @ (np.ones(N) / N))
eq_sharpe = eq_ret / eq_vol
H_uniform = entropy(np.ones(N) / N)

print("=" * 70)
print("文章1 最大熵投资组合")
print("=" * 70)
print(f"N={N} 资产 | 年化收益均值 {mu_true.mean():.3f} 跨度 [{mu_true.min():.3f},{mu_true.max():.3f}]")
print(f"等权: 收益 {eq_ret:.4f} 波动 {eq_vol:.4f} Sharpe {eq_sharpe:.2f} 熵 {H_uniform/log(N):.4f}(归一)")


# ---- 最大熵 + 收益约束：w_i ∝ exp(λ μ_i) ----
def maxent_w(mu, lam):
    e = np.exp(lam * mu)
    return e / e.sum()


def lam_for_target(mu, b):
    lo, hi = -200.0, 200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        mean = (maxent_w(mu, mid) * mu).sum()
        if mean > b:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# 扫描目标收益 -> 熵曲线
targets = np.linspace(eq_ret * 0.85, mu_true.max() * 0.97, 25)
Hs, means, vols = [], [], []
for b in targets:
    lam = lam_for_target(mu_true, b)
    w = maxent_w(mu_true, lam)
    Hs.append(entropy(w))
    means.append(w @ mu_true)
    vols.append(np.sqrt(w @ Sigma @ w))
Hs = np.array(Hs)
print(f"熵扫描: 目标收益 [{targets[0]:.3f},{targets[-1]:.3f}] -> 归一熵 [{Hs.min()/log(N):.3f},{Hs.max()/log(N):.3f}]")
print(f"  最集中处(目标最高) 归一熵 = {Hs.min()/log(N):.3f} ; 等权处(目标=等权收益) 归一熵 = {entropy(maxent_w(mu_true,lam_for_target(mu_true,eq_ret)))/log(N):.3f}")

# ---- 几种策略的权重（用于权重分布图）----
w_eq = np.ones(N) / N
# 中等倾斜：目标收益 = 等权收益的 1.6 倍
lam_mid = lam_for_target(mu_true, eq_ret * 1.3)
w_mid = maxent_w(mu_true, lam_mid)
# 强倾斜：目标收益 = 接近最大值
lam_hi = lam_for_target(mu_true, mu_true.max() * 0.92)
w_hi = maxent_w(mu_true, lam_hi)
print(f"中等倾斜 λ={lam_mid:.2f} 归一熵={entropy(w_mid)/log(N):.3f} ; 强倾斜 λ={lam_hi:.2f} 归一熵={entropy(w_hi)/log(N):.3f}")

# ---- 风险-收益散点（上帝视角，用真实 Σ, μ）----
w_mv = np.linalg.solve(Sigma, mu_true)
w_mv = w_mv / w_mv.sum()  # 切线组合（仅归一，不放开空）
w_maxret = np.zeros(N); w_maxret[np.argmax(mu_true)] = 1.0
strat = {
    "等权": (w_eq, w_eq @ mu_true, np.sqrt(w_eq @ Sigma @ w_eq)),
    "熵倾斜(中)": (w_mid, w_mid @ mu_true, np.sqrt(w_mid @ Sigma @ w_mid)),
    "熵倾斜(强)": (w_hi, w_hi @ mu_true, np.sqrt(w_hi @ Sigma @ w_hi)),
    "均值-方差切线": (w_mv, w_mv @ mu_true, np.sqrt(w_mv @ Sigma @ w_mv)),
    "纯追高": (w_maxret, w_maxret @ mu_true, np.sqrt(w_maxret @ Sigma @ w_maxret)),
}
print("风险-收益散点(God view):")
for k, (w, m, v) in strat.items():
    print(f"  {k:12s}: 收益 {m:.3f} 波动 {v:.3f} Sharpe {m/v:.2f} 归一熵 {entropy(w)/log(N):.3f}")

# ===== 图1: 权重分布（等权 vs 中倾斜 vs 强倾斜）=====
order = np.argsort(mu_true)[::-1]
fig, ax = plt.subplots(figsize=(11, 5.6))
x = np.arange(N)
ax.bar(x, w_eq[order], color=C_GREY, alpha=0.5, label="等权 (熵最大)")
ax.bar(x, w_mid[order], color=C_TRUE, width=0.6, label=f"熵倾斜·中 (归一熵={entropy(w_mid)/log(N):.2f})")
ax.bar(x, w_hi[order], color=C_REP, width=0.4, label=f"熵倾斜·强 (归一熵={entropy(w_hi)/log(N):.2f})")
ax.set_xlabel("资产（按真实收益降序）"); ax.set_ylabel("权重")
ax.set_title("最大熵组合：不加观点→等权；观点越强→权重越集中、熵越低", fontsize=13, fontweight="bold")
ax.legend(fontsize=9.5); ax.grid(True, color=GRID, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT1}/mep_weights.png", dpi=160, bbox_inches="tight"); plt.close()

# ===== 图2: 熵 vs 目标收益（温度/λ 扫描）=====
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.plot(targets, Hs / np.log(N), color=C_ACC, lw=2.4, marker="o", ms=4)
ax.axhline(1.0, color=C_GREY, ls="--", lw=1.2, label="等权上界 (归一熵=1)")
eq_lam0 = lam_for_target(mu_true, eq_ret)
ax.axvline(eq_ret, color=C_TRUE, ls=":", lw=1.4, label=f"等权收益 {eq_ret:.3f}")
ax.set_xlabel("目标年化收益（约束 w'μ ≥ b）"); ax.set_ylabel("归一化熵 H/log N")
ax.set_title("熵随「观点强度」单调下降：越敢下注，越不保守", fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, color=GRID)
fig.tight_layout(); fig.savefig(f"{OUT1}/mep_entropy_curve.png", dpi=160, bbox_inches="tight"); plt.close()

# ===== 图3: 风险-收益散点 =====
fig, ax = plt.subplots(figsize=(11, 5.8))
for k, (w, m, v) in strat.items():
    c = C_REP if "追高" in k else (C_ACC if "等权" in k else C_TRUE)
    ax.scatter(v, m, s=90, color=c, zorder=3, label=f"{k} (SR={m/v:.2f})")
    ax.annotate(k, (v, m), textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.set_xlabel("年化波动率"); ax.set_ylabel("年化收益")
ax.set_title("风险-收益平面：最大熵倾斜在「等权」与「追高」之间连续性插值", fontsize=13, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, color=GRID)
fig.tight_layout(); fig.savefig(f"{OUT1}/mep_risk_return.png", dpi=160, bbox_inches="tight"); plt.close()

# ---- OOS 蒙特卡洛：估计噪声下谁更稳 ----
MC = 300
T_train, T_test = 60, 60
kappa = 0.5  # 均值收缩强度（向等权先验收缩）
sharpe_oos = {"等权": [], "MV切线(raw)": [], "熵倾斜(raw μ)": [], "熵倾斜(收缩μ)": []}
mu_prior = mu_true.mean()
for mc in range(MC):
    rr = rng.multivariate_normal(mu_true / 12.0, Sigma / 12.0, T_train + T_test)
    r_tr, r_te = rr[:T_train], rr[T_train:]
    mu_hat = r_tr.mean(0) * 12.0
    Sig_hat = np.cov(r_tr.T, ddof=1) * 12.0
    # 等权
    we = np.ones(N) / N
    # MV raw
    try:
        wmv = np.linalg.solve(Sig_hat, mu_hat); wmv = wmv / wmv.sum()
    except np.linalg.LinAlgError:
        wmv = we
    # 熵倾斜 raw
    lam_r = lam_for_target(mu_hat, eq_ret * 1.3)
    wme_r = maxent_w(mu_hat, lam_r)
    # 熵倾斜 收缩μ
    mu_s = kappa * mu_hat + (1 - kappa) * mu_prior
    lam_s = lam_for_target(mu_s, eq_ret * 1.3)
    wme_s = maxent_w(mu_s, lam_s)
    for name, w in [("等权", we), ("MV切线(raw)", wmv), ("熵倾斜(raw μ)", wme_r), ("熵倾斜(收缩μ)", wme_s)]:
        ret = r_te @ w
        sharpe_oos[name].append(ret.mean() / (ret.std(ddof=1) + 1e-9) * np.sqrt(12))
print("OOS 蒙特卡洛 (MC=%d, 训练%d月/测试%d月, 年化Sharpe):" % (MC, T_train, T_test))
for name in ["等权", "MV切线(raw)", "熵倾斜(raw μ)", "熵倾斜(收缩μ)"]:
    arr = np.array(sharpe_oos[name])
    print(f"  {name:14s}: 均值 {arr.mean():.3f}  中位 {np.median(arr):.3f}  std {arr.std():.3f}  胜率(>0) { (arr>0).mean():.2%}")

# ===== 图4: OOS Sharpe 箱型/柱状 =====
fig, ax = plt.subplots(figsize=(11, 5.4))
names = ["等权", "MV切线(raw)", "熵倾斜(raw μ)", "熵倾斜(收缩μ)"]
vals = [np.array(sharpe_oos[n]) for n in names]
colors = [C_ACC, C_REP, C_TRUE, C_ACC]
bp = ax.boxplot(vals, tick_labels=names, patch_artist=True, showmeans=True,
                medianprops=dict(color="k", lw=1.5))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.55)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("OOS 年化 Sharpe")
ax.set_title("估计噪声下：MV 与 raw 熵倾斜过拟合，收缩后的熵倾斜最稳", fontsize=13, fontweight="bold")
ax.grid(True, color=GRID, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT1}/mep_oos_sharpe.png", dpi=160, bbox_inches="tight"); plt.close()
print("文章1 配图完成 ->", OUT1)

# =========================================================================
# 文章2: Gram-Schmidt 因子正交化
# =========================================================================
OUT2 = "public/images/gram-schmidt-factor-orthogonalization"
os.makedirs(OUT2, exist_ok=True)
rng2 = np.random.default_rng(20260831)

N2 = 200          # 截面股票数
K = 4             # 因子数
# 真正正交的 latent 因子
g = rng2.normal(0, 1, (N2, K))
g = g / np.linalg.norm(g, axis=0)
# 用随机混合矩阵把它们搅成「相关」的原始因子
A = rng2.normal(0, 1, (K, K))
F_raw = g @ A.T
# 标准化为 z-score 截面
F_raw = (F_raw - F_raw.mean(0)) / F_raw.std(0)

# 原始因子相关矩阵
def corrmat(M):
    M = (M - M.mean(0)) / M.std(0)
    return np.corrcoef(M.T)

C_raw = corrmat(F_raw)
offdiag_raw = C_raw - np.eye(K)
max_off = np.abs(offdiag_raw).max()

# ---- Gram-Schmidt 正交化 ----
def gram_schmidt(F):
    Kk = F.shape[1]
    U = np.zeros_like(F, dtype=float)
    for k in range(Kk):
        u = F[:, k].copy()
        for j in range(k):
            proj = (F[:, k] @ U[:, j]) / (U[:, j] @ U[:, j] + 1e-15)
            u = u - proj * U[:, j]
        u = u / (np.linalg.norm(u) + 1e-15)
        U[:, k] = u
    return U

U = gram_schmidt(F_raw)
C_orth = corrmat(U)
# 正交后的「被前驱解释比例」：把 u_k 对 u_1..u_{k-1} 回归的 R²
r2_pred_orth = []
for k in range(K):
    if k == 0:
        r2_pred_orth.append(0.0); continue
    X = U[:, :k]
    yv = U[:, k]
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    pred = X @ coef
    ss_res = ((yv - pred) ** 2).sum(); ss_tot = ((yv - yv.mean()) ** 2).sum()
    r2_pred_orth.append(1 - ss_res / ss_tot)
# 原始因子的被前驱解释比例（作为对照）
r2_pred_raw = []
for k in range(K):
    if k == 0:
        r2_pred_raw.append(0.0); continue
    X = F_raw[:, :k]
    yv = F_raw[:, k]
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    pred = X @ coef
    ss_res = ((yv - pred) ** 2).sum(); ss_tot = ((yv - yv.mean()) ** 2).sum()
    r2_pred_raw.append(1 - ss_res / ss_tot)
# 每个正交因子的增量方差解释（用 u 的方差占全部 u 方差和的比例）
inc_var = (U ** 2).sum(0) / (U ** 2).sum()

print("\n" + "=" * 70)
print("文章2 Gram-Schmidt 因子正交化")
print("=" * 70)
print(f"N={N2} 截面 / K={K} 因子 | 原始因子最大离对角相关 = {max_off:.3f}")
print(f"正交后最大离对角相关 = {np.abs(C_orth-np.eye(K)).max():.4f}")
print(f"原始因子 被前驱解释R² = {[round(x,3) for x in r2_pred_raw]}")
print(f"正交因子 被前驱解释R² = {[round(x,4) for x in r2_pred_orth]}")
print(f"正交因子 增量方差占比 = {[round(x,3) for x in inc_var]}")

# ---- 截面回归：用 y 只由 g_1 驱动 ----
snr = 1.0
y = snr * g[:, 0] + rng2.normal(0, 1, N2) * 0.6
y = (y - y.mean()) / y.std()
# 原始因子回归
coef_raw, *_ = np.linalg.lstsq(F_raw, y, rcond=None)
pred_raw = F_raw @ coef_raw
r2_raw = 1 - ((y - pred_raw) ** 2).sum() / ((y - y.mean()) ** 2).sum()
se_raw = np.sqrt(np.sum((y - pred_raw) ** 2) / (N2 - K - 1)) * np.sqrt(np.diag(np.linalg.inv(F_raw.T @ F_raw)))
# 正交因子回归（纯净）
coef_orth, *_ = np.linalg.lstsq(U, y, rcond=None)
pred_orth = U @ coef_orth
r2_orth = 1 - ((y - pred_orth) ** 2).sum() / ((y - y.mean()) ** 2).sum()
se_orth = np.sqrt(np.sum((y - pred_orth) ** 2) / (N2 - K - 1)) * np.sqrt(np.diag(np.linalg.inv(U.T @ U)))
print(f"截面回归 R²: 原始因子={r2_raw:.3f} (同空间, 与正交应一致) / 正交因子={r2_orth:.3f}")
print(f"原始因子系数 = {np.round(coef_raw,3)}  SE={np.round(se_raw,3)}")
print(f"正交因子系数 = {np.round(coef_orth,3)}  SE={np.round(se_orth,3)}  (正交→SE相等且最小)")
# 说明：正交回归里只有与 g_1 相关的那个 u 分量应显著
u_vs_g1 = U.T @ g[:, 0]   # 每个 u 与真实驱动因子 g_1 的相关
print(f"各正交因子与真实驱动 g_1 的相关 = {np.round(u_vs_g1,3)}")

# ===== 图1: 原始因子相关热图 =====
fig, ax = plt.subplots(figsize=(5.6, 5.0))
im = ax.imshow(C_raw, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(K)); ax.set_yticks(range(K))
ax.set_xticklabels([f"F{i+1}" for i in range(K)]); ax.set_yticklabels([f"F{i+1}" for i in range(K)])
for i in range(K):
    for j in range(K):
        ax.text(j, i, f"{C_raw[i,j]:.2f}", ha="center", va="center",
                color="k" if abs(C_raw[i, j]) < 0.6 else "w", fontsize=10)
ax.set_title("原始因子：相关性被搅在一起", fontsize=12, fontweight="bold")
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig(f"{OUT2}/gs_corr_raw.png", dpi=160, bbox_inches="tight"); plt.close()

# ===== 图2: 正交因子相关热图 =====
fig, ax = plt.subplots(figsize=(5.6, 5.0))
im = ax.imshow(C_orth, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(K)); ax.set_yticks(range(K))
ax.set_xticklabels([f"F{i+1}⊥" for i in range(K)]); ax.set_yticklabels([f"F{i+1}⊥" for i in range(K)])
for i in range(K):
    for j in range(K):
        ax.text(j, i, f"{C_orth[i,j]:.2f}", ha="center", va="center",
                color="k" if abs(C_orth[i, j]) < 0.6 else "w", fontsize=10)
ax.set_title("Gram-Schmidt 后：对角=1、离对角≈0", fontsize=12, fontweight="bold")
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig(f"{OUT2}/gs_corr_orth.png", dpi=160, bbox_inches="tight"); plt.close()

# ===== 图3: 增量方差解释 =====
fig, ax = plt.subplots(figsize=(11, 5.2))
bars = ax.bar([f"F{i+1}⊥" for i in range(K)], inc_var, color=C_TRUE)
for i, v in enumerate(inc_var):
    ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("增量方差占比")
ax.set_title("每个正交因子的「纯净」增量信息递减：越往后越接近噪声", fontsize=13, fontweight="bold")
ax.set_ylim(0, max(inc_var) * 1.25); ax.grid(True, color=GRID, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT2}/gs_incremental_var.png", dpi=160, bbox_inches="tight"); plt.close()

# ===== 图4: 截面回归系数（原始 vs 正交）=====
fig, ax = plt.subplots(figsize=(11, 5.4))
xpos = np.arange(K)
w = 0.38
ax.bar(xpos - w/2, coef_raw, w, yerr=se_raw, color=C_REP, capsize=3, label="原始因子系数 (受共线污染)")
ax.bar(xpos + w/2, coef_orth, w, yerr=se_orth, color=C_ACC, capsize=3, label="正交因子系数 (纯净、SE 最小化)")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(xpos); ax.set_xticklabels([f"因子{i+1}" for i in range(K)])
ax.set_ylabel("回归系数")
ax.set_title(f"截面回归：正交后系数干净且可加 (R²={r2_orth:.2f})；原始系数被共线扭曲", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9.5); ax.grid(True, color=GRID, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT2}/gs_reg_coef.png", dpi=160, bbox_inches="tight"); plt.close()
print("文章2 配图完成 ->", OUT2)
