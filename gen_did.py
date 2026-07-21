#!/usr/bin/env python3
"""
为文章「双重差分 DID 在金融事件评估」(causal-did-finance) 生成真实配图 + 可复现指标。

所有图表都由文中代码真实计算生成（纯 numpy，无 sklearn 依赖）：

  1) cover.png              —— 2×2 DID 框架：处理组/控制组 × 事件前/后，效应=两组前后差分之差
  2) did_parallel_trends.png —— 平行趋势图：事件前处理组与控制组趋势平行，事件后处理组 diverge
  3) did_permutation.png     —— 置换（安慰剂）检验：把真实处理效应放进 1000 次随机分配的效应分布里看是否极端

机制（数据自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 面板：20 只股票（1 只被事件冲击 = 处理组，19 只同类 = 控制组），事件前 24 月 + 后 24 月
  - 平行趋势成立：事件前两组共享同一时间趋势；事件后处理组有 +0.8%/月的持续水平偏移
  - 估计：双向固定效应 Y_it = α_i + λ_t + τ·(treat_i × post_t) + ε_it，τ 即 DID 估计量（ATT）
  - 推断：置换检验把处理标签随机重排 1000 次，看真实 τ 在随机分布中的位置（安慰剂检验）
"""
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _cand in ["/System/Library/Fonts/STHeiti Medium.ttc",
              "/System/Library/Fonts/Supplemental/Songti SC.ttf"]:
    try:
        fm.fontManager.addfont(_cand)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Heiti SC", "Songti SC", "STHeiti", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight",
})

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "causal-did-finance")
os.makedirs(D, exist_ok=True)

C = {"treat": "#4C72B0", "ctrl": "#9E9E9E", "gold": "#E1A100",
     "pos": "#55A868", "neg": "#C44E52", "gap": "#C44E52"}

rng = np.random.default_rng(20260722)

# ---------------------------------------------------------------------------
# 合成面板数据
# ---------------------------------------------------------------------------
N_UNITS = 20          # 1 处理 + 19 控制
T_PRE = 24
T_POST = 24
T = T_PRE + T_POST
TRUE_TAU = 0.8        # 事件后处理组的真实持续水平偏移（%/月）

# 单元固定效应、共同时间趋势、特质噪声
alpha = rng.normal(0, 1.2, N_UNITS)
lam = np.linspace(0, 0.6, T)          # 轻微上行趋势（两组共享 → 平行）
treat = np.zeros(N_UNITS); treat[0] = 1.0
post = np.concatenate([np.zeros(T_PRE), np.ones(T_POST)])

Y = np.zeros((N_UNITS, T))
for i in range(N_UNITS):
    eps = rng.normal(0, 1.0, T)
    base = alpha[i] + lam
    # 事件后处理组多一层持续偏移
    Y[i] = base + TRUE_TAU * treat[i] * post + eps
# 转成“月收益率(单位 %)”尺度的可读数字
Y = Y * 0.5

# ---------------------------------------------------------------------------
# 双向固定效应估计 DID (= ATT)
# ---------------------------------------------------------------------------
def did_twfe(Y, treat, post):
    N, Tt = Y.shape
    yv = Y.ravel()
    unit = np.repeat(np.arange(N), Tt)
    time = np.tile(np.arange(Tt), N)
    did = (treat[:, None] * post[None, :]).ravel()
    # 设计矩阵：单元哑变量 + 时间哑变量 + DID
    X = []
    for i in range(N):
        col = (unit == i).astype(float); X.append(col)
    for t in range(Tt):
        col = (time == t).astype(float); X.append(col)
    X.append(did)
    X = np.array(X).T
    XtX = X.T @ X
    Xty = X.T @ yv
    beta, *_ = np.linalg.lstsq(XtX, Xty, rcond=None)
    tau = beta[-1]
    resid = yv - X @ beta
    dof = len(yv) - X.shape[1]
    se = np.sqrt(float(resid @ resid / dof) * np.linalg.inv(XtX)[-1, -1])
    return tau, se

tau, se = did_twfe(Y, treat, post)
tstat = tau / se
print(f"真实 τ (ATT) = {TRUE_TAU*0.5:.3f}  (合成时乘了 0.5 缩放)")
print(f"DID 估计 τ̂ = {tau:.4f}  标准误 = {se:.4f}  t = {tstat:.2f}")

# 经典 2×2 差分核对
pre_mean_treat = Y[0, :T_PRE].mean()
post_mean_treat = Y[0, T_PRE:].mean()
pre_mean_ctrl = Y[1:, :T_PRE].mean()
post_mean_ctrl = Y[1:, T_PRE:].mean()
d2 = (post_mean_treat - pre_mean_treat) - (post_mean_ctrl - pre_mean_ctrl)
print(f"2×2 差分核对 ATT = {d2:.4f}")

# ---------------------------------------------------------------------------
# 置换（安慰剂）检验
# ---------------------------------------------------------------------------
n_perm = 1000
perm_taus = np.zeros(n_perm)
idx = np.arange(N_UNITS)
for p in range(n_perm):
    if p == 0:
        tr = treat.copy()
    else:
        tr = np.zeros(N_UNITS); tr[rng.choice(N_UNITS, 1)] = 1.0
    tp, _ = did_twfe(Y, tr, post)
    perm_taus[p] = tp
pval = np.mean(np.abs(perm_taus) >= abs(tau))
print(f"置换检验 p 值 = {pval:.3f}   (perm 分布 sd = {perm_taus.std():.4f})")

# ===========================================================================
# 图 1: cover —— 2×2 DID 框架
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
panel = [("事件前 · 控制组", Y[1:].mean(0)[:T_PRE], C["ctrl"]),
         ("事件前 · 处理组", Y[0, :T_PRE], C["treat"]),
         ("事件后 · 控制组", Y[1:].mean(0)[T_PRE:], C["ctrl"]),
         ("事件后 · 处理组", Y[0, T_PRE:], C["treat"])]
for ax, (title, series, col) in zip(axes.ravel(), panel):
    ax.plot(np.arange(len(series)), series, color=col, lw=2.0)
    ax.axhline(series[0], color=col, ls=":", lw=1, alpha=0.6)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
# 在四格之间画 DID 箭头注解
fig.suptitle("双重差分 DID：效应 = (处理组后−前) − (控制组后−前)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(D, "cover.png"))
plt.close(fig)

# ===========================================================================
# 图 2: 平行趋势图
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 5))
tt = np.arange(T)
ctrl_mean = Y[1:].mean(0)
treat_mean = Y[0]
ax.plot(tt, ctrl_mean, color=C["ctrl"], lw=2.2, label="控制组（19 只同类）")
ax.plot(tt, treat_mean, color=C["treat"], lw=2.2, label="处理组（被冲击股票）")
ax.axvline(T_PRE - 0.5, color=C["gold"], ls="--", lw=1.6, label="事件时点")
# 事件前平行趋势拟合
pre_fit_ctrl = np.polyval(np.polyfit(np.arange(T_PRE), ctrl_mean[:T_PRE], 1), np.arange(T))
pre_fit_treat = np.polyval(np.polyfit(np.arange(T_PRE), treat_mean[:T_PRE], 1), np.arange(T))
ax.plot(np.arange(T_PRE), pre_fit_treat[:T_PRE], color=C["treat"], ls=":", lw=1.2)
ax.plot(np.arange(T_PRE), pre_fit_ctrl[:T_PRE], color=C["ctrl"], ls=":", lw=1.2)
# 事件后 gap 阴影
gap = treat_mean[T_PRE:] - ctrl_mean[T_PRE:]
ax.fill_between(np.arange(T_PRE, T), ctrl_mean[T_PRE:], treat_mean[T_PRE:],
                color=C["gap"], alpha=0.12, label=f"处理效应 ≈ {tau:.2f}%/月")
ax.set_xlabel("月份（事件前 24 → 事件后 24）"); ax.set_ylabel("月收益率 (%)")
ax.set_title("平行趋势：事件前两组趋势平行，事件后处理组系统性抬升", fontsize=12)
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(D, "did_parallel_trends.png"))
plt.close(fig)

# ===========================================================================
# 图 3: 置换（安慰剂）检验分布
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(perm_taus, bins=40, color=C["ctrl"], alpha=0.8, density=True,
        label=f"1000 次随机分配 (sd={perm_taus.std():.3f})")
ax.axvline(tau, color=C["treat"], lw=2.5, label=f"真实处理效应 τ̂={tau:.3f}")
ax.axvline(-tau, color=C["treat"], lw=1.2, ls=":")
ax.set_xlabel("随机重排下的 DID 估计量 τ"); ax.set_ylabel("密度")
ax.set_title(f"安慰剂/置换检验：真实效应落在随机分布极尾 (p={pval:.3f})", fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "did_permutation.png"))
plt.close(fig)

print("images saved to", D)
