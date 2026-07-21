#!/usr/bin/env python3
"""
为文章「合成控制法评估政策冲击」(synthetic-control-method) 生成真实配图 + 可复现指标。

所有图表都由文中代码真实计算生成（纯 numpy + scipy.optimize，无 sklearn 依赖）：

  1) cover.png               —— 受冲击单元 vs 合成控制：事件前贴合、事件后 gap 即处理效应
  2) scm_weights.png         —— 合成控制的权重是稀疏的：仅少数 donor 贡献，多数权重为 0
  3) scm_placebo.png         —— 安慰剂（留一）检验：把受冲击单元的 post-gap 放进所有 donor 的伪 gap 分布，看是否极端

机制（数据自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 1 个受冲击单元 + 39 个 donor（未受冲击）。共同因子驱动 + 单元载荷，事件前受冲击单元可被 donor 的加权组合逼近；
  - 事件后受冲击单元叠加 +1.2 的持续性水平偏移（真实处理效应），donor 没有。
  - 估计：在事件前最小化 ‖Y_treat − Σ W_j Y_j‖²，约束 W≥0, ΣW=1（凸问题，SLSQP 求解）。
  - 处理效应 = Y_treat,post − Σ W_j Y_j,post（合成控制）。
"""
import os
import numpy as np
from scipy.optimize import minimize

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
D = os.path.join(BASE, "synthetic-control-method")
os.makedirs(D, exist_ok=True)

C = {"treat": "#4C72B0", "synth": "#E1A100", "ctrl": "#9E9E9E",
     "pos": "#55A868", "neg": "#C44E52", "gap": "#C44E52"}

rng = np.random.default_rng(20260722)

# ---------------------------------------------------------------------------
# 合成面板数据：因子模型
# ---------------------------------------------------------------------------
K_DONORS = 39
N = K_DONORS + 1          # +1 受冲击单元
T_PRE = 24
T_POST = 24
T = T_PRE + T_POST
r = 3                    # 潜在共同因子数
TRUE_EFFECT = 1.8        # 事件后受冲击单元的持续性水平偏移

# 共同因子路径（pre + post）
F = rng.normal(0, 1, (T, r))
F = np.cumsum(F, axis=0) * 0.4          # 平滑因子，更像宏观/行业状态
# 单元载荷
L = rng.normal(0, 1, (N, r))
# 无噪声潜在轨迹
Y0 = F @ L.T                          # (T, N)

# 受冲击单元 post 叠加处理效应
Y = Y0.copy()
Y[T_PRE:, 0] += TRUE_EFFECT

# 加观测噪声（小），并缩放到可读尺度（%）
noise = rng.normal(0, 0.08, (T, N))
Y = (Y + noise) * 0.6

treat_series = Y[:, 0]
donors = Y[:, 1:]

# ---------------------------------------------------------------------------
# 合成控制估计（凸权重）
# ---------------------------------------------------------------------------
def scm_weights(y_treat_pre, X_pre):
    # 最小化 ‖y_treat_pre - X_pre @ w‖², s.t. w>=0, sum(w)=1
    K = X_pre.shape[1]
    def obj(w):
        return float(np.sum((y_treat_pre - X_pre @ w) ** 2))
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0, None)] * K
    w0 = np.full(K, 1.0 / K)
    res = minimize(obj, w0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"ftol": 1e-10, "maxiter": 1000})
    return res.x

w = scm_weights(treat_series[:T_PRE], donors[:T_PRE])
synth = donors @ w

pre_rmse = np.sqrt(np.mean((treat_series[:T_PRE] - synth[:T_PRE]) ** 2))
post_gap = treat_series[T_PRE:] - synth[T_PRE:]
effect_mean = post_gap.mean()
print(f"真实处理效应 (post 水平偏移) = {TRUE_EFFECT*0.6:.3f}  (合成时乘了 0.6 缩放)")
print(f"合成控制 pre-fit RMSE = {pre_rmse:.4f}")
print(f"事件后平均 gap (DID 式处理效应) = {effect_mean:.4f}")
print(f"非零权重数 / 总 donor 数 = {int(np.sum(w > 1e-3))} / {K_DONORS}")
print("Top-5 权重:", np.round(np.sort(w)[::-1][:5], 3))

# ---------------------------------------------------------------------------
# 安慰剂（留一）检验：每个 donor 当成假受冲击单元，算其 post-gap
# ---------------------------------------------------------------------------
n_perm = N               # 用全部单元做 placebo（含真实受冲击单元）
placebo_gaps = np.zeros(n_perm)
for i in range(n_perm):
    if i == 0:
        y_pre = treat_series[:T_PRE]; y_post = treat_series[T_PRE:]
        Xp = donors
    else:
        cols = list(range(N)); cols.remove(i)
        others = Y[:, cols]
        y_pre = Y[:T_PRE, i]; y_post = Y[T_PRE:, i]
        Xp = others
    wi = scm_weights(y_pre, Xp[:T_PRE])
    si = Xp @ wi
    placebo_gaps[i] = (y_post - si[T_PRE:]).mean()
pval = np.mean(np.abs(placebo_gaps) >= abs(effect_mean))
print(f"安慰剂检验：真实单元 gap 在 {n_perm} 个单元中排名 → p≈{pval:.3f}")

# ===========================================================================
# 图 1: cover —— 受冲击单元 vs 合成控制
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 5))
tt = np.arange(T)
ax.plot(tt, treat_series, color=C["treat"], lw=2.4, label="受冲击单元（真实）")
ax.plot(tt, synth, color=C["synth"], lw=2.2, ls="--", label="合成控制（donor 加权）")
ax.axvline(T_PRE - 0.5, color=C["ctrl"], ls=":", lw=1.6, label="事件时点")
ax.fill_between(np.arange(T_PRE, T), synth[T_PRE:], treat_series[T_PRE:],
                color=C["gap"], alpha=0.15, label=f"处理效应 ≈ {effect_mean:.2f}%/月")
ax.set_xlabel("时期（事件前 → 事件后）"); ax.set_ylabel("结果变量 (%)")
ax.set_title(f"合成控制：事件前贴合 (RMSE={pre_rmse:.3f})，事件后才显出 gap", fontsize=12)
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(D, "cover.png"))
plt.close(fig)

# ===========================================================================
# 图 2: 权重稀疏性
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 4.5))
idx = np.arange(K_DONORS)
ax.bar(idx, w, color=C["synth"], alpha=0.85)
ax.set_xlabel("donor 单元编号"); ax.set_ylabel("合成控制权重 W_j")
ax.set_title(f"合成控制权重是稀疏的：仅 {int(np.sum(w>1e-3))}/{K_DONORS} 个 donor 贡献", fontsize=12)
ax.set_ylim(0, max(0.5, w.max() * 1.1))
fig.tight_layout()
fig.savefig(os.path.join(D, "scm_weights.png"))
plt.close(fig)

# ===========================================================================
# 图 3: 安慰剂检验分布
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(placebo_gaps[1:], bins=25, color=C["ctrl"], alpha=0.8, label=f"{N-1} 个未受冲击单元")
ax.axvline(effect_mean, color=C["treat"], lw=2.6, label=f"受冲击单元 gap={effect_mean:.3f}")
ax.set_xlabel("伪处理效应（各单元 post-gap 均值）"); ax.set_ylabel("单元数")
ax.set_title(f"安慰剂检验：真实受冲击单元的 gap 是分布中的极端离群点", fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "scm_placebo.png"))
plt.close(fig)

print("images saved to", D)
