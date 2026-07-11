#!/usr/bin/env python3
"""
为文章「用 SHAP 做因子贡献归因与模型监控」(shap-factor-attribution) 生成真实配图。

核心叙事（与已发布的「SHAP 可解释性」文章区分）：
  本文聚焦「因子贡献归因」+「生产模型监控」两条线，而非单样本解释。
    - 归因：把模型的每一条预测（进而把组合 PnL）拆解到每个因子头上，
            并在因子组（价值/动量/质量/波动/其他）层面聚合。
    - 监控：把样本按时间分成若干周期，追踪各因子 SHAP 贡献的分布漂移，
            漂移即生产环境预警信号。
    - 完整性：用 Shapley 的效率性质 ΣSHAP ≈ 预测 − 基线，检验归因是否自洽。

图表（全部真实数值，非占位）：
  1. factor_attribution_waterfall.png  单条「强势买入」预测的因子级瀑布归因
  2. group_attribution.png             因子组层面的平均 |SHAP| 贡献强度
  3. monitoring_drift.png              各因子 SHAP 贡献随时间周期漂移（热力图 → 监控）
  4. attribution_efficiency.png        归因完整性检验：ΣSHAP vs 预测−基线（效率性质）
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.ensemble import GradientBoostingRegressor
from scipy.optimize import curve_fit

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "shap-factor-attribution")
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(20260711)

# ============================================================
# 1) 因子定义与相关性结构
# ============================================================
groups = {
    "价值": ["bp", "ep"],
    "动量": ["mom1m", "mom12m"],
    "质量": ["roe", "gross_margin"],
    "波动": ["vol_60d", "beta"],
    "其他": ["ln_mktcap", "sales_growth", "amihud_illiq"],
}
names = [f for g in groups for f in groups[g]]
p = len(names)
group_of = {f: g for g, fs in groups.items() for f in fs}

# 温和相关性协方差
cov = 0.6 * np.eye(p) + 0.4 / p
cov[names.index("bp"), names.index("mom12m")] = -0.15
cov[names.index("mom12m"), names.index("bp")] = -0.15

# ============================================================
# 2) 数据生成（含交互项与非线性） + 训练因子模型
# ============================================================
N_train = 3000
Xtr = rng.multivariate_normal(np.zeros(p), cov, size=N_train)


def dgp(Z):
    bp, ep, mom1m, mom12m, roe, gm, vol, beta, lnmc, sg, ami = Z.T
    r = (0.50 * bp + 0.35 * ep + 0.40 * mom12m + 0.30 * roe + 0.25 * gm
         - 0.35 * vol - 0.20 * beta + 0.15 * lnmc + 0.10 * sg - 0.10 * ami
         + 0.32 * mom1m * roe                       # 交互项：动量 × 质量
         + 0.10 * np.tanh(vol * 2.0) * mom12m)       # 轻度非线性
    return r


rtr = dgp(Xtr) + rng.normal(0, 0.40, N_train)
rtr = (rtr - rtr.mean()) / rtr.std()

model = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                  learning_rate=0.05, subsample=0.8, random_state=7)
model.fit(Xtr, rtr)
bg = Xtr.mean(0, keepdims=True)
base_pred = float(model.predict(bg)[0])
print("训练完成。基线预测(全因子均值):", round(base_pred, 4))


# ============================================================
# 3) 向量化蒙特卡洛 Shapley 估计
# ============================================================
def mc_shap(X, model, bg, K=120, seed=3):
    n = X.shape[0]
    rr = np.random.default_rng(seed)
    phi = np.zeros((p, n))
    for j in range(p):
        C = rr.random((n, K, p)) < 0.5            # (n, K, p)  coalition 掩码
        Cw = C.copy(); Cw[:, :, j] = True          # 含因子 j
        Cwo = C.copy(); Cwo[:, :, j] = False       # 不含因子 j
        Xw = np.where(Cw, X[:, None, :], bg[None, None, :]).reshape(n * K, p)
        Xwo = np.where(Cwo, X[:, None, :], bg[None, None, :]).reshape(n * K, p)
        pw = model.predict(Xw).reshape(n, K)
        pwo = model.predict(Xwo).reshape(n, K)
        phi[j] = (pw - pwo).mean(1)
    return phi


# 测试集：分成 10 个时间周期，制造 regime drift（因子均值跨周期漂移）
n_periods = 10
per = 90
N_te = n_periods * per
Xte = np.zeros((N_te, p))
period = np.repeat(np.arange(n_periods), per)
drift = {"bp": 0.30, "mom12m": -0.30, "roe": 0.20, "vol_60d": 0.15, "mom1m": -0.20}
for pp in range(n_periods):
    t = (pp - (n_periods - 1) / 2) / ((n_periods - 1) / 2)   # -1 .. 1
    mean = np.zeros(p)
    for f in names:
        if f in drift:
            mean[names.index(f)] = drift[f] * t
    Xte[period == pp] = rng.multivariate_normal(mean, cov, size=per)
rte = dgp(Xte) + rng.normal(0, 0.40, N_te)
rte = (rte - rte.mean()) / rte.std()

print("计算 SHAP（MC, K=%d, n=%d）..." % (120, N_te))
phi = mc_shap(Xte, model, bg, K=120, seed=3)
pred = model.predict(Xte)
print("效率性质 ΣSHAP vs 预测−基线 相关系数:",
      round(np.corrcoef(phi.sum(0), pred - base_pred)[0, 1], 4))

# ============================================================
# 4) 图1：单条预测的因子级瀑布归因
# ============================================================
idx = int(np.argmax(pred))                       # 取预测最高（最强买入信号）的样本
order = np.argsort(np.abs(phi[:, idx]))[::-1]    # 按 |SHAP| 降序
base = base_pred
cum = base
fig, ax = plt.subplots(figsize=(11, 6.2))
yticks, ylabels = [], []
running = base
for k, j in enumerate(order):
    v = phi[j, idx]
    color = "#d62728" if v > 0 else "#2ca02c"
    ax.barh(k, v, left=running, color=color, alpha=0.85, edgecolor="white")
    running += v
    yticks.append(k)
    ylabels.append(names[j])
ax.scatter([running], [len(order) - 0.5], color="black", zorder=5, s=40)
ax.plot([running, running], [-0.5, len(order) - 0.5], color="black", lw=1.0, ls="--")
ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel("对预测收益的贡献（标准化单位）", fontsize=11)
ax.set_title(f"单条最强买入信号的因子级归因瀑布（样本 #{idx}，预测={pred[idx]:.2f}）",
             fontsize=12.5, fontweight="bold")
ax.axvline(base, color="gray", lw=1.0, ls=":")
ax.text(base, len(order) - 0.3, f" 基线 {base:.2f}", fontsize=8.5, color="gray")
ax.grid(True, axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "factor_attribution_waterfall.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 5) 图2：因子组层面的平均 |SHAP| 贡献强度
# ============================================================
group_abs = {}
for g, fs in groups.items():
    idxs = [names.index(f) for f in fs]
    group_abs[g] = np.abs(phi[idxs, :]).mean()
items = sorted(group_abs.items(), key=lambda kv: kv[1])
g_names = [k for k, _ in items]
g_vals = [v for _, v in items]
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(g_names)))
fig, ax = plt.subplots(figsize=(9.5, 5.4))
bars = ax.barh(g_names, g_vals, color=colors)
for b, v in zip(bars, g_vals):
    ax.text(v + 0.002, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=10)
ax.set_xlabel("平均 |SHAP|（因子组贡献强度）", fontsize=11)
ax.set_title("因子组层面的 SHAP 贡献强度：Alpha 主要来自哪些大类", fontsize=12.5, fontweight="bold")
ax.grid(True, axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "group_attribution.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 6) 图3：生产监控 —— 各因子 SHAP 贡献随时间周期漂移（热力图）
# ============================================================
mean_phi_period = np.array([phi[:, period == pp].mean(1) for pp in range(n_periods)])  # (periods, p)
norm = mean_phi_period / (np.abs(mean_phi_period).max(0, keepdims=True) + 1e-9)        # 按行归一化
fig, ax = plt.subplots(figsize=(11.5, 6.0))
im = ax.imshow(norm.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(n_periods)); ax.set_xticklabels([f"P{i+1}" for i in range(n_periods)], fontsize=9)
ax.set_yticks(range(p)); ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("时间周期（P1 → P10，模拟周度/月度漂移）", fontsize=11)
ax.set_ylabel("因子", fontsize=11)
ax.set_title("生产监控：各因子 SHAP 贡献随时间漂移（按行归一化，红=正贡献/蓝=负贡献）",
             fontsize=12.5, fontweight="bold")
cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cb.set_label("归一化平均 SHAP", fontsize=9)
# 标注漂移最显著的因子
ax.text(0, names.index("bp"), "  ↑价值因子贡献抬升", fontsize=8, color="darkred", va="center")
ax.text(0, names.index("mom12m"), "  ↓动量因子贡献衰减", fontsize=8, color="navy", va="center")
plt.tight_layout()
plt.savefig(os.path.join(D, "monitoring_drift.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 7) 图4：归因完整性检验（效率性质）
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 6.0))
x = phi.sum(0)
y = pred - base_pred
ax.scatter(x, y, s=10, alpha=0.35, color="#1f77b4")
lims = [min(x.min(), y.min()), max(x.max(), y.max())]
ax.plot(lims, lims, "k--", lw=1.2, label="理想 y = x")
r = np.corrcoef(x, y)[0, 1]
ax.set_xlabel("ΣSHAP（各因子贡献之和）", fontsize=11)
ax.set_ylabel("模型预测 − 基线", fontsize=11)
ax.set_title(f"归因完整性检验：ΣSHAP ≈ 预测 − 基线（相关性 r = {r:.3f}）",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "attribution_efficiency.png"), dpi=150, bbox_inches="tight")
plt.close()

print("\n✅ 配图生成完成：", sorted(os.listdir(D)))
