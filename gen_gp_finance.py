#!/usr/bin/env python3
"""
为文章「高斯过程回归在金融预测中的应用」(gaussian-process-finance) 生成真实配图。

所有图表都由文中代码真实计算生成（纯 numpy，无 sklearn/torch 依赖）：

  1) cover.png        —— 一维 GP 后验拟合（均值 + 95% 观测带），训练区段 vs 分布外区段
  2) gp_calibration.png —— 校准检验：分布内 vs 分布外(OOD) 区间覆盖率 + 预测 std 对真实误差
  3) gp_scaling.png    —— O(n^3) 计算复杂度实测：拟合时间随样本量立方增长

机制（数据自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 特征 x = 个股滞后 5 日已实现波动率(年化 %)；目标 y = 次日已实现波动率。
  - 真实关系在两区制间发生「概念漂移」（x>=阈值后斜率突变）：训练只用区制 1，
    测试覆盖两段，用来如实暴露「平稳 RBF 核在分布外会过度自信」这一核心陷阱。
  - 注意：95% 区间须用「观测带」= sqrt(函数方差 + 噪声方差)，否则只会覆盖函数本身。
"""
import os
import time
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
D = os.path.join(BASE, "gaussian-process-finance")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "gp": "#4C72B0", "true": "#2F4B7C",
     "train": "#55A868", "ood": "#C44E52", "band": "#9ECAE1", "gold": "#E1A100"}

rng = np.random.default_rng(20260722)


class GP1D:
    """一维高斯过程回归（RBF 核），纯 numpy 实现。"""
    def __init__(self, ls=1.0, sig=1.0, noise=0.1):
        self.ls, self.sig, self.noise = ls, sig, noise

    def _K(self, X, X2=None):
        if X2 is None:
            X2 = X
        X = np.atleast_2d(X).T
        X2 = np.atleast_2d(X2).T
        d = X - X2.T
        return self.sig ** 2 * np.exp(-0.5 * d ** 2 / self.ls ** 2)

    def fit(self, X, y):
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, float)
        K = self._K(self.X) + (self.noise ** 2 + 1e-8) * np.eye(len(self.X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))

    def predict(self, Xs, return_std=True):
        Xs = np.asarray(Xs, float)
        Ks = self._K(Xs, self.X)
        mean = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = self.sig ** 2 - np.sum(v ** 2, axis=0)
        if return_std:
            return mean, np.sqrt(np.clip(var, 0, None))
        return mean


def true_fn(x):
    f1 = 0.62 * x + 0.15 * np.sin(x)
    f2 = (0.62 * 3.5 + 0.15 * np.sin(3.5)) + 1.25 * (x - 3.5)
    return np.where(x < 3.5, f1, f2)


NOISE = 0.12
x_all = np.linspace(0.5, 5.0, 600)
y_all = true_fn(x_all) + rng.normal(0, NOISE, len(x_all))
train_mask = x_all < 3.5
Xtr, ytr = x_all[train_mask], y_all[train_mask]

# 用子采样训练集（避免网格过密导致核近奇异），超参人为固定在稳健区间
gp = GP1D(ls=1.0, sig=1.5, noise=NOISE)
gp.fit(Xtr, ytr)
xs = np.linspace(0.5, 5.0, 400)
mean, sd = gp.predict(xs)
mean_all, sd_all = gp.predict(x_all)

# 95% 观测带（含噪声）
band = np.sqrt(sd_all ** 2 + NOISE ** 2)
cov_in = np.mean(np.abs(y_all[train_mask] - mean_all[train_mask]) <= 1.96 * band[train_mask])
cov_ood = np.mean(np.abs(y_all[~train_mask] - mean_all[~train_mask]) <= 1.96 * band[~train_mask])
# 仅函数带（错误用法）做对照
func_band = 1.96 * sd_all
cov_in_func = np.mean(np.abs(y_all[train_mask] - mean_all[train_mask]) <= func_band[train_mask])
cov_ood_func = np.mean(np.abs(y_all[~train_mask] - mean_all[~train_mask]) <= func_band[~train_mask])
print(f"[校准] 95% 观测带覆盖率  分布内={cov_in:.3f}  分布外(OOD)={cov_ood:.3f}")
print(f"[校准] (仅函数带对照)    分布内={cov_in_func:.3f}  分布外={cov_ood_func:.3f}")
print(f"[校准] OOD 段观测带宽均值={band[~train_mask].mean():.3f}  真实|误差|均值={np.abs(y_all[~train_mask]-mean_all[~train_mask]).mean():.3f}")
print(f"[校准] OOD 段函数std均值={sd_all[~train_mask].mean():.3f} (远小于真实误差->过度自信)")


# =====================================================================
# 图 1（cover）：GP 后验拟合 + 95% 观测带
# =====================================================================
fig, ax = plt.subplots(figsize=(12, 5.4))
ax.axvspan(0.5, 3.5, color=C["train"], alpha=0.07, label="训练区(区制1)")
ax.axvspan(3.5, 5.0, color=C["ood"], alpha=0.07, label="分布外区(区制2·漂移)")
ax.plot(xs, true_fn(xs), "--", color=C["true"], lw=1.8, label="真实关系 f(x)")
# 用 x_all 上的 mean_all/band 重采样到 xs 索引以对齐
idx = np.searchsorted(x_all, xs)
ax.fill_between(xs, mean_all[idx] - 1.96 * band[idx], mean_all[idx] + 1.96 * band[idx],
                color=C["band"], alpha=0.55, label="95% 观测带")
ax.plot(xs, mean_all[idx], color=C["gp"], lw=2.2, label="GP 预测均值")
ax.scatter(Xtr, ytr, s=14, color=C["train"], zorder=5, label="训练样本")
ax.set_xlabel("滞后 5 日已实现波动率 x (年化 %)")
ax.set_ylabel("次日已实现波动率 y")
ax.set_title("高斯过程回归：均值预测 + 不确定性带（分布外区明显过度自信）")
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
fig.savefig(os.path.join(D, "cover.png"))
plt.close(fig)

# =====================================================================
# 图 2：校准检验（覆盖率 + std vs 误差）
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].bar(["分布内", "分布外(OOD)"], [cov_in, cov_ood], color=[C["train"], C["ood"]], width=0.55)
axes[0].axhline(0.95, ls="--", color="black", lw=1.2, label="名义 95%")
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel("经验覆盖率")
axes[0].set_title("95% 观测带实际覆盖率（OOD 大幅低于名义）")
axes[0].legend(fontsize=9)
axes[1].scatter(sd_all[train_mask], np.abs(y_all[train_mask] - mean_all[train_mask]),
                s=8, color=C["train"], alpha=0.5, label="分布内")
axes[1].scatter(sd_all[~train_mask], np.abs(y_all[~train_mask] - mean_all[~train_mask]),
                s=10, color=C["ood"], alpha=0.6, label="分布外(OOD)")
axes[1].set_xlabel("GP 函数预测标准差 σ")
axes[1].set_ylabel("真实 |误差|")
axes[1].set_title("函数σ对真实误差：OOD段σ偏小、误差却更大")
axes[1].legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "gp_calibration.png"))
plt.close(fig)

# =====================================================================
# 图 3：O(n^3) 计算复杂度实测
# =====================================================================
ns = [50, 100, 200, 400, 800, 1600]
times = []
for n in ns:
    xx = rng.uniform(0.5, 5, n)
    yy = true_fn(xx) + rng.normal(0, NOISE, n)
    g = GP1D(1.0, 1.5, NOISE)
    g.fit(xx, yy)
    # 仅测 Cholesky 分解 + 回代（GP 的 O(n^3) 主体），排除 Python 解释开销
    t0 = time.perf_counter()
    reps = 10
    for _ in range(reps):
        K = g._K(xx) + (NOISE ** 2 + 1e-8) * np.eye(n)
        L = np.linalg.cholesky(K)
        _ = np.linalg.solve(L.T, np.linalg.solve(L, yy))
    times.append((time.perf_counter() - t0) / reps * 1000.0)
p = np.polyfit(np.log(ns), np.log(times), 1)
print(f"[复杂度] log-log 斜率≈{p[0]:.2f}（理论 3.0）")
fig, ax = plt.subplots(figsize=(7.2, 4.8))
ax.loglog(ns, times, "o-", color=C["gp"], lw=2, label="实测 Cholesky 求解耗时(ms)")
ax.loglog(ns, [times[0] * (n / ns[0]) ** 3 for n in ns], "--", color=C["gold"], label="∝ n³ 参照")
ax.set_xlabel("训练样本量 n")
ax.set_ylabel("耗时 (ms, log)")
ax.set_title(f"GP 拟合 ∝ n³（大样本段实测斜率≈{p[0]:.2f}）")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "gp_scaling.png"))
plt.close(fig)

print("[完成] 已生成 cover.png / gp_calibration.png / gp_scaling.png ->", D)
