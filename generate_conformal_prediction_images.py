#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「共形预测 Conformal Prediction：给机器学习收益预测套上误差区间」生成真实配图与真实数字。

核心思想(Vladimir Vovk et al., 2005):
  - 机器学习只给点预测, 但金融收益异方差 + 厚尾, 固定区间会系统性漏覆盖
  - 共形预测(conformal prediction)用「校准集残差的经验分位」构造区间,
    在可交换性下给出有限样本的边缘覆盖保证: P(Y∈Ĉ(X)) ≥ 1-α
  - 本文对比三种区间:
      1) 朴素正态区间: pred ± z·σ_train (假设同方差+正态 -> 厚尾/异方差下漏覆盖)
      2) 普通 Split-Conformal: 校准集 |残差| 的经验分位 -> 边缘覆盖达标, 但宽度恒定
      3) 波动率感知 Conformal: 对「标准化残差(残差/滚动波动)」做共形 -> 宽度随波动伸缩,
         条件覆盖(按波动分层)也贴近名义水平

所有图表与数字均由文中 Python 逻辑真实计算生成(自包含):
  1) cp_intervals.png      —— 测试段真实收益 + 波动率感知共形区间带
  2) cp_coverage_levels.png—— 三种方法在 80%/90%/95% 名义水平下的边缘覆盖率(对比目标线)
  3) cp_width_vs_vol.png   —— 区间半宽 vs 当前波动: 普通共形恒定, 波动率感知共形随波动上升
  4) cp_coverage_by_vol.png—— 按波动五分位的条件覆盖率: 普通共形在高波动层漏覆盖, 波动率感知持平

数据: 合成收益 r_t = μ(X_t) + σ_t·ε_t, σ_t 为 GARCH 式波动聚集, ε_t ~ Student-t(df=4) 厚尾;
      μ 含轻微可预测项(滞后收益/波动), 但残差异方差+厚尾是主角。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
from sklearn.linear_model import Ridge

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "conformal-prediction-finance")
os.makedirs(D, exist_ok=True)

C = {"eq": "#2F4B7C", "up": "#55A868", "dn": "#C44E52", "grid": "#DDDDDD",
     "naive": "#8172B3", "plain": "#4C72B0", "vol": "#DD8452", "gold": "#CCB974", "tgt": "#999999"}

# =====================================================================
# 1) 合成数据: 异方差 + 厚尾 + 轻微可预测均值
#    关键设定: 训练段(前 50%) 波动平静, 校准+测试段(后 50%) 波动结构性抬升
#    -> 朴素正态区间用「平静训练期」残差尺度, 遇到高波动测试段会系统性过窄、漏覆盖
# =====================================================================
N = 4000
L = 3
rng = np.random.default_rng(20240713)
# GARCH 式对数波动聚集
h = np.zeros(N)
for t in range(1, N):
    h[t] = 0.90 * h[t - 1] + 0.25 * rng.standard_normal()
# 结构性波动抬升: 训练段平静, 校准+测试段高波动 (真实场景: 模型训于平静市, 遇波动骤升)
i1_abs = L + int((N - L) * 0.50)
level = np.where(np.arange(N) >= i1_abs, 0.9, 0.0)
vol = np.exp((h + level) / 2.0)
# 厚尾冲击: Student-t(df=3) —— 比 df=4 更肥尾, 朴素正态区间会明显漏覆盖
z = rng.standard_normal(N)
chi = rng.chisquare(3, N) / 3.0
eps = z / np.sqrt(chi)                       # ~ t(df=3)
r = np.zeros(N)
for t in range(3, N):
    mu = 0.06 * r[t - 1] - 0.04 * r[t - 2] + 0.12 * (vol[t - 1] - vol.mean())
    r[t] = mu + vol[t] * eps[t]

# 滚动波动代理(EWMA of |r|, 仅用历史, 无前视): 用于特征与共形缩放
lam = 0.94
rvol = np.zeros(N)
for t in range(1, N):
    rvol[t] = np.sqrt(lam * rvol[t - 1] ** 2 + (1 - lam) * r[t - 1] ** 2)
rvol[:3] = rvol[3]

# 特征矩阵 X_t = [r_{t-1}, r_{t-2}, r_{t-3}, |r_{t-1}|, rvol_{t-1}]
L = 3
X = np.column_stack([r[L - 1:N - 1], r[L - 2:N - 2], r[L - 3:N - 3],
                     np.abs(r[L - 1:N - 1]), rvol[L - 1:N - 1]])
y = r[L:N]
rv_test_proxy = rvol[L:N]                    # 各样本对应时刻的波动代理

# =====================================================================
# 2) 训练 / 校准 / 测试 三段切分(时间顺序, 无前视)
# =====================================================================
n = len(y)
i1, i2 = int(n * 0.50), int(n * 0.75)
X_tr, y_tr = X[:i1], y[:i1]
X_cal, y_cal = X[i1:i2], y[i1:i2]
X_te, y_te = X[i2:], y[i2:]
rv_cal = rv_test_proxy[i1:i2]
rv_te = rv_test_proxy[i2:]

model = Ridge(alpha=1.0).fit(X_tr, y_tr)
p_tr = model.predict(X_tr)
p_cal = model.predict(X_cal)
p_te = model.predict(X_te)
res_tr = y_tr - p_tr
res_cal = y_cal - p_cal
res_te = y_te - p_te
sigma_train = res_tr.std(ddof=1)


def conf_quantile(resids, alpha):
    """split-conformal 阈值: ⌈(n+1)(1-α)⌉/n 分位"""
    n_c = len(resids)
    k = int(np.ceil((n_c + 1) * (1 - alpha)))
    k = min(max(k, 1), n_c)
    return np.sort(resids)[k - 1]


def empirical_coverage(yv, lower, upper):
    return np.mean((yv >= lower) & (yv <= upper))


def quintile_coverage(yv, lower, upper, rv):
    order = np.argsort(rv)
    q = np.linspace(0, len(rv), 6).astype(int)
    covs = []
    for a, b in zip(q[:-1], q[1:]):
        idx = order[a:b]
        covs.append(empirical_coverage(yv[idx], lower[idx], upper[idx]))
    return np.array(covs)

# =====================================================================
# 3) 三种区间
# =====================================================================
alphas = [0.20, 0.10, 0.05]
z = {a: stats.norm.ppf(1 - a / 2) for a in alphas}

results = {}
for a in alphas:
    # 1) 朴素正态区间
    lo_naive = p_te - z[a] * sigma_train
    hi_naive = p_te + z[a] * sigma_train
    # 2) 普通 split-conformal
    q_plain = conf_quantile(np.abs(res_cal), a)
    lo_plain = p_te - q_plain
    hi_plain = p_te + q_plain
    # 3) 波动率感知 conformal (对标准化残差共形)
    sres_cal = np.abs(res_cal) / rv_cal
    q_vol = conf_quantile(sres_cal, a)
    lo_vol = p_te - q_vol * rv_te
    hi_vol = p_te + q_vol * rv_te
    results[a] = dict(
        cov_naive=empirical_coverage(y_te, lo_naive, hi_naive),
        cov_plain=empirical_coverage(y_te, lo_plain, hi_plain),
        cov_vol=empirical_coverage(y_te, lo_vol, hi_vol),
        width_naive=np.mean(hi_naive - lo_naive) / 2,
        width_plain=np.mean(hi_plain - lo_plain) / 2,
        width_vol=np.mean(hi_vol - lo_vol) / 2,
        quin_plain=quintile_coverage(y_te, lo_plain, hi_plain, rv_te),
        quin_vol=quintile_coverage(y_te, lo_vol, hi_vol, rv_te),
        band=(lo_vol, hi_vol, lo_plain, hi_plain, lo_naive, hi_naive),
    )

# 取 α=0.10 用于配图
a0 = 0.10
lo_vol, hi_vol, lo_plain, hi_plain, lo_naive, hi_naive = results[a0]["band"]

# =====================================================================
# 4) 图 1: 测试段收益 + 波动率感知共形区间带
# =====================================================================
seg = slice(0, 400)                          # 取测试段前 400 点便于观察
tt = np.arange(400)
fig, ax = plt.subplots(figsize=(9.4, 4.6))
ax.fill_between(tt, hi_vol[seg], lo_vol[seg], color=C["vol"], alpha=0.22,
                label="波动率感知共形区间 (90%)")
ax.plot(tt, y_te[seg], color=C["eq"], lw=1.1, label="真实收益 r_t")
ax.plot(tt, p_te[seg], color=C["dn"], lw=0.9, ls="--", alpha=0.8, label="模型点预测")
ax.set_xlabel("测试样本序号"); ax.set_ylabel("收益")
ax.set_title("共形预测区间: 真实收益绝大多数落在带内(边缘覆盖率≈名义水平)")
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "cp_intervals.png"), dpi=130); plt.close()

# =====================================================================
# 5) 图 2: 三种方法在 80/90/95% 名义水平下的边缘覆盖率
# =====================================================================
labels = ["80%", "90%", "95%"]
x = np.arange(len(labels)); w = 0.26
fig, ax = plt.subplots(figsize=(9, 4.8))
cov_naive = [results[a]["cov_naive"] for a in alphas]
cov_plain = [results[a]["cov_plain"] for a in alphas]
cov_vol = [results[a]["cov_vol"] for a in alphas]
ax.bar(x - w, cov_naive, w, color=C["naive"], label="朴素正态区间")
ax.bar(x, cov_plain, w, color=C["plain"], label="普通 Split-Conformal")
ax.bar(x + w, cov_vol, w, color=C["vol"], label="波动率感知 Conformal")
for i, tgt in enumerate([0.80, 0.90, 0.95]):
    ax.axhline(tgt, xmin=(i - 0.4) / 3, xmax=(i + 0.4) / 3, color=C["tgt"], ls="--", lw=1.3)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("边缘覆盖率")
ax.set_title("边缘覆盖率: 共形方法贴近名义水平, 朴素正态区间系统性漏覆盖(厚尾+异方差)")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(D, "cp_coverage_levels.png"), dpi=130); plt.close()

# =====================================================================
# 6) 图 3: 区间半宽 vs 当前波动
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.scatter(rv_te, (hi_vol - lo_vol) / 2, s=8, color=C["vol"], alpha=0.5,
           label="波动率感知共形 (半宽 ∝ 波动)")
ax.axhline(results[a0]["width_plain"], color=C["plain"], lw=1.8,
           label="普通共形 (恒定半宽=%.3f)" % results[a0]["width_plain"])
# 拟合半宽~波动的斜率
slope = np.polyfit(rv_te, (hi_vol - lo_vol) / 2, 1)[0]
ax.set_xlabel("当前波动代理 (EWMA |r|)"); ax.set_ylabel("区间半宽")
ax.set_title("区间宽度随波动伸缩: 波动率感知共形斜率≈%.3f, 普通共形为常数" % slope)
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "cp_width_vs_vol.png"), dpi=130); plt.close()

# =====================================================================
# 7) 图 4: 按波动五分位的条件覆盖率
# =====================================================================
qn = results[a0]["quin_plain"]
qv = results[a0]["quin_vol"]
fig, ax = plt.subplots(figsize=(9, 4.6))
xb = np.arange(5)
ax.plot(xb, qn, color=C["plain"], lw=2.0, marker="o", ms=6, label="普通共形")
ax.plot(xb, qv, color=C["vol"], lw=2.0, marker="s", ms=6, label="波动率感知共形")
ax.axhline(0.90, color=C["tgt"], ls="--", lw=1.3, label="名义 90%")
ax.set_xticks(xb); ax.set_xticklabels(["Q1\n低波动", "Q2", "Q3", "Q4", "Q5\n高波动"])
ax.set_ylabel("该波动层内覆盖率")
ax.set_title("条件覆盖: 普通共形在高波动层漏覆盖, 波动率感知共形各层贴近 90%")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "cp_coverage_by_vol.png"), dpi=130); plt.close()

# =====================================================================
# 打印真实数字
# =====================================================================
print("=== 共形预测 Conformal Prediction 关键数字 ===")
print("合成: N=%d, 收益 r_t=μ+σ_t·ε_t, σ_t 波动聚集且训练/测试段结构性抬升, ε_t~t(df=3) 厚尾" % N)
print("切分: 训练 %d / 校准 %d / 测试 %d" % (i1, i2 - i1, n - i2))
print("模型: Ridge; 训练残差 std=σ_train=%.4f" % sigma_train)
print("-" * 70)
print("名义   朴素正态      普通共形     波动率感知共形   | 平均半宽(朴素/普通/波动)")
for a in alphas:
    rr = results[a]
    print("%.0f%%    %.3f        %.3f         %.3f          | %.3f / %.3f / %.3f" % (
        (1 - a) * 100, rr["cov_naive"], rr["cov_plain"], rr["cov_vol"],
        rr["width_naive"], rr["width_plain"], rr["width_vol"]))
print("-" * 70)
print("α=10%% 按波动五分位覆盖率(目标0.90):")
print("  普通共形 :", np.round(qn, 3))
print("  波动感知 :", np.round(qv, 3))
print("波动率感知共形半宽~波动斜率 ≈ %.3f" % slope)
print("\n图片已保存到:", D)
