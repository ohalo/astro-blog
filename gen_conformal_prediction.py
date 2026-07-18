#!/usr/bin/env python3
"""
为文章「共形预测在量化中的应用：给预测区间一个可验证的覆盖率保证」
(conformal-prediction-quant) 生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（自洽合成，仅用于演示方法；真实落地见文末路径）：
  * 构造异方差回归 y = sin(2x) + σ(x)·ε,σ(x) 随 x 变化(真实市场波动率聚类)
  * Split-Conformal(交叉拟合): 训练集拟模型 -> 校准集算残差分位 -> 测试集给 [pred±ĉ] 区间
  * 图1: 测试集预测点 + 共形区间带 + 真实值(命中/漏出着色),展示边际覆盖 ≈ 1-α
  * 图2: 经验覆盖 vs 名义 1-α(多次随机划分取均值±SD),点贴近 y=x —— 有限样本有效性的实证
  * 图3: 分布漂移下,普通(训练残差)区间 vs 共形(校准集)区间的覆盖对比:漂移使朴素法漏覆盖尾部
  * 图4: 区间宽度(效率) vs 名义覆盖 1-α:α 越大区间越宽(诚实的代价)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.ensemble import HistGradientBoostingRegressor

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "conformal-prediction-quant")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "pred": "#4C72B0", "band": "#C44E52", "true": "#55A868",
     "hit": "#55A868", "miss": "#C44E52", "naive": "#DD8452", "conf": "#4C72B0", "mk": "#8172B3"}


def gen_hetero(n, rng, scale=1.0):
    x = rng.uniform(-3, 3, n)
    f = np.sin(2 * x)
    sigma = (0.25 + 0.45 * (x > 0)) * scale   # x>0 区波动更大(异方差/波动聚类)
    y = f + rng.normal(0, sigma)
    return x, y


def split_conformal(x_tr, y_tr, x_cal, y_cal, x_te, y_te, alpha, model=None):
    if model is None:
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_depth=3)
    model.fit(x_tr.reshape(-1, 1), y_tr)
    pred_cal = model.predict(x_cal.reshape(-1, 1))
    resid = np.abs(y_cal - pred_cal)
    n = len(resid)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    c = np.quantile(resid, q, method="higher")  # 有限样本校正
    pred_te = model.predict(x_te.reshape(-1, 1))
    lo = pred_te - c
    hi = pred_te + c
    cover = np.mean((y_te >= lo) & (y_te <= hi))
    return model, pred_te, lo, hi, c, cover, pred_cal, resid


# =================== 图1：测试集区间与覆盖 ===================
rng = np.random.default_rng(20260719)
x_tr, y_tr = gen_hetero(1500, rng, 1.0)
x_cal, y_cal = gen_hetero(800, rng, 1.0)
x_te, y_te = gen_hetero(600, rng, 1.0)
alpha = 0.10
model, pred_te, lo, hi, c, cover, _, _ = split_conformal(x_tr, y_tr, x_cal, y_cal, x_te, y_te, alpha)
print("图1 经验覆盖(目标 %.2f): %.3f  区间半宽 c=%.3f" % (1 - alpha, cover, c))
order = np.argsort(x_te)
xo = x_te[order]
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.fill_between(xo, lo[order], hi[order], color=C["band"], alpha=0.18, label="共形 90%% 区间 [pred±ĉ]")
ax.plot(xo, pred_te[order], color=C["pred"], lw=1.8, label="点预测")
inside = (y_te >= lo) & (y_te <= hi)
ax.scatter(x_te[inside], y_te[inside], s=14, color=C["hit"], alpha=0.8, label="真实值·区间内(命中)")
ax.scatter(x_te[~inside], y_te[~inside], s=26, color=C["miss"], marker="x", label="真实值·区间外(漏出)")
ax.set_xlabel("特征 x", fontsize=11)
ax.set_ylabel("目标 y", fontsize=11)
ax.set_title("Split-Conformal 区间：测试集经验覆盖 = %.1f%% ≈ 目标 90%%" % (cover * 100), fontsize=11.5)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, color=C["grid"])
plt.tight_layout()
fig.savefig(os.path.join(D, "conformal_interval_coverage.png"), dpi=130)
plt.close(fig)

# =================== 图2：有效性(经验覆盖 vs 名义) ===================
nominal = np.array([0.80, 0.85, 0.90, 0.95])
emp_mean, emp_sd = [], []
for nm in nominal:
    a = 1 - nm
    covs = []
    for k in range(40):
        rk = np.random.default_rng(1000 + k)
        xt, yt = gen_hetero(1500, rk, 1.0)
        xc, yc = gen_hetero(800, rk, 1.0)
        xv, yv = gen_hetero(600, rk, 1.0)
        _, _, _, _, _, cov, _, _ = split_conformal(xt, yt, xc, yc, xv, yv, a)
        covs.append(cov)
    emp_mean.append(np.mean(covs))
    emp_sd.append(np.std(covs))
emp_mean = np.array(emp_mean)
emp_sd = np.array(emp_sd)
print("图2 经验覆盖 by 名义:", dict(zip(nominal, np.round(emp_mean, 3))))
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.errorbar(nominal, emp_mean, yerr=emp_sd, fmt="o", color=C["conf"], lw=1.8, capsize=4,
            markersize=7, label="共形经验覆盖(40 次划分 ±SD)")
ax.plot([0.78, 0.97], [0.78, 0.97], color="#333333", ls="--", lw=1.2, label="理想对角线(经验=名义)")
ax.set_xlabel("名义覆盖 1-α", fontsize=11)
ax.set_ylabel("经验覆盖", fontsize=11)
ax.set_title("有限样本有效性：经验覆盖贴在名义水平附近", fontsize=11.5)
ax.set_xlim(0.78, 0.97)
ax.set_ylim(0.75, 0.98)
ax.legend(fontsize=9)
ax.grid(True, color=C["grid"])
plt.tight_layout()
fig.savefig(os.path.join(D, "conformal_validity.png"), dpi=130)
plt.close(fig)

# =================== 图3：分布漂移下 朴素 vs 共形 ===================
# 训练/校准来自 scale=1.0,测试来自 scale=2.0(波动翻倍=分布漂移)
x_tr2, y_tr2 = gen_hetero(1500, np.random.default_rng(11), 1.0)
x_cal2, y_cal2 = gen_hetero(800, np.random.default_rng(12), 1.0)
# 漂移测试集:波动翻倍
x_te2, y_te2 = gen_hetero(800, np.random.default_rng(13), 2.0)

model2 = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_depth=3)
model2.fit(x_tr2.reshape(-1, 1), y_tr2)
pred_cal2 = model2.predict(x_cal2.reshape(-1, 1))
resid_cal = np.abs(y_cal2 - pred_cal2)
# 共形:用校准集残差分位
n = len(resid_cal)
a = 0.10
q = np.ceil((n + 1) * (1 - a)) / n
c_conf = np.quantile(resid_cal, q, method="higher")
# 朴素:用训练集残差(来自 scale=1.0,对 scale=2.0 测试过窄)
pred_tr2 = model2.predict(x_tr2.reshape(-1, 1))
resid_tr = np.abs(y_tr2 - pred_tr2)
c_naive = np.quantile(resid_tr, q, method="higher")
pred_te2 = model2.predict(x_te2.reshape(-1, 1))
cover_conf = np.mean((y_te2 >= pred_te2 - c_conf) & (y_te2 <= pred_te2 + c_conf))
cover_naive = np.mean((y_te2 >= pred_te2 - c_naive) & (y_te2 <= pred_te2 + c_naive))
print("图3 漂移测试 朴素覆盖=%.3f (区间半宽 %.3f) | 共形覆盖=%.3f (区间半宽 %.3f)" % (cover_naive, c_naive, cover_conf, c_conf))
ord2 = np.argsort(x_te2)
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(x_te2[ord2], pred_te2[ord2], color=C["pred"], lw=1.8, label="点预测")
ax.fill_between(x_te2[ord2], (pred_te2 - c_naive)[ord2], (pred_te2 + c_naive)[ord2],
                color=C["naive"], alpha=0.22, label="朴素区间(训练残差,过窄)")
ax.fill_between(x_te2[ord2], (pred_te2 - c_conf)[ord2], (pred_te2 + c_conf)[ord2],
                color=C["band"], alpha=0.22, label="共形区间(校准集,自适应)")
inside_c = (y_te2 >= pred_te2 - c_conf) & (y_te2 <= pred_te2 + c_conf)
ax.scatter(x_te2[~inside_c], y_te2[~inside_c], s=24, color=C["miss"], marker="x", label="漏出点")
ax.set_xlabel("特征 x", fontsize=11)
ax.set_ylabel("目标 y", fontsize=11)
ax.set_title("分布漂移(波动翻倍)：朴素法漏覆盖=%.1f%%, 共形法=%.1f%%" % (cover_naive * 100, cover_conf * 100), fontsize=11)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, color=C["grid"])
plt.tight_layout()
fig.savefig(os.path.join(D, "naive_vs_conformal_drift.png"), dpi=130)
plt.close(fig)

# =================== 图4：效率(区间宽度) vs α ===================
widths_conf, widths_naive = [], []
for nm in nominal:
    a = 1 - nm
    n = len(resid_cal)
    q = np.ceil((n + 1) * (1 - a)) / n
    c_conf = np.quantile(resid_cal, q, method="higher")
    c_naive = np.quantile(resid_tr, q, method="higher")
    widths_conf.append(2 * c_conf)
    widths_naive.append(2 * c_naive)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(nominal, widths_naive, "s-", color=C["naive"], lw=1.8, label="朴素(训练残差)")
ax.plot(nominal, widths_conf, "o-", color=C["conf"], lw=1.8, label="共形(校准集)")
for x, y in zip(nominal, widths_conf):
    ax.text(x, y + 0.05, "%.2f" % y, ha="center", fontsize=9, color=C["conf"])
ax.set_xlabel("名义覆盖 1-α", fontsize=11)
ax.set_ylabel("区间总宽度 (2ĉ)", fontsize=11)
ax.set_title("效率代价：越想要确定性(α 越大)，区间越宽", fontsize=11.5)
ax.legend(fontsize=9)
ax.grid(True, color=C["grid"])
plt.tight_layout()
fig.savefig(os.path.join(D, "interval_width_vs_alpha.png"), dpi=130)
plt.close(fig)

print("saved 4 figures to", D)
