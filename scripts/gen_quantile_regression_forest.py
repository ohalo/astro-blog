#!/usr/bin/env python3
"""
为文章「分位数回归森林：给出收益分布而非单点预测」(quantile-regression-forest)
生成真实配图 + 计算正文引用的所有关键数字。

机制（自洽合成，仅用于演示；落地见文末路径）：
  * 造一组异方差 + 左偏的收益面板：
        - 特征 s  (信号) 决定条件均值  mu = 0.6*s
        - 特征 v  (波动状态, 0=平静 1=紧张) 决定条件尺度 sigma = 0.3 + 1.7*v
        - 噪声用左偏 skew normal (a=-5) => 肥左尾(亏损尾更肥)
        y = mu + sigma * noise
  * 从零实现：
        - 回归树(CART 存叶节点全部 y 样本)
        - 随机森林(自助采样 + 随机特征子集)
        - 分位数回归森林 QRF(Meinshausen 2006): 预测 quantile tau = 各树落点叶
          节点样本的并集中取经验分位
        - 线性分位数回归 QR(从零, Pinball 损失的次梯度下降) 作为对照
        - OLS 点预测 + 全局正态假设 作为最朴素对照
  * 量化指标：
        - 90% 中心区间经验覆盖率：QRF / OLS-正态 / 线性QR
        - CRPS(连续秩概率分数)：QRF < OLS-正态 < 线性QR
        - 可靠性曲线(reliability diagram)：名义水平 vs 经验覆盖
"""
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "quantile-regression-forest")
os.makedirs(D, exist_ok=True)

C = {"qrf": "#4C72B0", "ols": "#C44E52", "lqr": "#55A868",
     "true": "#8172B3", "grid": "#DDDDDD", "warn": "#DD8452"}

rng = np.random.default_rng(20260723)
N = 3000
s = rng.normal(0, 1, N)                      # 信号特征
v = rng.uniform(0, 1, N)                     # 波动状态特征
noise = stats.skewnorm.rvs(a=-5, size=N, random_state=7)
noise = noise / np.std(noise) * 0.8          # 标准化到 ~单位尺度
sigma = 0.3 + 1.7 * v                        # 异方差：紧张时尺度翻数倍
mu = 0.6 * s                                 # 条件均值只取决于 s
y = mu + sigma * noise
X = np.column_stack([s, v])

# ---- 训练/测试切分 ----
n_tr = 2100
Xtr, ytr = X[:n_tr], y[:n_tr]
Xte, yte = X[n_tr:], y[n_tr:]


# ---------------------------------------------------------------------------
# 1) 回归树：存叶节点全部 y 样本
# ---------------------------------------------------------------------------
def best_split(X, y, feats, n_thr=12):
    parent_var = np.var(y) * len(y)
    best_gain, best = -1.0, None
    for f in feats:
        col = X[:, f]
        qs = np.quantile(col, np.linspace(0.15, 0.85, n_thr))
        for thr in qs:
            left = y[col <= thr]
            right = y[col > thr]
            if len(left) < 3 or len(right) < 3:
                continue
            gain = parent_var - (np.var(left) * len(left) + np.var(right) * len(right))
            if gain > best_gain:
                best_gain = gain
                best = (f, thr)
    return best


def build_tree(X, y, depth, max_depth, min_leaf, rng):
    n = len(y)
    if depth >= max_depth or n <= 2 * min_leaf or np.var(y) < 1e-7:
        return {"leaf": True, "vals": y.copy()}
    feats = rng.choice(X.shape[1], size=max(1, int(np.ceil(np.sqrt(X.shape[1])))), replace=False)
    sp = best_split(X, y, feats)
    if sp is None or sp[1] is None:
        return {"leaf": True, "vals": y.copy()}
    f, thr = sp
    mask = X[:, f] <= thr
    if mask.sum() < 3 or (~mask).sum() < 3:
        return {"leaf": True, "vals": y.copy()}
    left = build_tree(X[mask], y[mask], depth + 1, max_depth, min_leaf, rng)
    right = build_tree(X[~mask], y[~mask], depth + 1, max_depth, min_leaf, rng)
    return {"leaf": False, "split": (f, thr), "left": left, "right": right}


def tree_leaf_vals(tree, x):
    node = tree
    while not node["leaf"]:
        f, thr = node["split"]
        node = node["left"] if x[f] <= thr else node["right"]
    return node["vals"]


def tree_mean(tree, x):
    return np.mean(tree_leaf_vals(tree, x))


# ---------------------------------------------------------------------------
# 2) 随机森林 + QRF
# ---------------------------------------------------------------------------
B = 220
forest = []
for b in range(B):
    idx = rng.integers(0, n_tr, n_tr)         # 自助采样
    forest.append(build_tree(Xtr[idx], ytr[idx], 0, max_depth=7, min_leaf=8, rng=rng))


def qrf_quantile(x, tau):
    samples = []
    for tree in forest:
        samples.append(tree_leaf_vals(tree, x))
    allv = np.concatenate(samples)
    return np.quantile(allv, tau)


def qrf_mean(x):
    ms = np.array([tree_mean(t, x) for t in forest])
    return ms.mean()


# ---------------------------------------------------------------------------
# 3) 线性分位数回归 (Pinball 次梯度下降, 从零)
# ---------------------------------------------------------------------------
def fit_linear_qr(X, y, tau, iters=4000, lr=0.02):
    n, d = X.shape
    beta = np.zeros(d)
    for it in range(iters):
        pred = X @ beta
        resid = y - pred
        # 次梯度
        g = -X.T @ ((resid < 0).astype(float) - tau) / n
        lr_now = lr / (1 + it / 500.0)
        beta -= lr_now * g
    return beta


def linear_qr_quantile(beta, x):
    return float(x @ beta)


def _get_beta(d, key):
    """容忍浮点键微小误差"""
    for k in d:
        if abs(k - key) < 1e-9:
            return d[k]
    raise KeyError(key)


# ---------------------------------------------------------------------------
# 4) OLS 点预测 + 全局正态
# ---------------------------------------------------------------------------
beta_ols, _, _, _ = np.linalg.lstsq(Xtr, ytr, rcond=None)
ols_resid_std = np.std(ytr - Xtr @ beta_ols)


# ---------------------------------------------------------------------------
# 5) 评估指标
# ---------------------------------------------------------------------------
# 5a) 点预测 RMSE (均值)
qrf_means = np.array([qrf_mean(x) for x in Xte])
ols_means = Xte @ beta_ols
rmse_qrf = np.sqrt(np.mean((qrf_means - yte) ** 2))
rmse_ols = np.sqrt(np.mean((ols_means - yte) ** 2))

# 5b) 线性 QR 拟合一组细密分位 (用于覆盖 + CRPS)
taus = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                  0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
lqr_betas = {float(t): fit_linear_qr(Xtr, ytr, float(t)) for t in taus}

# 5c) 可靠性曲线：对一系列置信水平 alpha 算中心区间经验覆盖
alphas = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
cov_qrf, cov_ols, cov_lqr = [], [], []
for a in alphas:
    lo_t, hi_t = float((1 - a) / 2), float(1 - (1 - a) / 2)
    z = stats.norm.ppf(hi_t)
    # QRF
    in_qrf = 0
    for i, x in enumerate(Xte):
        lo, hi = qrf_quantile(x, lo_t), qrf_quantile(x, hi_t)
        if lo <= yte[i] <= hi:
            in_qrf += 1
    cov_qrf.append(in_qrf / len(yte))
    # OLS-正态
    in_ols = 0
    for i, x in enumerate(Xte):
        m = float(x @ beta_ols)
        if m - z * ols_resid_std <= yte[i] <= m + z * ols_resid_std:
            in_ols += 1
    cov_ols.append(in_ols / len(yte))
    # 线性 QR
    in_lqr = 0
    for i, x in enumerate(Xte):
        lo = linear_qr_quantile(_get_beta(lqr_betas, lo_t), x)
        hi = linear_qr_quantile(_get_beta(lqr_betas, hi_t), x)
        if lo <= yte[i] <= hi:
            in_lqr += 1
    cov_lqr.append(in_lqr / len(yte))

# 5d) 90% 区间覆盖 (主指标)
_i90 = int(np.where(np.isclose(alphas, 0.90))[0][0])
cov90_qrf = cov_qrf[_i90]
cov90_ols = cov_ols[_i90]
cov90_lqr = cov_lqr[_i90]


# 5e) CRPS
def crps_empirical(samples, y):
    n = len(samples)
    if n < 2:
        return abs(samples[0] - y)
    s = np.sort(samples)
    return np.mean(np.abs(s - y)) - 0.5 * np.mean(np.abs(s[:, None] - s[None, :]))


def crps_gaussian(mu, sigma, y):
    z = (y - mu) / sigma
    pdf = stats.norm.pdf(z)
    cdf = stats.norm.cdf(z)
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1.0 / np.sqrt(np.pi))


crps_qrf_list, crps_ols_list, crps_lqr_list = [], [], []
for i, x in enumerate(Xte):
    samp = np.concatenate([tree_leaf_vals(t, x) for t in forest])
    crps_qrf_list.append(crps_empirical(samp, yte[i]))
    crps_ols_list.append(crps_gaussian(float(x @ beta_ols), ols_resid_std, yte[i]))
    # 线性 QR: 由细网格分位构造经验分布
    qs = np.array([linear_qr_quantile(_get_beta(lqr_betas, float(t)), x) for t in taus])
    qs = np.sort(qs)
    crps_lqr_list.append(crps_empirical(qs, yte[i]))

crps_qrf = np.mean(crps_qrf_list)
crps_ols = np.mean(crps_ols_list)
crps_lqr = np.mean(crps_lqr_list)

print("="*60)
print("分位数回归森林 关键数字")
print("="*60)
print(f"点预测 RMSE : QRF={rmse_qrf:.4f}  OLS={rmse_ols:.4f}")
print(f"90% 区间经验覆盖: QRF={cov90_qrf:.3f}  OLS正态={cov90_ols:.3f}  线性QR={cov90_lqr:.3f}")
print(f"CRPS(越小越好): QRF={crps_qrf:.4f}  OLS正态={crps_ols:.4f}  线性QR={crps_lqr:.4f}")
print(f"CRPS: QRF 相对 OLS正态 改进 {100*(crps_ols-crps_qrf)/crps_ols:.1f}%")
print(f"CRPS: QRF 相对 线性QR  改进 {100*(crps_lqr-crps_qrf)/crps_lqr:.1f}%")

# ===========================================================================
# 配图 1: cover —— 不同波动状态下 QRF 预测分布带 vs 真实条件分布
# ===========================================================================
fig, ax = plt.subplots(figsize=(11, 6.2))
vv = np.linspace(0, 1, 9)
ss = 0.4
for vi, vval in enumerate(vv):
    xs = np.full(40, vval)
    pred_lo = [qrf_quantile(np.array([ss, vval]), 0.05) for _ in range(40)]
    pred_hi = [qrf_quantile(np.array([ss, vval]), 0.95) for _ in range(40)]
    pred_med = [qrf_quantile(np.array([ss, vval]), 0.50) for _ in range(40)]
    ax.fill_between([vval - 0.025, vval + 0.025],
                    np.mean(pred_lo), np.mean(pred_hi), color=C["qrf"], alpha=0.25)
    ax.plot([vval, vval], [np.mean(pred_lo), np.mean(pred_hi)], color=C["qrf"], lw=2)
    ax.plot(vval, np.mean(pred_med), "o", color=C["qrf"], ms=6)
# 真实条件尺度(参考)：sigma = 0.3+1.7v
true_med = 0.6 * ss
ax.plot(vv, true_med * np.ones_like(vv), "--", color=C["true"], lw=1.6, label="真实条件均值")
ax.plot(vv, true_med + 1.645 * (0.3 + 1.7 * vv), ":", color=C["ols"], lw=1.6,
        label="OLS-正态 90% 上界(恒定尺度)")
ax.plot(vv, true_med - 1.645 * (0.3 + 1.7 * vv), ":", color=C["ols"], lw=1.6)
ax.set_xlabel("波动状态 v (0=平静 → 1=紧张)")
ax.set_ylabel("预测收益")
ax.set_title("分位数回归森林：预测区间随波动状态自适应展宽", fontsize=14)
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "cover.png"), dpi=130)
plt.close()

# ===========================================================================
# 配图 2: 可靠性曲线 (reliability diagram)
# ===========================================================================
fig, ax = plt.subplots(figsize=(8.5, 6.2))
ax.plot(alphas, cov_qrf, "o-", color=C["qrf"], lw=2.2, ms=7, label="QRF 分位数森林")
ax.plot(alphas, cov_ols, "s-", color=C["ols"], lw=2.2, ms=7, label="OLS + 全局正态")
ax.plot(alphas, cov_lqr, "^-", color=C["lqr"], lw=2.2, ms=7, label="线性分位数回归 QR")
ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.5, label="完美校准")
ax.set_xlabel("名义置信水平 1 − α")
ax.set_ylabel("经验覆盖率")
ax.set_title("可靠性曲线：谁的中心区间覆盖对了？", fontsize=14)
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "reliability.png"), dpi=130)
plt.close()

# ===========================================================================
# 配图 3: 单点预测分布 —— QRF 经验分布 vs 正态拟合
# ===========================================================================
xq = np.array([0.4, 0.85])   # 紧张状态
samp = np.concatenate([tree_leaf_vals(t, xq) for t in forest])
samp = np.clip(samp, np.percentile(samp, 0.5), np.percentile(samp, 99.5))
fig, ax = plt.subplots(figsize=(9, 5.8))
ax.hist(samp, bins=50, density=True, color=C["qrf"], alpha=0.65,
        label=f"QRF 预测分布 (紧张状态 v={xq[1]})")
# 正态对照：OLS 均值 + 全局残差标准差
m_ols = float(xq @ beta_ols)
xs = np.linspace(samp.min(), samp.max(), 200)
ax.plot(xs, stats.norm.pdf(xs, m_ols, ols_resid_std), "--", color=C["ols"], lw=2,
        label=f"OLS 正态 (均值={m_ols:.2f}, σ={ols_resid_std:.2f} 恒定)")
ax.axvline(m_ols, color=C["ols"], lw=1)
ax.set_xlabel("次日收益")
ax.set_ylabel("概率密度")
ax.set_title("单点预测分布：QRF 给出厚左尾，正态假设掩盖它", fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "dist_single.png"), dpi=130)
plt.close()

print("✅ 配图已保存到", D)
print(["cover.png", "reliability.png", "dist_single.png"])
