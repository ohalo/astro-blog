#!/usr/bin/env python3
"""Meta-Labeling 实验：一级信号(MA金叉) + 二级模型(逻辑回归)过滤与仓位分配"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)
N_ASSETS = 24
OUT = "/Users/halo/workspace/astro-blog/public/images/meta-labeling-strategy"
os.makedirs(OUT, exist_ok=True)

# ---------- 1. 合成市场：regime 切换（趋势 / 震荡），多资产 ----------
N = 3000

def make_asset(rng):
    regime = np.zeros(N, dtype=int)  # 0=震荡, 1=趋势
    p_stay = 0.995
    for t in range(1, N):
        regime[t] = regime[t - 1] if rng.random() < p_stay else 1 - regime[t - 1]
    drift = np.where(regime == 1, 0.0013, 0.0)
    vol = np.where(regime == 1, 0.009, 0.016)
    trend_dir = np.zeros(N)
    cur = rng.choice([-1.0, 1.0])
    for t in range(N):
        if t > 0 and regime[t] == 1 and regime[t - 1] == 0:
            cur = rng.choice([-1.0, 1.0])
        trend_dir[t] = cur if regime[t] == 1 else 0.0
    # 震荡段是 OU 式均值回复：价格被拉回锚点，金叉追涨必被反噬
    log_p = np.zeros(N)
    anchor = 0.0
    for t in range(1, N):
        if regime[t] == 0 and regime[t - 1] == 1:
            anchor = log_p[t - 1]  # 进入震荡段时锁定锚点
        if regime[t] == 1:
            log_p[t] = log_p[t - 1] + trend_dir[t] * drift[t] + vol[t] * rng.standard_normal()
        else:
            log_p[t] = log_p[t - 1] + 0.05 * (anchor - log_p[t - 1]) + vol[t] * rng.standard_normal()
    ret = np.diff(log_p, prepend=0.0)
    price = 100 * np.exp(log_p)
    return price, ret, regime

assets = [make_asset(rng) for _ in range(N_ASSETS)]
price, ret, regime = assets[0]  # 第一只用于绘图

# ---------- 2. 一级模型：MA20/MA60 金叉 ----------
def sma(x, w):
    out = np.full_like(x, np.nan, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out

HOLD = 20

def extract_signals(price, ret):
    ma_f, ma_s = sma(price, 20), sma(price, 60)
    ma_120 = sma(price, 120)
    cross_up = np.zeros(N, dtype=bool)
    cross_up[1:] = (ma_f[1:] > ma_s[1:]) & (ma_f[:-1] <= ma_s[:-1]) & ~np.isnan(ma_s[1:]) & ~np.isnan(ma_s[:-1])
    sig_idx = np.where(cross_up)[0]
    sig_idx = sig_idx[(sig_idx < N - HOLD - 5) & (sig_idx >= 130)]
    rows = []
    for i in sig_idx:
        entry = price[i + 1]  # 次日入场
        path = price[i + 1:i + 1 + HOLD] / entry - 1
        sigma20 = np.std(ret[max(0, i - 20):i]) if i >= 20 else np.std(ret[:i + 1])
        up_b, dn_b = 2.0 * sigma20 * np.sqrt(HOLD), -2.0 * sigma20 * np.sqrt(HOLD)
        final = path[-1]
        for p in path:
            if p >= up_b:
                final = p; break
            if p <= dn_b:
                final = p; break
        y = 1 if final > 0 else 0
        f_vol = sigma20
        f_slope = (ma_s[i] - ma_s[i - 10]) / ma_s[i - 10] if i >= 10 and not np.isnan(ma_s[i - 10]) else 0
        f_mom = price[i] / price[i - 5] - 1 if i >= 5 else 0
        f_dist = (ma_f[i] - ma_s[i]) / ma_s[i]
        f_range = (price[max(0, i - 20):i + 1].max() / price[max(0, i - 20):i + 1].min() - 1)
        # Kaufman 效率比：净位移 / 路径总长，趋势市接近 1，震荡市接近 0
        lo = max(0, i - 30)
        seg = price[lo:i + 1]
        f_er = abs(seg[-1] - seg[0]) / (np.abs(np.diff(seg)).sum() + 1e-12)
        # 长动量：60/120日，以及与 120 日均线的距离（区分大趋势方向）
        f_mom60 = price[i] / price[i - 60] - 1
        f_mom120 = price[i] / price[i - 120] - 1
        f_dist120 = (price[i] - ma_120[i]) / ma_120[i]
        # 交互特征：趋势市（低波动）里大趋势方向才重要 —— 把两个弱信号乘起来
        # 只保留有真实判别力的少数特征，避免噂声特征淹没信号
        is_trendy = 1.0 if f_vol < 0.013 else 0.0  # 低波动 ≈ 趋势市
        f_inter = is_trendy * np.sign(f_mom120)
        rows.append((i, y, [f_vol, f_mom120, f_dist120, f_inter], final))
    return rows, ma_f, ma_s, sig_idx

all_rows = []
for a, (p_, r_, _) in enumerate(assets):
    rows, *_ = extract_signals(p_, r_)
    for (i, y_, f_, fr_) in rows:
        all_rows.append((i, y_, f_, fr_))
# 按信号发生时间排序，保证走前式切分不泄未来
all_rows.sort(key=lambda x: x[0])
X = np.array([f for _, _, f, _ in all_rows])
y = np.array([y_ for _, y_, _, _ in all_rows])
r = np.array([fr for _, _, _, fr in all_rows])
print(f"一级信号总数({N_ASSETS}只资产): {len(y)}")
print(f"胜率(一级信号全接): {y.mean():.3f}")

# 绘图用第一只资产
_, ma_f, ma_s, sig_idx = extract_signals(price, ret)

# ---------- 4. 二级模型：走前式（前60%训练，后40%测试）逻辑回归 ----------
split = int(len(y) * 0.6)
Xtr, Xte, ytr, yte, rte = X[:split], X[split:], y[:split], y[split:], r[split:]

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
clf.fit(Xtr, ytr)
p_te = clf.predict_proba(Xte)[:, 1]

from sklearn.metrics import roc_auc_score
print(f"AUC(测试段): {roc_auc_score(yte, p_te):.3f}")
# 分位数分桶胜率
qs = np.quantile(p_te, [0, .25, .5, .75, 1])
for lo, hi in zip(qs[:-1], qs[1:]):
    m = (p_te >= lo) & (p_te <= hi)
    print(f"  p∈[{lo:.2f},{hi:.2f}] n={m.sum()} 胜率={yte[m].mean():.3f} 均收益={rte[m].mean():+.4f}")

# ---------- 5. 指标对比 ----------
def prf(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return prec, rec, f1

base_pred = np.ones_like(yte)
thr = 0.5
meta_pred = (p_te >= thr).astype(int)
p0 = prf(yte, base_pred)
p1 = prf(yte, meta_pred)
print(f"一级全接  precision={p0[0]:.3f} recall={p0[1]:.3f} F1={p0[2]:.3f}")
print(f"元标注过滤 precision={p1[0]:.3f} recall={p1[1]:.3f} F1={p1[2]:.3f}")

# 权益曲线（测试段）：全接 vs 过滤 vs 概率仓位
eq_base = np.cumprod(1 + rte)
eq_filt = np.cumprod(1 + rte * meta_pred)
size = np.clip((p_te - 0.45) / 0.2, 0, 1)  # 概率->仓位：p≤0.45 空仓，p≥0.65 满仓
eq_size = np.cumprod(1 + rte * size)
print(f"测试段交易数: {len(rte)}, 过滤后: {meta_pred.sum()}")
print(f"全接终值 {eq_base[-1]:.3f}  过滤终值 {eq_filt[-1]:.3f}  概率仓位终值 {eq_size[-1]:.3f}")

def sharpe(x):
    if x.std() == 0: return 0
    return x.mean() / x.std() * np.sqrt(252 / HOLD)
print(f"Sharpe: 全接 {sharpe(rte):.2f} | 过滤 {sharpe(rte*meta_pred):.2f} | 仓位 {sharpe(rte*size):.2f}")

# ---------- 图1：价格与信号 ----------
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax[0].plot(price, lw=0.7, color="#334155", label="价格")
ax[0].plot(ma_f, lw=0.8, color="#0ea5e9", alpha=0.8, label="MA20")
ax[0].plot(ma_s, lw=0.8, color="#f59e0b", alpha=0.8, label="MA60")
ax[0].scatter(sig_idx, price[sig_idx], marker="^", color="#dc2626", s=28, zorder=5, label="金叉信号")
ax[0].legend(loc="upper left", fontsize=9)
ax[0].set_title("合成行情（趋势/震荡切换）与一级模型 MA 金叉信号")
ax[1].fill_between(range(N), regime, step="pre", color="#22c55e", alpha=0.5)
ax[1].set_ylabel("regime")
ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["震荡", "趋势"])
plt.tight_layout(); plt.savefig(f"{OUT}/signals-regime.png", dpi=110); plt.close()

# ---------- 图2：precision/recall/F1 对比 ----------
fig, ax = plt.subplots(figsize=(8, 4.5))
xpos = np.arange(3); wd = 0.35
ax.bar(xpos - wd/2, p0, wd, label="一级模型全接单", color="#94a3b8")
ax.bar(xpos + wd/2, p1, wd, label="元标注过滤 (p≥0.5)", color="#0ea5e9")
for i, (a0, a1) in enumerate(zip(p0, p1)):
    ax.text(i - wd/2, a0 + 0.01, f"{a0:.2f}", ha="center", fontsize=9)
    ax.text(i + wd/2, a1 + 0.01, f"{a1:.2f}", ha="center", fontsize=9)
ax.set_xticks(xpos); ax.set_xticklabels(["Precision", "Recall", "F1"])
ax.set_ylim(0, 1); ax.legend(); ax.set_title("测试段：元标注用 Recall 换 Precision")
plt.tight_layout(); plt.savefig(f"{OUT}/precision-recall.png", dpi=110); plt.close()

# ---------- 图3：权益曲线 ----------
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(eq_base, label=f"全接单 (终值 {eq_base[-1]:.2f})", color="#94a3b8", lw=1.4)
ax.plot(eq_filt, label=f"元标注过滤 (终值 {eq_filt[-1]:.2f})", color="#0ea5e9", lw=1.4)
ax.plot(eq_size, label=f"概率仓位 (终值 {eq_size[-1]:.2f})", color="#dc2626", lw=1.4)
ax.set_title("测试段逐笔复利权益：过滤与概率仓位 vs 全接单")
ax.set_xlabel("测试段交易序号"); ax.set_ylabel("权益 (初始=1)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/equity-compare.png", dpi=110); plt.close()

# ---------- 图4：概率校准/仓位 ----------
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
bins = np.linspace(0, 1, 8)
mids, obs = [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (p_te >= lo) & (p_te < hi)
    if m.sum() >= 5:
        mids.append((lo + hi) / 2); obs.append(yte[m].mean())
ax[0].plot([0, 1], [0, 1], "--", color="#94a3b8")
ax[0].plot(mids, obs, "o-", color="#0ea5e9")
ax[0].set_xlabel("预测胜率"); ax[0].set_ylabel("实际胜率"); ax[0].set_title("可靠性曲线（测试段）")
ax[1].hist(size, bins=20, color="#0ea5e9", alpha=0.8)
ax[1].set_xlabel("仓位 (clip(2p-1, 0, 1))"); ax[1].set_ylabel("笔数"); ax[1].set_title("概率映射出的仓位分布")
plt.tight_layout(); plt.savefig(f"{OUT}/calibration-sizing.png", dpi=110); plt.close()

print("图已生成:", os.listdir(OUT))
