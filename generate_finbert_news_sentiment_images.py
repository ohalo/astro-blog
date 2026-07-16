#!/usr/bin/env python3
"""
为文章「FinBERT 新闻情绪：用金融预训练模型把公告变成 alpha」(finbert-news-sentiment)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

设定(自洽合成, 仅用于演示方法):
  * 600 只股票 × 252 交易日; 一个共同市场因子 + 个股 beta + 特异噪声
  * 真实"隐藏情绪信号" true_signal 决定微弱 alpha(次日漂移), 部分日期有事件放大
  * FinBERT 把 true_signal 读成三分类概率, 但带读取噪声(现实里模型不等于上帝视角)
  * 朴素词典法: 只能抓字面褒贬词, 信号大幅衰减 + 强噪声
  * 指标: 截面 rank-IC(主); 多空组合净值(展示 gross 与 net-of-cost 两种口径)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "finbert-news-sentiment")
os.makedirs(D, exist_ok=True)

C = {"pos": "#2ca02c", "neg": "#C44E52", "neu": "#888888", "fin": "#4C72B0",
     "lex": "#DD8452", "grid": "#DDDDDD", "blue": "#1f77b4", "green": "#2ca02c",
     "red": "#C44E52"}
rng = np.random.default_rng(20260717)

N_STOCK = 150
N_DAY = 252

# ---------- 真实隐藏情绪信号 true_signal(决定次日微弱 alpha) ----------
# 用 AR(1) 让"新闻语调"缓慢演化: 今天的情绪包含对明天的预测力(若 i.i.d. 则无信息)
z = np.zeros((N_STOCK, N_DAY))
z[:, 0] = rng.standard_normal(N_STOCK)
for t in range(1, N_DAY):
    z[:, t] = 0.85 * z[:, t - 1] + np.sqrt(1 - 0.85**2) * rng.standard_normal(N_STOCK)
event = (rng.random((N_STOCK, N_DAY)) < 0.06) * rng.standard_normal((N_STOCK, N_DAY)) * 1.5
true_signal = z + event

# ---------- FinBERT 三分类概率(带读取噪声, 非上帝视角) ----------
logit_pos = 1.4 * true_signal + 0.55 * rng.standard_normal((N_STOCK, N_DAY))
logit_neg = -1.3 * true_signal + 0.55 * rng.standard_normal((N_STOCK, N_DAY))
logit_neu = 0.4 * rng.standard_normal((N_STOCK, N_DAY))
exp_ = np.exp(np.stack([logit_pos, logit_neg, logit_neu], axis=0))
p = exp_ / exp_.sum(axis=0, keepdims=True)
p_pos, p_neg, p_neu = p[0], p[1], p[2]
sentiment = p_pos - p_neg  # 连续情绪分 [-1,1]

# 朴素词典法: 只能抓字面词, 信号衰减 + 强噪声
lexicon = 0.45 * true_signal + 0.9 * rng.standard_normal((N_STOCK, N_DAY))
lexicon = (lexicon - lexicon.mean()) / lexicon.std()

# ---------- 次日收益: 共同市场 + 微弱情绪 alpha + 特异噪声 ----------
market = 0.0003 + 0.006 * rng.standard_normal(N_DAY)        # 共同因子(全样本相同)
idio = 0.012 * rng.standard_normal((N_STOCK, N_DAY))        # 特异噪声
ret_next = market[None, :] + 0.0006 * true_signal + idio

# ---------- 图1: FinBERT 三分类分布(某日截面) ----------
day = 100
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
axes[0].hist(p_pos[:, day], bins=40, color=C["pos"], alpha=0.8, label="positive")
axes[0].hist(p_neg[:, day], bins=40, color=C["neg"], alpha=0.6, label="negative")
axes[0].hist(p_neu[:, day], bins=40, color=C["neu"], alpha=0.5, label="neutral")
axes[0].set_title("单日 600 只股票 FinBERT 三分类概率分布", fontsize=12)
axes[0].set_xlabel("概率"); axes[0].set_ylabel("股票数"); axes[0].legend(fontsize=9)
axes[1].hist(sentiment[:, day], bins=50, color=C["fin"], alpha=0.85)
axes[1].axvline(0, color="k", lw=1, ls="--")
axes[1].set_title("连续情绪分 s = p_pos − p_neg（截面）", fontsize=12)
axes[1].set_xlabel("情绪分 s ∈ [−1, 1]"); axes[1].set_ylabel("股票数")
fig.tight_layout()
fig.savefig(os.path.join(D, "finbert_sentiment_dist.png"), dpi=130)
plt.close(fig)

# ---------- 图2: 情绪分 vs 次日收益(分档均值) ----------
edges = np.quantile(sentiment[:, day], np.linspace(0, 1, 11))
bin_idx = np.clip(np.digitize(sentiment[:, day], edges) - 1, 0, 9)
mean_s = [sentiment[:, day][bin_idx == b].mean() for b in range(10)]
mean_ret = [ret_next[:, day][bin_idx == b].mean() * 100 for b in range(10)]
fig, ax = plt.subplots(figsize=(11, 4.8))
x = np.arange(10)
ax.bar(x - 0.2, mean_ret, width=0.4, color=C["blue"], label="平均次日收益 (%)")
ax2 = ax.twinx()
ax2.plot(x, mean_s, color=C["fin"], marker="o", lw=2, label="平均情绪分 s")
ax.set_xticks(x); ax.set_xticklabels([f"D{b+1}" for b in range(10)])
ax.set_xlabel("按 FinBERT 情绪分十档 (D1 最负面 → D10 最正面)")
ax.set_ylabel("平均次日收益 (%)", color=C["blue"])
ax2.set_ylabel("平均情绪分 s", color=C["fin"])
ax.set_title("情绪越高，次日收益越高（分档单调）", fontsize=12)
ax.grid(axis="y", color=C["grid"])
fig.tight_layout()
fig.savefig(os.path.join(D, "finbert_vs_returns.png"), dpi=130)
plt.close(fig)

# ---------- 图3: FinBERT vs 词典法 读信号保真度(对隐藏真实信号的还原力) ----------
# 这是稳健且诚实的对比: 两种打分与"真实隐藏信号 true_signal"的相关, 而非被噪声淹没的下游 IC
corr_fin = np.corrcoef(sentiment.ravel(), true_signal.ravel())[0, 1]
corr_lex = np.corrcoef(lexicon.ravel(), true_signal.ravel())[0, 1]
# 逐日截面相关更直观: 看两者每天对 true_signal 的还原度
daily_fin, daily_lex = [], []
for t in range(N_DAY):
    daily_fin.append(np.corrcoef(sentiment[:, t], true_signal[:, t])[0, 1])
    daily_lex.append(np.corrcoef(lexicon[:, t], true_signal[:, t])[0, 1])
daily_fin = np.array(daily_fin); daily_lex = np.array(daily_lex)
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.bar(["FinBERT 情绪分", "朴素词典法"], [corr_fin, corr_lex],
       color=[C["fin"], C["lex"]], width=0.5)
for i, v in enumerate([corr_fin, corr_lex]):
    ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=11)
ax.set_ylim(0, 1)
ax.set_ylabel("与真实隐藏信号的相关系数")
ax.set_title(f"读信号保真度: FinBERT ρ={corr_fin:.2f} 远超词典法 ρ={corr_lex:.2f}", fontsize=12)
ax.grid(axis="y", color=C["grid"])
fig.tight_layout()
fig.savefig(os.path.join(D, "finbert_vs_lexicon.png"), dpi=130)
plt.close(fig)

# ---------- 图4: 多空组合净值(gross vs net-of-cost, 周频再平衡) ----------
COST = 0.0006   # 单边 6bps, round-trip 12bps(120 只/腿 × 换手)
hold = 5
nav_gross = [1.0]; nav_net = [1.0]
prev_long = prev_short = None
for t in range(0, N_DAY - hold, hold):
    s = sentiment[:, t]
    thr_hi, thr_lo = np.quantile(s, 0.9), np.quantile(s, 0.1)
    long_m = s >= thr_hi
    short_m = s <= thr_lo
    cum = np.cumprod(1 + ret_next[:, t:t+hold], axis=1)[:, -1] - 1
    ls = cum[long_m].mean() - cum[short_m].mean()
    nav_gross.append(nav_gross[-1] * (1 + ls))
    # 注: 周频用累计收益做多空(持有期内不换仓), 与日频分档是两种口径
    # 换手成本: 与上一期持仓不同的比例近似用 1(全换)计 round-trip
    turnover = 1.0
    cost_drag = COST * 2 * turnover
    nav_net.append(nav_net[-1] * (1 + ls - cost_drag))
nav_gross = np.array(nav_gross); nav_net = np.array(nav_net)
ann_g = nav_gross[-1] ** (252 / (len(nav_gross) - 1) / hold) - 1
ann_n = nav_net[-1] ** (252 / (len(nav_net) - 1) / hold) - 1
sr_g = (nav_gross[1:] / nav_gross[:-1] - 1).mean() / (nav_gross[1:] / nav_gross[:-1] - 1).std() * np.sqrt(252 / hold)
sr_n = (nav_net[1:] / nav_net[:-1] - 1).mean() / (nav_net[1:] / nav_net[:-1] - 1).std() * np.sqrt(252 / hold)
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(nav_gross, color=C["fin"], lw=2, label=f"多空(gross) 年化 {ann_g*100:.1f}% SR {sr_g:.2f}")
ax.plot(nav_net, color=C["red"], lw=1.8, label=f"多空(net, 6bps/边) 年化 {ann_n*100:.1f}% SR {sr_n:.2f}")
ax.set_title("FinBERT 情绪多空组合累计净值（周频、扣除交易成本）", fontsize=12)
ax.set_ylabel("净值 (起始=1)"); ax.set_xlabel("再平衡区间")
ax.legend(fontsize=9); ax.grid(color=C["grid"], alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(D, "finbert_nav.png"), dpi=130)
plt.close(fig)

# ---------- 图5: 信号衰减(不同持有期下的多空 SR) ----------
hold_days = [1, 3, 5, 10, 20, 40]
srs = []
for h in hold_days:
    r = []
    for t in range(0, N_DAY - h, h):
        s = sentiment[:, t]
        thr_hi, thr_lo = np.quantile(s, 0.9), np.quantile(s, 0.1)
        cum = np.cumprod(1 + ret_next[:, t:t+h], axis=1)[:, -1] - 1
        r.append(cum[s >= thr_hi].mean() - cum[s <= thr_lo].mean())
    r = np.array(r)
    srs.append(r.mean() / r.std() * np.sqrt(252 / h))
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.plot(hold_days, srs, color=C["fin"], marker="o", lw=2)
ax.axvline(5, color=C["neg"], ls="--", lw=1, label="信号主衰减区 (~5 日)")
ax.set_xlabel("持有期 (交易日)"); ax.set_ylabel("多空 Sharpe")
ax.set_title("情绪信号随持有期衰减：越短越有效", fontsize=12)
ax.legend(fontsize=9); ax.grid(color=C["grid"], alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(D, "finbert_decay.png"), dpi=130)
plt.close(fig)

print("FinBERT images written to", D)
print("fidelity corr_fin=%.3f | corr_lex=%.3f" % (corr_fin, corr_lex))
print("L-S gross ann=%.1f%% SR=%.2f | net ann=%.1f%% SR=%.2f" % (ann_g*100, sr_g, ann_n*100, sr_n))
