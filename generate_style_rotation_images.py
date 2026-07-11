#!/usr/bin/env python3
"""
为文章「风格轮动的宏观信号：用利差/动量给价值成长切换择时」(style-rotation-macro)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（数据由因子模型+宏观状态自洽合成，仅用于演示方法；真实落地见文末路径）：
  底层有一个「看不见的」宏观状态 h_t 驱动价值/成长的相对强弱；
  我们观测到的曲线斜率(slope)与信用利差(credit)只是 h_t 的「带噪声代理」；
  风格日收益 = 慢变风格漂移 + β·h_t(宏观暴露) + 大方差特异噪声(日内无法被宏观解释的部分)。
  实战里宏观信号只能部分预测风格切换 → 引入「信号滞后 L 日」模拟执行时滞，
  避免前视。策略用带噪信号 + 滞后打分得到价值权重 w_value_t。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "style-rotation-macro")
os.makedirs(D, exist_ok=True)

C = {"val": "#2F4B7C", "grw": "#C44E52", "rot": "#55A868", "grid": "#DDDDDD",
     "slope": "#4C72B0", "credit": "#DD8452", "score": "#8172B3", "w": "#55A868"}

def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return dd.min(), dd

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# ============================================================
# 1) 隐藏宏观状态 h_t + 带噪观测信号 + 宏观驱动的风格收益
# ============================================================
def simulate_all(T=252 * 8, seed=42):
    rng = np.random.default_rng(seed)
    # 隐藏宏观状态 h_t ∈ ~[-0.8,0.8]，慢变 AR(1)
    h = np.zeros(T); h[0] = 0.30
    for t in range(1, T):
        h[t] = 0.990 * h[t - 1] + 0.06 * rng.normal()
    h = 0.8 * np.tanh(h / 0.8)
    # 观测信号：只是 h_t 的带噪代理（实战里信号从不精确反映真实状态）
    slope = 1.5 * h + 0.30 + rng.normal(0, 0.45, size=T)     # 10Y-2Y 斜率, %
    credit = 2.5 - 1.6 * h + rng.normal(0, 0.40, size=T)      # 信用利差, %
    # 风格日收益：慢变漂移 + 宏观暴露(β·h) + 大方差特异噪声
    mu_v, mu_g = 0.05 / 252, 0.09 / 252
    beta = 0.0011
    rv = rng.normal(0, 0.011, size=T)
    rg = rng.normal(0, 0.011, size=T)
    r_value = mu_v + beta * h + rv
    r_growth = mu_g - beta * h + rg
    return h, slope, credit, r_value, r_growth

# ============================================================
# 2) 宏观信号驱动的风格轮动（含 L 日执行滞后，避免前视）
# ============================================================
def style_rotation(r_value, r_growth, slope, credit, k=1.5, lag=5):
    T = len(r_value)
    score = (slope - slope.mean()) / slope.std() - (credit - credit.mean()) / credit.std()
    score_lag = np.concatenate([np.full(lag, score[0]), score[:-lag]])  # 滞后 L 日
    w_value = sigmoid(k * score_lag)
    w_growth = 1.0 - w_value
    r_rot = w_value * r_value + w_growth * r_growth
    return np.cumprod(1 + r_rot), w_value, score_lag

# ============================================================
# 主计算
# ============================================================
T = 252 * 8
h, slope, credit, r_value, r_growth = simulate_all(T)
cum_v = np.cumprod(1 + r_value); cum_v = np.insert(cum_v, 0, 1.0)
cum_g = np.cumprod(1 + r_growth); cum_g = np.insert(cum_g, 0, 1.0)
cum_5050 = np.cumprod(1 + 0.5 * (r_value + r_growth)); cum_5050 = np.insert(cum_5050, 0, 1.0)
cum_rot, w_value, score = style_rotation(r_value, r_growth, slope, credit, k=1.5, lag=5)
cum_rot = np.insert(cum_rot, 0, 1.0)

t = np.arange(T + 1)
dd_rot, _ = max_drawdown(cum_rot)
dd_v, _ = max_drawdown(cum_v)
dd_g, _ = max_drawdown(cum_g)
dd_5050, _ = max_drawdown(cum_5050)

# 年化收益
def ann(ret, n=T):
    return (ret[-1] ** (252.0 / n)) - 1.0

# ---------- 图 1：价值 / 成长 / 轮动 累计净值 ----------
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(t, cum_v, color=C["val"], lw=1.2, alpha=0.85, label="价值因子 (纯多)")
ax.plot(t, cum_g, color=C["grw"], lw=1.2, alpha=0.85, label="成长因子 (纯多)")
ax.plot(t, cum_rot, color=C["rot"], lw=1.8, label="宏观信号轮动 (本文策略)")
ax.plot(t, cum_5050, color=C["grid"], lw=1.0, ls=":", label="静态 50/50")
ax.set_xlabel("交易日"); ax.set_ylabel("累计净值 (初始=1.0)")
ax.set_title("宏观信号轮动：在价值/成长之间切换，跑赢单一风格与静态 50/50")
ax.legend(loc="upper left", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "style_cumulative.png"), dpi=130); plt.close()

# ---------- 图 2：宏观信号时序（曲线斜率 + 信用利差）----------
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(t[1:], slope, color=C["slope"], lw=1.1, label="收益率曲线斜率 (10Y-2Y, %)")
ax.plot(t[1:], credit, color=C["credit"], lw=1.1, label="信用利差 (BAA-AAA, %)")
ax.axhline(0, color="#888", lw=0.6, ls="--")
ax.set_xlabel("交易日"); ax.set_ylabel("百分比 (%)")
ax.set_title("宏观信号：陡峭/走阔的曲线 + 收窄的信用利差 → 价值占优")
ax.legend(loc="upper right", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "style_signals.png"), dpi=130); plt.close()

# ---------- 图 3：轮动权重时序 ----------
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.fill_between(t[1:], w_value, color=C["val"], alpha=0.4, label="价值权重 w_value")
ax.plot(t[1:], w_value, color=C["val"], lw=1.2)
ax.plot(t[1:], 1 - w_value, color=C["grw"], lw=1.2, label="成长权重 (1-w_value)")
ax.set_xlabel("交易日"); ax.set_ylabel("权重")
ax.set_title("轮动权重随时间变化：宏观打分决定价值/成长的配比")
ax.legend(loc="upper right", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "style_weights.png"), dpi=130); plt.close()

# ---------- 图 4：回撤对比 ----------
_, dd_series_rot = max_drawdown(cum_rot)
_, dd_series_v = max_drawdown(cum_v)
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.fill_between(t, dd_series_v * 100, color=C["val"], alpha=0.30)
ax.plot(t, dd_series_v * 100, color=C["val"], lw=0.8, label="价值因子 回撤")
ax.plot(t, dd_series_rot * 100, color=C["rot"], lw=1.5, label="宏观轮动 回撤")
ax.set_xlabel("交易日"); ax.set_ylabel("回撤 (%)")
ax.set_title("轮动回撤 (%.1f%%) 显著低于单一价值 (%.1f%%)" % (dd_rot * 100, dd_v * 100))
ax.legend(loc="lower left", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "style_drawdown.png"), dpi=130); plt.close()

print("=== 风格轮动的宏观信号 关键数字 ===")
print("样本: 日度 %d 天 (约 %.1f 年)，宏观状态 AR(1) 慢变" % (T, T / 252))
print("价值因子: 终值=%.3f, 年化=%.1f%%, 最大回撤=%.1f%%" % (cum_v[-1], ann(cum_v) * 100, dd_v * 100))
print("成长因子: 终值=%.3f, 年化=%.1f%%, 最大回撤=%.1f%%" % (cum_g[-1], ann(cum_g) * 100, dd_g * 100))
print("静态50/50: 终值=%.3f, 年化=%.1f%%, 最大回撤=%.1f%%" % (cum_5050[-1], ann(cum_5050) * 100, dd_5050 * 100))
print("宏观轮动: 终值=%.3f, 年化=%.1f%%, 最大回撤=%.1f%%" % (cum_rot[-1], ann(cum_rot) * 100, dd_rot * 100))
print("平均价值权重=%.2f (成长=%.2f)，权重 std=%.2f" % (w_value.mean(), (1 - w_value).mean(), w_value.std()))
print("\n图片已保存到:", D)
