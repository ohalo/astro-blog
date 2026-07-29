#!/usr/bin/env python3
"""生成 Easley-O'Hara 信息模型文章配图"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "public/images/easley-ohara-microstructure"
rng = np.random.default_rng(7)

# ---- 图1: 事件树结构 ----
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.axis("off")

def node(x, y, text, color="#deebf7"):
    ax.annotate(text, xy=(x, y), ha="center", va="center", fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.45", fc=color, ec="#555"))

def arrow(x1, y1, x2, y2, label="", dx=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.4))
    ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2, label, fontsize=9.5, color="#c0392b", ha="center")

node(0.5, 0.92, "每个交易日开始", "#fee6ce")
node(0.28, 0.68, "无信息事件", "#deebf7")
node(0.72, 0.68, "发生信息事件", "#fdd0a2")
node(0.55, 0.42, "坏消息\n(真值 = 低)", "#fcbba1")
node(0.89, 0.42, "好消息\n(真值 = 高)", "#c7e9c0")
node(0.28, 0.16, "只有不知情交易者\n买 ε_b、卖 ε_s", "#deebf7")
node(0.55, 0.10, "不知情 + 知情者\n集中卖出 (μ)", "#fcbba1")
node(0.89, 0.10, "不知情 + 知情者\n集中买入 (μ)", "#c7e9c0")

arrow(0.46, 0.88, 0.30, 0.73, "1−α", dx=-0.04)
arrow(0.54, 0.88, 0.70, 0.73, "α", dx=0.04)
arrow(0.68, 0.63, 0.57, 0.48, "δ", dx=-0.04)
arrow(0.76, 0.63, 0.87, 0.48, "1−δ", dx=0.045)
arrow(0.28, 0.63, 0.28, 0.22)
arrow(0.55, 0.36, 0.55, 0.16)
arrow(0.89, 0.36, 0.89, 0.17)
ax.set_title("Easley-O'Hara 模型的信息事件树", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/eo-event-tree.png", dpi=130)
plt.close(fig)

# ---- 图2: 事件日 vs 非事件日的订单流分布 ----
days = 2000
alpha, delta, mu, eb, es = 0.3, 0.4, 60, 50, 50
labels_ev = rng.random(days) < alpha
good = rng.random(days) < (1 - delta)
B = rng.poisson(eb, days).astype(float)
S = rng.poisson(es, days).astype(float)
B[labels_ev & good] += rng.poisson(mu, (labels_ev & good).sum())
S[labels_ev & ~good] += rng.poisson(mu, (labels_ev & ~good).sum())

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.scatter(B[~labels_ev], S[~labels_ev], s=9, alpha=0.4, color="#3182bd", label="无事件日")
ax.scatter(B[labels_ev & good], S[labels_ev & good], s=9, alpha=0.5, color="#31a354", label="好消息日（买单激增）")
ax.scatter(B[labels_ev & ~good], S[labels_ev & ~good], s=9, alpha=0.5, color="#e6550d", label="坏消息日（卖单激增）")
ax.set_xlabel("日买单数 B")
ax.set_ylabel("日卖单数 S")
ax.set_title("模拟 2000 个交易日：三种状态在订单流平面上自然分簇")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/eo-orderflow-clusters.png", dpi=130)
plt.close(fig)

# ---- 图3: 价差随 PIN 变化 ----
alphas = np.linspace(0.05, 0.6, 100)
pin = alphas * mu / (alphas * mu + eb + es)
VH, VL = 101.0, 99.0
# 开盘价差近似（对称 delta=0.5）
spread = pin * (VH - VL)

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(pin, spread / 100 * 10000, lw=2.4, color="#e6550d")
ax.set_xlabel("知情交易概率 PIN")
ax.set_ylabel("模型隐含开盘价差 (bp)")
ax.set_title("信息风险直接定价：PIN 越高，做市商挂出的价差越宽")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/eo-pin-spread.png", dpi=130)
plt.close(fig)

# ---- 图4: 事件日内做市商贝叶斯更新报价 ----
T = 200
is_bad = True
p_good = np.zeros(T + 1)
p_event = np.zeros(T + 1)
p_event[0] = alpha
p_good[0] = 1 - delta
# 到达强度
lam_b_g = eb + mu; lam_s_g = es
lam_b_b = eb; lam_s_b = es + mu
lam_b_n = eb; lam_s_n = es

# 状态后验: (no, good, bad)
post = np.array([1 - alpha, alpha * (1 - delta), alpha * delta])
mid = [post[0] * 100 + post[1] * VH + post[2] * VL + 100 * 0]
mid = [(post[0] * 100 + post[1] * VH + post[2] * VL)]
trades = []
for i in range(T):
    tot = np.array([lam_b_n + lam_s_n, lam_b_g + lam_s_g, lam_b_b + lam_s_b])
    # 真实状态 bad: 生成交易
    pb = lam_b_b / (lam_b_b + lam_s_b)
    is_buy = rng.random() < pb
    if is_buy:
        lik = np.array([lam_b_n, lam_b_g, lam_b_b]) / tot
    else:
        lik = np.array([lam_s_n, lam_s_g, lam_s_b]) / tot
    post = post * lik
    post /= post.sum()
    mid.append(post[0] * 100 + post[1] * VH + post[2] * VL)

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(mid, lw=1.8, color="#3182bd", label="做市商期望价值（中间价）")
ax.axhline(VL, color="#e6550d", ls="--", lw=1.4, label="坏消息真值 99.0")
ax.axhline(100, color="gray", ls=":", lw=1.2, label="无条件期望 100.0")
ax.set_xlabel("成交笔数")
ax.set_ylabel("价格")
ax.set_title("坏消息日：做市商通过订单流贝叶斯学习，报价逐步收敛到真值")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/eo-bayesian-quotes.png", dpi=130)
plt.close(fig)

print("EO images done")
