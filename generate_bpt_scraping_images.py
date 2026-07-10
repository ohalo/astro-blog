#!/usr/bin/env python3
"""
为两篇新量化文章生成真实配图（matplotlib 渲染，非占位图）：
  1. behavioral-portfolio-theory  (行为组合理论：心智账户金字塔 vs 均值方差前沿)
  2. web-scraping-text-mining      (网页爬虫与文本挖掘：从抓取到情绪信号)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches

rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"


# ============================================================
# 文章 1：行为组合理论
# ============================================================
d1 = os.path.join(BASE, "behavioral-portfolio-theory")
os.makedirs(d1, exist_ok=True)

# ---------- 图1：心智账户金字塔 ----------
fig, ax = plt.subplots(figsize=(9, 7.5))
layers = [
    ("投机层 / 致富", "小概率博取极高收益", "预期收益 +25%~+∞", "个股 / 期权 / 加密货币", "#E45756"),
    ("成长层 / 增值", "跑赢通胀、长期资本增值", "预期收益 +8%~12%", "股票 / 股票基金", "#F58518"),
    ("收入层 / 安稳", "稳定现金流覆盖生活", "预期收益 +4%~6%", "高股息 / 债基 / REITs", "#54A24B"),
    ("安全层 / 保本", "绝对不能亏的底线", "预期收益 +1%~2%", "货币基金 / 国债 / 存款", "#4C78A8"),
]
# 金字塔：从上到下，塔尖最窄
top_y = 4.0
heights = [0.9, 0.95, 1.05, 1.2]
half_widths = [0.9, 1.5, 2.1, 2.8]
y = 0.0
poly_patches = []
for i, (name, goal, ret, asset, color) in enumerate(layers):
    hw_bottom = half_widths[3 - i]
    hw_top = half_widths[3 - i - 1] if i < 3 else 0.0
    yb = y
    yt = y + heights[3 - i]
    if i == 3:  # 最底层是矩形（塔基）
        poly = Polygon([(-hw_bottom, yb), (hw_bottom, yb),
                        (hw_bottom, yt), (-hw_bottom, yt)], closed=True,
                       facecolor=color, alpha=0.85, edgecolor="white", lw=2)
    else:
        poly = Polygon([(-hw_bottom, yb), (hw_bottom, yb),
                        (hw_top, yt), (-hw_top, yt)], closed=True,
                       facecolor=color, alpha=0.85, edgecolor="white", lw=2)
    ax.add_patch(poly)
    # 文字
    ax.text(0, (yb + yt) / 2, f"{name}\n{goal}\n{ret} · {asset}",
            ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
    y = yt

ax.set_xlim(-3.4, 3.4)
ax.set_ylim(-0.2, 5.2)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("行为组合理论的核心：财富被切分为多层「心智账户」金字塔\n（Shefrin & Statman, 2000）",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(d1, "bpt_pyramid.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------- 图2：行为组合前沿 vs 均值方差有效前沿 ----------
np.random.seed(20260710)
n_assets = 6
names = ["安全资产", "债券", "高股息", "宽基股", "成长股", "投机标的"]
# 各资产年化预期收益与波动
mu = np.array([0.02, 0.04, 0.06, 0.09, 0.12, 0.20])
sig = np.array([0.01, 0.05, 0.10, 0.16, 0.24, 0.45])
corr = np.array([
    [1.00, 0.30, 0.10, -0.05, -0.10, -0.15],
    [0.30, 1.00, 0.45, 0.20, 0.10, 0.00],
    [0.10, 0.45, 1.00, 0.55, 0.40, 0.20],
    [-0.05, 0.20, 0.55, 1.00, 0.75, 0.45],
    [-0.10, 0.10, 0.40, 0.75, 1.00, 0.65],
    [-0.15, 0.00, 0.20, 0.45, 0.65, 1.00],
])
cov = np.outer(sig, sig) * corr

# 蒙特卡洛随机组合 -> 均值方差云
mc = 8000
w = np.random.dirichlet(np.ones(n_assets), mc)
p_mu = w @ mu
p_sig = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))
idx = np.argsort(p_sig)
sorted_sig = p_sig[idx]
sorted_mu = p_mu[idx]
# 近似有效前沿（对每个风险水平取最大收益）
front_sig, front_mu = [], []
step = 0.004
s = step
while s < sorted_sig.max():
    mask = (sorted_sig >= s) & (sorted_sig < s + step)
    if mask.any():
        front_sig.append(s + step / 2)
        front_mu.append(sorted_mu[mask].max())
    s += step

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(p_sig, p_mu, s=6, color="#BFBFBF", alpha=0.5, label="随机组合 (可行域)")
ax.plot(front_sig, front_mu, color="#4C78A8", lw=2.6, label="均值方差有效前沿 (Markowitz)")

# 行为金字塔组合：按各层目标收益反推权重（示范值）
bpt_weights = np.array([0.30, 0.20, 0.20, 0.18, 0.09, 0.03])
bpt_mu = bpt_weights @ mu
bpt_sig = np.sqrt(bpt_weights @ cov @ bpt_weights)
ax.scatter([bpt_sig], [bpt_mu], color="#E45756", s=160, zorder=5,
           marker="*", label=f"行为金字塔组合 (BPT)\n收益 {bpt_mu*100:.1f}% / 波动 {bpt_sig*100:.1f}%")

# 标注：BPT 点落在有效前沿“内部”（均值方差角度看被支配）
ax.annotate("被均值方差前沿支配：\n同样波动下 MV 能给出更高收益",
            xy=(bpt_sig, bpt_mu), xytext=(bpt_sig + 0.06, bpt_mu + 0.03),
            fontsize=9, color="#E45756",
            arrowprops=dict(arrowstyle="->", color="#E45756", lw=1.4))

ax.set_xlabel("年化波动率 (风险)")
ax.set_ylabel("年化预期收益")
ax.set_title("行为组合前沿 vs 均值方差有效前沿：BPT 在 MV 意义上“非最优”\n但因为它匹配了多层心理目标，投资者主观上更满意",
             fontsize=12, fontweight="bold")
ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(d1, "bpt_vs_mv_frontier.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------- 图3：四层累积净值模拟（心智账户分层运作） ----------
np.random.seed(20260710)
T = 252 * 5  # 5 年日度
def layer_path(mu_a, sig_a, seed):
    np.random.seed(seed)
    r = np.random.normal(mu_a / 252, sig_a / np.sqrt(252), T)
    return np.cumprod(1 + r)

safe = layer_path(0.015, 0.008, 1)
bond = layer_path(0.045, 0.04, 2)
growth = layer_path(0.10, 0.18, 3)
spec = layer_path(0.22, 0.45, 4)

fig, ax = plt.subplots(figsize=(10, 5.8))
ax.plot(range(T), safe, color="#4C78A8", lw=2, label="安全层 (保本)")
ax.plot(range(T), bond, color="#54A24B", lw=2, label="收入层 (安稳)")
ax.plot(range(T), growth, color="#F58518", lw=2, label="成长层 (增值)")
ax.plot(range(T), spec, color="#E45756", lw=2, label="投机层 (致富)")
ax.set_xlabel("交易日"); ax.set_ylabel("累积净值 (起始=1)")
ax.set_title("四层心智账户的净值路径：风险越高的层，波动越剧烈\n——这正是“分层持有”而非“整体最优化”的行为逻辑",
             fontsize=12, fontweight="bold")
ax.legend(ncol=2); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(d1, "bpt_layer_paths.png"), dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 文章 2：网页爬虫与文本挖掘
# ============================================================
d2 = os.path.join(BASE, "web-scraping-text-mining")
os.makedirs(d2, exist_ok=True)

# ---------- 图1：数据采集与文本挖掘流水线 ----------
fig, ax = plt.subplots(figsize=(11, 4))
stages = [
    ("数据源", "新闻 / 财报 / 公告\n社交媒体 / 论坛", "#4C78A8"),
    ("采集", "Requests + BeautifulSoup\nSelenium / Playwright\n(遵守 robots.txt)", "#54A24B"),
    ("清洗", "去 HTML 噪声\n中文分词 (jieba)\n去停用词 / 实体识别", "#F58518"),
    ("特征化", "情感词典 / FinBERT\nTF-IDF / 词向量\n主题模型 (LDA)", "#B279A2"),
    ("信号", "日度情绪指数\n事件强度\n文本异常度", "#E45756"),
    ("回测", "领先-滞后检验\n多空组合\nIC / 分层", "#72B7B2"),
]
n = len(stages)
xpos = np.arange(n)
for i, (title, desc, color) in enumerate(stages):
    box = mpatches.FancyBboxPatch((i - 0.42, 0.25), 0.84, 0.5,
                                  boxstyle="round,pad=0.02,rounding_size=0.04",
                                  facecolor=color, alpha=0.88, edgecolor="white", lw=1.5)
    ax.add_patch(box)
    ax.text(i, 0.5, title, ha="center", va="center", color="white",
            fontsize=12, fontweight="bold")
    ax.text(i, 0.05, desc, ha="center", va="bottom", color="#333", fontsize=8.2)
    if i < n - 1:
        ax.annotate("", xy=(i + 0.5, 0.5), xytext=(i + 0.42, 0.5),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=2))
ax.set_xlim(-0.6, n - 0.4)
ax.set_ylim(-0.05, 0.92)
ax.axis("off")
ax.set_title("网页爬虫 + 文本挖掘：从原始网页到可回测信号的完整流水线",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(d2, "scraping_pipeline.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------- 图2：构造的日度情绪指数 + 价格叠加 ----------
np.random.seed(20260710)
days = 250
# 真实情绪：带主题的随机游走 + 偶发事件尖峰
sent = np.zeros(days)
shock_day = np.random.randint(40, days - 40, size=6)
for s in shock_day:
    sent[s:s + 3] += np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.0, size=3)
for t in range(1, days):
    sent[t] = 0.92 * sent[t - 1] + 0.08 * np.random.randn()
sent = np.clip(sent, -1, 1)
# 价格：情绪领先 3 日影响收益
ret = 0.0004 * np.roll(sent, 3) + np.random.normal(0.0003, 0.011, days)
price = 100 * np.cumprod(1 + ret)

fig, ax1 = plt.subplots(figsize=(10, 5.6))
ax1.plot(range(days), price, color="#4C78A8", lw=1.8, label="资产价格 (归一=100)")
ax1.set_ylabel("资产价格", color="#4C78A8")
ax1.tick_params(axis="y", labelcolor="#4C78A8")
ax2 = ax1.twinx()
ax2.fill_between(range(days), sent, 0, color="#E45756", alpha=0.30)
ax2.plot(range(days), sent, color="#E45756", lw=1.5, label="日度情绪指数 (-1~1)")
ax2.axhline(0, color="gray", lw=0.8, ls="--")
ax2.set_ylabel("情绪指数", color="#E45756")
ax2.tick_params(axis="y", labelcolor="#E45756")
ax2.set_ylim(-1.2, 1.2)
ax1.set_xlabel("交易日")
ax1.set_title("文本挖掘得到的日度情绪指数：领先于价格拐点的“软信息”\n（红：情绪 / 蓝：价格，情绪在多个拐点先于价格变化）",
              fontsize=11.5, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(d2, "sentiment_index.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------- 图3：金融文本 TF-IDF 主题-词项热力图 ----------
np.random.seed(11)
terms = ["营收", "增长", "风险", "下调", "亏损", "分红", "并购", "监管", "创新", "债务"]
docs = ["年报A", "季报B", "公告C", "新闻D", "研报E", "论坛F", "快讯G", "业绩H"]
# 构造一个有结构感的 TF-IDF 矩阵（让某些词在某些文档突出）
base = np.abs(np.random.randn(len(docs), len(terms))) * 0.4
# 注入主题：前3个文档偏“利好词”，后几个偏“风险词”
base[:3, :3] += np.array([[1.2, 1.0, 0.6]] * 3)
base[3:6, 3:6] += np.array([[1.1, 0.9, 0.7]] * 3)
base[6:, 6:] += np.array([[1.0, 0.8, 0.9, 0.6]] * 2)
mat = base / base.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(9, 5.5))
im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(terms)))
ax.set_xticklabels(terms, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(docs)))
ax.set_yticklabels(docs, fontsize=9)
ax.set_title("文档-词项 TF-IDF 矩阵（行归一）：\n文本挖掘把非结构文本压成可量化、可聚类的数值特征",
             fontsize=11, fontweight="bold")
for i in range(len(docs)):
    for j in range(len(terms)):
        ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                color="black" if mat[i, j] < 0.2 else "white", fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="TF-IDF 权重")
plt.tight_layout()
plt.savefig(os.path.join(d2, "tfidf_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

print("✅ 配图已生成：")
for d in (d1, d2):
    files = sorted(os.listdir(d))
    print(f"  {os.path.basename(d)}: {files}")
