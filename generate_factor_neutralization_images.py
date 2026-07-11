#!/usr/bin/env python3
"""
为文章「因子截面对冲：用行业市值中性化剔除风格暴露」(factor-neutralization-industry) 生成真实配图。
数据：模拟 300 只股票、6 个行业、不同市值；构造一个被「行业 + 市值」污染的 value 信号，
      未来收益同时由「纯净价值 alpha」与「行业 / 市值风格因子」驱动。
方法：原始多空组合 vs 回归中性化（对 行业哑变量 + log市值 取残差）后的多空组合，
      对比二者的行业净权重、市值暴露与多空收益，证明中性化剔除风格暴露而保留 alpha。
图表：
  1. neu_signal_scatter.png   信号 vs 市值散点（按行业着色）→ 直观显示信号被风格污染
  2. neu_exposure_raw.png     原始多空组合的行业净权重（大偏斜）
  3. neu_exposure_neu.png     中性化后多空组合的行业净权重（≈0）
  4. neu_compare.png          收益 vs 风格暴露对比：中性化后 alpha 保留、风格暴露归零
全部为真实数值计算，非占位图。
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
D = os.path.join(BASE, "factor-neutralization-industry")
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(20260711)

# ============================================================
# 1) 模拟股票池
# ============================================================
N = 300
INDUSTRIES = ["金融", "科技", "消费", "医药", "工业", "能源"]
K = len(INDUSTRIES)
ind = rng.integers(0, K, N)
# 市值：对数正态，跨度大（大/中/小盘）
logmcap = rng.normal(0, 1, N)
# 各行业「价值倾斜」不同：金融/能源天然偏 value，科技/消费偏 growth
ind_value_tilt = np.array([ 1.1, -0.9, -0.4, 0.0, 0.5, 0.9])
# 纯净价值 alpha（我们真正想要的）：与行业、市值独立
true_quality = rng.normal(0, 1, N)

# 观测到的 value 信号 = 纯净 alpha + 行业污染 + 市值污染 + 噪声
# 小盘(logmcap 小) book-to-market 高 → 信号更大，制造市值污染
signal = (1.0 * true_quality
          + 0.9 * ind_value_tilt[ind]
          - 0.7 * (logmcap - logmcap.mean())        # 小盘信号更高
          + rng.normal(0, 0.5, N))

# 未来收益：纯净 alpha 被定价(β_q>0) + 行业因子 + 市值因子(小盘溢价) + 噪声
# 注意：行业/市值因子本身也「给钱」，所以原始信号的高收益里混着风格溢价
ind_factor = ind_value_tilt[ind]                    # 行业收益（与行业价值倾斜同向）
size_factor = -(logmcap - logmcap.mean())           # 小盘溢价
fwd = (0.55 * true_quality
       + 0.45 * ind_factor
       + 0.35 * size_factor
       + rng.normal(0, 0.8, N))
print("信号-收益 IC（原始）:", round(np.corrcoef(signal, fwd)[0,1], 3))

# ============================================================
# 2) 原始多空 & 中性化多空
# ============================================================
def ls_portfolio(sig, fwd, frac=0.20):
    """做多信号最高 frac，做空最低 frac，返回权重向量（多 +1，空 -1，其余 0）"""
    order = np.argsort(sig)
    n = int(N * frac)
    w = np.zeros(N)
    w[order[:n]] = -1.0 / n     # 空頭（最低）
    w[order[-n:]] = +1.0 / n     # 多头（最高）
    ret = w @ fwd
    return w, ret

w_raw, ret_raw = ls_portfolio(signal, fwd)
# 中性化：对 行业哑变量 + log市值 回归取残差
X = np.column_stack([(ind == k).astype(float) for k in range(K)] + [logmcap])
X = np.column_stack([np.ones(N), X])
beta = np.linalg.lstsq(X, signal, rcond=None)[0]
signal_neu = signal - X @ beta
print("信号-收益 IC（中性化后）:", round(np.corrcoef(signal_neu, fwd)[0,1], 3))
w_neu, ret_neu = ls_portfolio(signal_neu, fwd)

# 行业净权重：多头行业占比 - 空头行业占比
def net_industry(w):
    long_mask = w > 0; short_mask = w < 0
    out = []
    for k in range(K):
        long_w = np.sum((ind == k) & long_mask) / np.sum(long_mask) if long_mask.any() else 0
        short_w = np.sum((ind == k) & short_mask) / np.sum(short_mask) if short_mask.any() else 0
        out.append(long_w - short_w)
    return np.array(out)

net_raw = net_industry(w_raw)
net_neu = net_industry(w_neu)
# 市值暴露：多头平均 log市值 - 空头平均 log市值
size_exp_raw = logmcap[w_raw > 0].mean() - logmcap[w_raw < 0].mean()
size_exp_neu = logmcap[w_neu > 0].mean() - logmcap[w_neu < 0].mean()
print(f"\n原始多空: 收益={ret_raw:.3f}  行业净权重范数={np.linalg.norm(net_raw):.3f}  市值暴露={size_exp_raw:.3f}")
print(f"中性化多空: 收益={ret_neu:.3f}  行业净权重范数={np.linalg.norm(net_neu):.3f}  市值暴露={size_exp_neu:.3f}")

# ============================================================
# 图1：信号 vs 市值 散点（按行业着色）
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6.0))
cmap = plt.cm.tab10
for k in range(K):
    m = ind == k
    ax.scatter(logmcap[m], signal[m], s=22, alpha=0.65,
               color=cmap(k), label=INDUSTRIES[k])
ax.set_xlabel("对数市值（左=小盘，右=大盘）", fontsize=11)
ax.set_ylabel("value 信号", fontsize=11)
ax.set_title("信号被风格污染：同行业抱团、小盘整体更高（单纯排序=变相押注风格）",
             fontsize=12.5, fontweight="bold")
ax.axhline(0, color="black", lw=0.6)
ax.legend(fontsize=9, ncol=3, loc="upper right")
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "neu_signal_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图2 & 3：行业净权重（原始 vs 中性化）
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.4))
cols = [cmap(k) for k in range(K)]
bars = ax.bar(INDUSTRIES, net_raw*100, color=cols, alpha=0.85)
ax.axhline(0, color="black", lw=0.8)
for b, v in zip(bars, net_raw*100):
    ax.text(b.get_x()+b.get_width()/2, v + (1.2 if v >= 0 else -2.2),
            f"{v:+.0f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("行业净权重 (%)", fontsize=11)
ax.set_title(f"原始多空组合的行业净权重：严重偏斜（金融 +{net_raw[0]*100:.0f}%，科技 {net_raw[1]*100:.0f}%）",
             fontsize=12, fontweight="bold")
ax.set_ylim(min(net_raw.min()*100, -30)-8, max(net_raw.max()*100, 30)+8)
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "neu_exposure_raw.png"), dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(11, 5.4))
bars = ax.bar(INDUSTRIES, net_neu*100, color=cols, alpha=0.85)
ax.axhline(0, color="black", lw=0.8)
for b, v in zip(bars, net_neu*100):
    ax.text(b.get_x()+b.get_width()/2, v + (0.6 if v >= 0 else -1.1),
            f"{v:+.0f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("行业净权重 (%)", fontsize=11)
ax.set_title(f"中性化后多空组合的行业净权重：全部贴近 0（行业暴露被剔除）",
             fontsize=12, fontweight="bold")
ax.set_ylim(min(net_neu.min()*100, -30)-8, max(net_neu.max()*100, 30)+8)
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "neu_exposure_neu.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图4：收益 vs 风格暴露对比
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
# 左：多空收益
ax[0].bar(["原始", "中性化"], [ret_raw, ret_neu], color=["#ff7f0e", "#2ca02c"], width=0.5)
for i, v in enumerate([ret_raw, ret_neu]):
    ax[0].text(i, v + (0.02 if v >= 0 else -0.04), f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")
ax[0].axhline(0, color="black", lw=0.6)
ax[0].set_ylabel("多空组合未来收益", fontsize=11)
ax[0].set_title("alpha 基本保留：中性化后收益仍有原始的 ~%.0f%%" % (100*ret_neu/ret_raw),
                fontsize=11.5, fontweight="bold")
ax[0].grid(True, axis="y", alpha=0.25)
# 右：风格暴露（行业权重范数 + |市值暴露|）
exp_raw_tot = np.linalg.norm(net_raw) + abs(size_exp_raw)
exp_neu_tot = np.linalg.norm(net_neu) + abs(size_exp_neu)
ax[1].bar(["原始", "中性化"], [exp_raw_tot, exp_neu_tot], color=["#d62728", "#1f77b4"], width=0.5)
for i, v in enumerate([exp_raw_tot, exp_neu_tot]):
    ax[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")
ax[1].set_ylabel("风格暴露综合指数", fontsize=11)
ax[1].set_title("风格暴露几乎清零：行业+市值暴露从 %.2f 降到 %.2f" % (exp_raw_tot, exp_neu_tot),
                fontsize=11.5, fontweight="bold")
ax[1].grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "neu_compare.png"), dpi=150, bbox_inches="tight")
plt.close()

print("\n✅ 因子中性化配图生成完成：", sorted(os.listdir(D)))
