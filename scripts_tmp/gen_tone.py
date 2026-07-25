# -*- coding: utf-8 -*-
"""财报电话会语调分析：用词频把管理层情绪变成因子 — 配图与统计"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

for f in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc"]:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/earnings-call-tone-analysis"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(7)

# ========= 面板设定 =========
n_stocks = 300
n_q = 40          # 40 个季度（10 年），每股每季一次电话会
# 每次电话会：真实基本面信号 f ~ N(0,1)（慢均值回归），语调 = 信号 + 管理层噪声/粉饰
f = np.zeros((n_stocks, n_q))
for q in range(1, n_q):
    f[:, q] = 0.6 * f[:, q-1] + 0.8 * rng.standard_normal(n_stocks)
f = (f - f.mean()) / f.std()

# 语调净值 tone = (正面词-负面词)/总词数，标准化后 = 0.35*真实信号 + 粉饰偏置 + 噪声
gloss = 0.8 + 0.80 * rng.standard_normal((n_stocks, 1))   # 管理层普遍报喜不报忧，且粉饰程度因人而异（横截面固定效应）
tone_raw = 0.35 * f + gloss + 0.55 * rng.standard_normal((n_stocks, n_q))
# 语调变化 ΔTone 才是信号（水平被粉饰污染）
dtone = np.diff(tone_raw, axis=1)          # (n_stocks, n_q-1)
df_sig = np.diff(f, axis=1)

# 未来一季度收益：真实信号变化驱动 + 市场 + 不可分散的风格因子（情绪 regime）
mkt = 0.015 + 0.06 * rng.standard_normal(n_q - 1)
style = 0.05 * rng.standard_normal(n_q - 1)              # 情绪/风格因子收益
beta_style = 0.5 * rng.standard_normal((n_stocks, 1))    # 个股对风格因子的暴露
# 语调改善股天然偏向高情绪暴露 → 多空腿无法完全对冲掉风格风险
beta_eff = beta_style + 0.35 * (dtone - dtone.mean()) / dtone.std()
fwd_ret = 0.02 * df_sig + mkt[None, :] + beta_eff * style[None, :] + 0.10 * rng.standard_normal((n_stocks, n_q - 1))

# ========= 1. 词频示意：Loughran-McDonald 风格正负词计数 =========
# 模拟两份 transcript 的词频条形（强势 vs 遮掩）
pos_words = ["strong", "growth", "record", "improve", "confident", "momentum"]
neg_words = ["decline", "challenge", "headwind", "weak", "uncertain", "delay"]
strong_counts = [38, 31, 17, 24, 15, 12]
weak_counts_p = [12, 9, 3, 8, 4, 2]
strong_neg = [4, 8, 5, 3, 6, 2]
weak_neg = [22, 27, 19, 14, 25, 11]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
y = np.arange(len(pos_words))
axes[0].barh(y - 0.2, strong_counts, 0.38, color="#6ACC65", label="正面词")
axes[0].barh(y + 0.2, strong_neg, 0.38, color="#D65F5F", label="负面词")
axes[0].set_yticks(y); axes[0].set_yticklabels([f"{p} / {n}" for p, n in zip(pos_words, neg_words)], fontsize=8)
axes[0].set_title("电话会 A：语调净值 +0.62"); axes[0].legend(fontsize=8); axes[0].set_xlabel("词频")
axes[1].barh(y - 0.2, weak_counts_p, 0.38, color="#6ACC65")
axes[1].barh(y + 0.2, weak_neg, 0.38, color="#D65F5F")
axes[1].set_title("电话会 B：语调净值 −0.48"); axes[1].set_xlabel("词频")
fig.suptitle("Loughran-McDonald 金融词典计数：同一家公司两个季度的电话会", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/tone_wordcount.png", dpi=110); plt.close(fig)

# ========= 2. 语调水平 vs 语调变化：与未来收益的 IC =========
def cs_ic(sig, ret):
    """逐期横截面 spearman IC"""
    from scipy import stats as ss
    ics = []
    for q in range(sig.shape[1]):
        ics.append(ss.spearmanr(sig[:, q], ret[:, q]).statistic)
    return np.array(ics)

try:
    from scipy import stats as ss
    have_scipy = True
except ImportError:
    have_scipy = False

tone_lvl = tone_raw[:, 1:]   # 对齐 ΔTone 的期
ic_lvl = cs_ic(tone_lvl, fwd_ret)
ic_chg = cs_ic(dtone, fwd_ret)
print(f"[stat] 语调水平 IC 均值 {ic_lvl.mean():.3f} (t={ic_lvl.mean()/ic_lvl.std()*np.sqrt(len(ic_lvl)):.1f})")
print(f"[stat] 语调变化 IC 均值 {ic_chg.mean():.3f} (t={ic_chg.mean()/ic_chg.std()*np.sqrt(len(ic_chg)):.1f})")

fig, ax = plt.subplots(figsize=(10, 5))
qs = np.arange(len(ic_chg))
ax.bar(qs - 0.2, ic_lvl, 0.4, color="#B0B0B0", label=f"语调水平 IC（均值 {ic_lvl.mean():.3f}）")
ax.bar(qs + 0.2, ic_chg, 0.4, color="#4878CF", label=f"语调变化 ΔTone IC（均值 {ic_chg.mean():.3f}）")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("季度"); ax.set_ylabel("横截面 Spearman IC")
ax.set_title("水平被粉饰污染、变化才有信息：ΔTone 的 IC 显著更高更稳")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/tone_ic.png", dpi=110); plt.close(fig)

# ========= 3. 五分组多空 =========
n_use = dtone.shape[1]
port_ret = {k: [] for k in range(5)}
ls_ret = []
for q in range(n_use):
    order = np.argsort(dtone[:, q])
    groups = np.array_split(order, 5)
    means = [fwd_ret[g, q].mean() for g in groups]
    for k in range(5):
        port_ret[k].append(means[k])
    ls_ret.append(means[4] - means[0])
ls_ret = np.array(ls_ret)
g_means = [np.mean(port_ret[k]) * 100 for k in range(5)]
print("[stat] ΔTone 五分组季度均收益(%):", [f"{v:.2f}" for v in g_means])
ann_ls = (1 + ls_ret.mean()) ** 4 - 1
sh_ls = ls_ret.mean() / ls_ret.std() * 2  # 季频 → 年化 sqrt(4)=2
eq_ls = np.cumprod(1 + ls_ret)
dd_ls = (eq_ls / np.maximum.accumulate(eq_ls) - 1).min()
print(f"[stat] 多空: 季均 {ls_ret.mean()*100:.2f}%, 年化 {ann_ls*100:.1f}%, Sharpe {sh_ls:.2f}, MDD {dd_ls*100:.1f}%, 胜率 {(ls_ret>0).mean()*100:.0f}%")

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#D65F5F" if v < np.mean(g_means) else "#6ACC65" for v in g_means]
ax.bar([f"Q{i+1}" for i in range(5)], g_means, color=colors)
ax.axhline(np.mean(g_means), color="gray", lw=0.8, ls="--", label="全样本均值")
ax.set_xlabel("ΔTone 分组（Q1 语调恶化 → Q5 语调改善）"); ax.set_ylabel("下季度平均收益（%）")
ax.set_title("语调改善组跑赢恶化组：单调阶梯")
for i, v in enumerate(g_means):
    ax.text(i, v + 0.05, f"{v:.2f}%", ha="center", fontsize=9)
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/tone_quintile.png", dpi=110); plt.close(fig)

# ========= 4. 多空净值 + 信号衰减 =========
# 衰减：持有 1/2/3/4 季的多空季均收益
decay = []
for h in range(1, 5):
    rets_h = []
    for q in range(n_use - h + 1):
        order = np.argsort(dtone[:, q])
        groups = np.array_split(order, 5)
        r = np.mean([fwd_ret[groups[4], min(q+j, n_use-1)].mean() - fwd_ret[groups[0], min(q+j, n_use-1)].mean() for j in range(h)])
        rets_h.append(r)
    decay.append(np.mean(rets_h) * 100)
print("[stat] 持有1-4季的季均多空收益(%):", [f"{v:.2f}" for v in decay])

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
axes[0].plot(np.arange(1, n_use + 1) / 4, eq_ls, lw=1.3, color="#4878CF")
axes[0].set_xlabel("年"); axes[0].set_ylabel("多空组合净值")
axes[0].set_title(f"ΔTone 多空净值：年化 {ann_ls*100:.1f}%、Sharpe {sh_ls:.2f}")
axes[1].bar(["1季", "2季", "3季", "4季"], decay, color="#4878CF")
axes[1].set_xlabel("持有期"); axes[1].set_ylabel("季均多空收益（%）")
axes[1].set_title("信号衰减：语调信息在 1-2 个季度内耗尽")
fig.tight_layout(); fig.savefig(f"{OUT}/tone_ls_decay.png", dpi=110); plt.close(fig)

print("done")
