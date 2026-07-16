#!/usr/bin/env python3
"""
为文章「净权益发行异象：公司回购越多未来越该涨」(composite-equity-issuance) 生成真实配图。

核心逻辑（净权益发行异象 / 回购异象，McConnell-M必由之路(al. 1990)、Pontiff & Schill 2003、FF 投资因子）：
  - 公司对外的「净权益发行」NI = 新股发行 − 股票回购 − 配股。NI>0 是净增发（稀释），
    NI<0 是净回购（缩股、回报股东）。
  - 异象事实：净增发越多的公司，未来回报越低；净回购越多的公司，未来回报越高。
    即横截面上 next-return 与 NI 显著负相关 —— 长期被低估的「聪明钱信号」。
  - 本文用自洽面板 DGP：每家公司 NI 有持续性（回购/增发是习惯动作），
    下期超额收益 = γ·(−NI) + 市场 β·Mkt + 噪声（γ>0 即异象成立）。
  - 策略：每月按 NI 排序分十档，多最低 NI（回购最多）档、空最高 NI（增发最多）档，
    下月再平衡，复利成净值；并给出分档收益单调性、L-S 年化与 Sharpe、NI 与未来回报散点。
  全部为合成数据，非占位图。
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
D = os.path.join(BASE, "composite-equity-issuance")
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(20260716)

# ---------- 1) 构造面板 DGP ----------
N = 300                      # 公司数
T = 300                      # 月数
mkt_beta_mu = 1.0
NI_persist = 0.75            # 公司净发行习惯的持续性
NI_idio = 0.45

# 共同的市场发行环境（融资周期）
common = rng.normal(0, 0.5, T)
# 每家公司横截面异质性（有些公司是「天然回购者」NI 偏负，有些是「天然增发者」）
firm_bias = rng.normal(0, 0.8, N)

NI = np.zeros((T, N))
for t in range(T):
    for i in range(N):
        prev = NI[t - 1, i] if t > 0 else firm_bias[i]
        NI[t, i] = NI_persist * prev + 0.25 * common[t] + rng.normal(0, NI_idio)

# 市场因子
mkt = rng.normal(0.006, 0.04, T)            # 月度市场超额收益
# 下期超额收益：把 t 期 NI 在截面上标准化成 z 分数（每月内 std=1），异象信号 = gamma*(-NI_z)
# 这样横截面信号量纲可控：全分档月度溢价 ≈ gamma*(~3个标准差)，设定 realistic 量级
gamma = 0.0025                          # 异象月度强度（每 std 的 NI_z 对应收益）
beta_i = rng.normal(mkt_beta_mu, 0.3, N)
ret = np.zeros((T, N))
for t in range(T - 1):
    ni_z = (NI[t] - NI[t].mean()) / (NI[t].std() + 1e-9)
    signal = gamma * (-ni_z)               # NI 越负（回购多）→ signal 越大
    sysfac = beta_i * mkt[t + 1]
    ret[t + 1] = signal + sysfac + rng.normal(0, 0.05, N)

# ---------- 2) 分十档：长低 NI、空高 NI ----------
def decile_labels(x):
    # 返回 0..9，0=最低 NI
    order = np.argsort(x)
    lab = np.empty(len(x), dtype=int)
    lab[order] = np.arange(len(x)) // (len(x) // 10 + 1)
    lab[order] = np.minimum(lab[order], 9)
    return lab

dec_ret = np.zeros((T, 10))
ls_ret = np.zeros(T)
for t in range(1, T):
    if np.all(np.isnan(ret[t])):
        continue
    lab = decile_labels(NI[t - 1])          # 用上一期 NI 排序（无前视）
    for d in range(10):
        mask = lab == d
        if mask.sum() > 0:
            dec_ret[t, d] = np.mean(ret[t, mask])
    low = lab == 0                            # 回购最多（NI 最低）
    high = lab == 9                           # 增发最多（NI 最高）
    ls_ret[t] = np.mean(ret[t, low]) - np.mean(ret[t, high])

# 分档平均收益（横截面均值，去掉 t=0）
valid = ~np.all(dec_ret == 0, axis=1)
dec_avg = dec_ret[valid].mean(axis=0)
ls_mean_month = ls_ret[1:].mean()
ls_std_month = ls_ret[1:].std()
ls_ann = ls_mean_month * 12
ls_sharpe = ls_mean_month * 12 / (ls_std_month * np.sqrt(12))
ls_tstat = ls_mean_month / (ls_std_month / np.sqrt(len(ls_ret[1:])))
win_rate = (ls_ret[1:] > 0).mean()

# 复利净值
nv = np.cumprod(1 + ls_ret[1:])
# 买入持有市场做对比
nv_mkt = np.cumprod(1 + mkt[1:])

print("=== 净权益发行异象 诊断 ===")
print(f"γ(异象月度强度) = {gamma}")
print(f"十档平均下月超额收益 (decile 0=最低NI/回购最多 → 9=最高NI/增发最多):")
for d in range(10):
    print(f"  decile {d}: {dec_avg[d]*100:+.3f}%")
print(f"单调性检查: dec0 - dec9 = {(dec_avg[0]-dec_avg[9])*100:+.3f}%")
print(f"L-S 月度均值={ls_mean_month*100:+.3f}%  年化={ls_ann*100:+.2f}%  Sharpe={ls_sharpe:.2f}  t={ls_tstat:.2f}  胜率={win_rate*100:.1f}%")
print(f"生成图片: {os.listdir(D)}")

# ---------- 绘图 ----------
plt.rcParams["figure.dpi"] = 150

# 图1：十档平均下月收益（单调性）
fig, ax = plt.subplots(figsize=(11, 5.2))
xs = np.arange(10)
colors = ["#c44e52" if d in (0, 9) else "#4c72b0" for d in range(10)]
bars = ax.bar(xs, dec_avg * 100, color=colors, alpha=0.88)
for b, v in zip(bars, dec_avg):
    ax.text(b.get_x() + b.get_width() / 2, v * 100 + (0.02 if v >= 0 else -0.05),
            f"{v*100:+.2f}", ha="center", fontsize=9, fontweight="bold")
ax.axhline(0, color="#333", lw=1)
ax.set_xticks(xs)
ax.set_xticklabels([f"D{d}\n({'回购最多' if d==0 else '增发最多' if d==9 else ''})" for d in range(10)], fontsize=9)
ax.set_xlabel("净权益发行分档（D0=回购最多 → D9=增发最多）", fontsize=10.5)
ax.set_ylabel("下月平均超额收益 (%)", fontsize=11)
ax.set_title("净权益发行异象：回购越多、增发越少，未来回报越高（十档单调递减）",
             fontsize=12.5, fontweight="bold")
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "issuance_decile_returns.png"), bbox_inches="tight")
plt.close()

# 图2：L-S 组合复利净值 vs 买入持有市场
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(nv, color="#c44e52", lw=2.2, label=f"净发行 L-S 组合 (年化 {ls_ann*100:+.1f}%, SR {ls_sharpe:.2f})")
ax.plot(nv_mkt, color="#4c72b0", lw=1.6, alpha=0.8, label="买入持有市场")
ax.set_ylabel("净值（起始=1）", fontsize=11)
ax.set_xlabel("月", fontsize=11)
ax.set_title("多回购/空增发组合：累积净值跑赢市场", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "issuance_long_short_nav.png"), bbox_inches="tight")
plt.close()

# 图3：NI 与未来 12 月累计回报散点（取某截面）
fig, ax = plt.subplots(figsize=(11, 5.2))
# 构造某期「未来12月累计」代理：用 NI 与对应下期收益的关系展示
sample_t = 150
sc_x = NI[sample_t]
sc_y = ret[sample_t + 1]
ax.scatter(sc_x, sc_y * 100, color="#55a868", alpha=0.5, s=18)
z = np.polyfit(sc_x, sc_y * 100, 1)
xs2 = np.linspace(sc_x.min(), sc_x.max(), 50)
ax.plot(xs2, np.polyval(z, xs2), color="#c44e52", lw=2.2, label=f"拟合斜率={z[0]:.2f}%/单位NI")
ax.set_xlabel("净权益发行 NI（负=净回购，正=净增发）", fontsize=11)
ax.set_ylabel("下月超额收益 (%)", fontsize=11)
ax.set_title(f"横截面：净发行 NI 与未来回报显著负相关（样本月 t={sample_t}）",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "issuance_ni_vs_return_scatter.png"), bbox_inches="tight")
plt.close()

# 图4：净权益发行分布（回购者 vs 增发者）
fig, ax = plt.subplots(figsize=(11, 5.0))
last_ni = NI[-1]
ax.hist(last_ni, bins=40, color="#6a3d9a", alpha=0.8)
ax.axvline(0, color="#c44e52", lw=2, ls="--", label="NI=0（既不增也不回购）")
ax.axvline(last_ni.mean(), color="#2ca02c", lw=2, label=f"均值 {last_ni.mean():.2f}")
ax.set_xlabel("净权益发行 NI", fontsize=11)
ax.set_ylabel("公司数", fontsize=11)
ax.set_title("横截面净发行分布：左尾=净回购者，右尾=净增发者", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "issuance_distribution.png"), bbox_inches="tight")
plt.close()

print("✅ 图片已生成:", sorted(os.listdir(D)))
