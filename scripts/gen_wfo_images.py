# -*- coding: utf-8 -*-
"""Walk-Forward 优化配图：滚动窗口示意 + 样本内外夏普对比 + 参数漂移 + WFE"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent.parent / "public/images/walk-forward-optimization"
OUT.mkdir(parents=True, exist_ok=True)

# ============ 图 1：Walk-Forward 滚动窗口示意图 ============
fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=130)
n_folds = 6
is_len, oos_len = 4, 1
for i in range(n_folds):
    start = i * oos_len
    ax.add_patch(Rectangle((start, n_folds - 1 - i - 0.35), is_len, 0.7,
                           facecolor="#1f77b4", alpha=0.75))
    ax.add_patch(Rectangle((start + is_len, n_folds - 1 - i - 0.35), oos_len, 0.7,
                           facecolor="#d62728", alpha=0.85))
    ax.text(-0.3, n_folds - 1 - i, f"第 {i+1} 轮", ha="right", va="center", fontsize=10)
ax.add_patch(Rectangle((0, -1.6), 0.6, 0.5, facecolor="#1f77b4", alpha=0.75))
ax.text(0.8, -1.35, "样本内（IS）：优化参数", va="center", fontsize=10)
ax.add_patch(Rectangle((4.5, -1.6), 0.6, 0.5, facecolor="#d62728", alpha=0.85))
ax.text(5.3, -1.35, "样本外（OOS）：冻结参数实测", va="center", fontsize=10)
ax.set_xlim(-1.6, is_len + n_folds * oos_len + 0.5)
ax.set_ylim(-2.0, n_folds - 0.3)
ax.set_xlabel("时间 →")
ax.set_yticks([])
ax.set_title("Walk-Forward：滚动地「样本内调参 → 冻结参数 → 样本外实测」，OOS 段拼成完整净值")
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "wfo-rolling-scheme.jpg")
plt.close(fig)
print("[图1] 滚动窗口示意图 done")

# ============ 模拟：动量策略 + 参数网格 walk-forward ============
rng = np.random.default_rng(20)
T = 2520  # 10 年日频

# 构造带 regime 的价格：前半段趋势市（动量有效），后半段震荡市（动量衰减）
mu_regime = np.concatenate([np.full(T // 2, 0.0006), np.full(T - T // 2, 0.0001)])
# 动量真实存在但强度随 regime 变化：AR(1) 正自相关前强后弱
phi = np.concatenate([np.full(T // 2, 0.06), np.full(T - T // 2, 0.01)])
eps = rng.normal(0, 0.012, T)
ret = np.empty(T)
ret[0] = eps[0]
for t in range(1, T):
    ret[t] = mu_regime[t] + phi[t] * (ret[t - 1] - mu_regime[t - 1]) * 8 + eps[t]

price = 100 * np.cumprod(1 + ret)

LOOKBACKS = [5, 10, 20, 40, 60, 90, 120]

def momo_returns(ret, lb):
    """lb 日累计收益为正则持有（次日生效）"""
    sig = np.zeros_like(ret)
    cum = np.convolve(ret, np.ones(lb), "full")[:len(ret)]
    pos = (cum > 0).astype(float)
    sig[1:] = pos[:-1]           # signal-on-i, execute-on-i+1
    sig[:lb + 1] = 0
    return sig * ret

def sharpe(x):
    x = x[np.abs(x) > 0] if np.any(np.abs(x) > 0) else x
    if len(x) < 20 or x.std() == 0:
        return 0.0
    return float(x.mean() / x.std() * np.sqrt(252))

def sharpe_full(x):
    if x.std() == 0:
        return 0.0
    return float(x.mean() / x.std() * np.sqrt(252))

IS_LEN, OOS_LEN = 504, 126  # 2年IS + 半年OOS
folds = []
start = 0
while start + IS_LEN + OOS_LEN <= T:
    folds.append((start, start + IS_LEN, start + IS_LEN + OOS_LEN))
    start += OOS_LEN

is_sharpes, oos_sharpes, best_lbs = [], [], []
oos_concat = []
for (a, b, c) in folds:
    r_is = ret[a:b]
    scores = [sharpe_full(momo_returns(r_is, lb)) for lb in LOOKBACKS]
    k = int(np.argmax(scores))
    best_lbs.append(LOOKBACKS[k])
    is_sharpes.append(scores[k])
    # OOS：用含 lookback 缓冲的段计算信号，但只取 OOS 段收益
    buf = max(LOOKBACKS[k] + 2, 0)
    seg = ret[max(0, b - buf):c]
    strat = momo_returns(seg, LOOKBACKS[k])[-(c - b):]
    oos_sharpes.append(sharpe_full(strat))
    oos_concat.append(strat)

oos_all = np.concatenate(oos_concat)
wfe = np.mean(oos_sharpes) / np.mean(is_sharpes) if np.mean(is_sharpes) != 0 else np.nan
print(f"折数={len(folds)}  IS 平均夏普={np.mean(is_sharpes):.2f}  OOS 平均夏普={np.mean(oos_sharpes):.2f}  WFE={wfe:.2f}")
print(f"最优 lookback 序列: {best_lbs}")
print(f"拼接 OOS 整体夏普={sharpe_full(oos_all):.2f}")

# 全样本一次性优化（对照组：过拟合基准）
scores_full = [sharpe_full(momo_returns(ret, lb)) for lb in LOOKBACKS]
k_full = int(np.argmax(scores_full))
print(f"全样本一次性优化: 最优 lb={LOOKBACKS[k_full]}, 全样本夏普={scores_full[k_full]:.2f}")

# ============ 图 2：每折 IS vs OOS 夏普 ============
fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=130)
x = np.arange(len(folds))
w = 0.38
ax.bar(x - w / 2, is_sharpes, w, color="#1f77b4", alpha=0.85, label="样本内夏普（调参后）")
ax.bar(x + w / 2, oos_sharpes, w, color="#d62728", alpha=0.85, label="样本外夏普（冻结参数）")
ax.axhline(0, color="k", lw=0.8)
ax.axhline(np.mean(is_sharpes), color="#1f77b4", ls="--", lw=1.2, alpha=0.7)
ax.axhline(np.mean(oos_sharpes), color="#d62728", ls="--", lw=1.2, alpha=0.7)
ax.text(len(folds) - 0.4, np.mean(is_sharpes) + 0.06, f"IS 均值 {np.mean(is_sharpes):.2f}", color="#1f77b4", ha="right")
ax.text(len(folds) - 0.4, np.mean(oos_sharpes) - 0.18, f"OOS 均值 {np.mean(oos_sharpes):.2f}", color="#d62728", ha="right")
ax.set_xticks(x, [f"{i+1}" for i in x])
ax.set_xlabel("Walk-Forward 轮次")
ax.set_ylabel("年化夏普")
ax.set_title(f"每轮样本内 vs 样本外夏普：WFE = OOS/IS = {wfe:.2f}（水分被挤掉的比例一目了然）")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "wfo-is-vs-oos.jpg")
plt.close(fig)
print("[图2] IS vs OOS done")

# ============ 图 3：最优参数漂移 ============
fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=130)
ax.step(x, best_lbs, where="mid", color="#2ca02c", lw=2, marker="o", ms=7)
ax.axvspan(-0.5, (T // 2 - IS_LEN) / OOS_LEN - 0.5, alpha=0.10, color="#1f77b4")
ax.text(0.2, max(best_lbs) * 0.95, "趋势市 regime（动量强）", color="#1f77b4", fontsize=10)
ax.text(len(folds) * 0.55, max(best_lbs) * 0.95, "震荡市 regime（动量弱）", color="#d62728", fontsize=10)
ax.set_xticks(x, [f"{i+1}" for i in x])
ax.set_xlabel("Walk-Forward 轮次")
ax.set_ylabel("样本内最优回看窗口（日）")
ax.set_title("每轮选出的最优参数：跳来跳去 = 参数不稳定的直接证据")
fig.tight_layout()
fig.savefig(OUT / "wfo-param-drift.jpg")
plt.close(fig)
print("[图3] 参数漂移 done, best_lbs =", best_lbs)

# ============ 图 4：三条净值对比 ============
# a) 全样本优化(过拟合口径，用全样本最优参数跑全样本)
strat_full = momo_returns(ret, LOOKBACKS[k_full])
eq_full = np.cumprod(1 + strat_full)
# b) walk-forward 拼接 OOS
oos_start = folds[0][1]
eq_wf = np.ones(T)
eq_wf[oos_start:oos_start + len(oos_all)] = np.cumprod(1 + oos_all)
eq_wf[oos_start + len(oos_all):] = eq_wf[oos_start + len(oos_all) - 1]
# c) buy & hold
eq_bh = np.cumprod(1 + ret)

days = np.arange(T) / 252
fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=130)
ax.plot(days, eq_full, color="#1f77b4", lw=1.6,
        label=f"全样本一次性优化（lb={LOOKBACKS[k_full]}，夏普 {scores_full[k_full]:.2f}，含水分）")
ax.plot(days[oos_start:], eq_wf[oos_start:], color="#d62728", lw=1.6,
        label=f"Walk-Forward 拼接 OOS（夏普 {sharpe_full(oos_all):.2f}，可信口径）")
ax.plot(days, eq_bh, color="gray", lw=1.2, alpha=0.8, label="买入持有")
ax.axvline(T // 2 / 252, color="k", ls=":", lw=1, alpha=0.6)
ax.text(T // 2 / 252 + 0.1, ax.get_ylim()[0] if False else eq_bh.max() * 0.55, "regime 切换", fontsize=9, alpha=0.7)
ax.set_xlabel("年")
ax.set_ylabel("净值（起点 1.0）")
ax.set_title("同一策略族的三条净值：全样本优化的曲线永远最好看，但它不可交易")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "wfo-equity-compare.jpg")
plt.close(fig)
print("[图4] 净值对比 done")
print("done ->", OUT)
