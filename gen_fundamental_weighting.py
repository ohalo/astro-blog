#!/usr/bin/env python3
"""
为文章「基本面指数加权 vs 市值加权」(fundamental-index-weighting) 生成真实配图 + 可复现指标。

所有图表都由文中代码真实计算生成（纯 numpy，无 sklearn/torch 依赖）：

  1) cover.png               —— 市值加权 vs 基本面加权 累计净值（对数轴）
  2) fw_concentration.png     —— 集中度(HHI)随时间 + 年换手率对照
  3) fw_valuation_tilt.png    —— 估值(P/F) vs 权重散点：市值加权追涨贵股、基本面加权中性

机制（数据自洽合成，仅用于演示方法；真实落地见文末路径）：
  - N=50 只股票，每只真实「基本面」F_t 以漂移 g 增长（企业真实价值）
  - 价格 P_t = F_t * m_t，m_t 是均值回复的「错误定价」(OU 过程，均值 1)：m>1 被高估
  - 市值加权权重 ∝ P_t（自动超配被高估、近期上涨的大块头）
  - 基本面加权权重 ∝ F_t（与价格无关，自动低配贵股、高配便宜股 → 价值倾斜）
  - 年度再平衡；收益来自价格相对基本面的「均值回复」
"""
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _cand in ["/System/Library/Fonts/STHeiti Medium.ttc",
              "/System/Library/Fonts/Supplemental/Songti SC.ttf"]:
    try:
        fm.fontManager.addfont(_cand)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Heiti SC", "Songti SC", "STHeiti", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight",
})

BASE = "/Users/halo/workspace/astro-blog/public/images"
IMG = os.path.join(BASE, "fundamental-index-weighting")
os.makedirs(IMG, exist_ok=True)

C = {"cap": "#4C72B0", "fund": "#55A868", "scatter": "#C44E52", "grid": "#DDDDDD"}

N = 50
T = 360  # 月


def gen_panel(seed=20260722):
    rng = np.random.default_rng(seed)
    # 每只股票的基本面漂移（真实增长），跨股票有差异
    g = rng.normal(0.006, 0.004, N)
    F = np.zeros((T + 1, N))
    F[0] = rng.uniform(0.8, 1.2, N)  # 初始基本面
    # 错误定价 m_t 均值回复到 1（OU），部分股票系统性偏贵
    kappa = rng.uniform(0.05, 0.15, N)         # 回复速度
    m = np.zeros((T + 1, N))
    m[0] = 1.0 + rng.normal(0, 0.25, N)
    for t in range(1, T + 1):
        F[t] = F[t - 1] * (1 + g) * np.exp(rng.normal(0, 0.01, N))  # 基本面带微噪增长
        m[t] = m[t - 1] + kappa * (1.0 - m[t - 1]) + rng.normal(0, 0.06, N)
        m[t] = np.clip(m[t], 0.5, 2.0)
    P = F * m
    # 月收益
    R = P[1:] / P[:-1] - 1.0
    return F[:-1], P[:-1], R


def weights(F, P):
    w_cap = P / P.sum(1, keepdims=True)
    w_fund = F / F.sum(1, keepdims=True)
    return w_cap, w_fund


def backtest(R, w_cap, w_fund, rebal=12):
    """买入持有、年度再平衡。返回两条净值序列。"""
    nav_cap = np.ones(T)
    nav_fund = np.ones(T)
    wc = w_cap[0].copy()
    wf = w_fund[0].copy()
    for t in range(1, T):
        nav_cap[t] = nav_cap[t - 1] * (1 + R[t - 1] @ wc)
        nav_fund[t] = nav_fund[t - 1] * (1 + R[t - 1] @ wf)
        # 持有期收益后更新权重（价格变动）
        wc = wc * (1 + R[t - 1])
        wf = wf * (1 + R[t - 1])
        wc /= wc.sum(); wf /= wf.sum()
        if t % rebal == 0:  # 年度再平衡回原始目标权重
            wc = w_cap[t].copy()
            wf = w_fund[t].copy()
    return nav_cap, nav_fund


def hhi(w):
    return np.sum(w ** 2, axis=1)


def turnover(w_target, w_prev):
    return 0.5 * np.sum(np.abs(w_target - w_prev))


# ---------------------------------------------------------------------------
# 1) 封面：累计净值
# ---------------------------------------------------------------------------
def make_cover(R, w_cap, w_fund):
    nav_cap, nav_fund = backtest(R, w_cap, w_fund)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(nav_cap, color=C["cap"], lw=2.2, label=f"市值加权 (终值 {nav_cap[-1]:.2f}x)")
    ax.plot(nav_fund, color=C["fund"], lw=2.2, label=f"基本面加权 (终值 {nav_fund[-1]:.2f}x)")
    ax.set_yscale("log")
    ax.set_xlabel("月份")
    ax.set_ylabel("累计净值（对数轴）")
    ax.set_title("基本面加权：不被价格牵着走，终值更高", fontsize=14, fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.5, -0.18, "合成数据：收益来自价格相对基本面的均值回复（量级仅演示）",
            transform=ax.transAxes, ha="center", fontsize=9.5, color="#555")
    fig.savefig(os.path.join(IMG, "cover.png"))
    plt.close(fig)
    return nav_cap[-1], nav_fund[-1]


# ---------------------------------------------------------------------------
# 2) 集中度 + 换手率
# ---------------------------------------------------------------------------
def make_concentration(w_cap, w_fund, R, rebal=12):
    hhi_cap = hhi(w_cap)
    hhi_fund = hhi(w_fund)
    # 换手率：每年初目标权重相对上一年末持有权重的变动
    to_cap, to_fund = [], []
    for t in range(rebal, T, rebal):
        to_fund.append(turnover(w_fund[t], w_fund[t - 1]))
        to_cap.append(turnover(w_cap[t], w_cap[t - 1]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(hhi_cap * 100, color=C["cap"], lw=1.6, label="市值加权")
    axes[0].plot(hhi_fund * 100, color=C["fund"], lw=1.6, label="基本面加权")
    axes[0].set_xlabel("月份"); axes[0].set_ylabel("HHI (×100，越高越集中)")
    axes[0].set_title("集中度：市值加权更押注少数大块头", fontsize=12, fontweight="bold")
    axes[0].legend(frameon=False)
    bars = axes[1].bar(["市值加权", "基本面加权"],
                       [np.mean(to_cap) * 100, np.mean(to_fund) * 100],
                       color=[C["cap"], C["fund"]], width=0.5, alpha=0.9)
    for bar, v in zip(bars, [np.mean(to_cap) * 100, np.mean(to_fund) * 100]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                     f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("年度再平衡换手率 (%)")
    axes[1].set_title("换手率：基本面加权锚定慢变量、更稳更省", fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(IMG, "fw_concentration.png"))
    plt.close(fig)
    return dict(mean_hhi_cap=hhi_cap.mean() * 100, mean_hhi_fund=hhi_fund.mean() * 100,
                mean_to_cap=np.mean(to_cap) * 100, mean_to_fund=np.mean(to_fund) * 100)


# ---------------------------------------------------------------------------
# 3) 估值倾斜散点
# ---------------------------------------------------------------------------
def make_valuation_tilt(w_cap, w_fund, F, P, t=240):
    val = (P[t] / F[t])  # P/F：>1 偏贵
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(val, w_cap[t] * 100, s=28, color=C["cap"], alpha=0.7)
    c0 = np.corrcoef(val, w_cap[t])[0, 1]
    axes[0].set_xlabel("估值 P/F（>1 = 偏贵）")
    axes[0].set_ylabel("市值加权权重 (%)")
    axes[0].set_title(f"市值加权：权重追涨贵股 (corr={c0:.2f})", fontsize=12, fontweight="bold")
    axes[1].scatter(val, w_fund[t] * 100, s=28, color=C["fund"], alpha=0.7)
    c1 = np.corrcoef(val, w_fund[t])[0, 1]
    axes[1].set_xlabel("估值 P/F（>1 = 偏贵）")
    axes[1].set_ylabel("基本面加权权重 (%)")
    axes[1].set_title(f"基本面加权：权重与估值脱钩 (corr={c1:.2f})", fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(IMG, "fw_valuation_tilt.png"))
    plt.close(fig)
    return c0, c1


if __name__ == "__main__":
    F, P, R = gen_panel()
    w_cap, w_fund = weights(F, P)
    nav_cap, nav_fund = backtest(R, w_cap, w_fund)
    end_cap, end_fund = make_cover(R, w_cap, w_fund)
    conc = make_concentration(w_cap, w_fund, R)
    corr_cap, corr_fund = make_valuation_tilt(w_cap, w_fund, F, P)
    ann_cap = (end_cap ** (12.0 / T) - 1) * 100
    ann_fund = (end_fund ** (12.0 / T) - 1) * 100
    print("=" * 60)
    print(f"市值加权终值          = {end_cap:.3f}x  (年化 {ann_cap:.2f}%)")
    print(f"基本面加权终值        = {end_fund:.3f}x  (年化 {ann_fund:.2f}%)")
    print(f"超额（基本面-市值）   = {(end_fund/end_cap-1)*100:.2f}% 累计 / "
          f"{(ann_fund-ann_cap):.2f}% 年化")
    print(f"平均 HHI 市值         = {conc['mean_hhi_cap']:.2f}  vs 基本面 {conc['mean_hhi_fund']:.2f}")
    print(f"年度换手率 市值       = {conc['mean_to_cap']:.1f}%  vs 基本面 {conc['mean_to_fund']:.1f}%")
    print(f"权重-估值相关 市值    = {corr_cap:.3f}  vs 基本面 {corr_fund:.3f}")
    print("=" * 60)
