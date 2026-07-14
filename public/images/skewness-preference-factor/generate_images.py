# -*- coding: utf-8 -*-
"""
偏度偏好因子 (Skewness Preference Factor) 配图生成脚本
生成 4 张 PNG 到本脚本同级 public/images/skewness-preference-factor/ 目录：
  1. skew_quintile_returns.png  - 五分位组合未来收益柱状图
  2. skew_long_short.png        - 多空组合净值曲线
  3. skew_distribution.png      - 截面偏度分布直方图 + 偏度 vs 未来收益散点
  4. skew_drawdown.png          - 多空组合回撤曲线

数据为自洽合成：每只股票带"彩票型"跳涨，形成期偏度(iskew)越高，
未来 1 月收益被设定为越低（彩票溢价透支），以此还原偏度偏好异象。
仅用于演示方法，真实落地请用实际日收益，见文章文末说明。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["PingFang SC", "Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # 屏蔽 SimHei 缺失噪声

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def skew(x):
    """样本偏度（三阶标准矩）。"""
    x = np.asarray(x, dtype=float)
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0:
        return 0.0
    return np.mean((x - m) ** 3) / (s ** 3)


def simulate_panel(n_months=180, n_stocks=500, seed=20260715):
    """模拟面板数据：每个月 N 只股票的 iskew 与未来 1 月收益。

    设计要点（仅用于演示方法）：
    - gamma：个股真实的"彩票偏度倾向"，横截面右偏（Gamma 分布）。
    - 形成期收益带偶发正向跳涨，跳涨概率/幅度随 gamma 上升 -> iskew 与 gamma 强正相关。
    - 未来收益由 TRUE gamma 驱动（保证经济效应清晰）：
        基础正 drift + 共同市场因子
        - 偏度偏好异象项：-slope * pref[t] * gamma  （彩票溢价透支，高偏度未来亏）
        - 散户泡沫挤兑项：+bubble * max(pref[t]-0.85,0) * gamma
          （散户狂热峰期，高偏度彩票股被继续爆买，做空端被挤，因子回撤）
      pref[t]：约 5 年一轮的"散户彩票需求"强度，0.3~1.5 之间波动。
      这样正常期异象强（多空为正），泡沫峰期异象反转（多空为负、回撤），更贴近真实。
      因子排序用含噪的 iskew（估计噪声是真实陷阱，文末单列）。
    """
    rng = np.random.default_rng(seed)
    records = []
    for t in range(n_months):
        gamma = rng.gamma(shape=2.0, scale=0.45, size=n_stocks)
        base = rng.normal(0.0004, 0.015, size=(n_stocks, 60))
        # 跳涨概率/幅度均随 gamma 上升，使 iskew 可靠地反映 lotteryness
        jump = (rng.random((n_stocks, 60)) < 0.022 * gamma[:, None]) * rng.exponential(0.085, size=(n_stocks, 60))
        form = base + jump
        iskew = np.array([skew(form[i]) for i in range(n_stocks)])
        # 散户彩票需求强度：约 5 年一轮的周期 + 噪声
        phase = 2 * np.pi * t / 60
        pref = np.clip(0.7 + 0.5 * np.sin(phase) + rng.normal(0, 0.05), 0.3, 1.5)
        mkt = rng.normal(0.008, 0.045)
        # 正常期异象主导(多空为正)；仅散户狂热峰期(pref>1.11)做空端被挤、多空转负(回撤)
        future = (0.010 + 0.55 * mkt
                  - 0.032 * pref * gamma
                  + 0.22 * max(pref - 0.95, 0.0) * gamma
                  + rng.normal(0, 0.05, size=n_stocks))
        df = pd.DataFrame({"month": t, "iskew": iskew, "true_skew": gamma,
                           "future": future, "pref": pref})
        records.append(df)
    return pd.concat(records, ignore_index=True)


def quintile_stats(df):
    """按 iskew 五分位，返回每组平均未来收益(%)与组内平均 iskew。"""
    df = df.copy()
    df["q"] = pd.qcut(df["iskew"].rank(method="first"), 5, labels=False)
    g = df.groupby("q").agg(iskew=("iskew", "mean"), future=("future", "mean"))
    g["future"] = g["future"] * 100
    return g


def long_short_series(df):
    """每月末按 iskew 排序：多最低 20%(低偏度)，空最高 20%(高偏度)，持有 1 月。"""
    ls = []
    months = sorted(df["month"].unique())
    for m in months:
        sub = df[df["month"] == m].copy()
        sub = sub.sort_values("iskew")
        n = len(sub)
        lo = sub.iloc[: max(1, int(n * 0.2))]
        hi = sub.iloc[-max(1, int(n * 0.2)):]
        ls.append(lo["future"].mean() - hi["future"].mean())
    return np.array(ls)


def perf(r):
    r = np.asarray(r)
    cum = np.cumprod(1 + r)
    ann = cum[-1] ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12)
    sharpe = (r.mean() * 12 - 0.02) / vol
    peak = np.maximum.accumulate(cum)
    mdd = (cum / peak - 1).min()
    return ann, vol, sharpe, mdd


def main():
    df = simulate_panel()
    qs = quintile_stats(df)
    ls = long_short_series(df)
    ann, vol, sharpe, mdd = perf(ls)

    # ---- 图 1：五分位组合未来收益柱状图 ----
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [f"Q{k+1}\n(低→高偏度)" for k in range(5)]
    colors = ["#2E86C1", "#5DADE2", "#F4D03F", "#EB984E", "#E74C3C"]
    bars = ax.bar(labels, qs["future"].values, color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, qs["future"].values):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.05 if v >= 0 else -0.12),
                f"{v:+.2f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("未来 1 月平均收益 (%)", fontsize=12)
    ax.set_title("按偏度(iskew)五分位：越高偏度组合未来收益越低", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(OUT_DIR, "skew_quintile_returns.png")
    plt.savefig(p1, dpi=150)
    plt.close(fig)
    print("Generated", p1)

    # ---- 图 2：多空组合净值曲线 ----
    fig, ax = plt.subplots(figsize=(10, 6))
    cum = np.cumprod(1 + ls) * 100
    ax.plot(range(1, len(cum) + 1), cum, color="#C0392B", linewidth=2)
    ax.set_xlabel("月份", fontsize=12)
    ax.set_ylabel("净值 (起点=100)", fontsize=12)
    ax.set_title(f"多低偏度 / 空高偏度 多空组合净值 (年化 {ann*100:.1f}%, 夏普 {sharpe:.2f})",
                 fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(OUT_DIR, "skew_long_short.png")
    plt.savefig(p2, dpi=150)
    plt.close(fig)
    print("Generated", p2)

    # ---- 图 3：截面偏度分布直方图 + 散点 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    samp = df.sample(n=min(6000, len(df)), random_state=1)
    axes[0].hist(samp["iskew"], bins=60, color="#2874A6", edgecolor="white", alpha=0.9)
    axes[0].set_title("截面 iskew 分布：右偏、厚尾", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("iskew (形成期偏度)", fontsize=11)
    axes[0].set_ylabel("股票数", fontsize=11)
    axes[0].axvline(samp["iskew"].median(), color="red", linestyle="--", linewidth=1.2,
                    label=f"中位数 {samp['iskew'].median():.2f}")
    axes[0].legend()

    # 散点（抽样）+ 回归线
    x = samp["iskew"].values
    y = samp["future"].values * 100
    axes[1].scatter(x, y, s=6, alpha=0.25, color="#117A65")
    b1, b0 = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    axes[1].plot(xs, b0 + b1 * xs, color="#C0392B", linewidth=2.2,
                 label=f"斜率 {b1:+.3f}%/单位偏度")
    axes[1].set_title("偏度越高 → 未来收益越低 (负斜率)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("iskew", fontsize=11)
    axes[1].set_ylabel("未来 1 月收益 (%)", fontsize=11)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    p3 = os.path.join(OUT_DIR, "skew_distribution.png")
    plt.savefig(p3, dpi=150)
    plt.close(fig)
    print("Generated", p3)

    # ---- 图 4：多空组合回撤 ----
    fig, ax = plt.subplots(figsize=(10, 6))
    cum = np.cumprod(1 + ls)
    peak = np.maximum.accumulate(cum)
    dd = (cum / peak - 1) * 100
    ax.fill_between(range(1, len(dd) + 1), dd, 0, color="#922B21", alpha=0.35)
    ax.plot(range(1, len(dd) + 1), dd, color="#922B21", linewidth=1.5)
    ax.set_xlabel("月份", fontsize=12)
    ax.set_ylabel("回撤 (%)", fontsize=12)
    ax.set_title(f"多空组合回撤 (最大回撤 {mdd*100:.1f}%)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p4 = os.path.join(OUT_DIR, "skew_drawdown.png")
    plt.savefig(p4, dpi=150)
    plt.close(fig)
    print("Generated", p4)

    # ---- 打印统计，供文章引用 ----
    print("\n==== 统计摘要 ====")
    print("五分位未来收益(%):", [f"{v:.2f}" for v in qs["future"].values])
    print(f"多空月均收益: {ls.mean()*100:.3f}%  -> 年化 {ann*100:.2f}%  夏普 {sharpe:.2f}  最大回撤 {mdd*100:.2f}%")
    print(f"散点斜率: {b1:+.4f}%/单位偏度  截面 iskew 中位数: {df['iskew'].median():.3f}")


if __name__ == "__main__":
    main()
