#!/usr/bin/env python3
"""
为文章「采样频率偏差：日频 vs 周频回测为何给出不同夏普」
(sampling-frequency-bias) 生成真实配图。所有图表均由文中方法真实计算生成。

机制（自洽合成，仅用于演示方法；真实落地见文末路径）：
  * 生成带自相关 (AR(1)) 的日收益序列:
        r_t = phi * r_{t-1} + eps_t,   eps_t ~ N(mu, sigma_eps)
  * 日频夏普:  S_d = mean(r)/std(r) ; 年化: S_d * sqrt(252)
  * 周频夏普:  周收益 = 连续 5 日日收益之和, 年化: S_w * sqrt(252/5)
  * 关键: 当 phi=0 (iid), 两种年化夏普一致;
          当 phi>0 (动量/正自相关), 周频年化夏普 < 日频 (聚合把同向波动叠加, 抬高了周收益方差, 夏普被压低);
          当 phi<0 (均值回复/负自相关), 周频年化夏普 > 日频 (聚合把反向波动抵消, 周收益方差被压低, 夏普被抬高)。
  * 通过蒙特卡洛 (多 phi) 展示两种频率年化夏普的系统性偏差。
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
D = os.path.join(BASE, "sampling-frequency-bias")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "x": "#4C72B0", "y": "#DD8452", "base": "#55A868",
     "band": "#C44E52", "eq": "#2F4B7C", "bh": "#CCB974", "mk": "#8172B3",
     "null": "#999999", "out": "#C44E52", "acc": "#4C72B0"}

rng = np.random.default_rng(2026)
N_DAYS = 252 * 12          # 约 12 年日数据
MU_D = 0.0004              # 日收益漂移 (主动 alpha)
SIGMA_D = 0.010            # 目标日波动率
N_MC = 200                 # 蒙特卡洛次数


def gen_ar1(phi, n=N_DAYS, mu=MU_D, sd=SIGMA_D, rng=rng):
    eps = rng.normal(mu, sd, n)
    r = np.zeros(n)
    r[0] = eps[0]
    for t in range(1, n):
        r[t] = phi * r[t - 1] + eps[t]
    return r


def ann_sharpe_daily(r):
    sd = r.std(ddof=1)
    return (r.mean() / sd) * np.sqrt(252)


def ann_sharpe_weekly(r, h=5):
    # 周收益 = 每 h 个日收益求和
    n = (len(r) // h) * h
    weekly = r[:n].reshape(-1, h).sum(axis=1)
    sw = weekly.std(ddof=1)
    return (weekly.mean() / sw) * np.sqrt(252 / h)


def main():
    phis = np.linspace(-0.3, 0.3, 25)
    daily_means, weekly_means, daily_std, weekly_std = [], [], [], []
    for phi in phis:
        sd_l, sw_l = [], []
        for _ in range(N_MC):
            r = gen_ar1(phi)
            sd_l.append(ann_sharpe_daily(r))
            sw_l.append(ann_sharpe_weekly(r))
        daily_means.append(np.mean(sd_l))
        weekly_means.append(np.mean(sw_l))
        daily_std.append(np.std(sd_l))
        weekly_std.append(np.std(sw_l))

    daily_means = np.array(daily_means)
    weekly_means = np.array(weekly_means)
    daily_std = np.array(daily_std)
    weekly_std = np.array(weekly_std)

    # ===== Fig 1: 日频 vs 周频年化夏普 随 phi 变化 =====
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(phis, daily_means, color=C["x"], lw=2, label="日频年化夏普 (×√252)")
    ax.plot(phis, weekly_means, color=C["band"], lw=2, label="周频年化夏普 (×√(252/5))")
    ax.axvline(0, color=C["null"], ls="--", lw=1.2, label="φ=0 (iid, 两线重合)")
    # 标注偏差方向
    ax.annotate("φ>0 动量:\n周频 < 日频\n(聚合抬升周波动, 夏普被压低)",
                xy=(0.22, weekly_means[phis > 0.2][0]), xytext=(0.05, weekly_means.max() * 0.92),
                fontsize=8, color=C["band"],
                arrowprops=dict(arrowstyle="->", color=C["band"]))
    ax.annotate("φ<0 回复:\n周频 > 日频\n(聚合压低周波动, 夏普被抬高)",
                xy=(-0.22, weekly_means[phis < -0.2][0]), xytext=(-0.32, weekly_means.min() * 0.92),
                fontsize=8, color=C["x"],
                arrowprops=dict(arrowstyle="->", color=C["x"]))
    ax.set_xlabel("日收益一阶自相关 φ (AR(1))")
    ax.set_ylabel("年化夏普比率")
    ax.set_title("同一策略在不同采样频率下年化夏普系统性分化 (φ=0 处重合)", fontsize=12)
    ax.legend(loc="center right", fontsize=9)
    ax.grid(True, color=C["grid"], ls=":")
    plt.tight_layout()
    fig.savefig(os.path.join(D, "sharpe_vs_autocorr.png"), dpi=130)
    plt.close(fig)

    # ===== Fig 2: 偏差百分比 (周频相对日频) 随 phi =====
    bias_pct = (weekly_means - daily_means) / np.abs(daily_means) * 100
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar(phis, bias_pct, width=0.018, color=np.where(bias_pct >= 0, C["band"], C["x"]))
    ax.axhline(0, color=C["null"], lw=1.2)
    ax.axvline(0, color=C["null"], ls="--", lw=1.2)
    ax.set_xlabel("日收益一阶自相关 φ (AR(1))")
    ax.set_ylabel("周频 vs 日频 年化夏普偏差 (%)")
    ax.set_title("频率偏差百分比: 正自相关抬高、负自相关压低周频夏普", fontsize=12)
    ax.grid(True, color=C["grid"], axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(D, "frequency_bias_pct.png"), dpi=130)
    plt.close(fig)

    # ===== Fig 3: 蒙特卡洛散布 (φ=+0.2 vs φ=-0.2) =====
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, phi, title in [(axes[0], 0.20, "φ = +0.20 (正自相关/动量)"),
                           (axes[1], -0.20, "φ = -0.20 (负自相关/回复)")]:
        sd_l, sw_l = [], []
        for _ in range(N_MC):
            r = gen_ar1(phi)
            sd_l.append(ann_sharpe_daily(r))
            sw_l.append(ann_sharpe_weekly(r))
        ax.scatter(sd_l, sw_l, s=14, alpha=0.5, color=C["mk"])
        lim = [min(min(sd_l), min(sw_l)) - 0.1, max(max(sd_l), max(sw_l)) + 0.1]
        ax.plot(lim, lim, color=C["null"], ls="--", lw=1.2, label="对角线 (两频一致)")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("日频年化夏普")
        ax.set_ylabel("周频年化夏普")
        ax.set_title(f"{title}\n点落在对角线{'' if phi>=0 else '上'}{'' if phi<0 else '下'}方", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, color=C["grid"], ls=":")
    plt.tight_layout()
    fig.savefig(os.path.join(D, "mc_scatter_phi.png"), dpi=130)
    plt.close(fig)

    print("==== DIAGNOSTICS (sampling-frequency-bias) ====")
    for phi in [-0.20, -0.10, 0.0, 0.10, 0.20]:
        i = np.argmin(np.abs(phis - phi))
        print(f"phi={phi:+.2f}  daily_annSR={daily_means[i]:.3f}  weekly_annSR={weekly_means[i]:.3f}  bias={bias_pct[i]:+.1f}%")
    # 单点精确演示
    r0 = gen_ar1(0.0)
    print(f"[iid检查] phi=0 daily={ann_sharpe_daily(r0):.3f} weekly={ann_sharpe_weekly(r0):.3f}")
    r_pos = gen_ar1(0.15)
    r_neg = gen_ar1(-0.15)
    print(f"[phi=+0.15] daily={ann_sharpe_daily(r_pos):.3f} weekly={ann_sharpe_weekly(r_pos):.3f}  gap={ann_sharpe_weekly(r_pos)-ann_sharpe_daily(r_pos):+.3f}")
    print(f"[phi=-0.15] daily={ann_sharpe_daily(r_neg):.3f} weekly={ann_sharpe_weekly(r_neg):.3f}  gap={ann_sharpe_weekly(r_neg)-ann_sharpe_daily(r_neg):+.3f}")
    print("images saved to", D)
    print(os.listdir(D))


if __name__ == "__main__":
    main()
