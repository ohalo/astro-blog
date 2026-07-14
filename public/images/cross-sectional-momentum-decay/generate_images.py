# -*- coding: utf-8 -*-
"""
截面动量衰减曲线 配图生成脚本
生成 4 张 PNG 到本脚本同级目录：
  1. momo_decay_curve.png     - 信号 IC 随持有天数衰减曲线
  2. momo_holding_sharpe.png  - 不同持有期的多空组合夏普（含/不含成本）
  3. momo_turnover_cost.png   - 换手率 vs 净收益（成本侵蚀）
  4. momo_cumret_hold.png     - 不同持有期多空组合净值曲线

数据为自洽合成：横截面动量信号真实但微弱(IC~0.04)，对未来收益的预测力随
持有天数指数衰减。短持有期能吃到更多峰值 alpha，但换手更高、被成本反噬；
中等持有期在 alpha 与成本间取得平衡。数量级刻意压到接近真实(IC 0.03~0.05,
毛夏普 1~3)。仅用于演示方法，真实落地请用真实截面收益，见文章文末说明。
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
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20260715)


def simulate_panel(n_days=1500, n_stocks=300):
    """
    模拟横截面面板：
      - 每只股票有一个『真实预期收益漂移』mu_t（缓变，AR 快速均值回复 → 动量会腐烂）；
      - 日收益 = mu_t 当期分量 + 大量特异噪声；
      - 我们用『过去 form 日累计收益』作为可观测动量信号（带估计噪声）。
    alpha 控制信号强度，noise 主导 → IC 落在真实量级(~0.04)。
    decay 控制 mu 的均值回复速度 → 决定动量腐烂快慢。
    返回 rets (T×N 日收益)。
    """
    phi, sigma_mu, idio = 0.965, 0.0011, 0.016
    n_factors, fac_load, fac_vol = 3, 0.012, 0.012
    innov = np.sqrt(1 - phi ** 2) * sigma_mu
    mu = rng.standard_normal(n_stocks) * sigma_mu
    # 固定因子暴露（产生横截面相关 → 降低有效分散、拉低多空夏普到真实量级）
    B = rng.standard_normal((n_stocks, n_factors)) * fac_load
    rets = np.zeros((n_days, n_stocks))
    for t in range(n_days):
        # 预期收益漂移 = 强持续 AR(1)，phi 越接近1 → 动量腐烂越慢
        mu = phi * mu + innov * rng.standard_normal(n_stocks)
        f = rng.standard_normal(n_factors) * fac_vol       # 当日因子收益
        common = B @ f
        rets[t] = mu + common + rng.standard_normal(n_stocks) * idio
    return rets


def rank_ic(a, b):
    """Spearman 秩相关。"""
    ar = pd.Series(a).rank().values
    br = pd.Series(b).rank().values
    if np.std(ar) == 0 or np.std(br) == 0:
        return 0.0
    return np.corrcoef(ar, br)[0, 1]


def main():
    rets = simulate_panel()
    T, N = rets.shape
    form = 60  # 形成期：过去 60 日累计动量（经典 12-1 的日频缩影）
    momentum = pd.DataFrame(rets).rolling(form).sum().shift(1).values

    # ---------- 图1: IC 衰减曲线 ----------
    # 关键：测『形成期信号 vs 未来第 k 天当日收益』的边际 IC，
    # 而非累计收益——后者会掉进『越长越大』的陷阱。边际 IC 才会真正衰减。
    horizons = list(range(1, 41))
    ic_by_h = []
    for k in horizons:
        fwd = pd.DataFrame(rets).shift(-k).values   # 未来第 k 天的当日收益
        ics = []
        for t in range(form + 5, T - k - 1, 3):
            rm, rf = momentum[t], fwd[t]
            mask = ~(np.isnan(rm) | np.isnan(rf))
            if mask.sum() > 30:
                ics.append(rank_ic(rm[mask], rf[mask]))
        ic_by_h.append(np.nanmean(ics))
    ic_by_h = np.array(ic_by_h)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(horizons, ic_by_h, marker="o", ms=3.5, color="#1f77b4", lw=1.4)
    ax.axhline(0, color="#888", lw=0.8)
    half = ic_by_h[0] / 2
    # 找半衰点
    below = np.where(ic_by_h <= half)[0]
    hl = horizons[below[0]] if len(below) else None
    ax.axhline(half, color="#d62728", ls="--", lw=1,
               label=f"半衰线 IC₁/2={half:.3f}" + (f"（约第 {hl} 天）" if hl else ""))
    ax.set_title("横截面动量信号的边际 IC 衰减：预测力随天数逐日『腐烂』", fontsize=13)
    ax.set_xlabel("向前第 k 天（当日收益）")
    ax.set_ylabel("秩相关 IC(动量, 未来第 k 日当日收益)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "momo_decay_curve.png"), dpi=130)
    plt.close(fig)

    # ---------- 多空组合 ----------
    def long_short(hold, cost_bps=0.0):
        """每 hold 天调仓，多 top20% 空 bottom20%，等权。返回日频组合收益与年化换手。"""
        port_ret = np.zeros(T)
        pos = np.zeros(N)
        total_turnover = 0.0
        n_rebal = 0
        for t in range(form + 1, T):
            if (t - form - 1) % hold == 0:
                sig = momentum[t]
                valid = ~np.isnan(sig)
                new_pos = np.zeros(N)
                if valid.sum() > 20:
                    idx_valid = np.where(valid)[0]
                    ranks = pd.Series(sig[valid]).rank(pct=True).values
                    longs = idx_valid[ranks >= 0.8]
                    shorts = idx_valid[ranks <= 0.2]
                    if len(longs) and len(shorts):
                        new_pos[longs] = 1.0 / len(longs)
                        new_pos[shorts] = -1.0 / len(shorts)
                turnover = np.sum(np.abs(new_pos - pos))
                total_turnover += turnover
                n_rebal += 1
                port_ret[t] -= turnover * cost_bps / 1e4
                pos = new_pos
            port_ret[t] += np.nansum(pos * rets[t])
        ann_turnover = (total_turnover / max(n_rebal, 1)) * (252 / hold)
        return port_ret[form + 1:], ann_turnover

    def sharpe(x):
        return np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(252)

    holds = [1, 2, 3, 5, 10, 15, 20, 30, 40]
    sharpe_nc, sharpe_c, turns, net_ann = [], [], [], []
    for h in holds:
        r0, _ = long_short(h, 0)
        r1, tovr = long_short(h, 10)  # 单边 10bps
        sharpe_nc.append(sharpe(r0))
        sharpe_c.append(sharpe(r1))
        turns.append(tovr)
        net_ann.append(np.mean(r1) * 252 * 100)

    # ---------- 图2: 持有期 vs 夏普 ----------
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(holds))
    ax.plot(x, sharpe_nc, marker="o", color="#2ca02c", lw=1.4, label="毛夏普（不含成本）")
    ax.plot(x, sharpe_c, marker="s", color="#d62728", lw=1.4, label="净夏普（单边 10bps）")
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in holds])
    ax.set_title("持有期 vs 多空夏普：高频毛 alpha 更高，净 alpha 被成本吃掉", fontsize=13)
    ax.set_xlabel("持有期（天）")
    ax.set_ylabel("年化夏普比率")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "momo_holding_sharpe.png"), dpi=130)
    plt.close(fig)

    # ---------- 图3: 换手率 vs 净收益 ----------
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(turns, net_ann, marker="o", color="#ff7f0e", lw=1.4)
    for h, tx, ny in zip(holds, turns, net_ann):
        ax.annotate(f"{h}日", (tx, ny), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_title("年化换手率 vs 净收益：换手越高，成本侵蚀越狠", fontsize=13)
    ax.set_xlabel("年化换手率（对数轴）")
    ax.set_ylabel("含成本年化净收益 (%)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "momo_turnover_cost.png"), dpi=130)
    plt.close(fig)

    # ---------- 图4: 净值曲线 ----------
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for h, color in zip([1, 5, 10, 20], ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]):
        r1, _ = long_short(h, 10)
        eq = np.cumprod(1 + r1)
        ax.plot(eq, label=f"持有 {h} 天", color=color, lw=1.3)
    ax.axhline(1, color="#888", lw=0.8)
    ax.set_title("含成本多空净值：中等持有期在 alpha 与成本间取得平衡", fontsize=13)
    ax.set_xlabel("交易日")
    ax.set_ylabel("净值（起始=1）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "momo_cumret_hold.png"), dpi=130)
    plt.close(fig)

    print("IC(h=1):", round(ic_by_h[0], 4), " IC(h=20):", round(ic_by_h[19], 4),
          " IC(h=40):", round(ic_by_h[-1], 4))
    print("Sharpe gross:", dict(zip(holds, [round(s, 2) for s in sharpe_nc])))
    print("Sharpe net  :", dict(zip(holds, [round(s, 2) for s in sharpe_c])))
    print("Turnover(ann):", dict(zip(holds, [round(t, 1) for t in turns])))
    print("Net ann ret %:", dict(zip(holds, [round(r, 2) for r in net_ann])))
    print("Saved 4 PNGs to", OUT_DIR)


if __name__ == "__main__":
    main()
