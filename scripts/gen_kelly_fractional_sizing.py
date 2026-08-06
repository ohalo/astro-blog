#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分数 Kelly 仓位：为什么全 Kelly 是理论最优却是实战自杀
所有图表由真实计算生成，固定随机种子可复现。
SEED = 20260807
"""
import json, os, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["figure.facecolor"] = "white"

IMG_DIR = "/Users/halo/workspace/astro-blog/public/images/kelly-fractional-sizing"
os.makedirs(IMG_DIR, exist_ok=True)

SEED = 20260807

C_FULL = "#dc2626"
C_HALF = "#1e3a5f"
C_QTR  = "#2563eb"
C_OVER = "#7f1d1d"
C_OK   = "#16a34a"
C_WARN = "#e05c2a"
C_PUR  = "#9333ea"
C_GREY = "#9ca3af"

TD = 252
N_YEARS = 20
N_STEPS = N_YEARS * TD
N_PATHS = 20000

# 真实优势（构造已知）：年化超额 6%，年化波动 20%
MU_TRUE  = 0.06
SIG_TRUE = 0.20
# 连续时间 Kelly：f* = mu / sigma^2
KELLY_TRUE = MU_TRUE / SIG_TRUE ** 2


def simulate(f, mu=MU_TRUE, sig=SIG_TRUE, n_paths=N_PATHS, n_steps=N_STEPS,
             seed=SEED, cost_bps=0.0, rebal=21, fat_tail=False, jump=None):
    """
    几何布朗运动下的固定比例下注。
    对数财富增量：f*mu*dt + f*sig*dW - 0.5*f^2*sig^2*dt
    （倒数第三项就是波动拖累，全 Kelly 之上它会吃掉全部增长）
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / TD
    logw = np.zeros(n_paths)
    peak = np.zeros(n_paths)
    maxdd = np.zeros(n_paths)
    ruin = np.zeros(n_paths, dtype=bool)
    track = np.zeros((n_steps // 21 + 1, n_paths))
    ti = 0

    for t in range(n_steps):
        if fat_tail:
            # 学生 t(4)，标准化到单位方差
            z = rng.standard_t(4, n_paths) / np.sqrt(4 / 2)
        else:
            z = rng.standard_normal(n_paths)
        r = mu * dt + sig * np.sqrt(dt) * z
        if jump is not None:
            # 补偿跳跃：把跳跃的无条件期望从漂移中减回去，
            # 使总期望收益与基准一致 —— 只改尾部形状，不改优势
            p_j, size_j = jump
            hit = rng.random(n_paths) < p_j * dt
            r = r + hit * size_j - p_j * dt * size_j

        # 简单收益 → 组合简单收益 → 对数
        port_r = f * r
        port_r = np.maximum(port_r, -0.9999)     # 破产下限
        logw += np.log1p(port_r)

        if cost_bps > 0 and (t % rebal == 0):
            logw += np.log1p(-abs(f) * cost_bps * 1e-4 * 0.5)

        peak = np.maximum(peak, logw)
        maxdd = np.minimum(maxdd, logw - peak)
        ruin |= (logw < np.log(0.10))            # 亏掉 90% 视为出局

        if t % 21 == 0:
            track[ti] = logw
            ti += 1

    w = np.exp(logw)
    cagr = np.exp(logw / N_YEARS) - 1.0
    return dict(logw=logw, w=w, cagr=cagr,
                maxdd=1.0 - np.exp(maxdd),
                ruin=ruin, track=track[:ti])


def theory(f, mu=MU_TRUE, sig=SIG_TRUE):
    """对数增长率解析解 g(f) = f*mu - 0.5*f^2*sig^2"""
    return f * mu - 0.5 * f ** 2 * sig ** 2


def main():
    print("=" * 62)
    print("分数 Kelly 仓位 — 受控仿真")
    print("=" * 62)
    print(f"\n真值（构造已知）：mu={MU_TRUE:.2%}  sigma={SIG_TRUE:.2%}  "
          f"f* = mu/sigma² = {KELLY_TRUE:.4f}（{KELLY_TRUE:.2f}x 杠杆）")
    print(f"理论最大对数增长率 g(f*) = {theory(KELLY_TRUE):.6f} = {theory(KELLY_TRUE):.4%}/年")

    # =====================================================================
    # 1. 主对比：0.25x / 0.5x / 1.0x / 1.5x / 2.0x Kelly
    # =====================================================================
    print("\n[1/7] 主对比（20 年 × 20000 路径）...")
    fracs = [0.25, 0.50, 1.00, 1.50, 2.00]
    main_res = {}
    rows = []
    for fr in fracs:
        f = fr * KELLY_TRUE
        r = simulate(f)
        main_res[fr] = r
        med = float(np.median(r["cagr"]))
        rows.append(dict(frac=fr, lev=f,
                         g_theory=theory(f),
                         cagr_med=med,
                         cagr_mean=float(r["cagr"].mean()),
                         cagr_p05=float(np.percentile(r["cagr"], 5)),
                         cagr_p95=float(np.percentile(r["cagr"], 95)),
                         maxdd_med=float(np.median(r["maxdd"])),
                         maxdd_p95=float(np.percentile(r["maxdd"], 95)),
                         ruin=float(r["ruin"].mean()),
                         p_lose=float((r["w"] < 1.0).mean()),
                         p_half=float((r["maxdd"] > 0.50).mean())))
        print(f"    {fr:.2f}x Kelly (杠杆 {f:.2f}x): 理论g={theory(f):.4%}  "
              f"中位CAGR={med:7.2%}  中位MDD={np.median(r['maxdd']):6.2%}  "
              f"MDD>50%概率={rows[-1]['p_half']:6.2%}  出局={r['ruin'].mean():.3%}")
    main_df = pd.DataFrame(rows)

    # 验证：仿真中位 CAGR 应逼近理论 g（对数正态中位数 = exp(g*T)）
    dev = np.abs(np.log1p(main_df["cagr_med"]) - main_df["g_theory"]).max()
    print(f"\n    验证：中位对数增长 vs 理论 g 最大偏差 = {dev:.6f}（应≈0，纯 MC 误差）")

    # =====================================================================
    # 2. 全 Kelly 的分布本质
    # =====================================================================
    print("\n[2/7] 全 Kelly 分布诊断...")
    fk = main_res[1.00]
    hk = main_res[0.50]
    print(f"    全 Kelly:  中位财富 {np.median(fk['w']):8.2f}x  均值 {fk['w'].mean():10.2f}x  "
          f"→ 均值/中位 = {fk['w'].mean()/np.median(fk['w']):.1f}")
    print(f"    半 Kelly:  中位财富 {np.median(hk['w']):8.2f}x  均值 {hk['w'].mean():10.2f}x  "
          f"→ 均值/中位 = {hk['w'].mean()/np.median(hk['w']):.1f}")
    print(f"    全 Kelly 有 {(fk['w'] < np.median(hk['w'])).mean():.1%} 的路径跑不过半 Kelly 的中位数")
    beat = float((fk["w"] > hk["w"]).mean())
    print(f"    逐路径对比（同一随机数）：全 Kelly 跑赢半 Kelly 的比例 = {beat:.1%}")

    # 半 Kelly 保留多少增长率
    keep = theory(0.5 * KELLY_TRUE) / theory(KELLY_TRUE)
    keep25 = theory(0.25 * KELLY_TRUE) / theory(KELLY_TRUE)
    print(f"    半 Kelly 保留理论增长率 {keep:.2%}，四分之一 Kelly 保留 {keep25:.2%}")

    # =====================================================================
    # 3. 安慰剂 A：参数估计误差（真正的杀手）
    # =====================================================================
    print("\n[3/7] 安慰剂 A：用估计的 mu 而非真值下注...")
    est_rows = []
    for est_years in [1, 3, 5, 10, 20, 50]:
        # mu 的估计标准误 = sig / sqrt(T)
        se = SIG_TRUE / np.sqrt(est_years)
        rng = np.random.default_rng(SEED + 999)
        n_sim = 4000
        mu_hat = rng.normal(MU_TRUE, se, n_sim)
        f_hat = np.clip(mu_hat / SIG_TRUE ** 2, -5, 10)
        # 真实增长率用真值 mu 评估估计出的 f
        g_real = f_hat * MU_TRUE - 0.5 * f_hat ** 2 * SIG_TRUE ** 2
        g_half = (0.5 * f_hat) * MU_TRUE - 0.5 * (0.5 * f_hat) ** 2 * SIG_TRUE ** 2
        est_rows.append(dict(years=est_years, se=se,
                             f_med=float(np.median(f_hat)),
                             f_p95=float(np.percentile(f_hat, 95)),
                             g_full=float(g_real.mean()),
                             g_half=float(g_half.mean()),
                             neg_rate=float((g_real < 0).mean()),
                             over2x=float((f_hat > 2 * KELLY_TRUE).mean())))
        print(f"    估计窗 {est_years:2d} 年 (SE={se:.3%}): 全Kelly平均真实增长={g_real.mean():+.4%}  "
              f"半Kelly={g_half.mean():+.4%}  增长为负概率={est_rows[-1]['neg_rate']:.1%}  "
              f"过度下注>2x概率={est_rows[-1]['over2x']:.1%}")
    est_df = pd.DataFrame(est_rows)

    # =====================================================================
    # 4. 安慰剂 B：零优势世界（真值 f*=0）
    # =====================================================================
    print("\n[4/7] 安慰剂 B：零优势世界（mu=0，真 Kelly=0）...")
    zero_rows = []
    for fr in [0.25, 0.5, 1.0]:
        f = fr * KELLY_TRUE
        r = simulate(f, mu=0.0, seed=SEED + 31)
        zero_rows.append(dict(frac=fr, cagr_med=float(np.median(r["cagr"])),
                              g_theory=theory(f, mu=0.0),
                              maxdd_med=float(np.median(r["maxdd"]))))
        print(f"    {fr:.2f}x: 理论g={theory(f, mu=0.0):+.4%}  实测中位CAGR={np.median(r['cagr']):+.4%}  "
              f"中位MDD={np.median(r['maxdd']):.2%}")
    zero_df = pd.DataFrame(zero_rows)

    # 零波动安慰剂：sigma→0 时 Kelly 应发散，增长率线性
    r_nv = simulate(1.0, sig=1e-6, n_paths=500, seed=SEED + 41)
    print(f"    零波动检验：f=1.0 下中位CAGR={np.median(r_nv['cagr']):.6%}  "
          f"理论={np.exp(MU_TRUE)-1:.6%}  偏差={abs(np.median(r_nv['cagr'])-(np.exp(MU_TRUE)-1)):.2e}")

    # =====================================================================
    # 5. 尾部与跳跃：连续 Kelly 公式失效的地方
    # =====================================================================
    print("\n[5/7] 厚尾与跳跃冲击...")
    print("    （跳跃已做期望补偿：总期望收益与基准相同，只改尾部形状）")
    tail_scens = [("正态（基准）", {}),
                  ("学生t(4) 厚尾", {"fat_tail": True}),
                  ("补偿跳跃 -20%/年1次", {"jump": (1.0, -0.20)}),
                  ("补偿跳跃 -35%/年0.5次", {"jump": (0.5, -0.35)})]
    tail_rows = []
    for label, kw in tail_scens:
        for fr in [0.5, 1.0]:
            f = fr * KELLY_TRUE
            r = simulate(f, n_paths=8000, seed=SEED + 77, **kw)
            tail_rows.append(dict(scenario=label, frac=fr,
                                  cagr_med=float(np.median(r["cagr"])),
                                  maxdd_med=float(np.median(r["maxdd"])),
                                  maxdd_p99=float(np.percentile(r["maxdd"], 99)),
                                  ruin=float(r["ruin"].mean())))
            print(f"    {label:20s} {fr:.1f}x: 中位CAGR={np.median(r['cagr']):+7.2%}  "
                  f"中位MDD={np.median(r['maxdd']):6.2%}  99%MDD={np.percentile(r['maxdd'],99):6.2%}  "
                  f"出局={r['ruin'].mean():.2%}")
    tail_df = pd.DataFrame(tail_rows)

    # 跳跃世界里的真实最优 f：数值搜索，看连续公式高估了多少
    print("\n    数值搜索：各情形下真实最优 Kelly 分数")
    opt_rows = []
    for label, kw in tail_scens:
        best_f, best_g = None, -9e9
        for fr in np.arange(0.10, 1.65, 0.05):
            rr = simulate(fr * KELLY_TRUE, n_paths=3000, n_steps=10 * TD,
                          seed=SEED + 123, **kw)
            g = float(np.mean(rr["logw"])) / 10.0
            if g > best_g:
                best_g, best_f = g, float(fr)
        opt_rows.append(dict(scenario=label, best_frac=best_f, best_g=best_g,
                             overbet=1.0 / best_f))
        print(f"    {label:20s} 真最优 = {best_f:.2f}x 连续公式 Kelly"
              f"（连续公式高估 {1/best_f:.2f} 倍）g={best_g:+.4%}")

    # =====================================================================
    # 6. 时间维度：全 Kelly 需要多久才「赢」
    # =====================================================================
    print("\n[6/7] 时间维度：全 Kelly 跑赢半 Kelly 的概率随时间...")
    horiz = [1, 2, 3, 5, 10, 20, 40, 80]
    hz_rows = []
    for h in horiz:
        rf = simulate(1.0 * KELLY_TRUE, n_paths=8000, n_steps=h * TD, seed=SEED + 55)
        rh = simulate(0.5 * KELLY_TRUE, n_paths=8000, n_steps=h * TD, seed=SEED + 55)
        beat_h = float((rf["logw"] > rh["logw"]).mean())
        hz_rows.append(dict(years=h, beat=beat_h,
                            full_med=float(np.median(np.exp(rf["logw"]))),
                            half_med=float(np.median(np.exp(rh["logw"]))),
                            full_dd=float(np.median(rf["maxdd"])),
                            half_dd=float(np.median(rh["maxdd"]))))
        print(f"    {h:2d} 年: 全Kelly跑赢半Kelly={beat_h:.1%}  "
              f"中位财富 {np.median(np.exp(rf['logw'])):7.2f}x vs {np.median(np.exp(rh['logw'])):7.2f}x  "
              f"中位MDD {np.median(rf['maxdd']):.1%} vs {np.median(rh['maxdd']):.1%}")
    hz_df = pd.DataFrame(hz_rows)

    # 成本影响
    cost_rows = []
    for c in [0, 5, 10, 25, 50]:
        for fr in [0.5, 1.0]:
            r = simulate(fr * KELLY_TRUE, n_paths=5000, cost_bps=c, seed=SEED + 88)
            cost_rows.append(dict(cost=c, frac=fr, cagr_med=float(np.median(r["cagr"]))))
    cost_df = pd.DataFrame(cost_rows)

    # =====================================================================
    # 绘图
    # =====================================================================
    print("\n[7/7] 绘图...")
    fgrid = np.linspace(0, 2.4 * KELLY_TRUE, 400)
    gcurve = theory(fgrid)

    # ---- cover ----
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
    ax[0].plot(fgrid / KELLY_TRUE, gcurve * 100, color=C_HALF, lw=2.4,
               label="理论对数增长率 g(f)")
    ax[0].axvline(1.0, color=C_FULL, ls="--", lw=1.8, label=f"全 Kelly f*={KELLY_TRUE:.2f}x")
    ax[0].axvline(2.0, color=C_OVER, ls=":", lw=1.8, label="2x Kelly：g 回到 0")
    ax[0].axhline(0, color="black", lw=1.1)
    ax[0].scatter(main_df["frac"], main_df["g_theory"] * 100, s=70, color=C_WARN, zorder=5)
    for _, r in main_df.iterrows():
        ax[0].annotate(f"{r['frac']:.2f}x", (r["frac"], r["g_theory"] * 100),
                       textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax[0].fill_between(fgrid / KELLY_TRUE, gcurve * 100, 0,
                       where=gcurve > 0, color=C_OK, alpha=0.10)
    ax[0].fill_between(fgrid / KELLY_TRUE, gcurve * 100, 0,
                       where=gcurve <= 0, color=C_FULL, alpha=0.12)
    ax[0].set_xlabel("Kelly 分数 f / f*"); ax[0].set_ylabel("年化对数增长率 %")
    ax[0].set_title("Kelly 曲线在峰值附近是平的，右侧是悬崖", fontsize=13, fontweight="bold")
    ax[0].legend(fontsize=9)

    for fr, c in [(0.25, C_QTR), (0.50, C_HALF), (1.00, C_FULL), (2.00, C_OVER)]:
        d = main_df[main_df["frac"] == fr].iloc[0]
        ax[1].scatter(d["maxdd_med"] * 100, d["cagr_med"] * 100, s=190, color=c, zorder=5)
        ax[1].annotate(f"{fr:.2f}x Kelly", (d["maxdd_med"] * 100, d["cagr_med"] * 100),
                       textcoords="offset points", xytext=(10, -4), fontsize=10, fontweight="bold")
    ax[1].plot(main_df["maxdd_med"] * 100, main_df["cagr_med"] * 100,
               color=C_GREY, lw=1.4, ls="--", zorder=1)
    ax[1].axhline(0, color="black", lw=1.0)
    ax[1].set_xlabel("中位最大回撤 %"); ax[1].set_ylabel("中位年化收益 %")
    ax[1].set_title("回撤翻倍，收益却在缩水（20年×2万路径中位数）",
                    fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{IMG_DIR}/cover.png"); plt.close()

    # ---- distribution ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for fr, c, lb in [(0.25, C_QTR, "0.25x"), (0.50, C_HALF, "0.50x"),
                      (1.00, C_FULL, "1.00x"), (2.00, C_OVER, "2.00x")]:
        lw_ = np.log10(np.maximum(main_res[fr]["w"], 1e-6))
        ax[0].hist(lw_, bins=90, alpha=0.45, color=c, label=f"{lb} Kelly", density=True)
    ax[0].axvline(0, color="black", lw=1.4)
    ax[0].set_xlim(-4, 4)
    ax[0].set_xlabel("20 年后财富倍数（log10）"); ax[0].set_ylabel("密度")
    ax[0].set_title("终值分布：全 Kelly 是又胖又长的两端", fontsize=12, fontweight="bold")
    ax[0].legend(fontsize=9)

    yr = np.arange(main_res[1.00]["track"].shape[0]) * 21 / TD
    for fr, c, lb in [(0.50, C_HALF, "半 Kelly"), (1.00, C_FULL, "全 Kelly")]:
        tk = np.exp(main_res[fr]["track"])
        ax[1].plot(yr, np.median(tk, axis=1), color=c, lw=2.2, label=f"{lb} 中位")
        ax[1].fill_between(yr, np.percentile(tk, 25, axis=1),
                           np.percentile(tk, 75, axis=1), color=c, alpha=0.16)
        ax[1].plot(yr, np.percentile(tk, 5, axis=1), color=c, lw=1.0, ls=":")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("年"); ax[1].set_ylabel("财富倍数（对数轴）")
    ax[1].set_title("阴影=四分位区间，虚线=5% 分位", fontsize=12, fontweight="bold")
    ax[1].legend(fontsize=9)

    x = np.arange(len(fracs)); wd = 0.36
    ax[2].bar(x - wd / 2, main_df["maxdd_med"] * 100, wd, color=C_HALF, label="中位最大回撤")
    ax[2].bar(x + wd / 2, main_df["maxdd_p95"] * 100, wd, color=C_FULL, label="95分位最大回撤")
    ax[2].axhline(50, color=C_WARN, ls="--", lw=1.6, label="腰斩线")
    ax[2].set_xticks(x); ax[2].set_xticklabels([f"{f:.2f}x" for f in fracs])
    ax[2].set_xlabel("Kelly 分数"); ax[2].set_ylabel("最大回撤 %")
    ax[2].set_title("全 Kelly 的 95 分位回撤已经不可执行", fontsize=12, fontweight="bold")
    ax[2].legend(fontsize=9)
    for i, v in enumerate(main_df["maxdd_p95"] * 100):
        ax[2].text(i + wd / 2, v + 1.2, f"{v:.0f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{IMG_DIR}/distribution.png"); plt.close()

    # ---- estimation ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].plot(est_df["years"], est_df["g_full"] * 100, color=C_FULL, lw=2.2,
               marker="o", label="全 Kelly（用估计值）")
    ax[0].plot(est_df["years"], est_df["g_half"] * 100, color=C_HALF, lw=2.2,
               marker="s", label="半 Kelly（用估计值）")
    ax[0].axhline(theory(KELLY_TRUE) * 100, color=C_OK, ls="--", lw=1.8,
                  label=f"已知真值上限 {theory(KELLY_TRUE):.2%}")
    ax[0].axhline(0, color="black", lw=1.0)
    ax[0].set_xscale("log"); ax[0].set_xticks(est_df["years"]); ax[0].set_xticklabels(est_df["years"])
    ax[0].set_xlabel("用于估计 μ 的历史长度（年）"); ax[0].set_ylabel("真实平均对数增长率 %")
    ax[0].set_title("参数不确定下，半 Kelly 全程压过全 Kelly", fontsize=12, fontweight="bold")
    ax[0].legend(fontsize=8)

    ax[1].plot(est_df["years"], est_df["neg_rate"] * 100, color=C_FULL, lw=2.2, marker="o",
               label="增长率为负的概率")
    ax[1].plot(est_df["years"], est_df["over2x"] * 100, color=C_OVER, lw=2.0, marker="^",
               ls="--", label="下注超过 2x 真 Kelly 的概率")
    ax[1].set_xscale("log"); ax[1].set_xticks(est_df["years"]); ax[1].set_xticklabels(est_df["years"])
    ax[1].set_xlabel("历史长度（年）"); ax[1].set_ylabel("概率 %")
    ax[1].set_title("样本越短，越容易把杠杆开到自毁区", fontsize=12, fontweight="bold")
    ax[1].legend(fontsize=8)

    zb = ax[2].bar([f"{r['frac']:.2f}x" for _, r in zero_df.iterrows()],
                   zero_df["cagr_med"] * 100,
                   color=[C_QTR, C_HALF, C_FULL], alpha=0.88)
    ax[2].axhline(0, color="black", lw=1.2)
    for r, (_, row) in zip(zb, zero_df.iterrows()):
        ax[2].text(r.get_x() + r.get_width() / 2, row["cagr_med"] * 100 - 0.25,
                   f"{row['cagr_med']:+.2%}\n(理论 {row['g_theory']:+.2%})",
                   ha="center", va="top", fontsize=9)
    ax[2].set_ylabel("中位年化收益 %")
    ax[2].set_title("安慰剂：零优势世界里 Kelly 只剩波动拖累", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{IMG_DIR}/estimation.png"); plt.close()

    # ---- tails & horizon ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    scen = tail_df["scenario"].unique()
    xx = np.arange(len(scen)); wd = 0.36
    h = tail_df[tail_df["frac"] == 0.5].set_index("scenario").loc[scen]
    f_ = tail_df[tail_df["frac"] == 1.0].set_index("scenario").loc[scen]
    ax[0].bar(xx - wd / 2, h["maxdd_p99"] * 100, wd, color=C_HALF, label="半 Kelly")
    ax[0].bar(xx + wd / 2, f_["maxdd_p99"] * 100, wd, color=C_FULL, label="全 Kelly")
    for i, row in enumerate(opt_rows):
        ax[0].annotate(f"真最优\n{row['best_frac']:.2f}x", (i, 4), ha="center",
                       va="bottom", fontsize=8, color=C_OK, fontweight="bold")
    ax[0].set_xticks(xx); ax[0].set_xticklabels(scen, fontsize=8, rotation=12)
    ax[0].set_ylabel("99 分位最大回撤 %")
    ax[0].set_title("同一优势下只换尾部形状", fontsize=12, fontweight="bold")
    ax[0].legend(fontsize=9)

    ax[1].plot(hz_df["years"], hz_df["beat"] * 100, color=C_FULL, lw=2.4, marker="o")
    ax[1].axhline(50, color="black", ls="--", lw=1.4, label="50% 分水岭")
    ax[1].set_xscale("log"); ax[1].set_xticks(horiz); ax[1].set_xticklabels(horiz)
    ax[1].set_xlabel("持有年限"); ax[1].set_ylabel("全 Kelly 跑赢半 Kelly 的路径比例 %")
    ax[1].set_title("同一批随机数下的逐路径胜率", fontsize=12, fontweight="bold")
    ax[1].legend(fontsize=9)

    for fr, c, lb in [(0.5, C_HALF, "半 Kelly"), (1.0, C_FULL, "全 Kelly")]:
        d = cost_df[cost_df["frac"] == fr]
        ax[2].plot(d["cost"], d["cagr_med"] * 100, color=c, lw=2.2, marker="o", label=lb)
    ax[2].axhline(0, color="black", lw=1.0)
    ax[2].set_xlabel("再平衡成本 (bp / 月)"); ax[2].set_ylabel("中位年化收益 %")
    ax[2].set_title("成本对高杠杆的伤害是线性放大的", fontsize=12, fontweight="bold")
    ax[2].legend(fontsize=9)
    plt.tight_layout(); plt.savefig(f"{IMG_DIR}/tails_horizon.png"); plt.close()

    # ---- stats.json ----
    stats = {
        "seed": SEED, "n_paths": N_PATHS, "n_years": N_YEARS,
        "truth": {"mu": MU_TRUE, "sigma": SIG_TRUE, "kelly": round(KELLY_TRUE, 6),
                  "g_max": round(theory(KELLY_TRUE), 8)},
        "main": main_df.round(6).to_dict("list"),
        "median_vs_theory_maxdev": float(dev),
        "growth_retention": {"half": round(keep, 6), "quarter": round(keep25, 6)},
        "full_vs_half": {
            "full_median_wealth": round(float(np.median(fk["w"])), 4),
            "full_mean_wealth": round(float(fk["w"].mean()), 4),
            "half_median_wealth": round(float(np.median(hk["w"])), 4),
            "half_mean_wealth": round(float(hk["w"].mean()), 4),
            "full_mean_over_median": round(float(fk["w"].mean() / np.median(fk["w"])), 4),
            "beat_rate_20y": round(beat, 6),
            "full_below_half_median": round(float((fk["w"] < np.median(hk["w"])).mean()), 6)},
        "estimation_scan": est_df.round(8).to_dict("list"),
        "zero_edge_placebo": zero_df.round(8).to_dict("list"),
        "zero_vol_check": {"median_cagr": float(np.median(r_nv["cagr"])),
                           "theory": float(np.exp(MU_TRUE) - 1),
                           "dev": float(abs(np.median(r_nv["cagr"]) - (np.exp(MU_TRUE) - 1)))},
        "tail_scan": tail_df.round(6).to_dict("records"),
        "jump_optimal_f": [{k: (round(v, 8) if isinstance(v, float) else v)
                            for k, v in r.items()} for r in opt_rows],
        "horizon_scan": hz_df.round(6).to_dict("list"),
        "cost_scan": cost_df.round(6).to_dict("records"),
    }
    with open(f"{IMG_DIR}/stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 62)
    for fn in sorted(os.listdir(IMG_DIR)):
        print(f"  {fn:24s} {os.path.getsize(os.path.join(IMG_DIR, fn))/1024:8.1f} KB")
    print("=" * 62)


if __name__ == "__main__":
    main()
