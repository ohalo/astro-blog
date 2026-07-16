#!/usr/bin/env python3
"""
为文章「Beneish M-Score 财务操纵识别：用 8 个比率给财报打假分」(beneish-m-score-fraud)
生成真实配图（自洽合成，非占位图）。

模型：Beneish (1999) M-Score
  M = −4.84 + 0.920·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI
            + 0.115·DEPI − 0.172·SGAI + 4.679·TATA − 0.327·LVGI
  M > −1.78 → 高操纵嫌疑
图表：
  1. beneish_ratios.png      8 个比率：健康公司 vs 操纵公司中位数对比
  2. beneish_mscore_dist.png M-Score 两组分布 + −1.78 阈值线
  3. beneish_waterfall.png   代表性操纵公司的系数贡献瀑布
  4. beneish_roc.png         两组的分离度（ROC 近似）
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

SLUG = "beneish-m-score-fraud"
BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, SLUG)
os.makedirs(D, exist_ok=True)

COEF = dict(DSRI=0.920, GMI=0.528, AQI=0.404, SGI=0.892,
            DEPI=0.115, SGAI=-0.172, TATA=4.679, LVGI=-0.327)
INTER = -4.84
THRESH = -1.78


def mscore(r):
    return INTER + sum(COEF[k] * r[k] for k in COEF)


def gen_firm(manip=False, seed=0):
    """生成一组相邻两年财报比率（t-1, t）。"""
    rng = np.random.default_rng(seed)
    if not manip:
        # 健康：稳健增长、应收/收入同步、现金流为正、折旧正常
        rev_g = rng.uniform(0.95, 1.12)
        dsri = rng.uniform(0.85, 1.12)
        gmi = rng.uniform(0.95, 1.05)
        aqi = rng.uniform(0.92, 1.06)
        sgi = rev_g
        depi = rng.uniform(0.92, 1.08)
        sgai = rng.uniform(0.90, 1.10)
        tata = rng.uniform(-0.02, 0.04)   # 应计占比低（现金流健康）
        lvgi = rng.uniform(0.85, 1.10)
    else:
        # 操纵：应收暴涨(DSRI↑)、毛利率恶化(GMI>1)、资产质量降(AQI↑)、
        #      营收虚增(SGI↑)、现金流为负(TATA<0)、杠杆抬升(LVGI↑)
        dsri = rng.uniform(1.60, 2.60)
        gmi = rng.uniform(1.10, 1.40)
        aqi = rng.uniform(1.20, 1.50)
        sgi = rng.uniform(1.30, 1.80)
        depi = rng.uniform(0.85, 1.00)
        sgai = rng.uniform(1.10, 1.30)
        tata = rng.uniform(-0.12, -0.02)  # 利润与现金流严重背离（仍为负）
        lvgi = rng.uniform(1.20, 1.60)
    return dict(DSRI=dsri, GMI=gmi, AQI=aqi, SGI=sgi,
                DEPI=depi, SGAI=sgai, TATA=tata, LVGI=lvgi)


def main():
    rng = np.random.default_rng(20260716)
    healthy = [gen_firm(False, s) for s in rng.integers(1, 9999, 220)]
    manip = [gen_firm(True, s) for s in rng.integers(10000, 19999, 220)]
    keys = list(COEF.keys())

    # ---------- 图1：8 比率中位数对比 ----------
    h_med = [np.median([f[k] for f in healthy]) for k in keys]
    m_med = [np.median([f[k] for f in manip]) for k in keys]
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(x - w/2, h_med, w, label="健康公司", color="#2ca02c")
    ax.bar(x + w/2, m_med, w, label="操纵公司", color="#d62728")
    ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=30, fontsize=9)
    ax.set_title("Beneish 8 比率：健康 vs 操纵公司中位数对比（虚线=1.0 基准）",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("比率值"); ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(D, "beneish_ratios.png"), dpi=130); plt.close(fig)

    # ---------- 图2：M-Score 分布 + 阈值 ----------
    mh = np.array([mscore(f) for f in healthy])
    mm = np.array([mscore(f) for f in manip])
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bins = np.linspace(-4, 2, 45)
    ax.hist(mh, bins=bins, alpha=0.6, color="#2ca02c", label=f"健康 (中位 {np.median(mh):.2f})")
    ax.hist(mm, bins=bins, alpha=0.6, color="#d62728", label=f"操纵 (中位 {np.median(mm):.2f})")
    ax.axvline(THRESH, color="black", lw=2, ls="--", label=f"阈值 M = {THRESH}")
    ax.set_title("M-Score 分布：两组清晰分离，阈值 −1.78 切分", fontsize=14, fontweight="bold")
    ax.set_xlabel("M-Score"); ax.set_ylabel("公司数"); ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(D, "beneish_mscore_dist.png"), dpi=130); plt.close(fig)

    # ---------- 图3：代表性操纵公司瀑布 ----------
    rep = gen_firm(True, 12345)
    contrib = {k: COEF[k] * rep[k] for k in keys}
    order = ["截距"] + keys
    vals = [INTER] + [contrib[k] for k in keys]
    cum = np.cumsum(vals)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(range(len(order)), vals, color=["#555555"] + ["#d62728" if v > 0 else "#1f77b4" for v in vals[1:]])
    ax.plot(range(len(order)), cum, "o-", color="black", lw=1.5, ms=4)
    ax.axhline(THRESH, color="black", ls="--", lw=1.5, label=f"阈值 {THRESH}")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=30, fontsize=9)
    ax.set_title("系数贡献瀑布：代表性操纵公司累计 M-Score 越过阈值", fontsize=13, fontweight="bold")
    ax.set_ylabel("对 M 的贡献"); ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(D, "beneish_waterfall.png"), dpi=130); plt.close(fig)

    # ---------- 图4：分离度（ROC）----------
    allm = np.concatenate([mh, mm])
    thresholds = np.linspace(allm.min(), allm.max(), 200)
    fpr = []; tpr = []
    P = len(mm); N = len(mh)
    for t in thresholds:
        tp = np.sum(mm > t); fp = np.sum(mh > t)
        tpr.append(tp / P); fpr.append(fp / N)
    fpr = np.array(fpr); tpr = np.array(tpr)
    # 按 fpr 升序排序后积分，避免阈值方向导致符号错误
    order = np.argsort(fpr)
    auc = float(np.trapezoid(tpr[order], fpr[order]))
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.plot(fpr, tpr, color="#d62728", lw=2.2, label=f"ROC (AUC ≈ {auc:.2f})")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
    ax.set_title("M-Score 两组分离度（ROC）", fontsize=14, fontweight="bold")
    ax.set_xlabel("假阳性率（健康被判操纵）"); ax.set_ylabel("真阳性率（操纵被正确识别）")
    ax.legend(fontsize=11, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(D, "beneish_roc.png"), dpi=130); plt.close(fig)

    print("beneish-m-score-fraud 配图已生成：", sorted(os.listdir(D)))
    print(f"  健康中位 M={np.median(mh):.2f}  操纵中位 M={np.median(mm):.2f}  AUC≈{auc:.2f}")


if __name__ == "__main__":
    main()
