#!/usr/bin/env python3
"""
为文章「Carhart 四因子模型实战拆解」(carhart-four-factor) 生成真实配图 + 可复现指标。

所有图表都由文中代码真实计算生成（纯 numpy + 时间序列回归，无 sklearn 依赖）：

  1) cover.png                      —— 四因子月均风险溢价条形图（Mkt/SMB/HML/Mom）
  2) carhart_size_bm_heatmap.png    —— 25 个 规模×账面市值比 组合 CAPM α 热力图（暴露价值/规模溢价）
  3) carhart_alpha_comparison.png   —— CAPM α vs Carhart 四因子 α：四因子把截面 α 几乎吃光

机制（数据自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 因子：Mkt/SMB/HML/Mom 月收益，带现实量级的风险溢价与轻度相关
  - 25 组合（5×5 规模 × B/M）：真实载荷 β_SMB 随规模单调、β_HML 随 B/M 单调、β_Mom 随 B/M 递增
  - 真实 α = 0（收益完全由四因子解释）；CAPM 漏掉 SMB/HML/Mom → 截面出现系统性 α
  - WML（赢家减输家）组合：β_Mom=1、其余≈0 → CAPM α 巨大、Carhart α≈0
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
IMG = os.path.join(BASE, "carhart-four-factor")
os.makedirs(IMG, exist_ok=True)

C = {"mkt": "#4C72B0", "smb": "#55A868", "hml": "#E1A100", "mom": "#C44E52",
     "neg": "#C44E52", "pos": "#55A868"}

T = 360  # 360 个月（30 年）

# 因子风险溢价（月均，量级参考历史）
LAM = {"Mkt": 0.006, "SMB": 0.0025, "HML": 0.004, "Mom": 0.006}
SD = {"Mkt": 0.044, "SMB": 0.026, "HML": 0.026, "Mom": 0.040}
FACTORS = ["Mkt", "SMB", "HML", "Mom"]


def gen_factors(seed=20260722):
    rng = np.random.default_rng(seed)
    # 轻度相关：Mkt 与 Mom 略负相关，HML 与 SMB 略正相关
    corr = np.array([
        [1.00, 0.20, 0.15, -0.25],
        [0.20, 1.00, 0.30, 0.10],
        [0.15, 0.30, 1.00, 0.05],
        [-0.25, 0.10, 0.05, 1.00],
    ])
    L = np.linalg.cholesky(corr)
    z = rng.normal(0, 1, size=(T, 4))
    f = z @ L.T
    F = np.empty((T, 4))
    for j, name in enumerate(FACTORS):
        F[:, j] = LAM[name] + SD[name] * f[:, j]
    return F  # 列：Mkt,SMB,HML,Mom


def gen_25_portfolios(F, seed=7):
    rng = np.random.default_rng(seed)
    beta = np.zeros((5, 5, 4))  # [size_row, bm_col, factor]
    ret = np.zeros((T, 5, 5))
    for i in range(5):       # 规模：0=最小,4=最大
        for j in range(5):   # B/M：0=成长,4=价值
            b_mkt = 1.0 + rng.normal(0, 0.03)
            b_smb = (1.0 - 2 * (i / 4)) + rng.normal(0, 0.015)  # 小盘+1,大盘-1
            b_hml = (-1.0 + 2 * (j / 4)) + rng.normal(0, 0.015) # 价值+1,成长-1
            b_mom = (0.3 + 0.4 * (j / 4)) + rng.normal(0, 0.015) # 价值股略带动量暴露
            beta[i, j] = [b_mkt, b_smb, b_hml, b_mom]
            # 真实 α=0：收益 = Σ β·factor + idio
            idio = rng.normal(0, 0.010, T)
            ret[:, i, j] = F @ beta[i, j] + idio
    return ret, beta


def gen_wml(F, seed=11):
    rng = np.random.default_rng(seed)
    # 赢家减输家：β_Mom=1, 其余≈0
    beta_v = np.array([0.0, 0.0, 0.0, 1.0])
    idio = rng.normal(0, 0.025, T)
    return F @ beta_v + idio, beta_v


def ts_regress(y, X):
    """时间序列回归 y = a + X@b + e，返回 alpha(年化) 与 beta。"""
    n = len(y)
    Xd = np.column_stack([np.ones(n), X])
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return coef[0] * 12.0, coef[1:]  # alpha 年化


# ---------------------------------------------------------------------------
# 1) 封面：四因子月均风险溢价
# ---------------------------------------------------------------------------
def make_cover(F):
    means = F.mean(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    cols = [C["mkt"], C["smb"], C["hml"], C["mom"]]
    bars = ax.bar(FACTORS, means * 100, color=cols, width=0.6, alpha=0.9)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m * 100 + 0.03,
                f"{m*100:.2f}%", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("月均风险溢价 (%)")
    ax.set_title("Carhart 四因子：四个可交易的风险溢价来源", fontsize=14,
                 fontweight="bold")
    ax.text(0.5, -0.22, "Mkt=市场  SMB=规模  HML=价值  Mom=动量（合成数据，量级参考历史）",
            transform=ax.transAxes, ha="center", fontsize=9.5, color="#555")
    fig.savefig(os.path.join(IMG, "cover.png"))
    plt.close(fig)
    return means


# ---------------------------------------------------------------------------
# 2) 25 组合 CAPM α 热力图（暴露规模/价值溢价）
# ---------------------------------------------------------------------------
def make_heatmap(F, ret):
    capm_alpha = np.zeros((5, 5))
    carhart_alpha = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            y = ret[:, i, j]
            a1, _ = ts_regress(y, F[:, 0:1])           # CAPM：仅 Mkt
            a4, _ = ts_regress(y, F[:, 0:4])           # Carhart：四因子
            capm_alpha[i, j] = a1
            carhart_alpha[i, j] = a4
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    lim = max(abs(capm_alpha).max(), 0.001)
    vmax = min(lim, 0.15) if lim > 0 else 0.15
    im0 = axes[0].imshow(capm_alpha, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                         aspect="auto")
    axes[0].set_title("CAPM α（%年，仅市场因子）", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("B/M：成长 → 价值")
    axes[0].set_ylabel("规模：小盘 → 大盘")
    axes[0].set_xticks(range(5)); axes[0].set_yticks(range(5))
    for i in range(5):
        for j in range(5):
            axes[0].text(j, i, f"{capm_alpha[i,j]*100:.1f}", ha="center",
                         va="center", fontsize=9,
                         color="white" if abs(capm_alpha[i,j]) > vmax*0.6 else "black")
    im1 = axes[1].imshow(carhart_alpha, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                         aspect="auto")
    axes[1].set_title("Carhart 四因子 α（%年）", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("B/M：成长 → 价值")
    axes[1].set_yticks(range(5))
    for i in range(5):
        for j in range(5):
            axes[1].text(j, i, f"{carhart_alpha[i,j]*100:.1f}", ha="center",
                         va="center", fontsize=9,
                         color="white" if abs(carhart_alpha[i,j]) > vmax*0.6 else "black")
    fig.colorbar(im1, ax=axes, fraction=0.025, pad=0.02)
    fig.suptitle("25 个规模×B/M 组合：CAPM 留大 α，四因子几乎清零",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(os.path.join(IMG, "carhart_size_bm_heatmap.png"))
    plt.close(fig)
    return capm_alpha, carhart_alpha


# ---------------------------------------------------------------------------
# 3) α 对比条形图（截面均值 + WML）
# ---------------------------------------------------------------------------
def make_alpha_comparison(F, ret, wml):
    capm_abs, car_abs = [], []
    for i in range(5):
        for j in range(5):
            a1, _ = ts_regress(ret[:, i, j], F[:, 0:1])
            a4, _ = ts_regress(ret[:, i, j], F[:, 0:4])
            capm_abs.append(abs(a1)); car_abs.append(abs(a4))
    wml_capm, _ = ts_regress(wml, F[:, 0:1])
    wml_car, _ = ts_regress(wml, F[:, 0:4])
    fig, ax = plt.subplots(figsize=(8.5, 5))
    labels = ["25 组合平均 |α|", "WML 动量组合 α"]
    capm_vals = [np.mean(capm_abs) * 100, wml_capm * 100]
    car_vals = [np.mean(car_abs) * 100, wml_car * 100]
    x = np.arange(2); w = 0.35
    ax.bar(x - w/2, capm_vals, w, label="CAPM（仅 Mkt）", color=C["neg"], alpha=0.85)
    ax.bar(x + w/2, car_vals, w, label="Carhart 四因子", color=C["pos"], alpha=0.85)
    for xi, c, k in zip(x - w/2, capm_vals, ["c", "c"]):
        ax.text(xi, c + 0.05, f"{c:.2f}%", ha="center", fontsize=10.5, fontweight="bold")
    for xi, c in zip(x + w/2, car_vals):
        ax.text(xi, c + 0.05, f"{c:.2f}%", ha="center", fontsize=10.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("年化 α (% )")
    ax.set_title("CAPM 漏掉规模/价值/动量 → 四因子几乎吃光 α", fontsize=13,
                 fontweight="bold")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(IMG, "carhart_alpha_comparison.png"))
    plt.close(fig)
    return dict(mean_capm_abs=np.mean(capm_abs)*100, mean_car_abs=np.mean(car_abs)*100,
                wml_capm=wml_capm*100, wml_car=wml_car*100)


if __name__ == "__main__":
    F = gen_factors()
    ret, beta = gen_25_portfolios(F)
    wml, beta_v = gen_wml(F)
    means = make_cover(F)
    capm_a, car_a = make_heatmap(F, ret)
    res = make_alpha_comparison(F, ret, wml)
    print("=" * 55)
    print("四因子月均溢价(%):", {n: round(m*100, 3) for n, m in zip(FACTORS, means)})
    print(f"25 组合 CAPM α 范围: [{capm_a.min()*100:.2f}%, {capm_a.max()*100:.2f}%]")
    print(f"25 组合 Carhart α 范围: [{car_a.min()*100:.2f}%, {car_a.max()*100:.2f}%]")
    print(f"截面平均 |CAPM α|      = {res['mean_capm_abs']:.3f}%")
    print(f"截面平均 |Carhart α|   = {res['mean_car_abs']:.3f}%")
    print(f"WML 组合 CAPM α       = {res['wml_capm']:+.3f}%")
    print(f"WML 组合 Carhart α    = {res['wml_car']:+.3f}%")
    print("=" * 55)
