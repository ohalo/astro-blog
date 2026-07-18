#!/usr/bin/env python3
"""
为文章「基本面定律 Grinold-Kahn：用信息比率拆解主动管理的收益上限」
(fundamental-law-active-management) 生成真实配图。所有图表均由文中方法真实计算生成。

机制（自洽合成，仅用于演示方法；真实落地见文末路径）：
  * 主动管理基本面定律 (Grinold 1989, Grinold-Kahn 2000):
        IR = IC * sqrt(BR)
    - IR  : 信息比率 = 年化主动收益 / 年化主动风险
    - IC  : 信息系数 = 预测值与实际收益的相关系数(预测技巧)
    - BR  : 广度 = 每年独立下注的次数
  * 扩展形式 (含约束): IR = IC * sqrt(BR) * TC
    - TC  : 转移系数 = 预测转化为实际持仓的程度 (0~1, 完全无约束=1)
  * 通过蒙特卡洛模拟: 一个经理每年做 BR 次独立下注, 每次预测与结果相关 IC,
    复现 IR 的真实分布, 验证 sqrt(BR) 缩放。
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
D = os.path.join(BASE, "fundamental-law-active-management")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "x": "#4C72B0", "y": "#DD8452", "base": "#55A868",
     "band": "#C44E52", "eq": "#2F4B7C", "bh": "#CCB974", "mk": "#8172B3",
     "null": "#999999", "out": "#C44E52", "acc": "#4C72B0"}


def simulate_manager(ic, br, years=20, rng=None):
    """模拟一个主动经理 years 年, 每年 BR 次独立下注。

    每次下注: 真实收益 z~N(0,1) (主动收益, 以标准差为单位),
    经理的预测信号 s 与 z 相关 IC: s = IC*z + sqrt(1-IC^2)*e.
    经理按信号幅度比例建仓 (w_t ∝ s_t), 主动盈亏 = w_t * z_t.
    IC = corr(s, z). 多年平均下: 年化主动 IR ≈ IC*sqrt(BR)/sqrt(1+IC^2) (BR 次独立下注).
    返回多年 annual active returns 序列。
    """
    if rng is None:
        rng = np.random.default_rng(2026)
    annual = np.zeros(years)
    for y in range(years):
        z = rng.normal(0, 1, br)
        e = rng.normal(0, 1, br)
        s = ic * z + np.sqrt(max(1 - ic ** 2, 0)) * e
        # 比例建仓: 仓位 w_t 正比于信号; 主动盈亏 = sum(w_t * z_t)
        bets = s * z
        # 标准化: 年主动收益 = 平均单笔盈亏 * sqrt(BR), 使年化主动波动≈1
        annual[y] = bets.mean() * np.sqrt(br)
    return annual


def main():
    # ===== 理论曲线: IR = IC * sqrt(BR) =====
    brs = np.linspace(1, 1000, 400)
    ics = [0.02, 0.05, 0.10, 0.15]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for ic in ics:
        ir = ic * np.sqrt(brs)
        ax.plot(brs, ir, lw=1.8, label=f"IC = {ic}")
    ax.set_xscale("log")
    ax.set_xlabel("广度 BR (每年独立下注次数, 对数轴)")
    ax.set_ylabel("信息比率 IR = IC × √BR")
    ax.set_title("主动管理基本面定律: IR 随 √BR 增长 (对数横轴下为直线)", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, color=C["grid"], which="both", ls=":")
    plt.tight_layout()
    fig.savefig(os.path.join(D, "ir_vs_breadth.png"), dpi=130)
    plt.close(fig)

    # ===== 热力图: IR 对 IC × BR 的响应 =====
    ic_grid = np.linspace(0.01, 0.20, 40)
    br_grid = np.linspace(5, 500, 40)
    IC, BR = np.meshgrid(ic_grid, br_grid)
    IR = IC * np.sqrt(BR)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    cm = ax.contourf(BR, IC, IR, levels=20, cmap="viridis")
    cs = ax.contour(BR, IC, IR, levels=[0.5, 1.0, 1.5, 2.0], colors="white", linewidths=1.2)
    ax.clabel(cs, inline=True, fontsize=9, fmt="IR=%.1f")
    cbar = fig.colorbar(cm, ax=ax)
    cbar.set_label("IR")
    # 标注真实经理点: IC=0.05, BR=100 -> IR=0.5 ; IC=0.10, BR=250 -> IR=1.58
    ax.plot(100, 0.05, "ro", ms=8, label="IC=0.05, BR=100 (IR≈0.5)")
    ax.plot(250, 0.10, "ys", ms=8, label="IC=0.10, BR=250 (IR≈1.58)")
    ax.set_xlabel("广度 BR")
    ax.set_ylabel("信息系数 IC")
    ax.set_title("IR = IC × √BR 响应曲面", fontsize=12)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(D, "ir_heatmap.png"), dpi=130)
    plt.close(fig)

    # ===== 蒙特卡洛验证 + 转移系数 TC 约束效应 =====
    rng = np.random.default_rng(2026)
    scenarios = [
        ("高技巧少机会", 0.15, 25),
        ("中等技巧中机会", 0.05, 250),
        ("低技巧广撒网", 0.02, 1000),
        ("高技巧广撒网", 0.10, 400),
    ]
    labels, ir_theory, ir_sim, ir_tc = [], [], [], []
    for name, ic, br in scenarios:
        ann = simulate_manager(ic, br, years=4000, rng=rng)
        ir_sim.append(ann.mean() / (ann.std(ddof=1) + 1e-12))
        ir_theory.append(ic * np.sqrt(br))
        # 约束: TC=0.6, 实际 IR 缩水
        ir_tc.append(ic * np.sqrt(br) * 0.6)
        labels.append(f"{name}\n(IC={ic}, BR={br})")

    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(x - w, ir_theory, w, label="理论 IR = IC×√BR", color=C["x"])
    ax.bar(x, ir_sim, w, label="蒙特卡洛实测 IR", color=C["base"])
    ax.bar(x + w, ir_tc, w, label="含约束 TC=0.6: IR×0.6", color=C["band"])
    for i, v in enumerate(ir_sim):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("信息比率 IR")
    ax.set_title("四种经理画像: 理论 vs 模拟, 以及约束(TC)如何砍掉 IR", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, color=C["grid"], axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(D, "ir_manager_scenarios.png"), dpi=130)
    plt.close(fig)

    print("==== DIAGNOSTICS (fundamental-law) ====")
    for (name, ic, br), sim in zip(scenarios, ir_sim):
        print(f"{name:18s} IC={ic} BR={br:5d}  theory IR={ic*np.sqrt(br):.3f}  sim IR={sim:.3f}  TC=0.6 -> {ic*np.sqrt(br)*0.6:.3f}")
    # 关键演示: BR 在根号下 -> 翻倍 IC 等价于翻 4 倍 BR
    print(f"IC 0.05 BR 100 IR={0.05*np.sqrt(100):.3f}  |  IC 0.05 BR 400 IR={0.05*np.sqrt(400):.3f}  (BR×4=IC×2 的效果)")
    print("images saved to", D)
    print(os.listdir(D))


if __name__ == "__main__":
    main()
