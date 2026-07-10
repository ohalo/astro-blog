# -*- coding: utf-8 -*-
"""为《气象数据对大宗商品预测》生成 3 张配图（合成数据,仅用于示意）。
输出目录: public/images/weather-data-commodity-prediction/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "public/images/weather-data-commodity-prediction"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260710)

C_TEMP = "#E45756"
C_HDD = "#4C78A8"
C_CDD = "#F58518"
C_LINE = "#333333"

# ---------------------------------------------------------------
# 图1: 气温与 HDD/CDD(采暖/制冷度日)
# ---------------------------------------------------------------
days = np.arange(365)
# 合成季节性气温(北半球)
base = 15 + 12 * np.sin(2 * np.pi * (days - 80) / 365)
temp = base + rng.normal(0, 2.2, 365)
T_ref = 18.0  # 基准温度(°C)
HDD = np.where(temp < T_ref, T_ref - temp, 0)
CDD = np.where(temp > T_ref, temp - T_ref, 0)
# 年化累计
cum_hdd = np.cumsum(HDD)
cum_cdd = np.cumsum(CDD)

fig, axes = plt.subplots(2, 1, figsize=(13, 8), facecolor="white", sharex=True)
ax = axes[0]
ax.plot(days, temp, color=C_TEMP, lw=1.8, label="日平均气温")
ax.axhline(T_ref, color="gray", ls="--", lw=1.4, label=f"基准温度 {T_ref:.0f}°C")
ax.fill_between(days, temp, T_ref, where=(temp < T_ref), color=C_HDD, alpha=0.35, label="低于基准→采暖(HDD)")
ax.fill_between(days, temp, T_ref, where=(temp > T_ref), color=C_CDD, alpha=0.35, label="高于基准→制冷(CDD)")
ax.set_ylabel("气温 (°C)", fontsize=12, fontweight="bold")
ax.set_title("日气温与采暖/制冷度日(HDD/CDD)的季节性", fontsize=14, fontweight="bold")
ax.legend(fontsize=9, loc="upper right", framealpha=0.92)
ax.grid(True, alpha=0.3, ls="--")

ax = axes[1]
ax.plot(days, cum_hdd, color=C_HDD, lw=2.4, label="累计 HDD(冬季供暖需求)")
ax.plot(days, cum_cdd, color=C_CDD, lw=2.4, label="累计 CDD(夏季制冷需求)")
ax.set_xlabel("年内第几天", fontsize=12, fontweight="bold")
ax.set_ylabel("累计度日", fontsize=12, fontweight="bold")
ax.legend(fontsize=10, loc="upper left", framealpha=0.92)
ax.grid(True, alpha=0.3, ls="--")
fig.tight_layout()
fig.savefig(f"{OUT}/hdd_cdd_temperature.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------
# 图2: ENSO(厄尔尼诺)指数 vs 农产品价格
# ---------------------------------------------------------------
T = 240
months = np.arange(T)
# ENSO 3.4 指数:缓慢振荡 + 偶发强事件
enso = 1.2 * np.sin(2 * np.pi * months / 48) + 0.8 * np.sin(2 * np.pi * months / 17)
enso += rng.normal(0, 0.4, T)
# 厄尔尼诺(ENSO>0)通常压制南美大豆供应预期 → 负相关
crop_price = 100 * np.exp(0.012 * months +
                          -0.16 * enso +
                          0.05 * np.roll(enso, 3) +
                          np.cumsum(rng.normal(0, 0.012, T)))
crop_ret = np.diff(np.log(crop_price), prepend=np.log(crop_price)[0])

fig, ax1 = plt.subplots(figsize=(13, 6), facecolor="white")
ax1.plot(months, enso, color="#72B7B2", lw=2.2, label="ENSO 3.4 海温距平 (°C)")
ax1.axhline(0, color="gray", lw=1, alpha=0.6)
ax1.axhspan(0, 2.5, color="#E45756", alpha=0.06)
ax1.set_ylabel("ENSO 3.4 距平 (°C)", fontsize=12, fontweight="bold", color="#2a7d77")
ax1.set_xlabel("月份", fontsize=12, fontweight="bold")
ax1.tick_params(axis="y", labelcolor="#2a7d77")
ax2 = ax1.twinx()
ax2.plot(months, crop_price, color="#B279A2", lw=2.2, label="大豆期货价格(示意)")
ax2.set_ylabel("农产品价格", fontsize=12, fontweight="bold", color="#7d3f6e")
ax2.tick_params(axis="y", labelcolor="#7d3f6e")
ax1.set_title("厄尔尼诺(ENSO)海温异常 vs 农产品价格:暖相位常压制供应预期", fontsize=14, fontweight="bold")
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=10, framealpha=0.92)
ax1.grid(True, alpha=0.3, ls="--")
fig.tight_layout()
fig.savefig(f"{OUT}/enso_commodity.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------
# 图3: 气象特征对商品收益预测的重要性
# ---------------------------------------------------------------
features = ["HDD(供暖)", "CDD(制冷)", "降水异常", "ENSO相位", "干旱指数", "积温", "风速", "霜冻天数"]
imp = np.array([0.31, 0.27, 0.14, 0.12, 0.07, 0.05, 0.02, 0.02]) + rng.normal(0, 0.01, 8)
imp = np.clip(imp, 0.005, None)
imp = imp / imp.sum()
order = np.argsort(imp)
features = [features[i] for i in order]
imp = imp[order]

fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
colors = [C_HDD if f.startswith("HDD") else (C_CDD if f.startswith("CDD") else "#54A24B") for f in features]
bars = ax.barh(features, imp, color=colors)
for b, v in zip(bars, imp):
    ax.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.1%}", va="center", fontsize=10, fontweight="bold")
ax.set_xlabel("特征重要性(归一化)", fontsize=12, fontweight="bold")
ax.set_title("气象特征对大宗商品收益预测的相对重要性(示意)", fontsize=14, fontweight="bold")
ax.set_xlim(0, max(imp) * 1.2)
ax.grid(True, alpha=0.3, ls="--", axis="x")
fig.tight_layout()
fig.savefig(f"{OUT}/weather_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("saved:", os.listdir(OUT))
