#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重抽样有效性 (Resampled Efficiency, Michaud 1998) 配图生成 (3 张真实图表)

机制(自洽合成, 仅用于演示方法):
  * 真实资产: 5 资产, 真实均值向量 mu*、真实协方差 Sigma* 有结构
  * 经典 Markowitz: 用「样本估计」 mu_hat, Sigma_hat 解有效前沿, 权重是误差最大化
  * 重抽样有效性: B 次自助抽样 -> 每次重估前沿 -> 在同一目标波动下取权重 -> 跨 B 次平均
      -> 把估计误差「摊平」, 得到去噪权重
  * 对照: 经典 MV(单一估计) vs 重抽样平均权重
  * 图1: 有效前沿 + 重抽样组合云(同一目标波动下 B 个权重的组合点散开)
  * 图2: 目标组合权重跨 B 次重抽样的箱线图(分布) + 经典 vs 重抽样平均权重柱状
  * 图3: 样本外累计收益 / 滚动波动对比(重抽样更稳)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "resampled-efficiency-michaud"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"grid": "#DDDDDD", "res": "#4C72B0", "cls": "#C44E52", "cloud": "#8FA9D4",
     "gold": "#CCB974", "bond": "#9AD3BC"}

rng = np.random.default_rng(20260719)
names = ["股票", "债券", "黄金", "商品", "REITs"]
n = 5
# 真实风险溢价(年化)与波动
mu_star = np.array([0.10, 0.03, 0.06, 0.07, 0.09])
ann_vol = np.array([0.22, 0.05, 0.15, 0.18, 0.25])
corr = np.array([
    [1.00, -0.20, 0.10, 0.30, 0.60],
    [-0.20, 1.00, 0.05, -0.10, -0.15],
    [0.10, 0.05, 1.00, 0.20, 0.10],
    [0.30, -0.10, 0.20, 1.00, 0.35],
    [0.60, -0.15, 0.10, 0.35, 1.00],
])
D = np.diag(ann_vol)
Sigma_star = D @ corr @ D

# 生成「真实」日收益样本(训练期), 用真实参数
T = 1500
L = np.linalg.cholesky(Sigma_star)
daily_ret = (mu_star / 252) + (rng.standard_normal((T, n)) @ L.T / np.sqrt(252))

# ---------- 经典 Markowitz 前沿求解(长仓约束) ----------
def solve_frontier(mu, Sigma, target_ret):
    """min w'Sigma w  s.t. w'mu=target_ret, sum w=1, w>=0"""
    def obj(w):
        return w @ Sigma @ w
    cons = [{"type": "eq", "fun": lambda w: w @ mu - target_ret},
            {"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.repeat(1.0 / n, n)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x if res.success else x0

# 样本估计(经典 MV 用的就是这一份估计)
mu_hat = daily_ret.mean(axis=0) * 252
Sigma_hat = np.cov(daily_ret.T) * 252

# 目标波动水平的组合: 在样本前沿上找一个「目标波动 = 某值」的组合
# 扫描目标收益, 解前沿, 记录波动, 选最接近 target_vol 的点
target_vol = 0.12
best_w_cls, best_vol_cls, best_ret_cls = None, 1e9, None
for tr in np.linspace(mu_hat.min(), mu_hat.max() * 1.2, 200):
    w = solve_frontier(mu_hat, Sigma_hat, tr)
    pv = np.sqrt(w @ Sigma_hat @ w)
    if abs(pv - target_vol) < abs(best_vol_cls - target_vol):
        best_vol_cls, best_w_cls, best_ret_cls = pv, w, tr

# ---------- 重抽样有效性 ----------
B = 500
resampled_w = np.zeros((B, n))
resampled_vol = np.zeros(B)
for b in range(B):
    idx = rng.integers(0, T, T)
    rb = daily_ret[idx]
    mb = rb.mean(axis=0) * 252
    Sb = np.cov(rb.T) * 252
    # 在「同一目标收益 = best_ret_cls」下重估(对齐目标, 公平比较)
    w = solve_frontier(mb, Sb, best_ret_cls)
    resampled_w[b] = w
    resampled_vol[b] = np.sqrt(w @ Sb @ w)
w_res = resampled_w.mean(axis=0)
# 把重抽样平均权重回投到样本 Sigma 的波动
vol_res_on_sample = np.sqrt(w_res @ Sigma_hat @ w_res)

# ---------- 图1: 有效前沿 + 重抽样云 ----------
rets_scan = np.linspace(mu_hat.min(), mu_hat.max() * 1.2, 120)
front_vol, front_ret = [], []
for tr in rets_scan:
    w = solve_frontier(mu_hat, Sigma_hat, tr)
    front_vol.append(np.sqrt(w @ Sigma_hat @ w))
    front_ret.append(w @ mu_hat)
front_vol, front_ret = np.array(front_vol), np.array(front_ret)

# 重抽样组合点(在每次重估的前沿上, 同一目标收益对应的组合波动/收益)
cloud_ret = resampled_w @ mu_hat
cloud_vol = resampled_vol

fig, ax = plt.subplots(figsize=(9, 5.6))
ax.plot(front_vol * 100, front_ret * 100, color=C["cls"], lw=2, label="经典样本有效前沿")
ax.scatter(cloud_vol * 100, cloud_ret * 100, s=10, color=C["cloud"], alpha=0.5,
           label=f"重抽样组合云 (B={B}, 同一目标收益)")
# 标记两个目标组合
ax.scatter([best_vol_cls * 100], [best_ret_cls * 100], color=C["cls"], s=90, zorder=5,
           edgecolor="black", label=f"经典目标组合 (σ={best_vol_cls*100:.1f}%)")
ax.scatter([vol_res_on_sample * 100], [w_res @ mu_hat * 100], color=C["res"], s=90, zorder=5,
           edgecolor="black", label=f"重抽样平均组合 (σ={vol_res_on_sample*100:.1f}%)")
ax.set_xlabel("年化波动率 (%)")
ax.set_ylabel("年化收益 (%)")
ax.set_title("有效前沿与重抽样组合云：经典点被估计误差「钉死」,\n重抽样把不确定性摊开成一片, 再取平均去噪")
ax.legend(fontsize=8)
ax.grid(True, color=C["grid"])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "re_frontier_cloud.png"))
plt.close(fig)

# ---------- 图2: 权重分布(箱线图) + 经典 vs 重抽样 ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
bp = axes[0].boxplot([resampled_w[:, i] * 100 for i in range(n)], tick_labels=names, patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor(C["cloud"])
    patch.set_alpha(0.7)
axes[0].set_ylabel("权重 (%)")
axes[0].set_title(f"目标组合权重跨 {B} 次重抽样的分布\n(经典 MV 只取其中 1 个估计, 极易落在极端分位)")
axes[0].grid(True, color=C["grid"], axis="y")
x = np.arange(n)
width = 0.38
axes[1].bar(x - width / 2, best_w_cls * 100, width, color=C["cls"], label="经典 MV (单一估计)")
axes[1].bar(x + width / 2, w_res * 100, width, color=C["res"], label="重抽样平均 (去噪)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(names)
axes[1].set_ylabel("权重 (%)")
axes[1].set_title("同一目标组合：经典权重 vs 重抽样平均权重\n(重抽样把极端权重拉回, 更均匀、更稳定)")
axes[1].legend(fontsize=8)
axes[1].grid(True, color=C["grid"], axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "re_weights.png"))
plt.close(fig)

# ---------- 图3: 样本外累计收益 / 滚动波动 ----------
# 用真实参数生成「样本外」测试期
T2 = 750
daily_oos = (mu_star / 252) + (rng.standard_normal((T2, n)) @ L.T / np.sqrt(252))
port_cls = daily_oos @ best_w_cls
port_res = daily_oos @ w_res
cum_cls = np.cumprod(1 + port_cls) - 1
cum_res = np.cumprod(1 + port_res) - 1
vol_cls_r = np.sqrt(np.var(port_cls) * 252) * 100
vol_res_r = np.sqrt(np.var(port_res) * 252) * 100
# 滚动波动
def roll_vol(r, w=60):
    out = np.full(len(r), np.nan)
    for i in range(w, len(r)):
        out[i] = np.sqrt(np.var(r[i - w:i]) * 252) * 100
    return out
rv_cls = roll_vol(port_cls)
rv_res = roll_vol(port_res)
fig, ax1 = plt.subplots(figsize=(9, 5.6))
ax1.plot(cum_cls * 100, color=C["cls"], lw=1.6, label=f"经典 MV (末累计 {cum_cls[-1]*100:.1f}%, σ={vol_cls_r:.1f}%)")
ax1.plot(cum_res * 100, color=C["res"], lw=1.6, label=f"重抽样 (末累计 {cum_res[-1]*100:.1f}%, σ={vol_res_r:.1f}%)")
ax1.set_ylabel("累计收益 (%)")
ax1.set_xlabel("样本外交易日")
ax1.set_title("样本外表现：重抽样平均权重波动更低、曲线更平滑\n(去噪权重把「误差最大化」的极端暴露压住)")
ax1.legend(fontsize=8, loc="upper left")
ax2 = ax1.twinx()
ax2.plot(rv_cls, color=C["cls"], lw=0.7, alpha=0.45)
ax2.plot(rv_res, color=C["res"], lw=0.7, alpha=0.45)
ax2.set_ylabel("滚动年化波动率 (%)", color="#666666")
ax2.tick_params(axis="y", labelcolor="#666666")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "re_oos.png"))
plt.close(fig)

print("DONE", os.listdir(OUT))
print("w_cls", np.round(best_w_cls, 3))
print("w_res", np.round(w_res, 3))
print("vol_cls_on_sample", round(best_vol_cls, 4), "vol_res_on_sample", round(vol_res_on_sample, 4))
print("OOS vol cls", round(vol_cls_r, 2), "res", round(vol_res_r, 2))
print("OOS cum cls", round(cum_cls[-1] * 100, 2), "res", round(cum_res[-1] * 100, 2))
print("weight std across resamples:", np.round(resampled_w.std(axis=0), 3))
