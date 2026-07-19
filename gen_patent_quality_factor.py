#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""专利质量因子 配图生成 (5 张真实图表, 自洽合成)

机制(自洽合成, 仅用于演示方法):
  * 合成 150 家公司面板, 每家公司有一个"专利质量" q_i = _forward citations per patent_
        (Kogan et al. 2017 用前向引用次数度量创新质量), 用 lognormal 抽样构造
  * 隐藏"创新溢价": 专利质量越高的公司, 持续性获得更高月收益 (kappa * q_i)
  * 隐藏"创新浪潮事件": 在 T≈60-72 注入一次对高专利质量公司的额外奖励冲击
        (高 q 公司吃到超额回报, 低 q 公司几乎无感)
  * 公司月度收益: r = MKT + beta*MKT + kappa*q_i + innov_load*q_i*innov_shock + eps
  * 专利质量因子组合: 每月按 q 分 5 组, 多高质(组5)/空低质(组1), 等权
  * 横截面回归: ret ~ q (+ 控制 MKT, size), t 统计量验证"专利质量溢价"
  * 机制分解: 高质组在创新浪潮期超额最明显; 五分位收益随质量单调上行
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "patent-quality-factor"
OUT = os.path.join("/Users/halo/workspace/astro-blog/public/images", SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"net": "#4C72B0", "red": "#C44E52", "green": "#55A868", "orange": "#DD8452",
     "purple": "#8172B3", "grid": "#DDDDDD", "line": "#999999"}

rng = np.random.default_rng(20260720)
N, T = 150, 240                      # 150 公司, 240 月(20 年)

# ---------- 专利质量 q = 前向引用/专利 ----------
q = rng.lognormal(mean=0.0, sigma=0.8, size=N)
q = np.clip(q, 0.1, None)            # 前向引用次数/专利, 右偏
size = rng.lognormal(0, 0.6, N)      # 公司规模(控制变量)

# ---------- 市场 & 创新浪潮 ----------
MKT = rng.normal(0.008, 0.04, T-1)
innov_shock = np.zeros(T-1)          # 创新浪潮(高质公司额外奖励)
for t in range(60, 73):
    innov_shock[t] = rng.normal(0.03, 0.01)

# ---------- 公司收益 ----------
beta = rng.normal(1.0, 0.3, N)
kappa = 0.0045                       # 专利质量持续溢价(系数)
innov_load = 1.6                     # 创新浪潮放大敏感度
eps = rng.normal(0, 0.02, (N, T-1))  # 特质噪声
R = (MKT[None, :]
     + beta[:, None] * MKT[None, :]
     + kappa * q[:, None]                       # 持续创新溢价
     + innov_load * q[:, None] * innov_shock[None, :]   # 创新浪潮: 高质吃超额
     + eps)

# ---------- 分组组合 ----------
order = np.argsort(q)
n = N // 5
groups = [order[i*n:(i+1)*n] for i in range(5)]
grp_ret = np.array([R[g, :].mean(0) for g in groups])   # (5, T-1)
low = grp_ret[0]; high = grp_ret[4]
pq_factor = high - low              # 多高质(组5)/空低质(组1)
nav_factor = np.cumprod(1 + pq_factor)
nav_mkt = np.cumprod(1 + MKT)

def ann(r): return (np.prod(1 + r))**(12/len(r)) - 1
def sharpe(r): return r.mean()/r.std()*np.sqrt(12)
dec_ann = np.array([ann(grp_ret[k]) for k in range(5)])

# ---------- 横截面回归 ----------
def xs_reg(y, Xcols):
    X = np.column_stack([np.ones(N)] + Xcols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = N - X.shape[1]
    se = np.sqrt((resid @ resid)/dof) * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
    return coef, se
coef, se = xs_reg(R.mean(1), [q, size])
t_q = coef[1] / se[1]

# ---------- 创新浪潮期机制分解 ----------
pre = range(0, 60); wave = range(60, 73); post = range(73, T-1)
def ann_in(g, idx): return (np.prod(1 + grp_ret[g][list(idx)]))**(12/len(list(idx))) - 1
low_wave = ann_in(0, wave); high_wave = ann_in(4, wave)
# 更稳健的窗口内月度超额(避免短窗口年化爆炸)
wave_spread_m = grp_ret[4][list(wave)].mean() - grp_ret[0][list(wave)].mean()
normal_spread_m = grp_ret[4][list(range(0,60))+list(range(73,T-1))].mean() - grp_ret[0][list(range(0,60))+list(range(73,T-1))].mean()

summary = {
    "pq_factor_ann": ann(pq_factor), "pq_factor_sharpe": sharpe(pq_factor),
    "mkt_ann": ann(MKT), "t_q": float(t_q), "coef_q": float(coef[1]),
    "dec_ann": [float(x) for x in dec_ann],
    "low_wave_ann": low_wave, "high_wave_ann": high_wave,
    "wave_spread_m": float(wave_spread_m), "normal_spread_m": float(normal_spread_m),
    "q_median": float(np.median(q)), "q_p90": float(np.sort(q)[int(0.9*N)]),
}
print(json.dumps(summary, indent=2))

# ================= 图 1: 专利质量分布 =================
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.hist(q, bins=30, color=C["net"], alpha=0.8, edgecolor="white")
ax.axvline(np.median(q), color=C["red"], ls="--", lw=1.3, label=f"中位数 q={np.median(q):.2f}")
ax.set_title("专利质量分布(合成 150 家, q=前向引用/专利, lognormal)", fontsize=12)
ax.set_xlabel("专利质量 q (前向引用次数 / 专利)"); ax.set_ylabel("公司数")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/q_dist.png"); plt.close(fig)

# ================= 图 2: 因子净值 vs 市场 =================
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(nav_factor, color=C["green"], lw=2, label=f"专利质量因子(多高/空低) {ann(pq_factor)*100:.1f}%/yr")
ax.plot(nav_mkt, color=C["line"], lw=1.5, ls="--", label=f"市场 {ann(MKT)*100:.1f}%/yr")
ax.axhline(1.0, color=C["grid"], lw=1)
ax.set_title("专利质量因子净值: 把创新质量变成可交易 alpha", fontsize=12)
ax.set_xlabel("月份"); ax.set_ylabel("净值(起点=1)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/pq_factor_nav.png"); plt.close(fig)

# ================= 图 3: 创新浪潮期 高质 vs 低质 累计 =================
pre_nav_low = np.cumprod(1 + grp_ret[0])
pre_nav_high = np.cumprod(1 + grp_ret[4])
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(pre_nav_high, color=C["green"], lw=2, label="高专利质量组(组5)")
ax.plot(pre_nav_low, color=C["red"], lw=2, label="低专利质量组(组1)")
ax.axvspan(60, 73, color=C["orange"], alpha=0.18, label="创新浪潮窗口")
ax.set_title("20 年累计净值: 高质组持续跑赢, 浪潮期加速拉开", fontsize=12)
ax.set_xlabel("月份"); ax.set_ylabel("净值(起点=1)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/quality_cumret.png"); plt.close(fig)

# ================= 图 4: 横截面回归 =================
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.scatter(q, R.mean(1)*100, s=22, alpha=0.55, color=C["net"])
xs = np.linspace(q.min(), q.max(), 50)
ax.plot(xs, (coef[0] + coef[1]*xs)*100, color=C["red"], lw=2,
        label=f"斜率 t={t_q:.1f}")
ax.set_title("横截面: 专利质量越高, 月均收益越高(t 显著为正)", fontsize=12)
ax.set_xlabel("专利质量 q"); ax.set_ylabel("月均收益 (%)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/xs_regression.png"); plt.close(fig)

# ================= 图 5: 五分位年化收益 =================
fig, ax = plt.subplots(figsize=(9, 4.2))
colors = [C["red"] if i==0 else (C["green"] if i==4 else C["net"]) for i in range(5)]
qs = np.sort(q)
ax.bar([f"Q{i+1}\n(q={qs[i*n]:.2f}-{qs[min((i+1)*n,N)-1]:.2f})" for i in range(5)],
       dec_ann*100, color=colors)
for i, v in enumerate(dec_ann*100):
    ax.text(i, v + (0.3 if v>=0 else -1.0), f"{v:.1f}%", ha="center", fontsize=9)
ax.axhline(0, color=C["line"], lw=1)
ax.set_title("五分位组合年化收益: 专利质量单调(低→高 收益升)", fontsize=12)
ax.set_ylabel("年化收益 (%)")
ax.grid(alpha=0.3, color=C["grid"], axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/decile_returns.png"); plt.close(fig)

print("IMAGES_WRITTEN:", sorted(os.listdir(OUT)))
