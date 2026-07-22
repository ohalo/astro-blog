#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「核心卫星策略配置框架：用 80% 被动核心 + 20% 主动卫星，把稳健和锐度都装上」
生成真实配图与统计数字。

核心逻辑:
  - 纯被动(100% 宽基指数)稳健但「全程裸多一个 beta」，拿不到任何主动 alpha
  - 纯主动(全仓卫星策略)锐度够但波动大、回撤深、容易在风格逆风里裸奔
  - 核心-卫星(Core-Satellite): 大部分仓位(核心)买宽基指数获得市场 beta + 低成本，
    小部分仓位(卫星)做主动策略/行业/因子/个股，用来博取超额收益
  - 用「成分风险贡献」拆解: 核心 sleeve 提供多少收益、又贡献多少总波动
  - 对照: 纯被动 vs 核心-卫星(70/18/12) vs 纯卫星，看期望收益/波动/回撤/Sharpe

说明: 期望指标由合成宇宙的 mu/sigma 解析计算(不依赖随机路径)；累计净值曲线为
      固定随机种子的「示意路径」，用于展示形态，不代表任何真实历史。
图片:
  cover.png                  —— 核心-卫星结构图(核心 70% 箱体 + 卫星 30% 三块)
  cs_equity_curve.png        —— 三种配置累计净值曲线(示意路径)
  cs_risk_decomp.png         —— 成分风险贡献(权重 vs 对总风险的贡献)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "core-satellite-strategy")
os.makedirs(D, exist_ok=True)

# ================= 资产设定(年化 mu / sigma, 月度相关) =================
# 核心 = 宽基指数; 卫星1 = 动量因子 tilt; 卫星2 = 价值因子 tilt
ann_mu = np.array([0.070, 0.120, 0.100])
ann_sd = np.array([0.150, 0.190, 0.170])
C = np.array([
    [1.00, 0.62, 0.55],
    [0.62, 1.00, 0.50],
    [0.55, 0.50, 1.00],
])
Cov = (ann_sd[:, None] * ann_sd[None, :]) * C  # 年化协方差
L = np.linalg.cholesky(Cov)

W_passive = np.array([1.0, 0.0, 0.0])
W_cs = np.array([0.70, 0.18, 0.12])
W_sat = np.array([0.0, 0.50, 0.50])

# ================= 期望指标(解析, mu/sigma) =================
def exp_metrics(w):
    mu_p = w @ ann_mu
    sd_p = np.sqrt(w @ Cov @ w)
    sharpe = (mu_p - 0.02) / sd_p
    return dict(ann_ret=mu_p, ann_vol=sd_p, sharpe=sharpe)
mv = exp_metrics(W_passive)
mcs = exp_metrics(W_cs)
msat = exp_metrics(W_sat)
# 卫星 sleeve(30% 仓位内的主动部分)
w_sat_sleeve = np.array([0.0, 18/30, 12/30])
msat_sleeve = exp_metrics(w_sat_sleeve)

print("=== 期望指标(解析) ===")
for name, m in [("纯被动(100%核心)", mv), ("核心-卫星(70/18/12)", mcs),
                ("纯卫星(50/50)", msat), ("卫星 sleeve(占组合30%)", msat_sleeve)]:
    print(f"{name:22s} 期望年化 {m['ann_ret']*100:5.2f}%  波动 {m['ann_vol']*100:5.2f}%  Sharpe {m['sharpe']:.2f}")

# ================= 成分风险贡献(核心-卫星, 年化) =================
pv = W_cs @ Cov @ W_cs
mrc = Cov @ W_cs
rc = W_cs * mrc
rc_pct = rc / pv
print("\n=== 核心-卫星(70/18/12) 成分风险贡献 ===")
print("权重 %:", np.round(W_cs*100,1))
print("成分风险贡献 %:", np.round(rc_pct*100,1))
print("年化成分波动 %:", np.round(np.sqrt(rc)*100,2))

# ================= 示意路径(固定种子采样) =================
# 选一个「终值贴近期望年化」的种子做示意路径(期望 20 年终值: 被动≈3.87 / 核心-卫星≈5.06 / 纯卫星≈8.06)
M = 240
exp_term = np.array([(1+ann_mu@W_passive)**20, (1+ann_mu@W_cs)**20, (1+ann_mu@W_sat)**20])
best_seed, best_err = None, 1e18
for s in range(5000):
    rng = np.random.default_rng(s)
    Z = rng.standard_normal((M,3))
    monthly = (ann_mu/12) - 0.5*(ann_sd**2)/12 + (Z @ L.T)/np.sqrt(12)
    terms = np.array([(1+monthly@W_passive).cumprod()[-1],
                      (1+monthly@W_cs).cumprod()[-1],
                      (1+monthly@W_sat).cumprod()[-1]])
    err = ((terms - exp_term)**2).sum()
    if err < best_err:
        best_err, best_seed = err, s
rng = np.random.default_rng(best_seed)
Z = rng.standard_normal((M,3))
monthly = (ann_mu/12) - 0.5*(ann_sd**2)/12 + (Z @ L.T)/np.sqrt(12)
ret_passive = monthly @ W_passive
ret_cs = monthly @ W_cs
ret_sat = monthly @ W_sat
nv_p = (1+ret_passive).cumprod()
nv_cs = (1+ret_cs).cumprod()
nv_s = (1+ret_sat).cumprod()
print(f"\n示意路径种子: {best_seed} (期望终值 被动{exp_term[0]:.2f}x / 核心-卫星{exp_term[1]:.2f}x / 纯卫星{exp_term[2]:.2f}x)")
print(f"  纯被动终值 {nv_p[-1]:.2f}x | 核心-卫星 {nv_cs[-1]:.2f}x | 纯卫星 {nv_s[-1]:.2f}x")

# ================= 图 1: 结构图 =================
fig, ax = plt.subplots(figsize=(8.4, 4.6))
ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis("off")
ax.add_patch(plt.Rectangle((0.3,0.8),9.4,4.6,fill=True,color="#E8EEF7",ec="#2E5AAC",lw=2.5))
ax.text(5.0,5.05,"核心 Core (70%)  —— 宽基指数 ETF，获取市场 beta + 低成本",
        ha="center",fontsize=12,color="#1B3A6B",weight="bold")
sat_cols=["#F4A261","#E76F51"]
sat_lbl=["动量因子 tilt\n(18%)","价值因子 tilt\n(12%)"]
for i,(x,c,l) in enumerate(zip([1.1,4.0],sat_cols, sat_lbl)):
    ax.add_patch(plt.Rectangle((x,1.2),2.5,3.4,fill=True,color=c,ec="white",lw=2,alpha=0.92))
    ax.text(x+1.25,2.9,l,ha="center",va="center",fontsize=10,color="white",weight="bold")
ax.text(7.6,2.9,"卫星合计\n30%\n主动 alpha\n来源",ha="center",va="center",fontsize=9.5,
        color="#444",weight="bold")
ax.text(5.0,0.45,"核心吃「稳健」，卫星吃「锐度」—— 一笔钱同时满足两个互相冲突的目标",
        ha="center",fontsize=10,color="#555")
fig.tight_layout(); fig.savefig(os.path.join(D,"cover.png"),dpi=140); plt.close(fig)

# ================= 图 2: 累计净值(示意) =================
years=np.arange(M)/12
fig, ax = plt.subplots(figsize=(8.6,4.6))
ax.plot(years,nv_cs,color="#2E5AAC",lw=2.4,label=f"核心-卫星 (终值 {nv_cs[-1]:.2f}x)")
ax.plot(years,nv_p,color="#888",lw=1.8,ls="--",label=f"纯被动 (终值 {nv_p[-1]:.2f}x)")
ax.plot(years,nv_s,color="#E76F51",lw=1.8,ls=":",label=f"纯卫星 (终值 {nv_s[-1]:.2f}x)")
ax.set_xlabel("年"); ax.set_ylabel("累计净值 (起点=1)")
ax.set_title("三种配置累计净值(示意路径)：核心-卫星在接近被动的波动下略跑赢被动",fontsize=11)
ax.legend(fontsize=9,loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D,"cs_equity_curve.png"),dpi=140); plt.close(fig)

# ================= 图 3: 成分风险贡献 =================
labels=["核心(70%)","动量(18%)","价值(12%)"]
x=np.arange(3); w_pct=W_cs*100; width=0.38
fig, ax = plt.subplots(figsize=(8.2,4.4))
b1=ax.bar(x-width/2,w_pct,width,label="仓位权重 %",color="#9DB8E0")
b2=ax.bar(x+width/2,rc_pct*100,width,label="对总风险贡献 %",color="#E76F51")
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+1.5,f"{b.get_height():.1f}",ha="center",fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("百分比 (%)")
ax.set_title(f"成分风险贡献：核心占 {w_pct[0]:.0f}% 权重贡献 {rc_pct[0]*100:.0f}% 风险，卫星只占 {w_pct[1:].sum():.0f}% 却贡献 {(rc_pct[1:].sum())*100:.0f}%",fontsize=10.5)
ax.legend(fontsize=9); ax.grid(alpha=0.3,axis="y")
fig.tight_layout(); fig.savefig(os.path.join(D,"cs_risk_decomp.png"),dpi=140); plt.close(fig)

print("\n图片已保存至:",D)
