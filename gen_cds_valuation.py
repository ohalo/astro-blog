#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""信用违约互换 CDS 估值配图生成 (4 张真实图表, 信用曲线 bootstrap 全自洽计算)

机制 (约化型 / 危险率模型):
  * 生存概率 S(t)=exp(−∫λ du); 违约强度 λ(t) 分段常数
  * 保护 leg PV = (1−R)·Σ_j DF_j·(S_{j-1}−S_j)
  * 保费 leg PV(RPV01) = Σ_j DF_j·(S_{j-1}+S_j)/2·Δt_j
  * 平价 CDS 利差 s = 保护leg / RPV01
  * bootstrap: 已知各期限平价利差 s_i, 逐段反解 λ_i —— 第 i 段仅用 1..mat[i] 期限
    (前段 λ 已定, 后段 λ 置 0, 因为对应 CDS 在 mat[i] 已到期)
  * 参数: 期限 1..10y, 市场 CDS 利差曲线由浅到深, R=40%, r=3%, 年付
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "cds-valuation-credit-curve"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"grid": "#DDDDDD", "path": "#4C72B0", "mean": "#C44E52", "fit": "#55A868",
     "true": "#999999", "curve": "#8172B3", "scatter": "#DD8452", "black": "#333333"}


def cds_legs_upto(h, maturities, df, R, upto):
    """只算 1..upto 段 (含) 的 legs, 用于单期限 CDS 平价校验。"""
    surv = 1.0
    prev = 0.0
    rpv = 0.0
    prot = 0.0
    for i in range(upto):
        m = maturities[i]
        dt = m - prev
        s_end = surv * np.exp(-h[i] * dt)
        prot += (1 - R) * df[i] * (surv - s_end)
        rpv += df[i] * (surv + s_end) / 2.0 * dt
        surv = s_end
        prev = m
    return rpv, prot


def par_spread_upto(h, maturities, df, R, upto):
    rpv, prot = cds_legs_upto(h, maturities, df, R, upto)
    return prot / rpv if rpv > 0 else np.inf


def bootstrap_credit_curve(maturities, spreads_bps, r, R):
    df = np.exp(-r * maturities)
    h = np.zeros_like(maturities, dtype=float)
    for i in range(len(maturities)):
        # 第 i 段: 仅用 1..mat[i] 期限 CDS 平价条件反解 h[i]
        target = spreads_bps[i] / 1e4
        lo, hi = 1e-7, 5.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            h_try = h.copy()
            h_try[i] = mid
            s = par_spread_upto(h_try, maturities, df, R, i + 1)
            if s > target:
                hi = mid
            else:
                lo = mid
        h[i] = (lo + hi) / 2.0
    # 累计违约概率
    surv = 1.0
    prev = 0.0
    cum_def = np.zeros_like(maturities, dtype=float)
    for i, m in enumerate(maturities):
        dt = m - prev
        seg = surv * (1 - np.exp(-h[i] * dt))
        cum_def[i] = cum_def[i - 1] + seg if i > 0 else seg
        surv *= np.exp(-h[i] * dt)
        prev = m
    return maturities, h, df, cum_def


# ---------- 参数 ----------
r, R = 0.03, 0.40
maturities = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
spreads = np.array([40, 55, 70, 85, 100, 115, 130, 150, 165, 180], dtype=float)

times, h, df, cum_def = bootstrap_credit_curve(maturities, spreads, r, R)

# 自洽校验: 用 bootstrap 出的 h, 逐期限重算平价利差, 应≈市场报价
chk = [par_spread_upto(h.copy(), maturities, df, R, i + 1) * 1e4 for i in range(len(maturities))]
max_dev = max(abs(np.array(chk) - spreads))

# ---------- 图1: 市场 CDS 溢价曲线 ----------
fig1, ax = plt.subplots(figsize=(11, 6))
ax.plot(maturities, spreads, "-o", color=C["path"], lw=2.4, ms=5, label="市场报价")
ax.plot(maturities, chk, "--s", color=C["true"], lw=1.4, ms=4, label="bootstrap 还原")
ax.set_title("市场 CDS 溢价曲线 (bps)：信用越差，长端越陡", fontsize=12.5)
ax.set_xlabel("期限 (年)"); ax.set_ylabel("CDS 溢价 (bps)")
ax.legend(fontsize=9); ax.grid(color=C["grid"], lw=0.5)
fig1.tight_layout(); fig1.savefig(os.path.join(OUT, "cds_spread_curve.png")); plt.close(fig1)

# ---------- 图2: bootstrap 违约强度 λ(t) ----------
fig2, ax2 = plt.subplots(figsize=(11, 6))
ax2.step(maturities, h * 1e4, where="post", color=C["mean"], lw=2.4)
ax2.set_title("Bootstrap 违约强度 λ(t) (年化 bps)：由溢价曲线反推", fontsize=12.5)
ax2.set_xlabel("期限 (年)"); ax2.set_ylabel("违约强度 λ (bps/年)")
ax2.grid(color=C["grid"], lw=0.5)
fig2.tight_layout(); fig2.savefig(os.path.join(OUT, "cds_hazard_curve.png")); plt.close(fig2)

# ---------- 图3: 累计违约概率 ----------
fig3, ax3 = plt.subplots(figsize=(11, 6))
ax3.plot(maturities, cum_def * 100, "-o", color=C["fit"], lw=2.4, ms=5)
ax3.set_title("累计违约概率 P(τ ≤ t)：10y 约违约 %.1f%%" % (cum_def[-1] * 100), fontsize=12.5)
ax3.set_xlabel("期限 (年)"); ax3.set_ylabel("累计违约概率 %")
ax3.grid(color=C["grid"], lw=0.5)
fig3.tight_layout(); fig3.savefig(os.path.join(OUT, "cds_cum_default.png")); plt.close(fig3)

# ---------- 图4: 给定 5y 报价 100bps, 公允 NPV 随合约利差 ----------
idx5 = list(maturities).index(5.0)
h5 = h[:idx5 + 1]
m5v = maturities[:idx5 + 1]
df5 = df[:idx5 + 1]
rpv5, _ = cds_legs_upto(h5, m5v, df5, R, idx5 + 1)
notional = 1e7
s_mkt = 100.0 / 1e4
contract_spreads = np.linspace(60, 160, 100) / 1e4
npv = (contract_spreads - s_mkt) * rpv5 * notional / 1e4   # 万元
fig4, ax4 = plt.subplots(figsize=(11, 6))
ax4.plot(contract_spreads * 1e4, npv, color=C["curve"], lw=2.4)
ax4.axvline(100, color=C["true"], ls=":", lw=1.4, label="市场 5y=100bps (平价 NPV=0)")
ax4.axhline(0, color=C["black"], lw=0.8)
ax4.set_title("5y CDS 公允净现值：以 120bps 卖保护赚, 以 80bps 买保护亏 (RPV01=%.3f)" % rpv5, fontsize=11.2)
ax4.set_xlabel("合约 CDS 溢价 (bps)"); ax4.set_ylabel("净现值 (万元)")
ax4.legend(fontsize=9); ax4.grid(color=C["grid"], lw=0.5)
fig4.tight_layout(); fig4.savefig(os.path.join(OUT, "cds_npv.png")); plt.close(fig4)

print("cds images done. 10y cum_default=%.2f%%  RPV01_5y=%.4f  max|chk-spread|=%.4fbps"
      % (cum_def[-1] * 100, rpv5, max_dev))
print("h(bps)=", np.round(h * 1e4, 1))
print("chk(bps)=", np.round(chk, 1))
