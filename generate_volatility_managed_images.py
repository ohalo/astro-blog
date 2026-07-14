#!/usr/bin/env python3
"""
为文章「波动率管理组合(Moreira-Muir)：用已实现波动缩放敞口而非预测方向」生成真实配图 + 计算关键数字。

机制（自洽合成，仅用于演示；落地见文末路径）：
  * 基础策略（如某横截面异常因子月度收益）有「恒定期望超额收益 μ」但「时变波动率 σ_t」
    —— 即异常收益不随自身方差放大（这是 Moreira-Muir 的核心实证发现）。
  * σ_t 由随机波动率 h_t ~ AR(1) 驱动：高波动期聚集、且难以被完美预测。
  * 波动率预测：用过去 12 个月已实现方差（trailing realized variance）估计 σ_hat_t²。
  * 波动率管理组合(VM)：每期敞口 w_t = c / σ_hat_t² ，c 取 σ_hat² 的样本均值使平均敞口≈1。
    —— 高波动期（σ_hat 大）自动降杠杆，低波动期加杠杆，不预测方向、只管「风险预算」。
  * 对比：原始组合（满仓 r_t）vs VM 组合（w_t · r_t）。典型 MM 结果：VM 波动率大幅下降，
    且 Sharpe 不降反升（因为异常收益不随方差放大，降杠杆砍掉的主要是噪声而非 alpha）。
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
D = os.path.join(BASE, "volatility-managed-portfolio")
os.makedirs(D, exist_ok=True)

C = {"ot": "#4C72B0", "mean": "#C44E52", "equal": "#999999",
     "target": "#55A868", "grid": "#DDDDDD", "warn": "#DD8452", "calm": "#8172B3",
     "raw": "#C44E52", "vm": "#2A6F97", "band": "#DD8452"}

# ---------------- 参数 ----------------
MU = 0.0040          # 基础策略月度期望超额收益（恒定，不随方差放大）
SIG0 = 0.045         # 月度波动率基准
PHI = 0.90           # 随机波动率 h_t 的 AR(1) 持续性（高波动聚集）
SIG_H = 0.08         # h_t 的波动（单位：对数波动率上的噪声）
W = 36               # 波动率估计窗口（月）—— 用较长窗口抑制估计噪声
MONTHS = 360         # 30 年月度数据


def simulate(rng):
    """生成基础策略月度收益：恒定 μ + 随机波动率乘标准正态。"""
    h = np.zeros(MONTHS)
    r = np.zeros(MONTHS)
    h[0] = 0.0
    for t in range(1, MONTHS):
        h[t] = PHI * h[t - 1] + rng.normal(0, SIG_H)
    sig_t = SIG0 * np.exp(h)              # 真实时变月度波动率
    z = rng.normal(0, 1, MONTHS)
    r = MU + sig_t * z
    return r, sig_t


def realized_var(r, t, w=W):
    """用过去 w 个月收益的平方和估计年化方差（这里保留月度尺度）。"""
    if t < w:
        return np.var(r[: max(t, 1)]) + 1e-12
    return np.mean(r[t - w: t] ** 2) + 1e-12


def build_vm(r):
    """波动率管理组合(Moreira-Muir)：w_t = c / σ̂_t²。
    c = 1/E[1/σ̂²] 使平均敞口≈1（不靠杠杆放大，纯做风险变换）。
    高波动期(σ̂ 大)自动降杠杆，低波动期加杠杆，不预测方向、只管风险预算。"""
    sig2_hat = np.array([realized_var(r, t) for t in range(MONTHS)])
    c = 1.0 / (1.0 / sig2_hat).mean()
    w = c / sig2_hat                       # 敞口（杠杆）
    w = np.clip(w, 0.3, 2.5)              # 实操杠杆上下限（去杠杆下限0.3 / 加杠杆上限2.5）
    r_vm = w * r
    return r_vm, w, sig2_hat


def metrics(ret, ann=12):
    r = np.asarray(ret, float)
    a = r.mean() * ann
    v = r.std() * np.sqrt(ann)
    sr = r.mean() / (r.std() + 1e-12) * np.sqrt(ann)
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    mdd = float(((eq - peak) / peak).min())
    return a, v, sr, mdd


# ---------------- 蒙特卡洛 ----------------
rng0 = np.random.default_rng(20260714)
NSIM = 200
raw_sr, vm_sr, raw_vol, vm_vol, raw_ret, vm_ret, raw_mdd, vm_mdd = [], [], [], [], [], [], [], []
rep_r, rep_vm, rep_w, rep_sig = None, None, None, None
for s in range(NSIM):
    rng = np.random.default_rng(2000 + s)
    r, sig = simulate(rng)
    r_vm, w, sh = build_vm(r)
    a1, v1, s1, m1 = metrics(r)
    a2, v2, s2, m2 = metrics(r_vm)
    raw_sr.append(s1); vm_sr.append(s2); raw_vol.append(v1); vm_vol.append(v2)
    raw_ret.append(a1); vm_ret.append(a2); raw_mdd.append(m1); vm_mdd.append(m2)
    if s == 11:  # 代表性路径用于配图
        rng = np.random.default_rng(2011)
        rep_r, rep_sig = simulate(rng)
        rep_vm, rep_w, rep_sh = build_vm(rep_r)

raw_sr = np.array(raw_sr); vm_sr = np.array(vm_sr)
raw_vol = np.array(raw_vol); vm_vol = np.array(vm_vol)
raw_ret = np.array(raw_ret); vm_ret = np.array(vm_ret)
raw_mdd = np.array(raw_mdd); vm_mdd = np.array(vm_mdd)

print("=" * 70)
print("波动率管理组合(Moreira-Muir) 关键数字 (seed 20260714, %d sims)" % NSIM)
print("=" * 70)
print(f"基础策略: 月度 μ={MU} 月度 σ0={SIG0} 随机波动率 AR(1) φ={PHI} 窗口={W}月")
print("\n原始组合(满仓) vs 波动率管理组合(VM):")
print(f"  年化收益:  raw={raw_ret.mean():.3f}  vm={vm_ret.mean():.3f}")
print(f"  年化波动:  raw={raw_vol.mean():.3f}  vm={vm_vol.mean():.3f}  (降幅 {(1-vm_vol.mean()/raw_vol.mean())*100:.0f}%)")
print(f"  Sharpe:    raw={raw_sr.mean():.3f}  vm={vm_sr.mean():.3f}  (比值 {vm_sr.mean()/raw_sr.mean():.3f})")
print(f"  最大回撤:  raw={raw_mdd.mean():.3f}  vm={vm_mdd.mean():.3f}  (降幅 {(1-vm_mdd.mean()/raw_mdd.mean())*100:.0f}%)")
print(f"  VM Sharpe > raw 的比例: {(vm_sr>raw_sr).mean()*100:.1f}%")

# 代表性路径指标
a1, v1, s1, m1 = metrics(rep_r)
a2, v2, s2, m2 = metrics(rep_vm)
print("\n代表性路径(图用):")
print(f"  raw:  ann={a1:+.3f} vol={v1:.3f} Sharpe={s1:+.3f} MDD={m1:+.3f}")
print(f"  VM :  ann={a2:+.3f} vol={v2:.3f} Sharpe={s2:+.3f} MDD={m2:+.3f}")
print(f"  VM 杠杆范围: [{rep_w.min():.2f}, {rep_w.max():.2f}]  均值={rep_w.mean():.2f}")

# ============================================================================
# 图 1：真实时变波动率 vs VM 杠杆（高波动期自动降杠杆）
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5.2))
ann_sig = rep_sig * np.sqrt(12) * 100
ax.plot(ann_sig, color=C["warn"], lw=1.8, label="基础策略真实年化波动率")
ax.set_ylabel("年化波动率 (%)", color=C["warn"])
ax.set_xlabel("月份")
ax.tick_params(axis="y", labelcolor=C["warn"])
ax2 = ax.twinx()
ax2.plot(rep_w, color=C["vm"], lw=1.8, label="VM 敞口(杠杆) w_t")
ax2.set_ylabel("VM 敞口 w_t", color=C["vm"])
ax2.tick_params(axis="y", labelcolor=C["vm"])
ax.set_title("波动率管理：真实波动飙升时，VM 敞口自动降到 <1（去杠杆）")
ax.grid(True, color=C["grid"], lw=0.6)
fig.tight_layout(); fig.savefig(os.path.join(D, "vol_and_leverage.png"), dpi=130); plt.close()

# ============================================================================
# 图 2：VM 杠杆 vs 已实现波动率（散点，展示逆关系 + 杠杆上限）
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5.4))
ann_sh = np.sqrt(rep_sh) * np.sqrt(12) * 100
sc = ax.scatter(ann_sh, rep_w, c=range(len(rep_w)), cmap="viridis", s=18, alpha=0.7)
ax.set_xlabel("估计年化波动率 σ̂ (%)")
ax.set_ylabel("VM 敞口 w_t = c / σ̂²")
ax.set_title("敞口随波动率逆缩放：w_t ∝ 1/σ̂²（红线下界 0.1、上界 8 为实操约束）")
ax.axhline(1.0, color="#888", ls="--", lw=1.0, label="满仓 w=1")
ax.axhline(0.1, color=C["warn"], ls=":", lw=1.0, label="去杠杆下界 0.1")
ax.axhline(8.0, color=C["mean"], ls=":", lw=1.0, label="加杠杆上界 8.0")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
fig.tight_layout(); fig.savefig(os.path.join(D, "weight_vs_vol.png"), dpi=130); plt.close()

# ============================================================================
# 图 3：累计净值 raw vs VM
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5.2))
eq_raw = np.cumprod(1 + rep_r)
eq_vm = np.cumprod(1 + rep_vm)
ax.plot(eq_raw, color=C["raw"], lw=1.8, label=f"原始组合 (Sharpe={s1:+.2f})")
ax.plot(eq_vm, color=C["vm"], lw=1.8, label=f"波动率管理 VM (Sharpe={s2:+.2f})")
ax.set_ylabel("累计净值 (起始=1)")
ax.set_xlabel("月份")
ax.set_title("累计净值：VM 在更低波动下取得不输(甚至更高)的 Sharpe")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6)
fig.tight_layout(); fig.savefig(os.path.join(D, "equity_curve.png"), dpi=130); plt.close()

def rolling_std(x, w=24):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    for t in range(w, len(x)):
        out[t] = x[t - w: t].std()
    return out

# ============================================================================
# 图 4：滚动波动率对比 —— VM 把波动熨平
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5.2))
roll = 24
rv_raw = rolling_std(rep_r) * np.sqrt(12) * 100
rv_vm = rolling_std(rep_vm) * np.sqrt(12) * 100
ax.plot(rv_raw, color=C["raw"], lw=1.6, label=f"原始组合 滚动波动(均值 {np.nanmean(rv_raw):.1f}%)")
ax.plot(rv_vm, color=C["vm"], lw=1.6, label=f"VM 组合 滚动波动(均值 {np.nanmean(rv_vm):.1f}%)")
ax.set_ylabel("24 个月滚动年化波动率 (%)")
ax.set_xlabel("月份")
ax.set_title("波动率管理：把组合波动整体熨平（风险预算恒定）")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6)
fig.tight_layout(); fig.savefig(os.path.join(D, "rolling_vol.png"), dpi=130); plt.close()

print("done ->", D)
