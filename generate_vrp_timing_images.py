#!/usr/bin/env python3
"""
为文章「波动率风险溢价择时：用 VRP 开关股债配置」(vrp-timing-allocation) 生成真实配图。

核心逻辑（波动率风险溢价 VRP）：
  VRP_t = σ_implied²_t − σ_realized²_t
  其中 σ_implied 来自期权隐含波动率（如 VIX），σ_realized 来自已实现收益方差。
  VRP>0 = 市场为「保险」付费（恐慌溢价）→ 历史上对应权益未来超额收益为正。
  择时规则：VRP 高于阈值 → 配股票；VRP 低（恐慌消失/自满）→ 配债券。
  全部为合成但贴合真实结构的数值（波动率聚集、VIX 在危机飙升、VRP 月度均值正），
  非占位图。
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
D = os.path.join(BASE, "vrp-timing-allocation")
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(20260715)

T = 240  # 240 个月（20 年）

# ---------- 1) 构造已实现方差路径（波动率聚集 + 危机尖峰）----------
state_vol = 0.04
realized_var = np.zeros(T)
for t in range(T):
    state_vol += 0.85 * (0.04 - state_vol) + rng.normal(0, 0.004)
    state_vol = max(state_vol, 0.02)
    realized_var[t] = state_vol ** 2
# 注入危机：已实现方差尖峰（崩盘时波动飙升）
crises = [(48, 56), (120, 128), (200, 208)]
for a, b in crises:
    realized_var[a:b] *= 5.0

# ---------- 2) 隐含方差 + VRP（恐慌时 VRP 系统性为正）----------
vrp = (0.0006 + 0.8 * np.maximum(realized_var - 0.04 ** 2, 0)
       + rng.normal(0, 0.0002, T))
implied_var = realized_var + vrp
implied_vol = np.sqrt(implied_var) * np.sqrt(12) * 100   # 年化隐含波动 %
realized_vol = np.sqrt(realized_var) * np.sqrt(12) * 100
VRP = implied_var - realized_var                         # 波动率风险溢价（方差差）
VRP_volpts = implied_vol - realized_vol                  # 波动率点（用于解读/择时）

# ---------- 3) 权益收益：条件预期溢价由「上月 VRP(波动率点)」正向驱动（BTZ 机制）----------
# 危机窗口首月急跌，其后高 VRP 区连续反弹 → 滞后 VRP 与下月权益超额正相关。
zvrp_lag = (VRP_volpts - VRP_volpts.mean()) / VRP_volpts.std()
e_ret = np.zeros(T)
e_ret[0] = 0.009 + rng.normal(0, 0.045)
for t in range(1, T):
    prem = 0.009 + 0.006 * zvrp_lag[t - 1]     # 上月 VRP 越高 → 本月条件预期溢价越高（温和）
    crash = -0.11 if any(a <= t < b for a, b in crises) and (t - a) == 0 else 0.0
    e_ret[t] = prem + crash + rng.normal(0, 0.042)

# 债券收益：温和票息 + 危机避险（崩盘首月走强）
b_ret = 0.004 + 0.3 * (-(e_ret - 0.009)) + rng.normal(0, 0.008, T)
for a, b in crises:
    b_ret[a] += 0.014

# ---------- 4) 择时规则：VRP 高于阈值 → 股票，否则债券 ----------
threshold = np.percentile(VRP_volpts, 60)   # 60 分位为开关
alloc_eq = (VRP_volpts > threshold).astype(float)
timing_ret = alloc_eq * e_ret + (1 - alloc_eq) * b_ret

# 对比基准
bh_eq = e_ret.copy()
alloc_6040 = 0.6 * e_ret + 0.4 * b_ret

def perf(r):
    r = np.array(r)
    cum = np.cumprod(1 + r)
    total = cum[-1] - 1
    ann = cum[-1] ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12)
    sharpe = (r.mean() * 12 - 0.02) / vol if vol > 0 else 0
    peak = np.maximum.accumulate(cum)
    mdd = (cum / peak - 1).min()
    return dict(total=total, ann=ann, vol=vol, sharpe=sharpe, mdd=mdd)

p_t = perf(timing_ret)
p_e = perf(bh_eq)
p_6 = perf(alloc_6040)

# ---------- 图1：隐含波动 vs 已实现波动 + 危机阴影 ----------
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(implied_vol, color="#c44e52", lw=1.8, label="隐含波动 (VIX 式)")
ax.plot(realized_vol, color="#4c72b0", lw=1.8, label="已实现波动")
ax.fill_between(range(T), realized_vol, implied_vol,
                where=(implied_vol >= realized_vol), color="#c44e52", alpha=0.12,
                label="VRP = 隐含 − 已实现 > 0")
for a, b in crises:
    ax.axvspan(a, b, color="#888", alpha=0.12)
ax.set_xlabel("月份", fontsize=11)
ax.set_ylabel("年化波动率 (%)", fontsize=11)
ax.set_title("波动率风险溢价：隐含波动系统性高于已实现波动（恐慌溢价）",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "vrp_series.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------- 图2：VRP 与下月权益超额收益（正相关 = 可择时）----------
fwd_ex = e_ret[1:] - b_ret[1:]
vrp_lag = VRP_volpts[:-1]
b1, b0 = np.polyfit(vrp_lag, fwd_ex, 1)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(vrp_lag * 100, fwd_ex * 100, s=12, alpha=0.25, color="#4c72b0", edgecolors="none")
xs = np.linspace(vrp_lag.min(), vrp_lag.max(), 50)
ax.plot(xs, (b0 + b1 * xs) * 100, color="#c44e52", lw=2.2,
        label=f"OLS 斜率 = {b1*100:.2f} %/(vol点)")
ax.set_xlabel("当月 VRP（波动率点 ×100）", fontsize=11)
ax.set_ylabel("下月权益−债券 超额收益 (%)", fontsize=11)
ax.set_title("VRP 越高，下月权益相对债券越赚：择时的统计基础",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "vrp_forward.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------- 图3：三条净值曲线对比 ----------
fig, ax = plt.subplots(figsize=(11, 6))
def cum(r):
    return np.cumprod(1 + np.array(r))
ax.plot(cum(bh_eq), color="#4c72b0", lw=1.8, label=f"买入持有股票 (年化 {p_e['ann']*100:.1f}%, SR {p_e['sharpe']:.2f})")
ax.plot(cum(alloc_6040), color="#55a868", lw=1.8, label=f"60/40 (年化 {p_6['ann']*100:.1f}%, SR {p_6['sharpe']:.2f})")
ax.plot(cum(timing_ret), color="#c44e52", lw=2.0, label=f"VRP 择时 (年化 {p_t['ann']*100:.1f}%, SR {p_t['sharpe']:.2f})")
ax.set_xlabel("月份", fontsize=11)
ax.set_ylabel("净值（起始=1）", fontsize=11)
ax.set_title("VRP 股债择时：守住回撤又不丢太多收益",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "vrp_equity.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------- 图4：回撤对比 ----------
def drawdown(r):
    cum = np.cumprod(1 + np.array(r))
    peak = np.maximum.accumulate(cum)
    return cum / peak - 1

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.fill_between(range(T), drawdown(bh_eq) * 100, color="#4c72b0", alpha=0.4, label="买入持有股票")
ax.fill_between(range(T), drawdown(timing_ret) * 100, color="#c44e52", alpha=0.6, label="VRP 择时")
ax.set_xlabel("月份", fontsize=11)
ax.set_ylabel("回撤 (%)", fontsize=11)
ax.set_title(f"回撤对比：VRP 择时最大回撤 {p_t['mdd']*100:.1f}% vs 股票 {p_e['mdd']*100:.1f}%",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="lower left", fontsize=10)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "vrp_drawdown.png"), dpi=150, bbox_inches="tight")
plt.close()

print("=== VRP 择时 诊断 ===")
print(f"VRP 月度均值(波动率点): {VRP_volpts.mean():.2f}  | >0 占比: {(VRP_volpts>0).mean()*100:.0f}%")
print(f"VRP vs 下月权益超额 SLOPE: {b1*100:.3f} %/(vol点) | 相关性: {np.corrcoef(vrp_lag, fwd_ex)[0,1]:.3f}")
print(f"买入持有股票: 年化 {p_e['ann']*100:.1f}% / SR {p_e['sharpe']:.2f} / MDD {p_e['mdd']*100:.1f}%")
print(f"60/40:        年化 {p_6['ann']*100:.1f}% / SR {p_6['sharpe']:.2f} / MDD {p_6['mdd']*100:.1f}%")
print(f"VRP 择时:     年化 {p_t['ann']*100:.1f}% / SR {p_t['sharpe']:.2f} / MDD {p_t['mdd']*100:.1f}%")
print(f"股票配置月份占比: {alloc_eq.mean()*100:.0f}%")
print(f"生成图片: {sorted(os.listdir(D))}")
