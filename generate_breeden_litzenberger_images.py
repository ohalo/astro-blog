#!/usr/bin/env python3
"""
为文章「Breeden-Litzenberger 隐含风险中性密度：从期权价格反推市场预期的崩盘」
(breeden-litzenberger-implied-density) 生成真实配图与真实统计数字。

核心方法：
  Breeden-Litzenberger(1978)：风险中性密度 q(K) = e^{rT} * d²C/dK²
  即看涨期权价格对行权价的二阶导，直接反推出市场对到期日标的分布的预期。
  - 用 Black-Scholes(带波动率微笑) 生成一条 C(K) 曲线
  - 数值二阶差分 → 隐含密度 q(K)
  - 对比：无微笑(常数 IV) 得到对数正态密度；有微笑(左偏) 得到肥左尾/崩盘预期
数据：全部由 BS 公式 + 指定波动率微笑真实计算，确定性。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

# NumPy 2.x 兼容
trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "breeden-litzenberger-implied-density")
os.makedirs(D, exist_ok=True)

C = {"eq": "#2F4B7C", "up": "#55A868", "dn": "#C44E52", "grid": "#DDDDDD",
     "smile": "#C44E52", "flat": "#4C72B0", "shade": "#F2C0C0", "rn": "#8172B3"}

# ---------- 自包含正态 CDF / PDF ----------
def norm_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)

def norm_cdf(x):
    # Abramowitz-Stegun 7.1.26 误差函数近似
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    z = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    erf = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-z * z)
    return 0.5 * (1.0 + sign * erf)

# ---------- Black-Scholes 看涨 ----------
def bs_call(S, K, r, sigma, T):
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)

# ---------- 参数 ----------
S0 = 100.0
r = 0.02
T = 0.5
atm_vol = 0.20

# 行权价网格（细网格用于数值二阶导）
K = np.linspace(55, 150, 951)
moneyness = np.log(K / S0)

# 波动率微笑：左偏（skew），OTM put(低行权价) 波动率更高 -> 崩盘预期
# sigma(K) = atm + skew*(-m) + smile*m^2
skew = 0.55
curv = 0.60
sig_smile = atm_vol + skew * (-moneyness) + curv * moneyness ** 2
sig_smile = np.clip(sig_smile, 0.08, 0.80)
sig_flat = np.full_like(K, atm_vol)

# 期权价
C_smile = bs_call(S0, K, r, sig_smile, T)
C_flat = bs_call(S0, K, r, sig_flat, T)

# Breeden-Litzenberger: q(K) = e^{rT} d²C/dK²
def rn_density(Cvals, Kvals, r, T):
    d2 = np.gradient(np.gradient(Cvals, Kvals), Kvals)
    q = np.exp(r * T) * d2
    q = np.clip(q, 0, None)
    # 归一化
    area = trapz(q, Kvals)
    return q / area if area > 0 else q

q_smile = rn_density(C_smile, K, r, T)
q_flat = rn_density(C_flat, K, r, T)

# =====================================================================
# 图1：波动率微笑本身（左偏）
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(K, sig_smile * 100, color=C["smile"], lw=2.4, label="真实微笑 IV（左偏）")
ax.plot(K, sig_flat * 100, color=C["flat"], lw=2.0, ls="--", label="常数 IV（BS 假设）")
ax.axvline(S0, color="#888888", lw=1.2, ls=":", label="平值 S₀=100")
ax.set_xlabel("行权价 K")
ax.set_ylabel("隐含波动率 (%)")
ax.set_title("波动率微笑：OTM 看跌期权 IV 系统性更高 = 市场为崩盘付溢价")
ax.grid(True, color=C["grid"], alpha=0.6)
ax.legend(loc="upper right")
ax.set_xlim(55, 150)
fig.tight_layout()
fig.savefig(os.path.join(D, "bl_smile.png"), dpi=130)
plt.close(fig)

# =====================================================================
# 图2：看涨期权价格曲线 C(K) 及其凸性
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(K, C_smile, color=C["smile"], lw=2.4, label="C(K) 含微笑")
ax.plot(K, C_flat, color=C["flat"], lw=2.0, ls="--", label="C(K) 常数 IV")
ax.set_xlabel("行权价 K")
ax.set_ylabel("看涨期权价格 C(K)")
ax.set_title("看涨期权价格对 K 单调下降且凸——二阶导即风险中性密度")
ax.grid(True, color=C["grid"], alpha=0.6)
ax.legend(loc="upper right")
ax.set_xlim(55, 150)
fig.tight_layout()
fig.savefig(os.path.join(D, "bl_call_curve.png"), dpi=130)
plt.close(fig)

# =====================================================================
# 图3：反推的风险中性密度（核心图）——微笑 vs 常数
# =====================================================================
fig, ax = plt.subplots(figsize=(9.2, 5.4))
ax.plot(K, q_flat, color=C["flat"], lw=2.2, ls="--", label="常数 IV → 对数正态（对称尾）")
ax.plot(K, q_smile, color=C["smile"], lw=2.6, label="含微笑 → 肥左尾（崩盘预期）")
# 阴影标注左尾
mask = K < 80
ax.fill_between(K[mask], 0, q_smile[mask], color=C["shade"], alpha=0.7,
                label="左尾崩盘概率质量")
ax.axvline(S0, color="#888888", lw=1.1, ls=":", label="现价 S₀=100")
ax.set_xlabel("到期标的价格 S_T")
ax.set_ylabel("风险中性密度 q(S_T)")
ax.set_title("Breeden-Litzenberger 反推密度：微笑把概率质量搬到左尾")
ax.grid(True, color=C["grid"], alpha=0.6)
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(55, 150)
fig.tight_layout()
fig.savefig(os.path.join(D, "bl_rn_density.png"), dpi=130)
plt.close(fig)

# =====================================================================
# 图4：左尾概率对比柱状——市场预期的崩盘概率
# =====================================================================
def tail_prob(q, Kv, thresh):
    m = Kv <= thresh
    return trapz(q[m], Kv[m])

levels = [70, 75, 80, 85, 90]
p_smile = [tail_prob(q_smile, K, S0 * l / 100 * 100 / 100 * 1.0) for l in levels]
# 更直接：阈值就是价格水平
p_smile = [tail_prob(q_smile, K, l) for l in levels]
p_flat = [tail_prob(q_flat, K, l) for l in levels]

x = np.arange(len(levels))
w = 0.38
fig, ax = plt.subplots(figsize=(9, 5.2))
b1 = ax.bar(x - w / 2, np.array(p_flat) * 100, w, color=C["flat"], label="常数 IV（低估崩盘）")
b2 = ax.bar(x + w / 2, np.array(p_smile) * 100, w, color=C["smile"], label="含微笑（真实崩盘预期）")
ax.set_xticks(x)
ax.set_xticklabels([f"S_T<{l}\n(−{100-l}%)" for l in levels])
ax.set_ylabel("到期概率 (%)")
ax.set_title("市场预期的下跌概率：微笑把尾部风险显性化")
ax.grid(True, axis="y", color=C["grid"], alpha=0.6)
ax.legend(loc="upper right")
for b in list(b1) + list(b2):
    h = b.get_height()
    ax.annotate(f"{h:.1f}", (b.get_x() + b.get_width() / 2, h),
                ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(D, "bl_tail_prob.png"), dpi=130)
plt.close(fig)

# ---------- 打印真实统计数字 ----------
mean_smile = trapz(K * q_smile, K)
mean_flat = trapz(K * q_flat, K)
def skew_of(q, Kv):
    m = trapz(Kv * q, Kv)
    v = trapz((Kv - m) ** 2 * q, Kv)
    s = trapz((Kv - m) ** 3 * q, Kv) / v ** 1.5
    return s, np.sqrt(v)

sk_s, sd_s = skew_of(q_smile, K)
sk_f, sd_f = skew_of(q_flat, K)

print("=== Breeden-Litzenberger 真实统计 ===")
print(f"含微笑密度: 均值={mean_smile:.2f}, 标准差={sd_s:.2f}, 偏度={sk_s:.3f}")
print(f"常数IV密度: 均值={mean_flat:.2f}, 标准差={sd_f:.2f}, 偏度={sk_f:.3f}")
print(f"P(S_T<80) 含微笑={tail_prob(q_smile,K,80)*100:.2f}%  常数IV={tail_prob(q_flat,K,80)*100:.2f}%")
print(f"P(S_T<75) 含微笑={tail_prob(q_smile,K,75)*100:.2f}%  常数IV={tail_prob(q_flat,K,75)*100:.2f}%")
print(f"P(S_T<70) 含微笑={tail_prob(q_smile,K,70)*100:.2f}%  常数IV={tail_prob(q_flat,K,70)*100:.2f}%")
print(f"密度积分校验: 微笑={trapz(q_smile,K):.4f}, 常数={trapz(q_flat,K):.4f}")
print("图片已生成:", os.listdir(D))
