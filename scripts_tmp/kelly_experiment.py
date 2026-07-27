# -*- coding: utf-8 -*-
"""状态条件 Kelly：两状态 regime-switching 市场下的最优杠杆"""
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)

# ---- 两状态马尔可夫 regime：牛市 / 熊市 ----
# 日频参数（年化：牛 mu=12%, sig=12%; 熊 mu=-18%, sig=32%）
mu_b, sig_b = 0.12 / 252, 0.12 / np.sqrt(252)
mu_c, sig_c = -0.18 / 252, 0.32 / np.sqrt(252)
# 转移概率：牛市平均持续 ~1年, 熊市 ~4个月
p_bb, p_cc = 1 - 1 / 252, 1 - 1 / 84
P = np.array([[p_bb, 1 - p_bb], [1 - p_cc, p_cc]])
pi_stat = np.array([ (1-p_cc), (1-p_bb) ]); pi_stat = pi_stat / pi_stat.sum()
print(f"平稳分布: 牛 {pi_stat[0]:.3f}, 熊 {pi_stat[1]:.3f}")

# 无条件矩
mu_u = pi_stat[0]*mu_b + pi_stat[1]*mu_c
var_u = pi_stat[0]*(sig_b**2 + mu_b**2) + pi_stat[1]*(sig_c**2 + mu_c**2) - mu_u**2
print(f"无条件: mu_ann={mu_u*252:.3%}, sig_ann={np.sqrt(var_u*252):.3%}")

# Kelly 公式（连续近似 f* = mu/sigma^2，无风险利率设 0）
f_uncond = mu_u / var_u
f_bull = mu_b / sig_b**2
f_bear = mu_c / sig_c**2
print(f"Kelly: 无条件 f={f_uncond:.2f}, 牛市 f={f_bull:.2f}, 熊市 f={f_bear:.2f}")

# ---- 模拟 60 年市场 ----
n = 252 * 60
state = np.empty(n, dtype=int)
state[0] = 0
for t in range(1, n):
    state[t] = rng.random() > (P[state[t-1], state[t-1]]) and 1 - state[t-1] or state[t-1]
# 更正：上面写法有坑，重写
state = np.empty(n, dtype=int); state[0] = 0
u01 = rng.random(n)
for t in range(1, n):
    s = state[t-1]
    stay = P[s, s]
    state[t] = s if u01[t] < stay else 1 - s
r = np.where(state == 0,
             rng.normal(mu_b, sig_b, n),
             rng.normal(mu_c, sig_c, n))
print(f"模拟 {n} 天, 熊市占比 {state.mean():.3f}")

# ---- 贝叶斯滤波估计 regime 概率（观察者不知道真实状态）----
# 用已知参数的 Hamilton filter
lik_b = stats.norm.pdf(r, mu_b, sig_b)
lik_c = stats.norm.pdf(r, mu_c, sig_c)
p_bull = np.empty(n)
prior = pi_stat[0]
for t in range(n):
    # 预测
    if t > 0:
        prior = p_bull[t-1] * p_bb + (1 - p_bull[t-1]) * (1 - p_cc)
    num = prior * lik_b[t]
    den = num + (1 - prior) * lik_c[t]
    p_bull[t] = num / den
acc = ((p_bull > 0.5) == (state == 0)).mean()
print(f"滤波准确率: {acc:.3f}")

# ---- 各策略回测（信号滞后一天执行：用 t-1 的滤波概率决定 t 的杠杆）----
def run(lev):
    lev = np.clip(lev, 0.0, 2.0)  # 杠杆上限 2 倍，禁止做空（可讨论）
    gr = 1 + lev * r
    gr = np.maximum(gr, 1e-9)
    return np.cumprod(gr)

# 1) 无条件 Kelly（半 Kelly 也算）
lev_uncond = np.full(n, f_uncond)
# 2) 状态条件 Kelly：f_t = E[mu|p]/E[var|p] 用滤波概率混合矩
p_lag = np.concatenate([[pi_stat[0]], p_bull[:-1]])  # 滞后一天
mu_mix = p_lag * mu_b + (1 - p_lag) * mu_c
var_mix = p_lag * (sig_b**2 + mu_b**2) + (1 - p_lag) * (sig_c**2 + mu_c**2) - mu_mix**2
lev_regime = mu_mix / var_mix
# 3) 全知（用真实状态，作弊上限）
state_lag = np.concatenate([[0], state[:-1]])
lev_oracle = np.where(state_lag == 0, f_bull, f_bear)
# 4) 半状态 Kelly
lev_half = 0.5 * lev_regime
# 5) buy and hold
lev_bh = np.full(n, 1.0)

curves = {}
for name, lv in [("无条件 Kelly", lev_uncond), ("状态条件 Kelly", lev_regime),
                 ("半状态 Kelly", lev_half), ("全知 Kelly（作弊上限）", lev_oracle),
                 ("买入持有", lev_bh)]:
    eq = run(lv)
    lg = np.log(eq[-1]) / (n / 252)
    dd = 1 - eq / np.maximum.accumulate(eq)
    ret = np.diff(np.log(eq))
    sh = ret.mean() / ret.std() * np.sqrt(252)
    curves[name] = (eq, lg, dd.max(), sh)
    print(f"{name}: 年化对数增长 {lg:.3%}, 最大回撤 {dd.max():.1%}, Sharpe {sh:.2f}")

# ---- 参数误差敏感性：把熊市 mu 估错 ----
errs = np.linspace(-0.15, 0.15, 13)  # 年化误差
growth_err_regime, growth_err_uncond = [], []
for e in errs:
    mu_c_hat = (-0.18 + e) / 252
    mu_mix_e = p_lag * mu_b + (1 - p_lag) * mu_c_hat
    var_mix_e = p_lag * (sig_b**2 + mu_b**2) + (1 - p_lag) * (sig_c**2 + mu_c_hat**2) - mu_mix_e**2
    lv = mu_mix_e / var_mix_e
    eq = run(lv)
    growth_err_regime.append(np.log(eq[-1]) / (n / 252))
    mu_u_e = pi_stat[0]*mu_b + pi_stat[1]*mu_c_hat
    var_u_e = pi_stat[0]*(sig_b**2+mu_b**2) + pi_stat[1]*(sig_c**2+mu_c_hat**2) - mu_u_e**2
    eq2 = run(np.full(n, mu_u_e / var_u_e))
    growth_err_uncond.append(np.log(eq2[-1]) / (n / 252))
print("熊市均值误差敏感性 (err_ann, regime, uncond):")
for e, g1, g2 in zip(errs, growth_err_regime, growth_err_uncond):
    print(f"  {e:+.2f}  {g1:.3%}  {g2:.3%}")

# ---- 滤波延迟成本：转移后多少天才识别 ----
switch_idx = np.where(np.diff(state) != 0)[0] + 1
delays = []
for si in switch_idx:
    tgt = state[si]
    for k in range(si, min(si + 60, n)):
        if (p_bull[k] > 0.5) == (tgt == 0):
            delays.append(k - si)
            break
delays = np.array(delays)
print(f"状态切换 {len(switch_idx)} 次, 识别延迟中位数 {np.median(delays):.0f} 天, 均值 {delays.mean():.1f} 天, 90分位 {np.percentile(delays,90):.0f} 天")

# ================= 图 =================
import os
OUT = "/Users/halo/workspace/astro-blog/public/images/regime-conditional-kelly"
os.makedirs(OUT, exist_ok=True)
yrs = np.arange(n) / 252

# 图1：滤波概率 vs 真实状态（截取前 15 年）
m = 252 * 15
fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, height_ratios=[1, 1.4])
axes[0].fill_between(yrs[:m], 0, 1, where=state[:m] == 1, color="#f2c3c3", alpha=0.8, label="真实熊市")
axes[0].plot(yrs[:m], 1 - p_bull[:m], color="#7a1e1e", lw=0.7, label="滤波熊市概率")
axes[0].set_ylabel("P(熊市)")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_title("Hamilton 滤波：从收益率里实时推断 regime（前 15 年）")
axes[1].plot(yrs[:m], np.clip(lev_regime[:m], 0, 2), color="#2a6f97", lw=0.7, label="状态条件 Kelly 杠杆")
axes[1].axhline(np.clip(f_uncond, 0, 2), color="#d64545", ls="--", lw=1.2, label=f"无条件 Kelly = {f_uncond:.2f}")
axes[1].set_ylabel("杠杆倍数")
axes[1].set_xlabel("年")
axes[1].legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/kelly-filter-leverage.jpg", dpi=110, pil_kwargs={"quality": 88})
plt.close(fig)

# 图2：权益曲线（对数）
fig, ax = plt.subplots(figsize=(11, 5))
colors = {"无条件 Kelly": "#d64545", "状态条件 Kelly": "#2a6f97", "半状态 Kelly": "#5fa877",
          "全知 Kelly（作弊上限）": "#999999", "买入持有": "#c9a227"}
styles = {"全知 Kelly（作弊上限）": ":"}
for name, (eq, lg, mdd, sh) in curves.items():
    ax.plot(yrs, eq, lw=1.1, color=colors[name], ls=styles.get(name, "-"),
            label=f"{name}（年化 {lg:.1%}，回撤 {mdd:.0%}）")
ax.set_yscale("log")
ax.set_title("60 年权益曲线（对数轴）：把 regime 概率写进杠杆的差距")
ax.set_xlabel("年"); ax.set_ylabel("净值（初始=1）")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/kelly-equity-curves.jpg", dpi=110, pil_kwargs={"quality": 88})
plt.close(fig)

# 图3：参数误差敏感性
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.plot(errs * 100, np.array(growth_err_regime) * 100, "o-", color="#2a6f97", label="状态条件 Kelly")
ax.plot(errs * 100, np.array(growth_err_uncond) * 100, "s--", color="#d64545", label="无条件 Kelly")
ax.axvline(0, color="#888", lw=0.8, ls=":")
ax.set_title("熊市均值估计误差 vs 年化对数增长：Kelly 对参数误差的不对称惩罚")
ax.set_xlabel("熊市年化均值估计误差（百分点）")
ax.set_ylabel("年化对数增长 (%)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/kelly-parameter-error.jpg", dpi=110, pil_kwargs={"quality": 88})
plt.close(fig)

# 图4：识别延迟分布
fig, ax = plt.subplots(figsize=(9.5, 4.4))
ax.hist(delays, bins=np.arange(0, 45, 2), color="#a8c6df", edgecolor="white")
ax.axvline(np.median(delays), color="#d64545", ls="--", lw=1.4, label=f"中位数 {np.median(delays):.0f} 天")
ax.axvline(delays.mean(), color="#7a5195", ls="--", lw=1.4, label=f"均值 {delays.mean():.1f} 天")
ax.set_title(f"regime 切换识别延迟分布（{len(switch_idx)} 次切换）：滤波不是免费的千里眼")
ax.set_xlabel("识别延迟（交易日）"); ax.set_ylabel("次数")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/kelly-detection-delay.jpg", dpi=110, pil_kwargs={"quality": 88})
plt.close(fig)

print("figures saved to", OUT)
