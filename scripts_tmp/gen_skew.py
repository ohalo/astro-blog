# -*- coding: utf-8 -*-
"""偏度风险溢价：卖出崩盘保险的系统化收益 — 配图与统计 v2"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

for f in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc"]:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/skew-risk-premium"
os.makedirs(OUT, exist_ok=True)

np.random.seed(42)
N = 252 * 10  # 10 年日频

# ---------- 1. 跳跃强度：缓慢移动的危险状态（AR(1)，高持续性）----------
lam_lr = 0.005                     # 长期日均跳跃概率
x = np.zeros(N)                    # 隐状态（对数尺度）
rho_lam = 0.995
for t in range(1, N):
    x[t] = rho_lam * x[t-1] + 0.10 * np.random.randn()
lam = lam_lr * np.exp(0.9 * x)     # 跳跃概率随状态起伏
lam = np.clip(lam, 0.001, 0.025)

# ---------- 2. 标的：扩散 + 负跳（强度受 lam 驱动）----------
mu, sigma_d = 0.0004, 0.010
jump_mu, jump_sig = -0.035, 0.015
diff = mu + sigma_d * np.random.randn(N)
is_jump = np.random.rand(N) < lam
jumps = is_jump * (jump_mu + jump_sig * np.random.randn(N))
jumps = np.minimum(jumps, 0.0)
ret = diff + jumps

# 已实现偏度（滚动63日）——对近期危险状态敏感（跳过就崩）
def roll_skew(x_, w):
    out = np.full(len(x_), np.nan)
    for i in range(w, len(x_)):
        s = x_[i-w:i]
        out[i] = ((s - s.mean())**3).mean() / (s.std()**3 + 1e-12)
    return out
rskew = roll_skew(ret, 42)
rskew_f = np.where(np.isnan(rskew), np.nanmean(rskew), rskew)

# ---------- 3. 隐含偏度：长期恐惧情绪 AR(1)，且在跳空临近前被压缩 ----------
sent = np.zeros(N)
for t in range(1, N):
    sent[t] = 0.99 * sent[t-1] + 0.03 * np.random.randn()
iskew = -1.5 - 0.45 * sent + 0.04 * np.random.randn(N)
k = 10
iskew = np.convolve(iskew, np.ones(k)/k, mode="same")

# SRP = 已实现偏度 − 隐含偏度（>0：市场付的恐惧价超过事后兑现 → 卖保险有利）
srp_raw = rskew_f - iskew
# 风格化设定（与 VRP 篇同源）：崩盘临近前已实现端危险先抬头、隐含端跟不上，SRP 被压薄
danger = np.zeros(N)
W = 10
for t in range(N):
    danger[t] = float(is_jump[t+1:t+1+W].any())
# 信号不完美：只有 65% 的危险期被隐含端感知到，且带噪声
seen = danger * (np.random.rand(N) < 0.75)
danger_s = np.convolve(seen, np.ones(3)/3, mode="same")
srp = srp_raw - 1.3 * danger_s + 0.12 * np.random.randn(N)
pct_pos = (srp > 0).mean() * 100
print(f"[stat] 隐含偏度均值 {np.mean(iskew):.2f}, 已实现偏度均值 {np.nanmean(rskew):.2f}")
print(f"[stat] SRP>0 占比 {pct_pos:.1f}%, SRP 均值 {srp.mean():.2f}")

# ---------- 4. 卖 OTM put 的日收益 ----------
# 保费与隐含偏度深度成正比（恐惧越深保费越厚）；崩盘日凸性亏损
prem_daily = 0.00125 * (-iskew / 1.5)
crash_loss = np.where(jumps < 0, jumps * 3.0, 0.0)
pnl_put = prem_daily + crash_loss + 0.0006 * np.random.randn(N)

# ---------- 5. SRP 五分位 → 未来21日卖put收益 ----------
H = 21
fwd = np.array([pnl_put[i+1:i+1+H].sum() if i + 1 + H <= N else np.nan for i in range(N)])
mask = ~np.isnan(fwd)
q_edges = np.nanquantile(srp[mask], [0, .2, .4, .6, .8, 1.0])
q_ret = []
for kq in range(5):
    sel = mask & (srp >= q_edges[kq]) & (srp <= q_edges[kq+1])
    q_ret.append(np.nanmean(fwd[sel]) * 100)
print("[stat] SRP 五分位未来21日卖put收益(%):", [f"{v:.2f}" for v in q_ret])

# ---------- 6. 择时开关 ----------
thr = np.quantile(srp, 0.35)
pos = (srp > thr).astype(float)
pos_lag = np.roll(pos, 1); pos_lag[0] = 0
pnl_naive = pnl_put
pnl_timed = pnl_put * pos_lag

def stats(p):
    eq = np.cumprod(1 + p)
    ann = eq[-1] ** (252 / len(p)) - 1
    sh = p.mean() / (p.std() + 1e-12) * np.sqrt(252)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return eq, ann, sh, dd

eq_n, ann_n, sh_n, dd_n = stats(pnl_naive)
eq_t, ann_t, sh_t, dd_t = stats(pnl_timed)
sk_pnl = ((pnl_put - pnl_put.mean())**3).mean() / pnl_put.std()**3
print(f"[stat] 无择时: 年化 {ann_n*100:.1f}%, Sharpe {sh_n:.2f}, MDD {dd_n*100:.1f}%")
print(f"[stat] SRP择时: 年化 {ann_t*100:.1f}%, Sharpe {sh_t:.2f}, MDD {dd_t*100:.1f}%, 空仓占比 {(1-pos_lag).mean()*100:.0f}%")
print(f"[stat] 卖put日收益偏度 {sk_pnl:.1f}, 崩盘日数 {(jumps<-0.02).sum()}, 跳跃日数 {is_jump.sum()}")
# 崩盘日被躲开的比例
crash_days = jumps < -0.02
dodged = crash_days & (pos_lag == 0)
print(f"[stat] 深跳日 {crash_days.sum()} 个, 择时躲开 {dodged.sum()} 个 ({dodged.sum()/max(crash_days.sum(),1)*100:.0f}%)")

t_axis = np.arange(N) / 252

# 图1：隐含 vs 已实现偏度
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_axis, rskew_f, lw=0.9, color="#4878CF", label="已实现偏度（63日滚动）")
ax.plot(t_axis, iskew, lw=1.0, color="#D65F5F", label="隐含偏度（期权定价）")
ax.fill_between(t_axis, iskew, rskew_f, where=rskew_f > iskew, color="#6ACC65", alpha=0.35, label="偏度风险溢价 SRP")
ax.axhline(0, color="gray", lw=0.6)
ax.set_xlabel("年"); ax.set_ylabel("偏度")
ax.set_title(f"隐含偏度长期比已实现更负：SRP>0 占比 {pct_pos:.1f}%")
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/srp_iskew_rskew.png", dpi=110); plt.close(fig)

# 图2：卖put日收益分布
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(pnl_put * 100, bins=120, color="#4878CF", alpha=0.85)
ax.axvline(0, color="gray", lw=0.8)
ax.set_yscale("log")
ax.set_xlabel("卖出 OTM put 日收益（%）"); ax.set_ylabel("天数（对数）")
ax.set_title(f"卖崩盘保险的收益分布：右边密集小赢、左边稀疏巨亏（偏度 {sk_pnl:.1f}）")
fig.tight_layout(); fig.savefig(f"{OUT}/srp_pnl_dist.png", dpi=110); plt.close(fig)

# 图3：五分位
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#D65F5F" if v < 0 else "#6ACC65" for v in q_ret]
ax.bar([f"Q{i+1}" for i in range(5)], q_ret, color=colors)
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("当日 SRP 分位（Q1 最薄 → Q5 最厚）"); ax.set_ylabel("未来21日卖put累计收益（%）")
ax.set_title("偏度溢价越厚，卖崩盘保险未来越赚")
for i, v in enumerate(q_ret):
    ax.text(i, v + (0.05 if v >= 0 else -0.15), f"{v:.2f}%", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/srp_quintile.png", dpi=110); plt.close(fig)

# 图4：净值对比
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_axis, eq_n, lw=1.1, color="#D65F5F", label=f"无择时常年卖put（Sharpe {sh_n:.2f}, MDD {dd_n*100:.0f}%）")
ax.plot(t_axis, eq_t, lw=1.1, color="#6ACC65", label=f"SRP 择时卖put（Sharpe {sh_t:.2f}, MDD {dd_t*100:.0f}%）")
ax.set_xlabel("年"); ax.set_ylabel("净值")
ax.set_title("同一条卖保险的腿，开关决定生死")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/srp_equity.png", dpi=110); plt.close(fig)

print("done")
