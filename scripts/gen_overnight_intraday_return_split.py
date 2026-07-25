#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「隔夜与日内收益拆分：同一只股票的两副面孔」
(overnight-intraday-return-split) 生成真实配图与真实统计数字。

核心角度（区别于站内已有的时间序列拆解文章）：
  横截面上的『拔河』——Lou, Polk & Skouras (2019, JFE) "A Tug of War"。
  同一只股票，隔夜段收益（收盘→次日开盘）呈现动量/延续，
  日内段收益（开盘→收盘）呈现反转；两股力量方向相反，
  分别由散户（隔夜）与机构（日内）的交易时段偏好驱动。

所有图与数字均由文中 Python 逻辑真实计算生成：
  1) split_scatter.png     —— 上期隔夜收益 vs 本期隔夜收益（动量，正斜率）
                              上期隔夜收益 vs 本期日内收益（反转，负斜率）
  2) split_quintile.png    —— 按上期隔夜收益五分组，本期隔夜/日内平均收益条形图
  3) split_equity.png      —— 隔夜动量多空 / 日内反转多空 / 合并策略 三条净值
  4) split_heatmap.png     —— 动量强度 β × 成本 的多空 Sharpe 热力图

仅为机制演示的自洽合成数据，非真实行情。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rc = matplotlib.rcParams
rc["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rc["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "overnight-intraday-return-split")
os.makedirs(D, exist_ok=True)

C = {"over": "#4C72B0", "intra": "#DD8452", "combo": "#2F4B7C",
     "up": "#55A868", "dn": "#C44E52", "grid": "#E2E2E2"}

# =====================================================================
# 合成横截面面板：N 只股票 × T 天，每天拆成隔夜段 + 日内段
# 机制：
#   本期隔夜 =  +PHI * 上期隔夜   (动量，散户情绪延续)   + 噪声
#   本期日内 =  -PSI * 上期隔夜   (反转，机构逆向承接)   + 噪声
# =====================================================================
N_STOCK = 200
T_DAY = 1260            # ≈ 5 年
PHI = 0.055            # 隔夜→隔夜 动量传导
PSI = 0.042            # 隔夜→日内 反转传导
SEED = 20260725


def simulate(n=N_STOCK, t=T_DAY, phi=PHI, psi=PSI, seed=SEED):
    rng = np.random.default_rng(seed)
    # 市场公共因子（当天所有股票共享）
    mkt = rng.standard_normal(t) * 0.010
    # regime：传导强度本身逐日波动（有些日子动量反噬、有些日子加倍）——
    # 这是 factor-level 波动，长短组合无法通过增加标的数来分散掉，
    # 决定了策略真实 Sharpe 的上限，避免出现 10+ 的虚高读数。
    regime = 1.0 + rng.standard_normal(t) * 2.6
    # 长短腿共同因子：每日一个不可分散的频度因子（多空组合难以对冲）
    spread_factor_o = rng.standard_normal(t) * 0.0018
    spread_factor_i = rng.standard_normal(t) * 0.0018
    # 隔夜与日内的特质波动
    ov = np.zeros((t, n))
    idr = np.zeros((t, n))
    prev_ov = rng.standard_normal(n) * 0.012
    for d in range(t):
        eps_o = rng.standard_normal(n) * 0.012
        eps_i = rng.standard_normal(n) * 0.014
        # 按上期隔夜排名施加共同 spread 因子（高排名多、低排名空，无法分散）
        r_prev = np.argsort(np.argsort(prev_ov)) / n - 0.5
        cur_ov = phi * regime[d] * prev_ov + eps_o + 0.4 * mkt[d] + r_prev * spread_factor_o[d]
        cur_idr = -psi * regime[d] * prev_ov + eps_i + 0.6 * mkt[d] - r_prev * spread_factor_i[d]
        ov[d] = cur_ov
        idr[d] = cur_idr
        prev_ov = cur_ov
    return ov, idr


ov, idr = simulate()
# 上期隔夜（滞后一天）
prev_ov = np.vstack([np.full((1, N_STOCK), np.nan), ov[:-1]])

# ---------- 统计：斜率与 t 值（横截面回归逐日平均，Fama-MacBeth 风格）----------
def fama_macbeth(x, y):
    """逐日横截面回归 y~x，返回斜率序列。"""
    slopes = []
    for d in range(1, x.shape[0]):
        xd, yd = x[d], y[d]
        m = np.isfinite(xd) & np.isfinite(yd)
        if m.sum() < 20:
            continue
        xx = xd[m] - xd[m].mean()
        yy = yd[m] - yd[m].mean()
        denom = (xx * xx).sum()
        if denom == 0:
            continue
        slopes.append((xx * yy).sum() / denom)
    return np.array(slopes)

b_mom = fama_macbeth(prev_ov, ov)      # 上期隔夜 -> 本期隔夜
b_rev = fama_macbeth(prev_ov, idr)     # 上期隔夜 -> 本期日内
t_mom = b_mom.mean() / (b_mom.std(ddof=1) / np.sqrt(len(b_mom)))
t_rev = b_rev.mean() / (b_rev.std(ddof=1) / np.sqrt(len(b_rev)))
print(f"隔夜动量斜率 {b_mom.mean():.4f}  t={t_mom:.2f}")
print(f"日内反转斜率 {b_rev.mean():.4f}  t={t_rev:.2f}")

# =====================================================================
# 图 1：双散点（动量 vs 反转）
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 抽一天做散点（取中间某天，样本量足够）
day = 600
x = prev_ov[day] * 100
for ax, yv, title, col, slope in [
    (axes[0], ov[day] * 100, f"上期隔夜 → 本期隔夜（动量，斜率均值 {b_mom.mean():.3f}, t={t_mom:.1f}）", C["over"], b_mom.mean()),
    (axes[1], idr[day] * 100, f"上期隔夜 → 本期日内（反转，斜率均值 {b_rev.mean():.3f}, t={t_rev:.1f}）", C["intra"], -b_rev.mean()),
]:
    ax.scatter(x, yv, s=14, alpha=0.5, color=col, edgecolors="none")
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
    # 用全样本平均斜率画趋势线
    coef = np.polyfit(x[np.isfinite(x)], yv[np.isfinite(yv)], 1)
    ax.plot(xs, np.polyval(coef, xs), color=C["dn"], lw=2)
    ax.axhline(0, color="#999", lw=0.8); ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("上期隔夜收益 (%)"); ax.set_ylabel("本期收益 (%)")
    ax.set_title(title, fontsize=10.5)
    ax.grid(True, color=C["grid"], lw=0.6)
fig.suptitle("同一信号，两副面孔：隔夜延续、日内反转", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(D, "split_scatter.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# =====================================================================
# 图 2：五分组条形（按上期隔夜分组，看本期隔夜 & 本期日内）
# =====================================================================
def quintile_means(signal, target):
    q_ov, q_id = np.zeros(5), np.zeros(5)
    cnt = np.zeros(5)
    for d in range(1, signal.shape[0]):
        s, tv1, tv2 = signal[d], target[0][d], target[1][d]
        m = np.isfinite(s) & np.isfinite(tv1) & np.isfinite(tv2)
        if m.sum() < 25:
            continue
        s2 = s[m]
        ranks = np.argsort(np.argsort(s2))
        grp = (ranks / len(s2) * 5).astype(int).clip(0, 4)
        for g in range(5):
            sel = grp == g
            if sel.any():
                q_ov[g] += tv1[m][sel].mean()
                q_id[g] += tv2[m][sel].mean()
                cnt[g] += 1
    return q_ov / cnt * 100, q_id / cnt * 100

qov, qid = quintile_means(prev_ov, (ov, idr))
fig, ax = plt.subplots(figsize=(9, 5))
xpos = np.arange(5)
w = 0.38
ax.bar(xpos - w/2, qov, w, label="本期隔夜平均收益", color=C["over"])
ax.bar(xpos + w/2, qid, w, label="本期日内平均收益", color=C["intra"])
ax.axhline(0, color="#666", lw=1)
ax.set_xticks(xpos)
ax.set_xticklabels(["Q1\n(最低)", "Q2", "Q3", "Q4", "Q5\n(最高)"])
ax.set_xlabel("按上期隔夜收益分组")
ax.set_ylabel("本期平均收益 (%)")
ax.set_title("五分组检验：隔夜段随分组递增（动量），日内段随分组递减（反转）", fontsize=11.5, fontweight="bold")
ax.legend()
ax.grid(True, axis="y", color=C["grid"], lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(D, "split_quintile.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# =====================================================================
# 图 3：三条多空净值
# 隔夜动量：买上期隔夜高、卖上期隔夜低，持有本期隔夜段
# 日内反转：买上期隔夜低、卖上期隔夜高，持有本期日内段
# 合并：两条腿相加
# =====================================================================
COST = 0.0005  # 单边 5bp（每日换手）
def ls_returns(signal, target, direction):
    """direction=+1 动量, -1 反转。返回每日多空组合收益(扣成本)。"""
    rets = []
    for d in range(1, signal.shape[0]):
        s, tv = signal[d], target[d]
        m = np.isfinite(s) & np.isfinite(tv)
        if m.sum() < 25:
            rets.append(0.0); continue
        s2, t2 = s[m], tv[m]
        ranks = np.argsort(np.argsort(s2))
        top = ranks >= 0.8 * len(s2)
        bot = ranks < 0.2 * len(s2)
        long_leg = t2[top].mean()
        short_leg = t2[bot].mean()
        r = direction * (long_leg - short_leg) - 2 * COST
        rets.append(r)
    return np.array(rets)

r_mom = ls_returns(prev_ov, ov, +1)
r_rev = ls_returns(prev_ov, idr, -1)
r_combo = r_mom + r_rev

def stats(r):
    ann = r.mean() * 252
    sharpe = r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(252)
    eq = np.cumprod(1 + r)
    mdd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min()
    return ann, sharpe, mdd, eq

a_m, s_m, d_m, eq_m = stats(r_mom)
a_r, s_r, d_r, eq_r = stats(r_rev)
a_c, s_c, d_c, eq_c = stats(r_combo)
print(f"隔夜动量  年化{a_m:.1%} Sharpe{s_m:.2f} MDD{d_m:.1%}")
print(f"日内反转  年化{a_r:.1%} Sharpe{s_r:.2f} MDD{d_r:.1%}")
print(f"合并策略  年化{a_c:.1%} Sharpe{s_c:.2f} MDD{d_c:.1%}")

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(eq_m, color=C["over"], lw=1.8, label=f"隔夜动量多空  年化{a_m:.1%} / Sharpe {s_m:.2f}")
ax.plot(eq_r, color=C["intra"], lw=1.8, label=f"日内反转多空  年化{a_r:.1%} / Sharpe {s_r:.2f}")
ax.plot(eq_c, color=C["combo"], lw=2.4, label=f"合并策略  年化{a_c:.1%} / Sharpe {s_c:.2f}")
ax.axhline(1, color="#999", lw=0.8)
ax.set_xlabel("交易日")
ax.set_ylabel("净值（起始=1）")
ax.set_title("两条腿方向相反、却都赚钱：合并后 Sharpe 更高", fontsize=12, fontweight="bold")
ax.legend(loc="upper left")
ax.grid(True, color=C["grid"], lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(D, "split_equity.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# =====================================================================
# 图 4：β(动量强度) × 成本 的合并策略 Sharpe 热力图
# =====================================================================
betas = np.array([0.02, 0.04, 0.06, 0.08, 0.12, 0.16])
costs = np.array([0.0, 0.0002, 0.0005, 0.0010, 0.0020])
grid = np.zeros((len(betas), len(costs)))
for i, b in enumerate(betas):
    ov_b, idr_b = simulate(phi=b, psi=b*0.75, seed=SEED + i)
    prev_b = np.vstack([np.full((1, N_STOCK), np.nan), ov_b[:-1]])
    # 计算无成本多空日收益
    rm, rr = [], []
    for d in range(1, T_DAY):
        for tgt, store, sgn in [(ov_b, rm, +1), (idr_b, rr, -1)]:
            s, tv = prev_b[d], tgt[d]
            mm = np.isfinite(s) & np.isfinite(tv)
            if mm.sum() < 25:
                store.append(0.0); continue
            s2, t2 = s[mm], tv[mm]
            ranks = np.argsort(np.argsort(s2))
            top = ranks >= 0.8 * len(s2); bot = ranks < 0.2 * len(s2)
            store.append(sgn * (t2[top].mean() - t2[bot].mean()))
    rm = np.array(rm); rr = np.array(rr)
    for j, c in enumerate(costs):
        r = rm + rr - 4 * c  # 两条腿各双边一次
        grid[i, j] = r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(252)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", origin="lower",
               vmin=-max(abs(grid.min()), abs(grid.max())),
               vmax=max(abs(grid.min()), abs(grid.max())))
ax.set_xticks(range(len(costs)))
ax.set_xticklabels([f"{c*1e4:.0f}bp" for c in costs])
ax.set_yticks(range(len(betas)))
ax.set_yticklabels([f"{b:.2f}" for b in betas])
ax.set_xlabel("单边交易成本")
ax.set_ylabel("动量传导强度 β")
for i in range(len(betas)):
    for j in range(len(costs)):
        ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                color="black", fontsize=9)
ax.set_title("生死线：合并策略 Sharpe（β × 成本）", fontsize=12, fontweight="bold")
fig.colorbar(im, ax=ax, label="年化 Sharpe")
fig.tight_layout()
fig.savefig(os.path.join(D, "split_heatmap.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

print("\n=== 图表已生成 ===")
print(f"输出目录: {D}")
for f in sorted(os.listdir(D)):
    print(" ", f)

# 供文章引用的关键数字落盘
with open(os.path.join(D, "_stats.txt"), "w") as fh:
    fh.write(f"mom_slope={b_mom.mean():.4f} t={t_mom:.2f}\n")
    fh.write(f"rev_slope={b_rev.mean():.4f} t={t_rev:.2f}\n")
    fh.write(f"mom ann={a_m:.4f} sharpe={s_m:.4f} mdd={d_m:.4f}\n")
    fh.write(f"rev ann={a_r:.4f} sharpe={s_r:.4f} mdd={d_r:.4f}\n")
    fh.write(f"combo ann={a_c:.4f} sharpe={s_c:.4f} mdd={d_c:.4f}\n")
    fh.write(f"quintile_ov={qov.tolist()}\n")
    fh.write(f"quintile_id={qid.tolist()}\n")
