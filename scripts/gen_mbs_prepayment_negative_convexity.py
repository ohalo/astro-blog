#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MBS 提前偿付与负凸性：为什么利率下行时你的债券不涨反跌

受控仿真：30 年 pass-through MBS 现金流引擎 + S 曲线提前偿付模型（含 burnout），
平行移动利率 -300bp ~ +300bp，计算有效久期 / 有效凸性 / 对冲残差。
所有图表由真实计算生成，固定随机种子可复现。
"""
import json
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["figure.facecolor"] = "white"

OUT = "/Users/halo/workspace/astro-blog/public/images/mbs-prepayment-negative-convexity"
os.makedirs(OUT, exist_ok=True)

SEED = 20260806
np.random.seed(SEED)

# ---------- 配色 ----------
C_MBS   = "#dc2626"   # MBS 红
C_TSY   = "#2563eb"   # 国债蓝
C_ZERO  = "#16a34a"   # 零提前偿付 绿
C_CONST = "#f59e0b"   # 恒定 CPR 橙
C_GREY  = "#64748b"

# =============================================================================
# 一、池子参数
# =============================================================================
TERM_M     = 360        # 30 年
WAC        = 0.0600     # 借款人加权平均票息 6.00%
NET_CPN    = 0.0550     # 投资人净票息 5.50%（0.5% 服务费/担保费）
DISC0      = 0.0550     # 基准贴现率 = 净票息 → 基准价格精确等于面值 100
MTG0       = 0.0600     # 基准市场按揭利率 6.00% → 基准 refi incentive = 0
ORIG_BAL   = 100.0

# S 曲线参数（CPR = f(refi incentive)）
CPR_MIN    = 0.04       # 换房/违约/自然周转形成的地板速度 4% CPR
CPR_MAX    = 0.55       # 再融资潮饱和速度 55% CPR
K_STEEP    = 3.0        # S 曲线陡度
X0_INCEN   = 0.70       # 触发阈值：需要约 70bp 的利差才明显启动再融资

# Burnout（燃尽）参数
BURN_BETA  = 2.2        # 燃尽衰减速度
BURN_FLOOR = 0.40       # 燃尽后再融资敏感度残留下限
SEASON_M   = 30         # PSA 式 seasoning 爬坡月数


# =============================================================================
# 二、提前偿付模型
# =============================================================================
def scurve_cpr(incentive_pp, burnout=0.0, k=K_STEEP, x0=X0_INCEN):
    """
    S 曲线提前偿付模型。
    incentive_pp : refi incentive，单位「百分点」= (WAC - 市场按揭利率) * 100
    burnout      : 燃尽程度 ∈ [0,1)，= 1 - 实际余额/计划余额
    返回年化 CPR。
    """
    s = 0.5 + np.arctan(k * (incentive_pp - x0)) / np.pi     # ∈ (0,1)
    cpr_fresh = CPR_MIN + (CPR_MAX - CPR_MIN) * s
    # 燃尽只削弱「再融资超额部分」，不削弱自然周转地板
    mult = BURN_FLOOR + (1.0 - BURN_FLOOR) * np.exp(-BURN_BETA * burnout)
    return CPR_MIN + (cpr_fresh - CPR_MIN) * mult


def cpr_to_smm(cpr):
    """年化 CPR → 月度单月死亡率 SMM"""
    return 1.0 - (1.0 - cpr) ** (1.0 / 12.0)


def psa_cpr(age_m, psa=100.0):
    """PSA 基准曲线：100 PSA = 前 30 个月 0.2%*age 线性爬坡，之后 6% CPR"""
    base = np.where(age_m < 30, 0.002 * age_m, 0.06)
    return base * psa / 100.0


# =============================================================================
# 三、现金流引擎
# =============================================================================
def run_pool(shift_bp, mode="scurve", const_cpr=0.06, k=K_STEEP,
             burnout_on=True, seasoning_on=True):
    """
    生成 30 年 pass-through 的完整月度现金流。

    mode:
      "scurve"  : 利率敏感的 S 曲线提前偿付（含 burnout）
      "zero"    : 安慰剂 —— CPR ≡ 0，纯计划摊销
      "const"   : 对照组 —— CPR ≡ const_cpr，与利率无关
    """
    mtg_rate  = MTG0 + shift_bp / 10000.0
    incentive = (WAC - mtg_rate) * 100.0        # 百分点

    wac_m = WAC / 12.0
    bal       = ORIG_BAL      # 实际余额
    sched_bal = ORIG_BAL      # 无提前偿付的计划余额（用于度量 burnout）

    cf    = np.zeros(TERM_M)
    prin  = np.zeros(TERM_M)
    intr  = np.zeros(TERM_M)
    prepay= np.zeros(TERM_M)
    cprs  = np.zeros(TERM_M)
    bals  = np.zeros(TERM_M)

    for i in range(TERM_M):
        if bal <= 1e-10:
            break
        age = i + 1
        rem = TERM_M - i                        # 剩余期数
        # 沉基金公式：level-pay 抵押贷款当期计划本金占比
        factor = wac_m / ((1.0 + wac_m) ** rem - 1.0)

        sched_prin_actual = min(bal * factor, bal)
        sched_prin_track  = min(sched_bal * factor, sched_bal)

        # --- CPR ---
        if mode == "zero":
            cpr = 0.0
        elif mode == "const":
            cpr = const_cpr
        else:
            burn = max(0.0, 1.0 - bal / sched_bal) if (burnout_on and sched_bal > 0) else 0.0
            cpr = scurve_cpr(incentive, burnout=burn, k=k)
            if seasoning_on:
                cpr *= min(1.0, age / SEASON_M)

        smm = cpr_to_smm(cpr)
        pp  = (bal - sched_prin_actual) * smm

        interest_inv = bal * NET_CPN / 12.0     # 投资人拿净票息
        p_tot = sched_prin_actual + pp

        cf[i]     = interest_inv + p_tot
        prin[i]   = p_tot
        intr[i]   = interest_inv
        prepay[i] = pp
        cprs[i]   = cpr
        bals[i]   = bal

        bal       -= p_tot
        sched_bal -= sched_prin_track

    return dict(cf=cf, prin=prin, intr=intr, prepay=prepay, cpr=cprs,
                bal=bals, incentive=incentive, mtg_rate=mtg_rate)


def price_pool(shift_bp, **kw):
    """按 (DISC0 + shift) 平坦曲线贴现该情景下的现金流"""
    pool = run_pool(shift_bp, **kw)
    y_m  = (DISC0 + shift_bp / 10000.0) / 12.0
    t    = np.arange(1, TERM_M + 1)
    df   = (1.0 + y_m) ** (-t)
    return float(np.sum(pool["cf"] * df))


def wal(shift_bp, **kw):
    """加权平均寿命（年）"""
    pool = run_pool(shift_bp, **kw)
    p = pool["prin"]
    t = np.arange(1, TERM_M + 1)
    return float(np.sum(t * p) / np.sum(p) / 12.0)


def eff_dur_conv(shift_bp, h=25.0, **kw):
    """有效久期 / 有效凸性：按揭利率与贴现率同步 ±h bp 平行冲击"""
    p0  = price_pool(shift_bp,     **kw)
    pu  = price_pool(shift_bp + h, **kw)
    pd  = price_pool(shift_bp - h, **kw)
    dy  = h / 10000.0
    dur = (pd - pu) / (2.0 * p0 * dy)
    cvx = (pd + pu - 2.0 * p0) / (p0 * dy * dy)
    return dur, cvx, p0


# =============================================================================
# 四、久期匹配的国债（bullet，无提前偿付期权）
# =============================================================================
def tsy_price(shift_bp, mat_yr, cpn=NET_CPN):
    n  = int(round(mat_yr * 12))
    y_m = (DISC0 + shift_bp / 10000.0) / 12.0
    t  = np.arange(1, n + 1)
    cf = np.full(n, 100.0 * cpn / 12.0)
    cf[-1] += 100.0
    return float(np.sum(cf * (1.0 + y_m) ** (-t)))


def tsy_dur_conv(shift_bp, mat_yr, h=25.0):
    p0 = tsy_price(shift_bp, mat_yr)
    pu = tsy_price(shift_bp + h, mat_yr)
    pd = tsy_price(shift_bp - h, mat_yr)
    dy = h / 10000.0
    return ((pd - pu) / (2 * p0 * dy),
            (pd + pu - 2 * p0) / (p0 * dy * dy),
            p0)


# =============================================================================
# 五、主计算
# =============================================================================
print("=" * 70)
print("MBS 提前偿付与负凸性 —— 受控仿真")
print("=" * 70)

# ---- 基准点 ----
base_price = price_pool(0)
base_dur, base_cvx, _ = eff_dur_conv(0)
base_wal = wal(0)
base_cpr_0 = scurve_cpr(0.0)

print(f"\n[基准] 价格 = {base_price:.4f}（理论应精确 = 100，因贴现率 = 净票息）")
print(f"[基准] 有效久期 = {base_dur:.3f} 年，有效凸性 = {base_cvx:.2f}，WAL = {base_wal:.2f} 年")
print(f"[基准] 起始 CPR（未 seasoning）= {base_cpr_0*100:.2f}%")

# ---- 求久期匹配的国债期限 ----
mat_star = brentq(lambda T: tsy_dur_conv(0, T)[0] - base_dur, 0.5, 30.0, xtol=1e-8)
t_dur0, t_cvx0, t_p0 = tsy_dur_conv(0, mat_star)
print(f"[国债] 久期匹配期限 = {mat_star:.3f} 年，久期 = {t_dur0:.3f}，凸性 = {t_cvx0:.2f}，价格 = {t_p0:.4f}")

# ---- 情景扫描 ----
SHIFTS = np.arange(-300, 301, 25).astype(float)

mbs_px, mbs_dur, mbs_cvx, mbs_wal, mbs_cpr1 = [], [], [], [], []
zero_px, zero_cvx, zero_dur = [], [], []
const_px, const_cvx, const_dur = [], [], []
tsy_px, tsy_dur, tsy_cvx = [], [], []

for s in SHIFTS:
    d, c, p = eff_dur_conv(s)
    mbs_px.append(p); mbs_dur.append(d); mbs_cvx.append(c)
    mbs_wal.append(wal(s))
    pool = run_pool(s)
    mbs_cpr1.append(float(np.mean(pool["cpr"][:12])))   # 首年平均 CPR

    dz, cz, pz = eff_dur_conv(s, mode="zero")
    zero_px.append(pz); zero_cvx.append(cz); zero_dur.append(dz)

    dc, cc, pc = eff_dur_conv(s, mode="const", const_cpr=0.10)
    const_px.append(pc); const_cvx.append(cc); const_dur.append(dc)

    dt, ct, pt = tsy_dur_conv(s, mat_star)
    tsy_px.append(pt); tsy_dur.append(dt); tsy_cvx.append(ct)

mbs_px   = np.array(mbs_px);   mbs_dur = np.array(mbs_dur);   mbs_cvx = np.array(mbs_cvx)
mbs_wal  = np.array(mbs_wal);  mbs_cpr1 = np.array(mbs_cpr1)
zero_px  = np.array(zero_px);  zero_cvx = np.array(zero_cvx); zero_dur = np.array(zero_dur)
const_px = np.array(const_px); const_cvx = np.array(const_cvx); const_dur = np.array(const_dur)
tsy_px   = np.array(tsy_px);   tsy_dur = np.array(tsy_dur);   tsy_cvx = np.array(tsy_cvx)

i0    = int(np.where(SHIFTS == 0)[0][0])
i_m300= int(np.where(SHIFTS == -300)[0][0])
i_m100= int(np.where(SHIFTS == -100)[0][0])
i_p100= int(np.where(SHIFTS == 100)[0][0])
i_p300= int(np.where(SHIFTS == 300)[0][0])

mbs_ret = mbs_px / mbs_px[i0] - 1.0
tsy_ret = tsy_px / tsy_px[i0] - 1.0

capture_m300 = mbs_ret[i_m300] / tsy_ret[i_m300]
capture_m100 = mbs_ret[i_m100] / tsy_ret[i_m100]
loss_p300    = mbs_ret[i_p300] / tsy_ret[i_p300]

print(f"\n[-300bp] MBS {mbs_ret[i_m300]*100:+.2f}%  国债 {tsy_ret[i_m300]*100:+.2f}%  "
      f"上行捕获率 = {capture_m300*100:.1f}%")
print(f"[-100bp] MBS {mbs_ret[i_m100]*100:+.2f}%  国债 {tsy_ret[i_m100]*100:+.2f}%  "
      f"上行捕获率 = {capture_m100*100:.1f}%")
print(f"[+100bp] MBS {mbs_ret[i_p100]*100:+.2f}%  国债 {tsy_ret[i_p100]*100:+.2f}%")
print(f"[+300bp] MBS {mbs_ret[i_p300]*100:+.2f}%  国债 {tsy_ret[i_p300]*100:+.2f}%  "
      f"下行放大倍数 = {loss_p300:.2f}x")

neg_mask = mbs_cvx < 0
neg_range = (SHIFTS[neg_mask].min(), SHIFTS[neg_mask].max()) if neg_mask.any() else (np.nan, np.nan)
min_cvx_i = int(np.argmin(mbs_cvx))
print(f"\n[凸性] MBS 基准凸性 = {mbs_cvx[i0]:.2f}，最小值 = {mbs_cvx[min_cvx_i]:.2f} "
      f"@ {SHIFTS[min_cvx_i]:+.0f}bp")
print(f"[凸性] 负凸性区间 = [{neg_range[0]:+.0f}bp, {neg_range[1]:+.0f}bp]，"
      f"占扫描格点 {neg_mask.sum()}/{len(SHIFTS)}")
print(f"[凸性] 国债基准凸性 = {tsy_cvx[i0]:.2f}（同久期，恒正）")

# ---- 久期反转 ----
dur_max_i = int(np.argmax(mbs_dur))
print(f"[久期] MBS 久期峰值 = {mbs_dur[dur_max_i]:.2f} 年 @ {SHIFTS[dur_max_i]:+.0f}bp；"
      f"-300bp 时压缩到 {mbs_dur[i_m300]:.2f} 年（缩短 "
      f"{(1-mbs_dur[i_m300]/mbs_dur[dur_max_i])*100:.1f}%）")
print(f"[WAL ] -300bp: {mbs_wal[i_m300]:.2f}y | 0bp: {mbs_wal[i0]:.2f}y | "
      f"+300bp: {mbs_wal[i_p300]:.2f}y")

# =============================================================================
# 六、对抗式检验 / 安慰剂
# =============================================================================
print("\n" + "=" * 70)
print("对抗式检验")
print("=" * 70)

# --- 安慰剂 1：关掉提前偿付 → 凸性应精确转正 ---
zero_min_cvx = float(np.min(zero_cvx))
print(f"[安慰剂 1｜零提前偿付] 凸性最小值 = {zero_min_cvx:.2f}，"
      f"全区间为正: {bool(np.all(zero_cvx > 0))}")

# --- 对照 2：恒定 CPR=10%（有提前偿付，但与利率无关）---
const_min_cvx = float(np.min(const_cvx))
print(f"[对照 2｜恒定 CPR=10%] 凸性最小值 = {const_min_cvx:.2f}，"
      f"全区间为正: {bool(np.all(const_cvx > 0))}  ← 说明负凸性来自「期权」不是「速度」")

# --- 检验 3：S 曲线陡度扫描 ---
K_SCAN = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
k_min_cvx, k_base_cvx, k_cap = [], [], []
for kk in K_SCAN:
    cvs, pxs = [], []
    for s in SHIFTS:
        d, c, p = eff_dur_conv(s, mode="scurve", k=kk)
        cvs.append(c); pxs.append(p)
    cvs = np.array(cvs); pxs = np.array(pxs)
    k_min_cvx.append(float(cvs.min()))
    k_base_cvx.append(float(cvs[i0]))
    r = pxs / pxs[i0] - 1.0
    k_cap.append(float(r[i_m300] / tsy_ret[i_m300]))
    print(f"[敏感性 3｜k={kk:>4.2f}] 基准凸性={cvs[i0]:>8.2f}  最小凸性={cvs.min():>8.2f}  "
          f"-300bp 捕获率={r[i_m300]/tsy_ret[i_m300]*100:>6.1f}%")

# --- 检验 4：久期对冲有效性（DV01 中性）---
hedge_ratio = (base_dur * base_price) / (t_dur0 * t_p0)     # 每 1 单位 MBS 对应的国债空头名义
hedge_pnl   = (mbs_px - mbs_px[i0]) - hedge_ratio * (tsy_px - tsy_px[i0])
# 二阶理论预测：-0.5*(C_mbs - C_tsy)*Δy²*P
dy_arr   = SHIFTS / 10000.0
pred_pnl = 0.5 * (mbs_cvx[i0] - t_cvx0) * dy_arr ** 2 * base_price
print(f"\n[对冲 4] DV01 中性对冲比率 = {hedge_ratio:.4f}")
print(f"[对冲 4] 残差 P&L: -300bp={hedge_pnl[i_m300]:+.3f}  -100bp={hedge_pnl[i_m100]:+.3f}  "
      f"+100bp={hedge_pnl[i_p100]:+.3f}  +300bp={hedge_pnl[i_p300]:+.3f}（面值 100）")
print(f"[对冲 4] 残差全为负: {bool(np.all(hedge_pnl[SHIFTS != 0] < 0))}  "
      f"→ 双边亏损 = 空 gamma 特征")
print(f"[对冲 4] 二阶理论预测 ±100bp = {pred_pnl[i_p100]:+.3f}，实测 = {hedge_pnl[i_p100]:+.3f}")

# --- 检验 4b：局部凸性 vs 路径平均凸性 ---
# P/P0 - 1 = -D0*Δy + 0.5*C_path*Δy²  →  反解出「这段位移实际兑现的平均凸性」
with np.errstate(divide="ignore", invalid="ignore"):
    c_path = 2.0 * (mbs_ret + base_dur * dy_arr) / (dy_arr ** 2)
c_path[i0] = np.nan
c_path_m300 = float(c_path[i_m300])
c_path_p300 = float(c_path[i_p300])
print(f"[对冲 4b] 局部有效凸性谷底 = {mbs_cvx.min():.1f}（仅在 {SHIFTS[min_cvx_i]:+.0f}bp 附近成立）")
print(f"[对冲 4b] -300bp 整段真正兑现的路径平均凸性 = {c_path_m300:.1f}")
print(f"[对冲 4b] +300bp 整段真正兑现的路径平均凸性 = {c_path_p300:.1f}")

# --- 检验 5：差分步长收敛性 ---
H_SCAN = [5.0, 10.0, 25.0, 50.0, 100.0]
h_dur, h_cvx = [], []
for hh in H_SCAN:
    d, c, _ = eff_dur_conv(0, h=hh)
    h_dur.append(float(d)); h_cvx.append(float(c))
    print(f"[数值 5｜h={hh:>5.1f}bp] 久期={d:.4f}  凸性={c:>9.2f}")
cvx_spread = max(h_cvx) - min(h_cvx)
print(f"[数值 5] 凸性在 h∈[5,100]bp 间的极差 = {cvx_spread:.2f}"
      f"（负凸性结论对步长稳健: {bool(max(h_cvx) < 0)}）")

# --- 检验 6：Burnout 开关 ---
nb_px, nb_cvx = [], []
for s in SHIFTS:
    d, c, p = eff_dur_conv(s, burnout_on=False)
    nb_px.append(p); nb_cvx.append(c)
nb_px = np.array(nb_px); nb_cvx = np.array(nb_cvx)
nb_ret = nb_px / nb_px[i0] - 1.0
print(f"\n[Burnout 6] 关掉燃尽后：最小凸性 {nb_cvx.min():.2f}（含燃尽 {mbs_cvx.min():.2f}），"
      f"-300bp 捕获率 {nb_ret[i_m300]/tsy_ret[i_m300]*100:.1f}%"
      f"（含燃尽 {capture_m300*100:.1f}%）")

# =============================================================================
# 七、绘图
# =============================================================================
XT = np.arange(-300, 301, 100)

# ---------- 图 1：cover —— 价格曲线对比 ----------
fig, ax = plt.subplots(figsize=(11, 6.2))
ax.plot(SHIFTS, tsy_ret * 100, "-o", color=C_TSY, lw=2.6, ms=4.5,
        label=f"久期匹配国债（{mat_star:.1f}年 bullet，凸性 {t_cvx0:+.0f}）")
ax.plot(SHIFTS, mbs_ret * 100, "-o", color=C_MBS, lw=2.8, ms=4.5,
        label=f"30年 MBS pass-through（凸性 {mbs_cvx[i0]:+.0f}）")
ax.fill_between(SHIFTS, mbs_ret * 100, tsy_ret * 100,
                where=(SHIFTS <= 0), color=C_MBS, alpha=0.13,
                label="被提前偿付「压扁」的上行空间")
ax.fill_between(SHIFTS, mbs_ret * 100, tsy_ret * 100,
                where=(SHIFTS >= 0), color="#7c3aed", alpha=0.13,
                label="被久期拉长「放大」的下行损失")
ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
ax.annotate(f"-300bp\nMBS {mbs_ret[i_m300]*100:+.1f}%  vs  国债 {tsy_ret[i_m300]*100:+.1f}%\n"
            f"只吃到 {capture_m300*100:.0f}% 的涨幅",
            xy=(-300, mbs_ret[i_m300] * 100), xytext=(-292, -13.5),
            fontsize=10.5, color=C_MBS, weight="bold",
            arrowprops=dict(arrowstyle="->", color=C_MBS, lw=1.4))
ax.annotate(f"+300bp\nMBS {mbs_ret[i_p300]*100:+.1f}%  vs  国债 {tsy_ret[i_p300]*100:+.1f}%\n"
            f"多亏 {abs(mbs_ret[i_p300]-tsy_ret[i_p300])*100:.1f} 个百分点",
            xy=(300, mbs_ret[i_p300] * 100), xytext=(55, 6.5),
            fontsize=10.5, color="#7c3aed", weight="bold",
            arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1.4))
ax.margins(y=0.10)
ax.set_xlabel("利率平行移动（bp）", fontsize=12)
ax.set_ylabel("价格变动（%）", fontsize=12)
ax.set_title("负凸性的真面目：同样久期，MBS 涨不动、跌得狠", fontsize=15, weight="bold", pad=14)
ax.set_xticks(XT)
ax.legend(loc="upper right", fontsize=10, framealpha=0.94)
plt.tight_layout()
plt.savefig(f"{OUT}/cover.png")
plt.close()
print("\n✓ cover.png")

# ---------- 图 2：S 曲线 ----------
inc = np.linspace(-3.0, 3.5, 400)
cpr_fresh = scurve_cpr(inc, burnout=0.0)
cpr_b30   = scurve_cpr(inc, burnout=0.30)
cpr_b60   = scurve_cpr(inc, burnout=0.60)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
ax = axes[0]
ax.plot(inc, cpr_fresh * 100, color=C_MBS, lw=2.8, label="新池子（burnout=0）")
ax.plot(inc, cpr_b30 * 100, color=C_CONST, lw=2.4, ls="--", label="中度燃尽（burnout=0.30）")
ax.plot(inc, cpr_b60 * 100, color=C_ZERO, lw=2.4, ls=":", label="深度燃尽（burnout=0.60）")
ax.axhline(CPR_MIN * 100, color=C_GREY, lw=1.1, ls="-.", label=f"自然周转地板 {CPR_MIN*100:.0f}% CPR")
ax.axvline(X0_INCEN, color=C_GREY, lw=1.0, ls="--")
ax.text(X0_INCEN + 0.06, 50, f"触发阈值\n{X0_INCEN*100:.0f}bp", fontsize=9.5, color=C_GREY)
ax.scatter([0], [scurve_cpr(0.0) * 100], s=90, color="k", zorder=5)
ax.annotate(f"基准点 CPR={scurve_cpr(0.0)*100:.1f}%", xy=(0, scurve_cpr(0.0) * 100),
            xytext=(-2.85, 24), fontsize=10,
            arrowprops=dict(arrowstyle="->", color="k", lw=1.1))
ax.set_xlabel("再融资激励 refi incentive = WAC − 市场按揭利率（百分点）", fontsize=11)
ax.set_ylabel("年化 CPR（%）", fontsize=11)
ax.set_title("S 曲线：提前偿付速度对利率高度非线性", fontsize=13, weight="bold")
ax.legend(fontsize=9.5, loc="upper left")

ax = axes[1]
for s, col, ls in [(-300, C_MBS, "-"), (-150, C_CONST, "-"), (0, "k", "-"),
                   (150, "#7c3aed", "--"), (300, C_TSY, "--")]:
    pool = run_pool(s)
    m = np.arange(1, TERM_M + 1) / 12.0
    ax.plot(m, pool["cpr"] * 100, color=col, ls=ls, lw=2.2,
            label=f"{s:+d}bp（首年均值 {np.mean(pool['cpr'][:12])*100:.1f}%）")
ax.plot(np.arange(1, TERM_M + 1) / 12.0, psa_cpr(np.arange(1, TERM_M + 1)) * 100,
        color=C_GREY, lw=1.6, ls="-.", label="100 PSA 基准")
ax.set_xlabel("池龄（年）", fontsize=11)
ax.set_ylabel("年化 CPR（%）", fontsize=11)
ax.set_title("实际路径：seasoning 爬坡 → 峰值 → burnout 衰减", fontsize=13, weight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(0, 30)
plt.tight_layout()
plt.savefig(f"{OUT}/scurve_cpr.png")
plt.close()
print("✓ scurve_cpr.png")

# ---------- 图 3：有效凸性 ----------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
ax = axes[0]
ax.plot(SHIFTS, mbs_cvx, "-o", color=C_MBS, lw=2.8, ms=4.5, label="MBS（S 曲线提前偿付）")
ax.plot(SHIFTS, zero_cvx, "-s", color=C_ZERO, lw=2.2, ms=4,
        label=f"安慰剂：CPR≡0（最小凸性 {zero_min_cvx:+.0f}）")
ax.plot(SHIFTS, const_cvx, "-^", color=C_CONST, lw=2.2, ms=4,
        label=f"对照：CPR≡10% 恒定（最小凸性 {const_min_cvx:+.0f}）")
ax.plot(SHIFTS, tsy_cvx, "--", color=C_TSY, lw=2.0, label=f"久期匹配国债（{t_cvx0:+.0f}）")
ax.axhline(0, color="k", lw=1.2)
ax.fill_between(SHIFTS, mbs_cvx, 0, where=(mbs_cvx < 0), color=C_MBS, alpha=0.16)
ax.axvspan(neg_range[0], neg_range[1], color=C_MBS, alpha=0.05)
ax.annotate(f"负凸性区间 [{neg_range[0]:+.0f}bp, {neg_range[1]:+.0f}bp]\n谷底 {mbs_cvx[min_cvx_i]:.0f} @ {SHIFTS[min_cvx_i]:+.0f}bp",
            xy=(SHIFTS[min_cvx_i], mbs_cvx[min_cvx_i]),
            xytext=(75, mbs_cvx[min_cvx_i] * 0.80), fontsize=10.5, color=C_MBS, weight="bold",
            arrowprops=dict(arrowstyle="->", color=C_MBS, lw=1.3))
ax.set_xlabel("利率平行移动（bp）", fontsize=11)
ax.set_ylabel("有效凸性", fontsize=11)
ax.set_title("负凸性 100% 来自「利率敏感的」提前偿付期权", fontsize=13, weight="bold")
ax.set_xticks(XT)
ax.legend(fontsize=9, loc="lower left", framealpha=0.94)

ax = axes[1]
ax.plot(SHIFTS, mbs_dur, "-o", color=C_MBS, lw=2.8, ms=4.5, label="MBS 有效久期")
ax.plot(SHIFTS, zero_dur, "-s", color=C_ZERO, lw=2.0, ms=3.5, label="CPR≡0 久期")
ax.plot(SHIFTS, np.full_like(SHIFTS, base_dur), "--", color=C_TSY, lw=2.0,
        label=f"对冲用的静态久期假设 {base_dur:.2f}y")
ax.scatter([SHIFTS[dur_max_i]], [mbs_dur[dur_max_i]], s=90, color="k", zorder=5)
ax.annotate(f"久期峰值 {mbs_dur[dur_max_i]:.2f}y\n@ {SHIFTS[dur_max_i]:+.0f}bp",
            xy=(SHIFTS[dur_max_i], mbs_dur[dur_max_i]), xytext=(120, mbs_dur[dur_max_i] - 2.2),
            fontsize=10, arrowprops=dict(arrowstyle="->", color="k", lw=1.1))
ax.annotate(f"-300bp 久期塌到 {mbs_dur[i_m300]:.2f}y",
            xy=(-300, mbs_dur[i_m300]), xytext=(-292, mbs_dur[i_m300] + 3.2),
            fontsize=10, color=C_MBS,
            arrowprops=dict(arrowstyle="->", color=C_MBS, lw=1.1))
ax.set_xlabel("利率平行移动（bp）", fontsize=11)
ax.set_ylabel("有效久期（年）", fontsize=11)
ax.set_title("久期反转：利率越跌久期越短，你的对冲比率一直在错", fontsize=13, weight="bold")
ax.set_xticks(XT)
ax.margins(y=0.12)
ax.legend(fontsize=9.5, loc="upper right", framealpha=0.94)
plt.tight_layout()
plt.savefig(f"{OUT}/effective_convexity.png")
plt.close()
print("✓ effective_convexity.png")

# ---------- 图 4：现金流剖面 ----------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
ax = axes[0]
scen = [(-300, C_MBS), (-150, C_CONST), (0, "k"), (150, "#7c3aed"), (300, C_TSY)]
years = np.arange(1, 31)
w = 0.16
for j, (s, col) in enumerate(scen):
    pool = run_pool(s)
    ann = pool["prin"].reshape(30, 12).sum(axis=1)
    ax.bar(years + (j - 2) * w, ann, width=w, color=col, alpha=0.85,
           label=f"{s:+d}bp（WAL {wal(s):.2f}y）")
ax.set_xlabel("年份", fontsize=11)
ax.set_ylabel("当年本金回收（占原始面值 100）", fontsize=11)
ax.set_title("现金流剖面：收缩风险 vs 展期风险", fontsize=13, weight="bold")
ax.set_xlim(0.3, 30.7)
ax.legend(fontsize=9.5)

ax = axes[1]
ax.plot(SHIFTS, mbs_wal, "-o", color=C_MBS, lw=2.8, ms=4.5, label="MBS 加权平均寿命 WAL")
ax.axhline(mbs_wal[i0], color=C_GREY, ls="--", lw=1.4, label=f"基准 WAL {mbs_wal[i0]:.2f}y")
ax.fill_between(SHIFTS, mbs_wal, mbs_wal[i0], where=(SHIFTS < 0),
                color=C_MBS, alpha=0.14, label="收缩风险（钱提前回来，只能低息再投）")
ax.fill_between(SHIFTS, mbs_wal, mbs_wal[i0], where=(SHIFTS > 0),
                color="#7c3aed", alpha=0.14, label="展期风险（钱被锁死，错过高息）")
ax.annotate(f"{mbs_wal[i_m300]:.2f}y", xy=(-300, mbs_wal[i_m300]), xytext=(-290, mbs_wal[i_m300] + 0.9),
            fontsize=11, color=C_MBS, weight="bold")
ax.annotate(f"{mbs_wal[i_p300]:.2f}y", xy=(300, mbs_wal[i_p300]), xytext=(238, mbs_wal[i_p300] - 1.2),
            fontsize=11, color="#7c3aed", weight="bold")
ax.set_xlabel("利率平行移动（bp）", fontsize=11)
ax.set_ylabel("WAL（年）", fontsize=11)
ax.set_title(f"WAL 从 {mbs_wal[i_m300]:.2f}y 拉伸到 {mbs_wal[i_p300]:.2f}y（{mbs_wal[i_p300]/mbs_wal[i_m300]:.1f}倍）",
             fontsize=13, weight="bold")
ax.set_xticks(XT)
ax.legend(fontsize=9.5, loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT}/cashflow_profile.png")
plt.close()
print("✓ cashflow_profile.png")

# ---------- 图 5：对冲残差 ----------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
ax = axes[0]
cols = [C_MBS if v < 0 else C_ZERO for v in hedge_pnl]
ax.bar(SHIFTS, hedge_pnl, width=18, color=cols, alpha=0.85, label="实测对冲残差 P&L")
ax.plot(SHIFTS, pred_pnl, "--", color="k", lw=2.0,
        label="二阶理论预测 ½·(C_MBS − C_国债)·Δy²·P")
ax.axhline(0, color="k", lw=1.0)
for ii, xx in [(i_m300, -300), (i_m100, -100), (i_p100, 100), (i_p300, 300)]:
    ax.annotate(f"{hedge_pnl[ii]:.2f}", xy=(xx, hedge_pnl[ii]), xytext=(xx, 1.15),
                fontsize=10, color=C_MBS, weight="bold", ha="center")
ax.set_ylim(min(pred_pnl.min(), hedge_pnl.min()) * 1.15, 3.6)
ax.set_xlabel("利率平行移动（bp）", fontsize=11)
ax.set_ylabel("残差 P&L（每 100 面值）", fontsize=11)
ax.set_title("DV01 中性对冲：两个方向都亏 —— 这就是空 gamma", fontsize=13, weight="bold")
ax.set_xticks(XT)
ax.legend(fontsize=9.5, loc="lower center")

ax = axes[1]
mask = SHIFTS != 0
r2 = 1 - np.sum((hedge_pnl[mask] - pred_pnl[mask]) ** 2) / \
     np.sum((hedge_pnl[mask] - hedge_pnl[mask].mean()) ** 2)
small = mask & (np.abs(SHIFTS) <= 100)
r2_small = 1 - np.sum((hedge_pnl[small] - pred_pnl[small]) ** 2) / \
           np.sum((hedge_pnl[small] - hedge_pnl[small].mean()) ** 2)
ax.plot(SHIFTS, mbs_cvx, "-o", color=C_MBS, lw=2.6, ms=4,
        label=f"局部有效凸性（±25bp 冲击，谷底 {mbs_cvx.min():.0f}）")
ax.plot(SHIFTS[mask], c_path[mask], "-s", color="#7c3aed", lw=2.4, ms=4,
        label="路径平均凸性（该段位移真正兑现的）")
ax.axhline(0, color="k", lw=1.2)
ax.axhline(t_cvx0, color=C_TSY, ls="--", lw=1.6, label=f"国债凸性 {t_cvx0:+.0f}（恒正、稳定）")
ax.annotate(f"-300bp 整段只兑现 {c_path_m300:.0f}", xy=(-300, c_path_m300),
            xytext=(-285, 330), fontsize=10,
            color="#7c3aed", weight="bold",
            arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1.2))
ax.annotate(f"局部谷底 {mbs_cvx.min():.0f}", xy=(SHIFTS[min_cvx_i], mbs_cvx.min()),
            xytext=(105, mbs_cvx.min() * 0.80), fontsize=10, color=C_MBS, weight="bold",
            arrowprops=dict(arrowstyle="->", color=C_MBS, lw=1.2))
ax.set_xlabel("利率平行移动（bp）", fontsize=11)
ax.set_ylabel("凸性", fontsize=11)
ax.set_title(f"别信单点凸性：二阶泰勒全区间 R²={r2:.2f}，±100bp 内 R²={r2_small:.2f}",
             fontsize=12.5, weight="bold")
ax.set_xticks(XT)
ax.set_ylim(float(np.nanmin(np.r_[mbs_cvx, c_path])) * 1.16,
            float(np.nanmax(np.r_[mbs_cvx, c_path])) * 1.75)
ax.legend(fontsize=9, loc="upper right", framealpha=0.94)
plt.tight_layout()
plt.savefig(f"{OUT}/hedge_error.png")
plt.close()
print("✓ hedge_error.png")

# ---------- 图 6：安慰剂 / 敏感性 ----------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
ax = axes[0]
ax.plot(SHIFTS, mbs_ret * 100, "-o", color=C_MBS, lw=2.6, ms=4, label="MBS（S 曲线）")
ax.plot(SHIFTS, (zero_px / zero_px[i0] - 1) * 100, "-s", color=C_ZERO, lw=2.4, ms=4,
        label="安慰剂：CPR≡0（纯摊销）")
ax.plot(SHIFTS, (const_px / const_px[i0] - 1) * 100, "-^", color=C_CONST, lw=2.4, ms=4,
        label="对照：CPR≡10% 恒定")
ax.plot(SHIFTS, tsy_ret * 100, "--", color=C_TSY, lw=2.0, label="久期匹配国债")
ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("利率平行移动（bp）", fontsize=11)
ax.set_ylabel("价格变动（%）", fontsize=11)
ax.set_title("安慰剂：只要 CPR 不随利率变，曲线就重新「向上凸」", fontsize=13, weight="bold")
ax.set_xticks(XT)
ax.legend(fontsize=9.5, loc="upper right")

ax = axes[1]
ax.plot(K_SCAN, k_min_cvx, "-o", color=C_MBS, lw=2.8, ms=7, label="最小有效凸性")
ax.plot(K_SCAN, k_base_cvx, "-s", color=C_CONST, lw=2.4, ms=6, label="基准点（0bp）有效凸性")
ax.axhline(0, color="k", lw=1.2)
ax.axvline(K_STEEP, color=C_GREY, ls="--", lw=1.4)
ax.text(K_STEEP + 0.12, k_min_cvx[0] * 0.55, f"本文基准 k={K_STEEP}", fontsize=10, color=C_GREY)
ax.annotate(f"k=0（CPR 完全不敏感）\n凸性 {k_min_cvx[0]:+.0f} → 正",
            xy=(0, k_min_cvx[0]), xytext=(0.55, k_min_cvx[0] * 0.30 + abs(k_min_cvx[-1]) * 0.10),
            fontsize=10, color=C_ZERO, weight="bold",
            arrowprops=dict(arrowstyle="->", color=C_ZERO, lw=1.3))
ax2 = ax.twinx()
ax2.plot(K_SCAN, np.array(k_cap) * 100, ":", color=C_TSY, lw=2.4, marker="d", ms=6,
         label="-300bp 上行捕获率（右轴）")
ax2.set_ylabel("-300bp 上行捕获率（%）", fontsize=11, color=C_TSY)
ax2.tick_params(axis="y", labelcolor=C_TSY)
ax2.grid(False)
ax.set_xlabel("S 曲线陡度参数 k（越大 = 借款人再融资越敏捷）", fontsize=11)
ax.set_ylabel("有效凸性", fontsize=11)
ax.set_title("参数敏感性：负凸性强度由借款人的「行权效率」决定", fontsize=13, weight="bold")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=9.5, loc="center right")
plt.tight_layout()
plt.savefig(f"{OUT}/placebo_sensitivity.png")
plt.close()
print("✓ placebo_sensitivity.png")

# ---------- 图 7：数值稳健性 ----------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
ax = axes[0]
ax.plot(H_SCAN, h_cvx, "-o", color=C_MBS, lw=2.6, ms=7)
ax.axhline(0, color="k", lw=1.2)
_sgn = [-1, 1, 1, -1, 1]     # h=5 标在下方、h=10 标在上方，避免两个标签重叠
for hh, cc, sg in zip(H_SCAN, h_cvx, _sgn):
    ax.annotate(f"{cc:.0f}", xy=(hh, cc),
                xytext=(hh, cc + sg * abs(min(h_cvx)) * 0.085),
                fontsize=10, ha="center", va="center", color=C_MBS)
ax.margins(y=0.20)
ax.set_xlabel("中心差分步长 h（bp）", fontsize=11)
ax.set_ylabel("基准点有效凸性", fontsize=11)
ax.set_title(f"步长收敛性：h∈[5,100]bp 极差仅 {cvx_spread:.0f}，结论不靠调参",
             fontsize=12.5, weight="bold")

ax = axes[1]
ax.plot(SHIFTS, mbs_cvx, "-o", color=C_MBS, lw=2.6, ms=4,
        label=f"含 burnout：基准 {mbs_cvx[i0]:.0f}，-300bp 捕获 {capture_m300*100:.0f}%")
ax.plot(SHIFTS, nb_cvx, "-s", color="#7c3aed", lw=2.4, ms=4,
        label=f"关掉 burnout：基准 {nb_cvx[i0]:.0f}，-300bp 捕获 "
              f"{nb_ret[i_m300]/tsy_ret[i_m300]*100:.0f}%")
ax.axhline(0, color="k", lw=1.2)
ax.annotate(f"基准点恶化\n{mbs_cvx[i0]:.0f} → {nb_cvx[i0]:.0f}", xy=(0, nb_cvx[i0]),
            xytext=(95, nb_cvx[i0] * 1.05), fontsize=10, color="#7c3aed", weight="bold",
            arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1.2))
ax.set_xlabel("利率平行移动（bp）", fontsize=11)
ax.set_ylabel("有效凸性", fontsize=11)
ax.set_title("Burnout 是负凸性的「安全气囊」：老池子没那么毒", fontsize=12.5, weight="bold")
ax.set_xticks(XT)
ax.legend(fontsize=9, loc="lower right", framealpha=0.94)
plt.tight_layout()
plt.savefig(f"{OUT}/numerical_robustness.png")
plt.close()
print("✓ numerical_robustness.png")

# =============================================================================
# 八、导出 stats.json
# =============================================================================
stats_out = {
    "seed": SEED,
    "pool_params": {
        "term_months": TERM_M, "wac": WAC, "net_coupon": NET_CPN,
        "base_discount_rate": DISC0, "base_mortgage_rate": MTG0,
        "orig_balance": ORIG_BAL,
        "cpr_min": CPR_MIN, "cpr_max": CPR_MAX,
        "scurve_k": K_STEEP, "scurve_x0_pp": X0_INCEN,
        "burnout_beta": BURN_BETA, "burnout_floor": BURN_FLOOR,
        "seasoning_months": SEASON_M
    },
    "base_case": {
        "price": float(base_price),
        "effective_duration": float(base_dur),
        "effective_convexity": float(base_cvx),
        "wal_years": float(base_wal),
        "cpr_at_zero_incentive": float(base_cpr_0),
        "first_year_avg_cpr": float(mbs_cpr1[i0])
    },
    "duration_matched_treasury": {
        "maturity_years": float(mat_star),
        "coupon": NET_CPN,
        "price": float(t_p0),
        "effective_duration": float(t_dur0),
        "effective_convexity": float(t_cvx0)
    },
    "scenario_grid": {
        "shifts_bp": SHIFTS.tolist(),
        "mbs_price": [float(x) for x in mbs_px],
        "mbs_return_pct": [float(x * 100) for x in mbs_ret],
        "mbs_eff_duration": [float(x) for x in mbs_dur],
        "mbs_eff_convexity": [float(x) for x in mbs_cvx],
        "mbs_wal_years": [float(x) for x in mbs_wal],
        "mbs_first_year_cpr": [float(x) for x in mbs_cpr1],
        "tsy_price": [float(x) for x in tsy_px],
        "tsy_return_pct": [float(x * 100) for x in tsy_ret],
        "tsy_eff_duration": [float(x) for x in tsy_dur],
        "tsy_eff_convexity": [float(x) for x in tsy_cvx],
        "hedge_residual_pnl": [float(x) for x in hedge_pnl],
        "hedge_residual_pred_2nd_order": [float(x) for x in pred_pnl]
    },
    "headline": {
        "ret_minus300_mbs_pct": float(mbs_ret[i_m300] * 100),
        "ret_minus300_tsy_pct": float(tsy_ret[i_m300] * 100),
        "upside_capture_minus300": float(capture_m300),
        "ret_minus100_mbs_pct": float(mbs_ret[i_m100] * 100),
        "ret_minus100_tsy_pct": float(tsy_ret[i_m100] * 100),
        "upside_capture_minus100": float(capture_m100),
        "ret_plus100_mbs_pct": float(mbs_ret[i_p100] * 100),
        "ret_plus100_tsy_pct": float(tsy_ret[i_p100] * 100),
        "ret_plus300_mbs_pct": float(mbs_ret[i_p300] * 100),
        "ret_plus300_tsy_pct": float(tsy_ret[i_p300] * 100),
        "downside_amplification_plus300": float(loss_p300),
        "convexity_negative_range_bp": [float(neg_range[0]), float(neg_range[1])],
        "convexity_negative_gridpoints": int(neg_mask.sum()),
        "convexity_gridpoints_total": int(len(SHIFTS)),
        "min_convexity": float(mbs_cvx[min_cvx_i]),
        "min_convexity_at_bp": float(SHIFTS[min_cvx_i]),
        "duration_peak_years": float(mbs_dur[dur_max_i]),
        "duration_peak_at_bp": float(SHIFTS[dur_max_i]),
        "duration_at_minus300": float(mbs_dur[i_m300]),
        "duration_shrink_pct": float((1 - mbs_dur[i_m300] / mbs_dur[dur_max_i]) * 100),
        "wal_minus300": float(mbs_wal[i_m300]),
        "wal_base": float(mbs_wal[i0]),
        "wal_plus300": float(mbs_wal[i_p300]),
        "wal_stretch_ratio": float(mbs_wal[i_p300] / mbs_wal[i_m300]),
        "cpr_first_year_minus300": float(mbs_cpr1[i_m300]),
        "cpr_first_year_base": float(mbs_cpr1[i0]),
        "cpr_first_year_plus300": float(mbs_cpr1[i_p300])
    },
    "adversarial_tests": {
        "placebo_zero_prepay": {
            "min_convexity": zero_min_cvx,
            "base_convexity": float(zero_cvx[i0]),
            "all_positive": bool(np.all(zero_cvx > 0)),
            "base_duration": float(zero_dur[i0]),
            "base_wal": float(wal(0, mode="zero"))
        },
        "control_constant_cpr_10pct": {
            "min_convexity": const_min_cvx,
            "base_convexity": float(const_cvx[i0]),
            "all_positive": bool(np.all(const_cvx > 0)),
            "base_duration": float(const_dur[i0])
        },
        "k_sensitivity_scan": {
            "k_values": K_SCAN,
            "min_convexity": k_min_cvx,
            "base_convexity": k_base_cvx,
            "upside_capture_minus300": k_cap
        },
        "hedge_effectiveness": {
            "hedge_ratio": float(hedge_ratio),
            "residual_minus300": float(hedge_pnl[i_m300]),
            "residual_minus100": float(hedge_pnl[i_m100]),
            "residual_plus100": float(hedge_pnl[i_p100]),
            "residual_plus300": float(hedge_pnl[i_p300]),
            "all_negative_off_base": bool(np.all(hedge_pnl[SHIFTS != 0] < 0)),
            "second_order_r2_full": float(r2),
            "second_order_r2_within_100bp": float(r2_small),
            "pred_2nd_order_plus100": float(pred_pnl[i_p100]),
            "worst_residual": float(hedge_pnl.min()),
            "total_residual_abs_sum": float(np.abs(hedge_pnl).sum())
        },
        "local_vs_path_convexity": {
            "local_min_convexity": float(mbs_cvx.min()),
            "local_min_at_bp": float(SHIFTS[min_cvx_i]),
            "path_convexity_minus300": c_path_m300,
            "path_convexity_plus300": c_path_p300,
            "path_convexity_minus100": float(c_path[i_m100]),
            "path_convexity_plus100": float(c_path[i_p100])
        },
        "step_size_convergence": {
            "h_bp": H_SCAN,
            "duration": h_dur,
            "convexity": h_cvx,
            "convexity_spread": float(cvx_spread),
            "all_negative": bool(max(h_cvx) < 0)
        },
        "burnout_off": {
            "min_convexity": float(nb_cvx.min()),
            "base_convexity": float(nb_cvx[i0]),
            "upside_capture_minus300": float(nb_ret[i_m300] / tsy_ret[i_m300])
        }
    }
}

with open(f"{OUT}/stats.json", "w", encoding="utf-8") as f:
    json.dump(stats_out, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 70)
print("✅ 所有图表和 stats.json 生成完毕！")
print(f"输出目录: {OUT}")
print("=" * 70)

s = stats_out["headline"]
print("\n===== 关键数字摘要 =====")
print(f"基准价格 {stats_out['base_case']['price']:.4f} | 久期 {stats_out['base_case']['effective_duration']:.3f}y "
      f"| 凸性 {stats_out['base_case']['effective_convexity']:.1f} | WAL {stats_out['base_case']['wal_years']:.2f}y")
print(f"-300bp: MBS {s['ret_minus300_mbs_pct']:+.2f}% vs 国债 {s['ret_minus300_tsy_pct']:+.2f}% "
      f"→ 捕获率 {s['upside_capture_minus300']*100:.1f}%")
print(f"+300bp: MBS {s['ret_plus300_mbs_pct']:+.2f}% vs 国债 {s['ret_plus300_tsy_pct']:+.2f}% "
      f"→ 放大 {s['downside_amplification_plus300']:.2f}x")
print(f"负凸性区间 {s['convexity_negative_range_bp']}，谷底 {s['min_convexity']:.1f}")
print(f"久期 {s['duration_peak_years']:.2f}y @ {s['duration_peak_at_bp']:+.0f}bp → "
      f"{s['duration_at_minus300']:.2f}y @ -300bp（缩 {s['duration_shrink_pct']:.1f}%）")
print(f"WAL {s['wal_minus300']:.2f}y → {s['wal_plus300']:.2f}y（{s['wal_stretch_ratio']:.2f}x）")
print(f"对冲最差残差 {stats_out['adversarial_tests']['hedge_effectiveness']['worst_residual']:.2f}/100面值，"
      f"二阶 R²(全区间)={r2:.3f} / R²(±100bp)={r2_small:.3f}")
print(f"路径平均凸性：-300bp {c_path_m300:.1f} | +300bp {c_path_p300:.1f}（局部谷底 {mbs_cvx.min():.1f}）")
print(f"安慰剂 CPR≡0 最小凸性 {zero_min_cvx:+.1f}；对照 CPR≡10% 最小凸性 {const_min_cvx:+.1f}")
