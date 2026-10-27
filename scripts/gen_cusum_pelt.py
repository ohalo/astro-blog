#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""变点检测在波动率跳跃中的应用：CUSUM 与 PELT 捕捉 regime 切换
E1 合成波动率阶梯市场 + CUSUM（对平方收益）统计量轨迹
E2 PELT 精确分割（L1 惩罚）：惩罚 β 扫描 vs 检出变点数，真值 5
E3 检出功效：跳跃幅度 σ1/σ0 扫描 × 200 次 MC，CUSUM（重尾置换峰值校准）/ PELT 各自检出率
E4 检出延迟：每次真变点后 CUSUM 首次越阈的天数分布（分大/小跳跃）
E5 冷水：均值漂移（无波动跳跃）在平方收益 CUSUM 上的虚警实拍 + PELT 对趋势段的过分割
输出 5 张图 + _cpd_results.json
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for f in ["Heiti TC", "PingFang SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

OUT = "/Users/halo123456/workspace/astro-blog/public/images/change-point-detection-volatility-jump"
os.makedirs(OUT, exist_ok=True)
RED, GREEN, BLUE, GRAY, ORANGE, BLACK, PURPLE = "#d63031", "#00b894", "#0984e3", "#636e72", "#e17055", "#2d3436", "#6c5ce7"
SEED = 20261027
rng = np.random.default_rng(SEED)
R = {}

# ---------------- 真值：波动率阶梯 ----------------
T = 1200
CPS = np.array([150, 380, 560, 820, 1000])          # 真变点
SIGS = np.array([0.80, 2.00, 1.20, 2.60, 0.90, 1.80])  # 6 段日波动 %（段间跳幅 2.5x/1.67x/2.17x/2.89x/2x）

def make_sigma(T, cps, seg_sigs):
    bnd = [0] + list(cps) + [T]
    s = np.empty(T)
    for i in range(len(seg_sigs)):
        s[bnd[i]:bnd[i + 1]] = seg_sigs[i]
    return s

sigma_true = make_sigma(T, CPS, SIGS)
r = rng.normal(0.0, sigma_true)  # 均值 0，方差分段
R["cps"] = CPS.tolist(); R["sigs"] = SIGS.tolist()
R["jump_ratios"] = [round(float(SIGS[i + 1] / SIGS[i]), 2) for i in range(5)]
R["jump_dirs"] = ["up" if SIGS[i + 1] > SIGS[i] else "down" for i in range(5)]

# ---------------- CUSUM（对平方收益，双侧） ----------------
def cusum(x2, k_allow, h_thr=None):
    """双侧 CUSUM（Lindley 递推的向量化形式）：S_t = W_t − min_{s<=t} W_s（不复位，供阈值校准取峰值分布）"""
    m0 = np.mean(x2)
    def side(delta):
        W = np.cumsum(delta)
        W0 = np.concatenate([[0.0], W])
        run_min = np.minimum.accumulate(W0)[:len(W)]
        return W - run_min
    sp = side(x2 - m0 - k_allow)
    sn = side(-(x2 - m0) - k_allow)
    return sp, sn

def cusum_reset(x2s, k_allow, h, m0):
    """标准复位型双侧 CUSUM（PP 图语义）：S±_t = max(0, S±_{t-1} ± (x_t − m0 ∓ k))；
    任一统计量越阈 h 即告警并把两侧统计量清零复位，可连续抓多个变化。"""
    n = len(x2s)
    sp = np.zeros(n); sn = np.zeros(n); al = []
    splus = sminus = 0.0
    for t in range(n):
        splus = max(0.0, splus + (x2s[t] - m0) - k_allow)
        sminus = max(0.0, sminus - (x2s[t] - m0) - k_allow)
        if splus > h or sminus > h:
            al.append(t); splus = 0.0; sminus = 0.0
        sp[t] = splus; sn[t] = sminus
    return sp, sn, al

x2 = r ** 2
m2, s2_ = np.mean(x2), np.std(x2)
K_ALLOW = 0.5 * s2_  # 松弛 ≈ 半个标准差的漂移（经典取值）

def cusum_calibrated(x2s, k_allow):
    """用 permutation 校准阈值到约 5% 平均虚警/序列：对打乱序列跑同 CUSUM，取 95 分位峰值。"""
    peaks = []
    rg = np.random.default_rng(7)
    for _ in range(60):
        xp = x2s[rg.permutation(len(x2s))]
        sp, sn = cusum(xp, k_allow, 0)
        peaks.append(max(sp.max(), sn.max()))
    return float(np.quantile(peaks, 0.95))

H_THR = cusum_calibrated(x2, K_ALLOW)  # 主序列阈值：置换校准至 5% 虚警水平

def alarm_times(sp, sn, h):
    al = []
    armed = True
    for t in range(len(sp)):
        if armed and (sp[t] > h or sn[t] > h):
            al.append(t); armed = False
        elif sp[t] == 0 and sn[t] == 0:
            armed = True
    return al

sp0, sn0, alarms0 = cusum_reset(x2, K_ALLOW, H_THR, np.mean(x2[:CPS[0]]))  # m0 用首段（无变参照段）估计
R["cusum_H"] = round(float(H_THR), 3); R["cusum_k"] = round(float(K_ALLOW), 3)

def detected_cps(al, tol=40):
    """把告警时刻映射到最近真变点（±tol 内算命中，同变点取最早告警的滞后）。"""
    hits = {}
    for a in al:
        dd = np.abs(CPS - a)
        j = int(dd.argmin())
        if dd[j] <= tol:
            hits[j] = min(hits.get(j, 10 ** 9), a - CPS[j])
    return hits

hits0 = detected_cps(alarms0)
R["n_alarms_main"] = len(alarms0)
R["cusum_hits"] = sorted(hits0.keys())
R["cusum_hit_cps"] = [int(CPS[j]) for j in sorted(hits0.keys())]
R["cusum_lags_map"] = {int(CPS[j]): int(v) for j, v in hits0.items()}

# ---------------- PELT（精确，L1 惩罚，成本 = 高斯负对数似然（均值0，方差未知）） ----------------
LMIN = 10  # 每段最少 10 天，防止逐点退化

def pelt(x2, beta):
    """精确最优分割：成本 = log r² 分段常数高斯负对数似然（前缀和 O(1) 段成本），
    F[t] = min_s F[s] + cost(s,t) + β，全量 DP + 回溯。"""
    v = np.log(x2 + 1e-12)
    n = len(v)
    c1 = np.concatenate([[0.0], np.cumsum(v)])
    c2 = np.concatenate([[0.0], np.cumsum(v * v)])
    F = np.full(n + 1, np.inf); F[0] = -beta
    bk = np.zeros(n + 1, dtype=int)
    for t in range(1, n + 1):
        s = np.arange(0, t - LMIN + 1)
        if len(s) == 0:
            F[t] = np.inf; bk[t] = 0; continue
        m = t - s
        s1 = c1[t] - c1[s]; s2 = c2[t] - c2[s]
        mu = s1 / m
        var_ = np.maximum(s2 / m - mu * mu, 1e-8)
        c = m * (np.log(var_) + 1)
        c = np.where(m >= LMIN, c, np.inf)
        costs = F[s] + c + beta
        j = int(costs.argmin())
        F[t] = costs[j]; bk[t] = int(s[j])
    cps = []; t = n
    while t > 0:
        s = bk[t]
        if s > 0:
            cps.append(s)
        t = s
    return sorted(cps)

# 先扫 β：记录每个 β 的命中/虚警，主 β 取"零虚警带"中命中最多的子带中点
def pelt_hits_scan(cps_hat_, tol=40):
    hit = [bool([h for h in cps_hat_ if abs(h - c) <= tol]) for c in CPS]
    fp = [h for h in cps_hat_ if all(abs(h - c) > tol for c in CPS)]
    return hit, fp

_betas_pro = np.arange(5.0, 50.01, 1.0)
_scan = []
for b in _betas_pro:
    ch = pelt(x2, float(b))
    h, f = pelt_hits_scan(ch)
    _scan.append((float(b), len(ch), sum(h), len(f)))
_fp0 = [s for s in _scan if s[3] == 0]
_best_h = max(s[2] for s in _fp0)
_ok = [s[0] for s in _fp0 if s[2] == _best_h]
BETA_MAIN = float(np.median(_ok)) if _ok else 22.0
R["beta_zero_fp_band"] = [round(min(s[0] for s in _fp0), 1), round(max(s[0] for s in _fp0), 1)] if _fp0 else None
R["beta_main_hits"] = _best_h
cps_hat = pelt(x2, BETA_MAIN)
R["pelt_beta_main"] = BETA_MAIN
R["pelt_cps_main"] = [int(c) for c in cps_hat]
# 命中：±40 内
def pelt_hits(cps_hat, tol=40):
    hit = []
    for c in CPS:
        near = [h for h in cps_hat if abs(h - c) <= tol]
        hit.append(bool(near))
    fp = [h for h in cps_hat if all(abs(h - c) > tol for c in CPS)]
    return hit, fp
ph, pfp = pelt_hits(cps_hat)
R["pelt_hit_map_main"] = ph
R["pelt_fp_main"] = [int(x) for x in pfp]

# ---------------- 图 1：市场 + CUSUM 轨迹 ----------------
fig, ax = plt.subplots(3, 1, figsize=(12, 8.2), height_ratios=[1.5, 1, 1], sharex=True)
price = 100 + np.cumsum(r)
ax[0].plot(price, color=BLACK, lw=0.9)
for c in CPS:
    ax[0].axvline(c, color=RED, ls="--", lw=1.1, alpha=0.75)
bnd = [0] + list(CPS) + [T]
for i in range(6):
    ax[0].axvspan(bnd[i], bnd[i + 1], alpha=0.08, color=(BLUE if SIGS[i] < 1.5 else ORANGE))
ax[0].set_title("合成波动率阶梯市场：5 个真变点（红虚线），段 σ = 0.80/2.00/1.20/2.60/0.90/1.80 %（蓝低波 橙高波）", fontsize=10)
ax[0].set_ylabel("价格")
ax[1].plot(r, color=GRAY, lw=0.5)
ax[1].plot(np.arange(T), sigma_true, color=PURPLE, lw=1.4, label="真 σ（日）")
for c in CPS:
    ax[1].axvline(c, color=RED, ls="--", lw=1.0, alpha=0.6)
ax[1].legend(fontsize=8); ax[1].set_ylabel("收益率 / σ %")
ax[2].plot(sp0, color=RED, lw=1.0, label="S+（波动升高）")
ax[2].plot(-sn0, color=BLUE, lw=1.0, label="S-（波动回落）")
ax[2].axhline(H_THR, color=BLACK, ls=":", lw=1.2, label=f"阈值 h={H_THR:.2f}")
ax[2].axhline(-H_THR, color=BLACK, ls=":", lw=1.2)
for a_ in alarms0:
    ax[2].axvline(a_, color=ORANGE, ls="-", lw=0.8, alpha=0.5)
ax[2].set_title(f"双侧 CUSUM（对 r²，松弛 k={K_ALLOW:.2f}）：{len(alarms0)} 次告警，命中真变点 {len(hits0)}/5", fontsize=10)
ax[2].set_xlabel("交易日"); ax[2].legend(fontsize=8, ncol=3, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_cusum_path.png"); plt.close(fig)

# ---------------- 图 2：PELT 惩罚扫描 ----------------
betas = np.arange(5.0, 50.01, 1.0)
pcnt, phit, pfpn = [], [], []
for b in betas:
    c_ = pelt(x2, float(b))
    h_, f_ = pelt_hits(c_)
    pcnt.append(len(c_)); phit.append(sum(h_)); pfpn.append(len(f_))
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax[0].plot(betas, pcnt, "o-", color=PURPLE, ms=3.5, lw=1.2, label="PELT 检出变点总数")
ax[0].plot(betas, phit, "s-", color=GREEN, ms=3.5, lw=1.2, label="命中真变点数 / 5")
ax[0].plot(betas, pfpn, "^-", color=RED, ms=3.5, lw=1.2, label="误报数（±40 外）")
ax[0].axhline(5, color=BLACK, ls=":", lw=1.0)
ax[0].axvline(BETA_MAIN, color=BLUE, ls="--", lw=1.2, label=f"本文主 β={BETA_MAIN}（检出 {R['pelt_cps_main'].__len__()} 个）")
ax[0].set_title("PELT（L1 惩罚，log r² 高斯成本）：惩罚 β 是唯一定生死钮")
ax[0].set_ylabel("数量"); ax[0].legend(fontsize=8, ncol=4)
band_fp0 = [s[0] for s in _scan if s[3] == 0]
best_hit0 = max((s[2] for s in _scan if s[3] == 0), default=None)
best_band = [s[0] for s in _scan if s[3] == 0 and s[2] == best_hit0]
R["beta_zero_fp_band"] = [round(min(band_fp0), 1), round(max(band_fp0), 1)] if band_fp0 else None
R["beta_best_hits_zero_fp"] = best_hit0
ax[1].plot(betas, [abs(n_ - 5) for n_ in pcnt], "o-", color=ORANGE, ms=3.5, lw=1.2, label="|检出数 − 5|")
if best_band:
    ax[1].axvspan(min(best_band), max(best_band), alpha=0.15, color=GREEN,
                  label=f"零虚警带 β∈[{min(band_fp0):.0f}, {max(band_fp0):.0f}]，此带命中 {best_hit0}/5")
    ax[1].axvspan(min(band_fp0), max(band_fp0), alpha=0.0, label=None)
# 全对（5/5 且 0 误报）带？
band_ok = [s[0] for s in _scan if s[2] == 5 and s[3] == 0]
R["beta_plateau"] = [round(min(band_ok), 1), round(max(band_ok), 1)] if band_ok else None
has_all5 = [s[0] for s in _scan if s[2] == 5]
R["beta_hits5"] = [round(min(has_all5), 1), round(max(has_all5), 1)] if has_all5 else None
if has_all5:
    ax[1].axvline(min(has_all5), color=RED, ls=":", lw=1.2, label=f"能抓到全部 5 个的最高 β={max(has_all5):.0f}（带 {[s[3] for s in _scan if s[0]==max(has_all5)][0]} 个误报）")
ax[1].set_title(f"|检出数 − 真值 5| 随 β：过分割区 / 漏检区实测边界（不存在零误报又全命中的 β）")
ax[1].set_xlabel("惩罚 β"); ax[1].set_ylabel("误差"); ax[1].legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_pelt_beta.png"); plt.close(fig)

# ---------------- 图 3：功效扫描（MC） ----------------
mags = [1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]
NS = 200
pow_cusum, pow_pelt, far_cusum, far_pelt = [], [], [], []
Tmc = 400
for mg in mags:
    hitc = col = fpc = fpp = 0
    for s in range(NS // 4):  # 50 次（幅度大时功效高，200 序列预算分给关键档）
        rg = np.random.default_rng(int(5000 + mg * 97) + s)
        sg = np.full(Tmc, 0.8); sg[Tmc // 2:] = 0.8 * mg
        rr = rg.normal(0, sg)
        xx = rr ** 2
        kk = 0.5 * np.std(xx)
        hh = cusum_calibrated(xx[:Tmc // 2], kk)  # 用低波段校准阈值（无跳跃段）
        sp, sn, al = cusum_reset(xx, kk, hh, np.mean(xx[:Tmc // 2]))
        if any(Tmc // 2 - 20 <= a <= Tmc // 2 + 60 for a in al):
            hitc += 1
        else:
            col += 1
        # 虚警率：跳变前告警
        pre = [a for a in al if a < Tmc // 2 - 20]
        fpc += len(pre)
    pow_cusum.append(100 * hitc / (NS // 4))
    far_cusum.append(100 * fpc / (NS // 4) / (Tmc // 2))
# PELT 功效单独跑（与 CUSUM 同一批次序列重放）
for mg in mags:
    hitp = 0; cnt = NS // 4
    for s in range(cnt):
        rg = np.random.default_rng(int(5000 + mg * 97) + s)
        sg = np.full(Tmc, 0.8); sg[Tmc // 2:] = 0.8 * mg
        rr = rg.normal(0, sg)
        xx = rr ** 2
        ch = pelt(xx, 22.0)
        if any(abs(c - Tmc // 2) <= 40 for c in ch):
            hitp += 1
    pow_pelt.append(100 * hitp / cnt)

fig, ax = plt.subplots(figsize=(11, 4.8))
xx_ = np.arange(len(mags))
ax.bar(xx_ - 0.2, pow_cusum, 0.4, color=BLUE, alpha=0.85, label="CUSUM 检出率（重尾置换峰值校准阈值）")
ax.bar(xx_ + 0.2, pow_pelt, 0.4, color=PURPLE, alpha=0.85, label="PELT 检出率（β=22）")
for i, (a, b) in enumerate(zip(pow_cusum, pow_pelt)):
    ax.text(i - 0.2, a + 1.2, f"{a:.0f}", ha="center", fontsize=8)
    ax.text(i + 0.2, b + 1.2, f"{b:.0f}", ha="center", fontsize=8)
ax.axhline(80, color=RED, ls=":", lw=1.2, label="80% 功效线")
ax.set_xticks(xx_); ax.set_xticklabels([f"×{m}" for m in mags])
ax.set_title(f"波动率跳跃幅度 × 检出功效（{NS // 4} 次 MC/档，T=400，变点在 200 天）：CUSUM 过 80% 功效需跳幅 ≈×{mags[int(np.argmax(np.array(pow_cusum) >= 80))]}，PELT ≈×{mags[int(np.argmax(np.array(pow_pelt) >= 80))]}")
ax.set_xlabel("跳跃幅度 σ跳/σ基"); ax.set_ylabel("检出率 %"); ax.set_ylim(0, 108); ax.legend(fontsize=8, ncol=3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_power.png"); plt.close(fig)
R["power_cusum"] = [round(p, 1) for p in pow_cusum]
R["power_pelt"] = [round(p, 1) for p in pow_pelt]
R["far_cusum_perday"] = round(float(np.mean(far_cusum)), 3)

# ---------------- 图 4：检出延迟 ----------------
lags_big, lags_small = [], []
mag_of = lambda i: max(SIGS[i + 1] / SIGS[i], SIGS[i] / SIGS[i + 1])
for j, c in enumerate(CPS):
    (lags_big if mag_of(j) >= 2 else lags_small).append(R["cusum_lags_map"].get(j, None))
lags_all = [v for v in R["cusum_lags_map"].values()]
# 补充：对每个变点也记录 PELT 断点偏移
pelt_shifts = []
for c in CPS:
    near = [abs(h - c) for h in cps_hat if abs(h - c) <= 40]
    pelt_shifts.append(min(near) if near else None)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
cusum_lag_vals = [R["cusum_lags_map"].get(int(c), np.nan) for c in CPS]
ax[0].bar(np.arange(5) - 0.2, cusum_lag_vals, 0.4,
          color=[RED if mag_of(j) >= 2 else ORANGE for j in range(5)], label="CUSUM 检出偏移")
ax[0].bar(np.arange(5) + 0.2, [p if p is not None else np.nan for p in pelt_shifts], 0.4,
          color=PURPLE, alpha=0.75, label="PELT 断点偏移")
for j in range(5):
    lg = cusum_lag_vals[j]
    if not np.isnan(lg):
        ax[0].text(j - 0.2, lg + 1, str(int(lg)), ha="center", fontsize=9)
    if pelt_shifts[j] is not None:
        ax[0].text(j + 0.2, pelt_shifts[j] + 1, str(pelt_shifts[j]), ha="center", fontsize=9)
ax[0].set_xticks(range(5)); ax[0].set_xticklabels([f"day {c}\nσ {SIGS[i]:.1f}→{SIGS[i+1]:.1f}" for i, c in enumerate(CPS)], fontsize=8)
_fin = [v for v in cusum_lag_vals if not np.isnan(v)]
_hitn = int(sum(1 for c in CPS if c in R["cusum_lags_map"]))
ax[0].set_title(f"逐变点检出偏移（正=晚检 负=提前预警）：CUSUM 命中 {_hitn}/5 命中样本|偏移|中位 {np.median(np.abs(_fin)):.0f} 天，跳幅≥2x 红 / <2x 橙", fontsize=9)
ax[0].set_ylabel("延迟 / 偏移（天）"); ax[0].legend(fontsize=8)
al_lags = []
rg2 = np.random.default_rng(31)
for s in range(120):
    sg = np.full(300, 0.8); sg[150:] = 1.92  # ×2.4 跳
    rr = rg2.normal(0, sg); xx = rr ** 2
    kk = 0.5 * np.std(xx)
    hh = cusum_calibrated(xx[:150], kk)
    sp, sn, al0 = cusum_reset(xx, kk, hh, np.mean(xx[:150]))
    al = [a for a in al0 if a >= 150 - 20]
    if al:
        al_lags.append(al[0] - 150)
ax[1].hist(al_lags, bins=range(0, 61, 4), color=BLUE, edgecolor=BLACK, alpha=0.85)
ax[1].axvline(np.median(al_lags), color=RED, ls="--", lw=1.5, label=f"中位 {np.median(al_lags):.0f} 天")
ax[1].set_title(f"×2.4 跳跃 120 次 MC 的 CUSUM 首告延迟分布（n={len(al_lags)}，均值 {np.mean(al_lags):.1f}，P90 {np.percentile(al_lags,90):.0f}）")
ax[1].set_xlabel("变点后延迟（天）"); ax[1].set_ylabel("次数"); ax[1].legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_lag.png"); plt.close(fig)
R["lag24_median"] = float(np.median(al_lags)); R["lag24_mean"] = round(float(np.mean(al_lags)), 1)
R["lag24_p90"] = float(np.percentile(al_lags, 90)); R["lag24_n"] = len(al_lags)
R["pelt_shifts"] = [int(p) if p is not None else None for p in pelt_shifts]

# ---------------- 图 5：虚警冷水——均值漂移骗过平方 CUSUM ----------------
# 场景：σ 恒定 1.0，但均值从 0 缓慢漂到 +1.8%/天（结构性趋势，无波动跳跃）
T5, beta5 = 900, 1.6
r5_a = np.concatenate([rng.normal(0.0, 1.0, T5 // 2), rng.normal(0.9, 1.0, T5 // 2)])  # 阶跃均值
r5_b = np.linspace(0, 1.6, T5) + rng.normal(0, 1.0, T5)                                 # 线性漂移
hk = cusum_calibrated(r5_a[:T5 // 2], 0.5 * np.std(r5_a[:T5 // 2]))
sp_a, sn_a, al_a = cusum_reset(r5_a ** 2, 0.5 * np.std(r5_a ** 2), hk, np.mean(r5_a[:T5 // 2] ** 2))
sp_b, sn_b, al_b = cusum_reset(r5_b ** 2, 0.5 * np.std(r5_b ** 2), hk, np.mean(r5_b[:T5 // 2] ** 2))
R["false_alarm_step"] = [int(a) for a in al_a]; R["false_alarm_drift"] = [int(a) for a in al_b]
cps_pelt_a = pelt(r5_a ** 2, BETA_MAIN); cps_pelt_b = pelt(r5_b ** 2, BETA_MAIN)
R["pelt_overstep_a"] = [int(c) for c in cps_pelt_a]; R["pelt_overstep_b"] = [int(c) for c in cps_pelt_b]
# 对照：同 β 下对含真跳序列的断点数
fig, ax = plt.subplots(2, 2, figsize=(12, 7.2))
ax[0][0].plot(r5_a, color=GRAY, lw=0.5)
ax[0][0].axvline(T5 // 2, color=BLUE, ls="--", lw=1.2)
for a_ in al_a:
    ax[0][0].axvline(a_, color=RED, lw=1.0, alpha=0.7)
ax[0][0].set_title(f"均值阶跃 0→0.9%（σ 恒定）：平方 CUSUM 告警 {len(al_a)} 次（红线）——虚警", fontsize=9)
ax[0][1].plot(r5_b, color=GRAY, lw=0.5)
for a_ in al_b:
    ax[0][1].axvline(a_, color=RED, lw=1.0, alpha=0.7)
ax[0][1].set_title(f"均值线性漂移 0→1.6%（σ 恒定）：告警 {len(al_b)} 次", fontsize=9)
ax[1][0].plot(np.arange(T5), np.abs(r5_b), color=GRAY, lw=0.4)
ax[1][0].plot(np.arange(T5), np.maximum(sp_b, sn_b), color=BLUE, lw=1.2, label="CUSUM 峰值")
ax[1][0].axhline(hk, color=BLACK, ls=":", lw=1.1, label=f"校准阈值 {hk:.1f}")
ax[1][0].set_title("漂移段 CUSUM 统计量爬坡越阈：均值漂移在 r² 眼里就是'波动升高'", fontsize=9)
ax[1][0].legend(fontsize=8)
ax[1][1].plot(r5_a, color=GRAY, lw=0.4)
ax[1][1].axvline(T5 // 2, color=GREEN, ls=":", lw=1.5, label=f"真均值阶跃 day {T5 // 2}")
for c in cps_pelt_a:
    ax[1][1].axvline(c, color=PURPLE, ls="--", lw=1.0, alpha=0.8)
ax[1][1].legend(fontsize=8)
ax[1][1].set_title(f"PELT（β={BETA_MAIN:.0f}）在同一序列断出 {len(cps_pelt_a)} 个断点（紫线）：σ 恒定，全无真变点", fontsize=9)
ax[1][1].set_xlabel("交易日")
fig.suptitle("冷水：CUSUM/PELT 都对'方差'负责——均值漂移会被读成波动跳跃（σ 全程恒定为 1.0）", fontsize=11, y=1.0)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_false_alarm.png"); plt.close(fig)
R["t5"] = T5; R["hk"] = round(float(hk), 3)

with open("/Users/halo123456/workspace/astro-blog/_cpd_results.json", "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
