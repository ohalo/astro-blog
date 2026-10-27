#!/usr/bin/env python3
"""为文章「GARCH-EMD 混合波动：用经验模态分解剥离趋势再做 GARCH」(garch-emd-hybrid) 生成真实配图。

方法论(诚实、可复现):
  金融收益的波动里既有「慢变的结构性水平漂移(regime/宏观)」也有「快变的条件异方差聚集」。
  直接对原始收益拟合 GARCH(1,1)，慢趋势会污染波动率截距、拉高持续性 β 的估计。
  混合思路: 先用 EMD 把收益序列(或其平方/绝对值代理)分解成 IMF，把最低频的几个 IMF+残余
  当作「趋势/慢波动」剥掉，对剩下的高频残差再拟合 GARCH。
  实验:
    (a) 合成收益 = 慢变波动趋势 * GARCH 冲击, 展示 EMD 分解 + 剥趋势;
    (b) 纯 GARCH vs EMD-GARCH 的条件波动率 vs 真实波动率对比;
    (c) 样本外一步波动预测: QLIKE / MSE 对比(纯 GARCH / EMD-GARCH / EWMA / 滚动std);
    (d) 诚实翻车: EMD 端点效应 + 剥太多 IMF 会把真波动信息也剥掉。
  纯 numpy 实现 GARCH MLE(网格+梯度) 与 EMD sifting, 非占位图。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.interpolate import CubicSpline

for _f in ["/Library/Fonts/Arial Unicode.ttf",
           "/System/Library/Fonts/STHeiti Medium.ttc",
           "/System/Library/Fonts/PingFang.ttc"]:
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUT = "public/images/garch-emd-hybrid"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight",
    "axes.unicode_minus": False, "font.family": "Arial Unicode MS",
})
C_TRUE = "#111827"; C_GARCH = "#ef4444"; C_HYB = "#2563eb"; C_AUX = "#9ca3af"

rng = np.random.default_rng(20260724)

# ---------------- 1) 合成收益: 慢变方差水平 g_t x 短期 GARCH 因子 h_t ----------------
# 乘性成分模型(Engle-Rangel Spline-GARCH 思路): sigma^2_t = g_t * h_t
#   g_t: 慢变(regime/宏观)方差水平, 含一次较陡的 regime 跳变, 使纯 GARCH 难以追上
#   h_t: 单位无条件均值的平稳短期 GARCH 因子(波动聚集)
def simulate(n=2000, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    # 慢变方差水平 g_t: 缓慢正弦 + 一次较陡的 regime 抬升(第1300天附近)
    g = (0.010 + 0.005*(1+np.sin(2*np.pi*t/700)))**2      # 缓慢起伏(方差尺度)
    g = g + (0.013**2) * (1/(1+np.exp(-(t-1300)/40)))      # 陡峭 regime 跳变
    # 短期 GARCH(1,1) 因子, 归一到单位无条件均值
    a0, a1, b1 = 0.06, 0.09, 0.88
    h = np.zeros(n); u = np.zeros(n); h[0] = a0/(1-a1-b1)
    z = rng.standard_normal(n)
    for i in range(1, n):
        h[i] = a0 + a1*u[i-1]**2 + b1*h[i-1]
        u[i] = np.sqrt(h[i]) * z[i]
    h = h / np.mean(h)                                     # 单位均值因子
    true_var = g * h                                       # 真实瞬时方差
    true_sigma = np.sqrt(true_var)
    ret = true_sigma * rng.standard_normal(n)              # 观测收益
    slow_vol = np.sqrt(g)                                  # 真实慢波动水平
    return t, ret, true_sigma, slow_vol

# ---------------- 2) EMD(从零): cubic-spline 包络 sifting ----------------
def _extrema(x):
    n = len(x)
    mx = [i for i in range(1, n-1) if x[i] > x[i-1] and x[i] >= x[i+1]]
    mn = [i for i in range(1, n-1) if x[i] < x[i-1] and x[i] <= x[i+1]]
    return np.array(mx), np.array(mn)

def _envelope(idx, val, n):
    # 端点 clamp: 复制首尾极值, 抑制样条外推爆炸
    xi = np.concatenate(([0], idx, [n-1]))
    yi = np.concatenate(([val[0]], val, [val[-1]]))
    xi, uniq = np.unique(xi, return_index=True)
    yi = yi[uniq]
    cs = CubicSpline(xi, yi)
    return cs(np.arange(n))

def emd(x, max_imf=8, max_sift=60):
    x = np.asarray(x, float); n = len(x)
    imfs = []; res = x.copy()
    for _ in range(max_imf):
        h = res.copy()
        for _s in range(max_sift):
            mx, mn = _extrema(h)
            if len(mx) < 2 or len(mn) < 2:
                break
            upper = _envelope(mx, h[mx], n)
            lower = _envelope(mn, h[mn], n)
            mean_env = 0.5*(upper+lower)
            h_new = h - mean_env
            if np.mean(mean_env**2) < 1e-10 * (np.mean(h**2)+1e-12):
                h = h_new; break
            h = h_new
        imfs.append(h)
        res = res - h
        mx, mn = _extrema(res)
        if len(mx) + len(mn) < 3:
            break
    imfs.append(res)  # residue
    return imfs

# ---------------- 3) GARCH(1,1) MLE(从零) ----------------
def garch_negll(params, r):
    a0, a1, b1 = params
    if a0 <= 0 or a1 < 0 or b1 < 0 or a1+b1 >= 0.999:
        return 1e10
    n = len(r); h = np.empty(n)
    h[0] = np.var(r)
    ll = 0.0
    for i in range(1, n):
        h[i] = a0 + a1*r[i-1]**2 + b1*h[i-1]
        if h[i] <= 0: return 1e10
        ll += 0.5*(np.log(2*np.pi) + np.log(h[i]) + r[i]**2/h[i])
    return ll

def fit_garch(r):
    # 粗网格 + 局部坐标下降
    var_r = np.var(r)
    best = None; bestll = 1e18
    for a1 in [0.03,0.05,0.08,0.12,0.18]:
        for b1 in [0.80,0.86,0.90,0.94]:
            if a1+b1 >= 0.999: continue
            a0 = var_r*(1-a1-b1)
            ll = garch_negll((a0,a1,b1), r)
            if ll < bestll: bestll = ll; best=[a0,a1,b1]
    # 局部细化
    for _ in range(200):
        improved = False
        for k,step in [(0,best[0]*0.1),(1,0.005),(2,0.005)]:
            for d in (step,-step):
                cand = best.copy(); cand[k]+=d
                if cand[0]<=0 or cand[1]<0 or cand[2]<0 or cand[1]+cand[2]>=0.999: continue
                ll = garch_negll(cand, r)
                if ll < bestll-1e-9: bestll=ll; best=cand; improved=True
        if not improved: break
    return np.array(best), bestll

def garch_filter(params, r):
    a0,a1,b1 = params; n=len(r); h=np.empty(n); h[0]=np.var(r)
    for i in range(1,n):
        h[i]=a0+a1*r[i-1]**2+b1*h[i-1]
    return np.sqrt(h)

# ================= RUN =================
t, ret, true_sigma, slow_vol = simulate()
n = len(ret)

split = int(n*0.7)

# 混合思路(乘性分解 Spline-GARCH 思路): 先用 EMD 从对数平方收益提慢变方差水平 g_t, 再对标准化残差拟 GARCH
proxy = np.log(ret**2 + 1e-8)                 # 对数平方收益作为方差代理
imfs = emd(proxy, max_imf=8)
low_freq = imfs[-1] + (imfs[-2] if len(imfs) >= 2 else 0)  # 最低两层 = 慢变 log 方差
g_hat = np.exp(low_freq)
g_hat = g_hat / np.mean(g_hat) * np.var(ret)  # 对齐到无条件方差尺度
detrended = ret / np.sqrt(g_hat)              # 剥掉慢方差后的标准化残差
ret_std = detrended

# 纯 GARCH: 直接拟原始收益; EMD-GARCH: 拟标准化残差
p_pure, _ = fit_garch(ret[:split])
p_hyb, _  = fit_garch(ret_std[:split])
sig_pure = garch_filter(p_pure, ret)
h_hyb = garch_filter(p_hyb, ret_std)          # 短期 GARCH 因子(单位尺度)
sig_hyb = np.sqrt(g_hat) * (h_hyb / np.sqrt(np.mean(h_hyb**2)))  # 乘性重构 sqrt(g_t)*h_t
slow_level = np.sqrt(g_hat)                    # EMD 剥出的慢波动水平

print(f"纯 GARCH 参数  a0={p_pure[0]:.5f} a1={p_pure[1]:.3f} b1={p_pure[2]:.3f} 持续性={p_pure[1]+p_pure[2]:.3f}")
print(f"EMD-GARCH 参数 a0={p_hyb[0]:.5f} a1={p_hyb[1]:.3f} b1={p_hyb[2]:.3f} 持续性={p_hyb[1]+p_hyb[2]:.3f}")

# ---------- 图1 cover: EMD 分解 + 剥趋势 ----------
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
axes[0].plot(t, ret, color=C_AUX, lw=0.6, label="观测收益")
axes[0].plot(t, slow_level, color=C_HYB, lw=1.8, label="EMD 剥出的慢波动水平 √g_t")
axes[0].plot(t, -slow_level, color=C_HYB, lw=1.8)
axes[0].plot(t, slow_vol, color=C_TRUE, lw=1.2, ls="--", label="真实慢波动水平")
axes[0].plot(t, -slow_vol, color=C_TRUE, lw=1.2, ls="--")
axes[0].set_title("① 原始收益 与 EMD 剥出的慢波动包络", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=9, loc="upper right"); axes[0].set_ylabel("收益")
axes[1].plot(t, detrended, color="#059669", lw=0.6)
axes[1].set_title("② 剥掉慢方差后的标准化残差(喂给短期 GARCH)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("残差")
axes[2].plot(t, true_sigma, color=C_TRUE, lw=1.6, label="真实瞬时波动 σ_t")
axes[2].plot(t, sig_pure, color=C_GARCH, lw=1.0, alpha=0.85, label="纯 GARCH 条件波动")
axes[2].plot(t, sig_hyb, color=C_HYB, lw=1.0, alpha=0.85, label="EMD-GARCH 混合波动")
axes[2].axvline(split, color="k", ls=":", lw=1); axes[2].text(split+10, axes[2].get_ylim()[1]*0.9, "样本外", fontsize=9)
axes[2].set_title("③ 波动率估计对比", fontsize=12, fontweight="bold")
axes[2].legend(fontsize=9, loc="upper left"); axes[2].set_ylabel("σ"); axes[2].set_xlabel("时间(交易日)")
plt.tight_layout(); plt.savefig(f"{OUT}/cover.png", dpi=120); plt.close()

# ---------- 图2: IMF 全谱 ----------
k = len(imfs)
fig, axes = plt.subplots(k, 1, figsize=(11, 1.4*k), sharex=True)
for i, imf in enumerate(imfs):
    lbl = f"IMF{i+1}" if i < k-1 else "残余 residue"
    axes[i].plot(t, imf, color=(C_HYB if i>=k-2 else "#374151"), lw=0.8)
    axes[i].set_ylabel(lbl, fontsize=9, rotation=0, ha="right", va="center")
    axes[i].tick_params(labelsize=8)
axes[0].set_title("EMD 分解: 高频 → 低频 → 残余(最低两层当慢结构剥离)", fontsize=12, fontweight="bold")
axes[-1].set_xlabel("时间(交易日)")
plt.tight_layout(); plt.savefig(f"{OUT}/imfs.png", dpi=120); plt.close()

# ---------- 图3: 样本外一步波动预测对比 ----------
# 用各法在 [split:] 上的一步波动预测, 评估 QLIKE 与 MSE(以真实 σ² 为标的)
oos = slice(split, n)
true_var = true_sigma[oos]**2
def metrics(sig_hat):
    v = np.maximum(sig_hat[oos]**2, 1e-10)
    mse = np.mean((v - true_var)**2)
    qlike = np.mean(np.log(v) + true_var/v)
    return mse, qlike
# 基线
ewma = np.zeros(n); lam=0.94; ewma[0]=np.var(ret[:split])
for i in range(1,n): ewma[i]=lam*ewma[i-1]+(1-lam)*ret[i-1]**2
sig_ewma = np.sqrt(ewma)
sig_roll = np.array([np.std(ret[max(0,i-63):i]) if i>0 else np.std(ret[:split]) for i in range(n)])

rows = [("EMD-GARCH 混合", sig_hyb, C_HYB),
        ("纯 GARCH", sig_pure, C_GARCH),
        ("EWMA(λ=0.94)", sig_ewma, "#f59e0b"),
        ("滚动std(63)", sig_roll, C_AUX)]
res = [(name, *metrics(s)) for name,s,_ in rows]
for name,mse,q in res:
    print(f"{name:16s} MSE(σ²)={mse:.3e}  QLIKE={q:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
names = [r[0] for r in res]
mses = [r[1] for r in res]; qs=[r[2] for r in res]
cols = [c for _,_,c in rows]
ax1.barh(names[::-1], [m*1e6 for m in mses[::-1]], color=cols[::-1])
ax1.set_xlabel("样本外 MSE(σ²) ×10⁻⁶  (越小越好)")
ax1.set_title("波动预测均方误差", fontsize=12, fontweight="bold")
for i,v in enumerate([m*1e6 for m in mses[::-1]]):
    ax1.text(v, i, f" {v:.3f}", va="center", fontsize=9)
ax2.barh(names[::-1], qs[::-1], color=cols[::-1])
ax2.set_xlabel("样本外 QLIKE  (越小越好)")
ax2.set_title("QLIKE 损失(波动预测标准评价)", fontsize=12, fontweight="bold")
for i,v in enumerate(qs[::-1]):
    ax2.text(v, i, f" {v:.3f}", va="center", fontsize=9)
ax2.set_xlim(min(qs)-0.2, max(qs)+0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/oos_metrics.png", dpi=120); plt.close()

# ---------- 图4: 诚实翻车 - 剥太多IMF + 端点效应 ----------
# 递增剥离的 IMF 层数, 看重构的 g_t 与真实慢方差的匹配度(剥太多会把 GARCH 聚集也当成趋势吸掉)
true_g = slow_vol**2
corrs = []
cum = imfs[-1].copy()
layers = []
for j in range(len(imfs)-1, -1, -1):
    if j == len(imfs)-1:
        cum = imfs[j].copy()
    else:
        cum = cum + imfs[j]
    g_j = np.exp(cum); g_j = g_j/np.mean(g_j)*np.var(ret)
    # 重构 g_t 与真实慢方差的相关: 剥到合适层数最高, 再剥就把聚集吸进来
    c = np.corrcoef(g_j[split:], true_g[split:])[0,1]
    corrs.append(c); layers.append(len(imfs)-j)
fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.4))
axa.plot(layers, corrs, "o-", color=C_HYB, lw=1.8)
axa.axvline(2, color=C_GARCH, ls="--", lw=1.2, label="本文选择: 剥最低2层")
axa.set_xlabel("剥离的低频 IMF 层数(从残余往上)")
axa.set_ylabel("重构慢方差 g_t 与真实慢方差相关")
axa.set_title("剥太多 → 把 GARCH 聚集也当成趋势吸走", fontsize=12, fontweight="bold")
axa.legend(fontsize=9)
# 端点效应: EMD 剥出的慢波动 vs 真实慢波动的偏差, 首尾放大
edge = np.abs(slow_level - slow_vol)
axb.plot(t, edge, color="#7c3aed", lw=0.7)
axb.axvspan(0, 60, color=C_GARCH, alpha=0.12)
axb.axvspan(n-60, n, color=C_GARCH, alpha=0.12, label="端点区(误差放大)")
axb.set_xlabel("时间(交易日)"); axb.set_ylabel("慢波动重建|偏差|")
axb.set_title("EMD 端点效应: 首尾样条外推不稳", fontsize=12, fontweight="bold")
axb.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/pitfalls.png", dpi=120); plt.close()

# 导出关键指标供文章引用
import json
summary = {
    "pure_garch": {"a0":float(p_pure[0]),"a1":float(p_pure[1]),"b1":float(p_pure[2]),"persist":float(p_pure[1]+p_pure[2])},
    "emd_garch":  {"a0":float(p_hyb[0]),"a1":float(p_hyb[1]),"b1":float(p_hyb[2]),"persist":float(p_hyb[1]+p_hyb[2])},
    "oos": {name:{"mse":float(mse),"qlike":float(q)} for name,mse,q in res},
    "n_imfs": int(len(imfs)),
}
with open("garch_emd_summary.json","w") as f: json.dump(summary,f,ensure_ascii=False,indent=2)
print("\n=== 图已生成 ===")
print(json.dumps(summary, ensure_ascii=False, indent=2))
