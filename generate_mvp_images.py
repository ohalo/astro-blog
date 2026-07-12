#!/usr/bin/env python3
"""
为文章「最小方差组合：把估计误差也当成风险来管理」(minimum-variance-portfolio)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

设定(自洽合成, 仅用于演示方法):
  * 真实协方差 Sigma_true = B B' + D  (B 为因子载荷, D 为异质方差), 固定不随试验变
  * 资产数 N=25, 每次试验从 N(0, Sigma_true) 抽 T 条历史观测, 估计样本协方差 S
  * 估计误差: 当 T 与 N 可比时, S 病态(存在伪小特征值), S^-1 把噪声放大 ->
        直接用 S 算最小方差组合会得到极端权重, 样本外真实方差暴涨
  * 对比四种组合权重求解:
        (1) 样本最小方差(S 直接求逆)         —— 被估计误差坑最惨
        (2) Ledoit-Wolf 风格相关性收缩(S->F)  —— 把估计误差当风险管
        (3) 岭正则最小方差(S + λI 求逆)     —— 显式惩罚极端权重
        (4)  oracle(用 Sigma_true 求逆)        —— 不可达下界, 作参照
  * 评价指标: 样本外真实方差 = w' Sigma_true w  (越小越好)
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
D = os.path.join(BASE, "minimum-variance-portfolio")
os.makedirs(D, exist_ok=True)

C = {"eq": "#4C72B0", "vix": "#C44E52", "grid": "#DDDDDD", "fv": "#55A868",
     "rv": "#DD8452", "hedge": "#9467bd", "thr": "#888888", "green": "#2ca02c",
     "orange": "#FF7F0E", "blue": "#1f77b4"}

rng0 = np.random.default_rng(20240713)
N_AST = 25
B = rng0.normal(0.0, 1.0, size=(N_AST, 3))
Ddiag = np.diag(rng0.uniform(0.5, 1.5, N_AST))
SIGMA_TRUE = B @ B.T + Ddiag          # 真实协方差(固定)
ONES = np.ones(N_AST)

def mvp(Sigma):
    """无约束最小方差组合: w = Sigma^-1 1 / (1' Sigma^-1 1)。用伪逆保数值稳定。"""
    w = np.linalg.pinv(Sigma) @ ONES
    return w / w.sum()

def const_corr_target(S):
    """Ledoit-Wolf 常相关性目标矩阵 F: 对角不变, 非对角 = 平均相关 × √(var_i var_j)。"""
    n = S.shape[0]
    sd = np.sqrt(np.diag(S))
    corr = S / np.outer(sd, sd)
    rho = np.mean([corr[i, j] for i in range(n) for j in range(i + 1, n)])
    F = np.diag(np.diag(S)).astype(float).copy()
    for i in range(n):
        for j in range(i + 1, n):
            F[i, j] = F[j, i] = rho * sd[i] * sd[j]
    return F

def oos_var(w):
    return float(w @ SIGMA_TRUE @ w)

# ============================================================
# 单次试验: 返回四种方法的 (权重, 样本外方差, 权重集中度H, 选中的δ/λ)
# ============================================================
def trial(T, seed):
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(np.zeros(N_AST), SIGMA_TRUE, size=T)
    S = np.cov(X, rowvar=False)
    F = const_corr_target(S)
    folds = 5
    idx = rng.permutation(T)
    chunks = np.array_split(idx, folds)
    # (1) 样本最小方差
    w_s = mvp(S)
    # (4) oracle
    w_o = mvp(SIGMA_TRUE)
    # (2) LW 收缩: δ 用 5 折交叉验证按样本外方差最小化来定
    dgrid = np.linspace(0.0, 1.0, 11)
    cv_scores = np.zeros(len(dgrid))
    for fi, d in enumerate(dgrid):
        sc = 0.0
        for k in range(folds):
            tr = X[np.concatenate([chunks[j] for j in range(folds) if j != k])]
            va = X[chunks[k]]
            Sr = np.cov(tr, rowvar=False)
            Sv = np.cov(va, rowvar=False)
            w = mvp(d * const_corr_target(Sr) + (1.0 - d) * Sr)
            sc += w @ Sv @ w
        cv_scores[fi] = sc / folds
    dstar = dgrid[int(np.argmin(cv_scores))]
    w_lw = mvp(dstar * F + (1.0 - dstar) * S)
    # (3) 岭正则: λ 用同样的 CV 定
    lgrid = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0])
    cv_scores_l = np.zeros(len(lgrid))
    for fi, lam in enumerate(lgrid):
        sc = 0.0
        for k in range(folds):
            tr = X[np.concatenate([chunks[j] for j in range(folds) if j != k])]
            va = X[chunks[k]]
            Sr = np.cov(tr, rowvar=False)
            Sv = np.cov(va, rowvar=False)
            w = mvp(Sr + lam * np.eye(N_AST))
            sc += w @ Sv @ w
        cv_scores_l[fi] = sc / folds
    lamstar = lgrid[int(np.argmin(cv_scores_l))]
    w_r = mvp(S + lamstar * np.eye(N_AST))
    H = lambda w: float(np.sum(w ** 2))
    return {
        "w_s": w_s, "w_lw": w_lw, "w_r": w_r, "w_o": w_o,
        "v_s": oos_var(w_s), "v_lw": oos_var(w_lw), "v_r": oos_var(w_r), "v_o": oos_var(w_o),
        "H_s": H(w_s), "H_lw": H(w_lw), "H_r": H(w_r), "H_o": H(w_o),
        "dstar": dstar, "lamstar": lamstar, "S": S, "F": F,
    }

# ============================================================
# 图1: 单次试验三种方法的排序后权重(降序) —— 样本法少数资产权重飙升
# ============================================================
r1 = trial(30, 12345)   # T=30: 25 资产只用 30 条观测, 估计误差极端 -> 权重极端集中
ws = np.sort(np.abs(r1["w_s"]))[::-1]
wl = np.sort(np.abs(r1["w_lw"]))[::-1]
wr = np.sort(np.abs(r1["w_r"]))[::-1]
xs = np.arange(1, N_AST + 1)
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(xs, ws, color=C["vix"], lw=2.2, marker="o", ms=3, label="样本最小方差(直接 S 求逆)")
ax.plot(xs, wl, color=C["blue"], lw=2.2, marker="s", ms=3, label="LW 收缩协方差")
ax.plot(xs, wr, color=C["fv"], lw=2.2, marker="^", ms=3, label="岭正则 (S+λI)")
ax.axhline(1.0 / N_AST, color=C["thr"], ls="--", lw=1.2, label="等权基准 1/N")
ax.set_xlabel("资产序号(按 |权重| 降序排列)")
ax.set_ylabel("|组合权重| |w_i|")
ax.set_title("估计误差的代价(T=30): 样本最小方差把权重压到极少数资产")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "mvp_weights.png"), dpi=130)
plt.close()

# ============================================================
# 图2: 样本外方差箱线图(多试验) —— 样本法最差, 收缩/岭更接近 oracle
# ============================================================
M = 500
Vs, Vlw, Vr, Vo = [], [], [], []
Hs, Hlw, Hr = [], [], []
Ms, Mlw, Mr = [], [], []
for i in range(M):
    r = trial(120, 20000 + i)
    Vs.append(r["v_s"]); Vlw.append(r["v_lw"]); Vr.append(r["v_r"]); Vo.append(r["v_o"])
    Hs.append(r["H_s"]); Hlw.append(r["H_lw"]); Hr.append(r["H_r"])
    Ms.append(np.max(np.abs(r["w_s"]))); Mlw.append(np.max(np.abs(r["w_lw"]))); Mr.append(np.max(np.abs(r["w_r"])))
Vs, Vlw, Vr, Vo = map(np.array, (Vs, Vlw, Vr, Vo))
Hs, Hlw, Hr = map(np.array, (Hs, Hlw, Hr))
Ms, Mlw, Mr = map(np.array, (Ms, Mlw, Mr))
data = [Vs, Vlw, Vr, Vo]
labels = ["样本最小方差", "LW 收缩", "岭正则", "Oracle(不可达)"]
fig, ax = plt.subplots(figsize=(9.5, 5.4))
bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.6)
cols = [C["vix"], C["blue"], C["fv"], C["thr"]]
for patch, c in zip(bp["boxes"], cols):
    patch.set_facecolor(c); patch.set_alpha(0.55)
for med in bp["medians"]:
    med.set_color("black")
ax.set_ylabel("样本外真实方差  w'Σ_true w")
ax.set_title("把估计误差当风险管理后: 样本外方差显著下降(样本法最差)")
ax.grid(True, color=C["grid"], lw=0.6, axis="y")
ax.text(1, Vs.mean() * 1.02, "均值 %.3f" % Vs.mean(), ha="center", fontsize=8, color=C["vix"])
ax.text(2, Vlw.mean() * 1.02, "均值 %.3f" % Vlw.mean(), ha="center", fontsize=8, color=C["blue"])
ax.text(3, Vr.mean() * 1.02, "均值 %.3f" % Vr.mean(), ha="center", fontsize=8, color=C["fv"])
ax.text(4, Vo.mean() * 1.02, "均值 %.3f" % Vo.mean(), ha="center", fontsize=8, color=C["thr"])
plt.tight_layout()
plt.savefig(os.path.join(D, "mvp_oos_box.png"), dpi=130)
plt.close()

# ============================================================
# 图3: 样本外方差 vs 训练样本量 T —— 样本法随 T 增大追上, 但小样本时差距骇人
# ============================================================
Ts = [30, 60, 120, 250, 500]
mean_s, mean_lw, mean_r, mean_o = [], [], [], []
for T in Ts:
    Vs_, Vlw_, Vr_, Vo_ = [], [], [], []
    for i in range(150):
        r = trial(T, 30000 + T * 1000 + i)
        Vs_.append(r["v_s"]); Vlw_.append(r["v_lw"]); Vr_.append(r["v_r"]); Vo_.append(r["v_o"])
    mean_s.append(np.mean(Vs_)); mean_lw.append(np.mean(Vlw_))
    mean_r.append(np.mean(Vr_)); mean_o.append(np.mean(Vo_))
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.plot(Ts, mean_s, color=C["vix"], marker="o", lw=2.2, label="样本最小方差")
ax.plot(Ts, mean_lw, color=C["blue"], marker="s", lw=2.2, label="LW 收缩")
ax.plot(Ts, mean_r, color=C["fv"], marker="^", lw=2.2, label="岭正则")
ax.plot(Ts, mean_o, color=C["thr"], marker="d", lw=2.2, ls="--", label="Oracle(不可达下界)")
ax.set_xscale("log", base=2)
ax.set_xlabel("训练样本量 T (交易日, log2 刻度)")
ax.set_ylabel("平均样本外方差")
ax.set_title("样本越少, 估计误差越致命: 小样本下收缩/岭远胜样本法")
ax.legend(fontsize=8.5); ax.grid(True, color=C["grid"], lw=0.6)
for x, y in zip(Ts, mean_s):
    ax.text(x, y * 1.03, "%.2f" % y, ha="center", fontsize=7.5, color=C["vix"])
for x, y in zip(Ts, mean_lw):
    ax.text(x, y * 1.03, "%.2f" % y, ha="center", fontsize=7.5, color=C["blue"])
for x, y in zip(Ts, mean_r):
    ax.text(x, y * 1.03, "%.2f" % y, ha="center", fontsize=7.5, color=C["fv"])
plt.tight_layout()
plt.savefig(os.path.join(D, "mvp_vs_T.png"), dpi=130)
plt.close()

# ============================================================
# 图4: 单次试验, 样本外方差 vs 收缩强度 δ —— U 形, 存在最优收缩
# ============================================================
r4 = trial(120, 54321)
S4, F4 = r4["S"], r4["F"]
dgrid = np.linspace(0.0, 1.0, 41)
vs_d = [oos_var(mvp(d * F4 + (1.0 - d) * S4)) for d in dgrid]
best = int(np.argmin(vs_d))
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.plot(dgrid, vs_d, color=C["blue"], lw=2.2, label="样本外方差 vs δ")
ax.axvline(dgrid[best], color=C["vix"], ls="--", lw=1.4,
            label="最优收缩 δ*≈%.2f" % dgrid[best])
ax.axvline(0.0, color=C["thr"], ls=":", lw=1.2, label="δ=0 样本最小方差(最差)")
ax.axvline(1.0, color=C["green"], ls=":", lw=1.2, label="δ=1 全收缩到目标(也差)")
ax.set_xlabel("收缩强度 δ (0=纯样本, 1=全收缩到目标)")
ax.set_ylabel("样本外真实方差  w'Σ_true w")
ax.set_title("收缩是门艺术: 太少(过拟合)与太多(欠拟合)都糟, 中间有最优")
ax.legend(fontsize=8.5); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "mvp_shrinkage_curve.png"), dpi=130)
plt.close()

# ============================================================
# 关键数字输出
# ============================================================
print("=== 最小方差组合 关键数字 (N=%d 资产, 试验数 M=%d, T=120) ===" % (N_AST, M))
print("样本外方差均值: 样本=%.4f  LW=%.4f  岭=%.4f  Oracle=%.4f"
      % (Vs.mean(), Vlw.mean(), Vr.mean(), Vo.mean()))
print("相对 Oracle 的超额方差: 样本 +%.1f%%  LW +%.1f%%  岭 +%.1f%%"
      % ((Vs.mean()/Vo.mean()-1)*100, (Vlw.mean()/Vo.mean()-1)*100, (Vr.mean()/Vo.mean()-1)*100))
print("权重集中度 Herfindahl(越大越集中): 样本=%.2f  LW=%.2f  岭=%.2f  等权=%.3f"
      % (Hs.mean(), Hlw.mean(), Hr.mean(), 1.0/N_AST))
print("最大单资产权重 |w|_max 均值: 样本=%.2f  LW=%.2f  岭=%.2f"
      % (Ms.mean(), Mlw.mean(), Mr.mean()))
print("--- 样本外方差 vs 训练样本量 T ---")
for T, s, l, rr, o in zip(Ts, mean_s, mean_lw, mean_r, mean_o):
    print("  T=%4d : 样本=%.3f  LW=%.3f  岭=%.3f  Oracle=%.3f" % (T, s, l, rr, o))
print("图4 单次试验最优收缩 δ* = %.2f (样本外方差由 %.3f 降到 %.3f)"
      % (dgrid[best], vs_d[0], min(vs_d)))
print("\n图片已保存到:", D)
