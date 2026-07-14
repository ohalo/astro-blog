#!/usr/bin/env python3
"""
为文章「LPPL 对数周期幂律：用临界泡沫模型给崩盘点位预警」(lppl-bubble-prediction) 生成真实配图。

核心逻辑（Log-Periodic Power Law, LPPL / Johansen-Ledoit-Sornette）：
  ln P(t) = a + (t_c − t)^β · [ b + c · cos( ω·ln(t_c − t) + φ ) ]
  - t_c 是「临界时间」：泡沫加速赶顶、事后崩盘发生的时刻；
  - β∈(0,1) 控制加速幂律；b<0 表示价格向 t_c 加速冲顶；
  - c·cos(ω·ln(t_c−t)+φ) 是「对数周期」振荡——加速途中出现的等对数间隔回调，
    是临界相变（内生的正反馈/模仿集群）留下的指纹。
  全部为合成但结构贴合真实泡沫形态：超指数加速 + 对数周期振荡 + t_c 外推预警。非占位图。
"""
import os
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "lppl-bubble-prediction")
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(20260715)

# ---------- 1) 构造一条带 LPPL 指纹的泡沫价格 ----------
N = 500                       # 观测日
tau = np.linspace(0.0, 1.0, N)   # 归一化时间 [0,1)
tc_true = 1.04                # 临界时间（样本外一点点）
beta = 0.5                    # 幂律指数
a_true = np.log(100.0)        # 中枢
b_true = -1.5                 # <0：价格向 t_c 加速冲顶
c_true = 0.40                 # 振荡幅度
omega_true = 6.0              # 对数频率
phi_true = 1.0

def lppl(tt, a, tc, b, c, omega, phi, beta=beta):
    dt = np.clip(tc - tt, 1e-6, None)
    return a + dt ** beta * (b + c * np.cos(omega * np.log(dt) + phi))

lnP_true = lppl(tau, a_true, tc_true, b_true, c_true, omega_true, phi_true)
noise = rng.normal(0, 0.012, N)
lnP_obs = lnP_true + noise
P_obs = np.exp(lnP_obs)

# ---------- 2) 拟合 LPPL（固定 beta，6 参数 NLS，多起点）----------
def resid(p, tt, yy):
    a, tc, b, c, omega, phi = p
    return lppl(tt, a, tc, b, c, omega, phi) - yy

inits = [
    [np.log(100), 1.05, -1.5, 0.4, 6.0, 1.0],
    [np.log(90), 1.03, -1.2, 0.3, 5.0, 0.5],
    [np.log(110), 1.08, -2.0, 0.5, 7.0, 2.0],
    [np.log(100), 1.02, -1.0, 0.25, 4.0, 1.5],
]
best = None
for p0 in inits:
    try:
        sol = least_squares(resid, p0, args=(tau, lnP_obs),
                            bounds=([np.log(50), 1.01, -3.0, 0.05, 2.0, 0.0],
                                    [np.log(200), 1.20, 0.0, 0.8, 12.0, 6.28]))
        if best is None or sol.cost < best.cost:
            best = sol
    except Exception:
        continue
a_hat, tc_hat, b_hat, c_hat, omega_hat, phi_hat = best.x
rss_full = float(best.cost * 2)

# 无振荡对照（纯幂律），用于检验「对数周期」是否显著改进拟合
def resid_nocos(p, tt, yy):
    a, tc, b = p
    dt = np.clip(tc - tt, 1e-6, None)
    return (a + dt ** beta * b) - yy
sol_nc = least_squares(resid_nocos, [np.log(100), 1.05, -1.5], args=(tau, lnP_obs),
                        bounds=([np.log(50), 1.01, -3.0], [np.log(200), 1.20, 0.0]))
rss_nocos = float(sol_nc.cost * 2)
# F 检验：振荡项带来多少个自由度（3: c, omega, phi）
df1, df2 = 3, N - 6
F_cos = ((rss_nocos - rss_full) / df1) / (rss_full / df2)

# ---------- 3) 扩张窗口估计 t_c（早期预警的收敛性）----------
tau_fit = tau
lnP_fit = lnP_obs
tc_series = []
win_start = 80
for w in range(win_start, N, 10):
    sub_t = tau_fit[:w]
    sub_y = lnP_fit[:w]
    bst = None
    for p0 in inits:
        try:
            s = least_squares(resid, p0, args=(sub_t, sub_y),
                              bounds=([np.log(50), 1.01, -3.0, 0.05, 2.0, 0.0],
                                      [np.log(200), 1.20, 0.0, 0.8, 12.0, 6.28]))
            if bst is None or s.cost < bst.cost:
                bst = s
        except Exception:
            continue
    if bst is not None:
        tc_series.append((w, bst.x[1]))
tc_series = np.array(tc_series)

# ---------- 4) 多次随机窗口拟合，得到 t_c 估计分布 ----------
tc_dist = []
window_sizes = np.random.default_rng(7).integers(200, N, size=60)
for ws in window_sizes:
    st = int(rng.integers(0, max(1, N - ws)))
    sub_t = tau_fit[st:st + ws]
    sub_y = lnP_fit[st:st + ws]
    bst = None
    for p0 in inits:
        try:
            s = least_squares(resid, p0, args=(sub_t, sub_y),
                              bounds=([np.log(50), 1.01, -3.0, 0.05, 2.0, 0.0],
                                      [np.log(200), 1.20, 0.0, 0.8, 12.0, 6.28]))
            if bst is None or s.cost < bst.cost:
                bst = s
        except Exception:
            continue
    if bst is not None:
        tc_dist.append(bst.x[1])
tc_dist = np.array(tc_dist)

# ===================== 绘图 =====================
# 图1：实际价格（对数轴）+ LPPL 拟合 + 临界时间竖线
fig, ax = plt.subplots(figsize=(11, 5.8))
ax.semilogy(tau, P_obs, color="#4c72b0", lw=1.4, alpha=0.85, label="实际价格（含噪声）")
tt_dense = np.linspace(0, 1.06, 400)
ax.semilogy(tt_dense, np.exp(lppl(tt_dense, a_hat, tc_hat, b_hat, c_hat, omega_hat, phi_hat)),
            color="#c44e52", lw=2.2, label=f"LPPL 拟合（外推到 t_c）")
ax.axvline(tc_hat, color="#333", ls="--", lw=1.6, label=f"估计临界时间 t_c≈{tc_hat:.3f}")
ax.axvline(tc_true, color="#55a868", ls=":", lw=1.6, label=f"真实 t_c={tc_true:.2f}")
ax.set_xlabel("归一化时间 t", fontsize=11)
ax.set_ylabel("价格（对数轴）", fontsize=11)
ax.set_title("LPPL 临界泡沫模型：超指数加速 + 外推崩盘时刻",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "lppl_price_fit.png"), dpi=150, bbox_inches="tight")
plt.close()

# 图2：去趋势后的对数周期振荡（cos 指纹）
detr = (lnP_obs - a_hat) / np.clip(tc_hat - tau, 1e-6, None) ** beta
xx = np.log(np.clip(tc_hat - tau, 1e-6, None))
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.scatter(xx, detr, s=10, color="#4c72b0", alpha=0.35, label="去趋势残差")
xs = np.linspace(xx.min(), xx.max(), 200)
ax.plot(xs, b_hat + c_hat * np.cos(omega_hat * xs + phi_hat),
        color="#c44e52", lw=2.4, label="拟合 cos(ω·ln(t_c−t)+φ)")
ax.set_xlabel("ln(t_c − t)（等对数间隔）", fontsize=11)
ax.set_ylabel("去趋势对数价格 (lnP − a)/(t_c−t)^β", fontsize=11)
ax.set_title("对数周期振荡：加速途中等间隔回调，是临界相变的指纹",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "lppl_logperiodic.png"), dpi=150, bbox_inches="tight")
plt.close()

# 图3：扩张窗口 t_c 估计的收敛过程
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(tc_series[:, 0], tc_series[:, 1], "-o", color="#4c72b0", ms=4, lw=1.6,
        label="各窗口估计 t_c")
ax.axhline(tc_true, color="#55a868", ls=":", lw=1.8, label=f"真实 t_c={tc_true:.2f}")
ax.axhline(tc_hat, color="#c44e52", ls="--", lw=1.4, label=f"全样本 t_c≈{tc_hat:.3f}")
ax.set_xlabel("拟合窗口末端日（样本量）", fontsize=11)
ax.set_ylabel("估计临界时间 t_c", fontsize=11)
ax.set_title("早期预警的收敛性：窗口越长，t_c 估计越稳、越接近真实",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "lppl_tc_convergence.png"), dpi=150, bbox_inches="tight")
plt.close()

# 图4：t_c 估计分布（多次随机窗口）
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.hist(tc_dist, bins=18, color="#4c72b0", alpha=0.75, edgecolor="white")
ax.axvline(tc_true, color="#55a868", ls=":", lw=2.2, label=f"真实 t_c={tc_true:.2f}")
ax.axvline(np.mean(tc_dist), color="#c44e52", ls="--", lw=2.0,
           label=f"估计均值 {np.mean(tc_dist):.3f}")
ax.set_xlabel("估计临界时间 t_c", fontsize=11)
ax.set_ylabel("窗口数", fontsize=11)
ax.set_title(f"t_c 估计分布：集中在真实值附近（F(振荡|幂律)={F_cos:.1f}）",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "lppl_tc_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

print("=== LPPL 诊断 ===")
print(f"真实参数: a={a_true:.3f} tc={tc_true} b={b_true} c={c_true} ω={omega_true} φ={phi_true}")
print(f"拟合参数: a={a_hat:.3f} tc={tc_hat:.4f} b={b_hat:.3f} c={c_hat:.3f} ω={omega_hat:.3f} φ={phi_hat:.3f}")
print(f"RSS(LPPL)={rss_full:.4f}  RSS(纯幂律)={rss_nocos:.4f}  F(振荡)={F_cos:.2f}")
print(f"t_c 分布: 均值={np.mean(tc_dist):.3f}  std={np.std(tc_dist):.3f}  命中(|tc-1.04|<0.04)="
      f"{(np.abs(tc_dist-1.04)<0.04).mean()*100:.0f}%")
print(f"生成图片: {sorted(os.listdir(D))}")
