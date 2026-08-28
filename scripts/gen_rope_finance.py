#!/usr/bin/env python3
"""
为文章「Transformer 旋转位置编码金融改进：让周期与相位进入注意力」生成真实配图。

所有图表均由下文代码真实计算（纯 numpy + sklearn，无占位图）：

  1) rope_attention_heatmap.png  —— 绝对位置 / 默认RoPE / 周期调制RoPE 三种编码下
                                     query·key 内积热力图：默认RoPE只依赖相对距离(m-n)，
                                     周期调制RoPE额外呈现周期为 P 的等相位条纹。
  2) rope_cycle_fit.png          —— 周期目标 y_t=sin(2π t/P) 上，无位置 / 绝对线性位置 /
                                     周期RoPE 三种编码的线性回归：训练区 vs 外推区预测。
  3) rope_freq_sweep.png         —— 扫描 RoPE 频率 θ，外推区 RMSE 在真实周期频率 θ* 处呈 V 形谷。
  4) rope_multi_cycle.png        —— 日/周/月多周期叠加信号，多维 RoPE（每维配一组谐波频率）
                                     训练区+外推区的重建。

机制（合成数据，仅用于演示方法；落地见文末路径）：
  RoPE 对位置 t 的向量做旋转：q_t = R(θ t) q，R 为分块旋转矩阵，θ_i = base^{-2i/d}。
  内积 q_t·k_s = q·k·cos(θ(t-s))，天然只依赖相对距离 (t-s)，且 cos 周期结构把
  「相距 P 天 = 同相位」编码进注意力。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "rope-rotary-position-finance")
os.makedirs(D, exist_ok=True)

C = {"raw": "#9E9E9E", "rope": "#4C72B0", "gold": "#E1A100",
     "pos": "#55A868", "neg": "#C44E52", "abs": "#8172B3", "none": "#CCB974"}
rng = np.random.default_rng(20260828)

# ---------------------------------------------------------------------------
# 图1：三种位置编码下的 query·key 内积热力图
# ---------------------------------------------------------------------------
T = 40
P = 20
theta_def = 1.0            # 默认 RoPE 频率（d=2 时 base^0）
theta_cyc = 2 * np.pi / P  # 周期调制 RoPE 频率

M, N = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
abs_inner = M * N
rope_def_inner = np.cos(theta_def * (M - N))
rope_cyc_inner = np.cos(theta_cyc * (M - N))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
specs1 = [
    (abs_inner, "绝对位置嵌入\n内积 = m·n（依赖绝对位置，无相对距离结构）", "viridis"),
    (rope_def_inner, "默认 RoPE (θ=1)\n内积 = cos(m−n)（只依赖相对距离）", "RdBu_r"),
    (rope_cyc_inner, f"周期调制 RoPE (θ=2π/{P})\n内积 = cos(2π(m−n)/{P})（等相位条纹）", "RdBu_r"),
]
for ax, (mat, title, cmap) in zip(axes, specs1):
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("位置 n"); ax.set_ylabel("位置 m")
    ax.set_xticks([0, 10, 20, 30, 39]); ax.set_yticks([0, 10, 20, 30, 39])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(f"{D}/rope_attention_heatmap.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 图2：周期目标上的线性回归（无位置 / 绝对线性 / 周期RoPE）
# ---------------------------------------------------------------------------
P = 20
t = np.arange(400)
y = np.sin(2 * np.pi * t / P)
t_train = t[:200]
y_train = y[:200]
t_test = t[200:]
y_test = y[200:]

def fit_predict(Phi_tr, Phi_te, y_tr):
    w, *_ = np.linalg.lstsq(Phi_tr, y_tr, rcond=None)
    return Phi_tr @ w, Phi_te @ w

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

# 无位置：只有偏置
Phi_none_tr = np.ones((len(t_train), 1))
Phi_none_te = np.ones((len(t_test), 1))
pnone_tr, pnone_te = fit_predict(Phi_none_tr, Phi_none_te, y_train)

# 绝对线性位置：[1, t]
Phi_abs_tr = np.stack([np.ones_like(t_train, dtype=float), t_train], axis=1)
Phi_abs_te = np.stack([np.ones_like(t_test, dtype=float), t_test], axis=1)
pabs_tr, pabs_te = fit_predict(Phi_abs_tr, Phi_abs_te, y_train)

# 周期 RoPE：两个基 [cos(θt), sin(θt)]，θ=2π/P
th = 2 * np.pi / P
Phi_rope_tr = np.stack([np.cos(th * t_train), np.sin(th * t_train)], axis=1)
Phi_rope_te = np.stack([np.cos(th * t_test), np.sin(th * t_test)], axis=1)
prope_tr, prope_te = fit_predict(Phi_rope_tr, Phi_rope_te, y_train)

rmse_none = rmse(pnone_te, y_test)
rmse_abs = rmse(pabs_te, y_test)
rmse_rope = rmse(prope_te, y_test)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
panels = [
    (axes[0], pnone_tr, pnone_te, "无位置编码", C["none"], rmse_none),
    (axes[1], pabs_tr, pabs_te, "绝对线性位置", C["abs"], rmse_abs),
    (axes[2], prope_tr, prope_te, "周期 RoPE (θ=2π/P)", C["rope"], rmse_rope),
]
for ax, ptr, pte, name, col, r in panels:
    ax.plot(t_train, y_train, color=C["raw"], lw=1.0, label="真实（训练区）")
    ax.plot(t_train, ptr, color=col, lw=1.6, label="拟合（训练区）")
    ax.plot(t_test, y_test, color=C["raw"], lw=1.0, ls="--", label="真实（外推区）")
    ax.plot(t_test, pte, color=col, lw=1.6, ls="--", label="预测（外推区）")
    ax.axvline(199.5, color="k", lw=0.8, ls=":")
    ax.set_title(f"{name}\n外推 RMSE={r:.3f}", fontsize=10)
    ax.set_xlabel("时间步 t")
    ax.legend(fontsize=7, loc="upper right")
axes[0].set_ylabel("y_t = sin(2π t / 20)")
plt.tight_layout()
plt.savefig(f"{D}/rope_cycle_fit.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 图3：RoPE 频率扫描 —— 外推区 RMSE 在真周期频率处呈 V 形
# ---------------------------------------------------------------------------
th_grid = np.linspace(0.02, 0.6, 120)
rmses = []
for th in th_grid:
    Phi_tr = np.stack([np.cos(th * t_train), np.sin(th * t_train)], axis=1)
    Phi_te = np.stack([np.cos(th * t_test), np.sin(th * t_test)], axis=1)
    w, *_ = np.linalg.lstsq(Phi_tr, y_train, rcond=None)
    pred = Phi_te @ w
    rmses.append(rmse(pred, y_test))
rmses = np.array(rmses)
th_star = 2 * np.pi / P

fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.plot(th_grid, rmses, color=C["rope"], lw=1.8)
ax.axvline(th_star, color=C["gold"], lw=1.4, ls="--",
           label=f"真实周期频率 θ* = 2π/{P} ≈ {th_star:.3f}")
ax.set_title("RoPE 频率扫描：外推区 RMSE 在真周期频率处出现 V 形谷")
ax.set_xlabel("RoPE 频率 θ"); ax.set_ylabel("外推区 RMSE")
ax.legend(fontsize=9)
ax.annotate(f"谷底 RMSE={rmses.min():.3f}",
            xy=(th_star, rmses.min()), xytext=(th_star + 0.08, rmses.min() + 0.15),
            arrowprops=dict(arrowstyle="->", color="k"), fontsize=9)
plt.tight_layout()
plt.savefig(f"{D}/rope_freq_sweep.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 图4：日/周/月多周期叠加信号 + 多维 RoPE 重建
# ---------------------------------------------------------------------------
P_day, P_week, P_month = 5, 20, 60
w_day, w_week, w_month = 0.5, 0.3, 0.2
y_multi = (w_day * np.sin(2 * np.pi * t / P_day)
           + w_week * np.sin(2 * np.pi * t / P_week)
           + w_month * np.sin(2 * np.pi * t / P_month))
freqs = [2 * np.pi / P_day, 2 * np.pi / P_week, 2 * np.pi / P_month]
cols_ph = []
for fr in freqs:
    cols_ph.append(np.cos(fr * t))
    cols_ph.append(np.sin(fr * t))
Phi_m = np.stack(cols_ph, axis=1)
w_m, *_ = np.linalg.lstsq(Phi_m[:200], y_multi[:200], rcond=None)
recon = Phi_m @ w_m

train_err = rmse(recon[:200], y_multi[:200])
test_err = rmse(recon[200:], y_multi[200:])

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(t, y_multi, color=C["raw"], lw=1.0, label="真实多周期信号")
ax.plot(t, recon, color=C["rope"], lw=1.5, label="多维 RoPE 线性重建")
ax.axvline(199.5, color="k", lw=0.8, ls=":")
ax.set_title("日/周/月多周期叠加：多维 RoPE（每维一组谐波频率）训练区+外推区重建")
ax.set_xlabel("时间步 t"); ax.set_ylabel("信号值")
ax.legend(fontsize=9, loc="upper right")
ax.text(0.01, 0.06, f"训练区 RMSE={train_err:.4f}   外推区 RMSE={test_err:.4f}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec=C["rope"]))
plt.tight_layout()
plt.savefig(f"{D}/rope_multi_cycle.png", dpi=120)
plt.close()

print("=== RoPE 关键数字 ===")
print(f"图2 外推 RMSE: 无位置={rmse_none:.4f}  绝对线性位置={rmse_abs:.4f}  周期RoPE={rmse_rope:.4f}")
print(f"图3 频率扫描谷底 θ={th_star:.4f} RMSE={rmses.min():.4f}  两端(0.02)={rmses[0]:.4f} (0.6)={rmses[-1]:.4f}")
print(f"图4 多周期 训练区RMSE={train_err:.4f} 外推区RMSE={test_err:.4f}")
print("图片已写入:", D)
print(os.listdir(D))
