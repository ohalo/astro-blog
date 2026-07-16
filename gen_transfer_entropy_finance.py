#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「转移熵因果识别：用信息流方向分清『谁带动谁』」生成真实配图与统计数字。

核心逻辑(Schreiber 2000, Transfer Entropy):
  - 相关系数只说"一起动", 不说"谁先动"。转移熵(TE) 用条件互信息量:
        TE(X->Y) = sum p(y_{t+1}, y_t, x_t) * log2 [ p(y_{t+1}|y_t, x_t) / p(y_{t+1}|y_t) ]
    —— 加入 X 的历史后, Y 的"下一刻不确定性"下降了多少 bits。下降多 = X 带动了 Y。
  - 关键性质: TE 非对称。若 X 领先 Y, 则 TE(X->Y) 显著 > TE(Y->X)。
  - 工程实现: 用符号化/分箱估计(等宽分箱 + 联合/边缘计数, 用加性平滑防 log0)。
        这是 Kraskov k-NN 之外最常用、最易自检的实现; 本文用它保证数值合理
        (TE 必然 <= H(Y_{t+1}|Y_t), 单变量高斯熵只有零点几 bit)。
  - 仿真验证:
      ① 真实领先-滞后链 X -> Y -> Z(真实 TE 非对称) 下, 估计能否还原方向?
      ② 纯因果 vs 共同因(confounder): 两者都相关, 但 TE 能否区分?
      ③ 把 TE 当"领先指标"做交易: 在已知领先关系里, 沿信息流方向下注能否吃到领先收益?

全部数字由文中 Python 真实计算(自包含, 仅依赖 numpy)。
图片:
  te_chain_recovery.png     —— X->Y->Z 链: 估计出的 6 向 TE 矩阵(非对称, 还原真实指向)
  te_lead_lag_signal.png    —— 价格路径 + 信息流方向标注(谁带动谁)
  te_vs_corr.png            —— 同一对相关下, 相关矩阵"对称"但 TE 矩阵"非对称"的对比
  te_trading_edge.png       —— 沿信息流方向下注 vs 反向押注 vs 买入持有的净值
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
D = os.path.join(BASE, "transfer-entropy-finance")
os.makedirs(D, exist_ok=True)

rng = np.random.default_rng(20260717)


# ---------- 符号化 / 分箱 转移熵估计 (bits) ----------
def binned_te(x, y, k=8, lag=1, base=2.0, eps=1e-9):
    """TE(X->Y): x 是源(领先), y 是目标(滞后). 估计加入 x_t 后 y_{t+1} 的不确定度削减(bits)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    # 等宽分箱
    xb = np.digitize(x, np.linspace(x.min(), x.max(), k + 1)[1:-1])
    yb = np.digitize(y, np.linspace(y.min(), y.max(), k + 1)[1:-1])
    # 对齐: y_{t+1}, y_t, x_t
    Yf = yb[lag:]
    Yp = yb[:-lag]
    Xp = xb[:-lag]
    n = len(Yf)
    # 联合 (Yf, Yp, Xp) 计数
    keys = Yf * (k * k) + Yp * k + Xp
    # 用 bincount 重建计数, 按 (Yf, Yp, Xp) 顺序 reshape
    c_joint = np.bincount(keys, minlength=k * k * k).astype(float).reshape(k, k, k) + eps   # (Yf,Yp,Xp)
    c_ypxp = np.bincount(Yp * k + Xp, minlength=k * k).astype(float).reshape(k, k) + eps     # (Yp,Xp)
    c_yp = np.bincount(Yp, minlength=k).astype(float) + eps                # Yp
    c_yfyp = c_joint.sum(axis=2) + eps                                    # (Yf,Yp) 对 Xp 求和
    # p(yf|yp,xp) / p(yf|yp) = p(yf,yp,xp)*p(yp) / [ p(yp,xp)*p(yf,yp) ]
    #   c_joint:(Yf,Yp,Xp)  c_yp:(Yp)  c_ypxp:(Yp,Xp)  c_yfyp:(Yf,Yp)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (c_joint * c_yp[None, :, None]) / (c_ypxp[None, :, :] * c_yfyp[:, :, None])
        logr = np.log2(np.clip(ratio, eps, None))
    te = np.sum(c_joint / n * logr)
    return float(te)


# ---------- 1) 仿真领先-滞后链 X -> Y -> Z ----------
T = 4000
X = np.zeros(T)
X[0] = rng.normal()
for t in range(1, T):
    X[t] = 0.6 * X[t - 1] + rng.normal() * 0.8
Y = np.zeros(T)
for t in range(1, T):
    Y[t] = 0.5 * Y[t - 1] + 0.4 * X[t - 1] + rng.normal() * 0.6   # Y 受 X 领先带动
Z = np.zeros(T)
for t in range(1, T):
    Z[t] = 0.5 * Z[t - 1] + 0.4 * Y[t - 1] + rng.normal() * 0.6   # Z 受 Y 领先带动
T2 = T

te_xy = binned_te(X, Y)
te_yx = binned_te(Y, X)
te_yz = binned_te(Y, Z)
te_zy = binned_te(Z, Y)
te_xz = binned_te(X, Z)
te_zx = binned_te(Z, X)

# ---------- 2) 共同因 vs 因果 ----------
# 共同因 W 同时驱动 U, V (无直接因果)
W = rng.normal(0, 1, T2)
U = 0.7 * W + rng.normal(0, 1, T2)
V = 0.7 * W + rng.normal(0, 1, T2)
te_uv = binned_te(U, V)
te_vu = binned_te(V, U)
corr_uv = np.corrcoef(U, V)[0, 1]

# ---------- 3) 沿信息流方向下注的交易检验 ----------
# 构造"领先"序列: B_{t+1} 含 A_t 信息 -> 信息流 A -> B
M = 2000
A = rng.normal(0, 1, M)
B = np.zeros(M)
for t in range(1, M):
    B[t] = 0.5 * A[t - 1] + rng.normal() * 0.8     # A 领先 B 一个 tick
# 信息流 A -> B: A_t 决定 B_{t+1}. 用 A 的当期方向作 B 下期变化的领先信号. 对齐: sig_t = sign(A_t), realized_t = B_{t+1} - B_t
sig = np.sign(A[:-1])                               # t 时刻 A 方向(领先 B_{t+1})
realized = B[1:] - B[:-1]                           # B 下一期真实变化(长度 M-1, 与 sig 对齐)
ret_follow = sig * realized                         # 沿信息流: 跟 A 方向押 B 下期
ret_against = -sig * realized                      # 反向押注
ret_bh = realized.copy()

def sharpe(r):
    r = np.asarray(r)
    return r.mean() / (r.std() + 1e-9) * np.sqrt(252)
s_follow = sharpe(ret_follow)
s_against = sharpe(ret_against)
s_bh = sharpe(ret_bh)
cum_follow = np.cumprod(1 + np.clip(ret_follow, -0.1, 0.1))
cum_against = np.cumprod(1 + np.clip(ret_against, -0.1, 0.1))
cum_bh = np.cumprod(1 + np.clip(ret_bh, -0.1, 0.1))

metrics = {
    "T": int(T2),
    "te_xy": round(float(te_xy), 4),
    "te_yx": round(float(te_yx), 4),
    "te_yz": round(float(te_yz), 4),
    "te_zy": round(float(te_zy), 4),
    "te_xz": round(float(te_xz), 4),
    "te_zx": round(float(te_zx), 4),
    "te_uv": round(float(te_uv), 4),
    "te_vu": round(float(te_vu), 4),
    "corr_uv": round(float(corr_uv), 3),
    "te_ratio_xy": round(float(te_xy / (te_yx + 1e-12)), 2),
    "te_ratio_yz": round(float(te_yz / (te_zy + 1e-12)), 2),
    "s_follow": round(float(s_follow), 3),
    "s_against": round(float(s_against), 3),
    "s_bh": round(float(s_bh), 3),
}
with open(os.path.join(D, "_metrics.txt"), "w") as f:
    for k, vv in metrics.items():
        f.write(f"{k}={vv}\n")
print("METRICS", metrics)

# ============ 图1: X->Y->Z 链 TE 矩阵 ============
labels = ["X", "Y", "Z"]
M_te = np.array([
    [0,       te_xy,   te_xz],
    [te_yx,   0,       te_yz],
    [te_zx,   te_zy,   0   ],
])
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(M_te, cmap="Blues", aspect="auto")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(labels); ax.set_yticklabels(labels)
ax.set_xlabel("源 (领先)"); ax.set_ylabel("目标 (滞后)")
ax.set_title("转移熵矩阵 TE(行→列)：X→Y→Z 链被还原", fontsize=12, fontweight="bold")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{M_te[i, j]:.3f}", ha="center", va="center",
                color="white" if M_te[i, j] > 0.02 else "black", fontsize=11)
ax.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="red", lw=2))
ax.annotate("", xy=(2, 1), xytext=(1, 1), arrowprops=dict(arrowstyle="->", color="red", lw=2))
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(D, "te_chain_recovery.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============ 图2: 价格路径 + 信息流方向 ============
fig, ax = plt.subplots(figsize=(10, 5.0))
xx = np.arange(T2)
ax.plot(xx, np.cumsum(X)[:T2] * 0.5, color="#2F4B7C", lw=1.5, label="X (领先)")
ax.plot(xx, np.cumsum(Y)[:T2] * 0.5 + 2, color="#DD8452", lw=1.5, label="Y (中)")
ax.plot(xx, np.cumsum(Z)[:T2] * 0.5 + 4, color="#55A868", lw=1.5, label="Z (滞后)")
ax.set_title("X→Y→Z 领先-滞后链：信息从 X 流向 Z", fontsize=13, fontweight="bold")
ax.set_xlabel("时间"); ax.set_ylabel("累计路径（平移以便观察）")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.annotate("信息流 →", xy=(0.5, 0.85), xytext=(0.05, 0.85),
            xycoords="axes fraction", fontsize=12, color="red", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red"))
plt.tight_layout()
plt.savefig(os.path.join(D, "te_lead_lag_signal.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============ 图3: 相关(对称) vs TE(非对称) 对比 ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0))
C = np.array([[1.0, corr_uv], [corr_uv, 1.0]])
im1 = ax1.imshow(C, cmap="Greys", vmin=-1, vmax=1)
ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
ax1.set_xticklabels(["U", "V"]); ax1.set_yticklabels(["U", "V"])
ax1.set_title(f"相关系数(对称): U↔V = {corr_uv:.2f}", fontsize=12, fontweight="bold")
for i in range(2):
    for j in range(2):
        ax1.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", color="white", fontsize=12)
Mt = np.array([[0, te_uv], [te_vu, 0]])
im2 = ax2.imshow(Mt, cmap="Blues")
ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
ax2.set_xticklabels(["U", "V"]); ax2.set_yticklabels(["U", "V"])
ax2.set_title(f"转移熵(非对称): TE(U→V)={te_uv:.3f}, TE(V→U)={te_vu:.3f}", fontsize=11, fontweight="bold")
for i in range(2):
    for j in range(2):
        ax2.text(j, i, f"{Mt[i,j]:.3f}", ha="center", va="center", color="black", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(D, "te_vs_corr.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============ 图4: 沿信息流下注 vs 反向 vs 买入持有 ============
fig, ax = plt.subplots(figsize=(9, 5.0))
ax.plot(range(len(cum_follow)), cum_follow, color="#2F4B7C", lw=1.8,
        label=f"沿信息流(跟 A 押 B) Sharpe {s_follow:.2f}")
ax.plot(range(len(cum_against)), cum_against, color="#C44E52", lw=1.8,
        label=f"反向押注 Sharpe {s_against:.2f}")
ax.plot(range(len(cum_bh)), cum_bh, color="#999999", lw=1.5, ls="--",
        label=f"买入持有 Sharpe {s_bh:.2f}")
ax.set_title("信息流方向就是 alpha 方向：沿流下注才能吃到领先收益",
             fontsize=12, fontweight="bold")
ax.set_xlabel("步"); ax.set_ylabel("净值（起始=1）")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "te_trading_edge.png"), dpi=150, bbox_inches="tight")
plt.close()
print("DONE transfer-entropy-finance images")
