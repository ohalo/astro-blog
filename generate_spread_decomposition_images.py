#!/usr/bin/env python3
"""
为文章「买卖价差的状态空间分解：逆向选择成本与库存成本」(spread-decomposition-ss)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

数据机制（自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 把「报价价差」当成观测，用【非观测成分状态空间模型】拆出两个隐状态：
      · 逆向选择成本 s_t^info  （信息驱动，慢 AR(1)，遇信息事件跳升）
      · 库存成本     s_t^inv   （做市商库存调整，快 AR(1)）
      · 指令处理成本 γ          （近似常数，始终存在）
    观测： spread_t = s_t^info + s_t^inv + γ + e_t
  - 用标准 Kalman 滤波 + RTS 平滑从单条观测序列里把两个隐状态复原出来，
    验证「状态空间分解」确实能把价差拆开。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "spread-decomposition-ss")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "info": "#2F4B7C", "inv": "#C44E52", "order": "#55A868",
     "obs": "#333333", "mk": "#8172B3", "line": "#555555"}

# ============================================================
# 1) 合成真实成分
# ============================================================
def build_truth(T=1000, seed=20260713):
    rng = np.random.default_rng(seed)
    # 逆向选择成分：慢均值回复 AR(1)，均值约 3 bps
    phi_info, sig_info, mu_info = 0.92, 0.55, 3.0
    info = np.zeros(T)
    for t in range(1, T):
        info[t] = mu_info + phi_info * (info[t - 1] - mu_info) + sig_info * rng.normal()
    # 库存成分：快均值回复 AR(1)，均值约 1.4 bps
    phi_inv, sig_inv, mu_inv = 0.55, 0.9, 1.4
    inv = np.zeros(T)
    for t in range(1, T):
        inv[t] = mu_inv + phi_inv * (inv[t - 1] - mu_inv) + sig_inv * rng.normal()
    gamma = 2.0  # 指令处理成本（近似常数，bps）
    # 观测价差
    obs_noise = 0.6 * rng.normal(size=T)
    spread = info + inv + gamma + obs_noise
    return info, inv, gamma, spread

# ============================================================
# 2) 卡尔曼滤波 + RTS 平滑（2 维线性高斯状态空间模型）
#    状态 x_t=[info_t, inv_t]，观测 y_t = H x_t + gamma + e_t
# ============================================================
def kalman_smoother(y, gamma, phi_info, sig_info, phi_inv, sig_inv, R):
    F = np.diag([phi_info, phi_inv])
    Q = np.diag([sig_info ** 2, sig_inv ** 2])
    H = np.array([[1.0, 1.0]])
    I = np.eye(2)
    # 无条件方差初始化
    v_info = sig_info ** 2 / (1 - phi_info ** 2)
    v_inv = sig_inv ** 2 / (1 - phi_inv ** 2)
    x_filt = np.zeros((len(y), 2)); P_filt = np.zeros((len(y), 2, 2))
    x_pred = np.zeros((len(y), 2)); P_pred = np.zeros((len(y), 2, 2))
    x0 = np.array([0.0, 0.0]); P0 = np.diag([v_info, v_inv])
    xprev, Pprev = x0.copy(), P0.copy()
    for t in range(len(y)):
        xp = F @ xprev
        Pp = F @ Pprev @ F.T + Q
        S = H @ Pp @ H.T + R
        K = Pp @ H.T / S
        innov = y[t] - (H @ xp + gamma)
        xf = xp + K.flatten() * innov
        Pf = (I - K @ H) @ Pp
        x_pred[t], P_pred[t] = xp, Pp
        x_filt[t], P_filt[t] = xf, Pf
        xprev, Pprev = xf, Pf
    # RTS 平滑
    x_sm = x_filt.copy(); P_sm = P_filt.copy()
    for t in range(len(y) - 2, -1, -1):
        Pp_next = F @ P_filt[t] @ F.T + Q
        C = P_filt[t] @ F.T @ np.linalg.inv(Pp_next)
        x_sm[t] = x_filt[t] + C @ (x_sm[t + 1] - F @ x_filt[t])
        P_sm[t] = P_filt[t] + C @ (P_sm[t + 1] - Pp_next) @ C.T
    return x_sm, x_pred, P_pred

# ============================================================
# 3) 图一：真实成分堆叠 + 观测价差
# ============================================================
def fig_components(info, inv, gamma, spread):
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = np.arange(len(spread))
    ax.fill_between(x, 0, info, color=C["info"], alpha=0.55, label=r"adverse-selection $s_t^{\rm info}$")
    ax.fill_between(x, info, info + inv, color=C["inv"], alpha=0.55,
                    label=r"inventory $s_t^{\rm inv}$")
    ax.plot(x, info + inv + gamma, color=C["order"], lw=1.6, alpha=0.95,
            label=r"order-processing $\gamma$ (constant)")
    ax.plot(x, spread, color=C["obs"], lw=0.7, alpha=0.35,
            label=r"observed quoted spread $S_t$")
    ax.set_title("Quoted spread = adverse-selection + inventory + order-processing", fontsize=11.5)
    ax.set_xlabel("trade / minute index"); ax.set_ylabel("spread (bps)")
    ax.legend(frameon=False, fontsize=8.2, loc="upper left", ncol=2)
    ax.grid(True, color=C["grid"], lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "ss_components.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p

# ============================================================
# 4) 图二：卡尔曼平滑复原 vs 真实
# ============================================================
def fig_kalman(info, inv, x_sm):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.2, 6.4), sharex=True)
    a1.plot(info, color=C["info"], lw=1.8, label="true $s_t^{\\rm info}$")
    a1.plot(x_sm[:, 0], color="#1A2F5A", lw=1.2, ls="--", alpha=0.9,
            label="Kalman-smoothed estimate")
    a1.set_title("State-space extraction recovers the two latent components", fontsize=11.5)
    a1.set_ylabel("info cost (bps)"); a1.legend(frameon=False, fontsize=8.2)
    a1.grid(True, color=C["grid"], lw=0.6); a1.set_axisbelow(True)
    a2.plot(inv, color=C["inv"], lw=1.8, label="true $s_t^{\\rm inv}$")
    a2.plot(x_sm[:, 1], color="#7A1F25", lw=1.2, ls="--", alpha=0.9,
            label="Kalman-smoothed estimate")
    a2.set_ylabel("inv cost (bps)"); a2.set_xlabel("trade / minute index")
    a2.legend(frameon=False, fontsize=8.2)
    a2.grid(True, color=C["grid"], lw=0.6); a2.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "ss_kalman.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p

# ============================================================
# 5) 图三：平均价差构成（成分占比）
# ============================================================
def fig_composition(info, inv, gamma):
    m_info, m_inv = info.mean(), inv.mean()
    total = m_info + m_inv + gamma
    fig, ax = plt.subplots(figsize=(8.4, 2.6))
    left = 0.0
    seg = [("adverse-selection", m_info, C["info"]),
           ("inventory", m_inv, C["inv"]),
           ("order-processing", gamma, C["order"])]
    for name, val, col in seg:
        ax.barh(0, val, left=left, color=col, alpha=0.9,
                label=f"{name} {val/total*100:.0f}%")
        ax.text(left + val / 2, 0, f"{val/total*100:.0f}%", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
        left += val
    ax.set_xlim(0, total * 1.02); ax.set_yticks([])
    ax.set_title(f"Mean spread {total:.1f} bps decomposed into three costs", fontsize=11.5)
    ax.legend(frameon=False, fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3)
    fig.tight_layout()
    p = os.path.join(D, "ss_composition.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p, total, m_info, m_inv

# ============================================================
# 6) 图四：信息事件窗口——逆向选择跳升、库存平稳
# ============================================================
def fig_event(info, inv, spread, gamma, ev0=620, ev1=648):
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    idx = np.arange(ev0 - 30, ev1 + 20)
    ax.axvspan(ev0, ev1, color="#FFE08A", alpha=0.35, label="information event window")
    ax.plot(idx, info[idx], color=C["info"], lw=1.8, label=r"$s_t^{\rm info}$ (adverse selection)")
    ax.plot(idx, inv[idx], color=C["inv"], lw=1.8, label=r"$s_t^{\rm inv}$ (inventory)")
    ax.plot(idx, spread[idx], color=C["obs"], lw=0.9, alpha=0.5, label=r"observed $S_t$")
    ax.set_title("Around an information event, adverse-selection cost spikes; inventory stays flat",
                 fontsize=10.8)
    ax.set_xlabel("trade / minute index"); ax.set_ylabel("cost (bps)")
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    ax.grid(True, color=C["grid"], lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "ss_event.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p

if __name__ == "__main__":
    info, inv, gamma, spread = build_truth(T=1000)
    # 注入一个信息事件：在 t∈[620,648) 给逆向选择成分叠加冲击
    info[620:648] += np.linspace(2.0, 7.0, 28)
    info[620:648] += 0.5 * (np.arange(28) % 7 - 3)

    x_sm, _, _ = kalman_smoother(spread, gamma, 0.92, 0.55, 0.55, 0.9, 0.6 ** 2)
    # 复原质量
    corr_info = np.corrcoef(info, x_sm[:, 0])[0, 1]
    corr_inv = np.corrcoef(inv, x_sm[:, 1])[0, 1]
    rmse_info = np.sqrt(np.mean((info - x_sm[:, 0]) ** 2))
    rmse_inv = np.sqrt(np.mean((inv - x_sm[:, 1]) ** 2))
    print(f"recovery corr: info={corr_info:.3f}  inv={corr_inv:.3f}")
    print(f"recovery RMSE (bps): info={rmse_info:.3f}  inv={rmse_inv:.3f}")

    p1 = fig_components(info, inv, gamma, spread)
    p2 = fig_kalman(info, inv, x_sm)
    p3, total, m_info, m_inv = fig_composition(info, inv, gamma)
    p4 = fig_event(info, inv, spread, gamma)
    print("composition bps: info=%.2f inv=%.2f gamma=%.2f total=%.2f" % (m_info, m_inv, gamma, total))
    print("saved:", p1); print("saved:", p2); print("saved:", p3); print("saved:", p4)
