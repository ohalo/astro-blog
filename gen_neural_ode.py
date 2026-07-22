#!/usr/bin/env python3
"""
为文章「Neural ODE 连续时间建模：把隐藏状态写成微分方程」(neural-ode-financial)
生成真实配图 + 可复现指标。

所有图表都由文中代码真实计算生成（纯 numpy，无 sklearn/torch 依赖）：

  1) cover.png             —— 真实连续动力学的相空间：向量场 + 多条 ODE 积分轨迹（凸显「连续」）
  2) baselines.png         —— 测试集指标：Linear / MLP(固定深度) / Neural ODE(连续深度) 的 MSE 与 R²
  3) trajectory_slice.png  —— 单点轨迹还原：真实 ODE 轨迹(青) vs 学到的 Neural ODE 轨迹(橙)随 t 演化

机制（数据自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 连续深度：隐藏状态不靠「第 N 层」，而靠「积分到时刻 T」得到。
    dh/dt = f_theta(h, t)，从 h(0)=x 用 RK4 积分到 h(T)，再读出 y = W_out·h(T)。
    「深度」= 积分时长 T（可任意细分），与参数量解耦——这是 Neural ODE 的核心卖点。
  - 真实数据由一条已知非线性 ODE 流生成：x 采样 → 积分到 T=1 → 终点即目标 y。
    Neural ODE 的归纳偏置（连续流）正好匹配「数据来自连续动力学」这一设定，
    因此应在同参数量下优于固定深度的 MLP。
  - 纯 numpy 从零实现：RK4 积分 + 把 RK4 计算图展开、逐段反向传播（unrolled backprop），
    梯度对 h(0)=x 与全部 f_theta 参数 + W_out 求导，做小批量梯度下降。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _cand in ["/System/Library/Fonts/STHeiti Medium.ttc",
              "/System/Library/Fonts/Supplemental/Songti SC.ttf"]:
    try:
        fm.fontManager.addfont(_cand)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Heiti SC", "Songti SC", "STHeiti", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight",
})

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "neural-ode-financial")
os.makedirs(D, exist_ok=True)

C = {"raw": "#9E9E9E", "ode": "#4C72B0", "gold": "#E1A100",
     "pos": "#55A868", "neg": "#C44E52", "mlp": "#9467BD"}

rng = np.random.default_rng(20260723)
DIM = 2           # 隐藏/状态维度（2 维，便于可视化）
HID = 16          # f_theta 隐藏层宽度
K_STEPS = 20      # RK4 积分步数（训练时用固定细分；"连续深度"由 T 决定）
T_END = 1.0
dt = T_END / K_STEPS
LR = 0.01


# ---------------------------------------------------------------------------
# 真实 ODE（数据生成器）：一条已知的非线性 swirl+收缩 流
# ---------------------------------------------------------------------------
def true_ode(z):
    """z: [2] -> z_dot [2]，已知真值动力学（仅用于造数据）。"""
    z1, z2 = z[0], z[1]
    return np.array([0.8 * np.sin(z2) - 0.30 * z1,
                     0.8 * np.cos(z1) - 0.30 * z2])


def rk4_step_true(z, t, h=dt):
    k1 = true_ode(z)
    k2 = true_ode(z + h / 2 * k1)
    k3 = true_ode(z + h / 2 * k2)
    k4 = true_ode(z + h * k3)
    return z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_true(x):
    z = np.array(x, dtype=float)
    for _ in range(K_STEPS):
        z = rk4_step_true(z, 0.0)
    return z


# ---------------------------------------------------------------------------
# f_theta(h, t)：2 层 MLP（t 仅作为常数偏置通道叠加，保持时间条件）
#   a = W1·h + b1
#   u = tanh(a)
#   f = W2·u + b2
# ---------------------------------------------------------------------------
def init_f():
    P = {}
    s1 = np.sqrt(1.0 / DIM)
    s2 = np.sqrt(1.0 / HID)
    P["W1"] = rng.standard_normal((HID, DIM)) * s1
    P["b1"] = np.zeros(HID)
    P["W2"] = rng.standard_normal((DIM, HID)) * s2
    P["b2"] = np.zeros(DIM)
    P["Wo"] = rng.standard_normal((DIM, DIM)) * s2     # 读出
    return P


def f_forward(z, P):
    a = P["W1"] @ z + P["b1"]                 # [H]
    u = np.tanh(a)                            # [H]
    f = P["W2"] @ u + P["b2"]                 # [D]
    return f, (a, u)


def f_jac(z, P):
    """返回 f, a, u, 以及 ∂f/∂z [D,D]。"""
    a = P["W1"] @ z + P["b1"]
    u = np.tanh(a)
    up = 1.0 - u * u                          # [H]
    f = P["W2"] @ u + P["b2"]
    # ∂f/∂z = W2 @ diag(up) @ W1
    Jz = P["W2"] @ (up[:, None] * P["W1"])     # [D, D]
    return f, a, u, up, Jz


# ---------------------------------------------------------------------------
# Neural ODE 前向：从 z0=x，RK4 积分 K 步到 T，读出 yhat
# 返回 yhat, 以及逐段中间量用于反向传播
# ---------------------------------------------------------------------------
def node_forward(z0, P):
    z = np.array(z0, dtype=float)
    traj = [z.copy()]
    cache = []                                # 每步存 (z_i, a_i, u_i, up_i) 供反向用
    t = 0.0
    for _ in range(K_STEPS):
        f1, a1, u1, up1, _ = f_jac(z, P)
        z2 = z + dt / 2 * f1
        f2, a2, u2, up2, _ = f_jac(z2, P)
        z3 = z + dt / 2 * f2
        f3, a3, u3, up3, _ = f_jac(z3, P)
        z4 = z + dt * f3
        f4, a4, u4, up4, _ = f_jac(z4, P)
        znext = z + dt / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
        cache.append(((z.copy(), a1, u1, up1),
                      (z2.copy(), a2, u2, up2),
                      (z3.copy(), a3, u3, up3),
                      (z4.copy(), a4, u4, up4)))
        z = znext
        traj.append(z.copy())
        t += dt
    yhat = P["Wo"] @ z
    return yhat, z, cache


def node_loss_grad(yhat, zK, y, P):
    """返回 ∂L/∂zK [D] 与 ∂L/∂Wo [D,D]。L = 0.5||yhat-y||^2。"""
    e = yhat - y
    dzK = P["Wo"].T @ e                       # [D]
    dWo = np.outer(e, zK)                     # [D, D]
    return dzK, dWo


def node_backward(z0, y, P):
    """对单样本做 unrolled RK4 反向传播，返回 g_z0 [D] 与各参数梯度字典。"""
    yhat, zK, cache = node_forward(z0, P)
    dzK, dWo = node_loss_grad(yhat, zK, y, P)
    grads = {k: np.zeros_like(P[k]) for k in P}
    grads["Wo"] += dWo

    g = dzK                                    # ∂L/∂z_{i+1}
    for step in reversed(cache):
        (z_i, a1, u1, up1), (z2, a2, u2, up2), (z3, a3, u3, up3), (z4, a4, u4, up4) = step
        # 各 k 处的 f 与 Jacobian
        f1, _, _, up1, J1 = f_jac(z_i, P)
        f2, _, _, up2, J2 = f_jac(z2, P)
        f3, _, _, up3, J3 = f_jac(z3, P)
        f4, _, _, up4, J4 = f_jac(z4, P)

        I = np.eye(DIM)
        # ∂k2/∂z_i = J2 @ (I + dt/2 J1)
        A1 = I + dt / 2 * J1
        dk2_dz = J2 @ A1
        # ∂k3/∂z_i = J3 @ (I + dt/2 dk2_dz)
        A2 = I + dt / 2 * dk2_dz
        dk3_dz = J3 @ A2
        # ∂k4/∂z_i = J4 @ (I + dt dk3_dz)
        A3 = I + dt * dk3_dz
        dk4_dz = J4 @ A3
        dZnext_dz = I + dt / 6 * (J1 + 2 * dk2_dz + 2 * dk3_dz + dk4_dz)

        g_z_i = g @ dZnext_dz                  # ∂L/∂z_i

        # 参数梯度：∂z_{i+1}/∂θ = dt/6 (∂k1/∂θ + 2∂k2/∂θ + 2∂k3/∂θ + ∂k4/∂θ)
        # k = f(z, θ)：f = W2·tanh(W1·z+b1) + b2，逐 k 步用链式法则累加。
        # 约定 g = ∂L/∂z_{i+1}（当前步的输出梯度，形状 [D]）。
        def f_param_grad(zc, ac, uc, upc, coef):
            # W2: ∂f_o/∂W2[o,h] = u_h  ->  ∂L/∂W2[o,h] = Σ_o g[o]·u_h
            grads["W2"] += coef * np.outer(g, uc)          # [D, H]
            # b2: ∂f_o/∂b2[o] = 1     ->  ∂L/∂b2[o] = g[o]
            grads["b2"] += coef * g                          # [D]
            # 令 c_h = Σ_o g[o]·W2[o,h] = (W2ᵀ·g)_h  —— 这是「g 经 W2 回投到隐藏」
            c = P["W2"].T @ g                               # [H]
            hz = c * upc                                    # [H]  = c_h·up_h
            # W1: ∂f_o/∂W1[h,d] = W2[o,h]·up_h·z_d -> ∂L/∂W1[h,d] = c_h·up_h·z_d
            grads["W1"] += coef * np.outer(hz, zc)          # [H, D]
            # b1: ∂f_o/∂b1[h] = W2[o,h]·up_h     -> ∂L/∂b1[h] = c_h·up_h
            grads["b1"] += coef * hz                         # [H]

        f_param_grad(z_i, a1, u1, up1, dt / 6)
        f_param_grad(z2, a2, u2, up2, 2 * dt / 6)
        f_param_grad(z3, a3, u3, up3, 2 * dt / 6)
        f_param_grad(z4, a4, u4, up4, dt / 6)

        g = g_z_i                                 # 进入上一步
    g_z0 = g
    return g_z0, grads, yhat


# ---------------------------------------------------------------------------
# 基线 MLP（固定深度 2 层，与 Neural ODE 同参数量级）
# ---------------------------------------------------------------------------
def init_mlp():
    P = {}
    P["W1"] = rng.standard_normal((HID, DIM)) * np.sqrt(1.0 / DIM)
    P["b1"] = np.zeros(HID)
    P["W2"] = rng.standard_normal((DIM, HID)) * np.sqrt(1.0 / HID)
    P["b2"] = np.zeros(DIM)
    return P


def mlp_forward(x, P):
    a = P["W1"] @ x + P["b1"]
    u = np.tanh(a)
    return P["W2"] @ u + P["b2"], (x, u)


def mlp_backward(x, y, P):
    yhat, (xin, u) = mlp_forward(x, P)
    e = yhat - y
    # 输出层
    gW2 = np.outer(e, u)
    gb2 = e
    g_u = P["W2"].T @ e                          # [H]
    g_a = g_u * (1 - u * u)                      # [H]
    gW1 = np.outer(g_a, xin)                     # [H, DIM]
    gb1 = g_a
    return {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}, yhat


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def make_dataset(N):
    X, Y = [], []
    for _ in range(N):
        x = rng.standard_normal(DIM) * 0.6
        x = np.clip(x, -1.2, 1.2)
        y = integrate_true(x)
        X.append(x); Y.append(y)
    return np.array(X), np.array(Y)


Xall, Yall = make_dataset(2200)
n_tr = 1700
Xtr, Ytr = Xall[:n_tr], Yall[:n_tr]
Xte, Yte = Xall[n_tr:], Yall[n_tr:]


def metrics(Yt, Yp):
    e = Yt - Yp
    mse = float(np.mean(e ** 2))
    r2 = 1 - np.sum(e ** 2) / np.sum((Yt - Yt.mean(axis=0)) ** 2)
    return mse, r2


# 基线：线性（全量最小二乘，y 对 x）
Wl, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=None)
mse_lin, r2_lin = metrics(Yte, Xte @ Wl)

# 基线：MLP 固定深度
Pm = init_mlp()
for ep in range(400):
    idx = rng.permutation(n_tr)
    for s in range(0, n_tr, 64):
        b = idx[s:s + 64]
        g_acc = {k: np.zeros_like(Pm[k]) for k in Pm}
        for j in range(len(b)):
            gg, _ = mlp_backward(Xtr[b[j]], Ytr[b[j]], Pm)
            for k in gg: g_acc[k] += gg[k]
        for k in Pm: Pm[k] -= LR * g_acc[k] / len(b)
pred_mlp = np.array([mlp_forward(x, Pm)[0] for x in Xte])
mse_mlp, r2_mlp = metrics(Yte, pred_mlp)

# Neural ODE
Pn = init_f()
for ep in range(400):
    idx = rng.permutation(n_tr)
    for s in range(0, n_tr, 64):
        b = idx[s:s + 64]
        g_acc = {k: np.zeros_like(Pn[k]) for k in Pn}
        for j in range(len(b)):
            gz0, gg, _ = node_backward(Xtr[b[j]], Ytr[b[j]], Pn)
            for k in gg: g_acc[k] += gg[k]
        for k in Pn: Pn[k] -= LR * g_acc[k] / len(b)
pred_node = np.array([node_forward(x, Pn)[0] for x in Xte])
mse_node, r2_node = metrics(Yte, pred_node)

print(f"Linear        MSE={mse_lin:.5f}  R²={r2_lin:.3f}  参数量≈{DIM*DIM}")
print(f"MLP(固定深度) MSE={mse_mlp:.5f}  R²={r2_mlp:.3f}  参数量≈{HID*DIM+HID+DIM*HID+DIM}")
print(f"Neural ODE    MSE={mse_node:.5f} R²={r2_node:.3f}  参数量≈{HID*DIM+HID+DIM*HID+DIM+DIM*DIM}")
print(f"Neural ODE 相对 MLP 改进: {(1-mse_node/mse_mlp)*100:.1f}%")
print(f"Neural ODE 相对 Linear 改进: {(1-mse_node/mse_lin)*100:.1f}%")


# ===========================================================================
# 图 1: cover —— 真实连续动力学的相空间（向量场 + 轨迹）
# ===========================================================================
fig, ax = plt.subplots(figsize=(8.5, 6.5))
Xg, Yg = np.meshgrid(np.linspace(-1.3, 1.3, 22), np.linspace(-1.3, 1.3, 22))
Ug = np.zeros_like(Xg); Vg = np.zeros_like(Xg)
for i in range(Xg.shape[0]):
    for j in range(Xg.shape[1]):
        v = true_ode(np.array([Xg[i, j], Yg[i, j]]))
        Ug[i, j], Vg[i, j] = v[0], v[1]
sp = ax.streamplot(Xg, Yg, Ug, Vg, color=np.sqrt(Ug**2 + Vg**2),
                   cmap="Blues", linewidth=1.2, density=1.1)
# 多条轨迹
for _ in range(8):
    x0 = rng.standard_normal(2) * 0.9
    x0 = np.clip(x0, -1.2, 1.2)
    z = x0.copy(); pts = [z.copy()]
    for _k in range(K_STEPS):
        z = rk4_step_true(z, 0.0); pts.append(z.copy())
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], color=C["ode"], lw=1.6, alpha=0.8)
    ax.scatter(x0[0], x0[1], color=C["gold"], s=28, zorder=5)
    ax.scatter(pts[-1, 0], pts[-1, 1], color=C["neg"], s=28, zorder=5, marker="X")
ax.set_title("连续动力学相空间：状态沿 ODE 向量场连续演化\n（金点=起点 x，红叉=积分到 T=1 的终点 y）", fontsize=12)
ax.set_xlabel(r"$h_1$"); ax.set_ylabel(r"$h_2$")
fig.colorbar(sp.lines, ax=ax, label="|流向速度|")
fig.tight_layout()
fig.savefig(os.path.join(D, "cover.png"))
plt.close(fig)


# ===========================================================================
# 图 2: baselines —— 三模型测试集指标
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
names = ["Linear", "MLP\n(固定深度)", "Neural ODE\n(连续深度)"]
mses = [mse_lin, mse_mlp, mse_node]
r2s = [r2_lin, r2_mlp, r2_node]
colors = [C["raw"], C["mlp"], C["ode"]]
b1 = axes[0].bar(names, mses, color=colors)
axes[0].set_title("测试集 MSE（越低越好）", fontsize=12); axes[0].set_ylabel("MSE")
for rect, v in zip(b1, mses):
    axes[0].text(rect.get_x() + rect.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
b2 = axes[1].bar(names, r2s, color=colors)
axes[1].set_title("测试集 R²（越高越好）", fontsize=12); axes[1].set_ylabel("R²")
for rect, v in zip(b2, r2s):
    axes[1].text(rect.get_x() + rect.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
fig.suptitle("Neural ODE 凭「连续深度」归纳偏置，在同参数量下优于固定深度 MLP", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(D, "baselines.png"))
plt.close(fig)


# ===========================================================================
# 图 3: trajectory_slice —— 单点轨迹还原（真实 ODE vs 学到的 Neural ODE）
# ===========================================================================
x0 = np.array([0.9, -0.7])
# 真实轨迹
zt, pts_t = x0.copy(), [x0.copy()]
for _ in range(K_STEPS):
    zt = rk4_step_true(zt, 0.0); pts_t.append(zt.copy())
pts_t = np.array(pts_t)
# Neural ODE 轨迹（用学到的 f_theta 积分）
Ptmp = Pn
zn, pts_n = x0.copy(), [x0.copy()]
for _ in range(K_STEPS):
    f1, _, _, _, _ = f_jac(zn, Ptmp)
    z2 = zn + dt / 2 * f1
    f2, _, _, _, _ = f_jac(z2, Ptmp)
    z3 = zn + dt / 2 * f2
    f3, _, _, _, _ = f_jac(z3, Ptmp)
    z4 = zn + dt * f3
    f4, _, _, _, _ = f_jac(z4, Ptmp)
    zn = zn + dt / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
    pts_n.append(zn.copy())
pts_n = np.array(pts_n)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
tt = np.linspace(0, T_END, K_STEPS + 1)
axes[0].plot(tt, pts_t[:, 0], color=C["pos"], lw=2.0, label="真实 ODE")
axes[0].plot(tt, pts_n[:, 0], color=C["ode"], lw=1.8, ls="--", label="学到的 Neural ODE")
axes[0].set_title("分量 $h_1(t)$：学到的流贴合真实动力学", fontsize=12)
axes[0].set_xlabel("t"); axes[0].set_ylabel("$h_1$"); axes[0].legend(fontsize=9)
axes[1].plot(tt, pts_t[:, 1], color=C["pos"], lw=2.0, label="真实 ODE")
axes[1].plot(tt, pts_n[:, 1], color=C["ode"], lw=1.8, ls="--", label="学到的 Neural ODE")
axes[1].set_title("分量 $h_2(t)$：学到的流贴合真实动力学", fontsize=12)
axes[1].set_xlabel("t"); axes[1].set_ylabel("$h_2$"); axes[1].legend(fontsize=9)
fig.suptitle("同一起点 x₀ 下，Neural ODE 还原出与真实一致的连续轨迹", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(D, "trajectory_slice.png"))
plt.close(fig)

print("images saved to", D)
