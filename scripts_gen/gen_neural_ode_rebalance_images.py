#!/usr/bin/env python3
"""神经微分方程组合再平衡：生成 4 张真实计算图。

实现一个从零开始的 Neural ODE 连续时间组合控制器：
- 1 个隐藏层 MLP 参数化漂移 f_theta(w, t) = dw/dt
- 用 Adam 在合成「目标权重路径」上做监督回归
- RK4 积分得到平滑的连续调仓路径
- 与「月度离散再平衡」和「买入持有」对比：跟踪误差、换手、净财富、相图
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import font_manager

# 注册中文字体（macOS 自带 Hiragino Sans GB）
_CJK = "/System/Library/Fonts/Hiragino Sans GB.ttc"
if os.path.exists(_CJK):
    font_manager.fontManager.addfont(_CJK)
    _name = font_manager.FontProperties(fname=_CJK).get_name()
    plt.rcParams["font.family"] = _name
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/neural-ode-rebalance"
import os
os.makedirs(OUT, exist_ok=True)
np.random.seed(7)

# ============ 1. 合成目标权重路径（风险偏好在两种 regime 间缓慢漂移） ============
T = 252 * 2                       # 2 年日频
tau = np.linspace(0.0, 2.0, T)    # 时间（年）
def target_w1(t):                 # 资产 1 的目标权重：慢正弦在 [0.30,0.80] 间摆动
    return 0.55 + 0.25 * np.sin(np.pi * 0.9 * t)
w1_star = target_w1(tau)
w2_star = 1.0 - w1_star
dw1dt_star = 0.25 * np.pi * 0.9 * np.cos(np.pi * 0.9 * tau)   # 解析梯度真值

# ============ 2. 从零实现 1 隐藏层 MLP + Adam ============
class MLP:
    def __init__(self, din, dh, dout, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((dh, din)) * 0.6
        self.b1 = np.zeros(dh)
        self.W2 = rng.standard_normal((dout, dh)) * 0.6
        self.b2 = np.zeros(dout)
    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1.T + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2.T + self.b2
        return self.z2
    def backward(self, dout):
        B = self.x.shape[0]
        dW2 = dout.T @ self.a1 / B
        db2 = dout.sum(0) / B
        da1 = dout @ self.W2
        dz1 = da1 * (1.0 - self.a1 ** 2)
        dW1 = dz1.T @ self.x / B
        db1 = dz1.sum(0) / B
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

def adam_train(model, X, Y, iters=4000, lr=0.02):
    m = {k: np.zeros_like(v) for k, v in vars(model).items()}
    v = {k: np.zeros_like(v) for k, v in vars(model).items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, iters + 1):
        out = model.forward(X)
        loss = np.mean((out - Y) ** 2)
        dout = 2.0 * (out - Y) / X.shape[0]
        g = model.backward(dout)
        for k in g:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * (g[k] ** 2)
            mh = m[k] / (1 - b1 ** t)
            vh = v[k] / (1 - b2 ** t)
            setattr(model, k, getattr(model, k) - lr * mh / (np.sqrt(vh) + eps))
        if t % 1000 == 0:
            print(f"  [train] iter {t:5d}  MSE={loss:.3e}")
    return loss

# 监督数据：(w1, tau) -> dw1/dtau
Xtr = np.column_stack([w1_star, tau])
Ytr = dw1dt_star.reshape(-1, 1)
mlp = MLP(din=2, dh=24, dout=1, seed=3)
print("训练 Neural ODE 漂移网络 f_theta ...")
adam_train(mlp, Xtr, Ytr, iters=5000, lr=0.02)

# ============ 3. RK4 积分得到连续调仓路径 ============
def f_theta(w1, t):
    return float(mlp.forward(np.array([[w1, t]]))[0, 0])

def rk4(w0, tgrid):
    w = np.zeros_like(tgrid)
    w[0] = w0
    for i in range(len(tgrid) - 1):
        h = tgrid[i + 1] - tgrid[i]
        t = tgrid[i]; y = w[i]
        k1 = f_theta(y, t)
        k2 = f_theta(y + 0.5 * h * k1, t + 0.5 * h)
        k3 = f_theta(y + 0.5 * h * k2, t + 0.5 * h)
        k4 = f_theta(y + h * k3, t + h)
        w[i + 1] = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        w[i + 1] = min(max(w[i + 1], 0.0), 1.0)   # 投影到单纯形
    return w

w1_ode = rk4(w1_star[0], tau)
w2_ode = 1.0 - w1_ode

# ============ 4. 月度离散再平衡（再平衡点骤变、之间权重漂移 stale） ============
rebal = 21
w1_disc = np.empty_like(w1_star)
cur = w1_star[0]
for i in range(T):
    if i % rebal == 0:
        cur = w1_star[i]
    w1_disc[i] = cur
w2_disc = 1.0 - w1_disc

# 指标
track_err_ode = np.sqrt((w1_ode - w1_star) ** 2)
track_err_disc = np.sqrt((w1_disc - w1_star) ** 2)
turn_ode = np.abs(np.diff(w1_ode, prepend=w1_ode[0]))
turn_disc = np.abs(np.diff(w1_disc, prepend=w1_disc[0]))
print(f"  RMSE 跟踪误差  ODE={track_err_ode.mean():.4f}  Discrete={track_err_disc.mean():.4f}")
print(f"  累计换手      ODE={turn_ode.sum():.3f}  Discrete={turn_disc.sum():.3f}")

# ============ 5. 合成资产市场 + 净财富对比（含交易成本） ============
mu = np.array([0.09, 0.04]); Sigma = np.array([[0.05, 0.0], [0.0, 0.012]])
L = np.linalg.cholesky(Sigma)
dt = 1.0 / 252
ret = (mu * dt).reshape(2, 1) + (L @ np.random.randn(2, T)) * np.sqrt(dt)   # 2 x T 日收益
c = 0.0012   # 单位权重变动的交易成本

def wealth(weights):
    V = np.ones(T)
    for i in range(1, T):
        # 用权重持有 + 每日资产收益
        growth = weights[:, i - 1] @ ret[:, i - 1] + 1.0
        cost = c * abs(weights[:, i] - weights[:, i - 1]).sum()
        V[i] = V[i - 1] * growth - cost
    return V

V_bh   = wealth(np.tile([w1_star[0], w2_star[0]], (T, 1)).T)         # 买入持有
V_tgt  = wealth(np.vstack([w1_star, w2_star]))                       # 理想目标跟随（无滞后）
V_disc = wealth(np.vstack([w1_disc, w2_disc]))                       # 月度离散
V_ode  = wealth(np.vstack([w1_ode, w2_ode]))                         # Neural ODE 连续
def sharpe(V):
    r = np.diff(V) / V[:-1]
    return r.mean() / (r.std() + 1e-12) * np.sqrt(252)
print(f"  净财富终值  BH={V_bh[-1]:.3f} TGT={V_tgt[-1]:.3f} DISC={V_disc[-1]:.3f} ODE={V_ode[-1]:.3f}")
print(f"  年化 Sharpe BH={sharpe(V_bh):.2f} TGT={sharpe(V_tgt):.2f} DISC={sharpe(V_disc):.2f} ODE={sharpe(V_ode):.2f}")

# ========================= 绘图 =========================
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 110, "savefig.bbox": "tight"})
C = {"star": "#222222", "disc": "#d1495b", "ode": "#2a9d8f", "bh": "#8d99ae"}

# 图1：权重轨迹
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(tau, w1_star, "--", color=C["star"], lw=2, label="目标 w1*(t)")
ax[0].plot(tau, w1_disc, color=C["disc"], lw=1.3, alpha=0.9, label="月度离散再平衡")
ax[0].plot(tau, w1_ode, color=C["ode"], lw=1.8, label="Neural ODE 连续")
ax[0].set_title("资产 1 权重轨迹 w1(t)"); ax[0].set_xlabel("时间（年）"); ax[0].set_ylabel("权重"); ax[0].legend(fontsize=9)
ax[1].plot(tau, w2_star, "--", color=C["star"], lw=2, label="目标 w2*(t)")
ax[1].plot(tau, w2_disc, color=C["disc"], lw=1.3, alpha=0.9, label="月度离散再平衡")
ax[1].plot(tau, w2_ode, color=C["ode"], lw=1.8, label="Neural ODE 连续")
ax[1].set_title("资产 2 权重轨迹 w2(t)"); ax[1].set_xlabel("时间（年）"); ax[1].set_ylabel("权重"); ax[1].legend(fontsize=9)
fig.suptitle("连续时间控制 vs 离散再平衡：权重路径", fontsize=13)
fig.savefig(f"{OUT}/weights_trajectory.png"); plt.close(fig)

# 图2：跟踪误差 + 换手
fig, ax = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
ax[0].plot(tau, track_err_disc, color=C["disc"], lw=1.3, label="月度离散")
ax[0].plot(tau, track_err_ode, color=C["ode"], lw=1.8, label="Neural ODE")
ax[0].set_ylabel("|w − w*| 跟踪误差"); ax[0].set_title("跟踪误差：连续控制显著更低"); ax[0].legend(fontsize=9)
ax[1].plot(tau, turn_disc, color=C["disc"], lw=1.0, alpha=0.8, label="月度离散（月末尖峰）")
ax[1].plot(tau, turn_ode, color=C["ode"], lw=1.3, label="Neural ODE（平滑）")
ax[1].set_ylabel("单日 |Δw| 换手"); ax[1].set_xlabel("时间（年）"); ax[1].set_title("换手：连续控制把尖峰摊成平滑流"); ax[1].legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/tracking_turnover.png"); plt.close(fig)

# 图3：净财富
fig, ax = plt.subplots(figsize=(9.5, 5))
ax.plot(tau, V_bh, color=C["bh"], lw=1.5, label=f"买入持有 ({V_bh[-1]:.2f})")
ax.plot(tau, V_tgt, color=C["star"], lw=1.8, ls="--", label=f"理想目标跟随 ({V_tgt[-1]:.2f})")
ax.plot(tau, V_disc, color=C["disc"], lw=1.6, label=f"月度离散再平衡 ({V_disc[-1]:.2f})")
ax.plot(tau, V_ode, color=C["ode"], lw=2.0, label=f"Neural ODE 连续 ({V_ode[-1]:.2f})")
ax.set_title("净财富对比（含交易成本）"); ax.set_xlabel("时间（年）"); ax.set_ylabel("组合价值（起点=1）"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/cumulative_wealth.png"); plt.close(fig)

# 图4：相图 + ODE 向量场
fig, ax = plt.subplots(figsize=(7.2, 6.2))
Xg, Yg = np.meshgrid(np.linspace(0.2, 0.85, 18), np.linspace(0.15, 0.8, 18))
Ug = np.zeros_like(Xg); Vg = np.zeros_like(Yg)
for i in range(Xg.shape[0]):
    for j in range(Xg.shape[1]):
        w1 = Xg[i, j]; w2 = Yg[i, j]; t = 1.0
        dw1 = f_theta(w1, t); dw2 = -dw1
        Ug[i, j] = dw1; Vg[i, j] = dw2
speed = np.sqrt(Ug ** 2 + Vg ** 2) + 1e-9
ax.streamplot(Xg, Yg, Ug / speed, Vg / speed, color=speed, cmap="viridis", density=1.1, linewidth=1.1)
ax.plot(w1_ode, w2_ode, color=C["ode"], lw=2.2, label="Neural ODE 轨迹")
ax.plot(w1_star, w2_star, "--", color=C["star"], lw=1.6, label="目标路径")
ax.scatter([w1_ode[0]], [w2_ode[0]], color="black", zorder=5, s=45, label="起点")
ax.set_xlim(0.2, 0.85); ax.set_ylim(0.15, 0.8)
ax.set_xlabel("w1"); ax.set_ylabel("w2"); ax.set_title("相图：权重空间中的连续控制流"); ax.legend(fontsize=9, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/phase_portrait.png"); plt.close(fig)

print("✅ 已生成 4 张图表到", OUT)
print(os.listdir(OUT))
