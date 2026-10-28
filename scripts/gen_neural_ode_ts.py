#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""神经微分方程时序建模：Neural ODE 用 ODE 求解器拟合连续时间动态
真实计算 5 张图 + _node_results.json（torch 自动微分反向传播 unrolled RK4）

E1 阻尼振荡外推：Neural ODE(RK4) vs Neural ODE(Euler) vs 离散单步 MLP 自回归
E2 数值解代价：RK4 4 阶 vs Euler 1 阶 步长收敛；粗步 Euler 训练失败代价
E3 不规则采样：mask 损失 Neural ODE vs 前向填充 + 自回归离散模型
E4 OU 均值回归：学漂移场 dx/dt=f(x) 恢复 -κ
E5 Lorenz 混沌：短窗训练、6 倍外推，误差指数增长，拟合李雅普诺夫增长率

依赖：torch（venv：~/.workbuddy/binaries/python/envs/default/bin/python）
"""
import os, json, time, math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = "mps" if (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()) else "cpu"
print("device =", device)

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

OUT = "/Users/halo123456/workspace/astro-blog/public/images/neural-ode-time-series"
os.makedirs(OUT, exist_ok=True)
RED, GREEN, BLUE, GRAY, ORANGE, BLACK, PURPLE = "#d63031", "#00b894", "#0984e3", "#636e72", "#e17055", "#2d3436", "#6c5ce7"
SEED = 20261028
np.random.seed(SEED); torch.manual_seed(SEED)
R = {}


# ---------------- 通用：RK4 / Euler 展开（torch 自动微分）----------------
def rk4_step(f, x, h):
    k1 = f(x); k2 = f(x + 0.5 * h * k1); k3 = f(x + 0.5 * h * k2); k4 = f(x + h * k3)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def euler_step(f, x, h):
    return x + h * f(x)

def rollout(f, x0, n, h, method="rk4"):
    step = rk4_step if method == "rk4" else euler_step
    xs = [x0]; x = x0
    for _ in range(n):
        x = step(f, x, h); xs.append(x)
    return torch.stack(xs, 1)  # (B, n+1, D)


class Net(nn.Module):
    def __init__(self, D=2, W=64):
        super().__init__()
        self.m = nn.Sequential(nn.Linear(D, W), nn.Tanh(), nn.Linear(W, W), nn.Tanh(), nn.Linear(W, D))
    def forward(self, x):
        return self.m(x)


def train_node(x0, y_obs, mask=None, epochs=600, lr=1e-2, h=0.04, method="rk4", D=2, W=64, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model = Net(D, W).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    T = y_obs.shape[1]; Dh = y_obs.shape[-1]
    if mask is None:
        mask = torch.ones(x0.shape[0], T, device=device)
    loss_hist = []; step_times = []
    x0d = x0.to(device); yd = y_obs.to(device); maskd = mask.to(device)
    for ep in range(epochs):
        opt.zero_grad()
        t0 = time.time()
        traj = rollout(model, x0d, T, h, method)
        pred = traj[:, 1:, :]
        diff = ((pred - yd) ** 2).sum(-1)
        loss = (diff * maskd).sum() / maskd.sum() / Dh
        loss.backward(); opt.step()
        if ep >= 5:
            step_times.append(time.time() - t0)
        loss_hist.append(loss.item())
    return model, np.array(loss_hist), float(np.mean(step_times))


# ============ E1 数据：阻尼振荡器 ============
W0, ZETA = 2.0, 0.08
def osc_truth(x):
    return torch.stack([x[..., 1], -W0**2 * x[..., 0] - 2 * ZETA * W0 * x[..., 1]], -1)

h_data, N_T, N_EXT, B1, NOISE = 0.04, 100, 300, 160, 0.02
torch.manual_seed(SEED)
x0s = torch.stack([torch.empty(B1).uniform_(-1.6, 1.6), torch.empty(B1).uniform_(-1.6, 1.6)], -1)
true_full = rollout(osc_truth, x0s, (N_T + N_EXT) * 40, 0.001, "rk4")
true_dense = true_full[:, ::40]
Y_TRAIN = true_dense[:, 1:N_T + 1]
Y_TEST_GT = true_dense[:, N_T + 1:]
Y_TRAIN_N = Y_TRAIN + NOISE * torch.randn_like(Y_TRAIN)

t0 = time.time()
m_node, lh_node, step_t_rk4 = train_node(x0s, Y_TRAIN_N, epochs=400, h=h_data, method="rk4")
t_node_train = time.time() - t0
m_euler, lh_euler, step_t_eul = train_node(x0s, Y_TRAIN_N, epochs=400, h=h_data, method="euler")

# ---- 离散单步 MLP 自回归基线：x_t -> x_{t+1} ----
WIN = 40
noisy_all = true_dense[:, 1:N_T + WIN + 1] + NOISE * torch.randn(B1, N_T + WIN, 2)
wins = torch.stack([noisy_all[:, i:i + WIN] for i in range(N_T - WIN + 1)], 1)
wins_f = wins.reshape(-1, WIN, 2)
tgt = torch.stack([noisy_all[:, i + WIN] for i in range(N_T - WIN + 1)], 1).reshape(-1, 2)

class OneStep(nn.Module):
    def __init__(self, D=2, W=64):
        super().__init__()
        self.m = nn.Sequential(nn.Linear(D, W), nn.Tanh(), nn.Linear(W, W), nn.Tanh(), nn.Linear(W, D))
    def forward(self, x):
        # 单步预测：取窗口最后一步状态 -> 下一步
        return self.m(x[:, -1, :])

torch.manual_seed(SEED)
lstm = OneStep().to(device)
optL = torch.optim.Adam(lstm.parameters(), lr=3e-3)
lh_lstm = []
dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(wins_f.to(device), tgt.to(device)),
                                 batch_size=512, shuffle=True)
t0 = time.time()
for ep in range(150):
    tot = 0.0
    for xb, yb in dl:
        optL.zero_grad()
        loss = ((lstm(xb) - yb) ** 2).mean()
        loss.backward(); optL.step(); tot += loss.item() * len(xb)
    lh_lstm.append(tot / len(wins_f))
t_lstm_train = time.time() - t0

def gru_autoreg(seed_win, n_future):
    cur = seed_win.clone(); out = []
    with torch.no_grad():
        for _ in range(n_future):
            nxt = lstm(cur)
            out.append(nxt); cur = torch.cat([cur[:, 1:, :], nxt[:, None, :]], 1)
    return torch.stack(out, 1)

SUB = slice(0, 64)
node_traj = rollout(m_node, x0s[SUB].to(device), N_T + N_EXT, h_data, "rk4").detach().cpu()
node_ext = node_traj[:, N_T + 1:, :]
lstm_ext = gru_autoreg(Y_TRAIN_N[SUB, -WIN:].to(device), N_EXT).detach().cpu()

def rmse_seq(pred, gt):
    return ((pred - gt) ** 2).mean(-1).detach().sqrt().mean(0).numpy()

err_node = rmse_seq(node_ext, Y_TEST_GT[SUB])
err_lstm = rmse_seq(lstm_ext, Y_TEST_GT[SUB])
R["e1"] = {"rmse_node_end": round(float(err_node[-1]), 4), "rmse_lstm_end": round(float(err_lstm[-1]), 4),
           "rmse_node_q": round(float(np.quantile(err_node, 0.25)), 4),
           "rmse_lstm_q": round(float(np.quantile(err_lstm, 0.25)), 4),
           "gru_one_step_rmse": round(float(lh_lstm[-1]) ** 0.5, 4),
           "node_train_s": round(t_node_train, 1), "gru_train_s": round(t_lstm_train, 1)}

# ---- 图 1 ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
tt = (np.arange(1, N_T + N_EXT + 1)) * h_data
i = 0
ax[0].plot(tt, true_dense[i, 1:, 0].numpy(), color=BLACK, lw=1.2, label="真值(全程)")
ax[0].plot(tt[:N_T], Y_TRAIN_N[i, :, 0].numpy(), color=GRAY, lw=0.8, alpha=0.8, label="含噪训练数据")
ax[0].plot(tt[N_T:], node_ext[i, :, 0].numpy(), color=RED, lw=1.4, ls="--", label="Neural ODE")
ax[0].plot(tt[N_T:], lstm_ext[i, :, 0].numpy(), color=BLUE, lw=1.2, ls=":", label="单步 MLP 自回归")
ax[0].axvline(4.0, color=ORANGE, lw=1, ls="--")
ax[0].text(4.1, 1.35, "训练终点 t=4s", fontsize=8, color=ORANGE)
ax[0].set_title(f"样本#0 位移：外推段末端 RMSE  ODE {R['e1']['rmse_node_end']:.3f} vs 自回归 {R['e1']['rmse_lstm_end']:.3f}")
ax[0].set_xlabel("t (s)"); ax[0].legend(fontsize=8)
sq = max(1, N_EXT // 80)
ax[1].semilogy(tt[N_T:][::sq], np.maximum(err_node[::sq], 1e-4), color=RED, label="Neural ODE (RK4)")
ax[1].semilogy(tt[N_T:][::sq], np.maximum(err_lstm[::sq], 1e-4), color=BLUE, label="单步 MLP 自回归")
ax[1].axhline(NOISE, color=GRAY, ls="--", lw=1, label=f"观测噪声 σ={NOISE}")
ax[1].set_title("外推误差演化（逐步 RMSE，64 样本平均）"); ax[1].set_xlabel("t (s)"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_extrapolation.png"); plt.close(fig)

# ---- E2 步长扫描 ----
hs = [0.04, 0.02, 0.01, 0.005, 0.0025]
sweep = {"rk4": [], "euler": []}
gt_fine = true_dense[:, N_T + 1:]
for h in hs:
    n_h = int(round((N_T + N_EXT) * 0.04 / h))
    tr_rk = rollout(m_node, x0s[SUB].to(device), n_h, h, "rk4").detach().cpu()
    tr_eu = rollout(m_euler, x0s[SUB].to(device), n_h, h, "euler").detach().cpu()
    idx = np.arange(N_T + N_EXT + 1) * round(0.04 / h)
    tr_rk = tr_rk[:, idx]; tr_eu = tr_eu[:, idx]
    sweep["rk4"].append(float(((tr_rk[:, N_T + 1:] - gt_fine[SUB]) ** 2).mean().sqrt()))
    sweep["euler"].append(float(((tr_eu[:, N_T + 1:] - gt_fine[SUB]) ** 2).mean().sqrt()))
euler_big = []
for h in [0.16, 0.08]:
    m_tmp, lh_tmp, st = train_node(x0s, Y_TRAIN_N, epochs=400, h=h, method="euler")
    euler_big.append({"h": h, "loss_end": round(float(lh_tmp[-1]), 5), "step_ms": round(st * 1000, 2)})
R["e2"] = {"hs": hs, "rmse_rk4": [round(v, 6) for v in sweep["rk4"]],
           "rmse_euler": [round(v, 6) for v in sweep["euler"]],
           "rk4_loglog_slope": round(float(np.polyfit(np.log(hs), np.log(np.maximum(sweep["rk4"], 1e-12)), 1)[0]), 2),
           "euler_loglog_slope": round(float(np.polyfit(np.log(hs), np.log(np.maximum(sweep["euler"], 1e-12)), 1)[0]), 2),
           "euler_big_h_train": euler_big,
           "step_ms_rk4": round(step_t_rk4 * 1000, 2), "step_ms_euler": round(step_t_eul * 1000, 2),
           "rk4_loss_end": round(float(lh_node[-1]), 5), "euler_loss_end": round(float(lh_euler[-1]), 5)}

# ---- 图 2 ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].loglog(hs, sweep["rk4"], "o-", color=RED, label=f"RK4 模型（log-log 斜率 {R['e2']['rk4_loglog_slope']}）")
ax[0].loglog(hs, sweep["euler"], "s--", color=BLUE, label=f"Euler 模型（log-log 斜率 {R['e2']['euler_loglog_slope']}）")
ax[0].axvline(0.04, color=GRAY, ls=":", lw=1)
ax[0].text(0.041, 2e-4, "训练所用步长", fontsize=8, color=GRAY)
ax[0].set_title("推理步长 h vs 外推段 RMSE（同一个训练好的网络）")
ax[0].set_xlabel("h (s)"); ax[0].set_ylabel("RMSE"); ax[0].legend(fontsize=9)
et2 = euler_big + [{"h": "0.04(RK4)", "loss_end": R["e2"]["rk4_loss_end"], "step_ms": R["e2"]["step_ms_rk4"]}]
xs = np.arange(len(et2))
bars = ax[1].bar(xs, [d["loss_end"] for d in et2], 0.55, color=[ORANGE, ORANGE, RED])
for x0_, d in zip(xs, et2):
    ax[1].text(x0_, d["loss_end"], f"损失 {d['loss_end']:.5f}\n单步 {d['step_ms']:.0f} ms",
               ha="center", va="bottom", fontsize=8.5)
ax[1].set_xticks(xs)
ax[1].set_xticklabels([f"Euler h={d['h']}" if i < 2 else "RK4 h=0.04" for i, d in enumerate(et2)])
ax[1].set_ylim(0, et2[0]["loss_end"] * 1.35)
ax[1].set_title("粗步 Euler 训练的隐性代价：h 太粗连数据都拟合不上")
ax[1].set_ylabel("训练损失末值 (MSE)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_solver_step.png"); plt.close(fig)

# ---- E3 不规则采样 ----
dens = [1, 2, 4, 8]
node_rmse = {}; lstm_by_lag = {}
B3 = 96
x03 = x0s[:B3]
for s in dens:
    cols = np.arange(N_T)
    mask = torch.zeros(B1, N_T)
    mask[:, (cols % s) == 0] = 1.0
    m_m, lh_m, _ = train_node(x0s, Y_TRAIN_N, mask=mask, epochs=300, h=h_data, seed=SEED + 7 * s)
    traj_m = rollout(m_m, x03.to(device), N_T, h_data, "rk4").detach().cpu()[:, 1:, :]
    eval_cols = np.arange(48, N_T)
    node_rmse[s] = float(((traj_m[:B3][:, eval_cols] - Y_TRAIN[:B3][:, eval_cols]) ** 2).mean((-1, 0, 1)).sqrt())
    ff = Y_TRAIN_N[:B3].clone()
    for t in range(1, N_T):
        ff[:, t] = torch.where(mask[:B3, [t]] > 0, Y_TRAIN_N[:B3, t], ff[:, t - 1])
    pts = []
    for t_eval in eval_cols:
        lo = t_eval - (t_eval % s)
        lo = max(lo, WIN - 1)
        lag = t_eval - lo
        seedw = ff[:, lo - WIN + 1: lo + 1]
        pr = seedw
        with torch.no_grad():
            for _ in range(max(1, lag)):
                nxt = lstm(pr.to(device))
                pr = torch.cat([pr[:, 1:, :], nxt[:, None, :].cpu()], 1)
        err = float(((pr[:, -1] - true_dense[:B3, t_eval + 1]) ** 2).mean().sqrt())
        pts.append((lag, err))
    arr = np.array(pts)
    lstm_by_lag[s] = {int(lg): float(arr[:, 1][arr[:, 0] == lg].mean()) for lg in np.unique(arr[:, 0]).astype(int)}
    R.setdefault("e3", {})[f"dens{s}"] = {
        "node_rmse": round(node_rmse[s], 4),
        "lstm_rmse_mean": round(float(arr[:, 1].mean()), 4),
        "lstm_rmse_at_maxlag": round(lstm_by_lag[s][max(lstm_by_lag[s])], 4)}

# ---- 图 3 ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
xg = np.arange(len(dens)); w2 = 0.38
nvals = [node_rmse[s] for s in dens]
lvals = [R["e3"][f"dens{s}"]["lstm_rmse_mean"] for s in dens]
b1_ = ax[0].bar(xg - w2 / 2, nvals, w2, color=RED, label="Neural ODE（mask 损失）")
b2_ = ax[0].bar(xg + w2 / 2, lvals, w2, color=BLUE, label="单步 MLP（前向填充+自回归）")
for b, v in list(zip(b1_, nvals)) + list(zip(b2_, lvals)):
    ax[0].text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", fontsize=7.5, va="bottom")
ax[0].set_xticks(xg); ax[0].set_xticklabels([f"1/{d}" for d in dens])
ax[0].set_xlabel("观测密度（每 d 步 1 个观测）"); ax[0].set_ylabel("测试段 RMSE")
ax[0].set_title("观测变稀：Neural ODE 几乎不痛，离散模型明显受伤"); ax[0].legend(fontsize=9)
for s, c in zip(dens, [GRAY, GREEN, PURPLE, ORANGE]):
    d = lstm_by_lag[s]; lgs = sorted(d)
    ax[1].plot(lgs, [d[l] for l in lgs], "o-", ms=3, color=c, label=f"1/{s} 密度")
ax[1].set_xlabel("距上一个真实观测的步数（lag）"); ax[1].set_ylabel("测试点 RMSE")
ax[1].set_title("离散模型误差随『距上次观测的步数』爬升"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_irregular.png"); plt.close(fig)

# ---- E4 OU 漂移场 ----
KAPPA, SIG_OU = 0.8, 0.5
B4, T4 = 256, 60
torch.manual_seed(SEED + 1)
x = torch.empty(B4, 1).normal_(0, 1.0)
X4 = [x.clone()]
for t in range(T4):
    for _ in range(25):
        x = x - KAPPA * x * 0.002 + SIG_OU * math.sqrt(0.002) * torch.randn(B4, 1)
    X4.append(x.clone())
X4 = torch.stack(X4, 1)
Y4 = X4[:, 1:] + 0.15 * torch.randn(B4, T4, 1)
m_ou, lh_ou, _ = train_node(X4[:, 0], Y4, epochs=400, h=0.05, lr=5e-3, D=1)
with torch.no_grad():
    gx = torch.linspace(-2.2, 2.2, 221)[:, None]
    fhat = m_ou(gx.to(device)).cpu().numpy().squeeze()
slope = float(np.polyfit(gx.numpy().squeeze(), fhat, 1)[0])
R["e4"] = {"kappa_true": KAPPA, "kappa_hat": round(-slope, 4),
           "bias_pct": round(abs(slope + KAPPA) / KAPPA * 100, 2),
           "half_life_true": round(math.log(2) / KAPPA, 4),
           "half_life_hat": round(math.log(2) / max(-slope, 1e-9), 4),
           "ou_stationary_sd": round(SIG_OU / math.sqrt(2 * KAPPA), 4),
           "half_life_bias_pct": round(abs(math.log(2) / max(-slope, 1e-9) - math.log(2) / KAPPA) / (math.log(2) / KAPPA) * 100, 1)}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
samp = X4[:, ::15].numpy().ravel()
sd = SIG_OU / math.sqrt(2 * KAPPA)
xs_ = np.linspace(-2.6, 2.6, 220)
ax[0].hist(samp, bins=70, density=True, color=GRAY, alpha=0.55, label="OU 样本（每 15 步取样）")
ax[0].plot(xs_, np.exp(-xs_**2 / (2 * sd * sd)) / (sd * np.sqrt(2 * math.pi)), color=BLACK, lw=1.5, label=f"平稳密度 σ={sd:.3f}")
ax[0].legend(fontsize=8); ax[0].set_title(f"OU 数据：κ={KAPPA}, σ={SIG_OU}，Euler–Maruyama 细步模拟，15% 观测噪声")
ax[1].plot(gx.numpy(), fhat, color=RED, lw=2, label=f"学到的 f(x)：斜率 {slope:.3f}")
ax[1].plot(gx.numpy(), -KAPPA * gx.numpy(), color=BLACK, ls="--", lw=1.5, label=f"真漂移 −κx (κ={KAPPA})")
ax[1].set_xlabel("x"); ax[1].set_ylabel("f(x) = dx/dt"); ax[1].legend(fontsize=9)
ax[1].set_title(f"漂移场恢复：κ̂={-slope:.3f}（偏差 {R['e4']['bias_pct']}%）→ 半衰期 {R['e4']['half_life_hat']:.3f} vs 真 {R['e4']['half_life_true']:.3f}")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_ou_drift.png"); plt.close(fig)

# ---- E5 Lorenz 外推时限 ----
SIG_L, RHO_L, BETA_L = 10.0, 28.0, 8.0 / 3.0
def lorenz(x):
    s_, r_ = x[..., 1] - x[..., 0], x[..., 0] * (RHO_L - x[..., 2]) - x[..., 1]
    return torch.stack([SIG_L * s_, r_, x[..., 0] * x[..., 1] - BETA_L * x[..., 2]], -1)
H5, N5, B5 = 0.01, 50, 256
torch.manual_seed(SEED + 2)
x0l = (torch.rand(B5, 3) * 2 - 1) * 6 + torch.tensor([0., 0., 25.])
l_true = rollout(lorenz, x0l, N5 * 6, H5, "rk4")
Yl = l_true[:, 1:N5 + 1] + 0.01 * torch.randn(B5, N5, 3)
m_l, lh_l, _ = train_node(x0l, Yl, epochs=1200, h=H5, lr=8e-3, D=3, W=64)
l_pred = rollout(m_l, x0l.to(device), N5 * 6, H5, "rk4")[:, 1:, :].detach().cpu()
scale = l_true[:, 1:, :].std(dim=(0, 1), keepdim=True)
err5 = (((l_pred - l_true[:, 1:, :]) / scale) ** 2).mean(-1).sqrt().mean(0).numpy()
t5 = np.arange(1, N5 * 6 + 1) * H5
over = np.where(err5 > 0.5)[0]
t_usable = float(t5[over[0]]) if len(over) else float(t5[-1])
fitm = (t5 > N5 * H5 * 0.5) & (err5 > 1e-3) & (err5 < 0.45)
if fitm.sum() < 8:
    fitm = (t5 > N5 * H5 * 0.5) & (err5 > 1e-3) & (err5 < err5.max() * 0.5)
lg_slope = float(np.polyfit(t5[fitm], np.log(err5[fitm]), 1)[0])
R["e5"] = {"train_window": round(N5 * H5, 3), "error_at_win_end": round(float(err5[N5 - 1]), 4),
           "error_at_3win": round(float(err5[3 * N5 - 1]), 4), "error_at_end": round(float(err5[-1]), 3),
           "t_usable": round(t_usable, 3), "usable_over_trainwin": round(t_usable / (N5 * H5), 2),
           "fitted_growth_rate": round(lg_slope, 3), "lyapunov_true": 0.906,
           "l10_cross": round(float(t5[np.where(err5 > 0.1)[0][0]]), 3)}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
sc = ax[0].scatter(l_true[12, :N5, 0], l_true[12, :N5, 1], c=np.arange(N5), cmap="viridis", s=7)
ax[0].scatter(l_pred[12, :, 0], l_pred[12, :, 1], s=2, c=RED, alpha=0.35, label="Neural ODE 外推(6 倍长)")
ax[0].legend(fontsize=8); ax[0].set_title("Lorenz 吸引子：训练窗(彩, t≤0.512) vs 外推(红)")
ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
ax[1].semilogy(t5, np.maximum(err5, 1e-5), color=RED, lw=1.5)
ax[1].axhline(0.5, color=GRAY, ls="--", lw=1); ax[1].axvline(N5 * H5, color=ORANGE, ls="--", lw=1)
ax[1].text(N5 * H5 + 0.02, 1.1, "训练终点", fontsize=8, color=ORANGE)
ax[1].text(0.02, 0.62, f"误差过 0.5：t≈{t_usable:.2f}（训练窗 {t_usable/(N5*H5):.1f} 倍）\n指数拟合增长率 {lg_slope:.2f}/单位时间", fontsize=8)
ax[1].set_title("混沌外推时限：误差 e^λt 放大直到完全跑偏（状态归一化 RMSE）")
ax[1].set_xlabel("t (时间单位)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_lorenz_horizon.png"); plt.close(fig)

R["train_loss"] = {"node_rk4_end": round(float(lh_node[-1]), 5),
                   "gru_one_step_end": round(float(lh_lstm[-1]), 5),
                   "lorenz_end": round(float(lh_l[-1]), 5), "ou_end": round(float(lh_ou[-1]), 5)}
with open(f"{OUT}/_node_results.json", "w") as fp:
    json.dump(R, fp, ensure_ascii=False, indent=1, default=float)
print(json.dumps(R, ensure_ascii=False, indent=1, default=float))
s = json.dumps(R)
print("NAN/inf check:", ("nan" in s.lower()) or ("inf" in s.lower()))
