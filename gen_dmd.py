#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""oct26b-B (v2): koopman-operator-dmd
Koopman 算子与动态模态分解（DMD）：把非线性动力学搬回线性空间。
v2 修正：
- E1 噪声扫描改 20 次重复取均值（单次实现的随机性会伪造非单调曲线）
- E2 换成制度切换波动率模型（阈值型条件期望=折线，强非线性）：
  低波制度慢回复 / 高波制度快回复；一步 MSE 看偏差 vs 不可约噪声分解，
  多步递推（确定性路径、无新噪声）看偏差累积
- E3 主导模态选择改为按模长（旧版按角度，被 angle≈π 的伪特征值劫持）
实验：
  E1  线性系统：DMD vs VAR(1) 特征值恢复 + 噪声扫描 + 多步预测
  E1b 高维低秩：50 维观测 / 5 因子真动力学，DMD(秩截断) vs VAR(1)
  E2  制度切换波动率：DMD/VAR vs EDMD（二次字典）
  E3  延迟嵌入：从单条含噪振荡序列恢复阻尼 γ 与频率 ω
输出 5 张图 + results json
"""
import os
import json
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

OUT = "/Users/halo/workspace/astro-blog/public/images/koopman-operator-dmd"
os.makedirs(OUT, exist_ok=True)
RED, GREEN, BLUE, GRAY, ORANGE, BLACK, PURPLE = "#d63031", "#00b894", "#0984e3", "#636e72", "#e17055", "#2d3436", "#6c5ce7"
rng = np.random.default_rng(20261026)
R = {}

# ============================================================
# DMD 基础
# ============================================================
def dmd(X, Y, rank=None, tol=1e-10):
    """标准 DMD：X, Y 为 n×m 快照矩阵"""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if rank is None:
        rank = int(np.sum(s > tol * s[0]))
    r = min(rank, len(s))
    Ur, sr, Vr = U[:, :r], s[:r], Vt[:r, :]
    At = Ur.T @ Y @ Vr.T @ np.diag(1.0 / sr)
    lam, W = np.linalg.eig(At)
    Phi = Y @ Vr.T @ np.diag(1.0 / sr) @ W
    return lam, Phi, s

def dmd_operator(lam, Phi):
    return np.real(Phi @ np.diag(lam) @ np.linalg.pinv(Phi))

def mse_multi(A, x, horizons):
    res = []
    cur = x[0].copy()
    for k in range(max(horizons)):
        cur = A @ cur
        if (k + 1) in horizons:
            res.append(float(np.real(np.mean((cur - x[k + 1]) ** 2))))
    return res

# ============================================================
# E1: 线性系统 + 噪声扫描（20 次重复）
# ============================================================
def make_M(lam1, lam_re, lam_im, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    J = np.array([[lam1, 0, 0], [0, lam_re, -lam_im], [0, lam_im, lam_re]])
    return Q @ J @ Q.T

M_true = make_M(0.95, 0.40, 0.40 * np.sqrt(3), rng)   # 复对: |λ|=0.8, angle=60°
ev_true = np.sort_complex(np.linalg.eigvals(M_true))
print(f"[B] E1 true eigenvalues: {ev_true.round(4)}")

def sim_linear(T, M, sig, rng):
    x = np.zeros((T, 3))
    x[0] = rng.standard_normal(3)
    for t in range(T - 1):
        x[t + 1] = M @ x[t] + sig * rng.standard_normal(3)
    return x

def eig_err(lam_hat, ev_true):
    errs = []
    used = [False] * len(lam_hat)
    for lt in ev_true:
        d = [abs(lt - lh) if not used[i] else 9 for i, lh in enumerate(lam_hat)]
        j = int(np.argmin(d)); used[j] = True
        errs.append(abs(lt - lam_hat[j]))
    return float(np.mean(errs))

noise_grid = [0.0, 0.01, 0.03, 0.10, 0.30]
H = [1, 5, 10, 20]
NREP = 20
e1_res = {}
for sig in noise_grid:
    ed, ev, md, mv = [], [], [], []
    for rep in range(NREP):
        x = sim_linear(2000, M_true, sig, rng)
        X, Y = x[:-1].T, x[1:].T
        lam_d, Phi_d, _ = dmd(X, Y, rank=3)
        A_var = Y @ np.linalg.pinv(X)
        lam_v = np.linalg.eigvals(A_var)
        ed.append(eig_err(lam_d, ev_true)); ev.append(eig_err(lam_v, ev_true))
        md.append(mse_multi(dmd_operator(lam_d, Phi_d), x, H))
        mv.append(mse_multi(A_var, x, H))
    md = np.mean(md, axis=0); mv = np.mean(mv, axis=0)
    e1_res[str(sig)] = {"eig_dmd": float(np.mean(ed)), "eig_dmd_sd": float(np.std(ed)),
                        "eig_var": float(np.mean(ev)), "eig_var_sd": float(np.std(ev)),
                        "mse_dmd": [float(v) for v in md], "mse_var": [float(v) for v in mv]}
    print(f"[B] E1 sig={sig}: eig err DMD={np.mean(ed):.4f}±{np.std(ed):.4f} VAR={np.mean(ev):.4f}±{np.std(ev):.4f} | "
          f"MSE1 DMD={md[0]:.5f} VAR={mv[0]:.5f} | MSE20 DMD={md[-1]:.5f} VAR={mv[-1]:.5f}")
R["e1"] = e1_res
R["e1_ev_true"] = [float(abs(v)) for v in ev_true]

# ============================================================
# E1b: 高维低秩：50 维观测 = 5 因子 × 载荷 + 噪声
# ============================================================
n_obs, n_fac = 50, 5
L_load = rng.standard_normal((n_obs, n_fac)) / np.sqrt(n_fac)
Qf, _ = np.linalg.qr(rng.standard_normal((n_fac, n_fac)))
Mf = Qf @ np.diag([0.9, 0.85, 0.8, 0.75, 0.7]) @ Qf.T
T_hi = 400
f_hi = np.zeros((T_hi, n_fac))
f_hi[0] = rng.standard_normal(n_fac)
for t in range(T_hi - 1):
    f_hi[t + 1] = Mf @ f_hi[t] + 0.1 * rng.standard_normal(n_fac)
x_hi = f_hi @ L_load.T + 0.3 * rng.standard_normal((T_hi, n_obs))
tr_hi, te_hi = x_hi[:300], x_hi[300:]
Xh, Yh = tr_hi[:-1].T, tr_hi[1:].T
lam_h5, Phi_h5, sv_h5 = dmd(Xh, Yh, rank=5)
A_dmd5 = dmd_operator(lam_h5, Phi_h5)
A_var_hi = Yh @ np.linalg.pinv(Xh)
A_ridge = Yh @ Xh.T @ np.linalg.inv(Xh @ Xh.T + 1.0 * np.eye(n_obs))
def test_mse(A):
    pred = np.real(A @ te_hi[:-1].T).T
    return float(np.mean((pred - te_hi[1:]) ** 2))
mse_hi = {"dmd_r5": test_mse(A_dmd5), "var": test_mse(A_var_hi), "ridge": test_mse(A_ridge)}
rank_sweep = {}
for rk in [1, 2, 3, 5, 8, 12, 20, 50]:
    lam_r, Phi_r, _ = dmd(Xh, Yh, rank=rk)
    rank_sweep[str(rk)] = test_mse(dmd_operator(lam_r, Phi_r))
R["e1b"] = {"mse": mse_hi, "rank_sweep": rank_sweep,
            "irreducible": 0.09}
print(f"[B] E1b: {mse_hi}")
print(f"[B] E1b rank sweep: { {k: round(v,4) for k,v in rank_sweep.items()} }")

# 画图用数据
x_clean = sim_linear(600, M_true, 0.0, rng)
lam_c, Phi_c, _ = dmd(x_clean[:-1].T, x_clean[1:].T, rank=3)
x_noisy = sim_linear(600, M_true, 0.10, rng)
lam_n, Phi_n, _ = dmd(x_noisy[:-1].T, x_noisy[1:].T, rank=3)

# ============================================================
# E2: 制度切换波动率（阈值型条件期望）
#   v<0.02: 慢回复到 0.010 (kappa=0.02);  v>=0.02: 快回复到 0.035 (kappa=0.30)
# ============================================================
T2 = 100000
v = np.zeros(T2); v[0] = 0.015
noise_v = 0.0008
for t in range(T2 - 1):
    if v[t] < 0.02:
        v[t + 1] = 0.010 + (v[t] - 0.010) * np.exp(-0.02) + noise_v * rng.standard_normal()
    else:
        v[t + 1] = 0.035 + (v[t] - 0.035) * np.exp(-0.30) + noise_v * rng.standard_normal()
    v[t + 1] = max(v[t + 1], 1e-4)
r = -0.5 * v + np.sqrt(v) * 0.008 * rng.standard_normal(T2)
obs = np.vstack([r, v]).T

def true_ce(vt):
    return np.where(vt < 0.02, 0.010 + (vt - 0.010) * np.exp(-0.02),
                    0.035 + (vt - 0.035) * np.exp(-0.30))

# 线性 VAR(1)（含截距）
Xl = np.hstack([np.ones((T2 - 1, 1)), obs[:-1]])
A_lin = np.linalg.lstsq(Xl, obs[1:], rcond=None)[0]     # (3,2)
pred_lin_v = (Xl @ A_lin)[:, 1]
ce_true = true_ce(v[:-1])
bias2_lin = float(np.mean((pred_lin_v - ce_true) ** 2))
tot_lin = float(np.mean((pred_lin_v - v[1:]) ** 2))

# EDMD 二次字典
def poly_dict(G):
    G = np.atleast_2d(G)
    r_, v_ = G[:, 0], G[:, 1]
    return np.vstack([np.ones_like(r_), r_, v_, r_ ** 2, r_ * v_, v_ ** 2]).T

D, Dn = poly_dict(obs[:-1]), poly_dict(obs[1:])
K2 = Dn.T @ np.linalg.pinv(D.T)
Psi = obs[:-1].T @ np.linalg.pinv(D.T)
pred_edmd = np.real(Psi @ (K2 @ D.T)).T
bias2_edmd = float(np.mean((pred_edmd[:, 1] - ce_true) ** 2))
tot_edmd = float(np.mean((pred_edmd[:, 1] - v[1:]) ** 2))
# 条件期望曲线（网格）
grid = np.linspace(0.004, 0.045, 200)
Xg = np.hstack([np.ones((len(grid), 1)), np.zeros((len(grid), 1)), grid.reshape(-1, 1)])
curve_lin = Xg @ A_lin
curve_lin_v = curve_lin[:, 1]
Dg = poly_dict(np.vstack([np.zeros(len(grid)), grid]).T)
curve_edmd = np.real(Psi @ (K2 @ Dg.T)).T[:, 1]

# 多步递推（60 步，6 个起点）
def rec_lin(x0, steps):
    cur = np.concatenate([[1.0], x0]); out = []
    for k in range(steps):
        nxt = A_lin.T @ cur
        out.append(nxt)
        cur = np.concatenate([[1.0], nxt])
    return np.array(out)

def rec_edmd(x0, steps):
    g = poly_dict(np.asarray(x0, dtype=float).reshape(1, -1))[0]; out = []
    for k in range(steps):
        g = K2 @ g
        out.append(Psi @ g)
    return np.array(out)

hi_idx = np.where(v[:-1] > 0.028)[0]
lo_idx = np.where((v[:-1] > 0.008) & (v[:-1] < 0.012))[0]
start_idxs = list(hi_idx[::len(hi_idx) // 3][:3]) + list(lo_idx[::len(lo_idx) // 3][:3])
Hm = 60
mh_res = []
for i0 in start_idxs:
    tl = rec_lin(obs[i0], Hm); te = rec_edmd(obs[i0], Hm); tv = obs[i0 + 1:i0 + 1 + Hm]
    mh_res.append({"v0": float(v[i0]),
                   "mse_lin": float(np.mean((tl[:, 1] - tv[:, 1]) ** 2)),
                   "mse_edmd": float(np.mean((te[:, 1] - tv[:, 1]) ** 2))})
mse_lin_mh = float(np.mean([x["mse_lin"] for x in mh_res]))
mse_edmd_mh = float(np.mean([x["mse_edmd"] for x in mh_res]))
R["e2"] = {"bias2_lin": bias2_lin, "bias2_edmd": bias2_edmd,
           "tot_lin": tot_lin, "tot_edmd": tot_edmd,
           "noise_var": noise_v ** 2,
           "mse_lin_mh": mse_lin_mh, "mse_edmd_mh": mse_edmd_mh,
           "multi": mh_res, "v_mean": float(v.mean()), "v_std": float(v.std())}
print(f"[B] E2 1-step: bias^2 lin={bias2_lin:.3e} edmd={bias2_edmd:.3e} | total lin={tot_lin:.3e} edmd={tot_edmd:.3e}")
print(f"[B] E2 60-step: lin={mse_lin_mh:.3e} edmd={mse_edmd_mh:.3e} ratio={mse_lin_mh/mse_edmd_mh:.2f}")
for x in mh_res:
    print(f"     v0={x['v0']:.4f}: lin={x['mse_lin']:.3e} edmd={x['mse_edmd']:.3e}")

# 画路径用一个高波起点
i0_show = start_idxs[0]
mh_show_lin = rec_lin(obs[i0_show], 40)
mh_show_edmd = rec_edmd(obs[i0_show], 40)
mh_show_true = obs[i0_show + 1:i0_show + 41]

# ============================================================
# E3: 延迟嵌入恢复阻尼振荡
# ============================================================
gamma_t, omega_t = 0.02, 0.6
a1 = 2 * np.exp(-gamma_t) * np.cos(omega_t)
a2 = -np.exp(-2 * gamma_t)
T3 = 1500

def sim_osc(T, sn, rng):
    x = np.zeros(T)
    x[0] = rng.standard_normal() * 0.5
    x[1] = rng.standard_normal() * 0.5
    for t in range(1, T - 1):
        x[t + 1] = a1 * x[t] + a2 * x[t - 1] + sn * rng.standard_normal()
    return x

def hankel(x, m):
    n = len(x) - m + 1
    return np.vstack([x[i:i + m] for i in range(n)]).T

def dominant_pair(lam, tol=1e-8):
    nz = lam[np.abs(lam) > tol]
    if len(nz) < 2:
        return None
    order = np.argsort(-np.abs(nz))
    return nz[order[:2]]

x3 = sim_osc(T3, 0.1, rng)
e3_res = {}
for m in [4, 6, 10, 20, 40]:
    Hm3 = hankel(x3, m)
    lam_h, _, sv_h = dmd(Hm3[:, :-1], Hm3[:, 1:], rank=None, tol=1e-8)
    pair = dominant_pair(lam_h)
    om = abs(np.angle(pair[0])); gm = -np.log(np.abs(pair[0]))
    e3_res[str(m)] = {"omega": float(om), "gamma": float(gm),
                      "n_eig": int(np.sum(np.abs(lam_h) > 1e-8)),
                      "sv": [float(s) for s in sv_h[:6]]}
    print(f"[B] E3 m={m}: omega={om:.4f} (true {omega_t}) gamma={gm:.4f} (true {gamma_t}) rank={e3_res[str(m)]['n_eig']}")

e3_noise = {}
NREP3 = 10
for sn in [0.0, 0.1, 0.3, 0.5, 1.0]:
    oms, gms = [], []
    for rep in range(NREP3):
        x3n = sim_osc(T3, sn, rng)
        Hm3 = hankel(x3n, 10)
        lam_h, _, _ = dmd(Hm3[:, :-1], Hm3[:, 1:], rank=None, tol=1e-8)
        pair = dominant_pair(lam_h)
        oms.append(abs(np.angle(pair[0]))); gms.append(-np.log(np.abs(pair[0])))
    e3_noise[str(sn)] = {"omega": float(np.mean(oms)), "omega_sd": float(np.std(oms)),
                         "gamma": float(np.mean(gms)), "gamma_sd": float(np.std(gms)),
                         "snr": float(x3n.std() / max(sn, 1e-12))}
    print(f"[B] E3 noise={sn}: omega={np.mean(oms):.4f}±{np.std(oms):.4f} gamma={np.mean(gms):.4f}±{np.std(gms):.4f}")
R["e3"] = {"by_m": e3_res, "by_noise": e3_noise,
           "gamma_true": gamma_t, "omega_true": omega_t}

# ============================================================
# 图 1：E1 线性系统
# ============================================================
fig = plt.figure(figsize=(11, 7))
ax1 = fig.add_subplot(2, 3, 1)
for i in range(3):
    ax1.plot(x_clean[:, i], lw=0.8)
ax1.set_title("E1 干净线性系统轨迹", fontsize=9)
ax2 = fig.add_subplot(2, 3, 2)
for i in range(3):
    ax2.plot(x_noisy[:, i], lw=0.6, alpha=0.85)
ax2.set_title("E1 含噪系统（σ=0.10）", fontsize=9)
th = np.linspace(0, 2 * np.pi, 200)
for col, (lam_hat, ttl) in enumerate([(lam_c, "干净数据 DMD 特征值"), (lam_n, "含噪数据 DMD 特征值")]):
    ax = fig.add_subplot(2, 3, 4 + col)
    ax.plot(np.cos(th), np.sin(th), "k-", lw=0.8)
    ax.scatter(ev_true.real, ev_true.imag, marker="o", s=70, facecolors="none",
               edgecolors=BLACK, lw=1.5, label="真实")
    ax.scatter(np.real(lam_hat), np.imag(lam_hat), marker="x", s=60, color=RED, label="DMD")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal")
    ax.set_title(ttl, fontsize=9); ax.legend(fontsize=8)
ax5 = fig.add_subplot(2, 3, 6)
A_c = dmd_operator(lam_c, Phi_c)
rec = np.zeros((200, 3)); cur = x_clean[0].copy()
for k in range(200):
    cur = A_c @ cur; rec[k] = cur
for i in range(3):
    ax5.plot(x_clean[:200, i], lw=1.2, color=[RED, GREEN, BLUE][i], label=f"x{i+1} 真实")
    ax5.plot(rec[:, i], lw=0.9, ls="--", color=[RED, GREEN, BLUE][i], alpha=0.7)
ax5.set_title("DMD 重建（200 步）", fontsize=9); ax5.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "dmd_linear.png"), bbox_inches="tight")
plt.close()

# ============================================================
# 图 2：E1 噪声扫描 + E1b 高维低秩
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
sigs = [float(k) for k in e1_res.keys()]
axes[0, 0].errorbar(sigs, [e1_res[k]["eig_dmd"] for k in e1_res],
                    yerr=[e1_res[k]["eig_dmd_sd"] for k in e1_res],
                    fmt="o-", color=RED, label="DMD", capsize=3)
axes[0, 0].errorbar(sigs, [e1_res[k]["eig_var"] for k in e1_res],
                    yerr=[e1_res[k]["eig_var_sd"] for k in e1_res],
                    fmt="s-", color=BLUE, label="VAR(1)", capsize=3)
axes[0, 0].set_title("特征值平均误差 vs 噪声（20 次重复）", fontsize=9)
axes[0, 0].set_xlabel("σ_η"); axes[0, 0].legend(fontsize=8)
axes[0, 1].plot(H, e1_res["0.01"]["mse_dmd"], "o-", color=RED, label="DMD σ=0.01")
axes[0, 1].plot(H, e1_res["0.01"]["mse_var"], "s-", color=BLUE, label="VAR σ=0.01")
axes[0, 1].plot(H, e1_res["0.10"]["mse_dmd"], "o--", color=ORANGE, label="DMD σ=0.10")
axes[0, 1].plot(H, e1_res["0.10"]["mse_var"], "s--", color=GRAY, label="VAR σ=0.10")
axes[0, 1].set_yscale("log"); axes[0, 1].set_title("多步预测 MSE vs 步长", fontsize=9)
axes[0, 1].set_xlabel("预测步长"); axes[0, 1].legend(fontsize=8)
names_hi = ["DMD\n(秩5)", "VAR(1)\n(50×50)", "VAR\n(ridge)"]
vals_hi = [mse_hi["dmd_r5"], mse_hi["var"], mse_hi["ridge"]]
axes[1, 0].bar(names_hi, vals_hi, color=[RED, GRAY, BLUE])
axes[1, 0].axhline(0.09, color=BLACK, ls=":", lw=1, label="观测噪声下限 0.09")
axes[1, 0].set_title("E1b 高维低秩：测试集一步 MSE", fontsize=9)
axes[1, 0].set_yscale("log"); axes[1, 0].legend(fontsize=8)
for i, v in enumerate(vals_hi):
    axes[1, 0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
rks = [int(k) for k in rank_sweep.keys()]
axes[1, 1].plot(rks, [rank_sweep[k] for k in rank_sweep], "o-", color=RED)
axes[1, 1].axvline(5, color=BLACK, ls="--", lw=1, label="真实秩 = 5")
axes[1, 1].axhline(0.09, color=BLACK, ls=":", lw=1, label="噪声下限")
axes[1, 1].set_xscale("log"); axes[1, 1].set_yscale("log")
axes[1, 1].set_title("E1b DMD 秩扫描：测试 MSE vs 截断秩", fontsize=9)
axes[1, 1].set_xlabel("秩 r"); axes[1, 1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "dmd_noise.png"), bbox_inches="tight")
plt.close()

# ============================================================
# 图 3：E2 制度切换波动率
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
axes[0, 0].plot(v[:2000] * 100, lw=0.7, color=BLUE)
axes[0, 0].axhline(2.0, color=RED, ls="--", lw=1, label="制度阈值 2%")
axes[0, 0].set_title("E2 合成波动率 v（制度切换，前 2000 日）", fontsize=9)
axes[0, 0].set_xlabel("交易日"); axes[0, 0].legend(fontsize=8)
axes[0, 1].plot(grid * 100, true_ce(grid) * 100, color=BLACK, lw=1.8, label="真实条件期望（折线）")
axes[0, 1].plot(grid * 100, curve_lin_v * 100, color=ORANGE, lw=1.3, ls="--", label="线性 VAR/DMD")
axes[0, 1].plot(grid * 100, curve_edmd * 100, color=RED, lw=1.3, ls=":", label="EDMD（二次字典）")
axes[0, 1].set_title("一步条件期望：线性拉不弯折线", fontsize=9)
axes[0, 1].set_xlabel("v_t（%/日）"); axes[0, 1].set_ylabel("E[v_(t+1)|v_t]（%/日）")
axes[0, 1].legend(fontsize=8)
tt40 = np.arange(1, 41)
axes[1, 0].plot(tt40, mh_show_true[:, 1] * 100, color=BLACK, lw=1.5, label="真实")
axes[1, 0].plot(tt40, mh_show_lin[:, 1] * 100, color=ORANGE, lw=1.1, ls="--", label="线性 VAR/DMD")
axes[1, 0].plot(tt40, mh_show_edmd[:, 1] * 100, color=RED, lw=1.1, ls=":", label="EDMD")
axes[1, 0].set_title(f"40 步递推预测（起点 v0={v[i0_show]*100:.1f}%）", fontsize=9)
axes[1, 0].set_xlabel("步长"); axes[1, 0].legend(fontsize=8)
names3 = ["线性\n一步", "EDMD\n一步", "线性\n60步", "EDMD\n60步"]
vals = [tot_lin, tot_edmd, mse_lin_mh, mse_edmd_mh]
axes[1, 1].bar(names3, vals, color=[ORANGE, RED, ORANGE, RED])
axes[1, 1].set_title("预测 MSE（v 分量）", fontsize=9)
axes[1, 1].set_yscale("log")
for i, vv in enumerate(vals):
    axes[1, 1].text(i, vv, f"{vv:.1e}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "dmd_nonlinear.png"), bbox_inches="tight")
plt.close()

# ============================================================
# 图 4：E3 延迟嵌入
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
axes[0, 0].plot(x3, lw=0.7, color=BLUE, label="含噪振荡信号")
Hm10 = hankel(x3, 10)
lam_h10, _, sv_h10 = dmd(Hm10[:, :-1], Hm10[:, 1:], rank=None, tol=1e-8)
# DMD 算子直接对 Hankel 快照做最小二乘（满秩时与特征分解等价）
Kh = Hm10[:, 1:] @ np.linalg.pinv(Hm10[:, :-1])
rec3 = np.zeros((200, Hm10.shape[0]))
cur = Hm10[:, 0].copy()
for k in range(200):
    cur = Kh @ cur
    rec3[k] = cur
axes[0, 0].plot(np.arange(200), rec3[:, 0], lw=1.2, color=RED, ls="--", label="DMD 重建")
axes[0, 0].set_title("E3 阻尼振荡 + DMD 重建", fontsize=9); axes[0, 0].legend(fontsize=8)
axes[0, 1].stem(np.arange(1, len(sv_h10) + 1), sv_h10 / sv_h10[0], linefmt="-", markerfmt="o", basefmt=" ")
axes[0, 1].set_yscale("log")
axes[0, 1].set_title("Hankel 矩阵奇异值谱（m=10）", fontsize=9)
axes[0, 1].set_xlabel("序号"); axes[0, 1].set_ylabel("相对奇异值")
ms_ = [int(k) for k in e3_res.keys()]
axes[1, 0].plot(ms_, [e3_res[k]["omega"] for k in e3_res], "o-", color=RED, label="ω̂")
axes[1, 0].axhline(omega_t, color=BLACK, ls="--", lw=1, label="ω 真值")
axes[1, 0].set_title("恢复频率 vs 嵌入维数 m", fontsize=9); axes[1, 0].set_xlabel("m"); axes[1, 0].legend(fontsize=8)
axes[1, 1].plot(ms_, [e3_res[k]["gamma"] for k in e3_res], "s-", color=PURPLE, label="γ̂")
axes[1, 1].axhline(gamma_t, color=BLACK, ls="--", lw=1, label="γ 真值")
axes[1, 1].set_title("恢复阻尼 vs 嵌入维数 m", fontsize=9); axes[1, 1].set_xlabel("m"); axes[1, 1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "dmd_delay.png"), bbox_inches="tight")
plt.close()

# ============================================================
# 图 5：E2 EDMD 提升空间模态
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
lamK, WK = np.linalg.eig(K2)
g_init = poly_dict(obs[100])[0]
c0 = np.linalg.solve(WK, g_init)
modes = []
for k in range(500):
    gk = WK @ (lamK ** k * c0)
    modes.append(np.abs(gk))
modes = np.array(modes)
top2 = np.argsort(-np.abs(lamK))[:2]
for j, mi in enumerate(top2):
    axes[0].plot(modes[:, mi], lw=1.2, color=[RED, BLUE][j],
                 label=f"模式 {j+1} (|λ|={abs(lamK[mi]):.3f})")
axes[0].set_title("EDMD 提升空间模态幅度", fontsize=9); axes[0].set_xlabel("步"); axes[0].legend(fontsize=8)
axes[1].stem(np.abs(lamK), linefmt="-", markerfmt="o", basefmt=" ")
axes[1].axhline(1.0, color=BLACK, ls="--", lw=1)
axes[1].set_title("EDMD 算子特征值模长", fontsize=9); axes[1].set_xlabel("模式编号")
axes[2].scatter(np.angle(lamK), np.abs(lamK), color=BLUE, s=40)
axes[2].axhline(1.0, color=BLACK, ls="--", lw=1)
axes[2].set_title("特征值：角度（频率）vs 模长（衰减）", fontsize=9)
axes[2].set_xlabel("angle(λ)"); axes[2].set_ylabel("|λ|")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "dmd_modes.png"), bbox_inches="tight")
plt.close()

with open("/Users/halo/workspace/astro-blog/_dmd_results.json", "w") as f:
    json.dump(R, f, indent=1, ensure_ascii=False)
print("[B] all done. figures in", OUT)
