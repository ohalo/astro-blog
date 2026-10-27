#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""oct26b-A: Realized GARCH 文章配图 + 核心统计
模型（Hansen-Huang-Shek 2012 的乘法形式）:
  r_t = sqrt(h_t) z_t,  h_t = w + a r_{t-1}^2 + b h_{t-1}
  x_t = h_t * exp(delta z_t + s_u eta_t - 0.5(delta^2+s_u^2)),  E[x_t|F_{t-1}] = h_t
x_t = 已实现波动测度（日内平方收益和），含杠杆通道 delta 与乘性噪声 s_u。
对比估计量: GARCH(1,1) 收益QMLE / RV-GARCH 递推用x / RGARCH 联合QMLE(5参数)。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize

for f in ["Heiti TC", "PingFang SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130
rcParams["figure.facecolor"] = "white"
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.3

OUT = "/Users/halo123456/workspace/astro-blog/public/images/realized-garch"
os.makedirs(OUT, exist_ok=True)
RED, GREEN, BLUE, GRAY, ORANGE, BLACK, PURPLE = "#d63031", "#00b894", "#0984e3", "#636e72", "#e17055", "#2d3436", "#6c5ce7"
LOG2PI = np.log(2 * np.pi)

# ---------------- DGP ----------------
W0, A0, B0, D0, SU0 = 0.02, 0.08, 0.90, -0.15, 0.30


def simulate(n, seed, w=W0, a=A0, b=B0, delta=D0, su=SU0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    eta = rng.standard_normal(n)
    h = np.empty(n); rv = np.empty(n); ret = np.empty(n)
    h[0] = w / (1 - a - b)
    for t in range(n):
        ret[t] = np.sqrt(h[t]) * z[t]
        u = np.exp(delta * z[t] + su * eta[t] - 0.5 * (delta ** 2 + su ** 2))
        rv[t] = h[t] * u
        if t + 1 < n:
            h[t + 1] = w + a * ret[t] ** 2 + b * h[t]
    return ret, rv, h, z


# ---------------- 估计器 ----------------
def _unpack_garch(p):
    return np.exp(p[0]), 1 / (1 + np.exp(-p[1])), 1 / (1 + np.exp(-p[2]))


def garch_qmle(ret, x0s=((-3.0, -2.6, 2.0), (-3.5, -2.2, 2.5))):
    """标准 GARCH(1,1)，只用收益率"""
    def negll(p):
        w, a, b = _unpack_garch(p)
        if a + b >= 0.999:
            return 1e12
        n = len(ret)
        h = w / (1 - a - b)
        ll = -0.5 * (LOG2PI + np.log(h) + ret[0] ** 2 / h)
        for t in range(1, n):
            h = w + a * ret[t - 1] ** 2 + b * h
            ll += -0.5 * (LOG2PI + np.log(h) + ret[t] ** 2 / h)
        return -ll
    best = None
    for x0 in x0s:
        res = minimize(negll, x0, method="Nelder-Mead",
                       options={"maxiter": 3000, "xatol": 1e-7, "fatol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    return _unpack_garch(best.x)


def rv_garch_qmle(rv, x0s=((-3.0, -2.6, 2.0, -1.2), (-3.5, -2.2, 2.5, -0.5))):
    """RV-GARCH: h_t = w + a x_{t-1} + b h_{t-1}，x_t = h_t u_t, u 对数正态"""
    def negll(p):
        w = np.exp(p[0]); a = 1 / (1 + np.exp(-p[1])); b = 1 / (1 + np.exp(-p[2])); s = np.exp(p[3])
        if a + b >= 0.999 or s > 3:
            return 1e12
        n = len(rv)
        h = w / (1 - a - b)
        ll = -np.log(rv[0]) - np.log(s) - 0.5 * LOG2PI \
            - 0.5 * ((np.log(rv[0]) - np.log(h) + 0.5 * s ** 2) / s) ** 2
        for t in range(1, n):
            h = w + a * rv[t - 1] + b * h
            ll += -np.log(rv[t]) - np.log(s) - 0.5 * LOG2PI \
                - 0.5 * ((np.log(rv[t]) - np.log(h) + 0.5 * s ** 2) / s) ** 2
        return -ll
    best = None
    for x0 in x0s:
        res = minimize(negll, x0, method="Nelder-Mead",
                       options={"maxiter": 3000, "xatol": 1e-7, "fatol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    return np.exp(best.x[0]), 1 / (1 + np.exp(-best.x[1])), 1 / (1 + np.exp(-best.x[2])), np.exp(best.x[3])


def rgarch_qmle(ret, rv, s_hint=None, x0s=None):
    """Realized GARCH 联合 QMLE: 收益方程 + 测量方程, 参数 (w,a,b,delta,s_u)"""
    if x0s is None:
        s_h = np.log(s_hint) if s_hint is not None else -1.2
        x0s = ((-3.0, -2.4, 2.0, -1.5, s_h), (-3.5, -2.0, 2.5, -0.8, s_h + 0.5),
               (-2.5, -2.8, 1.5, -2.0, s_h - 0.3))
    def negll(p):
        w = np.exp(p[0]); a = 1 / (1 + np.exp(-p[1])); b = 1 / (1 + np.exp(-p[2]))
        dl = p[3]; s = np.exp(p[4])
        if a + b >= 0.999 or s > 3:
            return 1e12
        n = len(ret)
        h = w / (1 - a - b)
        z = ret[0] / np.sqrt(h)
        ll = -0.5 * (LOG2PI + np.log(h) + ret[0] ** 2 / h) \
            - np.log(rv[0]) - np.log(s) - 0.5 * LOG2PI \
            - 0.5 * ((np.log(rv[0]) - np.log(h) - dl * z + 0.5 * (dl ** 2 + s ** 2)) / s) ** 2
        for t in range(1, n):
            h = w + a * ret[t - 1] ** 2 + b * h
            z = ret[t] / np.sqrt(h)
            ll += -0.5 * (LOG2PI + np.log(h) + ret[t] ** 2 / h) \
                - np.log(rv[t]) - np.log(s) - 0.5 * LOG2PI \
                - 0.5 * ((np.log(rv[t]) - np.log(h) - dl * z + 0.5 * (dl ** 2 + s ** 2)) / s) ** 2
        return -ll
    best = None
    for x0 in x0s:
        res = minimize(negll, x0, method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    return (np.exp(best.x[0]), 1 / (1 + np.exp(-best.x[1])), 1 / (1 + np.exp(-best.x[2])),
            best.x[3], np.exp(best.x[4]))


def filter_garch(ret, params):
    """返回 h_t (F_{t-1} 可测) 与一步预测序列 h_fc[t] = E[h_t|F_{t-1}]"""
    w, a, b = params
    n = len(ret)
    h = np.empty(n); h[0] = w / (1 - a - b)
    for t in range(1, n):
        h[t] = w + a * ret[t - 1] ** 2 + b * h[t - 1]
    return h


def filter_rvgarch(rv, params):
    w, a, b = params[0], params[1], params[2]
    n = len(rv)
    h = np.empty(n); h[0] = w / (1 - a - b)
    for t in range(1, n):
        h[t] = w + a * rv[t - 1] + b * h[t - 1]
    return h


def qlike(f, h):
    f = np.maximum(f, 1e-12)
    return np.mean(h / f - np.log(h / f) - 1.0)


def mse(f, h):
    return np.mean((f - h) ** 2)


# ================= E0 主实验数据 =================
N, OOS = 2500, 500
ret, rv, h, z = simulate(N, 20261026)
tr = slice(0, N - OOS)
te = slice(N - OOS, N)

g_par = garch_qmle(ret[tr])
x_par = rv_garch_qmle(rv[tr])
r_par = rgarch_qmle(ret[tr], rv[tr])
print("[E0] GARCH   w,a,b =", np.round(g_par, 4))
print("[E0] RV-GARCH w,a,b,s =", np.round(x_par, 4))
print("[E0] RGARCH  w,a,b,delta,s =", np.round(r_par, 4))
print("[E0] true    w,a,b,delta,s =", (W0, A0, B0, D0, SU0))

# 样本内过滤 + OOS 预测（h_f[t] = E[h_t|F_{t-1}]，与真实 h_t 对齐）
h_g = filter_garch(ret, g_par)
h_x = filter_rvgarch(rv, x_par)
h_r = filter_garch(ret, r_par[:3])

fc_g = h_g[N - OOS + 1:N]  # E[h_{t+1}|F_t], t = N-OOS..N-2
fc_x = h_x[N - OOS + 1:N]
fc_r = h_r[N - OOS + 1:N]
h_true = h[N - OOS + 1:N]

print("\n[E2] OOS 预测真实 h_{t+1}:")
print(f"  GARCH    QLIKE={qlike(fc_g, h_true):.5f}  MSE={mse(fc_g, h_true):.3e}")
print(f"  RV-GARCH QLIKE={qlike(fc_x, h_true):.5f}  MSE={mse(fc_x, h_true):.3e}")
print(f"  RGARCH   QLIKE={qlike(fc_r, h_true):.5f}  MSE={mse(fc_r, h_true):.3e}")

# 对 x_{t+1} 的预测 (RV-GARCH/RGARCH 的 E[x]=h; HAR 基准)
def har_fit_forecast(rv, oos_start):
    """HAR: x_{t+1} = c + cd*mean(rv[t-3:t+1]) + cw*mean(rv[t-20:t+1]) + cm*mean(rv[t-62:t+1])"""
    def feats(t):
        return [1.0, rv[t - 3:t + 1].mean(), rv[t - 20:t + 1].mean(), rv[t - 62:t + 1].mean()]
    X = np.array([feats(t) for t in range(62, oos_start)])
    y = rv[63:oos_start + 1]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    preds = np.array([np.dot(feats(t), beta) for t in range(oos_start, len(rv) - 1)])
    return preds

har_pred = har_fit_forecast(rv, N - OOS)
x_true = rv[N - OOS + 1:N]
print("\n[E2b] OOS 预测已实现测度 x_{t+1} (QLIKE):")
print(f"  GARCH(代理)   {qlike(fc_g, x_true):.5f}")
print(f"  RV-GARCH      {qlike(fc_x, x_true):.5f}")
print(f"  RGARCH        {qlike(fc_r, x_true):.5f}")
print(f"  HAR-RV        {qlike(har_pred, x_true):.5f}")

# ================= E1 参数恢复 (20 seeds) =================
NS = 20
rec_g = np.zeros((NS, 3)); rec_r = np.zeros((NS, 5))
for i in range(NS):
    rt, xv, hh, zz = simulate(N, 5000 + i)
    rec_g[i] = garch_qmle(rt[tr])
    rec_r[i] = rgarch_qmle(rt[tr], xv[tr])
rmse_g = np.sqrt(np.mean((rec_g - np.array([W0, A0, B0])) ** 2, axis=0))
rmse_r = np.sqrt(np.mean((rec_r - np.array([W0, A0, B0, D0, SU0])) ** 2, axis=0))
print("\n[E1] 参数 RMSE (20 seeds):")
print("  GARCH  (w,a,b):", np.round(rmse_g, 5))
print("  RGARCH (w,a,b,delta,s):", np.round(rmse_r, 5))
print(f"  delta 估计均值={rec_r[:,3].mean():.4f} (真值 {D0}), std={rec_r[:,3].std():.4f}")

# ================= E3 噪声扫描 =================
su_grid = [0.10, 0.20, 0.30, 0.50, 0.80]
n_scan = 6
scan_g, scan_x, scan_r = [], [], []
for su in su_grid:
    qg, qx, qr = [], [], []
    for i in range(n_scan):
        rt, xv, hh, zz = simulate(N, 9000 + i, su=su)
        gp = garch_qmle(rt[tr]); xp = rv_garch_qmle(xv[tr]); rp = rgarch_qmle(rt[tr], xv[tr], s_hint=su)
        fg = filter_garch(rt, gp)[N - OOS + 1:N]
        fx = filter_rvgarch(xv, xp)[N - OOS + 1:N]
        fr = filter_garch(rt, rp[:3])[N - OOS + 1:N]
        ht = hh[N - OOS + 1:N]
        qg.append(qlike(fg, ht)); qx.append(qlike(fx, ht)); qr.append(qlike(fr, ht))
    scan_g.append(np.mean(qg)); scan_x.append(np.mean(qx)); scan_r.append(np.mean(qr))
    print(f"[E3] s_u={su:.2f}  QLIKE GARCH={np.mean(qg):.6f}  RV-GARCH={np.mean(qx):.6f}  RGARCH={np.mean(qr):.6f}")
scan_g = np.array(scan_g); scan_x = np.array(scan_x); scan_r = np.array(scan_r)
print("[E3] ratio RV/GARCH:", np.round(scan_x / scan_g, 4))
print("[E3] ratio RG/GARCH:", np.round(scan_r / scan_g, 4))

# ================= 图 1: 数据面板 =================
fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.2), sharex=True)
w = slice(500, 1500)
axes[0].plot(np.arange(N)[w], ret[w], color=GRAY, lw=0.7, label="日收益 $r_t$")
axes[0].plot(np.arange(N)[w], h[w], color=RED, lw=1.6, label="真实条件方差 $h_t$")
axes[0].set_ylabel("收益率 / 方差")
axes[0].legend(loc="upper right", fontsize=9)
axes[0].set_title("合成市场：日收益与已实现波动测度（含杠杆通道 δ=−0.15）", fontsize=11)
axes[1].plot(np.arange(N)[w], rv[w], color=BLUE, lw=0.8, label="已实现测度 $x_t$（带噪的 $h_t$）")
axes[1].plot(np.arange(N)[w], h[w], color=RED, lw=1.6, label="真实 $h_t$")
axes[1].set_ylabel("$x_t$"); axes[1].set_xlabel("交易日")
axes[1].legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sim_panel.png"), bbox_inches="tight")
plt.close(fig)

# ================= 图 2: 测量方程 =================
hh = filter_garch(ret, r_par[:3])
zz = ret / np.sqrt(hh)
lx = np.log(rv) - np.log(hh)
fig, ax = plt.subplots(figsize=(7.6, 5.0))
ax.scatter(zz, lx, s=9, alpha=0.45, color=BLUE, edgecolors="none")
sl, ic = np.polyfit(zz, lx, 1)
xs = np.linspace(zz.min(), zz.max(), 50)
ax.plot(xs, sl * xs + ic, color=RED, lw=2,
        label=f"OLS 斜率 = {sl:.3f}（真值 δ = {D0}）")
ax.axhline(0, color=GRAY, lw=0.8, ls="--")
ax.set_xlabel("标准化收益冲击 $z_t$")
ax.set_ylabel("$\\log(x_t/h_t)$")
ax.set_title("测量方程：坏消息日（z<0）的已实现波动系统性偏高——杠杆效应", fontsize=11)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "measurement_eq.png"), bbox_inches="tight")
plt.close(fig)
print(f"[图2] 测量方程 OLS 斜率 = {sl:.4f}")

# ================= 图 3: 参数恢复 =================
labels = ["$\\omega$", "$a$", "$b$", "$\\delta$", "$s_u$"]
true_v = np.array([W0, A0, B0, D0, SU0])
rg_rmse = np.concatenate([rmse_r[:3], [rec_r[:, 3].std()], [rmse_r[4]]])
gg_rmse = np.concatenate([rmse_g, [np.nan, np.nan]])
fig, ax = plt.subplots(figsize=(8.2, 4.6))
xx = np.arange(5)
ax.bar(xx - 0.19, gg_rmse, width=0.38, color=GRAY, label="GARCH(1,1)（只用收益）")
ax.bar(xx + 0.19, rg_rmse, width=0.38, color=RED, label="Realized GARCH（联合 QMLE）")
for i, v in enumerate(rg_rmse):
    ax.text(i + 0.19, v + 0.0012, f"{v:.4f}", ha="center", fontsize=8.5, color=RED)
for i, v in enumerate(gg_rmse[:3]):
    ax.text(i - 0.19, v + 0.0012, f"{v:.4f}", ha="center", fontsize=8.5, color=GRAY)
ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("估计误差 RMSE（20 条独立路径）")
ax.set_title("已实现测度把 (a, b) 的估计误差压下去；δ 与 s_u 只有联合模型能估", fontsize=11)
ax.legend(fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "param_recovery.png"), bbox_inches="tight")
plt.close(fig)

# ================= 图 4: 预测评估 =================
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
models = ["GARCH", "RV-GARCH", "Realized\nGARCH"]
ql = [qlike(fc_g, h_true), qlike(fc_x, h_true), qlike(fc_r, h_true)]
ms = [mse(fc_g, h_true), mse(fc_x, h_true), mse(fc_r, h_true)]
axes[0].bar(models, ql, color=[GRAY, BLUE, RED], width=0.55)
for i, v in enumerate(ql):
    axes[0].text(i, v + max(ql) * 0.01, f"{v:.4f}", ha="center", fontsize=9.5)
axes[0].set_ylabel("QLIKE（越低越好）")
axes[0].set_title("预测真实 $h_{t+1}$：QLIKE", fontsize=11)
axes[0].set_ylim(0, max(ql) * 1.18)
ax2 = axes[1]
ww = slice(100, 300)
ax2.plot(np.arange(200), h_true[ww], color=BLACK, lw=1.4, label="真实 $h_{t+1}$")
ax2.plot(np.arange(200), fc_g[ww], color=GRAY, lw=1.1, label="GARCH")
ax2.plot(np.arange(200), fc_x[ww], color=RED, lw=1.1, label="Realized GARCH 类")
ax2.set_xlabel("样本外第 100–300 天")
ax2.set_title("波动率预测轨迹对比", fontsize=11)
ax2.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "forecast_eval.png"), bbox_inches="tight")
plt.close(fig)

# ================= 图 5: 噪声扫描 =================
fig, ax = plt.subplots(figsize=(8.6, 5.0))
ax.plot(su_grid, scan_x / scan_g, "o-", color=ORANGE, lw=1.8, ms=6,
        label="RV-GARCH（递推用 $x_t$）/ GARCH")
ax.plot(su_grid, scan_r / scan_g, "s-", color=RED, lw=1.8, ms=6,
        label="Realized GARCH（联合 QMLE）/ GARCH")
ax.axhline(1.0, color=GRAY, lw=1.2, ls=":")
ax.set_yscale("log")
ax.set_yticks([0.01, 0.1, 1.0, 10.0, 100.0])
ax.set_yticklabels(["0.01", "0.1", "1（打平）", "10", "100"])
for xx, yy in zip(su_grid, scan_x / scan_g):
    ax.annotate(f"{yy:.0f}×", (xx, yy), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8.5, color=ORANGE)
for xx, yy in zip(su_grid, scan_r / scan_g):
    ax.annotate(f"{yy:.3f}", (xx, yy), textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=8.5, color=RED)
ax.set_xlabel("测量噪声 $s_u$（对数正态标准差）")
ax.set_ylabel("QLIKE 比值（对数轴）")
ax.set_title("同一个已实现测度，两种用法两种命：递推换入始终惨败，联合估计始终占优", fontsize=11)
ax.legend(fontsize=9.5, loc="center left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "noise_scan.png"), bbox_inches="tight")
plt.close(fig)

print("\n[图] 5 张已保存至", OUT)
