#!/usr/bin/env python3
"""Gerber 统计量配图生成"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = ["PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "..", "public", "images", "gerber-statistic-covariance")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(7)

# ---------- 市场构造：N=20, 4 板块, t(4) 肥尾 ----------
N, n_sec = 20, 4
sec = np.repeat(np.arange(n_sec), N // n_sec)
beta_m = rng.uniform(0.8, 1.2, N)
beta_s = rng.uniform(0.5, 0.9, N)
sig_m, sig_s = 0.15, 0.10
vol_idio = 0.18 * rng.uniform(0.7, 1.4, N)

Sigma_true = np.outer(beta_m, beta_m) * sig_m**2
for s in range(n_sec):
    bs = beta_s * (sec == s)
    Sigma_true += np.outer(bs, bs) * sig_s**2
Sigma_true += np.diag(vol_idio**2)
d_true = np.sqrt(np.diag(Sigma_true))
R_true = Sigma_true / np.outer(d_true, d_true)
L = np.linalg.cholesky(Sigma_true / 252.0)

def sim_returns(T, df=4, seed=None):
    r = np.random.default_rng(seed)
    z = r.standard_t(df, size=(T, N))
    z /= np.sqrt(df / (df - 2))
    return z @ L.T

def gerber_matrix(X, c=0.5, variant="2022"):
    """variant='naive': g=(nc-nd)/(nc+nd)  最初版本，非 PSD 且高估共动
       variant='2022' : g=(nc-nd)/(T-n_nn)  Gerber-Markowitz-Sargen 2022 正式定义"""
    T, n = X.shape
    s = X.std(axis=0, ddof=1)
    U = (X >= c * s).astype(float) - (X <= -c * s).astype(float)  # +1/-1/0
    pos = (U > 0).astype(float); neg = (U < 0).astype(float); neu = (U == 0).astype(float)
    conc = pos.T @ pos + neg.T @ neg
    disc = pos.T @ neg + neg.T @ pos
    if variant == "naive":
        denom = np.maximum(conc + disc, 1)
    else:
        n_nn = neu.T @ neu  # 双双落在噪声区
        denom = np.maximum(T - n_nn, 1)
    G = (conc - disc) / denom
    np.fill_diagonal(G, 1.0)
    return G

def nearest_psd(A, eps=1e-8):
    w, V = np.linalg.eigh((A + A.T) / 2)
    w = np.clip(w, eps, None)
    B = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(B))
    return B / np.outer(d, d)

def frob(A, B):
    return np.linalg.norm(A - B, "fro")

def lw_shrink_corr(X):
    """极简 Ledoit-Wolf 收缩到常相关目标（用于对比）"""
    T = X.shape[0]
    S = np.corrcoef(X.T)
    iu = np.triu_indices(N, 1)
    rho_bar = S[iu].mean()
    F = np.full_like(S, rho_bar); np.fill_diagonal(F, 1.0)
    # 简化 delta：固定基于 T 的经验值
    delta = min(1.0, max(0.0, 6.0 / np.sqrt(T)))
    return delta * F + (1 - delta) * S

# ---------- 图1：Gerber vs Pearson 对异常日的稳健性 ----------
T = 252
X = sim_returns(T, seed=11)
# 注入 1 个极端日：所有资产 -8σ 同向 + 少量噪声（市场崩盘日）
X_dirty = X.copy()
crash = -8 * X.std(axis=0) * rng.uniform(0.8, 1.2, N)
X_dirty[100] = crash

n_pair = N * (N - 1) // 2
iu = np.triu_indices(N, 1)

P_clean = np.corrcoef(X.T)[iu]
P_dirty = np.corrcoef(X_dirty.T)[iu]
G_clean = gerber_matrix(X)[iu]
G_dirty = gerber_matrix(X_dirty)[iu]
print("corr(G_clean, P_clean) =", np.corrcoef(G_clean, P_clean)[0, 1])
print("G mean vs P mean:", G_clean.mean(), P_clean.mean())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
axes[0].scatter(P_clean, P_dirty, s=14, alpha=0.6, color="#d62728")
axes[0].plot([-0.2, 1], [-0.2, 1], "k--", lw=1)
axes[0].set_xlabel("干净样本的皮尔逊相关")
axes[0].set_ylabel("注入 1 个崩盘日后")
axes[0].set_title(f"皮尔逊：单日污染平均抬升 {np.mean(P_dirty-P_clean):+.3f}")
axes[1].scatter(G_clean, G_dirty, s=14, alpha=0.6, color="#1f77b4")
axes[1].plot([-0.2, 1], [-0.2, 1], "k--", lw=1)
axes[1].set_xlabel("干净样本的 Gerber 统计量")
axes[1].set_ylabel("注入 1 个崩盘日后")
axes[1].set_title(f"Gerber：平均漂移仅 {np.mean(G_dirty-G_clean):+.3f}")
fig.suptitle("一个 −8σ 崩盘日对两种相关度量的冲击（T=252, N=20）", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/gerber-outlier-robust.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("图1 done", np.mean(P_dirty - P_clean), np.mean(G_dirty - G_clean))

# ---------- 图2：阈值 c 敏感性 ----------
cs = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
n_mc = 40
err_clean = {c: [] for c in cs}
err_noisy = {c: [] for c in cs}
daily_vol = np.sqrt(np.diag(Sigma_true) / 252)
for mc in range(n_mc):
    Xm = sim_returns(252, seed=100 + mc)
    noise = np.random.default_rng(9000 + mc).normal(0, 0.6 * daily_vol, size=Xm.shape)
    Xn = Xm + noise  # 叠加独立微观结构噪声（买卖价弹跳）
    for c in cs:
        err_clean[c].append(frob(gerber_matrix(Xm, c), R_true))
        err_noisy[c].append(frob(gerber_matrix(Xn, c), R_true))

fig, ax1 = plt.subplots(figsize=(9, 5))
m_c = [np.mean(err_clean[c]) for c in cs]
m_n = [np.mean(err_noisy[c]) for c in cs]
ax1.errorbar(cs, m_c, yerr=[np.std(err_clean[c]) for c in cs], marker="o", color="#1f77b4", capsize=3, label="干净数据")
ax1.errorbar(cs, m_n, yerr=[np.std(err_noisy[c]) for c in cs], marker="s", color="#d62728", capsize=3, label="叠加微观结构噪声（0.6×日波动）")
ax1.set_xlabel("阈值 c（单位：σ）")
ax1.set_ylabel("与真实相关矩阵的 Frobenius 距离")
for c_, mc_, mn_ in zip(cs, m_c, m_n):
    ax1.annotate(f"+{(mn_/mc_-1)*100:.0f}%", (c_, mn_), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=8, color="#d62728")
ax1.set_title("阈值 c 的权衡：小 c 基线误差低但被噪声重创，大 c 抗噪但样本饥饿")
ax1.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/gerber-threshold-scan.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("图2 done clean", dict(zip(cs, np.round(m_c, 3))))
print("图2 noisy", dict(zip(cs, np.round(m_n, 3))))

# ---------- 图3：GMV 组合样本外波动率 ----------
def gmv_weights(Sig):
    ones = np.ones(N)
    w = np.linalg.solve(Sig, ones)
    return w / w.sum()

Ts = [63, 126, 252, 504]
n_mc2 = 60
res = {"样本协方差": {t: [] for t in Ts}, "LW 收缩": {t: [] for t in Ts},
       "Gerber 2022 (c=0.5)": {t: [] for t in Ts}, "Gerber 原始分母": {t: [] for t in Ts},
       "真实协方差": {t: [] for t in Ts}}
for mc in range(n_mc2):
    for T_ in Ts:
        Xm = sim_returns(T_, seed=2000 + mc * 10 + Ts.index(T_))
        s = Xm.std(axis=0, ddof=1)
        S_samp = np.cov(Xm.T)
        R_lw = lw_shrink_corr(Xm)
        S_lw = R_lw * np.outer(s, s)
        G = nearest_psd(gerber_matrix(Xm, 0.5))
        S_g = G * np.outer(s, s)
        Gn = nearest_psd(gerber_matrix(Xm, 0.5, variant="naive"))
        S_gn = Gn * np.outer(s, s)
        for name, Sig in [("样本协方差", S_samp), ("LW 收缩", S_lw),
                          ("Gerber 2022 (c=0.5)", S_g), ("Gerber 原始分母", S_gn),
                          ("真实协方差", Sigma_true / 252)]:
            try:
                w = gmv_weights(Sig)
                vol = np.sqrt(w @ (Sigma_true / 252) @ w * 252)
                res[name][T_].append(vol * 100)
            except np.linalg.LinAlgError:
                pass

fig, ax = plt.subplots(figsize=(9, 5))
colors = {"样本协方差": "#d62728", "LW 收缩": "#2ca02c", "Gerber 2022 (c=0.5)": "#1f77b4",
          "Gerber 原始分母": "#9467bd", "真实协方差": "#7f7f7f"}
for name in res:
    m = [np.mean(res[name][t]) for t in Ts]
    ax.plot(Ts, m, marker="o", label=name, color=colors[name],
            ls="--" if name == "真实协方差" else "-")
ax.set_xscale("log"); ax.set_xticks(Ts); ax.set_xticklabels(Ts)
ax.set_xlabel("估计窗口 T（天）")
ax.set_ylabel("GMV 组合真实年化波动率 (%)")
ax.set_title("GMV 判决：t(4) 肥尾市场下的样本外真实波动率")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/gerber-gmv-vol.png", dpi=110, bbox_inches="tight")
plt.close(fig)
for name in res:
    print(name, {t: round(np.mean(res[name][t]), 2) for t in Ts})

# ---------- 图4：三区间分类示意（散点 + 阈值带） ----------
Xp = sim_returns(504, seed=42)
i, j = 0, 1  # 同板块两只
si, sj = Xp[:, i].std(), Xp[:, j].std()
c = 0.5
ui = (Xp[:, i] >= c * si).astype(int) - (Xp[:, i] <= -c * si).astype(int)
uj = (Xp[:, j] >= c * sj).astype(int) - (Xp[:, j] <= -c * sj).astype(int)
conc = (ui * uj) > 0
disc = (ui * uj) < 0
both_neu = (ui == 0) & (uj == 0)
neut = ~conc & ~disc

fig, ax = plt.subplots(figsize=(8.5, 6.5))
mixed = neut & ~both_neu
ax.scatter(Xp[both_neu, i] * 100, Xp[both_neu, j] * 100, s=12, color="#bbbbbb", alpha=0.6, label=f"双双静止（剔出分母，{both_neu.sum()}）")
ax.scatter(Xp[mixed, i] * 100, Xp[mixed, j] * 100, s=12, color="#8c8c3c", alpha=0.5, label=f"单边动（计入分母，{mixed.sum()}）")
ax.scatter(Xp[conc, i] * 100, Xp[conc, j] * 100, s=16, color="#1f77b4", alpha=0.8, label=f"共动样本 n_c（{conc.sum()}）")
ax.scatter(Xp[disc, i] * 100, Xp[disc, j] * 100, s=16, color="#d62728", alpha=0.8, label=f"反向样本 n_d（{disc.sum()}）")
for v, s_ in [(c * si, "x"), (-c * si, "x")]:
    ax.axvline(v * 100, color="k", lw=0.8, ls=":")
for v in [c * sj, -c * sj]:
    ax.axhline(v * 100, color="k", lw=0.8, ls=":")
g_naive = (conc.sum() - disc.sum()) / (conc.sum() + disc.sum())
g_2022 = (conc.sum() - disc.sum()) / (len(Xp) - both_neu.sum())
ax.set_xlabel("资产 1 日收益 (%)")
ax.set_ylabel("资产 2 日收益 (%)")
ax.set_title(f"Gerber 分类（c=0.5σ）：原始 g={g_naive:.3f} vs 2022 定义 g={g_2022:.3f}，皮尔逊={np.corrcoef(Xp[:,i],Xp[:,j])[0,1]:.3f}")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/gerber-classification.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("图4 done g_naive=", g_naive, "g_2022=", g_2022)
print("ALL DONE")
