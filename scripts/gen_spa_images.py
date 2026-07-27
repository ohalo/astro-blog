# -*- coding: utf-8 -*-
"""Hansen SPA 检验配图：WRC vs SPA 对比模拟"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)
OUT = Path(__file__).resolve().parent.parent / "public/images/hansen-spa-test"
OUT.mkdir(parents=True, exist_ok=True)

T = 1250          # 5 年日频
B = 500           # bootstrap 次数
BLOCK = 20        # 块长
ANN = np.sqrt(252)
SIG = 0.01        # 日波动 1%


def make_pool(n_noise, n_junk, true_alpha_sharpe=None, seed=0):
    # 各组件独立 RNG：junk 数量变化不影响 noise / alpha 的随机实现
    r_noise = np.random.default_rng(seed * 1000 + 1)
    r_junk = np.random.default_rng(seed * 1000 + 2)
    r_alpha = np.random.default_rng(seed * 1000 + 3)
    cols = [r_noise.normal(0, SIG, size=(T, n_noise))]
    if n_junk > 0:
        # 垃圾策略：显著为负（年化约 -25%）
        cols.append(r_junk.normal(-0.001, SIG, size=(T, n_junk)))
    if true_alpha_sharpe is not None:
        mu = true_alpha_sharpe * SIG / ANN
        cols.append(r_alpha.normal(mu, SIG, size=(T, 1)))
    return np.hstack(cols)


def block_bootstrap_idx(r, T, block):
    n_blocks = T // block + 1
    starts = r.integers(0, T, n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % T
    return idx[:T]


def wrc_spa_pvalues(X, B=B, seed=1):
    """返回 (wrc_p, spa_p)。统计量：sqrt(T)*mean/std（学生化）"""
    r = np.random.default_rng(seed)
    Tn, M = X.shape
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    stat_obs = np.sqrt(Tn) * mu / sd
    V_obs = stat_obs.max()

    # SPA 重心化阈值：mu_k^c = mu_k * 1{ sqrt(T)*mu/sd <= -sqrt(2 loglog T) }
    thresh = -np.sqrt(2 * np.log(np.log(Tn)))
    keep_neg = stat_obs <= thresh          # 差策略：保留其负均值（不参与零分布抬升）
    mu_spa = np.where(keep_neg, mu, 0.0)   # 好/中性策略重心化到 0

    V_wrc = np.empty(B)
    V_spa = np.empty(B)
    for b in range(B):
        idx = block_bootstrap_idx(r, Tn, BLOCK)
        Xb = X[idx]
        mub = Xb.mean(axis=0)
        sdb = Xb.std(axis=0, ddof=1)
        # WRC：全部重心化到 0
        V_wrc[b] = (np.sqrt(Tn) * (mub - mu) / sdb).max()
        # SPA：仅对非差策略重心化
        V_spa[b] = np.maximum(np.sqrt(Tn) * (mub - mu + (mu - mu_spa) * 0 + (mu_spa - mu)) / sdb, 0).max() if False else (np.sqrt(Tn) * (mub - mu_spa - (mu - mu_spa)) / sdb).max()
    # 上式化简：mub - mu + (mu - mu_spa)？ 重新明确写：
    # WRC 零分布样本：sqrt(T)(mub - mu)/sdb
    # SPA 零分布样本：sqrt(T)(mub - mu)/sdb，但仅对 keep_neg=False 的策略取 max；
    # 差策略以其观测均值为中心，其对 max 的贡献为 sqrt(T)(mub - mu)/sdb + sqrt(T)mu/sd << 0，可忽略
    return (np.mean(V_wrc >= V_obs), None, stat_obs, V_wrc)


def wrc_spa_full(X, B=B, seed=1):
    """规范实现：返回 wrc_p, spa_p, 观测统计量, 两套零分布样本"""
    r = np.random.default_rng(seed)
    Tn, M = X.shape
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    stat_obs = np.sqrt(Tn) * mu / sd
    V_obs = max(stat_obs.max(), 0.0)

    thresh = -np.sqrt(2 * np.log(np.log(Tn)))
    is_junk = stat_obs <= thresh

    V_wrc = np.empty(B)
    V_spa = np.empty(B)
    for b in range(B):
        idx = block_bootstrap_idx(r, Tn, BLOCK)
        Xb = X[idx]
        mub = Xb.mean(axis=0)
        sdb = Xb.std(axis=0, ddof=1)
        z = np.sqrt(Tn) * (mub - mu) / sdb   # 以观测均值为中心的波动
        V_wrc[b] = max(z.max(), 0.0)
        # SPA：差策略中心移回其(负)观测统计量 → 对 max 几乎无贡献
        z_spa = z + np.where(is_junk, stat_obs, 0.0)
        V_spa[b] = max(z_spa.max(), 0.0)
    return (np.mean(V_wrc >= V_obs), np.mean(V_spa >= V_obs), stat_obs, V_wrc, V_spa)


# ============ 实验 1：含垃圾策略时两套零分布对比 ============
X1 = make_pool(n_noise=20, n_junk=80, true_alpha_sharpe=0.9, seed=5)
wrc_p1, spa_p1, stat1, Vw1, Vs1 = wrc_spa_full(X1, seed=11)
print(f"[图1] 20噪声+80垃圾+1真alpha(SR0.9): WRC p={wrc_p1:.3f}, SPA p={spa_p1:.3f}, 最优观测统计量={stat1.max():.2f}")

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=130)
bins = np.linspace(0, max(Vw1.max(), Vs1.max()) * 1.05, 45)
ax.hist(Vw1, bins=bins, alpha=0.55, color="#d62728", label=f"WRC 零分布（全池重心化）p={wrc_p1:.2f}")
ax.hist(Vs1, bins=bins, alpha=0.55, color="#1f77b4", label=f"SPA 零分布（剔除差策略）p={spa_p1:.2f}")
ax.axvline(stat1.max(), color="k", lw=2, ls="--", label=f"观测最优统计量 = {stat1.max():.2f}")
ax.set_xlabel("最大学生化统计量")
ax.set_ylabel("频数")
ax.set_title("同一份数据：80 个垃圾策略把 WRC 零分布推向右侧，SPA 几乎不受影响")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "spa-null-dist.jpg")
plt.close(fig)

# ============ 实验 2：p 值 vs 垃圾策略数量 ============
junk_counts = [0, 10, 20, 40, 80, 150]
wrc_ps, spa_ps = [], []
for k in junk_counts:
    Xk = make_pool(n_noise=20, n_junk=k, true_alpha_sharpe=0.9, seed=5)
    wp, sp, _, _, _ = wrc_spa_full(Xk, seed=11)
    wrc_ps.append(wp)
    spa_ps.append(sp)
    print(f"[图2] junk={k}: WRC p={wp:.3f}, SPA p={sp:.3f}")

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=130)
ax.plot(junk_counts, wrc_ps, "o-", color="#d62728", lw=2, ms=7, label="WRC p 值")
ax.plot(junk_counts, spa_ps, "s-", color="#1f77b4", lw=2, ms=7, label="SPA（consistent）p 值")
ax.axhline(0.05, color="gray", ls=":", lw=1.5, label="显著性阈值 0.05")
ax.set_xlabel("池中垃圾策略数量（年化约 -25% 的明显差策略）")
ax.set_ylabel("p 值")
ax.set_title("真 alpha（SR 0.9）固定不变：垃圾策略越多 WRC 越迟钝，SPA 稳定显著")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "spa-pvalue-vs-junk.jpg")
plt.close(fig)

# ============ 实验 3：SPA 三个 p 值（lower/consistent/upper）============
def spa_three(X, B=B, seed=1):
    r = np.random.default_rng(seed)
    Tn, M = X.shape
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    stat_obs = np.sqrt(Tn) * mu / sd
    V_obs = max(stat_obs.max(), 0.0)
    thresh = -np.sqrt(2 * np.log(np.log(Tn)))

    # 三种重心化：l=只留非负策略为0其余按观测, c=阈值法, u=全部为0(=WRC)
    shift_l = np.where(stat_obs < 0, stat_obs, 0.0)       # 所有负策略都按其观测中心
    shift_c = np.where(stat_obs <= thresh, stat_obs, 0.0) # 仅显著差的
    shift_u = np.zeros(M)                                  # 全部重心化（WRC）
    Vl = np.empty(B); Vc = np.empty(B); Vu = np.empty(B)
    for b in range(B):
        idx = block_bootstrap_idx(r, Tn, BLOCK)
        Xb = X[idx]
        z = np.sqrt(Tn) * (Xb.mean(0) - mu) / Xb.std(0, ddof=1)
        Vl[b] = max((z + shift_l).max(), 0.0)
        Vc[b] = max((z + shift_c).max(), 0.0)
        Vu[b] = max((z + shift_u).max(), 0.0)
    return (np.mean(Vl >= V_obs), np.mean(Vc >= V_obs), np.mean(Vu >= V_obs))

configs = {
    "纯噪声池\n(50 噪声)": make_pool(50, 0, None, seed=3),
    "噪声+垃圾\n(20 噪声+80 垃圾)": make_pool(20, 80, None, seed=3),
    "含真 alpha\n(20 噪声+80 垃圾+SR0.9)": make_pool(20, 80, 0.9, seed=5),
}
labels3 = list(configs.keys())
res3 = [spa_three(X, seed=13) for X in configs.values()]
for lab, (pl, pc, pu) in zip(labels3, res3):
    print(f"[图3] {lab.replace(chr(10),' ')}: lower={pl:.3f} consistent={pc:.3f} upper={pu:.3f}")

fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=130)
xpos = np.arange(len(labels3))
w = 0.25
ax.bar(xpos - w, [r[0] for r in res3], w, color="#2ca02c", label="SPA lower（下界）")
ax.bar(xpos,     [r[1] for r in res3], w, color="#1f77b4", label="SPA consistent（推荐）")
ax.bar(xpos + w, [r[2] for r in res3], w, color="#d62728", label="SPA upper（=WRC，上界）")
ax.axhline(0.05, color="gray", ls=":", lw=1.5)
ax.text(2.35, 0.06, "0.05", color="gray")
ax.set_xticks(xpos, labels3)
ax.set_ylabel("p 值")
ax.set_title("SPA 的三个 p 值：consistent 夹在 lower 与 upper（WRC）之间")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "spa-three-pvalues.jpg")
plt.close(fig)

print("done ->", OUT)
