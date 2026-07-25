import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)
OUT = "public/images/tail-risk-parity-cvar"
os.makedirs(OUT, exist_ok=True)

BLUE = "#3b6ea5"; RED = "#c0392b"; GREEN = "#27ae60"; ORANGE = "#e67e22"; PURPLE = "#8e44ad"; GRAY = "#7f8c8d"

# ---------- synthetic assets: 关键设计——「短波动/信用」资产波动低但左尾极肥 ----------
# 波动率平价会因其低波动而超配它，尾部风险平价则识破其肥尾并压低权重。
N = 4
T = 3000
names = ["股票", "债券", "商品", "短波动信用"]
# 日波动率：短波动信用日常波动最低（比债券还低）——这正是 VP 会跳进去的陷阱
vols = np.array([0.16, 0.06, 0.20, 0.035]) / np.sqrt(252)
# 相关矩阵
corr = np.array([
    [1.00, -0.20, 0.35, 0.45],
    [-0.20, 1.00, -0.10, 0.05],
    [0.35, -0.10, 1.00, 0.25],
    [0.45, 0.05, 0.25, 1.00],
])
L = np.linalg.cholesky(corr)
z = rng.standard_normal((T, N)) @ L.T
# 基础高斯收益（正漂移，模拟长期上行市场）
mu_daily = np.array([0.0006, 0.00015, 0.0005, 0.00055])
rets = mu_daily + z * vols
# 共同崩盘日（股票/商品中等左尾）：频率低但确实发生
crash = rng.random(T) < 0.010
jump = np.where(crash, -np.abs(rng.normal(0.022, 0.012, T)), 0.0)
rets[:, 0] += jump * 0.9          # 股票：中等左尾
rets[:, 2] += jump * 0.6          # 商品：中等左尾
# 短波动信用：独立的、更稀的、灾难性左尾（平时收租看似平静，尾部巨亏）
credit_crash = rng.random(T) < 0.008
rets[:, 3] += np.where(credit_crash, -np.abs(rng.normal(0.09, 0.04, T)), 0.0)

# ---------- risk metrics ----------
def cvar(x, alpha=0.05):
    var = np.quantile(x, alpha)
    tail = x[x <= var]
    return -tail.mean() if len(tail) else -var

def vol(x):
    return x.std()

# 1. return distributions (highlight fat tails)
fig, ax = plt.subplots(figsize=(9, 4.2))
colors = [RED, BLUE, ORANGE, PURPLE]
for i in range(N):
    ax.hist(rets[:, i], bins=80, density=True, alpha=0.45, color=colors[i], label=names[i])
ax.axvline(0, color=GRAY, lw=0.8)
ax.set_xlim(-0.08, 0.05)
ax.set_title("四资产日收益分布：股票/另类左尾明显更肥")
ax.set_xlabel("日收益率"); ax.set_ylabel("密度")
ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/return_distributions.png", dpi=120); plt.close()

# 2. vol vs CVaR ranking (show they differ)
vols_ann = np.array([vol(rets[:, i]) for i in range(N)]) * np.sqrt(252)
cvars = np.array([cvar(rets[:, i]) for i in range(N)])
x = np.arange(N)
fig, ax1 = plt.subplots(figsize=(9, 4.2))
w = 0.38
ax1.bar(x - w/2, vols_ann, w, color=BLUE, label="年化波动率")
ax1.set_ylabel("年化波动率", color=BLUE)
ax1.tick_params(axis="y", labelcolor=BLUE)
ax2 = ax1.twinx()
ax2.bar(x + w/2, cvars*100, w, color=RED, label="日 CVaR(5%)")
ax2.set_ylabel("日 CVaR 5% (%)", color=RED)
ax2.tick_params(axis="y", labelcolor=RED)
ax1.set_xticks(x); ax1.set_xticklabels(names)
ax1.set_title("波动率 vs CVaR：排序不同——波动率低估肥尾资产的尾部风险")
plt.tight_layout(); plt.savefig(f"{OUT}/vol_vs_cvar.png", dpi=120); plt.close()

# ---------- CVaR risk contribution via Euler allocation ----------
def portfolio_cvar(w, R, alpha=0.05):
    pr = R @ w
    var = np.quantile(pr, alpha)
    tail_mask = pr <= var
    return -pr[tail_mask].mean(), tail_mask

def cvar_risk_contrib(w, R, alpha=0.05):
    # Euler: RC_i = -w_i * E[r_i | portfolio in tail]
    _, tail_mask = portfolio_cvar(w, R, alpha)
    cond_mean = R[tail_mask].mean(axis=0)  # E[r_i | tail]
    mrc = -cond_mean  # 边际贡献方向
    rc = w * mrc
    return rc

# ---------- optimize tail risk parity via projected gradient / iterative ----------
def tail_risk_parity(R, alpha=0.05, iters=6000, lr=0.03):
    n = R.shape[1]
    w = np.ones(n) / n
    for _ in range(iters):
        rc = cvar_risk_contrib(w, R, alpha)
        target = rc.mean()
        grad = rc - target
        w = w - lr * grad
        w = np.clip(w, 1e-4, None)
        w = w / w.sum()
    return w

w_trp = tail_risk_parity(rets)
w_ew = np.ones(N) / N
# vol parity (inverse vol)
iv = 1.0 / (vols_ann)
w_vp = iv / iv.sum()

# risk contributions comparison
rc_ew = cvar_risk_contrib(w_ew, rets); rc_ew /= rc_ew.sum()
rc_vp = cvar_risk_contrib(w_vp, rets); rc_vp /= rc_vp.sum()
rc_trp = cvar_risk_contrib(w_trp, rets); rc_trp /= rc_trp.sum()

fig, ax = plt.subplots(figsize=(9, 4.2))
w_bar = 0.26
ax.bar(x - w_bar, rc_ew*100, w_bar, color=GRAY, label="等权")
ax.bar(x, rc_vp*100, w_bar, color=BLUE, label="波动率平价")
ax.bar(x + w_bar, rc_trp*100, w_bar, color=GREEN, label="尾部风险平价")
ax.axhline(25, color=RED, ls="--", lw=1.0, label="均衡目标 25%")
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_title("CVaR 风险贡献占比：尾部风险平价把每个资产拉到 25%")
ax.set_ylabel("CVaR 风险贡献占比 (%)")
ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/risk_contribution.png", dpi=120); plt.close()

# ---------- weights comparison ----------
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.bar(x - w_bar, w_ew*100, w_bar, color=GRAY, label="等权")
ax.bar(x, w_vp*100, w_bar, color=BLUE, label="波动率平价")
ax.bar(x + w_bar, w_trp*100, w_bar, color=GREEN, label="尾部风险平价")
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_title("配置权重对比：尾部风险平价进一步压低肥尾资产")
ax.set_ylabel("权重 (%)")
ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/weights_comparison.png", dpi=120); plt.close()

# ---------- backtest equity curves ----------
def equity(w, R):
    pr = R @ w
    return np.cumprod(1 + pr)

eq_ew = equity(w_ew, rets)
eq_vp = equity(w_vp, rets)
eq_trp = equity(w_trp, rets)
t = np.arange(T)
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(t, eq_ew, color=GRAY, lw=1.2, label="等权")
ax.plot(t, eq_vp, color=BLUE, lw=1.2, label="波动率平价")
ax.plot(t, eq_trp, color=GREEN, lw=1.4, label="尾部风险平价")
ax.set_title("样本内净值曲线（同一段肥尾市场）")
ax.set_xlabel("交易日"); ax.set_ylabel("净值")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/equity_curves.png", dpi=120); plt.close()

# metrics
def stats(w, R):
    pr = R @ w
    ann_ret = pr.mean() * 252
    ann_vol = pr.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol
    eq = np.cumprod(1 + pr)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    cv = cvar(pr)
    return ann_ret, ann_vol, sharpe, dd, cv

for nm, w in [("等权", w_ew), ("波动率平价", w_vp), ("尾部风险平价", w_trp)]:
    ar, av, sh, dd, cv = stats(w, rets)
    print(f"{nm}: ret={ar:.3f} vol={av:.3f} sharpe={sh:.2f} maxDD={dd:.3f} CVaR={cv*100:.3f}%")

print("weights TRP:", np.round(w_trp, 3), "VP:", np.round(w_vp, 3))
print("done tail risk parity")
