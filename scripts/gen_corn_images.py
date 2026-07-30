#!/usr/bin/env python3
"""生成 CORN 模式匹配组合文章配图（向量化版）"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "public/images/corn-pattern-matching"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

# ================= 合成市场：反相位周期模式 + 噪声 =================
# 偶数日：资产A +0.3%、资产B -0.25%；奇数日反过来。噪声 0.85%。
# 这是一个真实存在可匹配模式的市场——模式匹配应该能吃到，CRP 吃不到。
T = 1250
amp_a, amp_b, sig = 0.003, 0.0025, 0.0085
parity = np.arange(T) % 2
mu = np.zeros((T, 2))
mu[parity == 0, 0] = +amp_a
mu[parity == 0, 1] = -amp_b
mu[parity == 1, 0] = -amp_b
mu[parity == 1, 1] = +amp_a
rets = mu + sig * rng.standard_normal((T, 2))
X = 1.0 + rets
prices = np.cumprod(X, axis=0)

# 对照市场：同波动、无模式
X_ctrl = 1.0 + sig * rng.standard_normal((T, 2))

w_len = 5
rho = 0.2


def corn_run(X, w=5, rho=0.2, grid=101, min_samples=5):
    """向量化 CORN：返回每日权重 b_t（资产A占比）与匹配数"""
    T = len(X)
    wm = w * X.shape[1]
    # 窗口矩阵：行 i 对应预测日 t=i+w，窗口 X[t-w:t]
    n_rows = T - w
    W = np.empty((n_rows, wm))
    for i in range(n_rows):
        W[i] = X[i:i + w].ravel()
    Wz = (W - W.mean(axis=1, keepdims=True)) / (W.std(axis=1, keepdims=True) + 1e-12)
    C = (Wz @ Wz.T) / wm  # 相关系数矩阵

    b_grid = np.linspace(0, 1, grid)
    B = np.vstack([b_grid, 1 - b_grid])  # (2, grid)
    bs = np.full(T, 0.5)
    n_match = np.zeros(T, dtype=int)
    for i in range(n_rows):
        t = i + w
        jmax = i - w  # 只允许不重叠的历史窗口
        if jmax <= 0:
            continue
        mask = C[i, :jmax] >= rho
        cand_days = np.nonzero(mask)[0] + w
        n_match[t] = len(cand_days)
        if len(cand_days) >= min_samples:
            R = X[cand_days]            # (n, 2)
            wealth = np.log(R @ B).sum(axis=0)
            bs[t] = b_grid[int(np.argmax(wealth))]
    return bs, n_match


bs, n_match = corn_run(X, w=w_len, rho=rho)
bs_ctrl, _ = corn_run(X_ctrl, w=w_len, rho=rho)

wealth_corn = np.cumprod(X[:, 0] * bs + X[:, 1] * (1 - bs))
wealth_corn_ctrl = np.cumprod(X_ctrl[:, 0] * bs_ctrl + X_ctrl[:, 1] * (1 - bs_ctrl))
wealth_5050 = np.cumprod(0.5 * X[:, 0] + 0.5 * X[:, 1])
wealth_5050_ctrl = np.cumprod(0.5 * X_ctrl[:, 0] + 0.5 * X_ctrl[:, 1])

b_grid_full = np.linspace(0, 1, 201)
finals = [np.prod(b * X[:, 0] + (1 - b) * X[:, 1]) for b in b_grid_full]
b_star = b_grid_full[int(np.argmax(finals))]
wealth_bcrp = np.cumprod(b_star * X[:, 0] + (1 - b_star) * X[:, 1])

# 完美先知（知道 parity 的分段最优）作为上限参考
bs_oracle = np.where(parity == 0, 1.0, 0.0)
wealth_oracle = np.cumprod(X[:, 0] * bs_oracle + X[:, 1] * (1 - bs_oracle))

print(f"b* = {b_star:.3f}")
print(f"CORN 终值 {wealth_corn[-1]:.3f}")
print(f"BCRP 终值 {wealth_bcrp[-1]:.3f}")
print(f"50/50 终值 {wealth_5050[-1]:.3f}")
print(f"A only {prices[-1,0]:.3f}  B only {prices[-1,1]:.3f}")
print(f"先知上限 {wealth_oracle[-1]:.3f}")
print(f"对照市场: CORN {wealth_corn_ctrl[-1]:.3f}  50/50 {wealth_5050_ctrl[-1]:.3f}")
print(f"平均匹配数(后半段) {n_match[T//2:].mean():.0f}")

# 匹配集的 parity 纯度
Wtmp = np.empty((T - w_len, w_len * 2))
for i in range(T - w_len):
    Wtmp[i] = X[i:i + w_len].ravel()
Wz = (Wtmp - Wtmp.mean(axis=1, keepdims=True)) / (Wtmp.std(axis=1, keepdims=True) + 1e-12)
purity_list = []
for i in range(600, T - w_len):
    t = i + w_len
    jmax = i - w_len
    mask = (Wz[i, None] @ Wz[:jmax].T).ravel() / (w_len * 2) >= rho
    cd = np.nonzero(mask)[0] + w_len
    if len(cd) >= 5:
        purity_list.append((parity[cd] == parity[t]).mean())
purity = np.mean(purity_list)
print(f"匹配集同相位纯度 {purity:.3f}")

# ---- 图1: 模式匹配示意 ----
fig, ax = plt.subplots(figsize=(10, 4.6))
pa = prices[:400, 0]
ax.plot(pa, color="#3182bd", lw=1.2, label="资产A价格")
t0 = 380
ax.axvspan(t0 - w_len, t0, color="#e6550d", alpha=0.4, label="当前窗口（最近5日）")
cur = X[t0 - w_len:t0].ravel()
curz = (cur - cur.mean()) / cur.std()
sims = []
for s in range(w_len, t0 - w_len):
    h = X[s - w_len:s].ravel()
    hz = (h - h.mean()) / h.std()
    sims.append((float(curz @ hz) / len(cur), s))
sims.sort(reverse=True)
for c, s in sims[:8]:
    ax.axvspan(s - w_len, s, color="#31a354", alpha=0.30)
ax.plot([], [], color="#31a354", lw=6, alpha=0.4, label="相关性最高的历史窗口")
ax.set_title(f"CORN 核心动作：拿最近 {w_len} 日模式去历史里找『相似市场』", fontsize=13)
ax.set_xlabel("交易日")
ax.set_ylabel("价格")
ax.legend(loc="best", fontsize=9.5)
fig.tight_layout()
fig.savefig(f"{OUT}/corn-matching-idea.png", dpi=130)
plt.close(fig)

# ---- 图2: 财富曲线对比（含对照市场）----
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(wealth_oracle, color="#bbb", lw=1.0, ls="-.", label=f"先知上限（已知相位） 终值 {wealth_oracle[-1]:.1f}")
ax.plot(wealth_corn, color="#e6550d", lw=1.7, label=f"CORN (ρ={rho}, w={w_len})  终值 {wealth_corn[-1]:.2f}")
ax.plot(wealth_bcrp, color="#31a354", lw=1.3, label=f"事后最优 CRP b*={b_star:.2f}  终值 {wealth_bcrp[-1]:.2f}")
ax.plot(wealth_5050, color="#3182bd", lw=1.1, label=f"50/50 再平衡  终值 {wealth_5050[-1]:.2f}")
ax.plot(wealth_corn_ctrl, color="#756bb1", lw=1.0, ls="--",
        label=f"CORN 在无模式对照市场  终值 {wealth_corn_ctrl[-1]:.2f}")
ax.set_yscale("log")
ax.set_title("有真实模式的市场：CORN 大幅甩开一切常数组合；无模式市场：CORN 无超额", fontsize=12.5)
ax.set_xlabel("交易日")
ax.set_ylabel("财富（初始=1，对数轴）")
ax.legend(fontsize=9.5)
fig.tight_layout()
fig.savefig(f"{OUT}/corn-wealth-curves.png", dpi=130)
plt.close(fig)

# ---- 图3: 权重轨迹（全程 + 局部放大）----
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(bs, color="#e6550d", lw=0.5)
axes[0].axhline(0.5, color="#999", lw=0.8, ls="--")
axes[0].set_ylabel("资产A权重 b_t")
axes[0].set_title("CORN 持仓轨迹全程：学会模式后在 0/1 之间逐日切换", fontsize=12.5)
z0, z1 = 800, 860
axes[1].step(range(z0, z1), bs[z0:z1], color="#e6550d", lw=1.4, where="mid", label="CORN 权重")
axes[1].step(range(z0, z1), (parity[z0:z1] == 0).astype(float), color="#3182bd",
             lw=1.0, ls="--", where="mid", label="真实相位（A强=1）")
axes[1].set_ylim(-0.1, 1.1)
axes[1].set_xlabel("交易日")
axes[1].set_ylabel("权重 / 相位")
axes[1].legend(fontsize=9.5)
axes[1].set_title(f"局部放大（第{z0}-{z1}日）：权重切换与真实相位对齐", fontsize=12.5)
fig.tight_layout()
fig.savefig(f"{OUT}/corn-weights-matches.png", dpi=130)
plt.close(fig)

# ---- 图4: ρ 阈值敏感性 ----
rhos = [-1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
finals_rho = []
for r in rhos:
    b_r, _ = corn_run(X, w=w_len, rho=r)
    fv = float(np.prod(X[:, 0] * b_r + X[:, 1] * (1 - b_r)))
    finals_rho.append(fv)
    print(f"rho={r:+.1f}  final={fv:.3f}")

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(rhos, finals_rho, "o-", color="#e6550d", lw=1.5)
ax.axhline(wealth_bcrp[-1], color="#31a354", lw=1.2, ls="--", label=f"事后最优 CRP 终值 {wealth_bcrp[-1]:.2f}")
ax.set_yscale("log")
ax.set_xlabel("相关性阈值 ρ")
ax.set_ylabel("终值财富（对数轴）")
ax.set_title("ρ 太松=混入噪声样本，ρ 太紧=样本饥荒：阈值是真超参数", fontsize=13)
ax.legend(fontsize=9.5)
fig.tight_layout()
fig.savefig(f"{OUT}/corn-rho-sensitivity.png", dpi=130)
plt.close(fig)

print("CORN images done")
