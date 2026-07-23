#!/usr/bin/env python3
"""深度状态空间交易实验：多时间尺度对角线性 RNN vs tanh 储备池 vs OLS 滞后回归"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, json

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/deep-state-space-trading"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

# ---------- 1. 合成任务：长程依赖 ----------
# x_t 是一条模拟"订单流冲击"序列，目标 y_t 同时依赖短期 (lag1,2) 和长程 (lag64) 结构
T = 12000
x = rng.standard_normal(T)
# 加一点自相关让它更像金融信号
for t in range(1, T):
    x[t] = 0.3 * x[t - 1] + np.sqrt(1 - 0.09) * x[t]

LONG_LAG = 64
y = np.zeros(T)
for t in range(LONG_LAG, T):
    y[t] = (0.5 * x[t - 1] - 0.3 * x[t - 2] + 0.8 * np.tanh(x[t - LONG_LAG])
            + 0.15 * rng.standard_normal())

WARM = 200  # warmup 丢弃
train_end = 8000
test_start, test_end = 8000, 12000

# ---------- 2. 模型 A：多时间尺度对角线性 SSM ----------
# h_t = a ⊙ h_{t-1} + b ⊙ x_t，a 按对数均匀铺满 [短记忆, 长记忆]
def ssm_features(x, n_states=64, min_tau=1.0, max_tau=256.0):
    taus = np.exp(np.linspace(np.log(min_tau), np.log(max_tau), n_states))
    a = np.exp(-1.0 / taus)                       # 衰减率
    b = np.sqrt(1 - a ** 2)                       # 能量归一
    H = np.zeros((len(x), n_states))
    h = np.zeros(n_states)
    for t in range(len(x)):
        h = a * h + b * x[t]
        H[t] = h
    return H, a, taus

H_ssm, a_vec, taus = ssm_features(x)

# ---------- 3. 模型 B：tanh 随机储备池（同状态数） ----------
def reservoir_features(x, n_states=64, rho=0.9, seed=7):
    r = np.random.default_rng(seed)
    W = r.standard_normal((n_states, n_states))
    eig = np.max(np.abs(np.linalg.eigvals(W)))
    W = W * (rho / eig)
    win = r.standard_normal(n_states) * 0.5
    H = np.zeros((len(x), n_states))
    h = np.zeros(n_states)
    for t in range(len(x)):
        h = np.tanh(W @ h + win * x[t])
        H[t] = h
    return H

H_res = reservoir_features(x)

# ---------- 4. Ridge 读出（统一口径） ----------
def ridge_fit_eval(H, y, lam=1e-3):
    Xtr = np.hstack([H[WARM:train_end], x[WARM:train_end, None]])
    ytr = y[WARM:train_end]
    Xte = np.hstack([H[test_start:test_end], x[test_start:test_end, None]])
    yte = y[test_start:test_end]
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ ytr)
    pred = Xte @ w
    ss_res = np.sum((yte - pred) ** 2)
    ss_tot = np.sum((yte - yte.mean()) ** 2)
    return 1 - ss_res / ss_tot, pred, yte

r2_ssm, pred_ssm, yte = ridge_fit_eval(H_ssm, y)
r2_res, pred_res, _ = ridge_fit_eval(H_res, y)

# ---------- 5. OLS 滞后基线（K 个滞后） ----------
def ols_lags(x, y, K):
    rows = []
    for t in range(max(K, WARM), len(x)):
        rows.append(x[t - K:t][::-1])
    X = np.array(rows)
    yy = y[max(K, WARM):]
    off = max(K, WARM)
    tr = slice(0, train_end - off)
    te = slice(test_start - off, test_end - off)
    A = X[tr].T @ X[tr] + 1e-6 * np.eye(K)
    w = np.linalg.solve(A, X[tr].T @ yy[tr])
    pred = X[te] @ w
    yte2 = yy[te]
    return 1 - np.sum((yte2 - pred) ** 2) / np.sum((yte2 - yte2.mean()) ** 2)

r2_ols8 = ols_lags(x, y, 8)
r2_ols32 = ols_lags(x, y, 32)
r2_ols128 = ols_lags(x, y, 128)

print(f"SSM R2={r2_ssm:.4f}  Reservoir R2={r2_res:.4f}  OLS8={r2_ols8:.4f} OLS32={r2_ols32:.4f} OLS128={r2_ols128:.4f}")

# ---------- 6. 状态数扫描 ----------
state_list = [8, 16, 32, 64, 128]
r2_by_states = []
for n in state_list:
    Hn, _, _ = ssm_features(x, n_states=n)
    r2n, _, _ = ridge_fit_eval(Hn, y)
    r2_by_states.append(r2n)
print("states sweep:", dict(zip(state_list, [round(v, 4) for v in r2_by_states])))

# ---------- 图 1：记忆核 ----------
fig, ax = plt.subplots(figsize=(9, 4.5))
ks = np.arange(0, 300)
for i in [0, 16, 32, 48, 63]:
    kernel = (np.sqrt(1 - a_vec[i] ** 2)) * a_vec[i] ** ks
    ax.plot(ks, kernel, label=f"τ≈{taus[i]:.0f} 步 (a={a_vec[i]:.3f})")
ax.set_xlabel("滞后步数 k")
ax.set_ylabel("冲击响应 b·a^k")
ax.set_title("对角线性 SSM 的记忆核：不同衰减率 = 不同时间尺度的指数记忆")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/memory_kernels.png", dpi=130)
plt.close(fig)

# ---------- 图 2：合成任务示意 ----------
fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
seg = slice(9000, 9400)
axes[0].plot(np.arange(seg.start, seg.stop), x[seg], lw=0.9, color="#4472c4")
axes[0].set_ylabel("输入 x")
axes[0].set_title("合成任务：目标同时依赖 lag-1/2（短程）与 lag-64（长程）")
axes[0].grid(alpha=0.3)
axes[1].plot(np.arange(seg.start, seg.stop), y[seg], lw=0.9, color="#c00000")
axes[1].set_ylabel("目标 y")
axes[1].set_xlabel("时间步")
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/synthetic_task.png", dpi=130)
plt.close(fig)

# ---------- 图 3：模型对比条形图 ----------
fig, ax = plt.subplots(figsize=(9, 4.8))
names = ["OLS 8滞后", "OLS 32滞后", "OLS 128滞后", "tanh 储备池-64", "线性 SSM-64"]
vals = [r2_ols8, r2_ols32, r2_ols128, r2_res, r2_ssm]
colors = ["#a6a6a6", "#a6a6a6", "#7f7f7f", "#ed7d31", "#4472c4"]
bars = ax.bar(names, vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=11)
ax.set_ylabel("测试集 R²")
ax.set_title("长程依赖任务：64 状态线性 SSM vs 储备池 vs OLS 滞后回归")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/model_comparison.png", dpi=130)
plt.close(fig)

# ---------- 图 4：状态数扫描 + 预测对比 ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(state_list, r2_by_states, "o-", color="#4472c4")
for s, v in zip(state_list, r2_by_states):
    axes[0].annotate(f"{v:.3f}", (s, v), textcoords="offset points", xytext=(0, 8), ha="center")
axes[0].set_xscale("log", base=2)
axes[0].set_xlabel("状态维度 N")
axes[0].set_ylabel("测试集 R²")
axes[0].set_title("状态数扫描：多少个指数记忆才够覆盖 lag-64")
axes[0].grid(alpha=0.3)
seg2 = slice(200, 320)
axes[1].plot(yte[seg2], label="真实值", lw=1.2, color="#404040")
axes[1].plot(pred_ssm[seg2], label=f"线性 SSM (R²={r2_ssm:.3f})", lw=1.1, color="#4472c4")
axes[1].plot(pred_res[seg2], label=f"tanh 储备池 (R²={r2_res:.3f})", lw=1.0, color="#ed7d31", alpha=0.8)
axes[1].set_title("测试集预测 vs 真实（局部放大）")
axes[1].set_xlabel("测试集时间步")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/states_sweep_pred.png", dpi=130)
plt.close(fig)

# ---------- 7. 任务 B：平滑多尺度记忆 + 小样本 ----------
# y_t 依赖两个不同时间尺度的 EWMA（τ=8 与 τ=96），且只有 1500 个训练样本
ew_fast = np.zeros(T)
ew_slow = np.zeros(T)
af, aslow = np.exp(-1 / 8), np.exp(-1 / 96)
for t in range(1, T):
    ew_fast[t] = af * ew_fast[t - 1] + (1 - af) * x[t]
    ew_slow[t] = aslow * ew_slow[t - 1] + (1 - aslow) * x[t]
yB = 0.6 * ew_fast - 0.9 * ew_slow + 0.4 * np.tanh(2 * ew_slow) + 0.1 * rng.standard_normal(T)

train_end_B = 2000   # 小样本：仅 ~1800 有效训练点

def ridge_fit_eval_B(H, yv, lam=1e-3):
    Xtr = np.hstack([H[WARM:train_end_B], x[WARM:train_end_B, None]])
    ytr = yv[WARM:train_end_B]
    Xte = np.hstack([H[test_start:test_end], x[test_start:test_end, None]])
    yte = yv[test_start:test_end]
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ ytr)
    pred = Xte @ w
    return 1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2)

def ols_lags_B(x, yv, K, lam=1e-6):
    rows = []
    for t in range(max(K, WARM), len(x)):
        rows.append(x[t - K:t][::-1])
    X = np.array(rows)
    yy = yv[max(K, WARM):]
    off = max(K, WARM)
    tr = slice(0, train_end_B - off)
    te = slice(test_start - off, test_end - off)
    A = X[tr].T @ X[tr] + lam * np.eye(K)
    w = np.linalg.solve(A, X[tr].T @ yy[tr])
    pred = X[te] @ w
    yte2 = yy[te]
    return 1 - np.sum((yte2 - pred) ** 2) / np.sum((yte2 - yte2.mean()) ** 2)

H16, _, _ = ssm_features(x, n_states=16)
r2B_ssm16 = ridge_fit_eval_B(H16, yB)
r2B_ssm64 = ridge_fit_eval_B(H_ssm, yB)
r2B_res = ridge_fit_eval_B(H_res, yB)
r2B_ols32 = ols_lags_B(x, yB, 32)
r2B_ols128 = ols_lags_B(x, yB, 128)
r2B_ols256 = ols_lags_B(x, yB, 256)
print(f"TaskB: SSM16={r2B_ssm16:.4f} SSM64={r2B_ssm64:.4f} Res={r2B_res:.4f} OLS32={r2B_ols32:.4f} OLS128={r2B_ols128:.4f} OLS256={r2B_ols256:.4f}")

# ---------- 图 5：任务 B 对比 ----------
fig, ax = plt.subplots(figsize=(9.5, 4.8))
namesB = ["OLS 32滞后", "OLS 128滞后", "OLS 256滞后", "tanh 储备池-64", "线性 SSM-16", "线性 SSM-64"]
valsB = [r2B_ols32, r2B_ols128, r2B_ols256, r2B_res, r2B_ssm16, r2B_ssm64]
colorsB = ["#a6a6a6", "#a6a6a6", "#7f7f7f", "#ed7d31", "#70ad47", "#4472c4"]
bars = ax.bar(namesB, valsB, color=colorsB)
for b, v in zip(bars, valsB):
    ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.01, f"{v:.3f}", ha="center", fontsize=10)
ax.set_ylabel("测试集 R²")
ax.set_title("任务 B（平滑多尺度记忆 + 仅 1800 训练样本）：SSM 16 状态即接近饱和")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/taskB_comparison.png", dpi=130)
plt.close(fig)

json.dump({"r2_ssm": r2_ssm, "r2_res": r2_res, "r2_ols8": r2_ols8, "r2_ols32": r2_ols32,
           "r2_ols128": r2_ols128, "states": dict(zip(map(str, state_list), r2_by_states)),
           "taskB": {"ssm16": r2B_ssm16, "ssm64": r2B_ssm64, "res": r2B_res,
                      "ols32": r2B_ols32, "ols128": r2B_ols128, "ols256": r2B_ols256}},
          open(f"{OUT}/results.json", "w"), indent=2)
print("done")
