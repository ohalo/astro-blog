#!/usr/bin/env python3
"""DeepSets 置换不变集合建模实验：预测『资产集合（组合）』的聚合属性
任务：给定一个由 K 只资产（每只带特征向量）组成的无序集合，预测组合层面的目标
（例如组合尾部风险的非线性聚合）。对比：
  A) DeepSets: rho(sum(phi(x_i)))  —— 置换不变
  B) MLP-concat: 把集合按输入顺序摊平 —— 对顺序敏感
  C) 均值特征 + 线性回归 —— 朴素聚合基线
验证点：
  1) DeepSets 对置换后的同一集合输出严格不变；MLP-concat 输出漂移
  2) 泛化到不同集合大小 K（训练 K=8，测试 K=5/8/12）
  3) sum-pooling vs mean-pooling 在变 K 下的差异
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, json

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/permutation-invariant-set"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(123)

D = 4          # 每只资产特征维度: [波动率, 动量, 流动性, 尾部beta]
KTRAIN = 8

def gen_set(K, rng):
    """生成一个 K 只资产的集合与组合目标。
    目标 = 平滑非线性聚合：log(1 + sum(exp(w·x_i))) 型 soft-max 风险聚合 + 交互项"""
    X = rng.standard_normal((K, D)) * np.array([0.8, 1.0, 0.6, 0.9])
    vol, mom, liq, tbeta = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    # 组合尾部风险：软最大化的尾部beta（大者主导）+ 平均波动 + 动量分散度
    risk = np.log(np.sum(np.exp(1.2 * tbeta))) + 0.5 * np.mean(vol ** 2) \
           + 0.3 * np.std(mom) - 0.2 * np.mean(liq)
    return X, risk

def make_dataset(n, K, rng):
    Xs = np.zeros((n, K, D))
    ys = np.zeros(n)
    for i in range(n):
        Xs[i], ys[i] = gen_set(K, rng)
    return Xs, ys

Xtr, ytr = make_dataset(6000, KTRAIN, rng)
Xte, yte = make_dataset(2000, KTRAIN, rng)
Xte5, yte5 = make_dataset(2000, 5, rng)
Xte12, yte12 = make_dataset(2000, 12, rng)

# ---------- DeepSets 模型（纯 numpy，手写反向传播） ----------
class DeepSets:
    def __init__(self, d_in, d_phi=32, d_rho=32, pool="sum", seed=0):
        r = np.random.default_rng(seed)
        s1 = np.sqrt(2 / d_in); s2 = np.sqrt(2 / d_phi); s3 = np.sqrt(2 / d_rho)
        self.W1 = r.standard_normal((d_in, d_phi)) * s1; self.b1 = np.zeros(d_phi)
        self.W2 = r.standard_normal((d_phi, d_phi)) * s2; self.b2 = np.zeros(d_phi)
        self.W3 = r.standard_normal((d_phi, d_rho)) * s2; self.b3 = np.zeros(d_rho)
        self.W4 = r.standard_normal((d_rho, 1)) * s3;    self.b4 = np.zeros(1)
        self.pool = pool
        self.params = ["W1","b1","W2","b2","W3","b3","W4","b4"]

    def forward(self, X):
        # X: (B, K, D)
        self.X = X
        self.Z1 = X @ self.W1 + self.b1          # (B,K,H)
        self.A1 = np.maximum(0, self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = np.maximum(0, self.Z2)          # phi 输出 (B,K,H)
        if self.pool == "sum":
            self.S = self.A2.sum(axis=1)          # (B,H)
        else:
            self.S = self.A2.mean(axis=1)
        self.Z3 = self.S @ self.W3 + self.b3
        self.A3 = np.maximum(0, self.Z3)
        self.out = (self.A3 @ self.W4 + self.b4).ravel()
        return self.out

    def backward(self, dout, lr):
        B, K, _ = self.X.shape
        dout = dout[:, None]                      # (B,1)
        gW4 = self.A3.T @ dout / B; gb4 = dout.mean(0)
        dA3 = dout @ self.W4.T
        dZ3 = dA3 * (self.Z3 > 0)
        gW3 = self.S.T @ dZ3 / B; gb3 = dZ3.mean(0)
        dS = dZ3 @ self.W3.T                      # (B,H)
        if self.pool == "sum":
            dA2 = np.repeat(dS[:, None, :], K, axis=1)
        else:
            dA2 = np.repeat(dS[:, None, :], K, axis=1) / K
        dZ2 = dA2 * (self.Z2 > 0)
        gW2 = np.einsum("bkh,bkj->hj", self.A1, dZ2) / B
        gb2 = dZ2.mean(axis=(0, 1)) * K if self.pool=="sum" else dZ2.mean(axis=(0,1))*K
        gb2 = dZ2.sum(axis=(0,1)) / B
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0)
        gW1 = np.einsum("bkd,bkh->dh", self.X, dZ1) / B
        gb1 = dZ1.sum(axis=(0,1)) / B
        for p, g in zip(self.params, [gW1,gb1,gW2,gb2,gW3,gb3,gW4,gb4]):
            setattr(self, p, getattr(self, p) - lr * g)

def train(model, Xtr, ytr, Xval, yval, epochs=80, bs=128, lr=0.02):
    n = len(Xtr); hist = []
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, bs):
            j = idx[s:s+bs]
            pred = model.forward(Xtr[j])
            err = np.clip(2 * (pred - ytr[j]), -10, 10)   # 梯度裁剪防发散
            model.backward(err, lr)
        pv = model.forward(Xval)
        mse = np.mean((pv - yval) ** 2)
        hist.append(mse)
        lr *= 0.97
    return hist

def r2(model, X, y):
    p = model.forward(X)
    return 1 - np.sum((p - y)**2) / np.sum((y - y.mean())**2)

# ---------- MLP-concat 基线（摊平，顺序敏感） ----------
class MLPConcat:
    def __init__(self, d_in, hidden=128, seed=1):
        r = np.random.default_rng(seed)
        self.W1 = r.standard_normal((d_in, hidden)) * np.sqrt(2/d_in); self.b1 = np.zeros(hidden)
        self.W2 = r.standard_normal((hidden, hidden)) * np.sqrt(2/hidden); self.b2 = np.zeros(hidden)
        self.W3 = r.standard_normal((hidden, 1)) * np.sqrt(2/hidden); self.b3 = np.zeros(1)
    def forward(self, Xflat):
        self.X = Xflat
        self.Z1 = Xflat @ self.W1 + self.b1; self.A1 = np.maximum(0, self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2; self.A2 = np.maximum(0, self.Z2)
        self.out = (self.A2 @ self.W3 + self.b3).ravel()
        return self.out
    def backward(self, dout, lr):
        B = len(self.X); dout = dout[:, None]
        gW3 = self.A2.T @ dout / B; gb3 = dout.mean(0)
        dA2 = dout @ self.W3.T; dZ2 = dA2 * (self.Z2 > 0)
        gW2 = self.A1.T @ dZ2 / B; gb2 = dZ2.mean(0)
        dA1 = dZ2 @ self.W2.T; dZ1 = dA1 * (self.Z1 > 0)
        gW1 = self.X.T @ dZ1 / B; gb1 = dZ1.mean(0)
        self.W1 -= lr*gW1; self.b1 -= lr*gb1
        self.W2 -= lr*gW2; self.b2 -= lr*gb2
        self.W3 -= lr*gW3; self.b3 -= lr*gb3

def train_mlp(model, Xtr, ytr, epochs=60, bs=128, lr=0.02):
    n = len(Xtr)
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, bs):
            j = idx[s:s+bs]
            pred = model.forward(Xtr[j])
            model.backward(2*(pred-ytr[j]), lr)
        lr *= 0.97

# ---------- 训练 ----------
ds_sum = DeepSets(D, pool="sum", seed=0)
hist_sum = train(ds_sum, Xtr, ytr, Xte, yte, lr=0.004)   # sum-pool 梯度随 K 放大，降低 lr
ds_mean = DeepSets(D, pool="mean", seed=0)
hist_mean = train(ds_mean, Xtr, ytr, Xte, yte, lr=0.02)

Xtr_flat = Xtr.reshape(len(Xtr), -1)
Xte_flat = Xte.reshape(len(Xte), -1)
mlp = MLPConcat(KTRAIN * D)
train_mlp(mlp, Xtr_flat, ytr)

# 均值特征 + 线性
def meanfeat_ridge(Xtr, ytr, Xte, yte):
    Ftr = np.hstack([Xtr.mean(1), Xtr.std(1), Xtr.max(1)])
    Fte = np.hstack([Xte.mean(1), Xte.std(1), Xte.max(1)])
    lam = 1e-3
    w = np.linalg.solve(Ftr.T@Ftr + lam*np.eye(Ftr.shape[1]), Ftr.T@(ytr - ytr.mean()))
    pred = Fte @ w + ytr.mean()
    return 1 - np.sum((pred-yte)**2)/np.sum((yte-yte.mean())**2)

r2_meanfeat = meanfeat_ridge(Xtr, ytr, Xte, yte)

r2_ds_sum = r2(ds_sum, Xte, yte)
r2_ds_mean = r2(ds_mean, Xte, yte)
p_mlp = mlp.forward(Xte_flat)
r2_mlp = 1 - np.sum((p_mlp-yte)**2)/np.sum((yte-yte.mean())**2)
print(f"K=8: DeepSets-sum={r2_ds_sum:.4f} DeepSets-mean={r2_ds_mean:.4f} MLP-concat={r2_mlp:.4f} 均值特征Ridge={r2_meanfeat:.4f}")

# ---------- 置换不变性测试 ----------
n_perm = 50
sample = Xte[:200]
base_ds = ds_sum.forward(sample)
base_mlp = mlp.forward(sample.reshape(len(sample), -1))
ds_dev, mlp_dev = [], []
for _ in range(n_perm):
    perm = rng.permutation(KTRAIN)
    sp = sample[:, perm, :]
    ds_dev.append(np.abs(ds_sum.forward(sp) - base_ds).max())
    mlp_dev.append(np.abs(mlp.forward(sp.reshape(len(sp), -1)) - base_mlp).mean())
ds_dev_max = float(np.max(ds_dev))
mlp_dev_mean = float(np.mean(mlp_dev))
mlp_dev_rel = mlp_dev_mean / float(np.std(yte))
print(f"置换偏移: DeepSets max|Δ|={ds_dev_max:.2e}  MLP mean|Δ|={mlp_dev_mean:.4f} (≈{mlp_dev_rel:.1%} 的目标std)")

# ---------- 变 K 泛化 ----------
r2_sum_5, r2_sum_12 = r2(ds_sum, Xte5, yte5), r2(ds_sum, Xte12, yte12)
r2_mean_5, r2_mean_12 = r2(ds_mean, Xte5, yte5), r2(ds_mean, Xte12, yte12)
print(f"变K: sum-pool K5={r2_sum_5:.4f} K12={r2_sum_12:.4f} | mean-pool K5={r2_mean_5:.4f} K12={r2_mean_12:.4f}")

# ---------- 图 1：任务示意（集合 → 组合风险） ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
Xdemo, _ = gen_set(8, np.random.default_rng(9))
im = axes[0].imshow(Xdemo, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
axes[0].set_xticks(range(D)); axes[0].set_xticklabels(["波动率","动量","流动性","尾部β"])
axes[0].set_yticks(range(8)); axes[0].set_yticklabels([f"资产{i+1}" for i in range(8)])
axes[0].set_title("输入：8 只资产的特征矩阵（行序无意义）")
plt.colorbar(im, ax=axes[0], shrink=0.85)
axes[1].hist(ytr, bins=50, color="#4472c4", alpha=0.85)
axes[1].set_xlabel("组合尾部风险目标 y")
axes[1].set_ylabel("频数")
axes[1].set_title("目标：软最大化尾部β + 波动/分散度聚合")
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/task_setup.png", dpi=130)
plt.close(fig)

# ---------- 图 2：模型对比 ----------
fig, ax = plt.subplots(figsize=(9, 4.8))
names = ["均值/std/max特征\n+Ridge", "MLP-concat\n(摊平顺序敏感)", "DeepSets\nmean-pool", "DeepSets\nsum-pool"]
vals = [r2_meanfeat, r2_mlp, r2_ds_mean, r2_ds_sum]
colors = ["#a6a6a6", "#ed7d31", "#70ad47", "#4472c4"]
bars = ax.bar(names, vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}", ha="center", fontsize=11)
ax.set_ylabel("测试集 R² (K=8)")
ax.set_title("组合风险聚合任务：置换不变结构 vs 摊平 MLP vs 手工聚合特征")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/model_comparison.png", dpi=130)
plt.close(fig)

# ---------- 图 3：置换不变性 ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].bar(["DeepSets-sum", "MLP-concat"], [ds_dev_max, mlp_dev_mean],
            color=["#4472c4", "#ed7d31"])
axes[0].set_yscale("log")
axes[0].set_ylabel("同一集合置换后输出偏移 |Δ|")
axes[0].set_title(f"置换稳定性：DeepSets={ds_dev_max:.1e}（数值零）\nMLP-concat={mlp_dev_mean:.3f}（≈{mlp_dev_rel:.0%} 目标std）")
axes[0].grid(alpha=0.3, axis="y")
# 变K泛化
ks = [5, 8, 12]
axes[1].plot(ks, [r2_sum_5, r2_ds_sum, r2_sum_12], "o-", color="#4472c4", label="sum-pooling")
axes[1].plot(ks, [r2_mean_5, r2_ds_mean, r2_mean_12], "s-", color="#70ad47", label="mean-pooling")
for k, v in zip(ks, [r2_sum_5, r2_ds_sum, r2_sum_12]):
    axes[1].annotate(f"{v:.3f}", (k, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color="#4472c4")
for k, v in zip(ks, [r2_mean_5, r2_ds_mean, r2_mean_12]):
    axes[1].annotate(f"{v:.3f}", (k, v), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, color="#70ad47")
axes[1].axvline(8, color="gray", ls="--", alpha=0.5)
axes[1].text(8.05, axes[1].get_ylim()[0]+0.05, "训练 K=8", fontsize=9, color="gray")
axes[1].set_xlabel("测试集合大小 K")
axes[1].set_ylabel("测试集 R²")
axes[1].set_title("变 K 泛化：训练只见过 K=8，测试 K=5/12")
axes[1].legend()
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/invariance_and_generalization.png", dpi=130)
plt.close(fig)

# ---------- 图 4：训练曲线 ----------
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(hist_sum, label="DeepSets sum-pool", color="#4472c4")
ax.plot(hist_mean, label="DeepSets mean-pool", color="#70ad47")
ax.set_xlabel("epoch"); ax.set_ylabel("验证集 MSE"); ax.set_yscale("log")
ax.set_title("训练曲线（K=8 验证集）")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/training_curves.png", dpi=130)
plt.close(fig)

json.dump({"r2_ds_sum": r2_ds_sum, "r2_ds_mean": r2_ds_mean, "r2_mlp": r2_mlp,
           "r2_meanfeat": r2_meanfeat, "ds_dev_max": ds_dev_max, "mlp_dev_mean": mlp_dev_mean,
           "mlp_dev_rel": mlp_dev_rel,
           "varK": {"sum": [r2_sum_5, r2_ds_sum, r2_sum_12], "mean": [r2_mean_5, r2_ds_mean, r2_mean_12]}},
          open(f"{OUT}/results.json", "w"), indent=2)
print("done")
