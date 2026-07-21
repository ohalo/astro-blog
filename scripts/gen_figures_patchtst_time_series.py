# -*- coding: utf-8 -*-
"""PatchTST 时序分块预测 —— 生成配图 + 打印统计(自洽合成, 纯 numpy)
核心演示: 把长序列切成 patch(分块) + 通道独立, 相比逐点建模的收益.
用一个简化的 "patch + 线性注意力聚合" 前向来体现思想(非完整 Transformer 训练).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Heiti SC", "PingFang SC", "STHeiti", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 110

OUT = "public/images/patchtst-time-series"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260722)

# ---------------- 合成时序: 趋势 + 双周期 + 噪声 ----------------
T = 2600
t = np.arange(T)
trend = 0.02 * t
season1 = 6.0 * np.sin(2*np.pi*t/50)          # 中周期 50
season2 = 3.0 * np.sin(2*np.pi*t/12 + 0.7)    # 短周期 12
noise = rng.normal(0, 1.5, T)
series = 50 + trend + season1 + season2 + noise
series = series.astype(float)

L = 336        # 回看窗口(look-back)
H = 48         # 预测步长(horizon)
P = 16         # patch 长度
S = 8          # patch 步长(stride) -> 有重叠
n_patch = (L - P) // S + 1
print(f"序列长度 {T} | 回看 L={L} | 预测 H={H} | patch长 P={P} 步长 S={S} | patch数 {n_patch}")

# ---------------- 构造样本 ----------------
def make_windows(x, L, H):
    X, Y = [], []
    for i in range(L, len(x) - H + 1):
        X.append(x[i-L:i]); Y.append(x[i:i+H])
    return np.array(X), np.array(Y)

X, Y = make_windows(series, L, H)
n = len(X)
ntr = int(n * 0.7)
Xtr, Ytr = X[:ntr], Y[:ntr]
Xte, Yte = X[ntr:], Y[ntr:]
# 实例归一化(RevIN 思想): 每个窗口减自己的均值
mu_tr = Xtr.mean(1, keepdims=True); mu_te = Xte.mean(1, keepdims=True)
Xtr_n = Xtr - mu_tr; Ytr_n = Ytr - mu_tr
Xte_n = Xte - mu_te; Yte_n = Yte - mu_te

# ---------------- 方法A: 逐点线性(把整段 L 展平 -> H, 岭回归) ----------------
lam = 5.0
def ridge_fit(A, B, lam):
    d = A.shape[1]
    return np.linalg.solve(A.T @ A + lam*np.eye(d), A.T @ B)

Wp = ridge_fit(Xtr_n, Ytr_n, lam)
pred_point = Xte_n @ Wp + mu_te

# ---------------- 方法B: PatchTST 思想 = 分块嵌入 + 岭回归 ----------------
# 把 L 切成重叠 patch, 每个 patch 做同一个线性嵌入(共享权重) -> 再拼起来预测
def to_patches(Xn):
    m = Xn.shape[0]
    out = np.zeros((m, n_patch, P))
    for k in range(n_patch):
        st = k*S
        out[:, k, :] = Xn[:, st:st+P]
    return out

Ptr = to_patches(Xtr_n)   # (m, n_patch, P)
Pte = to_patches(Xte_n)

# patch 内共享线性嵌入到 D 维(用 PCA-like 固定基简化), 这里用一个学到的嵌入: 先堆叠所有 patch 学 embed
D = 8
flatP = Ptr.reshape(-1, P)
# 用岭回归学 patch->下一段的局部预测? 简化: 直接学 patch 嵌入基
# 嵌入: E (P x D), 用训练 patch 的主成分作为基(无监督, 稳定)
Xc = flatP - flatP.mean(0)
U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
E = Vt[:D].T                       # (P, D) 前 D 个主成分
emb_tr = (Ptr @ E).reshape(ntr, n_patch*D)          # 通道独立地嵌入每个 patch 再拼接
emb_te = (Pte @ E).reshape(len(Xte_n), n_patch*D)
Wb = ridge_fit(emb_tr, Ytr_n, lam)
pred_patch = emb_te @ Wb + mu_te

# ---------------- 评估 ----------------
def mse(a, b): return np.mean((a-b)**2)
def mae(a, b): return np.mean(np.abs(a-b))
naive = np.repeat(Xte[:, -1:], H, axis=1)           # 朴素: 最后值持平
print(f"[MSE] 朴素持平={mse(naive,Yte):.3f} | 逐点线性={mse(pred_point,Yte):.3f} | PatchTST分块={mse(pred_patch,Yte):.3f}")
print(f"[MAE] 朴素持平={mae(naive,Yte):.3f} | 逐点线性={mae(pred_point,Yte):.3f} | PatchTST分块={mae(pred_patch,Yte):.3f}")
print(f"分块相对逐点 MSE 降幅: {(1-mse(pred_patch,Yte)/mse(pred_point,Yte))*100:.1f}%")

# ================= 图1 cover: 分块示意 =================
fig, ax = plt.subplots(figsize=(10, 4.8))
seg = series[500:500+L]
ax.plot(np.arange(L), seg, color="#2c3e50", lw=1.3, label="回看窗口 (L=336)")
cmap = plt.cm.viridis(np.linspace(0.15, 0.9, n_patch))
for k in range(0, n_patch, 3):     # 画部分 patch 框避免太挤
    st = k*S
    ax.axvspan(st, st+P, color=cmap[k], alpha=0.18)
ax.set_title(f"PatchTST 核心：把长序列切成 {n_patch} 个重叠 patch（每块长 {P}、步长 {S}），再送进 Transformer", fontsize=11.5)
ax.set_xlabel("回看窗口内时间步"); ax.set_ylabel("价格")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/cover.png", bbox_inches="tight")
plt.close()

# ================= 图2: 预测对比 =================
fig, ax = plt.subplots(figsize=(10.5, 4.8))
idx = 40
hx = np.arange(H)
ax.plot(hx, Yte[idx], color="#2c3e50", lw=2.2, label="真实未来")
ax.plot(hx, pred_patch[idx], color="#e74c3c", lw=1.8, ls="-", label="PatchTST 分块")
ax.plot(hx, pred_point[idx], color="#3498db", lw=1.6, ls="--", label="逐点线性")
ax.plot(hx, naive[idx], color="#95a5a6", lw=1.4, ls=":", label="朴素持平")
ax.set_title("单个测试样本的 48 步预测对比：分块预测更贴合周期结构")
ax.set_xlabel("预测步 (horizon)"); ax.set_ylabel("价格")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/forecast_compare.png", bbox_inches="tight")
plt.close()

# ================= 图3: 误差随 horizon 增长 =================
err_patch = np.sqrt(((pred_patch - Yte)**2).mean(0))
err_point = np.sqrt(((pred_point - Yte)**2).mean(0))
err_naive = np.sqrt(((naive - Yte)**2).mean(0))
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.plot(err_naive, color="#95a5a6", lw=1.8, ls=":", label="朴素持平")
ax.plot(err_point, color="#3498db", lw=1.8, ls="--", label="逐点线性")
ax.plot(err_patch, color="#e74c3c", lw=2.2, label="PatchTST 分块")
ax.set_title("预测 RMSE 随步长增长：分块在长步长上衰减更慢")
ax.set_xlabel("预测步 (1→48)"); ax.set_ylabel("RMSE")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/rmse_by_horizon.png", bbox_inches="tight")
plt.close()

print("图已生成:", os.listdir(OUT))
