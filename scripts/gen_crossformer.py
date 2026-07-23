import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)
OUT = "public/images/crossformer-financial"
os.makedirs(OUT, exist_ok=True)

# ---- synthetic multivariate financial series: D assets driven by shared factors ----
T, D = 600, 6
t = np.arange(T)
f1 = np.sin(2*np.pi*t/50)            # slow cycle
f2 = np.sin(2*np.pi*t/13 + 0.7)      # fast cycle
loadings = rng.uniform(-1, 1, size=(D, 2))
X = loadings @ np.vstack([f1, f2]) + 0.35*rng.standard_normal((D, T))
X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)

# 1. multivariate series
fig, ax = plt.subplots(figsize=(9, 4.2))
for d in range(D):
    ax.plot(t, X[d] + d*3.0, lw=1.0)
    ax.text(-8, d*3.0, f"资产{d+1}", va="center", ha="right", fontsize=9)
ax.set_title("6 个资产的标准化价量序列（共享两个隐因子驱动）")
ax.set_xlabel("时间步"); ax.set_yticks([]); ax.set_xlim(-40, T)
plt.tight_layout(); plt.savefig(f"{OUT}/multivariate_series.png", dpi=120); plt.close()

# 2. dimension-segment-wise attention: patch the time axis into segments
seg = 20
n_seg = T // seg
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(t, X[0], color="#3b6ea5", lw=1.0)
for s in range(n_seg):
    ax.axvspan(s*seg, (s+1)*seg, color="#ffb347" if s % 2 else "#ffe0a3", alpha=0.25)
ax.set_title(f"时间轴分段（每段 {seg} 步）：段内建模局部形态，段间做跨段注意力")
ax.set_xlabel("时间步"); ax.set_ylabel("资产1 标准化值"); ax.set_xlim(0, T)
plt.tight_layout(); plt.savefig(f"{OUT}/time_segmentation.png", dpi=120); plt.close()

# 3. cross-dimension attention heatmap (learned variable interaction ~ correlation)
C = np.corrcoef(X)
fig, ax = plt.subplots(figsize=(5.4, 4.6))
im = ax.imshow(np.abs(C), cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(D)); ax.set_yticks(range(D))
ax.set_xticklabels([f"资产{i+1}" for i in range(D)], rotation=45, ha="right", fontsize=8)
ax.set_yticklabels([f"资产{i+1}" for i in range(D)], fontsize=8)
for i in range(D):
    for j in range(D):
        ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                color="white" if abs(C[i,j])<0.6 else "black", fontsize=7)
ax.set_title("跨维度注意力权重（≈变量间关联强度）")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.savefig(f"{OUT}/cross_dim_attention.png", dpi=120); plt.close()

# ---- tiny two-stage attention forecaster vs baselines (numpy, honest) ----
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=axis, keepdims=True)

L, H = 40, 1
Xtr = X[:, :450]; Xte = X[:, 450:]
def make(series):
    xs, ys = [], []
    for i in range(L, series.shape[1]-H):
        xs.append(series[:, i-L:i]); ys.append(series[:, i+H-1])
    return np.array(xs), np.array(ys)
xtr, ytr = make(Xtr); xte, yte = make(Xte)

# cross-dim attention feature: weight each asset's last value by learned-ish corr
Ctr = np.corrcoef(Xtr)
def predict_cross(xb):
    last = xb[:, :, -1]                    # (N, D)
    w = softmax(np.abs(Ctr) * 3.0, axis=1) # attention over variables
    mixed = last @ w.T                     # blend across correlated assets
    trend = xb[:, :, -1] - xb[:, :, -3]
    return 0.6*last + 0.25*mixed + 0.15*(last+trend)
pred_cross = predict_cross(xte)
pred_last = xte[:, :, -1]
def r2(y, p): 
    ss = ((y-p)**2).sum(); tot = ((y-y.mean())**2).sum(); return 1-ss/tot
r2_cross, r2_last = r2(yte, pred_cross), r2(yte, pred_last)

# 4. prediction vs truth for one asset
fig, ax = plt.subplots(figsize=(9, 4.0))
k = 0
ax.plot(yte[:, k], label="真实", color="#222", lw=1.4)
ax.plot(pred_cross[:, k], label=f"Crossformer 风格 (R²={r2_cross:.3f})", color="#e0563b", lw=1.2)
ax.plot(pred_last[:, k], label=f"朴素持平 (R²={r2_last:.3f})", color="#3b6ea5", lw=1.0, ls="--")
ax.set_title("资产1 单步预测：跨变量注意力 vs 朴素基线（测试集）")
ax.set_xlabel("测试样本序号"); ax.set_ylabel("标准化值"); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/prediction_vs_truth.png", dpi=120); plt.close()

print("crossformer images done", r2_cross, r2_last)
