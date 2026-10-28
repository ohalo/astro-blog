#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比学习自监督预训练：用 SimCLR 把时序片段映射到语义空间
真实计算 5 张图 + _simclr_results.json（torch 实现：增广 + 1D-CNN 编码器 + NT-Xent + 线性探测）

E1 增广视图：同一样本的两个视角的时域形态
E2 NT-Xent 对比损失曲线（train）
E3 测试集表征的 PCA 二维投影（按真类别着色，看语义聚类）
E4 线性探测 / KNN 准确率：SimCLR 表征 vs 原始信号 vs 随机基线
E5 线性探测准确率随训练轮次演化（收敛曲线）

依赖：torch（venv：~/.workbuddy/binaries/python/envs/default/bin/python）
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = "mps" if (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()) else "cpu"
print("device =", device)

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

OUT = "/Users/halo123456/workspace/astro-blog/public/images/contrastive-self-supervised-pretraining"
os.makedirs(OUT, exist_ok=True)
RED, GREEN, BLUE, GRAY, ORANGE, BLACK, PURPLE = "#d63031", "#00b894", "#0984e3", "#636e72", "#e17055", "#2d3436", "#6c5ce7"
SEED = 20261028
np.random.seed(SEED); torch.manual_seed(SEED)
R = {}

# ---------------- 数据：4 类不同频率的正弦（语义=频率）----------------
L = 64
FREQS = [0.10, 0.18, 0.28, 0.40]
N_PER = 250
N_CLASS = len(FREQS)

def make_class(c, n):
    out = []
    for _ in range(n):
        f = FREQS[c]
        amp = np.random.uniform(0.7, 1.3)
        ph = np.random.uniform(0, 2 * np.pi)
        t = np.arange(L)
        sig = amp * np.sin(2 * np.pi * f * t + ph)
        sig += np.random.normal(0, 0.12, L)          # 观测噪声
        out.append(sig)
    return np.array(out)

Xraw, Yraw = [], []
for c in range(N_CLASS):
    Xraw.append(make_class(c, N_PER)); Yraw.append(np.full(N_PER, c))
Xraw = np.concatenate(Xraw, 0); Yraw = np.concatenate(Yraw, 0)
perm = np.random.permutation(len(Xraw))
Xraw, Yraw = Xraw[perm], Yraw[perm]
n_tr = int(0.7 * len(Xraw))
Xtr, Ytr = Xraw[:n_tr], Yraw[:n_tr]
Xte, Yte = Xraw[n_tr:], Yraw[n_tr:]

# ---------------- 增广 ----------------
def augment(x, jitter=0.10, scale=(0.7, 1.3), mask_p=0.15):
    x = x + torch.randn_like(x) * jitter
    x = x * torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(*scale)
    if np.random.rand() < 0.5:
        x = -x
    if mask_p > 0:
        m = (torch.rand_like(x) > mask_p).float()
        x = x * m
    return x

# ---------------- 编码器 + 投影头 ----------------
class Encoder(nn.Module):
    def __init__(self, rep_dim=64):
        super().__init__()
        self.c1 = nn.Conv1d(1, 16, 5, padding=2)
        self.c2 = nn.Conv1d(16, 32, 5, padding=2)
        self.fc = nn.Linear(32, rep_dim)
    def forward(self, x):
        h = F.relu(self.c1(x))
        h = F.max_pool1d(h, 2)
        h = F.relu(self.c2(h))
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        return self.fc(h)

class Projector(nn.Module):
    def __init__(self, rep_dim=64, proj_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(rep_dim, rep_dim), nn.ReLU(), nn.Linear(rep_dim, proj_dim))
    def forward(self, z):
        z = self.net(z)
        return F.normalize(z, dim=1)

enc = Encoder(64).to(device)
proj = Projector(64, 32).to(device)
params = list(enc.parameters()) + list(proj.parameters())
opt = torch.optim.Adam(params, lr=1e-3)
TEMP = 0.5

def nt_xent(z1, z2):
    z = torch.cat([z1, z2], 0)              # 2N
    sim = z @ z.t() / TEMP
    N = z1.shape[0]
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)
    # 正确正样本对：i 与 i+N
    pos = torch.cat([torch.sum(z1 * z2, 1, keepdim=True) / TEMP,
                     torch.sum(z2 * z1, 1, keepdim=True) / TEMP], 0)
    denom = torch.logsumexp(sim, 1, keepdim=True)
    loss = -torch.mean(pos - denom)
    # 对比准确率（正样本相似度是否为同行最大）
    labels = torch.arange(N, device=z.device)
    full = torch.cat([z1, z2], 0)
    s2 = full @ full.t() / TEMP
    s2.masked_fill_(torch.eye(2 * N, dtype=torch.bool, device=z.device), -9e15)
    pred = torch.argmax(s2, 1)
    # 对 i< N，正样本是 i+N；对 i>=N，正样本是 i-N
    target = torch.cat([torch.arange(N, 2 * N, device=device), torch.arange(0, N, device=device)])
    acc = (pred == target).float().mean().item()
    return loss, acc

# ---------------- 线性探测 / KNN 辅助（numpy 实现，供训练期探针与终评共用）----------------
@torch.no_grad()
def extract(X):
    return enc(X.to(device)).cpu().numpy()

def softmax_regression_fit(X, y, epochs=300, lr=0.1, lam=1e-3):
    N, D = X.shape; C = int(y.max()) + 1
    W = np.zeros((D, C))
    for _ in range(epochs):
        scores = X @ W
        scores -= scores.max(1, keepdims=True)
        e = np.exp(scores); p = e / e.sum(1, keepdims=True)
        p[np.arange(N), y] -= 1
        grad = (X.T @ p) / N + lam * W
        W -= lr * grad
    return W

def knn(Xtr, ytr, Xte, k=5):
    acc = 0
    for i in range(0, len(Xte), 256):
        d = ((Xte[i:i+256][:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
        idx = np.argpartition(d, k, 1)[:, :k]
        votes = ytr[idx]
        pred = np.array([np.bincount(v).argmax() for v in votes])
        acc += (pred == Yte[i:i+256]).sum()
    return acc / len(Xte)

def sep_ratio(rep, y):
    mu = np.array([rep[y == c].mean(0) for c in range(N_CLASS)])
    intra = np.mean([np.mean(((rep[y == c] - mu[c]) ** 2).sum(1)) for c in range(N_CLASS)])
    inter = np.mean([((mu[i] - mu[j]) ** 2).sum() for i in range(N_CLASS) for j in range(i + 1, N_CLASS)])
    return inter / max(intra, 1e-12)

# ---------------- 训练 ----------------
BATCH = 128
EPOCHS = 150
Xt = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(1).to(device)   # (Ntr,1,L)
Yt = torch.tensor(Ytr, dtype=torch.long)
Xe = torch.tensor(Xte, dtype=torch.float32).unsqueeze(1).to(device)
Ye = torch.tensor(Yte, dtype=torch.long)

def probe_acc(enc_net, epochs_lr=200, lr=0.5):
    with torch.no_grad():
        rtr = enc_net(Xt).cpu().numpy(); rte = enc_net(Xe).cpu().numpy()
    Wr = softmax_regression_fit(rtr, Ytr, epochs=epochs_lr, lr=lr)
    return float(((rte @ Wr).argmax(1) == Yte).mean())

loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, Yt), batch_size=BATCH, shuffle=True)
loss_hist = []; probe_curve = []; probe_epochs = []
for ep in range(EPOCHS):
    tot = 0.0; cnt = 0
    for xb, _ in loader:
        v1 = augment(xb); v2 = augment(xb)
        z1 = proj(enc(v1)); z2 = proj(enc(v2))
        loss, _ = nt_xent(z1, z2)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * xb.shape[0]; cnt += xb.shape[0]
    loss_hist.append(tot / cnt)
    if ep % 20 == 0 or ep == EPOCHS - 1:
        probe_curve.append(probe_acc(enc)); probe_epochs.append(ep + 1)
R["train_loss"] = {"final": round(float(loss_hist[-1]), 4), "first": round(float(loss_hist[0]), 4),
                   "drop_pct": round((1 - loss_hist[-1] / loss_hist[0]) * 100, 1)}
R["probe_curve"] = {"epochs": probe_epochs, "acc": [round(a, 4) for a in probe_curve]}

# ---------------- 线性探测 / KNN（终评，复用上方辅助函数）----------------
repr_tr = extract(Xt); repr_te = extract(Xe)
raw_tr = Xtr.reshape(len(Xtr), -1); raw_te = Xte.reshape(len(Xte), -1)

Wrep = softmax_regression_fit(repr_tr, Ytr, epochs=300, lr=0.5)
acc_rep = float(((repr_te @ Wrep).argmax(1) == Yte).mean())
Wraw = softmax_regression_fit(raw_tr, Ytr, epochs=300, lr=0.1)
acc_raw = float(((raw_te @ Wraw).argmax(1) == Yte).mean())
acc_knn_rep = knn(repr_tr, Ytr, repr_te, k=5)
acc_knn_raw = knn(raw_tr, Ytr, raw_te, k=5)
R["eval"] = {"n_class": N_CLASS, "L": L, "n_train": int(n_tr), "n_test": int(len(Xte)),
             "acc_probe_simclr": round(acc_rep, 4), "acc_probe_raw": round(acc_raw, 4),
             "acc_random": round(1.0 / N_CLASS, 4),
             "acc_knn_simclr": round(acc_knn_rep, 4), "acc_knn_raw": round(acc_knn_raw, 4)}

# 类内/类间分离度（表征空间）
R["eval"]["sep_ratio_rep"] = round(float(sep_ratio(repr_te, Yte)), 2)
R["eval"]["sep_ratio_raw"] = round(float(sep_ratio(raw_te, Yte)), 2)

# ---------------- 图 1：增广视图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for j, axj in enumerate(ax):
    idx = 3
    x_np = Xtr[idx]
    if j == 0:
        axj.plot(x_np, color=BLACK, lw=1); axj.set_title("原始样本")
    else:
        v = augment(torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device),
                    jitter=0.12, scale=(0.6, 1.4), mask_p=0.2)
        axj.plot(v.cpu().numpy().ravel(), color=BLUE, lw=1)
        axj.set_title(f"增广视图 {j}")
    axj.set_xlabel("t"); axj.set_yticks([])
fig.suptitle(f"同一段时序的两个增广视角（SimCLR 要学到『变的是噪声、不变的是频率语义』）", fontsize=12, y=1.03)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_augmentations.png"); plt.close(fig)

# ---------------- 图 2：NT-Xent 损失 ----------------
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(np.arange(1, EPOCHS + 1), loss_hist, color=RED, lw=1.6)
ax.set_title(f"NT-Xent 对比损失：从 {R['train_loss']['first']:.2f} 降到 {R['train_loss']['final']:.2f}（降 {R['train_loss']['drop_pct']}%）")
ax.set_xlabel("epoch"); ax.set_ylabel("NT-Xent loss"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_loss.png"); plt.close(fig)

# ---------------- 图 3：PCA 二维投影 ----------------
def pca2(Xmat):
    Xc = Xmat - Xc_mean if False else None
    mean = Xmat.mean(0)
    Xc = Xmat - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T
pc = pca2(repr_te)
fig, ax = plt.subplots(figsize=(7, 5.5))
cmap = [RED, GREEN, BLUE, ORANGE, PURPLE, BLACK]
for c in range(N_CLASS):
    m = Yte == c
    ax.scatter(pc[m, 0], pc[m, 1], s=10, color=cmap[c], label=f"频率 {FREQS[c]:.2f}")
ax.set_title(f"测试集表征 PCA 投影（类内/类间分离比 {R['eval']['sep_ratio_rep']:.1f} vs 原始 {R['eval']['sep_ratio_raw']:.1f}）")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_embedding_pca.png"); plt.close(fig)

# ---------------- 图 4：探针 / KNN 对比 ----------------
fig, ax = plt.subplots(figsize=(8, 4.6))
labels = ["SimCLR\n表征", "原始信号", "随机\n基线"]
accs = [R["eval"]["acc_probe_simclr"], R["eval"]["acc_probe_raw"], R["eval"]["acc_random"]]
bars = ax.bar(np.arange(3) - 0.2, accs, 0.4, color=[RED, GRAY, BLACK], label="线性探测")
knns = [R["eval"]["acc_knn_simclr"], R["eval"]["acc_knn_raw"], R["eval"]["acc_random"]]
ax.bar(np.arange(3) + 0.2, knns, 0.4, color=[GREEN, BLUE, "#b2bec3"], label="KNN(k=5)")
for i, (a, k) in enumerate(zip(accs, knns)):
    ax.text(i - 0.2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + 0.2, k, f"{k:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(labels)
ax.set_ylabel("分类准确率"); ax.set_ylim(0, 1.05)
ax.set_title("自监督表征 vs 原始信号：线性探测 / KNN 准确率")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_probe.png"); plt.close(fig)

# ---------------- 图 5：线性探测随训练轮次 ----------------
fig, ax = plt.subplots(figsize=(7, 4.2))
xs = R["probe_curve"]["epochs"]
ax.plot(xs, R["probe_curve"]["acc"], "o-", color=RED, lw=1.6)
ax.axhline(acc_raw, color=GRAY, ls="--", lw=1, label=f"原始信号基线 {acc_raw:.3f}")
ax.axhline(1.0 / N_CLASS, color=BLACK, ls=":", lw=1, label=f"随机 {1.0/N_CLASS:.2f}")
ax.set_title("线性探测准确率随训练轮次（冻结编码器后学一个逻辑回归）")
ax.set_xlabel("epoch"); ax.set_ylabel("测试分类准确率"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_probe_curve.png"); plt.close(fig)

with open(f"{OUT}/_simclr_results.json", "w") as fp:
    json.dump(R, fp, ensure_ascii=False, indent=1, default=float)
print(json.dumps(R, ensure_ascii=False, indent=1, default=float))
s = json.dumps(R)
print("NAN/inf check:", ("nan" in s.lower()) or ("inf" in s.lower()))
