#!/usr/bin/env python3
"""生成 Shapelet 形态特征文章配图 v2"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "public/images/shapelet-time-series"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2026)

# ================= 合成数据集：两类价格片段 =================
L = 120          # 每个样本长度
n_per = 120      # 每类样本数
shape_len = 24
WALK = 0.45      # 随机游走步长
NOISE = 0.3      # 加性噪声
DEPTH = (5.0, 7.0)

def v_shape(m, depth):
    half = m // 2
    down = np.linspace(0, -depth, half)
    up = np.linspace(-depth, depth * 0.3, m - half)
    return np.concatenate([down, up])

def make_dataset(n_per, walk, noise, depth_range, seed_rng):
    Xs, ys = [], []
    for i in range(n_per * 2):
        label = 1 if i < n_per else 0
        series = np.cumsum(seed_rng.standard_normal(L) * walk)
        if label == 1:
            pos = seed_rng.integers(10, L - shape_len - 10)
            depth = seed_rng.uniform(*depth_range)
            series[pos:pos + shape_len] += v_shape(shape_len, depth)
        Xs.append(series + seed_rng.standard_normal(L) * noise)
        ys.append(label)
    Xs, ys = np.array(Xs), np.array(ys)
    idx = seed_rng.permutation(len(ys))
    return Xs[idx], ys[idx]

Xd, yd = make_dataset(n_per, WALK, NOISE, DEPTH, rng)
n_train = 160
Xtr, ytr = Xd[:n_train], yd[:n_train]
Xte, yte = Xd[n_train:], yd[n_train:]


def znorm(v):
    return (v - v.mean()) / (v.std() + 1e-9)


def min_dist(series, cand):
    m = len(cand)
    c = znorm(cand)
    best = np.inf
    for j in range(len(series) - m + 1):
        w = znorm(series[j:j + m])
        d = np.sqrt(np.mean((w - c) ** 2))
        if d < best:
            best = d
    return best


def info_gain(dists, labels):
    order = np.argsort(dists)
    d_sorted, l_sorted = dists[order], labels[order]
    n = len(labels)
    p = labels.mean()
    H = lambda q: 0.0 if q in (0.0, 1.0) else -(q*np.log2(q) + (1-q)*np.log2(1-q))
    H0 = H(p)
    best_gain, best_thr = 0.0, None
    for k in range(1, n):
        pl = l_sorted[:k].mean()
        pr = l_sorted[k:].mean()
        gain = H0 - (k/n)*H(pl) - ((n-k)/n)*H(pr)
        if gain > best_gain:
            best_gain = gain
            best_thr = (d_sorted[k-1] + d_sorted[k]) / 2
    return best_gain, best_thr


# ---- shapelet 搜索（子采样加速）----
cand_series = rng.choice(np.where(ytr == 1)[0], 30, replace=False)
cand_lens = [16, 20, 24, 28]
best = dict(gain=-1)
sub = Xtr[::2]; sub_y = ytr[::2]
for si in cand_series:
    s = Xtr[si]
    for m in cand_lens:
        for start in range(0, L - m + 1, 4):
            cand = s[start:start + m]
            dists = np.array([min_dist(x, cand) for x in sub])
            gain, thr = info_gain(dists, sub_y)
            if gain > best["gain"]:
                best = dict(gain=gain, thr=thr, cand=cand.copy(), m=m,
                            src=si, start=start)
print(f"best shapelet: len={best['m']} gain={best['gain']:.3f}")

dtr = np.array([min_dist(x, best["cand"]) for x in Xtr])
gain_full, thr_full = info_gain(dtr, ytr)
dte = np.array([min_dist(x, best["cand"]) for x in Xte])
pred = (dte <= thr_full).astype(int)
acc = (pred == yte).mean()
tp = ((pred == 1) & (yte == 1)).sum(); fp = ((pred == 1) & (yte == 0)).sum()
fn = ((pred == 0) & (yte == 1)).sum()
prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
f1 = 2 * prec * rec / max(prec + rec, 1e-9)
print(f"train gain={gain_full:.3f} thr={thr_full:.3f}")
print(f"test acc={acc:.3f} precision={prec:.3f} recall={rec:.3f} F1={f1:.3f}")

# ---- 图1: 两类样本 ----
fig, axes = plt.subplots(2, 2, figsize=(10.5, 6))
for k in range(2):
    i1 = np.where(yd == 1)[0][k]
    i0 = np.where(yd == 0)[0][k]
    axes[0, k].plot(Xd[i1], color="#e6550d", lw=1.0)
    axes[0, k].set_title(f"类别1样本（含V形形态）#{k+1}", fontsize=11)
    axes[1, k].plot(Xd[i0], color="#3182bd", lw=1.0)
    axes[1, k].set_title(f"类别0样本（纯随机游走）#{k+1}", fontsize=11)
fig.suptitle("形态位置随机、深度随机——全序列距离度量在这类数据上失效", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/shapelet-dataset.png", dpi=130)
plt.close(fig)

# ---- 图2: 学到的 shapelet 与匹配位置 ----
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
axes[0].plot(znorm(best["cand"]), color="#e6550d", lw=2.0)
axes[0].set_title(f"学到的最优 shapelet（长度 {best['m']}，信息增益 {gain_full:.2f}）", fontsize=11.5)
axes[0].set_xlabel("窗口内位置")
demo = Xte[np.where(yte == 1)[0][0]]
m = best["m"]; c = znorm(best["cand"])
dists_pos = [np.sqrt(np.mean((znorm(demo[j:j+m]) - c)**2)) for j in range(L - m + 1)]
jstar = int(np.argmin(dists_pos))
axes[1].plot(demo, color="#555", lw=1.0, label="测试集正类样本")
axes[1].plot(range(jstar, jstar + m), demo[jstar:jstar + m], color="#e6550d", lw=2.2, label="最佳匹配位置")
axes[1].legend(fontsize=9.5)
axes[1].set_title("shapelet 在新样本上滑窗匹配，取最小距离", fontsize=11.5)
fig.tight_layout()
fig.savefig(f"{OUT}/shapelet-learned.png", dpi=130)
plt.close(fig)

# ---- 图3: 距离分布 + 分类边界 ----
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.hist(dte[yte == 1], bins=24, alpha=0.6, color="#e6550d", label="正类（含形态）")
ax.hist(dte[yte == 0], bins=24, alpha=0.6, color="#3182bd", label="负类（随机游走）")
ax.axvline(thr_full, color="#333", lw=1.5, ls="--", label=f"分割阈值 d*={thr_full:.2f}")
ax.set_xlabel("到 shapelet 的最小 z-norm 距离")
ax.set_ylabel("样本数")
ax.set_title(f"单个 shapelet 距离特征分类：测试集准确率 {acc:.1%}，F1 {f1:.2f}", fontsize=12.5)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/shapelet-orderline.png", dpi=130)
plt.close(fig)

# ---- 图4: 形态深度/背景波动 信噪比压力测试 ----
depth_levels = [8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
accs = []
r2 = np.random.default_rng(99)
for dp in depth_levels:
    Xn, yn = make_dataset(80, WALK, NOISE, (dp, dp), r2)
    dn = np.array([min_dist(x, best["cand"]) for x in Xn])
    g, t = info_gain(dn, yn)
    accs.append(((dn <= t).astype(int) == yn).mean())
    print(f"depth={dp} acc={accs[-1]:.3f}")

snr = [d / (WALK * np.sqrt(shape_len)) for d in depth_levels]
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(snr, accs, "o-", color="#e6550d", lw=1.6)
ax.axhline(0.5, color="#999", lw=1.0, ls="--", label="随机猜测基线 50%")
ax.set_xlabel("信噪比：形态深度 / 背景游走同期波动")
ax.set_ylabel("分类准确率")
ax.set_title("信噪比压力测试：形态幅度接近背景波动后，判别力坍塌", fontsize=12.5)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/shapelet-noise-stress.png", dpi=130)
plt.close(fig)

print("Shapelet images done")
