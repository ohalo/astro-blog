#!/usr/bin/env python3
"""SAX (Symbolic Aggregate approXimation) experiment + figures."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from collections import Counter

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)
OUT = "/Users/halo/workspace/astro-blog/public/images/symbolic-aggregate-sax"
os.makedirs(OUT, exist_ok=True)

# ---------- SAX implementation ----------
def paa(x, w):
    n = len(x)
    idx = (np.arange(n) * w) // n
    out = np.zeros(w)
    for i in range(w):
        out[i] = x[idx == i].mean()
    return out

def sax_breakpoints(a):
    return norm.ppf(np.arange(1, a) / a)

def sax_word(x, w, a):
    z = (x - x.mean()) / (x.std() + 1e-12)
    p = paa(z, w)
    bp = sax_breakpoints(a)
    syms = np.digitize(p, bp)
    return "".join(chr(ord("a") + s) for s in syms), z, p, syms

def mindist_table(a):
    bp = np.concatenate([[-np.inf], sax_breakpoints(a), [np.inf]])
    tab = np.zeros((a, a))
    inner = sax_breakpoints(a)
    for r in range(a):
        for c in range(a):
            if abs(r - c) <= 1:
                tab[r, c] = 0.0
            else:
                tab[r, c] = inner[max(r, c) - 1] - inner[min(r, c)]
    return tab

def mindist(w1, w2, n, tab):
    d2 = sum(tab[ord(c1) - 97, ord(c2) - 97] ** 2 for c1, c2 in zip(w1, w2))
    return np.sqrt(n / len(w1)) * np.sqrt(d2)

# ---------- Fig 1: SAX pipeline illustration ----------
n, w, a = 128, 8, 4
t = np.arange(n)
x = np.sin(2 * np.pi * t / 64) * 1.5 + 0.3 * rng.standard_normal(n) + 0.01 * t
word, z, p, syms = sax_word(x, w, a)
bp = sax_breakpoints(a)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, z, color="#1f77b4", lw=1.0, alpha=0.7, label="z-normalized 序列")
seg = n // w
for i in range(w):
    ax.hlines(p[i], i * seg, (i + 1) * seg, color="#d62728", lw=2.5)
for b in bp:
    ax.axhline(b, color="gray", ls="--", lw=0.8)
labels = "abcd"
regions_y = [-2.0, (bp[0] + bp[1]) / 2, (bp[1] + bp[2]) / 2, 2.0]
for lab, ry in zip(labels, regions_y):
    ax.text(n + 2, ry, lab, fontsize=13, color="purple", va="center", weight="bold")
for i in range(w):
    ax.text(i * seg + seg / 2, p[i] + 0.25, chr(97 + syms[i]), fontsize=14,
            color="darkgreen", ha="center", weight="bold")
ax.set_title(f"SAX 流水线：z-norm → PAA(w={w}) → 高斯断点离散(a={a}) → 词 “{word}”")
ax.set_xlabel("时间"); ax.set_ylabel("标准化值")
ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUT}/sax-pipeline.png", dpi=110)
plt.close()
print(f"demo word: {word}")

# ---------- Equiprobability check ----------
big = rng.standard_normal(200000)
counts = np.histogram(big, bins=np.concatenate([[-10], bp, [10]]))[0]
print("symbol freq (should be ~equal):", counts / counts.sum())

# ---------- Fig 2: mindist lower bound check ----------
n2, w2, a2 = 64, 8, 6
tab = mindist_table(a2)
N_pairs = 3000
true_d, lb_d = [], []
for _ in range(N_pairs):
    x1 = np.cumsum(rng.standard_normal(n2)); x2 = np.cumsum(rng.standard_normal(n2))
    z1 = (x1 - x1.mean()) / x1.std(); z2 = (x2 - x2.mean()) / x2.std()
    w_1, _, _, _ = sax_word(x1, w2, a2)
    w_2, _, _, _ = sax_word(x2, w2, a2)
    true_d.append(np.sqrt(((z1 - z2) ** 2).sum()))
    lb_d.append(mindist(w_1, w_2, n2, tab))
true_d = np.array(true_d); lb_d = np.array(lb_d)
viol = (lb_d > true_d + 1e-9).mean()
tight = (lb_d / np.maximum(true_d, 1e-9)).mean()
print(f"lower-bound violations: {viol*100:.3f}%  mean tightness={tight:.3f}")

fig, ax = plt.subplots(figsize=(7.5, 6))
ax.scatter(true_d, lb_d, s=4, alpha=0.25, color="#1f77b4")
lim = max(true_d.max(), lb_d.max()) * 1.05
ax.plot([0, lim], [0, lim], "r--", lw=1.2, label="y = x（下界不可越过的线）")
ax.set_xlabel("真实欧氏距离（z-norm 后）")
ax.set_ylabel("MINDIST（SAX 词距离）")
ax.set_title(f"{N_pairs} 对随机游走：MINDIST 全部落在对角线下方\n违反率 {viol*100:.2f}%，平均紧度 {tight:.2f}")
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/mindist-lowerbound.png", dpi=110)
plt.close()

# ---------- Fig 3: motif retrieval on price series via SAX words ----------
T = 3000
ret = 0.0004 + 0.012 * rng.standard_normal(T)
# plant a pattern: sharp drop then V-recovery, length 40, at 5 spots
pat_len = 40
base_pat = np.concatenate([
    -0.010 - 0.003 * np.abs(rng.standard_normal(12)),
    0.0065 + 0.002 * np.abs(rng.standard_normal(20)),
    0.001 * rng.standard_normal(8),
])
plant_at = [400, 900, 1500, 2100, 2700]
for p0 in plant_at:
    noise = 0.004 * rng.standard_normal(pat_len)
    ret[p0:p0 + pat_len] = base_pat + noise
price = 100 * np.cumprod(1 + ret)

win, w3, a3 = 40, 8, 4
words = {}
for s in range(0, T - win):
    wd, _, _, _ = sax_word(price[s:s + win], w3, a3)
    words.setdefault(wd, []).append(s)

# find hot words with numerosity reduction (skip consecutive same-word starts)
def dedup(starts, min_gap=win):
    out = []
    for s in sorted(starts):
        if not out or s - out[-1] >= min_gap:
            out.append(s)
    return out

hot = sorted(((wd, dedup(ss)) for wd, ss in words.items()), key=lambda kv: -len(kv[1]))
# find the word that captures planted pattern
target_word, _, _, _ = sax_word(price[plant_at[0]:plant_at[0] + win], w3, a3)
hits = dedup(words.get(target_word, []))
print(f"target word: {target_word}, occurrences (dedup): {len(hits)} at {hits}")
detected = [h for h in hits if any(abs(h - p0) <= 5 for p0 in plant_at)]
print(f"planted spots recovered: {len(detected)}/5, false alarms: {len(hits) - len(detected)}")

fig, axes = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[2, 1])
ax = axes[0]
ax.plot(price, color="#333", lw=0.8)
for p0 in plant_at:
    ax.axvspan(p0, p0 + win, color="orange", alpha=0.25)
for h in hits:
    ax.plot(range(h, h + win), price[h:h + win], color="#d62728", lw=1.6)
ax.set_title(f"SAX 词检索模体：植入 5 处 V 形反转（橙色带），词 “{target_word}” 命中 {len(detected)}/5")
ax.set_ylabel("价格"); ax.grid(alpha=0.25)
ax = axes[1]
for h in hits:
    seg_ = price[h:h + win]
    z_ = (seg_ - seg_.mean()) / seg_.std()
    ax.plot(z_, alpha=0.7, lw=1.2)
ax.set_title("命中窗口 z-norm 叠放：同一个 8 字符词背后的形态族")
ax.set_xlabel("窗口内位置"); ax.set_ylabel("z 值"); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUT}/sax-motif-retrieval.png", dpi=110)
plt.close()

# ---------- Fig 4: parameter sensitivity (w,a) heatmap of retrieval F1 ----------
ws = [4, 6, 8, 10, 12]
as_ = [3, 4, 5, 6, 8]
F1 = np.zeros((len(ws), len(as_)))
for i, w_ in enumerate(ws):
    for j, a_ in enumerate(as_):
        wd_map = {}
        for s in range(0, T - win):
            wd, _, _, _ = sax_word(price[s:s + win], w_, a_)
            wd_map.setdefault(wd, []).append(s)
        tw, _, _, _ = sax_word(price[plant_at[0]:plant_at[0] + win], w_, a_)
        hs = dedup(wd_map.get(tw, []))
        tp = len([h for h in hs if any(abs(h - p0) <= 5 for p0 in plant_at)])
        fp = len(hs) - tp
        fn = 5 - tp
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / 5
        F1[i, j] = 2 * prec * rec / (prec + rec) if prec + rec else 0

fig, ax = plt.subplots(figsize=(8, 5.5))
im = ax.imshow(F1, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(as_)), [str(a_) for a_ in as_])
ax.set_yticks(range(len(ws)), [str(w_) for w_ in ws])
ax.set_xlabel("字母表大小 a"); ax.set_ylabel("词长 w（PAA 段数）")
ax.set_title("参数敏感性：模体检索 F1（查询词=首个植入窗口的 SAX 词）")
for i in range(len(ws)):
    for j in range(len(as_)):
        ax.text(j, i, f"{F1[i,j]:.2f}", ha="center", va="center",
                color="white" if F1[i, j] < 0.6 else "black", fontsize=10)
plt.colorbar(im, ax=ax, label="F1")
plt.tight_layout()
plt.savefig(f"{OUT}/sax-param-sensitivity.png", dpi=110)
plt.close()
print("F1 matrix:"); print(np.round(F1, 2))
print("DONE SAX")
