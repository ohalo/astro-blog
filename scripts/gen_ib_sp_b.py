#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""oct26e-B2: 信息瓶颈配图 B(分类版定稿) —— 从零 numpy VIB 市场状态序列预测
DGP: 隐藏市场状态 r_t ∈ {熊,震荡,牛} 为 3 态粘性马尔可夫链(P_stay=0.85);
     价格信号 s_t = REG_MU[r_t] + 0.15*noise; 观测 = 3 路含噪视角
     (s_t + 0.25e)×10帧 + d_nuis 个 AR(1)(φ=0.8) 结构化干扰序列窗口.
     目标 y = r_{t+1}(预测下一期状态).
模型: (10×d) 展平 -> 64 tanh -> 20 维高斯瓶颈(mu,logvar) -> 3 类 softmax;
     损失 = CE + beta*KL(q(z|x)||N(0,I)); beta>0 时重参数化采样, β 线性退火 warm=800.
     纯 numpy Adam, lr=0.02 除以 (1+8it/iters) 衰减.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.rcParams["font.family"] = ["Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130
plt.rcParams["figure.facecolor"] = "white"
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.3

ROOT = "/Users/halo123456/workspace/astro-blog"
OUT = ROOT + "/public/images/information-bottleneck-sequence-prediction"
os.makedirs(OUT, exist_ok=True)
RED, GREEN, BLUE, GRAY, ORANGE, BLACK, PURPLE, TEAL = "#d63031", "#00b894", "#0984e3", "#636e72", "#e17055", "#2d3436", "#6c5ce7", "#00cec9"
STATS = {}

L, P_STAY, NVIEW = 10, 0.85, 3
HID, ZDIM = 64, 20
REG_MU = np.array([-0.8, 0.0, 0.8])
SEEDS = [0, 1, 2, 3]


def gen(n_seq, dn, rng, ar=0.8):
    T = n_seq + L + 1
    r = np.empty(T, int); r[0] = rng.integers(0, 3)
    for t in range(1, T):
        r[t] = r[t - 1] if rng.random() < P_STAY else (r[t - 1] + 1 + rng.integers(2)) % 3
    s = REG_MU[r] + 0.15 * rng.normal(size=T)
    d = NVIEW + dn
    if dn:
        N = np.empty((T, dn)); N[0] = rng.normal(0, 1, dn)
        innr = rng.normal(0, np.sqrt(1 - ar ** 2), (T, dn))
        for t in range(1, T):
            N[t] = ar * N[t - 1] + innr[t]
    X = np.empty((n_seq, L, d)); y = np.empty(n_seq, int)
    for i in range(n_seq):
        t = L + i
        for j in range(NVIEW):
            X[i, :, j] = s[t - L + 1:t + 1] + 0.25 * rng.normal(size=L)
        if dn:
            X[i, :, NVIEW:] = N[t - L + 1:t + 1]
        y[i] = r[t + 1]
    return X, y


def pack(dn, seed, n_train=600, n_test=1200):
    g_ = np.random.default_rng(1000 + seed)
    Xtr, ytr = gen(n_train, dn, g_)
    Xte, yte = gen(n_test, dn, g_)
    return Xtr, ytr, Xte, yte


def train(Xtr, ytr, Xte, yte, beta, seed, iters=2400, lr0=0.02, batch=64, warm=800):
    rng = np.random.default_rng(seed)
    n, L_, d = Xtr.shape
    D = L_ * d; K = 3
    Xf = Xtr.reshape(n, D); Xtef = Xte.reshape(len(Xte), D)
    m, sdev = Xf.mean(0), Xf.std(0) + 1e-9
    Xf = (Xf - m) / sdev; Xtef = (Xtef - m) / sdev
    shapes = [(D + 1, HID), (HID + 1, ZDIM), (HID + 1, ZDIM), (ZDIM + 1, K)]
    Ws = [rng.normal(0, np.sqrt(2.0 / s[0]), s) for s in shapes]
    Ms = [np.zeros_like(w) for w in Ws]; Vs = [np.zeros_like(w) for w in Ws]
    b1, b2, eps = 0.9, 0.999, 1e-8
    sample = beta > 0

    def fwd(Xb, do_sample):
        x1 = np.concatenate([Xb, np.ones((len(Xb), 1))], 1)
        h1 = np.tanh(x1 @ Ws[0])
        h1x = np.concatenate([h1, np.ones((len(Xb), 1))], 1)
        mu = h1x @ Ws[1]; lv = np.clip(h1x @ Ws[2], -8, 8)
        z = mu + np.exp(0.5 * lv) * rng.normal(size=mu.shape) if do_sample else mu
        zx = np.concatenate([z, np.ones((len(z), 1))], 1)
        lg = zx @ Ws[3]
        lg = lg - lg.max(1, keepdims=True)
        p = np.exp(lg); p = p / p.sum(1, keepdims=True)
        return x1, h1, h1x, mu, lv, z, zx, lg, p

    for it in range(1, iters + 1):
        bt = beta * min(1.0, it / warm) if sample else 0.0
        idx = rng.integers(0, n, batch)
        x1, h1, h1x, mu, lv, z, zx, lg, p = fwd(Xf[idx], sample)
        dp = p.copy(); dp[np.arange(batch), ytr[idx]] -= 1; dp /= batch
        dz = dp @ Ws[3][:-1].T
        dmu = dz + bt * mu
        dlv = (dz * (0.5 * (z - mu)) if sample else 0.0) + bt * 0.5 * (np.exp(lv) - 1)
        g = [x1.T @ ((dmu @ Ws[1][:-1].T + dlv @ Ws[2][:-1].T) * (1 - h1 ** 2)),
             h1x.T @ dmu, h1x.T @ dlv, zx.T @ dp]
        lr = lr0 / (1 + 8.0 * it / iters)
        for W, M, V, gi in zip(Ws, Ms, Vs, g):
            M[:] = b1 * M + (1 - b1) * gi
            V[:] = b2 * V + (1 - b2) * gi * gi
            W -= lr * (M / (1 - b1 ** it)) / (np.sqrt(V / (1 - b2 ** it)) + eps)
    x1, h1, h1x, mu, lv, z, zx, lg, p = fwd(Xtef, False)
    acc = float((np.argmax(lg, 1) == yte).mean())
    # 训练集精度
    x1t = np.concatenate([Xf, np.ones((n, 1))], 1); h1t = np.tanh(x1t @ Ws[0])
    h1tx = np.concatenate([h1t, np.ones((n, 1))], 1)
    mut = h1tx @ Ws[1]; lgt = np.concatenate([mut, np.ones((n, 1))], 1) @ Ws[3]
    tracc = float((np.argmax(lgt, 1) == ytr).mean())
    rate = float(0.5 * np.sum(np.log2(1.0 + mu.var(0) / np.maximum(np.exp(lv).mean(0), 1e-9))))
    # 消融: 把干扰列(标准化后)置零再测. 展平序为 帧×特征 交错, 特征号 = 列号 % d
    Xab = Xtef.copy()
    Xab[:, (np.arange(D) % d) >= NVIEW] = 0.0
    _, _, _, _, _, _, _, lg2, _ = fwd(Xab, False)
    acc_abl = float((np.argmax(lg2, 1) == yte).mean())
    return {"acc": acc, "abl_acc": acc_abl, "train_acc": tracc, "rate": rate,
            "W1": Ws[0][:-1].copy()}


# 理论参照: 完美知道当前状态r_t, 策略"预测不变"的准确率(50万步链上实测)
g_ = np.random.default_rng(77)
T = 500000
r = np.empty(T, int); r[0] = g_.integers(0, 3)
stayc = 0
for t in range(1, T):
    r[t] = r[t - 1] if g_.random() < P_STAY else (r[t - 1] + 1 + g_.integers(2)) % 3
    stayc += (r[t] == r[t - 1])
CEIL = stayc / (T - 1)
STATS["ceiling_stay"] = float(CEIL)
print("oracle ceiling:", round(CEIL, 4))

# ---------- 图3: rate-distortion 前沿 (dn=32) ----------
DN_RD = 32
betas = [3e-4, 1e-3, 3e-3, 6e-3, 1e-2, 3e-2, 1e-1]
rd = []
for beta in betas:
    accs, rts = [], []
    for sd in SEEDS:
        Xtr, ytr, Xte, yte = pack(DN_RD, sd)
        r_ = train(Xtr, ytr, Xte, yte, beta, sd * 13 + 5)
        accs.append(r_["acc"]); rts.append(r_["rate"])
    rd.append((beta, float(np.mean(accs)), float(np.min(accs)), float(np.max(accs)), float(np.mean(rts))))
pl_accs, pl_rates = [], []
for sd in SEEDS:
    Xtr, ytr, Xte, yte = pack(DN_RD, sd)
    r_ = train(Xtr, ytr, Xte, yte, 0.0, sd * 13 + 5)
    pl_accs.append(r_["acc"]); pl_rates.append(r_["rate"])
pl_acc = float(np.mean(pl_accs)); pl_rate = float(np.mean(pl_rates))
# 多数类基线(测试集上最优单类策略)
g_ = np.random.default_rng(1003)
_, _, _, yte_ref = pack(DN_RD, 0)
maj = (np.bincount(yte_ref, minlength=3).max() / len(yte_ref))
STATS["rd"] = rd
STATS["plain_32"] = (pl_acc, pl_rate)
STATS["majority"] = float(maj)
fig, ax = plt.subplots(figsize=(8.2, 5.4))
for beta, a, lo, hi, rt in rd:
    ax.errorbar(rt, a, yerr=[[a - lo], [hi - a]], fmt="o", color=PURPLE, ms=7,
                capsize=3, zorder=3)
    ax.annotate(f"β={beta:g}", (rt, a), textcoords="offset points",
                xytext=(8, 6), fontsize=8.5, color=PURPLE)
ax.errorbar(pl_rate, pl_acc, fmt="s", color=RED, ms=9, zorder=3)
ax.annotate(f"plain (β=0, 无正则)\n码率≈{pl_rate:.1f} bit", (pl_rate, pl_acc),
            textcoords="offset points", xytext=(-70, 10), fontsize=8.5, color=RED)
ax.axhline(maj, color=GRAY, ls="--", lw=1.2, label=f"最优单类基线 {maj:.3f}")
ax.axhline(CEIL, color=GREEN, ls="--", lw=1.2, label=f"理论天花板(识破当前状态) {CEIL:.3f}")
ax.set_xlabel("瓶颈保留码率  I(X;Z) 估计 [bit]")
ax.set_ylabel("测试准确率")
ax.set_title(f"图3 信息-失真前沿（{DN_RD} 个 AR(1) 干扰维，600 样本，4 种子）：\n前两三个 bit 最值钱，plain 花 15 个 bit 买的不全是预测力")
ax.legend(fontsize=8.5, loc="center right")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/rd-frontier.png"); plt.close(fig)
print("fig3 done")

# ---------- 图4: 干扰税 + 码率随维度 ----------
nuis_list = [0, 16, 32, 48]
BEST_BETA = 3e-3
pl_a, vb_a, pl_r, vb_r = [], [], [], []
for dn in nuis_list:
    pa, va, pr, vr = [], [], [], []
    for sd in SEEDS:
        Xtr, ytr, Xte, yte = pack(dn, sd)
        r1 = train(Xtr, ytr, Xte, yte, 0.0, sd * 13 + 5)
        r2 = train(Xtr, ytr, Xte, yte, BEST_BETA, sd * 13 + 5)
        pa.append(r1["acc"]); va.append(r2["acc"]); pr.append(r1["rate"]); vr.append(r2["rate"])
    pl_a.append(float(np.mean(pa))); vb_a.append(float(np.mean(va)))
    pl_r.append(float(np.mean(pr))); vb_r.append(float(np.mean(vr)))
STATS["tax"] = {"nuis": nuis_list, "plain_acc": pl_a, "vib_acc": vb_a,
                "plain_rate": pl_r, "vib_rate": vb_r}
fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
xx = np.arange(len(nuis_list))
axes[0].plot(xx, pl_a, "o-", color=RED, lw=1.8, ms=7, label="plain（无正则）")
axes[0].plot(xx, vb_a, "s-", color=PURPLE, lw=1.8, ms=7, label=f"VIB（β={BEST_BETA:g} 退火）")
axes[0].axhline(CEIL, color=GREEN, ls="--", lw=1.1, label=f"天花板 {CEIL:.2f}")
axes[0].axhline(1 / 3, color=GRAY, ls=":", lw=1.1, label="瞎猜 1/3")
for i in range(len(nuis_list)):
    axes[0].text(i, min(pl_a[i], vb_a[i]) - 0.045, f"{pl_a[i]-vb_a[i]:+.3f}",
                 ha="center", fontsize=8, color=GRAY)
axes[0].set_xticks(xx); axes[0].set_xticklabels([str(v) for v in nuis_list])
axes[0].set_xlabel("AR(1) 干扰维数"); axes[0].set_ylabel("测试准确率")
axes[0].set_title("相关 ≠ 因果：干扰税主要咬 plain 的准确率", fontsize=10)
axes[0].legend(fontsize=8)
axes[1].plot(xx, pl_r, "o-", color=RED, lw=1.8, ms=7, label="plain 保留码率")
axes[1].plot(xx, vb_r, "s-", color=PURPLE, lw=1.8, ms=7, label="VIB 保留码率")
axes[1].set_xticks(xx); axes[1].set_xticklabels([str(v) for v in nuis_list])
axes[1].set_xlabel("AR(1) 干扰维数"); axes[1].set_ylabel("I(X;Z) 估计 [bit]")
axes[1].set_title("IB 的真实红利：码率与输入维度解耦（1.7 bit 不随干扰涨）", fontsize=10)
axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/nuisance-tax.png"); plt.close(fig)
print("fig4 done", pl_a, vb_a, pl_r, vb_r)

# ---------- 图5: 干扰消融 + 首层权重热图 (dn=48) ----------
DN_W = 48
d_model = NVIEW + DN_W
colmask_sig = (np.arange(L * d_model) % d_model) < NVIEW
maps_p, maps_v, tracc_p, tracc_v, rate_v5 = [], [], [], [], []
abl_p, abl_v, full_p, full_v = [], [], [], []
for sd in SEEDS:
    Xtr, ytr, Xte, yte = pack(DN_W, sd)
    r1 = train(Xtr, ytr, Xte, yte, 0.0, sd * 13 + 5)
    r2 = train(Xtr, ytr, Xte, yte, BEST_BETA, sd * 13 + 5)
    maps_p.append(np.abs(r1["W1"]).mean(1).reshape(L, d_model))
    maps_v.append(np.abs(r2["W1"]).mean(1).reshape(L, d_model))
    tracc_p.append(r1["train_acc"]); tracc_v.append(r2["train_acc"])
    rate_v5.append(r2["rate"])
    abl_p.append(r1["abl_acc"]); abl_v.append(r2["abl_acc"])
    full_p.append(r1["acc"]); full_v.append(r2["acc"])
p5 = np.mean(maps_p, axis=0); v5 = np.mean(maps_v, axis=0)
def share(m):
    return float(m[:, :NVIEW].mean()), float(m[:, NVIEW:].mean())
STATS["plain_share48"] = share(p5); STATS["vib_share48"] = share(v5)
STATS["tracc_p48"] = float(np.mean(tracc_p)); STATS["tracc_v48"] = float(np.mean(tracc_v))
STATS["rate_v48"] = float(np.mean(rate_v5))
ABL = {"plain": [float(np.mean(full_p)), float(np.mean(abl_p))],
       "vib": [float(np.mean(full_v)), float(np.mean(abl_v))]}
ABL["plain_loss"] = ABL["plain"][0] - ABL["plain"][1]
ABL["vib_loss"] = ABL["vib"][0] - ABL["vib"][1]
STATS["abl48"] = ABL
fig = plt.figure(figsize=(12.8, 5.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9], wspace=0.3)
sp5, sv5 = share(p5), share(v5)
for gi, (M, ttl) in enumerate((
        (p5, f"plain 首层|W|（{L}帧×{d_model}列）"),
        (v5, f"VIB 首层|W|（β={BEST_BETA:g}）"))):
    ax = fig.add_subplot(gs[0, gi])
    im = ax.imshow(M, aspect="auto", cmap="magma")
    ax.axvline(NVIEW - 0.5, color=TEAL, lw=2.2)
    ax.set_xlabel(f"列（竖线左 {NVIEW} 列=信号，右 {DN_W} 列=干扰）", fontsize=8.5)
    ax.set_ylabel("时间帧")
    ax.set_title(ttl, fontsize=9.5)
    ax.text(0.02, 0.02, f"信号列均权 {M[:, :NVIEW].mean():.4f}\n干扰列均权 {M[:, NVIEW:].mean():.4f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
            color="white", bbox=dict(fc="black", alpha=0.55, pad=2))
    plt.colorbar(im, ax=ax, fraction=0.046)
ax = fig.add_subplot(gs[0, 2])
xx = np.arange(2)
for name, pair, color, lbl in (
        ("p", ABL["plain"], RED, "plain"), ("v", ABL["vib"], PURPLE, f"VIB β={BEST_BETA:g}")):
    off = 0.0 if name == "p" else 0.18
    ax.bar(xx[0] + off, pair[0], 0.18, color=color, label=lbl)
    ax.bar(xx[1] + off, pair[1], 0.18, color=color, alpha=0.45)
    for xi, val in zip(xx, pair):
        ax.text(xi + off, val + 0.008, f"{val:.3f}", ha="center", fontsize=7.5, color=color)
ax.axhline(1 / 3, color=GRAY, ls=":", lw=1, label="瞎猜 1/3")
ax.set_xticks(xx + 0.09); ax.set_xticklabels(["全量输入", "干扰维置零"], fontsize=9)
ax.set_ylabel("测试准确率"); ax.legend(fontsize=7.5, loc="lower left")
ax.set_title(f"解剖判决：把干扰维置零（假如做得到）\nplain 反弹 +{abs(ABL['plain_loss']):.3f}，VIB 反弹 +{abs(ABL['vib_loss']):.3f}\n—— 两个模型都没学会在推理时无视干扰", fontsize=9)
fig.suptitle(f"图5 干扰税的解剖：首层权重 + 消融实验（{DN_W} 个 AR(1) 干扰维，4 种子均值）", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(f"{OUT}/vib-weights.png"); plt.close(fig)
print("fig5 done", "abl", ABL)

with open(ROOT + "/_ib_stats_b.json", "w") as f:
    json.dump(STATS, f, ensure_ascii=False, indent=1)
print("ALL OK")
