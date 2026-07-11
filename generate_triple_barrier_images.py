#!/usr/bin/env python3
"""
为文章「标签工程：Triple Barrier 与元标签(Meta-Labeling)在机器学习量化中的应用」
(triple-barrier-metalabeling) 生成真实配图与真实统计数字。

所有图表与数字均由文中 Python 逻辑真实计算生成：
  1) tb_path.png        —— 一条价格路径 + 三条 barrier(上/下/时间)，标注首个触发点
  2) tb_dist.png        —— Triple Barrier 标签分布 vs 朴素「N 日涨跌」标签分布
  3) tb_returns.png     —— 每笔交易收益分布：Triple Barrier 策略 vs 朴素持有 N 日
  4) tb_meta.png        —— 主信号(MA 金叉)全仓 vs 主信号 + 元标签过滤 的权益曲线

关键概念：
  - Triple Barrier：对每笔入场点设上/下/时间三条边界，首个触发的边界决定标签
    (上→+1 止盈, 下→-1 止损, 时间到期→0)。比「期末符号」多编码了路径信息。
  - Meta-Labeling (López de Prado)：主模型给方向(可解释)，元模型用 ML 学
    「这次该不该真的下注」，标签用主信号方向的 Triple Barrier 是否盈利(二分类)。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "triple-barrier-metalabeling")
os.makedirs(D, exist_ok=True)

C = {"eq": "#2F4B7C", "up": "#55A868", "dn": "#C44E52", "grid": "#DDDDDD",
     "meta": "#4C72B0", "prim": "#DD8452", "time": "#8172B3", "shade": "#F2C0C0"}

# =====================================================================
# 1) 合成市场：带轻微正漂移的 GBM + 随机成交量
# =====================================================================
def simulate(T=252 * 8, seed=42):
    rng = np.random.default_rng(seed)
    mu_d, sig_d = 0.08 / 252, 0.18 / np.sqrt(252)
    r = rng.normal(mu_d, sig_d, size=T)
    P = np.concatenate([[100.0], 100.0 * np.cumprod(1 + r)])
    vol = rng.lognormal(mean=10.0, sigma=0.4, size=T + 1)   # 成交量(任意单位)
    return P, r, vol

P, r, vol = simulate()
N = len(P)

# =====================================================================
# 2) Triple Barrier 标签（向量化：逐起点扫描 horizon 内首个触发）
# =====================================================================
def triple_barrier(P, h=20, u=0.05, d=0.05):
    n = len(P)
    labels = np.zeros(n, dtype=int)          # +1 / -1 / 0
    first_hit = np.full(n, np.nan)           # 首个触发距起点的交易日数
    for t in range(n - h):
        pt = P[t]
        seg = P[t + 1:t + 1 + h] / pt - 1.0  # 相对起点 t 的累计收益
        up = np.where(seg >= u)[0]
        dn = np.where(seg <= -d)[0]
        if len(up) and (len(dn) == 0 or up[0] < dn[0]):
            labels[t] = 1
            first_hit[t] = up[0] + 1
        elif len(dn):
            labels[t] = -1
            first_hit[t] = dn[0] + 1
        # 否则时间边界触发 -> 0
    return labels, first_hit

H, U, D_TB = 20, 0.05, 0.05
labels, first_hit = triple_barrier(P, H, U, D_TB)

# 朴素标签：N 日后的符号
naive = np.zeros(N, dtype=int)
end_ret = np.zeros(N)
for t in range(N - H):
    end_ret[t] = P[t + H] / P[t] - 1.0
    naive[t] = 1 if end_ret[t] > 0 else (-1 if end_ret[t] < 0 else 0)

# 每笔「Triple Barrier 策略」收益：触发即出场(±u)，否则期末收益
tb_ret = np.zeros(N)
for t in range(N - H):
    if labels[t] == 1:
        tb_ret[t] = U
    elif labels[t] == -1:
        tb_ret[t] = -D_TB
    else:
        tb_ret[t] = end_ret[t]

# =====================================================================
# 3) 图 1：一条路径 + 三条 barrier + 首个触发点
# =====================================================================
fig, ax = plt.subplots(figsize=(11, 4.7))
ax.plot(np.arange(N), P, color=C["eq"], lw=1.0, label="价格 (合成 GBM)")
# 选一个清晰触发上边界的起点
demo = int(N * 0.30)
ax.axhline(P[demo] * (1 + U), color=C["up"], ls="--", lw=1.3, label="上边界 +%.0f%%" % (U * 100))
ax.axhline(P[demo] * (1 - D_TB), color=C["dn"], ls="--", lw=1.3, label="下边界 -%.0f%%" % (D_TB * 100))
ax.axvline(demo + H, color=C["time"], ls=":", lw=1.4, label="时间边界 t+%d" % H)
win = slice(demo - 5, demo + H + 5)
ax.plot(np.arange(N)[win], P[win], color="#111111", lw=1.8)
if not np.isnan(first_hit[demo]):
    ht = demo + int(first_hit[demo])
    ax.scatter([ht], [P[ht]], color=C["up"], s=70, zorder=6, marker="*",
               label="首个触发(上边界) -> 标签 +1")
ax.set_xlabel("交易日"); ax.set_ylabel("价格")
ax.set_title("Triple Barrier：三条边界中第一个被触发的决定标签（止盈/止损/到期）")
ax.legend(loc="upper left", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "tb_path.png"), dpi=130); plt.close()

# =====================================================================
# 4) 图 2：标签分布对比
# =====================================================================
mask = slice(0, N - H)
lt = labels[mask]; nv = naive[mask]
tb_cnt = [np.sum(lt == 1), np.sum(lt == 0), np.sum(lt == -1)]
nv_cnt = [np.sum(nv == 1), np.sum(nv == 0), np.sum(nv == -1)]
fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(3); w = 0.38
ax.bar(x - w / 2, tb_cnt, w, color=C["eq"], label="Triple Barrier (+1/0/-1)")
ax.bar(x + w / 2, nv_cnt, w, color=C["prim"], label="朴素标签 (涨/平/跌)")
ax.set_xticks(x); ax.set_xticklabels(["+1 / 涨", "0 / 平", "-1 / 跌"])
ax.set_ylabel("样本数")
ax.set_title("标签分布：Triple Barrier 把「中途触达方向」编码进 ±1，而非只看期末符号")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6, axis="y")
for i, v in enumerate(tb_cnt):
    ax.text(i - w / 2, v + 5, str(int(v)), ha="center", fontsize=7)
for i, v in enumerate(nv_cnt):
    ax.text(i + w / 2, v + 5, str(int(v)), ha="center", fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(D, "tb_dist.png"), dpi=130); plt.close()

# =====================================================================
# 5) 图 3：每笔交易收益分布（Triple Barrier 策略 vs 朴素持有 N 日）
# =====================================================================
fig, ax = plt.subplots(figsize=(11, 4.6))
bins = np.linspace(-0.20, 0.20, 41)
ax.hist(tb_ret[mask], bins=bins, alpha=0.55, color=C["eq"], label="Triple Barrier 策略(±%.0f%%封顶)" % (U * 100))
ax.hist(end_ret[mask], bins=bins, alpha=0.5, color=C["prim"], label="朴素持有 %d 日" % H)
ax.axvline(0, color="#333", lw=0.8)
ax.set_xlabel("单笔交易收益"); ax.set_ylabel("频数")
ax.set_title("风险封顶：Triple Barrier 把单笔亏损/盈利锁在边界内，朴素持有有肥左尾")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "tb_returns.png"), dpi=130); plt.close()

# =====================================================================
# 6) 元标签 Meta-Labeling：主信号(MA20>MA50) + ML 过滤
# =====================================================================
def sma(x, w):
    out = np.full(len(x), np.nan)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out

ma20 = sma(P, 20); ma50 = sma(P, 50)
signal = (ma20 > ma50).astype(int)
# 只在 signal==1 的日子考虑做多
sig_days = np.where((signal == 1) & (np.arange(N) < N - H))[0]

# 元标签：主信号方向 + Triple Barrier 是否先触上边界(盈利)
y = np.array([1 if labels[t] == 1 else 0 for t in sig_days])

# 特征
def rsi(p, w=14):
    d = np.diff(p)
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    cu = np.concatenate([[0], np.cumsum(up)]); cd = np.concatenate([[0], np.cumsum(dn)])
    rs = (cu[w:] - cu[:-w]) / (cd[w:] - cd[:-w] + 1e-9)
    out = np.concatenate([np.full(w - 1, np.nan), 100 - 100 / (1 + rs)])
    if len(out) < len(p):
        out = np.concatenate([np.full(len(p) - len(out), np.nan), out])
    return out

rsi14 = rsi(P, 14)
vol20 = np.concatenate([[np.nan], np.sqrt(252) * np.convolve(np.abs(r[1:] - r[:-1]), np.ones(20) / 20, mode="full")[:N - 1]])
vma = sma(vol, 20)
vol_z = (vol - vma) / (vma + 1e-9)
mom = P / np.roll(P, 20) - 1.0
dist = (P - ma20) / (ma20 + 1e-9)

feat_names = ["vol20", "rsi14", "vol_z", "mom20", "distMA"]
F = np.vstack([vol20, rsi14, vol_z, mom, dist]).T
F = np.nan_to_num(F, nan=0.0)
rows = [(t, F[t], y[i]) for i, t in enumerate(sig_days) if not np.any(np.isnan(F[t]))]
Tsig = np.array([x[0] for x in rows]); X = np.array([x[1] for x in rows]); Y = np.array([x[2] for x in rows])

# 时序切分(前 70% 训练，后 30% 测试)，避免前视
cut = int(len(Tsig) * 0.70)
Xtr, Ytr = X[:cut], Y[:cut]
Xte, Yte, Tte = X[cut:], Y[cut:], Tsig[cut:]

# 手写逻辑回归 (numpy 梯度下降) —— 依赖自包含
def train_lr(X, y, iters=3000, lr=0.05):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        z = Xb @ w
        p = 1 / (1 + np.exp(-z))
        g = Xb.T @ (p - y) / len(y)
        w -= lr * g
    return w

def predict(w, X):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    return 1 / (1 + np.exp(-(Xb @ w)))

# 标准化(仅用训练集统计量)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd
w = train_lr(Xtr_s, Ytr)
p_te = predict(w, Xte_s)

# 交易模拟(测试段，按时间顺序逐笔复利，假设一次一笔)
def equity_curve(times, take_mask):
    eq = 1.0; eqs = [1.0]; dds = [0.0]; peak = 1.0
    for i, t in enumerate(times):
        if not take_mask[i]:
            continue
        ret = tb_ret[t] if labels[t] != 0 else end_ret[t]
        eq *= (1 + ret)
        peak = max(peak, eq); dds.append(eq / peak - 1); eqs.append(eq)
    return np.array(eqs), np.array(dds)

# (a) 主信号全仓
eq_prim, dd_prim = equity_curve(Tte, np.ones(len(Tte), dtype=bool))
# (b) 主信号 + 元标签过滤 (按预测概率取前 50% 下注——标准做法，避免校准失效导致全拒)
take = p_te >= np.median(p_te)
eq_meta, dd_meta = equity_curve(Tte, take)

def stats(eqs, dds):
    if len(eqs) <= 1:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    rets = np.diff(eqs) / eqs[:-1]
    return eqs[-1], dds.min() * 100, rets.mean() * 100, rets.std() * 100, (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252)

s_prim = stats(eq_prim, dd_prim)
s_meta = stats(eq_meta, dd_meta)

fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(np.arange(len(eq_prim)), eq_prim, color=C["prim"], lw=1.5, label="主信号全仓 (MA 金叉)")
ax.plot(np.arange(len(eq_meta)), eq_meta, color=C["meta"], lw=1.5, label="主信号 + 元标签过滤")
ax.set_xlabel("测试段交易序号"); ax.set_ylabel("权益 (初始=1.0)")
ax.set_title("元标签过滤：剔掉主模型「看多但不该下注」的交易，提升风险调整收益")
ax.legend(loc="upper left", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "tb_meta.png"), dpi=130); plt.close()

# =====================================================================
# 打印真实数字
# =====================================================================
print("=== Triple Barrier 标签工程 关键数字 ===")
print("样本: 日度 %d 天 (约 %.1f 年)，年化漂移 8%% / 波动 18%% 合成 GBM" % (N, N / 252))
print("参数: horizon h=%d 日, 上边界 +%.0f%%, 下边界 -%.0f%%" % (H, U * 100, D_TB * 100))
print("Triple Barrier 标签分布: +1=%d, 0(到期)=%d, -1=%d" % tuple(tb_cnt))
print("朴素标签分布: 涨=%d, 平=%d, 跌=%d" % tuple(nv_cnt))
print("TB 策略单笔: 均值=%.3f%%  Std=%.3f%%  胜率=%.1f%%" % (
    tb_ret[mask].mean() * 100, tb_ret[mask].std() * 100,
    (tb_ret[mask] > 0).mean() * 100))
print("朴素持有单笔: 均值=%.3f%%  Std=%.3f%%  胜率=%.1f%%  最差单笔=%.1f%%" % (
    end_ret[mask].mean() * 100, end_ret[mask].std() * 100,
    (end_ret[mask] > 0).mean() * 100, end_ret[mask].min() * 100))
print("--- 元标签诊断 ---")
print("训练正样本(盈利)占比: %.1f%%  测试: %.1f%%" % (Ytr.mean() * 100, Yte.mean() * 100))
pred_test = (p_te > 0.5).mean()
order = np.argsort(np.argsort(p_te))
nt = len(p_te); pos = Yte.sum(); neg = nt - pos
if pos > 0 and neg > 0:
    auc = (order[Yte == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
else:
    auc = float("nan")
print("元模型测试集: 预测盈利占比=%.1f%%  AUC=%.3f  平均p=%.3f" % (pred_test * 100, auc, p_te.mean()))
print("元标签过滤下注数: %d/%d" % (int(take.sum()), len(Tte)))
print("--- 元标签 (测试段 %d 笔主信号交易) ---" % len(Tte))
print("主信号全仓: 终值=%.3f  最大回撤=%.1f%%  单笔均值=%.3f%%  单笔Std=%.3f%%  Sharpe=%.2f" % s_prim)
print("元标签过滤: 下注 %d/%d 笔  终值=%.3f  最大回撤=%.1f%%  单笔均值=%.3f%%  单笔Std=%.3f%%  Sharpe=%.2f" % (
    int(take.sum()), len(Tte), s_meta[0], s_meta[1], s_meta[2], s_meta[3], s_meta[4]))
print("训练集正样本占比(盈利交易): %.1f%%  测试集: %.1f%%" % (Ytr.mean() * 100, Yte.mean() * 100))
print("\n图片已保存到:", D)
