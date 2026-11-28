---
title: "递归图与递归量化分析 RQA：把收益序列嵌进相空间，为什么层流率比确定性率更早看见结构切换、又在阈值与窗口选择下失真"
description: "在 1500 天合成行情里让自相关先抬升、波动率后跳升：DET 报警在第 555 天，反而比 40 日波动率的第 545 天晚 10 天；LAM 在第 405 天报警，领先波动率 140 天，但此前 100 天平静期已有 6 个瞬时越界。平静段 LAM 均值 0.1444、预压段 0.2850、危机段 0.3889，DET 三段仅 0.2843、0.2907、0.3198。半径分位从 0.05 放到 0.20，危机与平静的 DET 差从 0.0747 收缩到 0.0204；嵌入维 m=5 时差距反号。按窗口分位定半径会让 RR 恒等于约 0.1000，不能当状态信号。附完整代码与五张真实计算图。"
publishDate: '2026-11-28T18:10:00Z'
tags:
  - 量化
  - 风险监测
  - 非线性时序
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

多数风险监控只看波动率：滚动标准差、EWMA、GARCH 都是在问「价格摆动有多大」。但市场在真正剧烈波动之前，往往先改变的是「怎么摆动」——收益之间的依赖结构先变紧，幅度随后才放大。递归图与递归量化分析（RQA）把一维收益展开成相空间轨迹，再数轨迹重新访问自己附近区域的次数，正好对准这个变化。

本文用一个 1500 天的确定性合成实验把这件事拆开。第 450 到 570 天是预压段：波动率只从 0.0060 抬到 0.0062，几乎没变；一阶自相关却从 0.05 跳到 0.55。第 570 天起才进入危机段，波动率升到 0.0200。结果不是教科书式的「RQA 全面领先」：对角线确定性率 DET 在第 555 天才报警，比 40 日波动率的第 545 天还晚 10 天；垂直线层流率 LAM 却在第 405 天报警，领先波动率 140 天。领先有代价——报警前 100 天的平静期里已有 6 个瞬时越界，后段平静期还有 1 次。

因此正确的用法不是拿 RQA 替代波动率，而是把它当作「结构早期预警 + 明确误报预算」的补充通道。

![合成收益与三个状态段](/images/recurrence-quantification-regime/fig1_returns.png)

## 二、相空间递归与四个指标

对收益序列 $r_t$ 做延迟嵌入，取维度 m=3、延迟 tau=1，得到状态向量 $x_t=(r_t,r_{t-1},r_{t-2})$。两个状态在相空间里距离小于半径 eps 时，递归矩阵对应位置记为 1：

$$
R_{ij}=\Theta(\varepsilon-\|x_i-x_j\|_2),\quad i\neq j.
$$

为了避免「相邻两天当然靠近」这种平凡递归，需要排除 Theiler 窗内的点，本文取 5 天。半径不是固定绝对长度，而是每个窗口内所有有效点对距离的 5% 分位；这样波动幅度整体放大时，递归密度不会被机械地压到零。

在递归矩阵上数四种结构。递归率 RR 是 1 的占比；确定性率 DET 是长度不小于 2 的对角线上的递归点占全部递归点的比例，衡量轨迹重复走出相似片段的程度；平均对角线长度 Lmean 衡量这些片段持续多久；层流率 LAM 是垂直线上长线的递归点占比，衡量状态卡在同一邻域多少步。DET 看「重复轨迹」，LAM 看「停留时间」，两者在状态切换中的表现可以完全不同。

$$
DET=\frac{\sum_{\ell\ge \ell_{\min}}\ell\,N_{\ell}}{\sum_{ij}R_{ij}},\quad
LAM=\frac{\sum_{\nu\ge \nu_{\min}}\nu\,P_{\nu}}{\sum_{ij}R_{ij}}.
$$

实现上有两个细节容易被忽略。第一，对角线长度只统计主对角线以外的偏移对角线，否则每个点都会因为自己和自己递归而贡献一条长度为 1 的平凡线；本文还用 Theiler 窗把时间上相隔 5 天以内的点对整体剔除。第二，最短线长设为 2，若取 1，DET 几乎等于递归率，指标失去区分度；若取 5 以上，短窗口里合格线段太少，估计会剧烈抖动。本文把 lmin 固定为 2，并把所有可调参数写进结果文件，便于复核。

## 三、先变结构、后变波动的合成市场

生成过程是一段分段 AR(1)：$r_t=\phi r_{t-1}+\sigma_t \epsilon_t$。平静段 phi=0.05、sigma=0.0060；预压段 phi=0.55、sigma=0.0062；危机段 phi=0.72、sigma=0.0200；恢复段 phi=0.30、sigma=0.0100；最后 500 天回到平静。随机种子固定为 20261128，所有数值都可复现。

为什么要让预压段只改 phi？因为自相关上升意味着收益方向更持久，相空间轨迹会更倾向于沿着某些方向缓慢游走，垂直线因此变长；而波动率几乎没动，滚动标准差看不到异常。这正是检验两类指标分工的干净对照。

窗口长度取 100 天、步长 5 天，得到 281 个窗口。平静段 DET 均值 0.2843、LAM 均值 0.1444；预压段 DET 只升到 0.2907，LAM 已升到 0.2850；危机段 DET 0.3198、LAM 0.3889。换句话说，DET 的预压段信噪比只有 0.0064，LAM 是 0.1406，相差二十多倍。

![窗口化 RQA 指标与滚动波动率](/images/recurrence-quantification-regime/fig3_window_metrics.png)

## 四、谁先报警：DET 输给波动率，LAM 大幅领先

报警规则一致：用第 450 天之前的窗口估计基线，阈值取均值加三倍标准差，并要求连续 3 个窗口越界。DET 阈值 0.3374，首次连续报警在第 555 天；40 日波动率阈值 0.007216，首次报警在第 545 天。DET **晚了 10 天**，原因是危机早期的大幅震荡让轨迹在相空间里被拉开，对角线片段反而被打断；DET 在幅度冲击下并不是单调的风险计。

LAM 阈值 0.3341，首次连续报警在第 405 天，比波动率早 140 天，比预压段起点还早 45 天。这个「过早」需要认真解释：在报警前的 100 天平静窗口里，LAM 已有 6 次单点越界，只是没有连续 3 次；后段平静期也有 1 次。也就是说，LAM 的高灵敏度是用误报换来的。若策略每次报警都降杠杆，140 天的提前量对应的是多次空转。

连续 3 窗的规则本身就是一种折中：允许单点越界，能滤掉大部分由个别极端收益造成的垂直线；要求连续出现，则把真正持续的结构变化保留下来。若改成连续 1 窗，首次报警会提前到那 6 次单点越界中的最早一次，误报预算无法控制；若改成连续 10 窗，报警又会拖到危机深处。实盘里应当先固定可承受的误报次数，再反推连续窗数，而不是反过来挑一个好看的提前量。

![平静与危机状态的递归矩阵](/images/recurrence-quantification-regime/fig2_recurrence_matrices.png)

递归矩阵图给出了直观解释。平静段的点散成细碎噪点，长垂直线很少；危机段的点聚成更粗的块状结构，说明状态在同一邻域停留更久。DET 只看对角线，块状结构未必贡献长对角线；LAM 直接数垂直线，因此先感知到变化。

## 五、半径和嵌入维会改写结论

半径分位从 0.02、0.05、0.10 扫到 0.20：平静段 DET 分别是 0.1512、0.2593、0.2967、0.3744，危机段是 0.2500、0.2991、0.3341、0.3949。差距从 0.0988 缩到 0.0747、0.0374、0.0204。半径越大，越多点对被算作递归，指标越接近平滑的密度估计，状态差异被稀释；半径太小时递归点太少，DET 的抽样噪声又会放大。5% 到 10% 是这段数据里较稳的区间。

嵌入维从 2 扫到 5，平静段 DET 为 0.1986、0.2593、0.3167、0.3659，危机段为 0.2306、0.2991、0.3548、0.3585。到 m=5 时危机段反而比平静段低 0.0073。高维嵌入会稀释邻近点，维度越高越容易把真实结构判成噪声。任何 RQA 结论都必须连同 m、tau、eps、Theiler 窗一起报告，否则无法复现。

![半径分位敏感性](/images/recurrence-quantification-regime/fig4_threshold_sensitivity.png)

还有一个容易踩的坑：本文按窗口分位定半径，RR 在四个状态段都恒为约 0.1000，敏感性地上的 RR 也完全跟着分位走。这是归一化的数学后果，不是市场状态不变。若把 RR 当预警信号，只会测到分位参数本身。要比较递归密度，必须使用跨窗口统一的绝对半径，或者同时报告原始距离尺度。

## 六、可交易的含义与边界

一个可行的用法是把 LAM 当作结构监测器、把波动率当作幅度确认器：LAM 连续越界先行提示「依赖结构已变」，波动率随后越界才执行减仓。本文样本里这套顺序能把预警提前 140 天，但必须预留至少 7 次平静期误报的处理规则。只靠 DET 做单指标报警在本例里没有任何领先性，甚至慢于最简单的滚动波动率。

上线时基线也不能一次算完就固定。本文为了可复现，用第 450 天之前的平静窗口估计阈值；实盘更稳妥的做法是用过去一年滚动重估均值与标准差，并在阈值被突破后进入冷却期，避免同一段结构变化被反复计数。另一个工程约束是计算量：每个窗口要算 100×100 的距离矩阵，281 个窗口在普通笔记本上可以秒级完成，但若改成逐日更新上百个品种，就需要用近似近邻或稀疏递归矩阵来控制开销。

需要强调边界。这是合成数据，自相关的阶跃是我们手工设定的，真实市场没有如此干净的断点；RQA 对窗口长度同样敏感，100 天窗口覆盖约一个季度，短窗口会让垂直线统计失去意义；交易成本、流动性和事件驱动跳空都没有纳入。RQA 的价值在于提供一个不同于波动率的观察角度，而不是一张免检验的预警许可证。

![报警提前量与嵌入维敏感性](/images/recurrence-quantification-regime/fig5_lead_and_embedding.png)

## 七、完整可运行代码

下面的代码生成全部数据、五张图与 results.json，可独立运行。

```python
# -*- coding: utf-8 -*-
"""RQA on a synthetic market with an autocorrelation shift before volatility jumps."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 20261128
N = 1500
W = 100
STEP = 5
M = 3
TAU = 1
THEILER = 5
LMIN = 2
Q = 0.05
IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "public", "images", "recurrence-quantification-regime")

PHASES = [
    ("calm", 0, 450, 0.05, 0.0060),
    ("pre-stress", 450, 570, 0.55, 0.0062),
    ("crisis", 570, 750, 0.72, 0.0200),
    ("recovery", 750, 1000, 0.30, 0.0100),
    ("calm", 1000, N, 0.05, 0.0060),
]


def generate_returns():
    rng = np.random.default_rng(SEED)
    shocks = rng.standard_normal(N)
    r = np.zeros(N)
    for _, a, b, phi, sig in PHASES:
        for t in range(a, b):
            prev = r[t - 1] if t > a else 0.0
            r[t] = phi * prev + sig * shocks[t]
    return r


def embed(x, m, tau):
    n = len(x) - (m - 1) * tau
    return np.column_stack([x[i * tau:n + i * tau] for i in range(m)])


def runs_1d(v):
    edges = np.diff(np.r_[False, v, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return ends - starts


def rqa_metrics(vec, q=Q, m=M, theiler=THEILER, lmin=LMIN):
    y = embed(vec, m, TAU)
    w = len(y)
    d = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=2)
    valid = np.ones((w, w), dtype=bool)
    for k in range(theiler + 1):
        valid &= np.abs(np.eye(w, k=k, dtype=bool)) == False
        valid &= np.abs(np.eye(w, k=-k, dtype=bool)) == False
    iu = np.triu_indices(w, 1)
    pool = d[iu][valid[iu]]
    eps = np.quantile(pool, q)
    rmat = (d <= eps) & valid

    n_rec = int(rmat.sum())
    diag = []
    for k in range(1, w):
        diag.extend(runs_1d(np.diag(rmat, k)))
    diag = np.asarray(diag, dtype=int)
    vert = []
    for j in range(w):
        vert.extend(runs_1d(rmat[:, j]))
    vert = np.asarray(vert, dtype=int)
    long_d = diag[diag >= lmin]
    long_v = vert[vert >= lmin]
    return {
        "eps": float(eps),
        "rr": n_rec / len(pool),
        "det": float(long_d.sum() / n_rec) if n_rec else 0.0,
        "lmean": float(long_d.mean()) if len(long_d) else 0.0,
        "lam": float(long_v.sum() / n_rec) if n_rec else 0.0,
        "tt": float(long_v.mean()) if len(long_v) else 0.0,
        "matrix": rmat,
    }


def rolling_vol(r, win=40):
    out = np.full(len(r), np.nan)
    for i in range(win, len(r)):
        out[i] = r[i - win:i].std(ddof=1)
    return out


def first_alert(centers, values, calm_end=450, k=3):
    base = values[(centers < calm_end - W) & np.isfinite(values)]
    thr = base.mean() + 3.0 * base.std(ddof=1)
    for i in range(len(values)):
        if not np.isfinite(values[i]) or values[i] <= thr:
            continue
        if np.all(values[i:i + k] > thr):
            false_count = int(((centers >= calm_end - W) & (centers < calm_end) & (values > thr)).sum())
            return float(centers[i]), float(thr), false_count
    return float("nan"), float(thr), 0


def main():
    os.makedirs(IMG, exist_ok=True)
    r = generate_returns()
    rv = rolling_vol(r)
    centers = []
    dets, rrs, lmeans, lams = [], [], [], []
    for start in range(0, len(r) - W + 1, STEP):
        met = rqa_metrics(r[start:start + W])
        centers.append(start + W // 2)
        dets.append(met["det"])
        rrs.append(met["rr"])
        lmeans.append(met["lmean"])
        lams.append(met["lam"])
    centers = np.asarray(centers, dtype=float)
    dets = np.asarray(dets)
    rrs = np.asarray(rrs)
    lmeans = np.asarray(lmeans)
    lams = np.asarray(lams)
    vol_win = np.asarray([rv[int(c)] if np.isfinite(rv[int(c)]) else np.nan for c in centers])

    det_day, det_thr, det_false = first_alert(centers, dets)
    lam_day, lam_thr, lam_false = first_alert(centers, lams)
    vol_day, vol_thr, vol_false = first_alert(centers, vol_win)

    seg = {}
    for name, a, b, _, _ in PHASES:
        sel = (centers >= a) & (centers < b)
        seg[name] = {
            "det": float(dets[sel].mean()),
            "rr": float(rrs[sel].mean()),
            "lmean": float(lmeans[sel].mean()),
            "lam": float(lams[sel].mean()),
            "n": int(sel.sum()),
        }

    q_grid = [0.02, 0.05, 0.10, 0.20]
    sens = {}
    for qv in q_grid:
        c = rqa_metrics(r[350:350 + W], q=qv)
        s = rqa_metrics(r[620:620 + W], q=qv)
        sens[str(qv)] = {"det_calm": c["det"], "det_crisis": s["det"],
                         "rr_calm": c["rr"], "rr_crisis": s["rr"]}

    emb = {}
    for m in (2, 3, 4, 5):
        c = rqa_metrics(r[350:350 + W], m=m)
        s = rqa_metrics(r[620:620 + W], m=m)
        emb[str(m)] = {"det_calm": c["det"], "det_crisis": s["det"]}

    calm_mat = rqa_metrics(r[350:350 + W])["matrix"]
    crisis_mat = rqa_metrics(r[620:620 + W])["matrix"]

    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(r, lw=0.7, color="#3b5b8c")
    for a, b in [(450, 570), (570, 750), (750, 1000)]:
        ax.axvspan(a, b, alpha=0.16, color="#c0392b" if b == 750 else "#e0a33e")
    ax.set_title("Synthetic returns: autocorrelated pre-stress before the volatility jump")
    ax.set_xlabel("day")
    ax.set_ylabel("return")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig1_returns.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, mat, title in zip(axes, [calm_mat, crisis_mat],
                              ["Calm recurrence", "Crisis recurrence"]):
        ax.imshow(mat, cmap="Greys", interpolation="none")
        ax.set_title(title)
        ax.set_xlabel("state index")
        ax.set_ylabel("state index")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig2_recurrence_matrices.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(centers, dets, lw=1.4, color="#2c6e7f", label="DET (RQA)")
    ax.plot(centers, lams, lw=1.0, color="#8e6cbb", label="LAM (RQA)")
    ax2 = ax.twinx()
    ax2.plot(centers, vol_win, lw=1.1, color="#c0392b", label="40d vol")
    ax.axhline(det_thr, ls="--", lw=0.9, color="#2c6e7f")
    ax2.axhline(vol_thr, ls="--", lw=0.9, color="#c0392b")
    for a, b in [(450, 570), (570, 750)]:
        ax.axvspan(a, b, alpha=0.12, color="#e0a33e")
    ax.set_title("Windowed RQA metrics versus rolling volatility")
    ax.set_xlabel("day")
    ax.set_ylabel("RQA metric")
    ax2.set_ylabel("rolling vol")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig3_window_metrics.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].plot(q_grid, [sens[str(q)]["det_calm"] for q in q_grid], "o-", color="#3b5b8c", label="calm")
    axes[0].plot(q_grid, [sens[str(q)]["det_crisis"] for q in q_grid], "o-", color="#c0392b", label="crisis")
    axes[0].set_title("DET across recurrence radius quantiles")
    axes[0].set_xlabel("eps quantile")
    axes[0].set_ylabel("DET")
    axes[0].legend()
    axes[1].plot(q_grid, [sens[str(q)]["rr_calm"] for q in q_grid], "o-", color="#3b5b8c", label="calm")
    axes[1].plot(q_grid, [sens[str(q)]["rr_crisis"] for q in q_grid], "o-", color="#c0392b", label="crisis")
    axes[1].set_title("RR across recurrence radius quantiles")
    axes[1].set_xlabel("eps quantile")
    axes[1].set_ylabel("RR")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig4_threshold_sensitivity.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ms = [2, 3, 4, 5]
    axes[0].plot(ms, [emb[str(m)]["det_calm"] for m in ms], "o-", color="#3b5b8c", label="calm")
    axes[0].plot(ms, [emb[str(m)]["det_crisis"] for m in ms], "o-", color="#c0392b", label="crisis")
    axes[0].set_title("Embedding dimension sensitivity")
    axes[0].set_xlabel("m")
    axes[0].set_ylabel("DET")
    axes[0].legend()
    axes[1].bar(["DET", "LAM", "40d vol"], [det_day - 450, lam_day - 450, vol_day - 450],
                color=["#2c6e7f", "#8e6cbb", "#c0392b"])
    axes[1].axhline(0, color="0.3", lw=0.8)
    axes[1].set_title("First alert relative to regime start")
    axes[1].set_ylabel("days after day 450")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig5_lead_and_embedding.png"), dpi=150)
    plt.close(fig)

    out = {
        "seed": SEED,
        "n_days": N,
        "window": W,
        "step": STEP,
        "embedding_dim": M,
        "theiler": THEILER,
        "lmin": LMIN,
        "eps_quantile": Q,
        "segments": seg,
        "det_alert_day": round(det_day, 3),
        "lam_alert_day": round(lam_day, 3),
        "vol_alert_day": round(vol_day, 3),
        "det_lead_days": round(vol_day - det_day, 3),
        "lam_lead_days": round(vol_day - lam_day, 3),
        "det_threshold": round(det_thr, 6),
        "lam_threshold": round(lam_thr, 6),
        "vol_threshold": round(vol_thr, 6),
        "pre_alert_false_alarms": det_false,
        "lam_false_alarms": lam_false,
        "vol_false_alarms": vol_false,
        "calm2_false_alarms": int(((centers >= 1000) & (dets > det_thr)).sum()),
        "calm2_lam_false_alarms": int(((centers >= 1000) & (lams > lam_thr)).sum()),
        "threshold_sensitivity": {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in sens.items()},
        "embedding_sensitivity": {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in emb.items()},
    }
    with open(os.path.join(IMG, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
```

