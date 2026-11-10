---
title: "二阶随机占优组合：样本内几乎完美占优，样本外 81.7% 的轮次破功"
description: "把二阶随机占优写成线性规划，用 20 个阈值约束逼组合在样本内不比等权基准差——60 次独立训练/测试切分里，样本内在约束点上的平均 violation 是 −1.9084e−05（严格占优），可换到 200 点诊断网格上样本内就已经漏出 +2.5443e−06，样本外平均 0.0003171、81.7% 的轮次直接破功。代价也没换来业绩：样本外 Sharpe 0.5118 只比等权 0.4609 高一点，远输最小方差 0.8449；5% CVaR 0.018353 对 0.019157 仅改善 4.2%，最大回撤 0.274433 反而比等权 0.260584 更差。训练场景从 60 加到 960 时 violation 降 68%，但样本内收益优势同步缩水 74%；给约束加松弛项只会把样本外 violation 抬到 5.5 倍，收益增益并不稳定。附完整可运行的 HiGHS 线性规划代码、结果 JSON 与五张真实计算图。"
publishDate: '2026-11-10T09:12:00Z'
tags:
  - 量化
  - 交易
  - 组合优化
  - 随机占优
  - Python
language: zh
difficulty: advanced
---

## 一、理论承诺很漂亮，落地形态是一个 LP

二阶随机占优（SSD）的定义可以写得很干净：$X$ 二阶占优 $Y$ 当且仅当对所有阈值 $t$ 都有 $\mathbb E[(t-X)_+]\le \mathbb E[(t-Y)_+]$。它等价于「所有递增凹效用函数都更偏好 $X$」，不需要指定效用函数形式，也不假设收益正态——这正是它比均值-方差诱人的地方。Dentcheva 与 Ruszczynski 把它搬进组合优化后，做法是用经验场景的样本均值替代期望，在每个阈值上加一条约束，解一个线性规划。

具体形态：设训练集收益矩阵 $R\in\mathbb R^{S\times N}$，组合权重 $w$ 落在单纯形上，等权基准 $b$。取基准收益的 $K$ 个分位点 $t_k$，引入辅助变量 $z_{s,k}\ge t_k-r_{w,s}$、$z_{s,k}\ge0$，则短落约束线性化为

$$\sum_s z_{s,k}\le \sum_s (t_k-r_{b,s})_+ + \varepsilon S,\qquad k=1..K$$

最大化 $\hat\mu^\top w$ 即得 SSD 组合。因为 $w=b$ 永远可行（所有约束取等号），$\varepsilon\ge0$ 时 LP 不会不可行。本文用 $N=30$、$S=240$、$K=20$，变量数 $30+240\times20=4830$，约束 $4800+20+1$ 条，交给 HiGHS 求解。四个组合同台对照：等权、SSD-LP、最小方差、均值-方差（后两者用同一单纯形约束下的 SLSQP，保证比较的是结构假设而不是约束集差异）。

数据生成刻意做成「收益与风险正相关」：30 个资产中 6 个是低/负 beta 的防御股，其余 $\beta\in[0.35,1.65]$，$\alpha_i=8\times10^{-5}+2.8\times10^{-4}\beta_i+$ 噪声；市场因子 1.5% 概率出现约 −5% 的崩盘。于是「样本均值高」往往意味着「市场 beta 高、左尾更肥」。6000 条场景池，每轮随机抽 240 条训练、1500 条测试，重复 60 次独立切分。

有一个评估细节必须先定死：**诊断阈值网格取自全场景池的基准收益分布**（0.1%～99.9% 共 200 点，下界 −0.06729927），而不是训练集。训练窗口的基准收益最小值平均只有 −0.043106——240 条场景里平均仅 1.9 个崩盘日，有些窗口甚至一个都没有；若诊断网格也从训练集取，等于把样本外的左尾整段切掉，会人为制造「占优成立」的假象。这是诊断口径，不参与任何拟合。

约束阈值的取法同样有讲究。本文取基准收益 2%～98% 的 20 个分位点，而不是在收益区间上等距铺网格：分位点把约束集中到概率质量密集的地方，等距网格会把大量约束浪费在几乎没有观测的远端。代价是低于 2% 分位的极端左尾没有显式约束，而样本外的崩盘恰好落进这段空白。把 $K$ 从 20 提到 200，辅助变量就从 4830 涨到 48030，约束更密、可行域更小，也更贴近「对所有 $t$」的原意——这是精度与计算量的直接交换，不是免费改进。本文把它留在边界一节，而不是偷偷调大以使结果好看。

## 二、样本内「满足」其实有两层含义

代表性那轮（图 1）把绝对短落曲线放在左侧、把「组合短落减去基准短落」放在右侧。左侧四条线几乎完全叠在一起，因为 violation 只有 $10^{-4}$ 量级，不画差值根本看不见；右侧一加零线就清楚了：样本内 SSD 曲线全程压在零线下方，样本外却在 $t\approx-0.008$ 到 $0.013$ 之间抬到零线以上，最大 violation $4.7452\times10^{-4}$，红色阴影就是破功区。

![样本内与样本外的短落曲线及其差值，红色阴影为占优破功区](/images/stochastic-dominance-portfolio-lp/fig1_shortfall_in_vs_out.png)

更细地拆，60 轮平均的结果是：在那 20 个**约束点**上，SSD 组合的样本内 violation 为 $-1.9084\times10^{-5}$，严格占优，LP 解对了；可一旦换到 200 点诊断网格，样本内 violation 就变成 $+2.5443\times10^{-6}$。也就是说，即使完全不看样本外，只用 20 个分位点近似「对所有 $t$」，网格之间和网格之外的区间已经漏了。这是离散化误差，不是估计误差。

样本外才是主戏：60 轮平均 violation $0.0003171$，中位数 $0.0002576$，**81.7% 的轮次为正**（图 3）。机制就写在数据里：训练集平均只有 1.9 个崩盘日，测试集有 15.63 个。LP 在几乎没见过崩盘的窗口里最大化样本均值，把仓位压到高 alpha、高 beta 的名字上；等样本外的崩盘一来，左尾短落立刻反超基准。经验分布占优是关于**这份样本**的陈述，不是关于总体的陈述。

![60 轮样本外 violation 分布，以及四个组合的平均 violation 对比](/images/stochastic-dominance-portfolio-lp/fig3_oos_violation_distribution.png)

必须补一句诚实的话：SSD 并不是唯一漏的。均值-方差组合的样本外 violation 平均 $0.00069865$，比 SSD 还高一倍多；最小方差 $0.00013655$ 最低但不是零。凡是在同一份样本上既估计收益又挑组合，分布外都会漏，只是形式不同。

## 三、占优破了，业绩也没赢

60 轮样本外平均表现如下：

| 组合 | Sharpe | 5% CVaR | 最大回撤 | 年化均值 | 有效持仓数 | 与等权 L1 距离 |
| --- | --- | --- | --- | --- | --- | --- |
| 等权基准 | 0.4609 | 0.019157 | 0.260584 | 0.054795 | 30.00 | 0.0000 |
| SSD-LP | 0.5118 | 0.018353 | 0.274433 | 0.063171 | 3.82 | 1.7624 |
| 均值-方差 | 0.5102 | 0.019613 | 0.279258 | 0.068228 | 3.05 | 1.8061 |
| 最小方差 | 0.8449 | 0.003976 | 0.049046 | 0.026160 | 15.88 | 1.1357 |

SSD 相对等权的 Sharpe 优势只有 0.0509，5% CVaR 仅改善 4.2%，最大回撤反而从 0.260584 恶化到 0.274433；最小方差以 0.8449 的 Sharpe、0.003976 的 CVaR 大幅领先。换句话说，为了「对所有凹效用都更好」这一强承诺付了线性规划的代价，换来的样本外结果还不如一个不带效用假设的方差最小化。

权重结构解释了原因：SSD 解平均只有 3.82 个有效持仓，与等权的 L1 距离 1.7624，是典型的角点解。图 2 左的代表性轮次里，单只资产权重超过 0.82；右图把权重对市场 beta 画散点，被选中的恰恰是 beta 约 0.4 的高 alpha 名字。SSD 约束没有带来分散化，反而把估计误差集中到了少数样本最优资产上——这正是优化问题里经典的「误差最大化」。

![SSD 组合权重与等权基准对比，以及权重对市场 beta 的散点](/images/stochastic-dominance-portfolio-lp/fig2_ssd_weights.png)

![四个组合 60 轮样本外 Sharpe、CVaR、最大回撤与年化均值的箱线图](/images/stochastic-dominance-portfolio-lp/fig4_oos_performance_boxplots.png)

## 四、两个旋钮：样本量与松弛项

**加数据会同时消掉泄漏和优势。** 训练场景从 60 扫到 960（图 5 左）：样本外 violation 从 $0.00062383$ 降到 $0.00019801$，降幅约 68%；但样本内年化均值相对等权的优势从 $0.558927-0.060211=0.498716$ 缩到 $0.194506-0.065949=0.128557$，缩水约 74%。换句话说，那个诱人的「样本内占优还多赚 0.5」主要是小样本噪声，数据一多就回归现实。L1 距离始终在 1.77～1.82 之间，说明集中度不是样本量能治的，得靠显式的分散化约束。

**给约束加松弛项买不到稳定收益。** $\varepsilon$ 从 0 加到 0.004（图 5 右）：样本内 violation 从 $2.39\times10^{-6}$ 涨到 $0.00151274$，样本外 violation 从 $0.00036784$ 涨到 $0.00200655$（5.5 倍），几乎是线性恶化；可样本内年化均值是 0.322153 → 0.353010 → 0.341878 → 0.393071 → 0.313497，**非单调**，样本外年化均值也在 0.048093～0.084870 之间摆动。松弛项买到的是更大的占优破裂，不是可靠的收益增益。那种「稍微放松一点约束换取更高收益」的调参直觉，在这里没有数据支持。

![训练场景数与松弛项 epsilon 的双扫描](/images/stochastic-dominance-portfolio-lp/fig5_sensitivity_scans.png)

## 五、边界与工程建议

先把适用范围钉住：这是单一合成因子模型、单期收益、无交易成本、无换手约束的场景；真实组合还要面对收益预测本身来自估计、多期复利下占优不可传递、崩盘分布非平稳等问题。本文的 violation 度量是「200 点诊断网格上短落差的最大值」，网格之外仍可能有更大的破功点，所以这些数字是下界而不是上界；60 轮与 25 轮的蒙特卡洛也有抽样噪声，扫描曲线的小幅起伏不宜过度解读。

真要把 SSD 用在实盘研究里，至少有四件事不能省：阈值网格要覆盖到样本外的极端左尾，并且单独报告网格密度敏感性；在约束之外加分散化或权重上限，压住角点解；把「样本外占优是否成立」当成一个需要检验的假设（配一个对依赖稳健的重抽样，而不是拿同一份样本再算一遍）；永远把最小方差这类简单基线放进对照——本文里它赢了所有指标。想看带清洗与隔离的切分做法可参考 [Purged K-Fold 交叉验证](/blog/purged-kfold-cv/) 与 [组合对称交叉验证](/blog/combinatorial-purged-cv/)，约束型组合优化的一般写法见 [凸优化组合约束](/blog/convex-optimization-portfolio-constraints/)，另一种下行风险目标见 [条件回撤优化](/blog/conditional-drawdown-optimization/)，多重比较下的过拟合度量见 [模型置信集](/blog/model-confidence-set/)。本篇不重复这些，只回答「经验 SSD 约束到样本外还成不成立、值不值」。

下面的代码即全部实验源码：HiGHS 线性规划、SLSQP 基线、60 轮切分、两组扫描、五张图与 `results.json` 全部现场计算，文中每个数字都能在结果文件里逐项对上。

```python
"""Second-order stochastic dominance portfolio: in-sample vs out-of-sample.

This file IS the runnable code block embedded in the article. Run it directly:

    python3 article_core_ssd_portfolio_20261110.py

It writes five real PNG figures and results.json into ARTICLE_OUT (default: cwd).
The LP is solved with scipy.optimize.linprog (HiGHS), not faked or hardcoded.
All numbers quoted in the article come from results.json.

Dependencies: numpy, scipy, matplotlib.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

OUT = Path(os.environ.get("ARTICLE_OUT", ".")).resolve()
SEED = 20261110
ANN = 252.0
N_ASSETS = 30
POOL = 6000
K_CONSTRAINT = 20
K_EVAL = 200


# ---------- synthetic return panel ----------
def build_panel():
    """Risk and expected return are positively linked; crashes are rare (1.5%).

    This is the setting where maximising the *sample* mean under sample shortfall
    constraints is genuinely dangerous: a small training window can contain zero
    or one crash, so the LP happily loads on high-beta high-alpha names whose
    left tail only shows up out of sample.
    """
    rng = np.random.default_rng([SEED, 0, 0])
    beta = rng.uniform(0.35, 1.65, size=N_ASSETS)
    beta[:6] = rng.uniform(-0.45, 0.10, size=6)          # defensive names
    alpha = 0.00008 + 0.00028 * beta + rng.normal(0.0, 0.00015, size=N_ASSETS)
    idio_vol = rng.uniform(0.004, 0.010, size=N_ASSETS)
    S = POOL
    crisis = rng.random(S) < 0.015
    f = 0.0006 + 0.008 * rng.standard_normal(S) - crisis * (0.050 + 0.015 * rng.standard_normal(S))
    noise = rng.standard_normal((S, N_ASSETS)) * idio_vol[None, :]
    R = alpha[None, :] + f[:, None] * beta[None, :] + noise
    return R, beta, alpha, idio_vol


# ---------- SSD linear program ----------
def ssd_lp(R_train, b, eps=0.0, K=K_CONSTRAINT):
    """max mu'w  s.t. w in simplex and E[(t_k - r_w)_+] <= E[(t_k - r_b)_+] + eps.

    Epigraph variables z[s,k] >= t_k - r_w[s] linearise the shortfall.
    w = b is always feasible, so the LP is never infeasible for eps >= 0.
    """
    S, N = R_train.shape
    rb = R_train @ b
    mu = R_train.mean(axis=0)
    t = np.quantile(rb, np.linspace(0.02, 0.98, K))
    cap = np.maximum(t[None, :] - rb[:, None], 0.0).sum(axis=0) + eps * S

    M = S * K
    nv = N + M
    s_idx = np.repeat(np.arange(S), K)
    k_idx = np.tile(np.arange(K), S)
    row_base = np.arange(M)
    var_z = N + s_idx * K + k_idx
    col_w = np.broadcast_to(np.arange(N), (M, N)).ravel()
    rows = np.concatenate([row_base, np.repeat(row_base, N),
                           M + np.repeat(np.arange(K), S)])
    cols = np.concatenate([var_z, col_w,
                           (N + np.arange(S)[None, :] * K + np.arange(K)[:, None]).ravel()])
    data = np.concatenate([-np.ones(M), -R_train[s_idx].ravel(),
                           np.ones(M)])
    A_ub = coo_matrix((data, (rows, cols)), shape=(M + K, nv)).tocsr()
    b_ub = np.concatenate([-t[k_idx], cap])

    A_eq = np.zeros((1, nv))
    A_eq[0, :N] = 1.0
    c = np.zeros(nv)
    c[:N] = -mu
    bounds = [(0.0, 1.0)] * N + [(0.0, None)] * M
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    return res.x[:N], t


def simplex_opt(R_train, kind, gamma=5.0):
    """Long-only min-variance / mean-variance via SLSQP, same simplex as the LP."""
    S, N = R_train.shape
    mu = R_train.mean(axis=0)
    cov = np.cov(R_train, rowvar=False) + 1e-9 * np.eye(N)
    if kind == "minvar":
        fun = lambda w: float(w @ cov @ w)
    else:
        fun = lambda w: float(-mu @ w + gamma * w @ cov @ w)
    cons = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
    x0 = np.ones(N) / N
    r = minimize(fun, x0, method="SLSQP", bounds=[(0.0, 1.0)] * N,
                 constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
    w = np.clip(r.x, 0.0, None)
    return w / w.sum()


# ---------- evaluation ----------
def shortfall_curve(r, thresholds):
    return np.maximum(thresholds[None, :] - r[:, None], 0.0).mean(axis=0)


def ssd_violation(rw, rb, thresholds):
    """Positive value means the portfolio FAILS to dominate the benchmark."""
    return float(np.max(shortfall_curve(rw, thresholds) - shortfall_curve(rb, thresholds)))


def perf(r):
    sr = float(r.mean() / r.std(ddof=1) * np.sqrt(ANN))
    ordered = np.sort(r)
    k = max(1, int(np.ceil(0.05 * len(r))))
    cvar = float(-ordered[:k].mean())          # positive number = 5% tail loss
    equity = np.concatenate([[0.0], np.cumsum(r)])
    mdd = float(-np.min(equity - np.maximum.accumulate(equity)))
    return {"sharpe": round(sr, 4), "cvar5": round(cvar, 6),
            "max_dd": round(mdd, 6), "ann_mean": round(float(r.mean() * ANN), 6)}


def evaluate_round(R, S_train, S_test, round_id, eps=0.0, K=K_CONSTRAINT):
    rng = np.random.default_rng([SEED, 1, round_id])
    pick = rng.choice(len(R), size=S_train + S_test, replace=False)
    tr, te = R[pick[:S_train]], R[pick[S_train:]]
    b = np.ones(N_ASSETS) / N_ASSETS
    w_ssd, t_constraint = ssd_lp(tr, b, eps=eps, K=K)
    w_minvar = simplex_opt(tr, "minvar")
    w_mv = simplex_opt(tr, "meanvar")
    weights = {"benchmark": b, "ssd": w_ssd, "minvar": w_minvar, "meanvar": w_mv}

    rb_tr, rb_te = tr @ b, te @ b
    # Fixed diagnostic grid from the WHOLE scenario pool (the "population"),
    # never from the training window. Otherwise a training sample with no crash
    # would produce a truncated grid and hide the out-of-sample left tail.
    rb_pool = R @ b
    eval_t = np.quantile(rb_pool, np.linspace(0.001, 0.999, K_EVAL))
    out = {"weights": {k: [round(float(v), 6) for v in w] for k, w in weights.items()},
           "in_sample_violation": {}, "oos_violation": {}, "perf": {},
           "train_mean": {}, "l1_to_benchmark": {}, "n_active": {},
           "shortfall_train": {}, "shortfall_test": {}, "eval_t": [round(float(v), 8) for v in eval_t],
           "in_sample_violation_on_constraint_grid": round(
               ssd_violation(tr @ w_ssd, rb_tr, t_constraint), 12),
           "train_benchmark_min": round(float(rb_tr.min()), 8),
           "pool_benchmark_min": round(float(rb_pool.min()), 8),
           "train_crash_days": int((rb_tr < -0.03).sum()),
           "test_crash_days": int((rb_te < -0.03).sum())}
    for name, w in weights.items():
        rw_tr, rw_te = tr @ w, te @ w
        out["in_sample_violation"][name] = round(ssd_violation(rw_tr, rb_tr, eval_t), 8)
        out["oos_violation"][name] = round(ssd_violation(rw_te, rb_te, eval_t), 8)
        out["perf"][name] = perf(rw_te)
        out["train_mean"][name] = round(float(rw_tr.mean() * ANN), 6)
        out["l1_to_benchmark"][name] = round(float(np.abs(w - b).sum()), 4)
        out["n_active"][name] = int((w > 1e-3).sum())
        out["shortfall_train"][name] = [round(float(v), 8) for v in shortfall_curve(rw_tr, eval_t)]
        out["shortfall_test"][name] = [round(float(v), 8) for v in shortfall_curve(rw_te, eval_t)]
    out["shortfall_train"]["benchmark"] = [round(float(v), 8) for v in shortfall_curve(rb_tr, eval_t)]
    out["shortfall_test"]["benchmark"] = [round(float(v), 8) for v in shortfall_curve(rb_te, eval_t)]
    out["constraint_thresholds"] = [round(float(v), 8) for v in t_constraint]
    return out


def summarise(rounds):
    keys = ["benchmark", "ssd", "minvar", "meanvar"]
    out = {}
    for name in keys:
        oos = [r["oos_violation"][name] for r in rounds]
        inv = [r["in_sample_violation"][name] for r in rounds]
        out[name] = {
            "oos_violation_mean": round(float(np.mean(oos)), 8),
            "oos_violation_median": round(float(np.median(oos)), 8),
            "oos_violation_positive_frac": round(float(np.mean(np.array(oos) > 1e-8)), 4),
            "in_sample_violation_mean": round(float(np.mean(inv)), 8),
            "sharpe_mean": round(float(np.mean([r["perf"][name]["sharpe"] for r in rounds])), 4),
            "cvar5_mean": round(float(np.mean([r["perf"][name]["cvar5"] for r in rounds])), 6),
            "max_dd_mean": round(float(np.mean([r["perf"][name]["max_dd"] for r in rounds])), 6),
            "ann_mean_mean": round(float(np.mean([r["perf"][name]["ann_mean"] for r in rounds])), 6),
            "train_mean_mean": round(float(np.mean([r["train_mean"][name] for r in rounds])), 6),
            "l1_to_benchmark_mean": round(float(np.mean([r["l1_to_benchmark"][name] for r in rounds])), 4),
            "n_active_mean": round(float(np.mean([r["n_active"][name] for r in rounds])), 2),
        }
    return out


def run_all(verbose=True):
    R, beta, alpha, idio_vol = build_panel()
    stats = {"meta": {
        "seed": SEED, "n_assets": N_ASSETS, "scenario_pool": POOL,
        "K_constraint": K_CONSTRAINT, "K_eval": K_EVAL,
        "main": {"rounds": 60, "S_train": 240, "S_test": 1500},
        "size_scan": {"rounds": 25, "S_train_values": [60, 120, 240, 480, 960]},
        "eps_scan": {"rounds": 25, "eps_values": [0.0, 0.0005, 0.001, 0.002, 0.004]},
        "beta_first6": [round(float(v), 4) for v in beta[:6]],
    }}

    main_rounds = []
    for i in range(60):
        main_rounds.append(evaluate_round(R, 240, 1500, i))
    stats["main"] = summarise(main_rounds)
    stats["representative"] = main_rounds[0]
    stats["main_diagnostics"] = {
        "in_sample_violation_on_constraint_grid_mean": round(float(np.mean(
            [r["in_sample_violation_on_constraint_grid"] for r in main_rounds])), 12),
        "eval_grid_in_sample_violation_mean": round(float(np.mean(
            [r["in_sample_violation"]["ssd"] for r in main_rounds])), 10),
        "train_crash_days_mean": round(float(np.mean([r["train_crash_days"] for r in main_rounds])), 2),
        "test_crash_days_mean": round(float(np.mean([r["test_crash_days"] for r in main_rounds])), 2),
        "pool_benchmark_min": main_rounds[0]["pool_benchmark_min"],
        "train_benchmark_min_mean": round(float(np.mean([r["train_benchmark_min"] for r in main_rounds])), 6),
    }
    if verbose:
        print("[1/3] main 60 rounds done; diagnostics:", stats["main_diagnostics"])
        for k, v in stats["main"].items():
            print("   ", k, "oos viol", v["oos_violation_mean"],
                  "frac>0", v["oos_violation_positive_frac"],
                  "sharpe", v["sharpe_mean"], "cvar", v["cvar5_mean"])

    size_scan = {}
    for S in [60, 120, 240, 480, 960]:
        rr = [evaluate_round(R, S, 1500, 1000 + S * 10 + j) for j in range(25)]
        s = summarise(rr)
        size_scan[str(S)] = {
            "ssd_oos_violation": s["ssd"]["oos_violation_mean"],
            "ssd_oos_positive_frac": s["ssd"]["oos_violation_positive_frac"],
            "ssd_train_mean": s["ssd"]["train_mean_mean"],
            "benchmark_train_mean": s["benchmark"]["train_mean_mean"],
            "ssd_sharpe": s["ssd"]["sharpe_mean"],
            "benchmark_sharpe": s["benchmark"]["sharpe_mean"],
            "l1_to_benchmark": s["ssd"]["l1_to_benchmark_mean"],
        }
    stats["size_scan"] = size_scan
    if verbose:
        print("[2/3] training-size scan done:", size_scan)

    eps_scan = {}
    for j, eps in enumerate([0.0, 0.0005, 0.001, 0.002, 0.004]):
        rr = [evaluate_round(R, 240, 1500, 5000 + j * 100 + kk, eps=eps) for kk in range(25)]
        s = summarise(rr)
        eps_scan[f"{eps:.4f}"] = {
            "ssd_in_sample_violation": s["ssd"]["in_sample_violation_mean"],
            "ssd_oos_violation": s["ssd"]["oos_violation_mean"],
            "ssd_train_mean": s["ssd"]["train_mean_mean"],
            "ssd_ann_mean_oos": s["ssd"]["ann_mean_mean"],
            "l1_to_benchmark": s["ssd"]["l1_to_benchmark_mean"],
        }
    stats["eps_scan"] = eps_scan
    if verbose:
        print("[3/3] epsilon scan done:", eps_scan)

    figs = {"main": stats["main"], "rep": main_rounds[0],
            "size_scan": size_scan, "eps_scan": eps_scan,
            "oos_violations": [r["oos_violation"]["ssd"] for r in main_rounds],
            "all_rounds": main_rounds,
            "beta": beta.tolist()}
    return stats, figs


# ---------- figures ----------
def make_figures(figs):
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
    rep = figs["rep"]
    t = np.array(rep["eval_t"])
    names = ["benchmark", "ssd", "minvar", "meanvar"]
    colors = {"benchmark": "#6b7280", "ssd": "#d62728",
              "minvar": "#1f77b4", "meanvar": "#f59e0b"}

    # Fig 1: absolute shortfall curves (left) and their differences (right).
    # The violation is ~1e-4, invisible on the absolute scale, so the
    # difference panels with a zero line are the honest way to show it.
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.0), sharex=True)
    for row, (key, tag) in enumerate([("shortfall_train", "In sample"),
                                      ("shortfall_test", "Out of sample")]):
        bench = np.array(rep[key]["benchmark"])
        a_abs, a_dif = ax[row, 0], ax[row, 1]
        a_abs.plot(t, bench, color=colors["benchmark"], lw=2.2, label="equal-weight benchmark")
        for nm in ["ssd", "minvar", "meanvar"]:
            d = np.array(rep[key][nm]) - bench
            a_abs.plot(t, np.array(rep[key][nm]), color=colors[nm], lw=1.6, label=nm)
            a_dif.plot(t, d, color=colors[nm], lw=1.7, label=nm)
        a_dif.axhline(0, color="#111827", lw=1.4, ls="--", label="dominance boundary")
        a_dif.fill_between(t, 0, np.array(rep[key]["ssd"]) - bench,
                           where=(np.array(rep[key]["ssd"]) - bench) > 0,
                           color=colors["ssd"], alpha=0.25)
        a_abs.set_ylabel("E[(t - r)+]  shortfall")
        a_dif.set_ylabel("shortfall  minus benchmark")
        a_abs.set_title(f"{tag}: absolute shortfall")
        a_dif.set_title(f"{tag}: violation becomes visible")
        a_abs.legend(frameon=False, fontsize=8)
        a_dif.legend(frameon=False, fontsize=8)
        viol = rep["in_sample_violation"]["ssd"] if row == 0 else rep["oos_violation"]["ssd"]
        a_dif.text(0.03, 0.90, f"max SSD violation = {viol:.2e}", transform=a_dif.transAxes,
                   fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                         fc="#fef2f2" if row else "#ecfdf5",
                                         ec="#fecaca" if row else "#a7f3d0"))
    ax[1, 0].set_xlabel("threshold t (daily return)")
    ax[1, 1].set_xlabel("threshold t (daily return)")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_shortfall_in_vs_out.png")
    plt.close(fig)

    # Fig 2: weights
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(N_ASSETS)
    w_ssd = np.array(rep["weights"]["ssd"])
    w_b = np.array(rep["weights"]["benchmark"])
    ax[0].bar(x - 0.2, w_b, width=0.4, color=colors["benchmark"], label="equal weight")
    ax[0].bar(x + 0.2, w_ssd, width=0.4, color=colors["ssd"], label="SSD LP")
    ax[0].set_xlabel("asset id")
    ax[0].set_ylabel("portfolio weight")
    ax[0].set_title(f"SSD weights: L1 distance to benchmark = {rep['l1_to_benchmark']['ssd']:.3f}")
    ax[0].legend(frameon=False, fontsize=8)
    lb = np.array(figs["beta"])
    ax[1].scatter(lb, w_ssd, s=32, color=colors["ssd"], edgecolor="white", zorder=3)
    for i in range(N_ASSETS):
        if w_ssd[i] > 0.02:
            ax[1].annotate(str(i), (lb[i], w_ssd[i]), fontsize=7,
                           xytext=(3, 3), textcoords="offset points")
    ax[1].axhline(1 / N_ASSETS, color=colors["benchmark"], ls="--", lw=1.2, label="equal weight")
    ax[1].set_xlabel("market beta (first 6 assets are defensive)")
    ax[1].set_ylabel("SSD LP weight")
    ax[1].set_title("The LP concentrates on a handful of names")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_ssd_weights.png")
    plt.close(fig)

    # Fig 3: distribution of OOS violations over 60 rounds
    v = np.array(figs["oos_violations"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].hist(v, bins=22, color="#d62728", alpha=0.8, edgecolor="white")
    ax[0].axvline(0, color="#111827", lw=1.5, ls="--", label="dominance boundary")
    ax[0].axvline(v.mean(), color="#1f77b4", lw=1.5, label=f"mean = {v.mean():.2e}")
    ax[0].set_xlabel("max OOS SSD violation (positive = dominance fails)")
    ax[0].set_ylabel("number of rounds")
    ax[0].set_title(f"60 rounds: {(v > 1e-8).mean() * 100:.1f}% of rounds break dominance")
    ax[0].legend(frameon=False, fontsize=8)
    main = figs["main"]
    ax[1].bar(range(4), [main[n]["oos_violation_mean"] for n in names],
              color=[colors[n] for n in names], edgecolor="white")
    ax[1].axhline(0, color="#111827", lw=1)
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels(names)
    ax[1].set_ylabel("mean max OOS violation")
    ax[1].set_title("SSD is not uniquely bad: fitted distributions all leak")
    for i, n in enumerate(names):
        ax[1].text(i, main[n]["oos_violation_mean"], f"{main[n]['oos_violation_mean']:.1e}",
                   ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_oos_violation_distribution.png")
    plt.close(fig)

    # Fig 4: OOS performance across rounds (boxplots)
    rounds = figs["all_rounds"]
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.2))
    metrics = [("sharpe", "Out-of-sample Sharpe (annualised)"),
               ("cvar5", "5% CVaR (tail loss, lower is better)"),
               ("max_dd", "Max cumulative drawdown (lower is better)"),
               ("ann_mean", "Out-of-sample annualised mean")]
    for a, (m, title) in zip(ax.ravel(), metrics):
        data = [[r["perf"][n][m] for r in rounds] for n in names]
        bp = a.boxplot(data, tick_labels=names, patch_artist=True, widths=0.6,
                       medianprops=dict(color="#111827", lw=1.4))
        for patch, n in zip(bp["boxes"], names):
            patch.set_facecolor(colors[n])
            patch.set_alpha(0.75)
        a.set_title(title, fontsize=10)
        a.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_oos_performance_boxplots.png")
    plt.close(fig)

    # Fig 5: two sensitivity scans
    ss = figs["size_scan"]
    sizes = [60, 120, 240, 480, 960]
    es = figs["eps_scan"]
    eps_keys = [f"{e:.4f}" for e in [0.0, 0.0005, 0.001, 0.002, 0.004]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    axA = ax[0]
    axA.plot(sizes, [ss[str(s)]["ssd_oos_violation"] for s in sizes], "o-",
             color="#d62728", label="OOS violation (left)")
    axA.set_xscale("log", base=2)
    axA.set_xticks(sizes)
    axA.set_xticklabels([str(s) for s in sizes])
    axA.set_xlabel("training scenarios S")
    axA.set_ylabel("mean max OOS SSD violation", color="#d62728")
    axA2 = axA.twinx()
    axA2.plot(sizes, [ss[str(s)]["ssd_train_mean"] - ss[str(s)]["benchmark_train_mean"] for s in sizes],
              "s--", color="#1f77b4", label="in-sample mean edge (right)")
    axA2.set_ylabel("in-sample annualised mean edge", color="#1f77b4")
    axA2.grid(False)
    axA.set_title("More training data shrinks both the leak and the edge")
    lines = axA.get_lines() + axA2.get_lines()
    axA.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=8, loc="upper right")
    ax[1].plot(range(len(eps_keys)), [es[k]["ssd_oos_violation"] for k in eps_keys], "o-",
               color="#d62728", label="OOS violation")
    ax[1].set_xticks(range(len(eps_keys)))
    ax[1].set_xticklabels(eps_keys)
    ax[1].set_xlabel("slack epsilon in the shortfall constraint")
    ax[1].set_ylabel("mean max OOS SSD violation", color="#d62728")
    axB2 = ax[1].twinx()
    axB2.plot(range(len(eps_keys)), [es[k]["ssd_train_mean"] for k in eps_keys], "s--",
              color="#1f77b4", label="in-sample annualised mean")
    axB2.set_ylabel("in-sample annualised mean", color="#1f77b4")
    axB2.grid(False)
    ax[1].set_title("Slack buys violation, not a reliable return edge")
    lines2 = ax[1].get_lines() + axB2.get_lines()
    ax[1].legend(lines2, [l.get_label() for l in lines2], frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_sensitivity_scans.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats, figs = run_all()
    make_figures(figs)
    (OUT / "results.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\nMain summary (60 independent train/test splits):")
    for k, v in stats["main"].items():
        print(f"  {k:10s} in-sample viol={v['in_sample_violation_mean']:.2e}  "
              f"OOS viol={v['oos_violation_mean']:.2e}  "
              f"frac>0={v['oos_violation_positive_frac']:.3f}  "
              f"Sharpe={v['sharpe_mean']:.3f}  CVaR5={v['cvar5_mean']:.5f}  "
              f"L1={v['l1_to_benchmark_mean']:.3f}")
    print("Wrote 5 PNGs + results.json to", OUT)


if __name__ == "__main__":
    main()
```
