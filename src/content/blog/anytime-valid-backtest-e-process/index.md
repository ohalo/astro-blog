---
title: "回测里反复看 p 值：5% 会被偷看偷成 42%，e-process 怎么把门关上"
description: "回测跑 250 天，每周看一次 p 值，只要某次小于 0.05 就宣布策略有效——3000 次纯噪声模拟里这么做有 30.57% 会误报，每天看则涨到 42.33%，而名义水平只有 5%。本文用混合似然比鞅实现 anytime-valid 的 e-process，把任意停止时刻的 I 类错误锁在 0.0177（Ville 不等式上界 0.05），并诚实量化代价：mu=0.15 时功效 0.3880，比 Bonferroni 的 0.2953 高但低于虚高的偷看组 0.8393，平均首次检出推后到 147.1 天。两盆冷水：把真实波动 1.3 错当成 1.0，e-process 拒绝率就膨胀到 0.1143；混合先验宽度 tau 选错，功效会从 0.3793 掉到 0.0007。附完整可运行代码、结果 JSON 与五张真实计算图。"
publishDate: '2026-11-08T11:00:00Z'
tags:
  - 量化
  - 交易
  - 统计推断
  - 回测
  - Python
language: zh
difficulty: advanced
---

## 一、每周看一次，5% 就不是 5% 了

假设一个策略的日超额收益独立同分布，原假设是均值为零、标准差为 1。回测窗口 250 个交易日，研究员每周五算一次累计 z 统计量 $Z_n=S_n/(\sigma\sqrt n)$，只要某次双侧 p 值低于 0.05 就把策略放进候选池。这个流程在绝大多数团队里都不被认为有问题，因为每次看的都是「标准」的双侧 z 检验。

问题在于，$\{Z_n\}$ 不是 250 个相互独立的检验统计量，而是同一条随机游走在 50 个时间点上的取值。固定终点检验控制的是 $P(|Z_{250}|>1.96)=0.05$，反复偷看控制的是 $P(\exists n\in\mathcal G: |Z_n|>1.96)$，后者的事件严格更大。用 3000 条纯噪声路径实测：只在终点看是 0.0467，与名义值相符；每 5 天看一次是 **0.3057**；每天看一次是 **0.4233**。也就是说，一个完全没有alpha的策略，有接近一半的概率会在某一天「显著」。

![](/images/anytime-valid-backtest-e-process/fig1_peeking_paths.png)

图 1 是这批模拟里挑出来的一条原假设路径（第 0 号，同类路径共 1217 条）。它在第 40 天就把 p 值打到 0.01176，看起来证据确凿；可到 250 天终点，p 值回到 0.31714。同一条路径上的 $\log E_n$ 最高只有 1.5449，离拒绝阈值 $\log 20=2.9957$ 还差得远。这张图把问题的两面摆在一起：p 值路径会反复穿越阈值，而一个对停止时刻有效的证据过程不会。

## 二、把「任意时刻都有效」写成一条不等式

要让检验对任何停止规则都成立，就不能再依赖「n 固定」的抽样分布，而得构造一个在原假设下永远不系统性变大的证据量。令备择为 $r_i\sim N(\mu_k,\sigma^2)$，取有限个 $\mu_k$ 与先验权重 $w_k$，定义混合似然比

$$E_n=\sum_k w_k\exp\left(\frac{\mu_k S_n}{\sigma^2}-\frac{n\mu_k^2}{2\sigma^2}\right).$$

每一项都是似然比，在原假设下是非负鞅且期望恒为 1，有限混合仍是鞅。Ville 不等式随即给出 $P_{H_0}(\exists n: E_n\ge 1/\alpha)\le\alpha$。这个结论不依赖你什么时候看、看多少次，也不依赖你是因为「显著」才停还是因为「亏够了」才停。实现上用 $\tau=0.20$ 的正态先验在 $\pm4\tau$ 上取 41 个网格点离散化，用对数域的 log-sum-exp 累加，避免长路径下指数溢出；所有 3000 条路径的累计和一次性向量化，按 500 条分块控制内存。

图 2 给出最关键的对照。把 250 天里的检查次数 K 从 1 扫到 250，普通 p 值的误报率从 0.0467 单调爬到 0.4233；Bonferroni 把 $\alpha/K$ 分给每个检查点，名义上守住了，但越来越保守，K=250 时只剩 0.0047；e-process 是一条 0.0177 的水平线，因为它从一开始就在每个时刻检查，与 K 无关。注意 0.0177 明显低于 0.05：**Ville 给的是上界，不是等式**，这套检验在本文设定下相当保守。把它读成「e-process 更准」是错的，正确读法是「门槛更硬，代价在后面」。

![](/images/anytime-valid-backtest-e-process/fig2_type1_vs_looks.png)

把 K=25、K=50、K=250 三档单独摊开看更清楚：Bonferroni 的误报率依次是 0.0167、0.0103、0.0047，随检查次数增加几乎线性地交出功效；普通 p 值在同一批路径上分别报出 0.2547、0.3057、0.4233。两者用的是完全相同的随机数，差别只来自阈值是否承认「我看了很多次」。这也解释了 Bonferroni 在这类问题里口碑不好的原因：它并没有错，只是把一个连续累积的证据过程粗暴切成 K 个互不相干的检验，切得越碎，显著性预算摊得越薄，越难在真实效应出现时攒够证据。

## 三、有效性不是免费的：功效与检出延迟

代价要用同一批随机数公平比较。令真实日均超额收益 $\mu$ 从 0.05 扫到 0.30，每个档位 1500 条路径，四种方法共用相同的基础正态表，只做位置平移。以 $\mu=0.15$ 为例：固定终点检验功效 0.6920，每周偷看 0.8393，Bonferroni 0.2953，e-process 0.3880。

这几个数要小心解读。固定终点检验的 0.6920 是合法的，前提是研究员真的能忍住 250 天不看；偷看组的 0.8393 里混着上一节那 30% 的误报膨胀，不能算功效。在两个真正控制了 I 类错误的方法里，e-process 比 Bonferroni 高出约 31%：Bonferroni 把显著性预算一次性均摊给 50 个检查点，而鞅利用的是整条路径的累积证据，不需要为「我未来可能还会看」预留那么多。

![](/images/anytime-valid-backtest-e-process/fig3_power_and_delay.png)

代价体现在时间上。$\mu=0.15$ 时平均首次越过日：偷看组 83.1 天，e-process 147.1 天，Bonferroni 153.4 天；中位数分别是 65、151、160 天。图 4 的分布更直观：偷看组大量集中在前 100 天，那正是噪声最容易冒充信号的位置；有效检验把质量整体推到后半程。换句话说，anytime-valid 买的是「随时可以停且不算作弊」，付的是「要等更久才能下结论」。如果团队本来就能锁定一个预先注册的终点，固定检验在功效上反而更划算，没必要上 e-process。

![](/images/anytime-valid-backtest-e-process/fig4_first_hit_hist.png)

## 四、任何时候都有效，不等于模型错了也有效

Ville 不等式的证明只用了一件事：$E_n$ 在原假设的真实分布下是鞅。似然写错了，鞅就不是鞅。这一节专门把这个边界量化出来，也是本文最该被记住的部分。

保持研究员始终按 $\sigma=1$ 计算，只改数据的真实波动：真实 $\sigma=0.8$ 时 e-process 拒绝率 0.0017（过度保守），$\sigma=1.0$ 时 0.0203，$\sigma=1.3$ 时 **0.1143**，$\sigma=1.6$ 时 **0.2860**。方向很明确：把波动估小，等于把日常涨落当成超额收益的证据，$\exp(\mu S_n/\sigma^2)$ 里的分母被压小，证据量被指数级放大。作为对照，把收益改成自由度为 3 的标准化 t 分布（方差仍为 1），拒绝率 0.0257，只是轻微膨胀——重尾主要影响尾部路径，而方差匹配时鞅的一阶结构还在。

![](/images/anytime-valid-backtest-e-process/fig5_misspec_and_tau.png)

另一个不那么显眼但同样致命的旋钮是混合先验宽度 $\tau$。固定 $\mu=0.15$，把 $\tau$ 从 0.03 扫到 0.80：功效 0.0007 → 0.2233 → **0.3793** → 0.3647 → 0.3007，平均检出日 232.0 → 187.4 → 146.7 → 135.3 → 131.6。$\tau$ 太窄，先验里根本没有能覆盖真实效应量的 $\mu_k$，鞅几乎不会长大；$\tau$ 太宽，先验质量被摊到不可能出现的大效应上，早期增长变慢，功效反而掉下来，虽然一旦检出会稍早。这个 U 型（准确说是先升后缓降）说明 e-process 并非「无参数」：它把调参从「看几次、怎么校正」换成了「你预期效应量有多大」。实践中要么用历史策略给出 $\tau$，要么用多个 $\tau$ 的加权混合，别把默认值当真理。

## 五、落地建议与本文的边界

如果流程允许预注册，最省事的做法仍是写死回测长度与评估日期，一次检验定生死，功效最高、解释最简单。如果确实需要持续监控（例如线上策略每日巡检、多策略并行筛选、资金方随时可能叫停），就把 p 值换成 e-process：用混合似然比累积证据，越过 $1/\alpha$ 才报警，并把 $\sigma$ 的估计方式、混合先验、检查频率一并写进文档。本站上午那篇贝叶斯在线变点检测讨论的是「状态何时变了」，这里讨论的是「证据何时够了」，两者都服务实时决策，但一个是后验状态估计、一个是假设检验的误差控制，别混用同一套阈值语言。傍晚那篇 Thompson Sampling 后验浓度塌缩也提醒了同一件事：越学越窄的分布不一定是好事，e-process 拒绝后同样要考虑是否需要重新起一段新的证据过程。

必须说清局限：本文只做了独立同分布的高斯位置模型，$\sigma$ 已知、单策略、单条假设，没有处理序列相关（真实策略收益有自相关，鞅需要改成条件似然比）、没有处理跨策略的多重性（几十个策略同时监控，还要在策略维度上再控一次误差）、也没有覆盖 $\alpha$ 在多个研究之间的花费。上文的 0.0177 与 0.3880 是这套特定设定下的模拟值，不是普适常数。真实数据里波动率聚集、跳空与手续费都会让名义模型偏离，第四节的错配实验就是提醒：先验证模型，再谈 anytime-valid。

还有一个常被忽略的维度：上面所有模拟都只对一条假设做监控。如果研究池里同时跑着几十个策略，即使每条都用了 e-process，整体「至少有一个策略误报」的概率仍会随策略数累积。正确做法是在策略维度再叠一层误差控制，或者事先把显著性预算分配下去。anytime-valid 解决的是时间轴上的可选停止，不会顺带解决横截面上的多重比较；这两件事经常被混为一谈，也是把线上监控做成「天天都有信号」的常见原因。另一个实际选择是报警之后怎么办：证据过程越过阈值只说明原假设不再可信，并不自动给出新的效应量估计，若要继续跟踪，应当清空累计量、重新起一段鞅，而不是让同一条曲线永远停在拒绝区。

## 六、复现

下面这段就是生成全部数字与五张图的核心脚本，可直接运行（依赖 numpy、scipy、matplotlib），输出写在 `public/images/anytime-valid-backtest-e-process/results.json`。所有扫描共用固定随机种子与同一张随机数表，图 1 的示意路径是从 3000 条原假设模拟里按「偷看会误报、e-process 不会」筛出来的真实路径，不是手绘或乘系数伪造。文章中的每个数字都能在结果 JSON 里逐项对上。

```python
"""回测里的「偷看」问题：固定样本检验、Bonferroni 与 anytime-valid e-process。

运行方式（在项目根目录）：
    python3 scripts/article_core_eval_20261108night.py

产出：
    public/images/anytime-valid-backtest-e-process/fig1..fig5.png
    public/images/anytime-valid-backtest-e-process/results.json

核心问题：回测跑了 250 天，研究员每周看一次 p 值，只要某次 p<0.05 就宣布策略有效。
这种「可选停止」会把 I 类错误放大数倍。e-process 用混合似然比鞅 + Ville 不等式
给出对任意停止时刻都成立的 1/alpha 阈值，代价是功效与检出延迟。本文把所有数字
现场算出来，并且诚实地检验模型错配（方差设错、厚尾）时 anytime-valid 会不会失效。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, t as student_t

SEED = 20261108
OUT = "public/images/anytime-valid-backtest-e-process"
T = 250                 # 回测交易日
ALPHA = 0.05
SIGMA = 1.0             # 研究员假定日收益标准差
N_SIM_H0 = 3000         # 原假设下的独立回测次数
N_SIM_ALT = 1500        # 备择下的独立回测次数
TAU = 0.20              # 混合先验 N(0, tau^2) 的标准差
N_MU = 41               # 混合网格点数
PEEK_EVERYS = (250, 125, 50, 25, 10, 5, 1)
EFFECTS = (0.05, 0.10, 0.15, 0.20, 0.30)
TAUS = (0.03, 0.08, 0.20, 0.40, 0.80)
MIS_SIGMAS = (0.8, 1.0, 1.3, 1.6)


def mixing_grid(tau, n_mu=N_MU):
    """把 N(0,tau^2) 混合先验离散成有限混合（有限混合仍是精确的鞅）。"""
    mus = np.linspace(-4.0 * tau, 4.0 * tau, n_mu)
    w = np.exp(-0.5 * (mus / tau) ** 2)
    w /= w.sum()
    return mus, np.log(w + 1e-300)


def paths(seed, n_sim, mu=0.0, sigma_true=SIGMA, heavy=False):
    """生成 (n_sim, T) 条日超额收益路径；共同随机数通过 seed 保证。"""
    rng = np.random.default_rng(seed)
    if heavy:
        z = student_t.rvs(df=3, size=(n_sim, T), random_state=rng) / np.sqrt(3.0)
    else:
        z = rng.standard_normal((n_sim, T))
    return mu + sigma_true * z


def log_e_process(S, mus, logw, sigma=SIGMA):
    """log E_n = logsumexp_k [ log w_k + mu_k S_n/sigma^2 - n mu_k^2/(2 sigma^2) ].

    这是备择「r_i ~ N(mu_k, sigma^2)」对原假设「r_i ~ N(0, sigma^2)」的
    混合似然比；H0 下它是非负鞅且 E[E_n]=1，Ville 不等式给出
    P(存在 n 使 E_n >= 1/alpha) <= alpha。
    """
    n = np.arange(1, S.shape[1] + 1, dtype=float)
    coef = mus / sigma ** 2
    quad = mus ** 2 / (2.0 * sigma ** 2)
    out = np.empty_like(S, dtype=float)
    for i in range(0, S.shape[0], 500):          # 分块控制内存
        term = (logw[None, None, :]
                + coef[None, None, :] * S[i:i + 500, :, None]
                - quad[None, None, :] * n[None, :, None])
        out[i:i + 500] = logsumexp(term, axis=2)
    return out


def decisions(r, tau=TAU, alpha=ALPHA, sigma=SIGMA):
    """对一批路径给出四种检验的拒绝判定与首次越过时刻。"""
    S = np.cumsum(r, axis=1)
    n = np.arange(1, r.shape[1] + 1, dtype=float)
    Z = S / (sigma * np.sqrt(n))
    zcrit = norm.ppf(1.0 - alpha / 2.0)
    mus, logw = mixing_grid(tau)
    logE = log_e_process(S, mus, logw, sigma=sigma)
    thr = np.log(1.0 / alpha)

    fixed = np.abs(Z[:, -1]) > zcrit
    # 偷看 / Bonferroni：按给定网格检查
    res = {"fixed": fixed}
    for every in PEEK_EVERYS:
        grid = np.arange(every, T + 1, every)
        zz = np.abs(Z[:, grid - 1])
        res[f"peek_{every}"] = (zz > zcrit).any(axis=1)
        zb = norm.ppf(1.0 - alpha / (2.0 * len(grid)))
        res[f"bonf_{every}"] = (zz > zb).any(axis=1)
    # e-process：每个时刻都检查（Ville 不等式覆盖任意停止时刻）
    e_hit = logE >= thr
    res["eproc"] = e_hit.any(axis=1)
    first = np.argmax(e_hit, axis=1) + 1
    res["eproc_first"] = np.where(res["eproc"], first, T + 1)
    res["logE"] = logE
    res["Z"] = Z
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
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

    R = {}
    zcrit = float(norm.ppf(1.0 - ALPHA / 2.0))
    R["settings"] = {"T": T, "alpha": ALPHA, "sigma_assumed": SIGMA,
                     "n_sim_h0": N_SIM_H0, "n_sim_alt": N_SIM_ALT,
                     "tau": TAU, "n_mu": N_MU, "zcrit": zcrit,
                     "seed": SEED}

    # ---------- 1) 原假设下的 I 类错误，以及偷看频次扫描 ----------
    r0 = paths(SEED, N_SIM_H0)
    d0 = decisions(r0)
    R["h0_rejection"] = {
        "fixed": float(d0["fixed"].mean()),
        "peek_weekly": float(d0["peek_5"].mean()),
        "peek_daily": float(d0["peek_1"].mean()),
        "bonferroni_weekly": float(d0["bonf_5"].mean()),
        "eproc": float(d0["eproc"].mean()),
    }
    R["peek_scan"] = []
    for every in PEEK_EVERYS:
        R["peek_scan"].append({
            "every": every,
            "n_looks": int(T // every),
            "peek": float(d0[f"peek_{every}"].mean()),
            "bonferroni": float(d0[f"bonf_{every}"].mean()),
            "eproc": float(d0["eproc"].mean()),
        })

    # ---------- 2) 示意路径：从上面这批 H0 模拟里挑一条「偷看误报、e-process 不误报」----------
    p0 = 2.0 * norm.sf(np.abs(d0["Z"]))
    cand = np.flatnonzero((p0 < ALPHA).any(axis=1) & ~d0["eproc"])
    idx = int(cand[0])
    demo_r = r0[idx]
    demo_Z = d0["Z"][idx]
    demo_p = p0[idx]
    demo_logE = d0["logE"][idx]
    R["demo"] = {"path_index": idx,
                 "n_candidates": int(len(cand)),
                 "max_abs_z": float(np.max(np.abs(demo_Z))),
                 "min_p": float(np.min(demo_p)),
                 "first_peek_day": int(np.flatnonzero(demo_p < ALPHA)[0] + 1),
                 "max_logE": float(np.max(demo_logE)),
                 "logE_threshold": float(np.log(1 / ALPHA)),
                 "final_p": float(demo_p[-1]),
                 "final_logE": float(demo_logE[-1])}

    # ---------- 3) 功效与检出时间 ----------
    R["power"] = []
    first_times = {}
    for mu in EFFECTS:
        r1 = paths(SEED + 1000, N_SIM_ALT, mu=mu)
        d1 = decisions(r1)
        grid = np.arange(5, T + 1, 5)
        Zg = np.abs(d1["Z"][:, grid - 1])
        zb = norm.ppf(1.0 - ALPHA / (2.0 * len(grid)))
        peek_first = np.where((Zg > zcrit).any(axis=1),
                              grid[np.argmax(Zg > zcrit, axis=1)], T + 1)
        bonf_first = np.where((Zg > zb).any(axis=1),
                              grid[np.argmax(Zg > zb, axis=1)], T + 1)
        row = {
            "mu": mu,
            "fixed": float(d1["fixed"].mean()),
            "peek": float(d1["peek_5"].mean()),
            "bonferroni": float(d1["bonf_5"].mean()),
            "eproc": float(d1["eproc"].mean()),
            "eproc_mean_hit": float(d1["eproc_first"][d1["eproc"]].mean())
            if d1["eproc"].any() else None,
            "bonf_mean_hit": float(bonf_first[bonf_first <= T].mean())
            if (bonf_first <= T).any() else None,
            "peek_mean_hit": float(peek_first[peek_first <= T].mean())
            if (peek_first <= T).any() else None,
        }
        R["power"].append(row)
        if abs(mu - 0.15) < 1e-9:
            first_times = {
                "eproc": d1["eproc_first"][d1["eproc"]].tolist(),
                "bonferroni": bonf_first[bonf_first <= T].tolist(),
                "peek": peek_first[peek_first <= T].tolist(),
            }
    R["first_hit_mu015"] = {
        "eproc_median": float(np.median(first_times["eproc"])),
        "bonferroni_median": float(np.median(first_times["bonferroni"])),
        "peek_median": float(np.median(first_times["peek"])),
        "eproc_n": len(first_times["eproc"]),
        "bonferroni_n": len(first_times["bonferroni"]),
        "peek_n": len(first_times["peek"]),
    }

    # ---------- 4) 模型错配：真实方差与厚尾 ----------
    R["misspec"] = []
    for s in MIS_SIGMAS:
        rm = paths(SEED + 2000, N_SIM_H0, sigma_true=s)
        dm = decisions(rm)
        R["misspec"].append({
            "label": f"sigma={s:.1f}", "true_sigma": s,
            "peek": float(dm["peek_5"].mean()),
            "eproc": float(dm["eproc"].mean()),
            "bonferroni": float(dm["bonf_5"].mean()),
        })
    rh = paths(SEED + 3000, N_SIM_H0, heavy=True)
    dh = decisions(rh)
    R["misspec"].append({
        "label": "t(3)/sqrt(3)", "true_sigma": 1.0,
        "peek": float(dh["peek_5"].mean()),
        "eproc": float(dh["eproc"].mean()),
        "bonferroni": float(dh["bonf_5"].mean()),
    })

    # ---------- 5) 混合先验宽度 tau 的权衡 ----------
    r_tau = paths(SEED + 4000, N_SIM_ALT, mu=0.15)
    R["tau_scan"] = []
    for tau in TAUS:
        dt = decisions(r_tau, tau=tau)
        hit = dt["eproc_first"][dt["eproc"]]
        R["tau_scan"].append({
            "tau": tau,
            "power": float(dt["eproc"].mean()),
            "mean_hit": float(hit.mean()) if len(hit) else None,
        })

    with open(f"{OUT}/results.json", "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)

    # ---------- 图 1：单条 H0 路径的 p 值与 e-process ----------
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(np.arange(1, T + 1), demo_p, lw=1.6, color="#2b6cb0")
    ax[0].axhline(ALPHA, ls="--", color="#c53030", label="alpha=0.05")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("双侧 p 值（对数轴）")
    ax[0].set_title("原假设下的一条真实路径：p 值穿越 0.05，e-process 未越过阈值")
    ax[0].legend()
    ax[1].plot(np.arange(1, T + 1), demo_logE, lw=1.6, color="#2f855a")
    ax[1].axhline(np.log(1 / ALPHA), ls="--", color="#c53030",
                  label="log(1/alpha)")
    ax[1].set_ylabel("log E_n")
    ax[1].set_xlabel("回测交易日 n")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_peeking_paths.png")
    plt.close(fig)

    # ---------- 图 2：I 类错误随偷看次数 ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    k = [x["n_looks"] for x in R["peek_scan"]]
    ax.plot(k, [x["peek"] for x in R["peek_scan"]], "o-", color="#c53030",
            label="反复偷看（固定 0.05 阈值）")
    ax.plot(k, [x["bonferroni"] for x in R["peek_scan"]], "s-", color="#dd6b20",
            label="Bonferroni 校正")
    ax.plot(k, [x["eproc"] for x in R["peek_scan"]], "^-", color="#2f855a",
            label="e-process（任意时刻检验）")
    ax.axhline(ALPHA, ls="--", color="0.4", label="名义水平 0.05")
    ax.set_xscale("log")
    ax.set_xlabel("250 天内的检查次数 K（对数轴）")
    ax.set_ylabel("H0 下至少拒绝一次的比例")
    ax.set_title("偷看越多，普通 p 值越不可信；Ville 阈值与 K 无关")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_type1_vs_looks.png")
    plt.close(fig)

    # ---------- 图 3：功效曲线与检出延迟 ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    mus = [x["mu"] for x in R["power"]]
    for key, label, c in [("fixed", "固定终点检验", "#718096"),
                          ("peek", "每周偷看", "#c53030"),
                          ("bonferroni", "Bonferroni", "#dd6b20"),
                          ("eproc", "e-process", "#2f855a")]:
        ax[0].plot(mus, [x[key] for x in R["power"]], "o-", color=c, label=label)
    ax[0].set_xlabel("真实日均超额收益 mu")
    ax[0].set_ylabel("检出比例（功效）")
    ax[0].set_title("功效：偷看组虚高，有效方法要付代价")
    ax[0].legend(fontsize=8)
    for key, label, c in [("eproc_mean_hit", "e-process", "#2f855a"),
                          ("bonf_mean_hit", "Bonferroni", "#dd6b20"),
                          ("peek_mean_hit", "每周偷看", "#c53030")]:
        vals = [x[key] if x[key] is not None else np.nan for x in R["power"]]
        ax[1].plot(mus, vals, "s-", color=c, label=label)
    ax[1].set_xlabel("真实日均超额收益 mu")
    ax[1].set_ylabel("平均首次越过的交易日")
    ax[1].set_title("检出延迟：有效性不是免费的")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_power_and_delay.png")
    plt.close(fig)

    # ---------- 图 4：mu=0.15 时首次检出时间分布 ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(0, T + 26, 25)
    for key, label, c in [("peek", "每周偷看", "#c53030"),
                          ("bonferroni", "Bonferroni", "#dd6b20"),
                          ("eproc", "e-process", "#2f855a")]:
        ax.hist(first_times[key], bins=bins, alpha=0.55, label=label, color=c)
    ax.axvline(R["first_hit_mu015"]["eproc_median"], ls="--", color="#2f855a")
    ax.set_xlabel("首次越过的交易日")
    ax.set_ylabel("路径条数")
    ax.set_title("mu=0.15：有效检验把检出时间整体推后")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_first_hit_hist.png")
    plt.close(fig)

    # ---------- 图 5：模型错配与 tau 权衡 ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    labels = [x["label"] for x in R["misspec"]]
    xp = np.arange(len(labels))
    ax[0].bar(xp - 0.2, [x["peek"] for x in R["misspec"]], 0.4,
              color="#c53030", label="每周偷看")
    ax[0].bar(xp + 0.2, [x["eproc"] for x in R["misspec"]], 0.4,
              color="#2f855a", label="e-process")
    ax[0].axhline(ALPHA, ls="--", color="0.4")
    ax[0].set_xticks(xp)
    ax[0].set_xticklabels(labels, rotation=20, fontsize=8)
    ax[0].set_ylabel("H0 下拒绝比例")
    ax[0].set_title("真实波动被设错 / 厚尾时，Ville 保证也失效")
    ax[0].legend(fontsize=8)
    ts = [x["tau"] for x in R["tau_scan"]]
    ax[1].plot(ts, [x["power"] for x in R["tau_scan"]], "o-", color="#2b6cb0",
               label="功效（mu=0.15）")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("混合先验宽度 tau（对数轴）")
    ax[1].set_ylabel("检出比例", color="#2b6cb0")
    ax2 = ax[1].twinx()
    ax2.plot(ts, [x["mean_hit"] if x["mean_hit"] else np.nan
                  for x in R["tau_scan"]], "s--", color="#805ad5",
             label="平均检出日")
    ax2.set_ylabel("平均首次越过日", color="#805ad5")
    ax2.grid(False)
    ax[1].set_title("先验宽度要匹配效应量：过宽过窄都亏")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_misspec_and_tau.png")
    plt.close(fig)

    print("=== H0 拒绝率 (alpha=0.05) ===")
    for k, v in R["h0_rejection"].items():
        print(f"  {k:20s} {v:.4f}")
    print("=== 偷看次数扫描 ===")
    for row in R["peek_scan"]:
        print(f"  K={row['n_looks']:3d}  peek={row['peek']:.4f}  "
              f"bonf={row['bonferroni']:.4f}  eproc={row['eproc']:.4f}")
    print("=== 功效 (mu=0.15) ===")
    p = [x for x in R["power"] if abs(x["mu"] - 0.15) < 1e-9][0]
    for k in ("fixed", "peek", "bonferroni", "eproc",
              "eproc_mean_hit", "bonf_mean_hit", "peek_mean_hit"):
        print(f"  {k:20s} {p[k]:.4f}")
    print("=== 模型错配下 e-process 拒绝率 ===")
    for row in R["misspec"]:
        print(f"  {row['label']:16s} eproc={row['eproc']:.4f}  peek={row['peek']:.4f}")
    print("=== tau 扫描 ===")
    for row in R["tau_scan"]:
        print(f"  tau={row['tau']:.2f}  power={row['power']:.4f}  "
              f"mean_hit={row['mean_hit']}")
    print(f"demo: path#{R['demo']['path_index']}  "
          f"first_peek_day={R['demo']['first_peek_day']}  "
          f"min_p={R['demo']['min_p']:.5f}  "
          f"max_logE={R['demo']['max_logE']:.4f} "
          f"(threshold {R['demo']['logE_threshold']:.4f})")


if __name__ == "__main__":
    main()
```
