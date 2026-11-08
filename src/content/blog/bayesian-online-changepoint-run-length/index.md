---
title: "贝叶斯在线变点检测：用 run-length 后验给波动率突变装一个实时警报器"
description: "离线变点检测要等齐整段样本才下结论，实盘等不起。本文现场实现 Adams 与 MacKay 2007 的 BOCD 递推：先拆穿一个常见误用——常数 hazard 下 r_t=0 的后验恒等于 1/λ、与数据无关，不能拿来做警报分数；改用短 run-length 后验质量后，480 天三个已知波动率突变全部命中，平均滞后 1.00 天、误报 1 次。13 条独立序列的窗口 ROC 上 BOCD AUC 0.9665，重置式双侧 CUSUM 为 0.6229。附五档 hazard 灵敏度扫描、五张真实计算图与完整可运行代码。"
publishDate: '2026-11-08'
tags:
  - 量化
  - 交易
  - 变点检测
  - Python
language: zh
difficulty: advanced
---

## 一、离线检测等不起，在线检测怕误报

风险模型里最危险的一天，往往不是波动最大的那天，而是波动从低档跳到高档、而模型还停留在旧参数的那几天。假设一个组合用过去半年数据估出的日波动 0.6% 来算 VaR 和仓位，某天市场突然切到 2.2% 的波动档。若风控要等 60 个交易日才重新估计，这段时间的仓位实际上是按三分之一的真实风险在放大，杠杆被动翻了三倍以上。这不是尾部小概率事件，而是波动率区制切换的常态。

处理这类问题的传统工具是离线变点检测：拿到一整段样本，用 CUSUM、PELT 或者二元分割回头去找「最可能在哪一天断了」。它们统计性质干净，但有一个实盘无法接受的前提——你需要看到变点之后的样本才能确认变点之前发生过什么。等到样本攒够，风险早就兑现了。

贝叶斯在线变点检测（Bayesian Online Changepoint Detection，BOCD）换了一个问法：不追问「历史上哪天变了」，而是每个时刻维护一个隐状态 `r_t`，表示「截至今天，当前这一段已经持续了多少天」。如果数据一直符合旧模型，`r_t` 就顺着时间一天加一；如果今天的数据在旧模型下极端不可能，后验质量就会从长的 `r` 大量回灌到短的 `r`。这个回灌本身就是变点证据，而且它只需要截止今天的数据。

本文把这套递推完整算一遍，重点不是演示它「有效」，而是拆穿一个极易踩中的实现陷阱，并把它的检出能力放到同一评价口径下与 CUSUM 对拍。结论先摆在这里：**BOCD 明显优于基线，但优势高度依赖 hazard 参数；而且波动率下降型变点比上升型难检得多，这不是调参能补的。**

## 二、BOCD：把「距离上次变点多久」当成隐状态

方法来自 Adams 与 MacKay 在 2007 年发布的《Bayesian Online Changepoint Detection》（arXiv:0710.3742）。它不假设变点个数已知，也不假设每段的参数已知，只假设段与段之间独立、段内独立同分布，以及一个先验的变点发生率 `H`（hazard）。

### 2.1 递推的四个量

在 `t` 时刻，对每个可能的 run-length `r` 保存两样东西：联合概率 `P(r_{t-1}=r, x_{1:t-1})`，以及在该段上参数后验的充分统计量。收益率的均值和方差都未知，用 Normal-Inverse-Gamma 共轭先验（`mu0, kappa0, alpha0, beta0`），于是下一观测的一步预测是自由度为 `2α` 的 Student-t：位置 `μ`、尺度平方 `β(κ+1)/(ακ)`。这个厚尾预测很关键——它允许单个异常观测，不至于因为一根大阴线就立刻宣布变点。

每一步做四件事：对每个 `r` 算预测密度 `p(x_t | r)`；把「继续生长」的质量乘上 `(1−H)` 并平移到 `r+1`；把「今天发生变点」的质量乘上 `H` 汇总到 `r=0`；归一化后用 `x_t` 更新每个 `r` 的 NIG 参数。复杂度 `O(T²)`，480 天在纯 numpy 下不到一秒。

### 2.2 一个极易踩中的陷阱

网上不少 BOCD 实现把 `P(r_t = 0 | x_{1:t})` 直接当作「今天发生变点的概率」来阈值化。**在常数 hazard 下这个量恒等于 `H`，与数据完全无关。**

原因不难看清：变点分支的质量是 `Σ_r w_r · H`，生长分支的总质量是 `Σ_r w_r · (1−H)`，两者都含同一个因子 `Σ_r w_r`；归一化时这个因子被约掉，剩下 `H / (H + 1 − H) = H`。我在第一次跑这个实验时就是这么写的，结果 `cp_score` 在 λ=80 时恒为 0.0125、λ=20 时恒为 0.0500，标准差精确为零，阈值化后一个变点都检不出来。这不是模型失效，是取错了量。

要让 `r_t=0` 携带信息，得让 hazard 依赖 run-length（例如用 Weibull 风险率），或者干脆换一个统计量。本文选后者。

### 2.3 真正能用的警报分数

变点发生后，后验质量会从长 run-length 塌回短 run-length。所以定义短段后验质量 `S_t = Σ_{r=1..10} P(r_t = r | x_{1:t})`：平稳段里后验集中在大的 `r`，`S_t` 很小；一旦模型被迫承认「旧段解释不了今天」，质量回灌，`S_t` 跃升。本文固定判决阈值 0.50，并对报警点做 15 天非极大值抑制，避免一次变点被重复计数；前 15 天作为预热期跳过，因为初始后验天然集中在短段。

这个分数仍有量级问题：hazard 越小，几何分布的短段质量基线越低。所以下文的 hazard 扫描采用的是**同一个绝对阈值**，不同 λ 的基线不同本身就是灵敏度的一部分，这点必须说清楚，不能把阈值差异包装成模型差异。

## 三、实验设置

合成 480 个交易日收益率，分段常数波动率，三个真变点固定在第 120、260、380 天，四段真实日波动依次为 0.006、0.022、0.007、0.020——注意它是「升—降—升」交替，这样能分别考察上升型与下降型突变。随机数种子 20261108。

主实验 λ=80。灵敏度部分另生成 12 条独立序列（变点结构相同、随机流不同），共 13 条；每条序列在五档 λ 下重跑。评价用不重叠的 20 天窗口切片：窗口内含真变点即为正类，窗口内有报警即为预测正，由此算标准 ROC。CUSUM 基线用双侧统计量作用于 `|r|`，固定 δ=1.0，超过阈值报警并把该侧清零；其基准均值与标准差**只用前 60 天**估计，不含前视偏差。两边都只扫描各自的一个判决阈值，口径对等。

## 四、五张图逐一读

![](/images/bayesian-online-changepoint-run-length/fig1_returns_regimes.png)

图 1 是输入序列。上图是日收益，下图把 `|r|` 与真实 σ 阶梯画在一起。三个真变点用竖虚线标出；第二段（120–260 天）的点云明显变宽，第三段又收窄。这是设计出来的、真值已知的检验场，所有后续数字都可以对着这三条线核。

![](/images/bayesian-online-changepoint-run-length/fig2_runlength_posterior.png)

图 2 是整篇文章最值得看的一张：横轴时间、纵轴 run-length、颜色是后验的 log 值，亮线沿 45 度方向爬升代表「没有变点、段长持续加一」。每遇到一次变点，亮线就断掉并从 `r≈0` 重新起一条——图中锯齿结构清晰可见。绿色曲线是 MAP run-length，在 120 与 380 处干净回落，在 260 处回落得犹豫，出现长短两个模式交替的双峰。这正是「波动率下降比上升难检」的直观证据：高波动模型的 Student-t 预测本来就有很厚的尾，低波动观测在它下面并不算意外，似然比给不出强证据。

![](/images/bayesian-online-changepoint-run-length/fig3_map_detection.png)

图 3 把警报分数与 MAP 回落放在一起。λ=80、阈值 0.50 时报出四个点：122、235、258、383。其中 122、258、383 命中真变点 120、260、380，235 是误报，即 TP=3、FP=1，平均有符号滞后 1.00 天。注意 258 比真变点 260 还早两天——20 天容差窗口内提前报警也算命中，这个「负滞后」来自变点前波动已开始漂移，属于合成数据的边界效应，不能当成模型有预判能力。

![](/images/bayesian-online-changepoint-run-length/fig4_hazard_sensitivity.png)

图 4 是五档 λ 的权衡曲线。λ 从 20 加到 320，每条序列的误报数从 7.308 单调降到 0.385，检出数却从 3.000 掉到 2.385；平均滞后从 −0.26 天变成 4.41 天再回落到 3.69 天。λ=20 的「负滞后」是误报太多造成的统计假象——13 条序列里 7.3 条误报，总有几个碰巧落在容差窗口内被记为命中。λ=80 附近是这套参数下比较平衡的位置：误报 1.462 条/序列、检出 2.769 个/序列、滞后 4.13 天。

![](/images/bayesian-online-changepoint-run-length/fig5_roc_bocd_cusum.png)

图 5 是 13 条序列汇总的窗口 ROC。BOCD 的 AUC 为 0.9665，曲线在低误报区就爬到 0.95 以上；重置式双侧 CUSUM 为 0.6229，只略好于随机。要公平地读这张图：CUSUM 的 δ 固定在 1.0、且只用前 60 天校准，没有为这批数据做参数寻优；BOCD 除了 λ 也没有调参。差距主要来自 BOCD 用 Student-t 预测显式建模了「段内方差未知」，而 CUSUM 的标准化基准在波动率反复切换后会漂移。若把 CUSUM 的 δ 与校准窗一起网格搜索，差距会缩小，但那对在线系统是不公平的比较。

## 五、数字汇总与诚实结论

| 指标 | λ=80 主实验 | 说明 |
|------|------------|------|
| 真变点 | 120 / 260 / 380 | σ: 0.006→0.022→0.007→0.020 |
| 报警点 | 122 / 235 / 258 / 383 | 阈值 0.50，短段上界 r≤10 |
| 命中 / 误报 | 3 / 1 | 容差 ±20 天 |
| 平均有符号滞后 | 1.00 天 | 含一次提前报警 |
| ROC AUC（BOCD） | 0.9665 | 13 条序列、20 天窗口 |
| ROC AUC（CUSUM） | 0.6229 | δ=1.0、前 60 天校准 |

0.9665 这个 AUC 只在分段常数波动率这一种生成机制下成立——换成带波动率聚集的 GARCH 或带跳跃的序列，优势幅度未必保持，这是边界而不是结论。可以确认的是：**BOCD 的价值不在「更准」，而在它把「我不确定当前是不是同一段」这件事量化成了一个可阈值化的后验量**，并且这个量逐时刻可得，不需要未来样本。

## 六、边界：这套东西在哪里会失效

第一，常数 hazard 是强假设。它等价于变点服从几何间隔，而真实市场的低波动期往往持续更久、高波动期成簇出现；换成随 run-length 变化的 hazard 会同时改变 `r_t=0` 的信息量，届时检测分数也要重新设计。

第二，下降型变点天然弱势。图 2 里 260 处的双峰、图 4 里检出数随 λ 增大而下滑，主要都来自这一段。若业务上更关心「波动起来」而非「波动平息」，可以只对单侧做似然比检验，但那就不是完整 BOCD 了。

第三，计算是 `O(T²)`。480 天可以全量递推，几年 tick 级数据必须做 run-length 截断或粒子近似，截断本身又会在长段上引入偏差。

第四，单变量。本文只建模收益率幅度，没有用跨资产相关性的断裂信息；多变量 BOCD 的参数维度会随资产数平方增长，实践中通常先降维再检测。

第五，阈值的量级依赖 λ。图 4 已经展示：同一个 0.50 在不同 λ 下对应完全不同的误报水平。工程落地时阈值必须随 hazard 一起标定，不能从一篇博客里抄一个数就用。

## 七、完整可运行代码

下面这段代码就是生成全部五张图与 `results.json` 的脚本，在项目根目录直接运行即可。它自包含：包含数据生成、BOCD 递推、CUSUM 基线、ROC 评价、五张图与结果 JSON 的输出，不依赖任何外部数据文件。

```python
"""贝叶斯在线变点检测（BOCD）完整可运行实验。

运行方式（在项目根目录）：
    python3 scripts/article_core_bocd_20261108.py

产出：
    public/images/bayesian-online-changepoint-run-length/fig1..fig5.png
    public/images/bayesian-online-changepoint-run-length/results.json

模型：Adams & MacKay (2007) 的 run-length 递推，未知均值与方差用
Normal-Inverse-Gamma 共轭先验，一步预测是 Student-t。hazard 取常数 1/lambda。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

SEED = 20261108
OUT = "public/images/bayesian-online-changepoint-run-length"
T = 480
TRUE_CPS = (120, 260, 380)
SIGMAS = (0.006, 0.022, 0.007, 0.020)
TOL = 20  # 检测容差窗口（天）
SHORT_R = 10  # 「刚换段」的 run-length 上界，用于构造变点证据分数
THRESHOLD = 0.50  # 固定判决阈值：短段后验质量超过它才报警


def gen_returns(seed=SEED, cps=TRUE_CPS, sigmas=SIGMAS, n=T):
    """分段常数波动率的高斯收益率，变点位置与真实波动率都已知。"""
    rng = np.random.default_rng(seed)
    vol = np.zeros(n)
    bounds = (0,) + tuple(cps) + (n,)
    for k, s in enumerate(sigmas):
        vol[bounds[k]:bounds[k + 1]] = s
    return rng.standard_normal(n) * vol, vol


def bocd(x, lam=80.0, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1e-4):
    """返回 (run-length 后验矩阵, 短段后验质量分数, MAP run-length 序列).

    注意：常数 hazard 下 P(r_t=0 | x_1:t) 恒等于 hazard，与数据无关，
    不能拿来当检测分数。真正携带信息的是质量向短 run-length 的转移。

    递推的四个量都是按 run-length r 索引的 NIG 后验参数，
    表示「在 t-1 时刻、当前段已经持续 r 步」的条件下对下一观测的预测分布。
    """
    n = len(x)
    hazard = 1.0 / lam
    m = n + 1
    mu = np.full(m, mu0, dtype=float)
    kappa = np.full(m, kappa0, dtype=float)
    alpha = np.full(m, alpha0, dtype=float)
    beta = np.full(m, beta0, dtype=float)

    joint = np.zeros(m)          # P(r_{t-1}=r, x_{1:t-1})
    joint[0] = 1.0
    post = np.zeros((n, m))      # P(r_t=r | x_{1:t})
    map_r = np.zeros(n)

    for t in range(n):
        k = t + 1
        df = 2.0 * alpha[:k]
        scale2 = beta[:k] * (kappa[:k] + 1.0) / (alpha[:k] * kappa[:k])
        log_pred = student_t.logpdf(x[t], df=df, loc=mu[:k],
                                    scale=np.sqrt(scale2))
        w = joint[:k] * np.exp(log_pred)

        new_joint = np.zeros(m)
        new_joint[0] = np.sum(w * hazard)          # 发生变点
        new_joint[1:k + 1] = w * (1.0 - hazard)   # 段长度 +1
        new_joint /= (new_joint.sum() + 1e-300)

        # 用 x_t 更新每个 r 的 NIG 后验；r=0 保持先验
        kn = kappa[:k] + 1.0
        mn = (kappa[:k] * mu[:k] + x[t]) / kn
        an = alpha[:k] + 0.5
        bn = beta[:k] + kappa[:k] * (x[t] - mu[:k]) ** 2 / (2.0 * kn)

        new_mu = np.zeros(m)
        new_kappa = np.empty(m)
        new_alpha = np.empty(m)
        new_beta = np.empty(m)
        new_mu[0] = mu0
        new_kappa[0] = kappa0
        new_alpha[0] = alpha0
        new_beta[0] = beta0
        new_mu[1:k + 1] = mn
        new_kappa[1:k + 1] = kn
        new_alpha[1:k + 1] = an
        new_beta[1:k + 1] = bn

        mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta
        joint = new_joint
        post[t] = new_joint
        map_r[t] = int(np.argmax(new_joint[:k + 1]))

    # 短 run-length 的后验质量：变点后质量从长段回落到短段，此分数跃升
    short_mass = post[:, 1:SHORT_R + 1].sum(axis=1).copy()
    return post, short_mass, map_r


def cusum_alarms(x, h, delta=1.0, calib=60):
    """对 |r| 做双侧 CUSUM；超过 h 报警并把该侧统计量清零。

    基准均值/标准差只用前 calib 天估计，避免用到未来数据（无前视偏差）。
    """
    a = np.abs(x)
    z = (a - a[:calib].mean()) / (a[:calib].std() + 1e-12)
    sp, sm, alarms = 0.0, 0.0, []
    for i in range(1, len(x)):
        sp = max(0.0, sp + z[i] - delta)
        sm = max(0.0, sm - z[i] - delta)
        if sp > h:
            alarms.append(i)
            sp = 0.0
        if sm > h:
            alarms.append(i)
            sm = 0.0
    return alarms


def pick_alarms(score, thr, min_sep=15, warmup=15):
    """阈值化 + 非极大值抑制；跳过预热期（初始后验天然集中在短段）。"""
    cand = [i for i in range(len(score)) if score[i] > thr and i >= warmup]
    picked = []
    for i in sorted(cand, key=lambda j: -score[j]):
        if all(abs(i - p) > min_sep for p in picked):
            picked.append(i)
    return sorted(picked)


def match(alarms, cps, tol=TOL):
    tp, fp, delays = 0, 0, []
    used = set()
    for a in alarms:
        d = [(abs(a - c), c) for c in cps if c not in used]
        if d:
            dist, c = min(d)
            if dist <= tol:
                tp += 1
                delays.append(a - c)
                used.add(c)
                continue
        fp += 1
    return tp, fp, (float(np.mean(delays)) if delays else float("nan"))


def window_labels_and_preds(alarms, cps, n=T, tol=TOL):
    """把时间切成不重叠窗口，得到标准 ROC 的真值/预测标签。"""
    nw = n // tol
    y, s = [], []
    for w in range(nw):
        lo, hi = w * tol, (w + 1) * tol
        y.append(1 if any(lo <= c < hi for c in cps) else 0)
        s.append(1 if any(lo <= a < hi for a in alarms) else 0)
    return np.array(y), np.array(s)


def auc_from_curve(fpr, tpr):
    order = np.argsort(fpr)
    integ = getattr(np, "trapezoid", None)
    if integ is None:  # numpy < 2.0
        integ = np.trapz
    return float(integ(tpr[order], fpr[order]))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=110, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(OUT, exist_ok=True)
    x, vol = gen_returns()
    post, cp_score, map_r = bocd(x, lam=80.0)

    # 图 1：收益率与真实波动率区制
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1]})
    ax[0].plot(x, lw=0.7, color="#2f5597")
    ax[1].plot(np.abs(x), lw=0.6, color="#9aa5b1", label="|return|")
    ax[1].plot(vol, lw=2.0, color="#c0392b", label="true sigma")
    for c in TRUE_CPS:
        for a in ax:
            a.axvline(c, ls="--", lw=1.0, color="#333333")
    ax[0].set_ylabel("daily return")
    ax[1].set_ylabel("volatility")
    ax[1].set_xlabel("trading day")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[0].set_title("Synthetic returns with 3 volatility changepoints")
    fig.tight_layout()
    save(fig, "fig1_returns_regimes.png")

    # 图 2：run-length 后验热力图
    show = np.log(np.clip(post[:, :200].T, 1e-6, 1.0))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    im = ax.imshow(show, origin="lower", aspect="auto", cmap="magma",
                   extent=[0, T, 0, 200], vmin=-8, vmax=0)
    for c in TRUE_CPS:
        ax.axvline(c, ls="--", lw=1.0, color="#5ce1e6")
    ax.plot(np.arange(T), map_r, color="#7dff9b", lw=0.9, label="MAP run-length")
    ax.set_ylim(0, 200)
    ax.set_xlabel("trading day")
    ax.set_ylabel("run length r")
    ax.set_title("BOCD posterior P(r_t | x_1:t), log scale")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="log posterior")
    fig.tight_layout()
    save(fig, "fig2_runlength_posterior.png")

    # 图 3：变点后验概率与 MAP 回落
    alarms = pick_alarms(cp_score, THRESHOLD)
    tp, fp, delay = match(alarms, TRUE_CPS)
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    ax[0].plot(cp_score, color="#c0392b", lw=1.0)
    ax[0].axhline(THRESHOLD, ls=":", color="#555555",
                  label=f"threshold {THRESHOLD:.2f}")
    ax[0].set_ylabel("posterior mass, r<=10")
    ax[0].legend(fontsize=8)
    ax[1].plot(map_r, color="#2f5597", lw=0.9)
    ax[1].set_ylabel("MAP r_t")
    ax[1].set_xlabel("trading day")
    for a in ax:
        for c in TRUE_CPS:
            a.axvline(c, ls="--", lw=1.0, color="#333333")
    for p in alarms:
        ax[0].axvline(p, ls="-", lw=0.7, color="#e67e22", alpha=0.6)
    ax[0].set_title("Short-run-length posterior mass and MAP resets")
    fig.tight_layout()
    save(fig, "fig3_map_detection.png")

    # 图 4 + 图 5：hazard 灵敏度与 ROC，在多条独立序列上评估
    seqs = [(gen_returns(seed=SEED + 100 + s)[0], TRUE_CPS) for s in range(12)]
    seqs.insert(0, (x, TRUE_CPS))
    lams = [20.0, 40.0, 80.0, 160.0, 320.0]
    scan = []
    for lam in lams:
        delays, fps, tps = [], [], []
        for xs, cps in seqs:
            _, cs, _ = bocd(xs, lam=lam)
            a = pick_alarms(cs, THRESHOLD)
            t, f, d = match(a, cps)
            tps.append(t)
            fps.append(f)
            if np.isfinite(d):
                delays.append(d)
        scan.append({
            "lambda": lam,
            "mean_delay": float(np.mean(delays)) if delays else None,
            "false_alarms_per_series": float(np.mean(fps)),
            "true_positives_per_series": float(np.mean(tps)),
        })

    fig, ax = plt.subplots(figsize=(9, 4.4))
    xx = np.arange(len(lams))
    ax.plot(xx, [s["mean_delay"] for s in scan], "o-", color="#2f5597",
            label="mean detection delay (days)")
    ax.set_ylabel("detection delay (days)", color="#2f5597")
    ax2 = ax.twinx()
    ax2.plot(xx, [s["false_alarms_per_series"] for s in scan], "s--",
             color="#c0392b", label="false alarms / series")
    ax2.set_ylabel("false alarms per series", color="#c0392b")
    ax.set_xticks(xx)
    ax.set_xticklabels([str(int(v)) for v in lams])
    ax.set_xlabel("expected run length lambda")
    ax.set_title("Hazard sensitivity: delay vs false alarms")
    fig.tight_layout()
    save(fig, "fig4_hazard_sensitivity.png")

    # ROC：BOCD 的短段后验质量 vs 报警后重置的双侧 CUSUM
    # 两边都用「不重叠 20 天窗口」计标签，阈值各自扫描，评价口径一致。
    scores_b = [bocd(xs, 80.0)[1] for xs, _ in seqs]
    thrs_b = np.unique(np.quantile(np.concatenate(scores_b),
                                   np.linspace(0.50, 0.9995, 80)))
    thrs_c = np.linspace(0.2, 8.0, 100)

    def roc_curve(thrs, alarms_fn):
        pts = []
        for thr in thrs:
            Y, S = [], []
            for i, (_, cps) in enumerate(seqs):
                y, s = window_labels_and_preds(alarms_fn(i, thr), cps)
                Y.append(y)
                S.append(s)
            Y, S = np.concatenate(Y), np.concatenate(S)
            tpr = S[Y == 1].mean() if (Y == 1).any() else 0.0
            fpr = S[Y == 0].mean() if (Y == 0).any() else 0.0
            pts.append((float(fpr), float(tpr)))
        pts = sorted(set(pts))
        return (np.array([p[0] for p in pts]),
                np.array([p[1] for p in pts]))

    f_b, t_b = roc_curve(thrs_b, lambda i, thr: pick_alarms(scores_b[i], thr))
    f_c, t_c = roc_curve(thrs_c, lambda i, h: cusum_alarms(seqs[i][0], h))
    auc_b = auc_from_curve(np.r_[0.0, f_b, 1.0], np.r_[0.0, t_b, 1.0])
    auc_c = auc_from_curve(np.r_[0.0, f_c, 1.0], np.r_[0.0, t_c, 1.0])

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(np.r_[0.0, f_b, 1.0], np.r_[0.0, t_b, 1.0], "o-", ms=3,
            color="#c0392b", label=f"BOCD (AUC={auc_b:.3f})")
    ax.plot(np.r_[0.0, f_c, 1.0], np.r_[0.0, t_c, 1.0], "s-", ms=3,
            color="#2f5597", label=f"CUSUM (AUC={auc_c:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", color="#888888")
    ax.set_xlabel("false positive rate (per 20-day window)")
    ax.set_ylabel("true positive rate")
    ax.set_title("Detection ROC over 13 independent series")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    save(fig, "fig5_roc_bocd_cusum.png")

    results = {
        "seed": SEED,
        "n_days": T,
        "true_changepoints": list(TRUE_CPS),
        "true_sigmas": list(SIGMAS),
        "lambda_main": 80.0,
        "threshold_main": THRESHOLD,
        "short_runlength_cutoff": SHORT_R,
        "detected_changepoints": alarms,
        "true_positives": tp,
        "false_alarms": fp,
        "mean_signed_delay": delay,
        "hazard_scan": scan,
        "roc_auc_bocd": auc_b,
        "roc_auc_cusum": auc_c,
        "n_series": len(seqs),
        "tol_days": TOL,
    }
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("true changepoints :", list(TRUE_CPS))
    print("detected          :", alarms)
    print(f"threshold={THRESHOLD:.2f}  TP={tp}  FP={fp}  "
          f"mean_signed_delay={delay:.2f} days")
    for s in scan:
        d = ("nan" if s["mean_delay"] is None
             else f"{s['mean_delay']:.2f}")
        print(f"lambda={s['lambda']:>5.0f}  delay={d}  "
              f"FA/series={s['false_alarms_per_series']:.3f}  "
              f"TP/series={s['true_positives_per_series']:.3f}")
    print(f"ROC AUC: BOCD={auc_b:.4f}  CUSUM={auc_c:.4f}")
    return results


if __name__ == "__main__":
    run()
```

## 八、小结

BOCD 把「当前段还成不成立」变成了一个每天更新的概率分布，这比离线回溯更贴合风控的真实约束。但本文最想留下的不是 0.9665 这个 AUC，而是两件事：常数 hazard 下 `P(r_t=0)` 是个与数据无关的常数，照抄网上实现会得到一台永远不会响的警报器；以及，任何检出率数字都必须连同 hazard、判决阈值、容差窗口和基线配置一起报告，否则就只是调参调出来的好看曲线。相关方法可与站内已有的离线变点篇对照阅读，两者是互补而非替代关系。
