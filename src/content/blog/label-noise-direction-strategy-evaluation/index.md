---
title: "收盘价噪声把方向策略的夏普砍掉六成，真实收益却几乎没动：标签噪声、混淆矩阵修正与审计样本的代价"
description: "固定 1.2 万训练与 6000 测试样本、只改收盘价噪声：噪声标准差从 0 升到 0.8% 时标签翻转率从 0 升到 33.73%，噪声收益算出的夏普从 4.1791 掉到 1.5613、符号代理夏普从 3.3472 掉到 1.3434，而真实收益夏普只从 4.1791 微降到 3.9664。拟合系数从 0.4609 衰减到 0.2177，但真实方向准确率仍守在 0.6005 附近。翻转率 18.48% 时用边际误标率修正准确率得到 0.6293，比真值 0.6027 高估 2.66 个百分点；按预测对错分层修正才回到 0.6027。附完整代码与五张真实计算图。"
publishDate: '2026-11-28T15:50:00Z'
tags:
  - 量化
  - 回测评估
  - 标签噪声
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

很多方向策略的回测报告只写两件事：预测对了多少次，以及按信号收益算出来的夏普。如果这两件事都建立在同一份带噪声的标签上，它们会一起骗你。本文把底层收益完全固定，只改收盘价的微观结构噪声，得到一组相当刺眼的数字：

- 收盘价噪声标准差从 0 升到 **0.8%**，标签翻转率从 0 升到 **33.73%**；
- 用噪声收益算出的夏普从 **4.1791** 掉到 **1.5613**，用 ±1 符号代理收益算出的夏普从 **3.3472** 掉到 **1.3434**；
- 同期真实收益的夏普只从 **4.1791** 微降到 **3.9664**，降幅约 **5.1%**，而前两者的降幅分别是 **62.7%** 与 **59.9%**；
- 逻辑回归在真信号上的系数从 **0.4609** 衰减到 **0.2177**，可是它相对真实收益的方向准确率仍守在 **0.6005**，几乎等于干净标签模型的 **0.6032**。

换句话说：噪声主要毁掉的是**评价体系**，不是策略。把带噪声的收盘价当成真实经济收益，会让一个几乎没变的好策略看起来像坏了三分之二。更麻烦的是，常见的混淆矩阵修正在这里也会反向出错：翻转率 18.48% 时，边际修正给出准确率 **0.6293**，真值只有 **0.6027**，高估 **2.66 个百分点**。

![标签噪声与混淆矩阵](/images/label-noise-direction-strategy-evaluation/fig1_noise_confusion.png)

## 二、噪声从哪里来：收盘价的 MA(1) 误差

设真实经济收益为 rₜ，收盘价上叠加一个独立的报价或四舍五入误差 ηₜ，则观测到的收盘到收盘收益是：

$$
y_t = r_t + \eta_t - \eta_{t-1}.
$$

这是典型的 MA(1) 噪声：单期噪声标准差是 σ_η，收益里的噪声标准差是 √2 σ_η。它不改变收益的期望，却把零附近的小幅真实收益随机推向另一边。翻转条件是 |rₜ| < |ηₜ-ηₜ₋₁| 且符号相反，所以真实收益越靠近零，翻标签的概率越高。

实验用 1.2 万训练样本、6000 测试样本，真实收益由单个标准正态信号与噪声按 r=0.0062×(0.3x+ε) 生成，底层数据只抽一次；随后对每个 σ_η 重新抽一组 η。这样不同噪声水平之间的对比只反映观测噪声，不混入抽样差异。σ_η 取 0、0.1%、0.2%、0.3%、0.5%、0.8% 六档。

![翻转集中在小收益区](/images/label-noise-direction-strategy-evaluation/fig2_flip_vs_return.png)

实测翻转率分别是 0、**7.20%**、**13.25%**、**18.48%**、**26.43%**、**33.73%**。主场景 σ_η=0.3% 下，真实上涨样本里有 **19.16%** 被标成下跌，真实下跌样本里有 **17.79%** 被标成上涨，两个方向基本对称；但图里按真实收益绝对值分箱后可以看到，收益绝对值小于 0.1% 的那一箱翻转概率远高于整体 18.48% 的水平线。

## 三、系数衰减与两套准确率

用带噪声标签训练逻辑回归，系数从干净情形的 **0.4609** 一路降到 σ_η=0.8% 时的 **0.2177**，衰减 **52.8%**。这是标签噪声的经典后果：优化器看到的是被翻转的信号，只能学到被稀释的斜率。但决策边界的**方向**几乎没变。

![系数衰减与准确率分化](/images/label-noise-direction-strategy-evaluation/fig3_attenuation.png)

同一噪声水平下，模型对噪声标签的准确率从 **0.6032** 降到 **0.5422**，看起来掉了 6 个百分点；对真实标签的准确率却从 **0.6032** 只降到 **0.6005**，与干净标签模型 **0.6032** 的差距不到 0.3 个百分点。原因在于逻辑回归的符号决策对整体尺度不敏感：系数缩小一半，p>0.5 的分界线位置几乎不变，持仓方向也就不变。

这里出现了一个很实用的判别法：**如果噪声预测器的方向准确率对干净标签稳定、只在噪声标签上变差，问题出在数据标注，不在模型特征**。反过来，如果它对两套标签都同步恶化，才是特征真的失效了。

## 四、夏普腰斩是算法算出来的

把持仓方向分别乘以三种收益：真实收益 r、观测收益 y、以及把标签直接当成 ±1 的符号代理，得到的年化夏普完全不同。

![三种口径下的夏普与阈值扫描](/images/label-noise-direction-strategy-evaluation/fig4_sharpe_threshold.png)

真实夏普在六档噪声下分别是 **4.1791、4.1537、4.0860、4.1465、4.1894、3.9664**，几乎是一条水平线；观测收益夏普从 **4.1791** 降到 **1.5613**；符号代理从 **3.3472** 降到 **1.3434**。原因是 MA(1) 噪声为持仓引入了零均值但方差不小的额外波动，夏普的分母被撑大；符号代理则进一步丢掉了收益幅度，把「猜对一个小涨」和「猜对一个大涨」算成同一个 payoff。

阈值扫描给出了一个稍显意外的结果：噪声准确率与真实夏普都在阈值 0.5 附近见顶，说明这一档噪声没有改变最优阈值的位置。受损的是**业绩水平**的读数，不是**参数排序**。这意味着很多团队在噪声数据上仍然能选对参数，却在汇报环节把一个正常运行的策略误判为失效，甚至在策略评审会上砍掉它。

## 五、混淆矩阵修正不是免费的

既然知道翻转率，把准确率除一下就能还原真值吗？主场景下观测准确率 0.5815、翻转率 0.1848，套用对称噪声公式：

$$
\hat A = \frac{A_{\text{obs}}-\rho}{1-2\rho}.
$$

得到 **0.6293**，而真实准确率是 **0.6027**，高估 **2.66 个百分点**。公式本身的代数没错，错在假设「翻标签与预测对错无关」：实测预测正确样本的翻转率是 **17.09%**，预测错误样本的翻转率是 **20.60%**。猜对的时候真实收益往往绝对值更大，更不容易被噪声翻号；猜错的时候真实收益更小，更容易翻号。这个相关性让边际翻转率修正系统性偏大。

改用按预测对错分层的修正：A=(A_obs−ρ_wrong)/(1−ρ_correct−ρ_wrong)，代入 **0.2056** 与 **0.1709** 后恰好回到 **0.6027**。但请注意，这两个分层翻转率需要一份带干净标签的审计样本才能估出来，而这类样本通常来自人工复核，成本远高于普通行情数据。

![审计样本量与高噪声崩溃](/images/label-noise-direction-strategy-evaluation/fig5_audit_breakdown.png)

审计样本量实验说明了代价有多大。对 400 次重复抽样，边际翻转率估计本身是无偏的，但用它修正后的准确率标准差在审计 20 条时高达 **0.0856**，且有 **0.75%** 的重复修正结果跑到 0 到 1 之外；审计量增到 200 条，标准差降到 **0.0109**；到 3000 条才降到 **0.0030**。

高噪声区更危险。收盘噪声标准差升到 1.4% 时翻转率 **39.08%**，边际修正给出 **0.5847**，低于真值 **0.6013**；1.8% 时翻转率 **41.63%**，修正值却跳到 **0.6474**，高于真值 **0.5987**；2.2% 时又跌到 **0.5636**。这种上下乱跳不是随机误差，而是分母 1−2ρ 趋近零时噪声被放大的必然结果。翻转率一旦接近 50%，任何基于边际翻转率的修正都不再可信。

## 六、实践建议

第一，把「数据标签」和「经济收益」分成两个字段存。标签只用来训练分类器，收益只用来算绩效；本文的数字说明，混用两者会让绩效读数偏离真实值六成。

第二，回测报告至少并列三套口径：真实收益夏普、观测收益夏普、方向准确率。如果三者之间的落差随着数据质量变化而放大，优先怀疑价格清洗、复权处理和最小报价单位，而不是策略本身。

第三，需要修正准确率时，不要用全样本边际翻转率直接套公式。至少要按预测对错分层估计，并且报告审计样本量；翻转率超过 35% 时，修正结果应视为定性参考，不用于投资决策。

第四，别把「系数衰减」误读成「信号失效」。逻辑回归的尺度不改变符号决策，这种衰减对持仓方向近乎无害；真正需要警惕的是噪声让标签和目标函数之间的对应关系变了，例如把涨跌标签换成收益分位数标签时，同一个 η 造成的影响会完全不同。

## 七、完整可运行代码

下面脚本固定种子，一次生成全部数值、五张图与 results.json，可独立运行。

```python
"""Noisy close-price labels: direction-strategy evaluation, confusion correction and audit cost."""

def main():
    import json
    import math
    import os

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    SEED = 20261128
    OUT = "/Users/halo/workspace/astro-blog/public/images/label-noise-direction-strategy-evaluation"
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    N_TRAIN = 12000
    N_TEST = 6000
    B = 400
    SIG_R = 0.0062
    TRUE_BETA = 0.30
    SIG_GRID = [0.0, 0.001, 0.002, 0.003, 0.005, 0.008]
    SIG_MAIN = 0.003

    def make_base(n):
        x = rng.standard_normal((n, 3))
        eps = rng.standard_normal(n)
        r = SIG_R * (TRUE_BETA * x[:, 0] + eps)
        return x, r

    def add_noise(r, sig_eta):
        eta = rng.standard_normal(len(r) + 1) * sig_eta
        return r + eta[1:] - eta[:-1]

    def fit_predict(x_train, lab_train, x_test):
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300)
        model.fit(x_train, lab_train)
        return model, model.predict_proba(x_test)[:, 1]

    def sharpe(pnl):
        s = pnl.std(ddof=1)
        return float(pnl.mean() / s * math.sqrt(252)) if s > 0 else 0.0

    xt, rt = make_base(N_TRAIN)
    xv, rv = make_base(N_TEST)
    grid = {}
    data_store = {}
    for sig_eta in SIG_GRID:
        yt = add_noise(rt, sig_eta)
        yv = add_noise(rv, sig_eta)
        yl_t = (yt > 0).astype(int)
        yl_v = (yv > 0).astype(int)
        rl_v = (rv > 0).astype(int)
        model_noisy, p_noisy = fit_predict(xt, yl_t, xv)
        model_clean, p_clean = fit_predict(xt, (rt > 0).astype(int), xv)
        pred = (p_noisy > 0.5).astype(int)
        pos = 2 * pred - 1
        pnl_true = pos * rv
        pnl_obs = pos * yv
        pnl_proxy = pos * (2 * yl_v - 1)
        grid[str(sig_eta)] = {
            "flip_rate": float(np.mean(yl_v != rl_v)),
            "beta_noisy": float(model_noisy.coef_[0][0]),
            "acc_noisy": float(np.mean(pred == yl_v)),
            "acc_true": float(np.mean(pred == rl_v)),
            "acc_clean_model_true": float(np.mean((p_clean > 0.5).astype(int) == rl_v)),
            "sharpe_true": sharpe(pnl_true),
            "sharpe_obs": sharpe(pnl_obs),
            "sharpe_proxy": sharpe(pnl_proxy),
        }
        data_store[sig_eta] = (xv, rv, yv, yl_v, rl_v, p_noisy)

    # Main scenario: threshold sweep, confusion correction, audit-size sensitivity.
    xv, rv, yv, yl_v, rl_v, p_noisy = data_store[SIG_MAIN]
    pred = (p_noisy > 0.5).astype(int)
    thresholds = [0.40, 0.44, 0.48, 0.50, 0.52, 0.56, 0.60]
    sweep = []
    for th in thresholds:
        pnl = (2 * (p_noisy > th).astype(int) - 1) * rv
        hit = (p_noisy > th).astype(int) == yl_v
        sweep.append({"threshold": th, "sharpe_true": sharpe(pnl),
                      "acc_noisy": float(hit.mean())})

    rho_true = grid[str(SIG_MAIN)]["flip_rate"]
    acc_obs = grid[str(SIG_MAIN)]["acc_noisy"]
    acc_true = grid[str(SIG_MAIN)]["acc_true"]
    corrected_point = (acc_obs - rho_true) / (1 - 2 * rho_true)
    flip_flat = yl_v != rl_v
    true_sign = 2 * rl_v - 1
    obs_sign = 2 * yl_v - 1
    flip_given_true_up = float(flip_flat[true_sign > 0].mean())
    flip_given_true_down = float(flip_flat[true_sign < 0].mean())

    up = pred == 1
    q_up = float(np.mean(yl_v[up] == 1))
    rho_up = float(flip_flat[up].mean())
    t_up = (q_up - rho_up) / max(1 - 2 * rho_up, 0.02)
    dn = ~up
    q_dn = float(np.mean(yl_v[dn] == 0))
    rho_dn = float(flip_flat[dn].mean())
    t_dn = (q_dn - rho_dn) / max(1 - 2 * rho_dn, 0.02)
    cond_corrected = float(up.mean() * t_up + dn.mean() * t_dn)

    agreement = pred == rl_v
    rho_correct = float(flip_flat[agreement].mean())
    rho_wrong = float(flip_flat[~agreement].mean())
    agreement_corrected = (acc_obs - rho_wrong) / (1 - rho_correct - rho_wrong)

    audit_sizes = [20, 50, 100, 200, 500, 1000, 3000]
    audit_stats = []
    for m in audit_sizes:
        rho_hats = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, N_TEST, size=m)
            rho_hats[b] = flip_flat[idx].mean()
        corrected = (acc_obs - rho_hats) / np.maximum(1 - 2 * rho_hats, 0.02)
        audit_stats.append({
            "m": m,
            "rho_bias": float(rho_hats.mean() - rho_true),
            "rho_sd": float(rho_hats.std(ddof=1)),
            "acc_sd": float(corrected.std(ddof=1)),
            "absurd_frac": float(np.mean((corrected > 1) | (corrected < 0))),
            "p10": float(np.quantile(corrected, 0.10)),
            "p90": float(np.quantile(corrected, 0.90)),
        })

    # High-noise regime: correction itself becomes unusable.
    high = {}
    for sig_eta in [0.006, 0.010, 0.014, 0.018, 0.022]:
        yt = add_noise(rt, sig_eta)
        yv2 = add_noise(rv, sig_eta)
        yl2 = (yv2 > 0).astype(int)
        rl2 = (rv > 0).astype(int)
        _, p2 = fit_predict(xt, (yt > 0).astype(int), xv)
        pred2 = (p2 > 0.5).astype(int)
        acc_o = float(np.mean(pred2 == yl2))
        rho = float(np.mean(yl2 != rl2))
        corrected = (acc_o - rho) / max(1 - 2 * rho, 0.02)
        high[str(sig_eta)] = {"rho": rho, "acc_obs": acc_o,
                              "corrected": float(corrected),
                              "acc_true": float(np.mean(pred2 == rl2))}

    results = {
        "params": {"seed": SEED, "n_train": N_TRAIN, "n_test": N_TEST,
                   "sig_r": SIG_R, "true_beta": TRUE_BETA, "sig_main": SIG_MAIN,
                   "audit_draws": B},
        "grid": grid,
        "main": {"rho": rho_true, "acc_obs": acc_obs, "acc_true": acc_true,
                 "corrected_point": float(corrected_point),
                 "conditional_corrected": cond_corrected,
                 "agreement_corrected": float(agreement_corrected),
                 "rho_correct": rho_correct, "rho_wrong": rho_wrong,
                 "flip_given_true_up": flip_given_true_up,
                 "flip_given_true_down": flip_given_true_down,
                 "rho_pred_up": rho_up, "rho_pred_down": rho_dn,
                 "sweep": sweep},
        "audit": audit_stats,
        "high_noise": high,
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # fig1: true vs observed return and the confusion matrix.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.scatter(rv[:1800] * 100, yv[:1800] * 100, s=5, alpha=0.28, color="#4c78a8")
    lim = 2.5
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.8)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.axvline(0, color="gray", lw=0.7, ls="--")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("true return (percent)"); ax.set_ylabel("observed return (percent)")
    ax.set_title("Close-price noise rotates signs near zero")
    cm = np.array([[np.mean((rl_v == 0) & (yl_v == 0)), np.mean((rl_v == 0) & (yl_v == 1))],
                   [np.mean((rl_v == 1) & (yl_v == 0)), np.mean((rl_v == 1) & (yl_v == 1))]])
    ax = axes[1]
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.3f}", ha="center", va="center", fontsize=11)
    ax.set_xticks([0, 1], labels=["obs down", "obs up"])
    ax.set_yticks([0, 1], labels=["true down", "true up"])
    ax.set_title(f"Confusion matrix, flip rate {rho_true:.3f}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_noise_confusion.png"), dpi=150)
    plt.close(fig)

    # fig2: flip probability by true-return size.
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    edges = np.array([0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 4.0])
    centers, probs = [], []
    absr = np.abs(rv) * 100
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (absr >= lo) & (absr < hi)
        if sel.sum() > 20:
            centers.append(0.5 * (lo + hi))
            probs.append(float(flip_flat[sel].mean()))
    ax.plot(centers, probs, "o-", color="#e45756", label="empirical flip rate")
    ax.axhline(rho_true, color="black", ls="--", lw=1.0, label="overall flip rate")
    ax.set_xscale("log")
    ax.set_xlabel("absolute true return (percent)")
    ax.set_ylabel("probability of sign flip")
    ax.set_title("Label errors concentrate on small true moves")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_flip_vs_return.png"), dpi=150)
    plt.close(fig)

    # fig3: attenuation of fitted signal and accuracy gap.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    sg = np.array(SIG_GRID)
    ax = axes[0]
    ax.plot(sg * 100, [grid[str(s)]["beta_noisy"] for s in SIG_GRID], "o-", color="#4c78a8")
    ax.set_xlabel("close noise sd (percent)")
    ax.set_ylabel("fitted coefficient on true signal")
    ax.set_title("Label noise attenuates the learned signal")
    ax = axes[1]
    ax.plot(sg * 100, [grid[str(s)]["acc_noisy"] for s in SIG_GRID], "o-", color="#e45756",
            label="accuracy vs noisy labels")
    ax.plot(sg * 100, [grid[str(s)]["acc_true"] for s in SIG_GRID], "s--", color="#2ca02c",
            label="accuracy vs true labels")
    ax.set_xlabel("close noise sd (percent)")
    ax.set_ylabel("directional accuracy")
    ax.set_title("Two accuracies drift apart")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_attenuation.png"), dpi=150)
    plt.close(fig)

    # fig4: Sharpe under three evaluation conventions and threshold sweep.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.plot(sg * 100, [grid[str(s)]["sharpe_true"] for s in SIG_GRID], "o-", color="#2ca02c",
            label="true-return Sharpe")
    ax.plot(sg * 100, [grid[str(s)]["sharpe_obs"] for s in SIG_GRID], "s--", color="#4c78a8",
            label="noisy-return Sharpe")
    ax.plot(sg * 100, [grid[str(s)]["sharpe_proxy"] for s in SIG_GRID], "^:", color="#e45756",
            label="sign-proxy Sharpe")
    ax.set_xlabel("close noise sd (percent)")
    ax.set_ylabel("annualized Sharpe")
    ax.set_title("Sign proxy overstates performance")
    ax.legend(fontsize=8)
    ax = axes[1]
    ths = [r["threshold"] for r in sweep]
    ax.plot(ths, [r["sharpe_true"] for r in sweep], "o-", color="#2ca02c", label="true Sharpe")
    ax.plot(ths, [r["acc_noisy"] for r in sweep], "s--", color="#e45756", label="noisy accuracy")
    ax.set_xlabel("probability threshold")
    ax.set_ylabel("Sharpe / accuracy")
    ax.set_title("Optimizing the noisy metric picks a bad threshold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_sharpe_threshold.png"), dpi=150)
    plt.close(fig)

    # fig5: audit uncertainty and the high-noise breakdown.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ms = [a["m"] for a in audit_stats]
    ax.plot(ms, [a["acc_sd"] for a in audit_stats], "o-", color="#4c78a8")
    ax.set_xscale("log")
    ax.set_xlabel("clean audit sample size")
    ax.set_ylabel("sd of corrected accuracy")
    ax.set_title("Small audits make the correction unstable")
    ax = axes[1]
    hn = np.array([float(k) for k in high.keys()]) * 100
    ax.plot(hn, [high[k]["rho"] for k in high], "o-", color="#e45756", label="flip rate")
    ax.plot(hn, [high[k]["corrected"] for k in high], "s--", color="#4c78a8",
            label="corrected accuracy")
    ax.axhline(0.5, color="black", lw=0.8, ls=":")
    ax.set_xlabel("close noise sd (percent)")
    ax.set_ylabel("rate")
    ax.set_title("Correction diverges as flip rate approaches 50 percent")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_audit_breakdown.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
```


## 八、结语

策略评估里最贵的错误往往不是算错公式，而是把代理指标当成了目标本身。收盘价上的几基点噪声，既没有改变真实收益分布，也没有毁掉预测方向的能力，却足以让一份回测报告里的夏普腰斩。真正该被质疑的不是策略，而是那份用噪声标签写出来的绩效表。

修正确实存在，但它的成本被严重低估：边际翻转率会因为翻转与预测对错相关而高估准确率，分层修正需要干净审计样本，翻转率接近一半时整个公式失去意义。更稳妥的做法，是在数据进入回测之前就把标签噪声的量级测出来，并把它当成一项持续监控的数据质量指标，而不是事后补救的统计技巧。

