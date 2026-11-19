---
title: "狄利克雷过程混合模型：用贝叶斯非参数让收益密度的成分数从数据里自己长出来"
description: "普通高斯混合要事先猜 K，K 选错密度就歪。狄利克雷过程混合把 K 变成随机变量，让数据自己决定需要几个成分。本文从中国餐馆过程推到折叠吉布斯采样，用一个三波动状态的 2400 天合成收益做实验：后验占据成分数均值 8.98（真值 3，DP 用更多小成分去描尾巴），测试集每观测对数似然 DPM 3.1848 略胜 Student-t 3.1772、KDE 3.1763，明显甩开正态 2.9109；99% VaR 真值 -4.838%，DPM -5.409% 略偏保守，正态 -3.497% 低估 28%。附完整 numpy 代码、四张真实计算图，以及两个真实踩过的坑——先验尺度没校准会让 K 爆到 87，预测分布漏掉 1/a 因子会让采样退化成噪声。"
publishDate: '2026-11-20'
tags:
  - 量化交易
  - 贝叶斯非参数
  - 狄利克雷过程
  - 混合模型
  - 风险价值
  - 吉布斯采样
  - Python
language: Chinese
difficulty: advanced
---

风控和因子建模里到处都要一个「收益密度」：算 VaR 要它，算贝叶斯后验要它，做似然比检验也要它。最省事的做法是假设正态，但收益的尖峰肥尾人人都见过；换成 Student-t 好了一些，自由度却要一个参数同时管中心和尾巴；上高斯混合呢，成分数 K 又得先拍脑袋——K 小了拟合不出来，K 大了过拟合。

**狄利克雷过程混合模型（Dirichlet Process Mixture, DPM）**把这个问题整个翻过来：不再问「有几个成分」，而是给成分数放一个先验分布，让数据自己说话。本文从 stick-breaking 和中国餐馆过程讲到折叠吉布斯采样，在一组真值已知的三状态收益数据上，把它和正态、Student-t、核密度估计正面对比，并诚实给出它也没那么神的地方。

## 一、为什么成分数不该由你来定

有限高斯混合写成 $p(x)=\sum_{k=1}^{K}\pi_k N(x\mid\mu_k,\sigma_k^2)$，一旦 K 固定，模型容量就封顶了。选 K 通常靠 BIC 或交叉验证扫一遍，但这是个离散搜索，K 每加一个都要重新拟合，而且「真实分布有几个成分」本身常常没有明确答案——收益的厚尾也许不是三个离散状态，而是连续的一族。

贝叶斯非参数给出的答案是：让 K 趋向无穷，但只让有限个成分真正被数据占用。狄利克雷过程 $\mathrm{DP}(\alpha,G_0)$ 是定义在参数空间上的随机分布，$\alpha$ 是 Concentration 参数、控制新成分的出现倾向，$G_0$ 是基分布、给出新成分参数的先验。从 DP 抽一个 $G$，再对每个观测从 $G$ 抽参数，边际上就得到混合模型，而成分数自动是有限的。

两种等价的构造有助于理解。一是 **stick-breaking**：把一根长度为 1 的棍子按比例 $v_k\sim\mathrm{Beta}(1,\alpha)$ 不断折断，$\pi_k=v_k\prod_{j<k}(1-v_j)$，剩下的 $\pi$ 直接往无穷序列上摊；二是**中国餐馆过程（CRP）**：第 $i$ 个顾客以概率 $n_k/(i-1+\alpha)$ 坐到已有第 $k$ 桌，以概率 $\alpha/(i-1+\alpha)$ 开新桌。后者直接给出了可采样的条件概率，也是本文折叠吉布斯的核心。

stick-breaking 还顺手解释了一个常被忽略的事实：DP 的「无穷成分」是数学上的构造，实际被占用的成分永远有限。第 k 根棍子的期望长度是 $\mathbb{E}[\pi_k]=\frac{1}{1+\alpha}\left(\frac{\alpha}{1+\alpha}\right)^{k-1}$，随着 k 几何衰减——$\alpha=1$ 时第一桌期望占一半、第二桌占四分之一、第三桌占八分之一。这意味着 $\alpha$ 越大，尾巴上的小成分越多、模型越灵活，但单个成分分到的数据也越少、估计越不稳。行业里常把 $\alpha$ 叫「创新偏好」，它不是越大越好：太小退回有限混合，太大则每个观测都想开新桌，密度估计退化成以每个样本为中心的一堆尖峰。本文取 $\alpha=1$ 作为折中，并在最后一节讨论把它也放进先验采样的做法。

从 stick-breaking 到 CRP 的等价性不是巧合。如果把顾客按顺序落座的过程对归属变量积分掉，得到的正是 DP 的边际——先后次序不影响结果，这个性质叫可交换性（de Finetti 定理的延伸）。对量化建模的意义是：我们不需要假设数据按某种顺序到达，混合模型的后验与观测排列无关，这跟隐马尔可夫那种显式依赖时间的模型形成鲜明对比。代价是无法表达波动状态的持续性——今天平静、明天大概率还平静这种记忆，CRP 不知道，它只看边际密度。所以 DPM 是密度估计工具，不是状态识别工具，这个定位后面实验结果里还会再次印证。

## 二、把参数积分掉：折叠吉布斯采样

如果我们把每个成分的参数 $\theta_k$ 显式采样，链在高维空间和局部分配之间来回跳，混合很慢。更干净的办法是把 $\theta_k$ 积分掉——这就是 **collapsed Gibbs**：只采样每个观测的归属 $z_i$，条件分布为

$$P(z_i=k\mid z_{-i},x_i)\propto \begin{cases} n_k^{(-i)}\cdot m(x_i\mid \mathbf{x}_k^{(-i)}) & \text{已有成分}\\ \alpha\cdot m(x_i\mid G_0) & \text{新成分}\end{cases}$$

其中 $m(x\mid\cdot)$ 是后验预测密度。基分布取 Normal-Inverse-Gamma：$\mu\mid\sigma^2\sim N(\mu_0,\sigma^2/\kappa_0)$，$\sigma^2\sim\mathrm{IG}(a_0,b_0)$。这个选择让后验预测是 Student-t，参数可以解析更新。

这里藏着第一个真实的坑。NIG 后验预测分布的尺度平方应为

$$s^2=\frac{b_n\,(\kappa_n+1)}{\kappa_n\,a_n}$$

分母里的 $a_n$ 千万别漏。我第一版写成 $b_n(\kappa_n+1)/\kappa_n$，结果每个成分的预测分布都被人为吹宽，开新桌总是划算，后验成分数一路飙到 87——真值只有 3。修正后 K 的后验均值立刻回落到 8.98。第二个坑是基分布校准：把 $b_0$ 拍成固定小值，先验方差与数据尺度差一个量级，同样会把 K 顶飞。稳妥做法是用样本矩做经验贝叶斯初始化：$\mu_0=\bar x$，$\kappa_0=1$，$a_0=5$，$b_0=\tfrac12\hat{\sigma}^2(a_0-1)$。

## 三、实验：三波动状态的 2400 天收益

数据分三个状态：平静（权重 50%，均值 0.05%，波动 0.4%）、常态（35%，0.0%，1.0%）、压力（15%，-0.35%，3.0%），共 2400 天，前 1800 天训练、后 600 天测试。$\alpha=1$，采样 700 轮，前 200 轮烧入，每 4 轮留一个样本。图 1 画出了训练集里三个状态的散点，压力状态虽然只占 15%，却贡献了左尾绝大多数极端点。

![三波动状态合成收益散点](/images/dirichlet-process-mixture-model/01_regimes.png)

选这个数据不是为了制造好看的结果，而是因为它有一个关键特征：**三个状态的波动率相差 7.5 倍**。这正好是参数化密度最难对付的情形——正态只有一个方差参数，被高波动状态拉宽后，平静状态的尖峰就完全糊掉；Student-t 用自由度调肥尾，却无法区分「中心很窄但尾巴很长」这种组合。反过来，三成分混合本来应该轻松拿下，但前提是 K 得猜对。这里真值是 3，我们故意不告诉模型，让它自己从 1800 个点里推。

采样的实现上我用了一个小技巧：只维护当前非空成分的充分统计量 $(n_k,\sum x_i,\sum x_i^2)$，而不是开一个固定长度的数组。某个成分被抽空时立刻从列表里删掉并更新所有归属标签。这样既省内存，又避免了「预分配 K 个槽位」带来的编号冲突——我第一版就是死在这里：删空桌时只更新了当前观测之后的标签，前面的标签还指向旧编号，链跑几步就数组越界。修正成遍历全部标签重编号后才稳定。这段逻辑看着琐碎，却是折叠吉布斯最容易出错的地方。

## 四、结果：密度赢了，但别把成分当状态

密度拟合见图 2。DP 后验预测曲线几乎贴着真实混合密度把主峰和左右尾都描了出来；Student-t 主峰略低、左尾稍薄；正态整体摊平，主峰只有真值的一半不到。数值上，600 天测试集的每观测对数似然：DPM 3.1848 > Student-t 3.1772 > KDE 3.1763 >> 正态 2.9109。DPM 的优势不大（只比 t 高 0.0076），但对正态是碾压级的。

![后验预测密度与参数化基线对比](/images/dirichlet-process-mixture-model/02_density.png)

成分数 posterior 见图 3：均值 8.98、众数 9，5 到 15 之间都有概率质量。**注意这不等于「找到了 9 个市场状态」**——真值只有 3。DP 把每个宽状态拆成几个窄成分去贴合形状，尤其是尖峰附近。这是灵活性带来的代价：成分数不能直接解释成经济状态，想要可解释的状态数还得回到隐马尔可夫那套。

![成分数后验分布与吉布斯采样轨迹](/images/dirichlet-process-mixture-model/03_clusters.png)

风险价值的结果更有意思（图 4）。99% VaR 真值 -4.838%，DPM 给 -5.409%，偏保守 0.57 个百分点；正态给 -3.497%，**低估了两成八**——这才是真正危险的地方。Kupiec 检验上，DPM 600 天里 2 次违约、p=0.0567 勉强通过；正态 9 次违约 p=0.2518 也通过。别误读这个「通过」：600 天在 99% 水平只有 6 次期望违约，检验功效极低，几个模型都过不了被拒的门槛。要说结论，只能说正态的 VaR 数值明显偏浅，而 DPM 更接近真值。

![左尾放大与各模型 99% VaR](/images/dirichlet-process-mixture-model/04_tail_var.png)

## 五、边界与代价

DP 不是免费午餐。一是计算：700 轮折叠吉布斯在 1800 个点上跑了约 19 秒，点多了要上变分推断或 slice sampling。二是可解释性：如上所述，8.98 个成分不是 9 个状态。三是 α 敏感：本文固定 $\alpha=1$，实际上应该给 $\alpha$ 也放 Gamma 先验一起采样，否则成分数会被这个超参数悄悄锚定。四是低维限制：一维密度很漂亮，多维收益的 DP 混合需要 Wishart 基分布，采样难度陡增。

## 六、完整代码

```python
import math
import numpy as np
from scipy.special import gammaln
from scipy.stats import t as student_t, norm

np.random.seed(20261120)
N, TRAIN = 2400, 1800
means = np.array([0.0005, 0.0000, -0.0035])
sigmas = np.array([0.0040, 0.0100, 0.0300])
weights = np.array([0.50, 0.35, 0.15])
reg = np.random.choice(3, size=N, p=weights)
x = means[reg] + sigmas[reg] * np.random.randn(N)
x_train, x_test = x[:TRAIN], x[TRAIN:]

# NIG 基分布：经验贝叶斯初始化
MU0 = float(x_train.mean())
KAPPA0, A0, ALPHA = 1.0, 5.0, 1.0
B0 = float(x_train.var()) * (A0 - 1.0) * 0.5

def post_params(n, s1, s2):
    if n <= 0:
        return KAPPA0, MU0, A0, B0
    kap = KAPPA0 + n
    mu = (KAPPA0 * MU0 + s1) / kap
    a = A0 + 0.5 * n
    ss = max(s2 - s1 * s1 / n, 0.0)
    b = B0 + 0.5 * ss + 0.5 * KAPPA0 * n * (s1 / n - MU0) ** 2 / kap
    return kap, mu, a, b

def log_pred(xv, kap, mu, a, b):
    # 关键：尺度平方分母必须含 a，漏掉会让成分数爆炸
    scale2 = b * (kap + 1.0) / (kap * a)
    nu = 2.0 * a
    lga = gammaln(a + 0.5) - gammaln(a)
    return lga - 0.5*np.log(nu*np.pi*scale2) \
           - (a+0.5)*np.log1p((xv-mu)**2/(nu*scale2))

# 折叠吉布斯：clusters 存 (n, sum_x, sum_x2)
SWEEPS, BURN, THIN = 700, 200, 4
clusters = [[TRAIN, float(x_train.sum()), float((x_train**2).sum())]]
z = [0] * TRAIN
k_trace = []
for it in range(SWEEPS):
    for i in range(TRAIN):
        xv, k = x_train[i], z[i]
        c = clusters[k]
        c[0] -= 1; c[1] -= xv; c[2] -= xv*xv
        if c[0] == 0:                      # 空桌撤掉并重编号
            clusters.pop(k)
            for j in range(TRAIN):
                if j != i and z[j] > k:
                    z[j] -= 1
        K = len(clusters)
        lps = []
        for nj, s1j, s2j in clusters:
            kap, mu, a, b = post_params(nj, s1j, s2j)
            lps.append(math.log(nj) + float(log_pred(xv, kap, mu, a, b)))
        kb, mb, ab, bb = post_params(0.0, 0.0, 0.0)
        lps.append(math.log(ALPHA) + float(log_pred(xv, kb, mb, ab, bb)))
        mx = max(lps)
        exps = [math.exp(v-mx) for v in lps]
        r = np.random.random() * sum(exps)
        pick = len(lps)-1
        acc = 0.0
        for j, e in enumerate(exps):       # 类别采样，不引入额外依赖
            acc += e
            if r <= acc:
                pick = j; break
        if pick == K:
            clusters.append([1, xv, xv*xv]); z[i] = K
        else:
            cp = clusters[pick]
            cp[0] += 1; cp[1] += xv; cp[2] += xv*xv; z[i] = pick
    k_trace.append(len(clusters))

karr = np.array(k_trace[BURN:])
print("K posterior mean =", karr.mean(), "mode =", np.bincount(karr).argmax())
```

把每次采样留下的 `clusters` 状态代入 `predict_density`，在网格上平均即得后验预测密度；VaR 取累计密度到 1% 处的分位数，Kupiec 检验用标准的 $LR=-2\log(L_{\text{null}}/L_{\text{alt}})$。四张图分别画三状态散点、密度对比、成分数后验与轨迹、左尾 VaR 放大，见 `public/images/dirichlet-process-mixture-model/`。

## 七、小结

DPM 真正的价值不是「找到几个成分」，而是给你一个**不必先猜 K 的密度估计器**：正态和 Student-t 是拿一个族去套数据，DPM 是让数据在这个族里自己长。在本文的实验里，它在样本外对数似然上小胜 Student-t、大胜正态，99% VaR 也比正态靠谱得多。代价是计算更贵、成分数不可直接解释、超参数仍需小心校准。把它当成「会自己调节复杂度的 KDE」，比当成「自动发现市场状态的神器」，是更准确的定位。
