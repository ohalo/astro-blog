---
title: "Variance-Gamma 混合分布：用 Gamma 混合时间把高斯收益变成肥尾并回测风险价值"
description: "高斯分布算出的 99% VaR 是 -1.88%，真值却是 -2.36%——低估两成，意味着风控额度从一开始就设错了。Variance-Gamma 的做法很优雅：不让布朗运动按均匀时钟走，而是让它跑在一个随机的 Gamma 时钟上，时钟走得越快、瞬时方差越大，于是尖峰肥尾和偏斜都从机制里自然长出来。本文从随机时钟构造推到闭式密度（含 x=0 的极限处理与 Bessel 缩放技巧），并用数值积分对拍验证密度公式，相对误差低至 1e-15。6000 训练、2000 测试的已知参数实验里：四个参数都拟合到真值附近（θ=-0.001496 对 -0.0015、ν=0.793 对 0.8），样本外每观测对数似然 VG 3.5157 略胜 Student-t 的 3.5078、明显高于正态 3.4402；99% VaR -2.319% 对真值 -2.357%，误差仅 1.6%，而正态 -1.880% 低估 20.2%；Kupiec 检验 99% 水平 VG 25 次违约 p=0.279 通过，t 分布 33 次 p=0.0076 被拒，正态 51 次 p=5.6e-9 惨败。附完整可运行代码、四张真实计算图与模型边界的诚实讨论。"
publishDate: '2026-11-20'
tags:
  - 量化交易
  - Variance-Gamma
  - 肥尾
  - 风险价值
  - 极大似然
  - 收益分布
  - Python
language: Chinese
difficulty: advanced
---

风控里有个反复出现的尴尬：用历史波动率算出来的 99% VaR，遇到真正极端的一天就被击穿。问题往往不在数据量，而在分布族的假设。高斯分布尾巴衰减得太快，收益的尖峰、肥尾、轻微偏斜它一样都解释不了；Student-t 加个自由度能兜住对称性肥尾，却无法表达「跌得比涨得更凶」这种非对称，而且自由度一个参数要同时管中心厚度和尾部厚度，常常顾此失彼。

**Variance-Gamma（VG）分布**提供了一条更有物理直觉的路。它不问「收益分布长什么样」，而是问「如果时间本身是随机的会怎样」。把布朗运动的时钟换成 Gamma 过程——市场平静时钟走得慢，恐慌时钟飞快——高斯噪声经过随机时间的混合，就自动变成有尖峰、肥尾、可偏斜的分布。参数不多，只有四个，但每个都有明确含义，似然有闭式表达，VaR 和 ES 也能算。本文从构造推到密度，用数值积分验证公式，再在一组真值已知的数据上和正态、Student-t 正面对比，最后用 Kupiec 检验看谁的风控额度真的靠谱。

## 一、随机时钟：把高斯揉成肥尾

设 $G\sim\mathrm{Gamma}(t/\nu,\ \nu)$，形状参数为 $t/\nu$、尺度参数为 $\nu$，于是 $E[G]=t$，$\mathrm{Var}[G]=t\nu$。定义收益

$$X=\mu+\theta G+\sigma W_G$$

其中 $W_G$ 是在时间 $G$ 处的标准布朗运动。条件于 $G=g$ 时，$X$ 只是普通正态 $N(\mu+\theta g,\ \sigma^2 g)$；把 $g$ 按 Gamma 分布积分掉，边际分布就是 VG。四个参数的含义一目了然：$\mu$ 是漂移基准，$\sigma$ 控制扩散强度，$\nu$ 控制时钟的随机程度也就是尾部厚度（$\nu\to 0$ 退化为正态），$\theta$ 控制漂移随时间走的方向也就是偏度——本文取负值，意味着时钟走得越快、漂移越负，左尾被拉长。

一阶矩和二阶矩可以手算：

$$E[X]=\mu+\theta t,\qquad \mathrm{Var}[X]=(\sigma^2+\theta^2\nu)t$$

注意均值不等于 $\mu$：随机时钟的期望是 1，所以均值里有 $\theta$ 的贡献。这点很容易在参数解释时搞错。超额峰度则是 $\nu$ 的单调递增函数，这正是「时钟越随机、尾巴越肥」的数学表达。之所以叫 Variance-Gamma，正因为条件方差本身是 Gamma 分布的。

## 二、闭式密度与两个数值陷阱

对 $G$ 做积分并整理指数项，可以得到闭式密度。记 $k=1/\nu$，$w=\sqrt{\theta^2+2\sigma^2/\nu}$，$d=x-\mu$，则

$$f(x)=\frac{2\exp(\theta d/\sigma^2)}{\sigma\sqrt{2\pi}\,\Gamma(k)\nu^k}\left(\frac{|d|}{w}\right)^{k-\frac12}K_{k-\frac12}\!\left(\frac{w|d|}{\sigma^2}\right)$$

其中 $K_v$ 是第二类修正贝塞尔函数。对数的形式更适合优化：

$$\log f=\frac{\theta d}{\sigma^2}-\log\sigma-\tfrac12\log(2\pi)-\log\Gamma(k)-k\log\nu+\log 2+(k-\tfrac12)\log\frac{|d|}{w}+\log K_{k-\frac12}(z)$$

这里有两个坑，都会在实盘里悄悄毁掉你的似然。**第一个坑是 $d=0$**：$\log|d|$ 发散，必须用极限。当 $z\to 0$ 时 $K_v(z)\sim 2^{v-1}\Gamma(v)z^{-v}$，代回去得到 $(|d|/w)^vK_v(z)$ 的极限为 $2^{v-1}\Gamma(v)(\sigma^2/w)^v$，有限且可算；代码里对 $z<10^{-5}$ 的点走这个分支。**第二个坑是贝塞尔函数的溢出**：$K_v(z)$ 在 $z$ 大时指数衰减，直接用 `scipy.special.kv` 会下溢成 0。要用指数缩放版本 `kve(v,z)=K_v(z)e^z`，并在对数里减掉 $z$——本文初版忘了减这个 $z$，密度直接错到离谱，拟合跑到 $\sigma=10^{-5}$ 的荒谬角落。

严谨起见，我们把解析密度和数值积分 $\int_0^\infty N(x;\mu+\theta g,\sigma^2g)\,\mathrm{Gamma}(g;k,\nu)\,dg$ 逐点对拍：在 $x=-0.020,\ -0.008,\ 0,\ 0.006,\ 0.018$ 五个点上，相对误差分别是 $4.8\times10^{-15}$、$5.4\times10^{-15}$、$5.5\times10^{-12}$、$7.5\times10^{-15}$、$6.6\times10^{-16}$。**公式对了，后面的结论才有意义**——这一步花十分钟，胜过事后调三天参数。

为什么偏偏选 Gamma 当随机时钟？因为它属于广义双曲分布族里最简洁的一个成员：时钟分布换成逆高斯就得到 NIG，换成别的广义逆高斯就得到更灵活的 GH 族。Gamma 时钟是其中参数最少、密度最易算、又足以产生偏斜与肥尾的折中点，而且它有无穷可加性——把两个独立的 VG 时钟加起来还是 Gamma，多日收益的聚合保持同一分布族，这对日频风控是难得的便利。代价是它无法像 GH 那样单独调节尾部与偏度，两者被 $\nu$ 与 $\theta$ 绑在一起，实战中要根据问题取舍。

![拟合密度与样本外收益直方图](/images/variance-gamma-mixture-model/01_density.png)

## 三、赛马设置

真值取 $\mu=0.0003$、$\theta=-0.0015$、$\sigma=0.0075$、$\nu=0.8$，生成 8000 条日收益，前 6000 条训练、后 2000 条测试。三个模型在同一份训练数据上做极大似然：高斯（估均值和标准差）、Student-t（自由度、位置、尺度一起估）、VG（四个参数用 Nelder-Mead 优化，取三组不同初值中最优的一组防止局部极值）。所有评价都在测试集上做，避免「拟合得好」和「预测得好」混为一谈。

## 四、拟合结果与样本外似然

![左尾放大：正态系统性低估极端损失](/images/variance-gamma-mixture-model/02_tail.png)

拟合出的参数是：$\hat\mu=0.000169$、$\hat\theta=-0.001496$、$\hat\sigma=0.007337$、$\hat\nu=0.793$，四个都和真值相当接近，尤其 $\theta$ 和 $\nu$ 几乎分毫不差，说明似然曲面对尾部参数有足够识别力。对照模型方面，高斯均值 -0.001326（正好是 $\mu+\theta=-0.0012$ 附近，验证了前面的一阶矩公式），标准差 0.007500；Student-t 估出自由度 3.822、位置 -0.000940、尺度 0.005391。

样本外每观测对数似然：VG **3.5157**，Student-t **3.5078**，正态 **3.4402**。VG 赢 t 的幅度不大（每观测 0.008，2000 条合计约 15.7 个对数单位），但方向稳定；对正态的优势则是压倒性的。这说明在真值就是 VG 的世界里，t 分布能近似得不错，却始终差一口气——它没法把偏斜和尾部厚度分开调。样本测试集的超额峰度是 3.489（波动抽样使然），真模型理论值 2.605，而高斯样本只有约 0.002，差距的来源一目了然。

## 五、VaR 与 ES：谁在低估尾部

![Gamma 随机时钟示意](/images/variance-gamma-mixture-model/03_mixing.png)

用各模型拟合参数分别模拟 40 万条收益，算 99% 分位数作为 VaR、再取低于该阈值样本的均值作为 ES，和真模型对照：

| 模型 | 99% VaR | 相对真值误差 | 99% ES |
|---|---|---|---|
| 真模型 | -2.357% | — | -2.921% |
| VG 拟合 | -2.319% | -1.6% | -2.865% |
| Student-t | -2.172% | -7.8% | -3.005% |
| 正态 | -1.880% | -20.2% | -2.128% |

VG 几乎是贴着真值走的；Student-t 的 VaR 偏浅但 ES 反而偏深，暴露出它「尾部形状对不准」的毛病；正态则两头都严重低估，ES 只给到 -2.128%，连真值的 73% 都不到。对风控来说，ES 低估意味着一旦发生极端事件，你准备的资金远不够——这比 VaR 低估更危险。

## 六、Kupiec 回测：通过检验才算数

点估计看着漂亮还不够，VaR 是要拿违约次数说话的。在 2000 条测试集上统计四档置信水平下的违约次数，并用 Kupiec 似然比检验判断违约频率是否显著偏离名义水平。

![VaR 违约次数与 Kupiec 检验](/images/variance-gamma-mixture-model/04_backtest.png)

99% 水平：VG 违约 25 次（名义 20 次），p=0.279，**通过**；Student-t 33 次，p=0.0076，**被拒**；正态 51 次，p=$5.6\times10^{-9}$，惨败。95% 水平更能看出结构差异：正态 111 次 p=0.267 侥幸通过，Student-t 却有 134 次 p=0.00089 被拒——t 分布把概率堆在极端尾和中心，中间区域反而偏薄，于是「不太极端但超过 95% 阈值」的损失频繁击穿它。VG 在 95%、97.5%、99%、99.5% 四档全部通过（p 值分别 0.837、0.264、0.279、0.232），说明它的整条分布形状都对，而不只是恰好蒙对了某一个分位数。这才是密度建模相对于「只估一个 VaR」的真正价值。

## 七、完整实现

```python
import numpy as np
from scipy.special import gammaln, kve
from scipy.optimize import minimize
from scipy import stats

MU, THETA, SIGMA, NU = 0.0003, -0.0015, 0.0075, 0.8
rng = np.random.default_rng(1120)

def vg_sim(n, mu=MU, theta=THETA, sigma=SIGMA, nu=NU):
    g = rng.gamma(shape=1/nu, scale=nu, size=n)     # 随机时钟
    return mu + theta*g + sigma*np.sqrt(g)*rng.standard_normal(n)

def vg_logpdf(x, mu, sigma, nu, theta):
    k, w = 1/nu, np.sqrt(theta**2 + 2*sigma**2/nu)
    d, ad = np.asarray(x) - mu, np.abs(np.asarray(x) - mu)
    z, v = w*ad/sigma**2, k - 0.5
    term = np.empty_like(ad)
    small = z < 1e-5
    term[small] = ((v-1)*np.log(2) + gammaln(v) + v*np.log(sigma**2/w))
    zs = np.maximum(z[~small], 1e-300)
    term[~small] = (v*(np.log(ad[~small]) - np.log(w))
                    + np.log(kve(v, zs)) - zs)     # kve 必须减掉 z
    return (theta*d/sigma**2 - np.log(sigma) - 0.5*np.log(2*np.pi)
            - gammaln(k) - k*np.log(nu) + np.log(2) + term)

def fit_vg(x):
    def nll(p):
        mu, ls, lnu, th = p
        sig, nu = np.exp(ls), np.exp(lnu)
        if not (0.02 < nu < 8 and 1e-5 < sig < 0.1):
            return 1e8
        return -vg_logpdf(x, mu, sig, nu, th).mean()
    p0 = [x.mean(), np.log(x.std()*0.8), np.log(1.5), 0.0]
    r = minimize(nll, p0, method='Nelder-Mead',
                 options={'maxiter': 6000, 'xatol': 1e-8, 'fatol': 1e-10})
    mu, ls, lnu, th = r.x
    return mu, np.exp(ls), np.exp(lnu), th

def var_es(samples, q=0.01):
    v = np.quantile(samples, q)
    return v, samples[samples <= v].mean()

def kupiec(viol, n, p):
    ph = viol/n
    lr = -2*((n-viol)*np.log1p(-p) + viol*np.log(p)
             - (n-viol)*np.log1p(-ph) - viol*np.log(ph)) if 0 < ph < 1 else 0.0
    return lr, stats.chi2.sf(lr, 1)
```

用 `var_es()` 时务必从**拟合模型各自模拟**大样本取分位数，别拿训练样本的历史分位数代替——后者混进了抽样噪声，会让你分不清是模型差还是样本少。

## 八、边界与实务建议

把局限说清楚，才不会把合成实验的漂亮数字误当实盘承诺。第一，本文是**静态同分布**模型，没有时变波动；真实收益有明显波动聚集，实务中应把 VG 作为 GARCH 标准化残差的分布，即 `r_t = mu_t + sigma_t * z_t, z_t ~ VG`。第二，VG 的尾部虽肥，但对**极端跳空**仍不如带独立跳跃分量的模型灵活，$\nu$ 要同时管中间厚度和尾部，仍是一种折中。第三，MLE 对初值敏感，$\nu$ 逼近边界时密度数值不稳，建议多初值搜索并检查海森矩阵是否正定。第四，样本只有两千条时尾部参数的标准误不小，本文能精确还原真值是因为数据就由 VG 生成——真实数据里永远要做参数不确定性的敏感性分析。

尽管如此，VG 的思路值得记住：面对复杂的收益分布，与其不断往分布上贴补丁，不如换一个生成机制——**随机时间**这个单一改动，就同时解释了尖峰、肥尾和偏斜，而且似然、采样、风险指标全都能算。对需要给尾部定价的风控岗位来说，这种「机制清楚、参数可解释、检验能过关」的模型，往往比黑箱更好用。
