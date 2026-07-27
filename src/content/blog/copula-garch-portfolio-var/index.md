---
title: "Copula-GARCH 组合 VaR：边缘与依赖分开建模的组合风险"
description: "组合 VaR 的经典做法是方差-协方差法：正态边缘 + 一个相关系数矩阵包打天下。但相关系数只刻画线性依赖的平均强度，说不出「两个资产会不会一起崩」。Copula-GARCH（Sklar 定理 + Patton 2006 工程化）把问题拆成三层独立零件：GARCH 管各自的波动动态、t 边缘管各自的肥尾、copula 管它们怎么绑在一起。双资产 GARCH-t + t-copula(ν=4) 模拟实测：三个模型的 95% VaR 全部无罪（突破 70/72/75 次，期望 62.5），99% VaR 立刻分层——方差-协方差法突破 29 次（期望 12.5）Kupiec p=6×10⁻⁵ 处决，换 t 边缘（保留高斯 copula）降到 21 次，再换 t-copula 降到 18 次通过。教训：普通日子里所有模型都差不多，深尾层级上边缘肥尾贡献最大改进、尾部依赖贡献最后一截。附完整 Python 与伪均匀量、Kendall τ 校准、copula 自由度估计等工程细节（中阶）。"
publishDate: '2026-07-27'
tags:
  - 量化交易
  - Copula
  - GARCH
  - 组合VaR
  - 尾部依赖
  - 风险管理
  - Python
language: Chinese
difficulty: intermediate
---

单资产 VaR 的工具链已经很成熟：[GARCH 管波动动态](/blog/garch-filtered-historical-simulation/)，t 分布或经验分位数管肥尾。但组合 VaR 多出一个单资产没有的问题：**资产之间怎么绑在一起**。教科书答案是协方差矩阵——用一个相关系数 ρ 概括两个资产的全部依赖关系。这个答案在正态世界里是对的（多元正态由均值和协方差完全决定），在真实市场里是危险的：**相关系数只度量线性依赖的平均强度，它对"平时相关 0.5、崩盘时一起跳水"和"永远均匀地相关 0.5"给出同一个数字**。而组合的深尾风险恰恰由前者决定。

结论先放这：**Copula-GARCH 框架把组合建模拆成三层可独立更换的零件——GARCH 边缘管各自波动、t 分布管各自肥尾、copula 管联合依赖结构**。双资产模拟实测（真实世界：GARCH-t(6) 边缘 + t-copula ν=4、ρ=0.5，估计窗 750 天，评估窗 1250 天）：**95% VaR 层级三个模型全部通过 Kupiec 检验（突破 70/72/75 次，期望 62.5）；99% 层级立刻拉开——方差-协方差法 29 次突破（期望 12.5，p=6×10⁻⁵）处决，t 边缘 + 高斯 copula 21 次（p=0.028）仍拒绝，t 边缘 + t-copula 18 次（p=0.14）通过**。

## Sklar 定理：把联合分布拧成两截

Copula 方法的数学根基是 Sklar（1959）定理：任何联合分布 $F(x_1, x_2)$ 都可以唯一分解为

$$F(x_1, x_2) = C\big(F_1(x_1),\, F_2(x_2)\big)$$

其中 $F_1, F_2$ 是边缘分布，$C$ 是一个定义在 $[0,1]^2$ 上的 copula 函数——它吃进均匀分布的量，吐出联合概率。翻译成人话：**把每个资产各自的分布形状（肥尾、偏度）和资产之间的依赖结构（谁跟谁一起动）拆开，分别建模，再拼回去**。

这个分解的工程价值在于零件可以独立升级。方差-协方差法等价于"正态边缘 + 高斯 copula"，它错了两处：边缘太瘦 + copula 没有尾部依赖。Copula 框架允许你一次修一处，看清楚每处错误各自的代价——这正是下面实验的设计。

## 尾部依赖：高斯 copula 的结构性缺陷

两个 copula 可以有完全相同的相关系数，但深尾行为天差地别。度量指标是**下尾依赖系数**：

$$\lambda_L = \lim_{q \to 0^+} P\big(U_1 < q \,\big|\, U_2 < q\big)$$

——当资产 2 跌进最极端的 q 分位时，资产 1 也在自己最极端 q 分位里的条件概率。高斯 copula 无论 ρ 多大（只要 <1），**λ 恒等于 0**：极端事件渐近独立，"一起崩"的概率随层级加深衰减到零。t-copula 则有解析的正尾部依赖，本实验校准出的 t-copula（ρ=0.48, ν=3）给出 λ≈0.30——**即使在无穷深的尾部，一个资产崩盘时另一个跟着崩的概率仍有三成**。

![同相关系数下高斯与 t-copula 的散点对比](/images/copula-garch-portfolio-var/copula-scatter.jpg)

左图是实际标准化残差的伪均匀量散点（后文解释怎么来的），左下 5% 联合角里有 17 个点；中图高斯 copula 模拟只有 8 个；右图 t-copula 模拟 14 个。相关系数几乎相同，联合尾角的人口密度差一倍。

条件联合尾概率 $P(U_1<q, U_2<q)/q$ 随 q 变化的曲线把这个差异画得更直白：

![条件联合尾概率曲线](/images/copula-garch-portfolio-var/copula-tail-dependence.jpg)

q→0 时高斯 copula 的曲线一路衰减向 0，t-copula 收敛到 λ≈0.27 的正平台，实际残差的经验曲线贴着 t-copula 走。**组合 VaR 低估的直接来源就是这条差距：正态模型以为分散化在极端日子里仍然有效，实际上极端日子恰恰是分散化失效的日子。**

## 工程管线：五步从收益到组合 VaR

完整流程（Patton 2006 之后的标准做法）：

```python
# 第 1 步：每个资产独立拟合 GARCH(1,1)，QML，方差目标化减一个参数
def qml_garch(x):
    vbar = x.var()
    def nll(p):
        al, be = p
        if al <= 1e-4 or be <= 0 or al + be >= 0.999:
            return 1e9
        om = vbar * (1 - al - be)          # 方差目标化
        s2 = np.empty(len(x)); s2[0] = vbar
        for t in range(1, len(x)):
            s2[t] = om + al*x[t-1]**2 + be*s2[t-1]
        return 0.5 * np.sum(np.log(s2) + x**2/s2)
    # 多起点 Nelder-Mead，避免卡在 alpha=0 的退化解
    ...

# 第 2 步：标准化残差 z = r/sigma，逐资产拟合 t 边缘
z = r / np.sqrt(s2hat)
nu, _, scale = stats.t.fit(z, floc=0)

# 第 3 步：概率积分变换 → 伪均匀量
U = stats.t.cdf(z, nu, 0, scale)

# 第 4 步：在 U 上拟合 copula
#   ρ 用 Kendall τ 反演：rho = sin(pi*tau/2)（对肥尾稳健）
#   copula 自由度 ν 用网格 + профile 似然
tau = stats.kendalltau(U[:,0], U[:,1]).statistic
rho = np.sin(np.pi * tau / 2)

# 第 5 步：蒙特卡洛组合 VaR——从 copula 抽 U，逆变换回残差，
#   乘各自明日波动率预报，加权成组合，读分位数
zz = rng.standard_normal((NSIM, 2)) @ cholesky(R).T
ww = rng.chisquare(nu_cop, NSIM) / nu_cop
uu = stats.t.cdf(zz / np.sqrt(ww)[:, None], nu_cop)   # t-copula 抽样
eps = [stats.t.ppf(uu[:,j], nu[j], 0, sc[j]) for j in (0,1)]
loss = -(np.column_stack(eps) * sigma_next @ weights)
var99 = np.quantile(loss, 0.99)
```

三个值得停下来的细节：

**伪均匀量的质量决定一切**。copula 是在 $U = F(z)$ 上估计的，如果边缘 $F$ 估歪了，U 不均匀，copula 参数跟着歪。本实验一个真实插曲：初版代码拟合 t 分布时错误固定了 scale，边缘自由度被高估到 15 以上（残差显得"不那么肥"），修正为 `floc=0` 自由 scale 后估出 ν≈6.2/7.5，贴近真值 6。**边缘层的小错误会静默传染到依赖层**。

**Kendall τ 反演比矩相关稳健**。肥尾数据的皮尔逊相关被极端点绑架，τ 只依赖次序，反演公式 $\rho = \sin(\pi\tau/2)$ 在椭圆 copula 族内精确成立。

**copula 自由度靠 profile 似然网格**。ν 的似然面很平（3 和 5 的差异要大样本才分得开），网格 + 单参数 profile 比联合优化稳定。本实验从真值 4 估出 3.0——有偏差，但方向上保住了尾部依赖。

## 判决：三个模型追同一条损失序列

评估窗 1250 天，逐日用三个模型算 95%/99% VaR（蒙特卡洛 2 万条路径），与实际组合损失对账，[Kupiec 频率检验](/blog/kupiec-pof-test/)定罪：

![99% VaR 路径对比](/images/copula-garch-portfolio-var/copula-var-paths.jpg)

| 模型 | 95% 突破（期望 62.5） | Kupiec p | 99% 突破（期望 12.5） | Kupiec p |
|---|---|---|---|---|
| 方差-协方差（正态） | 70 | 0.34 ✅ | **29** | **6×10⁻⁵ ❌** |
| t 边缘 + 高斯 copula | 72 | 0.23 ✅ | 21 | 0.028 ❌ |
| t 边缘 + t-copula | 75 | 0.12 ✅ | **18** | **0.14 ✅** |

![突破记分板](/images/copula-garch-portfolio-var/copula-breach-scoreboard.jpg)

三行结果各有一条教训：

**95% 层级是钝刀**。三个模型统统过关——普通日子里正态近似够用，模型差异被推到深尾才显形。只在 95% 层级回测过的组合模型，等于没有测过它真正被雇来管的那部分风险。

**从 29 到 21：边缘肥尾贡献最大的一截改进**。保持高斯 copula 不动、只把正态边缘换成 t 边缘，99% 突破从 29 降到 21。单资产各自的极端波动是深尾损失的第一来源。

**从 21 到 18：尾部依赖贡献最后一截**。换上 t-copula 后，模拟世界里"两个资产同一天都掉进深尾"的联合事件频率终于和真实世界匹配，99% 突破回到不可拒绝区。注意这一截比上一截小——**在 ρ=0.5、双资产的设定下，依赖结构的错误排在边缘肥尾之后**。资产数越多、权重越分散，copula 层的相对重要性越大，因为分散化声称消掉的那部分风险恰恰全押在依赖结构上。

## 已知边界

**t-copula 是对称的**。它给上尾和下尾相同的依赖系数，而真实股市"一起崩"强于"一起涨"（相关性不对称，Longin & Solnik 2001）。修法是 Clayton copula（只有下尾依赖）或 skew-t copula，代价是估计更脆。

**静态 copula 假设依赖结构不随时间变**。本实验的 DGP 恰好如此，真实市场的相关性在危机中会系统性抬升——那是 [DCC-GARCH](/blog/dcc-garch-multivariate/) 或动态 copula（Patton 2006）的领地。

**维度灾难**。t-copula 到几十个资产还能撑（相关矩阵 + 一个自由度），几百个资产就需要因子 copula 或 [vine copula](/blog/vine-copula-dependence/) 的分层构造。

**两阶段估计的效率损失**。先边缘后 copula（IFM 方法）不是联合极大似然，参数不确定性会从第一阶段传到第二阶段，VaR 的置信区间比表面上更宽。

组合风险建模的分工哲学和单资产 FHS 一脉相承：**让每个零件只干自己擅长的事**——GARCH 管动态、t 边缘管肥尾、copula 管联合。方差-协方差法不是"简单所以稳健"，它是三个零件全部焊死在正态假设上，而正态假设恰好在你最需要模型的那些日子里失效。
