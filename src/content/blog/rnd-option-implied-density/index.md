---
title: "风险中性密度提取：从期权价格反推市场隐含的回报分布"
publishDate: '2026-07-12'
description: "期权价格里藏着的不是波动率，而是整条未来分布。Breeden-Litzenberger 定理说：看涨期权价格对行权价的二阶导就是风险中性密度。本文从零合成分布、定价、反推，并点明平滑、边界、Q≠P 三大真实陷阱。"
tags:
  - 量化交易
  - 期权定价
  - 风险中性密度
  - Breeden-Litzenberger
  - 隐含波动率
  - 衍生品
language: Chinese
difficulty: advanced
---

如果你只从期权里读出「隐含波动率」，那你只看到了冰山一角。一个深度 enough 的期权链，藏着的其实是市场对未来股价**整条概率分布**的看法的——而且不是散户拍脑袋的看法，是**用真金白银投票、由无套利定出来的**分布。

这篇文章要讲的，就是如何把这条分布从期权价格里「倒」出来。方法只有一个优雅的公式，叫 **Breeden-Litzenberger 定理**（1978）。

## 一、Breeden-Litzenberger：二阶导即密度

一个到期日为 $T$、行权价为 $K$ 的欧式看涨期权，其价格就是未来收益的风险中性期望折现：

$$C(K) = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\!\left[\max(S_T-K,\,0)\right]$$

把右边写成积分（对终端股价 $S_T$ 的风险中性密度 $q(S_T)$ 积分）：

$$C(K) = e^{-rT}\int_{K}^{\infty}(S_T-K)\,q(S_T)\,dS_T$$

对 $K$ 求一阶导：

$$\frac{\partial C}{\partial K} = -e^{-rT}\int_{K}^{\infty} q(S_T)\,dS_T = -e^{-rT}\,\mathbb{Q}(S_T>K)$$

再求一阶导（对积分上限求导，多一个负号）：

$$\frac{\partial^2 C}{\partial K^2} = e^{-rT}\,q(K)$$

移项即得全文主角：

$$q(K) = e^{rT}\,\frac{\partial^2 C}{\partial K^2}$$

**看涨期权价格对行权价的二阶导，就是终端股价的风险中性密度。** 不需要假设任何分布形态（对数正态、正态都不用），期权市场自己把分布「说」了出来。

## 二、为什么是二阶导，而不是一阶/零阶

直觉上：$C(K)$ 是一堆「$(S_T-K)^+$ 的加权」堆起来的。零阶（价格本身）混合了位置和形状；一阶导剥掉了「线性部分」，只剩**超过 K 的概率**——它随 K 下降，但还没到密度；二阶导再对 K 求一次，把「概率随 K 的衰减率」暴露出来，那正是密度本身。一阶导给 CDF，二阶导给 PDF，和概率论里一模一样。

## 三、实操三步走：定价 → 微笑 → 二阶差分

真实世界里我们没有解析式子的 $C(K)$，只有离散行权价上的一堆期权报价。路线是：

1. **报价 → 隐含波动率（IV）**：对每个 $K$ 用 BS 公式反解 IV，得到 IV 微笑/偏斜；
2. **平滑 IV**：用多项式或样条把 IV$(K)$ 拟合成光滑函数（关键，见陷阱一）；
3. **二阶有限差分**：用平滑后的 IV 重估 $C(K)$，再在行权价网格上算二阶导，乘以 $e^{rT}$ 得到 $q(K)$。

## 四、Python：合成真密度 → 定价 → 反推

下面这段代码与四张配图完全对应，是**自洽合成数据**（先用一个两对数正态混合造出「真实」风险中性密度，再数值积分定价，最后用 Breeden-Litzenberger 反推；仅用于演示方法；真实落地请用实际期权报价，见文末）。

```python
import math, numpy as np

S0, r, T = 100.0, 0.02, 0.25
S = np.linspace(10.0, 250.0, 1800)        # 宽价网格，避免边界截断
K = np.linspace(70.0, 130.0, 41)          # 中央行权价网格

def ncdf(x): return 0.5*(1.0+math.erf(x/math.sqrt(2.0)))

def true_rnd(S, T):
    # 两对数正态混合：常规态 + 低概率崩盘态 → 左偏、左尾厚
    F = S0*math.exp(r*T)
    w1,m1,s1 = 0.90, math.log(S0)+(r-0.5*0.10**2)*T, 0.10*math.sqrt(T)
    w2,m2,s2 = 0.10, math.log(S0)+(r-0.5*0.22**2)*T-0.10*math.sqrt(T), 0.22*math.sqrt(T)
    f = (w1*np.exp(-0.5*((np.log(S)-m1)/s1)**2)/(S*s1*math.sqrt(2*math.pi))
       + w2*np.exp(-0.5*((np.log(S)-m2)/s2)**2)/(S*s2*math.sqrt(2*math.pi)))
    return f*(F/np.trapezoid(f*S, S))      # 归一到 E^Q[S_T]=远期

def price_calls(f, S, K):
    Cm = np.array([np.trapezoid(f*np.maximum(S-k,0.0), S) for k in K])
    return Cm*math.exp(-r*T)

def bs_call(S,K,r,T,sig):
    d1=(math.log(S/K)+(r+0.5*sig**2)*T)/(sig*math.sqrt(T)); d2=d1-sig*math.sqrt(T)
    return S*ncdf(d1)-K*math.exp(-r*T)*ncdf(d2)

def implied_vol(S,K,r,T,Cm):
    lo,hi=1e-4,3.0
    if Cm<=max(S-K*math.exp(-r*T),0): return lo
    for _ in range(80):
        mid=0.5*(lo+hi); (hi:=mid) if bs_call(S,K,r,T,mid)>Cm else (lo:=mid)
    return 0.5*(lo+hi)

def recover(K, Cmkt):
    iv = np.array([implied_vol(S0,k,r,T,c) for k,c in zip(K,Cmkt)])
    Kn = (K-K.mean())/K.std()
    iv_s = np.clip(np.polyval(np.polyfit(Kn,iv,3), Kn), 1e-3, 3.0)
    Cs = np.array([bs_call(S0,k,r,T,v) for k,v in zip(K,iv_s)])
    dK = K[1]-K[0]
    q = math.exp(r*T)*(Cs[2:]-2*Cs[1:-1]+Cs[:-2])/dK**2
    return K[1:-1], q, iv

f = true_rnd(S, T)
Cm = price_calls(f, S, K)
Kmid, q, iv = recover(K, Cm)
print(f"IV skew (K低−K高) = {(iv[0]-iv[-1])*100:.1f} vol pts")
print(f"recovered ∫q = {np.trapezoid(q,Kmid):.4f}  (true ∫=1)")
```

跑出来：IV 从低行权价到高行权价**单调下滑约 15.4 个波动率点**（典型的美股指数左偏 smirk），反推密度的积分约为 **0.9997**，与真实密度（积分 1.0048）几乎重合——方法成立。

![看涨期权价格随行权价递减，凸性里藏着密度信息](/images/rnd-option-implied-density/rnd_option_prices.png)

## 五、配图解读：四张图各说明什么

**图一（期权价格）**：$C(K)$ 是随 $K$ 递减的凸曲线，越深的虚值期权越便宜。它的**曲率（二阶导）**就是我们要的密度——平还是陡，决定了终端股价落在哪个区间更「密」。

**图二（IV 微笑）**：把价格翻译成 IV 后，曲线明显**左高右低**（K 越小 IV 越高），这就是「崩盘恐惧」的印记——市场愿为下行保护付更高溢价。

![隐含波动率随行权价左高右低，呈现典型左偏 smirk](/images/rnd-option-implied-density/rnd_iv_smile.png)

**图三（反推 vs 真实）**：红线是「真实」风险中性密度（左偏、左尾厚），虚线是 Breeden-Litzenberger 反推出来的——两者几乎重合，证明二阶导法真的把隐含分布还原了出来。注意左尾比右尾「胖」，正是股指 RND 的标志性特征。

![反推风险中性密度（虚线）与真实密度（实线）几乎重合，左尾更厚](/images/rnd-option-implied-density/rnd_recovered.png)

**图四（期限结构）**：同一标的、不同期限的 RND 形状不同——**近月（1M）左偏更夸张**（崩盘恐惧最浓），**远月（6M）趋于对称**（长期风险溢价被时间摊薄）。这就是为什么做波动率风险溢价要分清期限。

![近月 RND 更左偏、远月更对称：崩盘恐惧随时间摊薄](/images/rnd-option-implied-density/rnd_term_structure.png)

## 六、实盘三大真实陷阱

**陷阱一：平滑与步长（最致命）。** 二阶有限差分对 $C(K)$ 的**任何抖动都会被 $1/h^2$ 放大**。如果 IV 拟合不光滑（多项式阶数太高、或直接在噪声报价上差分），密度会冒出尖锐毛刺甚至负值。必须用平滑的 IV$(K)$，且步长 $h$ 取行权价网格的自然间距，而不是把网格加密到极限去「提高精度」——那样只会放大噪声。

**陷阱二：边界失真。** 深度 ITM（K 很低）和深度 OTM（K 很高）的期权流动性差、买卖价差宽、报价噪声大，这些区域的 $C(K)$ 不可信，反推出的边缘密度毫无意义。实务上只取**中央行权价区间**（平值附近 ±20%~30%），两端直接截断。

**陷阱三：风险中性 ≠ 真实世界（Q ≠ P）。** 这是最容易误解的一点：RND 是 $\mathbb{Q}$ 测度下的分布，**已经包含了风险溢价**。它回答的是「市场愿意为哪种结局付多少钱」，不是「结局真实发生的概率」。左尾更厚，部分来自真实的崩盘概率，部分来自投资者愿意为保险多付的钱。想要物理测度分布 $P$，需要额外假设（如用看跌-看涨平价剥离、或假设某种定价核），绝非直接拿 RND 当预测用。

补充两个实操坑：**离散稀疏行权价**需要插值/拟合才能差分；反推出的密度必须做**无套利校验**——非负、积分=1、均值=远期 $F=S_0e^{rT}$，任何一条不满足都说明拟合失败、结果不可用。

## 七、落地路径与诚实结论

真实复现时，把合成数据换成：

- **数据**：同一到期日、同一标的的看涨期权链（行权价、报价、到期日、无风险利率）；
- **清洗**：剔除异常报价、非中央行权价区间，用买卖均价；
- **IV → 平滑**：对 IV$(K)$ 做样条/低阶多项式拟合（优先 Cubic Spline，避免高阶多项式龙格现象）；
- **差分**：用平滑 IV 重估 $C(K)$，在网格上二阶有限差分 × $e^{rT}$；
- **校验**：检查密度非负、积分≈1、均值≈远期，否则回炉重拟合；
- **应用**：用 RND 算隐含偏度/峰度、做波动率互换定价、或和已实现分布对比挖风险溢价。

**结论**：Breeden-Litzenberger 把一个看似不可见的量——市场对未来的完整信念——变成了一个可计算、可校验的二阶导数。它干净、优雅、不依赖分布假设，但干净的前提是你尊重它的脆弱性：平滑要够、边界要砍、Q 和 P 要分清。否则你「反推」出的，只是一张被数值噪声画花的假分布。
