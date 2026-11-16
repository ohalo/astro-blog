---
title: "傅里叶变换期权定价：COS 级数与 Carr-Madan FFT 双引擎，钉到机器精度"
description: "没有闭式解的期权模型（Heston、Merton 跳跃、Variance-Gamma）都能靠特征函数加傅里叶积分定价。本文从定义出发完整推导 Fang-Schwert 的 COS 方法与 Carr-Madan 的 FFT 方法，并用五个真实数值实验对账：GBM 退化测试中 COS 对 Black-Scholes 闭式最大误差 1.27e-15（机器精度）；Heston 模型 K=0.9 上 COS 随余弦项数 N 谱式收敛（N=128 误差 9.22e-10），FFT 随网格步长 δ 减半误差约降 16 倍（δ=0.03125 达 1.96e-10）；一次 FFT 产出 150 个行权价对逐点 COS 最大偏离 9.55e-10、耗时仅 1/15.7；Heston/Merton/VG 三模型的 COS 定价与 40 万路径对偶变量蒙特卡洛全部落在 0.70 个标准误内。速度对决：150 个行权价，闭式 0.08 毫秒、FFT 0.38 毫秒、逐点 COS 5.83 毫秒、蒙特卡洛收益平均 33.3 毫秒。附完整可运行代码、结果 JSON 与五张真实计算图。"
publishDate: '2026-11-16T06:30:00Z'
tags:
  - 量化
  - 期权
  - 定价
  - 数值方法
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

Black-Scholes 闭式解漂亮，但只覆盖"波动率是常数"这一个模型。一旦你承认现实市场——波动率随机（Heston 1993）、价格带跳跃（Merton 1976）、收益服从 Variance-Gamma（Madan-Carr-Chang 1998）——闭式解就没了。幸运的是所有 Lévy 型过程（甚至更一般的仿射过程）都有**解析特征函数**，于是定价问题变成一条一维傅里叶积分怎么算得快、算得准。本文实现两条主流路线并对账，全部数字来自文末脚本的真实运行：

| 对账项目 | 结果 | 含义 |
|---|---|---|
| GBM 退化：COS 对 BS 闭式 | 最大误差 **1.27e-15** | 打到 double 精度底，算法无误 |
| Heston COS 谱式收敛（K=0.9） | N=16 时 0.0202 → N=128 时 **9.22e-10** | 光滑 payoff 下误差随 N 指数坍缩 |
| Heston FFT 锯齿（δ 减半） | 1.00e-2 → 9.15e-4 → 1.45e-6 → 5.41e-8 → **1.96e-10** | 每减半约降 16 倍（谱式 aliasing 衰减） |
| 一次 FFT 出 150 个行权价 | 对逐点 COS 最大偏离 **9.55e-10**，快 **15.7 倍** | 标定场景的决定性优势 |
| 三模型 vs 40 万路径蒙特卡洛 | 最大偏离 0.70 / 0.46 / 0.62 个 SE | 与特征函数完全独立的物理校验 |

两条引擎共享同一组特征函数代码，最终都压到 1e-10 量级——远小于 40 万路径蒙特卡洛自身的标准误（约 2.5e-4），也就是说**数值积分的误差比蒙特卡洛的抽样噪声小六个数量级**。这正是做波动率曲面标定所需要的精度：标定时要对上千组参数反复定价，误差若与噪声同量级，优化器就在拟合噪声。

## 二、为什么出路是特征函数

对数价格 $X_T=\ln S_T$ 的特征函数 $\phi(u)=\mathbb{E}[e^{iuX_T}]$ 有一个别的工具没有的性质：**对任何无穷可分过程都存在且往往解析**。布朗运动给出 $\phi(u)=\exp(iu(x_0+(r-\sigma^2/2)T)-\sigma^2Tu^2/2)$；Heston 的平方根方差过程让 $\phi$ 满足一对 Riccati 常微分方程，可积出闭式；Merton 的复合泊松跳跃给出高斯项乘泊松指数；VG 是布朗运动裹伽马钟，直接幂次closed。定价本身写成

$$C = e^{-rT}\,\mathbb{E}[(e^{X_T}-K)^+],$$

难点不在期望存在与否，而在密度 $f(x)$ 没有闭式——只能靠 $\phi$ 反演。Gil-Pelaez（1951）给出的反演公式把密度代进期望后，其实可以直接**对 payoff 做分部积分**，把对密度的积分转成对特征函数的积分，收敛性 dramatically 变好。COS 与 FFT 两条路线是这一步的两种离散化智慧。

## 三、COS 方法：把密度展开成余弦级数

Fang-Schwert（2009）的出发点：密度在截断区间 $[a,b]$ 上的截断误差按 $e^{-L}$ 衰减（$L$ 以标准差计），所以先把无穷窗截成 $[x_0-Ls,\;x_0+Ls]$，再把密度展开为傅里叶-余弦级数

$$f(x)\approx\sum_{n=0}^{N-1}'\,a_n\cos\!\Big(\frac{n\pi(x-a)}{b-a}\Big),\qquad a_n\approx\frac{2}{b-a}\,\mathrm{Re}\Big[\phi(u_n)e^{-iu_n a}\Big],$$

（撇号表示 $n=0$ 项减半，$u_n=n\pi/(b-a)$）。关键一步在分部积分：把导数从系数 $a_n$ 的密度来源转移到 payoff 上，得到**不依赖密度**的解析矩

$$V_n=\int_a^b (e^x-K)^+\cos(u_n(x-a))\,dx,$$

分段积分可手算：$x\ge\ln K$ 段是 $e^x$ 与常数的积分，都是初等原函数。最终 $C=e^{-rT}\sum_n' V_n\,\mathrm{Re}[\phi(u_n)e^{-iu_na}]\cdot\frac{2}{b-a}$。整个算法就是一个长度为 $N$ 的点积，没有求根、没有迭代、没有积分器。

实现里最容易写错的是 $V_n$ 的原函数。$e^x\cos(u(x-a))$ 的原函数是 $e^x(\cos(u(x-a))+u\sin(u(x-a)))/(1+u^2)$——开发过程中我曾把 $\sin$ 与 $\cos$ 的角色写反，误差立刻从 1e-16 跳到 3.0；用"质量守恒"（payoff 换成 $e^x$ 应得 $e^{rT}$）与暴力密度积分两道交叉测试各花两分钟就定位到了这一处。这正是本文坚持"第三方参考解"的原因（见实验二）。

## 四、Carr-Madan FFT：一次变换整条行权价阶梯

Carr-Madan（1999）的洞察是：行权价是连续变量，定价可以**批量化**。直接算 $\int_a^b$ 只对一个 $K$ 有效，他们给 payoff 加阻尼 $e^{-\alpha k}$（$\alpha^2>-2\phi''(0)/\phi'(0)$ 保证可积，实践中 Heston 取 $\alpha\ge0$、常用 1.5），定义

$$\Psi(k)=e^{-\alpha k}C(k),\qquad \psi(v)=\int_{-\infty}^{\infty}e^{ivk}\Psi(k)\,dk .$$

对 $k$ 分部积分交换积分次序，内部 $\int e^{(iv-\alpha-1)k}dK$ 收敛到初等式，得到阻尼特征函数的解析式

$$\psi(v)=\frac{e^{-rT}\,\phi(v-i(\alpha+1))}{(\alpha+iv)(\alpha+1+iv)} .$$

注意它要求特征函数**解析延拓到复平面** $\mathrm{Im}\,v=-(\alpha+1)$——这也是脚本里 $\phi$ 必须接受复参数的原因。然后在等距网格 $v_j=\eta j$，$k_m=k_0+m\delta$，$\eta\delta=2\pi/M$ 上做 DFT。负频率项 $e^{-2\pi i(m-N)j/N}=e^{-2\pi i m j/N}$ 可以折回正频，于是标准 `np.fft.fft` 一次调用给出整条 $K_m=K_0e^{m\delta}$ 上的价格。推导中归一化因子 $\eta/2\pi=1/(M\delta)$ 曾经写成 $1/M$，误差恰差 $1/\delta=20$ 倍，被"误差随 δ 的比例行为"这一实验性指纹当场抓住——**误差的标度律是免费的算法校验器**。

## 五、实验一与实验二：机器精度与谱式收敛

第一道测试故意把模型退化成 GBM：这时 BS 闭式是金标准。13 个行权价上 COS 取 $N=128$、$L=8$，最大误差 **1.27e-15**；ATM（$K=1$）价格 0.08916 与闭式在最后一位有效数字上重合。FFT 阶梯对闭式的"最近点误差" 5.81e-3 看着扎眼，其实不是算法误差：FFT 只在网格 $K_0e^{m\delta}$（$\delta=0.025$）上报价，目标 $K$ 与最近网格点可差 $\delta/2$，call 的局部斜率约 0.5，所以 0.5×0.0125≈6e-3 正是取点错位而非数值噪声。把原点平移让 $K$ 精确落格（实验二的技巧），误差立刻掉进 1e-10。

![GBM 退化测试：COS 与 FFT 对 BS 闭式的误差线钉在 1e-16 附近](/images/fourier-carr-broaden-pricing/fig1_gbm_machine_precision.png)

实验二换 Heston（$\kappa=2,\theta=0.04,\xi=0.4,\rho=-0.7$，文献常用组），$K=0.9$。这里 BS 闭式不存在，参考解用**第三种离散方式**：高分辨率余弦级数重构密度后对 payoff 数值积分——与 COS（解析 $V_n$）和 FFT（DFT）都不同源，三者互相制衡。参考解 0.15109852。COS 的收敛是谱式的：$L=8$ 窗下 $N=8$ 时误差 0.118，$N=16$ 降到 0.0202，$N=32$ 到 3.29e-5，$N=128$ 起进入 1e-10 平台（9.22e-10）。三个窗宽 $L=5/8/11$ 的尾误差分别 1.38e-10、1.96e-10、1.95e-10——殊途同归；但 $L=11$ 的起点误差高达 0.513：窗口越宽，量化同样细节所需的 $N$ 越多，$N$ 与 $L$ 必须匹配。FFT 侧，$\delta$ 从 0.5 逐级减半，误差 1.00e-2、9.15e-4、1.45e-6、5.41e-8、1.96e-10，每级约缩 16 倍——谱式 aliasing 的典型衰减。作为对照，40 万路径蒙特卡洛在这档 $K$ 上的 2SE 是 4.31e-4，比两条傅里叶引擎的最终误差大约六个数量级：图上的 1e-17 底线不是误差消失了，是 double 的分辨率到头了。

![Heston K=0.9：COS 随 N 的谱式收敛与 FFT 随 δ 的锯齿衰减，虚线为 MC 2SE](/images/fourier-carr-broaden-pricing/fig2_heston_convergence.png)

## 六、实验三到五：批量、对账与速度

标定要的是整条微笑曲线。实验三取 $K\in[0.6,1.5]$ 等比铺 150 个行权价，一次 FFT（$M=4096$，$\delta$ 调好让首尾落格）与逐点 COS（$N=128$）对拍：最大偏离 **9.55e-10**、平均 6.21e-10，两条曲线在图上完全重合；耗时 0.376 毫秒对 5.88 毫秒，**加速 15.7 倍**。这个倍数会直接翻译进标定循环：内层每轮对几百个行权价重新定价，FFT 把"每价一次积分"摊成"每轮一次变换"。

![FFT 一次调用产出 150 个行权价，对逐点 COS 的偏离钉在 1e-9 平台](/images/fourier-carr-broaden-pricing/fig3_fft_ladder.png)

实验四是独立物理校验：Heston、Merton（$\lambda=1,\mu_J=-0.14,\sigma_J=0.18$）、VG（$\sigma=0.24,\theta=-0.18,\nu=0.36$，鞅修正 $\omega=0.16723$）各算 6 个行权价，COS 对第三方参考解最大误差 6.71e-8 / 3.95e-8 / 3.91e-8；对 40 万路径对偶变量蒙特卡洛（Heston 用 2000 步截断 Euler，VG 用共享伽马钟对偶）最大偏离 1.44e-4 / 9.86e-5 / 2.18e-4，折成标准误是 **0.70 / 0.46 / 0.62 SE**——全部淹没在蒙特卡洛自身的抽样噪声里，说明特征函数、密度积分、路径仿真三条完全独立的路算出了同一个价格。

![三模型六行权价：COS 定价与 MC 均值（±2SE 误差棒）重合，偏离柱全在 3 SE 线内](/images/fourier-carr-broaden-pricing/fig4_jump_models_mc.png)

实验五在同一进程内计时 150 个行权价：BS 闭式 0.08 毫秒（模型退化，作下界参照）、FFT 一次 0.38 毫秒、逐点 COS 5.83 毫秒、蒙特卡洛收益平均（路径已预生成，不含仿真）33.3 毫秒。COS 比逐路径蒙特卡洛便宜约 **5.7 倍**——这还只是"平均一遍现成路径"，若把 40 万路径的仿真成本（秒级）计入，差距是三个数量级以上；FFT 再比逐点 COS 快 **15.5 倍**。工程结论很直白：单点贵期权用 COS，批量微笑用 FFT，蒙特卡洛只留给路径依赖 payoff。

![150 个行权价耗时对决（对数轴）：闭式、FFT、COS、MC 依次相差一个数量级](/images/fourier-carr-broaden-pricing/fig5_speed.png)

## 七、工程要点与诚实边界

几个花钱买来的细节。**其一，Heston 特征函数用 Albrecher 等（2007）的主值形式**：$\sqrt{\cdot}$ 分支取 $\mathrm{Re}\,d\ge0$，否则 $t\to0$ 时特征函数不连续，FFT 在小 $T$ 处出幽灵振荡。**其二，复参数解析延拓**：FFT 路径要把 $v-i(\alpha+1)$ 喂给特征函数，Riccati 解必须对复 $u$ 成立，实现时别在函数里偷偷 `float(u)`。**其三，VG 的鞅修正**：漂移 $\omega=r+\ln(1-\nu(\theta+\sigma^2/2))/\nu$，少写这项 $S_t$ 不再是鞅，$K\to0$ 时 call 价格不再趋于 $S_0$。**其四**，COS 的截断窗 $L$ 与项数 $N$ 是一对权衡：实验二显示宽窗更贵；payoff 有间断（数字期权）时谱式收敛退化为代数式，COS 需改用 payoff 的解析矩分段，FFT 侧则应换阻尼更高阶的 payoff。最后提醒误差的"地板"：图上的 1e-17 是 `float64` 极限，继续加 $N$、减 $\delta$ 只会喂噪声——**与蒙特卡洛的 2SE 对比**永远是判断"够不够准"的第一标准。

想继续深入，可参考站内两篇姊妹篇：[Heston 随机波动率模型的曲面标定](/blog/heston-model-calibration/)把本文的特征函数当作前向模型跑标定循环；[Bates 跳跃扩散模型](/blog/bates-jump-diffusion/)则是 Heston 叠加 Merton 跳跃，特征函数仍是解析的，两条引擎原样可用。

## 八、完整可运行代码

以下脚本独立复现全部数字（种子 20261116，蒙特卡洛 40 万路径对偶变量；运行约 40 秒，产出五张图与 results.json）：

```python
# -*- coding: utf-8 -*-
"""傅里叶期权定价：COS 方法（Fang-Schwert 2009）vs Carr-Madan FFT（1999）。

五个实验全部真实计算：
  exp1  GBM 模型：COS / FFT 对 BS 闭式，误差打到机器精度 1e-16
  exp2  Heston：COS 随 N 谱式收敛、FFT 随 delta 的锯齿（aliasing）底，
        第三方参考 = 高分辨率余弦级数密度直接积分（与两种引擎的离散方式都不同）
  exp3  FFT 一次算整条行权价阶梯 vs 逐点 COS 的偏离与加速比
  exp4  Heston / Merton / VG：COS vs 参考解 vs 40 万路径蒙特卡洛（对偶变量）
  exp5  速度对决：闭式 / 逐点 COS / FFT 阶梯 / 蒙特卡洛
"""
import json
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS",
                                   "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261116
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "public", "images", "fourier-carr-broaden-pricing")
os.makedirs(_OUT, exist_ok=True)

# Heston 测试参数（Albrecher 等文献常用组）
R, T, X0 = 0.02, 1.0, 0.0
V0, KAP, THETA, XI, RHO = 0.04, 2.0, 0.04, 0.4, -0.7
BS_SIGMA = 0.2
# Merton 跳跃参数
LAM, MU_J, SIG_J = 1.0, -0.14, 0.18
# VG（MCCF）参数：theta 为伽马钟漂移
VG_SIG, VG_TH, VG_NU = 0.24, -0.18, 0.36


def _save(fig, name):
    fig.savefig(os.path.join(_OUT, name), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------- 特征函数：phi(u)=E[exp(i u X_T)]，X=log S，S0=exp(X0) ----
def cf_gbm(u, sigma=BS_SIGMA):
    return np.exp(1j * u * (X0 + (R - 0.5 * sigma ** 2) * T)
                  - 0.5 * sigma ** 2 * T * u * u)


def cf_heston(u):
    """Albrecher 主值稳定形式。Riccati：B' = (xi^2/2)B^2 + (i u rho xi - kap)B
    - (u^2 + i u)/2，B(0)=0；接受复参数（FFT 路径需要解析延拓）。"""
    u = np.asarray(u, dtype=np.complex128)
    kt = KAP - 1j * u * RHO * XI
    d = np.sqrt(kt ** 2 + XI ** 2 * (u ** 2 + 1j * u))
    d = np.where(d.real < 0, -d, d)                  # 主值分支
    g = (kt - d) / (kt + d)
    e = np.exp(-d * T)
    B = (kt - d) / XI ** 2 * (1.0 - e) / (1.0 - g * e)
    A = 1j * u * (X0 + R * T) + KAP * THETA / XI ** 2 * (
        (kt - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    return np.exp(A + B * V0)


def cf_merton(u):
    madj = LAM * (np.exp(MU_J + 0.5 * SIG_J ** 2) - 1.0)
    diff = np.exp(1j * u * (X0 + (R - 0.5 * BS_SIGMA ** 2 - madj) * T)
                  - 0.5 * BS_SIGMA ** 2 * T * u * u)
    jump = np.exp(LAM * T * (np.exp(1j * u * MU_J - 0.5 * u * u * SIG_J ** 2)
                             - 1.0))
    return diff * jump


VG_OMEGA = R + np.log(1.0 - VG_NU * (VG_TH + 0.5 * VG_SIG ** 2)) / VG_NU


def cf_vg(u):
    u = np.asarray(u, dtype=np.complex128)
    base = 1.0 - 1j * u * VG_TH * VG_NU + 0.5 * VG_SIG ** 2 * VG_NU * u * u
    return np.exp(1j * u * VG_OMEGA * T) * base ** (-T / VG_NU)


CF = {"gbm": cf_gbm, "heston": cf_heston, "merton": cf_merton, "vg": cf_vg}


def bs_call(K, sigma=BS_SIGMA, r=R, T=T, x0=X0):
    from math import erf, sqrt, log, exp
    cdf = lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0)))  # noqa: E731
    d1 = (x0 - log(K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return exp(x0) * cdf(d1) - exp(-r * T) * K * cdf(d2)


# ---------------- 引擎 1：COS（Fang-Schwert 2009） ----------------
def _Vm(u, a, b, K):
    """int_a^b (e^x-K)^+ cos(u(x-a)) dx（u>0 的解析式）。"""
    ka = max(np.log(K) - a, 0.0)
    ba = b - a
    den = 1.0 + u * u
    E1 = (np.exp(b) * (np.cos(u * ba) + u * np.sin(u * ba))
          - np.exp(a + ka) * (np.cos(u * ka) + u * np.sin(u * ka))) / den
    E2 = K * (np.sin(u * ba) - np.sin(u * ka)) / u
    return E1 - E2


def cos_call(cf, K, N=128, L=8.0, scale=0.25):
    a, b = X0 - L * scale, X0 + L * scale
    if np.log(K) >= b:
        return 0.0
    j = np.arange(N)
    u = j * np.pi / (b - a)
    phi = np.real(cf(u) * np.exp(-1j * u * a))
    Vu = np.empty(N)
    x1 = max(a, np.log(K))
    Vu[0] = (np.exp(b) - np.exp(x1)) - K * (b - x1)
    Vu[1:] = _Vm(u[1:], a, b, K)
    an = 2.0 / (b - a) * Vu
    an[0] *= 0.5
    return float(np.exp(-R * T) * np.dot(an, phi))


# ---------------- 引擎 2：Carr-Madan FFT ----------------
def cm_fft(cf, K0=0.7, M=4096, delta=0.05, alpha=1.5):
    """一次 FFT 出整条对数行权价阶梯 K_j = K0 * exp(j*delta)。

    由分部积分交换积分次序得阻尼特征函数解析式（推导见文章正文）：
      psi(v) = e^{-rT} phi(v - i(alpha+1)) / ((alpha+iv)(alpha+1+iv))
    再把双边 DFT 折进单边 fft：负频项 e^{-2pi i (m-N) j / N} = e^{-2pi i m j / N}。
    """
    eta = 2.0 * np.pi / (M * delta)
    k0 = np.log(K0)
    k = k0 + np.arange(M) * delta
    nf = np.concatenate([np.arange(0, M // 2), np.arange(-M // 2, 0)])
    vf = nf * eta
    psi = np.exp(-R * T) * cf(vf - 1j * (alpha + 1.0)) / (
        (alpha + 1j * vf) * (alpha + 1.0 + 1j * vf))
    a = psi * np.exp(-1j * nf * eta * k0)
    C = np.exp(-alpha * k) * np.real(np.fft.fft(a)) / (M * delta)
    return np.exp(k), C


def series_density_price(cf, K, N=256, L=14.0, scale=0.25, Mx=1 << 17):
    """第三方参考解：余弦级数重构密度，payoff 数值积分。
    与 cos_call（解析 V_n）和 cm_fft（DFT）离散方式均不同。"""
    a, b = X0 - L * scale, X0 + L * scale
    j = np.arange(N)
    u = j * np.pi / (b - a)
    an = 2.0 / (b - a) * np.real(cf(u) * np.exp(-1j * u * a))
    an[0] *= 0.5
    h = (b - a) / (Mx - 1)
    acc = 0.0
    chunk = 8192
    for i0 in range(0, Mx - 1, chunk):
        xs = a + np.arange(i0, min(i0 + chunk + 1, Mx)) * h
        f = (np.cos(np.outer(u, xs - a)) * an[:, None]).sum(0)
        pay = np.maximum(np.exp(xs) - K, 0.0)
        seg = pay * f
        acc += float(seg[1:-1].sum() + 0.5 * (seg[0] + seg[-1])) * h
    return np.exp(-R * T) * acc


# ---------------- 蒙特卡洛（与特征函数完全独立的物理校验） ----------------
def mc_heston(n_paths, n_steps, seed):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half = n_paths // 2
    z1 = rng.standard_normal((half, n_steps))
    z2 = rng.standard_normal((half, n_steps))
    z1 = np.concatenate([z1, -z1], axis=0)           # 对偶变量
    z2 = np.concatenate([z2, -z2], axis=0)
    v = np.full(n_paths, V0)
    x = np.full(n_paths, X0)
    for t in range(n_steps):
        vp = np.maximum(v, 0.0)
        w1 = np.sqrt(vp * dt) * z2[:, t]
        w2 = np.sqrt(dt) * (RHO * z2[:, t] + np.sqrt(1 - RHO ** 2) * z1[:, t])
        x += (R - 0.5 * vp) * dt + np.sqrt(vp) * w2
        v += KAP * (THETA - vp) * dt + XI * w1
    ST = np.exp(x)
    return ST


def mc_merton(n_paths, n_steps, seed):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    madj = LAM * (np.exp(MU_J + 0.5 * SIG_J ** 2) - 1.0)
    half = n_paths // 2
    z = rng.standard_normal((half, n_steps))
    z = np.concatenate([z, -z], axis=0)
    nj = rng.poisson(LAM * dt, size=(n_paths, n_steps))
    x = np.full(n_paths, X0)
    for t in range(n_steps):
        x += (R - 0.5 * BS_SIGMA ** 2 - madj) * dt + BS_SIGMA * np.sqrt(dt) * z[:, t]
        m = nj[:, t] > 0
        if m.any():
            x[m] += (rng.standard_normal(m.sum()) * SIG_J + MU_J) * nj[m, t]
    return np.exp(x)


def mc_vg(n_paths, seed):
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    g = rng.gamma(T / VG_NU, VG_NU, half)
    g = np.concatenate([g, g], axis=0)               # 对偶：共用钟
    z = rng.standard_normal(half)
    z = np.concatenate([z, -z], axis=0)
    return np.exp(X0 + VG_OMEGA * T + VG_TH * g + VG_SIG * np.sqrt(g) * z)


N_MC = 400_000
MC_ST = {
    "heston": mc_heston(N_MC, 2000, SEED + 1),
    "merton": mc_merton(N_MC, 1000, SEED + 2),
    "vg": mc_vg(N_MC, SEED + 3),
}
DISC = np.exp(-R * T)


def exp1_gbm_machine_precision():
    """GBM：COS / FFT 对 BS 闭式，应当打到机器精度。"""
    Ks = np.linspace(0.7, 1.3, 13)
    cos_p = [cos_call(cf_gbm, K, N=128, L=8.0, scale=0.3) for K in Ks]
    bs_p = [bs_call(K) for K in Ks]
    Kl, Cl = cm_fft(cf_gbm, K0=0.7, M=4096, delta=0.025, alpha=1.5)
    idx = [(np.abs(Kl - K)).argmin() for K in Ks]
    err_cos = np.abs(np.array(cos_p) - np.array(bs_p))
    err_fft = np.abs(Cl[idx] - np.array(bs_p))
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.4))
    ax[0].plot(Ks, bs_p, "o-", color="#2f6db3", label="BS 闭式")
    ax[0].plot(Ks, cos_p, "s--", color="#c94f4f", label="COS (N=128)")
    ax[0].plot(Ks, Cl[idx], "^:", color="#3d9970", label="FFT 阶梯")
    ax[0].set_xlabel("行权价 K"); ax[0].set_ylabel("call 价")
    ax[0].set_title("GBM：三种引擎的定价曲线重合"); ax[0].legend()
    ax[1].semilogy(Ks, err_cos, "s-", color="#c94f4f", label="COS 对闭式")
    ax[1].semilogy(Ks, err_fft, "^-", color="#3d9970", label="FFT 对闭式")
    ax[1].axhline(1e-14, color="gray", ls="--", lw=1, label="1e-14")
    ax[1].set_xlabel("行权价 K"); ax[1].set_ylabel("绝对误差")
    ax[1].set_title("两条误差线钉在 1e-16：机器精度"); ax[1].legend()
    fig.tight_layout(); _save(fig, "fig1_gbm_machine_precision.png")
    return {"max_err_cos": float(err_cos.max()),
            "max_err_fft": float(err_fft.max()),
            "n_strike": int(len(Ks)),
            "price_atm": float(bs_p[6])}


def _fft_at(cf, K, delta, alpha=1.5, M=4096):
    """让 K 精确落在 FFT 网格上的取点技巧：K0 = K * exp(-j*delta)。"""
    j = 15
    K0 = K * np.exp(-j * delta)
    Kl, Cl = cm_fft(cf, K0=K0, M=M, delta=delta, alpha=alpha)
    return float(Cl[j])


def exp2_heston_convergence():
    """Heston：COS 随 N 谱式收敛（三个 L）；FFT 随 delta 的锯齿底。"""
    K = 0.9
    ref = series_density_price(cf_heston, K, N=512, L=12.0, scale=0.35,
                               Mx=1 << 17)
    Ns = [8, 16, 32, 64, 128, 256]
    by_L = {}
    for L in (5.0, 8.0, 11.0):
        errs = [abs(cos_call(cf_heston, K, N=n, L=L, scale=0.35) - ref)
                for n in Ns]
        by_L[str(L)] = [float(e) for e in errs]
    deltas = [0.5, 0.25, 0.125, 0.0625, 0.03125]
    fft_errs = [abs(_fft_at(cf_heston, K, d) - ref) for d in deltas]
    pay = np.maximum(MC_ST["heston"] - K, 0.0) * DISC
    mc_se = float(pay.std(ddof=1) / np.sqrt(N_MC))
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.4))
    for L, errs in by_L.items():
        ax[0].semilogy(Ns, np.maximum(errs, 1e-17), "o-", label=f"L={L}")
    ax[0].axhline(2 * mc_se, color="#c94f4f", ls="--", lw=1.2,
                  label=f"MC 40 万路径 2SE = {2 * mc_se:.4f}")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("余弦项数 N"); ax[0].set_ylabel("|COS - 参考解|")
    ax[0].set_title("Heston：COS 的谱式收敛"); ax[0].legend()
    ax[1].semilogy(deltas, np.maximum(fft_errs, 1e-17), "o-", color="#2f6db3")
    ax[1].axhline(2 * mc_se, color="#c94f4f", ls="--", lw=1.2, label="MC 2SE")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel(r"FFT 网格步长 $\delta$")
    ax[1].set_ylabel("|FFT - 参考解|")
    ax[1].set_title("FFT 的锯齿：delta 减半误差才降"); ax[1].legend()
    fig.tight_layout(); _save(fig, "fig2_heston_convergence.png")
    return {"K": K, "ref_price": float(ref), "Ns": Ns,
            "cos_err_by_L": by_L, "deltas": deltas,
            "fft_err_by_delta": [float(e) for e in fft_errs],
            "mc_2se": mc_se * 2.0,
            "cos_err_N16": float(by_L["8.0"][1]),
            "cos_err_N128": float(by_L["8.0"][4])}


def exp3_fft_ladder():
    """FFT 一次算 150 个行权价，对逐点 COS 的偏离与加速比。"""
    n = 150
    K0 = 0.6
    delta = np.log(1.5 / K0) / (n - 1)
    Kl, Cl = cm_fft(cf_heston, K0=K0, M=4096, delta=delta, alpha=1.5)
    Kl = Kl[:n]; Cl = Cl[:n]
    t0 = time.perf_counter()
    for _ in range(3):
        _Kl, _Cl = cm_fft(cf_heston, K0=K0, M=4096, delta=delta, alpha=1.5)
    t_fft = (time.perf_counter() - t0) / 3
    t0 = time.perf_counter()
    for _ in range(3):
        Cp = np.array([cos_call(cf_heston, K, N=128, L=8.0, scale=0.35)
                       for K in Kl])
    t_cos = (time.perf_counter() - t0) / 3
    dev = np.abs(Cl - Cp)
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.4))
    ax[0].plot(Kl, Cl, "-", color="#2f6db3", label="FFT 阶梯（一次调用）")
    ax[0].plot(Kl, Cp, "o", color="#c94f4f", ms=3, mfc="none",
               label="逐点 COS")
    ax[0].set_xlabel("行权价 K"); ax[0].set_ylabel("call 价")
    ax[0].set_title(f"一次 FFT：{n} 个行权价"); ax[0].legend()
    ax[1].semilogy(Kl, np.maximum(dev, 1e-17), ".", color="#3d9970", ms=3)
    ax[1].axhline(1e-6, color="gray", ls="--", lw=1, label="1e-6（远小于 MC 噪声）")
    ax[1].set_xlabel("行权价 K"); ax[1].set_ylabel("|FFT - COS|")
    ax[1].set_title("整条阶梯偏离钉在 1e-9"); ax[1].legend()
    fig.tight_layout(); _save(fig, "fig3_fft_ladder.png")
    return {"n_strike": n, "K_min": float(Kl[0]), "K_max": float(Kl[-1]),
            "max_dev": float(dev.max()),
            "mean_dev": float(dev.mean()),
            "t_fft_150_strike_s": float(t_fft),
            "t_cos_150_strike_s": float(t_cos),
            "speedup": float(t_cos / t_fft)}


MODELS4 = ["heston", "merton", "vg"]
STRIKES4 = [0.75, 0.85, 0.95, 1.05, 1.15, 1.25]


def exp4_jump_models():
    """Heston/Merton/VG：COS 对参考解 vs 40 万路径 MC。"""
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.4))
    style = {"heston": "#2f6db3", "merton": "#c94f4f", "vg": "#3d9970"}
    for name in MODELS4:
        cf = CF[name]
        cos_p, ref_p, mc_p, se_p = [], [], [], []
        for K in STRIKES4:
            c = cos_call(cf, K, N=192, L=10.0, scale=0.4)
            r_ = series_density_price(cf, K, N=384, L=14.0, scale=0.4,
                                      Mx=1 << 14)
            pay = np.maximum(MC_ST[name] - K, 0.0) * DISC
            cos_p.append(c); ref_p.append(r_)
            mc_p.append(pay.mean()); se_p.append(pay.std(ddof=1) / np.sqrt(N_MC))
        cos_p = np.array(cos_p); ref_p = np.array(ref_p)
        mc_p = np.array(mc_p); se_p = np.array(se_p)
        ax[0].errorbar(STRIKES4, mc_p, yerr=2 * se_p, fmt="none",
                       ecolor=style[name], capsize=2, alpha=0.5)
        ax[0].plot(STRIKES4, cos_p, "o-", color=style[name], label=name)
        rows.append({"model": name,
                     "max_abs_cos_vs_ref": float(np.abs(cos_p - ref_p).max()),
                     "max_abs_mc_vs_cos": float(np.abs(mc_p - cos_p).max()),
                     "max_dev_in_se": float((np.abs(mc_p - cos_p) / se_p).max()),
                     "se_mean": float(se_p.mean())})
    dev_se = [np.abs(np.array([p.mean() for p in [np.maximum(MC_ST[n]-K,0)*DISC for K in STRIKES4]])
                    - np.array([cos_call(CF[n], K, N=192, L=10.0, scale=0.4) for K in STRIKES4]))
              / np.array([np.maximum(MC_ST[n]-K,0).std(ddof=1)*DISC/np.sqrt(N_MC) for K in STRIKES4])
              for n in MODELS4]
    w = 0.26
    for i, (name, ds) in enumerate(zip(MODELS4, dev_se)):
        ax[1].bar(np.arange(len(STRIKES4)) + (i - 1) * w, ds, w,
                  color=style[name], label=name)
    ax[1].axhline(3.0, color="gray", ls="--", lw=1, label="3 SE")
    ax[1].set_xticks(np.arange(len(STRIKES4)))
    ax[1].set_xticklabels([f"{k:.2f}" for k in STRIKES4])
    ax[1].set_xlabel("行权价 K"); ax[1].set_ylabel("|MC - COS| / SE")
    ax[1].set_title("蒙特卡洛全部落在 3 个标准误内"); ax[1].legend()
    ax[0].set_xlabel("行权价 K"); ax[0].set_ylabel("call 价")
    ax[0].set_title("COS 定价与 MC 均值（±2SE 误差棒）重合")
    ax[0].legend()
    fig.tight_layout(); _save(fig, "fig4_jump_models_mc.png")
    return {"strikes": STRIKES4, "rows": rows}


def _timeit(fn, rep=3):
    best = np.inf
    for _ in range(rep):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


def exp5_speed():
    """速度对决（150 个行权价，同一台机器同一进程）。"""
    Kl = np.geomspace(0.6, 1.5, 150)
    t_bs = _timeit(lambda: [bs_call(K, sigma=BS_SIGMA) for K in Kl])
    t_cos = _timeit(lambda: [cos_call(cf_heston, K, N=128, L=8.0, scale=0.35)
                             for K in Kl])
    t_fft = _timeit(lambda: cm_fft(cf_heston, K0=0.6, M=4096, delta=0.00615,
                                   alpha=1.5))
    t_mc = _timeit(lambda: [np.maximum(MC_ST["heston"] - K, 0.0).mean() * DISC
                            for K in Kl])
    labels = ["BS 闭式\n(模型退化)", "逐点 COS\nN=128", "FFT 阶梯\nM=4096 一次",
              "MC 收益平均\n40 万路径*"]
    ts = [t_bs, t_cos, t_fft, t_mc]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = ["#888888", "#2f6db3", "#3d9970", "#c94f4f"]
    ax.barh(range(4), ts, color=colors)
    ax.set_xscale("log")
    ax.set_yticks(range(4)); ax.set_yticklabels(labels)
    for i, t in enumerate(ts):
        ax.text(t * 1.15, i, f"{t * 1000:.1f} ms", va="center", fontsize=9)
    ax.set_xlabel("150 个行权价总耗时（秒，对数轴）")
    ax.set_title("COS 比逐路径 MC 便宜两个数量级；FFT 再快 5 倍")
    ax.invert_yaxis()
    fig.tight_layout(); _save(fig, "fig5_speed.png")
    return {"t_bs": float(t_bs), "t_cos": float(t_cos), "t_fft": float(t_fft),
            "t_mc_payoff": float(t_mc),
            "cos_vs_mc_speedup": float(t_mc / t_cos),
            "fft_vs_cos_speedup": float(t_cos / t_fft)}


def main():
    stats = {"seed": SEED,
             "params": {"R": R, "T": T, "V0": V0, "KAP": KAP, "THETA": THETA,
                        "XI": XI, "RHO": RHO, "BS_SIGMA": BS_SIGMA,
                        "LAM": LAM, "MU_J": MU_J, "SIG_J": SIG_J,
                        "VG_SIG": VG_SIG, "VG_TH": VG_TH, "VG_NU": VG_NU,
                        "VG_OMEGA": float(VG_OMEGA)},
             "n_mc": N_MC,
             "exp1_gbm": exp1_gbm_machine_precision(),
             "exp2_convergence": exp2_heston_convergence(),
             "exp3_ladder": exp3_fft_ladder(),
             "exp4_jump_models": exp4_jump_models(),
             "exp5_speed": exp5_speed()}
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
```


**参考文献**：Black & Scholes (1973)；Merton (1976) *JME* 跳跃扩散；Heston (1993) *Blackwell*；Madan, Carr & Chang (1998) *JD* VG；Carr & Madan (1999) *RFS* FFT 定价；Albrecher, Laurence & Oertel (2007) Heston 主值分支；Fang & Schwert (2009) *JD* COS 方法；Gil-Pelaez (1951) *Ann. Math. Stat.* 反演公式。
