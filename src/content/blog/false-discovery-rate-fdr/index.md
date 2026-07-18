---
title: "False Discovery Rate：用 BH 程序在多因子检验里控制假阳性"
description: "你筛出 1000 个因子,每个跑 IC 显著性检验,挑 p<0.05 的 95 个——其中 45 个是假阳性,真阳性率不到一半。经典多重检验要么放宽到失控、要么收紧到全灭。Benjamini-Hochberg 用 FDR 在'发现力'和'假阳性率'之间找平衡:同样 1000 个因子,它找出 52 个、只含 2 个假阳性(FDR 0.038),且不漏掉任何一个真信号。附完整 Python 与五类真实陷阱(中阶)。"
publishDate: '2026-07-18'
tags:
  - 量化交易
  - 多重检验
  - FDR
  - Benjamini-Hochberg
  - 假阳性
  - 因子筛选
  - p值
  - Python
language: Chinese
difficulty: intermediate
---

你是一个因子矿工。你手上有 **1000 个候选因子**(动量、估值、质量、文本情绪、各种技术指标变形),每个都算了一遍 IC(信息系数),并做了显著性 t 检验。

你按老办法挑:**p < 0.05 的因子,我就当真有预测力。** 结果挑出 95 个,兴冲冲写进组合。

三个月后实盘:这 95 个因子里,一大半的 IC 塌回 0。**你精心选的"显著因子",大半是运气。**

这不是你一个人的悲剧。这是**多重检验(multiple testing)的必然**——你对 1000 个假设各做一次检验,即使它们全都是噪声,平均也会有 $1000 \times 0.05 = 50$ 个纯靠运气落到 p<0.05。本文用真实合成数据告诉你:朴素阈值下你挑出的 95 个里 **45 个是假阳性**;而 Benjamini-Hochberg(BH, 1995)的 FDR 控制把它压到 **2 个**,且不漏掉任何一个真信号。

## 一、问题:为什么"每个都 p<0.05"还是错一大片

先分清两个你可能混淆的错误率:

- **FWER(族系误差率)**:至少犯**一次**第一类错误(假阳性)的概率。控制 FWER 的方法(Bonferroni)非常严格。
- **FDR(错误发现率, False Discovery Rate)**:在你**最终宣布发现的那一堆**里,假阳性的**比例**。FDR 允许你错,但控制"错的比例"。

金融场景里你要的是 FDR,不是 FWER。为什么?

> 假设你筛 1000 个因子,最终选 50 个进组合。你**宁可**这 50 个里混进 2 个无效的(组合被略微稀释),也**不愿**因为 Bonferroni 把阈值收到 0.05/1000=0.00005 而**一个因子都选不出来**。FDR 是"务实派":我承认会错几个,但保证错的比例可控。

## 二、合成数据:1000 个因子,50 个真有效

为了能"对答案",我们造一组**已知 ground truth** 的数据:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(2026)
m, n_obs, true_sig = 1000, 250, 50
ic_true = 0.05

ic = np.zeros((m, n_obs))
for i in range(m):
    mu = ic_true if i < true_sig else 0.0     # 前 50 个真有预测力
    ic[i] = rng.normal(mu, 0.10, n_obs)

# 每个因子做"IC 是否非零"的单样本 t 检验
tstat = np.sqrt(n_obs) * ic.mean(axis=1) / (ic.std(axis=1, ddof=1) + 1e-12)
pvals = 2 * (1 - stats.norm.cdf(np.abs(tstat)))

is_true = np.arange(m) < true_sig             # ground truth 掩码
```

这样我们**知道**前 50 个是真信号,后 950 个是噪声。后面所有"真阳性/假阳性"都能精确数出来。

## 三、三种校正的实跑对比

```python
def bh_reject(pvals, q=0.05):
    """返回 BH 拒绝集(布尔)与 BH 阈值 p_(k*)。"""
    m = len(pvals)
    order = np.argsort(pvals)
    ps = pvals[order]
    k_star = 0
    for k in range(1, m + 1):
        if ps[k - 1] <= k / m * q:        # BH 临界线
            k_star = k
    rej = np.zeros(m, dtype=bool)
    rej[order[:k_star]] = True
    return rej, (k_star / m * q if k_star else 0.0)

rej_naive, _ = (pvals < 0.05), 0.05
rej_bonf, _  = (pvals < 0.05 / m), 0.05 / m
rej_bh, thr  = bh_reject(pvals, q=0.05)

def tally(rej):
    tp = int((rej & is_true).sum())
    fp = int((rej & ~is_true).sum())
    return tp, fp, fp / max(tp + fp, 1)

for name, rj in [("朴素 α=0.05", rej_naive), ("Bonferroni", rej_bonf), ("BH q=0.05", rej_bh)]:
    tp, fp, fdr = tally(rj)
    print(f"{name:12s}: 发现 {tp+fp:3d}  真阳性 {tp:3d}  假阳性 {fp:3d}  FDR={fdr:.3f}")
print(f"BH 阈值 p_(k*) = {thr:.5f}")
```

跑出来:

```
朴素 α=0.05  : 发现  95  真阳性  50  假阳性  45  FDR=0.474
Bonferroni   : 发现  50  真阳性  50  假阳性   0  FDR=0.000
BH q=0.05     : 发现  52  真阳性  50  假阳性   2  FDR=0.038
BH 阈值 p_(k*) = 0.00260
```

三行数字值得逐字读:

- **朴素**:挑了 95 个,**近一半(45 个)是假阳性**。你以为在选因子,其实在选噪声。
- **Bonferroni**:假阳性 0,但代价是**只发现 50 个,且全靠真信号刚好够显著**——它太严,真实世界里很多弱有效因子(IC=0.03)会被它直接砍掉。
- **BH**:发现 52 个,其中 50 个真、2 个假,**FDR 实测 0.038 < 目标 0.05**,且**一个真信号都没漏**。

![1000 个因子中 50 个真有效: 三种多重检验校正对比](/images/false-discovery-rate-fdr/fdr_methods_compare.png)

> 图 1:绿色真阳性、红色假阳性。朴素方法红了一大片;Bonferroni 干净但保守;BH 几乎全绿,只有零星两个红点。

## 四、BH 到底在干什么:一张排序 p 值图

BH 的美在于它极其简单,且能画出来。把 1000 个 p 值从小到大排,画一条临界线 $y = \frac{k}{m} \cdot q$(k 是排名)。**最后一个"p 值穿过临界线下方"的点,就是拒绝集的边界**——它前面所有的假设都被拒绝。

```python
order = np.argsort(pvals)
ps = pvals[order]
ks = np.arange(1, m + 1)
bh_line = ks / m * 0.05

# 画出 ps vs ks(对数轴),叠加 bh_line 与朴素阈值 0.05
```

![BH 程序：在排序 p 值图上, 最后一个穿过临界线的点决定拒绝集](/images/false-discovery-rate-fdr/bh_sorted_pvals.png)

> 图 2:对数轴上的 p 值散点(绿=真信号,红=噪声)。蓝线是 BH 临界线。**真信号的 p 值(绿点)整体压在更低处,贴着临界线穿过;噪声的 p 值(红点)大多在线上方被挡住。** 那个穿过的交点,精确定位了"该拒绝到哪为止"。

直觉:BH 不是给每个 p 值单独加罚(像 Bonferroni 的 `0.05/m`),而是**让"拒绝的比例"和"p 值的排序位置"对齐**。你越靠后拒绝(越不显著的 p),要求的 p 值越小——这正好惩罚了"为了多发现而硬凑边界"的行为。

## 五、q 是旋钮:想多发现,就接受更多假阳性

BH 的目标 FDR 水平 `q` 是你亲手拧的旋钮。q 越大越宽松:

```python
qs = np.linspace(0.001, 0.20, 60)
disc, fp, tp = [], [], []
for q in qs:
    rj, _ = bh_reject(pvals, q=q)
    disc.append(int(rj.sum())); fp.append(int((rj & ~is_true).sum())); tp.append(int((rj & is_true).sum()))
```

![q 越大越宽松: 发现数上升, 但假阳性同步上升](/images/false-discovery-rate-fdr/fdr_vs_q_curve.png)

> 图 3:q=0.05 处标了竖线。左边 q 很小→发现少;右边 q 大→发现多但假阳性(红线)同步爬升。这条曲线就是你"发现力 vs 纯度"的取舍全景图。

实操建议:

- **因子初筛(宽进)**:q=0.20,宁可多带点噪声进下一轮,别漏真信号。
- **最终入池(严出)**:q=0.05 或更低,保证进组合的因子纯度。
- **分层使用**:先用大 q 粗筛,再对幸存因子用小 q 复检,等价于两阶段 FDR。

## 六、五类真实陷阱

1. **p 值本身不可靠,BH 救不了**:BH 假设你的 p 值来自正确的零分布。如果你的 IC 检验忽略了**自相关**(日频 IC 高度序列相关),t 检验的 p 值本身就偏小, BH 会在错误的基础上"精确地"多拒。先对收益/IC 做 Newey-West 调整或块 bootstrap 再算 p。
2. **依赖结构会轻微抬高 FDR**:BH 的理论保障在 p 值**相互独立**时最干净。因子之间高度相关(动量家族几十个变体)会让实际 FDR 略超 q。补救:用 **Benjamini-Yekutieli** 校正,或先做聚类(把相关因子并成一个"家族")再做 BH。
3. **q 不是"假阳性占比的上界"那么简单**:BH 保证的是**期望** FDR ≤ q(在独立且部分零假设下)。单次实验可能偶尔超过。别把 q=0.05 当"保证最多 5% 假阳性"的硬承诺。
4. **真信号被"挤掉"的边缘效应**:当大量强信号存在时,它们的低 p 值会把临界线拉低,反而可能让一些**弱真信号**被拒。这不是 bug,是 FDR 的固有权衡——弱信号本就难与噪声区分。
5. **不要对 BH 拒绝集再做第二次 BH**:"先 BH 一遍,把拒绝的再 BH 一遍"会破坏校准,实际 FDR 失控。多重阶段请用专门的分层/两阶段程序(如 stratified BH)。

## 七、结论

多重检验不是学术细节,是因子研究的**生死线**。本文用 1000 因子、50 真信号的合成面板给出三个可记的数字:

- **朴素 p<0.05**:发现 95 个,**45 个假阳性(FDR 0.47)**——你选的因子一半是噪声;
- **Bonferroni**:假阳性 0,但**太严,弱信号全灭**;
- **BH(q=0.05)**:发现 52 个,**仅 2 个假阳性(FDR 0.038)**,且不漏真信号。

BH 的价值不是"更准的 p 值",而是把决策从"每个假设单独判"升级到"**对发现集整体控比例**"。在金融这种"宁可错几个、不能全错过"的场景里,这正是你要的尺度。

---

*本文数据均为自洽合成(50 真信号 + 950 噪声,IC 正态基底),仅用于演示方法。真实落地时:p 值先经 Newey-West / 块 bootstrap 校正、相关因子先做聚类再 BH、并对入选因子做样本外验证(purged k-fold)防止过拟合。代码已全部跑通,数字即输出。*
