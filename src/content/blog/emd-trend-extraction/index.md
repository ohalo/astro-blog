---
title: "经验模态分解 EMD 趋势提取：把价格拆成「趋势+一堆周期+噪声」再拼回你要的趋势"
description: "移动平均会滞后、HP 滤波端点漂移、小波要选基——经验模态分解（EMD, Huang et al. 1998）换个思路：不预设基函数，直接按数据自身的局部极值反复「筛」出一组本征模态函数（IMF），高频在前、低频在后，最后一块是趋势残差。本文用 numpy 从零实现 sifting，在「慢饱和趋势+双周期+噪声」合成价格上把趋势 RMSE 压到约 4（接近 HP 滤波的 2.3），并诚实对比 MA / HP，拆穿端点漂移、模态混叠、噪声敏感、停止准则、计算成本五类真实陷阱（中阶）。"
publishDate: '2026-07-21'
tags:
  - 量化交易
  - 趋势提取
  - 经验模态分解
  - EMD
  - 本征模态函数
  - 信号处理
  - Python
language: Chinese
difficulty: intermediate
cover: "/images/emd-trend-extraction/cover.png"
---

你拿到一段价格序列，想抽出「长期趋势」做择时、做均值回归的中轴、或者单纯把噪音滤掉看结构。最朴素的工具是移动平均——但它**滞后**（均线永远慢半拍）、**端点缺数据**（头尾几根线直接断掉）。再进一步是 HP 滤波，可它**端点会漂**（序列末尾被狠狠拽向最后一点）。

有没有一种方法，**不预设任何基函数**（不像小波要挑母小波、傅里叶要整段周期），直接从数据自身的波动里把「趋势 / 周期 / 噪声」一层层剥出来？

结论先放这：**经验模态分解 EMD（Huang et al. 1998）做得到。** 它不要求你先验地知道有几个周期、周期多长，而是靠「局部极大/极小值 → 上下包络线 → 取均值 → 反复筛（sift）」把序列拆成一组**本征模态函数（IMF）**：高频 IMF 在前、低频 IMF 在后，最后剩下的一块**残差就是趋势**。把最低的若干块加起来，就是自适应提取的趋势。**在我们合成的价格（慢饱和趋势 + 双周期 + 噪声）上，EMD 趋势相对真实趋势的 RMSE≈4.0，与 HP 滤波的 2.3 同量级、远优于 MA(80) 的 13.0；且它不被端点拖拽、不依赖预设基。** 诚实地说：**EMD 不是银弹——模态混叠、端点漂移、停止准则主观仍是真问题，后文五类陷阱要一一拆穿，但它确实是趋势提取工具箱里「自适应、不预设结构」的那把利器。**

![EMD 把价格拆成趋势+周期+噪声，再拼回趋势](/images/emd-trend-extraction/cover.png)

---

## 1. 为什么 MA / HP / 小波都不够「自适应」

- **移动平均**：固定窗口，等于假设「趋势是局部常数」。窗口越长越平滑也越滞后，端点直接缺数据。
- **HP 滤波**：解 $\min_\tau \sum (y_t-\tau_t)^2 + \lambda\sum(\Delta^2\tau_t)^2$，靠惩罚项平滑。$\lambda$ 要拍、且**端点严重漂移**（末点被拉向最后观测）。
- **小波 / 傅里叶**：要预设基——小波母函数、傅里叶整段周期。非平稳、突变多的金融序列里，基选错就全错。

EMD 的反叛在于：**基由数据自己长出来**。它只对 IMF 提两个「物理上合理」的要求：
1. 极值点数与过零点数相等（或最多差 1）；
2. 上下包络线的均值处处为 0（局部对称）。

满足这两点的分量，就是「一个固有振荡模态」——IMF。

---

## 2. Sifting：EMD 的核心循环

对信号 $x(t)$，重复以下步骤得到第一个 IMF：

1. 找所有局部极大 / 极小值；
2. 用插值（本文用**端点钳制**的线性插值，避免样条外推炸端点）画出上、下包络线；
3. 取包络均值 $m(t)=\frac12(\text{上包络}+\text{下包络})$；
4. 剥离：$h(t)=x(t)-m(t)$；
5. 若 $h$ 还不符合 IMF 两条（有残余的局部不对称 / 极值数不匹配），把 $h$ 当新 $x$ 回到第 1 步；否则 $h$ 就是第 1 个 IMF。

把这第 1 个 IMF 从原信号减去，对残差重复整套流程，得到 IMF₂、IMF₃……直到残差变成单调或常数——那块**残差就是趋势**。

```python
import numpy as np
from scipy.interpolate import interp1d

def _env_mean(x):
    """上下包络线均值；端点 CLAMP（不外推），避免样条在头尾炸开。"""
    t = np.arange(len(x))
    d = np.diff(np.sign(np.diff(x)))
    maxima = np.where(d[:-1] < 0)[0] + 1
    minima = np.where(d[:-1] > 0)[0] + 1
    if len(maxima) < 2 or len(minima) < 2:
        return x
    fmax = interp1d(maxima, x[maxima], bounds_error=False,
                      fill_value=(x[maxima[0]], x[maxima[-1]]))
    fmin = interp1d(minima, x[minima], bounds_error=False,
                      fill_value=(x[minima[0]], x[minima[-1]]))
    return 0.5 * (fmax(t) + fmin(t))

def _is_imf(x):
    nz = np.sum(np.diff(np.sign(x)) != 0)
    d = np.sign(np.diff(x))
    ne = np.sum((d[1:] != d[:-1]) & (d[1:] != 0))
    return abs(nz - ne) <= 1

def emd(x, max_imf=10, max_iter=100, tol=1e-4):
    x = np.asarray(x, float).copy(); resid = x.copy(); imfs = []
    for _ in range(max_imf):
        h = resid.copy()
        for _ in range(max_iter):
            m = _env_mean(h); h_new = h - m
            if np.max(np.abs(h_new - h)) < tol * (np.std(resid) + 1e-12):
                h = h_new; break
            h = h_new
        if np.std(h) < 1e-9: break
        imfs.append(h); resid = resid - h
        if _is_imf(resid) or np.std(resid) < 1e-7: break
    imfs.append(resid)                 # 最后一块 = 趋势残差
    return imfs

def emd_trend(price, keep_low=2):
    imfs = emd(price); n = len(price); trend = np.zeros(n)
    for imf in imfs[-(keep_low + 1):]:   # 最低频的 keep_low 个 IMF + 残差
        trend += imf
    return trend
```

> 实务提醒：`_env_mean` 用 `fill_value` 把端点钳到最近的极值，**这是 EMD 实战的关键一招**——原始 Huang 用样条端点外推，金融序列头尾常因外推产生巨大假象，钳制能消掉绝大部分端点漂移。

---

## 3. 同一价格的 IMF 分层

我们造一条合成价格：慢饱和趋势（logistic 型爬升）+ 两个周期（120 步的「商业周期」+ 37 步的短振荡）+ 噪声：

```python
def make_signal(n=600, seed=1, noise_sd=0.05):
    rng = np.random.default_rng(seed); t = np.arange(n)
    trend = 100 + 40 / (1 + np.exp(-(t - 300) / 70))      # 慢饱和趋势
    cycle = 6 * np.sin(2 * np.pi * t / 120) + 3 * np.sin(2 * np.pi * t / 37)
    noise = noise_sd * rng.standard_normal(n) * np.std(trend)
    return t.astype(float), trend + cycle + noise, trend, cycle
```

把 EMD 拆出的所有 IMF 画出来——注意能量（std）从高频 IMF₁ 到低频 IMF 的变化：

![同一价格的 IMF 分层：高频在前、低频在后](/images/emd-trend-extraction/emd_imfs.png)

在我们这条 600 点序列上，EMD 拆出约 9 块：前面几块 std 在 0.2–0.5（高频噪声与短周期），中间有几块 std≈1（对应 37 步短周期与 120 步周期），最后一块 std≈14、**就是纯趋势残差**。IMF 按「局部振荡频率」自然排序——这是 EMD 最迷人的特性：**你不用告诉它周期在哪，它自己按频率分层**。

---

## 4. EMD 趋势 vs MA / HP：不被端点拖拽

把 EMD 趋势（取最低频 2 块 + 残差）和三类基准放一起比：

```python
from scipy.linalg import solve_banded
def hp_filter(y, lam=1600.0):
    y = np.asarray(y, float); n = len(y)
    A = np.zeros((n, n))
    for i in range(n - 2):
        A[i+1, i] += lam; A[i+1, i+1] -= 2*lam; A[i+1, i+2] += lam
    A += np.eye(n)
    ab = np.zeros((5, n))
    for j in range(n):
        for off in (-2,-1,0,1,2):
            i = j + off
            if 0 <= i < n: ab[2+off, j] = A[i, j]
    return solve_banded((2, 2), ab, y)
```

![EMD 趋势 vs HP / 移动平均](/images/emd-trend-extraction/emd_compare.png)

相对真实趋势的 RMSE（越小越好）：

| 方法 | EMD (keep_low=2) | HP 滤波 | 长期 MA(80) | 短期 MA(20) |
|---|---|---|---|---|
| RMSE | **3.60** | 2.34 | 12.99 | 7.52 |

读这张表要诚实：
- **HP 略优于 EMD**（2.3 vs 3.6）——因为 HP 是「为平滑量身定做」的凸优化，而我们这条趋势正好是它擅长的平滑曲线；
- **EMD 远胜 MA**（3.6 vs 13.0 / 7.5）——MA 的滞后与端点缺失让它在这条慢变趋势上误差巨大；
- **EMD 的关键优势不在「最小 RMSE」，而在「自适应 + 不预设基 + 端点稳」**：它没被告知周期长度、没被调 $\lambda$，却把趋势抓得和 HP 同量级，且对序列末尾的处理不依赖外推。

---

## 5. 对噪声的鲁棒性

把噪声强度从 0.02 调到 0.20（相对趋势标准差），看 EMD 趋势 RMSE 怎么变：

![EMD 趋势对噪声的鲁棒性](/images/emd-trend-extraction/emd_robust.png)

RMSE 随噪声：`0.02→4.11`、`0.05→4.05`、`0.10→3.91`、`0.20→4.09`。**噪声翻倍，趋势误差几乎不动（稳定在 ~4）**——因为 EMD 把高频噪声单独筛进前几块 IMF，提取趋势时只取最低频块，天然对加性噪声有隔离作用。这正是它比简单平滑更适合「脏数据」的原因。

---

## 6. 五类真实陷阱（中阶）

1. **模态混叠（mode mixing）**：若某个 IMF 里混进了本该属于另一频率的成分（比如一个突变同时污染多个 IMF），趋势提取会失真。标准解法是 EEMD（集合 EMD）——对数据加多次不同白噪声后做集合平均，把混叠「平均掉」。本文基础 EMD 不集合，突变多的真实价格上要警惕。
2. **端点漂移**：哪怕用了钳制，序列头尾几个点仍因极值稀疏而不稳。长周期趋势的端点尤其要复核——可补一段镜像/延拓再分解，或只信任中段趋势。
3. **停止准则主观**：「筛到什么时候算 IMF」靠 tol（本文 1e-4）和最大迭代次数。tol 太松→IMF 没筛干净（残存不对称）；太紧→迭代爆炸、把噪声当 IMF。需对 tol 做敏感性检查。
4. **噪声敏感与 IMF 数量**：噪声大时前几块 IMF 几乎全是噪声，取哪几块当「趋势」要按频率（而非固定块数）判断。本文 `keep_low=2` 是在合成数据上调出的，真实数据上应画 IMF 谱、按拐决定。
5. **计算成本**：每次 sift 都要插值包络，长序列（万点以上日线、或 ticks）比 HP / MA 慢一到两个数量级。实务上先降采样或只对中段做 EMD，再插值回原频。

---

## 7. 实战落地点：EMD 能做什么、不能做什么

EMD 在量化工作流里最适合三类场景，但边界同样清晰：

- **自适应趋势中轴**：做均值回归 / 通道策略时，EMD 给出的趋势比 MA 滞后小、比 HP 端点稳，可直接当「公允价值中轴」；尤其是非平稳、结构突变多的品种。
- **周期 / 状态分解**：把 IMF 按频率分组（高频=噪声、中频=交易周期、低频=商业周期、残差=趋势），能直观看「当前波动是噪声还是真周期」——比硬套傅里叶基灵活。
- **去噪预处理的「保真」版**：需要滤高频噪声但不想动低频结构时，EMD 只丢前几块 IMF、保留所有低频信息，比一刀切的 MA 平滑更少扭曲趋势形状。

但它**不能**直接告诉你「现在该买还是该卖」。EMD 只是把信号拆开，趋势块本身不携带方向预测；而且模态混叠、端点不稳会直接污染趋势。把 EMD 当成「把价格拆成可解释分量」的拆解工具，趋势块再喂给你的择时 / 回归逻辑，而不是把它当决策黑箱。

## 8. 完整 Python 代码

与本文全部数字、配图一一对应的端到端复现脚本（自洽合成数据，仅演示方法；真实落地请替换为真实价格、按需上 EEMD 与端点延拓）：

```python
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import solve_banded

def make_signal(n=600, seed=1, noise_sd=0.05):
    rng = np.random.default_rng(seed); t = np.arange(n)
    trend = 100 + 40 / (1 + np.exp(-(t - 300) / 70))
    cycle = 6 * np.sin(2 * np.pi * t / 120) + 3 * np.sin(2 * np.pi * t / 37)
    noise = noise_sd * rng.standard_normal(n) * np.std(trend)
    return t.astype(float), trend + cycle + noise, trend, cycle

def _env_mean(x):
    t = np.arange(len(x)); d = np.diff(np.sign(np.diff(x)))
    maxima = np.where(d[:-1] < 0)[0] + 1; minima = np.where(d[:-1] > 0)[0] + 1
    if len(maxima) < 2 or len(minima) < 2: return x
    fmax = interp1d(maxima, x[maxima], bounds_error=False,
                      fill_value=(x[maxima[0]], x[maxima[-1]]))
    fmin = interp1d(minima, x[minima], bounds_error=False,
                      fill_value=(x[minima[0]], x[minima[-1]]))
    return 0.5 * (fmax(t) + fmin(t))

def _is_imf(x):
    nz = np.sum(np.diff(np.sign(x)) != 0); d = np.sign(np.diff(x))
    ne = np.sum((d[1:] != d[:-1]) & (d[1:] != 0))
    return abs(nz - ne) <= 1

def emd(x, max_imf=10, max_iter=100, tol=1e-4):
    x = np.asarray(x, float).copy(); resid = x.copy(); imfs = []
    for _ in range(max_imf):
        h = resid.copy()
        for _ in range(max_iter):
            m = _env_mean(h); h_new = h - m
            if np.max(np.abs(h_new - h)) < tol * (np.std(resid) + 1e-12):
                h = h_new; break
            h = h_new
        if np.std(h) < 1e-9: break
        imfs.append(h); resid = resid - h
        if _is_imf(resid) or np.std(resid) < 1e-7: break
    imfs.append(resid); return imfs

def emd_trend(price, keep_low=2):
    imfs = emd(price); n = len(price); trend = np.zeros(n)
    for imf in imfs[-(keep_low + 1):]: trend += imf
    return trend

def hp_filter(y, lam=1600.0):
    y = np.asarray(y, float); n = len(y); A = np.zeros((n, n))
    for i in range(n - 2):
        A[i+1, i] += lam; A[i+1, i+1] -= 2*lam; A[i+1, i+2] += lam
    A += np.eye(n); ab = np.zeros((5, n))
    for j in range(n):
        for off in (-2,-1,0,1,2):
            i = j + off
            if 0 <= i < n: ab[2+off, j] = A[i, j]
    return solve_banded((2, 2), ab, y)

for sd in [0.02, 0.05, 0.10, 0.20]:
    t, price, trend, _ = make_signal(600, 3, sd)
    tr = emd_trend(price, 2)
    print(f"noise={sd} EMD RMSE={np.sqrt(np.mean((tr-trend)**2)):.3f}")

t, price, trend, _ = make_signal(600, 2, 0.05)
tr, hp = emd_trend(price, 2), hp_filter(price, 1600.0)
ma80 = np.convolve(price, np.ones(80) / 80, mode="same")
ma20 = np.convolve(price, np.ones(20) / 20, mode="same")
rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))
print(f"RMSE: EMD={rmse(tr, trend):.2f} HP={rmse(hp, trend):.2f} "
      f"MA80={rmse(ma80, trend):.2f} MA20={rmse(ma20, trend):.2f}")
```

> 真实落地提示：突变多的数据上用 EEMD（加噪集合平均）抑制模态混叠；头尾做镜像延拓再分解以稳端点；tol 与 keep_low 做敏感性检查；万点以上先降采样。EMD 是「自适应拆解」工具，趋势块要再喂给你的择时 / 回归逻辑，而非直接当决策黑箱。
