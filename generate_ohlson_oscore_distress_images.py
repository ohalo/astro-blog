#!/usr/bin/env python3
"""
为文章「Ohlson O-score 破产概率建模」(ohlson-oscore-distress) 生成真实配图。
所有图表均由文中 Python 代码真实计算生成。

机制（数据由自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 构造面板：N=600 只股票 × M=144 个月（约 12 年），每只股票有一个持续性
    的潜在财务健康度 q_i（q 越负 → 越接近困境）。
  - 用 Ohlson(1980) 的 9 个变量口径各生成一个与 q 相关的代理，代入标准 O-score
    线性组合，再用 logistic 映射成「未来两年破产/困境概率」P。
  - 未来 1 月收益被设定为随困境概率负相关（困境风险有定价、且高负债公司更脆弱），
    叠加市场因子与噪声。
  - 验证：O-score 截面分布、按 O-score 分桶的「已实现困境频率」、低困境/高困境
    长短因子净值、按 O-score 分的未来收益单调性。
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "ohlson-oscore-distress")
os.makedirs(D, exist_ok=True)

C = {
    "grid": "#DDDDDD",
    "pos": "#2F4B7C",
    "neg": "#C44E52",
    "ls": "#55A868",
    "mk": "#8172B3",
    "gold": "#E1A100",
    "blue": "#4C72B0",
}


# ============================================================
# 1) 合成面板 + Ohlson 9 个变量 + O-score
# ============================================================
def simulate_panel(N=600, M=144, seed=20260720):
    rng = np.random.default_rng(seed)
    # 潜在财务健康度：负=困境倾向
    q = rng.normal(0.0, 1.0, size=N)
    drift = rng.normal(0, 0.12, size=(N, M)).cumsum(axis=1) * 0.04

    def sig(beta, noise=0.6):
        return beta * q[:, None] + 0.3 * drift + rng.normal(0, noise, size=(N, M))

    # 资产规模（用对数规模作为 X1；规模越大 -log 项越负 → 降分，符合大公司更稳）
    TA = np.exp(rng.normal(12.0, 1.0, size=N))[:, None] * np.ones((1, M))
    X1 = np.log(TA)                         # ln(总资产) —— 与困境弱相关

    # X2 总负债/总资产（杠杆）：越困境越高
    X2 = np.clip(0.35 + 0.15 * (-sig(1.0)) + rng.normal(0, 0.05, size=(N, M)), 0.05, 0.95)
    # X3 营运资本/总资产：越困境越低（可为负）
    X3 = np.clip(0.20 + 0.12 * sig(1.0) + rng.normal(0, 0.05, size=(N, M)), -0.4, 0.6)
    # X4 流动资产/流动负债（流动比率）：越困境越低
    X4 = np.clip(2.0 + 0.8 * sig(1.0) + rng.normal(0, 0.25, size=(N, M)), 0.3, 6.0)
    # X5 (息税前净利-优先股)/流动负债：越困境越低
    X5 = np.clip(0.15 + 0.12 * sig(1.0) + rng.normal(0, 0.06, size=(N, M)), -0.5, 0.8)
    # X6 经营活动现金流/总负债：越困境越低
    X6 = np.clip(0.20 + 0.15 * sig(1.0) + rng.normal(0, 0.06, size=(N, M)), -0.3, 0.9)
    # X7 资不抵债哑变量（TL>TA）：困境公司更可能
    neg_worth = (X2 > 1.0)
    X7 = neg_worth.astype(float)
    # X8 连续两年净亏损哑变量：困境公司更可能
    ni = sig(1.2, noise=0.8)               # 净收益代理
    X8 = ((ni < 0) & (np.roll(ni, 1, axis=1) < 0)).astype(float)
    X8[:, 0] = (ni[:, 0] < 0).astype(float)
    # X9 净收益/总资产的变化：越困境越可能下滑
    nit_ta = ni / (TA + 1e-9)
    X9 = (nit_ta - np.roll(nit_ta, 1, axis=1))
    X9[:, 0] = 0.0

    # Ohlson(1980) 标准 O-score 线性组合
    O = (-1.32
         - 0.407 * X1
         + 6.03 * X2
         - 1.43 * X3
         + 0.0757 * X4
         - 1.72 * X5
         - 2.37 * X6
         + 0.285 * X7
         - 1.83 * X8
         + 0.285 * X9)
    # logistic 映射成「未来 2 年困境概率」
    P = 1.0 / (1.0 + np.exp(-O))

    # 市场序列
    mkt = rng.normal(0.005, 0.04, size=M)
    # 未来 1 月收益：随困境概率负相关（低 O-score=安全公司跑赢）
    O_z = (O - O.mean(axis=0, keepdims=True)) / (O.std(axis=0, keepdims=True) + 1e-6)
    future = (0.005 - 0.006 * O_z + 0.35 * mkt
              + rng.normal(0, 0.03, size=(N, M))
              + rng.normal(0, 0.003, size=N)[:, None])
    # 已实现困境事件：随 P 的 Bernoulli（用于校准图）
    distress = (rng.uniform(0, 1, size=(N, M)) < np.clip(P, 0, 0.9)).astype(float)
    return dict(O=O, P=P, X=X1, future=future, mkt=mkt, distress=distress)


PAN = simulate_panel()
O = PAN["O"]
P = PAN["P"]
future = PAN["future"]
mkt = PAN["mkt"]
distress = PAN["distress"]
M = O.shape[1]


# ============================================================
# 图一：O-score 截面分布（右尾=困境区）
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(O.flatten(), bins=60, color=C["blue"], alpha=0.85, edgecolor="white")
ax.axvline(O.flatten().mean(), color="black", lw=1.0, ls="--", label=f"均值 {O.flatten().mean():.2f}")
ax.set_xlabel("O-score（越大 → 困境概率越高）")
ax.set_ylabel("股票-月数")
ax.set_title("O-score 截面分布：右尾聚集高困境概率公司")
ax.grid(True, color=C["grid"], axis="y")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(D, "ohlson_oscore_distribution.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图二：按 O-score 十分位的「已实现困境频率」（模型区分能力）
# ============================================================
dec_dist = np.zeros(10)
dec_p = np.zeros(10)
for t in range(M):
    order = np.argsort(O[:, t])
    for d in range(10):
        idx = order[d * 60:(d + 1) * 60]
        dec_dist[d] += distress[:, t][idx].mean()
        dec_p[d] += P[:, t][idx].mean()
dec_dist /= M
dec_p /= M

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(10), dec_p * 100, "o-", color=C["blue"], lw=2, label="模型预测 P(困境)")
ax.plot(range(10), dec_dist * 100, "s--", color=C["neg"], lw=2, label="已实现困境频率")
ax.set_xlabel("O-score 十分位（D1 安全 → D10 高困境）")
ax.set_ylabel("困境概率 / 频率（%）")
ax.set_title("O-score 十分位：模型预测与已实现困境频率同步单调上升")
ax.grid(True, color=C["grid"])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(D, "ohlson_distress_calibration.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图三：长短因子净值（多安全 / 空困境）
# ============================================================
def ls_curve(signal, n=60):
    ret = np.zeros(M)
    for t in range(M):
        order = np.argsort(signal[:, t])          # 升序：左=安全(低O) 右=困境(高O)
        ret[t] = future[order[:n], t].mean() - future[order[-n:], t].mean()
    return np.cumprod(1 + ret)


nav_ls = ls_curve(O)   # 低 O-score 做多、高 O-score 做空
months = np.arange(M)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(months, nav_ls, color=C["ls"], lw=2.2, label="低困境多/高困境空 长短因子")
ax.set_xlabel("月份（约 12 年）")
ax.set_ylabel("累计净值（起始=1）")
ax.set_title("O-score 长短因子：做多安全、做空困境，长期为正")
ax.grid(True, color=C["grid"])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(D, "ohlson_ls_curve.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图四：按 O-score 十分位的未来收益（单调递减 = 困境溢价）
# ============================================================
dec_avg = np.zeros(10)
for t in range(M):
    order = np.argsort(O[:, t])
    for d in range(10):
        idx = order[d * 60:(d + 1) * 60]
        dec_avg[d] += future[:, t][idx].mean()
dec_avg /= M

fig, ax = plt.subplots(figsize=(8, 5))
colors = [C["neg"] if v < 0 else C["pos"] for v in dec_avg]
ax.bar(range(10), dec_avg * 100, color=colors)
ax.set_xlabel("O-score 十分位（D1 安全 → D10 高困境）")
ax.set_ylabel("平均未来 1 月收益（%）")
ax.set_title("O-score 十分组：收益单调递减，安全组显著跑赢困境组")
ax.grid(True, color=C["grid"], axis="y")
fig.tight_layout()
fig.savefig(os.path.join(D, "ohlson_decile_returns.png"), dpi=130)
plt.close(fig)


print("ohlson-oscore-distress images saved:", sorted(os.listdir(D)))
print(f"O mean={O.flatten().mean():.3f} std={O.flatten().std():.3f}")
print(f"P median={np.median(P):.3f} max={P.max():.3f}")
print(f"decile distress freq % = {np.round(dec_dist*100,2)}")
print(f"decile future ret %  = {np.round(dec_avg*100,3)}")
print(f"ls final = {nav_ls[-1]:.3f}")
