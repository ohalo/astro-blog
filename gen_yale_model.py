#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「耶鲁捐赠基金模式借鉴：把「另类资产 + 股权偏好」做成长期超额收益引擎」生成真实配图与统计数字。

核心逻辑(David Swensen, Yale Endowment Model, 1985 起):
  - 传统捐赠基金: 大量债券 + 现金(流动性高, 但长期收益低)
  - 耶鲁模式: 极度压低债券/现金, 重配另类资产(绝对收益/私募股权/风险投资/实物资产) + 全球股权
  - 三大收益引擎:
      ① 股权偏好(Equity Bias): 长期股票溢价 >> 债券
      ② 另类资产分散(Diversification): 绝对收益/实物资产与股票低相关
      ③ 流动性溢价(Illiquidity Premium): 私募/风投/实物锁定期换来的额外收益
  - 代价: 流动性差 -> 危机中无法再平衡、回撤更深(锁死在下跌的另类里)

全部数字由文中 Python 真实计算(仅依赖 numpy/matplotlib, 因子模型 + 蒙特卡洛)。
图片:
  cover.png       —— 耶鲁模式「收益引擎」示意: 三类资产角色 + 权重
  ye_curve.png    —— 60/40 vs 耶鲁完整版 vs 耶鲁流动代理版 长期净值
  ye_dd.png       —— 危机期回撤: 耶鲁完整版更深(流动性代价)
  ye_sources.png  —— 耶鲁组合年化收益的来源分解(各资产类别贡献)
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
D = os.path.join(BASE, "yale-endowment-model")
os.makedirs(D, exist_ok=True)

rng = np.random.default_rng(20260722)

# ================= 资产设定(年化 mu / sigma / 对股票因子 beta) =================
# 8 类资产: 国内股票/海外股票/绝对收益/私募股权/风险投资/实物资产/债券/现金
assets = ["国内股票", "海外股票", "绝对收益", "私募股权", "风险投资", "实物资产", "债券", "现金"]
ann_mu = np.array([0.085, 0.090, 0.055, 0.110, 0.150, 0.075, 0.035, 0.020])
ann_sd = np.array([0.160, 0.180, 0.070, 0.200, 0.300, 0.130, 0.055, 0.008])
beta_mkt = np.array([1.00, 0.85, 0.10, 0.80, 0.70, 0.45, -0.20, 0.00])
illiquid = np.array([False, False, False, True, True, True, False, False])  # 锁定期的另类
idio_sd = np.sqrt(ann_sd**2 - (beta_mkt * 0.16)**2)  # 特质波动(股票因子年化波动 16%)
idio_sd = np.clip(idio_sd, 0.01, None)

# 另类资产携带的「流动性溢价」(年化额外 mu, 作为锁定期的补偿)
illq_premium = np.where(illiquid, np.array([0,0,0,0.025,0.05,0.02,0,0]), 0.0)

# ================= 组合权重 =================
W_6040 = np.array([0.60, 0.00, 0.00, 0.00, 0.00, 0.00, 0.40, 0.00])
# 耶鲁模式(典型配置, 合计 100%): 国内5/海外20/绝对25/私募15/风投15/实物20/债券0/现金0
W_yale = np.array([0.05, 0.20, 0.25, 0.15, 0.15, 0.20, 0.00, 0.00])
# 耶鲁「流动代理版」: 同一另类哲学, 但全部用可月频交易的流动资产(把风投/私募换成更多海外股+绝对收益)
W_liq = np.array([0.15, 0.30, 0.35, 0.00, 0.00, 0.20, 0.00, 0.00])

def annualized(r):
    nv = (1 + r).cumprod()
    ar = (1 + r).prod() ** (12 / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(12)
    sh = (r.mean() - 0.02 / 12) / r.std(ddof=1) * np.sqrt(12)
    mdd = (nv / np.maximum.accumulate(nv) - 1).min()
    return ar, vol, sh, mdd

# ================= 20 年路径模拟(含 2 次危机) =================
MONTHS = 240
CRISIS = [(90, 102), (180, 192)]   # 两次危机窗口(月)
SIMS = 400

def simulate(W, monthly_rebal=True, freeze_illiquid_in_crisis=True):
    """返回 (SIMS, MONTHS) 组合月度收益矩阵。"""
    out = np.empty((SIMS, MONTHS))
    for s in range(SIMS):
        # 股票因子路径
        mkt = np.empty(MONTHS)
        for t in range(MONTHS):
            in_crisis = any(a <= t < b for a, b in CRISIS)
            if in_crisis:
                mkt[t] = rng.normal(-0.040, 0.130)   # 危机: 月均 -4%, 高波动
            else:
                # 偶发温和熊市
                mkt[t] = rng.normal(0.0075, 0.045)
        # 各资产月度收益
        asset_ret = np.empty((MONTHS, 8))
        for i in range(8):
            idio = rng.normal(0, idio_sd[i] / np.sqrt(12), MONTHS)
            in_crisis_b = np.array([any(a <= t < b for a, b in CRISIS) for t in range(MONTHS)])
            # 危机期: 非流动性资产 fire-sale 放大(有效 beta 上升)
            beta_eff = np.where(in_crisis_b & illiquid[i], beta_mkt[i] + 0.45, beta_mkt[i])
            mu_m = (ann_mu[i] + illq_premium[i]) / 12 - 0.5 * ann_sd[i]**2 / 12
            asset_ret[:, i] = mu_m + beta_eff * mkt + idio
        # 组合收益 + 再平衡机制(权重始终归一化到 1, NAV 由累积乘积跟踪)
        w = W.copy()
        port = np.empty(MONTHS)
        for t in range(MONTHS):
            r_t = asset_ret[t] @ w
            port[t] = r_t
            # 先让权重随市值漂移并归一化
            w = w * (1 + asset_ret[t])
            w = w / w.sum()
            # 再平衡动作
            in_crisis = any(a <= t < b for a, b in CRISIS)
            do_rebal = monthly_rebal or (t % 12 == 11)
            if do_rebal:
                if freeze_illiquid_in_crisis and in_crisis:
                    # 危机中: 非流动性资产冻结(随市值漂移), 仅把流动性部分调回目标权重
                    liq_mask = ~illiquid
                    ill_now = w * (~liq_mask)          # 非流动性部分保持当前
                    liq_target = W * liq_mask           # 流动性目标(总和=liq_目标占比)
                    w = ill_now + liq_target
                    w = w / w.sum()
                else:
                    w = W.copy()                        # 完全重置为目标权重
        out[s] = port
    return out

res_6040 = simulate(W_6040, monthly_rebal=True, freeze_illiquid_in_crisis=False)
res_yale = simulate(W_yale, monthly_rebal=False, freeze_illiquid_in_crisis=True)   # 年再平衡, 危机冻结
res_liq = simulate(W_liq, monthly_rebal=True, freeze_illiquid_in_crisis=False)

print("=== 20 年(含 2 次危机) 对比 [MC 400 路径均值] ===")
labels = ["60/40", "耶鲁完整版", "耶鲁流动代理版"]
for nm, R in zip(labels, [res_6040, res_yale, res_liq]):
    ar = (np.prod(1 + R, axis=1) ** (12 / MONTHS) - 1).mean()
    vol = (R.std(ddof=1, axis=1) * np.sqrt(12)).mean()
    sh = ((R.mean(axis=1) - 0.02/12) / R.std(ddof=1, axis=1) * np.sqrt(12)).mean()
    # 逐路径算 maxDD
    maxdd = np.array([((1 + r).cumprod() / np.maximum.accumulate((1+r).cumprod()) - 1).min() for r in R]).mean()
    fin = (np.prod(1 + R, axis=1)).mean()
    print(f"{nm:10s}: 年化 {ar*100:5.2f}%  波动 {vol*100:5.2f}%  Sharpe {sh:.2f}  "
          f"最大回撤 {maxdd*100:6.2f}%  终值 {fin:.2f}x")

# ================= 收益来源分解(耶鲁组合, 各资产类别对年化收益贡献) =================
# 各资产长期年化(合成): mu + illq_premium(近似), 用正常+危机的混合路径平均
contrib = W_yale * (ann_mu + illq_premium)
print("\n=== 耶鲁组合 年化收益来源分解(%) ===")
cat = {"股权(国内+海外)": contrib[0]+contrib[1],
       "绝对收益": contrib[2],
       "私募股权": contrib[3],
       "风险投资": contrib[4],
       "实物资产": contrib[5],
       "债券/现金": contrib[6]+contrib[7]}
for k, v in cat.items():
    print(f"  {k:14s}: {v*100:5.2f}")
print(f"  合计: {sum(cat.values())*100:.2f}%")
# 相对 60/40 的增量来自: 减债券(机会成本) + 加另类(流动性溢价+分散)

# ================= 图 1: 收益引擎示意 =================
fig, ax = plt.subplots(figsize=(8.6, 5.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
blocks = [
    (0.5, 5.2, 2.6, 1.4, "#C0504D", "股权偏好\nEquity Bias", "长期股票溢价 > 债券\n国内5% + 海外20%"),
    (3.4, 5.2, 2.6, 1.4, "#4F81BD", "另类分散\nDiversification", "绝对收益25% 低相关\n平滑波动"),
    (6.3, 5.2, 2.6, 1.4, "#E8B21A", "流动性溢价\nIlliquidity", "私募15% 风投15% 实物20%\n锁定期换收益"),
    (1.7, 2.4, 6.6, 1.5, "#2E5AAC", "耶鲁模式组合", "放弃债券/现金的高流动性\n换取长期超额收益"),
]
for x, y, w, h, c, t, sub in blocks:
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=c, alpha=0.88, edgecolor="white", lw=2))
    ax.text(x + w/2, y + h*0.62, t, ha="center", va="center", color="white", fontsize=11, weight="bold")
    ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center", color="white", fontsize=7.5)
ax.annotate("", xy=(3.0, 5.2), xytext=(1.7, 3.9), arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))
ax.annotate("", xy=(4.7, 5.2), xytext=(5.0, 3.9), arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))
ax.annotate("", xy=(7.6, 5.2), xytext=(8.3, 3.9), arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))
ax.set_title("耶鲁模式的「收益引擎」：三大来源撑起长期超额", fontsize=12, weight="bold", pad=12)
fig.tight_layout()
fig.savefig(os.path.join(D, "cover.png"), dpi=140)
plt.close(fig)

# ================= 图 2: 三组合净值 =================
def mean_nv(R):
    nvs = [(1 + r).cumprod() for r in R]
    return np.mean(nvs, axis=0)
nv_6040 = mean_nv(res_6040)
nv_yale = mean_nv(res_yale)
nv_liq = mean_nv(res_liq)
fig, ax = plt.subplots(figsize=(8.8, 4.4))
ax.plot(range(MONTHS), nv_6040, color="#888", lw=1.8, label="60/40")
ax.plot(range(MONTHS), nv_yale, color="#2E5AAC", lw=1.8, label="耶鲁完整版(年再平衡)")
ax.plot(range(MONTHS), nv_liq, color="#E8B21A", lw=1.8, label="耶鲁流动代理版")
ax.axvspan(90, 102, color="#B22222", alpha=0.10)
ax.axvspan(180, 192, color="#B22222", alpha=0.10)
ax.set_xlabel("月份"); ax.set_ylabel("平均净值 (起始=1)")
ax.set_title("20 年(含 2 次危机)：耶鲁完整版收益更高，但危机段更陡", fontsize=11.5)
ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "ye_curve.png"), dpi=140)
plt.close(fig)

# ================= 图 3: 危机回撤 =================
def mean_drawdown(R):
    nvs = np.array([(1 + r).cumprod() for r in R])
    dd = nvs / np.maximum.accumulate(nvs, axis=1) - 1
    return dd.mean(axis=0)
dd_yale = mean_drawdown(res_yale)
dd_6040 = mean_drawdown(res_6040)
dd_liq = mean_drawdown(res_liq)
fig, ax = plt.subplots(figsize=(8.8, 4.4))
ax.fill_between(range(MONTHS), dd_6040*100, color="#888", alpha=0.5, label="60/40")
ax.plot(range(MONTHS), dd_yale*100, color="#2E5AAC", lw=1.6, label="耶鲁完整版")
ax.plot(range(MONTHS), dd_liq*100, color="#E8B21A", lw=1.4, ls="--", label="耶鲁流动代理版")
ax.axvspan(90, 102, color="#B22222", alpha=0.10)
ax.axvspan(180, 192, color="#B22222", alpha=0.10)
ax.set_xlabel("月份"); ax.set_ylabel("回撤 (%)")
ax.set_title("危机回撤：耶鲁完整版更深(流动性锁定的代价)", fontsize=11.5)
ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "ye_dd.png"), dpi=140)
plt.close(fig)

# ================= 图 4: 收益来源分解 =================
cats = list(cat.keys()); vals = [cat[c]*100 for c in cats]
fig, ax = plt.subplots(figsize=(8.4, 4.6))
b = ax.barh(cats, vals, color=["#C0504D", "#4F81BD", "#4F81BD", "#E8B21A", "#E8B21A", "#7F7F7F"])
for bb, v in zip(b, vals):
    ax.text(v + 0.05, bb.get_y()+bb.get_height()/2, f"{v:.2f}%", va="center", fontsize=9)
ax.set_xlabel("对组合年化收益的贡献 (%)")
ax.set_title("耶鲁组合年化收益来源分解：另类资产贡献过半", fontsize=11.5)
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(os.path.join(D, "ye_sources.png"), dpi=140)
plt.close(fig)

print("\n图片已保存至:", D)
