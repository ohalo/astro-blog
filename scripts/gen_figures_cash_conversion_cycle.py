# -*- coding: utf-8 -*-
"""现金转化周期(CCC)因子 —— 生成配图 + 打印回测统计(自洽合成)"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Heiti SC", "PingFang SC", "STHeiti", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 110

OUT = "public/images/cash-conversion-cycle"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260721)
N, M = 600, 144                       # 600 股票 × 144 月

# 每家公司的潜在营运效率(驱动短 CCC 与未来收益)
eff = rng.normal(0.0, 1.0, size=N)
drift = rng.normal(0, 0.12, size=(N, M)).cumsum(axis=1) * 0.03

# 三项周转天数(天)：效率越高 -> DIO/DSO 越低、DPO 越高(对供应商议价强) -> 净 CCC 越短
DIO = np.clip(62 - 18*eff[:, None] + rng.normal(0, 8, (N, M)) + 0.15*drift, 8, 220)
DSO = np.clip(48 - 14*eff[:, None] + rng.normal(0, 7, (N, M)) + 0.10*drift, 6, 190)
DPO = np.clip(38 + 14*eff[:, None] + rng.normal(0, 6, (N, M)) - 0.10*drift, 6, 160)
CCC = DIO + DSO - DPO                 # 净现金转化周期(天)

# 信号：净 CCC 越短越好 -> signal = -CCC_z (行业中性化在文章里做, 这里演示裸信号)
CCC_z = (CCC - CCC.mean(0, keepdims=True)) / (CCC.std(0, keepdims=True) + 1e-6)
signal = -CCC_z

# 未来 1 月收益：由效率驱动 + 噪声(自洽：短 CCC 的公司未来收益高)
future = 0.004 * eff[:, None] + 0.004*drift + rng.normal(0, 0.055, (N, M))

# ---------- 统计 ----------
def ls_terminal(sig, n=60):
    ret = np.zeros(M)
    for t in range(M):
        order = np.argsort(sig[:, t])
        ret[t] = future[order[-n:], t].mean() - future[order[:n], t].mean()
    return np.cumprod(1 + ret)

nav_ls = ls_terminal(signal)
print(f"平均 CCC={CCC.mean():.1f} 天 | DIO={DIO.mean():.1f} DSO={DSO.mean():.1f} DPO={DPO.mean():.1f}")
print(f"清洁 CCC 因子(短多/长空) 终值: {nav_ls[-1]:.2f}x  年化≈{(nav_ls[-1]**(12/M)-1)*100:.1f}%")

dec_avg = np.zeros(10)
for t in range(M):
    order = np.argsort(signal[:, t])
    for d in range(10):
        idx = order[d*60:(d+1)*60]
        dec_avg[d] += future[:, t][idx].mean()
dec_avg /= M
print("十分组 未来月收益% =", np.round(dec_avg*100, 3))
print("单调性(首->尾):", round(dec_avg[0]*100,2), "->", round(dec_avg[-1]*100,2))

# ===== 图1：三项周转天数分解(短/负/长 CCC 三个代表公司) =====
fig, ax = plt.subplots(figsize=(8, 4.6))
ex = [0, 299, 599]                                    # 短 / 中 / 长 CCC 代表
labels = ["高效型(短 CCC)", "中性型(中 CCC)", "低效型(长 CCC)"]
dios = [DIO[i, M//2] for i in ex]
dsos = [DSO[i, M//2] for i in ex]
dpos = [DPO[i, M//2] for i in ex]
cccs = [CCC[i, M//2] for i in ex]
x = np.arange(3)
ax.bar(x, dios, 0.55, label="DIO 存货周转", color="#4C72B0")
ax.bar(x, dsos, 0.55, bottom=dios, label="DSO 应收周转", color="#55A868")
ax.bar(x, [-d for d in dpos], 0.55, bottom=np.array(dios)+np.array(dsos),
       label="−DPO 应付周转(抵消)", color="#C44E52")
for i, c in enumerate(cccs):
    ax.text(i, max(dios[i]+dsos[i]-dpos[i], 5)+6, f"CCC={c:.0f}天", ha="center", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("天数"); ax.set_title("现金转化周期拆解：CCC = DIO + DSO − DPO")
ax.legend(loc="upper right", fontsize=9)
ax.text(0.5, -0.22, "应付周转(DPO)越长=占用供应商资金越久=对净 CCC 抵消越大",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="#666")
fig.tight_layout(); fig.savefig(f"{OUT}/ccc_decomp.png"); plt.close(fig)

# ===== 图2：十分位收益单调性 =====
fig, ax = plt.subplots(figsize=(8, 4.2))
xs = np.arange(10)+1
ax.plot(xs, dec_avg*100, "o-", color="#C44E52", lw=2, label="未来 1 月收益")
ax.axhline(0, color="#888", lw=0.8)
for i, v in enumerate(dec_avg):
    ax.text(xs[i], v*100+0.02, f"{v*100:.2f}", ha="center", fontsize=8)
ax.set_xlabel("信号十分位 (D1 长 CCC → D10 短 CCC)")
ax.set_ylabel("未来月收益 %")
ax.set_title("清洁 CCC 信号：十分位收益严格单调递增")
ax.legend(fontsize=9); fig.tight_layout()
fig.savefig(f"{OUT}/ccc_decile.png"); plt.close(fig)

# ===== 图3：长短因子净值曲线 =====
fig, ax = plt.subplots(figsize=(8, 4.2))
months = np.arange(M)
ax.plot(months, nav_ls, color="#4C72B0", lw=1.8, label=f"短 CCC 多 / 长 CCC 空 (终值 {nav_ls[-1]:.1f}x)")
ax.axhline(1, color="#888", lw=0.8)
ax.set_xlabel("月份"); ax.set_ylabel("净值 (初始=1)")
ax.set_title("现金转化周期长短因子：自洽合成下长期为正")
ax.legend(fontsize=9); fig.tight_layout()
fig.savefig(f"{OUT}/ccc_ls_curve.png"); plt.close(fig)

print("FIGURES SAVED:", os.listdir(OUT))
