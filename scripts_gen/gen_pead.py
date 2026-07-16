"""盈余公告漂移 PEAD 配图生成 —— 基于合成财报+价格数据真实计算。
经典结论: 标准化未预期盈余(SUE)越高, 公告后 60 日累计异常收益越高(阶梯)。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_FP = "/System/Library/Fonts/STHeiti Medium.ttc"
fm.fontManager.addfont(_FP)
_CJK = fm.FontProperties(fname=_FP).get_name()
plt.rcParams["font.family"] = _CJK
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(20260717)
OUT = "public/images/post-earnings-drift"
os.makedirs(OUT, exist_ok=True)

ACCENT = "#2563eb"; GREY = "#6b7280"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "normal", "font.weight": "normal",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.dpi": 110,
})

# ---- 合成: 600 只股票 × 48 个季度 ----
n_stocks = 600
n_q = 48
quarters = pd.date_range("2013-03-31", periods=n_q, freq="QE")
true_eps = np.zeros((n_q, n_stocks))
for s in range(n_stocks):
    level = np.random.normal(2.0, 1.0)
    for q in range(n_q):
        if q == 0:
            true_eps[q, s] = level + np.random.normal(0, 0.15)
        else:
            gr = np.random.normal(0.02, 0.06)
            jump = np.random.normal(0, 0.12)
            true_eps[q, s] = true_eps[q-1, s]*(1+gr) + jump

def rolling_sue(eps):
    diff = eps[4:] - eps[:-4]
    std = pd.DataFrame(diff).rolling(8).std().shift(4)
    sue = (eps[4:] - eps[:-4]) / (std.values + 1e-6)
    valid = ~np.isnan(sue).any(axis=1)      # 丢弃滚动标准差尚为 NaN 的早期季度
    return sue[valid]

sue = rolling_sue(true_eps)   # numpy array, shape (n_valid, n_stocks)
n_valid = sue.shape[0]

# 公告后 60 日 CAR: 由 SUE 强度 + 噪声(市场仅部分消化)
n_days = 60
car = np.zeros((n_valid, n_stocks, n_days))
for i in range(n_valid):
    for s in range(n_stocks):
        beta = 0.9
        daily_drift = beta * np.clip(sue[i, s], -3, 3) * 0.02 / n_days
        noise = np.random.normal(0, 0.012, n_days)
        car[i, s] = np.cumsum(daily_drift + noise)

sue_vals = sue
flat_sue = sue_vals.flatten()
flat_car = car.reshape(-1, n_days)
order = np.argsort(flat_sue)
n = len(flat_sue)
deciles = 10
car_by_dec, sue_by_dec = [], []
for d in range(deciles):
    lo = d*n//deciles; hi = (d+1)*n//deciles
    idx = order[lo:hi]
    car_by_dec.append(flat_car[idx].mean(axis=0))
    sue_by_dec.append(flat_sue[idx].mean())
car_by_dec = np.array(car_by_dec)
sue_by_dec = np.array(sue_by_dec)

# 图 1: 各十分位 CAR 曲线(阶梯式分层)
fig, ax = plt.subplots(figsize=(11, 5.2))
cmap = plt.cm.viridis(np.linspace(0, 1, deciles))
for d in range(deciles):
    ax.plot(np.arange(1, n_days+1), car_by_dec[d]*100, color=cmap[d], lw=1.8,
            label=f"D{d+1} (SUE≈{sue_by_dec[d]:+.2f})")
ax.set_title("PEAD：公告后 60 日累计异常收益(CAR) 按 SUE 十分位分层")
ax.set_xlabel("公告后交易日"); ax.set_ylabel("平均 CAR %")
ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/pead-car-by-decile.png"); plt.close(fig)

# 图 2: 多空组合净值(做多最高十分位, 做空最低十分位)
# car_by_dec[d] 本身已是累积异常收益路径, 多空差值直接取差(无需再 cumsum)
long_short = car_by_dec[-1] - car_by_dec[0]
ls_final = long_short[-1]*100
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.plot(np.arange(1, n_days+1), long_short*100, color=ACCENT, lw=2.4)
ax.axhline(0, color="k", lw=0.6, alpha=0.3)
ax.set_title("PEAD 多空净值：做多 SUE 最高十分位 / 做空最低十分位（单期 60 日）")
ax.set_xlabel("公告后交易日"); ax.set_ylabel("累计异常收益差值 %")
fig.tight_layout(); fig.savefig(f"{OUT}/pead-long-short.png"); plt.close(fig)

# 图 3: SUE 分布直方图
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.hist(flat_sue, bins=60, color=ACCENT, alpha=0.85)
ax.axvline(0, color=GREY, lw=1.2, ls="--")
ax.set_title("标准化未预期盈余(SUE) 样本分布")
ax.set_xlabel("SUE"); ax.set_ylabel("股票-季度数")
fig.tight_layout(); fig.savefig(f"{OUT}/pead-sue-distribution.png"); plt.close(fig)

ls_total = ls_final
car_top = car_by_dec[-1][-1]*100
car_bot = car_by_dec[0][-1]*100
print("D10 CAR60=%.2f%%  D1 CAR60=%.2f%%  L-S=%.2f%%" % (car_top, car_bot, ls_total))
print("charts written to", OUT)
