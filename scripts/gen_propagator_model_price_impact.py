import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(2026)
OUT = "public/images/propagator-model-price-impact"
os.makedirs(OUT, exist_ok=True)

BLUE = "#3b6ea5"; RED = "#c0392b"; GREEN = "#27ae60"; ORANGE = "#e67e22"; PURPLE = "#8e44ad"; GRAY = "#7f8c8d"

# ============================================================
# 图 1：单笔成交的冲击核 G(t)——幂律衰减 vs 指数衰减 vs 永久冲击
# 传播子模型的核心：一笔成交的价格冲击不是瞬时的，而是按 G(t) 随时间衰减
# ============================================================
lag = np.arange(1, 200)
G_power = 1.0 / (lag ** 0.5)          # 幂律衰减 G(t) ~ t^{-beta}, beta=0.5
G_exp = np.exp(-lag / 15.0)           # 指数衰减
G_perm = np.ones_like(lag, dtype=float) * 0.35  # 永久冲击（不衰减）

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
ax[0].plot(lag, G_power, color=RED, lw=2, label=r"幂律 $G(t)\sim t^{-0.5}$（传播子）")
ax[0].plot(lag, G_exp, color=BLUE, lw=2, label=r"指数 $G(t)\sim e^{-t/15}$")
ax[0].plot(lag, G_perm, color=GRAY, lw=2, ls="--", label="永久冲击（不衰减）")
ax[0].set_title("冲击核 G(t)：一笔成交的冲击如何随时间消退", fontsize=12)
ax[0].set_xlabel("成交后经过的事件数 t"); ax[0].set_ylabel("残余冲击 G(t)")
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.25)

# log-log 看幂律的直线特征
ax[1].loglog(lag, G_power, color=RED, lw=2, label="幂律（log-log 上是直线）")
ax[1].loglog(lag, G_exp, color=BLUE, lw=2, label="指数（log-log 上迅速下坠）")
ax[1].set_title("log-log 坐标：幂律尾 vs 指数尾", fontsize=12)
ax[1].set_xlabel("t (log)"); ax[1].set_ylabel("G(t) (log)")
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.25, which="both")
plt.tight_layout(); plt.savefig(f"{OUT}/impact_kernel.png", dpi=110); plt.close()

# ============================================================
# 图 2：传播子模型价格构造 —— 价格 = 所有历史成交冲击的叠加
# p(t) = sum_{s<t} G(t-s) * f(v_s) * eps_s
# ============================================================
N = 300
signs = rng.choice([-1, 1], size=N)          # 成交方向
vols = rng.gamma(2.0, 1.0, size=N)           # 成交量
f_v = np.sqrt(vols)                          # 冲击强度 ~ sqrt(volume)（凹）
Y0 = 100.0                                   # 冲击尺度系数(bp)

# 预计算幂律核
beta = 0.5
def kernel(dt):
    return 1.0 / np.power(dt, beta)

# 构造价格：每个时点是过去所有成交冲击的加权和
price = np.zeros(N)
for t in range(N):
    contrib = 0.0
    for s in range(t):
        dt = t - s
        contrib += kernel(dt) * f_v[s] * signs[s]
    price[t] = Y0 * 1e-4 * contrib  # 转成价格增量
price = 100.0 + np.cumsum(np.zeros(N)) + price  # 基准价 100 + 冲击叠加

# 也画出单笔冲击的贡献分解（挑第 50 笔大单）
big = np.argmax(f_v[:100])
single = np.zeros(N)
for t in range(big + 1, N):
    single[t] = Y0 * 1e-4 * kernel(t - big) * f_v[big] * signs[big]

fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(price, color=BLUE, lw=1.6, label="传播子模型价格（所有成交冲击叠加）")
ax.axvline(big, color=RED, ls=":", lw=1.2)
ax.plot(np.arange(N), 100.0 + single, color=ORANGE, lw=1.5, alpha=0.8,
        label=f"第 {big} 笔大单的单独冲击轨迹（衰减中）")
ax.set_title("传播子模型：价格是历史成交冲击按 G(t) 衰减后的累加", fontsize=12)
ax.set_xlabel("事件时间（成交序号）"); ax.set_ylabel("价格")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/price_construction.png", dpi=110); plt.close()

# ============================================================
# 图 3：订单流自相关的长记忆 —— 传播子模型为什么必须存在
# 成交方向 sign 有长记忆（幂律自相关），若冲击永久则价格会爆炸/趋势过强
# ============================================================
# 造一个有长记忆的订单流（用分数差分近似：正相关持续）
def long_memory_signs(n, gamma=0.5, seed=1):
    r = np.random.default_rng(seed)
    # 简单 AR-like 长记忆：新方向以概率 p 延续上一个
    s = np.zeros(n)
    s[0] = r.choice([-1, 1])
    for i in range(1, n):
        # 延续概率随机但整体偏高，制造正自相关
        p_cont = 0.62
        s[i] = s[i-1] if r.random() < p_cont else -s[i-1]
    return s

lm = long_memory_signs(5000)
# 计算自相关
def acf(x, maxlag):
    x = x - x.mean()
    denom = np.sum(x*x)
    out = []
    for k in range(1, maxlag+1):
        out.append(np.sum(x[:-k]*x[k:]) / denom)
    return np.array(out)

maxlag = 100
ac_lm = acf(lm, maxlag)
# 对照：独立订单流
indep = rng.choice([-1,1], size=5000)
ac_indep = acf(indep, maxlag)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
lags = np.arange(1, maxlag+1)
ax[0].plot(lags, ac_lm, color=RED, lw=1.8, label="真实订单流（长记忆）")
ax[0].plot(lags, ac_indep, color=GRAY, lw=1.2, alpha=0.7, label="独立订单流（无记忆）")
ax[0].axhline(0, color="k", lw=0.6)
ax[0].set_title("成交方向的自相关：订单流有长记忆", fontsize=12)
ax[0].set_xlabel("滞后 lag"); ax[0].set_ylabel("方向自相关 C(lag)")
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.25)

# 若冲击永久，长记忆订单流会让价格产生强趋势（可预测→套利）；
# 传播子的衰减核恰好抵消长记忆，让价格接近鞅（无套利）
# 演示：永久冲击 vs 传播子冲击下的价格轨迹
lm2 = long_memory_signs(400, seed=7)
perm_price = np.cumsum(lm2) * 0.02 + 100        # 永久冲击：直接累加→强趋势
# 传播子：带幂律衰减核
prop_price = np.zeros(400)
for t in range(400):
    c = 0.0
    for s in range(max(0, t-60), t):  # 截断到 60 步加速
        c += (1.0/np.sqrt(t-s)) * lm2[s]
    prop_price[t] = 100 + c * 0.05
ax[1].plot(perm_price, color=ORANGE, lw=1.6, label="永久冲击：长记忆→强趋势（可套利）")
ax[1].plot(prop_price, color=BLUE, lw=1.6, label="传播子衰减：价格接近鞅（无套利）")
ax[1].set_title("为什么冲击必须衰减：抵消订单流长记忆", fontsize=12)
ax[1].set_xlabel("事件时间"); ax[1].set_ylabel("价格")
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/orderflow_memory.png", dpi=110); plt.close()

# ============================================================
# 图 4：核函数标定 —— 从模拟成交数据反推 G(t)
# 用响应函数 R(l) = E[ (p_{t+l}-p_t) * sign_t ] 估计冲击的滞后响应
# ============================================================
# 用图2的模拟数据算响应函数
def response_function(price, signs, maxl):
    R = []
    for l in range(1, maxl+1):
        dp = price[l:] - price[:-l]
        s = signs[:-l]
        R.append(np.mean(dp * s))
    return np.array(R)

# 重新造一段更长、信噪比更高的传播子价格用于标定
Nc = 20000
sg = rng.choice([-1, 1], size=Nc).astype(float)
# 注入长记忆（方向持续）
for i in range(1, Nc):
    if rng.random() < 0.35:
        sg[i] = sg[i-1]
fv = np.sqrt(rng.gamma(2.0, 1.0, size=Nc))
# 用幂律核构造价格增量（截断 100 步加速）
KW = 100
ker = 1.0 / np.sqrt(np.arange(1, KW + 1))     # G(t) = t^{-0.5}
contrib = fv * sg
dp = np.zeros(Nc)                             # 每步价格增量 = 过去成交冲击的滞后叠加
for t in range(Nc):
    lo = max(0, t - KW)
    k = ker[:t - lo][::-1]
    dp[t] = np.dot(k, contrib[lo:t])
pc = np.cumsum(dp) * 0.01
pc = pc + rng.standard_normal(Nc) * 0.05      # 加独立微观结构噪声（不累积）

maxl = 120
R = response_function(pc, sg, maxl)
lags2 = np.arange(1, maxl + 1)
# 传播子模型的响应函数应随 l 先升后趋平（凹饱和），用饱和形式拟合
from scipy.optimize import curve_fit
def sat(l, a, b): return a * (1.0 - np.exp(-l / b))
try:
    popt, _ = curve_fit(sat, lags2, R, p0=[R[-1], 8.0], maxfev=5000)
    fit = sat(lags2, *popt); tau = popt[1]
except Exception:
    fit = None; tau = float("nan")

fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.plot(lags2, R, color=BLUE, lw=2, marker="o", ms=3, label="标定出的响应函数 R(l)")
if fit is not None:
    ax.plot(lags2, fit, color=RED, lw=1.5, ls="--",
            label=f"饱和拟合  R(l)=a(1-e^{{-l/{tau:.1f}}})")
ax.axhline(R[-1], color=GRAY, ls=":", lw=1, alpha=0.7)
ax.set_title("从成交数据反推冲击响应：R(l) 随滞后先升后趋平", fontsize=12)
ax.set_xlabel("滞后 l（事件数）"); ax.set_ylabel("响应函数 R(l)")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/kernel_calibration.png", dpi=110); plt.close()

print("done. tau=%.2f  R1=%.4f Rlast=%.4f  price range=[%.2f,%.2f]" %
      (tau, R[0], R[-1], price.min(), price.max()))
print("acf long-mem lag1=%.3f lag50=%.3f | indep lag1=%.3f" % (ac_lm[0], ac_lm[49], ac_indep[0]))
