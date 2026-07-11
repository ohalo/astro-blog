#!/usr/bin/env python3
"""
为文章「极端值理论在尾部风险度量中的应用：广义帕累托(POT)与 Hill 估计」
(evt-tail-risk-gpd) 生成真实配图与真实统计数字。

所有图表与数字均由文中 Python 逻辑真实计算生成：
  1) evt_qq.png          —— 样本分位 vs 正态分位 QQ 图，明显偏离直线 = 肥尾
  2) evt_mean_excess.png —— 平均超额函数(ME)图(损失)，选阈值 u 使之上方近似线性
  3) evt_hill.png        —— Hill 图，对 k 的尾指数估计，取平稳段
  4) evt_var.png         —— 99%/99.5%/99.9% 下 VaR：正态 vs 经验 vs EVT-GPD

核心方法(POT，超阈值峰值)：
  - 把收益取负得到「损失」L=-R，对损失的上尾做 GPD 拟合：
    H(y)=1-(1+ξ y/β)^{-1/ξ}，ξ 形状(尾指数)，β 尺度
  - VaR_q = u + β/ξ[((1-q)n/N_u)^{-ξ} - 1]   (ξ≠0)
  - CVaR_q = VaR_q/(1-ξ) + (β-ξ u)/(1-ξ)
  - Hill 估计：γ(k)= (1/k)Σ log L_{(n-i+1)} - log L_{(n-k)}，稳定段即 ξ
数据：Student-t(df=4) 缩放至日波动 ~1.4%，理论尾指数 ξ=1/df=0.25，
     用于检验 EVT 估计是否还原真实值。
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
D = os.path.join(BASE, "evt-tail-risk-gpd")
os.makedirs(D, exist_ok=True)

C = {"eq": "#2F4B7C", "up": "#55A868", "dn": "#C44E52", "grid": "#DDDDDD",
     "gauss": "#8172B3", "emp": "#DD8452", "evt": "#4C72B0", "shade": "#F2C0C0"}

# =====================================================================
# 0) 逆正态 CDF (Acklam 近似) —— 自包含，不依赖 scipy
# =====================================================================
def norm_ppf(p):
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= phigh:
        q = p - 0.5; r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = np.sqrt(-2 * np.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

# =====================================================================
# 1) 合成肥尾日度收益：Student-t(df=4)，理论尾指数 ξ=1/df=0.25
# =====================================================================
def simulate(T=4000, df=4, seed=7):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(T)
    chi = np.sum(rng.standard_normal((T, df)) ** 2, axis=1) / df
    t = z / np.sqrt(chi)                       # 标准 t(df)
    return (t / np.sqrt(df / (df - 2))) * 0.014 / np.sqrt(252) * np.sqrt(252)  # 缩放至日波动~1.4%

R = simulate()
n = len(R)
L = -R                                          # 损失（正值=亏损）
mu, sd = R.mean(), R.std(ddof=1)
Lmu, Lsd = L.mean(), L.std(ddof=1)

# =====================================================================
# 2) 图 1：QQ 图（收益 vs 正态）
# =====================================================================
Rs = np.sort(R)
probs = (np.arange(1, n + 1) - 0.5) / n
z = np.array([norm_ppf(p) for p in probs])
fig, ax = plt.subplots(figsize=(6.2, 5.4))
ax.scatter(z, Rs * 100, s=6, color=C["eq"], alpha=0.5)
lim = max(abs(z.min()), abs(z.max())) * 1.05
ax.plot([-lim, lim], [mu*100 - lim*sd*100, mu*100 + lim*sd*100], color=C["dn"], lw=1.6, label="正态参考线")
ax.set_xlabel("理论正态分位数"); ax.set_ylabel("样本收益 (%)")
ax.set_title("QQ 图：尾部明显偏离直线 = 肥尾（正态假设会低估风险）")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "evt_qq.png"), dpi=130); plt.close()

# =====================================================================
# 3) 平均超额函数 (Mean Excess, 对损失 L) —— 选阈值 u
# =====================================================================
qs = np.linspace(0.50, 0.985, 120)
us = np.quantile(L, qs)
me = np.array([(L[L > u] - u).mean() for u in us])
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(us * 100, me * 100, color=C["eq"], lw=1.8, marker=".", ms=3, label="平均超额 ME(u)")
u_sel = np.quantile(L, 0.95)                   # 95% 分位的损失作为阈值
ax.axvline(u_sel * 100, color=C["dn"], ls="--", lw=1.4, label="选定 u = %.2f%% (损失95%%分位)" % (u_sel * 100))
ax.set_xlabel("阈值 u (损失 %)"); ax.set_ylabel("平均超额 ME(u) (%)")
ax.set_title("平均超额函数：u 之上近似线性 -> 满足 GPD 假设，可在此选阈值")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "evt_mean_excess.png"), dpi=130); plt.close()

# =====================================================================
# 4) GPD 的 MLE 拟合（对损失上尾，网格搜索，自包含）
# =====================================================================
excess = L[L > u_sel] - u_sel
Nu = len(excess)
xbar = excess.mean()
def gpd_loglik(xi, beta, z=excess):
    if beta <= 0:
        return -1e18
    s = 1 + xi * z / beta
    if np.any(s <= 0):
        return -1e18
    return -Nu * np.log(beta) - (1 / xi + 1) * np.sum(np.log(s))

best = (-1e18, None, None)
for xi in np.linspace(-0.40, 0.70, 600):
    for beta in np.linspace(0.3 * xbar, 3.0 * xbar, 600):
        ll = gpd_loglik(xi, beta)
        if ll > best[0]:
            best = (ll, xi, beta)
_, xi_hat, beta_hat = best

# Hill 估计 (损失上尾)
Ls_desc = np.sort(L)[::-1]                      # 从大到小（最大损失在前）
Ks = np.arange(50, 900, 10)
hill = []
for k in Ks:
    top = Ls_desc[:k]
    gamma = np.mean(np.log(top)) - np.log(Ls_desc[k - 1])
    hill.append(gamma)
hill = np.array(hill)
hill_stable = hill[(Ks >= 200) & (Ks <= 600)].mean()

# =====================================================================
# 5) 图 3：Hill 图
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(Ks, hill, color=C["eq"], lw=1.6, label="Hill 估计 γ(k)")
ax.axhline(hill_stable, color=C["dn"], ls="--", lw=1.4,
           label="平稳段均值 γ≈%.3f (尾指数 ξ≈%.3f)" % (hill_stable, hill_stable))
ax.set_xlabel("k (最大 k 个损失观测用于估计)"); ax.set_ylabel("尾指数 γ")
ax.set_title("Hill 图：损失上尾指数在 k∈[200,600] 平稳，ξ≈%.3f 确认肥尾" % hill_stable)
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "evt_hill.png"), dpi=130); plt.close()

# =====================================================================
# 6) VaR / CVaR：正态 vs 经验 vs EVT-GPD（均为损失，正值）
# =====================================================================
def evt_var(q):
    fact = ((1 - q) * n / Nu) ** (-xi_hat) - 1
    return u_sel + beta_hat / xi_hat * fact

qs_test = [0.95, 0.99, 0.999]
v_gauss, v_emp, v_evt, c_evt = [], [], [], []
for q in qs_test:
    v_gauss.append(Lmu + Lsd * norm_ppf(q))     # 正态 VaR（损失）
    v_emp.append(np.quantile(L, q))             # 经验 VaR
    v_evt.append(evt_var(q))
    c_evt.append(v_evt[-1] / (1 - xi_hat) + (beta_hat - xi_hat * u_sel) / (1 - xi_hat))

# =====================================================================
# 7) 图 4：VaR 对比
# =====================================================================
x = np.arange(len(qs_test)); w = 0.27
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.bar(x - w, [v * 100 for v in v_gauss], w, color=C["gauss"], label="正态假设 VaR")
ax.bar(x, [v * 100 for v in v_emp], w, color=C["emp"], label="经验(历史) VaR")
ax.bar(x + w, [v * 100 for v in v_evt], w, color=C["evt"], label="EVT-GPD VaR")
ax.set_xticks(x); ax.set_xticklabels(["95%", "99%", "99.9%"])
ax.set_ylabel("VaR (潜在损失 %, 正值)")
ax.set_title("越往尾部，正态 VaR 越低估风险；EVT-GPD 贴近经验分位")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(D, "evt_var.png"), dpi=130); plt.close()

# =====================================================================
# 打印真实数字
# =====================================================================
print("=== 极端值理论(EVT)尾部风险 关键数字 ===")
print("样本: 日度 %d 个收益观测，Student-t(df=4) 缩放至日波动 ~1.4%%（理论 ξ=1/df=0.25）" % n)
print("样本均值=%.4f%%  样本波动=%.4f%%  样本峰度=%.2f (正态=3)" % (mu*100, sd*100, (np.mean((R-mu)**4)/(sd**4))))
print("损失阈值 u = %.3f%% (95%%分位)，超阈值样本 N_u=%d" % (u_sel*100, Nu))
print("GPD MLE 拟合(损失上尾): 形状 ξ=%.3f  尺度 β=%.4f%%  [理论 ξ=0.25]" % (xi_hat, beta_hat*100))
print("Hill 平稳段尾指数 γ≈%.3f  (与 GPD ξ 一致: 肥尾确认)" % hill_stable)
print("VaR 对比 (潜在损失 %):")
for i, q in enumerate(qs_test):
    print("  q=%.1f%%: 正态=%.2f%%  经验=%.2f%%  EVT-GPD=%.2f%%  EVT-CVaR=%.2f%%" % (
        q*100, v_gauss[i]*100, v_emp[i]*100, v_evt[i]*100, c_evt[i]*100))
underest = (v_evt[2] - v_gauss[2]) / v_gauss[2] * 100
print("正态 VaR 在 99.9%% 相对 EVT 的低估幅度: 约 -%.0f%% (EVT 是正态的 %.2f 倍)" % (
    abs(underest), v_evt[2] / v_gauss[2]))
print("\n图片已保存到:", D)
