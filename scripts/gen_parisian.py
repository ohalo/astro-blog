#!/usr/bin/env python3
"""
巴黎期权与部分触及障碍 - 配图生成
生成 4 张真实图表到 public/images/parisian-barrier-options/
标准障碍 vs 巴黎式(部分触及)障碍 的 MC 定价 + 触及窗 D 敏感性 + 触及路径示意 + 收敛性
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Hiragino Sans GB", "PingFang SC", "STHeiti", "Songti SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
from scipy.stats import norm

OUT = "public/images/parisian-barrier-options"
os.makedirs(OUT, exist_ok=True)

# ---------- 参数 ----------
S0, K, r, sig, T = 100.0, 100.0, 0.05, 0.25, 1.0
B = 90.0                       # 下行障碍 (B < K)
n_steps = 252                  # 日频
dt = T / n_steps
rng = np.random.default_rng(20260720)

def bs_call(S, K, r, sig, T):
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def mc_price(option="parisian", D=5 * dt, n_paths=40000, seed=0):
    """MC 定价下行敲出看涨。option='standard' 任一时点触障即敲出;
    option='parisian' 需连续低于障碍累计 >= D 才敲出; option='vanilla' 不敲出。"""
    rg = np.random.default_rng(seed)
    Z = rg.standard_normal((n_paths, n_steps))
    drift = (r - 0.5 * sig ** 2) * dt
    log_s = np.log(S0) + np.cumsum(drift + sig * np.sqrt(dt) * Z, axis=1)
    S = np.exp(log_s)
    # 期末价格
    ST = S[:, -1]
    if option == "vanilla":
        payoff = np.maximum(ST - K, 0.0)
        return np.exp(-r * T) * np.mean(payoff)
    # 敲出判定
    below = S < B                      # (n_paths, n_steps) 是否低于障碍
    if option == "standard":
        knocked = below.any(axis=1)
    else:  # parisian: 连续低于障碍的累计时长 >= D
        D_steps = max(1, int(round(D / dt)))
        knocked = np.zeros(n_paths, dtype=bool)
        run = np.zeros(n_paths)
        for j in range(n_steps):
            run = np.where(below[:, j], run + dt, 0.0)
            knocked = knocked | (run >= D)
    payoff = np.where(knocked, 0.0, np.maximum(ST - K, 0.0))
    return np.exp(-r * T) * np.mean(payoff)

# 自洽校验: 巴黎式价格应介于标准障碍与 vanilla 之间
van = mc_price("vanilla", seed=1)
std = mc_price("standard", seed=1)
par1 = mc_price("parisian", D=5 * dt, seed=1)
par2 = mc_price("parisian", D=21 * dt, seed=1)
print(f"Vanilla BS 看涨闭式 = {bs_call(S0,K,r,sig,T):.4f}   MC={van:.4f}")
print(f"标准下行敲出看涨   = {std:.4f} (敲出概率最高, 最便宜)")
print(f"巴黎式(D=1周=5日)   = {par1:.4f}")
print(f"巴黎式(D=1月=21日)  = {par2:.4f}")
print(f"单调性检查: vanilla({van:.3f}) >= par1({par1:.3f}) >= std({std:.3f}) ? "
      f"{van>=par1>=std}")

# ============ 图 1: 障碍水平 B 对三类价格的影响 ============
B_list = np.linspace(80, 99, 12)
van_arr = np.array([bs_call(S0, K, r, sig, T) for _ in B_list])
std_arr, par_arr = [], []
for b in B_list:
    Bb = b
    # 用闭式近似标准障碍(下行敲出看涨, 连续监测)
    # Merton 公式: c_do = c - (S0/K)^{...} ... 直接用 MC 更稳妥
    std_arr.append(mc_price("standard", D=0, n_paths=20000, seed=7))
    par_arr.append(mc_price("parisian", D=10 * dt, n_paths=20000, seed=7))
std_arr, par_arr = np.array(std_arr), np.array(par_arr)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(B_list, van_arr, "k-", lw=2.4, label="普通欧式看涨 (不敲出)")
ax.plot(B_list, par_arr, "r--", lw=1.8, label="巴黎式下行敲出 (D=10日)")
ax.plot(B_list, std_arr, "b:", lw=1.8, label="标准下行敲出 (瞬时)")
ax.axvline(K, color="gray", ls=":", alpha=0.6, label="行权价 K=100")
ax.set_xlabel("障碍水平 $B$"); ax.set_ylabel("看涨期权价值")
ax.set_title("障碍水平对三类下行敲出看涨的影响")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/parisian_vs_standard.png", dpi=130); plt.close(fig)

# ============ 图 2: 触及窗 D 敏感性 ============
D_days = np.array([0, 1, 3, 5, 10, 15, 21, 42, 63])
price_D = [mc_price("parisian", D=d * dt, n_paths=30000, seed=3) for d in D_days]
price_D = np.array(price_D)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(D_days, price_D, "o-", color="darkred", label="巴黎式 (窗口 D)")
ax.axhline(van, color="k", ls="--", lw=1.6, label="普通欧式上限")
ax.axhline(std, color="b", ls=":", lw=1.4, label="标准障碍下限 (D=0)")
ax.set_xlabel("触及窗 $D$ (交易日)"); ax.set_ylabel("巴黎式看涨价值")
ax.set_title("巴黎式障碍的触及窗敏感性: D=0→标准, D→∞→欧式")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/parisian_window.png", dpi=130); plt.close(fig)

# ============ 图 3: 触及路径示意 (含累计触及计时) ============
rg = np.random.default_rng(42)
Z = rg.standard_normal((n_steps))
S_path = S0 * np.exp(np.cumsum((r - 0.5 * sig ** 2) * dt + sig * np.sqrt(dt) * Z))
t = np.arange(1, n_steps + 1) * dt
D_demo = 10 * dt
timer = np.zeros(n_steps)
for j in range(n_steps):
    timer[j] = timer[j - 1] + dt if (S_path[j] < B) else 0.0
knocked = timer >= D_demo
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(t, S_path, "b-", lw=1.6, label="标的路径 $S_t$")
ax.axhline(B, color="r", ls="--", lw=1.4, label=f"障碍 $B={B}$")
ax.fill_between(t, B, S_path, where=(S_path < B), color="red", alpha=0.18, label="低于障碍区间")
ax2 = ax.twinx()
ax2.plot(t, timer, "g-", lw=1.0, alpha=0.7, label="累计触及计时")
ax2.axhline(D_demo, color="g", ls=":", lw=1.2, label=f"触及窗 $D$")
ax2.set_ylabel("累计低于障碍时长", color="g")
ax2.tick_params(axis="y", labelcolor="g")
ax.set_xlabel("时间 $t$ (年)"); ax.set_ylabel("标的 $S_t$")
ax.set_title("巴黎式障碍: 仅当连续低于障碍≥D 才敲出")
ax.legend(loc="upper left"); fig.tight_layout()
fig.savefig(f"{OUT}/parisian_excursion.png", dpi=130); plt.close(fig)

# ============ 图 4: 路径数收敛性 ============
n_list = [2000, 5000, 10000, 20000, 40000, 80000]
par_conv, par_se = [], []
for n in n_list:
    # 多次重复取均值与 std 估计
    reps = [mc_price("parisian", D=10 * dt, n_paths=n, seed=s) for s in range(5, 9)]
    par_conv.append(np.mean(reps)); par_se.append(np.std(reps))
par_conv, par_se = np.array(par_conv), np.array(par_se)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.errorbar(n_list, par_conv, yerr=1.96 * par_se, fmt="o-", color="darkred",
            capsize=3, label="巴黎式价格 $\\pm 1.96\\sigma$")
ax.set_xlabel("蒙特卡洛路径数 $N$"); ax.set_ylabel("巴黎式看涨价值")
ax.set_title("巴黎式障碍 MC 收敛: 价格随路径数稳定")
ax.set_xscale("log"); ax.grid(True, which="both", alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/parisian_convergence.png", dpi=130); plt.close(fig)

print("✅ 巴黎期权配图已生成:", os.listdir(OUT))
