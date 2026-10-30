#!/usr/bin/env python3
"""
平均场博弈多代理人做市策略 — 完整可运行脚本
依赖：numpy, matplotlib
输出：5 张 PNG 图 + _mfg_results.json
"""
import numpy as np
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(20261030)

# ====== 1. 参数 ======
T = 1.0
gamma = 2.0
c = 0.5
sigma = 0.01
lam_plus = 10.0
lam_minus = 10.0
C_terminal = gamma / 2

alpha = gamma / (2 * c)
beta = lam_plus / (2 * c)

def analytical_spread(q, q_bar):
    return 2 * (alpha * q + beta * q_bar)

# ====== 2. Agent 仿真 ======
N_agents = 500
dt = 0.001
n_steps = int(T / dt)

q_init = np.random.uniform(-5, 5, N_agents)
q = q_init.copy()
spread_history = []
utility_history = np.zeros(N_agents)

Z = np.random.randn(N_agents, n_steps) * np.sqrt(dt) * sigma * 100

for step in range(n_steps):
    t = step * dt
    q_bar = np.mean(q)

    spreads = analytical_spread(q, q_bar)
    spread_history.append(np.mean(spreads))

    rel_comp = np.clip((q_bar - q) * beta / (c + 1e-8), -3, 3)

    lam_buy = lam_plus * np.exp(rel_comp)
    lam_sell = lam_minus * np.exp(-rel_comp)

    buys = np.random.poisson(np.clip(lam_buy * dt, 0, 1), N_agents)
    sells = np.random.poisson(np.clip(lam_sell * dt, 0, 1), N_agents)

    q += (buys - sells)
    q += Z[:, step]

    utility_history += (gamma / 2) * q**2 + c * np.abs(spreads)

terminal_penalty = C_terminal * q**2
total_utility = utility_history + terminal_penalty

# ====== 3. 收敛分析 ======
N_values = [10, 20, 50, 100, 200, 500, 1000]
converged_spreads = []

for N in N_values:
    q_test = np.random.uniform(-5, 5, N)
    q_bar_test = np.mean(q_test)
    spread_test = np.mean(analytical_spread(q_test, q_bar_test))
    converged_spreads.append(spread_test)

convergence_error = np.abs(np.array(converged_spreads[:-1]) - converged_spreads[-1]) / converged_spreads[-1]

simulation_error = np.std(spread_history) / np.mean(spread_history)
avg_utility = np.mean(total_utility)
utility_std = np.std(total_utility)

results = {
    "N_agents": N_agents,
    "alpha": float(alpha),
    "beta": float(beta),
    "reference_spread_N1000": float(converged_spreads[-1]),
    "convergence_error_N500": float(convergence_error[-1]),
    "simulation_noise_level": float(simulation_error),
    "avg_utility": float(avg_utility),
    "utility_std": float(utility_std),
}

os.makedirs("public/images/mean-field-game-multiaagent", exist_ok=True)
with open("_mfg_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ====== 4. 绘图 ======
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"平均场博弈多代理人做市策略 (N={N_agents}, α={alpha:.2f}, β={beta:.2f})",
             fontsize=14, fontweight="bold")

q_range = np.linspace(-10, 10, 200)
q_bar_values = np.array([-5, 0, 5])
for q_bar_val in q_bar_values:
    spread_vals = analytical_spread(q_range, q_bar_val)
    axes[0, 0].plot(q_range, spread_vals, label=f"q̄ = {q_bar_val}", linewidth=2)
axes[0, 0].set_xlabel("个体库存 q")
axes[0, 0].set_ylabel("均衡价差 spread")
axes[0, 0].set_title("解析均衡价差：spread = 2(αq + βq̄)")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].semilogy(N_values[:-1], convergence_error, "o-", linewidth=2, markersize=6)
axes[0, 1].set_xlabel("Agent 数量 N")
axes[0, 1].set_ylabel("相对收敛误差")
axes[0, 1].set_title(f"价差收敛：N→∞ (误差 {convergence_error[-1]*100:.2f}%)")
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(np.arange(len(spread_history)), spread_history, linewidth=0.8, alpha=0.7)
axes[0, 2].axhline(np.mean(spread_history), color="red", linestyle="--", linewidth=2,
                   label=f"均值 {np.mean(spread_history):.3f}")
axes[0, 2].set_xlabel("时间步")
axes[0, 2].set_ylabel("平均价差")
axes[0, 2].set_title(f"500 Agent 仿真 (噪声 {simulation_error*100:.2f}%)")
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

axes[1, 0].hist(total_utility, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
axes[1, 0].axvline(avg_utility, color="red", linestyle="--", linewidth=2,
                   label=f"均值 {avg_utility:.1f}")
axes[1, 0].set_xlabel("总效用")
axes[1, 0].set_ylabel("频数")
axes[1, 0].set_title(f"Agent 效用分布 (μ={avg_utility:.1f}, σ={utility_std:.1f})")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

alpha_range = np.linspace(0.5, 5.0, 50)
spread_sensitivity = [2 * a * 3 for a in alpha_range]
axes[1, 1].plot(alpha_range, spread_sensitivity, linewidth=2)
axes[1, 1].set_xlabel("α = γ/(2c)")
axes[1, 1].set_ylabel("均衡价差 (q=3)")
axes[1, 1].set_title("价差对 α 的线性敏感性")
axes[1, 1].grid(True, alpha=0.3)

N_test = np.logspace(1, 3.5, 50)
vol_N = 1.0 / np.sqrt(N_test)
axes[1, 2].semilogy(N_test, vol_N, linewidth=2)
axes[1, 2].axvline(N_agents, color="red", linestyle="--", linewidth=2,
                   label=f"N={N_agents}, σ={1/np.sqrt(N_agents):.4f}")
axes[1, 2].set_xlabel("Agent 数量 N")
axes[1, 2].set_ylabel("波动率 ~ 1/√N")
axes[1, 2].set_title("有限 N 波动衰减")
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("public/images/mean-field-game-multiaagent/mfg-overview.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()

# 补充图
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle("平均场博弈 — 详细分析", fontsize=14, fontweight="bold")

axes2[0, 0].hist(q, bins=40, alpha=0.7, edgecolor="black", linewidth=0.5, density=True)
axes2[0, 0].set_xlabel("期末库存")
axes2[0, 0].set_ylabel("密度")
axes2[0, 0].set_title(f"期末库存分布 (N={N_agents})")
axes2[0, 0].grid(True, alpha=0.3)

sample_idx = np.random.choice(N_agents, min(200, N_agents), replace=False)
final_spreads = analytical_spread(q[sample_idx], np.mean(q))
axes2[0, 1].scatter(q[sample_idx], final_spreads, alpha=0.5, s=10)
axes2[0, 1].set_xlabel("库存 q")
axes2[0, 1].set_ylabel("报价 spread")
axes2[0, 1].set_title("价差-库存散点 (200 采样)")
axes2[0, 1].grid(True, alpha=0.3)

axes2[1, 0].semilogy(N_values, converged_spreads, "o-", linewidth=2, markersize=6)
axes2[1, 0].set_xlabel("Agent 数量 N")
axes2[1, 0].set_ylabel("平均价差")
axes2[1, 0].set_title("不同 N 下的均衡价差")
axes2[1, 0].grid(True, alpha=0.3)

utility_by_N = []
N_plot = [10, 50, 200, 500]
for N in N_plot:
    q_test = np.random.uniform(-5, 5, N)
    q_bar_test = np.mean(q_test)
    spread_test = analytical_spread(q_test, q_bar_test)
    util_test = -np.mean((gamma / 2) * q_test**2 + c * np.abs(spread_test))
    utility_by_N.append(util_test)

axes2[1, 1].plot(N_plot, utility_by_N, "o-", linewidth=2, markersize=8)
axes2[1, 1].set_xlabel("Agent 数量 N")
axes2[1, 1].set_ylabel("平均效用")
axes2[1, 1].set_title("效用随 N 的收敛")
axes2[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("public/images/mean-field-game-multiaagent/mfg-detail.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()

fig3, ax3 = plt.subplots(figsize=(8, 6))
beta_range = np.linspace(0.5, 5.0, 50)
spread_beta = [2 * (alpha * 3 + b * 2) for b in beta_range]
ax3.plot(beta_range, spread_beta, linewidth=2)
ax3.set_xlabel("β = λ/(2c)")
ax3.set_ylabel("均衡价差 (q=3, q̄=2)")
ax3.set_title("价差对 β 的敏感性")
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("public/images/mean-field-game-multiaagent/mfg-beta-sensitivity.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()

fig4, ax4 = plt.subplots(figsize=(8, 6))
C_range = np.linspace(0, gamma / 2, 50)
inv_cost = [C * 25 for C in C_range]
ax4.plot(C_range, inv_cost, linewidth=2)
ax4.set_xlabel("终端库存惩罚 C")
ax4.set_ylabel("终端成本 (q²=25)")
ax4.set_title("终端惩罚对成本的影响")
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("public/images/mean-field-game-multiaagent/mfg-terminal-penalty.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()

print(f"✅ MFG 脚本完成")
print(f"   N={N_agents}, α={alpha:.3f}, β={beta:.3f}")
print(f"   收敛误差 (N=500 vs 1000): {convergence_error[-1]*100:.2f}%")
print(f"   平均效用: {avg_utility:.1f} ± {utility_std:.1f}")
print(f"   仿真噪声: {simulation_error*100:.2f}%")
