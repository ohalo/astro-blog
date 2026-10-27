#!/usr/bin/env python3
"""生成 少数者博弈金融市场模型 文章：基于 Agent 的 MG 模拟 + 3 张真实图表（CJK 字体）。

少数者博弈（Minority Game, Challet & Zhang 1997）：N 个 Agent 每轮同时选 0/1，
选到「少数人那一侧」的获胜。每个 Agent 持有若干策略（历史→动作查表），用虚拟得分选最好策略。
核心现象（本文可复现的诚实版本）：记忆 m 越长，Agent 越能把到场人数 a_t 协调到容量
L=N/2 附近——「协调效率」eff = P(|a_t−L|≤容差) 随 m 从 ~5%（m 极小、几乎乱选）单调跃升到
~55%（m 较大、自我组织成功）。这是「异质、有限理性、自适应策略的 Agent 如何内生产生接近有效、
却不锁死的协调」的经典模型，对应金融市场中「逆向 / 少数者 / 流动性提供」型交易者的自组织。
（注：完整 MG 在波动率 σ 上还表现出 m*≈log₂N 的最优记忆相变；本文聚焦协调效率这一更稳健、
可复现的序参量视角。）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False
fm._load_fontmanager()

N = 101
L = N // 2
WARM = 2000
STEPS = 60000
SEEDS = 8

def run_mg(N, m, S=2, L=None, steps=60000, warm=2000, seed=0, rtie=True):
    if L is None:
        L = N // 2
    rng = np.random.default_rng(seed)
    n_states = 2 ** m
    strat = rng.integers(0, 2, size=(N, S, n_states))
    score = np.zeros((N, S))
    hist = rng.integers(0, 2, size=m)
    att = np.empty(steps, dtype=int)
    for t in range(steps):
        key = 0
        for i in range(m - 1, -1, -1):
            key = (key << 1) | int(hist[i])
        if rtie:
            best = np.argmax(score + 1e-6 * rng.random((N, S)), axis=1)
        else:
            best = np.argmax(score, axis=1)
        a = int(strat[np.arange(N), best, key].sum())
        win = 0 if a < L else 1
        score += (strat[np.arange(N), :, key] == win).astype(float)
        att[t] = a
        hist = np.roll(hist, 1)
        hist[0] = win
    return att[warm:]

# ---- 扫描 m：协调效率序参量 ----
m_grid = list(range(1, 12))
mean_a, sig, eff5, eff10 = [], [], [], []
for m in m_grid:
    aa = [run_mg(N, m, seed=7000 + s) for s in range(SEEDS)]
    A = np.concatenate(aa)
    mean_a.append(A.mean())
    sig.append(A.std())
    eff5.append(np.mean(np.abs(A - L) <= 5) * 100)
    eff10.append(np.mean(np.abs(A - L) <= 10) * 100)
mean_a = np.array(mean_a); sig = np.array(sig)
eff5 = np.array(eff5); eff10 = np.array(eff10)

m_small, m_large = 2, 10
att_small = run_mg(N, m_small, seed=909)
att_large = run_mg(N, m_large, seed=909)

import os
outdir = "public/images/minority-game-model"
os.makedirs(outdir, exist_ok=True)

# ===== 图1：cover —— 大记忆 m=10 时到场人数紧贴容量 L =====
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(att_large, color="#1f3a5f", lw=0.8)
ax.axhline(L, color="#d73027", ls="--", lw=1.4, label=f"容量 L = N/2 = {L}")
ax.set_title(f"少数者博弈（m={m_large}）：Agent 把到场人数协调到容量 L 附近，不崩溃也不锁死",
             fontsize=13.5, fontweight="bold", color="#1f3a5f")
ax.set_ylabel("选 0 的人数 a_t")
ax.set_xlabel("轮次（已过预热）")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.25)
ax.set_ylim(0, N)
fig.tight_layout()
fig.savefig(f"{outdir}/cover.png", dpi=130)
plt.close(fig)

# ===== 图2：phase —— 协调效率序参量 eff10 随记忆 m 单调跃升 =====
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.plot(m_grid, eff10, color="#1a9850", marker="o", lw=1.8, label="协调效率 eff = P(|a_t−L|≤10)")
ax.axhline(100 * (21 / N), color="gray", ls=":", lw=1.3,
            label=f"随机基线 ≈ {100*21/N:.0f}%（均匀不相关）")
ax.set_title("相变：记忆 m 越长，Agent 越能自我组织到容量附近（从 ~5% 跃升到 ~55%）",
             fontsize=13.5, fontweight="bold", color="#1f3a5f")
ax.set_xlabel("记忆长度 m（用最近 m 轮少数者动作做历史）")
ax.set_ylabel("协调效率（%）")
ax.set_xticks(m_grid)
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{outdir}/phase_transition.png", dpi=130)
plt.close(fig)

# ===== 图3：分布对比 —— m=2（几乎乱选、偏置）vs m=10（紧贴 L）=====
fig, ax = plt.subplots(figsize=(11, 5.0))
bins = np.arange(0, N + 4, 4)
ax.hist(att_small, bins=bins, alpha=0.55, color="#d73027", label=f"m={m_small}（记忆过短：偏置、低效）")
ax.hist(att_large, bins=bins, alpha=0.55, color="#1a9850", label=f"m={m_large}（记忆充足：协调到 L）")
ax.axvline(L, color="black", ls="--", lw=1.2, label=f"容量 L={L}")
ax.set_title("分布对比：小记忆时到场人数发散偏置（浪费），大记忆时紧紧聚在 L 附近",
             fontsize=13.5, fontweight="bold", color="#1f3a5f")
ax.set_xlabel("选 0 的人数 a_t")
ax.set_ylabel("频次")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(f"{outdir}/attendance_dist.png", dpi=130)
plt.close(fig)

# ===== 指标 =====
print("=== Minority Game metrics ===")
print(f"N={N}, L={L}, S=2, steps={STEPS}, seeds={SEEDS}")
for m in range(1, 12):
    print(f"  m={m:>2}: mean_a={mean_a[m-1]:5.1f}  sigma={sig[m-1]:5.2f}  eff5={eff5[m-1]:5.1f}%  eff10={eff10[m-1]:5.1f}%")
print(f"  m={m_small}: mean_a={mean_a[m_small-1]:.1f} eff10={eff10[m_small-1]:.1f}%")
print(f"  m={m_large}: mean_a={mean_a[m_large-1]:.1f} eff10={eff10[m_large-1]:.1f}%")
print("Minority game images written.")
