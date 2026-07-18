#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""粒子滤波状态估计 配图生成 (3 张真实图表)"""
import os
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy.stats import norm

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "particle-filter-state-estimation"
OUT = f"public/images/{SLUG}"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260719)

# ---------- 1) 模拟随机波动状态(隐含波动率) + 噪声观测 ----------
T = 800
# 真实隐含波动率: 慢变随机游走 + 两段 regime 切换
true_iv = np.zeros(T); true_iv[0] = 0.20
for t in range(1, T):
    drift = 0.00002 if t < T*0.6 else -0.00003   # 先升后降的结构
    true_iv[t] = true_iv[t-1] + drift + 0.004 * rng.normal(0, 1)
    true_iv[t] = np.clip(true_iv[t], 0.05, 0.45)
# 观测: 用 IV 生成"期权报价"式代理, 加异方差噪声(波动越高噪声越大)
obs = true_iv + true_iv * 0.06 * rng.normal(0, 1, T)

# ---------- 2) SIR 粒子滤波 ----------
def sir_filter(y, N=2000, proc_sd=0.004, obs_sd_scale=0.06):
    particles = np.full(N, 0.20) + 0.002 * rng.normal(0, 1, N)
    mean_est = np.zeros(len(y)); var_est = np.zeros(len(y))
    ll = 0.0
    for t in range(len(y)):
        # 预测: 状态转移
        particles = particles + proc_sd * rng.normal(0, 1, N)
        particles = np.clip(particles, 0.02, 0.6)
        # 更新: 似然权重(异方差: 噪声随状态变大)
        obs_sd = obs_sd_scale * particles
        logw = -0.5 * ((y[t] - particles) / obs_sd)**2 - np.log(obs_sd)
        maxw = logw.max(); w = np.exp(logw - maxw); w /= w.sum()
        # 重采样(系统重采样)
        idx = np.searchsorted(np.cumsum(w), rng.uniform(0, 1, N))
        particles = particles[idx]
        mean_est[t] = particles.mean(); var_est[t] = particles.var()
        ll += np.log(np.exp(logw).mean()) + maxw
    return mean_est, var_est, ll

mean_est, var_est, ll = sir_filter(obs)
rmse = np.sqrt(np.mean((mean_est - true_iv)**2))

# ---------- 3) 对照: 卡尔曼滤波(线性高斯近似) ----------
# 把状态近似为线性高斯, 观测方差用平均观测尺度
def kalman(y, q=0.004**2, r=(obs.std()*0.06)**2):
    x = 0.20; P = 0.01; est = np.zeros(len(y)); var = np.zeros(len(y))
    F, H = 1.0, 1.0
    for t in range(len(y)):
        x = F * x; P = F * P * F + q
        S = H * P * H + r
        K = P * H / S
        x = x + K * (y[t] - H * x); P = (1 - K * H) * P
        est[t] = x; var[t] = P
    return est, var

kf_est, kf_var = kalman(obs)
kf_rmse = np.sqrt(np.mean((kf_est - true_iv)**2))

# ---------- 4) 图1: 状态轨迹 + 滤波估计 + 95%区间 ----------
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(true_iv, color="#27ae60", lw=1.6, label="真实隐含波动率")
ax.scatter(range(T), obs, s=8, color="#bdc3c7", alpha=0.5, label="噪声观测")
ax.plot(mean_est, color="#c0392b", lw=1.4, label="粒子滤波估计")
ax.fill_between(range(T), mean_est - 1.96*np.sqrt(var_est), mean_est + 1.96*np.sqrt(var_est),
                color="#c0392b", alpha=0.15, label="95% 置信带")
ax.set_xlabel("时间"); ax.set_ylabel("隐含波动率")
ax.set_title("粒子滤波从噪声观测中还原隐含波动率路径", fontweight="bold")
ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/pf_state_trajectory.png", bbox_inches="tight"); plt.close(fig)

# ---------- 5) 图2: 局部放大(前120步) 看跟踪质量 ----------
seg = slice(60, 200)
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(true_iv[seg], color="#27ae60", lw=1.8, label="真实")
ax.plot(obs[seg], color="#bdc3c7", lw=0.8, alpha=0.7, label="观测")
ax.plot(mean_est[seg], color="#c0392b", lw=1.6, label="粒子滤波")
ax.plot(kf_est[seg], color="#2980b9", lw=1.4, ls="--", label="卡尔曼滤波(线性近似)")
ax.set_xlabel("时间"); ax.set_ylabel("隐含波动率")
ax.set_title("局部放大: 粒子滤波在非高斯噪声下更贴真实状态", fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/pf_local_zoom.png", bbox_inches="tight"); plt.close(fig)

# ---------- 6) 图3: 粒子云快照(某时刻的后验分布) ----------
t_snap = 300
# 用观测重新跑一次并保存该时刻粒子(简化: 重跑并在 t_snap 存粒子)
def sir_filter_snapshot(y, t_snap, N=2000):
    particles = np.full(N, 0.20) + 0.002 * rng.normal(0, 1, N)
    saved = None
    for t in range(len(y)):
        particles = particles + 0.004 * rng.normal(0, 1, N)
        particles = np.clip(particles, 0.02, 0.6)
        obs_sd = 0.06 * particles
        logw = -0.5 * ((y[t] - particles) / obs_sd)**2 - np.log(obs_sd)
        maxw = logw.max(); w = np.exp(logw - maxw); w /= w.sum()
        if t == t_snap:
            saved = (particles.copy(), w.copy())
        idx = np.searchsorted(np.cumsum(w), rng.uniform(0, 1, N))
        particles = particles[idx]
    return saved

parts, wts = sir_filter_snapshot(obs, t_snap)
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.hist(parts, bins=60, weights=wts, color="#8e44ad", alpha=0.7, density=True)
ax.axvline(true_iv[t_snap], color="#27ae60", lw=2, label=f"真实 IV={true_iv[t_snap]:.3f}")
ax.axvline(mean_est[t_snap], color="#c0392b", lw=2, ls="--", label=f"滤波均值={mean_est[t_snap]:.3f}")
ax.set_xlabel("隐含波动率粒子 (时刻 t=300)"); ax.set_ylabel("后验密度")
ax.set_title("粒子云: 滤波在单个时刻给出的后验分布", fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/pf_particle_cloud.png", bbox_inches="tight"); plt.close(fig)

print(f"粒子滤波 RMSE={rmse:.5f}  对数似然={ll:.1f}")
print(f"卡尔曼(线性近似) RMSE={kf_rmse:.5f}")
print(f"真实IV @300={true_iv[t_snap]:.4f} 滤波均值@300={mean_est[t_snap]:.4f} 95%半宽={1.96*np.sqrt(var_est[t_snap]):.4f}")
print("DONE images:", os.listdir(OUT))
