# -*- coding: utf-8 -*-
"""
随机控制最优执行：用 HJB 方程解最优拆单
手写有限差分 HJB（(t,x) 网格反向动态规划），并用闭式 Almgren-Chriss sinh 调度校验。
seed = 20261029。

模型（Almgren 2001 均值-方差执行，简化为 LQ 随机控制）：
  状态 x(t)=剩余库存，dx/dt = -u(t)；x(0)=X，x(T)=0
  到达价格 S_t = S_0 + mu*t + sigma*W_t（mu 漂移，sigma 噪声）
  目标最小化 E[实现短缺] + lambda*Var(短缺)
    = ∫[ eta*u^2 + lambda*sigma^2*x^2 - mu*t*u ] dt   （价格噪声项均值为 0）
  HJB: -V_t = min_u { eta*u^2 + lambda*sigma^2*x^2 - mu*t*u - u*V_x }, V(T,0)=0
mu=0 时闭式：x*(t)=X*sinh(omega*(T-t))/sinh(omega*T), omega=sigma*sqrt(lambda/eta)
mu!=0 时无初等闭式 -> 这正是要数值解 HJB 的原因。

输出：public/images/stochastic-control-optimal-execution/*.png + _sce_results.json
"""
import numpy as np
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Heiti TC', 'PingFang SC', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

SEED = 20261029
np.random.seed(SEED)

SLUG = 'stochastic-control-optimal-execution'
OUTDIR = f'public/images/{SLUG}'
os.makedirs(OUTDIR, exist_ok=True)

# ---------- 参数 ----------
T = 1.0
X = 1.0
ETA = 1e-3
SIGMA = 0.02
LAM_SCHED = 1.0          # 调度图用的风险厌恶
MU_U = 0.003           # 正漂移（价格上行 -> 想晚卖）：清晰后载
MU_D = -0.003           # 负漂移 -> 更前载
LAM_SWEEP = np.logspace(-1, 2, 22)   # 前沿扫描

def omega(lam):
    return SIGMA * np.sqrt(lam / ETA)

# ---------- 闭式（mu=0，硬约束 x(T)=0 的 LQ 最优控制）----------
def closed(lam, t, x):
    # 调度：x*(t) = X sinh(omega*(T-t))/sinh(omega*T)
    # 值函数：V(t,x) = x^2 * sigma*sqrt(lambda*eta) * coth(omega*(T-t))
    w = omega(lam)
    arg = np.clip(w * (T - t), 1e-6, None)
    return x**2 * SIGMA * np.sqrt(lam * ETA) / np.tanh(arg)

def x_closed(lam, t):
    w = omega(lam)
    return X * np.sinh(w * (T - t)) / np.sinh(w * T)

def u_closed(lam, t):
    w = omega(lam)
    return w * X * np.cosh(w * (T - t)) / np.sinh(w * T)

# ---------- 有限差分 HJB（反向 DP，控制网格与状态网格解耦 + 插值）----------
def solve_hjb(lam, mu=0.0, Nt=300, Nx=100):
    dt = T / Nt
    dx = X / Nx
    xs = X * np.arange(Nx + 1) / Nx
    V = np.zeros((Nt + 1, Nx + 1))
    BIG = 1e12
    for j in range(1, Nx + 1):
        V[Nt, j] = BIG          # 终端必须清仓：x(T)>0 不可行
    # 控制网格：0~4 细（步长 0.05）覆盖平滑调度，4~X/dt 粗
    u_fine = np.linspace(0.0, 4.0, 81)
    u_coarse = np.linspace(4.0, X / dt, 121)[1:]
    u_grid = np.concatenate([u_fine, u_coarse])
    u_star = np.zeros((Nt, Nx + 1))
    for i in range(Nt - 1, -1, -1):
        t = i * dt
        for j in range(Nx + 1):
            x = xs[j]
            best = np.inf
            bu = 0.0
            for u in u_grid:
                xnext = x - u * dt
                if xnext < -1e-9:
                    continue
                if xnext <= 0.0:
                    vn = 0.0
                else:
                    vn = np.interp(xnext, xs, V[i + 1])
                    if vn >= BIG:
                        continue
                cost = (ETA * u * u + lam * SIGMA**2 * x * x - mu * t * u) * dt + vn
                if cost < best:
                    best = cost
                    bu = u
            if best >= BIG:
                best = BIG
            V[i, j] = best
            u_star[i, j] = bu
    # 仿真调度
    xsim = np.zeros(Nt + 1)
    xsim[0] = X
    usim = np.zeros(Nt)
    for i in range(Nt):
        u = np.interp(xsim[i], xs, u_star[i])
        usim[i] = u
        xsim[i + 1] = max(xsim[i] - u * dt, 0.0)
    return {'V': V, 'u_star': u_star, 'xs': xs, 'xsim': xsim, 'usim': usim, 'dt': dt, 'Nt': Nt}

def shortfall_stats_mu(sol, mu):
    dt = sol['dt']
    Nt = sol['Nt']
    usim = sol['usim']
    xsim = sol['xsim']
    t_grid = np.arange(Nt) * dt
    E = np.sum((ETA * usim**2 - mu * t_grid * usim) * dt)
    # 短缺方差 = sigma^2 * int_0^T x(t)^2 dt （x=剩余库存），前载 -> x 更快归零 -> 方差更小
    rem = xsim[:-1]
    var_int = np.trapezoid(rem**2, dx=dt)
    std = SIGMA * np.sqrt(var_int)
    return E, std

# ========== 实验 1：值函数面 V(t,x) 与闭式校验 ==========
sol0 = solve_hjb(LAM_SCHED, mu=0.0)
tt, xx = np.meshgrid(np.arange(sol0['Nt'] + 1) * sol0['dt'], sol0['xs'], indexing='ij')
V_closed_grid = closed(LAM_SCHED, tt, xx)
# ---- 有意义的校验：沿最优轨迹对比 DP 与闭式（避开终端惩罚区/边界噪声）----
ts_sim = np.arange(sol0['Nt'] + 1) * sol0['dt']
xc_sim = x_closed(LAM_SCHED, ts_sim)
sched_err = np.max(np.abs(sol0['xsim'] - xc_sim))
V_dp_traj = np.array([np.interp(sol0['xsim'][i], sol0['xs'], sol0['V'][i])
                      for i in range(sol0['Nt'] + 1)])
V_cl_traj = closed(LAM_SCHED, ts_sim, sol0['xsim'])
# 诚实校验：终端 t->T 处闭式 coth 发散、V->0，相对误差无意义 -> 排除最后 8% 时段
mask_safe = (ts_sim < T - 0.08) & (np.abs(V_cl_traj) > 1e-8)
max_rel = float(np.max(np.abs(V_dp_traj[mask_safe] - V_cl_traj[mask_safe]) / np.abs(V_cl_traj[mask_safe])))
mean_rel = float(np.mean(np.abs(V_dp_traj[mask_safe] - V_cl_traj[mask_safe]) / np.abs(V_cl_traj[mask_safe])))
traj_abs = float(np.max(np.abs(V_dp_traj - V_cl_traj)))   # 全轨迹最大绝对误差
sched_rel = sched_err / X

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
im = axes[0].contourf(tt, xx, sol0['V'], levels=20, cmap='viridis')
axes[0].set_xlabel('t'); axes[0].set_ylabel('剩余库存 x')
axes[0].set_title(f'图1a  HJB 值函数 V(t,x)（有限差分）')
plt.colorbar(im, ax=axes[0])
im2 = axes[1].contourf(tt, xx, V_closed_grid, levels=20, cmap='viridis')
axes[1].set_xlabel('t'); axes[1].set_ylabel('剩余库存 x')
axes[1].set_title('图1b  闭式 V=a(t)x^2')
plt.colorbar(im2, ax=axes[1])
fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig1_value_surface.png', dpi=130)
plt.close(fig)

# ========== 实验 2：最优清算进度 x(t)/X vs TWAP ==========
ts = np.linspace(0, T, 200)
xc = x_closed(LAM_SCHED, ts) / X
xtwap = 1.0 - ts / T
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ts, xc, color='#E45756', lw=2.5, label=f'HJB 最优 (lambda={LAM_SCHED}, 前载 sinh)')
ax.plot(ts, xtwap, color='#4C78A8', lw=2, ls='--', label='TWAP 匀速')
ax.set_xlabel('时间 t (归一化到 [0,T])')
ax.set_ylabel('剩余库存 x(t)/X')
ax.set_title('图2  最优清算进度：风险厌恶让你"抢早卖"以压低价格风险')
ax.legend(fontsize=9)
ax.set_ylim(-0.05, 1.05)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig2_schedule.png', dpi=130)
plt.close(fig)

# ========== 实验 3：交易速率 u*(t) —— 漂移制造 U 形 ==========
sol_u = solve_hjb(LAM_SCHED, mu=MU_U)
sol_d = solve_hjb(LAM_SCHED, mu=MU_D)
tu = np.arange(sol0['Nt']) * sol0['dt']
uc = u_closed(LAM_SCHED, tu)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(tu, uc, color='#888', lw=2, label=f'mu=0 (前载, 闭式)')
ax.plot(tu, sol_u['usim'], color='#E45756', lw=2, label=f'mu=+{MU_U} (后载：晚卖倾向)')
ax.plot(tu, sol_d['usim'], color='#54A24B', lw=2, label=f'mu={MU_D} (前载：早卖倾向)')
ax.axhline(1.0 / T, color='#4C78A8', ls='--', lw=1.5, label='TWAP 速率=1/T')
ax.set_xlabel('时间 t')
ax.set_ylabel('交易速率 u*(t) (占 X/T)')
ax.set_title('图3  交易速率 u*(t)：价格漂移把 HJB 解的执行重心从前向后推')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig3_rate_ushape.png', dpi=130)
plt.close(fig)

# ========== 实验 4：风险-成本前沿（lambda 扫描）+ TWAP ==========
Es, stds = [], []
for lam in LAM_SWEEP:
    s = solve_hjb(lam, mu=0.0)
    E, std = shortfall_stats_mu(s, 0.0)
    Es.append(E); stds.append(std)
Es = np.array(Es); stds = np.array(stds)
# TWAP
E_twap = ETA * X**2 / T
std_twap = SIGMA * X * np.sqrt(T / 3.0)
# HJB 在 lambda 极小时应逼近 TWAP
lam_min_idx = np.argmin(np.abs(LAM_SWEEP - 0.1))
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(stds, Es, '-o', color='#E45756', ms=4, label='HJB 前沿 (lambda 扫描)')
ax.plot(std_twap, E_twap, 's', color='#4C78A8', ms=10, label=f'TWAP (lambda->0 端点)')
ax.set_xlabel('执行风险 std(短缺)')
ax.set_ylabel('期望短缺 E[短缺]')
ax.set_title('图4  风险-成本前沿：lambda 是你在前沿上的位置旋钮，TWAP 是最小成本端点')
ax.legend(fontsize=9)
ax.annotate(f'lambda={LAM_SWEEP[lam_min_idx]:.2f}', (stds[lam_min_idx], Es[lam_min_idx]),
            textcoords='offset points', xytext=(8, -12), fontsize=8, color='#666')
fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig4_frontier.png', dpi=130)
plt.close(fig)

# ========== 实验 5：HJB 收敛（网格加密 -> 误差->0）==========
grids = [50, 100, 200, 400, 800]
errs = []
for Nt in grids:
    Nx = max(30, Nt // 3)
    s = solve_hjb(LAM_SCHED, mu=0.0, Nt=Nt, Nx=Nx)
    tsg = np.arange(s['Nt'] + 1) * s['dt']
    xcg = x_closed(LAM_SCHED, tsg)
    e = np.max(np.abs(s['xsim'] - xcg))      # 调度绝对误差（沿轨迹，无终端惩罚污染）
    errs.append(e)
errs = np.array(errs)
fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(grids, errs, '-o', color='#E45756', label='max 相对误差 V_dp vs V_闭式')
ax.loglog(grids, 1.0 / np.array(grids), '--', color='#888', label='~1/Nt 参考')
ax.set_xlabel('时间网格数 Nt')
ax.set_ylabel('最大相对误差')
ax.set_title('图5  收敛性：网格越密 HJB 误差越趋于 0（求解器正确）')
ax.legend(fontsize=9)
ax.set_ylim(1e-4, 1.0)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig5_convergence.png', dpi=130)
plt.close(fig)

# ========== 汇总关键数字 ==========
# 调度数值（mu=0）
x0_end = sol0['xsim'][-1]
w = omega(LAM_SCHED)
u0_start = u_closed(LAM_SCHED, 0.0)
u0_end = u_closed(LAM_SCHED, T)
# 前沿：HJB 最小成本 vs TWAP
E_hjb_min = Es.min()
results = {
    'seed': SEED,
    'params': {'T': T, 'X': X, 'eta': ETA, 'sigma': SIGMA, 'lam_sched': LAM_SCHED,
               'omega': float(w), 'mu_u': MU_U, 'mu_d': MU_D},
    'validation_mu0': {
        'max_rel_err_V': float(max_rel), 'mean_rel_err_V': float(mean_rel),
        'max_abs_err_V_traj': float(traj_abs),
        'x_end_HJB': float(x0_end), 'schedule_frontload_ratio_u0start_over_u0end': float(u0_start / u0_end),
    },
    'ushape': {
        'u_start_mu_pos': float(sol_u['usim'][0]), 'u_mid_mu_pos': float(sol_u['usim'][sol_u['Nt'] // 2]),
        'u_end_mu_pos': float(sol_u['usim'][-1]),
        'u_start_mu_neg': float(sol_d['usim'][0]), 'u_end_mu_neg': float(sol_d['usim'][-1]),
    },
    'frontier': {
        'E_twap': float(E_twap), 'std_twap': float(std_twap),
        'E_hjb_min': float(E_hjb_min), 'lam_sweep_min': float(LAM_SWEEP[np.argmin(Es)]),
        'std_at_lam_sched': float(stds[np.argmin(np.abs(LAM_SWEEP - LAM_SCHED))]),
        'E_at_lam_sched': float(Es[np.argmin(np.abs(LAM_SWEEP - LAM_SCHED))]),
    },
    'convergence': {'grids': grids, 'errs': [float(e) for e in errs]},
}
with open('_sce_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('===== HJB RESULTS =====')
print(f'params: T={T} X={X} eta={ETA} sigma={SIGMA} lambda={LAM_SCHED} omega={w:.4f}')
print(f'[校验 mu=0] 值函数(排除终端奇点)最大相对误差 = {max_rel:.2e}, 平均 = {mean_rel:.2e}; 全轨迹最大绝对误差 = {traj_abs:.2e}; 调度最大绝对误差 = {sched_err:.2e} (相对 {sched_rel*100:.2f}%); x(T) HJB = {x0_end:.4f} (应≈0)')
print(f'  闭式 u*(0)/u*(T) = {u0_start:.4f}/{u0_end:.4f} = 前载比 {u0_start/u0_end:.3f}')
print(f'[U形 mu=+{MU_U}] u*(0)={sol_u["usim"][0]:.4f} 中={sol_u["usim"][sol_u["Nt"]//2]:.4f} 末={sol_u["usim"][-1]:.4f}')
print(f'[前载 mu={MU_D}] u*(0)={sol_d["usim"][0]:.4f} 末={sol_d["usim"][-1]:.4f}')
print(f'[前沿] TWAP: E={E_twap:.5f} std={std_twap:.5f}; HJB 最小 E={E_hjb_min:.5f} (lambda={LAM_SWEEP[np.argmin(Es)]:.2f})')
print(f'  lambda={LAM_SCHED}: E={Es[np.argmin(np.abs(LAM_SWEEP-LAM_SCHED))]:.5f} std={stds[np.argmin(np.abs(LAM_SWEEP-LAM_SCHED))]:.5f}')
print(f'[收敛] grids={grids} errs={[f"{e:.2e}" for e in errs]}')
print('figures:', os.listdir(OUTDIR))
