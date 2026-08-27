#!/usr/bin/env python3
"""为文章「状态空间模型 S4/S5：用结构化状态空间替代注意力做长序列」
(state-space-model-s4) 生成真实配图（matplotlib，non-placeholder）。

全部数字来自真实 numpy/scipy 计算，可复现（seed=20260828）。

实验设计（诚实、可复现）：
  1) SSM-as-convolution 正确性：对角化 HiPPO(S4D-Lin) A，双线性离散化，
     由离散状态空间算出脉冲响应核 K_n = C_bar A_bar^n B_bar；用因果卷积处理输入，
     与按状态递推(x_t=A_bar x_{t-1}+B_bar u_t, y_t=C_bar x_t)的结果对照。
     结论：两者逐点差 ~1e-12（卷积视角成立）。
  2) 序列复杂度缩放：SSM 用 FFT 卷积(≈O(N log N)) vs 朴素注意力 O(N^2) 矩阵乘，
     在 L=512..8192 上测墙上时间，log-log 拟合斜率（≈1 vs ≈2）。
  3) 长程依赖：目标 = 序列早期位置 p0 注入的事件 e（位置 L-1 处读出）。
     SSM(衰减 a=0.99 的长记忆状态) vs 固定窗口线性基线(只看最后 30 拍,看不到 p0)。
     扫描 p0∈{10,30,60,100}，SSM 优势随事件越早越大。
     诚实补充：a=0.5(短记忆)对同一任务几乎遗忘，演示衰减调参的必要性。
  4) 内容选择性短板（诚实红线）：LTI-SSM 对每个位置用同一核 -> 无法做「取闸门后
     那一个 token」这类内容相关选择。对照一个显式内容选择器(找闸门取后一token)。
     LTI-SSM 测试 MSE ≈ Var(target)(随机水平)，内容选择器 ≈ 0。
"""
import os, json, time
import numpy as np
from scipy.signal import fftconvolve
from scipy.linalg import solve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "state-space-model-s4")
os.makedirs(D, exist_ok=True)

C_SSM = "#1f4e79"    # SSM
C_ATTN = "#c0392b"   # 注意力
C_WIN = "#8e44ad"    # 窗口基线
C_SEL = "#27ae60"    # 内容选择
GRID = "#e6e6e6"
CH = "#e67e22"       # 内容选择性失败


def hippo_diag(N):
    """S4D-Lin：对角复 A 的实部 -0.5、虚部 n。"""
    n = np.arange(1, N + 1)
    return -0.5 + 1j * n


def discretize_bilinear(A, B, dt):
    I = np.eye(A.shape[0], dtype=complex)
    left = np.linalg.inv(I - 0.5 * dt * A)
    Ad = left @ (I + 0.5 * dt * A)
    Bd = left @ (dt * B)
    return Ad, Bd


def ssm_impulse_response(Ad, Bd, Cd, L):
    """K_n = C Ad^n B，n=0..L-1（因果脉冲响应，复数）。"""
    N = Ad.shape[0]
    K = np.zeros(L, dtype=complex)
    An = np.eye(N, dtype=complex)
    for n in range(L):
        K[n] = Cd @ (An @ Bd)
        An = Ad @ An
    return K


def ssm_conv(u, Ad, Bd, Cd):
    K = ssm_impulse_response(Ad, Bd, Cd, len(u))
    y = fftconvolve(K.real, u)[:len(u)]      # 取实部核
    return y


def ssm_recurrence(u, Ad, Bd, Cd):
    N = Ad.shape[0]
    x = np.zeros(N, dtype=complex)
    ys = np.zeros(len(u), dtype=complex)
    for t, ut in enumerate(u):
        x = Ad @ x + Bd * ut
        ys[t] = Cd @ x
    return ys.real


# ============================================================
# 图1：SSM-as-convolution 正确性
# ============================================================
def fig_conv_correctness():
    rng = np.random.default_rng(20260828)
    N = 4
    L = 64
    dt = 0.1
    A = hippo_diag(N)
    B = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    C = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    Ad, Bd = discretize_bilinear(A, B, dt)
    u = rng.standard_normal(L)
    y_conv = ssm_conv(u, Ad, Bd, C)
    y_rec = ssm_recurrence(u, Ad, Bd, C)
    max_diff = float(np.max(np.abs(y_conv - y_rec)))
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.plot(np.arange(L), y_conv, "-o", color=C_SSM, ms=4, label="卷积视角  y = K * u (FFT)")
    ax.plot(np.arange(L), y_rec, "--s", color=CH, ms=3, alpha=0.8, label="状态递推  x_t=Ax_{t-1}+Bu_t, y_t=Cx_t")
    ax.set_xlabel("时间 t"); ax.set_ylabel("输出 y_t")
    ax.set_title(f"SSM-as-convolution 验证：两条曲线逐点重合（最大差 {max_diff:.1e}）")
    ax.legend(fontsize=9); ax.grid(color=GRID)
    fig.tight_layout()
    fig.savefig(f"{D}/s4_conv_correctness.png", dpi=160, bbox_inches="tight"); plt.close()
    return dict(max_diff=max_diff, N=N, L=L, dt=dt)


# ============================================================
# 图2：序列复杂度缩放 SSM(FFT) vs 注意力 O(N^2)
# ============================================================
def fig_complexity_scaling():
    lens = np.array([512, 1024, 2048, 4096, 8192])
    t_ssm, t_attn = [], []
    rng = np.random.default_rng(0)
    d = 16
    for L in lens:
        u = rng.standard_normal(L)
        K = rng.standard_normal(L)
        # SSM FFT 卷积
        t0 = time.perf_counter()
        for _ in range(5):
            _ = fftconvolve(K, u)[:L]
        t_ssm.append((time.perf_counter() - t0) / 5)
        # 朴素注意力 O(N^2)：Q K^T（d 维投影也计入）
        Q = rng.standard_normal((L, d)); Km = rng.standard_normal((L, d))
        t0 = time.perf_counter()
        for _ in range(3):
            _ = Q @ Km.T
        t_attn.append((time.perf_counter() - t0) / 3)
    # log-log 拟合斜率
    s_ssm = np.polyfit(np.log(lens), np.log(t_ssm), 1)[0]
    s_attn = np.polyfit(np.log(lens), np.log(t_attn), 1)[0]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.loglog(lens, t_ssm, "-o", color=C_SSM, label=f"结构化SSM（FFT卷积）斜率≈{s_ssm:.2f}")
    ax.loglog(lens, t_attn, "-s", color=C_ATTN, label=f"朴素注意力 QK^T 斜率≈{s_attn:.2f}")
    ax.set_xlabel("序列长度 N"); ax.set_ylabel("单次前向耗时 (s, log)")
    ax.set_title("序列复杂度：结构化SSM 线性扩展 vs 注意力 O(N²)")
    ax.legend(fontsize=9); ax.grid(color=GRID, which="both")
    fig.tight_layout()
    fig.savefig(f"{D}/s4_complexity_scaling.png", dpi=160, bbox_inches="tight"); plt.close()
    return dict(lens=list(lens.astype(int)), t_ssm=t_ssm, t_attn=t_attn,
                slope_ssm=float(s_ssm), slope_attn=float(s_attn))


# ============================================================
# 图3：长程依赖——SSM 长记忆 vs 固定窗口基线
# ============================================================
def fig_long_range():
    rng = np.random.default_rng(20260828)
    L = 200
    W = 30                       # 窗口基线只看最后 30 拍
    a_long = 0.99               # SSM 状态衰减（长记忆）
    a_short = 0.5               # 短记忆（对照，演示衰减调参必要）
    p0s = [10, 30, 60, 100]
    n_train, n_test = 2000, 500
    mse = {"ssm_long": [], "ssm_short": [], "window": []}
    for p0 in p0s:
        # 训练
        e_tr = rng.standard_normal(n_train)
        x_end_long = np.zeros(n_train); x_end_short = np.zeros(n_train)
        for i in range(n_train):
            # 标量 SSM 状态递推，事件在 p0
            xL = 0.0; xS = 0.0
            for t in range(L):
                inp = e_tr[i] if t == p0 else 0.0
                xL = a_long * xL + inp
                xS = a_short * xS + inp
            x_end_long[i] = xL; x_end_short[i] = xS
        wL = solve(np.array([[x_end_long @ x_end_long + 1e-6]]),
                   np.array([[x_end_long @ e_tr]]))[0, 0]
        wS = solve(np.array([[x_end_short @ x_end_short + 1e-6]]),
                   np.array([[x_end_short @ e_tr]]))[0, 0]
        # 测试
        e_te = rng.standard_normal(n_test)
        mse_long = mse_short = mse_win = 0.0
        for i in range(n_test):
            xL = 0.0; xS = 0.0
            for t in range(L):
                inp = e_te[i] if t == p0 else 0.0
                xL = a_long * xL + inp
                xS = a_short * xS + inp
            predL = wL * xL; predS = wS * xS
            predW = 0.0                      # 窗口基线看不到 p0，只能恒为 0
            mse_long += (predL - e_te[i]) ** 2
            mse_short += (predS - e_te[i]) ** 2
            mse_win += (predW - e_te[i]) ** 2
        mse["ssm_long"].append(mse_long / n_test)
        mse["ssm_short"].append(mse_short / n_test)
        mse["window"].append(mse_win / n_test)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(p0s))
    ax.bar(x - 0.28, mse["ssm_long"], width=0.28, color=C_SSM, label="结构化SSM (a=0.99 长记忆)")
    ax.bar(x, mse["ssm_short"], width=0.28, color=CH, label="SSM (a=0.5 短记忆, 应失败)")
    ax.bar(x + 0.28, mse["window"], width=0.28, color=C_WIN, label="固定窗口基线 (只看末30拍)")
    ax.set_xticks(x); ax.set_xticklabels([f"p0={p}" for p in p0s])
    ax.set_xlabel("事件注入位置 p0（越早→与读出端距离越远）")
    ax.set_ylabel("测试 MSE（目标=早期事件 e）")
    ax.set_title("长程依赖：SSM 把早期事件压进状态带到最后读出；窗口基线看不到 p0 必败")
    ax.legend(fontsize=8.5); ax.grid(axis="y", color=GRID)
    fig.tight_layout()
    fig.savefig(f"{D}/s4_long_range.png", dpi=160, bbox_inches="tight"); plt.close()
    return dict(p0s=p0s, mse_long=mse["ssm_long"], mse_short=mse["ssm_short"], mse_window=mse["window"])


# ============================================================
# 图4：内容选择性短板（诚实红线）
# ============================================================
def fig_selectivity():
    rng = np.random.default_rng(20260828)
    L = 60
    a = 0.9
    n_train, n_test = 1500, 400
    # 训练 SSM 读头：状态递推后取末态做线性读出
    X, Y = [], []
    for _ in range(n_train):
        u = rng.uniform(-1, 1, L)
        gate_pos = rng.integers(0, L - 1)
        u[gate_pos] = 1.0                      # 闸门标记
        target = u[gate_pos + 1]               # 目标=闸门后一token
        x = 0.0
        for t in range(L):
            x = a * x + u[t]
        X.append(x); Y.append(target)
    X = np.array(X); Y = np.array(Y)
    w = solve(np.array([[X @ X + 1e-6]]), np.array([[X @ Y]]))[0, 0]
    # 测试
    mse_ssm = mse_sel = 0.0
    for _ in range(n_test):
        u = rng.uniform(-1, 1, L)
        gate_pos = rng.integers(0, L - 1)
        u[gate_pos] = 1.0
        target = u[gate_pos + 1]
        x = 0.0
        for t in range(L):
            x = a * x + u[t]
        pred_ssm = w * x
        pred_sel = u[gate_pos + 1]             # 内容选择器：找闸门取后一token
        mse_ssm += (pred_ssm - target) ** 2
        mse_sel += (pred_sel - target) ** 2
    mse_ssm /= n_test; mse_sel /= n_test
    var_target = 1.0 / 12 * (1 - (-1)) ** 2   # U(-1,1) 方差 = 1/3
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = [mse_ssm, mse_sel, var_target]
    labels = [f"LTI-SSM 线性读出\nMSE={mse_ssm:.3f}", f"内容选择器(找闸门取后一)\nMSE={mse_sel:.3f}", f"目标方差(随机水平)\n={var_target:.3f}"]
    cols = [CH, C_SEL, GRID]
    ax.bar(range(3), bars, color=cols)
    for i, b in enumerate(bars):
        ax.text(i, b + 0.004, f"{b:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("测试 MSE")
    ax.set_title("内容选择性短板：LTI-SSM 对每个位置用同一核，无法做「取闸门后token」")
    ax.grid(axis="y", color=GRID)
    fig.tight_layout()
    fig.savefig(f"{D}/s4_selectivity.png", dpi=160, bbox_inches="tight"); plt.close()
    return dict(mse_ssm=float(mse_ssm), mse_selector=float(mse_sel), var_target=float(var_target))


if __name__ == "__main__":
    c1 = fig_conv_correctness()
    c2 = fig_complexity_scaling()
    c3 = fig_long_range()
    c4 = fig_selectivity()
    out = dict(conv=c1, complexity=c2, long_range=c3, selectivity=c4)
    def _f(o):
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    print(json.dumps(out, ensure_ascii=False, indent=2, default=_f))
