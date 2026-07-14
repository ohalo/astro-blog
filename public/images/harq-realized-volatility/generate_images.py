# -*- coding: utf-8 -*-
"""
已实现半方差 / HARQ 配图生成脚本
生成 4 张 PNG 到本脚本同级目录：
  1. harq_semivariance_decomp.png - 上行/下行半方差分解时序
  2. harq_forecast_compare.png    - HAR vs HARQ 预测对比
  3. harq_qlike_bars.png          - 样本外 QLIKE/RMSE 对比柱状图
  4. harq_jump_continuous.png     - 隔夜跳变 vs 日内连续波动散点

数据为自洽合成：潜在波动带杠杆效应（下行冲击抬升次日波动更多）+
测量误差随 realized quarticity 放大。因此：
  - 拆下行半方差(HAR-RS) 能提升预测；
  - 用 RQ 修正测量误差(HARQ) 能进一步提升。
仅用于演示方法，真实落地请用真实高频 tick / 分钟数据，见文章文末说明。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["PingFang SC", "Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
np.random.seed(20260715)


def simulate(n_days=1500, n_intraday=78):
    """
    潜在真实方差 h_t 自回归，且对『前一日下行冲击』响应更强（杠杆效应）。
    每日观测的 RV = h_t 的高频估计，带测量误差；误差方差 ∝ realized quarticity。
    返回 DataFrame: RV, RSV_pos, RSV_neg, JumpVar, RQ, h_true。
    """
    h = 1e-4
    rows = []
    prev_down = 0.0
    for _ in range(n_days):
        # 杠杆效应：下行半方差抬升次日潜在波动更多（保证平稳：系数和 < 1）
        h = 3e-5 + 0.55 * h + 0.55 * prev_down
        # 用潜在波动生成日内 5 分钟收益
        sig = np.sqrt(h / n_intraday)
        cont = np.random.randn(n_intraday) * sig
        # 偶发跳变，负跳更频繁更大
        jumps = np.zeros(n_intraday)
        if np.random.rand() < 0.12:
            idx = np.random.randint(n_intraday)
            sign = -1 if np.random.rand() < 0.62 else 1
            jumps[idx] = sign * abs(np.random.randn()) * 0.9 * np.sqrt(h)
        r = cont + jumps
        rv = np.sum(r ** 2)
        rsv_pos = np.sum(r[r > 0] ** 2)
        rsv_neg = np.sum(r[r < 0] ** 2)
        jump_var = np.sum(jumps ** 2)
        rq = (n_intraday / 3.0) * np.sum(r ** 4)  # realized quarticity 估计
        rows.append((rv, rsv_pos, rsv_neg, jump_var, rq, h))
        prev_down = rsv_neg
    return pd.DataFrame(rows, columns=["RV", "RSV_pos", "RSV_neg", "JumpVar", "RQ", "h_true"])


def refit_split(target, Xcols, split):
    """样本内拟合，样本外预测。返回全序列预测（含 NaN 对齐）。"""
    Xf = np.column_stack(Xcols)
    valid = ~(np.isnan(Xf).any(axis=1) | np.isnan(target))
    idx = np.where(valid)[0]
    tr = idx[idx < split]
    Xd_tr = np.column_stack([np.ones(len(tr)), Xf[tr]])
    beta, *_ = np.linalg.lstsq(Xd_tr, target[tr], rcond=None)
    pred = np.full(len(target), np.nan)
    te = idx[idx >= split]
    Xd_te = np.column_stack([np.ones(len(te)), Xf[te]])
    pred[te] = Xd_te @ beta
    return pred


def qlike(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, None)
    y_true = np.clip(y_true, 1e-10, None)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)


def main():
    df = simulate()
    rv = df["RV"].values
    target = rv.copy()
    n = len(rv)
    split = int(n * 0.6)

    d = pd.Series(rv).shift(1).values
    w = pd.Series(rv).rolling(5).mean().shift(1).values
    m = pd.Series(rv).rolling(22).mean().shift(1).values
    d_pos = df["RSV_pos"].shift(1).values
    d_neg = df["RSV_neg"].shift(1).values
    rq = df["RQ"].shift(1).values
    interact = np.sqrt(np.clip(rq, 0, None)) * d  # HARQ 测量误差修正项

    p_har = refit_split(target, [d, w, m], split)
    p_hars = refit_split(target, [d_pos, d_neg, w, m], split)
    p_harq = refit_split(target, [d, w, m, interact], split)
    p_harq_rs = refit_split(target, [d_pos, d_neg, w, m, interact], split)

    def metrics(pred):
        mm = ~np.isnan(pred)
        yt, yp = target[mm], pred[mm]
        rmse = np.sqrt(np.mean((yt - yp) ** 2))
        return qlike(yt, yp), rmse

    # ---------- 图1: 半方差分解时序 ----------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sl = slice(300, 500)
    ax.plot(df["RSV_pos"].values[sl] * 1e4, label="上行半方差 RSV⁺", color="#2ca02c", lw=1.1)
    ax.plot(-df["RSV_neg"].values[sl] * 1e4, label="下行半方差 RSV⁻（镜像）", color="#d62728", lw=1.1)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_title("已实现半方差分解：下行冲击抬升次日波动更多（杠杆效应）", fontsize=13)
    ax.set_xlabel("交易日")
    ax.set_ylabel("半方差 ×10⁴")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "harq_semivariance_decomp.png"), dpi=130)
    plt.close(fig)

    # ---------- 图2: 预测对比 ----------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sl2 = slice(split + 50, split + 200)
    ax.plot(target[sl2] * 1e4, label="真实 RV", color="#333", lw=1.4)
    ax.plot(p_har[sl2] * 1e4, label="HAR", color="#1f77b4", lw=1.0, alpha=0.85)
    ax.plot(p_harq[sl2] * 1e4, label="HARQ", color="#ff7f0e", lw=1.2)
    ax.set_title("样本外 HAR vs HARQ：HARQ 在波动突变段跟随更快、过冲更小", fontsize=13)
    ax.set_xlabel("交易日（样本外）")
    ax.set_ylabel("已实现方差 RV ×10⁴")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "harq_forecast_compare.png"), dpi=130)
    plt.close(fig)

    # ---------- 图3: QLIKE / RMSE 柱状 ----------
    names = ["HAR", "HAR-RS", "HARQ", "HARQ-RS"]
    preds = [p_har, p_hars, p_harq, p_harq_rs]
    ql = [metrics(p)[0] for p in preds]
    rm = [metrics(p)[1] for p in preds]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    c = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    axes[0].bar(names, ql, color=c)
    axes[0].set_title("样本外 QLIKE（越小越好）")
    axes[0].grid(alpha=0.3, axis="y")
    for i, v in enumerate(ql):
        axes[0].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    axes[1].bar(names, np.array(rm) * 1e4, color=c)
    axes[1].set_title("样本外 RMSE ×10⁴（越小越好）")
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("样本外损失对比：拆半方差 + RQ 测量误差修正逐级降低损失", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "harq_qlike_bars.png"), dpi=130)
    plt.close(fig)

    # ---------- 图4: 跳变 vs 连续 散点 ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    cont_var = (df["RV"] - df["JumpVar"]).clip(lower=0)
    ax.scatter(cont_var * 1e4, df["JumpVar"] * 1e4, s=10, alpha=0.4, color="#9467bd")
    ax.set_xlabel("连续波动部分（RV − 跳变方差）×10⁴")
    ax.set_ylabel("跳变方差 JumpVar ×10⁴")
    ax.set_title("隔夜/日内跳变 vs 连续扩散：跳变贡献尖峰厚尾", fontsize=13)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "harq_jump_continuous.png"), dpi=130)
    plt.close(fig)

    print("Out-of-sample QLIKE:", dict(zip(names, [round(x, 4) for x in ql])))
    print("Out-of-sample RMSE(x1e4):", dict(zip(names, [round(x * 1e4, 3) for x in rm])))
    print("Saved 4 PNGs to", OUT_DIR)


if __name__ == "__main__":
    main()
