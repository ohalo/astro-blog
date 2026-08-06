#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
波动率目标风险平价：把组合波动锁在固定区间
所有图表由真实计算生成，固定随机种子可复现。
SEED = 20260807
"""
import json, os, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["figure.facecolor"] = "white"

IMG_DIR = "/Users/halo/workspace/astro-blog/public/images/risk-parity-vol-targeting"
os.makedirs(IMG_DIR, exist_ok=True)

SEED = 20260807

C_EW   = "#9ca3af"
C_IV   = "#2563eb"
C_ERC  = "#1e3a5f"
C_VT   = "#dc2626"
C_OK   = "#16a34a"
C_WARN = "#e05c2a"
C_PUR  = "#9333ea"

TD = 252
NAMES = ["股票", "债券", "商品", "信用"]
MU_ANN  = np.array([0.070, 0.025, 0.030, 0.045])
VOL_ANN = np.array([0.180, 0.050, 0.220, 0.080])
CORR = np.array([
    [1.00, -0.15, 0.35, 0.55],
    [-0.15, 1.00, 0.05, 0.25],
    [0.35,  0.05, 1.00, 0.25],
    [0.55,  0.25, 0.25, 1.00],
])
N_A = len(NAMES)

# GARCH(1,1) 参数：alpha+beta=0.98 → 波动高度持续（真实市场的典型量级）
G_ALPHA, G_BETA = 0.080, 0.900
N_DAYS = 25 * TD          # 25 年
LEV_CAP = 3.0
TARGET_VOL = 0.10
COST_BPS = 8.0            # 单边换手成本
EWMA_LAMBDA = 0.94        # 波动预测（约 60 日半衰）
COV_LOOKBACK = 250        # 协方差估计窗


# =============================================================================
# 1. 数据生成：带 GARCH 波动聚集 + 固定相关结构
#    真值全部由构造已知：无条件年化波动 = VOL_ANN，相关矩阵 = CORR
# =============================================================================
def simulate_returns(seed, garch=True, n_days=N_DAYS):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(CORR)
    z = rng.standard_normal((n_days, N_A)) @ L.T      # 标准化冲击，相关结构由构造给定

    mu_d = MU_ANN / TD
    var_d_uncond = (VOL_ANN ** 2) / TD

    if not garch:
        # 同方差世界：无波动聚集
        ret = mu_d + np.sqrt(var_d_uncond) * z
        sig_path = np.tile(np.sqrt(var_d_uncond), (n_days, 1))
        return ret, sig_path

    omega = var_d_uncond * (1.0 - G_ALPHA - G_BETA)   # 保证无条件方差 = var_d_uncond
    sig2 = np.tile(var_d_uncond, (n_days, 1))
    eps = np.zeros((n_days, N_A))
    s2 = var_d_uncond.copy()
    for t in range(n_days):
        sig2[t] = s2
        e = np.sqrt(s2) * z[t]
        eps[t] = e
        s2 = omega + G_ALPHA * e ** 2 + G_BETA * s2
    ret = mu_d + eps
    return ret, np.sqrt(sig2)


# =============================================================================
# 2. 权重方案
# =============================================================================
def erc_weights(cov, iters=3000, tol=1e-14):
    """等风险贡献（ERC）。对角协方差时应精确退化为逆波动率加权。"""
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(iters):
        mrc = cov @ w
        mrc = np.maximum(mrc, 1e-18)
        w_new = (1.0 / n) / mrc
        w_new = w_new / w_new.sum()
        step = 0.5 * w + 0.5 * w_new
        if np.max(np.abs(step - w)) < tol:
            w = step
            break
        w = step
    return w / w.sum()


def inv_vol_weights(cov):
    v = np.sqrt(np.diag(cov))
    w = 1.0 / np.maximum(v, 1e-18)
    return w / w.sum()


def rc_dispersion(cov, w):
    """风险贡献的离散度（相对），用于验证 ERC 求解质量。"""
    mrc = cov @ w
    rc = w * mrc
    rc = rc / rc.sum()
    return float(rc.max() - rc.min())


# =============================================================================
# 3. 回测引擎
#    signal-on-t / execute-on-t（权重用 t 日之前的数据估计，收益记在 t 日）
#    warmup 段（前 COV_LOOKBACK 天）不计入评估
# =============================================================================
def run_backtest(ret, mode, target_vol=TARGET_VOL, lev_cap=LEV_CAP,
                 cost_bps=COST_BPS, cov_lb=COV_LOOKBACK, rebal=21,
                 vol_lb=None, lev_band=0.0, fixed_lev=None):
    n_days = ret.shape[0]
    w_prev = np.zeros(N_A)
    lev_prev = 0.0

    port_ret = np.zeros(n_days)
    lev_path = np.full(n_days, np.nan)
    fcst_path = np.full(n_days, np.nan)
    turnover = np.zeros(n_days)
    w_hist = np.full((n_days, N_A), np.nan)

    # EWMA 组合方差（用于波动预测）
    ew_var = None
    lam = EWMA_LAMBDA if vol_lb is None else np.exp(np.log(0.5) / max(vol_lb, 1))

    for t in range(n_days):
        if t < cov_lb:
            port_ret[t] = 0.0
            continue

        # ---- 权重决策：只用 [t-cov_lb, t) 的数据 ----
        if (t - cov_lb) % rebal == 0 or w_prev.sum() == 0:
            win = ret[t - cov_lb:t]
            cov = np.cov(win, rowvar=False) * TD
            if mode == "ew":
                w = np.ones(N_A) / N_A
            elif mode == "6040":
                w = np.array([0.60, 0.40, 0.0, 0.0])
            elif mode in ("iv", "iv_vt"):
                w = inv_vol_weights(cov)
            else:
                w = erc_weights(cov)
        else:
            w = w_prev.copy()

        # ---- 杠杆决策：波动目标 ----
        if mode.endswith("_vt"):
            fcst = np.sqrt(max(ew_var, 1e-12) * TD) if ew_var is not None else \
                   float(np.std(ret[t - cov_lb:t] @ w) * np.sqrt(TD))
            lev_tgt = min(target_vol / max(fcst, 1e-6), lev_cap)
            # 无交易带：目标杠杆偏离当前杠杆不足 band 时不动
            if lev_band > 0 and lev_prev > 0 and abs(lev_tgt / lev_prev - 1.0) < lev_band:
                lev = lev_prev
            else:
                lev = lev_tgt
            fcst_path[t] = fcst
        elif mode.endswith("_fix"):
            # 常数杠杆：仅用 warmup 窗口估计，无前视
            if fixed_lev is None:
                v0 = float(np.std(ret[:cov_lb] @ w) * np.sqrt(TD))
                fixed_lev = min(target_vol / max(v0, 1e-6), lev_cap)
            lev = fixed_lev
            if ew_var is not None:
                fcst_path[t] = np.sqrt(max(ew_var, 1e-12) * TD)
        else:
            lev = 1.0
            if ew_var is not None:
                fcst_path[t] = np.sqrt(max(ew_var, 1e-12) * TD)

        pos = lev * w
        turnover[t] = np.abs(pos - lev_prev * w_prev).sum()
        gross = float(pos @ ret[t])
        port_ret[t] = gross - turnover[t] * cost_bps * 1e-4

        lev_path[t] = lev
        w_hist[t] = w
        w_prev, lev_prev = w, lev

        # ---- 更新 EWMA（用已实现的当日组合收益，t 日之后才可用）----
        r_w = float(w @ ret[t])
        ew_var = r_w ** 2 if ew_var is None else lam * ew_var + (1 - lam) * r_w ** 2

    return dict(port_ret=port_ret, lev=lev_path, fcst=fcst_path,
                turnover=turnover, w=w_hist, warmup=cov_lb)


def metrics(res, rf=0.0):
    """评估窗口指标必须在 warmup 切片之后重算。"""
    r = res["port_ret"][res["warmup"]:]
    eq = np.cumprod(1.0 + r)
    yrs = len(r) / TD
    cagr = eq[-1] ** (1 / yrs) - 1
    vol = r.std(ddof=1) * np.sqrt(TD)
    sharpe = (r.mean() * TD - rf) / vol if vol > 0 else np.nan
    dd = eq / np.maximum.accumulate(eq) - 1.0
    to = res["turnover"][res["warmup"]:].sum() / yrs
    lv = res["lev"][res["warmup"]:]
    lv = lv[~np.isnan(lv)]
    # 已实现波动的稳定性：63 日滚动年化波动的标准差
    roll = pd.Series(r).rolling(63).std().dropna().values * np.sqrt(TD)
    return dict(cagr=float(cagr), vol=float(vol), sharpe=float(sharpe),
                maxdd=float(dd.min()), calmar=float(cagr / abs(dd.min())),
                turnover=float(to), lev_mean=float(np.mean(lv)) if len(lv) else 1.0,
                lev_max=float(np.max(lv)) if len(lv) else 1.0,
                vol_of_vol=float(roll.std()), roll_vol=roll,
                eq=eq, dd=dd, r=r,
                vol_hit=float(np.mean(np.abs(roll - TARGET_VOL) < 0.02)))


def main():
    print("=" * 62)
    print("波动率目标风险平价 — 受控仿真")
    print("=" * 62)

    print("\n[1/8] 生成 25 年多资产收益（GARCH 波动聚集）...")
    ret, sig = simulate_returns(SEED, garch=True)
    print(f"    实际年化波动: {dict(zip(NAMES, (ret.std(0)*np.sqrt(TD)).round(4)))}")
    print(f"    目标（构造）: {dict(zip(NAMES, VOL_ANN))}")
    emp_corr = np.corrcoef(ret, rowvar=False)
    print(f"    相关矩阵最大偏差: {np.abs(emp_corr - CORR).max():.4f}")

    # ---------- 主对比 ----------
    print("\n[2/8] 运行 4 个策略...")
    strats = {}
    for key, mode, label in [("6040", "6040", "60/40 股债"),
                             ("iv", "iv", "逆波动率 RP"),
                             ("erc", "erc", "等风险贡献 ERC"),
                             ("fix", "erc_fix", "ERC + 常数杠杆（对照）"),
                             ("vt", "erc_vt", "ERC + 波动目标 10%")]:
        strats[key] = metrics(run_backtest(ret, mode))
        m = strats[key]
        print(f"    {label:22s} CAGR={m['cagr']:7.2%}  Vol={m['vol']:6.2%}  "
              f"SR={m['sharpe']:5.3f}  MDD={m['maxdd']:7.2%}  换手={m['turnover']:5.2f}x/y  "
              f"波动命中={m['vol_hit']:5.1%}")

    # 同波动对照（零成本，剔除换手干扰）
    fix0 = metrics(run_backtest(ret, "erc_fix", cost_bps=0.0))
    vt0 = metrics(run_backtest(ret, "erc_vt", cost_bps=0.0))
    print(f"\n    【同波动、零成本】常数杠杆 SR={fix0['sharpe']:.4f} Vol={fix0['vol']:.2%} MDD={fix0['maxdd']:.2%}")
    print(f"    【同波动、零成本】波动目标 SR={vt0['sharpe']:.4f} Vol={vt0['vol']:.2%} MDD={vt0['maxdd']:.2%}")
    print(f"    → 真实增量 SR {vt0['sharpe']-fix0['sharpe']:+.4f}，MDD {vt0['maxdd']-fix0['maxdd']:+.2%}")

    # ---------- ERC 求解质量 ----------
    win = ret[-COV_LOOKBACK:]
    cov_full = np.cov(win, rowvar=False) * TD
    w_erc = erc_weights(cov_full)
    disp_full = rc_dispersion(cov_full, w_erc)
    cov_diag = np.diag(np.diag(cov_full))
    w_erc_d = erc_weights(cov_diag)
    w_iv_d = inv_vol_weights(cov_diag)
    dev_diag = float(np.abs(w_erc_d - w_iv_d).max())
    print(f"\n[3/8] ERC 求解校验：")
    print(f"    完整协方差风险贡献离散度 = {disp_full:.3e}（应≈0）")
    print(f"    对角协方差下 ERC vs 逆波动率最大偏差 = {dev_diag:.3e}（应精确为0）")

    # ---------- 安慰剂 A：抹掉波动聚集 ----------
    print("\n[4/8] 安慰剂 A：时间置换（保留边际分布与同期相关，销毁波动聚集）...")
    rng = np.random.default_rng(SEED + 1)
    perm = rng.permutation(N_DAYS)
    ret_shuf = ret[perm]
    pl_erc = metrics(run_backtest(ret_shuf, "erc"))
    pl_vt = metrics(run_backtest(ret_shuf, "erc_vt"))
    gain_real = strats["vt"]["sharpe"] - strats["erc"]["sharpe"]
    gain_shuf = pl_vt["sharpe"] - pl_erc["sharpe"]
    print(f"    真实世界  ERC SR={strats['erc']['sharpe']:.4f} → VT SR={strats['vt']['sharpe']:.4f}  增益={gain_real:+.4f}")
    print(f"    置换世界  ERC SR={pl_erc['sharpe']:.4f} → VT SR={pl_vt['sharpe']:.4f}  增益={gain_shuf:+.4f}")

    # ---------- 安慰剂 B：同方差世界 ----------
    print("\n[5/8] 安慰剂 B：同方差世界（GARCH 关闭，真值增益=0）...")
    ret_iid, _ = simulate_returns(SEED + 7, garch=False)
    iid_erc = metrics(run_backtest(ret_iid, "erc"))
    iid_vt = metrics(run_backtest(ret_iid, "erc_vt"))
    gain_iid = iid_vt["sharpe"] - iid_erc["sharpe"]
    print(f"    ERC SR={iid_erc['sharpe']:.4f} → VT SR={iid_vt['sharpe']:.4f}  增益={gain_iid:+.4f}")

    # ---------- 安慰剂 C：零成本 ----------
    zc_vt = metrics(run_backtest(ret, "erc_vt", cost_bps=0.0))
    zc_erc = metrics(run_backtest(ret, "erc", cost_bps=0.0))
    print(f"\n[6/8] 安慰剂 C：零成本  ERC SR={zc_erc['sharpe']:.4f}  VT SR={zc_vt['sharpe']:.4f}  "
          f"增益={zc_vt['sharpe']-zc_erc['sharpe']:+.4f}")

    # ---------- 扫描 ----------
    print("\n[7/8] 参数扫描...")
    # 成本扫描
    cost_grid = [0, 2, 5, 8, 12, 20, 30, 50]
    cost_rows = []
    for c in cost_grid:
        a = metrics(run_backtest(ret, "erc", cost_bps=c))
        b = metrics(run_backtest(ret, "erc_vt", cost_bps=c))
        cost_rows.append(dict(cost=c, erc=a["sharpe"], vt=b["sharpe"],
                              gain=b["sharpe"] - a["sharpe"]))
        print(f"    成本 {c:2d}bp: ERC={a['sharpe']:.4f}  VT={b['sharpe']:.4f}  增益={b['sharpe']-a['sharpe']:+.4f}")
    cost_df = pd.DataFrame(cost_rows)
    be = None
    for i in range(1, len(cost_df)):
        g0, g1 = cost_df["gain"].iloc[i - 1], cost_df["gain"].iloc[i]
        if g0 > 0 >= g1:
            c0, c1 = cost_df["cost"].iloc[i - 1], cost_df["cost"].iloc[i]
            be = c0 + (c1 - c0) * g0 / (g0 - g1)
            break

    # 杠杆上限扫描
    cap_grid = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
    cap_rows = []
    for cp in cap_grid:
        m = metrics(run_backtest(ret, "erc_vt", lev_cap=cp))
        cap_rows.append(dict(cap=cp, sharpe=m["sharpe"], cagr=m["cagr"],
                             vol=m["vol"], maxdd=m["maxdd"], lev_mean=m["lev_mean"],
                             to=m["turnover"]))
        print(f"    杠杆上限 {cp:.1f}x: SR={m['sharpe']:.4f}  CAGR={m['cagr']:6.2%}  "
              f"Vol={m['vol']:6.2%}  MDD={m['maxdd']:7.2%}  均值杠杆={m['lev_mean']:.2f}")
    cap_df = pd.DataFrame(cap_rows)

    # 波动预测半衰期扫描
    hl_grid = [5, 10, 21, 42, 63, 126, 250]
    hl_rows = []
    for hl in hl_grid:
        m = metrics(run_backtest(ret, "erc_vt", vol_lb=hl))
        m0 = metrics(run_backtest(ret, "erc_vt", vol_lb=hl, cost_bps=0.0))
        hl_rows.append(dict(hl=hl, sharpe=m["sharpe"], sharpe_free=m0["sharpe"],
                            to=m["turnover"], vov=m["vol_of_vol"],
                            hit=m["vol_hit"], maxdd=m["maxdd"]))
        print(f"    半衰期 {hl:3d}日: SR={m['sharpe']:.4f} (零成本 {m0['sharpe']:.4f})  "
              f"换手={m['turnover']:5.2f}x/y  命中率={m['vol_hit']:.1%}")
    hl_df = pd.DataFrame(hl_rows)

    # 无交易带扫描 —— 修复换手拖累的关键旋钮
    print("\n    无交易带扫描（杠杆偏离多少才调仓）...")
    band_grid = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    band_rows = []
    for bd in band_grid:
        m = metrics(run_backtest(ret, "erc_vt", lev_band=bd))
        m0 = metrics(run_backtest(ret, "erc_vt", lev_band=bd, cost_bps=0.0))
        band_rows.append(dict(band=bd, sharpe=m["sharpe"], sharpe_free=m0["sharpe"],
                              to=m["turnover"], vol=m["vol"], vov=m["vol_of_vol"],
                              hit=m["vol_hit"], maxdd=m["maxdd"], cagr=m["cagr"]))
        print(f"    带宽 {bd:4.0%}: SR={m['sharpe']:.4f} (零成本 {m0['sharpe']:.4f})  "
              f"换手={m['turnover']:6.2f}x/y  已实现波动={m['vol']:.2%}  "
              f"命中率={m['vol_hit']:.1%}")
    band_df = pd.DataFrame(band_rows)
    best_band = band_df.loc[band_df["sharpe"].idxmax()]

    # 波动预测质量：预测 vs 未来21日已实现
    res_probe = run_backtest(ret, "erc_vt")
    fc = res_probe["fcst"]; w_probe = res_probe["w"]
    fut = np.full(N_DAYS, np.nan)
    for t in range(COV_LOOKBACK, N_DAYS - 21):
        if np.isnan(w_probe[t]).any():
            continue
        fut[t] = np.std(ret[t + 1:t + 22] @ w_probe[t]) * np.sqrt(TD)
    ok = (~np.isnan(fc)) & (~np.isnan(fut))
    fc_corr = float(np.corrcoef(fc[ok], fut[ok])[0, 1])
    fc_r2 = fc_corr ** 2
    fc_bias = float(np.mean(fc[ok] - fut[ok]))
    print(f"\n    波动预测质量：corr={fc_corr:.4f}  R²={fc_r2:.4f}  "
          f"平均偏差={fc_bias*100:+.3f}pp  n={ok.sum()}")

    # 目标波动扫描
    tv_grid = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    tv_rows = []
    for tv in tv_grid:
        m = metrics(run_backtest(ret, "erc_vt", target_vol=tv))
        tv_rows.append(dict(tv=tv, sharpe=m["sharpe"], cagr=m["cagr"],
                            vol=m["vol"], maxdd=m["maxdd"], lev=m["lev_mean"]))
    tv_df = pd.DataFrame(tv_rows)

    # 多路径稳健性
    print("\n    多路径检验（40 条独立路径）...")
    paths = []
    for k in range(40):
        r_k, _ = simulate_returns(SEED + 100 + k, garch=True, n_days=12 * TD)
        a = metrics(run_backtest(r_k, "erc"))
        b = metrics(run_backtest(r_k, "erc_vt"))
        paths.append(dict(erc=a["sharpe"], vt=b["sharpe"], gain=b["sharpe"] - a["sharpe"],
                          dd_erc=a["maxdd"], dd_vt=b["maxdd"]))
    pdf = pd.DataFrame(paths)
    win_rate = float((pdf["gain"] > 0).mean())
    dd_better = float((pdf["dd_vt"] > pdf["dd_erc"]).mean())
    print(f"    Sharpe 增益均值={pdf['gain'].mean():+.4f}  中位={pdf['gain'].median():+.4f}  "
          f"胜率={win_rate:.1%}  回撤改善比例={dd_better:.1%}")

    # 危机期行为：找到 ERC 最大回撤区间，看 VT 的杠杆
    res_vt = run_backtest(ret, "erc_vt")
    res_erc = run_backtest(ret, "erc")
    m_erc = metrics(res_erc)
    trough = int(np.argmin(m_erc["dd"])) + COV_LOOKBACK
    peak = int(np.argmax(m_erc["eq"][:trough - COV_LOOKBACK])) + COV_LOOKBACK
    lev_crisis = np.nanmean(res_vt["lev"][peak:trough + 1])
    lev_normal = np.nanmean(res_vt["lev"][COV_LOOKBACK:])
    dd_vt_same = strats["vt"]["dd"][peak - COV_LOOKBACK:trough - COV_LOOKBACK + 1].min()
    print(f"\n    ERC 最大回撤区间：第 {peak} → {trough} 天（{(trough-peak)/TD:.2f} 年）")
    print(f"    该区间 VT 平均杠杆 = {lev_crisis:.3f}（全样本均值 {lev_normal:.3f}，"
          f"降低 {1-lev_crisis/lev_normal:.1%}）")

    # =========================================================================
    # 绘图
    # =========================================================================
    print("\n[8/8] 绘图...")
    idx = np.arange(len(strats["erc"]["eq"])) / TD

    # ---- cover ----
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
    for k, c, lb in [("6040", C_EW, "60/40 股债"), ("iv", C_IV, "逆波动率 RP"),
                     ("erc", C_ERC, "等风险贡献 ERC"), ("fix", C_PUR, "ERC + 常数杠杆"),
                     ("vt", C_VT, "ERC + 波动目标 10%")]:
        ax[0].plot(idx, strats[k]["eq"], color=c, lw=1.9 if k == "vt" else 1.4,
                   label=f"{lb}  SR={strats[k]['sharpe']:.2f}")
    ax[0].set_yscale("log")
    ax[0].set_title("净值曲线（对数轴，含 8bp 单边成本）", fontsize=13, fontweight="bold")
    ax[0].set_xlabel("年"); ax[0].set_ylabel("净值（起始=1）")
    ax[0].legend(fontsize=9, loc="upper left")

    for k, c, lb in [("fix", C_PUR, "ERC + 常数杠杆"), ("vt", C_VT, "ERC + 波动目标")]:
        ax[1].plot(idx[62:], strats[k]["roll_vol"], color=c, lw=1.2, alpha=0.9, label=lb)
    ax[1].axhline(TARGET_VOL, color=C_OK, ls="--", lw=1.8, label=f"目标 {TARGET_VOL:.0%}")
    ax[1].axhspan(TARGET_VOL - 0.02, TARGET_VOL + 0.02, color=C_OK, alpha=0.10)
    ax[1].set_title("63 日滚动已实现年化波动", fontsize=13, fontweight="bold")
    ax[1].set_xlabel("年"); ax[1].set_ylabel("年化波动")
    ax[1].legend(fontsize=9)
    ax[1].text(0.98, 0.96,
               f"波动命中率:\n常数杠杆 {strats['fix']['vol_hit']:.1%}\n"
               f"波动目标 {strats['vt']['vol_hit']:.1%}",
               transform=ax[1].transAxes, ha="right", va="top", fontsize=9,
               bbox=dict(fc="white", ec=C_VT, alpha=0.9))
    plt.tight_layout(); plt.savefig(f"{IMG_DIR}/cover.png"); plt.close()

    # ---- mechanism ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    lv = res_vt["lev"][COV_LOOKBACK:]
    ax[0].plot(idx, lv, color=C_VT, lw=0.8)
    ax[0].axhline(1.0, color=C_GREY if False else "#6b7280", ls=":", lw=1.2, label="1x（不加杠杆）")
    ax[0].axhline(LEV_CAP, color=C_WARN, ls="--", lw=1.4, label=f"上限 {LEV_CAP:.0f}x")
    ax[0].axvspan((peak - COV_LOOKBACK) / TD, (trough - COV_LOOKBACK) / TD,
                  color=C_WARN, alpha=0.15, label="ERC 最大回撤区间")
    ax[0].set_title(f"杠杆路径（均值 {lev_normal:.2f}x，危机区间 {lev_crisis:.2f}x）",
                    fontsize=12, fontweight="bold")
    ax[0].set_xlabel("年"); ax[0].set_ylabel("杠杆倍数"); ax[0].legend(fontsize=8)

    w_avg = np.nanmean(res_erc["w"][COV_LOOKBACK:], axis=0)
    cov_avg = np.cov(ret[COV_LOOKBACK:], rowvar=False) * TD
    mrc = cov_avg @ w_avg
    rc = w_avg * mrc; rc = rc / rc.sum()
    w_6040 = np.array([0.60, 0.40, 0.0, 0.0])
    rc_6040 = w_6040 * (cov_avg @ w_6040); rc_6040 = rc_6040 / rc_6040.sum()
    x = np.arange(N_A); wd = 0.36
    ax[1].bar(x - wd / 2, rc_6040 * 100, wd, color=C_EW, label="60/40")
    ax[1].bar(x + wd / 2, rc * 100, wd, color=C_ERC, label="ERC")
    ax[1].axhline(25, color=C_OK, ls="--", lw=1.5, label="等风险 25%")
    ax[1].set_xticks(x); ax[1].set_xticklabels(NAMES)
    ax[1].set_title("风险贡献占比（%）", fontsize=12, fontweight="bold")
    ax[1].set_ylabel("占组合总风险 %"); ax[1].legend(fontsize=9)
    for i, v in enumerate(rc_6040 * 100):
        ax[1].text(i - wd / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    for i, v in enumerate(rc * 100):
        ax[1].text(i + wd / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8)

    ax[2].plot(idx, strats["fix"]["dd"] * 100, color=C_PUR, lw=1.1, label="ERC + 常数杠杆")
    ax[2].plot(idx, strats["vt"]["dd"] * 100, color=C_VT, lw=1.1, label="ERC + 波动目标")
    ax[2].fill_between(idx, strats["vt"]["dd"] * 100, 0, color=C_VT, alpha=0.12)
    ax[2].set_title(f"同波动对照的回撤（{strats['fix']['maxdd']:.1%} vs {strats['vt']['maxdd']:.1%}）",
                    fontsize=12, fontweight="bold")
    ax[2].set_xlabel("年"); ax[2].set_ylabel("回撤 %"); ax[2].legend(fontsize=9)
    plt.tight_layout(); plt.savefig(f"{IMG_DIR}/mechanism.png"); plt.close()

    # ---- placebo ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    labels = ["真实世界\n(GARCH 波动聚集)", "安慰剂A\n时间置换", "安慰剂B\n同方差世界"]
    gains = [gain_real, gain_shuf, gain_iid]
    cols = [C_VT, C_PUR, C_OK]
    b = ax[0].bar(labels, gains, color=cols, alpha=0.88)
    ax[0].axhline(0, color="black", lw=1.1)
    for r, g in zip(b, gains):
        ax[0].text(r.get_x() + r.get_width() / 2, g + (0.008 if g >= 0 else -0.02),
                   f"{g:+.4f}", ha="center", fontsize=10, fontweight="bold")
    ax[0].set_title("波动目标带来的 Sharpe 增益", fontsize=12, fontweight="bold")
    ax[0].set_ylabel("ΔSharpe")

    # 波动可预测性：|r_t| 的自相关
    def acf_absr(rr, lags=40):
        s = pd.Series(np.abs(rr @ (np.ones(N_A) / N_A)))
        return [s.autocorr(l) for l in range(1, lags + 1)]
    ax[1].plot(range(1, 41), acf_absr(ret), color=C_VT, lw=1.8, marker="o", ms=3,
               label="真实世界")
    ax[1].plot(range(1, 41), acf_absr(ret_shuf), color=C_PUR, lw=1.4, marker="s", ms=3,
               label="时间置换")
    ax[1].plot(range(1, 41), acf_absr(ret_iid), color=C_OK, lw=1.4, marker="^", ms=3,
               label="同方差世界")
    ax[1].axhline(0, color="black", lw=1.0)
    ax[1].set_title("|收益| 自相关 = 波动可预测性", fontsize=12, fontweight="bold")
    ax[1].set_xlabel("滞后（日）"); ax[1].set_ylabel("自相关"); ax[1].legend(fontsize=9)

    ax[2].hist(pdf["gain"], bins=14, color=C_VT, alpha=0.75, edgecolor="white")
    ax[2].axvline(0, color="black", lw=1.4)
    ax[2].axvline(pdf["gain"].mean(), color=C_OK, ls="--", lw=1.8,
                  label=f"均值 {pdf['gain'].mean():+.3f}")
    ax[2].set_title(f"40 条独立路径的 Sharpe 增益（胜率 {win_rate:.0%}）",
                    fontsize=12, fontweight="bold")
    ax[2].set_xlabel("ΔSharpe"); ax[2].set_ylabel("路径数"); ax[2].legend(fontsize=9)
    plt.tight_layout(); plt.savefig(f"{IMG_DIR}/placebo.png"); plt.close()

    # ---- scan ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].plot(cost_df["cost"], cost_df["gain"], color=C_VT, lw=2, marker="o")
    ax[0].axhline(0, color="black", lw=1.2)
    if be:
        ax[0].axvline(be, color=C_WARN, ls="--", lw=1.6, label=f"盈亏平衡 {be:.1f}bp")
    ax[0].fill_between(cost_df["cost"], cost_df["gain"], 0,
                       where=cost_df["gain"] > 0, color=C_OK, alpha=0.15)
    ax[0].fill_between(cost_df["cost"], cost_df["gain"], 0,
                       where=cost_df["gain"] <= 0, color=C_WARN, alpha=0.15)
    ax[0].set_title("成本扫描：波动目标的净增益", fontsize=12, fontweight="bold")
    ax[0].set_xlabel("单边成本 (bp)"); ax[0].set_ylabel("ΔSharpe"); ax[0].legend(fontsize=9)

    a2 = ax[1]
    a2.plot(hl_df["hl"], hl_df["sharpe"], color=C_VT, lw=2, marker="o", label="含8bp成本")
    a2.plot(hl_df["hl"], hl_df["sharpe_free"], color=C_OK, lw=1.6, ls="--",
            marker="^", ms=4, label="零成本")
    a2.set_xscale("log"); a2.set_xticks(hl_grid); a2.set_xticklabels(hl_grid)
    a2.set_xlabel("波动预测半衰期（日）"); a2.set_ylabel("Sharpe")
    a3 = a2.twinx(); a3.grid(False)
    a3.plot(hl_df["hl"], hl_df["to"], color=C_IV, lw=1.4, ls=":", marker="s", ms=4)
    a3.set_ylabel("年换手（倍）", color=C_IV)
    a2.set_title("波动预测窗口：反应速度 vs 换手", fontsize=12, fontweight="bold")
    a2.axhline(strats["erc"]["sharpe"], color=C_ERC, ls=":", lw=1.4)
    a2.text(hl_grid[0], strats["erc"]["sharpe"], " ERC 基准", fontsize=8, va="bottom")
    a2.legend(fontsize=8, loc="lower right")

    a4 = ax[2]
    a4.plot(cap_df["cap"], cap_df["sharpe"], color=C_VT, lw=2, marker="o", label="Sharpe")
    a4.set_xlabel("杠杆上限（倍）"); a4.set_ylabel("Sharpe", color=C_VT)
    a5 = a4.twinx(); a5.grid(False)
    a5.plot(cap_df["cap"], cap_df["maxdd"] * 100, color=C_WARN, lw=1.6, ls="--",
            marker="s", label="最大回撤")
    a5.set_ylabel("最大回撤 %", color=C_WARN)
    a4.set_title("杠杆上限：Sharpe 封顶但回撤不封顶", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{IMG_DIR}/scan.png"); plt.close()

    # ---- target vol ----
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    ax[0].plot(tv_df["tv"] * 100, tv_df["vol"] * 100, color=C_VT, lw=2, marker="o",
               label="已实现波动")
    ax[0].plot(tv_df["tv"] * 100, tv_df["tv"] * 100, color=C_OK, ls="--", lw=1.6,
               label="45°（完美跟踪）")
    ax[0].set_xlabel("目标波动 %"); ax[0].set_ylabel("已实现波动 %")
    ax[0].set_title("目标 vs 已实现：高目标处被杠杆上限截断", fontsize=12, fontweight="bold")
    ax[0].legend(fontsize=9)
    for _, r in tv_df.iterrows():
        ax[0].annotate(f"{r['lev']:.1f}x", (r["tv"] * 100, r["vol"] * 100),
                       textcoords="offset points", xytext=(6, -10), fontsize=8)

    ax[1].plot(tv_df["vol"] * 100, tv_df["cagr"] * 100, color=C_ERC, lw=2, marker="o")
    for _, r in tv_df.iterrows():
        ax[1].annotate(f"目标{r['tv']:.0%}", (r["vol"] * 100, r["cagr"] * 100),
                       textcoords="offset points", xytext=(5, 6), fontsize=8)
    ax[1].set_xlabel("已实现波动 %"); ax[1].set_ylabel("CAGR %")
    ax[1].set_title("风险-收益前沿：目标波动只是沿线滑动", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{IMG_DIR}/target_vol.png"); plt.close()

    # ---- band ----
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    ax[0].plot(band_df["band"] * 100, band_df["sharpe"], color=C_VT, lw=2,
               marker="o", label="含 8bp 成本")
    ax[0].plot(band_df["band"] * 100, band_df["sharpe_free"], color=C_OK, lw=1.6,
               ls="--", marker="^", label="零成本")
    ax[0].axhline(strats["erc"]["sharpe"], color=C_ERC, ls=":", lw=1.6,
                  label=f"ERC 不加波动目标 {strats['erc']['sharpe']:.3f}")
    ax[0].scatter([best_band["band"] * 100], [best_band["sharpe"]], s=140,
                  facecolor="none", edgecolor=C_WARN, lw=2.2, zorder=5)
    ax[0].set_xlabel("无交易带宽（杠杆相对偏离 %）"); ax[0].set_ylabel("Sharpe")
    ax[0].set_title("无交易带：把换手拖累压回来", fontsize=12, fontweight="bold")
    ax[0].legend(fontsize=8)

    a6 = ax[1]
    a6.plot(band_df["band"] * 100, band_df["to"], color=C_IV, lw=2, marker="s")
    a6.set_xlabel("无交易带宽 %"); a6.set_ylabel("年换手（倍）", color=C_IV)
    a6.set_yscale("log")
    a7 = a6.twinx(); a7.grid(False)
    a7.plot(band_df["band"] * 100, band_df["hit"] * 100, color=C_WARN, lw=1.8,
            ls="--", marker="o")
    a7.set_ylabel("波动命中率 %", color=C_WARN)
    a6.set_title("换手骤降，波动控制几乎没变差", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{IMG_DIR}/band.png"); plt.close()

    # ---- stats.json ----
    stats = {
        "seed": SEED, "n_days": N_DAYS, "n_years": N_DAYS / TD,
        "assets": NAMES, "mu_ann": MU_ANN.tolist(), "vol_ann": VOL_ANN.tolist(),
        "garch": {"alpha": G_ALPHA, "beta": G_BETA, "persistence": G_ALPHA + G_BETA},
        "params": {"target_vol": TARGET_VOL, "lev_cap": LEV_CAP, "cost_bps": COST_BPS,
                   "ewma_lambda": EWMA_LAMBDA, "cov_lookback": COV_LOOKBACK, "rebal_days": 21},
        "main": {k: {kk: round(float(vv), 6) for kk, vv in v.items()
                     if kk in ("cagr", "vol", "sharpe", "maxdd", "calmar", "turnover",
                               "lev_mean", "lev_max", "vol_of_vol", "vol_hit")}
                 for k, v in strats.items()},
        "erc_check": {"rc_dispersion_full": disp_full,
                      "diag_vs_invvol_maxdev": dev_diag},
        "corr_check": float(np.abs(emp_corr - CORR).max()),
        "placebo": {
            "real_gain": round(gain_real, 6),
            "shuffle_gain": round(gain_shuf, 6),
            "iid_gain": round(gain_iid, 6),
            "shuffle_erc_sharpe": round(pl_erc["sharpe"], 6),
            "shuffle_vt_sharpe": round(pl_vt["sharpe"], 6),
            "iid_erc_sharpe": round(iid_erc["sharpe"], 6),
            "iid_vt_sharpe": round(iid_vt["sharpe"], 6),
            "zerocost_gain": round(zc_vt["sharpe"] - zc_erc["sharpe"], 6),
        },
        "cost_scan": cost_df.round(6).to_dict("list"),
        "cost_breakeven_bps": round(float(be), 2) if be else None,
        "cap_scan": cap_df.round(6).to_dict("list"),
        "halflife_scan": hl_df.round(6).to_dict("list"),
        "band_scan": band_df.round(6).to_dict("list"),
        "best_band": {k: round(float(v), 6) for k, v in best_band.items()},
        "forecast_quality": {"corr": round(fc_corr, 6), "r2": round(fc_r2, 6),
                             "bias_pp": round(fc_bias * 100, 6), "n": int(ok.sum())},
        "matched_vol_control": {
            "fix_sharpe_free": round(fix0["sharpe"], 6), "vt_sharpe_free": round(vt0["sharpe"], 6),
            "fix_vol": round(fix0["vol"], 6), "vt_vol": round(vt0["vol"], 6),
            "fix_maxdd": round(fix0["maxdd"], 6), "vt_maxdd": round(vt0["maxdd"], 6),
            "fix_hit": round(fix0["vol_hit"], 6), "vt_hit": round(vt0["vol_hit"], 6),
            "sharpe_gain": round(vt0["sharpe"] - fix0["sharpe"], 6),
            "maxdd_gain": round(vt0["maxdd"] - fix0["maxdd"], 6)},
        "target_vol_scan": tv_df.round(6).to_dict("list"),
        "paths": {"n": len(pdf), "gain_mean": round(float(pdf["gain"].mean()), 6),
                  "gain_median": round(float(pdf["gain"].median()), 6),
                  "gain_sd": round(float(pdf["gain"].std()), 6),
                  "win_rate": win_rate, "dd_worse_rate": dd_better,
                  "gain_p05": round(float(pdf["gain"].quantile(0.05)), 6),
                  "gain_p95": round(float(pdf["gain"].quantile(0.95)), 6)},
        "crisis": {"peak_day": peak, "trough_day": trough,
                   "len_years": round((trough - peak) / TD, 3),
                   "lev_crisis": round(float(lev_crisis), 4),
                   "lev_full": round(float(lev_normal), 4),
                   "lev_cut_pct": round(float(1 - lev_crisis / lev_normal), 4),
                   "erc_dd": round(float(m_erc["dd"].min()), 6),
                   "vt_dd_same_window": round(float(dd_vt_same), 6)},
        "rc_6040": [round(float(v), 6) for v in rc_6040],
        "rc_erc": [round(float(v), 6) for v in rc],
        "w_erc_avg": [round(float(v), 6) for v in w_avg],
    }
    with open(f"{IMG_DIR}/stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 62)
    for fn in sorted(os.listdir(IMG_DIR)):
        print(f"  {fn:22s} {os.path.getsize(f'{IMG_DIR}/{fn})' if False else f'{IMG_DIR}/{fn}')/1024:8.1f} KB")
    print("=" * 62)
    print("\n关键数字：")
    print(f"  60/40      SR={strats['6040']['sharpe']:.4f} MDD={strats['6040']['maxdd']:.2%}")
    print(f"  逆波动率   SR={strats['iv']['sharpe']:.4f} MDD={strats['iv']['maxdd']:.2%}")
    print(f"  ERC        SR={strats['erc']['sharpe']:.4f} MDD={strats['erc']['maxdd']:.2%} CAGR={strats['erc']['cagr']:.2%}")
    print(f"  ERC+VT     SR={strats['vt']['sharpe']:.4f} MDD={strats['vt']['maxdd']:.2%} CAGR={strats['vt']['cagr']:.2%}")
    print(f"  60/40 风险贡献: {dict(zip(NAMES, (np.array(rc_6040)*100).round(1)))}")
    print(f"  盈亏平衡成本 ≈ {be:.1f} bp" if be else "  无盈亏平衡点（全区间为负）")
    print(f"  波动命中率 ERC={strats['erc']['vol_hit']:.1%}  VT={strats['vt']['vol_hit']:.1%}")
    print(f"  最优无交易带 = {best_band['band']:.0%}  SR={best_band['sharpe']:.4f}  "
          f"换手={best_band['to']:.2f}x/y")
    print(f"  波动预测 R² = {fc_r2:.4f}")
    print(f"  【关键】同波动零成本对照: 常数杠杆 SR={fix0['sharpe']:.4f} MDD={fix0['maxdd']:.2%} "
          f"vs 波动目标 SR={vt0['sharpe']:.4f} MDD={vt0['maxdd']:.2%}")


if __name__ == "__main__":
    main()
