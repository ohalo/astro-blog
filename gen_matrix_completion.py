#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""矩阵补全（Matrix Completion）填补因子面板缺失
E1 面板与缺失机制：MCAR vs 停牌块状缺失
E2 soft-impute：lambda 路径（验证误差 U 型 / 核范数 / 有效秩 / 收敛）
E3 填补精度对拍：列均值 / 前向填充 / 零填+SVD / soft-impute / oracle
E4 下游：协方差误差 + 全局最小方差组合样本外波动
E5 缺失率扫描 + 秩误设扫描
输出 5 张图 + _mc_results.json
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for f in ["Heiti TC", "PingFang SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

OUT = "/Users/halo/workspace/astro-blog/public/images/matrix-completion-factor-panel"
os.makedirs(OUT, exist_ok=True)
RES = {}

T, N, R = 120, 200, 4
SIG_B, SIG_E = 0.035, 0.060
T_OOS = 60


def make_panel(seed, T=T, N=N, r=R, sig_b=SIG_B, sig_e=SIG_E, t_oos=T_OOS):
    rg = np.random.default_rng(seed)
    F = rg.normal(0, 1, (T + t_oos, r))
    B = rg.normal(0, sig_b, (N, r))
    E = rg.normal(0, sig_e, (T + t_oos, N))
    Xall = F @ B.T + E
    return Xall[:T], Xall[T:], B, (F @ B.T)[:T]


def mask_mcar(T, N, m, rg):
    return rg.random((T, N)) > m


def mask_block(T, N, m, rg, lam_len=8):
    """停牌式块状缺失：每列若干段连续缺失，段长 ~ Poisson(lam_len)"""
    M = np.ones((T, N), dtype=bool)
    for j in range(N):
        need = int(m * T)
        while need > 0:
            L = min(int(rg.poisson(lam_len)) + 2, need)
            st = rg.integers(0, max(T - L, 1))
            M[st:st + L, j] = False
            need -= L
    return M


# ---------------------------------------------------------------- soft-impute
def soft_impute(Xobs, mask, lam, max_iter=400, tol=1e-7):
    Z = np.where(mask, Xobs, 0.0)
    hist, ranks = [], []
    for it in range(max_iter):
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        s_t = np.maximum(s - lam, 0.0)
        Zt = (U * s_t) @ Vt
        obj = 0.5 * float(np.sum((Xobs[mask] - Zt[mask]) ** 2)) + lam * float(s_t.sum())
        hist.append(obj)
        ranks.append(int((s_t > 1e-10).sum()))
        Zn = np.where(mask, Xobs, Zt)
        delta = float(np.linalg.norm(Zn - Z) / max(np.linalg.norm(Z), 1e-12))
        Z = Zn
        if delta < tol:
            break
    return Z, np.array(hist), np.array(ranks), it + 1


def fill_colmean(Xobs, mask):
    Z = np.array(Xobs, dtype=float)
    colmean = np.where(mask.sum(0) > 0, (np.where(mask, Xobs, 0.0)).sum(0) / np.maximum(mask.sum(0), 1), 0.0)
    Z = np.where(mask, Z, colmean[None, :])
    return Z


def fill_ffill(Xobs, mask):
    Z = np.where(mask, Xobs, np.nan)
    for j in range(Z.shape[1]):
        col = Z[:, j]
        last = np.nan
        if np.any(mask[:, j]):
            first = np.argmax(mask[:, j])
            last = col[first]
        out = np.empty(len(col))
        for t in range(len(col)):
            if mask[t, j]:
                last = col[t]
            out[t] = last
        Z[:, j] = out
    return Z


def fill_zero_svd(Xobs, mask, rank):
    Z0 = np.where(mask, Xobs, 0.0)
    U, s, Vt = np.linalg.svd(Z0, full_matrices=False)
    s[rank:] = 0.0
    Zt = (U * s) @ Vt
    return np.where(mask, Xobs, Zt)


def cov_from(Z):
    Zc = Z - Z.mean(0)
    return Zc.T @ Zc / (Z.shape[0] - 1)


def cov_pairwise(Xobs, mask):
    """逐对删失协方差：每对用两者都观测到的月份"""
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            ok = mask[:, i] & mask[:, j]
            if ok.sum() > 2:
                a, b = Xobs[ok, i], Xobs[ok, j]
                v = np.mean((a - a.mean()) * (b - b.mean())) * (ok.sum() / (ok.sum() - 1))
            else:
                v = 0.0
            C[i, j] = C[j, i] = v
    return C


def gmv_weights(Sig, ridge):
    Sr = Sig + ridge * np.eye(Sig.shape[0])
    w = np.linalg.solve(Sr, np.ones(Sig.shape[0]))
    return w / w.sum()


def subspace_cos(Sig_est, B, r=R):
    """top-r 特征子空间 vs 真载荷子空间的 principal angle cos"""
    w, V = np.linalg.eigh(Sig_est)
    Qe = V[:, ::-1][:, :r]
    Qb, _ = np.linalg.qr(B)
    sv = np.linalg.svd(Qe.T @ Qb, compute_uv=False)
    return float(sv.min()), float(sv.mean())


# ---------------------------------------------------------------- E1/E2 主实验
SEED0 = 20261027
Xfull, Xoos, Btrue, Ltrue = make_panel(SEED0)
SIG_TRUE = Btrue @ Btrue.T + SIG_E ** 2 * np.eye(N)

rg = np.random.default_rng(SEED0)
mask_m = mask_mcar(T, N, 0.30, rg)
mask_b = mask_block(T, N, 0.30, rg)
Xobs_m = np.where(mask_m, Xfull, np.nan)
RES["missing_rate_actual"] = {"mcar": round(float(1 - mask_m.mean()), 4),
                              "block": round(float(1 - mask_b.mean()), 4)}


def run_pipeline(Xfull, mask, seed, nlam=24, rank=R):
    rg = np.random.default_rng(seed)
    Xobs = np.where(mask, Xfull, 0.0)
    # 从已观测中再挖 10% 做验证
    obs_idx = np.argwhere(mask)
    val_sel = rg.random(len(obs_idx)) < 0.10
    mask_tr = mask.copy()
    mask_tr[obs_idx[val_sel, 0], obs_idx[val_sel, 1]] = False
    mask_val = np.zeros_like(mask); mask_val[obs_idx[val_sel, 0], obs_idx[val_sel, 1]] = True
    s1 = np.linalg.svd(np.where(mask_tr, Xfull, 0.0), compute_uv=False)[0]
    alphas = np.logspace(np.log10(0.02), np.log10(0.90), nlam)
    lams = alphas * s1
    val_err, ranks, nucs = [], [], []
    for lam in lams:
        Z, _, rp, _ = soft_impute(Xobs, mask_tr, lam)
        val_err.append(float(np.sqrt(np.mean((Z[mask_val] - Xfull[mask_val]) ** 2))))
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        ranks.append(int(rp[-1])); nucs.append(float(s.sum()))
    best = int(np.argmin(val_err))
    lam_star = float(lams[best])
    Zsi, hist, rank_path, nit = soft_impute(Xobs, mask, lam_star)
    return dict(lams=lams, alphas=alphas, val_err=np.array(val_err), ranks=np.array(ranks),
                nucs=np.array(nucs), best=best, lam_star=lam_star, Zsi=Zsi,
                hist=hist, rank_path=rank_path, nit=nit, mask_val=mask_val, mask_tr=mask_tr)


pipe_m = run_pipeline(Xfull, mask_m, 777)
pipe_b = run_pipeline(Xfull, mask_b, 778)
RES["lambda_star"] = {"mcar": round(pipe_m["lam_star"], 4), "block": round(pipe_b["lam_star"], 4)}
RES["lambda_best_alpha"] = {"mcar": round(float(pipe_m["alphas"][pipe_m["best"]]), 4),
                            "block": round(float(pipe_b["alphas"][pipe_b["best"]]), 4)}
RES["val_err_at_lambda_star"] = {
    "mcar": round(float(pipe_m["val_err"][pipe_m["best"]]), 5),
    "block": round(float(pipe_b["val_err"][pipe_b["best"]]), 5)}
RES["val_err_at_min_lambda"] = {"mcar": round(float(pipe_m["val_err"][0]), 5),
                                "block": round(float(pipe_b["val_err"][0]), 5)}
RES["val_err_at_max_lambda"] = {"mcar": round(float(pipe_m["val_err"][-1]), 5),
                                "block": round(float(pipe_b["val_err"][-1]), 5)}
RES["eff_rank_at_star"] = {"mcar": int(pipe_m["ranks"][pipe_m["best"]]),
                           "block": int(pipe_b["ranks"][pipe_b["best"]])}
RES["soft_impute_iters"] = {"mcar": int(pipe_m["nit"]), "block": int(pipe_b["nit"])}
RES["noise_floor"] = SIG_E

# 图 1：面板三连（真低秩 / 观测缺失 / soft-impute 补全）
fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.6))
sub = (slice(0, 40), slice(0, 60))
vmax = np.abs(Xfull[sub]).max()
im0 = ax[0].imshow(Xfull[sub], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax[0].set_title("真低秩结构 L（前 40 月 × 60 只）")
disp = np.where(mask_m, Xfull, np.nan)[sub]
im1 = ax[1].imshow(disp, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax[1].set_title(f"观测面板（灰色=缺失 {100*(1-mask_m.mean()):.0f}%）")
ax[1].imshow(~mask_m[sub], cmap="gray", alpha=0.55, aspect="auto", vmin=0, vmax=1)
im2 = ax[2].imshow(pipe_m["Zsi"][sub], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax[2].set_title("soft-impute 补全结果")
for a in ax:
    a.set_xlabel("股票"); a.set_ylabel("月份")
plt.colorbar(im2, ax=ax, fraction=0.02, pad=0.02)
plt.savefig(f"{OUT}/fig1_panel_missing.png", bbox_inches="tight"); plt.close()

# 图 2：lambda 路径
fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.3))
ax[0].plot(pipe_m["alphas"], pipe_m["val_err"], "o-", color="#1976d2", ms=3.5, label="MCAR 30%")
ax[0].plot(pipe_b["alphas"], pipe_b["val_err"], "s-", color="#c62828", ms=3.5, label="块状 30%")
ax[0].axvline(pipe_m["alphas"][pipe_m["best"]], color="#1976d2", ls="--", lw=1.0)
ax[0].axvline(pipe_b["alphas"][pipe_b["best"]], color="#c62828", ls="--", lw=1.0)
ax[0].axhline(SIG_E, color="grey", ls=":", lw=1.2, label=f"噪声地板 σ_ε={SIG_E}")
ax[0].set_xscale("log"); ax[0].set_xlabel("λ / σ₁（零填矩阵的 top 奇异值）")
ax[0].set_ylabel("验证集 RMSE"); ax[0].set_title("λ 的 U 型：不罚会过拟合，罚过头会欠拟合")
ax[0].legend(fontsize=8)
ax[1].plot(pipe_m["alphas"], pipe_m["ranks"], "o-", color="#1976d2", ms=3.5, label="有效秩（MCAR）")
ax[1].plot(pipe_m["alphas"], pipe_m["nucs"] / pipe_m["nucs"][0], "s-", color="#6a1b9a", ms=3.5,
           label="核范数（归一化）")
ax[1].axhline(R, color="black", ls="--", lw=1.0, label=f"真秩 r={R}")
ax[1].set_xscale("log"); ax[1].set_xlabel("λ / σ₁"); ax[1].set_title("λ 决定有效秩")
ax[1].legend(fontsize=8)
h = pipe_m["hist"]
ax[2].plot(np.arange(1, len(h) + 1), h, "-", color="#2e7d32", lw=1.8)
ax[2].set_xlabel("迭代次数"); ax[2].set_ylabel("目标函数")
ax[2].set_title(f"soft-impute 收敛（{pipe_m['nit']} 次迭代）")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_softimpute_path.png", bbox_inches="tight"); plt.close()

RES["objective_first"] = round(float(h[0]), 4)
RES["objective_last"] = round(float(h[-1]), 4)
RES["objective_monotone"] = bool(np.all(np.diff(h) <= 1e-8))


# ---------------------------------------------------------------- E3/E4 方法对拍
def evaluate(Xfull, mask, seed, nlam=12):
    rgl = np.random.default_rng(seed)
    pipe = run_pipeline(Xfull, mask, seed, nlam=nlam)
    Xobs = np.where(mask, Xfull, 0.0)
    miss = ~mask
    out = {}
    for name, Z in [("零填充", np.where(mask, Xfull, 0.0)),
                    ("列均值", fill_colmean(Xobs, mask)),
                    ("前向填充", fill_ffill(Xobs, mask)),
                    ("零填+截断SVD", fill_zero_svd(Xobs, mask, R)),
                    ("soft-impute", pipe["Zsi"]),
                    ("oracle L", np.where(mask, Xfull, Ltrue))]:
        out[name] = float(np.sqrt(np.mean((Z[miss] - Xfull[miss]) ** 2)))
    # 下游：协方差
    ridge = 0.05 * np.mean(np.diag(cov_from(Xfull)))
    base_full = cov_from(Xfull)
    pw = cov_pairwise(Xobs, mask)
    min_eig = float(np.linalg.eigvalsh(pw).min())
    errs = {"完整数据参照": base_full,
            "逐对删失": pw,
            "列均值": cov_from(fill_colmean(Xobs, mask)),
            "前向填充": cov_from(fill_ffill(Xobs, mask)),
            "零填+截断SVD": cov_from(fill_zero_svd(Xobs, mask, R)),
            "soft-impute": cov_from(pipe["Zsi"])}
    dn = np.linalg.norm(SIG_TRUE)
    cov_err = {k: float(np.linalg.norm(v - SIG_TRUE) / dn) for k, v in errs.items()}
    # GMV 样本外
    oos_vol = {}
    for k, S in errs.items():
        w = gmv_weights(S, ridge)
        oos_vol[k] = float(np.std(Xoos @ w, ddof=1) * np.sqrt(12))
    w_or = gmv_weights(SIG_TRUE, ridge)
    oos_vol["oracle Σ"] = float(np.std(Xoos @ w_or, ddof=1) * np.sqrt(12))
    sub_cos = {}
    for k in ["逐对删失", "列均值", "soft-impute", "完整数据参照"]:
        mn, mean = subspace_cos(errs[k], Btrue)
        sub_cos[k] = {"min_cos": round(mn, 4), "mean_cos": round(mean, 4)}
    return out, cov_err, oos_vol, sub_cos, pipe, min_eig


SEEDS = [101, 202, 303, 404, 505, 606]
imp_m, cov_m, vol_m, imp_b, cov_b, vol_b = [], [], [], [], [], []
sub_agg, min_eigs = {}, []
for sd in SEEDS:
    rg_s = np.random.default_rng(sd)
    mm = mask_mcar(T, N, 0.30, rg_s)
    mb = mask_block(T, N, 0.30, rg_s)
    o1, c1, v1, s1_, _, me1 = evaluate(Xfull, mm, sd)
    o2, c2, v2, s2_, _, me2 = evaluate(Xfull, mb, sd)
    imp_m.append(o1); cov_m.append(c1); vol_m.append(v1)
    imp_b.append(o2); cov_b.append(c2); vol_b.append(v2)
    min_eigs += [me1, me2]
    for k in s1_:
        if k not in sub_agg:
            sub_agg[k] = {"min_cos": s1_[k]["min_cos"], "mean_cos": 0.0}
        sub_agg[k]["mean_cos"] += s1_[k]["mean_cos"] / len(SEEDS)
for k in sub_agg:
    sub_agg[k]["mean_cos"] = round(sub_agg[k]["mean_cos"], 4)
RES["pairwise_min_eig"] = {"min": round(float(np.min(min_eigs)), 5),
                           "mean": round(float(np.mean(min_eigs)), 5)}

def agg(dicts):
    keys = dicts[0].keys()
    return {k: {"mean": float(np.mean([d[k] for d in dicts])),
                "sd": float(np.std([d[k] for d in dicts], ddof=1))} for k in keys}

RES["impute_rmse_mcar"] = agg(imp_m)
RES["impute_rmse_block"] = agg(imp_b)
RES["cov_err_mcar"] = agg(cov_m)
RES["cov_err_block"] = agg(cov_b)
RES["oos_vol_mcar"] = agg(vol_m)
RES["oos_vol_block"] = agg(vol_b)
RES["subspace_cos"] = sub_agg
RES["ridge"] = 0.05

meth = ["逐对删失", "列均值", "前向填充", "零填+截断SVD", "soft-impute", "完整数据参照", "oracle Σ"]

fig, ax = plt.subplots(1, 2, figsize=(13.0, 4.8))
KEYS_IMP = ["零填充", "列均值", "前向填充", "零填+截断SVD", "soft-impute", "oracle L"]
xs = np.arange(len(KEYS_IMP)); labels = ["零填充", "列均值", "前向填充", "零填+截断\nSVD", "soft-\nimpute", "oracle\nL"]
for j, (data, col, lab) in enumerate([(imp_m, "#1976d2", "MCAR 30%"), (imp_b, "#c62828", "块状 30%")]):
    m = [np.mean([d[k] for d in data]) for k in KEYS_IMP]
    e = [np.std([d[k] for d in data], ddof=1) for k in KEYS_IMP]
    ax[0].bar(xs + (j - 0.5) * 0.38, m, 0.38, yerr=e, capsize=3, color=col, alpha=0.85, label=lab)
ax[0].axhline(SIG_E, color="grey", ls="--", lw=1.2)
ax[0].text(4.6, SIG_E * 1.03, f"噪声地板 σ_ε={SIG_E}", fontsize=8, color="grey")
ax[0].set_xticks(xs); ax[0].set_xticklabels(labels, fontsize=8)
ax[0].set_ylabel("填补 RMSE（相对真信号 L，缺失格）")
ax[0].set_title("填补精度：均值的下界是噪声，不是零"); ax[0].legend(fontsize=9)

keys_cov = ["逐对删失", "列均值", "前向填充", "零填+截断SVD", "soft-impute", "完整数据参照"]
xs2 = np.arange(len(keys_cov))
for j, (data, col, lab) in enumerate([(cov_m, "#1976d2", "MCAR 30%"), (cov_b, "#c62828", "块状 30%")]):
    m = [np.mean([d[k] for d in data]) for k in keys_cov]
    e = [np.std([d[k] for d in data], ddof=1) for k in keys_cov]
    ax[1].bar(xs2 + (j - 0.5) * 0.38, m, 0.38, yerr=e, capsize=3, color=col, alpha=0.85, label=lab)
ax[1].set_xticks(xs2); ax[1].set_xticklabels(["逐对\n删失", "列均值", "前向\n填充", "零填\nSVD", "soft-\nimpute", "完整\n数据"], fontsize=8)
ax[1].set_ylabel("‖Σ̂−Σ‖_F / ‖Σ‖_F"); ax[1].set_title("下游协方差误差")
ax[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_impute_rmse.png", bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13.0, 4.8))
keys_vol = ["逐对删失", "列均值", "前向填充", "零填+截断SVD", "soft-impute", "完整数据参照", "oracle Σ"]
xs3 = np.arange(len(keys_vol))
for j, (data, col, lab) in enumerate([(vol_m, "#1976d2", "MCAR 30%"), (vol_b, "#c62828", "块状 30%")]):
    m = [np.mean([d[k] for d in data]) for k in keys_vol]
    e = [np.std([d[k] for d in data], ddof=1) for k in keys_vol]
    ax[0].bar(xs3 + (j - 0.5) * 0.38, m, 0.38, yerr=e, capsize=3, color=col, alpha=0.85, label=lab)
ax[0].set_xticks(xs3)
ax[0].set_xticklabels(["逐对\n删失", "列均值", "前向\n填充", "零填\nSVD", "soft-\nimpute", "完整\n数据", "oracle\nΣ"], fontsize=8)
ax[0].set_ylabel("样本外年化波动率"); ax[0].set_title("全局最小方差组合：60 个月样本外实现波动")
ax[0].legend(fontsize=9)
sc = [sub_agg[k]["mean_cos"] for k in ["逐对删失", "列均值", "soft-impute", "完整数据参照"]]
ax[1].bar(np.arange(4), sc, color=["#90a4ae", "#ffb74d", "#2e7d32", "#455a64"], alpha=0.9)
ax[1].set_xticks(np.arange(4)); ax[1].set_xticklabels(["逐对\n删失", "列均值", "soft-\nimpute", "完整\n数据"], fontsize=8)
ax[1].set_ylabel("主角度 cos 均值（越大越准）"); ax[1].set_ylim(0, 1)
ax[1].set_title("因子载荷子空间恢复（top-4）")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_covariance_gmv.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- 冷水：整列缺失（新股 / 长期停牌）
NC = 20
rgc = np.random.default_rng(31337)
cold_cols = rgc.choice(N, NC, replace=False)
mask_col = mask_mcar(T, N, 0.30, rgc)
mask_col[:, cold_cols] = False
pc = run_pipeline(Xfull, mask_col, 4243, nlam=12)
Zcold = pc["Zsi"]
cm = np.zeros((T, N), dtype=bool); cm[:, cold_cols] = True
RES["cold_start"] = {
    "n_cols": NC,
    "soft": round(float(np.sqrt(np.mean((Zcold[cm] - Xfull[cm]) ** 2))), 5),
    "zero": round(float(np.sqrt(np.mean(Xfull[cm] ** 2))), 5),
    "rms_signal": round(float(np.sqrt(np.mean(Ltrue ** 2))), 5),
    "identical_to_zero": bool(np.max(np.abs(Zcold[:, cold_cols])) < 1e-10),
}

# ---------------------------------------------------------------- E5 扫描
rates = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
scan = {}
for m in rates:
    rgs = np.random.default_rng(int(m * 1000))
    mm = mask_mcar(T, N, m, rgs)
    mb = mask_block(T, N, m, rgs)
    Xo = np.where(mm, Xfull, 0.0)
    row = {}
    pm = run_pipeline(Xfull, mm, 4242, nlam=10)
    pb = run_pipeline(Xfull, mb, 4242, nlam=10)
    for lab, msk, pp in [("mcar", mm, pm), ("block", mb, pb)]:
        row[lab] = {
            "soft": round(float(np.sqrt(np.mean((pp["Zsi"][~msk] - Xfull[~msk]) ** 2))), 5),
            "colmean": round(float(np.sqrt(np.mean((fill_colmean(Xo if lab == "mcar" else np.where(msk, Xfull, 0.0), msk)[~msk] - Xfull[~msk]) ** 2))), 5),
            "rank_star": int(pp["ranks"][pp["best"]]),
        }
    scan[m] = row
RES["missing_rate_scan"] = scan

rank_scan = {}
for rh in [1, 2, 3, 4, 5, 6, 8, 10, 15]:
    Z = fill_zero_svd(np.where(mask_m, Xfull, 0.0), mask_m, rh)
    rank_scan[rh] = round(float(np.sqrt(np.mean((Z[~mask_m] - Xfull[~mask_m]) ** 2))), 5)
RES["rank_scan_zero_svd"] = rank_scan

fig, ax = plt.subplots(1, 2, figsize=(13.0, 4.6))
rr = [str(int(r * 100)) for r in rates]
ax[0].plot(rr, [scan[m]["mcar"]["soft"] for m in rates], "o-", color="#1976d2", label="soft-impute（MCAR）")
ax[0].plot(rr, [scan[m]["block"]["soft"] for m in rates], "o--", color="#c62828", label="soft-impute（块状）")
ax[0].plot(rr, [scan[m]["mcar"]["colmean"] for m in rates], "s-", color="#90a4ae", label="列均值（MCAR）")
ax[0].plot(rr, [scan[m]["block"]["colmean"] for m in rates], "s--", color="#ffb74d", label="列均值（块状）")
ax[0].axhline(SIG_E, color="grey", ls=":", lw=1.2, label="噪声地板")
ax[0].set_xlabel("缺失率 %"); ax[0].set_ylabel("填补 RMSE")
ax[0].set_title("缺失率扫描：补全的红利在中等缺失率最大"); ax[0].legend(fontsize=8)
rk = list(rank_scan.keys()); rv = list(rank_scan.values())
ax[1].plot([str(x) for x in rk], rv, "o-", color="#6a1b9a")
ax[1].axvline(3, color="black", ls="--", lw=1.0, label=f"真秩 r={R}")
ax[1].set_xlabel("截断 SVD 的设定秩 r̂"); ax[1].set_ylabel("填补 RMSE")
ax[1].set_title("秩误设：欠设比过设更致命"); ax[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_rate_rank_scan.png", bbox_inches="tight"); plt.close()

with open("/Users/halo/workspace/astro-blog/_mc_results.json", "w") as f:
    json.dump(RES, f, ensure_ascii=False, indent=2)
print(json.dumps(RES, ensure_ascii=False, indent=2))
