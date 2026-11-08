"""贝叶斯在线变点检测（BOCD）完整可运行实验。

运行方式（在项目根目录）：
    python3 scripts/article_core_bocd_20261108.py

产出：
    public/images/bayesian-online-changepoint-run-length/fig1..fig5.png
    public/images/bayesian-online-changepoint-run-length/results.json

模型：Adams & MacKay (2007) 的 run-length 递推，未知均值与方差用
Normal-Inverse-Gamma 共轭先验，一步预测是 Student-t。hazard 取常数 1/lambda。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

SEED = 20261108
OUT = "public/images/bayesian-online-changepoint-run-length"
T = 480
TRUE_CPS = (120, 260, 380)
SIGMAS = (0.006, 0.022, 0.007, 0.020)
TOL = 20  # 检测容差窗口（天）
SHORT_R = 10  # 「刚换段」的 run-length 上界，用于构造变点证据分数
THRESHOLD = 0.50  # 固定判决阈值：短段后验质量超过它才报警


def gen_returns(seed=SEED, cps=TRUE_CPS, sigmas=SIGMAS, n=T):
    """分段常数波动率的高斯收益率，变点位置与真实波动率都已知。"""
    rng = np.random.default_rng(seed)
    vol = np.zeros(n)
    bounds = (0,) + tuple(cps) + (n,)
    for k, s in enumerate(sigmas):
        vol[bounds[k]:bounds[k + 1]] = s
    return rng.standard_normal(n) * vol, vol


def bocd(x, lam=80.0, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1e-4):
    """返回 (run-length 后验矩阵, 短段后验质量分数, MAP run-length 序列).

    注意：常数 hazard 下 P(r_t=0 | x_1:t) 恒等于 hazard，与数据无关，
    不能拿来当检测分数。真正携带信息的是质量向短 run-length 的转移。

    递推的四个量都是按 run-length r 索引的 NIG 后验参数，
    表示「在 t-1 时刻、当前段已经持续 r 步」的条件下对下一观测的预测分布。
    """
    n = len(x)
    hazard = 1.0 / lam
    m = n + 1
    mu = np.full(m, mu0, dtype=float)
    kappa = np.full(m, kappa0, dtype=float)
    alpha = np.full(m, alpha0, dtype=float)
    beta = np.full(m, beta0, dtype=float)

    joint = np.zeros(m)          # P(r_{t-1}=r, x_{1:t-1})
    joint[0] = 1.0
    post = np.zeros((n, m))      # P(r_t=r | x_{1:t})
    map_r = np.zeros(n)

    for t in range(n):
        k = t + 1
        df = 2.0 * alpha[:k]
        scale2 = beta[:k] * (kappa[:k] + 1.0) / (alpha[:k] * kappa[:k])
        log_pred = student_t.logpdf(x[t], df=df, loc=mu[:k],
                                    scale=np.sqrt(scale2))
        w = joint[:k] * np.exp(log_pred)

        new_joint = np.zeros(m)
        new_joint[0] = np.sum(w * hazard)          # 发生变点
        new_joint[1:k + 1] = w * (1.0 - hazard)   # 段长度 +1
        new_joint /= (new_joint.sum() + 1e-300)

        # 用 x_t 更新每个 r 的 NIG 后验；r=0 保持先验
        kn = kappa[:k] + 1.0
        mn = (kappa[:k] * mu[:k] + x[t]) / kn
        an = alpha[:k] + 0.5
        bn = beta[:k] + kappa[:k] * (x[t] - mu[:k]) ** 2 / (2.0 * kn)

        new_mu = np.zeros(m)
        new_kappa = np.empty(m)
        new_alpha = np.empty(m)
        new_beta = np.empty(m)
        new_mu[0] = mu0
        new_kappa[0] = kappa0
        new_alpha[0] = alpha0
        new_beta[0] = beta0
        new_mu[1:k + 1] = mn
        new_kappa[1:k + 1] = kn
        new_alpha[1:k + 1] = an
        new_beta[1:k + 1] = bn

        mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta
        joint = new_joint
        post[t] = new_joint
        map_r[t] = int(np.argmax(new_joint[:k + 1]))

    # 短 run-length 的后验质量：变点后质量从长段回落到短段，此分数跃升
    short_mass = post[:, 1:SHORT_R + 1].sum(axis=1).copy()
    return post, short_mass, map_r


def cusum_alarms(x, h, delta=1.0, calib=60):
    """对 |r| 做双侧 CUSUM；超过 h 报警并把该侧统计量清零。

    基准均值/标准差只用前 calib 天估计，避免用到未来数据（无前视偏差）。
    """
    a = np.abs(x)
    z = (a - a[:calib].mean()) / (a[:calib].std() + 1e-12)
    sp, sm, alarms = 0.0, 0.0, []
    for i in range(1, len(x)):
        sp = max(0.0, sp + z[i] - delta)
        sm = max(0.0, sm - z[i] - delta)
        if sp > h:
            alarms.append(i)
            sp = 0.0
        if sm > h:
            alarms.append(i)
            sm = 0.0
    return alarms


def pick_alarms(score, thr, min_sep=15, warmup=15):
    """阈值化 + 非极大值抑制；跳过预热期（初始后验天然集中在短段）。"""
    cand = [i for i in range(len(score)) if score[i] > thr and i >= warmup]
    picked = []
    for i in sorted(cand, key=lambda j: -score[j]):
        if all(abs(i - p) > min_sep for p in picked):
            picked.append(i)
    return sorted(picked)


def match(alarms, cps, tol=TOL):
    tp, fp, delays = 0, 0, []
    used = set()
    for a in alarms:
        d = [(abs(a - c), c) for c in cps if c not in used]
        if d:
            dist, c = min(d)
            if dist <= tol:
                tp += 1
                delays.append(a - c)
                used.add(c)
                continue
        fp += 1
    return tp, fp, (float(np.mean(delays)) if delays else float("nan"))


def window_labels_and_preds(alarms, cps, n=T, tol=TOL):
    """把时间切成不重叠窗口，得到标准 ROC 的真值/预测标签。"""
    nw = n // tol
    y, s = [], []
    for w in range(nw):
        lo, hi = w * tol, (w + 1) * tol
        y.append(1 if any(lo <= c < hi for c in cps) else 0)
        s.append(1 if any(lo <= a < hi for a in alarms) else 0)
    return np.array(y), np.array(s)


def auc_from_curve(fpr, tpr):
    order = np.argsort(fpr)
    integ = getattr(np, "trapezoid", None)
    if integ is None:  # numpy < 2.0
        integ = np.trapz
    return float(integ(tpr[order], fpr[order]))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=110, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(OUT, exist_ok=True)
    x, vol = gen_returns()
    post, cp_score, map_r = bocd(x, lam=80.0)

    # 图 1：收益率与真实波动率区制
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1]})
    ax[0].plot(x, lw=0.7, color="#2f5597")
    ax[1].plot(np.abs(x), lw=0.6, color="#9aa5b1", label="|return|")
    ax[1].plot(vol, lw=2.0, color="#c0392b", label="true sigma")
    for c in TRUE_CPS:
        for a in ax:
            a.axvline(c, ls="--", lw=1.0, color="#333333")
    ax[0].set_ylabel("daily return")
    ax[1].set_ylabel("volatility")
    ax[1].set_xlabel("trading day")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[0].set_title("Synthetic returns with 3 volatility changepoints")
    fig.tight_layout()
    save(fig, "fig1_returns_regimes.png")

    # 图 2：run-length 后验热力图
    show = np.log(np.clip(post[:, :200].T, 1e-6, 1.0))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    im = ax.imshow(show, origin="lower", aspect="auto", cmap="magma",
                   extent=[0, T, 0, 200], vmin=-8, vmax=0)
    for c in TRUE_CPS:
        ax.axvline(c, ls="--", lw=1.0, color="#5ce1e6")
    ax.plot(np.arange(T), map_r, color="#7dff9b", lw=0.9, label="MAP run-length")
    ax.set_ylim(0, 200)
    ax.set_xlabel("trading day")
    ax.set_ylabel("run length r")
    ax.set_title("BOCD posterior P(r_t | x_1:t), log scale")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="log posterior")
    fig.tight_layout()
    save(fig, "fig2_runlength_posterior.png")

    # 图 3：变点后验概率与 MAP 回落
    alarms = pick_alarms(cp_score, THRESHOLD)
    tp, fp, delay = match(alarms, TRUE_CPS)
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    ax[0].plot(cp_score, color="#c0392b", lw=1.0)
    ax[0].axhline(THRESHOLD, ls=":", color="#555555",
                  label=f"threshold {THRESHOLD:.2f}")
    ax[0].set_ylabel("posterior mass, r<=10")
    ax[0].legend(fontsize=8)
    ax[1].plot(map_r, color="#2f5597", lw=0.9)
    ax[1].set_ylabel("MAP r_t")
    ax[1].set_xlabel("trading day")
    for a in ax:
        for c in TRUE_CPS:
            a.axvline(c, ls="--", lw=1.0, color="#333333")
    for p in alarms:
        ax[0].axvline(p, ls="-", lw=0.7, color="#e67e22", alpha=0.6)
    ax[0].set_title("Short-run-length posterior mass and MAP resets")
    fig.tight_layout()
    save(fig, "fig3_map_detection.png")

    # 图 4 + 图 5：hazard 灵敏度与 ROC，在多条独立序列上评估
    seqs = [(gen_returns(seed=SEED + 100 + s)[0], TRUE_CPS) for s in range(12)]
    seqs.insert(0, (x, TRUE_CPS))
    lams = [20.0, 40.0, 80.0, 160.0, 320.0]
    scan = []
    for lam in lams:
        delays, fps, tps = [], [], []
        for xs, cps in seqs:
            _, cs, _ = bocd(xs, lam=lam)
            a = pick_alarms(cs, THRESHOLD)
            t, f, d = match(a, cps)
            tps.append(t)
            fps.append(f)
            if np.isfinite(d):
                delays.append(d)
        scan.append({
            "lambda": lam,
            "mean_delay": float(np.mean(delays)) if delays else None,
            "false_alarms_per_series": float(np.mean(fps)),
            "true_positives_per_series": float(np.mean(tps)),
        })

    fig, ax = plt.subplots(figsize=(9, 4.4))
    xx = np.arange(len(lams))
    ax.plot(xx, [s["mean_delay"] for s in scan], "o-", color="#2f5597",
            label="mean detection delay (days)")
    ax.set_ylabel("detection delay (days)", color="#2f5597")
    ax2 = ax.twinx()
    ax2.plot(xx, [s["false_alarms_per_series"] for s in scan], "s--",
             color="#c0392b", label="false alarms / series")
    ax2.set_ylabel("false alarms per series", color="#c0392b")
    ax.set_xticks(xx)
    ax.set_xticklabels([str(int(v)) for v in lams])
    ax.set_xlabel("expected run length lambda")
    ax.set_title("Hazard sensitivity: delay vs false alarms")
    fig.tight_layout()
    save(fig, "fig4_hazard_sensitivity.png")

    # ROC：BOCD 的短段后验质量 vs 报警后重置的双侧 CUSUM
    # 两边都用「不重叠 20 天窗口」计标签，阈值各自扫描，评价口径一致。
    scores_b = [bocd(xs, 80.0)[1] for xs, _ in seqs]
    thrs_b = np.unique(np.quantile(np.concatenate(scores_b),
                                   np.linspace(0.50, 0.9995, 80)))
    thrs_c = np.linspace(0.2, 8.0, 100)

    def roc_curve(thrs, alarms_fn):
        pts = []
        for thr in thrs:
            Y, S = [], []
            for i, (_, cps) in enumerate(seqs):
                y, s = window_labels_and_preds(alarms_fn(i, thr), cps)
                Y.append(y)
                S.append(s)
            Y, S = np.concatenate(Y), np.concatenate(S)
            tpr = S[Y == 1].mean() if (Y == 1).any() else 0.0
            fpr = S[Y == 0].mean() if (Y == 0).any() else 0.0
            pts.append((float(fpr), float(tpr)))
        pts = sorted(set(pts))
        return (np.array([p[0] for p in pts]),
                np.array([p[1] for p in pts]))

    f_b, t_b = roc_curve(thrs_b, lambda i, thr: pick_alarms(scores_b[i], thr))
    f_c, t_c = roc_curve(thrs_c, lambda i, h: cusum_alarms(seqs[i][0], h))
    auc_b = auc_from_curve(np.r_[0.0, f_b, 1.0], np.r_[0.0, t_b, 1.0])
    auc_c = auc_from_curve(np.r_[0.0, f_c, 1.0], np.r_[0.0, t_c, 1.0])

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(np.r_[0.0, f_b, 1.0], np.r_[0.0, t_b, 1.0], "o-", ms=3,
            color="#c0392b", label=f"BOCD (AUC={auc_b:.3f})")
    ax.plot(np.r_[0.0, f_c, 1.0], np.r_[0.0, t_c, 1.0], "s-", ms=3,
            color="#2f5597", label=f"CUSUM (AUC={auc_c:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", color="#888888")
    ax.set_xlabel("false positive rate (per 20-day window)")
    ax.set_ylabel("true positive rate")
    ax.set_title("Detection ROC over 13 independent series")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    save(fig, "fig5_roc_bocd_cusum.png")

    results = {
        "seed": SEED,
        "n_days": T,
        "true_changepoints": list(TRUE_CPS),
        "true_sigmas": list(SIGMAS),
        "lambda_main": 80.0,
        "threshold_main": THRESHOLD,
        "short_runlength_cutoff": SHORT_R,
        "detected_changepoints": alarms,
        "true_positives": tp,
        "false_alarms": fp,
        "mean_signed_delay": delay,
        "hazard_scan": scan,
        "roc_auc_bocd": auc_b,
        "roc_auc_cusum": auc_c,
        "n_series": len(seqs),
        "tol_days": TOL,
    }
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("true changepoints :", list(TRUE_CPS))
    print("detected          :", alarms)
    print(f"threshold={THRESHOLD:.2f}  TP={tp}  FP={fp}  "
          f"mean_signed_delay={delay:.2f} days")
    for s in scan:
        d = ("nan" if s["mean_delay"] is None
             else f"{s['mean_delay']:.2f}")
        print(f"lambda={s['lambda']:>5.0f}  delay={d}  "
              f"FA/series={s['false_alarms_per_series']:.3f}  "
              f"TP/series={s['true_positives_per_series']:.3f}")
    print(f"ROC AUC: BOCD={auc_b:.4f}  CUSUM={auc_c:.4f}")
    return results


if __name__ == "__main__":
    run()
