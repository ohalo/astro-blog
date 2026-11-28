# -*- coding: utf-8 -*-
"""RQA on a synthetic market with an autocorrelation shift before volatility jumps."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 20261128
N = 1500
W = 100
STEP = 5
M = 3
TAU = 1
THEILER = 5
LMIN = 2
Q = 0.05
IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "public", "images", "recurrence-quantification-regime")

PHASES = [
    ("calm", 0, 450, 0.05, 0.0060),
    ("pre-stress", 450, 570, 0.55, 0.0062),
    ("crisis", 570, 750, 0.72, 0.0200),
    ("recovery", 750, 1000, 0.30, 0.0100),
    ("calm", 1000, N, 0.05, 0.0060),
]


def generate_returns():
    rng = np.random.default_rng(SEED)
    shocks = rng.standard_normal(N)
    r = np.zeros(N)
    for _, a, b, phi, sig in PHASES:
        for t in range(a, b):
            prev = r[t - 1] if t > a else 0.0
            r[t] = phi * prev + sig * shocks[t]
    return r


def embed(x, m, tau):
    n = len(x) - (m - 1) * tau
    return np.column_stack([x[i * tau:n + i * tau] for i in range(m)])


def runs_1d(v):
    edges = np.diff(np.r_[False, v, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return ends - starts


def rqa_metrics(vec, q=Q, m=M, theiler=THEILER, lmin=LMIN):
    y = embed(vec, m, TAU)
    w = len(y)
    d = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=2)
    valid = np.ones((w, w), dtype=bool)
    for k in range(theiler + 1):
        valid &= np.abs(np.eye(w, k=k, dtype=bool)) == False
        valid &= np.abs(np.eye(w, k=-k, dtype=bool)) == False
    iu = np.triu_indices(w, 1)
    pool = d[iu][valid[iu]]
    eps = np.quantile(pool, q)
    rmat = (d <= eps) & valid

    n_rec = int(rmat.sum())
    diag = []
    for k in range(1, w):
        diag.extend(runs_1d(np.diag(rmat, k)))
    diag = np.asarray(diag, dtype=int)
    vert = []
    for j in range(w):
        vert.extend(runs_1d(rmat[:, j]))
    vert = np.asarray(vert, dtype=int)
    long_d = diag[diag >= lmin]
    long_v = vert[vert >= lmin]
    return {
        "eps": float(eps),
        "rr": n_rec / len(pool),
        "det": float(long_d.sum() / n_rec) if n_rec else 0.0,
        "lmean": float(long_d.mean()) if len(long_d) else 0.0,
        "lam": float(long_v.sum() / n_rec) if n_rec else 0.0,
        "tt": float(long_v.mean()) if len(long_v) else 0.0,
        "matrix": rmat,
    }


def rolling_vol(r, win=40):
    out = np.full(len(r), np.nan)
    for i in range(win, len(r)):
        out[i] = r[i - win:i].std(ddof=1)
    return out


def first_alert(centers, values, calm_end=450, k=3):
    base = values[(centers < calm_end - W) & np.isfinite(values)]
    thr = base.mean() + 3.0 * base.std(ddof=1)
    for i in range(len(values)):
        if not np.isfinite(values[i]) or values[i] <= thr:
            continue
        if np.all(values[i:i + k] > thr):
            false_count = int(((centers >= calm_end - W) & (centers < calm_end) & (values > thr)).sum())
            return float(centers[i]), float(thr), false_count
    return float("nan"), float(thr), 0


def main():
    os.makedirs(IMG, exist_ok=True)
    r = generate_returns()
    rv = rolling_vol(r)
    centers = []
    dets, rrs, lmeans, lams = [], [], [], []
    for start in range(0, len(r) - W + 1, STEP):
        met = rqa_metrics(r[start:start + W])
        centers.append(start + W // 2)
        dets.append(met["det"])
        rrs.append(met["rr"])
        lmeans.append(met["lmean"])
        lams.append(met["lam"])
    centers = np.asarray(centers, dtype=float)
    dets = np.asarray(dets)
    rrs = np.asarray(rrs)
    lmeans = np.asarray(lmeans)
    lams = np.asarray(lams)
    vol_win = np.asarray([rv[int(c)] if np.isfinite(rv[int(c)]) else np.nan for c in centers])

    det_day, det_thr, det_false = first_alert(centers, dets)
    lam_day, lam_thr, lam_false = first_alert(centers, lams)
    vol_day, vol_thr, vol_false = first_alert(centers, vol_win)

    seg = {}
    for name, a, b, _, _ in PHASES:
        sel = (centers >= a) & (centers < b)
        seg[name] = {
            "det": float(dets[sel].mean()),
            "rr": float(rrs[sel].mean()),
            "lmean": float(lmeans[sel].mean()),
            "lam": float(lams[sel].mean()),
            "n": int(sel.sum()),
        }

    q_grid = [0.02, 0.05, 0.10, 0.20]
    sens = {}
    for qv in q_grid:
        c = rqa_metrics(r[350:350 + W], q=qv)
        s = rqa_metrics(r[620:620 + W], q=qv)
        sens[str(qv)] = {"det_calm": c["det"], "det_crisis": s["det"],
                         "rr_calm": c["rr"], "rr_crisis": s["rr"]}

    emb = {}
    for m in (2, 3, 4, 5):
        c = rqa_metrics(r[350:350 + W], m=m)
        s = rqa_metrics(r[620:620 + W], m=m)
        emb[str(m)] = {"det_calm": c["det"], "det_crisis": s["det"]}

    calm_mat = rqa_metrics(r[350:350 + W])["matrix"]
    crisis_mat = rqa_metrics(r[620:620 + W])["matrix"]

    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(r, lw=0.7, color="#3b5b8c")
    for a, b in [(450, 570), (570, 750), (750, 1000)]:
        ax.axvspan(a, b, alpha=0.16, color="#c0392b" if b == 750 else "#e0a33e")
    ax.set_title("Synthetic returns: autocorrelated pre-stress before the volatility jump")
    ax.set_xlabel("day")
    ax.set_ylabel("return")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig1_returns.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, mat, title in zip(axes, [calm_mat, crisis_mat],
                              ["Calm recurrence", "Crisis recurrence"]):
        ax.imshow(mat, cmap="Greys", interpolation="none")
        ax.set_title(title)
        ax.set_xlabel("state index")
        ax.set_ylabel("state index")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig2_recurrence_matrices.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(centers, dets, lw=1.4, color="#2c6e7f", label="DET (RQA)")
    ax.plot(centers, lams, lw=1.0, color="#8e6cbb", label="LAM (RQA)")
    ax2 = ax.twinx()
    ax2.plot(centers, vol_win, lw=1.1, color="#c0392b", label="40d vol")
    ax.axhline(det_thr, ls="--", lw=0.9, color="#2c6e7f")
    ax2.axhline(vol_thr, ls="--", lw=0.9, color="#c0392b")
    for a, b in [(450, 570), (570, 750)]:
        ax.axvspan(a, b, alpha=0.12, color="#e0a33e")
    ax.set_title("Windowed RQA metrics versus rolling volatility")
    ax.set_xlabel("day")
    ax.set_ylabel("RQA metric")
    ax2.set_ylabel("rolling vol")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig3_window_metrics.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].plot(q_grid, [sens[str(q)]["det_calm"] for q in q_grid], "o-", color="#3b5b8c", label="calm")
    axes[0].plot(q_grid, [sens[str(q)]["det_crisis"] for q in q_grid], "o-", color="#c0392b", label="crisis")
    axes[0].set_title("DET across recurrence radius quantiles")
    axes[0].set_xlabel("eps quantile")
    axes[0].set_ylabel("DET")
    axes[0].legend()
    axes[1].plot(q_grid, [sens[str(q)]["rr_calm"] for q in q_grid], "o-", color="#3b5b8c", label="calm")
    axes[1].plot(q_grid, [sens[str(q)]["rr_crisis"] for q in q_grid], "o-", color="#c0392b", label="crisis")
    axes[1].set_title("RR across recurrence radius quantiles")
    axes[1].set_xlabel("eps quantile")
    axes[1].set_ylabel("RR")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig4_threshold_sensitivity.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ms = [2, 3, 4, 5]
    axes[0].plot(ms, [emb[str(m)]["det_calm"] for m in ms], "o-", color="#3b5b8c", label="calm")
    axes[0].plot(ms, [emb[str(m)]["det_crisis"] for m in ms], "o-", color="#c0392b", label="crisis")
    axes[0].set_title("Embedding dimension sensitivity")
    axes[0].set_xlabel("m")
    axes[0].set_ylabel("DET")
    axes[0].legend()
    axes[1].bar(["DET", "LAM", "40d vol"], [det_day - 450, lam_day - 450, vol_day - 450],
                color=["#2c6e7f", "#8e6cbb", "#c0392b"])
    axes[1].axhline(0, color="0.3", lw=0.8)
    axes[1].set_title("First alert relative to regime start")
    axes[1].set_ylabel("days after day 450")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig5_lead_and_embedding.png"), dpi=150)
    plt.close(fig)

    out = {
        "seed": SEED,
        "n_days": N,
        "window": W,
        "step": STEP,
        "embedding_dim": M,
        "theiler": THEILER,
        "lmin": LMIN,
        "eps_quantile": Q,
        "segments": seg,
        "det_alert_day": round(det_day, 3),
        "lam_alert_day": round(lam_day, 3),
        "vol_alert_day": round(vol_day, 3),
        "det_lead_days": round(vol_day - det_day, 3),
        "lam_lead_days": round(vol_day - lam_day, 3),
        "det_threshold": round(det_thr, 6),
        "lam_threshold": round(lam_thr, 6),
        "vol_threshold": round(vol_thr, 6),
        "pre_alert_false_alarms": det_false,
        "lam_false_alarms": lam_false,
        "vol_false_alarms": vol_false,
        "calm2_false_alarms": int(((centers >= 1000) & (dets > det_thr)).sum()),
        "calm2_lam_false_alarms": int(((centers >= 1000) & (lams > lam_thr)).sum()),
        "threshold_sensitivity": {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in sens.items()},
        "embedding_sensitivity": {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in emb.items()},
    }
    with open(os.path.join(IMG, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
