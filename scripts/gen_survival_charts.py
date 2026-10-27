"""
Generate 3 real charts for the Survival Analysis Default article.
Article: 生存分析在违约预测：用 Cox 比例风险把『何时违约』变成风险率
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/Users/halo/workspace/astro-blog/public/images/survival-analysis-default"
np.random.seed(2026)

# --------------------------------------------------------------------------
# Chart 1: km_survival_curves_by_rating.png
# --------------------------------------------------------------------------
def km_estimator(times, events):
    order = np.argsort(times)
    times_s = times[order]
    events_s = events[order]
    n = len(times_s)
    S = 1.0
    surv = [(0.0, 1.0)]
    at_risk = n
    for t, e in zip(times_s, events_s):
        if e == 1:
            S *= (1 - 1/at_risk)
        else:
            at_risk -= 1
            continue
        surv.append((t, S))
        at_risk -= 1
    return np.array(surv)

ratings = {
    "AAA":  0.001,
    "BBB":  0.025,
    "BB":   0.060,
    "CCC":  0.150,
}
fig, ax = plt.subplots(figsize=(11, 5.2))
colors = {"AAA": "#1F3B66", "BBB": "#4F8AC9", "BB": "#E4B660", "CCC": "#D9654C"}

for rating, h in ratings.items():
    n = 400
    T = np.random.exponential(1.0/h, n)
    C = np.random.uniform(2.0, 9.0, n)
    times = np.minimum(T, C)
    events = (T <= C).astype(int)
    km = km_estimator(times, events)
    ax.step(km[:,0], km[:,1], where="post", lw=2.4, label=rating, color=colors[rating])

ax.set_xlabel("Years from issuance")
ax.set_ylabel("Survival probability S(t)")
ax.set_title("Kaplan-Meier Survival Curves by Credit Rating",
             fontsize=13, fontweight="bold")
ax.set_ylim(-0.02, 1.02)
ax.axhline(0.5, color="gray", lw=0.6, ls="--")
ax.text(0.05, 0.51, "50% survival", fontsize=8, color="gray")
ax.legend(title="Rating", loc="lower left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/km_survival_curves_by_rating.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("OK 1/3: km_survival_curves_by_rating.png")

# --------------------------------------------------------------------------
# Chart 2: cox_hazard_ratio_factors.png
# --------------------------------------------------------------------------
n = 600
leverage    = np.random.normal(0.4, 0.15, n)
leverage2   = leverage**2
roa         = np.random.normal(0.08, 0.03, n)
intcov      = np.random.lognormal(1.0, 0.5, n)
logassets   = np.random.normal(8.0, 1.2, n)

beta_true = np.array([2.5, -3.0, -4.5, -0.6, -0.4])

Z = np.column_stack([leverage, leverage2, roa, intcov, logassets])
log_h = Z @ beta_true + np.random.normal(0, 0.1, n)
hazard = np.exp(log_h)
times = np.random.exponential(1.0/hazard, n)
C = np.random.uniform(0.5, 12.0, n)
T = np.minimum(times, C)
E = (times <= C).astype(int)

def cox_partial_nll(beta, T, E, Z):
    order = np.argsort(T)
    T_s, E_s, Z_s = T[order], E[order], Z[order]
    log_partial = 0.0
    grad = np.zeros_like(beta, dtype=float)
    sum_outer = 0.0
    sum_w = np.zeros_like(beta)
    for i in range(len(T_s)):
        if E_s[i] == 0:
            continue
        risk_mask = (T_s >= T_s[i])
        # The event at position i corresponds to Z_s[i], not Z[risk_mask][i]
        zr_all = Z_s[risk_mask]
        if zr_all.shape[0] == 0:
            continue
        etas = np.exp(zr_all @ beta)
        denom = etas.sum()
        log_partial -= Z_s[i] @ beta - np.log(denom)
        w_norm = etas / denom
        diff = Z_s[i] - (w_norm[:, None] * zr_all).sum(0)
        grad -= diff
        ww = (w_norm[:, None, None] * (zr_all[:, :, None] * zr_all[:, None, :])).sum(0)
        ew = (w_norm[:, None] * zr_all).sum(0)
        sum_outer += ww
        sum_w += ew
    return -log_partial, -grad, -(sum_outer - np.outer(sum_w, sum_w))

beta = np.zeros(Z.shape[1])
for it in range(40):
    _, grad, hess = cox_partial_nll(beta, T, E, Z)
    try:
        step = np.linalg.solve(hess, grad)
    except np.linalg.LinAlgError:
        step = np.linalg.lstsq(hess, grad, rcond=None)[0]
    beta_new = beta - 0.6 * step
    if np.linalg.norm(beta_new - beta) < 1e-6:
        beta = beta_new
        break
    beta = beta_new

betas = []
for b in range(80):
    idx = np.random.choice(n, n, replace=True)
    bb = np.zeros(Z.shape[1])
    Zi, Ti, Ei = Z[idx], T[idx], E[idx]
    try:
        for it in range(40):
            _, g, H = cox_partial_nll(bb, Ti, Ei, Zi)
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            new = bb - 0.5 * step
            if np.linalg.norm(new - bb) < 1e-5:
                bb = new
                break
            bb = new
        betas.append(bb)
    except Exception:
        pass
betas = np.array(betas)
ci_low, ci_high = np.percentile(betas, [2.5, 97.5], axis=0)

factor_names = ["Leverage", "Leverage²", "ROA", "Int. Coverage", "log(Assets)"]
hr = np.exp(beta)
hr_lo = np.exp(ci_low)
hr_hi = np.exp(ci_high)

fig, ax = plt.subplots(figsize=(10, 5.2))
y = np.arange(len(factor_names))
ax.errorbar(hr, y, xerr=[hr-hr_lo, hr_hi-hr], fmt="o", color="#1F3B66",
            ecolor="#4F8AC9", capsize=6, lw=2, markersize=9)
ax.axvline(1, color="gray", lw=1.0, ls="--")
ax.set_yticks(y)
ax.set_yticklabels(factor_names)
ax.set_xscale("log")
ax.set_xlabel("Hazard ratio (exp of coefficient)")
ax.set_title("Cox PH Hazard Ratios with Bootstrap 95% CIs",
             fontsize=13, fontweight="bold")
for i, (h, l, h_2) in enumerate(zip(hr, hr_lo, hr_hi)):
    ax.text(max(h, h_2)*1.02, i, f"{h:.2f} [{l:.2f}, {h_2:.2f}]",
            va="center", fontsize=9)
ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/cox_hazard_ratio_factors.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("OK 2/3: cox_hazard_ratio_factors.png")

# --------------------------------------------------------------------------
# Chart 3: expected_loss_curves.png
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))

def gen_loan_data(seed, hr_factor):
    rng = np.random.default_rng(seed)
    n = 300
    lev = rng.normal(0.4, 0.15, n)
    roa = rng.normal(0.08, 0.03, n)
    intcov = rng.lognormal(1.0, 0.5, n)
    logassets = rng.normal(8.0, 1.2, n)
    Z = np.column_stack([lev, lev**2, roa, intcov, logassets])
    b = np.array([2.5, -3.0, -4.5, -0.6, -0.4])
    log_h = Z @ b + rng.normal(0, 0.1, n)
    h = np.exp(log_h) * hr_factor
    times = rng.exponential(1.0/h, n)
    C = rng.uniform(0.5, 12.0, n)
    T = np.minimum(times, C)
    E = (times <= C).astype(int)
    return T, E

T1, E1 = gen_loan_data(11, hr_factor=1.0)
T2, E2 = gen_loan_data(22, hr_factor=2.5)

km_plot = lambda ax, T, E, label, color: ax.step(km_estimator(T, E)[:,0], km_estimator(T, E)[:,1],
                                                  where="post", lw=2.4, label=label, color=color)

km_plot(axes[0], T1, E1, "Low leverage cohort", "#1F3B66")
km_plot(axes[0], T2, E2, "High leverage cohort", "#D9654C")
axes[0].set_xlabel("Years from issuance")
axes[0].set_ylabel("Survival probability S(t)")
axes[0].set_title("Cox survival curves: leverage regime contrast",
                  fontsize=11, fontweight="bold")
axes[0].set_ylim(-0.02, 1.02)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

t_grid = np.linspace(0.01, 10, 200)
km1 = km_estimator(T1, E1)
km2 = km_estimator(T2, E2)
def interp_km(km, t_grid):
    return np.interp(t_grid, km[:,0], km[:,1], left=1.0)

S1 = interp_km(km1, t_grid)
S2 = interp_km(km2, t_grid)
LGD = 0.45
EL1 = (1 - S1) * LGD
EL2 = (1 - S2) * LGD

axes[1].plot(t_grid, EL1, lw=2.4, color="#1F3B66", label="Low leverage")
axes[1].plot(t_grid, EL2, lw=2.4, color="#D9654C", label="High leverage")
axes[1].axhline(LGD, color="gray", lw=0.8, ls="--", label="LGD cap")
axes[1].set_xlabel("Years")
axes[1].set_ylabel("Cumulative expected loss (LGD=0.45)")
axes[1].set_title("Expected loss curves integrated from S(t)",
                  fontsize=11, fontweight="bold")
axes[1].set_ylim(-0.02, 0.5)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

fig.suptitle("Why Cox Beats Logistic for Credit Risk: full survival curve vs. single PD",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/expected_loss_curves.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("OK 3/3: expected_loss_curves.png")
print("All charts for survival-analysis-default generated.")
