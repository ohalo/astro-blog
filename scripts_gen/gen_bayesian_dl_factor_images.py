#!/usr/bin/env python3
"""生成「贝叶斯深度学习因子：用 MC Dropout 给神经网络预测配不确定性」配图。

纯 numpy 从零实现一个小 MLP 收益预测模型，并对比：
  (A) 确定性 MLP（dropout 推理时关闭，只给点预测）
  (B) MC Dropout 网络（推理时 dropout 保持开启，T 次随机前向取均值+方差）
在受控合成因子任务上计算四类真实指标（全部由脚本真实运行产出）：
  1. OOD 检测：分布外样本的预测方差显著高于分布内（AUC 分离）
  2. 选择性预测（OOD 门控）：在 ID+OOD 混合池上按不确定性弃投高不确定样本后，
     保留子集 RMSE 明显下降
  3. 校准：MC Dropout 只建模 epistemic 方差，漏掉 aleatoric 噪声 -> 90% 区间
     实际仅覆盖 ~50%；叠加训练集残差估计的 aleatoric 噪声后覆盖回升
  4. 诚实边界：iid 同方差下 epistemic σ 几乎不预测单样本 |误差|（r≈0）——揭示
     MC Dropout 的不确定性是"模型对该区域有多不熟"，不是"这只股票会错多少"
"""
import numpy as np
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False
fm._load_fontmanager()

SEED = 20260828
rng = np.random.default_rng(SEED)

# ============ 1. 合成因子任务：X(12维) -> 下一期收益 y = f(X)+噪声 ============
D = 12
N_TR = 2000
N_ID = 1000
N_OOD = 1000

W1 = rng.standard_normal((3,)) * 1.2
W2 = rng.standard_normal((3, 3)) * 0.6

def true_fn(X):
    a = X[:, 0] + 0.5 * X[:, 1]
    b = X[:, 2] - 0.3 * X[:, 0] * X[:, 1]
    c = np.tanh(X[:, :3] @ W1)
    return 0.6 * a - 0.4 * b + 0.8 * c

def make_data(n, ood=False):
    if ood:
        X = rng.standard_normal((n, D)) * 2.2 + 1.5   # 协变量漂移
    else:
        X = rng.standard_normal((n, D))
    y = true_fn(X) + rng.standard_normal(n) * 0.35     # 同方差噪声
    return X, y

Xtr, ytr = make_data(N_TR)
Xid, yid = make_data(N_ID)
Xood, yood = make_data(N_OOD, ood=True)

# ============ 2. 从零 MLP（含 dropout） ============
H1, H2 = 32, 32
p_drop = 0.15

def init_params():
    np.random.seed(SEED)
    return {
        "W1": rng.standard_normal((D, H1)) * np.sqrt(2.0 / D),
        "b1": np.zeros(H1),
        "W2": rng.standard_normal((H1, H2)) * np.sqrt(2.0 / H1),
        "b2": np.zeros(H2),
        "W3": rng.standard_normal((H2, 1)) * np.sqrt(2.0 / H2),
        "b3": np.zeros(1),
    }

def forward(X, p, dropout_on, train_mask=None):
    if dropout_on:
        if train_mask is None:
            m1 = (rng.random((X.shape[0], H1)) > p_drop).astype(float) / (1 - p_drop)
            m2 = (rng.random((X.shape[0], H2)) > p_drop).astype(float) / (1 - p_drop)
        else:
            m1, m2 = train_mask
    else:
        m1 = np.ones((X.shape[0], H1)); m2 = np.ones((X.shape[0], H2))
    z1 = X @ p["W1"] + p["b1"]
    h1 = np.maximum(0, z1) * m1
    z2 = h1 @ p["W2"] + p["b2"]
    h2 = np.maximum(0, z2) * m2
    y = h2 @ p["W3"] + p["b3"]
    return y.ravel(), (m1, m2)

def forward_cache(X, p):
    m1 = (rng.random((X.shape[0], H1)) > p_drop).astype(float) / (1 - p_drop)
    m2 = (rng.random((X.shape[0], H2)) > p_drop).astype(float) / (1 - p_drop)
    z1 = X @ p["W1"] + p["b1"]; a1 = np.maximum(0, z1); h1 = a1 * m1
    z2 = h1 @ p["W2"] + p["b2"]; a2 = np.maximum(0, z2); h2 = a2 * m2
    y = h2 @ p["W3"] + p["b3"]
    return y.ravel(), dict(m1=m1, m2=m2, a1=a1, a2=a2, h1=h1, h2=h2, z1=z1, z2=z2)

def backward(X, y, p, cache):
    n = X.shape[0]
    yh = (cache["h2"] @ p["W3"] + p["b3"]).ravel()
    err = yh - y
    m1, m2, z1, z2 = cache["m1"], cache["m2"], cache["z1"], cache["z2"]
    gW3 = cache["h2"].T @ err[:, None]
    gb3 = err.sum()
    dh2 = (err[:, None] @ p["W3"].T) * (z2 > 0) * m2
    gW2 = cache["h1"].T @ dh2
    gb2 = dh2.sum(axis=0)
    dh1 = (dh2 @ p["W2"].T) * (z1 > 0) * m1
    gW1 = X.T @ dh1
    gb1 = dh1.sum(axis=0)
    return {"W1": gW1 / n, "b1": gb1 / n, "W2": gW2 / n, "b2": gb2 / n,
            "W3": gW3 / n, "b3": np.array([gb3 / n])}

def train(p, lr0=0.01, n_iter=1500):
    lr = lr0
    for it in range(n_iter):
        yh, c = forward_cache(Xtr, p)
        g = backward(Xtr, ytr, p, c)
        for k in p:
            p[k] -= lr * g[k]
        lr *= 0.9995
    return p

p_det = train(init_params())
p_mc = train(init_params())

# ============ 3. 推理 ============
def mc_predict(X, p, T=50):
    preds = np.zeros((X.shape[0], T))
    for t in range(T):
        yh, _ = forward(X, p, dropout_on=True)
        preds[:, t] = yh
    return preds.mean(axis=1), preds.std(axis=1)

yid_mc, sid_mc = mc_predict(Xid, p_mc)
yood_mc, sood_mc = mc_predict(Xood, p_mc)

# ============ 4. 指标计算 ============
# A. OOD 检测 AUC
all_std = np.concatenate([sid_mc, sood_mc])
labels = np.concatenate([np.zeros(len(sid_mc)), np.ones(len(sood_mc))])
order_a = all_std.argsort()[::-1]
tp = np.cumsum(labels[order_a] == 1)
fp = np.cumsum(labels[order_a] == 0)
auc = float(np.trapezoid(tp / tp[-1], fp / fp[-1])) if tp[-1] > 0 else 0.5
id_std_mean = float(sid_mc.mean())
ood_std_mean = float(sood_mc.mean())

# B. 选择性预测（ID+OOD 混合池按 σ 弃投）
Xmix = np.vstack([Xid, Xood])
ymix = np.concatenate([yid, yood])
ymix_mc, smix_mc = mc_predict(Xmix, p_mc)
full_rmse_mix = float(np.sqrt(np.mean((ymix - ymix_mc) ** 2)))
order_u = np.argsort(-smix_mc)
def rmse_retained(frac):
    k = int(len(ymix) * frac)            # 弃投数量 = 最高不确定样本数
    idx = order_u[k:]                     # 保留最低不确定样本（尾部）
    return float(np.sqrt(np.mean((ymix[idx] - ymix_mc[idx]) ** 2)))
abstain_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
rmse_curve = [rmse_retained(f) for f in abstain_grid]
rmse_at_30 = rmse_retained(0.30)
rmse_drop_pct = (full_rmse_mix - rmse_at_30) / full_rmse_mix

# C. 校准：仅 epistemic vs epistemic+aleatoric
ytr_hat, _ = forward(Xtr, p_mc, dropout_on=False)
sigma_aleat = float(np.std(ytr - ytr_hat))
preds_id_epi = np.zeros((Xid.shape[0], 50))
for t in range(50):
    yh, _ = forward(Xid, p_mc, dropout_on=True)
    preds_id_epi[:, t] = yh
preds_id_corr = preds_id_epi + rng.standard_normal((Xid.shape[0], 50)) * sigma_aleat
nominal = np.array([0.5, 0.7, 0.8, 0.9, 0.95])
def coverage(preds):
    out = []
    for q in nominal:
        lo = np.percentile(preds, 100*(1-q)/2, axis=1)
        hi = np.percentile(preds, 100*(1+q)/2, axis=1)
        out.append(float(np.mean((yid >= lo) & (yid <= hi))))
    return out
cov_epi = coverage(preds_id_epi)
cov_corr = coverage(preds_id_corr)
cov_90_epi = cov_epi[3]
cov_90_corr = cov_corr[3]

# D. epistemic σ 与 |误差|（iid 同方差下）
err_id = np.abs(yid - yid_mc)
pear_epi = float(np.corrcoef(sid_mc, err_id)[0, 1])
bins = np.linspace(sid_mc.min(), sid_mc.max(), 6)
bin_err = [float(err_id[(sid_mc >= bins[i]) & (sid_mc < bins[i+1])].mean()) for i in range(5)]

# ============ 5. 绘图 ============
outdir = "public/images/bayesian-deep-learning-factor"
os.makedirs(outdir, exist_ok=True)

# 图1 cover：ID vs OOD 不确定性分布
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.hist(sid_mc, bins=40, alpha=0.6, color="#1a9850", label=f"分布内 ID（μ={id_std_mean:.3f}）", density=True)
ax.hist(sood_mc, bins=40, alpha=0.6, color="#d73027", label=f"分布外 OOD（μ={ood_std_mean:.3f}）", density=True)
ax.set_xlabel("预测标准差 σ（MC Dropout 估计的 epistemic 不确定性）")
ax.set_ylabel("密度")
ax.set_title(f"OOD 检测：分布外样本不确定性显著抬升（分离 AUC={auc:.3f}）",
             fontsize=12.5, fontweight="bold", color="#1f3a5f")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{outdir}/ood_uncertainty.png", dpi=130)
plt.close(fig)

# 图2：选择性预测（OOD 门控）
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.plot([f*100 for f in abstain_grid], rmse_curve, "s-", color="#4393c3", lw=2, label="MC Dropout 保留子集 RMSE")
ax.axhline(full_rmse_mix, color="gray", ls="--", lw=1.2, label=f"全样本 RMSE={full_rmse_mix:.3f}")
ax.set_xlabel("按不确定性弃投比例 (%)")
ax.set_ylabel("保留样本 RMSE")
ax.set_title(f"选择性预测（OOD 门控）：弃投 30% 最不确定样本后 RMSE 降 {rmse_drop_pct*100:.0f}%",
             fontsize=12.5, fontweight="bold", color="#1f3a5f")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{outdir}/selective_prediction.png", dpi=130)
plt.close(fig)

# 图3：校准（仅 epistemic 欠覆盖 vs 叠加 aleatoric）
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.plot(nominal*100, nominal*100, "k--", lw=1.2, label="完美校准线")
ax.plot(nominal*100, np.array(cov_epi)*100, "o-", color="#d73027", lw=2, label=f"仅 epistemic（90%→{cov_90_epi*100:.0f}%）")
ax.plot(nominal*100, np.array(cov_corr)*100, "s-", color="#1a9850", lw=2, label=f"epistemic+aleatoric（90%→{cov_90_corr*100:.0f}%）")
ax.set_xlabel("名义预测区间置信度 (%)")
ax.set_ylabel("经验覆盖率 (%)")
ax.set_title("校准：MC Dropout 只建模 epistemic 风险，叠加 aleatoric 噪声后覆盖回升",
             fontsize=11.5, fontweight="bold", color="#1f3a5f")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{outdir}/interval_calibration.png", dpi=130)
plt.close(fig)

# 图4：epistemic σ 与 |误差|（诚实边界）
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.scatter(sid_mc, err_id, s=12, alpha=0.35, color="#4393c3")
centers = [(bins[i]+bins[i+1])/2 for i in range(5)]
ax.plot(centers, bin_err, "o-", color="#1a9850", lw=2, label="按 σ 分箱平均 |误差|")
ax.set_xlabel("预测标准差 σ（epistemic uncertainty）")
ax.set_ylabel("真实 |误差|")
ax.set_title(f"诚实边界：iid 同方差下 epistemic σ 几乎不预测单样本误差（r={pear_epi:.2f}）",
             fontsize=11.5, fontweight="bold", color="#1f3a5f")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{outdir}/uncertainty_vs_error.png", dpi=130)
plt.close(fig)

# ============ 6. stats ============
stats = {
    "D": D, "N_train": N_TR, "N_id": N_ID, "N_ood": N_OOD, "T_mc": 50, "p_drop": p_drop,
    "ood_auc": round(auc, 4),
    "id_std_mean": round(id_std_mean, 4),
    "ood_std_mean": round(ood_std_mean, 4),
    "full_rmse_mix": round(full_rmse_mix, 4),
    "rmse_at_30pct_abstain": round(rmse_at_30, 4),
    "rmse_drop_pct": round(float(rmse_drop_pct), 4),
    "sigma_aleat": round(sigma_aleat, 4),
    "nominal": [round(float(x), 3) for x in nominal],
    "cov_epi": [round(x, 4) for x in cov_epi],
    "cov_corr": [round(x, 4) for x in cov_corr],
    "cov_90_epi": round(cov_90_epi, 4),
    "cov_90_corr": round(cov_90_corr, 4),
    "pearson_epistemic_error": round(pear_epi, 4),
    "bin_err": [round(x, 4) for x in bin_err],
}
with open(f"{outdir}/stats.json", "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print("=== Bayesian DL (MC Dropout) metrics ===")
print(f"  OOD AUC = {auc:.4f}  (ID mu={id_std_mean:.3f}, OOD mu={ood_std_mean:.3f})")
print(f"  full RMSE(mix)={full_rmse_mix:.4f} -> @30% abstain {rmse_at_30:.4f} (drop {rmse_drop_pct*100:.0f}%)")
print(f"  sigma_aleat={sigma_aleat:.4f}")
print(f"  coverage@90%: epistemic={cov_90_epi*100:.1f}% | epistemic+aleatoric={cov_90_corr*100:.1f}%")
print(f"  pearson(sigma_epi,|err|)={pear_epi:.4f}  bin_err={np.round(bin_err,3)}")
print("Bayesian DL images written.")
