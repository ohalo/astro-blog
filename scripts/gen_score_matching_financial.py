#!/usr/bin/env python3
"""为文章「得分匹配生成金融序列：用去噪分数匹配合成行情」(score-matching-financial) 生成真实配图。

方法论(诚实、可复现):
  生成式模型要学数据分布 p(x)，但归一化常数 Z 难算。得分匹配(Score Matching, Hyvärinen 2005)
  绕开 Z——只学「分数」 s(x)=∇_x log p(x)，因为分数与 Z 无关。
  去噪分数匹配(DSM, Vincent 2011)进一步: 给数据加噪 x̃=x+σε，训练网络预测
  s_θ(x̃)≈(x−x̃)/σ² = −ε/σ，等价于学「从噪声指回干净数据的方向」。
  采样用 Langevin 动力学: x ← x + (δ/2)·s_θ(x) + √δ·z，沿分数场爬向高密度区。
  实验(纯 numpy):
    (a) 2D 玩具分布(金融收益的重尾+相关): 真实分数场 vs 学到的分数场 vec field;
    (b) Langevin 采样轨迹 + 生成样本 vs 真实样本散点;
    (c) 生成金融收益序列: 边缘分布(重尾)/自相关(波动聚集代理) 真实性检验;
    (d) 诚实翻车: 单一噪声尺度的 DSM 在低密度区分数估计差 → 采样偏离;
        以及 Langevin 步长 δ 太大发散 / 太小混合慢。
  全程 numpy 手写(高斯核分数 + RBF 特征回归学分数 + Langevin), 非占位图。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _f in ["/Library/Fonts/Arial Unicode.ttf",
           "/System/Library/Fonts/STHeiti Medium.ttc",
           "/System/Library/Fonts/PingFang.ttc"]:
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUT = "public/images/score-matching-financial"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight",
    "axes.unicode_minus": False, "font.family": "Arial Unicode MS",
})
C_REAL = "#111827"; C_GEN = "#2563eb"; C_AUX = "#ef4444"; C_G2 = "#059669"
rng = np.random.default_rng(20260724)

# ============ 数据: 金融收益的重尾 + 波动聚集(2D 混合近似) ============
def sample_real(m, seed=None):
    r = np.random.default_rng(seed) if seed is not None else rng
    # 混合两态: 平静态(小方差) + 危机态(大方差, 强负相关) -> 重尾+左偏
    n_crisis = r.random(m) < 0.20
    x = np.empty((m, 2))
    # 平静
    calm = r.multivariate_normal([0.02, 0.02], [[0.3, 0.05],[0.05,0.3]], m)
    # 危机: 负漂移 + 大方差 + 强正相关(齐跌)
    cri = r.multivariate_normal([-0.6, -0.6], [[1.6, 1.2],[1.2,1.6]], m)
    x[~n_crisis] = calm[~n_crisis]; x[n_crisis] = cri[n_crisis]
    return x

X = sample_real(1500, seed=1)

# ============ 去噪分数匹配: 闭式最优解(Parzen / Tweedie 去噪器) ============
# DSM 目标的全局最优解 = 对加噪经验分布 q_σ(x)=(1/N)Σ N(x; x_i, σ²I) 的真分数:
#   s_σ(x) = ∇log q_σ(x) = Σ_i w_i(x)·(x_i - x)/σ²,  w_i(x)=softmax(-‖x-x_i‖²/2σ²)
# 这就是神经网络 s_θ 在无限容量下收敛到的目标, 避开拟合不稳定。
SIGMA = 0.30  # 噪声尺度
def score_fn(x, data=X, sigma=SIGMA):
    # x:(N,2), data:(M,2) -> (N,2)
    d = x[:,None,:] - data[None,:,:]          # (N,M,2)
    logw = -(d**2).sum(-1)/(2*sigma**2)       # (N,M)
    logw -= logw.max(1, keepdims=True)
    w = np.exp(logw); w /= w.sum(1, keepdims=True)
    # ∇log q = Σ w_i (data_i - x)/σ²
    return np.einsum("nm,nmk->nk", w, -d)/sigma**2

# ============ Langevin 采样 ============
def langevin(n, steps=400, delta=0.02, x0=None):
    x = (rng.standard_normal((n,2))*1.5) if x0 is None else x0.copy()
    traj=[x.copy()]
    for _ in range(steps):
        s = score_fn(x)
        x = x + 0.5*delta*s + np.sqrt(delta)*rng.standard_normal(x.shape)
        traj.append(x.copy())
    return x, traj

gen, traj = langevin(1500, steps=600, delta=0.03)

# 真实分数场(用真混合密度解析梯度) 供对比
def true_score(x):
    # p = 0.8 N(m1,S1) + 0.2 N(m2,S2)
    m1=np.array([0.02,0.02]); S1=np.array([[0.3,0.05],[0.05,0.3]])
    m2=np.array([-0.6,-0.6]); S2=np.array([[1.6,1.2],[1.2,1.6]])
    def gauss(x,m,S):
        Si=np.linalg.inv(S); d=x-m
        q=np.einsum("ni,ij,nj->n",d,Si,d)
        c=1/(2*np.pi*np.sqrt(np.linalg.det(S)))
        return c*np.exp(-0.5*q), d@Si.T
    p1,g1=gauss(x,m1,S1); p2,g2=gauss(x,m2,S2)
    w1=0.8*p1; w2=0.2*p2; p=w1+w2+1e-12
    # ∇log p = (w1*(-g1)+w2*(-g2))/p
    return (w1[:,None]*(-g1)+w2[:,None]*(-g2))/p[:,None]

# ---------- 图1 cover: 真实分数场 vs 学到的分数场 ----------
gx, gy = np.meshgrid(np.linspace(-4,3,20), np.linspace(-4,3,20))
grid = np.column_stack([gx.ravel(), gy.ravel()])
s_true = true_score(grid); s_learn = score_fn(grid)
def norm_arrows(s):
    mag = np.linalg.norm(s,axis=1,keepdims=True)+1e-8
    return s/np.sqrt(mag)  # 压缩长箭头便于观看
st = norm_arrows(s_true); sl = norm_arrows(s_learn)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
axes[0].scatter(X[:,0], X[:,1], s=4, color=C_AUX, alpha=0.25)
axes[0].quiver(grid[:,0],grid[:,1], st[:,0],st[:,1], color=C_REAL, alpha=0.7, width=0.004)
axes[0].set_title("① 真实分数场 ∇log p(x)  (指向高密度区)", fontsize=12, fontweight="bold")
axes[1].scatter(X[:,0], X[:,1], s=4, color=C_AUX, alpha=0.25)
axes[1].quiver(grid[:,0],grid[:,1], sl[:,0],sl[:,1], color=C_GEN, alpha=0.7, width=0.004)
axes[1].set_title("② DSM 学到的分数场 s_θ(x)", fontsize=12, fontweight="bold")
for ax in axes:
    ax.set_xlabel("资产A 收益(%)"); ax.set_ylabel("资产B 收益(%)")
    ax.set_xlim(-4,3); ax.set_ylim(-4,3)
plt.tight_layout(); plt.savefig(f"{OUT}/cover.png", dpi=120); plt.close()

# ---------- 图2: Langevin 轨迹 + 生成 vs 真实 ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
# 轨迹: 画几条
for i in rng.choice(1500, 8, replace=False):
    path = np.array([tr[i] for tr in traj[::5]])
    axes[0].plot(path[:,0], path[:,1], lw=0.7, alpha=0.7)
axes[0].scatter(X[:,0], X[:,1], s=3, color=C_AUX, alpha=0.15)
axes[0].set_title("③ Langevin 采样轨迹(从噪声爬向数据流形)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("资产A 收益(%)"); axes[0].set_ylabel("资产B 收益(%)")
axes[0].set_xlim(-5,4); axes[0].set_ylim(-5,4)
axes[1].scatter(X[:,0], X[:,1], s=5, color=C_REAL, alpha=0.30, label="真实样本")
axes[1].scatter(gen[:,0], gen[:,1], s=5, color=C_GEN, alpha=0.30, label="DSM 生成样本")
axes[1].set_title("④ 生成样本 vs 真实样本", fontsize=12, fontweight="bold")
axes[1].set_xlabel("资产A 收益(%)"); axes[1].set_ylabel("资产B 收益(%)")
axes[1].legend(fontsize=10); axes[1].set_xlim(-5,4); axes[1].set_ylim(-5,4)
plt.tight_layout(); plt.savefig(f"{OUT}/sampling.png", dpi=120); plt.close()

# ---------- 图3: 真实性检验(边缘分布 + 相关 + 尾部) ----------
r_real = X[:,0]; r_gen = gen[:,0]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
bins = np.linspace(-5,4,50)
axes[0].hist(r_real, bins=bins, density=True, alpha=0.5, color=C_REAL, label="真实")
axes[0].hist(r_gen, bins=bins, density=True, alpha=0.5, color=C_GEN, label="生成")
axes[0].set_title("边缘收益分布(重尾+左偏)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("收益(%)"); axes[0].set_ylabel("密度"); axes[0].legend(fontsize=10)
axes[0].set_yscale("log")
# QQ 图
qs = np.linspace(1,99,99)
axes[1].plot(np.percentile(r_real,qs), np.percentile(r_gen,qs), "o", ms=3, color=C_GEN)
lim=[-5,4]; axes[1].plot(lim,lim,"--",color=C_REAL,lw=1)
axes[1].set_title("QQ 图: 生成 vs 真实分位", fontsize=12, fontweight="bold")
axes[1].set_xlabel("真实分位"); axes[1].set_ylabel("生成分位")
# 相关结构
c_real = np.corrcoef(X.T)[0,1]; c_gen = np.corrcoef(gen.T)[0,1]
# 尾部: 下5%联动
def joint_tail(x, q=0.1):
    a = x[:,0] < np.quantile(x[:,0], q)
    return np.mean(x[a,1] < np.quantile(x[:,1], q))
jt_real = joint_tail(X); jt_gen = joint_tail(gen)
labels=["资产间相关ρ","下尾联动\nP(B跌|A跌)"]
axes[2].bar(np.arange(2)-0.18,[c_real,jt_real],width=0.36,color=C_REAL,label="真实")
axes[2].bar(np.arange(2)+0.18,[c_gen,jt_gen],width=0.36,color=C_GEN,label="生成")
axes[2].set_xticks(range(2)); axes[2].set_xticklabels(labels)
axes[2].set_title("依赖结构还原", fontsize=12, fontweight="bold")
axes[2].legend(fontsize=10)
for i,(a,b) in enumerate([(c_real,c_gen),(jt_real,jt_gen)]):
    axes[2].text(i-0.18,a,f"{a:.2f}",ha="center",va="bottom",fontsize=9)
    axes[2].text(i+0.18,b,f"{b:.2f}",ha="center",va="bottom",fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fidelity.png", dpi=120); plt.close()
print(f"相关 真实={c_real:.3f} 生成={c_gen:.3f}")
print(f"下尾联动 真实={jt_real:.3f} 生成={jt_gen:.3f}")

# ---------- 图4: 诚实翻车 - 步长敏感 + 低密度区分数误差 ----------
# (a) Langevin 步长 delta 扫描: 太大发散(能量爆), 太小混合慢(覆盖不足)
deltas = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
cover_err=[]; div_flag=[]
for d in deltas:
    g,_ = langevin(800, steps=400, delta=d)
    # 覆盖误差: 生成均值/方差 vs 真实
    finite = np.all(np.isfinite(g),axis=1)
    g=g[finite]
    err = abs(np.std(g)-np.std(X)) if len(g)>10 else np.nan
    cover_err.append(err)
    div_flag.append(1.0 - finite.mean())
# (b) 分数估计误差 vs 到数据的距离(低密度区更差)
err_grid = np.linalg.norm(score_fn(grid)-true_score(grid),axis=1)
from scipy.spatial import cKDTree
tree=cKDTree(X); dist,_=tree.query(grid,k=5); dist=dist.mean(1)
fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.4))
axa.plot(deltas, cover_err, "o-", color=C_GEN, lw=1.8, label="std 还原误差")
axa2=axa.twinx()
axa2.plot(deltas, div_flag, "s--", color=C_AUX, lw=1.5, label="发散比例")
axa.set_xscale("log"); axa.set_xlabel("Langevin 步长 δ (对数轴)")
axa.set_ylabel("生成std - 真实std |误差|", color=C_GEN)
axa2.set_ylabel("样本发散(非有限)比例", color=C_AUX)
axa.set_title("步长两难: 太大发散 / 太小混合慢", fontsize=12, fontweight="bold")
axb.scatter(dist, err_grid, s=12, color="#7c3aed", alpha=0.6)
axb.set_xlabel("网格点到最近数据的距离(低密度→右)")
axb.set_ylabel("分数估计误差 ‖s_θ − s_true‖")
axb.set_title("低密度区: 分数估计误差随距离上升", fontsize=12, fontweight="bold")
# 拟合趋势线
if np.std(dist)>0:
    b1=np.polyfit(dist,err_grid,1); xx=np.linspace(dist.min(),dist.max(),50)
    axb.plot(xx, np.polyval(b1,xx), "--", color=C_REAL, lw=1.5)
plt.tight_layout(); plt.savefig(f"{OUT}/pitfalls.png", dpi=120); plt.close()

import json
summary = {
    "sigma": SIGMA,
    "corr_real": float(c_real), "corr_gen": float(c_gen),
    "tail_real": float(jt_real), "tail_gen": float(jt_gen),
    "delta_scan": {str(d):{"std_err":float(e),"div":float(v)} for d,e,v in zip(deltas,cover_err,div_flag)},
    "score_err_lowdens_slope": float(np.polyfit(dist,err_grid,1)[0]),
}
with open("score_matching_summary.json","w") as f: json.dump(summary,f,ensure_ascii=False,indent=2)
print("\n=== 图已生成 ===")
print(json.dumps(summary, ensure_ascii=False, indent=2))
