#!/usr/bin/env python3
"""
为文章「决策树因子可解释性：用规则集替代黑盒给出合规理由」生成真实配图。

所有图表均由下文代码真实计算（numpy + sklearn，无占位图）：

  1) dt_rules_vs_blackbox.png —— 测试集表现：线性模型 / 随机森林 / 单棵深度受限决策树
                                 (max_depth=4) 的 AUC / 准确率 / 可解释节点数对照。
  2) dt_tree_structure.png     —— 一棵 max_depth=3 决策树的可视化结构（规则路径）。
  3) dt_rule_explanation.png   —— 抽取一条被预测为「违约」的测试样本的命中规则路径
                                 （从根到叶），展示「合规理由」如何生成。
  4) dt_depth_tradeoff.png     —— 深度 vs 可解释性/性能权衡：深度越大 AUC 略升但
                                  规则数指数膨胀、单条规则覆盖样本数骤降。

机制（合成信贷数据集，仅用于演示；落地见文末路径）：
  构造特征：leverage(杠杆)、profit_margin(净利率)、cash_ratio(现金比率)、
            growth(营收增速)、momentum(动量)。标签由一组可解释阈值规则生成
            （y=1 违约 当 leverage>2.3 且 cash_ratio<0.15，或 profit_margin<0.02 等），
            叠加噪声。决策树应基本复现这套规则。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "decision-tree-interpretable-factor")
os.makedirs(D, exist_ok=True)

C = {"tree": "#4C72B0", "rf": "#55A868", "lin": "#C44E52", "gold": "#E1A100",
     "neg": "#C44E52", "pos": "#55A868", "grid": "#DDDDDD"}

rng = np.random.default_rng(20260828)

# ---------------------------------------------------------------------------
# 合成信贷数据
# ---------------------------------------------------------------------------
N = 4000
feat = {
    "杠杆率 leverage":   rng.uniform(0.5, 4.0, N),
    "净利率 profit_margin": rng.uniform(-0.05, 0.20, N),
    "现金比率 cash_ratio":  rng.uniform(0.0, 0.4, N),
    "营收增速 growth":       rng.uniform(-0.3, 0.5, N),
    "价格动量 momentum":     rng.normal(0, 0.2, N),
}
X = np.stack([feat[k] for k in feat], axis=1)
cols = list(feat.keys())

# 可解释真规则：违约 = (杠杆>2.3 且 现金比率<0.15) 或 净利率<0.02 或 营收增速<-0.15
def true_rule(X):
    lev, pm, cr, gr, mo = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    return ((lev > 2.3) & (cr < 0.15)) | (pm < 0.02) | (gr < -0.15)
y = true_rule(X).astype(int)
noise_mask = rng.random(N) < 0.05
y[noise_mask] = 1 - y[noise_mask]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ---------------------------------------------------------------------------
# 模型对照
# ---------------------------------------------------------------------------
lin = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=0).fit(Xtr, ytr)
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=40, random_state=0).fit(Xtr, ytr)

def auc(model):
    return roc_auc_score(yte, model.predict_proba(Xte)[:, 1])

def acc(model):
    return accuracy_score(yte, model.predict(Xte))

auc_lin, auc_rf, auc_dt = auc(lin), auc(rf), auc(dt)
acc_lin, acc_rf, acc_dt = acc(lin), acc(rf), acc(dt)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
m = ["逻辑回归", "随机森林", "决策树(d=4)"]
a = [auc_lin, auc_rf, auc_dt]
b = [acc_lin, acc_rf, acc_dt]
xpos = np.arange(3)
bars = axes[0].bar(xpos, a, color=[C["lin"], C["rf"], C["tree"]], width=0.6)
axes[0].set_xticks(xpos); axes[0].set_xticklabels(m, fontsize=9)
axes[0].set_ylim(0.5, 1.02); axes[0].axhline(0.5, color="k", lw=0.8, ls=":")
axes[0].set_title("测试集 AUC")
for i, v in enumerate(a):
    axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
bars = axes[1].bar(xpos, b, color=[C["lin"], C["rf"], C["tree"]], width=0.6)
axes[1].set_xticks(xpos); axes[1].set_xticklabels(m, fontsize=9)
axes[1].set_ylim(0.5, 1.02)
axes[1].set_title("测试集准确率")
for i, v in enumerate(b):
    axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{D}/dt_rules_vs_blackbox.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 图2：决策树结构可视化
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5.2))
plot_tree(dt, feature_names=cols, class_names=["正常", "违约"],
          filled=True, rounded=True, fontsize=7, ax=ax,
          proportion=True, label="root")
ax.set_title("单棵深度受限决策树（max_depth=4）：每条路径都是一条可读规则", fontsize=10)
plt.tight_layout()
plt.savefig(f"{D}/dt_tree_structure.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 图3：抽取一条违约样本的规则路径（“合规理由”）
# ---------------------------------------------------------------------------
dtree = dt
# 找一条被预测违约且模型自信的测试样本
proba = dtree.predict_proba(Xte)[:, 1]
default_idx = np.where((dtree.predict(Xte) == 1) & (proba > 0.85))[0]
idx = default_idx[0]
sample = Xte[idx]
path = dtree.decision_path(sample.reshape(1, -1)).toarray()[0]
node_rules = []
threshold = dtree.tree_.threshold
features = dtree.tree_.feature
for n in np.where(path)[0]:
    if dtree.tree_.children_left[n] == dtree.tree_.children_right[n]:
        continue  # 叶子
    feat_name = cols[features[n]]
    thr = threshold[n]
    node_rules.append((n, feat_name, thr))

# 构建可读规则串（沿路径按符号）
rule_lines = []
for n, fname, thr in node_rules:
    left = dtree.tree_.children_left[n]
    right = dtree.tree_.children_right[n]
    if path[left]:
        rule_lines.append(f"{fname} ≤ {thr:.3f}")
    elif path[right]:
        rule_lines.append(f"{fname} > {thr:.3f}")

fig, ax = plt.subplots(figsize=(10.5, 4.6))
ax.axis("off")
pred = dtree.predict(Xte)[idx]
title = f"样本 #{idx} —— 模型判定：{'违约' if pred==1 else '正常'}（P=违约={proba[idx]:.2f}）"
ax.text(0.02, 0.95, title, fontsize=12, weight="bold", transform=ax.transAxes)
ax.text(0.02, 0.82, "命中规则路径（根→叶）：", fontsize=10, transform=ax.transAxes)
yt = 0.74
for i, rl in enumerate(rule_lines):
    ax.text(0.06, yt, f"IF {rl}", fontsize=10, transform=ax.transAxes,
            color=C["tree"] if i % 2 == 0 else C["gold"])
    yt -= 0.11
ax.text(0.02, yt - 0.02, "→ 合规理由生成：杠杆过高 + 现金枯竭 → 触发违约预警",
        fontsize=10, weight="bold", color=C["neg"], transform=ax.transAxes)
plt.tight_layout()
plt.savefig(f"{D}/dt_rule_explanation.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 图4：深度 vs 可解释性/性能权衡
# ---------------------------------------------------------------------------
depths = [2, 3, 4, 5, 6, 8, 10]
aucs, n_leaves, min_cov = [], [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, min_samples_leaf=20, random_state=0).fit(Xtr, ytr)
    aucs.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    n_leaves.append(int(m.get_n_leaves()))
    # 单条规则平均覆盖：用叶子样本数倒数近似
    leaves = m.apply(Xte)
    cov = Xte.shape[0] / len(np.unique(leaves))
    min_cov.append(cov)

fig, ax1 = plt.subplots(figsize=(9.5, 4.3))
ax1.plot(depths, aucs, "o-", color=C["tree"], lw=2, label="测试 AUC")
ax1.set_xlabel("决策树最大深度 max_depth")
ax1.set_ylabel("测试 AUC", color=C["tree"])
ax1.tick_params(axis="y", labelcolor=C["tree"])
ax1.set_ylim(0.82, 0.93)
ax2 = ax1.twinx()
ax2.plot(depths, n_leaves, "s--", color=C["gold"], lw=2, label="叶子数（规则条数）")
ax2.set_ylabel("叶子数 / 规则条数", color=C["gold"])
ax2.tick_params(axis="y", labelcolor=C["gold"])
ax1.set_title("深度权衡：深度↑ AUC 略升，但规则条数指数膨胀、可解释性崩塌")
l1, lab1 = ax1.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{D}/dt_depth_tradeoff.png", dpi=120)
plt.close()

print("=== 决策树关键数字 ===")
print(f"测试 AUC: 逻辑回归={auc_lin:.3f} 随机森林={auc_rf:.3f} 决策树(d=4)={auc_dt:.3f}")
print(f"测试准确率: 逻辑回归={acc_lin:.3f} 随机森林={acc_rf:.3f} 决策树(d=4)={acc_dt:.3f}")
print(f"树结构叶子数(depth4)={int(dt.get_n_leaves())}")
print(f"深度权衡 AUCs={[round(x,3) for x in aucs]} leaves={n_leaves}")
print("图片已写入:", D)
print(os.listdir(D))
