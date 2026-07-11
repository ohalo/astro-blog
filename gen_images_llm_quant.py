#!/usr/bin/env python3
"""Generate images for 2 blog posts: llm-stock-prediction-finetuning & quant-backtesting-pitfalls"""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rng = np.random.default_rng(20260711)

BASE = "/Users/halo/workspace/astro-blog/public/images"

# ============================================================
# Article 1: llm-stock-prediction-finetuning
# ============================================================
D1 = os.path.join(BASE, "llm-stock-prediction-finetuning")
os.makedirs(D1, exist_ok=True)

# --- Image 1: Data labeling workflow ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")

boxes = [
    (1, 6, 2.5, 1.2, "原始数据\n(新闻/财报/公告)", "#E3F2FD"),
    (4.5, 6, 2.5, 1.2, "文本清洗\n(去噪/分词/对齐)", "#FFF3E0"),
    (1, 3.5, 2.5, 1.2, "时间对齐\n(文本↔股价时间戳)", "#E8F5E9"),
    (4.5, 3.5, 2.5, 1.2, "计算标签\n(N日后涨跌方向)", "#F3E5F5"),
    (1, 1, 2.5, 1.2, "样本平衡\n(过采样/欠采样)", "#FFEBEE"),
    (4.5, 1, 2.5, 1.2, "训练数据集\n(text, label)对", "#C8E6C9"),
    (8, 3.5, 1.5, 4, "质量\n检查\n去重\n去泄露", "#FFECB3"),
]
for x, y, w, h, txt, c in boxes:
    bb = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", facecolor=c, edgecolor="#333", linewidth=1.2)
    ax.add_patch(bb)
    ax.text(x+w/2, y+h/2, txt, ha="center", va="center", fontsize=9, fontweight="bold")

# Arrows
arrows = [((3.5, 6.6), (4.5, 6.6)), ((2.25, 6), (2.25, 4.7)), ((4.5, 4.7), (3.5, 4.7)),
          ((2.25, 3.5), (2.25, 2.2)), ((4.5, 2.2), (3.5, 2.2)), ((6, 3.5), (8, 3.5))]
for (x1,y1),(x2,y2) in arrows:
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

ax.set_title("金融文本数据标注流程", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
fig.savefig(os.path.join(D1, "data-labeling-workflow.jpg"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ data-labeling-workflow.jpg")

# --- Image 2: LoRA architecture ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")

# Original weight matrix
ax.add_patch(FancyBboxPatch((1, 4.5), 2.5, 2.5, boxstyle="round,pad=0.1", facecolor="#BBDEFB", edgecolor="#1976D2", lw=2))
ax.text(2.25, 5.75, "预训练权重\nW ∈ R^{d×k}", ha="center", va="center", fontsize=10, fontweight="bold")

# LoRA matrices
ax.add_patch(FancyBboxPatch((5, 6), 1.5, 1, boxstyle="round,pad=0.1", facecolor="#FFCDD2", edgecolor="#D32F2F", lw=2))
ax.text(5.75, 6.5, "A ∈ R^{d×r}", ha="center", va="center", fontsize=9, fontweight="bold")
ax.add_patch(FancyBboxPatch((5, 4.5), 1.5, 1, boxstyle="round,pad=0.1", facecolor="#FFCDD2", edgecolor="#D32F2F", lw=2))
ax.text(5.75, 5, "B ∈ R^{r×k}", ha="center", va="center", fontsize=9, fontweight="bold")

# Result
ax.add_patch(FancyBboxPatch((7.5, 4.5), 2, 2.5, boxstyle="round,pad=0.1", facecolor="#C8E6C9", edgecolor="#388E3C", lw=2))
ax.text(8.5, 5.75, "W + α·A·B\n(仅训练A,B)", ha="center", va="center", fontsize=9, fontweight="bold")

# Plus sign
ax.text(4.2, 5.75, "+", fontsize=20, fontweight="bold", ha="center", va="center")
# Equals
ax.text(6.9, 5.75, "=", fontsize=20, fontweight="bold", ha="center", va="center")

# Info box at bottom
ax.add_patch(FancyBboxPatch((1, 1.2), 8.5, 2.5, boxstyle="round,pad=0.2", facecolor="#FFF8E1", edgecolor="#F57F17", lw=1.5))
info_text = (
    "LoRA 核心思想：冻结原始权重 W，只训练低秩矩阵 A 和 B\n"
    "参数量级：r << min(d, k)，通常 r=8~64\n"
    "训练效率：相比全参数微调，显存需求降低 70%，训练速度提升 3 倍"
)
ax.text(5.25, 2.45, info_text, ha="center", va="center", fontsize=9, linespacing=1.8)
ax.set_title("LoRA（Low-Rank Adaptation）微调架构", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
fig.savefig(os.path.join(D1, "lora-architecture.jpg"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ lora-architecture.jpg")

# ============================================================
# Article 2: quant-backtesting-pitfalls
# ============================================================
D2 = os.path.join(BASE, "quant-backtesting-pitfalls")
os.makedirs(D2, exist_ok=True)

# --- Image 3: Overfitting vs Generalization ---
np.random.seed(42)
x = np.linspace(-3, 3, 30)
y_true = 2*np.sin(x) + 0.3*x**2
y_noisy = y_true + rng.normal(0, 0.5, len(x))

# Overfitted: high-degree polynomial
coef_overfit = np.polyfit(x, y_noisy, 15)
y_overfit = np.polyval(coef_overfit, x)

# Generalization: low-degree polynomial
coef_gen = np.polyfit(x, y_noisy, 3)
y_gen = np.polyval(coef_gen, x)

# Extend x for future prediction
x_ext = np.linspace(3, 5, 20)
y_true_ext = 2*np.sin(x_ext) + 0.3*x_ext**2
y_overfit_ext = np.polyval(coef_overfit, x_ext)
y_gen_ext = np.polyval(coef_gen, x_ext)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(x, y_noisy, c="#1976D2", s=25, alpha=0.7, label="训练样本(含噪声)")
ax1.plot(x, y_overfit, "r-", lw=2, label="过拟合模型(15次多项式)")
ax1.plot(x_ext, y_overfit_ext, "r--", lw=2, alpha=0.7)
ax1.plot(x_ext, y_true_ext, "gray", lw=1.5, ls="--", alpha=0.5, label="真实趋势")
ax1.axvline(x=3, color="gray", ls=":", lw=1)
ax1.text(3.05, 10, "→ 未来", fontsize=9, color="gray")
ax1.set_title("过拟合：完美拟合历史,\n但预测完全失控", fontsize=12, fontweight="bold", color="#D32F2F")
ax1.legend(fontsize=8, loc="upper left")
ax1.set_xlabel("时间"); ax1.set_ylabel("收益率")

ax2.scatter(x, y_noisy, c="#1976D2", s=25, alpha=0.7, label="训练样本(含噪声)")
ax2.plot(x, y_gen, "#388E3C", lw=2, label="泛化模型(3次多项式)")
ax2.plot(x_ext, y_gen_ext, "#388E3C", lw=2, ls="--", alpha=0.7)
ax2.plot(x_ext, y_true_ext, "gray", lw=1.5, ls="--", alpha=0.5, label="真实趋势")
ax2.axvline(x=3, color="gray", ls=":", lw=1)
ax2.text(3.05, 10, "→ 未来", fontsize=9, color="gray")
ax2.set_title("泛化：捕捉真实趋势,\n预测接近实际", fontsize=12, fontweight="bold", color="#2E7D32")
ax2.legend(fontsize=8, loc="upper left")
ax2.set_xlabel("时间"); ax2.set_ylabel("收益率")

fig.suptitle("过拟合 vs 泛化：回测中最隐蔽的陷阱", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(D2, "overfitting-vs-generalization.jpg"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ overfitting-vs-generalization.jpg")

# --- Image 4: Trading cost erosion ---
strategies = ["低频价值\n(年换手0.5倍)", "中频动量\n(年换手5倍)", "高频统计\n(年换手30倍)", "超高频\n(年换手100倍)"]
gross_return = [18, 35, 55, 80]
cost_impact = [0.5, 4, 20, 55]
net_return = [g - c for g, c in zip(gross_return, cost_impact)]

x_pos = np.arange(len(strategies))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x_pos - width/2, gross_return, width, label="回测毛收益(%/年)", color="#1976D2", alpha=0.85)
bars2 = ax.bar(x_pos + width/2, net_return, width, label="扣费后净收益(%/年)", color="#388E3C", alpha=0.85)

# Add cost labels in red
for i, (g, c, n) in enumerate(zip(gross_return, cost_impact, net_return)):
    ax.text(i, g+1, f"费用\n-{c}%", ha="center", fontsize=9, color="#D32F2F", fontweight="bold")
    ax.text(i + width/2, n+1, f"{n}%", ha="center", fontsize=10, color="#1B5E20", fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels(strategies, fontsize=10)
ax.set_ylabel("年化收益率 (%)", fontsize=11)
ax.set_title("交易成本对策略收益的侵蚀效应", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.axhline(y=0, color="black", lw=0.5)
ax.set_ylim(-10, 95)
ax.grid(axis="y", alpha=0.3)

# Add annotation
ax.annotate("超高频策略毛收益80%\n扣费后仅剩25%", xy=(3, net_return[3]), xytext=(2.5, 15),
            fontsize=9, ha="center", color="#D32F2F",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.5))

plt.tight_layout()
fig.savefig(os.path.join(D2, "trading-cost-erosion.jpg"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ trading-cost-erosion.jpg")

print("\n✅ All 4 images generated successfully!")
