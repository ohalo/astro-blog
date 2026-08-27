# -*- coding: utf-8 -*-
"""生成「评级迁移矩阵建模」一文的四张真实计算图 + 打印关键数值。
模型：8 状态（AAA/AA/A/BBB/BB/B/CCC/D）年度转移矩阵，用蒙特卡洛
模拟一个信用组合在 1/3/5 年 horizon 的多期损失分布。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

_CJK = [
    "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/53fe5be564086fefc7523ccd0a31200acf92e0e5.asset/AssetData/STHEITI.ttf",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/5feac9245cca79adaf638ded7a4994b1ddb33ca0.asset/AssetData/Hei.ttf",
]
for _p in _CJK:
    try:
        _fm.fontManager.addfont(_p)
    except Exception:
        pass
plt.rcParams["font.family"] = ["STHeiti", "Heiti TC", "Hei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(20260827)

states = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
n = len(states)
D = states.index("D")

# 原始年度转移百分比（行=期初评级，列=期末评级），代码里会归一化到和为1
raw = np.array([
    [89.0, 8.5, 1.4, 0.6, 0.3, 0.2, 0.0, 0.0],   # AAA
    [2.0, 89.0, 7.5, 0.9, 0.3, 0.2, 0.0, 0.1],   # AA
    [0.3, 3.5, 88.0, 6.5, 1.0, 0.5, 0.1, 0.1],   # A
    [0.1, 0.6, 4.5, 86.0, 6.5, 1.8, 0.3, 0.2],   # BBB
    [0.1, 0.3, 1.0, 6.0, 80.0, 9.5, 1.5, 1.6],   # BB
    [0.1, 0.2, 0.6, 1.2, 5.5, 81.0, 7.5, 3.9],   # B
    [0.2, 0.3, 0.6, 1.5, 3.0, 12.0, 56.0, 26.4], # CCC
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],  # D (吸收态)
], dtype=float)
P = raw / raw.sum(axis=1, keepdims=True)

# 各评级对应的点差（bp），D 用违约回收后的隐含点差近似
spread = np.array([30.0, 60.0, 90.0, 140.0, 300.0, 500.0, 900.0, 1500.0])
LGD = 0.60          # 违约损失率
duration = 4.0      # 组合平均久期（年）

# 初始组合：偏投资级（A/BBB 为主，少量 BB/B）
init_mix = np.array([0.05, 0.15, 0.30, 0.35, 0.10, 0.05, 0.0, 0.0])
init_mix = init_mix / init_mix.sum()

P_cum = np.cumsum(P, axis=1)

def step_states(cur):
    """向量化单年转移：输入一维评级数组，返回转移后的一维数组。"""
    out = np.empty_like(cur)
    for k in range(n):
        idx = np.where(cur == k)[0]
        if idx.size == 0:
            continue
        u = rng.random(idx.size)
        out[idx] = np.searchsorted(P_cum[k], u, side="right")
        out[idx] = np.clip(out[idx], 0, n - 1)
    return out

def simulate(n_paths, n_obligors, horizon):
    """返回每条路径的组合损失百分比（数组形状 n_paths）。"""
    init_states = rng.choice(n, size=(n_paths, n_obligors), p=init_mix).ravel()
    s = init_states.copy()
    for _ in range(horizon):
        s = step_states(s)
    sp_init = spread[init_states].reshape(n_paths, n_obligors)
    sp_final = spread[s].reshape(n_paths, n_obligors)
    mtm = duration * (sp_final - sp_init) / 10000.0
    defaulted = (s == D).reshape(n_paths, n_obligors).astype(float) * LGD
    per = mtm + defaulted
    return per.mean(axis=1) * 100.0   # 等权组合损失%

def default_rate(horizon):
    init_states = rng.choice(n, size=(5000, 1000), p=init_mix).ravel()
    s = init_states.copy()
    for _ in range(horizon):
        s = step_states(s)
    return (s == D).mean()

horizons = [1, 3, 5]
losses = {h: simulate(5000, 1000, h) for h in horizons}

def stats(x):
    x = np.sort(x)
    el = x.mean()
    var95 = np.percentile(x, 95)
    var99 = np.percentile(x, 99)
    es99 = x[x >= var99].mean()
    default_rate = np.mean(x)  # placeholder; 真正违约率在下方单独算
    return el, var95, var99, es99

el1, v95_1, v99_1, es1 = stats(losses[1])
el3, v95_3, v99_3, es3 = stats(losses[3])
el5, v95_5, v99_5, es5 = stats(losses[5])

# 各 horizon 累计违约率（最终状态为 D 的比例），复用上面向量化后的 default_rate
dr = {h: default_rate(h) for h in horizons}

# ---------------- 图1：转移矩阵热力图 ----------------
fig, ax = plt.subplots(figsize=(8.5, 6))
mat_pct = P * 100
im = ax.imshow(mat_pct, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(n)); ax.set_xticklabels(states)
ax.set_yticks(range(n)); ax.set_yticklabels(states)
ax.set_xlabel("期末评级")
ax.set_ylabel("期初评级")
ax.set_title("年度评级转移概率矩阵（%）")
for i in range(n):
    for j in range(n):
        v = mat_pct[i, j]
        if v >= 0.05:
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="black" if v < 55 else "white", fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="概率 %")
fig.tight_layout()
fig.savefig("public/images/rating-migration-matrix-modeling/rm_matrix_heatmap.png", dpi=140)
plt.close(fig)

# ---------------- 图2：评级分布随时间漂移 ----------------
years = np.arange(0, 6)
dist = np.zeros((len(years), n))
dist[0] = init_mix
cur = np.tile(init_mix, (5000, 1)).copy()
# 用大样本确定性演化（P^T 推进），避免噪声
det = init_mix.copy()
det_series = [det.copy()]
for y in range(1, 6):
    det = det @ P
    det_series.append(det.copy())
dist = np.array(det_series)

fig, ax = plt.subplots(figsize=(9, 5))
bottom = np.zeros(len(years))
colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#d62728",
          "#8c564b", "#7f7f7f", "#000000"]
for j in range(n):
    ax.fill_between(years, bottom, bottom + dist[:, j],
                    label=states[j], color=colors[j], alpha=0.85)
    bottom += dist[:, j]
ax.set_xlabel("年")
ax.set_ylabel("组合占比")
ax.set_title("组合评级分布随时间的漂移（确定性的 P^T 推进）")
ax.set_xticks(years)
ax.set_xlim(0, 5)
ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13))
fig.tight_layout()
fig.savefig("public/images/rating-migration-matrix-modeling/rm_drift.png", dpi=140)
plt.close(fig)

# ---------------- 图3：多期损失分布直方图 ----------------
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(-2, 14, 50)
ax.hist(losses[1], bins=bins, alpha=0.55, density=True, label="1 年", color="#1f77b4")
ax.hist(losses[3], bins=bins, alpha=0.55, density=True, label="3 年", color="#ff7f0e")
ax.hist(losses[5], bins=bins, alpha=0.55, density=True, label="5 年", color="#d62728")
for h, c, v99 in [(1, "#1f77b4", v99_1), (3, "#ff7f0e", v99_3), (5, "#d62728", v99_5)]:
    ax.axvline(v99, color=c, ls="--", lw=1.6)
ax.set_xlabel("组合损失（% 面值）")
ax.set_ylabel("概率密度")
ax.set_title("组合多期损失分布：尾部随 horizon 非线性变厚")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("public/images/rating-migration-matrix-modeling/rm_loss_dist.png", dpi=140)
plt.close(fig)

# ---------------- 图4：期望损失 vs 99% VaR 随 horizon ----------------
fig, ax1 = plt.subplots(figsize=(9, 5))
hs = np.array(horizons)
ax1.plot(hs, [el1, el3, el5], "o-", color="#2ca02c", lw=2.2, label="期望损失 EL")
ax1.plot(hs, [v99_1, v99_3, v99_5], "s-", color="#d62728", lw=2.2, label="99% VaR")
ax1.plot(hs, [es1, es3, es5], "^--", color="#7f7f7f", lw=2.0, label="99% ES（预期短缺）")
# 1年损失的线性外推（虚线），用来对照真实的超线性程度
ax1.plot(hs, [el1, el1*3, el1*5], ":", color="#2ca02c", lw=1.2, alpha=0.6,
         label="EL 线性外推（×1/×3/×5）")
ax1.set_xlabel("投资期限（年）")
ax1.set_ylabel("组合损失（% 面值）")
ax1.set_title("多期损失：组合层近似线性，真正的凸性藏在累计违约率里")
ax1.set_xticks(hs)
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(alpha=0.3)

# 右轴：累计违约率（真实超线性来源）+ 1年线性外推对照
ax2 = ax1.twinx()
dr_pct = [dr[h]*100 for h in horizons]
ax2.plot(hs, dr_pct, "D-", color="#1f77b4", lw=2.4, label="累计违约率")
ax2.plot(hs, [dr[1]*100, dr[1]*300, dr[1]*500], "--", color="#1f77b4",
         lw=1.2, alpha=0.6, label="违约率线性外推")
ax2.set_ylabel("累计违约率（%）", color="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#1f77b4")
ax2.legend(fontsize=8, loc="center right")
fig.tight_layout()
fig.savefig("public/images/rating-migration-matrix-modeling/rm_tail_horizon.png", dpi=140)
plt.close(fig)

# ---------------- 打印关键数值 ----------------
print("=== 评级迁移矩阵 关键数值 ===")
print(f"初始组合占比: A={init_mix[2]*100:.0f}% BBB={init_mix[3]*100:.0f}% "
      f"BB={init_mix[4]*100:.0f}% B={init_mix[5]*100:.0f}%")
print(f"1年: EL={el1:.2f}%  VaR95={v95_1:.2f}%  VaR99={v99_1:.2f}%  ES99={es1:.2f}%  违约率={dr[1]*100:.2f}%")
print(f"3年: EL={el3:.2f}%  VaR95={v95_3:.2f}%  VaR99={v99_3:.2f}%  ES99={es3:.2f}%  违约率={dr[3]*100:.2f}%")
print(f"5年: EL={el5:.2f}%  VaR95={v95_5:.2f}%  VaR99={v99_5:.2f}%  ES99={es5:.2f}%  违约率={dr[5]*100:.2f}%")
P5 = np.linalg.matrix_power(P, 5)   # 5 年转移矩阵，用于单名对照
# 自洽性检验：组合累计违约率 = 各单名 5y->D 概率的持仓加权平均（独立假设下精确成立）
weighted_default = float(init_mix @ P5[:, D]) * 100
print(f"VaR99 组合层凸性: 1y->{v99_1:.2f}  3y->{v99_3:.2f}  5y->{v99_5:.2f}  "
      f"(1年线性外推5y应为{v99_1*5:.2f}) -> 真实{v99_5:.2f} 略低于线性，因分散化平滑")
print(f"累计违约率凸性: 1y->{dr[1]*100:.2f}%  3y->{dr[3]*100:.2f}%  5y->{dr[5]*100:.2f}%  "
      f"(1年线性外推5y应为{dr[1]*500:.2f}%) -> 真实{dr[5]*100:.2f}% 高于线性，因边际违约率逐年上升")
print(f"违约率自洽: 模拟={dr[5]*100:.2f}% vs 加权单名={weighted_default:.2f}% (应一致)")
print(f"BBB 单名 5y->D={P5[3,D]*100:.2f}%；组合含 BB/B 倾斜，故整体违约率 {dr[5]*100:.2f}% 高于纯 BBB")
print(f"5年累计违约率: {dr[5]*100:.2f}%  其中 1y 仅 {dr[1]*100:.2f}%")
# 5年矩阵首行（AAA 5年转移）
P5 = np.linalg.matrix_power(P, 5)
print(f"AAA 5年后仍 AAA/AA: {P5[0,0]*100:.1f}% / {P5[0,1]*100:.1f}%  | 落入 D: {P5[0,D]*100:.2f}%")
print(f"BBB 5年后落入 HY(BB及以下): {(P5[3,4:7].sum())*100:.1f}%  | 落入 D: {P5[3,D]*100:.2f}%")
