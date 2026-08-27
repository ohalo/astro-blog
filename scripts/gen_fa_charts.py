# -*- coding: utf-8 -*-
"""生成「堕落天使的强制抛售」一文的四张真实计算图 + 打印关键数值。
模型：一只从投资级（IG）被下调到高收益（HY）的「堕落天使」债券。
重点把价格/利差下跌拆成两块：
  (1) fundamental：评级下调反映的真实信用恶化（OAS 从 IG 跳到 HY 合理水平）；
  (2) mechanical：指数基金/受限账户在窗口期被迫抛售造成的机械性超额利差。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

# 注册中文 CJK 字体，避免标签渲染成方块
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

# ---------------- 参数（可复现） ----------------
N = 60                      # 事件前后共 60 个交易日
t_dg = 30                   # 第 30 天发生评级下调
duration = 7.0              # 债券久期（年）
OAS_pre = 130.0             # 下调前 IG 利差（bp）
OAS_fund_post = 380.0       # 下调后"真实"信用状态对应的 HY 合理利差（bp）
window = 5                  # 强制抛售窗口（交易日）
forced_frac = 0.35          # 必须卖出的持仓占流通盘比例
depth_stress = 0.18         # 压力期每日可吸收流动性（占流通盘比例）
lambda_m = 60.0             # 机械冲击系数：每单位 (抛压/深度) 产生的超额利差 bp
rebound = 0.70              # 窗口结束后机械折价的均值回复比例

daily_forced = forced_frac / window   # 窗口内每日抛压（占流通盘比例）

def mech_impact_bps(sold_frac, depth_frac):
    """Kyle 式冲击：超额利差与 抛压/深度 成正比。"""
    if depth_frac <= 0:
        return 0.0
    return lambda_m * (sold_frac / depth_frac)

# ---------------- 构造 OAS 路径 ----------------
oas = np.zeros(N)
oas_fund = np.zeros(N)      # 仅含基本面的"公允"OAS
oas[0] = OAS_pre
oas_fund[0] = OAS_pre

for t in range(1, t_dg):
    oas[t] = oas[t-1] + rng.normal(0, 3.0)
    oas_fund[t] = oas[t]

# 下调当日：基本面跳到 HY 合理水平
oas[t_dg] = OAS_fund_post
oas_fund[t_dg] = OAS_fund_post

# 窗口期：叠加机械性超额利差
cum_mech = 0.0
daily_mech = []
for t in range(t_dg + 1, t_dg + 1 + window):
    oas_fund[t] = oas_fund[t-1] + rng.normal(0, 2.0)
    imp = mech_impact_bps(daily_forced, depth_stress)
    cum_mech += imp
    daily_mech.append(imp)
    oas[t] = oas_fund[t] + cum_mech

# 窗口后：机械折价的均值回复
post_start = t_dg + 1 + window
for i, t in enumerate(range(post_start, N)):
    oas_fund[t] = oas_fund[t-1] + rng.normal(0, 2.0)
    progress = (i + 1) / (N - post_start)
    oas[t] = oas_fund[t] + cum_mech * (1 - rebound * progress)

peak_mech = cum_mech                          # 峰值机械超额利差
resid_mech = cum_mech * (1 - rebound)         # 永久残留
price_hit = peak_mech * duration / 10000.0 * 100.0   # 机械造成的额外价格跌幅（点）

# ---------------- 图1：价格路径（含基本面公允 vs 实际指数价） ----------------
days = np.arange(N)
price = 100.0 - oas * duration / 10000.0 * 100.0
price_fund = 100.0 - oas_fund * duration / 10000.0 * 100.0

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(days, price, color="#1f77b4", lw=2.2, label="实际指数价（基本面+机械抛压）")
ax.plot(days, price_fund, color="#888888", lw=1.8, ls="--", label="基本面公允价（仅反映真实信用恶化）")
ax.axvline(t_dg, color="#d62728", lw=1.5, ls=":", label=f"评级下调（第{t_dg}日）")
ax.fill_between(days, price, price_fund, where=(price < price_fund),
                color="#d62728", alpha=0.18, label="机械性折价的超额部分")
ax.set_xlabel("交易日")
ax.set_ylabel("债券净价（元）")
ax.set_title("堕落天使：实际价格跌破基本面公允价，缺口=机械性抛压")
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("public/images/fallen-angel-forced-selling/fa_price_path.png", dpi=140)
plt.close(fig)

# ---------------- 图2：窗口期每日强制抛压 ----------------
wdays = np.arange(1, window + 1)
forced_pct = [daily_forced * 100] * window
cum_pct = np.cumsum(forced_pct)

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.bar(wdays, forced_pct, color="#1f77b4", alpha=0.75, label="每日强制抛压（占流通盘%）")
ax1.set_xlabel(f"强制抛售窗口（下调后第 1~{window} 日）")
ax1.set_ylabel("当日抛压占流通盘比例（%）", color="#1f77b4")
ax1.set_xticks(wdays)
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax2 = ax1.twinx()
ax2.plot(wdays, cum_pct, color="#d62728", lw=2.2, marker="o", label="累计强制卖出（%）")
ax2.set_ylabel("累计占流通盘比例（%）", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")
ax1.set_title(f"评级触发后的机械性抛压：{int(forced_frac*100)}% 流通盘在 {window} 日内被迫清仓")
ax1.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("public/images/fallen-angel-forced-selling/fa_forced_volume.png", dpi=140)
plt.close(fig)

# ---------------- 图3：机械 vs 基本面 利差分解 ----------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(days, oas, color="#1f77b4", lw=2.2, label="实际观测 OAS")
ax.plot(days, oas_fund, color="#888888", lw=1.8, ls="--", label="基本面公允 OAS")
ax.axvline(t_dg, color="#d62728", lw=1.5, ls=":")
ax.fill_between(days, oas_fund, oas, where=(oas > oas_fund),
                color="#ff7f0e", alpha=0.25, label="机械性超额利差（非基本面）")
ax.annotate(f"峰值机械超额 ≈ {peak_mech:.0f} bp",
            xy=(t_dg + window, OAS_fund_post + peak_mech),
            xytext=(t_dg - 18, OAS_fund_post + peak_mech + 40),
            fontsize=9, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728"))
ax.set_xlabel("交易日")
ax.set_ylabel("期权调整利差 OAS（bp）")
ax.set_title("把利差下跌拆开：多少是真实信用、多少是被迫卖出？")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("public/images/fallen-angel-forced-selling/fa_oas_decomp.png", dpi=140)
plt.close(fig)

# ---------------- 图4：流动性深度敏感性（压力期深度 vs 峰值机械利差） ----------------
depths = np.linspace(0.05, 0.60, 40)
peak_by_depth = [window * mech_impact_bps(daily_forced, d) for d in depths]
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(depths * 100, peak_by_depth, color="#2ca02c", lw=2.4)
ax.axvline(depth_stress * 100, color="#d62728", ls="--", lw=1.5,
           label=f"基准压力深度 = {depth_stress*100:.0f}%")
ax.axhline(peak_mech, color="#1f77b4", ls=":", lw=1.3, label=f"基准峰值 ≈ {peak_mech:.0f} bp")
ax.scatter([depth_stress * 100], [peak_mech], color="#d62728", zorder=5, s=50)
ax.set_xlabel("压力期每日可吸收流动性（占流通盘 %）")
ax.set_ylabel("峰值机械性超额利差（bp）")
ax.set_title("深度越薄，机械冲击越非线性：流动性是坠落的引擎")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("public/images/fallen-angel-forced-selling/fa_depth_sensitivity.png", dpi=140)
plt.close(fig)

# ---------------- 打印关键数值（供正文引用） ----------------
print("=== 堕落天使 关键数值 ===")
print(f"下调前 OAS (IG)          : {OAS_pre:.0f} bp")
print(f"下调后基本面 OAS (HY)    : {OAS_fund_post:.0f} bp  (真实信用恶化 +{OAS_fund_post-OAS_pre:.0f} bp)")
print(f"窗口期每日抛压           : {daily_forced*100:.1f}% 流通盘")
print(f"峰值机械性超额利差       : {peak_mech:.1f} bp")
print(f"机械造成的额外价格跌幅   : {price_hit:.2f} 元")
print(f"窗口结束后永久残留       : {resid_mech:.1f} bp")
print(f"实际观测峰值 OAS         : {oas.max():.0f} bp  (基本面 {OAS_fund_post:.0f} + 机械 {peak_mech:.0f})")
print(f"深度=5% 时峰值机械利差   : {window*mech_impact_bps(daily_forced,0.05):.0f} bp")
print(f"深度=60% 时峰值机械利差  : {window*mech_impact_bps(daily_forced,0.60):.0f} bp")
