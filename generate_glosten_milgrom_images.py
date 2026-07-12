#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「做市商库存风险与 Glosten-Milgrom 模型：报价如何随信息调整」
(glosten-milgrom-mm) 生成真实配图与真实统计数字。

核心主题：Glosten-Milgrom(1985) 序贯交易模型。
  - 资产真实价值 V ∈ {V_L, V_H}，先验 π=P(V=V_H)。
  - 两类交易者：知情者(占比 μ，知道 V，永远朝赚钱方向交易) +
                不知情/流动性者(占比 1-μ，买卖各半)。
  - 做市商只看到"买/卖"方向，用贝叶斯更新对 V 的信念，再报：
        Ask = E[V | 来了一笔买单]   Bid = E[V | 来了一笔卖单]
    => 买单价高(被知情买盘抬)、卖单价低(被知情卖盘压) => 买卖价差=逆向选择成本。
  - 库存风险扩展(Avellaneda-Stoikov 风)：风险厌恶的做市商把"保留价格"按库存 I 下移
        r(I) = E[V] - γ·I
    报价以 r(I) 为中心、再叠加逆向选择半价差；库存越大，报价整体下移(不愿再加仓)，
    且额外加宽价差 2κ|I|(为承担库存风险收费)。

所有图表与数字均由文中逻辑真实计算生成：
  1) gm_quote_revision.png —— 单笔买/卖后，买/卖/中间价如何移动(信息驱动漂移)
  2) gm_simulation.png     —— 一段交易序列：中间价收敛到真实 V，库存随之累积
  3) gm_inventory_risk.png —— 库存风险：价差随 |库存| 加宽、中间价随库存下移

参数(文中固定)：V_H=110, V_L=90, 先验 π=0.5, 知情占比 μ=0.4, 不知情买卖各半 δ=0.5,
                库存水平系数 γ=0.05, 库存加宽系数 κ=0.01。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 字体 / 配色 ----------
rcParams = matplotlib.rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "glosten-milgrom-mm")
os.makedirs(D, exist_ok=True)

C = {"eq": "#2F4B7C", "up": "#55A868", "dn": "#C44E52", "grid": "#E2E2E2",
     "buy": "#C44E52", "sell": "#4C72B0", "mid": "#2F4B7C", "inv": "#DD8452",
     "true": "#55A868", "info": "#C44E52", "risk": "#DD8452", "ink": "#2b2b2b"}

# =====================================================================
# 1) Glosten-Milgrom 贝叶斯报价 + 库存风险扩展
# =====================================================================
def make_model(V_H=110, V_L=90, mu=0.4, delta=0.5, gamma=0.05, kappa=0.01):
    dV = V_H - V_L
    def quotes(pi, I=0.0):
        # 知情者：V=V_H 必买，V=V_L 必卖
        a = mu * 1.0 + (1 - mu) * delta          # P(buy | V=V_H)
        b = (1 - mu) * delta                      # P(buy | V=V_L)
        c = (1 - mu) * (1 - delta)                # P(sell| V=V_H)
        d = mu * 1.0 + (1 - mu) * (1 - delta)     # P(sell| V=V_L)
        p_buy = a * pi + b * (1 - pi)
        p_sell = c * pi + d * (1 - pi)
        pi_buy = a * pi / p_buy if p_buy > 0 else pi      # P(V=V_H | buy)
        pi_sell = c * pi / p_sell if p_sell > 0 else pi   # P(V=V_H | sell)
        ask_gm = pi_buy * V_H + (1 - pi_buy) * V_L
        bid_gm = pi_sell * V_H + (1 - pi_sell) * V_L
        nu = pi * V_H + (1 - pi) * V_L                      # 后验均值
        s_info = ask_gm - bid_gm                            # 逆向选择价差
        # 库存风险：保留价格 r(I)=nu-γI；以 r(I) 为中心叠加半价差，并随 |I| 加宽
        reservation = nu - gamma * I
        spread = s_info + 2 * kappa * abs(I)
        half = spread / 2.0
        ask = reservation + half
        bid = reservation - half
        return dict(ask=ask, bid=bid, mid=(ask + bid) / 2, nu=nu,
                    ask_gm=ask_gm, bid_gm=bid_gm, spread=spread,
                    s_info=s_info, pi_buy=pi_buy, pi_sell=pi_sell)
    return quotes

# =====================================================================
# 2) 初始数字 + 单笔买/卖后的报价移动
# =====================================================================
Q = make_model()
pi0 = 0.5
q0 = Q(pi0, I=0.0)
print(f"[初始] 先验 π={pi0}   Ask={q0['ask_gm']:.2f}  Bid={q0['bid_gm']:.2f}  "
      f"价差={q0['s_info']:.2f}  中间价={q0['nu']:.2f}")
q_after_buy = Q(q0["pi_buy"], I=0.0)
q_after_sell = Q(q0["pi_sell"], I=0.0)
print(f"[来一笔买] 后验 π→{q0['pi_buy']:.3f}  Ask→{q_after_buy['ask_gm']:.2f}  "
      f"(中间价上移到 {q_after_buy['nu']:.2f})")
print(f"[来一笔卖] 后验 π→{q0['pi_sell']:.3f}  Bid→{q_after_sell['bid_gm']:.2f}  "
      f"(中间价下移到 {q_after_sell['nu']:.2f})")
# 连续两笔买 / 两笔卖的收敛
q2b = Q(q_after_buy["pi_buy"], I=0.0)
q2s = Q(q_after_sell["pi_sell"], I=0.0)
print(f"[连续两笔买] 后验 π→{q2b['pi_buy']:.3f}  中间价→{q2b['nu']:.2f}")
print(f"[连续两笔卖] 后验 π→{q2s['pi_sell']:.3f}  中间价→{q2s['nu']:.2f}")

# =====================================================================
# 3) 交易序列模拟：固定真实价值 V=V_H，看中间价如何收敛 + 库存累积
# =====================================================================
def simulate(V_true, n_trades=400, pi0=0.5, mu=0.4, delta=0.5, gamma=0.05,
             kappa=0.01, seed=20260712):
    rng = np.random.default_rng(seed)
    Q = make_model(mu=mu, delta=delta, gamma=gamma, kappa=kappa)
    pi = pi0
    I = 0.0
    mids, invs, pis = [Q(pi, I)["mid"]], [I], [pi]
    for _ in range(n_trades):
        # 交易者类型：知情(知道 V_true) 或 不知情
        if rng.random() < mu:
            side = "buy" if V_true == 110 else "sell"   # 知情永远朝赚钱方向
        else:
            side = "buy" if rng.random() < delta else "sell"
        q = Q(pi, I)
        if side == "buy":
            pi = q["pi_buy"]; I -= 1.0          # 做市商卖出 => 库存 -1
        else:
            pi = q["pi_sell"]; I += 1.0         # 做市商买入 => 库存 +1
        mids.append(q["mid"]); invs.append(I); pis.append(pi)
    return np.array(mids), np.array(invs), np.array(pis)

mids, invs, pis = simulate(110, n_trades=400)
nu_end = pis[-1] * 110 + (1 - pis[-1]) * 90
print(f"\n[模拟 V=110, 400 笔] 末后验 π={pis[-1]:.3f}  末后验均值 ν={nu_end:.2f}  "
      f"(真实 V=110，信息中间价收敛到真实价值)")
print(f"  库存：均值={invs.mean():.1f}  标准差={invs.std():.1f}  "
      f"min={invs.min():.0f}  max={invs.max():.0f}")
print(f"  库存风险效应(γ=0.05)：库存 +20 时中间价下移 {0.05*20:.2f} 点；"
      f"|I|=20 时价差额外加宽 {2*0.01*20:.2f} 点")

# =====================================================================
# 图1：单笔买/卖后的报价移动(信息驱动漂移)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
for ax, label, after, col in [
        (axes[0], "来一笔【买】单 → 后验上移", q_after_buy, C["buy"]),
        (axes[1], "来一笔【卖】单 → 后验下移", q_after_sell, C["sell"])]:
    ax.axhspan(q0["bid_gm"], q0["ask_gm"], color=C["grid"], alpha=0.5,
               label="初始买卖区间")
    ax.plot([0, 1], [q0["ask_gm"], q0["ask_gm"]], "s--", color=C["buy"], label="初始 Ask")
    ax.plot([0, 1], [q0["bid_gm"], q0["bid_gm"]], "s--", color=C["sell"], label="初始 Bid")
    ax.plot([1, 2], [after["ask_gm"], after["ask_gm"]], "o-", color=C["buy"],
            lw=2, label="更新后 Ask")
    ax.plot([1, 2], [after["bid_gm"], after["bid_gm"]], "o-", color=C["sell"],
            lw=2, label="更新后 Bid")
    ax.plot([0, 2], [q0["nu"], after["nu"]], "k:", lw=1.5, label="中间价(后验均值)")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["之前", "事件", "之后"])
    ax.set_ylabel("价格")
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
fig.suptitle("Glosten-Milgrom：买单价整体抬高、卖单价整体压低，价差=逆向选择成本",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(D, "gm_quote_revision.png"), dpi=130)
plt.close()

# =====================================================================
# 图2：模拟 —— 中间价收敛到真实 V + 库存累积(双轴)
# =====================================================================
fig, ax1 = plt.subplots(figsize=(9.0, 4.6))
nu = pis * 110 + (1 - pis) * 90          # 后验均值 = 纯信息中间价
ax1.plot(nu, color=C["mid"], lw=1.8, label="做市商信息中间价 ν=后验均值")
ax1.axhline(110, color=C["true"], ls="--", lw=1.5, label="真实价值 V=110")
ax1.set_xlabel("交易笔数")
ax1.set_ylabel("信息中间价 ν", color=C["mid"])
ax1.tick_params(axis="y", labelcolor=C["mid"])
ax1.legend(loc="upper left", fontsize=9)
ax2 = ax1.twinx()
ax2.plot(invs, color=C["inv"], lw=1.2, alpha=0.8, label="库存 I")
ax2.axhline(0, color=C["ink"], lw=0.8)
ax2.set_ylabel("库存(正=多/负=空)", color=C["inv"])
ax2.tick_params(axis="y", labelcolor=C["inv"])
ax2.legend(loc="lower right", fontsize=9)
ax1.set_title("序贯交易：信息中间价随知情买盘收敛到真实价值，做市商被'逆向选择'成净空头",
              fontsize=10.5)
fig.tight_layout()
plt.savefig(os.path.join(D, "gm_simulation.png"), dpi=130)
plt.close()

# =====================================================================
# 图3：库存风险 —— 价差随 |库存| 加宽、中间价随库存下移
# =====================================================================
Is = np.linspace(-40, 40, 81)
spreads, mids_inv = [], []
for I in Is:
    q = Q(0.5, I=I)
    spreads.append(q["spread"]); mids_inv.append(q["mid"])
fig, ax1 = plt.subplots(figsize=(8.6, 4.4))
ax1.plot(Is, spreads, color=C["risk"], lw=2, label="总价差 = 逆向选择 + 2κ|I|")
ax1.axhline(q0["s_info"], color=C["info"], ls="--", lw=1,
            label=f"纯逆向选择价差 {q0['s_info']:.1f}")
ax1.set_xlabel("库存 I (正=多仓 / 负=空仓)")
ax1.set_ylabel("价差", color=C["risk"])
ax1.tick_params(axis="y", labelcolor=C["risk"])
ax1.legend(loc="upper center", fontsize=9)
ax2 = ax1.twinx()
ax2.plot(Is, mids_inv, color=C["mid"], lw=1.6, ls=":", label="中间价 = ν − γI")
ax2.set_ylabel("中间价(随库存下移)", color=C["mid"])
ax2.tick_params(axis="y", labelcolor=C["mid"])
ax2.legend(loc="lower right", fontsize=9)
ax1.set_title("库存风险：库存越大做市商加宽价差收费，并把报价整体下移(不愿再加仓)",
              fontsize=10.5)
fig.tight_layout()
plt.savefig(os.path.join(D, "gm_inventory_risk.png"), dpi=130)
plt.close()

print("\nDONE ->", D)
for f in sorted(os.listdir(D)):
    print("  ", f, os.path.getsize(os.path.join(D, f)), "bytes")
