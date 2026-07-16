#!/usr/bin/env python3
"""
为文章「SABR 随机波动率模型：用四参数把波动率微笑张开」(sabr-stochastic-volatility)
生成真实配图（自洽合成，非占位图）。

模型：Hagan 等(2002) SABR 近似隐含波动率公式
  参数：alpha(ATM vol) / beta(backbone, 0=正态 1=对数正态) / rho(即期-波动相关) / nu(vol-of-vol)
图表：
  1. sabr_smile_rho.png       相关系数 rho 如何把微笑拧成偏斜
  2. sabr_smile_beta.png      backbone beta 如何改变微笑整体形态
  3. sabr_smile_nu.png        波动的波动 nu 如何改变微笑弯曲度
  4. sabr_iv_surface.png      IV 随 moneyness × 期限 的热力面
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

SLUG = "sabr-stochastic-volatility"
BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, SLUG)
os.makedirs(D, exist_ok=True)

F = 100.0          # 远期/即期
T = 1.0            # 期限（年）
ALPHA = 0.20       # ATM 波动率（约 20%）
NU = 0.50          # vol-of-vol
RHO = -0.40        # 即期-波动相关（默认负：跌时波动飙升）


def sabr_iv(F, K, T, alpha, beta, rho, nu):
    """Hagan et al. (2002) SABR 近似隐含波动率（Black 型式）。"""
    if abs(F - K) < 1e-9:  # ATM 极限
        Fmid = F
        z = 0.0
        if abs(beta - 1.0) < 1e-9:
            fk = 1.0
        else:
            fk = Fmid ** (1.0 - beta)
        num = (1 - beta) ** 2 / 24 * (alpha ** 2 / fk ** 2)
        num += 0.25 * rho * beta * nu * alpha / fk
        num += (2 - 3 * rho ** 2) / 24 * nu ** 2
        iv = alpha / fk * (1 + num * T)
        return iv
    Fmid = 0.5 * (F + K)
    if abs(beta - 1.0) < 1e-9:
        fk = 1.0
        logFK = np.log(F / K)
        z = nu / alpha * logFK
    else:
        fk = Fmid ** (1 - beta)
        z = nu / alpha * (F * K) ** ((1 - beta) / 2.0) * np.log(F / K)
    if abs(abs(rho) - 1.0) < 1e-9:
        rho = np.sign(rho) * (1 - 1e-6)
    xz = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
    # 高阶修正项
    num = (1 - beta) ** 2 / 24 * (alpha ** 2 / fk ** 2)
    num += 0.25 * rho * beta * nu * alpha / fk
    num += (2 - 3 * rho ** 2) / 24 * nu ** 2
    iv = alpha / fk * (z / xz) * (1 + num * T)
    return iv


def smile(Ks, beta, rho, nu, alpha=ALPHA, T=T, F=F):
    return np.array([sabr_iv(F, k, T, alpha, beta, rho, nu) for k in Ks])


def main():
    Ks = np.linspace(70, 130, 61)

    # ---------- 图1：rho 把微笑拧成偏斜 ----------
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for rho, c, lab in [(-0.6, "#d62728", "ρ = −0.6（强负：左偏）"),
                        (0.0, "#1f77b4", "ρ = 0（对称微笑）"),
                        (0.6, "#2ca02c", "ρ = 0.6（右偏）")]:
        iv = smile(Ks, beta=0.5, rho=rho, nu=0.5)
        ax.plot(Ks, iv * 100, color=c, lw=2.2, label=lab)
    ax.axvline(F, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_title("SABR 波动率微笑：相关系数 ρ 决定偏斜方向", fontsize=14, fontweight="bold")
    ax.set_xlabel("行权价 K"); ax.set_ylabel("隐含波动率 (%)")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(D, "sabr_smile_rho.png"), dpi=130); plt.close(fig)

    # ---------- 图2：backbone beta 决定整体形态 ----------
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for beta, c, lab in [(0.0, "#9467bd", "β = 0（正态骨架，平缓）"),
                         (0.5, "#ff7f0e", "β = 0.5（混合骨架）"),
                         (1.0, "#1f77b4", "β = 1（对数正态，BS 极限）")]:
        iv = smile(Ks, beta=beta, rho=-0.4, nu=0.5)
        ax.plot(Ks, iv * 100, color=c, lw=2.2, label=lab)
    ax.axvline(F, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_title("SABR 微笑骨架：backbone β 改变整条曲线形态", fontsize=14, fontweight="bold")
    ax.set_xlabel("行权价 K"); ax.set_ylabel("隐含波动率 (%)")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(D, "sabr_smile_beta.png"), dpi=130); plt.close(fig)

    # ---------- 图3：nu 改变弯曲度 ----------
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for nu, c, lab in [(0.2, "#1f77b4", "ν = 0.2（波动平静）"),
                       (0.5, "#ff7f0e", "ν = 0.5"),
                       (0.9, "#d62728", "ν = 0.9（波动剧烈，弯度大）")]:
        iv = smile(Ks, beta=0.5, rho=-0.4, nu=nu)
        ax.plot(Ks, iv * 100, color=c, lw=2.2, label=lab)
    ax.axvline(F, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_title("SABR 微笑曲率：vol-of-vol ν 决定弯曲幅度", fontsize=14, fontweight="bold")
    ax.set_xlabel("行权价 K"); ax.set_ylabel("隐含波动率 (%)")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(D, "sabr_smile_nu.png"), dpi=130); plt.close(fig)

    # ---------- 图4：IV 随 moneyness × 期限 的热力面 ----------
    moneys = np.linspace(0.7, 1.3, 40)   # K/F
    mats = np.array([0.25, 0.5, 1.0, 2.0, 3.0])
    grid = np.zeros((len(mats), len(moneys)))
    for i, Tv in enumerate(mats):
        for j, m in enumerate(moneys):
            grid[i, j] = sabr_iv(F, F * m, Tv, ALPHA, 0.5, RHO, NU) * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.contourf(moneys, mats, grid, levels=30, cmap="viridis")
    ax.set_title("SABR 隐含波动率面：moneyness × 期限（β=0.5, ρ=−0.4, ν=0.5）",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("moneyness  K / F"); ax.set_ylabel("期限 T（年）")
    ax.axvline(1.0, color="white", ls="--", lw=1, alpha=0.8)
    cb = fig.colorbar(im, ax=ax); cb.set_label("隐含波动率 (%)")
    fig.tight_layout(); fig.savefig(os.path.join(D, "sabr_iv_surface.png"), dpi=130); plt.close(fig)

    print("sabr-stochastic-volatility 配图已生成：", sorted(os.listdir(D)))


if __name__ == "__main__":
    main()
