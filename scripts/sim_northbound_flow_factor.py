"""北向资金流因子模拟：陆股通资金能否预测 A 股收益"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

import os
OUT = "/Users/halo/workspace/astro-blog/public/images/northbound-flow-factor"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(2026)

# ---------- 面板模拟：1500 天 x 300 股 ----------
# 北向资金持股偏好大市值/白马；资金流包含 信息成分 + 趋势追随成分 + 噪声
n_days, n_stocks = 1500, 300
mu_info = 0.0004          # 信息成分对未来收益的日度传导

# 股票特征
size = rng.lognormal(0, 1, n_stocks)                # 市值
nb_coverage = (size / size.max()) ** 0.5            # 北向覆盖度：大票覆盖高
beta = rng.normal(1.0, 0.25, n_stocks)

# 市场收益
mkt = rng.normal(0.0003, 0.012, n_days)

# 信息信号：北向部分知情（对未来 5 日收益有微弱预测）
info = rng.normal(0, 1, (n_days, n_stocks))
idio_future = np.zeros((n_days, n_stocks))
horizon = 5
for h in range(1, horizon + 1):
    rolled = np.roll(info, h, axis=0)
    rolled[:h] = 0
    idio_future += rolled * (mu_info * (horizon - h + 1) / horizon)
idio = idio_future + rng.normal(0, 0.018, (n_days, n_stocks))
ret = beta[None, :] * mkt[:, None] + idio

# 北向净流入：信息成分 + 动量追随（过去5日收益）+ 噪声，再乘覆盖度
mom5 = np.zeros((n_days, n_stocks))
for h in range(1, 6):
    r_ = np.roll(ret, h, axis=0); r_[:h] = 0
    mom5 += r_
flow = (0.5 * info + 0.35 * (mom5 / mom5.std()) + rng.normal(0, 1, (n_days, n_stocks)))
flow *= nb_coverage[None, :]

# 标准化因子：过去20日累计净流入 / 过去20日标准差（避免 look-ahead：只用 t-1 及以前）
window = 20
flow_cum = np.zeros((n_days, n_stocks))
for t in range(window, n_days):
    seg = flow[t - window:t]        # 不含 t 当天 -> 信号在 t-1 收盘可得
    flow_cum[t] = seg.sum(axis=0) / (seg.std(axis=0) + 1e-9)

# ---------- IC 分析 ----------
warm = window + 10
fwd1 = np.roll(ret, -1, axis=0)
fwd5 = np.zeros_like(ret)
for h in range(1, 6):
    fwd5 += np.roll(ret, -h, axis=0)

def daily_ic(sig, fwd, t0, t1):
    ics = []
    for t in range(t0, t1):
        s, f = sig[t], fwd[t]
        rs = np.argsort(np.argsort(s)); rf = np.argsort(np.argsort(f))
        ics.append(np.corrcoef(rs, rf)[0, 1])
    return np.array(ics)

ic1 = daily_ic(flow_cum, fwd1, warm, n_days - 6)
ic5 = daily_ic(flow_cum, fwd5, warm, n_days - 6)
print(f"IC(1d): mean={ic1.mean():.4f}, IR={ic1.mean()/ic1.std()*np.sqrt(252):.2f}")
print(f"IC(5d): mean={ic5.mean():.4f}, IR={ic5.mean()/ic5.std()*np.sqrt(252):.2f}")

# IC 衰减
decay = []
for h in range(1, 21):
    fh = np.roll(ret, -h, axis=0)
    ics = daily_ic(flow_cum, fh, warm, n_days - 21)
    decay.append(ics.mean())

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.bar(range(1, 21), np.array(decay) * 100, color="#1f77b4")
ax.set_xlabel("未来第 h 个交易日")
ax.set_ylabel("平均 Rank IC（%）")
ax.set_title("北向资金流因子 IC 衰减：信息在前 5 日基本释放完毕")
ax.axhline(0, color="gray", lw=0.8)
fig.tight_layout()
fig.savefig(f"{OUT}/ic-decay.png", dpi=130)
plt.close(fig)
print("IC decay 1-5d:", [f"{d*100:.2f}" for d in decay[:5]])

# ---------- 分组回测 ----------
n_groups = 5
group_ret = np.zeros((n_days, n_groups))
for t in range(warm, n_days - 1):
    sig = flow_cum[t]
    qs = np.quantile(sig, np.linspace(0, 1, n_groups + 1)[1:-1])
    gid = np.digitize(sig, qs)
    for g in range(n_groups):
        group_ret[t + 1, g] = ret[t + 1, gid == g].mean()

cum = np.cumprod(1 + group_ret[warm:], axis=0)
ls = group_ret[warm:, -1] - group_ret[warm:, 0]
cum_ls_gross = np.cumprod(1 + ls)

# 成本：双边换手估计。每天重构组合，组内换手率估计
turnover = []
prev_top = None
for t in range(warm, n_days - 1):
    sig = flow_cum[t]
    top = set(np.argsort(sig)[-n_stocks // n_groups:])
    if prev_top is not None:
        turnover.append(1 - len(top & prev_top) / len(top))
    prev_top = top
to = np.mean(turnover)
print(f"多头组日均换手率: {to*100:.1f}%")

cost_bp = 15  # 单边 15bp（A股佣金+印花税卖出+冲击）
daily_cost = 2 * to * 2 * cost_bp / 1e4   # 多空两条腿、双边
ls_net = ls - daily_cost
cum_ls_net = np.cumprod(1 + ls_net)

ann = 252
def stats(r):
    return r.mean() * ann * 100, r.mean() / r.std() * np.sqrt(ann)

for name, r in [("LS gross", ls), ("LS net", ls_net)]:
    a, s = stats(r)
    dd = 1 - np.cumprod(1 + r) / np.maximum.accumulate(np.cumprod(1 + r))
    print(f"{name}: 年化={a:.1f}%, Sharpe={s:.2f}, MaxDD={dd.max()*100:.1f}%")

fig, ax = plt.subplots(figsize=(9.5, 5))
for g in range(n_groups):
    ax.plot(cum[:, g], lw=1.1, label=f"G{g+1}" + ("（净流出最多）" if g == 0 else "（净流入最多）" if g == n_groups-1 else ""))
ax.set_yscale("log")
ax.set_xlabel("交易日")
ax.set_ylabel("累计净值（对数）")
ax.set_title("北向资金流五分组累计净值")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/group-nav.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9.5, 5))
ax.plot(cum_ls_gross, lw=1.5, label="多空组合（费前）", color="#2ca02c")
ax.plot(cum_ls_net, lw=1.5, label=f"多空组合（单边 {cost_bp}bp 费后）", color="#d62728")
ax.axhline(1, color="gray", lw=0.8)
ax.set_xlabel("交易日")
ax.set_ylabel("累计净值")
ax.set_title("北向流因子多空组合：交易成本吃掉多少")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/longshort-cost.png", dpi=130)
plt.close(fig)

# ---------- 动量正交化 ----------
# 北向流 = 信息 + 动量追随。剥离动量后还剩多少 alpha？
ic_raw, ic_orth = [], []
for t in range(warm, n_days - 6):
    sig = flow_cum[t]
    m = mom5[t]
    # 截面回归取残差
    X = np.column_stack([np.ones(n_stocks), m])
    beta_hat = np.linalg.lstsq(X, sig, rcond=None)[0]
    resid = sig - X @ beta_hat
    f = fwd5[t]
    rs = np.argsort(np.argsort(sig)); rr = np.argsort(np.argsort(resid)); rf = np.argsort(np.argsort(f))
    ic_raw.append(np.corrcoef(rs, rf)[0, 1])
    ic_orth.append(np.corrcoef(rr, rf)[0, 1])
ic_raw, ic_orth = np.array(ic_raw), np.array(ic_orth)
print(f"原始因子 IC(5d)={ic_raw.mean():.4f}, 动量正交后 IC(5d)={ic_orth.mean():.4f}, 保留比例={ic_orth.mean()/ic_raw.mean()*100:.0f}%")

fig, ax = plt.subplots(figsize=(8.5, 4.8))
w = 60
roll_raw = np.convolve(ic_raw, np.ones(w)/w, mode="valid")
roll_orth = np.convolve(ic_orth, np.ones(w)/w, mode="valid")
ax.plot(roll_raw * 100, label=f"原始北向流因子（均值 {ic_raw.mean()*100:.2f}%）", lw=1.3)
ax.plot(roll_orth * 100, label=f"剥离动量后（均值 {ic_orth.mean()*100:.2f}%）", lw=1.3)
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("交易日")
ax.set_ylabel("60 日滚动 Rank IC（%）")
ax.set_title("北向流因子的 alpha 有多少只是动量的马甲")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/orthogonal-ic.png", dpi=130)
plt.close(fig)

# ---------- 调仓频率 vs 费后收益 ----------
freqs = [1, 2, 5, 10, 20]
res = []
for f in freqs:
    gret = np.zeros(n_days)
    tos = []
    prev_top = None; prev_bot = None
    hold_top = None; hold_bot = None
    for t in range(warm, n_days - 1):
        if (t - warm) % f == 0:
            sig = flow_cum[t]
            order = np.argsort(sig)
            new_bot = set(order[:n_stocks // n_groups])
            new_top = set(order[-n_stocks // n_groups:])
            if hold_top is not None:
                tos.append(1 - len(new_top & hold_top) / len(new_top))
            hold_top, hold_bot = new_top, new_bot
        if hold_top is not None:
            it = list(hold_top); ib = list(hold_bot)
            gret[t + 1] = ret[t + 1, it].mean() - ret[t + 1, ib].mean()
    to_f = np.mean(tos) if tos else 0
    cost_daily = 2 * to_f * 2 * cost_bp / 1e4 / f
    net = gret[warm:] - cost_daily
    g = gret[warm:]
    res.append((f, g.mean()*252*100, net.mean()*252*100,
                net.mean()/net.std()*np.sqrt(252), to_f*100))
    print(f"freq={f}d: gross={res[-1][1]:.1f}%, net={res[-1][2]:.1f}%, SR_net={res[-1][3]:.2f}, TO={to_f*100:.0f}%")

fig, ax = plt.subplots(figsize=(9, 4.8))
xs = [str(r[0]) for r in res]
ax.bar(np.arange(len(res)) - 0.2, [r[1] for r in res], 0.4, label="费前年化（%）", color="#1f77b4")
ax.bar(np.arange(len(res)) + 0.2, [r[2] for r in res], 0.4, label="费后年化（%）", color="#d62728")
ax.set_xticks(range(len(res)), [f"{x}日" for x in xs])
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("调仓频率")
ax.set_ylabel("年化收益（%）")
ax.set_title("调仓频率与费后收益：北向流因子必须降频才能活")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/rebalance-freq.png", dpi=130)
plt.close(fig)
