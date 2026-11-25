"""向量化回测引擎 + 四类回测陷阱审计：文章单一代码源。

纯 NumPy 实现，固定种子，exec 输出与 results.json 深度相等。
生成 5 张图：速度对比、前视偏差、参数热图、成本扫描、蒙特卡洛扇形。
"""
import json
import os
import time

import numpy as np

SEED = 20261124
N_TOTAL = 1800
N_TRAIN = 900
N_VALID = 300
COST_BPS = 5.0
FREQ = 252
FAST_LIST = np.array([5, 10, 15, 20, 25, 30])
SLOW_LIST = np.array([20, 30, 40, 50, 60, 70, 80])
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "public", "images", "quant-backtesting-vectorbt-guide")


# ------------------------------------------------------------------ 数据与引擎
def make_price(seed=SEED, n=N_TOTAL):
    """带两段波动状态的合成日线：温暖期 +  turbulent 期。"""
    rng = np.random.default_rng(seed)
    vol = np.where(np.arange(n) < n // 2, 0.009, 0.015)
    drift = np.where(np.arange(n) < n // 2, 0.0005, -0.0002)
    shock = rng.normal(drift, 1.0, size=n) * vol
    return 100.0 * np.exp(np.cumsum(shock))


def sma(x, window):
    """累积和 O(N) 移动均线，warmup 段为 NaN。"""
    csum = np.concatenate([[0.0], np.cumsum(x)])
    out = (csum[window:] - csum[:-window]) / window
    return np.concatenate([np.full(window - 1, np.nan), out])


def sma_matrix(x, windows):
    return np.stack([sma(x, int(w)) for w in windows])


def make_position(price, fast, slow, lookahead=False):
    """双均线持仓：信号在 bar i 收盘确认，默认 bar i+1 生效。"""
    pos = (sma(price, fast) > sma(price, slow)).astype(np.float64)
    if lookahead:
        return pos
    shifted = np.zeros_like(pos)
    shifted[1:] = pos[:-1]
    return shifted


def returns(price):
    r = np.zeros_like(price)
    r[1:] = price[1:] / price[:-1] - 1.0
    return r


def apply_cost(strategy_ret, position, cost_bps):
    """按换手扣单边成本：换手率 = |今日仓位 - 昨日仓位|。"""
    turnover = np.abs(np.diff(position, prepend=0.0))
    return strategy_ret - turnover * (cost_bps / 1e4), turnover


def stats(strategy_ret, position):
    r = np.asarray(strategy_ret, dtype=np.float64)
    mean = float(np.mean(r))
    std = float(np.std(r))
    sharpe = mean / std * np.sqrt(FREQ) if std > 0 else 0.0
    equity = np.cumprod(1.0 + r)
    total_return = float(equity[-1] - 1.0)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity / peak - 1.0))
    return {
        "sharpe": round(sharpe, 6),
        "return": round(total_return, 6),
        "max_dd": round(max_dd, 6),
        "turnover": round(float(np.mean(np.abs(np.diff(position, prepend=0.0)))), 6),
        "exposure": round(float(np.mean(position)), 6),
        "equity": equity,
    }


def backtest(price, fast, slow, start=0, end=None, cost_bps=COST_BPS, lookahead=False):
    end = len(price) if end is None else end
    seg = price[start:end]
    pos = make_position(seg, fast, slow, lookahead=lookahead)
    strat = pos * returns(seg)
    net, turn = apply_cost(strat, pos, cost_bps)
    s = stats(net, pos)
    s["turnover"] = round(float(np.mean(turn)), 6)
    return s


def scan_grid(price, start, end, cost_bps=COST_BPS):
    """(F,S,N) 广播一次算完全部参数组合的净收益 Sharpe。"""
    seg = price[start:end]
    n = len(seg)
    mf = sma_matrix(seg, FAST_LIST)   # (F,N)
    ms = sma_matrix(seg, SLOW_LIST)   # (S,N)
    pos = (mf[:, None, :] > ms[None, :, :]).astype(np.float64)  # (F,S,N)
    shifted = np.zeros_like(pos)
    shifted[:, :, 1:] = pos[:, :, :-1]
    r = returns(seg)
    net, turn = apply_cost(shifted * r[None, None, :], shifted, cost_bps)
    mean = net.mean(axis=2)
    std = net.std(axis=2)
    sharpe = np.zeros_like(mean)
    np.divide(mean, std, out=sharpe, where=std > 0)
    sharpe *= np.sqrt(FREQ)
    illegal = FAST_LIST[:, None] >= SLOW_LIST[None, :]
    sharpe[illegal] = np.nan
    turnover = turn.mean(axis=2)
    return sharpe, turnover


# ------------------------------------------------------------------ 图
def fig_speed(price):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [1000, 3000, 10000, 30000]
    loop_t, vec_t = [], []
    base = price[:500]
    for n in sizes:
        x = np.tile(base, int(np.ceil(n / len(base))))[:n]
        t0 = time.perf_counter()
        ma_f = np.full(n, np.nan)
        for i in range(n):
            if i >= 19:
                ma_f[i] = x[i - 19:i + 1].mean()
        loop_t.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        _ = sma(x, 20)
        vec_t.append(time.perf_counter() - t0)
    # 墙钟时间只用于画图；JSON 只存确定性操作量，保证 exec 复跑一致
    window = 20
    loop_ops = [n * window + n for n in sizes]
    vec_ops = [2 * n + window for n in sizes]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.loglog(sizes, loop_t, "o-", color="#b23a2b", lw=2, label="Python loop (slice mean)")
    ax.loglog(sizes, vec_t, "s-", color="#1f4e79", lw=2, label="NumPy cumsum")
    ax.set_xlabel("number of bars")
    ax.set_ylabel("seconds (log scale)")
    ax.set_title("Loop vs vectorized moving average")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_speed.png"), dpi=140)
    plt.close(fig)
    return loop_t, vec_t, loop_ops, vec_ops


def fig_lookahead(price):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    honest = backtest(price, 10, 50, 0, N_TOTAL, COST_BPS)
    cheat = backtest(price, 10, 50, 0, N_TOTAL, COST_BPS, lookahead=True)
    t = np.arange(N_TOTAL)
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.plot(t, honest["equity"], color="#1f4e79", lw=1.7,
            label="next-bar execution (honest)")
    ax.plot(t, cheat["equity"], color="#b23a2b", lw=1.7,
            label="same-bar execution (look-ahead)")
    ax.axvline(N_TRAIN, color="#888", ls=":", lw=1.0)
    ax.axvline(N_TRAIN + N_VALID, color="#555", ls="--", lw=1.1)
    ax.text(N_TRAIN + N_VALID + 12, ax.get_ylim()[1] * 0.92, "test starts",
            fontsize=9, color="#444")
    ax.set_xlabel("trading day")
    ax.set_ylabel("equity (start = 1)")
    ax.set_title("One-bar shift changes the whole story")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_lookahead.png"), dpi=140)
    plt.close(fig)
    return honest, cheat


def fig_grid(sharpe, best_idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    data = np.ma.masked_invalid(sharpe)
    im = ax.imshow(data, origin="lower", aspect="auto", cmap="RdYlGn",
                   extent=[SLOW_LIST[0] - 5, SLOW_LIST[-1] + 5,
                           FAST_LIST[0] - 2.5, FAST_LIST[-1] + 2.5])
    ax.scatter([SLOW_LIST[best_idx[1]]], [FAST_LIST[best_idx[0]]],
               marker="*", s=230, color="black", edgecolor="white", lw=1.1,
               label="in-sample champion")
    ax.set_xlabel("slow window")
    ax.set_ylabel("fast window")
    ax.set_title("In-sample Sharpe grid (5bps cost, next-bar execution)")
    fig.colorbar(im, ax=ax, label="annualised Sharpe")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_param_grid.png"), dpi=140)
    plt.close(fig)


def fig_cost(price, best_f, best_s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    costs = [0.0, 2.0, 5.0, 10.0, 20.0, 40.0]
    rows = {}
    for c in costs:
        rows[str(int(c))] = backtest(price, best_f, best_s,
                                     N_TRAIN + N_VALID, N_TOTAL, cost_bps=c)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.plot(costs, [rows[str(int(c))]["sharpe"] for c in costs], "o-",
            color="#1f4e79", lw=2, label="Sharpe (left)")
    ax2 = ax.twinx()
    ax2.plot(costs, [rows[str(int(c))]["return"] for c in costs], "s--",
             color="#b23a2b", lw=1.8, label="test return (right)")
    ax.set_xlabel("single-side cost (bps)")
    ax.set_ylabel("annualised Sharpe", color="#1f4e79")
    ax2.set_ylabel("cumulative test return", color="#b23a2b")
    ax.set_title("Cost sensitivity of the in-sample champion")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_cost.png"), dpi=140)
    plt.close(fig)
    return rows


def fig_mc(strategy_ret, seed=SEED):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    n_paths, horizon = 2000, FREQ
    idx = rng.integers(0, len(strategy_ret), size=(n_paths, horizon))
    draws = strategy_ret[idx]
    paths = np.cumprod(1.0 + draws, axis=1)
    perc = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    t = np.arange(1, horizon + 1)
    ax.fill_between(t, perc[0], perc[4], color="#9ecae1", alpha=0.45,
                    label="5–95% (block bootstrap)")
    ax.fill_between(t, perc[1], perc[3], color="#3182bd", alpha=0.45,
                    label="25–75%")
    ax.plot(t, perc[2], color="#08306b", lw=1.8, label="median")
    ax.axhline(1.0, color="#777", ls="--", lw=1)
    ax.set_xlabel("trading day")
    ax.set_ylabel("equity")
    ax.set_title("2000 bootstrap paths of the daily strategy return")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_bootstrap_fan.png"), dpi=140)
    plt.close(fig)
    return [round(float(x), 6) for x in perc[:, -1]]


# ------------------------------------------------------------------ 主流程
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    price = make_price()

    honest = backtest(price, 10, 50, 0, N_TRAIN)
    cheat = backtest(price, 10, 50, 0, N_TRAIN, lookahead=True)
    honest_test = backtest(price, 10, 50, N_TRAIN + N_VALID, N_TOTAL)
    honest_valid = backtest(price, 10, 50, N_TRAIN, N_TRAIN + N_VALID)

    train_sharpe, _ = scan_grid(price, 0, N_TRAIN)
    flat = np.argwhere(~np.isnan(train_sharpe))
    best_flat = flat[np.nanargmax(train_sharpe[~np.isnan(train_sharpe)])]
    bi, sj = int(best_flat[0]), int(best_flat[1])
    best_f, best_s = int(FAST_LIST[bi]), int(SLOW_LIST[sj])
    best_train_sharpe = float(train_sharpe[bi, sj])
    # 高原判定：八邻域（去掉自身）的均值
    neigh = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii, jj = bi + di, sj + dj
            if (di, dj) == (0, 0) or not (0 <= ii < len(FAST_LIST) and 0 <= jj < len(SLOW_LIST)):
                continue
            if not np.isnan(train_sharpe[ii, jj]):
                neigh.append(float(train_sharpe[ii, jj]))
    plateau_mean = float(np.mean(neigh)) if neigh else 0.0

    test_sharpe, _ = scan_grid(price, N_TRAIN + N_VALID, N_TOTAL)
    champ_test_sharpe = float(test_sharpe[bi, sj]) if not np.isnan(test_sharpe[bi, sj]) else 0.0
    ranked = np.argsort(-np.nan_to_num(train_sharpe, nan=-1e9), axis=None)
    top5_train = []
    top5_test = []
    for k in ranked[:5]:
        i, j = divmod(int(k), len(SLOW_LIST))
        top5_train.append(round(float(train_sharpe[i, j]), 6))
        tv = test_sharpe[i, j]
        top5_test.append(round(float(tv), 6) if not np.isnan(tv) else None)

    # walk-forward：6 段，train 300 / test 120，滚动选参数
    wf = []
    for w in range(6):
        tr_end = N_TRAIN + w * 60
        te_end = tr_end + 120
        if te_end > N_TOTAL:
            break
        sh, _ = scan_grid(price, tr_end - 300, tr_end)
        valid = np.argwhere(~np.isnan(sh))
        b = valid[np.nanargmax(sh[~np.isnan(sh)])]
        wi, wj = int(b[0]), int(b[1])
        s_tr = backtest(price, int(FAST_LIST[wi]), int(SLOW_LIST[wj]),
                        tr_end - 300, tr_end)["sharpe"]
        s_te = backtest(price, int(FAST_LIST[wi]), int(SLOW_LIST[wj]),
                        tr_end, te_end)["sharpe"]
        wf.append({"window": w + 1, "fast": int(FAST_LIST[wi]),
                   "slow": int(SLOW_LIST[wj]), "train_sharpe": s_tr,
                   "test_sharpe": s_te})

    loop_t, vec_t, loop_ops, vec_ops = fig_speed(price)
    lo, ch = fig_lookahead(price)
    fig_grid(train_sharpe, (bi, sj))
    cost_rows = fig_cost(price, best_f, best_s)
    pos_best = make_position(price[N_TRAIN + N_VALID:N_TOTAL], best_f, best_s)
    r_best = pos_best * returns(price[N_TRAIN + N_VALID:N_TOTAL])
    net_best, _ = apply_cost(r_best, pos_best, COST_BPS)
    mc_final = fig_mc(net_best)

    result = {
        "setup": {
            "seed": SEED, "n_total": N_TOTAL, "n_train": N_TRAIN,
            "n_valid": N_VALID, "n_test": N_TOTAL - N_TRAIN - N_VALID,
            "cost_bps": COST_BPS, "freq": FREQ,
            "fast_list": FAST_LIST.tolist(), "slow_list": SLOW_LIST.tolist(),
        },
        "speed": {"bars": [1000, 3000, 10000, 30000],
                  "loop_elementary_ops": loop_ops,
                  "vector_elementary_ops": vec_ops,
                  "work_ratio": round(float(loop_ops[-1] / vec_ops[-1]), 6)},
        "lookahead": {"honest": {k: lo[k] for k in ("sharpe", "return", "max_dd", "turnover")},
                      "cheat": {k: ch[k] for k in ("sharpe", "return", "max_dd", "turnover")}},
        "fixed": {"train": {k: honest[k] for k in ("sharpe", "return", "max_dd")},
                  "valid": {k: honest_valid[k] for k in ("sharpe", "return", "max_dd")},
                  "test": {k: honest_test[k] for k in ("sharpe", "return", "max_dd")}},
        "grid": {
            "champion": {"fast": best_f, "slow": best_s,
                         "train_sharpe": round(best_train_sharpe, 6),
                         "test_sharpe": round(champ_test_sharpe, 6),
                         "neighbour_mean": round(plateau_mean, 6)},
            "top5_train_sharpe": top5_train,
            "top5_matched_test_sharpe": top5_test,
        },
        "cost": {c: {k: cost_rows[c][k] for k in ("sharpe", "return", "turnover")}
                 for c in cost_rows},
        "walk_forward": wf,
        "bootstrap_final_equity": {
            "p5": mc_final[0], "p25": mc_final[1], "p50": mc_final[2],
            "p75": mc_final[3], "p95": mc_final[4],
        },
    }
    out = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
