# -*- coding: utf-8 -*-
"""探测 v4：以 SAM 版 LZ76 为权威，校准全部叙事数字。"""
import math
import time

import numpy as np

from probe_20261114 import LZ76SAM, pe_norm


def lz76(seq):
    return LZ76SAM().complexity(seq)


def norm_c(z):
    n = len(z)
    return lz76(z) * math.log2(n) / n


if __name__ == "__main__":
    rng = np.random.default_rng(20261114)

    # 0) SAM 正确性抽查：全 0 串 n=60 应 11（0|00|000|...|0000000000|尾）
    print("allzero60:", lz76([0] * 60), "expect 11", flush=True)

    # 1) iid 渐近（SAM 版）：C log2 n / n 走向哪
    for n in (200, 800, 4000, 16000):
        s = (rng.random(n) < 0.5).astype(int).tolist()
        c = lz76(s)
        print(f"fair n={n}: C={c} norm={c*math.log2(n)/n:.4f} "
              f"C/n={c/n:.4f} 1/log2n={1/math.log2(n):.4f}", flush=True)

    # 2) 周期 sign 序列 + 加噪周期：PE vs LZ 谁瞎
    t = np.arange(1600)
    for p_ in (8, 15, 30, 60):
        base = np.sin(2 * np.pi * t / p_)
        for snr in (np.inf, 1.0, 0.5):
            y = base if snr == np.inf else base / np.std(base) + snr * rng.standard_normal(len(t))
            y = np.round(y, 6)
            pe = pe_norm(y, 5)
            c = lz76((y > 0).astype(int).tolist())
            print(f"period={p_} snr={snr}: PE={pe:.4f} signC={c} "
                  f"norm={c*math.log2(1600)/1600:.4f}", flush=True)

    # 3) 混沌 vs 随机（SAM 版）
    x = 0.314159
    xs = []
    for _ in range(2000):
        x = 4 * x * (1 - x)
        xs.append(x)
    xs = np.array(xs)
    cl = lz76((xs > 0.5).astype(int).tolist())
    wn = (np.random.default_rng(2).standard_normal(2000) > 0).astype(int).tolist()
    cw = lz76(wn)
    print(f"logistic: C={cl} norm={cl*math.log2(2000)/2000:.4f} PE={pe_norm(xs,5):.4f}",
          flush=True)
    print(f"white  : C={cw} norm={cw*math.log2(2000)/2000:.4f} PE={pe_norm(2*rng.standard_normal(2000)-1,5):.4f}",
          flush=True)

    # 4) 游程效应：趋势注入（长同向 run）应压低 C
    n = 400
    noise = (rng.random(n) < 0.5).astype(int).tolist()
    cn = lz76(noise)
    print(f"noise400: C={cn} norm={cn*math.log2(n)/n:.4f}", flush=True)
    for bias in (0.1, 0.05):
        seq2 = (noise[:100] + (rng.random(200) < bias).astype(int).tolist()
                + noise[100:300])
        ct = lz76(seq2)
        print(f"trend bias={bias}: C={ct} norm={ct*math.log2(n)/n:.4f}", flush=True)

    # 5) 置换零分布：有限 n 下朴素常数对照 vs 置换校准
    n = 200
    fp_naive = 0
    fp_perm = 0
    REP = 300
    for _ in range(REP):
        z = (rng.random(n) < 0.5).astype(int).tolist()
        c = lz76(z)
        # 朴素规则：norm < 0.95*1 判「有结构」（用渐近常数 1）
        if c * math.log2(n) / n < 0.95:
            fp_naive += 1
        # 置换校准：同序列 shuffle 作零分布
        nulls = []
        zz = z.copy()
        for _s in range(120):
            rng.shuffle(zz)
            nulls.append(lz76(zz))
        nulls = np.array(nulls)
        zscore = (c - nulls.mean()) / (nulls.std() + 1e-12)
        if zscore < -1.645:
            fp_perm += 1
    print(f"FP@n=200: naive={fp_naive/REP:.4f} perm(z<-1.645)={fp_perm/REP:.4f}",
          flush=True)

    # 6) 趋势检出：power 对比 LZ-z vs PE-z（trend: AR1 phi=0.85 首 100 天）
    def lz_z(z, rng_, nb=120):
        c = lz76(z)
        nulls = []
        for _ in range(nb):
            zz = z.copy()
            rng_.shuffle(zz)
            nulls.append(lz76(zz))
        nulls = np.array(nulls)
        return (c - nulls.mean()) / (nulls.std() + 1e-12)

    def pe_z(x, rng_, m=5, nb=120):
        pe = pe_norm(x, m)
        nulls = []
        xx = x.copy()
        for _ in range(nb):
            rng_.shuffle(xx)
            nulls.append(pe_norm(xx, m))
        nulls = np.array(nulls)
        return (pe - nulls.mean()) / (nulls.std() + 1e-12)

    REP = 60
    n = 200
    power_lz = power_pe = 0
    fp_lz = fp_pe = 0
    thr_lz = np.percentile([lz_z((rng.random(n) < 0.5).astype(int).tolist(),
                                 np.random.default_rng(1000 + i))
                            for i in range(40)], 5)
    thr_pe = np.percentile([pe_z(rng.standard_normal(n),
                                 np.random.default_rng(2000 + i))
                            for i in range(40)], 5)
    t0 = time.time()
    for i in range(REP):
        r = np.random.default_rng(3000 + i)
        # H1：前 100 天持续上涨（AR1 phi=.9 正偏），后 100 天噪声
        phi = 0.9
        e = r.standard_normal(n)
        x = np.zeros(n)
        x[0] = e[0]
        for k in range(1, 100):
            x[k] = phi * x[k - 1] + math.sqrt(1 - phi * phi) * e[k] + 0.6
        x[100:] = e[100:]
        sgn = (x > 0).astype(int).tolist()
        if lz_z(sgn, np.random.default_rng(4000 + i)) < thr_lz:
            power_lz += 1
        if pe_z(x, np.random.default_rng(5000 + i)) < thr_pe:
            power_pe += 1
    for i in range(REP):
        r = np.random.default_rng(6000 + i)
        x = r.standard_normal(n)
        sgn = (x > 0).astype(int).tolist()
        if lz_z(sgn, np.random.default_rng(7000 + i)) < thr_lz:
            fp_lz += 1
        if pe_z(x, np.random.default_rng(8000 + i)) < thr_pe:
            fp_pe += 1
    print(f"power: LZ={power_lz/REP:.3f} PE={power_pe/REP:.3f} "
          f"fp: LZ={fp_lz/REP:.3f} PE={fp_pe/REP:.3f} "
          f"(thr_lz={thr_lz:.3f} thr_pe={thr_pe:.3f}) {time.time()-t0:.0f}s",
          flush=True)
