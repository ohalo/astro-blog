# -*- coding: utf-8 -*-
"""2026-11-14 批次探测 v3：LZ76 两个独立实现互证 + 叙事假设验证。

lz76      ：贪心短语分割（哈希短语集合），O(n log n)，文章用
lz76_sam  ：后缀自动机版（Blumer et al.），独立实现，用于对拍
"""
import math
import time
from itertools import permutations

import numpy as np


def lz76(seq):
    """LZ76 复杂度 C(n)：贪心分割成「最短的、未出现过的短语」的个数。"""
    s = tuple(int(b) for b in seq)
    n = len(s)
    seen = set()
    c = 0
    i = 0
    while i < n:
        j = 1
        while i + j <= n and s[i:i + j] in seen:
            j += 1
        # 匹配到的前缀 s[i:i+j-1] 已见过；s[i:i+j]（若存在）首现
        if i + j <= n:
            seen.add(s[i:i + j])
            c += 1
            i += j
        else:
            # 剩余尾部全部见过：尾巴整体作为最后一个短语
            tail = s[i:n]
            if tail not in seen:
                seen.add(tail)
            c += 1
            break
    return c


class LZ76SAM:
    """后缀自动机在线版（独立实现，用于对拍）。"""

    def __init__(self):
        self.next = [{}]
        self.link = [-1]
        self.length = [0]

    def _extend(self, c):
        cur = len(self.next)
        self.next.append({})
        self.link.append(-1)
        self.length.append(self.length[-1] + 1)
        p = cur - 1
        while p != -1 and c not in self.next[p]:
            self.next[p][c] = cur
            p = self.link[p]
        if p == -1:
            self.link[cur] = 0
        else:
            q = self.next[p][c]
            if self.length[p] + 1 == self.length[q]:
                self.link[cur] = q
            else:
                clone = len(self.next)
                self.next.append(dict(self.next[q]))
                self.link.append(self.link[q])
                self.length.append(self.length[p] + 1)
                while p != -1 and self.next[p].get(c) == q:
                    self.next[p][c] = clone
                    p = self.link[p]
                self.link[q] = self.link[cur] = clone
        return cur

    def complexity(self, seq):
        seq = [int(b) for b in seq]
        n = len(seq)
        c = 0
        i = 0
        guard = 0
        while i < n:
            guard += 1
            if guard > 4 * n + 100:
                raise RuntimeError("SAM walk stuck")
            state = 0
            j = i
            while j < n and seq[j] in self.next[state]:
                state = self.next[state][seq[j]]
                j += 1
            if j == n:
                # 末尾剩余全部匹配：作为最后一个短语插入
                for sym in seq[i:n]:
                    self._extend(sym)
                c += 1
                break
            c += 1
            for sym in seq[i:j]:
                self._extend(sym)
            self._extend(seq[j])
            i = j + 1
        return c


def lz76_sam(seq):
    return LZ76SAM().complexity(seq)


def pe_norm(x, m=5, tau=1):
    """归一化排列熵（Bandt & Pompe 2002）。"""
    x = np.asarray(x, float)
    perms = list(permutations(range(m)))
    idx = {p: i for i, p in enumerate(perms)}
    counts = np.zeros(len(perms))
    emb = (m - 1) * tau + 1
    rng = np.random.default_rng(0)
    for t in range(len(x) - emb + 1):
        w = x[t:t + emb:tau] + 1e-12 * rng.standard_normal(m)
        counts[idx[tuple(np.argsort(w))]] += 1
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(len(perms)))


def ar1_sign(n, phi, seed):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.zeros(n)
    s = math.sqrt(1 - phi * phi)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + s * e[t]
    return (x > 0).astype(int).tolist()


if __name__ == "__main__":
    # ---- 1. 两个实现互证 ----
    cases = [[0] * 60, [1] + [0] * 59, [i % 2 for i in range(80)],
             [1, 0, 0, 1] * 20, [0, 0, 1, 1, 0] * 24]
    rng0 = np.random.default_rng(42)
    for _ in range(6):
        cases.append((rng0.random(150) < 0.5).astype(int).tolist())
    cases.append((rng0.random(400) < 0.35).astype(int).tolist())
    allok = True
    for s in cases:
        t0 = time.time()
        a = lz76(s)
        tb = time.time() - t0
        b = lz76_sam(s)
        ok = a == b
        allok &= ok
        print("hash", a, "sam", b, "OK" if ok else "MISMATCH", "n=", len(s),
              f"({tb*1000:.0f}ms)", "head=", s[:8], flush=True)
    print("ALL MATCH" if allok else "HAS MISMATCH", flush=True)

    # ---- 2. 渐近：iid 符号序列 C(n)log2(n)/n -> 1/H ----
    rng = np.random.default_rng(1)
    for n in (1000, 4000):
        s = (rng.random(n) < 0.5).astype(int).tolist()
        t0 = time.time()
        c = lz76(s)
        print(f"fair n={n}: C={c} C*log2(n)/n={c*math.log2(n)/n:.4f} -> 1 "
              f"({time.time()-t0:.2f}s)", flush=True)
    import hashlib
    stream = ""
    seed_i = 0
    while len(stream) < 1500:
        stream += hashlib.sha256(str(seed_i).encode()).hexdigest()
        seed_i += 1
    hexs = [int(ch, 16) for ch in stream[:1500]]
    ch_ = lz76(hexs)
    print("sha256hex n=1500: C=", ch_, "C*log2/n=",
          round(ch_ * math.log2(1500) / 1500, 4), "-> log2(16)=",
          round(math.log2(16), 4), flush=True)

    # ---- 3. 长周期：PE 瞎、LZ 看见 ----
    t = np.arange(1600)
    for p_ in (8, 30, 61, 120):
        y = np.round(np.sin(2 * np.pi * t / p_), 6)
        pe = pe_norm(y, m=5)
        c = lz76((y > 0).astype(int).tolist())
        print(f"sine period={p_}: PE={pe:.4f} signC={c} "
              f"Clog2n/n={c*math.log2(1600)/1600:.4f}", flush=True)

    # ---- 4. 混沌 vs 随机 ----
    x = 0.314159
    xs = []
    for _ in range(2000):
        x = 4 * x * (1 - x)
        xs.append(x)
    xs = np.array(xs)
    cl = lz76((xs > 0.5).astype(int).tolist())
    wn = np.random.default_rng(2).standard_normal(2000)
    cw = lz76((wn > 0).astype(int).tolist())
    n_ = 2000
    print(f"logistic: C={cl} norm={cl*math.log2(n_)/n_:.4f} PE={pe_norm(xs,5):.4f}",
          flush=True)
    print(f"white  : C={cw} norm={cw*math.log2(n_)/n_:.4f} PE={pe_norm(wn,5):.4f}",
          flush=True)

    # ---- 5. AR(1) 符号序列：LZ 熵率 vs 精确符号熵率 ----
    for phi in (0.0, 0.3, 0.6, 0.85):
        sgn = ar1_sign(2000, phi, 11)
        c = lz76(sgn)
        trans = np.zeros((2, 2))
        for a, b in zip(sgn[:-1], sgn[1:]):
            trans[a, b] += 1
        trans /= np.maximum(trans.sum(1, keepdims=True), 1)
        Pi = np.array([0.5, 0.5])
        for _ in range(200):
            Pi = Pi @ trans
        rowH = np.array([-(trans[i] * np.log2(np.where(trans[i] > 0, trans[i], 1)))
                         .sum() for i in range(2)])
        H_rate = float((Pi * rowH).sum())
        print(f"AR1 phi={phi}: C={c} norm={c*math.log2(2000)/2000:.4f} "
              f"H_sign={H_rate:.4f} 1/H={1/H_rate:.4f}", flush=True)

    # ---- 6. 游程效应：趋势注入压低 sign-C ----
    n = 400
    noise = (rng.random(n) < 0.5).astype(int).tolist()
    cn = lz76(noise)
    print(f"noise C={cn} norm={cn*math.log2(n)/n:.4f}", flush=True)
    for bias in (0.08, 0.12, 0.2):
        seq2 = (noise[:100] + (rng.random(200) < bias).astype(int).tolist()
                + noise[100:300])
        ct = lz76(seq2)
        print(f"bias={bias}: C={ct} norm={ct*math.log2(n)/n:.4f}", flush=True)

    # ---- 7. t 池：E[max|t|] vs sqrt(2 ln N)；相关影响 ----
    for N in (100, 500, 2000, 5000, 10000):
        for corr in (0.0, 0.2):
            reps = 100 if (N >= 5000 and corr > 0) else 300
            rng_ = np.random.default_rng(3)
            Rho = corr * np.ones((N, N)) + (1 - corr) * np.eye(N)
            L = np.linalg.cholesky(Rho)
            mxs, c196, c3 = [], [], []
            for _ in range(reps):
                z0 = rng_.standard_normal((250, N))
                z = z0 @ L.T
                ts = z.mean(0) / (z.std(0, ddof=1) / math.sqrt(250))
                mxs.append(np.max(np.abs(ts)))
                c196.append(np.sum(np.abs(ts) > 1.96))
                c3.append(np.sum(np.abs(ts) > 3.0))
            print(f"N={N} corr={corr}: E[max|t|]={np.mean(mxs):.3f}"
                  f"(sd {np.std(mxs):.3f}) sqrt(2lnN)={math.sqrt(2*math.log(N)):.3f}"
                  f" E[#>1.96]={np.mean(c196):.1f} E[#>3]={np.mean(c3):.2f}",
                  flush=True)
