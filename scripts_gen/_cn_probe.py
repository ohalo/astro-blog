#!/usr/bin/env python3
"""Probe: find an honest, decisive contrastive negative-sampling regime."""
import numpy as np

N_SECTOR = 5
PER = 10
N = N_SECTOR * PER
DIM = 8
OUT = 4
sector = np.repeat(np.arange(N_SECTOR), PER)
pos_mask = (sector[:, None] == sector[None, :]) & ~np.eye(N, dtype=bool)


def l2(x, e=1e-8):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + e)


def make_centroids(seed=1):
    # two sectors overlap (close centroids) -> hard negatives; three far -> easy
    rng = np.random.default_rng(seed)
    a = np.array([0.0, 0.45, 2.3, 3.3, 4.2])
    base = np.array([[np.cos(t), np.sin(t)] + [0] * (DIM - 2) for t in a]) * 1.5
    R = rng.normal(0, 1, (DIM, DIM))
    return base @ R


def make_raw(rng, cen, sig, noise):
    return np.array([sig * cen[sector[i]] + rng.normal(0, noise, DIM) for i in range(N)])


def make_pairs():
    pos_pair = np.zeros(N, dtype=int)
    for s in range(N_SECTOR):
        js = np.where(sector == s)[0]
        for k, i in enumerate(js):
            pos_pair[i] = js[(k + 5) % PER]
    return pos_pair


def pick_neg(z, strat, K, rng):
    sim = z @ z.T
    res = []
    for i in range(N):
        order = np.argsort(-sim[i])
        cands = [j for j in order if j != i and j != pos_pair[i]]
        if strat == "random":
            cand = np.array([j for j in range(N) if j != i and j != pos_pair[i]])
            pool = cand[rng.choice(len(cand), min(K, len(cand)), replace=False)]
        elif strat == "hard":
            tn = [j for j in cands if not pos_mask[i, j]][:K]
            pool = np.array(tn[:K])
        elif strat == "semihard":
            ps = float(sim[i, pos_pair[i]])
            band = [j for j in cands if (not pos_mask[i, j]) and ps < sim[i, j] <= ps + 0.35]
            pool = np.array(band[:K] if len(band) >= K else cands[:K])
        else:  # noisy: top-K most similar, NOT excluding same sector
            pool = np.array(cands[:K])
        res.append(pool)
    return res


pos_pair = make_pairs()


def train(strat, epochs, K, lr, tau, seed, cen, sig, noise):
    rng = np.random.default_rng(seed)
    raw = make_raw(rng, cen, sig, noise)
    z = l2(raw[:, :OUT]).copy()
    for ep in range(epochs):
        negs = pick_neg(z, strat, K, rng)
        gz = np.zeros_like(z)
        for i in range(N):
            zp = z[pos_pair[i]]
            zn = z[negs[i]]
            ap = z[i] @ zp / tau
            an = z[i] @ zn.T / tau
            lg = np.concatenate([[ap], an])
            ex = np.exp(lg - lg.max())
            p = ex / ex.sum()
            gzi = (1.0 / tau) * (-(1 - p[0]) * zp + (p[1:] @ zn))
            gz[i] += gzi
            gz[pos_pair[i]] += (-(1 - p[0]) / tau) * z[i]
            w = p[1:] / tau
            for k, j in enumerate(negs[i]):
                gz[j] += w[k] * z[i]
        z = l2(z - lr * gz)
    # final loss
    negs = pick_neg(z, strat, K, rng)
    loss = 0.0
    for i in range(N):
        ap = z[i] @ z[pos_pair[i]] / tau
        an = z[i] @ z[negs[i]].T / tau
        lg = np.concatenate([[ap], an])
        ex = np.exp(lg - lg.max())
        loss += -np.log(ex[0] / ex.sum())
    return z, loss / N


def loo_1nn(z):
    sim = z @ z.T
    np.fill_diagonal(sim, -9)
    pred = sector[np.argmax(sim, axis=1)]
    return float(np.mean(pred == sector))


if __name__ == "__main__":
    cen = make_centroids(1)
    for sig, noise in [(1.8, 1.0), (2.2, 1.2), (2.5, 1.4)]:
        for tau in [0.3, 0.5]:
            for lr in [0.3, 0.5]:
                row = []
                for st in ["random", "hard", "semihard", "noisy"]:
                    accs, losses = [], []
                    for sd in [0, 7, 42, 99, 123]:
                        z, lf = train(st, 700, 16, lr, tau, sd, cen, sig, noise)
                        accs.append(loo_1nn(z))
                        losses.append(lf)
                    row.append(f"{np.mean(accs):.3f}/{np.mean(losses):.3f}")
                print(f"sig={sig} noise={noise} tau={tau} lr={lr}: " +
                      f"rand={row[0]} hard={row[1]} semi={row[2]} noisy={row[3]}")
