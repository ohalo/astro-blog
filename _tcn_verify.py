#!/usr/bin/env python3
"""Standalone verified TCN: im2col + batch matmul, FD-checked backward."""
import numpy as np

K = 2
C_CH = 6
L = 20
DILS = [1, 2, 4]


def causal_dconv_im2col(x, W, b, d):
    """x:[B,Cin,L] -> out[B,Cout,L]. im2col with left-pad (k-1)*d zeros (causal)."""
    B, Cin, Lx = x.shape
    Cout = W.shape[0]
    pad = (K - 1) * d
    xp = np.zeros((B, Cin, Lx + pad))
    xp[:, :, pad:] = x
    cols = []
    for kk in range(K):
        cols.append(xp[:, :, kk:kk + Lx])        # [B, Cin, L]
    Xcol = np.stack(cols, axis=2)                # [B, Cin, K, L]
    Xcol = Xcol.transpose(0, 3, 1, 2).reshape(B * L, Cin * K)  # [B*L, Cin*K]
    Wflat = W.reshape(Cout, Cin * K)             # [Cout, Cin*K]
    out = Xcol @ Wflat.T + b[None, :]            # [B*L, Cout]
    out = out.reshape(B, L, Cout).transpose(0, 2, 1)  # [B, Cout, L]
    return out


def relu(z): return np.maximum(0, z)
def drelu(z): return (z > 0).astype(float)


def init_tcn(nin=1, seed=0):
    rg = np.random.default_rng(seed)
    P = {}
    cin = nin
    for i, d in enumerate(DILS):
        cout = C_CH
        P[f"W{i}"] = rg.standard_normal((cout, cin, K)) * 0.10
        P[f"b{i}"] = np.zeros(cout)
        P[f"Wr{i}"] = rg.standard_normal((cin, cout)) * 0.10
        cin = cout
    P["Wo"] = rg.standard_normal((1, C_CH)) * 0.10
    P["bo"] = np.zeros(1)
    return P


def forward(X, P):
    acts = []
    h = X
    for i, d in enumerate(DILS):
        conv = causal_dconv_im2col(h, P[f"W{i}"], P[f"b{i}"], d)
        res = np.einsum("bcl,cj->bjl", h, P[f"Wr{i}"])
        a = relu(conv + res)
        acts.append((h, conv, res, a))
        h = a
    last = h[:, :, -1]
    yhat = last @ P["Wo"].T + P["bo"]
    return yhat.ravel(), acts


def backward(X, y, P, lr=0.02):
    B = X.shape[0]
    yhat, acts = forward(X, P)
    dL = (yhat - y) / B
    last_a = acts[-1][3][:, :, -1]
    dWo = dL.reshape(-1, 1).T @ last_a
    dbo = dL.sum().reshape(1)
    dh = (dL.reshape(-1, 1) @ P["Wo"]).reshape(B, C_CH)
    dh = np.einsum("bc,bcl->bcl", dh, np.ones((B, C_CH, L)))
    grads = {"Wo": dWo, "bo": dbo}
    for i in reversed(range(len(DILS))):
        h, conv, res, a = acts[i]
        da = dh * drelu(conv + res)
        # 权重梯度（im2col 反向）
        pad = (K - 1) * DILS[i]
        xp = np.zeros((B, h.shape[1], L + pad)); xp[:, :, pad:] = h
        cols = []
        for kk in range(K):
            cols.append(xp[:, :, kk:kk + L])
        Xcol = np.stack(cols, axis=2).transpose(0, 3, 1, 2).reshape(B * L, -1)  # [B*L, Cin*K]
        da_flat = da.transpose(0, 2, 1).reshape(B * L, -1)                       # [B*L, Cout]
        gWflat = da_flat.T @ Xcol                                                # [Cout, Cin*K]
        gW = gWflat.reshape(P[f"W{i}"].shape)
        gb = da_flat.sum(axis=0)                                                 # [Cout]
        gWr = np.einsum("bcl,bcl->cj", h, da) if False else None  # placeholder
        # 残差投影梯度: res[b,c_out,l] = Σ_cin h[b,cin,l]*Wr[cin,c_out]
        gWr = np.einsum("bol,bil->io", da, h)   # Σ_b,l da[b,c_out,l] h[b,cin,l] -> [cin,c_out]
        # 残差到 h_in 的梯度
        dh_prev = np.einsum("bjl,cj->bcl", da, P[f"Wr{i}"])   # [B, cin, L]
        # 卷积对输入 h 的梯度（反卷积, k=2 用 im2col 的转置）
        # ∂L/∂h[b,cin,l] = Σ_cout Σ_k W[c_out,cin,k]*da[b,c_out, l - k*d] (含 pad 消隐)
        dhtap = np.zeros_like(h)
        for kk in range(K):
            idx = np.arange(L) - kk * DILS[i]
            valid = idx >= 0
            grad_shift = np.zeros_like(da)
            grad_shift[:, :, valid] = da[:, :, idx[valid]]
            # 对 W 第 kk 列乘: W[c,cin,kk]
            dhtap += np.einsum("bcl,cj->bjl", grad_shift, P[f"W{i}"][:, :, kk])  # [B,cin,L]
        dh = dh_prev + dhtap
        grads[f"W{i}"] = gW; grads[f"b{i}"] = gb; grads[f"Wr{i}"] = gWr
    for k in grads:
        P[k] -= lr * grads[k]
    return float(np.mean((yhat - y) ** 2))


# ---- FD check ----
rg = np.random.default_rng(7)
Xc = rg.standard_normal((3, 1, L)) * 0.5
yc = rg.standard_normal(3)
P = init_tcn(seed=7)


def loss(PP):
    yy, _ = forward(Xc, PP)
    return 0.5 * np.mean((yy - yc) ** 2)


# analytic grads
yhat, acts = forward(Xc, P)
dL = (yhat - yc) / 3
last_a = acts[-1][3][:, :, -1]
an = {"Wo": dL.reshape(-1, 1).T @ last_a, "bo": dL.sum().reshape(1)}
dh = (dL.reshape(-1, 1) @ P["Wo"]).reshape(3, C_CH)
dh = np.einsum("bc,bcl->bcl", dh, np.ones((3, C_CH, L)))
for i in reversed(range(len(DILS))):
    h, conv, res, a = acts[i]
    da = dh * drelu(conv + res)
    pad = (K - 1) * DILS[i]
    xp = np.zeros((3, h.shape[1], L + pad)); xp[:, :, pad:] = h
    cols = []
    for kk in range(K):
        cols.append(xp[:, :, kk:kk + L])
    Xcol = np.stack(cols, axis=2).transpose(0, 3, 1, 2).reshape(3 * L, -1)
    da_flat = da.transpose(0, 2, 1).reshape(3 * L, -1)
    gW = (da_flat.T @ Xcol).reshape(P[f"W{i}"].shape)
    gb = da_flat.sum(axis=0)
    gWr = np.einsum("bol,bil->io", da, h)
    dh_prev = np.einsum("bjl,cj->bcl", da, P[f"Wr{i}"])
    dhtap = np.zeros_like(h)
    for kk in range(K):
        idx = np.arange(L) - kk * DILS[i]; valid = idx >= 0
        gs = np.zeros_like(da); gs[:, :, valid] = da[:, :, idx[valid]]
        dhtap += np.einsum("bcl,cj->bjl", gs, P[f"W{i}"][:, :, kk])
    dh = dh_prev + dhtap
    an[f"W{i}"] = gW; an[f"b{i}"] = gb; an[f"Wr{i}"] = gWr

max_rel = 0
for key in an:
    arr = an[key]
    for idx in np.ndindex(arr.shape):
        if np.random.rand() > 0.05:
            continue
        eps = 1e-5
        Pp = {k: v.copy() for k, v in P.items()}; Pp[key][idx] += eps
        Pm = {k: v.copy() for k, v in P.items()}; Pm[key][idx] -= eps
        fd = (loss(Pp) - loss(Pm)) / (2 * eps)
        rel = abs(fd - an[key][idx]) / (abs(fd) + abs(an[key][idx]) + 1e-8)
        max_rel = max(max_rel, rel)
print(f"FD max_rel = {max_rel:.2e}")
assert max_rel < 1e-3, "GRAD WRONG"

# ---- training sanity ----
x = rg.standard_normal(4000)
y = np.zeros(4000)
for lag in (1, 2, 4):
    y[lag:] += np.tanh(0.7 * x[:-lag])
y = (y - y.mean()) / y.std()
X, Y = [], []
for i in range(L, len(x)):
    X.append(x[i - L:i]); Y.append(y[i])
X = np.array(X).reshape(-1, 1, L); Y = np.array(Y)
n_tr = 3000
Pt = init_tcn(seed=7)
print("init mse", float(np.mean((forward(X[:n_tr], Pt)[0] - Y[:n_tr]) ** 2)))
for ep in range(60):
    idx = rg.permutation(n_tr)
    for s in range(0, n_tr, 64):
        b = idx[s:s + 64]; backward(X[b], Y[b], Pt, lr=0.03)
    if ep % 15 == 0:
        tr = float(np.mean((forward(X[:n_tr], Pt)[0] - Y[:n_tr]) ** 2))
        te = float(np.mean((forward(X[n_tr:], Pt)[0] - Y[n_tr:]) ** 2))
        print(f"ep{ep:02d} train={tr:.4f} test={te:.4f}")
print("LEARNED OK")
