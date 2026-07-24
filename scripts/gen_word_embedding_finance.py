import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)
OUT = "public/images/word-embedding-finance"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# Synthetic "announcement" corpus.
# Three latent topics, each with its own vocabulary that co-occurs:
#   利好(pos) / 利空(neg) / 中性(neu)
# Words within a topic co-occur; cross-topic co-occurrence is rare.
# A word2vec skip-gram should place same-topic words near each other.
# ---------------------------------------------------------------
pos_words = ["超预期", "增长", "回购", "中标", "扩产", "订单", "提价", "盈利", "分红", "创新高"]
neg_words = ["下滑", "亏损", "减持", "诉讼", "违约", "预亏", "商誉", "退市", "问询", "停产"]
neu_words = ["公告", "会议", "换届", "选举", "变更", "登记", "备案", "披露", "参会", "议案"]
topics = {"利好": pos_words, "利空": neg_words, "中性": neu_words}
vocab = pos_words + neg_words + neu_words
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
topic_of = {}
for t, ws in topics.items():
    for w in ws:
        topic_of[w] = t

# Generate sentences: pick a topic, sample words mostly from it + a little leakage
def make_sentence(topic, length=8, leak=0.08):
    words = []
    for _ in range(length):
        if rng.random() < leak:
            src = rng.choice(list(topics.keys()))
        else:
            src = topic
        words.append(rng.choice(topics[src]))
    return words

n_sent = 4000
sentences = []
sent_topic = []
for _ in range(n_sent):
    t = rng.choice(list(topics.keys()))
    sentences.append(make_sentence(t))
    sent_topic.append(t)

# ---------------------------------------------------------------
# Skip-gram with negative sampling (pure numpy).
#   maximize sigma(v_c . v_w) for real pairs, sigma(-v_c . v_neg) for negs
# ---------------------------------------------------------------
D = 16          # embedding dim
window = 2
n_neg = 5
lr = 0.05
epochs = 5

# unigram^0.75 negative sampling table
counts = np.zeros(V)
for s in sentences:
    for w in s:
        counts[w2i[w]] += 1
neg_p = counts ** 0.75
neg_p /= neg_p.sum()

Win = rng.normal(0, 0.1, (V, D))   # center (input) vectors
Wout = rng.normal(0, 0.1, (V, D))  # context (output) vectors

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# build training pairs
pairs = []
for s in sentences:
    ids = [w2i[w] for w in s]
    for i, c in enumerate(ids):
        lo = max(0, i - window); hi = min(len(ids), i + window + 1)
        for j in range(lo, hi):
            if j != i:
                pairs.append((c, ids[j]))
pairs = np.array(pairs)
print("training pairs:", len(pairs))

loss_hist = []
for ep in range(epochs):
    perm = rng.permutation(len(pairs))
    tot = 0.0
    for k in perm:
        c, o = pairs[k]
        negs = rng.choice(V, size=n_neg, p=neg_p)
        vc = Win[c]
        # positive
        so = sigmoid(vc @ Wout[o])
        g_pos = (so - 1.0)
        # negatives
        sn = sigmoid(Win[c] @ Wout[negs].T)
        g_neg = sn
        # grads
        grad_c = g_pos * Wout[o] + (g_neg[:, None] * Wout[negs]).sum(0)
        Wout[o] -= lr * g_pos * vc
        Wout[negs] -= lr * g_neg[:, None] * vc
        Win[c] -= lr * grad_c
        tot += -np.log(so + 1e-9) - np.log(1 - sn + 1e-9).sum()
    loss_hist.append(tot / len(pairs))
    print(f"epoch {ep+1} loss {loss_hist[-1]:.4f}")

emb = Win.copy()
emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

# ---------------------------------------------------------------
# Evaluation 1: within-topic cosine >> cross-topic cosine
# ---------------------------------------------------------------
cos = emb @ emb.T
within, cross = [], []
for a in range(V):
    for b in range(a + 1, V):
        if topic_of[vocab[a]] == topic_of[vocab[b]]:
            within.append(cos[a, b])
        else:
            cross.append(cos[a, b])
within, cross = np.array(within), np.array(cross)
print(f"within-topic cos {within.mean():.3f}  cross-topic cos {cross.mean():.3f}")

# ---------- FIG 1 (cover): 2D PCA of embeddings, colored by topic ----------
Xc = emb - emb.mean(0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
P = Xc @ Vt[:2].T
colors = {"利好": "#2e7d32", "利空": "#c62828", "中性": "#616161"}
fig, ax = plt.subplots(figsize=(9.5, 7))
for t in topics:
    idx = [i for i, w in enumerate(vocab) if topic_of[w] == t]
    ax.scatter(P[idx, 0], P[idx, 1], s=90, c=colors[t], label=t, alpha=0.85,
               edgecolors="k", linewidths=0.5)
for i, w in enumerate(vocab):
    ax.annotate(w, (P[i, 0], P[i, 1]), fontsize=9,
                xytext=(4, 3), textcoords="offset points")
ax.set_title("词嵌入把语义相近的公告用词聚成簇（PCA 二维投影）")
ax.set_xlabel("主成分 1"); ax.set_ylabel("主成分 2")
ax.legend(title="真实语义类别"); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{OUT}/cover.png", dpi=120); plt.close(fig)

# ---------- FIG 2: within vs cross cosine distribution ----------
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(within, bins=25, alpha=0.75, color="#4C72B0", label=f"同类词对 (均值 {within.mean():.2f})")
ax.hist(cross, bins=25, alpha=0.75, color="#C44E52", label=f"异类词对 (均值 {cross.mean():.2f})")
ax.set_title("余弦相似度：同语义类词对显著高于跨类词对")
ax.set_xlabel("余弦相似度"); ax.set_ylabel("词对数")
ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/cosine_dist.png", dpi=120); plt.close(fig)

# ---------------------------------------------------------------
# Evaluation 2: the CORE value of embeddings = generalize to words
#   NOT in your lexicon. Split each class's vocab into:
#     known (first 5)  -> in the analyst's hand-built lexicon
#     novel (last 5)   -> NEW jargon never added to the lexicon
#   Held-out announcements use ONLY novel words: bag-of-words scores
#   them all 0 (IC~0), but embeddings still work because novel words
#   sit near known ones, so the sentiment axis still projects.
# ---------------------------------------------------------------
known_pos, novel_pos = pos_words[:5], pos_words[5:]
known_neg, novel_neg = neg_words[:5], neg_words[5:]

# sentiment axis built ONLY from KNOWN words (what an analyst could label)
sent_axis = emb[[w2i[w] for w in known_pos]].mean(0) - \
            emb[[w2i[w] for w in known_neg]].mean(0)
sent_axis /= np.linalg.norm(sent_axis)

def doc_vec(s):
    return emb[[w2i[w] for w in s]].mean(0)

known_lex_pos = set(known_pos); known_lex_neg = set(known_neg)
def bow_factor(s):
    return sum(w in known_lex_pos for w in s) - sum(w in known_lex_neg for w in s)

def ic(x, y):
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    return float((x * y).mean())

topic_sign = {"利好": 1.0, "利空": -1.0, "中性": 0.0}

# (a) in-sample: announcements mixing known + novel words
ret = np.array([topic_sign[t] for t in sent_topic]) * 1.0 + rng.normal(0, 1.0, n_sent)
emb_factor = np.array([doc_vec(s) @ sent_axis for s in sentences])
bow = np.array([bow_factor(s) for s in sentences], dtype=float)
ic_emb = ic(emb_factor, ret)
ic_bow = ic(bow, ret)

# (b) held-out: announcements using ONLY novel (out-of-lexicon) words
novel_topics = {"利好": novel_pos, "利空": novel_neg, "中性": neu_words[5:]}
n_hold = 1500
hold_sents, hold_topic = [], []
for _ in range(n_hold):
    t = rng.choice(list(novel_topics.keys()))
    hold_sents.append([rng.choice(novel_topics[t]) for _ in range(8)])
    hold_topic.append(t)
hold_ret = np.array([topic_sign[t] for t in hold_topic]) * 1.0 + rng.normal(0, 1.0, n_hold)
emb_hold = np.array([doc_vec(s) @ sent_axis for s in hold_sents])
bow_hold = np.array([bow_factor(s) for s in hold_sents], dtype=float)
ic_emb_hold = ic(emb_hold, hold_ret)
ic_bow_hold = ic(bow_hold, hold_ret) if bow_hold.std() > 1e-9 else 0.0

print(f"[in-sample] IC emb {ic_emb:.3f}  bow {ic_bow:.3f}")
print(f"[held-out novel-vocab] IC emb {ic_emb_hold:.3f}  bow {ic_bow_hold:.3f}")

# ---------- FIG 3: IC in-sample vs held-out novel vocab + held-out scatter ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(2); w = 0.35
axes[0].bar(x - w/2, [ic_emb, ic_emb_hold], w, color="#4C72B0", label="词嵌入语义因子")
axes[0].bar(x + w/2, [ic_bow, ic_bow_hold], w, color="#C44E52", label="词典计数(不完整)")
axes[0].set_xticks(x)
axes[0].set_xticklabels(["混合用词\n(含词典内词)", "仅新词\n(全在词典外)"])
axes[0].set_title("信息系数 IC：嵌入因子 vs 词典计数")
axes[0].set_ylabel("IC（与公告后收益的相关）"); axes[0].legend()
for xi, v in zip([x[0]-w/2, x[1]-w/2, x[0]+w/2, x[1]+w/2],
                 [ic_emb, ic_emb_hold, ic_bow, ic_bow_hold]):
    axes[0].text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)

order = np.argsort(emb_hold)
nb = 20
bx = [b.mean() for b in np.array_split(emb_hold[order], nb)]
by = [b.mean() for b in np.array_split(hold_ret[order], nb)]
axes[1].scatter(emb_hold, hold_ret, s=6, alpha=0.15, color="gray")
axes[1].plot(bx, by, "o-", color="#4C72B0", label="分组均值")
axes[1].axhline(0, color="k", lw=0.6)
axes[1].set_title("仅用『词典外新词』的公告，嵌入因子仍能排序收益")
axes[1].set_xlabel("文档情绪投影值"); axes[1].set_ylabel("公告后收益")
axes[1].legend()
fig.tight_layout(); fig.savefig(f"{OUT}/factor_ic.png", dpi=120); plt.close(fig)

# ---------- FIG 4: dimension sensitivity ----------
def quick_train(D, epochs=3):
    Wi = rng.normal(0, 0.1, (V, D)); Wo = rng.normal(0, 0.1, (V, D))
    for _ in range(epochs):
        for k in rng.permutation(len(pairs))[:len(pairs)//2]:
            c, o = pairs[k]
            negs = rng.choice(V, size=n_neg, p=neg_p)
            vc = Wi[c]
            so = sigmoid(vc @ Wo[o]); g_pos = so - 1.0
            sn = sigmoid(Wi[c] @ Wo[negs].T); g_neg = sn
            grad_c = g_pos * Wo[o] + (g_neg[:, None] * Wo[negs]).sum(0)
            Wo[o] -= lr * g_pos * vc
            Wo[negs] -= lr * g_neg[:, None] * vc
            Wi[c] -= lr * grad_c
    e = Wi / (np.linalg.norm(Wi, axis=1, keepdims=True) + 1e-9)
    c2 = e @ e.T
    wi, cr = [], []
    for a in range(V):
        for b in range(a + 1, V):
            (wi if topic_of[vocab[a]] == topic_of[vocab[b]] else cr).append(c2[a, b])
    return np.mean(wi) - np.mean(cr)   # separation margin

dims = [2, 4, 8, 16, 32, 64]
seps = [quick_train(d) for d in dims]
print("dim seps:", list(zip(dims, [round(s, 3) for s in seps])))
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(dims, seps, "o-", color="#55A868")
ax.set_xscale("log", base=2)
ax.set_title("嵌入维度敏感性：同类-异类相似度差随维度先升后平")
ax.set_xlabel("嵌入维度 D（log2 轴）"); ax.set_ylabel("同类均值 − 异类均值")
ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(f"{OUT}/dim_sensitivity.png", dpi=120); plt.close(fig)

print("saved figures to", OUT)
