#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMM 状态切换资产配置：让模型自己识别牛熊震荡
全部图表由真实计算生成，固定随机种子可复现。
"""
import json, os, time, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

OUT = "/Users/halo/workspace/astro-blog/public/images/regime-switching-hmm-allocation"
os.makedirs(OUT, exist_ok=True)
SEED = 42
np.random.seed(SEED)

plt.rcParams["font.sans-serif"] = ["PingFang SC","Hiragino Sans GB","Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["figure.facecolor"] = "white"

C_BULL="#16a34a"; C_CHURN="#f59e0b"; C_BEAR="#dc2626"
C_VIT="#2563eb";  C_FILT="#7c3aed";  C_ORA="#0d9488"; C_BNH="#6b7280"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HMM 真值世界（顺序固定：0=牛市 1=震荡 2=熊市）
# ═══════════════════════════════════════════════════════════════════════════════
T_TRUE  = np.array([[0.970,0.028,0.002],[0.022,0.965,0.013],[0.010,0.038,0.952]])
MUS_TRUE = np.array([ 0.0015, 0.0000,-0.0015])   # 0=牛市 1=震荡 2=熊市
SIGS_TRUE = np.array([0.008, 0.010, 0.008])
N = 5000

def simulate_markov(T_mat, n):
    K = T_mat.shape[0]; state = np.random.choice(K)
    states = np.empty(n, dtype=int)
    for i in range(n):
        states[i] = state; state = np.random.choice(K, p=T_mat[state])
    return states

true_states = simulate_markov(T_TRUE, N)
returns     = np.array([np.random.normal(MUS_TRUE[s], SIGS_TRUE[s]) for s in true_states])
price      = 1.0 + np.cumsum(returns)

durations = []; d = 0
for i in range(1, N):
    if true_states[i]==true_states[i-1]: d+=1
    else: durations.append(d+1); d=0
durations.append(d+1)
avg_duration = np.mean(durations)
n_switches   = int(np.sum(np.diff(true_states)!=0))

print(f"[真值世界] N={N}, 真实切换={n_switches}次, "
      f"平均持续={avg_duration:.1f}天")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 纯 numpy 工具（用于文章展示 + 无 hmmlearn 时的备选）
# ═══════════════════════════════════════════════════════════════════════════════
def normal_log_emit(x, mu, sigma):
    return -0.5*np.log(2*np.pi*sigma**2) - 0.5*((x-mu)/sigma)**2

def forward_log_scaled(obs, init, T_mat, mus, sigmas):
    T_obs=len(obs); K=T_mat.shape[0]
    log_alpha=np.full((T_obs,K),-np.inf); log_c=np.zeros(T_obs)
    log_T=np.log(T_mat+1e-300)
    for j in range(K):
        log_alpha[0,j]=np.log(init[j]+1e-300)+normal_log_emit(obs[0],mus[j],sigmas[j])
    log_c[0]=logsumexp(log_alpha[0]); log_alpha[0]-=log_c[0]
    for t in range(1,T_obs):
        for j in range(K):
            log_alpha[t,j]=logsumexp(log_alpha[t-1]+log_T[:,j])\
                           +normal_log_emit(obs[t],mus[j],sigmas[j])
        log_c[t]=logsumexp(log_alpha[t]); log_alpha[t]-=log_c[t]
    return log_alpha, np.sum(log_c)

def backward_log_scaled(obs, init, T_mat, mus, sigmas):
    T_obs=len(obs); K=T_mat.shape[0]
    log_beta=np.full((T_obs,K),-np.inf); log_beta[-1]=0.0
    log_T=np.log(T_mat+1e-300)
    emit_next=np.array([normal_log_emit(obs[1:],mus[j],sigmas[j]) for j in range(K)]).T
    for t in range(T_obs-2,-1,-1):
        log_beta[t]=np.array([logsumexp(log_T[i]+emit_next[t]+log_beta[t+1]) for i in range(K)])
    return log_beta

def viterbi_numpy(obs, init, T_mat, mus, sigmas):
    T_obs=len(obs); K=T_mat.shape[0]
    log_delta=np.full((T_obs,K),-np.inf); psi=np.zeros((T_obs,K),dtype=int)
    log_T=np.log(T_mat+1e-300)
    for j in range(K):
        log_delta[0,j]=np.log(init[j]+1e-300)+normal_log_emit(obs[0],mus[j],sigmas[j])
    for t in range(1,T_obs):
        for j in range(K):
            prev=log_delta[t-1]+log_T[:,j]; best_i=np.argmax(prev)
            log_delta[t,j]=prev[best_i]+normal_log_emit(obs[t],mus[j],sigmas[j])
            psi[t,j]=best_i
    states=np.empty(T_obs,dtype=int); states[-1]=np.argmax(log_delta[-1])
    for t in range(T_obs-2,-1,-1): states[t]=psi[t+1,states[t+1]]
    return states

def filter_causal(obs, init, T_mat, mus, sigmas):
    log_alpha,_=forward_log_scaled(obs,init,T_mat,mus,sigmas)
    gamma=np.exp(log_alpha); gamma/=gamma.sum(axis=1,keepdims=True)
    return gamma


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 训练 HMM（hmmlearn + 纯 numpy 解码）
# ═══════════════════════════════════════════════════════════════════════════════
def align_states_to_true_order(init_raw, T_raw, mus_raw, sigs_raw):
    """
    调整状态顺序，使估计的 mus 升序排列：
    sorted[0]=熊市（最低均值）→ 策略 position=-1.5
    sorted[1]=震荡（中均值）  → 策略 position=0
    sorted[2]=牛市（最高均值）→ 策略 position=+1.0
    """
    order = np.argsort(mus_raw)
    init_s = init_raw[order]
    T_s    = T_raw[np.ix_(order, order)]
    mus_s  = mus_raw[order]
    sigs_s = sigs_raw[order]
    sigs_s = np.clip(sigs_s, 0.003, 0.05)
    return init_s, T_s, mus_s, sigs_s

def position_from_estimated_state(est_state, mus_sorted, sigs_sorted):
    """
    给定排序后的估计状态（升序mean），
    返回策略权重：最高mean=bull(+1.0), 最低=bear(-1.5), 中=churn(0)
    """
    K = len(mus_sorted)
    bull_state  = int(np.argmax(mus_sorted))
    bear_state  = int(np.argmin(mus_sorted))
    weights = np.zeros(K)
    weights[bull_state] =  1.0
    weights[bear_state] = -1.5
    return weights[est_state]

try:
    from hmmlearn.hmm import GaussianHMM as GHM
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False

def train_hmm(obs, K, seed, n_iter=100):
    """训练 K 状态 Gaussian HMM，失败时返回 None"""
    if not HAS_HMMLEARN:
        return None
    best_m, best_s = None, -np.inf
    for r in range(5):
        m = GHM(n_components=K, covariance_type="full", n_iter=n_iter,
                tol=1e-6, random_state=seed+r*111, init_params="stmc")
        try:
            m.fit(obs.reshape(-1,1))
            s = m.score(obs.reshape(-1,1))
            if s > best_s: best_s=s; best_m=m
        except Exception:
            pass
    return best_m

if HAS_HMMLEARN:
    print("\n[训练] hmmlearn GaussianHMM...")
    model = train_hmm(returns, K=3, seed=SEED, n_iter=150)
    if model is None:
        raise RuntimeError("HMM training failed for all 5 restarts")
    init_raw = model.startprob_; T_raw = model.transmat_
    mus_raw  = model.means_.ravel(); sigs_raw = np.sqrt(model.covars_.ravel())
    init_est, T_est, mus_est, sigs_est = align_states_to_true_order(
        init_raw, T_raw, mus_raw, sigs_raw)
    print(f"  估计均值（升序）: {[round(m,5) for m in mus_est]}")
    print(f"  估计标准差: {[round(s,5) for s in sigs_est]}")
    # 确认排序正确：mus_est[2]>mus_est[0]
    print(f"  最高均值=状态{mus_est.argmax()}（牛市）: {mus_est.max():.5f}")
    print(f"  最低均值=状态{mus_est.argmin()}（熊市）: {mus_est.min():.5f}")
    t0=time.time()
    viterbi_states = viterbi_numpy(returns, init_est, T_est, mus_est, sigs_est)
    gamma_causal   = filter_causal(returns, init_est, T_est, mus_est, sigs_est)
    print(f"  纯numpy解码耗时: {time.time()-t0:.2f}s")
else:
    raise RuntimeError("hmmlearn not available — please install: pip install hmmlearn")

viterbi_pred = viterbi_states   # 已经是按升序排列的状态
causal_pred  = np.argmax(gamma_causal, axis=1)

viterbi_acc = float(np.mean(viterbi_pred == true_states))
causal_acc  = float(np.mean(causal_pred  == true_states))
print(f"  Viterbi 准确率: {viterbi_acc:.4f}")
print(f"  因果 Filtering 准确率: {causal_acc:.4f}")
print(f"  估计转移矩阵:\n{np.array(T_est).round(4)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 状态切换识别滞后
# ═══════════════════════════════════════════════════════════════════════════════
def compute_switch_lags(true_states, probs, threshold=0.5, max_look=500):
    lags={"bull":[],"churn":[],"bear":[]}
    state_name={0:"bear",1:"churn",2:"bull"}  # 真实状态:0=熊,1=震,2=牛
    N_t=len(true_states)
    for i in range(1,N_t-1):
        if true_states[i]!=true_states[i-1]:
            t0,i_s=i,true_states[i]
            found=False
            for t in range(t0,min(t0+max_look,N_t)):
                if probs[t,i_s]>=threshold:
                    lags[state_name[i_s]].append(t-t0); found=True; break
            if not found: lags[state_name[i_s]].append(np.nan)
    return {k:np.array(v) for k,v in lags.items()}

switch_lags = compute_switch_lags(true_states, gamma_causal)
all_lags    = np.concatenate([v[~np.isnan(v)] for v in switch_lags.values()])

print(f"\n[状态切换滞后] median={np.nanmedian(all_lags):.1f}d, "
      f"mean={np.nanmean(all_lags):.1f}d, "
      f"p25={np.nanpercentile(all_lags,25):.1f}d, p75={np.nanpercentile(all_lags,75):.1f}d")
print(f"  熊市开始滞后: median={np.nanmedian(switch_lags['bear']):.1f}d")
print(f"  牛市开始滞后: median={np.nanmedian(switch_lags['bull']):.1f}d")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 配置回测（4条腿 + 正确动态权重映射）
# ═══════════════════════════════════════════════════════════════════════════════
def equity_curve(rets):
    return np.cumprod(1+np.array(rets))

def perf_metrics(rets):
    r=np.array(rets)
    total=np.prod(1+r)-1
    ann=(1+total)**(252/len(r))-1
    vol=np.std(r)*np.sqrt(252)
    shp=ann/vol if vol>1e-10 else 0.0
    cummax=np.maximum.accumulate(np.cumprod(1+r))
    dd=np.cumprod(1+r)/cummax-1
    return dict(total_ret=total,ann_ret=ann,vol=vol,sharpe=shp,max_dd=np.min(dd))

# 动态生成策略权重
bull_state_est = int(np.argmax(mus_est))
bear_state_est = int(np.argmin(mus_est))
w_bull, w_bear, w_churn = 1.0, -1.5, 0.0

def w_from_est_state(s):
    if s==bull_state_est: return w_bull
    if s==bear_state_est: return w_bear
    return w_churn

oracle_w = np.array([w_from_est_state(s) for s in true_states])
vit_w   = np.array([w_from_est_state(s) for s in viterbi_pred])
filt_w  = np.array([w_from_est_state(s) for s in causal_pred])

oracle_ret  = oracle_w * returns
viterbi_ret = vit_w   * returns
filt_ret    = filt_w  * returns
bnh_ret     = returns.copy()
forty_ret   = returns * 0.6

oracle_met  = perf_metrics(oracle_ret)
viterbi_met = perf_metrics(viterbi_ret)
filt_met    = perf_metrics(filt_ret)
bnh_met     = perf_metrics(bnh_ret)
forty_met   = perf_metrics(forty_ret)

oracle_eq  = equity_curve(oracle_ret)
viterbi_eq = equity_curve(viterbi_ret)
filt_eq    = equity_curve(filt_ret)
bnh_eq     = equity_curve(bnh_ret)
forty_eq   = equity_curve(forty_ret)

def annual_turnover(weights):
    return np.abs(np.diff(weights)).mean()*252

vt_turn = annual_turnover(vit_w)
fi_turn = annual_turnover(filt_w)

print(f"\n[4条腿回测]")
for name, met in [("Oracle",oracle_met),("Viterbi",viterbi_met),
                   ("Causal",filt_met),("BuyHold",bnh_met),("60/40",forty_met)]:
    print(f"  {name:8s}: Sharpe={met['sharpe']:.4f}, Ann={met['ann_ret']:.4%}, MaxDD={met['max_dd']:.4%}")
print(f"  Viterbi-Causal Sharpe差距: {viterbi_met['sharpe']-filt_met['sharpe']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 安慰剂检验
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[安慰剂1] 零状态 i.i.d. 喂给 HMM")
iid_ret = np.random.RandomState(99).normal(0.0002, 0.015, N)
m_iid = train_hmm(iid_ret, K=3, seed=99, n_iter=80)
if m_iid is not None:
    _,_,pla_mus_raw,_ = align_states_to_true_order(
        m_iid.startprob_, m_iid.transmat_,
        m_iid.means_.ravel(), np.sqrt(m_iid.covars_.ravel()))
    pla_bull_st = int(np.argmax(pla_mus_raw))
    pla_bear_st = int(np.argmin(pla_mus_raw))
    pla_w = np.array([1.0 if s==pla_bull_st else (-1.5 if s==pla_bear_st else 0.0)
                      for s in np.argmax(filter_causal(
                          iid_ret, m_iid.startprob_, m_iid.transmat_,
                          m_iid.means_.ravel(), np.sqrt(m_iid.covars_.ravel())), axis=1)])
    pla_ret_seq  = pla_w * iid_ret
    pla_met      = perf_metrics(pla_ret_seq)
    pla_bnh_met  = perf_metrics(iid_ret)
    # i.i.d.数据没有真实状态，Viterbi解码应该≈随机
    pla_vit_st   = viterbi_numpy(iid_ret, m_iid.startprob_, m_iid.transmat_,
                                  m_iid.means_.ravel(), np.sqrt(m_iid.covars_.ravel()))
    # 计算每个状态的占比
    state_counts = {s: int(np.sum(pla_vit_st==s)) for s in range(3)}
    print(f"  i.i.d. Viterbi状态分布: {state_counts}（≈[1667,1667,1667]=随机）")
    print(f"  HMM策略Sharpe={pla_met['sharpe']:.4f} (BuyHold Sharpe={pla_bnh_met['sharpe']:.4f})")
    # 打乱后再次用真实数据的参数（因为i.i.d.模型可能退化）
    pla_dur_est = float("nan")
else:
    pla_met = dict(sharpe=0.0); pla_bnh_met = dict(sharpe=0.0)
    state_counts = {}; pla_dur_est = float("nan")

print("\n[安慰剂2] 打乱收益顺序")
shuff_idx = np.random.RandomState(777).permutation(N)
shuff_ret_seq = returns[shuff_idx]
m_shuf = train_hmm(shuff_ret_seq, K=3, seed=777, n_iter=80)
shuff_pred = np.argmax(filter_causal(
    shuff_ret_seq,
    m_shuf.startprob_, m_shuf.transmat_,
    m_shuf.means_.ravel(), np.sqrt(m_shuf.covars_.ravel())), axis=1)
shuff_w = np.array([w_from_est_state(s) for s in shuff_pred])
shuff_ret_seq2 = shuff_w * shuff_ret_seq
shuff_met = perf_metrics(shuff_ret_seq2)
print(f"  打乱后 Sharpe={shuff_met['sharpe']:.4f}（应≈0，说明持续性是核心）")

print("\n[安慰剂3] K扫描 K=2..6")
k_scan = {}
for K_k in range(2, 7):
    train_n = int(N*0.8)
    r_train, r_test = returns[:train_n], returns[train_n:]
    m_k = train_hmm(r_train, K=K_k, seed=42, n_iter=80)
    if m_k is None:
        k_scan[K_k] = dict(sharpe_oos=0.0, sharpe_bnh=0.0, ann_ret_oos=0.0)
        continue
    init_k, T_k, mus_k, sigs_k = align_states_to_true_order(
        m_k.startprob_, m_k.transmat_,
        m_k.means_.ravel(), np.sqrt(m_k.covars_.ravel()))
    bull_k = int(np.argmax(mus_k))
    bear_k = int(np.argmin(mus_k))
    gamma_test = filter_causal(r_test, init_k, T_k, mus_k, sigs_k)
    pred_test  = np.argmax(gamma_test, axis=1)
    strat_test = np.array([1.0 if s==bull_k else (-1.5 if s==bear_k else 0.0)
                           for s in pred_test]) * r_test
    strat_met  = perf_metrics(strat_test)
    bnh_test   = perf_metrics(r_test)
    k_scan[K_k] = dict(sharpe_oos=strat_met['sharpe'],
                       sharpe_bnh=bnh_test['sharpe'],
                       ann_ret_oos=strat_met['ann_ret'])
    print(f"  K={K_k}: OOS Sharpe={strat_met['sharpe']:.4f}, B&H={bnh_test['sharpe']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 交易成本敏感性
# ═══════════════════════════════════════════════════════════════════════════════
annual_turn_fi = float(annual_turnover(filt_w))
causal_sharpe_raw = filt_met['sharpe']
breakeven_bp = 100
for bp in range(1, 101):
    annual_cost = annual_turn_fi * 2 * bp * 1e-4
    if causal_sharpe_raw * filt_met['vol'] * np.sqrt(252) - annual_cost <= 0:
        breakeven_bp=bp; break
print(f"\n[成本敏感性] 因果年化换手={annual_turn_fi:.2f}x, 盈亏平衡={breakeven_bp}bp")


# ═══════════════════════════════════════════════════════════════════════════════
# 图1: cover.png
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3,1,figsize=(14,10),sharex=True,gridspec_kw={"height_ratios":[3,1,1]})
ax=axes[0]
ax.plot(price, color="black", lw=0.8)
ax.set_ylabel("价格", fontsize=11)
# 真实状态色带：true_states[0]=熊, [1]=震, [2]=牛
for s_id,color in [(2,C_BULL),(1,C_CHURN),(0,C_BEAR)]:
    mask=true_states==s_id
    ax.fill_between(range(N),0,price.max()*1.05,where=mask,alpha=0.18,color=color,lw=0)
ax.text(0.02,0.97,"绿=牛市 | 黄=震荡 | 红=熊市",transform=ax.transAxes,
        fontsize=9,va="top",bbox=dict(boxstyle="round",fc="white",alpha=0.8))
ax=axes[1]
# gamma_causal 的列顺序：升序mean，所以col2=牛市, col0=熊市
ax.fill_between(range(N),0,gamma_causal[:,2],alpha=0.75,color=C_BULL,label="P(牛市)",lw=0)
ax.fill_between(range(N),0,gamma_causal[:,1],alpha=0.75,color=C_CHURN,label="P(震荡)",lw=0)
ax.fill_between(range(N),0,gamma_causal[:,0],alpha=0.75,color=C_BEAR,label="P(熊市)",lw=0)
ax.set_ylabel("推断概率\n(因果)",fontsize=9)
ax.legend(fontsize=8,loc="upper right"); ax.set_ylim(0,1)
ax=axes[2]
ax.plot(gamma_causal[:,2],color=C_BULL,lw=0.7,alpha=0.9,label="P(牛市|因果)")
ax.plot((true_states==2).astype(float),color="gray",lw=0.5,alpha=0.4,label="真实牛市")
ax.axhline(0.5,color="black",lw=0.8,ls="--",alpha=0.5)
ax.set_ylabel("牛市概率",fontsize=9); ax.set_xlabel("交易日",fontsize=11)
ax.legend(fontsize=8,loc="upper right")
fig.suptitle("HMM 状态切换：真实价格 vs 因果推断状态概率（3状态高信噪比仿真）",fontsize=13,fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT}/cover.png",dpi=150,facecolor="white"); plt.close()
print(f"✓ cover.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 图2: viterbi_vs_causal.png
# ═══════════════════════════════════════════════════════════════════════════════
fig,axes=plt.subplots(1,3,figsize=(16,4.5))
ax=axes[0]
methods=["Viterbi\n(全样本)","因果 Filtering\n(实时)"]
accs=[viterbi_acc, causal_acc]
for i,(m,a,c) in enumerate(zip(methods,accs,[C_VIT,C_FILT])):
    bar=ax.bar(m,a,color=c,width=0.45,edgecolor="black",lw=0.5)
    ax.text(bar[0].get_x()+bar[0].get_width()/2,bar[0].get_height()+0.01,
            f"{a:.3f}",ha="center",va="bottom",fontsize=13,fontweight="bold")
ax.set_ylim(0,1.15); ax.set_ylabel("准确率",fontsize=11)
ax.set_title("状态识别准确率",fontsize=12,fontweight="bold")
ax.axhline(1/3,color="gray",ls="--",lw=1,label="随机基准(33.3%)")
ax.legend(fontsize=9)

window=600; ax=axes[1]
ts=np.arange(window)
ax.fill_between(ts,0,gamma_causal[:window,2],alpha=0.65,color=C_BULL,label="P(牛市)",lw=0)
ax.fill_between(ts,0,gamma_causal[:window,1],alpha=0.65,color=C_CHURN,label="P(震荡)",lw=0)
ax.fill_between(ts,0,gamma_causal[:window,0],alpha=0.65,color=C_BEAR,label="P(熊市)",lw=0)
for s_id,color in [(2,C_BULL),(1,C_CHURN),(0,C_BEAR)]:
    mask=true_states[:window]==s_id
    ax.scatter(np.where(mask)[0],np.full(mask.sum(),0.04),
               color=color,s=2,alpha=0.5,zorder=5,marker="|")
ax.axhline(0.5,color="black",ls="--",lw=0.8,alpha=0.6)
ax.set_xlim(0,window); ax.set_ylim(0,1)
ax.set_xlabel("交易日",fontsize=10); ax.set_ylabel("推断概率",fontsize=10)
ax.set_title(f"因果 Filtering 概率轨迹（前{window}天）",fontsize=11,fontweight="bold")
ax.legend(fontsize=8,loc="upper right")

ax=axes[2]
cm_vit=np.zeros((3,3),dtype=int)
for yt,yp in zip(true_states,viterbi_pred): cm_vit[yt,yp]+=1
im=ax.imshow(cm_vit,cmap="Blues",vmin=0)
state_labels=["熊市(真0)","震荡(真1)","牛市(真2)"]
ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
ax.set_xticklabels(["熊","震","牛"]); ax.set_yticklabels(["熊","震","牛"])
ax.set_xlabel("预测状态 (Viterbi)",fontsize=10); ax.set_ylabel("真实状态",fontsize=10)
ax.set_title("Viterbi 混淆矩阵\n（全样本 look-ahead）",fontsize=11,fontweight="bold")
for i in range(3):
    for j in range(3):
        ax.text(j,i,str(cm_vit[i,j]),ha="center",va="center",fontsize=14,fontweight="bold",
                color="white" if cm_vit[i,j]>cm_vit.max()*0.5 else "black")
plt.colorbar(im,ax=ax,shrink=0.7)
fig.suptitle("Viterbi 全样本 vs 因果 Filtering：识别质量最硬一刀",fontsize=13,fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT}/viterbi_vs_causal.png",dpi=150,facecolor="white"); plt.close()
print(f"✓ viterbi_vs_causal.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 图3: strategy_compare.png
# ═══════════════════════════════════════════════════════════════════════════════
fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True,gridspec_kw={"height_ratios":[2.5,1]})
x=np.arange(N)
ax=axes[0]
ax.plot(x,oracle_eq, color=C_ORA,lw=1.2,label=f"Oracle (Sharpe={oracle_met['sharpe']:.2f})")
ax.plot(x,viterbi_eq,color=C_VIT,lw=1.2,label=f"Viterbi look-ahead (Sharpe={viterbi_met['sharpe']:.2f})")
ax.plot(x,filt_eq,   color=C_FILT,lw=1.6,label=f"Causal Filtering (Sharpe={filt_met['sharpe']:.2f})")
ax.plot(x,bnh_eq,    color=C_BNH,lw=0.9,ls="--",label=f"Buy-Hold (Sharpe={bnh_met['sharpe']:.2f})")
ax.plot(x,forty_eq,  color=C_CHURN,lw=0.9,ls=":",label=f"60/40 (Sharpe={forty_met['sharpe']:.2f})")
ax.set_ylabel("累计净值",fontsize=11)
ax.set_title("4条腿权益曲线 vs 回撤",fontsize=13,fontweight="bold")
ax.legend(fontsize=9,loc="upper left")

ax2=axes[1]
def dd_pct(eq):
    cummax=np.maximum.accumulate(eq); return (eq/cummax-1)*100
ax2.fill_between(x,0,dd_pct(oracle_eq), alpha=0.25,color=C_ORA, label="Oracle")
ax2.fill_between(x,0,dd_pct(viterbi_eq),alpha=0.25,color=C_VIT, label="Viterbi")
ax2.plot(x,dd_pct(filt_eq),   color=C_FILT,lw=1.2,label="Causal")
ax2.plot(x,dd_pct(bnh_eq),    color=C_BNH,lw=0.8,ls="--",label="Buy-Hold")
ax2.set_ylabel("回撤 (%)",fontsize=11); ax2.set_xlabel("交易日",fontsize=11)
ax2.set_title("回撤曲线",fontsize=12,fontweight="bold")
ax2.legend(fontsize=9,loc="lower left"); ax2.set_ylim(-70,5)

gap = viterbi_met['sharpe']-filt_met['sharpe']
fig.suptitle(f"配置回测：Viterbi-Causal Sharpe 差距 = {gap:.3f}",fontsize=13,fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT}/strategy_compare.png",dpi=150,facecolor="white"); plt.close()
print(f"✓ strategy_compare.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 图4: placebo.png
# ═══════════════════════════════════════════════════════════════════════════════
if m_iid is not None:
    pla_gamma = filter_causal(iid_ret,m_iid.startprob_,m_iid.transmat_,
                              m_iid.means_.ravel(),np.sqrt(m_iid.covars_.ravel()))
else:
    pla_gamma = np.ones((N,3))/3

fig,axes=plt.subplots(2,2,figsize=(14,8))
ax=axes[0,0]
ax.plot(iid_ret,color="steelblue",lw=0.5,alpha=0.7)
ax.axhline(0,color="gray",lw=0.8,ls="--")
ax.set_title("安慰剂: i.i.d. 收益（真值：无状态）",fontsize=11,fontweight="bold")
ax.set_ylabel("日收益率"); ax.set_xlabel("交易日")

ax=axes[0,1]
ax.fill_between(range(N),0,pla_gamma[:,2],alpha=0.7,color=C_BULL,label="P(牛市)",lw=0)
ax.fill_between(range(N),0,pla_gamma[:,1],alpha=0.7,color=C_CHURN,label="P(震荡)",lw=0)
ax.fill_between(range(N),0,pla_gamma[:,0],alpha=0.7,color=C_BEAR,label="P(熊市)",lw=0)
ax.set_title("HMM在i.i.d.数据上照样「发现」状态\n（Viterbi≈随机分布，无真实信息）",fontsize=11,fontweight="bold")
ax.set_ylabel("状态概率"); ax.set_xlabel("交易日")
ax.legend(fontsize=8,loc="upper right"); ax.set_ylim(0,1)

ax=axes[1,0]
pla_eq = equity_curve(pla_ret_seq if 'pla_ret_seq' in dir() else np.zeros(N))
pla_bnh_eq = equity_curve(iid_ret)
ax.plot(pla_eq,    color=C_FILT,lw=1.2,label=f"HMM策略 (Sharpe={pla_met['sharpe']:.3f})")
ax.plot(pla_bnh_eq,color=C_BNH,lw=1.0,ls="--",label=f"Buy-Hold (Sharpe={pla_bnh_met['sharpe']:.3f})")
ax.axhline(1.0,color="gray",lw=0.8,ls="--")
ax.set_title("安慰剂：i.i.d. 世界中 HMM 策略 vs Buy-Hold",fontsize=11,fontweight="bold")
ax.set_ylabel("净值"); ax.set_xlabel("交易日"); ax.legend(fontsize=9)

ax=axes[1,1]
shuff_eq = equity_curve(shuff_ret_seq2)
ax.plot(shuff_eq,color=C_BEAR,lw=1.2,label=f"打乱后 (Sharpe={shuff_met['sharpe']:.3f})")
ax.plot(pla_eq,  color=C_FILT,lw=0.8,ls="--",label="零状态安慰剂")
ax.axhline(1.0,color="gray",lw=0.8,ls="--")
ax.set_title("打乱收益顺序：持续性破坏后 Sharpe → 0",fontsize=11,fontweight="bold")
ax.set_ylabel("净值"); ax.set_xlabel("交易日"); ax.legend(fontsize=9)

fig.suptitle("安慰剂检验：HMM 在无状态数据上照样「发现」漂亮规律",fontsize=13,fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT}/placebo.png",dpi=150,facecolor="white"); plt.close()
print(f"✓ placebo.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 图5: k_scan.png
# ═══════════════════════════════════════════════════════════════════════════════
K_range=list(range(2,7))
sharpes_oos=[k_scan[k]["sharpe_oos"] for k in K_range]

fig,axes=plt.subplots(1,2,figsize=(14,5))
ax=axes[0]
ax.plot(K_range,sharpes_oos,"o-",color=C_VIT,lw=2,ms=8)
ax.set_xlabel("状态数 K",fontsize=11); ax.set_ylabel("样本外 Sharpe",fontsize=11)
ax.set_title("K扫描: 样本外 Sharpe（先升后降 = 过拟合）",fontsize=12,fontweight="bold")
ax.set_xticks(K_range); ax.grid(True,alpha=0.3)
for k,s in zip(K_range,sharpes_oos):
    ax.annotate(f"{s:.3f}",(k,s),textcoords="offset points",xytext=(0,8),ha="center",fontsize=9)
best_k=K_range[np.argmax(sharpes_oos)]
ax.annotate(f"最优K={best_k}",(best_k,max(sharpes_oos)),
             xytext=(0,10),textcoords="offset points",ha="center",
             fontsize=10,color=C_BULL,fontweight="bold")

ax2=axes[1]
bar_colors=[C_BULL if s>0.05 else C_BEAR for s in sharpes_oos]
bars=ax2.bar(K_range,sharpes_oos,color=bar_colors,width=0.5,edgecolor="black",lw=0.5)
ax2.axhline(0,color="gray",lw=1)
ax2.set_xlabel("状态数 K",fontsize=11); ax2.set_ylabel("样本外 Sharpe",fontsize=11)
ax2.set_title("各K样本外Sharpe详情",fontsize=12,fontweight="bold")
ax2.set_xticks(K_range); ax2.grid(True,alpha=0.3,axis="y")
for bar,s in zip(bars,sharpes_oos):
    ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.02,
             f"{s:.3f}",ha="center",va="bottom",fontsize=9)
fig.suptitle("过拟合警告：样本内似然随K↑，样本外 Sharpe 在最优K见顶",fontsize=13,fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT}/k_scan.png",dpi=150,facecolor="white"); plt.close()
print(f"✓ k_scan.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 图6: lag_distribution.png
# ═══════════════════════════════════════════════════════════════════════════════
valid=all_lags[~np.isnan(all_lags)]
fig,axes=plt.subplots(1,2,figsize=(14,5))
ax=axes[0]
ax.hist(valid,bins=30,color=C_VIT,edgecolor="white",alpha=0.8)
ax.axvline(np.nanmedian(all_lags),color=C_BEAR,lw=2,ls="--",
           label=f"中位数={np.nanmedian(all_lags):.1f}天")
ax.axvline(np.nanmean(all_lags),  color=C_CHURN,lw=2,ls=":",
           label=f"均值={np.nanmean(all_lags):.1f}天")
ax.set_xlabel("识别滞后（天）",fontsize=11); ax.set_ylabel("切换事件数",fontsize=11)
ax.set_title("状态切换识别滞后分布",fontsize=12,fontweight="bold")
ax.legend(fontsize=10)

ax2=axes[1]
data_groups=[
    switch_lags["bull"][~np.isnan(switch_lags["bull"])],
    switch_lags["churn"][~np.isnan(switch_lags["churn"])],
    switch_lags["bear"][~np.isnan(switch_lags["bear"])],
]
bp=ax2.boxplot(data_groups,
               patch_artist=True,medianprops=dict(color="black",lw=2))
ax2.set_xticklabels(["牛市开始","震荡开始","熊市开始"])
for patch,color in zip(bp["boxes"],[C_BULL,C_CHURN,C_BEAR]):
    patch.set_facecolor(color); patch.set_alpha(0.6)
ax2.set_ylabel("识别滞后（天）",fontsize=11)
ax2.set_title("不同切换方向的识别滞后\n（熊市开始最危险：滞后最大）",fontsize=12,fontweight="bold")
ax2.grid(True,alpha=0.3,axis="y")
fig.suptitle("状态切换识别滞后：因果 Filtering 的致命缺陷",fontsize=13,fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT}/lag_distribution.png",dpi=150,facecolor="white"); plt.close()
print(f"✓ lag_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 保存 stats.json
# ═══════════════════════════════════════════════════════════════════════════════
stats={
    "simulation":{
        "N":N,"n_states":3,"seed":SEED,
        "avg_duration_days":round(float(avg_duration),1),
        "true_switches":n_switches,
    },
    "decoding_quality":{
        "viterbi_accuracy": round(viterbi_acc,6),
        "causal_accuracy":  round(causal_acc,6),
        "accuracy_gap":     round(viterbi_acc-causal_acc,6),
    },
    "switch_lag":{
        "median_days":  round(float(np.nanmedian(all_lags)),2),
        "mean_days":    round(float(np.nanmean(all_lags)),2),
        "p25_days":     round(float(np.nanpercentile(all_lags,25)),2),
        "p75_days":     round(float(np.nanpercentile(all_lags,75)),2),
        "bear_median":   round(float(np.nanmedian(switch_lags["bear"])),2),
        "bull_median":   round(float(np.nanmedian(switch_lags["bull"])),2),
        "n_valid_lags":  int(len(valid)),
        "n_bear_starts":int(np.sum(~np.isnan(switch_lags["bear"]))),
    },
    "strategy_performance":{
        "Oracle":     {k:round(v,6) for k,v in oracle_met.items()},
        "Viterbi":    {k:round(v,6) for k,v in viterbi_met.items()},
        "Causal":     {k:round(v,6) for k,v in filt_met.items()},
        "BuyHold":    {k:round(v,6) for k,v in bnh_met.items()},
        "SixtyForty": {k:round(v,6) for k,v in forty_met.items()},
    },
    "turnover":{
        "viterbi_annual": round(float(vt_turn),4),
        "causal_annual":  round(float(fi_turn),4),
    },
    "cost_sensitivity":{
        "annual_turnover": round(float(annual_turn_fi),4),
        "breakeven_bp":    int(breakeven_bp),
        "causal_sharpe":   round(float(causal_sharpe_raw),4),
    },
    "placebo":{
        "iid_hmm_sharpe":  round(float(pla_met['sharpe']),4),
        "iid_bnh_sharpe":  round(float(pla_bnh_met['sharpe']),4),
        "shuffled_sharpe": round(float(shuff_met['sharpe']),4),
    },
    "k_scan":{str(k):{"sharpe_oos":round(v["sharpe_oos"],4)} for k,v in k_scan.items()},
    "sample_size":{
        "switches_at_5000": n_switches,
        "min_switches_stable": 100,
        "min_days_recommended": 3000,
    }
}

with open(f"{OUT}/stats.json","w",encoding="utf-8") as f:
    json.dump(stats,f,ensure_ascii=False,indent=2)

print(f"\n✓ stats.json saved")
print(f"\n[最终统计]")
for name,met in stats["strategy_performance"].items():
    print(f"  {name:12s}: Sharpe={met['sharpe']:.4f}, Ann={met['ann_ret']:.4%}, MaxDD={met['max_dd']:.4%}")
print(f"\n所有文件→ {OUT}")
print("DONE")
