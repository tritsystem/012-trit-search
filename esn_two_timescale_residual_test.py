#!/usr/bin/env python3
"""Residual binary/ternary stage test -- the untried candidate fix named
in ternary-torus-arnold-finding.md for P1's failure (esn_two_timescale_bench.py:
protect readout NMSE 0.138 vs dense 0.0038, "sign-only weights can't do
precision regression").

IDEA: a SECOND independent TwoTimescaleLinear stage, summed with the
first, trained JOINTLY on the same online task loss -- backprop through
the sum means stage 2 naturally learns to correct whatever stage 1's
sign-only weights can't represent (the residual), the same principle as
residual/multi-stage quantization in the compression literature. No new
mechanism invented: this is two of the EXACT SAME already-validated
TwoTimescaleLinear layers, summed.

Cost: ~2x the weight-storage of single-stage protect (~3.2 bits/weight
instead of ~1.6), still far short of dense's 32 bits/weight -- an
honest ~10x compression instead of ~20x if this works, not free.

Same task/reservoir/dense/oracle setup as esn_two_timescale_bench.py,
for a directly comparable number, not a different benchmark.

PRE-REGISTERED (before running): residual-stage tail-NMSE meaningfully
better than single-stage protect's 0.138, ideally within 2x of dense's
0.0038 (P1's original bar). DISCONFIRM: residual ~= single-stage (the
correction has nothing left to learn -- sign-only is a hard structural
ceiling, not a capacity-starved one).
"""
import numpy as np
import torch
import torch.nn.functional as F

from tritkit.twotimescale import TwoTimescaleLinear

torch.manual_seed(0)
rng = np.random.default_rng(0)


def mackey_glass(total, t_regime=600, taus=(17, 30)):
    hist = 31
    x = list(1.2 + 0.2 * rng.standard_normal(hist))
    tau_seq = []
    for t in range(total):
        tau = taus[(t // t_regime) % len(taus)]
        tau_seq.append(tau)
        xt, xd = x[-1], x[-tau]
        x.append(xt + (0.2 * xd / (1 + xd ** 10) - 0.1 * xt))
    return np.array(x[hist:]), np.array(tau_seq)


TOTAL, T_REG = 24000, 600
series, tau_seq = mackey_glass(TOTAL + 1, T_REG)
series = (series - series.mean()) / series.std()

N_RES = 100
Wres = rng.standard_normal((N_RES, N_RES)) * (rng.random((N_RES, N_RES)) < 0.1)
Wres *= 0.9 / max(abs(np.linalg.eigvals(Wres)))
win = 0.5 * rng.standard_normal(N_RES)
LEAK = 0.3


def states(sig):
    s = np.zeros(N_RES)
    out = np.empty((len(sig), N_RES))
    for t, v in enumerate(sig):
        s = (1 - LEAK) * s + LEAK * np.tanh(Wres @ s + win * v)
        out[t] = s
    return out


S = states(series[:-1])
Y = series[1:]
WASH = 300
S, Y = S[WASH:], Y[WASH:]
T = len(Y)
St = torch.tensor(S, dtype=torch.float32)
Yt = torch.tensor(Y, dtype=torch.float32)


def metrics(err2):
    var = Y.var()
    return err2[T // 2:].mean() / var


def run_dense(lr):
    w = np.zeros(N_RES + 1)
    e2 = np.empty(T)
    for t in range(T):
        s = np.append(S[t], 1.0)
        e = Y[t] - w @ s
        e2[t] = e * e
        w += lr * e * s
    return e2


def run_ttt_single(lr, accum=8, t_gate=64):
    """Single-stage protect, reproduced here (same params as the
    original) for a same-run, directly comparable baseline."""
    lyr = TwoTimescaleLinear(N_RES, 1, density=0.5, bias=True, evidence_beta=0.9)
    opt = torch.optim.SGD(lyr.parameters(), lr=lr)
    e2 = np.empty(T)
    n_acc = 0
    for t in range(T):
        out = lyr(St[t:t + 1])
        loss = F.mse_loss(out, Yt[t:t + 1, None])
        e2[t] = loss.item()
        (loss / accum).backward()
        n_acc += 1
        if n_acc == accum:
            opt.step()
            opt.zero_grad()
            n_acc = 0
        if t % t_gate == t_gate - 1:
            lyr.step_gate()
    return e2


def run_ttt_residual(lr, accum=8, t_gate=64):
    """TWO independent TwoTimescaleLinear stages, summed, trained
    jointly on the same loss -- stage 2 sees the same input and the same
    gradient signal, so it learns whatever stage 1's sign-only weights
    left on the table."""
    lyr1 = TwoTimescaleLinear(N_RES, 1, density=0.5, bias=True, evidence_beta=0.9)
    lyr2 = TwoTimescaleLinear(N_RES, 1, density=0.5, bias=False, evidence_beta=0.9)
    opt = torch.optim.SGD(list(lyr1.parameters()) + list(lyr2.parameters()), lr=lr)
    e2 = np.empty(T)
    n_acc = 0
    for t in range(T):
        out = lyr1(St[t:t + 1]) + lyr2(St[t:t + 1])
        loss = F.mse_loss(out, Yt[t:t + 1, None])
        e2[t] = loss.item()
        (loss / accum).backward()
        n_acc += 1
        if n_acc == accum:
            opt.step()
            opt.zero_grad()
            n_acc = 0
        if t % t_gate == t_gate - 1:
            lyr1.step_gate()
            lyr2.step_gate()
    return e2


def run_oracle():
    e2 = np.empty(T)
    for tau in np.unique(tau_seq[WASH:WASH + T]):
        m = tau_seq[WASH:WASH + T] == tau
        A = np.hstack([S[m], np.ones((m.sum(), 1))])
        w = np.linalg.solve(A.T @ A + 1e-2 * np.eye(N_RES + 1), A.T @ Y[m])
        e2[m] = (Y[m] - A @ w) ** 2
    return e2


def best(fn, lrs):
    b = None
    for lr in lrs:
        e2 = fn(lr)
        tail = metrics(e2)
        if b is None or tail < b[1]:
            b = (lr, tail)
    return b


LRS = [0.002, 0.005, 0.01, 0.02, 0.05]
print(f"MG switching tau 17/30 every {T_REG}, {T} online steps\n")
d = best(run_dense, LRS)
p1 = best(run_ttt_single, LRS)
p2 = best(run_ttt_residual, LRS)
ot = metrics(run_oracle())

print(f"{'readout':>16} | {'tail NMSE':>9} | {'best lr':>7} | bits/weight")
print("-" * 55)
print(f"{'oracle':>16} | {ot:>9.4f} | {'--':>7} | 32 (offline ceiling)")
print(f"{'dense':>16} | {d[1]:>9.4f} | {d[0]:>7} | 32")
print(f"{'protect (1-stage)':>16} | {p1[1]:>9.4f} | {p1[0]:>7} | ~1.6")
print(f"{'residual (2-stage)':>16} | {p2[1]:>9.4f} | {p2[0]:>7} | ~3.2")

print("\n--- verdict ---")
improvement = (p1[1] - p2[1]) / p1[1] * 100 if p1[1] > 0 else float("nan")
print(f"residual vs single-stage: {p2[1]:.4f} vs {p1[1]:.4f} ({improvement:+.0f}%)")
print(f"residual within 2x dense (P1's original bar): {p2[1]:.4f} vs {2*d[1]:.4f} -> "
      f"{'CONFIRMED' if p2[1] <= 2*d[1] else 'NOT MET'}")
