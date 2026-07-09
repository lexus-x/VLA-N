"""Is the frozen backbone's pooled feature ACTUALLY information-poor, or does it
just look that way?

Both existing diagnostics (diag_pooled_variance, diag_raw_pixels) measure cosine
similarity on UNCENTERED vectors. That number is dominated by the shared mean
direction, not by the signal. Evidence it is an artifact, from our own logs:
raw pixels -- which are trivially information-rich -- score 0.9667. Non-negative
activations with a large common offset always score ~1.0 regardless of how much
information they carry. So "cos_sim = 0.996" is not evidence of a bottleneck.

Worse, the fix we credited (unfreeze + VICReg) optimizes
`relu(1 - std(pooled_batch, dim=0)).mean()`, i.e. it MAXIMIZES per-dim std of the
exact vector whose cosine similarity we then report as the outcome. 0.996 -> 0.28
is close to tautological: the penalty drives that metric by construction.

This script asks the question that actually matters, with no training and no GPU:
can a plain ridge regression read the action out of the frozen pooled feature?

  - centered vs uncentered cos_sim          (is the "collapse" an offset artifact?)
  - relative variance                        (already computed by diag_pooled_variance, never reported)
  - ridge probe R^2, frozen pooled -> action (is the information present at all?)
  - the same probe on SHUFFLED targets       (control; doubles as this script's self-check)

Splitting is BY DEMO, not by sample: adjacent timesteps within one demo are
near-duplicates, so a random sample split leaks and inflates R^2.

Read the result like this:
  probe R^2 clearly > 0  -> information IS there; 0% success is a head/training
                            problem, and unfreezing was never the required fix.
  probe R^2 ~ 0          -> information is genuinely absent from the POOLED vector.
                            Next question is whether pooling destroyed it (see the
                            token-grid variant) or the encoder never had it.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, r"C:\Users\islab01\vla-atlas")

CACHE = r"C:\Users\islab01\vla-atlas\experiments\firstpass_sweep\cache_{}.pt"
LAMBDAS = [1e-2, 1e0, 1e2, 1e4, 1e6]


def cos_stats(p):
    pn = torch.nn.functional.normalize(p, dim=-1)
    n = p.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    s = (pn @ pn.T)[iu[0], iu[1]]
    return s.mean().item()


def ridge_r2(Xtr, Ytr, Xte, Yte, lam):
    """Closed-form ridge. Centering means the null model (predict train mean)
    scores exactly R^2 = 0, so the number is directly interpretable."""
    xm, ym = Xtr.mean(0, keepdim=True), Ytr.mean(0, keepdim=True)
    Xc, Yc = Xtr - xm, Ytr - ym
    A = Xc.T @ Xc + lam * torch.eye(Xc.shape[1], dtype=Xc.dtype)
    W = torch.linalg.solve(A, Xc.T @ Yc)
    pred = (Xte - xm) @ W + ym
    ss_res = ((Yte - pred) ** 2).sum()
    ss_tot = ((Yte - ym) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def fit_probe(X, Y, demo_id, seed=0):
    """Demo-level 60/20/20 train/val/test. Val picks lambda; test reports R^2."""
    demos = torch.unique(demo_id)
    g = torch.Generator().manual_seed(seed)
    demos = demos[torch.randperm(len(demos), generator=g)]
    n_tr, n_va = int(0.6 * len(demos)), int(0.2 * len(demos))
    parts = [demos[:n_tr], demos[n_tr : n_tr + n_va], demos[n_tr + n_va :]]
    masks = [torch.isin(demo_id, d) for d in parts]
    (Xtr, Xva, Xte), (Ytr, Yva, Yte) = [X[m] for m in masks], [Y[m] for m in masks]

    best = max(LAMBDAS, key=lambda l: ridge_r2(Xtr, Ytr, Xva, Yva, l))
    return ridge_r2(Xtr, Ytr, Xte, Yte, best), best


for suite in ["libero_long", "libero_plus", "metaworld_push"]:
    try:
        cache = torch.load(CACHE.format(suite), weights_only=False)
    except FileNotFoundError:
        print(f"{suite}: cache not found, skip", flush=True)
        continue

    pooled, actions, demo_id = cache["pooled"], cache["actions"], cache["demo_id"]

    for shot in [10, 50]:
        m = demo_id < shot
        p = pooled[m].double()
        y = actions[m].double().reshape(p.shape[0], -1)
        d = demo_id[m]
        if len(torch.unique(d)) < 5:
            continue

        raw_cos = cos_stats(p)
        cent_cos = cos_stats(p - p.mean(0, keepdim=True))
        rel_var = (p.var(0).sum() / p.pow(2).sum(1).mean()).item()

        r2, lam = fit_probe(p, y, d)
        # Control: same pipeline, targets shuffled across samples. Must be <= ~0.
        # If this comes back positive, the probe leaks and every number above is void.
        y_shuf = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(1))]
        r2_shuf, _ = fit_probe(p, y_shuf, d)

        print(
            f"{suite} shot{shot}: n={p.shape[0]} "
            f"cos_raw={raw_cos:.4f} cos_centered={cent_cos:.4f} rel_var={rel_var:.2e} | "
            f"probe R^2={r2:+.4f} (lam={lam:g})  shuffled-control R^2={r2_shuf:+.4f}",
            flush=True,
        )
        assert r2_shuf < 0.05, f"CONTROL FAILED (R^2={r2_shuf:.3f}) -- probe leaks, ignore all results"
