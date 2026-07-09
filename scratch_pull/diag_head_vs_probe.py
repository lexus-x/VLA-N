"""Gate: diag_probe showed a ridge regression reads R^2=0.75 of held-out action
variance out of the FROZEN pooled features. The sweep's own l1_chunk head reaches
only 2-19% task success. Is the head failing to extract what a linear model finds?

Trains l1_chunk with common.py's exact recipe (EPOCHS/BATCH_SIZE/LR) on the SAME
demo-level split diag_probe uses, then reports held-out R^2 and MAE for both.

Caveat this script controls for: l1_chunk minimizes L1 (fits the conditional
median), ridge minimizes L2 (fits the mean). R^2 is an L2 metric and therefore
mildly favors ridge. So MAE is reported too -- l1_chunk should WIN on MAE if it
is training correctly at all. Judge underfitting on the pair, not on R^2 alone.

Read the result:
  head R^2 << ridge R^2 AND head MAE >= ridge MAE -> head/training underfits. A bug.
  head MAE < ridge MAE (head fits fine)           -> representation and readout are
                                                     both fine; the 0-19% success gap
                                                     lives in the closed loop.
"""
import random
import sys

import numpy as np
import torch

sys.path.insert(0, r"C:\Users\islab01\vla-atlas")
sys.path.insert(0, r"C:\Users\islab01\vla-atlas\experiments\firstpass_sweep")

import common  # noqa: E402
from heads.head_l1_chunk import L1ChunkHead  # noqa: E402

SUITE, SHOT, SEED = "libero_long", 50, 0
CACHE = rf"C:\Users\islab01\vla-atlas\experiments\firstpass_sweep\cache_{SUITE}.pt"
LAMBDAS = [1e-2, 1e0, 1e2, 1e4, 1e6]


def set_seed(s):
    # Defined here, not imported: the remote common.py predates the set_seed patch.
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def demo_split(demo_id, seed=SEED):
    """Byte-identical to diag_probe.fit_probe's split, so the two are comparable."""
    demos = torch.unique(demo_id)
    g = torch.Generator().manual_seed(seed)
    demos = demos[torch.randperm(len(demos), generator=g)]
    n_tr, n_va = int(0.6 * len(demos)), int(0.2 * len(demos))
    parts = [demos[:n_tr], demos[n_tr : n_tr + n_va], demos[n_tr + n_va :]]
    return [torch.isin(demo_id, d) for d in parts]


def scores(pred, Y, ym):
    ss_res = ((Y - pred) ** 2).sum()
    ss_tot = ((Y - ym) ** 2).sum()
    return (1 - ss_res / ss_tot).item(), (Y - pred).abs().mean().item()


def ridge(Xtr, Ytr, Xte, Yte, Xva, Yva):
    xm, ym = Xtr.mean(0, keepdim=True), Ytr.mean(0, keepdim=True)
    Xc, Yc = Xtr - xm, Ytr - ym

    def fit(lam):
        A = Xc.T @ Xc + lam * torch.eye(Xc.shape[1], dtype=Xc.dtype)
        return torch.linalg.solve(A, Xc.T @ Yc)

    best = max(LAMBDAS, key=lambda l: scores((Xva - xm) @ fit(l) + ym, Yva, ym)[0])
    return scores((Xte - xm) @ fit(best) + ym, Yte, ym), best


cache = torch.load(CACHE, weights_only=False, map_location="cpu")
m = cache["demo_id"] < SHOT
pooled, actions, demo_id = cache["pooled"][m], cache["actions"][m], cache["demo_id"][m]
tr, va, te = demo_split(demo_id)
H = pooled.shape[1]
print(f"{SUITE} shot{SHOT}: n={len(pooled)} hidden={H} train/val/test={tr.sum()}/{va.sum()}/{te.sum()}", flush=True)

# --- ridge reference (float64, CPU) ---
Xd, Yd = pooled.double(), actions.double().reshape(len(pooled), -1)
(r2_ridge, mae_ridge), lam = ridge(Xd[tr], Yd[tr], Xd[te], Yd[te], Xd[va], Yd[va])
print(f"ridge     : R^2={r2_ridge:+.4f}  MAE={mae_ridge:.4f}  (lam={lam:g})", flush=True)

# --- l1_chunk, common.py's exact recipe ---
dev = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(SEED)
head = L1ChunkHead(H).to(dev)
opt = torch.optim.Adam(head.parameters(), lr=common.LR)
Xtr, Ytr = pooled[tr].float().to(dev), actions[tr].float().to(dev)

for epoch in range(common.EPOCHS):
    perm = torch.randperm(len(Xtr), device=dev)
    for b in range(0, len(Xtr), common.BATCH_SIZE):
        idx = perm[b : b + common.BATCH_SIZE]
        opt.zero_grad()
        _, loss = head(Xtr[idx], Ytr[idx])
        loss.backward()
        opt.step()
    if epoch % 20 == 0 or epoch == common.EPOCHS - 1:
        print(f"  epoch {epoch:3d}  train L1={loss.item():.4f}", flush=True)

head.eval()
ym = Yd[tr].mean(0, keepdim=True)
with torch.no_grad():
    p_te = head.generate(pooled[te].float().to(dev)).cpu().double().reshape(int(te.sum()), -1)
    p_tr = head.generate(Xtr).cpu().double().reshape(int(tr.sum()), -1)
r2_te, mae_te = scores(p_te, Yd[te], ym)
r2_tr, mae_tr = scores(p_tr, Yd[tr], ym)

print(f"l1_chunk  : R^2={r2_te:+.4f}  MAE={mae_te:.4f}   (train R^2={r2_tr:+.4f} MAE={mae_tr:.4f})", flush=True)
print(
    f"\nVERDICT: head {'UNDERFITS' if mae_te >= mae_ridge else 'fits'} "
    f"(head MAE {mae_te:.4f} vs ridge {mae_ridge:.4f}); "
    f"train-vs-test R^2 gap {r2_tr - r2_te:+.4f}",
    flush=True,
)
