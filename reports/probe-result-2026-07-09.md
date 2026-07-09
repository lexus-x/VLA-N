# Linear-probe result — refutes the "frozen backbone collapse" premise

Run: 2026-07-09, `a6000-left`, CPU-only, `scratch_pull/diag_probe.py` fed over ssh stdin.
Ridge regression, frozen pooled SmolVLM2-500M features → action chunk (56-dim).
Split **by demo** (60/20/20 train/val/test); val picks λ; test reports R². Shuffled-target control.

| suite | shot | n | cos_raw | cos_centered | rel_var | probe R² | shuffled control |
|---|---|---|---|---|---|---|---|
| libero_long | 10 | 1896 | 0.9960 | **0.0040** | 4.18e-03 | **+0.3808** | −0.0000 |
| libero_long | 50 | 9470 | 0.9959 | **0.0040** | 4.26e-03 | **+0.7538** | −0.0000 |
| libero_plus | 10 | 1896 | 0.9960 | 0.0041 | 4.13e-03 | +0.3970 | +0.0000 |
| libero_plus | 50 | 9470 | 0.9960 | 0.0040 | 4.22e-03 | +0.7528 | +0.0000 |
| metaworld_push | 10 | 610 | 0.9988 | 0.0007 | 1.27e-03 | +0.3166 | +0.0009 |
| metaworld_push | 50 | 3035 | 0.9988 | 0.0022 | 1.21e-03 | +0.6680 | −0.0007 |

Control is ~0 everywhere → no demo-level leakage; the R² numbers are real.

## What this refutes

**`handoff.md` (2026-07-07 evening) claimed:** *"this is a real information bottleneck in the frozen
backbone's learned representations… No head can discriminate outputs that differ by ~3 L2 from an
input space that's >99% cosine-similar."*

**That is false.** Centering the pooled features drops pairwise cosine similarity from 0.996 to
**0.004** — they are nearly orthogonal. The 0.996 was a shared mean offset, nothing else. A plain
ridge regression reads **75% of held-out action variance** out of those "collapsed" features.

Consequences:

1. **Unfreezing the backbone was never required.** Direction A's core premise is dead.
2. **The VICReg cos-sim result (0.996 → 0.28) is not evidence of anything.** The penalty
   `relu(1 - std(pooled_batch, dim=0)).mean()` maximizes per-dim std of the exact vector whose
   cosine similarity was then reported as the outcome. It is circular, and it was never load-bearing.
3. **The v2-vs-v3 `var_loss` "robustness gap" was unseeded variance**, not a finding. `common.py`
   now seeds (`set_seed` at `train_head` entry).
4. **The real open question moved.** Features are linearly action-decodable at R²=0.75, yet trained
   heads reach 0–19% task success. The gap is in the readout or the closed loop, not the representation.

## Second finding: `libero_plus` is not a robustness eval as configured

`common.py` `SUITES` gives `libero_long` and `libero_plus` the **same `hdf5_path`** and the same
`init_states_path`. Only `instruction` (a paraphrase) and `bddl_path` differ. Identical n (1896/9470)
and near-identical probe R² confirm the visual data is the same. As configured this cell measures
language paraphrasing on identical frames — not viewpoint/lighting/distractor robustness. Any claim
about LIBERO-Plus robustness from these cells would be wrong.

Corollary worth checking: swapping the instruction moved pooled cos_raw by <0.0001, i.e. the text
side barely affects the pooled vector at all.

## Gate 2 (run 2026-07-09): the head is not underfitting either

`scratch_pull/diag_head_vs_probe.py`, same demo split, `common.py`'s exact recipe
(EPOCHS=60, BATCH_SIZE=32, LR=1e-4, Adam), `libero_long` shot50. Ridge reproduced at
R²=0.7538 exactly, confirming the split matches.

| model | held-out R² | held-out MAE |
|---|---|---|
| ridge (L2 fit) | +0.7538 | 0.1227 |
| **l1_chunk (L1 fit)** | **+0.7755** | **0.0830** |

l1_chunk train R²=0.8654 / MAE=0.0625 → train-test R² gap +0.090, so it is not badly
overfitting either. MAE is reported because l1_chunk minimizes L1 (conditional median)
while ridge minimizes L2, and R² is an L2 metric that mildly favors ridge. l1_chunk wins
on **both**.

**Conclusion: representation is fine, and the readout is fine.** Frozen pooled features
carry the signal, and the sweep's own head extracts it better than a linear model does.
Yet the same head scores 2–19% task success. The failure is in the **closed loop**.

## Gate 3 (run 2026-07-09): NOT a train/eval frame mismatch — hypothesis refuted

Training reads `agentview_rgb` from hdf5 (`sweep.py:96`); eval reads `agentview_image`
from `OffScreenRenderEnv` (`sweep.py:227`); neither applies a transform. OpenVLA's LIBERO
eval applies `img[::-1, ::-1]` to the env observation, so a 180° mismatch was the leading
suspect — it would exactly reproduce "great open-loop fit, ~0% closed-loop."

`scratch_pull/diag_frame_parity.py` resets the env to demo 0's recorded init state and
compares the rendered frame to the hdf5 frame under each candidate transform:

| transform | RMSE vs hdf5 frame |
|---|---|
| **identity** | **15.58** |
| flip horizontal | 35.59 |
| flip vertical | 43.84 |
| rotate 180 | 45.82 |

Identity wins by ~3×. **The frames already agree; there is no orientation bug.** (Residual
15.58 is state drift between `set_init_state` and the recorded frame, not orientation.)

## Where that leaves us

Eliminated: representation collapse, head underfitting, frame orientation.
Remaining suspects for the open-loop → closed-loop gap, in order of cheapness:

1. **Blind chunk execution.** `eval_libero` runs all 8 actions of a predicted chunk before
   re-observing (`sweep.py:233-236`). 8 steps × 20 Hz = 0.4 s open-loop per observation.
   Per-step MAE 0.083 compounds across them.
2. **`MAX_STEPS=250`** vs LIBERO's own 600 (`common.py:58`) — episodes may simply be truncated.
3. **No proprioception and no history.** The head conditions on a pooled image+text vector
   only. `obs["robot0_eef_pos"]` is available in the eval loop and never used. At a fixed
   image, demo actions are multimodal across task phases; 22% unexplained variance is
   consistent with that ambiguity.

Next: re-run one eval cell with (a) 1 action per chunk instead of 8, (b) MAX_STEPS=600.
If success moves, the gap is drift, and a frozen-backbone plug-in module that stabilizes
closed-loop execution is a real, well-scoped target. If it does not move, suspect (3).
