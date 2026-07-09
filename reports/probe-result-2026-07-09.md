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

## Next gate (cheap, decides whether a paper exists)

Train `l1_chunk` on the same cached features and compare its **held-out R²** against ridge's 0.7538.

- head R² ≪ 0.75 → the head/training pipeline is underfitting a signal a linear model finds. A bug, not a paper.
- head R² ≈ 0.75 but success stays 0–19% → representational sufficiency without closed-loop competence.
  That gap is the module target, and it is a real research question.
