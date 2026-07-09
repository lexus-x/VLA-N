# SmolVLA baseline on LIBERO — a real VLA, running

Host `a100`, conda env `lerobot` (lerobot 0.4.4), `MUJOCO_GL=egl`, seed 1000.
Policy: `HuggingFaceVLA/smolvla_libero` (SmolVLA 450M, LIBERO-finetuned, `lerobot/smolvla_base` parent).
Command: `lerobot-eval --policy.path=... --env.type=libero --env.task=libero_10`.

## Smoke test (2026-07-09): the pipeline closes

1 episode, `libero_10` task 0 ("put both the alphabet soup and the tomato sauce in the basket"),
`batch_size=1`.

| metric | value |
|---|---|
| success | **True** (reward 1.0 at step 391 / 520) |
| `pc_success` | 100.0 (n=1 — a smoke test, not a result) |
| wall clock | **485 s for one episode** |
| video | `eval_smoke_1ep/videos/libero_10_0/eval_episode_0.mp4` |

This is the first time this project has run a real VLA end-to-end on LIBERO.

## Cost, measured not estimated

485 s/episode at `batch_size=1`, i.e. ~1.16 s per control step. SmolVLA has `n_action_steps=1`, so
every one of the ≤520 steps costs a full VLM forward plus a 10-step flow-matching integration.

The standard LIBERO protocol is 50 episodes × 10 tasks = 500 rollouts → **~67 GPU-hours** at that
rate. `n_episodes` in `lerobot-eval` is **per task**, not total (loop at `lerobot_eval.py:754`).
Parallel envs (`--eval.batch_size`) is the only lever; the GPU sits at ~6 % util with one env.

Budget this before promising a full-protocol number.

## Checkpoint provenance (caveat for the paper)

The model card says `datasets: unknown`. Inspected the cached `HuggingFaceVLA/libero` dataset directly:
**40 tasks, 1693 episodes, 273465 frames, fps=10** — 40 tasks = 4 suites × 10, and 1693 ≈ 4 × 423
after no-noop filtering. So the checkpoint was trained on all four LIBERO suites, and `libero_10`
eval is **in-distribution**, not zero-shot. Say so in any table that uses this number.

`tasks.parquet` stores only `task_index`, not language strings, so the suite identity is inferred
from counts, not matched strings. Good enough to establish in-distribution; do not over-claim it.

## Config facts worth keeping

| | value |
|---|---|
| `chunk_size` | 50 |
| `n_action_steps` | **1** (re-observes every control step) |
| `n_obs_steps` | 1 |
| VLM | `HuggingFaceTB/SmolVLM2-500M-Instruct` |
| flow-matching steps | 10 |
| inputs | `observation.images.image` (agentview), `.image2` (eye-in-hand), `observation.state` |
| action | 7-dim |
| images resized to | 512×512 with padding |
| env fps | LiberoEnv default 30; training data fps **10** — check this mismatch before trusting timing-sensitive claims |
| max episode steps | 520 |

## Status

Baseline run launched detached: 10 episodes/task × 10 tasks, `batch_size=10`, seed 1000 →
`/home/user/vla-atlas/eval_baseline_libero10/run1`. Compare its `pc_success` against SmolVLA's
published LIBERO-Long figure (~71 %, protocol-dependent) before building anything on top.
