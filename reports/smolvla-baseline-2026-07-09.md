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

## Batching: measured ~7× throughput, and a setup hazard

| | `batch_size=1` | `batch_size=10` |
|---|---|---|
| per control step | 1.16 s/it | 1.66 s/it |
| episodes in flight | 1 | 10 |
| effective throughput | 1× | **~7×** |
| GPU util | ~6 % | ~38 % |

Batching 10 parallel envs costs only 1.43× per step, so it buys ~7× wall-clock. Worth doing.

**Setup hazard:** `lerobot_eval.py:754` iterates a *materialized* list of `(task_group, task_id, env)`,
so **all 10 tasks × `batch_size` envs are constructed before any rollout begins** — 100 MuJoCo envs at
`batch_size=10`. That is ~7 min of CPU-bound setup and ~42 GB RSS before the first step, and it scales
with `batch_size`. At `batch_size=50` (full protocol in one shot) this would build 500 envs; don't.
Loop over tasks with `--env.task_ids=[i]` instead.

GPU during rollout: 66/80 GB total, of which ~18 GB belongs to another user. Headroom is thin — a
larger `batch_size` risks OOM on a shared box.

## Revised cost model

520 steps × 1.66 s ≈ 14.4 min per task worst case (less when episodes succeed early). Full standard
protocol (50 eps × 10 tasks) at `batch_size=10` ≈ **12 GPU-hours**, not 67. Feasible, but plan it.

## Status

Baseline run in flight (PID 2594051, launched 13:36): 10 episodes/task × 10 tasks, `batch_size=10`,
seed 1000 → `/home/user/vla-atlas/eval_baseline_libero10/run1`. ETA ~2.5 h.
Compare its `pc_success` against SmolVLA's published LIBERO-Long figure (~71 %, protocol-dependent)
before building anything on top. **10 eps/task is not the standard 50 — label it as such.**
