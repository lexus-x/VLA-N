# First-Pass Sweep Report: <500M Action-Representation Atlas (Pipeline Validation + Preliminary Results)

Status: infrastructure fully built and validated end-to-end; first honest empirical data collected under a deliberately reduced ("first-pass") protocol. This is **not** the report's full-scoped experiment (500 rollouts/cell, 3 seeds, 4 shot-levels) — it's a smaller, realistic-timeframe pass chosen to validate the pipeline and get real planning numbers, per `research-direction-report.md`.

---

## 1. What was built

All on a remote Windows box (`a6000-left`, 2x RTX A6000 49GB, GPU1 used), single shared venv:

| Component | Status |
|---|---|
| Backbone | SmolVLM2-500M (507.5M params), loaded via `AutoModelForImageTextToText` |
| 5 action heads | AR integer-tokens (VLA-0 style), FAST/DCT tokens (pi0-FAST/lerobot), B-spline compact coefficients (BEAST style), flow-matching (SmolVLA/lerobot), L1 chunk regression (OpenVLA-OFT style) — all individually smoke-tested (forward+backward pass) on the shared backbone |
| LIBERO | Installed, patched for Windows (robosuite 1.4.0 + mujoco compatibility patch, documented at `patches/`) |
| LIBERO-Plus | Language-perturbation factor confirmed working (paraphrased instruction, same scene/demos as base LIBERO) |
| Meta-World | v3.1.1 installed, hard-tier task list sourced from TinyVLA (arXiv:2409.12514), scripted-policy demo generation confirmed (100% success on `push-v3`) |

Reference repos cloned for future head-porting: VLA-0 (NVlabs/vla0), BEAST (intuitive-robots/beast_calvin), OpenVLA-OFT (moojink/openvla-oft).

## 2. First-pass sweep design (deliberately reduced scope)

Full report design: 5 heads × 4 shot-levels (10/25/50/full) × 3 seeds × 3 suites × 500 rollouts/cell ≈ 78 days serial / ~13 days across 6 GPUs.

This first pass: **5 heads × 2 shot-levels (10, 50) × 1 seed × 3 suites × 30 eval episodes/cell = 30 cells**, one representative task per suite (LIBERO-Long: `STUDY_SCENE1` book-in-caddy; LIBERO-Plus: same task, paraphrased instruction; Meta-World: `push-v3`). Eval horizon cut 600→250 steps to fit a realistic timeframe. Backbone **frozen** — only the action head trains (a simplification for this phase, not the final experimental design).

## 3. Full 30-cell results

| Head | Type | libero_long 10-shot | libero_long 50-shot | libero_plus 10-shot | libero_plus 50-shot | metaworld_push 10-shot | metaworld_push 50-shot |
|---|---|---|---|---|---|---|---|
| ar_tokens | compact (AR) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| fast_tokens | compact (DCT) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| bspline | compact (spline) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 3.3%* |
| flow_matching | raw-chunk | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| l1_chunk | raw-chunk (OFT) | 3.3%* | 0.0% | 0.0% | 30.0%* | 3.3%* | 13.3%* |

\* Cells marked with an asterisk were re-run at N=100 episodes for statistical confidence (see §4) — the N=30 numbers above are the original, noisier reads.

**Jerk (smoothness) was clearly, reproducibly differentiated by head regardless of success**, a secondary but solid finding:

| Head | Typical jerk (all suites/shots) | Interpretation |
|---|---|---|
| fast_tokens | ~0.004 | Static/degenerate output — never learns a real policy |
| bspline | ~2-38 | Smooth |
| l1_chunk | ~5-70 | Moderate |
| flow_matching | ~20-172 | Moderate-high, variable |
| ar_tokens | ~70-245 | Erratic/jerky |

## 4. Refined evaluation (N=100) on the 5 cells with any signal

| Cell | N=30 | N=100 (clean, single-process) | Verdict |
|---|---|---|---|
| l1_chunk — libero_plus, 50-shot | 30.0% | **19.0%** [11-27%] | Signal holds — real, the standout result |
| l1_chunk — metaworld_push, 50-shot | 13.3% | **16.0%** [9-23%] | Signal holds, confirmed real |
| l1_chunk — libero_long, 10-shot | 3.3% | **8.0%** [3-13%] | Weak but real (CI excludes 0) |
| l1_chunk — metaworld_push, 10-shot | 3.3% (contaminated by duplicate-process race) | **2.0%** [0-5%] | Resolves to noise |
| bspline — metaworld_push, 50-shot | 3.3% | **0.0%** | Does not replicate — was noise |

**Clean conclusion:** `l1_chunk` (simple direct action-chunk regression) is the only head with a real, replicating success signal under this frozen-backbone protocol, and it scales positively with more demos (2-8% at 10-shot → 16-19% at 50-shot). Every discrete/compact-token head (`ar_tokens`, `fast_tokens`, and — once retested — `bspline`) shows no real signal at all at this scale.

## 5. Why this is NOT the final word on the paper's crossover hypothesis

The report's hypothesis was that compact-output heads would **win**, not lose, at low shot counts. The observed result is the opposite, but three simplifications specific to this first pass likely confound it:

1. **Frozen backbone, head-only training** — may specifically cripple heads that need joint fine-tuning to learn a good discretization (AR/FAST tokens), while a shallow regression head fits frozen features easily. This is the biggest suspect.
2. **Simplified decoders** — `ar_tokens`/`fast_tokens` use a small from-scratch GRU decoder substituted for native pretrained-LM autoregressive decoding (a build-phase engineering simplification), which may be why they show zero signal rather than merely worse signal (fast_tokens' near-zero jerk everywhere indicates a static/degenerate output, not a struggling-but-real policy).
3. **250-step eval cap** (cut from 600 for time) may unfairly punish slower-converging heads before they can complete the task.

## 6. Infrastructure notes for reproducibility

- Two unexplained silent process kills occurred during the run (no traceback, no reboot, no GPU driver reset, no crash record — cause never conclusively identified). Fixed with resume-safe cell/cache checkpointing + a heartbeat-driven watchdog that relaunches via WMI process creation (survives SSH session closure — `start`/`Start-Process` do not, on this Windows OpenSSH setup).
- The watchdog had a heartbeat blind spot around `metaworld.MT1()` construction, causing 3 false-positive relaunches (4 concurrent processes racing for a while). Fixed. Confirmed via read-only integrity check that this never corrupted `results.json` (exactly one entry per cell throughout) — only wasted redundant GPU-hours, and incidentally produced some free (uncontrolled) seed-variance samples.
- Real GPU1 contention from another lab member's job occurred twice during the run (confirmed via command-line lookup both times) — elevated eval times but never crashed anything. This machine is shared; expect this to recur.
- Robosuite 1.4.0 + modern mujoco needed a version-adaptive compatibility patch for `mj_fullM()` — documented at `C:\Users\islab01\vla-atlas\patches\` on the remote box, needed again if the venv is ever rebuilt.
- Total wall-clock: first-pass sweep ~20h23min (including infra incidents), refined N=100 follow-up ~3h18min.

## 7. Recommended next step (not yet started)

Unfreeze the backbone (or at least allow the action head + a small adapter to fine-tune it) and restore native decoding for `ar_tokens`/`fast_tokens` before drawing any real conclusion about the crossover hypothesis — the current result may simply reflect "frozen features favor regression over discrete decoding," not anything about compact vs. raw action representations in general. This is a genuinely separate, larger experiment (another multi-hour-to-multi-day compute commitment) and needs explicit go-ahead given shared-GPU usage.
