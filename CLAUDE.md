# vla-n

Novel <500M-param VLA contribution, target Q1/Q2 (CoRL → RA-L → TMLR). Shared lab GPUs.

**Hard constraint: the backbone stays frozen.** The deliverable is a plug-in module that boosts an existing VLA, not a finetuning recipe. Any proposal requiring backbone gradients is out of scope, however well it works.

## Read before doing anything

1. `handoff/handoff.md` — current state, open blockers, what ran and what hung.
2. `reports/research-direction-report.md` — verified SOTA tables, **6 killed directions with reasons**, venue ranking.

Do not re-research what those cover. Do not re-propose a killed direction without new evidence it is actually undone.

## Token discipline

Fan-out novelty sweeps cost real money (the last two were 75 and 99 agents). Rules:

- **Never sweep a claim whose premise is unvalidated.** Run the cheap diagnostic first; sweep the claim that survives it.
- One narrow sweep on one sharply-stated claim, immediately before writing. Not exploratory.
- `arxiv` + `semantic-scholar` MCP before web fan-out — they are targeted and cheap.

## Research integrity (non-negotiable)

- **No number without a run log or a fetched source.** Never estimate a result and present it as measured.
- **Never report a metric that your loss term directly optimizes.** The VICReg variance penalty maximizes per-dim std of the pooled vector; reporting its cos-sim drop (0.996→0.28) as the *finding* is circular. A reviewer catches this on page one.
- **Cosine similarity on uncentered activations is not a collapse measurement.** Raw pixels score 0.9667 on this data. Center first, or report relative variance / a linear probe instead.
- **No unseeded number is a result.** ≥3 seeds, report mean ± std. `sweep.py`/`common.py` currently do not seed — anything they produced is a pilot, not evidence. (`phase0_validate.py` does seed.)
- **Say what failed.** A negative result is publishable; a hidden one is misconduct.
- State assumptions before implementing. If two readings of a request exist, name both — don't pick silently.
- Every experiment gets its success criterion written down *before* it launches. "Run it and see" is how confounds survive.

## Shared-GPU rules

- `a6000-left` **GPU0 is off-limits** (reserved: fishonet). GPU1 only.
- All hosts are shared. `nvidia-smi` + check process owner before launching. `scripts/gpu_check.sh` = all-6 snapshot.
- SSH lands in `cmd.exe`, not bash: `ssh -o ConnectTimeout=10 -o BatchMode=yes <host> "<cmd>"`. `a100` is the one Linux box.
- Long jobs must survive disconnect → launch via WMI: `Invoke-CimMethod -ClassName Win32_Process -MethodName Create`. Not `start`/`Start-Process`.
- File transfer is `scp`/SSH only.
- Modal = real money ($50). Free lab GPUs first, Modal only on overflow.

## Environment gotchas

- robosuite 1.4.0 breaks on modern mujoco (`mj_fullM()` signature) → apply `patches/` on the host, re-apply after any venv rebuild.
- Unfrozen `vision_model`+`connector` training backprops through 32 frozen text layers → OOM-deadlocks a 49GB A6000 at any batch size. Gradient checkpointing on `text_model` (`use_reentrant=False`) is the only lever; `expandable_segments` is unsupported on Windows.
