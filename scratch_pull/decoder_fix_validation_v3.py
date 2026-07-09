"""Re-run ar_tokens/fast_tokens x {10,50}-shot on libero_long only, after
fixing fast_tokens' generate() to train on an eos_id target (see
heads/head_fast_tokens.py's BUGFIX docstring) so the GRU learns to stop
before the CHUNK_LEN*ACTION_DIM safety cap instead of always hitting it and
feeding physical-intelligence/fast's decode() an over-length sequence that
silently fell back to an all-zero action. Confirmed at the token level by
diag_fast_tokens.py (v3 log): eos now hit 10/10, decode succeeds. This script
confirms it at the real success_rate/mean_jerk level. Same protocol as
decoder_fix_validation_v2.py otherwise (same cache, same EPOCHS/BATCH_SIZE/LR,
30 eval episodes, 250-step cap). ar_tokens is included as a regression check
(should reproduce v2's numbers unchanged) -- fast_tokens is the one expected
to differ from v2's frozen 0.004191546799880047 mean_jerk constant.

Writes to decoder_fix_validation_v3_fasttokens.json -- does NOT touch v1/v2
or results.json.
"""
import os

import common  # noqa: F401 -- must precede any libero/metaworld import
import torch

import sweep  # reuse get_cache / make_env / close_env / run_cell

RESULTS_PATH = os.path.join(common.OUT_DIR, "decoder_fix_validation_v3_fasttokens.json")
LOG_PATH = os.path.join(common.OUT_DIR, "decoder_fix_validation_v3_fasttokens_log.txt")
sweep.RESULTS_PATH = RESULTS_PATH
sweep.LOG_PATH = LOG_PATH


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "expected CUDA (GPU1) to be available"
    sweep.log("=== decoder fix validation v3 (fast_tokens eos fix) start === device=" + device)
    common.heartbeat()

    bb = common.load_backbone(device=device)
    sweep.log("backbone loaded")
    common.heartbeat()

    suite_name = "libero_long"
    cfg = common.SUITES[suite_name]
    cache = sweep.get_cache(suite_name, cfg, bb)
    env_ctx = sweep.make_env(cfg)
    try:
        for shot in [10, 50]:
            for head_name in ["ar_tokens", "fast_tokens"]:
                sweep.run_cell(suite_name, cfg, shot, head_name, cache, bb, device, env_ctx)
    finally:
        sweep.close_env(env_ctx)

    sweep.log("=== decoder fix validation v3 complete ===")


if __name__ == "__main__":
    main()
