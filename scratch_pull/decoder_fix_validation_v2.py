"""Re-run ar_tokens/fast_tokens x {10,50}-shot on libero_long only, after
fixing eval-time decoding to use REAL autoregressive generation (head.generate())
instead of teacher-forcing off an all-zero dummy target (that was the bug found
after decoder_fix_validation.json: decoder_fix_validation.json itself was still
invalid because sweep.py's eval loop called `head(pooled, zeros)` for these two
AR heads, so its "predictions" were teacher-forced off ground-truth-shaped zeros,
not genuinely generated). Same protocol as decoder_fix_validation.py otherwise:
30 eval episodes, 250-step cap, same EPOCHS/BATCH_SIZE/LR, existing
cache_libero_long.pt (no re-caching). Writes to decoder_fix_validation_v2.json
-- does NOT touch decoder_fix_validation.json or results.json.

ponytail: import order matters -- common (torch/transformers) MUST be imported
before metaworld/libero native libs. common is imported first here; sweep.py
does the same at its own top, so no additional ordering risk from importing it.
"""
import os
import time

import common  # noqa: F401 -- must precede any libero/metaworld import
import torch

import sweep  # reuse get_cache / make_env / close_env / run_cell / cell_done

RESULTS_PATH = os.path.join(common.OUT_DIR, "decoder_fix_validation_v2.json")
LOG_PATH = os.path.join(common.OUT_DIR, "decoder_fix_validation_v2_log.txt")
sweep.RESULTS_PATH = RESULTS_PATH  # redirect all writes away from results.json / v1
sweep.LOG_PATH = LOG_PATH


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "expected CUDA (GPU1) to be available"
    sweep.log("=== decoder fix validation v2 (real generate()) start === device=" + device)
    common.heartbeat()

    bb = common.load_backbone(device=device)
    sweep.log("backbone loaded")
    common.heartbeat()

    suite_name = "libero_long"
    cfg = common.SUITES[suite_name]
    cache = sweep.get_cache(suite_name, cfg, bb)  # loads existing cache_libero_long.pt, no recaching
    env_ctx = sweep.make_env(cfg)
    try:
        for shot in [10, 50]:
            for head_name in ["ar_tokens", "fast_tokens"]:
                sweep.run_cell(suite_name, cfg, shot, head_name, cache, bb, device, env_ctx)
    finally:
        sweep.close_env(env_ctx)

    sweep.log("=== decoder fix validation v2 complete ===")


if __name__ == "__main__":
    main()
