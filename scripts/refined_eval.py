"""Refined re-eval at N=100 episodes for the 5 cells that showed any signal
in the first-pass 30-cell sweep. Reuses cached features + the exact same
train_head/eval_libero/eval_metaworld code from the original sweep -- only
N_EPISODES changes (30 -> 100). Writes to a NEW file (refined_eval_100ep.json),
does not touch the original results.json.

ponytail: reuses sweep.py's env-per-suite / cache-per-suite machinery as-is
(import it directly) instead of re-deriving eval loops -- same code path,
same correctness, less to get wrong.
"""
import os
import sys
import time

sys.path.insert(0, r"C:\Users\islab01\vla-atlas\experiments\firstpass_sweep")

import common  # noqa: E402 -- must precede metaworld import

SMOKE = os.environ.get("REFINED_SMOKE") == "1"
common.N_EPISODES = 2 if SMOKE else 100  # the whole point of this script

import sweep  # noqa: E402

RESULTS_PATH = os.path.join(common.OUT_DIR, "refined_smoke.json" if SMOKE else "refined_eval_100ep.json")
LOG_PATH = os.path.join(common.OUT_DIR, "refined_eval_log.txt")

# (head_name, suite_name, shot)
CELLS = [
    ("l1_chunk", "libero_plus", 50),
    ("l1_chunk", "libero_long", 10),
    ("l1_chunk", "metaworld_push", 10),
    ("l1_chunk", "metaworld_push", 50),
    ("bspline", "metaworld_push", 50),
]


def log(msg):
    common.log(msg, logfile=LOG_PATH)


def cell_done(cell_id):
    if not os.path.exists(RESULTS_PATH):
        return False
    import json
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    return any(r.get("cell_id") == cell_id and r.get("status") == "ok" for r in results)


def main():
    device = "cuda" if torch_cuda_available() else "cpu"
    log("=== refined_eval start (N_EPISODES=" + str(common.N_EPISODES) + ") device=" + device + " ===")
    common.heartbeat()

    bb = common.load_backbone(device=device)
    log("backbone loaded")
    common.heartbeat()

    suites_needed = sorted(set(s for _, s, _ in CELLS))
    for suite_name in suites_needed:
        cfg = common.SUITES[suite_name]
        cells_here = [(h, sh) for h, s, sh in CELLS if s == suite_name]
        cell_ids_here = [h + "__" + suite_name + "__shot" + str(sh) for h, sh in cells_here]
        if all(cell_done(cid) for cid in cell_ids_here):
            log("SKIP suite " + suite_name + " (all its cells already done)")
            continue

        log("--- suite " + suite_name + " ---")
        cache = sweep.get_cache(suite_name, cfg, bb)  # loads existing cache_*.pt, does not recompute
        env_ctx = sweep.make_env(cfg)
        try:
            for head_name, shot in cells_here:
                run_cell(suite_name, cfg, shot, head_name, cache, bb, device, env_ctx)
        finally:
            sweep.close_env(env_ctx)

    log("=== refined_eval complete ===")


def torch_cuda_available():
    import torch
    return torch.cuda.is_available()


def run_cell(suite_name, cfg, shot, head_name, cache, bb, device, env_ctx):
    cell_id = head_name + "__" + suite_name + "__shot" + str(shot)
    if cell_done(cell_id):
        log("SKIP (already done): " + cell_id)
        return

    mask = cache["demo_id"] < shot
    pooled_sub = cache["pooled"][mask].to(device)
    action_sub = cache["actions"][mask].to(device)
    hidden_size = cache["hidden_size"]

    t_cell_start = time.time()
    log("START " + cell_id + " (n_samples=" + str(pooled_sub.shape[0]) + ", n_episodes=" + str(common.N_EPISODES) + ")")
    head, loss_curve, train_seconds = common.train_head(head_name, pooled_sub, action_sub, hidden_size, device)
    head.eval()

    if cfg["kind"] == "libero":
        eval_summary = sweep.eval_libero(env_ctx, cfg, bb, head, hidden_size, device)
    else:
        eval_summary = sweep.eval_metaworld(env_ctx, cfg, bb, head, hidden_size, device)

    sr = eval_summary["success_rate"]
    n_success = sum(r["success"] for r in eval_summary["episodes"])
    entry = {
        "cell_id": cell_id,
        "status": "ok",
        "head": head_name,
        "suite": suite_name,
        "shot_level": shot,
        "success_rate": sr,
        "n_success": n_success,
        "mean_jerk": eval_summary["mean_jerk"],
        "achieved_hz": eval_summary["achieved_hz"],
        "train_time_s": train_seconds,
        "eval_time_s": eval_summary["eval_time_s"],
        "first_loss": loss_curve[0] if loss_curve else None,
        "last_loss": loss_curve[-1] if loss_curve else None,
        "n_train_samples": int(pooled_sub.shape[0]),
        "n_eval_episodes": common.N_EPISODES,
        "max_steps": common.MAX_STEPS,
        "cell_wall_s": time.time() - t_cell_start,
    }
    common.append_result(entry, results_path=RESULTS_PATH)
    log(
        "DONE  " + cell_id + "  success_rate=" + str(sr) + " (" + str(n_success) + "/" + str(common.N_EPISODES) + ")"
        + "  train=" + str(round(train_seconds, 1)) + "s eval=" + str(round(eval_summary["eval_time_s"], 1)) + "s"
    )


if __name__ == "__main__":
    main()
