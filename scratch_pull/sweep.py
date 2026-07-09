"""Main driver for the first-pass 5-head x 2-shot x 3-suite sweep.
See common.py module docstring for the caching/precision/MAX_STEPS design
notes and tradeoffs.

Usage:
    python sweep.py                        # full run (about 10-14h)
    set SWEEP_SMOKE=1
    python sweep.py                        # tiny smoke run, all 30 cells,
                                            # 2 epochs / 2 episodes / 20 steps,
                                            # writes to smoke_results.json
                                            # instead of results.json -- run
                                            # this once before the real thing.

ponytail: import order matters -- common (torch/transformers) MUST be
imported before metaworld, else metaworld/mujoco native DLL init collides
with torch/transformers DLL init and the process dies silently (exit code 5,
no traceback -- confirmed by isolation test while building this sweep). Keep
import common first.
"""
import os
import sys
import time
import json
import traceback

import common  # noqa: F401 -- must precede metaworld import, see docstring
import h5py
import numpy as np
import torch
from PIL import Image

SMOKE = os.environ.get("SWEEP_SMOKE") == "1"
if SMOKE:
    common.EPOCHS = 2
    common.N_EPISODES = 2
    common.MAX_STEPS = 20
    RESULTS_PATH = os.path.join(common.OUT_DIR, "smoke_results.json")
    LOG_PATH = os.path.join(common.OUT_DIR, "smoke_log.txt")
else:
    RESULTS_PATH = os.path.join(common.OUT_DIR, "results.json")
    LOG_PATH = os.path.join(common.OUT_DIR, "sweep_log.txt")


def log(msg):
    common.log(msg, logfile=LOG_PATH)


# ---------------------------------------------------------------------------
# Feature caching (one pass per suite at N_DEMOS_FULL=50; 10-shot slices demo_id<10)
# ---------------------------------------------------------------------------

def cache_path(suite_name):
    tag = "_smoke" if SMOKE else ""
    return os.path.join(common.OUT_DIR, "cache_" + suite_name + tag + ".pt")


def _partial_path(suite_name):
    tag = "_smoke" if SMOKE else ""
    return os.path.join(common.OUT_DIR, "cache_" + suite_name + tag + "_partial.pt")


def _load_partial(suite_name):
    # ponytail: caching a 50-demo suite takes 1-2h of GPU time; a crash mid-way
    # used to lose all of it since only the FINAL cache was ever saved to disk.
    # Checkpointing after every demo (below) fixes that -- resume from here.
    p = _partial_path(suite_name)
    if os.path.exists(p):
        blob = torch.load(p, weights_only=False)
        log("  [" + suite_name + "] resuming partial cache: " + str(blob["next_demo_idx"]) + " demos already done")
        return blob["pooled_list"], blob["action_list"], blob["demo_id_list"], blob["next_demo_idx"]
    return [], [], [], 0


def _save_partial(suite_name, pooled_list, action_list, demo_id_list, next_demo_idx):
    torch.save(
        {"pooled_list": pooled_list, "action_list": action_list, "demo_id_list": demo_id_list, "next_demo_idx": next_demo_idx},
        _partial_path(suite_name),
    )


def _cleanup_partial(suite_name):
    p = _partial_path(suite_name)
    if os.path.exists(p):
        os.remove(p)


def build_cache_libero(suite_name, cfg, bb):
    n_demos = 3 if SMOKE else common.N_DEMOS_FULL
    pooled_list, action_list, demo_id_list, start_idx = _load_partial(suite_name)
    with h5py.File(cfg["hdf5_path"], "r") as f:
        data = f["data"]
        demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))[:n_demos]
        for d_idx in range(start_idx, len(demo_keys)):
            dk = demo_keys[d_idx]
            demo = data[dk]
            imgs = demo["obs"]["agentview_rgb"][:]
            acts = demo["actions"][:].astype(np.float32)
            if SMOKE:
                imgs, acts = imgs[:6], acts[:6]
            chunks = common.build_chunks(acts)
            for t in range(imgs.shape[0]):
                pooled = common.backbone_pooled(bb, Image.fromarray(imgs[t]), cfg["instruction"])
                pooled_list.append(pooled.squeeze(0).cpu())
                action_list.append(torch.from_numpy(chunks[t]))
                demo_id_list.append(d_idx)
            log("  [" + suite_name + "] cached demo " + str(d_idx + 1) + "/" + str(n_demos) + " (" + str(imgs.shape[0]) + " frames)")
            _save_partial(suite_name, pooled_list, action_list, demo_id_list, d_idx + 1)
            common.heartbeat()
    blob = _finish_cache(suite_name, pooled_list, action_list, demo_id_list)
    _cleanup_partial(suite_name)
    return blob


def build_cache_metaworld(suite_name, cfg, bb):
    demos_blob = torch.load(cfg["demos_path"], weights_only=False)
    demos = demos_blob["demos"]
    n_demos = 3 if SMOKE else common.N_DEMOS_FULL
    demos = demos[:n_demos]
    pooled_list, action_list, demo_id_list, start_idx = _load_partial(suite_name)
    for d_idx in range(start_idx, len(demos)):
        demo = demos[d_idx]
        imgs = demo["frames"]
        acts4 = demo["actions"]
        if SMOKE:
            imgs, acts4 = imgs[:6], acts4[:6]
        acts7 = np.stack([common.pad_action_to_7(a) for a in acts4])
        chunks = common.build_chunks(acts7)
        for t in range(imgs.shape[0]):
            pooled = common.backbone_pooled(bb, Image.fromarray(imgs[t]), cfg["instruction"])
            pooled_list.append(pooled.squeeze(0).cpu())
            action_list.append(torch.from_numpy(chunks[t]))
            demo_id_list.append(d_idx)
        log("  [" + suite_name + "] cached demo " + str(d_idx + 1) + "/" + str(n_demos) + " (" + str(imgs.shape[0]) + " frames)")
        _save_partial(suite_name, pooled_list, action_list, demo_id_list, d_idx + 1)
    blob = _finish_cache(suite_name, pooled_list, action_list, demo_id_list)
    _cleanup_partial(suite_name)
    return blob


def _finish_cache(suite_name, pooled_list, action_list, demo_id_list):
    pooled = torch.stack(pooled_list)
    actions = torch.stack(action_list)
    demo_id = torch.tensor(demo_id_list, dtype=torch.long)
    hidden_size = pooled.shape[-1]
    blob = {"pooled": pooled, "actions": actions, "demo_id": demo_id, "hidden_size": hidden_size}
    torch.save(blob, cache_path(suite_name))
    log("  [" + suite_name + "] cache saved: " + str(pooled.shape[0]) + " samples, hidden_size=" + str(hidden_size))
    return blob


def get_cache(suite_name, cfg, bb):
    p = cache_path(suite_name)
    if os.path.exists(p):
        log("  [" + suite_name + "] loading existing cache from " + p)
        return torch.load(p, weights_only=False)
    t0 = time.time()
    if cfg["kind"] == "libero":
        blob = build_cache_libero(suite_name, cfg, bb)
    else:
        blob = build_cache_metaworld(suite_name, cfg, bb)
    log("  [" + suite_name + "] feature caching took " + str(round(time.time() - t0, 1)) + "s")
    return blob


# ---------------------------------------------------------------------------
# Eval loops. The env is created ONCE PER SUITE (see make_env below) and
# reused across all 5 heads x 2 shots for that suite, not recreated per cell.
#
# ponytail: this used to create+close a fresh OffScreenRenderEnv every cell.
# LIBERO/mujoco's offscreen GL render context doesn't fully release native
# memory on env.close() on Windows, so 30 create/close cycles leaked ~1.9GB
# resident RAM each (confirmed via Get-Process on a live run: PID 34288 grew
# 1.7GB -> 9.2GB private over 4 completed libero_long cells). Reusing one env
# per suite (3 envs total instead of 30) cuts the leak multiplier 10x for
# free. If a future, larger sweep still leaks meaningfully, the next lazy
# step up is running each cell as its own subprocess so the OS reclaims
# everything between cells (same principle as the detached-launch fix).
# ---------------------------------------------------------------------------

def make_env(cfg):
    # ponytail: env construction (esp. metaworld.MT1, which registers every
    # metaworld env) can block for a long stretch with zero heartbeat calls
    # inside it -- under GPU contention this tripped the watchdog's staleness
    # check and spawned duplicate sweep.py processes (confirmed live: 3 extra
    # processes running the same cells). heartbeat() right before each
    # blocking constructor closes that gap.
    common.heartbeat()
    if cfg["kind"] == "libero":
        from libero.libero.envs import OffScreenRenderEnv
        env = OffScreenRenderEnv(bddl_file_name=cfg["bddl_path"], camera_heights=128, camera_widths=128)
        return {"env": env}
    else:
        import metaworld
        e = metaworld.MT1(cfg["mw_task"])
        env = e.train_classes[cfg["mw_task"]](render_mode="rgb_array")
        return {"env": env, "tasks": e.train_tasks}


def close_env(env_ctx):
    env_ctx["env"].close()
    import gc
    gc.collect()


def eval_libero(env_ctx, cfg, bb, head, hidden_size, device):
    env = env_ctx["env"]
    init_states = torch.load(cfg["init_states_path"], weights_only=False)
    dt = 1.0 / cfg["control_hz"]

    n_episodes = common.N_EPISODES
    max_steps = common.MAX_STEPS
    episode_results = []
    total_sim_steps = 0
    t_eval_start = time.time()

    for ep in range(n_episodes):
        env.reset()
        init_state = init_states[ep % init_states.shape[0]]
        obs = env.set_init_state(init_state)
        for _ in range(5):
            obs, _, _, _ = env.step(np.zeros(common.ACTION_DIM))

        ee_pos_traj = [obs["robot0_eef_pos"].copy()]
        steps, success = 0, False
        with torch.no_grad():
            while steps < max_steps and not success:
                img = Image.fromarray(obs["agentview_image"])
                pooled = common.backbone_pooled(bb, img, cfg["instruction"])
                pred_chunk = head.generate(pooled)
                pred_chunk = pred_chunk.squeeze(0).cpu().numpy()
                for k in range(pred_chunk.shape[0]):
                    if steps >= max_steps:
                        break
                    obs, reward, done, info = env.step(pred_chunk[k])
                    steps += 1
                    ee_pos_traj.append(obs["robot0_eef_pos"].copy())
                    if env.check_success():
                        success = True
                        break
        total_sim_steps += steps
        jerk = common.trajectory_jerk(ee_pos_traj, dt)
        episode_results.append({"episode": ep, "success": bool(success), "steps": steps, "jerk": jerk})
        common.heartbeat()

    t_eval_total = time.time() - t_eval_start
    return _summarize_eval(episode_results, total_sim_steps, t_eval_total)


def eval_metaworld(env_ctx, cfg, bb, head, hidden_size, device):
    env = env_ctx["env"]
    tasks = env_ctx["tasks"]
    dt = 1.0 / cfg["control_hz"]

    n_episodes = common.N_EPISODES
    max_steps = common.MAX_STEPS
    episode_results = []
    total_sim_steps = 0
    t_eval_start = time.time()

    for ep in range(n_episodes):
        task = tasks[ep % len(tasks)]
        env.set_task(task)
        obs, info = env.reset()

        ee_pos_traj = [env.get_endeff_pos().copy()]
        steps, success = 0, False
        with torch.no_grad():
            while steps < max_steps and not success:
                img = Image.fromarray(env.render()).resize((128, 128))
                pooled = common.backbone_pooled(bb, img, cfg["instruction"])
                pred_chunk = head.generate(pooled)
                pred_chunk = pred_chunk.squeeze(0).cpu().numpy()
                for k in range(pred_chunk.shape[0]):
                    if steps >= max_steps:
                        break
                    a4 = np.clip(pred_chunk[k][:4], -1.0, 1.0).astype(np.float32)
                    obs, rew, term, trunc, info = env.step(a4)
                    steps += 1
                    ee_pos_traj.append(env.get_endeff_pos().copy())
                    if info.get("success", 0):
                        success = True
                        break
                    if term or trunc:
                        break
                if steps >= max_steps:
                    break
        total_sim_steps += steps
        jerk = common.trajectory_jerk(ee_pos_traj, dt)
        episode_results.append({"episode": ep, "success": bool(success), "steps": steps, "jerk": jerk})
        common.heartbeat()

    t_eval_total = time.time() - t_eval_start
    return _summarize_eval(episode_results, total_sim_steps, t_eval_total)


def _summarize_eval(episode_results, total_sim_steps, t_eval_total):
    n = len(episode_results)
    n_success = sum(r["success"] for r in episode_results)
    jerks = [r["jerk"] for r in episode_results if r["jerk"] is not None]
    return {
        "success_rate": n_success / n if n else None,
        "mean_jerk": float(np.mean(jerks)) if jerks else None,
        "achieved_hz": total_sim_steps / t_eval_total if t_eval_total > 0 else None,
        "eval_time_s": t_eval_total,
        "episodes": episode_results,
    }


# ---------------------------------------------------------------------------
# Grid driver
# ---------------------------------------------------------------------------

def cell_done(cell_id):
    if not os.path.exists(RESULTS_PATH):
        return False
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    for r in results:
        if r.get("cell_id") == cell_id and r.get("status") == "ok":
            return True
    return False


def run_cell(suite_name, cfg, shot, head_name, cache, bb, device, env_ctx):
    cell_id = head_name + "__" + suite_name + "__shot" + str(shot)
    if cell_done(cell_id):
        log("SKIP (already done): " + cell_id)
        return

    mask = cache["demo_id"] < shot
    pooled_sub = cache["pooled"][mask].to(device)
    action_sub = cache["actions"][mask].to(device)
    hidden_size = cache["hidden_size"]

    last_err = None
    for attempt in range(1, 4):
        try:
            log("START " + cell_id + " (attempt " + str(attempt) + ", n_samples=" + str(pooled_sub.shape[0]) + ")")
            head, loss_curve, train_seconds = common.train_head(head_name, pooled_sub, action_sub, hidden_size, device, bb=bb)
            head.eval()

            if cfg["kind"] == "libero":
                eval_summary = eval_libero(env_ctx, cfg, bb, head, hidden_size, device)
            else:
                eval_summary = eval_metaworld(env_ctx, cfg, bb, head, hidden_size, device)

            sr = eval_summary["success_rate"]
            mj = eval_summary["mean_jerk"]
            hz = eval_summary["achieved_hz"]
            ev_t = eval_summary["eval_time_s"]

            entry = {
                "cell_id": cell_id,
                "status": "ok",
                "head": head_name,
                "suite": suite_name,
                "task": cfg.get("task_name", cfg.get("mw_task")),
                "shot_level": shot,
                "success_rate": sr,
                "mean_jerk": mj,
                "achieved_hz": hz,
                "train_time_s": train_seconds,
                "eval_time_s": ev_t,
                "first_loss": loss_curve[0] if loss_curve else None,
                "last_loss": loss_curve[-1] if loss_curve else None,
                "n_train_samples": int(pooled_sub.shape[0]),
                "n_eval_episodes": common.N_EPISODES,
                "max_steps": common.MAX_STEPS,
            }
            common.append_result(entry, results_path=RESULTS_PATH)
            log("DONE  " + cell_id + "  success_rate=" + str(sr) + "  jerk=" + str(mj) + "  hz=" + str(hz) + "  train=" + str(round(train_seconds, 1)) + "s eval=" + str(round(ev_t, 1)) + "s")
            return
        except Exception as e:
            last_err = type(e).__name__ + ": " + str(e)
            log("FAIL  " + cell_id + " attempt " + str(attempt) + "/3: " + last_err)
            log(traceback.format_exc())

    entry = {
        "cell_id": cell_id,
        "status": "failed",
        "head": head_name,
        "suite": suite_name,
        "task": cfg.get("task_name", cfg.get("mw_task")),
        "shot_level": shot,
        "error": last_err,
    }
    common.append_result(entry, results_path=RESULTS_PATH)
    log("GIVEUP " + cell_id + " after 3 attempts: " + str(last_err))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("=== sweep start (smoke=" + str(SMOKE) + ") device=" + device + " ===")
    common.heartbeat()

    if not os.path.exists(common.SUITES["metaworld_push"]["demos_path"]):
        import metaworld_demos
        metaworld_demos.main()

    bb = common.load_backbone(device=device)
    log("backbone loaded")
    common.heartbeat()

    shots = [5] if SMOKE else common.SHOT_LEVELS
    for suite_name, cfg in common.SUITES.items():
        log("--- suite " + suite_name + " ---")
        cache = get_cache(suite_name, cfg, bb)
        env_ctx = make_env(cfg)  # one env per suite, reused across all its cells -- see eval loop docstring
        try:
            for shot in shots:
                for head_name in common.HEAD_NAMES:
                    run_cell(suite_name, cfg, shot, head_name, cache, bb, device, env_ctx)
        finally:
            close_env(env_ctx)

    log("=== sweep complete ===")


if __name__ == "__main__":
    main()
