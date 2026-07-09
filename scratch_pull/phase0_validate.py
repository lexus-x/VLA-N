"""Phase 0 validation gate for the <500M action-representation atlas.

THE question this answers (nothing downstream is worth GPU-days until it does):
  Does unfreezing vision_model+connector (+ a VICReg-style anti-collapse
  variance penalty) lift LIBERO success_rate over the frozen-backbone
  baseline, reproducibly across seeds?

Design: one fixed cell (l1_chunk head, shot10, libero_long), SAME data, two
conditions run back-to-back per seed --
  A = frozen backbone  (current pipeline's best head, l1_chunk)
  B = unfrozen vision_model+connector + VICReg variance penalty, joint w/ head
Reports success_rate for both + before/after pooled cos_sim for B (the
de-collapse metric). 3 seeds so run-to-run robustness (the v2-worked/v3-didn't
open question in the handoff) is MEASURED, not assumed.

Reuses sweep.eval_libero / make_env / close_env verbatim -- the eval rollout
reads whatever weights live in bb.model, so a trained backbone evaluates
automatically. Bypasses bb.forward() for training (it's @no_grad + detach);
uses a direct batched bb.model(...) call like smoke_unfreeze_vision_v3.py.
"""
import os
import sys
import json
import time
import random

sys.path.insert(0, r"C:\Users\islab01\vla-atlas")

import numpy as np
import torch
import torch.nn.functional as F
import h5py
from PIL import Image

import common  # noqa: F401  torch/transformers first (import-order gotcha)
import sweep   # reuse make_env / eval_libero / close_env
from heads.backbone import SmolVLM2Backbone
from heads.head_l1_chunk import L1ChunkHead

# ---- fixed cell + validation-scale knobs (see module docstring) -------------
SEEDS = [0, 1, 2]
SUITE = "libero_long"
SHOT = 10
FRAMES_PER_DEMO = 20      # subsample cap -> ~200 live samples, tractable to train live
UNFREEZE_EPOCHS = 25
HEAD_EPOCHS_FROZEN = 60   # frozen head is cheap (cached feats), let it converge
BATCH_SIZE = 4            # 8 DEADLOCKS the unfrozen backward at the 48.4GB ceiling
                         # (smoke hung 19min, 0 heartbeats). 4 halves retained
                         # activations -> ~24GB, clears it. Handoff-prescribed lever.
                         # ponytail: VICReg std() over 4 samples is noisier but the
                         # 3-seed sweep makes that visible; drop to grad-checkpointing
                         # only if de-collapse proves batch-size-sensitive.
LR = 1e-4
VAR_WEIGHT = 5.0
VAR_TARGET_STD = 1.0
N_EPISODES = 20
MAX_STEPS = 250

if os.environ.get("PHASE0_SMOKE") == "1":
    # fast preflight: exercise every path (imports, live fwd+grad, env, eval)
    # in a few minutes to catch integration bugs before the ~2h real run.
    SEEDS = [0]
    FRAMES_PER_DEMO = 3
    UNFREEZE_EPOCHS = 2
    HEAD_EPOCHS_FROZEN = 2
    N_EPISODES = 1
    MAX_STEPS = 10

device = "cuda"
OUT_DIR = common.OUT_DIR
RESULTS_PATH = os.path.join(OUT_DIR, "phase0_validate.json")
LOG_PATH = os.path.join(OUT_DIR, "phase0_log.txt")

# eval loops read these off common
common.N_EPISODES = N_EPISODES
common.MAX_STEPS = MAX_STEPS


def log(msg):
    common.log(msg, logfile=LOG_PATH)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def make_bb():
    """Fresh backbone with pretrained weights (fp32, all frozen)."""
    return SmolVLM2Backbone(device=device)


def load_data(cfg):
    images, actions = [], []
    with h5py.File(cfg["hdf5_path"], "r") as f:
        data = f["data"]
        demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))[:SHOT]
        for dk in demo_keys:
            demo = data[dk]
            imgs = demo["obs"]["agentview_rgb"][:]
            acts = demo["actions"][:].astype(np.float32)
            chunks = common.build_chunks(acts)
            step = max(1, imgs.shape[0] // FRAMES_PER_DEMO)
            for t in range(0, imgs.shape[0], step):
                images.append(Image.fromarray(imgs[t]))
                actions.append(chunks[t])
    return images, torch.from_numpy(np.stack(actions)).to(device)


def prompt_for(bb, cfg):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": cfg["instruction"]}]}]
    return bb.processor.apply_chat_template(msgs, add_generation_prompt=True)


def live_pooled(bb, images, idxs, prompt):
    """One batched forward -> pooled (B,H), WITH grad (caller controls no_grad)."""
    batch_images = [[images[i]] for i in idxs]
    inputs = bb.processor(text=[prompt] * len(idxs), images=batch_images,
                          return_tensors="pt", padding=True).to(device)
    out = bb.model(**inputs, output_hidden_states=True)
    return out.hidden_states[-1].mean(dim=1).float()


def cos_sim_stats(vecs):
    vn = F.normalize(vecs, dim=-1)
    sim = vn @ vn.T
    n = vecs.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    p = sim[iu[0], iu[1]]
    return float(p.mean()), float(p.min()), float(p.max())


def frozen_pooled_all(bb, images, prompt):
    n = len(images)
    chunks = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            chunks.append(live_pooled(bb, images, list(range(i, min(i + BATCH_SIZE, n))), prompt))
    return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# Condition A: frozen backbone + l1_chunk head (current pipeline's best)
# ---------------------------------------------------------------------------
def run_frozen(bb, frozen_pooled, actions, env_ctx, cfg, hidden_size, seed):
    set_seed(seed)
    head = L1ChunkHead(hidden_size).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    n = frozen_pooled.shape[0]
    for _ in range(HEAD_EPOCHS_FROZEN):
        perm = torch.randperm(n, device=device)
        for b in range(0, n, BATCH_SIZE):
            idx = perm[b:b + BATCH_SIZE]
            opt.zero_grad()
            _, loss = head(frozen_pooled[idx], actions[idx])
            loss.backward()
            opt.step()
        common.heartbeat()
    head.eval()
    ev = sweep.eval_libero(env_ctx, cfg, bb, head, hidden_size, device)
    return ev


# ---------------------------------------------------------------------------
# Condition B: unfreeze vision_model+connector + VICReg penalty, joint w/ head
# ---------------------------------------------------------------------------
def run_unfrozen(images, actions, env_ctx, cfg, prompt, seed):
    set_seed(seed)
    bb = make_bb()
    # THE memory fix: backprop to vision+connector flows through all 32 frozen
    # text layers; retaining those activations OOM-deadlocks a 49GB card at ANY
    # batch size (batch 8 AND 4 both hung at 48.5GB). Gradient checkpointing
    # recomputes text-layer activations in backward instead of storing them
    # (~48GB -> ~18GB, +~30% compute). use_reentrant=False is required because
    # the text_model params are frozen but grad must still flow THROUGH them.
    # Checkpointing only activates in train() mode; SmolLM2 dropout is 0 so
    # train() changes nothing else. Eval later runs under bb.model.eval().
    bb.model.model.text_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    bb.model.model.text_model.train()
    vparams = list(bb.model.model.vision_model.parameters()) + list(bb.model.model.connector.parameters())
    for p in vparams:
        p.requires_grad = True

    hidden_size = live_pooled(bb, images, [0], prompt).shape[-1]
    head = L1ChunkHead(hidden_size).to(device)
    opt = torch.optim.Adam(vparams + list(head.parameters()), lr=LR)
    n = len(images)

    before = cos_sim_stats(frozen_pooled_all(bb, images, prompt))
    log(f"  [seed {seed}] B before-train cos_sim mean/min/max = {before[0]:.4f}/{before[1]:.4f}/{before[2]:.4f}")

    for epoch in range(UNFREEZE_EPOCHS):
        perm = np.random.permutation(n)
        et, ev, ns = 0.0, 0.0, 0
        for b in range(0, n, BATCH_SIZE):
            idxs = perm[b:b + BATCH_SIZE].tolist()
            if len(idxs) < 2:
                continue
            opt.zero_grad()
            pooled = live_pooled(bb, images, idxs, prompt)
            _, task_loss = head(pooled, actions[idxs])
            var_loss = F.relu(VAR_TARGET_STD - pooled.std(dim=0)).mean()
            (task_loss + VAR_WEIGHT * var_loss).backward()
            opt.step()
            et += task_loss.item(); ev += var_loss.item(); ns += 1
        log(f"  [seed {seed}] B epoch {epoch}: task={et/ns:.4f} var={ev/ns:.4f}")
        common.heartbeat()

    after = cos_sim_stats(frozen_pooled_all(bb, images, prompt))
    log(f"  [seed {seed}] B after-train cos_sim mean/min/max = {after[0]:.4f}/{after[1]:.4f}/{after[2]:.4f}")

    bb.model.eval()
    head.eval()
    ev_summary = sweep.eval_libero(env_ctx, cfg, bb, head, hidden_size, device)
    del bb
    torch.cuda.empty_cache()
    return ev_summary, before, after


def save(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def main():
    log("=== phase0 validation start ===")
    cfg = common.SUITES[SUITE]
    results = {"config": {
        "suite": SUITE, "shot": SHOT, "frames_per_demo": FRAMES_PER_DEMO,
        "unfreeze_epochs": UNFREEZE_EPOCHS, "batch_size": BATCH_SIZE,
        "var_weight": VAR_WEIGHT, "n_episodes": N_EPISODES, "max_steps": MAX_STEPS,
        "seeds": SEEDS}, "A_frozen": {}, "B_unfrozen": {}}
    save(results)

    images, actions = load_data(cfg)
    log(f"loaded {len(images)} live samples ({SHOT} demos, ~{FRAMES_PER_DEMO} frames/demo)")

    env_ctx = sweep.make_env(cfg)
    log("env created")
    try:
        # --- A: frozen (one frozen backbone, cache feats once, reuse for all seeds) ---
        frozen_bb = make_bb()
        prompt = prompt_for(frozen_bb, cfg)
        frozen_pooled = frozen_pooled_all(frozen_bb, images, prompt)
        hidden_size = frozen_pooled.shape[-1]
        fb_before = cos_sim_stats(frozen_pooled)
        log(f"A frozen-backbone cos_sim mean/min/max = {fb_before[0]:.4f}/{fb_before[1]:.4f}/{fb_before[2]:.4f}")
        for seed in SEEDS:
            log(f"--- A frozen seed {seed} ---")
            t0 = time.time()
            ev = run_frozen(frozen_bb, frozen_pooled, actions, env_ctx, cfg, hidden_size, seed)
            results["A_frozen"][str(seed)] = {
                "success_rate": ev["success_rate"], "mean_jerk": ev["mean_jerk"],
                "achieved_hz": ev["achieved_hz"], "wall_s": round(time.time() - t0, 1)}
            log(f"A seed {seed} DONE success_rate={ev['success_rate']} jerk={ev['mean_jerk']}")
            save(results)
        del frozen_bb, frozen_pooled
        torch.cuda.empty_cache()

        # --- B: unfrozen + VICReg, fresh backbone per seed ---
        for seed in SEEDS:
            log(f"--- B unfrozen seed {seed} ---")
            t0 = time.time()
            ev, before, after = run_unfrozen(images, actions, env_ctx, cfg, prompt, seed)
            results["B_unfrozen"][str(seed)] = {
                "success_rate": ev["success_rate"], "mean_jerk": ev["mean_jerk"],
                "achieved_hz": ev["achieved_hz"],
                "cos_before_mean": before[0], "cos_after_mean": after[0],
                "cos_after_min": after[1], "wall_s": round(time.time() - t0, 1)}
            log(f"B seed {seed} DONE success_rate={ev['success_rate']} "
                f"cos {before[0]:.3f}->{after[0]:.3f} jerk={ev['mean_jerk']}")
            save(results)
    finally:
        sweep.close_env(env_ctx)

    # verdict
    a = [results["A_frozen"][str(s)]["success_rate"] for s in SEEDS if str(s) in results["A_frozen"]]
    b = [results["B_unfrozen"][str(s)]["success_rate"] for s in SEEDS if str(s) in results["B_unfrozen"]]
    decol = [results["B_unfrozen"][str(s)]["cos_after_mean"] for s in SEEDS if str(s) in results["B_unfrozen"]]
    results["verdict"] = {
        "A_mean_sr": float(np.mean(a)) if a else None,
        "B_mean_sr": float(np.mean(b)) if b else None,
        "B_decollapsed_seeds": int(sum(1 for c in decol if c < 0.9)),
        "n_seeds": len(SEEDS)}
    save(results)
    log(f"=== phase0 complete. verdict: {results['verdict']} ===")


if __name__ == "__main__":
    main()
