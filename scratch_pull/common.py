"""Shared library for the first-pass 5-head x 2-shot x 3-suite sweep.

Design notes (see SUMMARY.md for the full writeup):
- Feature caching runs ONCE per suite at the 50-demo level; the 10-shot
  condition slices the first 10 demos out of that same cache (demo_id < 10).
  This is why caching is a 3x cost (one per suite) instead of a naive
  5-head x 2-shot x 3-suite = 30x cost.
- Backbone runs in bfloat16 (frozen, inference-only, no training happens
  through it) for a modest ~15% throughput win with zero protocol change;
  heads themselves stay fp32 exactly as they were smoke-tested, so pooled
  features are cast back to float32 at the boundary.
  ponytail: tried batching multiple frames into one processor/model call
  first (the obvious bigger win) but measured NO speedup (0.73-0.77s/frame
  batched vs 0.74s/frame single, likely CPU-preprocessing-bound), so
  dropped it rather than keep chasing. bf16 was the only real, cheap win.
- MAX_STEPS=250 for eval rollouts (vs LIBERO own 600 default) -- approved
  tradeoff to fit a ~10-14h budget instead of ~25h. Applied uniformly to
  every cell in this sweep, so relative comparisons stay valid; absolute
  success rates are NOT comparable to a future full-protocol (600-step) run.
- Meta-World native action space is 4-dim (dx,dy,dz,gripper). To keep the
  shared 7-dim ACTION_DIM / 8-step CHUNK_LEN convention that every head file
  hardcodes, Meta-World actions are zero-padded to 7 dims for training
  targets and the predicted chunk is sliced back to [:4] to step the env.
"""
import sys
import os
import json
import time
import traceback

sys.path.insert(0, r"C:\Users\islab01\vla-atlas")

import numpy as np
import torch
from PIL import Image

from heads.backbone import ACTION_DIM, CHUNK_LEN, SmolVLM2Backbone
from heads.head_ar_tokens import ARTokenHead
from heads.head_fast_tokens import FASTTokenHead
from heads.head_bspline import BSplineHead
from heads.head_flow_matching import FlowMatchingHead
from heads.head_l1_chunk import L1ChunkHead

HEAD_CLASSES = {
    "ar_tokens": ARTokenHead,
    "fast_tokens": FASTTokenHead,
    "bspline": BSplineHead,
    "flow_matching": FlowMatchingHead,
    "l1_chunk": L1ChunkHead,
}
HEAD_NAMES = ["ar_tokens", "fast_tokens", "bspline", "flow_matching", "l1_chunk"]
COMPACT_HEADS = {"ar_tokens", "fast_tokens", "bspline"}
RAW_CHUNK_HEADS = {"flow_matching", "l1_chunk"}

OUT_DIR = r"C:\Users\islab01\vla-atlas\experiments\firstpass_sweep"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_STEPS = 250
N_EPISODES = 30
EPOCHS = 60
BATCH_SIZE = 32
LR = 1e-4
SHOT_LEVELS = [10, 50]
N_DEMOS_FULL = 50

_INIT_STATES = r"C:\Users\islab01\vla-atlas\LIBERO\libero\libero\init_files\libero_10\STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.pruned_init"

SUITES = {
    "libero_long": dict(
        kind="libero",
        hdf5_path=r"C:\Users\islab01\vla-atlas\LIBERO\datasets\libero_10\STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5",
        instruction="pick up the book and place it in the back compartment of the caddy",
        bddl_path=r"C:\Users\islab01\vla-atlas\LIBERO\libero\libero\bddl_files\libero_10\STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl",
        init_states_path=_INIT_STATES,
        task_name="STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        control_hz=20,
    ),
    "libero_plus": dict(
        kind="libero",
        hdf5_path=r"C:\Users\islab01\vla-atlas\LIBERO\datasets\libero_10\STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5",
        instruction="could you please put the book in the caddys back compartment",
        bddl_path=r"C:\Users\islab01\vla-atlas\LIBERO-Plus\libero\libero\bddl_files\libero_10\STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_language_1.bddl",
        init_states_path=_INIT_STATES,
        task_name="STUDY_SCENE1 language_1 paraphrase (same scene as libero_long)",
        control_hz=20,
    ),
    "metaworld_push": dict(
        kind="metaworld",
        instruction="push the puck to the goal position",
        mw_task="push-v3",
        demos_path=os.path.join(OUT_DIR, "metaworld_push_demos.pt"),
        control_hz=80,
    ),
}


def load_backbone(device="cuda"):
    bb = SmolVLM2Backbone(device=device)
    bb.model = bb.model.to(torch.bfloat16)
    return bb


def backbone_pooled(bb, image, instruction):
    _, pooled = bb.forward(image, instruction)
    return pooled.float()


def pad_action_to_7(a4):
    a4 = np.asarray(a4, dtype=np.float32)
    out = np.zeros(7, dtype=np.float32)
    out[:4] = a4
    return out


def build_chunks(actions_2d, chunk_len=CHUNK_LEN):
    T = actions_2d.shape[0]
    chunks = np.empty((T, chunk_len, actions_2d.shape[1]), dtype=np.float32)
    for t in range(T):
        end = t + chunk_len
        if end <= T:
            chunks[t] = actions_2d[t:end]
        else:
            pad = np.repeat(actions_2d[-1:], end - T, axis=0)
            chunks[t] = np.concatenate([actions_2d[t:], pad], axis=0)
    return chunks


# ponytail: ar_tokens/fast_tokens decode through the backbone's own frozen
# embed_tokens/lm_head (see heads/backbone.py docstring), so their __init__
# needs the live backbone object -- the other 3 heads don't. Branch here,
# once, instead of a bb-shaped param threaded through every head class.
BACKBONE_HEADS = {"ar_tokens", "fast_tokens"}


def train_head(head_name, pooled_cache, action_cache, hidden_size, device, bb=None):
    head_cls = HEAD_CLASSES[head_name]
    if head_name in BACKBONE_HEADS:
        head = head_cls(hidden_size, bb).to(device)
    else:
        head = head_cls(hidden_size).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    n = pooled_cache.shape[0]
    loss_curve = []
    bs1 = head_name == "fast_tokens"

    t0 = time.time()
    for epoch in range(EPOCHS):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_steps = 0
        step = 1 if bs1 else BATCH_SIZE
        for b in range(0, n, step):
            idx = perm[b : b + step]
            if idx.numel() == 0:
                continue
            pooled_b = pooled_cache[idx]
            target_b = action_cache[idx]
            opt.zero_grad()
            _, loss = head(pooled_b, target_b)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_steps += 1
        loss_curve.append(epoch_loss / max(1, n_steps))
        heartbeat()  # ponytail: per-epoch proof-of-life; fast_tokens at shot50 can take ~40min/cell with zero other output
    train_seconds = time.time() - t0
    return head, loss_curve, train_seconds


def heartbeat():
    # ponytail: sweep_log.txt only gets a line at cell START/DONE, and a single
    # cell can run ~30-40min (fast_tokens bs=1 training, or a 30-episode eval)
    # with nothing in between -- too coarse for a watchdog to trust. This
    # gives a cheap, frequent timestamp touch a supervisor process can poll
    # without guessing at process/PID liveness.
    with open(os.path.join(OUT_DIR, "heartbeat.txt"), "w") as f:
        f.write(str(time.time()))


def trajectory_jerk(pos_list, dt):
    pos = np.stack(pos_list)
    if pos.shape[0] < 4:
        return None
    vel = np.diff(pos, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    return float(np.mean(np.linalg.norm(jerk, axis=1)))


def append_result(entry, results_path=None):
    if results_path is None:
        results_path = os.path.join(OUT_DIR, "results.json")
    results = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    results = [r for r in results if r.get("cell_id") != entry["cell_id"]]
    results.append(entry)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def log(msg, logfile=None):
    if logfile is None:
        logfile = os.path.join(OUT_DIR, "sweep_log.txt")
    line = "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + str(msg)
    print(line, flush=True)
    with open(logfile, "a") as f:
        f.write(line + "\n")
