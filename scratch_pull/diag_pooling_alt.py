"""Cheap probe (~20 frames, no training): the pooled-variance diagnostic showed
cosine similarity ~0.996-0.999 across wildly different libero_long timesteps,
using the current pooling: hidden.mean(dim=1) over the FULL token sequence
(system/chat template + fixed instruction text + image patch tokens). If that
fixed-per-suite instruction text dominates the mean (same tokens every single
frame), the pooled vector would be nearly frame-invariant regardless of
backbone freezing -- a pooling-method bug, not a "frozen features" ceiling.
Test by re-running the backbone on a handful of real frames spanning several
demos and comparing full-sequence mean pooling vs. image-token-only mean
pooling vs. last-token hidden state.
"""
import sys

sys.path.insert(0, r"C:\Users\islab01\vla-atlas")
import h5py
import numpy as np
import torch
from PIL import Image

import common
from heads.backbone import SmolVLM2Backbone

device = "cuda"
bb = SmolVLM2Backbone(device=device)
print("backbone loaded", flush=True)

cfg = common.SUITES["libero_long"]
frames = []
with h5py.File(cfg["hdf5_path"], "r") as f:
    data = f["data"]
    demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))[:4]
    for dk in demo_keys:
        imgs = data[dk]["obs"]["agentview_rgb"][:]
        for t in [0, imgs.shape[0] // 2, imgs.shape[0] - 1]:
            frames.append(Image.fromarray(imgs[t]))
print(f"n_frames={len(frames)}", flush=True)

# Find the image-placeholder token id: whatever token id repeats a large,
# constant count in every prompt (the expanded image-feature slots) is it.
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": cfg["instruction"]}]}]
prompt = bb.processor.apply_chat_template(messages, add_generation_prompt=True)
probe_inputs = bb.processor(text=prompt, images=[frames[0]], return_tensors="pt").to(device)
ids = probe_inputs["input_ids"][0].tolist()
from collections import Counter
counts = Counter(ids)
image_token_id, image_token_count = counts.most_common(1)[0]
print(f"seq_len={len(ids)} most_common_token_id={image_token_id} count={image_token_count}", flush=True)
print(f"decoded most-common token: {bb.processor.tokenizer.decode([image_token_id])!r}", flush=True)

full_pooled, img_pooled, last_pooled = [], [], []
with torch.no_grad():
    for img in frames:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": cfg["instruction"]}]}]
        prompt = bb.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = bb.processor(text=prompt, images=[img], return_tensors="pt").to(device)
        out = bb.model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1][0].float()  # (seq_len, H)
        ids_t = inputs["input_ids"][0]
        img_mask = ids_t == image_token_id

        full_pooled.append(hidden.mean(dim=0).cpu())
        if img_mask.any():
            img_pooled.append(hidden[img_mask].mean(dim=0).cpu())
        else:
            img_pooled.append(hidden.mean(dim=0).cpu())
        last_pooled.append(hidden[-1].cpu())


def report(name, vecs):
    v = torch.stack(vecs)
    vn = torch.nn.functional.normalize(v, dim=-1)
    sim = vn @ vn.T
    n = v.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    pair_sim = sim[iu[0], iu[1]]
    print(f"{name}: cos_sim mean/min/max = {pair_sim.mean():.4f}/{pair_sim.min():.4f}/{pair_sim.max():.4f}", flush=True)


report("full-sequence mean pool (current)", full_pooled)
report("image-token-only mean pool", img_pooled)
report("last-token hidden state", last_pooled)
