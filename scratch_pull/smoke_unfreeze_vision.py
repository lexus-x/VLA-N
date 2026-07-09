"""Smoke-scale (~20 samples, ~15 epochs, minutes not hours) test of the
"unfreeze the vision encoder" hypothesis from diag_depth_variance.py /
diag_raw_pixels.py: raw pixels differ meaningfully (cos_sim ~0.97) but the
frozen vision_model+connector collapse them to ~0.999 before the LLM decoder
ever runs. This does NOT touch the real caching pipeline (common.py/sweep.py)
-- it's a standalone, throwaway check of "does gradient signal into
vision_model+connector measurably reduce that collapse at all", before
committing tens of GPU-hours to a real-scale run (full unfreeze loses the
epoch-to-epoch feature-caching optimization common.py relies on).
"""
import sys

sys.path.insert(0, r"C:\Users\islab01\vla-atlas")
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import common
from heads.backbone import SmolVLM2Backbone
from heads.head_l1_chunk import L1ChunkHead

device = "cuda"
bb = SmolVLM2Backbone(device=device)
print("backbone loaded", flush=True)

# Unfreeze vision_model + connector only; text_model + lm_head stay frozen
# (requires_grad=False) but MUST stay outside no_grad so gradients still
# flow back through them to the vision side.
vision_params = list(bb.model.model.vision_model.parameters()) + list(bb.model.model.connector.parameters())
for p in vision_params:
    p.requires_grad = True
n_trainable = sum(p.numel() for p in vision_params)
print(f"trainable (vision_model+connector) params: {n_trainable / 1e6:.1f}M", flush=True)

cfg = common.SUITES["libero_long"]
images, actions = [], []
with h5py.File(cfg["hdf5_path"], "r") as f:
    data = f["data"]
    demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))[:2]
    for dk in demo_keys:
        demo = data[dk]
        imgs = demo["obs"]["agentview_rgb"][:]
        acts = demo["actions"][:].astype(np.float32)
        chunks = common.build_chunks(acts)
        step = max(1, imgs.shape[0] // 10)
        for t in range(0, imgs.shape[0], step):
            images.append(Image.fromarray(imgs[t]))
            actions.append(chunks[t])
n = len(images)
print(f"n_samples={n}", flush=True)
action_t = torch.from_numpy(np.stack(actions)).to(device)


def live_pooled(img):
    """Same as backbone.forward()/common.backbone_pooled but WITHOUT
    @torch.no_grad()/.detach() -- gradients flow through vision_model+connector.
    """
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": cfg["instruction"]}]}]
    prompt = bb.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = bb.processor(text=prompt, images=[img], return_tensors="pt").to(device)
    out = bb.model(**inputs, output_hidden_states=True)
    hidden = out.hidden_states[-1]
    return hidden.mean(dim=1)


def measure_collapse(tag):
    with torch.no_grad():
        vecs = torch.cat([live_pooled(img) for img in images], dim=0).float()
    vn = F.normalize(vecs, dim=-1)
    sim = vn @ vn.T
    iu = torch.triu_indices(n, n, offset=1)
    pair_sim = sim[iu[0], iu[1]]
    print(f"[{tag}] pooled cos_sim mean/min/max = {pair_sim.mean():.4f}/{pair_sim.min():.4f}/{pair_sim.max():.4f}", flush=True)


measure_collapse("BEFORE training")

hidden_size = live_pooled(images[0]).shape[-1]
head = L1ChunkHead(hidden_size).to(device)
opt = torch.optim.Adam(vision_params + list(head.parameters()), lr=1e-4)

EPOCHS = 15
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for i in range(n):
        opt.zero_grad()
        pooled = live_pooled(images[i]).float()
        _, loss = head(pooled, action_t[i : i + 1])
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    print(f"epoch {epoch}: loss={epoch_loss / n:.4f}", flush=True)
    common.heartbeat()

measure_collapse("AFTER training")
