"""v3: same experiment as v2 (unfreeze vision_model+connector + VICReg-style
variance penalty, confirmed working -- BEFORE/AFTER cos_sim 0.9961->0.2845 in
v2), but fixes the batching bug that made v2 crawl and pin GPU1 at 98.5%: v2
looped over BATCH_SIZE images calling bb.model(...) once each and held all of
their forward graphs in memory simultaneously before one backward() call.
This version builds one real batched tensor input (stacked pixel_values +
matching repeated-instruction input_ids) and does ONE bb.model(...) call per
step -- the standard, memory-efficient way to batch a transformer forward.
Same 43-sample/4-demo data, same variance-penalty weight/target, same 15
epochs, batch_size=8 -- only the forward-pass mechanics changed, so a
matching BEFORE/AFTER cos_sim result (and much lower peak memory / faster
wall-clock) confirms the fix is correct, not just different.
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
BATCH_SIZE = 8
EPOCHS = 15
VAR_WEIGHT = 5.0
VAR_TARGET_STD = 1.0

bb = SmolVLM2Backbone(device=device)
print("backbone loaded", flush=True)

vision_params = list(bb.model.model.vision_model.parameters()) + list(bb.model.model.connector.parameters())
for p in vision_params:
    p.requires_grad = True
print(f"trainable (vision_model+connector) params: {sum(p.numel() for p in vision_params) / 1e6:.1f}M", flush=True)

cfg = common.SUITES["libero_long"]
images, actions = [], []
with h5py.File(cfg["hdf5_path"], "r") as f:
    data = f["data"]
    demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))[:4]
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

_messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": cfg["instruction"]}]}]
_prompt = bb.processor.apply_chat_template(_messages, add_generation_prompt=True)


def live_pooled_batch(idxs):
    """ONE batched forward call instead of a Python loop -- this is the fix.
    apply_chat_template's prompt is identical for every sample (same fixed
    per-suite instruction), so it's safe to just repeat it len(idxs) times.
    """
    batch_images = [[images[i]] for i in idxs]  # one inner list per text sample
    inputs = bb.processor(
        text=[_prompt] * len(idxs), images=batch_images, return_tensors="pt", padding=True
    ).to(device)
    out = bb.model(**inputs, output_hidden_states=True)
    return out.hidden_states[-1].mean(dim=1).float()


def measure_collapse(tag):
    with torch.no_grad():
        chunks = [live_pooled_batch(list(range(i, min(i + BATCH_SIZE, n)))) for i in range(0, n, BATCH_SIZE)]
    vecs = torch.cat(chunks, dim=0)
    vn = F.normalize(vecs, dim=-1)
    sim = vn @ vn.T
    iu = torch.triu_indices(n, n, offset=1)
    pair_sim = sim[iu[0], iu[1]]
    print(f"[{tag}] pooled cos_sim mean/min/max = {pair_sim.mean():.4f}/{pair_sim.min():.4f}/{pair_sim.max():.4f}", flush=True)


measure_collapse("BEFORE training")

hidden_size = live_pooled_batch([0]).shape[-1]
head = L1ChunkHead(hidden_size).to(device)
opt = torch.optim.Adam(vision_params + list(head.parameters()), lr=1e-4)

for epoch in range(EPOCHS):
    perm = np.random.permutation(n)
    epoch_task_loss, epoch_var_loss, n_steps = 0.0, 0.0, 0
    for b in range(0, n, BATCH_SIZE):
        idxs = perm[b : b + BATCH_SIZE].tolist()
        if len(idxs) < 2:
            continue
        opt.zero_grad()
        pooled = live_pooled_batch(idxs)
        _, task_loss = head(pooled, action_t[idxs])
        std = pooled.std(dim=0)
        var_loss = F.relu(VAR_TARGET_STD - std).mean()
        loss = task_loss + VAR_WEIGHT * var_loss
        loss.backward()
        opt.step()
        epoch_task_loss += task_loss.item()
        epoch_var_loss += var_loss.item()
        n_steps += 1
    print(
        f"epoch {epoch}: task_loss={epoch_task_loss / n_steps:.4f} var_loss={epoch_var_loss / n_steps:.4f}",
        flush=True,
    )
    common.heartbeat()

measure_collapse("AFTER training")
