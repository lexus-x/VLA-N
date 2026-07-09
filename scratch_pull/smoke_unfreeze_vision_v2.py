"""v2 of smoke_unfreeze_vision.py: the v1 result was a clean NEGATIVE finding
-- unfreezing vision_model+connector with plain L1 loss on 22 samples made
the collapse WORSE (cos_sim 0.9961 -> 1.0000), not better. Classic
representation collapse: with a big encoder, tiny data, and no pressure to
stay spread out, gradient descent finds it easier to collapse the encoder's
output to a single point and have the head just predict the mean/median
target than to actually learn to discriminate frames.

Standard fix from the self-supervised-learning literature (VICReg-style):
add an explicit per-dimension variance-preservation term over each batch of
pooled vectors, penalizing collapse directly instead of hoping the task loss
alone prevents it. This needs real batches (not single-sample SGD) to have a
batch-level std to penalize, so this version batches BATCH_SIZE frames/step
instead of v1's one-at-a-time updates, and uses somewhat more samples so
there's enough diversity per batch for the variance term to mean something.
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


def live_pooled_batch(idxs):
    vecs = []
    for i in idxs:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": cfg["instruction"]}]}]
        prompt = bb.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = bb.processor(text=prompt, images=[images[i]], return_tensors="pt").to(device)
        out = bb.model(**inputs, output_hidden_states=True)
        vecs.append(out.hidden_states[-1].mean(dim=1))
    return torch.cat(vecs, dim=0).float()


def measure_collapse(tag):
    with torch.no_grad():
        vecs = live_pooled_batch(range(n))
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
        idxs = perm[b : b + BATCH_SIZE]
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
