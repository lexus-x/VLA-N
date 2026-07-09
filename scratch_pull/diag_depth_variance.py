"""Cheap probe (~12 frames, no training): diag_pooling_alt.py showed even the
LAST layer's hidden states collapse (cos sim ~0.99) across different robot
states -- so where in the network does the collapse actually originate?
Check cosine similarity of the (last-token) hidden state at every layer depth
to find how many final layers would need to be unfrozen to fix it, before
committing to any backbone-surgery implementation.
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

per_layer_last_tok = None
per_layer_mean_pool = None
n_layers = None

with torch.no_grad():
    for img in frames:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": cfg["instruction"]}]}]
        prompt = bb.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = bb.processor(text=prompt, images=[img], return_tensors="pt").to(device)
        out = bb.model(**inputs, output_hidden_states=True)
        hs = out.hidden_states  # tuple of (n_layers+1) tensors, each (1, seq_len, H)
        if n_layers is None:
            n_layers = len(hs)
            per_layer_last_tok = [[] for _ in range(n_layers)]
            per_layer_mean_pool = [[] for _ in range(n_layers)]
        for i, h in enumerate(hs):
            h0 = h[0].float()
            per_layer_last_tok[i].append(h0[-1].cpu())
            per_layer_mean_pool[i].append(h0.mean(dim=0).cpu())

print(f"n_layers(incl. embeddings)={n_layers}", flush=True)


def pair_sim(vecs):
    v = torch.stack(vecs)
    vn = torch.nn.functional.normalize(v, dim=-1)
    sim = vn @ vn.T
    n = v.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return sim[iu[0], iu[1]]


for i in range(n_layers):
    s_last = pair_sim(per_layer_last_tok[i])
    s_mean = pair_sim(per_layer_mean_pool[i])
    print(
        f"layer {i:2d}/{n_layers - 1}: last_tok cos_sim mean/min={s_last.mean():.4f}/{s_last.min():.4f}  "
        f"mean_pool cos_sim mean/min={s_mean.mean():.4f}/{s_mean.min():.4f}",
        flush=True,
    )
