"""Cheapest possible probe (no model, no GPU): the per-layer diagnostic shows
near-total collapse (cos_sim ~1.0000) already at LAYER 0 (embeddings) --
before any language-model computation happens -- so check whether the RAW
PIXELS themselves are just this similar frame-to-frame (small foreground
robot arm, mostly-static scene/camera), vs. a vision-encoder/preprocessing
artifact.
"""
import h5py
import numpy as np
import sys

sys.path.insert(0, r"C:\Users\islab01\vla-atlas")
import common

cfg = common.SUITES["libero_long"]
frames = []
with h5py.File(cfg["hdf5_path"], "r") as f:
    data = f["data"]
    demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))[:4]
    for dk in demo_keys:
        imgs = data[dk]["obs"]["agentview_rgb"][:]
        for t in [0, imgs.shape[0] // 2, imgs.shape[0] - 1]:
            frames.append(imgs[t].astype(np.float32))

arr = np.stack(frames)  # (N, H, W, 3)
n = arr.shape[0]
print(f"n_frames={n} shape={arr.shape[1:]}", flush=True)

flat = arr.reshape(n, -1)
flat_n = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
sim = flat_n @ flat_n.T
iu = np.triu_indices(n, k=1)
pair_sim = sim[iu]
print(f"raw-pixel cosine sim: mean/min/max = {pair_sim.mean():.4f}/{pair_sim.min():.4f}/{pair_sim.max():.4f}", flush=True)

# per-pixel std across frames, normalized by mean pixel magnitude -- how much
# of the 224x224x3 frame actually changes at all, and what fraction of pixels
# account for most of the variance (foreground-object-is-small check)
std_map = arr.std(axis=0)  # (H, W, 3)
mean_mag = arr.mean()
print(f"per-pixel std: mean={std_map.mean():.3f} max={std_map.max():.3f} (mean pixel magnitude={mean_mag:.1f})", flush=True)
thresh = std_map.mean(axis=-1)
frac_changing = (thresh > thresh.mean() * 2).mean()
print(f"fraction of pixels with >2x average std (i.e. 'the moving part'): {frac_changing * 100:.1f}%", flush=True)

# pixel-space L2 distance vs. the already-measured action L2 distance, so
# they're at least qualitatively comparable in spirit
pix_pair_dist = np.linalg.norm(flat[iu[0]] - flat[iu[1]], axis=1)
print(f"raw-pixel mean pairwise L2 (unnormalized, 224*224*3-dim): {pix_pair_dist.mean():.1f}", flush=True)
