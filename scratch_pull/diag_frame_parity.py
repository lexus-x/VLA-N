"""Do the frames the head TRAINS on match the frames it SEES at eval?

Training reads `demo["obs"]["agentview_rgb"]` from the LIBERO hdf5 (sweep.py:96).
Eval reads `obs["agentview_image"]` from OffScreenRenderEnv (sweep.py:227).
Neither path applies any transform. OpenVLA's LIBERO eval applies `img[::-1, ::-1]`
to the env observation, with the comment "rotate 180 degrees to match train
preprocessing" -- i.e. the two sources are NOT in the same orientation.

If that holds here, the policy trains right-side-up and rolls out upside-down.
That reproduces the exact signature we measured: near-perfect held-out open-loop
fit (R^2=0.78, all from hdf5 frames) alongside ~0% closed-loop success.

Resets the env to demo 0's recorded init state and compares the rendered frame
against the hdf5 frame under each candidate transform. Lowest error wins.
"""
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, r"C:\Users\islab01\vla-atlas")
sys.path.insert(0, r"C:\Users\islab01\vla-atlas\experiments\firstpass_sweep")

import common  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

cfg = common.SUITES["libero_long"]

with h5py.File(cfg["hdf5_path"], "r") as f:
    hdf5_img = f["data"]["demo_0"]["obs"]["agentview_rgb"][0].astype(np.float64)
print(f"hdf5  agentview_rgb[0]   shape={hdf5_img.shape} range=[{hdf5_img.min():.0f},{hdf5_img.max():.0f}]", flush=True)

env = OffScreenRenderEnv(bddl_file_name=cfg["bddl_path"], camera_heights=128, camera_widths=128)
env.reset()
init_states = torch.load(cfg["init_states_path"], weights_only=False)
obs = env.set_init_state(init_states[0])
env_img = obs["agentview_image"].astype(np.float64)
print(f"env   agentview_image    shape={env_img.shape} range=[{env_img.min():.0f},{env_img.max():.0f}]", flush=True)

if env_img.shape != hdf5_img.shape:
    print(f"\nSHAPE MISMATCH -- resolution differs, that is a second bug. Stopping.", flush=True)
    env.close()
    sys.exit(1)

# ponytail: RMSE over raw pixels. A 180-deg rotation on a scene with a bright
# table and dark background is unmissable at this scale -- no perceptual metric needed.
candidates = {
    "identity            ": env_img,
    "flip vertical  [::-1]": env_img[::-1],
    "flip horizontal[:,::-1]": env_img[:, ::-1],
    "rotate 180 [::-1,::-1]": env_img[::-1, ::-1],
}
print("\nRMSE of each candidate against the hdf5 frame (lower = same orientation):", flush=True)
errs = {}
for name, cand in candidates.items():
    errs[name] = float(np.sqrt(((cand - hdf5_img) ** 2).mean()))
for name, e in sorted(errs.items(), key=lambda kv: kv[1]):
    print(f"  {name}  RMSE={e:8.3f}", flush=True)

best = min(errs, key=errs.get)
env.close()

print(f"\nBEST MATCH: {best.strip()}", flush=True)
if best.strip() != "identity":
    print(
        "\nCONFIRMED TRAIN/EVAL FRAME MISMATCH. The sweep's ~0% success rates are a\n"
        "preprocessing bug, not a model or representation finding. Apply the winning\n"
        "transform to obs['agentview_image'] in eval_libero, then re-run the sweep.",
        flush=True,
    )
else:
    print(
        "\nFrames already agree. The open-loop/closed-loop gap is NOT an orientation bug;\n"
        "next suspects are compounding error (covariate shift) and MAX_STEPS=250 truncation.",
        flush=True,
    )
