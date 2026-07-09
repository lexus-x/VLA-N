"""Cheap (~seconds, CPU-only) diagnostic: before committing to an expensive
backbone-unfreezing experiment, check whether the ALREADY-CACHED pooled
backbone features even vary enough across samples to explain why every GRU
head (ar_tokens, fast_tokens) collapses to a near-constant output. If pooled
vectors are nearly identical across different timesteps/samples within a
single task/scene, no head -- frozen or not -- can learn input-conditioned
outputs from them, and that points at "frozen features are too coarse" (or
"one task/scene gives too little visual variation") rather than "the head
architecture can't decode."
"""
import torch

for suite in ["libero_long", "libero_plus", "metaworld_push"]:
    path = rf"C:\Users\islab01\vla-atlas\experiments\firstpass_sweep\cache_{suite}.pt"
    try:
        cache = torch.load(path, weights_only=False)
    except FileNotFoundError:
        print(f"{suite}: cache not found, skip", flush=True)
        continue
    pooled = cache["pooled"]
    demo_id = cache["demo_id"]
    actions = cache["actions"]

    for shot in [10, 50]:
        mask = demo_id < shot
        p = pooled[mask].float()
        a = actions[mask].float()
        if p.shape[0] < 2:
            continue
        n = p.shape[0]
        pn = torch.nn.functional.normalize(p, dim=-1)
        sim = pn @ pn.T
        iu = torch.triu_indices(n, n, offset=1)
        pair_sim = sim[iu[0], iu[1]]

        # per-dim variance of pooled vectors vs. their mean norm, so the
        # number is scale-free and comparable across suites/hidden sizes
        var_per_dim = p.var(dim=0)
        mean_sq_norm = (p.pow(2).sum(dim=1)).mean()
        rel_var = (var_per_dim.sum() / mean_sq_norm).item()

        # same question for the actions themselves: are the TARGETS varied?
        a_flat = a.reshape(n, -1)
        a_pair_dist = torch.cdist(a_flat, a_flat).mean().item()

        print(
            f"{suite} shot{shot}: n={n} "
            f"pooled_cos_sim(mean/min/max)={pair_sim.mean():.4f}/{pair_sim.min():.4f}/{pair_sim.max():.4f} "
            f"pooled_rel_var={rel_var:.6f} "
            f"action_mean_pairwise_L2={a_pair_dist:.4f}",
            flush=True,
        )
