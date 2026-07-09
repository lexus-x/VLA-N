import json, math

r = json.load(open(r"C:\Users\islab01\vla-atlas\experiments\firstpass_sweep\refined_eval_100ep.json"))
print("total entries:", len(r))
for e in r:
    p = e["success_rate"]
    n = e["n_eval_episodes"]
    ci = 1.96 * math.sqrt(p * (1 - p) / n)
    print(
        e["cell_id"],
        "| n_success=%d/%d" % (e["n_success"], n),
        "| rate=%.3f" % p,
        "| CI=[%.3f, %.3f]" % (max(0, p - ci), p + ci),
        "| eval_s=%.0f" % e["eval_time_s"],
        "| train_s=%.1f" % e["train_time_s"],
    )
