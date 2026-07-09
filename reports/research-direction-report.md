# Research-Direction Report: A Novel <500M VLA / Module Under 1×A100, Publishable in Q1/Q2

## 1. TL;DR

**The honest starting point:** your literal goal — a novel <500M model that *beats* much larger SOTA on standard LIBERO — is already commoditized. VLA-Adapter (0.5B backbone) hits 97.3% LIBERO average, edging OpenVLA-OFT-7B (97.1%); VLA-0-Smol (~500M) reaches 94.1%. Standard LIBERO is saturated (94–99% across 0.5B–7B). So "small beats big on LIBERO average" is not a publishable headline anymore — someone already published it.

**Recommended direction #1:** Build a rigorous, single-protocol **action-representation atlas at a fixed <500M budget**, whose one load-bearing new result is a **few-shot data-efficiency crossover** (10/25/50 demos): which action head lets a sub-500M model win *where data is scarce and benchmarks still discriminate* — LIBERO-Long, LIBERO-Plus per-factor, MetaWorld-hard — reported with smoothness/latency/GPU-hours nobody co-reports. Target TMLR (or RA-L with a small real-robot slice). Win on discriminative axes, not headline average.

**One-line backup:** A **compute-normalized test-time-scaling study** for sub-500M VLAs (cached-backbone best-of-N + a genuinely-underexplored retrieved-waypoint progress verifier + test-time augmentation), reporting the success-per-FLOP / success-per-latency Pareto frontier with test-time compute applied to *both* the small and the large model for a fair comparison.

---

## 2. Landscape (verified)

Standard 4-suite LIBERO (Spatial/Object/Goal/Long → average). All numbers traced to sources opened during fact-checking; confidence and caveats marked.

| Model | Params | Action decoding | LIBERO avg | Notes / confidence |
|---|---|---|---|---|
| SimpleVLA-RL | RL-tuned OFT (~7B base) | online RL | **99.1%** | avg CONFIRMED; **per-suite breakdown REFUTED** — the widely-copied Spatial 99.4/Object 99.1/Goal 99.2/Long 98.5 is scrambled. Correct row (arXiv 2509.09674 Table 2): Spatial 99.1/Object 99.2/Goal 98.5/Long 99.1. |
| RLinf-VLA | RL-tuned | online RL | **98.11%** | across 130 tasks (2510.06710). CONFIRMED. |
| VLA-Adapter-Pro | ~0.6B | adapter/bridge head | 98.5% | self-reported (2509.09372). |
| PokeVLA | 1.22B | transformer head | 98.2% | 2026 preprint, self-reported, UNCERTAIN (no independent replication). Also 83.5% LIBERO-Plus (trained-on-Plus) / **79.3% transfer** — cite the right one per context. |
| GeoVLA | ~large | flow + 3D | 97.7% | self-reported. |
| VLA-Adapter | **~597M** (0.5B Qwen2.5 + 97M head) | bridge-attention L1 head | **97.3%** | CONFIRMED (2509.09372). ~8h/1 consumer GPU, no robotic pretraining, 219 Hz. **Slightly over 500M.** |
| OpenVLA-OFT | 7B | parallel decode + L1 chunks | **97.1%** | CONFIRMED (2502.19645). Protocol-dependent: reported 95.3% (GeoVLA table), 91.9% (VLA-0 no-pretrain table). |
| VLA-0 | ~3B (Qwen2.5-VL) | integer-text tokens | 94.7% | CONFIRMED (2510.13054). Best "no-pretraining" method. |
| **VLA-0-Smol** | **~500M** (SmolVLM2-500M) | integer-text tokens | **94.1%** | Spatial 92.2/Object 97.2/Goal 95.6/Long 91.2. **Best truly sub-500M.** UNCERTAIN: self-published project page, not peer-reviewed, no independent replication — but it is a real checkpoint with code/weights (some critics wrongly called it "fabricated"). |
| pi0 | ~3.3B | flow matching | 94.2% | CONFIRMED. Long only 85.2. |
| CogACT | ~7B | diffusion | 93.2% | |
| SmolVLA | **0.45B** | flow matching | **87.3%** | CONFIRMED (2506.01844); Long 71. UNCERTAIN: third-party re-evals report lower; protocol-dependent. Trains ~4h/20k steps on 1 A100. |
| pi0-FAST | ~3B | FAST/DCT tokens | 85.5% (or **71.8** with large-scale pretraining) | two numbers exist — 86.0 LIBERO-only vs 71.8 with-pretrain. Cite carefully. |
| OpenVLA (orig) | 7B | discrete AR tokens | 76.5% | standard baseline. CONFIRMED. |
| Octo-Base | 93M | diffusion | 75.1% | Long only 51.1. |

**Robustness (LIBERO-Plus, the real open axis):** models at ~95% standard LIBERO collapse to **17–70% overall** under viewpoint/initial-state/lighting/distractor perturbations; best is **79.6%** only after dedicated robustness post-training; PokeVLA-1.22B **83.5%** (trained-on-Plus). Worst factors: robot-initial-state (~32% for OFT) and camera viewpoint (~56%). CONFIRMED (2510.13626).

**Meta-World MT50 (LeRobot IL, 50 demos/task):** small purpose-built VLAs beat larger generalists — Evo-1 (0.77B) 80.6% > SmolVLA (2.25B) 68.2% > pi0 (3.5B) 47.9% > TinyVLA-H (1.3B) 31.6% > Diffusion Policy 10.5%. CONFIRMED (2511.04555, self-reported). ProgVLA (0.1B) 78.5%.

**The exact gap to close:** on standard LIBERO average, essentially **none** — a 0.5B model already ties/beats 7B. The gap that actually exists and discriminates: (a) **long-horizon** (SmolVLA Long 71 vs OFT 94.5, a ~23-pt gap); (b) **robustness** (best 79.6–83.5%, most models 17–70%); (c) **MetaWorld-hard tiers** (SmolVLA ~52–70%); (d) **few-shot / data-efficiency at <500M** (largely unmeasured under a fixed protocol). These are where a win is meaningful.

> Do **not** use the "success drops to 0.08% for five chained instructions" statistic — flagged unverifiable twice.

---

## 3. Where the whitespace is

Standard LIBERO average is dead as a target. The real open problems, filtered to those that *structurally favor* a small model:

1. **No controlled action-representation comparison at a fixed <500M budget.** Every small-VLA paper reports its head on its own backbone/protocol; cross-paper variance is 2–6 pts, so "which representation wins at 500M" is literally unanswerable. StarVLA-alpha did a controlled sweep — but at 2–8B, on saturated ID LIBERO, with only 4 heads. Sub-500M + newer heads (integer-text, B-spline, VQ) + the discriminative axes (LIBERO-Plus, MetaWorld-hard, jerk, Hz, few-shot) is genuinely un-done.
2. **Few-shot data-efficiency of action representations is uncharacterized.** At 10–25 demos the compact-output-dimensionality heads (K spline/DCT coefficients) plausibly win by a *wider* margin than at full data — the deployability evidence reviewers actually want. This is the one place a clean new empirical result lives.
3. **Compute-normalized test-time scaling on small VLAs.** The action head is ~5× cheaper than the backbone, so best-of-N over a cached backbone is nearly free — but the field has each lever (verifier BoN, self-consistency, retrieval) and no clean matched-FLOP comparison at <500M on both LIBERO-Long and LIBERO-Plus.
4. **Long-horizon collapse and action smoothness across chunk boundaries** are size-independent decoder problems (repeatedly flagged in surveys) — but the cleanest mechanisms here (RTC-class, continuation-conditioning) are already published (see §6).

The honest read: the whitespace is **measurement and characterization**, not a new mechanism. Almost every mechanism that would "beat bigger" has a 2025–2026 paper on it already.

---

## 4. Recommended Direction #1 — The <500M Action-Representation Atlas + Few-Shot Crossover

### Mechanism
Fix **one** sub-500M backbone (SmolVLM2-500M, the VLA-0-Smol / SmolVLA class), **one** finetune dataset, **one** frozen eval protocol (rollouts, seeds, checkpoint). Swap **only the action head** across a matched set — cap at **4–5 arms** (the 7-head version is a time sink and an integration-confound generator):
- (a) autoregressive integer-text tokens (VLA-0),
- (b) FAST/DCT discrete tokens (pi0-FAST),
- (c) B-spline / compact-coefficient tokens (BEAST),
- (d) flow-matching expert (SmolVLA/pi0),
- (e) L1 continuous-chunk regression (OFT recipe).

Report each on: LIBERO 4-suite + **Long**, **LIBERO-Plus per-factor**, **MetaWorld hard/very-hard**, and four axes nobody co-reports — **sample efficiency (10/25/50/full demos), trajectory jerk, control-rate Hz, single-A100 GPU-hours**. The single load-bearing new result: the **few-shot crossover** — at ≤25 demos, does the compact-output-dimensionality head (K≈4–5 coefficients) dominate the raw-chunk head, and by how much, and where does the curve cross?

### Why it's novel (vs named prior art)
- **StarVLA-alpha (2604.11757)** did a controlled head sweep — but at 2–8B, on saturated ID LIBERO, 4 heads, and concluded "given a strong VLM, head choice barely matters." That conclusion is a *loaded gun*: your paper must be framed as "does that hold at 500M and on discriminative axes?" not "first controlled sweep."
- **OpenVLA-OFT** ablated discrete/continuous/diffusion/L1 — at 7B, standard LIBERO only.
- **BEAST (2506.06072)** sells compact-spline sample-efficiency — but not as a controlled cross-head crossover-point measurement at fixed <500M.
- **AnchorRefine (2604.17787)** is the heterogeneous gripper-event head — cite it, don't claim it.

The residual, defensible contribution: **the sub-500M budget, the few-shot crossover point + best coefficient count for ≤25-demo adaptation, and the co-reported efficiency/robustness axes under one protocol.** That is real but incremental — position it honestly.

**Kill the overclaim the original pitch made:** "minimizing output dimensionality is a few-shot advantage *only a small model can claim*" is **logically false** — output dimensionality is orthogonal to backbone capacity (a 7B model with the same K-coefficient head reaps the same variance reduction). All three critics caught this. Demote it to: "at a fixed <500M budget, compact-output heads help few-shot; here is the measured crossover." Do not headline a scaling law.

### Why it fits <500M + 1 A100 + easy-to-build
Pure head swaps on one open 500M checkpoint reusing existing code (VLA-0, SmolVLA, BEAST, FAST, OFT all open). Each full-data finetune ~20h on 1 A100 (SmolVLA-verified 4h/20k steps). Few-shot (10-demo) runs are a few hours. No from-scratch pretraining. **Realistic budget: ~500–1000 A100-hours** once slow LIBERO/MetaWorld rollout *evaluation* is counted (the hidden cost — budget for it). Cap to 4–5 heads / 2–3 suites / 3 seeds to stay in weeks, not months.

### Build plan
- **Backbone:** SmolVLM2-500M (stay honestly under 500M; note VLA-0-Smol adds an action head, so verify total ≤500M).
- **Data/protocol:** LIBERO 4-suite standard demos; subsample to 10/25/50 for the few-shot arm. Freeze rollout count (≥500/suite), 3 seeds, report mean ± std.
- **The confound you must address head-on:** you cannot "swap only the head" — VLA-0 rides on temporal ensembling + whole-action masking; OFT on parallel decoding + FiLM; flow needs its own sampler. **Design decision to state explicitly:** for at least one head, report *both* a matched-protocol arm (isolates the head) *and* a native-recipe arm (each head at its reported strength), so reviewers see the "controlled = crippled variant" tension is handled, not hidden.
- **Baselines:** VLA-Adapter (0.5B, 97.3%), SmolVLA (450M), and the large anchors (OFT-7B 97.1, pi0 94.2) as reference lines — not strawmen.
- **Ablations:** coefficient count K sweep for the compact head; jerk vs SR; demos-vs-SR crossover per head; heterogeneous gripper-event variant to protect contact events.
- **Stats:** 3 seeds minimum, report variance; the few-shot crossover must be reproducible across seeds or it isn't a result.

### Expected results and honest probability of "beating bigger"
- Few-shot crossover characterized (compact head wins at 10–25 demos): **plausible, ~50–60%** it's clean and reproducible; the risk is StarVLA's "head barely matters" + strong pretrained prior washing out head differences at low data.
- Beating a 3B/pi0 or 7B/OFT on a *specific discriminative axis* (LIBERO-Long few-shot, or MetaWorld-hard) at <500M: **~30–40%.**
- Beating SOTA on LIBERO average as a novel claim: **near zero** — already done by others; do not promise it.

### Key risks + mitigations
- **Saturation erases signal on standard suites** → flee to Long/Plus/MetaWorld-hard/few-shot from the start; standard-LIBERO is a sanity check, not the result.
- **Integration confound ("your port or the head?")** → build on StarVLA's modular codebase if possible (also further erodes novelty — trade-off to accept), report native+matched arms.
- **Compact base underfits contact-rich MetaWorld-hard, flipping the win** → report where the crossover actually occurs rather than assuming a uniform win; the honest map *is* the contribution.
- **Reviewer reads it as an analysis paper** (RA-L/T-RO want a method/hardware) → target TMLR, or add a small real-robot data-efficiency slice for RA-L.

### Fallback if the main hypothesis fails
If the few-shot crossover is noisy/absent, the paper survives as the **Pareto/robustness atlas alone** (success-per-FLOP, jerk, Hz, LIBERO-Plus per-factor under one protocol) — a legitimate TMLR correctness-bar contribution that dissolves the 2–6 pt cross-paper confound. That deliverable is durable even if the sharp claim dies.

---

## 5. Runner-Up Direction #2 — Compute-Normalized Test-Time Scaling for Sub-500M VLAs

**Mechanism:** Base = SmolVLA / VLA-0-Smol. Run the expensive backbone once, KV-cache features, buy diversity cheaply in the ~100M action head: (a) K noise-seed candidates reranked by training-free action-space **medoid self-consistency**; (b) a small learned verifier on cached features; (c) a **retrieved-waypoint progress verifier** (FAISS index of demo states → stored next sub-goal; score candidates by progress toward the retrieved goal — zero verifier training); plus **test-time augmentation** (K action-preserving photometric augmentations, aggregated by medoid) for robustness. Adaptive gating: default K=1, escalate only on disagreement.

**Why it's (partly) novel — and where it isn't:** cached-backbone BoN is **MG-Select / RoVer** (already published, same latency argument); verifier BoN + majority vote is **RoboMonkey**; TTA on VLA action prediction *already exists* (2604.18107, +7.4% LIBERO) — so drop the "first/never" claims. The **one genuinely underexplored piece** is the retrieved-waypoint *progress* verifier (a twist on E-TTS's buffer / Retrieve-then-Steer). Lead with that plus the compute-normalized frontier.

**The non-negotiable fairness fix:** the "small+search beats 7B single-pass" claim is a rigged frontier. Apply test-time compute to **both** models (large models also gain from TTS). Replace "beats 7B" with an explicit **iso-FLOP / iso-latency Pareto** claim + an honest failure map (TTA cannot fix viewpoint/init-state — the dominant LIBERO-Plus failures — say so).

**Fit:** fully feasible; candidate sampling, medoid, retrieval, TTA are inference-only; FAISS index is a one-time embedding pass; small verifier trains in hours. All <500M active params.

**Honest probability:** matches a 7B at equal FLOP: likely. *Beats* it on raw SR on Long/Plus: **~20–30%** (dominant failures are exactly where these levers are weak). Publishable as a compute-normalized study: **medium** (TMLR/RA-L), landmark: no.

**Fallback:** the compute-normalized frontier + retrieved-waypoint verifier stand alone as a measurement paper even if no SR win over the large model materializes.

---

## 6. Directions considered and rejected

- **Budget-inverted geometry allocation (sub-500M 3D VLA + "geometry substitutes for LLM scale" law)** — KILLED. Action-head geometry shortcut = FALCON (2510.17439); depth-PE = SpatialVLA Ego3D-PE (2501.15830); "vision not LLM is the bottleneck" = VLM4VLA (2601.03309). Ground-truth sim depth is an oracle the field is abandoning (Spatial Forcing 2510.12276). Budget arithmetic incoherent (SigLIP tower alone busts <500M).
- **Robustness by structural freezing (frozen SSL encoder + thin task stream)** — KILLED. Exact architecture already published: "Preserving Pretrained Representations" (2509.11417); fully-frozen encoder *empirically collapses* (VLM4VLA: −24 to −42 pts). LIBERO-Plus's own failures are language/init-state, not encoder warping.
- **Continuation-conditioned AR decoding (sampler-free RTC for text-token VLAs)** — KILLED. = 2512.05964 (train-time prefix conditioning) + 2606.13355 (real-time AR execution); and VLA-0 already runs per-step temporal ensembling that suppresses boundary jerk, making the mechanism largely redundant.
- **Action-invariant consistency + free sim re-render** — KILLED. The re-render-preserves-EE-labels insight *is* LIBERO-Plus's own construction/training pipeline (2510.13626); action-invariance + init-state/layout exclusion = 2510.00037; camera-pose re-render on LIBERO = 2603.29192. "Held-out families" claim is self-contradictory (trains on the same families it tests).
- **Self-generated failure DPO + sim-privileged divergence localization** — KILLED (2/3, but killed). Offline failure-as-preference = GRAPE (2411.19309); self-generated recovery = PLD / AFIL (2605.08434); premise collapses because SimpleVLA-RL shows the free sim predicate makes *online* RL cheap and already SOTA (98.5% Long).
- **On-policy distillation from an RL-saturated 7B teacher into <500M** — KILLED. = VLA-OPD (2603.26666), same SimpleVLA-RL teacher; its distill-only ceiling was **78.9%** (needed added GRPO to reach parity), refuting the "student reaches 96–98% without RL" claim. A student is upper-bounded by its teacher, so it can't *beat* the large SOTA by construction.

---

## 7. Publication plan

**Targets (robotics, all Scimago Q1 for robotics 2024 per aggregators — verify on scimagojr.com before submission):**
- **IEEE RA-L (Q1)** — the most common venue for compact/efficient-VLA papers (TinyVLA precedent, RA-L 2025). Best fit **if** you add a small real-robot data-efficiency slice. Real-robot validation is *effectively expected* for a policy-learning contribution here.
- **TMLR** — the sim-only fallback, correctness-not-novelty bar, no fees, OpenReview. **Re-verified 2026-07-03:** Scimago SJR is a confirmed **Q2** (0.744, AI and CV categories) — not "unresolved" as originally reported. It is **not JCR/Web-of-Science indexed** (expected — launched 2022, WoS typically needs 2-3 years of coverage), so no Clarivate Impact Factor exists. If your institution requires a JCR-indexed venue specifically, TMLR won't count; if Scimago Q2 satisfies the requirement, it does. Confirm which standard your committee uses before choosing it.
- Autonomous Robots / RAS — **SJR Q1** but **JCR mixed** (Q2 in "CS, Artificial Intelligence", Q1 in "Robotics"); real-robot validation not mandatory at either.
- **Full venue ranking (quartile-verified 2026-07-03):** IEEE T-RO (Q1/Q1, IF 11.1) and IJRR (Q1/Q1, IF 7.7) are the highest-impact options but culturally expect mature/real-robot work, a poor fit for a compact sim study. IEEE T-ASE (Q1/Q1, IF 7.9) has a high IF but weak thematic fit (automation/deployed-systems focus). **RA-L has a genuine SJR-Q1-vs-JCR-Q2 discrepancy** (JCR figure moderately-confident only, Clarivate portal paywalled). For this paper's actual profile (sim-heavy, sub-500M), the realistic best-fit ranking is **CoRL (top-tier conference, CFP explicitly sanctions sim-only submissions) → RA-L → ICRA → RAS/Autonomous Robots → TMLR**, with CoRL/ICRA/RSS/NeurIPS workshops as a non-archival fallback tier only, not a standalone publication.

**Sim-only sufficient?** For TMLR, yes (benchmark/analysis contributions are in-scope). For RA-L/T-RO, plan a **small real-robot data-efficiency demo** — even efficiency-first small-VLA journal papers ship real hardware (TinyVLA: bimanual UR5). CoRL reviewer norms: sim-only survives only with credible transfer / data-efficiency / sound-analysis argument. A 10-demo real-robot few-shot slice directly supplies that and is the strongest evidence for Direction #1's crossover claim.

**The story/claims the paper makes (Direction #1):**
1. Under one fixed <500M protocol, we resolve the 2–6 pt cross-paper confound and report *which* action representation wins on the axes that still discriminate (Long, LIBERO-Plus per-factor, MetaWorld-hard, jerk, Hz, GPU-hours).
2. We characterize a **few-shot crossover**: at ≤25 demos, compact-output-dimensionality heads dominate raw-chunk heads at fixed <500M, with a measured best coefficient count — the deployability finding.
3. We do **not** claim to beat SOTA on LIBERO average (saturated/already done) or a size-coupled scaling law (logically unsupported); we claim a controlled, honest sub-500M atlas + a reproducible few-shot result.

---

## 8. Honest risk assessment

**Top 3 ways this fails:**
1. **Saturation + StarVLA's "head barely matters" hold at 500M too** → the atlas shows no meaningful ranking flip and the few-shot crossover is within seed noise. This is the single biggest risk (~40%). Mitigation: the axes chosen (few-shot, Plus per-factor, MetaWorld-hard) are the ones *least* saturated; a clean *negative* result ("head choice doesn't matter even at 500M, here's the honest Pareto") is still TMLR-publishable, just less exciting.
2. **Integration confound sinks the "controlled" claim** → a mis-ported head underperforms its paper and reviewers reject fairness. Mitigation: native+matched dual arms, build on existing modular code.
3. **Reviewers demand a method/real-robot and desk-reject the analysis framing** at RA-L/T-RO. Mitigation: TMLR primary, or the real-robot few-shot slice.

**Confidence that the researcher achieves a *publishable* result under these constraints (1×A100, <500M, LIBERO/MetaWorld): MEDIUM-HIGH.** The build is cheap, reproducible, and the fallback (atlas without the crossover) is itself publishable at TMLR.

**Confidence of a result that genuinely *beats much larger SOTA* as a novel headline: LOW.** That framing is largely foreclosed by 2025–2026 work (VLA-Adapter, SmolVLA, VLA-0-Smol already did "small ties/beats big"). The achievable and honest win is *discriminative-axis dominance + a controlled measurement*, targeted at TMLR/RA-L — not a leaderboard headline.

---

## 9. Sources

- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- VLA-Adapter: https://arxiv.org/abs/2509.09372
- VLA-0 / VLA-0-Smol: https://arxiv.org/abs/2510.13054 · https://robot-learning-collective.github.io/vla-0-smol
- SmolVLA: https://arxiv.org/abs/2506.01844 · https://huggingface.co/docs/lerobot/en/smolvla
- pi0: https://arxiv.org/abs/2410.24164
- LIBERO-Plus: https://arxiv.org/abs/2510.13626
- PokeVLA: https://arxiv.org/html/2604.20834
- SimpleVLA-RL: https://arxiv.org/abs/2509.09674 · RLinf-VLA: https://arxiv.org/abs/2510.06710
- Evo-1 (Meta-World MT50): https://arxiv.org/abs/2511.04555 · ProgVLA: https://arxiv.org/abs/2605.28231
- StarVLA-alpha (controlled head sweep): arXiv 2604.11757 · BEAST: https://arxiv.org/abs/2506.06072
- Rejected-direction prior art: FALCON 2510.17439 · VLM4VLA 2601.03309 · SpatialVLA 2501.15830 · Spatial Forcing 2510.12276 · "Preserving Pretrained Representations" 2509.11417 · Training-time RTC 2512.05964 · AR real-time execution 2606.13355 · Multi-modal robustness 2510.00037 · GRAPE 2411.19309 · VLA-OPD 2603.26666 · RoboMonkey 2506.17811 · MG-Select 2510.05681
- Venues: RA-L https://www.resurchify.com/impact/details/21100900379 · CoRL reviewer guidance https://www.corl.org/contributions/instruction-for-reviews · TinyVLA (RA-L 2025) https://arxiv.org/abs/2409.12514

> Caveats carried throughout: PokeVLA/SimpleVLA-RL/Evo-1 are 2025–2026 self-reported preprints (no independent replication); VLA-0-Smol's 94.1% rests on one non-peer-reviewed project page; SmolVLA's 87.3% is protocol-dependent; the SimpleVLA-RL per-suite breakdown circulating in aggregators is refuted (only the 99.1% avg and the corrected suite values are reliable). Several "Q1" quartile figures were read from aggregators (scimagojr returned 403) — verify before citing in a manuscript.

---

## 10. Novelty Sweep Addendum (run 2026-07-02)

A dedicated 99-agent novelty sweep (5 search angles, 17 primary sources fetched, 25 load-bearing claims independently 3-vote verified: 22 confirmed / 3 refuted / 0 unresolved) stress-tested Direction #1 against every named piece of prior art plus a scoop check of Feb–Apr 2026 preprints.

**Verdict: PARTIALLY NOVEL.** No single paper runs the full combination (5-head atlas × fixed <500M backbone × 10/25/50-demo few-shot crossover × joint LIBERO-Plus + Meta-World-hard evaluation × jerk/Hz/GPU-hour co-reporting). But most individual axes have partial precedent that must be cited and differentiated from, not claimed as untouched.

| Paper | What it actually verified to contain | Overlap with Direction #1 |
|---|---|---|
| **SmolVLA** (2506.01844, 450M) | 2-head ablation (flow-matching vs. L1) on standard LIBERO **+** separate Meta-World hard/very-hard eval — but disjoint parts of the paper, single head each, full-data only | **Closest single piece of prior art.** Must be cited as the nearest anchor point. |
| **BEAST** (2506.06072) | Controlled 3-way discrete-tokenizer comparison (FAST/binning/BEAST) on one fixed backbone | Backbone is **~0.77B (Florence-2-large)**, over the 500M budget; no flow-matching/L1 head; no LIBERO-Plus/Meta-World; no few-shot axis. A claimed sub-500M BEAST-SF variant running this comparison was checked and **refuted** (0-3 vote) — no confirmed sub-500M version exists. |
| **OpenVLA-OFT** (2502.19645) | L1 vs. discrete vs. diffusion head ablation | At **7B** only, standard LIBERO only, predates LIBERO-Plus by ~8 months. |
| **StarVLA-alpha** (2604.11757) | Head sweep (MLP/OFT-style/GR00T-style/flow-matching) concluding "head choice barely matters given a strong backbone" | At **2–8B**, on standard LIBERO/SimplerEnv/RoboTwin/RoboCasa. This conclusion is the single biggest risk to Direction #1 — it must be engaged head-on, not ignored (medium-confidence claim, 2-1 vote, re-verify exact wording before quoting). |
| **VLA-0** (2510.13054, the real paper) | Correction: actual backbone is **3B** (Qwen-VL-2.5), not ~500M. Ablations never swap the action-generation mechanism — not a cross-head sweep at all. "VLA-0-Smol" is a separate non-arXiv derivative (Robot Learning Collective), not part of this paper. | Low overlap once corrected — but the report's earlier framing of "VLA-0-Smol as ~500M integer-token head" needs a citation caveat distinguishing the arXiv paper from the third-party reproduction. |
| **LIBERO-Plus** (2510.13626) | Robustness benchmark | Tested on **10 model variants, all 3B+.** Zero sub-500M models. Never paired with Meta-World or any few-shot axis anywhere in the paper. |
| **LIBERO-X, VQ-VLA, "Not All Features Are Created Equal"** (Feb–Mar 2026 scoop check) | Different robustness taxonomy / tokenizer-scaling study / mechanistic-interpretability study confounding backbone and head | **No overlap found** — none run a controlled <500M head sweep, few-shot crossover, or the LIBERO-Plus+Meta-World pairing. |

**Confirmed-open slices (genuinely absent everywhere searched):**
1. A full 5-head taxonomy compared under one fixed <500M backbone (existing work caps at 2–3 heads at or near this scale).
2. Any few-shot/low-data (10/25/50-demo) crossover analysis for action-head comparison, at *any* backbone scale.
3. LIBERO-Plus evaluation of any sub-500M model.
4. Any paper pairing LIBERO-Plus-style robustness with Meta-World hard/very-hard in the same study.

**New feasibility flag (not just novelty):** Meta-World hard/very-hard and LIBERO-Plus-style perturbations have never been paired in one study in the surveyed literature — this may reflect real simulator/task-suite integration friction, not just an unclaimed research gap. Budget time to verify both harnesses install and run together before committing to the joint-evaluation design.

**Unresolved (not confirmed either way):** whether any paper anywhere co-reports trajectory jerk + control-rate Hz + GPU-hours alongside success rate for an action-head comparison. This specific metric-combination axis was not directly investigated — worth a targeted follow-up search, but don't claim it as a novelty pillar until checked.

**Caveats on the sweep itself:**
- Search coverage was arXiv/openaccess-centric; CoRL/ICRA/RSS 2025–2026 **workshop papers** were only spot-checked, not systematically searched — a dedicated workshop-proceedings pass is recommended close to submission time.
- The "recent preprints" scoop check surfaced 3 candidates in the Feb–Apr 2026 window; this is not confirmed exhaustive — re-run a fresh arXiv scan right before submission, since this is a fast-moving area.
- Several cited papers (StarVLA-alpha, LIBERO-X) are dated after this assistant's nominal knowledge cutoff (Jan 2026) — they were verified via live fetch of arXiv pages (direct quotes captured), not from memory, but independently re-confirm arXiv IDs 2604.11757 and 2602.06556 resolve correctly before citing them in a manuscript.

**Net effect on the recommendation:** Direction #1 survives as **novel-as-a-combination**, but the paper's framing must change from "first controlled sweep at this scale" to "resolves the SmolVLA/BEAST/StarVLA-alpha fragmentation under one protocol, and is the first to test whether StarVLA-alpha's 'head choice barely matters' conclusion holds at <500M and on the discriminative axes (few-shot, LIBERO-Plus, Meta-World-hard) where it hasn't been tested." That is a defensible, honest novelty claim — not the same as "nobody has done anything like this."
