# CheXpert Baseline Comparison — Frozen Protocol v1

Status: implemented, verified against real data and real GPU runs, live in
`Colab_Training_Notebooks/*.ipynb`. This document reflects what the code
**actually does**, not the original plan — see "Deviations from the original
plan" below for every place they differ and why.

Code: https://github.com/fei7fei/chexpert-baselines

## Models (6, not 5)

| Model | Architecture | Init | Notes |
|---|---|---|---|
| Ark+ | Swin-Large (V1) | random | Plain baseline — its 3 original "improvements" (per-class uncertain-label strategy, class-weighted BCE, co-occurrence loss) were stripped so it's directly comparable to the other 5 |
| SwinL | Swin-Large (V1) | random | Ark+'s architecture-matched, no-pretraining control. **Extra model beyond the original 5** — supports the free "does pretraining delay overfitting" analysis |
| SwinB | **Swin Transformer V2** Base | ImageNet-22k | Real Swin V2 (`microsoft/swinv2-base-patch4-window12-192-22k` via `transformers`, not `timm==0.5.4` which predates V2 support) |
| DenseNet-121 | DenseNet-121 | ImageNet-1k | |
| RadDino | RAD-DINO (DINOv2 ViT-B/14) | pretrained | |
| ConvNeXtV2 | ConvNeXt V2 Base | ImageNet-22k | Via `transformers` (also predates the `timm` pin) |

## Frozen settings (identical across all 6)

| Setting | Value |
|---|---|
| Input resolution | 320×320 (native CheXpert-small short side) |
| Uncertain (-1) labels | U-Zeros (headline); U-Ones is a one-line toggle for the appendix sensitivity check |
| Epochs | 30, no early stopping (`patience = epochs`) — reduced from 50 on 2026-08-03 for the Aug 5 SPIE abstract deadline; checkpoint selection is AUC-based so this is still a valid run, just less epoch budget to find the best one |
| Optimizer | AdamW, cosine schedule |
| Warmup epochs | **1** — ⚠️ open item, see below |
| LR / weight decay | 5e-5 / 0.05 |
| Checkpoint selection | **Best validation macro-AUC** across the 30-epoch budget (not val loss, not the epoch-30 endpoint) |
| Val vs. test | Official 234-study CheXpert valid.csv = validation (threshold tuning / checkpoint selection only). A **held-out test set carved from train by patient ID** (15% of patients) = final reporting, touched exactly once, after training, on the single selected checkpoint |
| Ablation subset | Fixed, stratified ~20% of train, patient-disjoint from the held-out test set |
| Seeds | 1 (screening tier) / 2 — seeds 13, 42 (headline tier) — reduced from 3 seeds on 2026-08-03 for the Aug 5 deadline. ⚠️ This overrides the earlier explicit guidance to keep 3 seeds even under time pressure (reviewers expect the 3-seed spread) — a deliberate, acknowledged trade-off, not an oversight |
| Metrics | Macro-AUC, macro-F1, per-class sensitivity (recall) at Youden's-J-optimal threshold, macro-accuracy, MCC, AP, ECE, epoch-to-best-AUC |
| Resume | `args.resume = True` — safe to re-run the same cell after a Colab disconnect, resumes from the last saved epoch |

## Per-model settings (deliberately vary)

| Model | Batch size | AMP dtype | Why it differs |
|---|---|---|---|
| Ark+, SwinL, SwinB, DenseNet-121, ConvNeXtV2 | 64 | fp16 | fp16 is the protocol default |
| RadDino | 32 | fp16 | Smaller batch — heaviest per-token attention cost of the 6 (ViT, ~485 tokens at 320px) |
| ConvNeXtV2 | 64 | **fp16** (verified necessary) | bf16 crashes: "no cuDNN engine found" for its depthwise convs |
| SwinB (Swin V2) | 64 | **bf16** (verified necessary) | fp16 underflows: cosine attention L2-normalizes Q/K, and normalizing near-zero-norm vectors goes non-finite under fp16's narrow range (22/473 grad tensors, reproducibly). bf16 (fp32's exponent range) is clean (0/473). |

Batch sizes assume the RTX PRO 6000 Blackwell's 96GB VRAM headroom — reduce if
running on smaller hardware.

## Two-tier discipline

1. **Screening** (1 seed, 20% ablation subset) — cheap, ranks variants/architectures.
2. **Headline** (3 seeds, 100% of train_pool) — confirms the result.

Both tiers are reported explicitly and separately in the results-summary cell,
labeled "Screening (1 seed, 20% subset)" / "Headline (3 seeds, 100% data)" — never
blended into one unlabeled number.

## Data pipeline (added this pass, for speed)

- `args.workers = 8` (was 2), `persistent_workers=True` (worker pool isn't
  respawned every one of the 30 epochs), `prefetch_factor=4`.
- `non_blocking=True` on host→device transfers (paired with the DataLoaders'
  existing `pin_memory=True`).
- Not yet verified for actual speedup on the target hardware — only confirmed
  correct (no crashes) locally on different hardware.

## Deviations from the original plan document

| Item | Plan said | Actually built |
|---|---|---|
| Warmup epochs | 2 ("since 50 epochs") | **1** — never updated to match; open decision, see below |
| Precision | Uniform bf16 (A100 assumption) | fp16 default + 2 verified, documented, hardware-forced per-model exceptions (see table above) |
| Model count | 5 | 6 (SwinL added as Ark+'s no-pretraining control) |
| Model architecture | "ConvNeXt-B" (ambiguous), "Swin-V2-B" | ConvNeXt **V2** Base (matches on param count, ~89M); Swin V2 Base implemented for real |
| Epochs | 50 | **30** — cut 2026-08-03 for the Aug 5 abstract deadline |
| Headline seeds | 3 (13, 42, 2024) | **2** (13, 42) — cut 2026-08-03 for the Aug 5 abstract deadline, overriding earlier explicit guidance to never reduce seeds |

## Open items

1. **Warmup epochs: 1 or 2?** Your v2 doc explicitly wants 2. Not yet changed — needs a decision.
2. **Gap #1 (calibration + per-label thresholds)** — not yet built. ~0 GPU cost, pure post-hoc analysis on cached logits once checkpoints exist. Needed for the minimal publishable plan.
3. **Gap #3 (ASL/focal loss)** — not yet built. Real engineering (new loss class) + ~18-25 GPU-hours (screening sweep across all 6 models). Needed for the minimal publishable plan.
4. Gap #2 (DenseNet fusion), Gap #4 (LoRA), Gap #5 (GRN) — deferred per the plan's own prioritization; unaffected by anything above.
5. Log-CPB (full transplant into Ark+) — still correctly deferred to January. Note: building real Swin V2 for SwinB already gives a small, free, verified data point supporting that story (zero dropped backbone weights transferring across window sizes at 320px, vs. V1's ~0.4% dropped) — worth a sentence in Methods/Related Work now.

## Where things live

- Code (this repo): `github.com/fei7fei/chexpert-baselines` — pull this on any machine to get the full, current implementation.
- Shared protocol split cache (patient partition, identical across all 6 models): Google Drive, `chexpert_protocol_v1/` folder.
- Per-model checkpoints: Google Drive, `<model>_chexpert_checkpoints/` folders, under `_protocol1_<model>_screening` / `_protocol1_<model>_headline` experiment names.
