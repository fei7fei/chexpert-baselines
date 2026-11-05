# CheXpert Baseline Comparison — Frozen Protocol v1

Status: implemented, verified against real data and real GPU runs, live in
`Colab_Training_Notebooks/*.ipynb`. This document reflects what the code
**actually does**, not the original plan — see "Deviations from the original
plan" below for every place they differ and why.

Code: https://github.com/fei7fei/chexpert-baselines

## Golden rule (2026-11-05): the submitted PDFs decide

The submitted abstract and supplemental PDF (`SPIE_Abstract/spie_abstract_text.*`, `SPIE_Abstract/spie_supplemental.*`) **cannot change**. Every experiment follows them: Table 1's protocol, the six models, questions (a)–(c) and the held-out test set. Where this file or `SESSION_NOTES.md` disagrees with the PDFs, the PDFs win.

Decisions taken under this rule on 2026-11-05:

| Decision | Why |
|---|---|
| Keep `engine.py`'s `lr_scheduler.step(val_loss)` exactly as is, for all 6 models | timm 0.5.4's `step()` reads its first argument as the *epoch*, so the cosine scheduler never leaves its 1-epoch warmup. The LR is ≈ 1e-6 + 4.9e-5 × (previous epoch's val loss): 1e-6 in epoch 1, then ≈ 2.0–2.5e-5 for Ark+ (reconstructed from its epoch tables). Ark+ already trained this way and can't be retrained, so every baseline does too. The configured values (cosine, warmup 1, lr 5e-5, wd 0.05) are exactly Table 1's. |
| Keep the `engine.py` / `utils.py` metric code exactly as is | The submitted Table 3's F1, Sensitivity, Accuracy and MCC are `engine.py`'s numbers, whose Youden thresholds are fit on the test labels (e.g. sensitivity 0.7791 and 0.7865 → "0.779–0.787"). Baselines are only comparable with Table 3 if they're computed the same way. Validation-tuned numbers come separately from `AllModels_Gaps_Colab.ipynb` (Figure 1's method). Never mix the two kinds in one column. |
| Test set = the patient-disjoint held-out 15% only | That is what Table 1 says. No official-CheXpert-test-set evaluation. |
| RAD-DINO batch size 32 → 64 | The PDF promises one identical recipe for all six models. |

## Models (6, not 5)

| Model | Architecture | Init | Notes |
|---|---|---|---|
| Ark+ | Swin-Large (V1) | **ark** (real pretrained checkpoint, `Ark6_swinLarge768_ep50.pth.tar`) | Its 3 original "improvements" (per-class uncertain-label strategy, class-weighted BCE, co-occurrence loss) were stripped for the baseline comparison so it's directly comparable to the other 5 — see "Results status" below for the one exception (class-weighted BCE reintroduced as a separate, non-protocol follow-up test) |
| SwinL | Swin-Large (V1) | random | Ark+'s architecture-matched, no-pretraining control. **Extra model beyond the original 5** — supports the free "does pretraining delay overfitting" analysis |
| SwinB | **Swin Transformer V2** Base | ImageNet-22k | Real Swin V2 (`microsoft/swinv2-base-patch4-window12-192-22k` via `transformers`, not `timm==0.5.4` which predates V2 support) |
| DenseNet-121 | DenseNet-121 | ImageNet-1k | |
| RadDino | RAD-DINO (DINOv2 ViT-B/14) | pretrained | |
| ConvNeXtV2 | ConvNeXt V2 Base | ImageNet-22k | Via `transformers` (also predates the `timm` pin) |

## Results status (as of 2026-08-05, the abstract deadline)

- **Ark+ only** has a complete result under this frozen protocol — headline
  tier (2 seeds, 30 epochs, full train pool), real Ark+ pretrained weights,
  reported in `SPIE_Abstract/spie_supplemental.tex` Table 3. This is the
  paper's one confirmed, protocol-compliant number.
- **SwinL, SwinB (V2), DenseNet-121, RadDino, ConvNeXtV2** have **not** been
  run under this frozen protocol at all yet. What exists for them instead is
  older **pilot** data (50 epochs, 1 seed, a different/non-frozen setup,
  predates this protocol document) — reported separately in the same PDF's
  Table 2, explicitly labeled non-comparable to Table 3. Running these 5
  under the actual frozen protocol above is the largest piece of remaining
  work (see `SESSION_NOTES.md`).
- **Class-weighted BCE follow-up (Ark+ only, screening tier)**: a quick,
  separate proof-of-concept test — not part of the 6-way frozen comparison —
  reintroducing `pos_weight` (one of Ark+'s 3 stripped components) to check
  whether it helps the Atelectasis/Consolidation weakness found in the
  headline result. Real result, 1 seed, 20% ablation subset only (not
  headline-tier confirmed). See `SESSION_NOTES.md` for the two bugs caught
  while building it, and `Colab_Training_Notebooks/ArkPlus_AbstractImprovementTest_Colab.ipynb`
  for the code. Uses a separate `exp_name` (`_protocol1_arkplus_cw_screening`)
  so it never touches the plain-baseline checkpoints this table describes.

## Frozen settings (identical across all 6)

| Setting | Value |
|---|---|
| Input resolution | 320×320 (native CheXpert-small short side) |
| Uncertain (-1) labels | U-Zeros (headline); U-Ones is a one-line toggle for the appendix sensitivity check |
| Epochs | 30, no early stopping (`patience = epochs`) — reduced from 50 on 2026-08-03 for the Aug 5 SPIE abstract deadline; checkpoint selection is AUC-based so this is still a valid run, just less epoch budget to find the best one |
| Optimizer | AdamW, cosine schedule (stepped with val loss, so the effective LR is ≈2e-5 after epoch 1 for every model; see "Golden rule" above) |
| Warmup epochs | **1** — ⚠️ open item, see below |
| LR / weight decay | 5e-5 / 0.05 |
| Checkpoint selection | **Best validation macro-AUC** across the 30-epoch budget (not val loss, not the epoch-30 endpoint) |
| Val vs. test | Official 234-study CheXpert valid.csv = validation (threshold tuning / checkpoint selection only). A **held-out test set carved from train by patient ID** (15% of patients) = final reporting, touched exactly once, after training, on the single selected checkpoint |
| Ablation subset | Fixed, stratified ~20% of train, patient-disjoint from the held-out test set |
| Seeds | 1 (screening tier) / 2 — seeds 13, 42 (headline tier) — reduced from 3 seeds on 2026-08-03 for the Aug 5 deadline. ⚠️ This overrides the earlier explicit guidance to keep 3 seeds even under time pressure (reviewers expect the 3-seed spread) — a deliberate, acknowledged trade-off, not an oversight |
| Metrics | Macro-AUC, macro-F1, per-class sensitivity (recall) at Youden's-J-optimal threshold, macro-accuracy, MCC, AP, ECE, epoch-to-best-AUC — ⚠️ as currently computed by `engine.py`, the Youden's-J threshold is fit on the *test* set's own labels then applied to that same set (a leak; see `SESSION_NOTES.md` bug #8), **deliberately kept as is (2026-11-05)**, because the submitted Table 3's F1 / Sensitivity / Accuracy / MCC are exactly these test-fit numbers (MCC uses the same test-fit threshold). The leak-free, validation-tuned numbers are **Figure 1 only**: `ArkPlus_Gaps_Colab.ipynb` (Ark+) and `AllModels_Gaps_Colab.ipynb` (all six) fit temperature + 3 threshold policies (naive 0.5, Youden's J, F1-optimal) on validation only and apply them once to test. AUC, AP and ECE are threshold-independent |
| Resume | `args.resume = True`. **Since 2026-11-05:** a run already listed in `models.log` is never retrained, and a finished experiment is not re-tested. An interrupted run continues exactly from the end of its last finished epoch: `<run>_last.pth.tar` is written every epoch and deleted when the run finishes, so only the best-validation-AUC checkpoint per run is kept, as with Ark+. Verified on CPU with a stubbed model: an uninterrupted run's outputs are byte-identical to the old engine's, and runs interrupted once or twice resume to identical outputs. **Before this**, resume reloaded the *best* checkpoint and retrained from that epoch, which silently replaced Ark+'s screening checkpoint on Drive on 2026-08-05 (the Mac copies in `checkpoints/arkplus/` and `results/arkplus/` are the clean originals) |

## Per-model settings (deliberately vary)

| Model | Batch size | AMP dtype | Why it differs |
|---|---|---|---|
| Ark+, SwinL, DenseNet-121 | 64 | fp16 | fp16 is the protocol default |
| RadDino | 64 (32 until 2026-11-05) | fp16 | Raised to 64 to match the other five: the submitted PDF promises one identical recipe. If it ever OOMs, stop and decide; don't silently drop it back to 32 |
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

1. **Warmup epochs: 1 or 2?** Your v2 doc explicitly wants 2. Not yet changed — needs a decision. Lowest priority of anything on this list.
2. ~~Gap #1 (calibration + per-label thresholds)~~ — **done.** Built, run, and is what `spie_supplemental.tex` Table 3/Figure 1 actually uses. See `SESSION_NOTES.md`.
3. **Gap #3 (ASL/focal loss)** — not yet built. Real engineering (new loss class) + ~18-25 GPU-hours (screening sweep across all 6 models). Needed for the minimal publishable plan; not needed for the Aug 5 abstract, which instead got a much smaller stand-in (class-weighted BCE screening-tier test, see "Results status" above and `SESSION_NOTES.md`).
4. Gap #2 (DenseNet fusion), Gap #4 (LoRA), Gap #5 (GRN) — deferred per the plan's own prioritization; unaffected by anything above.
5. Log-CPB (full transplant into Ark+) — still correctly deferred to January. Note: building real Swin V2 for SwinB already gives a small, free, verified data point supporting that story (zero dropped backbone weights transferring across window sizes at 320px, vs. V1's ~0.4% dropped) — worth a sentence in Methods/Related Work now.
6. **The big remaining item**: SwinL, SwinB (V2), DenseNet-121, RadDino, ConvNeXtV2 have never been run under this frozen protocol. Everything for them in the current PDF is old pilot data from before this protocol existed. This is the actual bottleneck between the Aug 5 abstract and a complete final manuscript (due 27 Jan 2027) — see "Results status" above.
7. PDF placeholders, updated 2026-08-05: authors filled in (Pei Lin Xie, Yi-an Su, UBCO). Still open — department name, corresponding-author email, and the Disclosure of Prior/Concurrent Submissions statement. All in `SPIE_Abstract/spie_supplemental.tex` (and mirrored in `spie_abstract_text.tex`'s author block — keep both in sync).
8. The 200-300 word technical-review abstract is drafted (`SPIE_Abstract/spie_abstract_text.tex`). The 50-150 word program-display summary is not yet drafted — separate, shorter, more audience-facing SPIE form field.

## Where things live

- Code (this repo): `github.com/fei7fei/chexpert-baselines` — pull this on any machine to get the full, current implementation.
- Shared protocol split cache (patient partition, identical across all 6 models): Google Drive, `chexpert_protocol_v1/` folder.
- Per-model checkpoints: Google Drive, `<model>_chexpert_checkpoints/` folders, under `_protocol1_<model>_screening` / `_protocol1_<model>_headline` experiment names. Ark+'s also has `_protocol1_arkplus_cw_screening` (the class-weighted follow-up).
- `AllModels_Gaps_Colab.ipynb` (added 2026-11-05): Figure 1's Gap #1 analysis (same functions, copied verbatim) for every finished run of all six models. It stops unless it re-derives the submitted Figure 1 for Ark+. Writes `chexpert_protocol_v1/gap1/gap1_all_models.csv` on Drive; copy that into `results/gap1/`. `Results_Dashboard_Colab.ipynb` (Michael) displays `*_results.txt`; Ark+'s screening file on Drive has 4 test blocks from the 2026-08-05 accidental resume, so the dashboard shows 4 "runs" there. Use the Mac copy in `results/arkplus/` instead.
- Notebooks: `Colab_Training_Notebooks/*_Colab_Baseline.ipynb` (one per model), `ArkPlus_Gaps_Colab.ipynb` (baseline + Gap #1 calibration/threshold analysis), `ArkPlus_AbstractImprovementTest_Colab.ipynb` (baseline, byte-identical through its training cell, + the class-weighted BCE follow-up — kept as a separate notebook specifically so the plain baseline stays untouched and runnable on its own).
- The actual deliverable: `SPIE_Abstract/spie_supplemental.tex` (+ compiled `spie_supplemental.pdf`), 4 pages, SPIE Letter-size margins verified. `SESSION_NOTES.md` has its current content/status and what's still a placeholder.
