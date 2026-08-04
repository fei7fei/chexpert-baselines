# Session Notes — Context for Continuing This Work

This file exists because Claude Code conversations don't sync across machines —
only the repo does. If you're picking this up on a different machine (or a fresh
Claude Code session anywhere), read this first; it has the context a plain
`git log` won't give you. The technical protocol itself is in `PROTOCOL.md`,
kept separate so it doesn't get buried in narrative.

## The deadline this is actually for

**SPIE Medical Imaging 2027, Computer-Aided Diagnosis track.** Verified via live
web search (not assumed):
- Abstract + required 2-4 page PDF: **5 August 2026**
- Acceptance notification: ~October 2026
- Award application + preliminary manuscript: 27 November 2026
- Final manuscript for proceedings: **27 January 2027**
- Conference: 14-18 Feb 2027, Vancouver BC

The 5 August date is the binding near-term constraint. Everything after
acceptance has a ~5.5 month runway — most of the deferred work (Log-CPB, GRN,
LoRA, RAD-DINO distillation) is legitimately for that window, not now.

## What this project actually is

Baseline comparison of 6 chest-X-ray classification architectures on CheXpert
(Ark+/Swin-L, SwinL-no-pretrain, Swin V2 Base, DenseNet-121, RAD-DINO, ConvNeXt
V2) under one frozen, identical protocol, plus a small number of "gap"
experiments (calibration/thresholds, asymmetric loss) that turn it from a pure
benchmark table into a publishable CAD paper. Framing: **questions, not
claims** — e.g. "does Ark+ still need a long-tail-aware loss?" not "we propose
X." This framing was explicitly requested and matters for review — see
`PROTOCOL.md`'s deviations table for what "frozen" actually means in practice.

## Compute environment

- GPU: **NVIDIA RTX PRO 6000 Blackwell Server Edition, 96GB GDDR7** — accessed
  through a standard Colab GPU tier that Google's own UI calls "G4" (this is
  *not* the same as AWS's G4 instance family, which is T4-based — a source of
  real confusion earlier in this session, resolved by directly checking specs).
  96GB means batch-size/OOM is basically not a constraint for any of these 6
  models.
- Platform: standard Colab (not Enterprise, not a custom/local runtime) — so
  everything Colab-specific in the notebooks (`google.colab.drive`, the
  interactive Kaggle-credential prompt, `/content/...` paths) works normally.

## Code flow — important operational detail

Code changes go: **local edit → git commit/push to
`github.com/fei7fei/chexpert-baselines` → manually resync to
`G:\My Drive\Project\models\Repos\ark_plus_repo\Ark_Plus\Finetuning`**. The
Colab notebooks pull their code from **Drive**, not git, directly (see cell 4 of
any `*_Colab_Baseline.ipynb`). Git push alone is *not* enough to make a change
live in Colab — this was missed at least once this session, causing a stale
Drive copy to run for a while. Always resync after pushing a Finetuning/ change:

```powershell
robocopy "A:\Project\models\Repos\ark_plus_repo\Ark_Plus\Finetuning" "G:\My Drive\Project\models\Repos\ark_plus_repo\Ark_Plus\Finetuning" /E /Z /MT:8 /R:2 /W:5 /XD __pycache__
```

Also: an already-running Colab session won't pick up a Drive update either,
because cell 4 only copies code `if /content/chexpert_code doesn't already
exist`. Picking up a code change requires a full **Runtime → Restart session**,
not just re-running cells.

## Real bugs found and fixed this session (worth knowing so they aren't rediscovered)

1. **Held-out-test-set leak**: the training loop was evaluating per-epoch
   metrics on the *test* set and using those numbers to report "best epoch" —
   defeats the point of holding it out. Fixed: per-epoch metrics (and now
   checkpoint selection) run on validation only; test is touched once, after
   training, on the one selected checkpoint.
2. **Checkpoint pickling crash**: `best_val_auc` was a numpy scalar
   (`meanAUC()` returns `np.mean(...)`), which breaks `torch.load`'s
   `weights_only=True` default in PyTorch ≥2.6. Fixed by casting to a plain
   `float()` before storing in the checkpoint dict.
3. **ConvNeXt V2 crashes under bf16** ("no cuDNN engine found" for its
   depthwise convs) — verified, fp16 avoids it. This model is the one fp16
   exception... but:
4. **Swin V2 crashes under fp16** — its cosine attention L2-normalizes Q/K
   before the dot product; normalizing near-zero-norm vectors underflows
   fp16's narrow range (verified: 22/473 grad tensors go non-finite). bf16 is
   clean. So Swin V2 is the mirror-image bf16 exception to ConvNeXtV2's fp16
   exception — both real, hardware-forced, not preferences.
5. **`transformers`' `Swinv2Config.hidden_size` is simply wrong** for this
   checkpoint (reports 768; actual pooled feature is 1024-dim). Fixed by
   computing `embed_dim * 2**(num_stages-1)` instead of trusting the config
   attribute.
6. Per-batch training time was suspiciously perfectly flat (0.136s, zero
   jitter over 950 batches) before this session's speedup changes — a strong
   signal of a CPU-bound (dataloader-starved) step, not a GPU-bound one.
   `args.workers` was only 2. Bumped to 8 + `persistent_workers` +
   `prefetch_factor=4` + `non_blocking=True` transfers. **Not yet verified for
   actual speedup on the target hardware** — only confirmed correct
   (no crashes) on different local hardware. The real test is the next
   restart's per-batch timing.
7. **STOP-THE-LINE, found 2026-08-03: Ark+ was never actually loading Ark+
   weights.** `ArkPlus_Colab_Baseline.ipynb`'s cell 12 had `USE_ARK_WEIGHTS =
   False` hardcoded, and cell 14 (rewritten by the frozen-protocol pass)
   never referenced `CKPT_PATH`/`USE_ARK_WEIGHTS` at all — it unconditionally
   set `args.pretrained_weights = None`. So Ark+ and SwinL were training as
   the exact same random-init model. Three real bugs, now fixed:
   - Cell 12: `USE_ARK_WEIGHTS` flipped to `True`.
   - Cell 14: now sets `args.init = "ark" if USE_ARK_WEIGHTS else "random"`,
     `args.pretrained_weights = CKPT_PATH`, `args.scale_up = True`.
   - `models.py`'s `swin_large` pretrained-load branch hardcoded
     `window_size=7`, ignoring the `_swin_window_size()` helper the
     random-init branch already used correctly — 320px needs `window_size=5`,
     not 7. Fixed to call the same helper. Separately, `load_pretrained_weights()`
     only filtered `attn_mask` keys for the window-size change, not the also
     window-size-dependent `relative_position_bias_table`/`relative_position_index`
     — `strict=False` doesn't save a same-key-different-shape error, only
     missing/unexpected keys. Added a shape-mismatch filter (same pattern
     already used in `_load_swin_pretrained_filtered` elsewhere in the file).
   - The real checkpoint (`Ark6_swinLarge768_ep50.pth.tar`, requires the
     Google Form request in the Ark repo README — not downloadable any other
     way) is now on Drive at `ark_plus_chexpert_checkpoints/`. **Verified
     locally**: loads with only expected missing keys (window-geometry
     buffers + `head.weight`/`head.bias`) and expected unexpected keys
     (Ark's pretraining-only `projector`/`omni_heads.*`, correctly dropped)
     — the actual backbone (patch embed, attention, MLP, norms) transfers
     cleanly. This was the single highest-priority item in the reconciled
     plan (v3 §1.1) — without it there was no real Ark+ in the comparison,
     and the Ark+-vs-SwinL pretraining control (the paper's strongest claim
     per v3 §5.1) didn't exist.
   - **Follow-up trap, also caught same day**: a stale checkpoint
     (`_protocol1_arkplus_screening_run_0.pth.tar`, 1 epoch, trained *before*
     this fix under the random-init bug) was still sitting on Drive. With
     `args.resume=True`, simply running the notebook would have silently
     resumed from that contaminated state instead of starting fresh from the
     real Ark+ weights — quietly undoing the whole fix. Deleted it (and its
     `_run_0_epoch_table.csv`) before any real run started. **If Ark+
     training is ever restarted from scratch, check for and clear any
     pre-fix checkpoint under this exp_name first** — `args.resume=True`
     will not warn you if it picks up a bad one.
8. **`engine.py`'s test-time F1/Recall/Accuracy leak the held-out test set
   into threshold selection** (found while building Gap #1): the Youden's-J
   threshold used to binarize predictions is picked from the *test* set's
   own labels (`meanF1(y_test, p_test)` etc.), then that same set is scored
   with it — same category of leak as bug #1 above, just for thresholds
   instead of checkpoint selection. AUC is unaffected (threshold-independent);
   F1/Recall/Accuracy as currently reported by the base training notebooks
   are optimistically biased and **should not be used as headline numbers**.
   Not yet fixed at the source (`engine.py`) — instead, `ArkPlus_Gaps_Colab.ipynb`
   (Gap #1) computes the leak-free version post-hoc: thresholds fit on
   validation only, applied once to test. If time allows, the same fix
   (separate val-fit threshold from test-time scoring) should go into
   `engine.py` itself so all 6 models' base pipeline numbers stop leaking,
   not just Ark+'s Gap #1 analysis.
9. **`torch.load` crashes on Colab's PyTorch (>=2.6) but not locally (2.5.1)** —
   found while actually running Ark+ in Colab. PyTorch 2.6 flipped `torch.load`'s
   `weights_only` default to `True`, which can't unpickle numpy scalars
   present in checkpoint dicts (`epoch`, `lossMIN`, etc.). This is **not**
   Ark+-specific: it hit every unguarded `torch.load` in the pipeline —
   `models.py`'s pretrained-checkpoint load (the Ark6 checkpoint itself),
   `trainer.py`'s `test_classification()` (final test-set reload, used by
   **every** model), and `engine.py`'s resume logic (used by every model,
   since `args.resume=True` everywhere). All three, plus the equivalent load
   in `ArkPlus_Gaps_Colab.ipynb`'s Gap #1 cell, now pass `weights_only=False`
   explicitly (safe: these are either our own saved checkpoints or the
   verified, officially-gated Ark+ release, not untrusted files). **Resynced
   to Drive — requires a full Colab Runtime → Restart to take effect**, same
   as any other code change (see "Code flow" above). Since this affects
   every model's resume/test path, not just Ark+, it likely would have hit
   SwinB/DenseNet-121/RadDino/ConvNeXtV2/SwinL too, whenever they first hit a
   resume or test-evaluation step — worth a quick sanity check on whichever
   model runs next, in case another `torch.load` site was missed.

## Minimal publishable plan — status

Per the original plan doc: 5(-6) baselines + Gap #1 (calibration/thresholds) +
Gap #3 (asymmetric/focal loss) + the free 50-epoch overfitting-dynamics
analysis (now trivial: `idxmax()` on the epoch table, since checkpoint
selection is AUC-based). Nothing built this session weakens this plan —
if anything the val/test leak fix and the extra SwinL control strengthen it.
**Gap #1 and Gap #3 are not yet built** — still fully outstanding.

## Explicit guidance received and followed

- Reviewers expect the 3-seed headline spread specifically; do not reduce
  headline seeds under time pressure, even though epochs can be shortened
  safely (AUC-based checkpoint selection makes a shorter run still valid,
  just with less chance of finding an even-better epoch).
  **Superseded 2026-08-03**: with the Aug 5 abstract deadline 2 days out,
  explicitly decided to override this and cut to 2 headline seeds (13, 42;
  dropped 2024) *and* 30 epochs (down from 50), across all 6
  `Colab_Training_Notebooks/*.ipynb`. This was a deliberate, acknowledged
  trade-off after being reminded of the original guidance — not an oversight.
  If there's runway after the abstract deadline, consider adding the 3rd
  seed back for the final manuscript (27 Jan 2027) to restore the spread
  reviewers expect.
- Frame everything as a question being tested, not a claim of a new method —
  applies to Log-CPB, ASL, GRN discussion equally.
- `CheX-DS` (arXiv:2505.11168) already covers DenseNet+Swin-B+ASL on this
  task — must be read and cited in Related Work, or a reviewer who knows it
  will reject for the omission.

## Open decisions (unresolved as of this file's writing)

1. Warmup epochs: currently 1, plan doc wants 2 ("since 50 epochs") — not yet
   changed, needs a call.
2. Whether to build Gap #1 (calibration/thresholds) now, in parallel with
   baseline training runs.
3. Real-world confirmation of the dataloader speedup changes (see bug #6
   above) — needs a fresh Colab restart and a shared progress-log snippet.

## How to pick this up cold

1. `git pull` — gets `PROTOCOL.md`, this file, and every notebook/code change.
2. Check whether any Colab training runs are already in progress (Drive
   folders `<model>_chexpert_checkpoints/` under experiment names
   `_protocol1_<model>_screening` / `_headline`) before starting anything new,
   to avoid an accidental duplicate/colliding run.
3. `PROTOCOL.md`'s "Open items" section is the actual TODO list.
