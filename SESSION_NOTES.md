# Session Notes — Context for Continuing This Work

This file exists because Claude Code conversations don't sync across machines —
only the repo does. If you're picking this up on a different machine (or a fresh
Claude Code session anywhere), read this first; it has the context a plain
`git log` won't give you. The technical protocol itself is in `PROTOCOL.md`,
kept separate so it doesn't get buried in narrative.

## 2026-11-06: how the baseline runs work now

- **Where they run.** From VS Code, using the Google Colab extension (`google.colab`, already installed on Ian's Mac): open a `*_Colab_Baseline.ipynb`, pick the Colab kernel with an A100, then Run All. Runs use the **signed-in account's** Drive.
- **Code: GitHub tag `protocol1-baselines-v1`, md5-checked.** No more robocopy to Drive for the baselines (the "Code flow" section below still applies to Ark+'s own notebooks). Changing training code now means: commit, create a new tag, and update `CODE_TAG` + `EXPECTED_CODE_MD5` in the 5 baseline notebooks and `AllModels_Gaps_Colab.ipynb`. Fingerprint at this tag: `engine.py c5b6e179169c35481be1d6e54c9179af`, `trainer.py 3b368affdd29b1b03475f82474ebfa6a`, `models.py e0ced0f809680aaebfc97ec3fd6d0e14`, `dataloader.py 6aba3a804bb10625a49c9027734a7b76`, `utils.py a7c17deb55d24e826c74b46ad848c441`.
- **Dataset = Ark+'s, enforced.** Fingerprints over every image path + all 14 labels, in order: train pool `fd210ea10a3f5c364de5d167b57144f3`, held-out test `03d149b5a3504245e7e9a2f42a1c2e9f`, screening subset `87c42c824823e16386ee0cbc026202db`, valid `bca0b7e132dfc73e7aab492cf40cc084`. They were computed from CheXpert-small with the protocol's split code, which reproduces Ark+'s held-out `gt.csv` row for row. A mismatch stops the notebook; tested both ways locally.
- **A100 (40 GB) vs Ark+'s 96 GB card.** The effective batch stays 64 via gradient accumulation (`args.grad_accum_steps`, opt-in; see `PROTOCOL.md`). Swin-L switches to 2 × 32 automatically on GPUs under 80 GB. The others use 64; if one runs out of memory, set its `MICRO_BATCH = 32`. DenseNet-121 must stay at real 64 (BatchNorm). Memory use on the A100 is an estimate until the first runs; nothing was measured on GPU here.
- **Drive must be mounted.** The config cell stops otherwise, so checkpoints can't silently land on the runtime's disk.
- **Paper outputs.** `code/analysis/paper_outputs.py`, run by each notebook's last cell, writes per-seed and per-class metrics, the Table 3-format min–max summary, training-length numbers (best-AUC epoch, lowest-val-loss epoch) and loss/AUC-per-epoch figures (PDF + PNG). Run it on the Mac over `results/` for cross-model figures (all six models; Ark+ vs Swin-L). Colours: the dataviz reference palette, validated.
- **Table 3 has two upper bounds that don't match the results file.** MCC 0.363–**0.375** (file: 0.3743 / 0.3633 → 0.363–0.374) and ECE 0.011–**0.015** (file: 0.0109 / 0.0141 → 0.011–0.014). The other five ranges match exactly. The PDF can't change; the manuscript should use the file-traceable values (`paper_outputs.py` prints them).
- **Ark+ training length (headline, from its epoch tables):** best validation AUC at epoch 13 (seed 13) and epoch 5 (seed 42); lowest validation loss at epochs 13 and 21.

## 2026-11-05: read this first

**Golden rule:** the submitted abstract and supplemental PDF cannot change, and everything follows them. See `PROTOCOL.md` → "Golden rule" for the decisions taken under it. In short: the LR-scheduler behaviour and `engine.py`'s metric code stay exactly as Ark+ ran; the test set is the held-out 15% only; RAD-DINO's batch size is now 64.

What was found and changed in this session:

1. **LR schedule.** `engine.py` calls `lr_scheduler.step(val_loss)`. timm 0.5.4 reads that as the epoch number, so the LR never leaves warmup: ≈2.0–2.5e-5 after epoch 1, no decay (Ark+'s epoch tables). This is inherited from upstream Ark (`Ark_MICCAI2023/engine.py` does the same). **Kept on purpose, so don't fix it.**
2. **Table 3 = `engine.py`'s test-fit metrics.** The submitted Table 3's F1, Sensitivity, Accuracy and MCC equal `results/arkplus/_protocol1_arkplus_headline_results.txt` exactly, with Youden thresholds fit on the test labels (bug #8 below). Figure 1 is the validation-tuned analysis. **Don't "fix" bug #8 in `engine.py`/`utils.py`**, or the baselines stop being comparable with Table 3.
3. **Ark+'s screening run on Drive was altered on 2026-08-05.** Re-running the baseline cell with `resume=True` reloaded the finished run's best checkpoint and retrained from there, three times. The Drive screening checkpoint now gives test mAUC 0.7922 (the protocol run gave 0.7914), its results file has 4 test blocks, and the early epoch-table rows are gone. The headline run_0 epoch table on Drive was cut to 2 rows; the headline checkpoints are intact. **The Mac copies (`checkpoints/arkplus/`, `results/arkplus/`, from the 4 Aug export) are the clean originals.** Also check whether the abstract's class-weighted BCE comparison (plain screening F1 at 0.5) used the altered checkpoint.
4. **`engine.py` resume and test fixes.** These don't change any number from an uninterrupted run; that was verified byte-for-byte against the old engine on CPU with a stubbed model:
   - runs already in `models.log` are skipped, never retrained;
   - an interrupted run resumes exactly from `<run>_last.pth.tar` (written every epoch, deleted at the end);
   - a finished experiment isn't re-tested;
   - duplicate `models.log` lines are tested once.
   - **Bug #11's real cause** was code, not stale files: cached test CSVs already hold only the 5 reported classes, and the test step subset them again with 14-class indices. Any re-run that found its own cached predictions crashed with that `IndexError`. Fixed.
5. **Split verified.** Rebuilding the split from CheXpert-small `train.csv` with the protocol's code reproduces Ark+'s held-out `gt.csv` row for row: held-out test 33,055 images / 9,681 patients; train pool 190,359 / 54,859; screening subset 38,186 / 10,971; valid 234. Every baseline notebook now asserts these counts before training.
6. **Baseline notebooks** (SwinL, SwinB, DenseNet121, RadDino, ConvNeXtV2):
   - the split guard above;
   - an `env_<model>_<utc>.json` record of torch/timm/transformers versions and md5s of the training code;
   - RAD-DINO batch size 32 → 64;
   - `args.init` labels corrected (RAD-DINO / ConvNeXtV2 always loaded their pretrained weights; the label said "random").

   The config cells now differ from Ark+'s only in model identity, plus Swin V2-B's bf16.
7. **`AllModels_Gaps_Colab.ipynb`** runs Figure 1's analysis for all six models and refuses to continue unless it re-derives Figure 1 for Ark+. Checked locally with the clean Ark+ seed-13 checkpoint: 0.5 → 0.8603 / 0.3194; Youden → 0.6634 / 0.8351; F1-optimal → 0.7180 / 0.7275 (accuracy / sensitivity; Figure 1: 0.86/0.32, 0.66/0.84, 0.72/0.73); T = 1.634 (PDF: "≈1.6").

(The training-code fingerprint listed here before was superseded on 2026-11-06; see below.)

Ark+ wall-clock times, from Drive file timestamps: screening ≈1.3 h, each headline seed ≈6.5–7 h.

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
10. **`pos_weight` computed against the wrong label semantics** (found while
    building the class-weighted BCE follow-up test). The first version
    computed `pos_weight = neg_count / pos_count` counting only rows with a
    *definite* 0 or 1 label, excluding uncertain (-1) and blank rows entirely.
    But this protocol trains with `uncertain_label="Zeros"` and
    `unknown_label=0` — both uncertain *and* blank become negative at
    training time. Excluding them from the weight computation understates
    the true negative count badly for any class with lots of uncertain
    labels. Verified against the real CheXpert train.csv: for Atelectasis,
    the buggy formula gave `pos_weight=0.04`; the correct one (matching what
    training actually does) gives `≈5.7` — off by ~140x, and in the wrong
    direction (the buggy version would have *down*-weighted the exact class
    the test was meant to help). Fixed in
    `ArkPlus_AbstractImprovementTest_Colab.ipynb` cell 16: `pos_weight =
    (n - pos_count) / pos_count`, i.e. everything that isn't a definite 1
    counts as negative, matching training-time semantics exactly. Caught
    before the flawed version finished training (interrupted a few epochs
    in), so no wasted GPU time, but this is exactly the kind of bug that
    would have produced a plausible-looking, silently-wrong result if it had
    run to completion.
11. **Stale cached test predictions crashed the plain-BCE screening run** —
    unrelated to bug #10, hit while finishing the screening tier's training.
    `engine.py`'s "start testing" phase checks whether a prediction/`gt.csv`
    already exists for the experiment and, if so, reads the cached file
    instead of recomputing. `_protocol1_arkplus_screening`'s folder had a
    leftover cache from a much older run (before the current "train on all
    14 labels, report on 5" convention), so the cached arrays had 5 columns;
    the current code then tried to index that 5-column array with 14-class
    indices (`test_diseases_5`, which goes up to index 8) → `IndexError`.
    Training itself was fine; only the cache-read shortcut was stale. Fixed
    by deleting the two stale files (`gt.csv` and the experiment's
    prediction `.csv`) so it recomputed fresh. **If any other model's
    screening/headline folder throws the same `IndexError` at the "start
    testing" stage, check for the same stale-cache cause before assuming a
    training bug.**
12. **Notebook cell had a literal newline embedded inside an f-string opener**
    — a self-inflicted bug from editing `ArkPlus_AbstractImprovementTest_Colab.ipynb`'s
    JSON directly: one source-list entry contained an actual `\n` byte
    instead of the two-character escape sequence, so the pasted/executed
    cell read as `print(f"` followed by a real line break — invalid syntax
    outside a triple-quoted string (`SyntaxError: unterminated f-string
    literal`). Root cause was a shell heredoc silently mangling backslash
    escaping during one edit pass. Fixed by writing the replacement through
    a standalone `.py` script file (no shell involved) instead of an inline
    heredoc, and verified by `compile()`-checking every code cell in the
    notebook. **If editing notebook JSON source lists directly again, avoid
    bash heredocs for anything containing `\n`/backslash escapes — write a
    script file instead and verify with `compile()` before telling the user
    it's fixed.**

## Minimal publishable plan — status

Per the original plan doc: 5(-6) baselines + Gap #1 (calibration/thresholds) +
Gap #3 (asymmetric/focal loss) + the free 50-epoch overfitting-dynamics
analysis (now trivial: `idxmax()` on the epoch table, since checkpoint
selection is AUC-based). Nothing built weakens this plan — if anything the
val/test leak fix and the extra SwinL control strengthen it.

**Updated 2026-08-05 (deadline day):**
- **Gap #1 — done.** Built, run, real data. Table 3's Sensitivity column and
  Figure 1 (accuracy-vs-sensitivity by threshold policy) both come from it.
- **Gap #3 (ASL/focal loss) — still not built.** Too much engineering for the
  Aug 5 deadline. A smaller stand-in was built instead: class-weighted BCE
  (one of Ark+'s original 3 stripped components, not a new loss), tested on
  the screening tier only. See "Class-weighted BCE proof-of-concept" below.
  Gap #3 itself is still fully outstanding, deferred to the Jan 2027
  manuscript.
- **The Ark+ headline run is complete** — real weights, 2 seeds, 30 epochs.
  This is the one number in the abstract that's fully protocol-compliant.
- **The other 5 architectures have not been run under this protocol at all.**
  What's in the current PDF for them is pilot data from before the frozen
  protocol existed (see "Data-integrity issues caught before submission"
  below for why that distinction had to be made explicit).
- **The SPIE supplemental PDF itself is built and submission-shaped** — see
  "SPIE supplemental PDF — status" below.

## SPIE supplemental PDF — status

`SPIE_Abstract/spie_supplemental.tex` (compiles to `spie_supplemental.pdf`).
Structure: Title/Authors → Purpose → Methods (frozen protocol table) →
Results (pilot 5-architecture table, then the real Ark+ headline result:
table, threshold-policy figure, calibration finding, the class-weighted BCE
follow-up) → Novel/Breakthrough Aspects → Related Work → Conclusion →
Disclosure of Prior/Concurrent Submissions → References.

Verified as of last compile: **4 pages** (within SPIE's 2-4 page spec —
references start on page 4, so it's within budget even before applying
SPIE's "not counting acknowledgements and references" rule), **Letter size**
(612×792pt, confirmed from the PDF's own MediaBox), **margins** match spec
(0.88in L/R, 1.0in top, 1.25in bottom, confirmed via `geometry`'s verbose
layout report), **zero LaTeX warnings**, every `\cite`/`\bibitem` pair and
every `\ref`/`\label` pair resolves (checked programmatically, not just by
eye) on the last clean compile, 2026-08-05.

**Updated 2026-08-05, later same day:**
- **Authors filled in**: Pei Lin Xie and Yi-an Su, University of British
  Columbia Okanagan (UBCO). Still open: department name and corresponding-
  author email (both currently `[Department]` / `[email]` placeholders).
- **Disclosure of Prior or Concurrent Submissions** — still `[FILL IN —
  required field]`. Only remaining true placeholder in the PDF.
- **A review-audit pass was run against the PDF** (not a rewrite — see
  "External academic-writing skill repos" below for why that distinction
  matters) using an adversarial 8-dimension challenge framework plus a
  5-dimension self-review checklist, both borrowed from third-party Claude
  Code skill repos. Two real, cheap fixes came out of it and are now in the
  PDF: (1) Purpose now discloses up front that only 1 of 6 architectures is
  complete, instead of only becoming clear once the reader reaches Results;
  (2) Table 1's Seeds row now states "reduced from 3" directly in the PDF
  (previously that fact only lived in this file / `PROTOCOL.md`, not in the
  submission itself). Full finding list, including the ones that were
  checked and came back clean, is in this conversation's transcript, not
  duplicated here.
- **`SPIE_Abstract/spie_abstract_text.tex`** — new, separate file. A clean,
  standalone 1-page PDF of just the 200-300 word technical-review abstract +
  keywords, for the user's own reference (SPIE's form wants this as pasted
  text, not a file — this PDF is not itself submitted anywhere). Same
  author block as the main PDF; keep both in sync if authors/affiliation
  change again.
- **The 200-300 word abstract is drafted** (in `spie_abstract_text.tex` and
  this conversation) — built by drafting three versions against three
  different structural methodologies (a direct version, a 5-component
  Background/Purpose/Method/Findings/Implications version, and a
  Challenge→Insight→Contribution version) and merging the strongest parts.
  All three drafts, and the merge, are real data only — no new numbers, just
  restructured prose. One overclaim was caught and fixed during this
  process: an early draft said the remaining 5 architectures "are in
  progress/being trained," which overstates their actual status (not yet
  started, per "Results status" in `PROTOCOL.md`) — corrected to "planned."
- **Still not drafted**: the 50-150 word program-display summary (shorter,
  more audience-facing than the technical abstract — separate SPIE form
  field, not part of any PDF).

## External academic-writing skill repos (used for methodology only, not installed as active tooling)

Two third-party Claude Code skill/plugin repos were cloned to
`~/.claude/external-skills/` on 2026-08-05, at the user's request, to help
draft the abstract and review the PDF:

- `Master-cai/Research-Paper-Writing-Skills` — a lightweight skill (no
  executable code, just `SKILL.md` + reference/example markdown). Also
  copied to `~/.claude/skills/research-paper-writing/` (user scope) so it's
  available as a real invocable skill in future sessions. Its abstract
  template (Challenge→Insight→Contribution) and its `paper-review.md`
  adversarial checklist were both used directly.
- `Imbad0202/academic-research-skills` — a much larger plugin (13-38 agent
  pipeline, hooks, Python enforcement scripts, real CI). **Not activated as
  a live plugin** — `/plugin install` was attempted and found unavailable in
  this specific environment (VSCode-hosted Claude Agent SDK session, not the
  standalone CLI where it may work). Instead, its `abstract_bilingual_agent`
  and `devils_advocate_reviewer_agent` persona files were read directly and
  their *substantive* review/writing methodology applied by hand — their
  machine-oriented orchestration scaffolding (contract JSON, sprint phases,
  synthesizer handoff format) was deliberately not replicated, since it only
  makes sense inside their full pipeline, which isn't running here.
- **If picking this up on a different machine**: neither repo is required
  for anything — everything they contributed (the abstract text, the two
  PDF fixes) is already committed as plain content. They're only relevant
  again if someone wants to draft *new* sections the same way, or actually
  get repo 1 running as a live plugin (would need the standalone CLI and a
  deliberate decision to trust its hooks/scripts — see this conversation's
  transcript for the full risk discussion before doing that).

## Data-integrity issues caught before submission

All three found by comparing an earlier slide/PDF against the underlying
result files — worth knowing about so they aren't reintroduced:

1. **A dataset labeled "Ark+" was actually the random-init (SwinL) run.**
   Proven two ways: it numerically matched a `random_50ep_results.txt` file
   exactly, and no real Ark+ checkpoint existed yet at the time that data was
   produced. A run without Ark+'s pretrained weights loaded is just SwinL —
   there's no such thing as "Ark+ trained from random init," that's a
   contradiction in terms for what makes it Ark+.
2. **An old "SwinB" checkpoint was actually a different, deprecated
   architecture.** Its `relative_position_index` shape (49×49, implying
   `window_size=7`) and patch-embed shape matched old Swin **V1** Base via
   `timm`, not the Swin **V2** Base (`transformers`) that "SwinB" means under
   the current protocol. Kept in the pilot table but explicitly relabeled
   "Swin Transformer V1 Base," not "SwinB," to avoid misrepresenting it as
   the current architecture.
3. **The first pilot table mixed two evaluation methodologies** — 3
   architectures' numbers came from one source, 2 from another, inconsistently.
   Rebuilt from a single consistent source (`results.txt`) for all 5 once the
   inconsistency was noticed.

Also refused, both times asked: stating any architecture is unqualifiedly
"best" (SwinL wins 3/6 pilot metrics but is *last* on sensitivity — the
honest framing is hedged, not a ranking claim), and fabricating a numeric
improvement estimate for any not-yet-run intervention.

## Class-weighted BCE proof-of-concept

Built same-day (Aug 5) as a fast, real alternative to fabricating an
improvement estimate for the abstract's "planned interventions" list — see
bugs #10-12 above for what went wrong building it, and `PROTOCOL.md`'s
"Results status" for where the code and checkpoints live.

**Real result** (1 seed, 20% screening subset, naive 0.5 threshold): F1 rose
substantially on both classes the intervention targets — Atelectasis
0.02→0.33, Consolidation 0.01→0.19 — while AP moved slightly *down* on every
one of the 5 competition classes (−0.01 to −0.02). Read honestly, not as an
unambiguous win: this pattern is consistent with `pos_weight` shifting the
model's output probabilities upward (so more predictions cross a fixed
threshold) rather than actually improving ranking quality, and it may
substantially overlap with what validation-tuned thresholding (already in
the PDF as Figure 1) recovers *without any retraining at all*. Whether
class-weighting adds anything beyond what free threshold-tuning already
gives is genuinely unresolved — flagged as such in the PDF rather than
claimed as a clean additive win. Screening-tier, 1-seed — not a headline-tier
confirmed result.

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
   changed, needs a call. Lowest-priority open item in the whole project.
2. ~~Whether to build Gap #1 (calibration/thresholds) now~~ — resolved, built,
   done. See "SPIE supplemental PDF — status" above.
3. Real-world confirmation of the dataloader speedup changes (see bug #6
   above) — needs a fresh Colab restart and a shared progress-log snippet.
   Still outstanding, still low-priority relative to everything below.
4. **New, highest priority**: the other 5 architectures (SwinL, SwinB V2,
   DenseNet-121, RadDino, ConvNeXtV2) still need to actually run under this
   frozen protocol — nothing currently reported for them beyond old pilot
   data. This is the real gap between the Aug 5 abstract and the Jan 2027
   final manuscript.
5. Whether class-weighted BCE's F1 gain (see "Class-weighted BCE
   proof-of-concept" above) is additive over validation-tuned thresholding
   or substantially redundant with it — not yet tested, would need running
   the plain-BCE screening checkpoint through the same threshold-tuning
   analysis Gap #1 already does and comparing.
6. Two PDF placeholders (author info, disclosure statement) — waiting on the
   user, not an engineering task.

## How to pick this up cold

1. `git pull` — gets `PROTOCOL.md`, this file, and every notebook/code change.
2. Read `PROTOCOL.md`'s "Results status" section first — it says, as of
   right now, which of the 6 architectures actually have a real result under
   the frozen protocol (as of 2026-08-05: only Ark+) vs. old pilot data.
3. Check whether any Colab training runs are already in progress (Drive
   folders `<model>_chexpert_checkpoints/` under experiment names
   `_protocol1_<model>_screening` / `_headline`) before starting anything new,
   to avoid an accidental duplicate/colliding run.
4. `PROTOCOL.md`'s "Open items" section is the actual TODO list.
5. The submittable deliverable is `SPIE_Abstract/spie_supplemental.tex` — see
   "SPIE supplemental PDF — status" above for its current state and the two
   remaining placeholders. If asked to work on "the abstract," this file (not
   a separate abstract document) is almost always what's meant — the actual
   200-300 word abstract/50-150 word summary are separate text fields typed
   directly into SPIE's submission form, not yet drafted anywhere in the repo.
