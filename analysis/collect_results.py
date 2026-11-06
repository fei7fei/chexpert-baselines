"""Copy each baseline model's results (and its finished checkpoints) from Google Drive into SPIE-CAD,
one folder per model, in the same layout as results/arkplus/ and checkpoints/arkplus/:

    results/<model_key>/_protocol1_<key>_<tier>_results.txt   test metrics written by engine.py
    results/<model_key>/_protocol1_<key>_<tier>/              epoch tables, models.log, gt.csv, test predictions
    results/<model_key>/paper/                                tables + figures from paper_outputs.py
    results/<model_key>/env/                                  versions + code/dataset fingerprints, one per session
    checkpoints/<model_key>/_protocol1_<key>_<tier>_run_<i>.pth.tar   best-val-AUC checkpoint, finished runs only
    results/gap1/gap1_all_models.csv                          AllModels_Gaps_Colab.ipynb output

model_key: swinl, swinv2b, densenet121, raddino, convnextv2.

Source = the folder that holds the Drive folders (densenet121_chexpert_checkpoints/, ...): Google Drive for
desktop's "My Drive" (found automatically), or a folder downloaded from drive.google.com (--source).

Safe by design: dry run unless --copy; never deletes; never touches Ark+ (its Drive copies were altered by
the 2026-08-05 accidental resume -- results/arkplus and checkpoints/arkplus are the clean originals); never
copies a checkpoint of a run that is still training; never overwrites a checkpoint that differs.

    python code/analysis/collect_results.py            # dry run: what would be copied
    python code/analysis/collect_results.py --copy     # copy
"""
import argparse
import glob
import os
import shutil
import sys

SPIE_CAD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = {  # model_key: (Drive folder written by its notebook, args.model_name used in env_*.json)
    "swinl":       ("ark_plus_chexpert_checkpoints",    "swin_large"),   # shares Ark+'s folder; only _protocol1_swinl_* is taken
    "swinv2b":     ("swinb_chexpert_checkpoints",       "swinv2_base"),
    "densenet121": ("densenet121_chexpert_checkpoints", "densenet121"),
    "raddino":     ("raddino_chexpert_checkpoints",     "rad_dino"),
    "convnextv2":  ("convnextv2_chexpert_checkpoints",  "convnextv2_base"),
}
TIERS = ("screening", "headline")


def find_drive_root():
    roots = sorted(glob.glob(os.path.expanduser("~/Library/CloudStorage/GoogleDrive-*/My Drive")))
    if len(roots) == 1:
        return roots[0]
    if len(roots) > 1:
        sys.exit("Several Google Drive accounts found -- pick one with --source:\n  " + "\n  ".join(roots))
    sys.exit("Google Drive for desktop not found. Install it, or download the model folders from drive.google.com "
             "and pass the folder that contains them with --source.")


def finished_runs(exp_dir):
    log = os.path.join(exp_dir, "models.log")
    if not os.path.isfile(log):
        return set()
    with open(log) as f:
        return {line.split()[0] for line in f if line.strip()}


def plan(source, results_dir, ckpt_dir):
    """List of (kind, src, dst, note) for every file to consider."""
    items = []
    for key, (folder, model_name) in MODELS.items():
        base = os.path.join(source, folder)
        models_dir = os.path.join(base, "Models", "Classification", "CheXpert")
        outputs_dir = os.path.join(base, "Outputs", "Classification", "CheXpert")
        for tier in TIERS:
            exp = f"_protocol1_{key}_{tier}"
            exp_dir = os.path.join(models_dir, exp)
            done = finished_runs(exp_dir)
            for src in sorted(glob.glob(os.path.join(exp_dir, "*"))):
                name = os.path.basename(src)
                if not os.path.isfile(src):
                    continue
                if name.endswith(".pth.tar") or name.endswith(".tmp"):
                    run = name[:-len(".pth.tar")] if name.endswith(".pth.tar") else name
                    in_progress = (name.endswith("_last.pth.tar") or name.endswith(".tmp")
                                   or run not in done or os.path.exists(os.path.join(exp_dir, run + "_last.pth.tar")))
                    if in_progress:
                        items.append(("skip", key, src, None, "run still training -- checkpoint not final yet"))
                    else:
                        items.append(("ckpt", key, src, os.path.join(ckpt_dir, key, name), ""))
                else:
                    items.append(("result", key, src, os.path.join(results_dir, key, exp, name), ""))
            res = os.path.join(outputs_dir, exp + "_results.txt")
            if os.path.isfile(res):
                items.append(("result", key, res, os.path.join(results_dir, key, os.path.basename(res)), ""))
            for src in sorted(glob.glob(os.path.join(outputs_dir, "paper", exp + "_*"))):
                items.append(("result", key, src, os.path.join(results_dir, key, "paper", os.path.basename(src)), ""))
        for src in sorted(glob.glob(os.path.join(outputs_dir, f"env_{model_name}_*.json"))):
            items.append(("result", key, src, os.path.join(results_dir, key, "env", os.path.basename(src)), ""))
    gap1 = os.path.join(source, "chexpert_protocol_v1", "gap1", "gap1_all_models.csv")
    if os.path.isfile(gap1):
        items.append(("result", "gap1", gap1, os.path.join(results_dir, "gap1", "gap1_all_models.csv"), ""))
    return items


def status(kind, src, dst):
    if kind == "skip":
        return "skip"
    if not os.path.exists(dst):
        return "new"
    s, d = os.stat(src), os.stat(dst)
    if kind == "ckpt":  # 2+ GB: same size and timestamp (copy2 keeps it) = same file; otherwise refuse
        return "same" if (s.st_size == d.st_size and int(s.st_mtime) == int(d.st_mtime)) else "conflict"
    with open(src, "rb") as a, open(dst, "rb") as b:
        return "same" if a.read() == b.read() else "update"


def main():
    ap = argparse.ArgumentParser(description="Collect baseline results from Google Drive into SPIE-CAD (one folder per model).")
    ap.add_argument("--source", help="folder containing the Drive folders (default: Google Drive for desktop's My Drive)")
    ap.add_argument("--copy", action="store_true", help="actually copy (default is a dry run)")
    ap.add_argument("--dest", default=SPIE_CAD, help="SPIE-CAD folder to copy into (default: this project)")
    ap.add_argument("--no-checkpoints", action="store_true", help="skip the .pth.tar checkpoints (2+ GB each for Swin-L)")
    a = ap.parse_args()

    source = a.source or find_drive_root()
    dest = os.path.abspath(a.dest)
    results_dir, ckpt_dir = os.path.join(dest, "results"), os.path.join(dest, "checkpoints")
    items = plan(source, results_dir, ckpt_dir)
    if a.no_checkpoints:
        items = [i for i in items if i[0] != "ckpt"]

    counts = {}
    for kind, key, src, dst, note in items:
        st = status(kind, src, dst)
        counts.setdefault(key, {}).setdefault(st, 0)
        counts[key][st] += 1
        if st in ("new", "update"):
            if a.copy:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
            print(f"{'copied ' if a.copy else 'would copy'} [{st}] {os.path.relpath(dst, dest)}")
        elif st == "conflict":
            print(f"CONFLICT (not copied): {os.path.relpath(dst, dest)} differs from Drive -- check by hand")
        elif st == "skip":
            print(f"skipped: {os.path.relpath(src, source)} ({note})")

    print(f"\nSource: {source}")
    if not items:
        print("Nothing found yet -- no baseline has written results to this Drive.")
    for key in sorted(counts):
        print(f"  {key:<12} " + "  ".join(f"{k} {v}" for k, v in sorted(counts[key].items())))
    if not a.copy:
        print("\nDry run only. Re-run with --copy to copy.")


if __name__ == "__main__":
    main()
