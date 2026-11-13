"""Confidence intervals for held-out test AUC, by patient-level bootstrap (no GPU, no model needed).

Why: the SPIE reviewers asked whether AUC differences between models are meaningful. Two seeds give a
min-max range (training variability); this adds the other source of uncertainty, the finite test set.

What it does
- Reads each headline run's cached test predictions (results/<model>/_protocol1_<model>_headline/<run>.csv
  and gt.csv, written once by engine.py's test step). Nothing is re-fitted and no threshold is involved:
  AUC is threshold-free, so the test set is not "touched again" in any way that selects anything.
- Rebuilds the held-out test list with the protocol's own split code and checks it against gt.csv row for
  row, to get each row's patient ID.
- Resamples PATIENTS with replacement (images of one patient are not independent), B times, and reports
  percentile 95% intervals for macro-AUC (mean of the 5 competition classes) and per-class AUC.
- Paired comparisons: the same resampled patients are applied to both models, so the interval is for the
  AUC difference on the same test set. "seed-mean" averages a model's two seeds inside each resample.

Usage: python code/analysis/uncertainty.py --results results --train-csv chestdata/train.csv --out results/paper/uncertainty
"""
import argparse, ast, json, os, re, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split as _protocol_train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATALOADER = os.path.join(HERE, "..", "Repos", "ark_plus_repo", "Ark_Plus", "Finetuning", "dataloader.py")
COMPETITION_5 = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
MODELS = [  # (results key, display name) -- same names and order as paper_outputs.py's tables
    ("arkplus", "Ark+"), ("swinl", "Swin-L (no pretraining)"), ("swinv2b", "Swin V2-B"),
    ("densenet121", "DenseNet-121"), ("raddino", "RAD-DINO"), ("convnextv2", "ConvNeXt V2-B"),
]
SEEDS = [13, 42]  # run_0, run_1
EXPECTED_TEST_ROWS = 33055


def protocol_split_code():
    """The split functions exactly as the training notebooks import them, without importing dataloader.py
    itself (it needs torch, SimpleITK, ...). Only the PROTOCOL_* constants and _protocol_* / split defs are run."""
    tree = ast.parse(open(DATALOADER).read())
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("_protocol_extract_patient_id", "build_protocol_splits", "filter_csv_by_patients"):
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith("PROTOCOL_") for t in node.targets):
            keep.append(node)
    ns = {"os": os, "re": re, "pd": pd, "np": np, "_protocol_train_test_split": _protocol_train_test_split}
    exec(compile(ast.Module(body=keep, type_ignores=[]), DATALOADER, "exec"), ns)
    return ns


def test_patients(train_csv, gt):
    """Patient ID of every held-out test row, in gt.csv's row order (checked against gt.csv)."""
    ns = protocol_split_code()
    splits = ns["build_protocol_splits"](train_csv, COMPETITION_5)
    df = pd.read_csv(train_csv)
    df["_patient"] = df["Path"].apply(ns["_protocol_extract_patient_id"])
    test = df[df["_patient"].isin(splits["held_out_test"])]
    assert len(test) == EXPECTED_TEST_ROWS, f"rebuilt held-out test has {len(test)} rows, expected {EXPECTED_TEST_ROWS}"
    labels = (test[COMPETITION_5].to_numpy() == 1).astype(float)  # U-Zeros: uncertain (-1) and blank count as negative
    assert np.array_equal(labels, gt), "rebuilt held-out test labels differ from gt.csv -- not the same test set or row order"
    return test["_patient"].to_numpy(), test["Path"].to_numpy()


class FastAUC:
    """Weighted AUC for one (labels, scores) pair with the sort done once; weights = how often each row is drawn."""

    def __init__(self, y, s):
        order = np.argsort(s, kind="mergesort")
        self.order = order
        self.y = y[order].astype(np.float64)
        ss = s[order]
        self.starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])  # first row of each group of tied scores
        self.group = np.repeat(np.arange(len(self.starts)), np.diff(np.r_[self.starts, len(ss)]))

    def __call__(self, w):
        w = w[self.order]
        wp, wn = w * self.y, w * (1.0 - self.y)
        gp, gn = np.add.reduceat(wp, self.starts), np.add.reduceat(wn, self.starts)
        below = np.cumsum(gn) - gn  # weighted negatives scoring strictly lower than each tie group
        return float(np.sum(gp * (below + 0.5 * gn)) / (wp.sum() * wn.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--train-csv", default="chestdata/train.csv")
    ap.add_argument("--out", default="results/paper/uncertainty")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261112)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    # ---- load ----
    gt, preds = None, {}
    for key, name in MODELS:
        d = os.path.join(a.results, key, f"_protocol1_{key}_headline")
        g = pd.read_csv(os.path.join(d, "gt.csv"))
        assert list(g.columns) == COMPETITION_5
        g = g.to_numpy()
        assert gt is None or np.array_equal(gt, g), f"{key}: gt.csv differs from the other models'"
        gt = g
        for i, seed in enumerate(SEEDS):
            p = pd.read_csv(os.path.join(d, f"_protocol1_{key}_headline_run_{i}.csv"))
            assert list(p.columns) == COMPETITION_5 and p.shape == gt.shape == (EXPECTED_TEST_ROWS, 5), (key, i, p.shape)
            preds[(name, seed)] = p.to_numpy()
    patients, paths = test_patients(a.train_csv, gt)
    uniq, inv = np.unique(patients, return_inverse=True)
    print(f"held-out test: {len(gt)} images, {len(uniq)} patients; rebuilt split matches gt.csv row for row")

    # ---- point estimates (must equal the logged values) and the sorted structures ----
    fast, point = {}, {}
    for (name, seed), p in preds.items():
        for k, c in enumerate(COMPETITION_5):
            fast[(name, seed, c)] = FastAUC(gt[:, k], p[:, k])
            point[(name, seed, c)] = roc_auc_score(gt[:, k], p[:, k])
            assert abs(fast[(name, seed, c)](np.ones(len(gt))) - point[(name, seed, c)]) < 1e-9
        point[(name, seed, "macro")] = float(np.mean([point[(name, seed, c)] for c in COMPETITION_5]))

    # ---- bootstrap over patients ----
    rng = np.random.default_rng(a.seed)
    keys = [(name, seed, c) for (name, seed) in preds for c in COMPETITION_5 + ["macro"]]
    boot = {k: np.empty(a.n_boot) for k in keys}
    for b in range(a.n_boot):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        w = counts[inv].astype(np.float64)  # every image of a drawn patient, as many times as the patient was drawn
        for (name, seed) in preds:
            vals = [fast[(name, seed, c)](w) for c in COMPETITION_5]
            for c, v in zip(COMPETITION_5, vals):
                boot[(name, seed, c)][b] = v
            boot[(name, seed, "macro")][b] = np.mean(vals)
        if (b + 1) % 500 == 0:
            print(f"  bootstrap {b + 1}/{a.n_boot}  ({time.time() - t0:.0f} s)")

    def ci(x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return float(lo), float(hi)

    # ---- per-run intervals ----
    rows = []
    for (name, seed, c) in keys:
        lo, hi = ci(boot[(name, seed, c)])
        rows.append({"model": name, "seed": seed, "class": c, "AUC": round(point[(name, seed, c)], 4), "CI95_low": round(lo, 4), "CI95_high": round(hi, 4)})
    per_run = pd.DataFrame(rows)
    per_run.to_csv(os.path.join(a.out, "auc_ci_headline.csv"), index=False)

    # ---- paired differences in macro-AUC, every pair of models ----
    names = [n for _, n in MODELS]
    rows = []
    for i, A in enumerate(names):
        for B in names[i + 1:]:
            for label, sel in [("seed 13", [13]), ("seed 42", [42]), ("seed-mean", SEEDS)]:
                pa = np.mean([point[(A, s, "macro")] for s in sel]); pb = np.mean([point[(B, s, "macro")] for s in sel])
                d = np.mean([boot[(A, s, "macro")] for s in sel], axis=0) - np.mean([boot[(B, s, "macro")] for s in sel], axis=0)
                lo, hi = ci(d)
                rows.append({"model_A": A, "model_B": B, "seeds": label, "AUC_A": round(float(pa), 4), "AUC_B": round(float(pb), 4),
                             "diff_A_minus_B": round(float(pa - pb), 4), "CI95_low": round(lo, 4), "CI95_high": round(hi, 4),
                             "CI_excludes_0": bool(lo > 0 or hi < 0)})
            # Every pairing of one seed of A with one seed of B (2 x 2). With only two seeds per model, seed-to-seed
            # variation can't be estimated, so a difference counts as holding across seeds only if all four pairings
            # agree in sign and each interval excludes 0.
            for sa in SEEDS:
                for sb in SEEDS:
                    if sa == sb:
                        continue  # the equal-seed pairings are the "seed 13" / "seed 42" rows above
                    pa, pb = point[(A, sa, "macro")], point[(B, sb, "macro")]
                    lo, hi = ci(boot[(A, sa, "macro")] - boot[(B, sb, "macro")])
                    rows.append({"model_A": A, "model_B": B, "seeds": f"A seed {sa} vs B seed {sb}", "AUC_A": round(float(pa), 4),
                                 "AUC_B": round(float(pb), 4), "diff_A_minus_B": round(float(pa - pb), 4), "CI95_low": round(lo, 4),
                                 "CI95_high": round(hi, 4), "CI_excludes_0": bool(lo > 0 or hi < 0)})
    diffs = pd.DataFrame(rows)
    diffs.to_csv(os.path.join(a.out, "auc_diff_ci_headline.csv"), index=False)

    # ---- markdown tables ----
    macro = per_run[per_run["class"] == "macro"]
    lines = ["# Held-out test macro-AUC with 95% confidence intervals, headline tier", "",
             f"Patient-level bootstrap ({a.n_boot} resamples of the {len(uniq)} test patients; percentile intervals). "
             "The interval reflects the finite test set only; the two seeds show training variability. Per-class values: `auc_ci_headline.csv`.", "",
             "| Model | Seed 13: AUC (95% CI) | Seed 42: AUC (95% CI) |", "|---|---|---|"]
    for n in names:
        cells = []
        for s in SEEDS:
            r = macro[(macro["model"] == n) & (macro["seed"] == s)].iloc[0]
            cells.append(f"{r['AUC']:.3f} ({r['CI95_low']:.3f}–{r['CI95_high']:.3f})")
        lines.append(f"| {n} | {cells[0]} | {cells[1]} |")
    lines += ["", "## Paired differences in macro-AUC, same resampled patients for both models", "",
              "\"Seed-mean\" averages each model's two seeds. The last two columns use all four pairings of one seed of A with one seed of B: "
              "a difference holds across seeds only if all four have the same sign and each 95% interval excludes 0. "
              "With two seeds, seed-to-seed variation cannot be estimated, so a seed-mean interval that excludes 0 is not enough on its own.", "",
              "| Model A | Model B | Seed-mean AUC A − AUC B (95% CI) | Range over the 4 seed pairings | Holds for all 4 pairings |", "|---|---|---|---|---|"]
    for _, r in diffs[diffs["seeds"] == "seed-mean"].iterrows():
        four = diffs[(diffs["model_A"] == r["model_A"]) & (diffs["model_B"] == r["model_B"]) & (diffs["seeds"] != "seed-mean")]
        assert len(four) == 4
        holds = bool(four["CI_excludes_0"].all() and (four["diff_A_minus_B"] > 0).nunique() == 1)
        lines.append(f"| {r['model_A']} | {r['model_B']} | {r['diff_A_minus_B']:+.4f} ({r['CI95_low']:+.4f} to {r['CI95_high']:+.4f}) | "
                     f"{four['diff_A_minus_B'].min():+.4f} to {four['diff_A_minus_B'].max():+.4f} | {'yes' if holds else 'no'} |")
    open(os.path.join(a.out, "table_auc_ci_headline.md"), "w").write("\n".join(lines) + "\n")

    json.dump({"n_boot": a.n_boot, "rng_seed": a.seed, "n_test_images": int(len(gt)), "n_test_patients": int(len(uniq)),
               "method": "patient-level (cluster) bootstrap, percentile 95% intervals; weights = number of times each patient is drawn",
               "inputs": "results/<model>/_protocol1_<model>_headline/{<run>.csv, gt.csv}", "script": "code/analysis/uncertainty.py"},
              open(os.path.join(a.out, "method.json"), "w"), indent=1)
    print(f"wrote {a.out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
