"""Tables (CSV) and figures (PDF + PNG) for the frozen-protocol runs.

Two uses, one implementation:

1. In each Colab training notebook, after training (last cell):
       save_experiment_outputs("swinl", "headline", model_path, output_path, out_dir)
   writes, for one experiment, per-seed metrics, per-class metrics, the min-max summary
   (the submitted Table 3's format), training-length numbers, and train/validation loss +
   validation AUC curves per epoch.

2. On the Mac, after copying each model's results into results/<model_key>/ (same layout as
   results/arkplus/):
       python code/analysis/paper_outputs.py --results results --out results/paper
   builds the cross-model tables and figures (all six models; Ark+ vs Swin-L for question (a)).

Numbers come only from engine.py's own files (<exp>_results.txt and <run>_epoch_table.csv) --
nothing is recomputed or re-rounded before the final 3-decimal range, so every value traces to
a file in results/.

F1 / Sensitivity / Accuracy / MCC here are engine.py's: Youden threshold fit on the test labels,
the same kind as the submitted Table 3. They are NOT the validation-tuned Figure 1 numbers
(AllModels_Gaps_Colab.ipynb). Keep the two kinds apart.
"""
import argparse
import os
import re
import sys
from decimal import Decimal, ROUND_HALF_UP

import matplotlib
if "ipykernel" not in sys.modules:  # script use: file-only backend; in a notebook keep its inline backend
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

# ---- fixed vocabulary -----------------------------------------------------------------------
# key -> (display name, colour). Colours: the dataviz reference categorical palette, slots 1-6 in
# fixed order, validated (adjacent pairs pass CVD / normal-vision checks on white). Slots 3-5 sit
# below 3:1 contrast, so every figure names models in titles/legends and every value is in a CSV.
MODELS = {
    "arkplus":     ("Ark+",                    "#2a78d6"),
    "swinl":       ("Swin-L (no pretraining)", "#eb6834"),
    "swinv2b":     ("Swin V2-B",               "#1baf7a"),
    "densenet121": ("DenseNet-121",            "#eda100"),
    "raddino":     ("RAD-DINO",                "#e87ba4"),
    "convnextv2":  ("ConvNeXt V2-B",           "#008300"),
}
SEEDS = {"screening": [13], "headline": [13, 42]}
CLASSES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
METRICS = ["AUC", "F1", "Sensitivity", "Accuracy", "MCC", "AP", "ECE"]   # submitted Table 3 order
THRESHOLD_NOTE = ("F1/Sensitivity/Accuracy/MCC from engine.py: Youden threshold fit on the test labels "
                  "(same kind as submitted Table 3; not the validation-tuned Figure 1 numbers)")

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
LINE_W = 1.5          # ~2 px
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": MUTED, "pdf.fonttype": 42,
})


def exp_name(model_key, tier):
    return f"_protocol1_{model_key}_{tier}"


# ---- reading engine.py's files ---------------------------------------------------------------
_RUN_RE = re.compile(
    r"^(?P<run>\S+_run_\d+): mAUC = (?P<AUC>-?[\d.]+), mMCC = (?P<MCC>-?[\d.]+), mAP = (?P<AP>-?[\d.]+), "
    r"mF1 = (?P<F1>-?[\d.]+), mRecall = (?P<Sensitivity>-?[\d.]+), mAccuracy = (?P<Accuracy>-?[\d.]+), "
    r"mECE = (?P<ECE>-?[\d.]+)\s*$")
_ARR_RE = re.compile(r"^\s*(?P<name>AUC|MCC|AP|F1|Recall|Accuracy|ECE) = \[(?P<vals>[^\]]*)\]\s*$")
_ARR_NAME = {"Recall": "Sensitivity"}


def parse_results_txt(path):
    """{run: {"macro": {metric: '0.8090', ...}, "per_class": {metric: [5 floats]}}} from an engine.py
    results file. Values stay the 4-decimal strings engine.py wrote. A run tested twice (an old
    accidental re-test) raises instead of silently picking one."""
    runs, pending = {}, {}
    with open(path) as f:
        for line in f:
            m = _ARR_RE.match(line)
            if m:
                pending[_ARR_NAME.get(m["name"], m["name"])] = [float(v) for v in m["vals"].replace(" ", "").split(",") if v]
                continue
            m = _RUN_RE.match(line)
            if m:
                run = m["run"]
                if run in runs:
                    raise ValueError(f"{path}: {run} was tested more than once -- resolve by hand before using these numbers")
                runs[run] = {"macro": {k: m[k] for k in METRICS}, "per_class": dict(pending)}
                pending = {}
    return runs


def read_epoch_table(path):
    return pd.read_csv(path)


def training_length(table):
    """Best-epoch numbers from a run's epoch table. Best = first epoch reaching the maximum validation
    AUC, which is the epoch engine.py checkpointed (it saves only on a strict improvement)."""
    best = table.loc[table["AUC"].idxmax()]
    min_vl = table.loc[table["val_loss"].idxmin()]
    last = table.iloc[-1]
    return {
        "epochs_logged": int(len(table)),
        "best_val_auc_epoch": int(best["Epoch"]), "best_val_auc": float(best["AUC"]),
        "min_val_loss_epoch": int(min_vl["Epoch"]), "min_val_loss": float(min_vl["val_loss"]),
        "final_train_loss": float(last["train_loss"]), "final_val_loss": float(last["val_loss"]),
    }


def _q3(s):
    return Decimal(s).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)


def value_range(values):
    """'0.803–0.809' from engine's 4-decimal strings (half-up to 3 decimals); one seed -> one value."""
    lo, hi = min(_q3(v) for v in values), max(_q3(v) for v in values)
    return f"{lo}" if len(values) == 1 or lo == hi else f"{lo}–{hi}"


def load_experiment(model_key, tier, model_path, output_path):
    """Everything engine.py wrote for one experiment. model_path/output_path are the
    .../Models/Classification/CheXpert and .../Outputs/Classification/CheXpert folders."""
    exp = exp_name(model_key, tier)
    res_txt = os.path.join(output_path, exp + "_results.txt")
    results = parse_results_txt(res_txt) if os.path.isfile(res_txt) else {}
    runs = []
    for i, seed in enumerate(SEEDS[tier]):
        run = f"{exp}_run_{i}"
        tpath = os.path.join(model_path, exp, run + "_epoch_table.csv")
        table = read_epoch_table(tpath) if os.path.isfile(tpath) else None
        if table is None and run not in results:
            continue
        runs.append({"run": run, "seed": seed, "table": table, "result": results.get(run),
                     "results_file": res_txt, "table_file": tpath})
    return {"model_key": model_key, "tier": tier, "exp": exp, "runs": runs}


# ---- tables ------------------------------------------------------------------------------------
def per_run_rows(e):
    rows = []
    for r in e["runs"]:
        row = {"model": MODELS[e["model_key"]][0], "tier": e["tier"], "seed": r["seed"], "run": r["run"]}
        if r["result"]:
            row.update({k: float(r["result"]["macro"][k]) for k in METRICS})
        if r["table"] is not None:
            row.update(training_length(r["table"]))
        row.update({"tested": r["result"] is not None, "results_file": os.path.basename(r["results_file"]),
                    "threshold_note": THRESHOLD_NOTE})
        rows.append(row)
    return rows


def per_class_rows(e):
    rows = []
    for r in e["runs"]:
        if not r["result"]:
            continue
        for metric in METRICS:
            vals = r["result"]["per_class"].get(metric)
            if vals:
                for cls, v in zip(CLASSES, vals):
                    rows.append({"model": MODELS[e["model_key"]][0], "tier": e["tier"], "seed": r["seed"],
                                 "class": cls, "metric": metric, "value": v})
    return rows


def summary_row(e):
    """Table 3 format: min–max over the tested seeds (a range, never mean ± std, with 2 seeds)."""
    tested = [r for r in e["runs"] if r["result"]]
    if not tested:
        return None
    row = {"model": MODELS[e["model_key"]][0], "tier": e["tier"], "seeds": " ".join(str(r["seed"]) for r in tested)}
    for k in METRICS:
        row[k] = value_range([r["result"]["macro"][k] for r in tested])
    complete = len(tested) == len(SEEDS[e["tier"]])
    row["complete"] = complete
    row["threshold_note"] = THRESHOLD_NOTE
    return row


# ---- figures -----------------------------------------------------------------------------------
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)


def _mark_best(ax, table, color, label=True, ls="-"):
    t = training_length(table)
    ax.plot([t["best_val_auc_epoch"]], [t["best_val_auc"]], "o", color=color, markersize=6.5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
    if label:  # above the dot (it is the curve's maximum), with headroom so it never touches the line
        ax.annotate(f"best {t['best_val_auc']:.3f} · epoch {t['best_val_auc_epoch']}",
                    (t["best_val_auc_epoch"], t["best_val_auc"]), xytext=(5, 6), textcoords="offset points",
                    fontsize=7.5, color=INK2, ha="left", va="bottom")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.15 * (hi - lo))


def _legend_below(fig, handles, labels=None):
    # Legend in its own row under the panels, so it can never collide with the title.
    kw = {"labels": labels} if labels is not None else {}
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 0.0), **kw)


def _save(fig, path_no_ext):
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    fig.savefig(path_no_ext + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [path_no_ext + ".pdf", path_no_ext + ".png"]


def plot_experiment_curves(e, path_no_ext):
    """Per experiment: columns = seeds; row 1 train vs validation loss, row 2 validation AUC with the
    checkpointed (best-AUC) epoch marked. Hairline at the best epoch in both rows."""
    runs = [r for r in e["runs"] if r["table"] is not None]
    if not runs:
        return []
    name, color = MODELS[e["model_key"]]
    fig, axes = plt.subplots(2, len(runs), figsize=(3.4 * len(runs), 4.6), sharex=True, squeeze=False)
    for c, r in enumerate(runs):
        t, best = r["table"], training_length(r["table"])["best_val_auc_epoch"]
        ax, ax2 = axes[0, c], axes[1, c]
        ax.plot(t["Epoch"], t["train_loss"], color=MUTED, lw=LINE_W, label="Train loss")
        ax.plot(t["Epoch"], t["val_loss"], color=color, lw=LINE_W, label="Validation loss")
        ax2.plot(t["Epoch"], t["AUC"], color=color, lw=LINE_W)
        _mark_best(ax2, t, color)
        for a in (ax, ax2):
            a.axvline(best, color=MUTED, lw=0.8, zorder=0)
            _style(a)
        ax.set_title(f"Seed {r['seed']}")
        ax2.set_xlabel("Epoch")
        if c == 0:
            ax.set_ylabel("Loss (BCE, 14 labels)")
            ax2.set_ylabel("Validation macro-AUC\n(5 classes)")
    h, l = axes[0, 0].get_legend_handles_labels()
    _legend_below(fig, h, l)
    fig.suptitle(f"{name}, {e['tier']} tier: loss and validation AUC per epoch", x=0.0, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return _save(fig, path_no_ext)


def plot_pair(exps, path_no_ext, title):
    """Question (a): models overlaid (e.g. Ark+ vs Swin-L), columns = seeds; rows = train loss,
    validation loss, validation AUC."""
    exps = [e for e in exps if any(r["table"] is not None for r in e["runs"])]
    if not exps:
        return []
    seeds = SEEDS[exps[0]["tier"]]
    fig, axes = plt.subplots(3, len(seeds), figsize=(3.4 * len(seeds), 6.4), sharex=True, squeeze=False)
    for c, seed in enumerate(seeds):
        for e in exps:
            name, color = MODELS[e["model_key"]]
            r = next((r for r in e["runs"] if r["seed"] == seed and r["table"] is not None), None)
            if r is None:
                continue
            t = r["table"]
            axes[0, c].plot(t["Epoch"], t["train_loss"], color=color, lw=LINE_W, label=name)
            axes[1, c].plot(t["Epoch"], t["val_loss"], color=color, lw=LINE_W, label=name)
            axes[2, c].plot(t["Epoch"], t["AUC"], color=color, lw=LINE_W, label=name)
            _mark_best(axes[2, c], t, color, label=False)
        axes[0, c].set_title(f"Seed {seed}")
        axes[2, c].set_xlabel("Epoch")
        for a in axes[:, c]:
            _style(a)
    axes[0, 0].set_ylabel("Train loss")
    axes[1, 0].set_ylabel("Validation loss")
    axes[2, 0].set_ylabel("Validation macro-AUC")
    h, l = axes[2, 0].get_legend_handles_labels()
    _legend_below(fig, h, l)
    fig.suptitle(title, x=0.0, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return _save(fig, path_no_ext)


def plot_small_multiples(exps_by_key, path_no_ext, kind):
    """One panel per model (fixed order), shared axes. kind='auc': validation AUC per seed;
    kind='loss': train (grey) and validation (model colour) loss. Seed 13 solid, seed 42 dashed."""
    keys = list(MODELS)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), sharex=True, sharey=True)
    for ax, key in zip(axes.ravel(), keys):
        name, color = MODELS[key]
        e = exps_by_key.get(key)
        runs = [r for r in e["runs"] if r["table"] is not None] if e else []
        _style(ax)
        ax.set_title(name)
        if not runs:
            ax.text(0.5, 0.5, "not run yet", transform=ax.transAxes, ha="center", va="center", color=MUTED)
            continue
        for r in runs:
            t, ls = r["table"], "-" if r["seed"] == 13 else "--"
            if kind == "auc":
                ax.plot(t["Epoch"], t["AUC"], color=color, lw=LINE_W, ls=ls)
                _mark_best(ax, t, color, label=False)
            else:
                ax.plot(t["Epoch"], t["train_loss"], color=MUTED, lw=LINE_W, ls=ls)
                ax.plot(t["Epoch"], t["val_loss"], color=color, lw=LINE_W, ls=ls)
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("Validation macro-AUC" if kind == "auc" else "Loss (BCE)")
    keys_legend = [plt.Line2D([], [], color=INK2, lw=LINE_W, ls="-", label="Seed 13"),
                   plt.Line2D([], [], color=INK2, lw=LINE_W, ls="--", label="Seed 42")]
    if kind == "loss":
        keys_legend += [plt.Line2D([], [], color=MUTED, lw=LINE_W, label="Train"),
                        plt.Line2D([], [], color=INK2, lw=LINE_W, label="Validation (model colour)")]
    _legend_below(fig, keys_legend)
    fig.suptitle("Validation AUC per epoch, headline tier" if kind == "auc" else "Train and validation loss per epoch, headline tier",
                 x=0.0, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return _save(fig, path_no_ext)


def _dot_axes(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, width=0.8)


def _seed_marker(ax, x, y, color, seed):
    # seed 13 filled, seed 42 open: the second encoding beside colour, so seeds are told apart without it
    ax.plot([x], [y], "o", markersize=7, color=color, markeredgecolor=color, markeredgewidth=1.6,
            markerfacecolor=color if seed == 13 else SURFACE, zorder=4)


def _seed_legend(fig, tier):
    h = [plt.Line2D([], [], ls="", marker="o", markersize=7, color=INK2, markerfacecolor=INK2, label="Seed 13")]
    if tier == "headline":
        h.append(plt.Line2D([], [], ls="", marker="o", markersize=7, color=INK2, markerfacecolor=SURFACE,
                            markeredgewidth=1.6, label="Seed 42"))
        h.append(plt.Line2D([], [], color=INK2, lw=LINE_W, alpha=0.5, label="Range over seeds"))
    _legend_below(fig, h)


def plot_best_epochs(exps, tier, path_no_ext):
    """Question (a): at which epoch does each model peak? Best validation AUC (the checkpoint kept) and
    lowest validation loss, one marker per seed."""
    keys = [k for k in MODELS if k in exps and any(r["table"] is not None for r in exps[k]["runs"])]
    if not keys:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.42 * len(keys) + 1.6), sharey=True)
    for ax, (field, title) in zip(axes, [("best_val_auc_epoch", "Best validation AUC (checkpoint kept)"),
                                         ("min_val_loss_epoch", "Lowest validation loss")]):
        for y, k in enumerate(keys):
            color = MODELS[k][1]
            vals = [(r["seed"], training_length(r["table"])[field]) for r in exps[k]["runs"] if r["table"] is not None]
            xs = [v for _, v in vals]
            if len(xs) > 1:
                ax.plot([min(xs), max(xs)], [y, y], color=color, lw=LINE_W, alpha=0.5, zorder=3)
            for seed, x in vals:
                _seed_marker(ax, x, y, color, seed)
        _dot_axes(ax)
        ax.set_title(title)
        ax.set_xlim(0, 31)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        ax.set_xlabel("Epoch (of 30)")
    axes[0].set_yticks(range(len(keys)))
    axes[0].set_yticklabels([MODELS[k][0] for k in keys])
    axes[0].set_ylim(len(keys) - 0.5, -0.5)
    _seed_legend(fig, tier)
    fig.suptitle(f"When each model peaks, {tier} tier", x=0.0, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    return _save(fig, path_no_ext)


def plot_metric_ranges(exps, tier, path_no_ext):
    """All seven metrics of the submitted Table 1, per model: min–max over seeds with one marker per seed.
    Overlapping ranges are visible as such -- no ranking is implied."""
    keys = [k for k in MODELS if k in exps and any(r["result"] for r in exps[k]["runs"])]
    if not keys:
        return []
    layout = [("AUC", "AUC"), ("AP", "AP"), ("ECE", "ECE (lower is better)"), (None, None),
              ("F1", "F1*"), ("Sensitivity", "Sensitivity*"), ("Accuracy", "Accuracy*"), ("MCC", "MCC*")]
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 2 * (0.36 * len(keys) + 1.0) + 0.9), sharey=True)
    for ax, (metric, title) in zip(axes.ravel(), layout):
        if metric is None:
            ax.axis("off")
            ax.text(0.0, 0.5, "* Threshold: Youden's J fit on the\ntest labels (engine.py), as in the\n"
                    "submitted Table 3. Validation-tuned\noperating points are reported\nseparately (Gap #1).",
                    transform=ax.transAxes, fontsize=7.5, color=INK2, va="center")
            continue
        for y, k in enumerate(keys):
            color = MODELS[k][1]
            vals = [(r["seed"], float(r["result"]["macro"][metric])) for r in exps[k]["runs"] if r["result"]]
            xs = [v for _, v in vals]
            if len(xs) > 1:
                ax.plot([min(xs), max(xs)], [y, y], color=color, lw=LINE_W, alpha=0.5, zorder=3)
            for seed, x in vals:
                _seed_marker(ax, x, y, color, seed)
        _dot_axes(ax)
        ax.set_title(title)
        lo, hi = ax.get_xlim()
        pad = max(0.01, 0.08 * (hi - lo))
        ax.set_xlim(lo - pad, hi + pad)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    axes[0, 0].set_yticks(range(len(keys)))
    axes[0, 0].set_yticklabels([MODELS[k][0] for k in keys])
    axes[0, 0].set_ylim(len(keys) - 0.5, -0.5)
    _seed_legend(fig, tier)
    fig.suptitle(f"Held-out test metrics (macro over the 5 competition classes), {tier} tier", x=0.0, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    return _save(fig, path_no_ext)


# ---- tables for the paper (Markdown + LaTeX) ---------------------------------------------------
def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def _tex_table(headers, rows, caption, label):
    esc = lambda s: str(s).replace("–", "--").replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")
    lines = [r"% generated by code/analysis/paper_outputs.py from engine.py's results files -- do not edit by hand",
             r"\begin{table}[ht]", r"\centering", rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{l" + "c" * (len(headers) - 1) + "}", r"\toprule",
             " & ".join(esc(h) for h in headers) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def write_paper_tables(exps, tier, out_dir):
    """Table 3 format for every model (all seven metrics of the submitted Table 1), per-class tables, and
    training-length numbers, as .md (to read) and .tex (to paste into the manuscript)."""
    ordered = [exps[k] for k in MODELS if k in exps]
    written = []
    seeds = ", ".join(str(s) for s in SEEDS[tier])
    rows = [s for s in (summary_row(e) for e in ordered) if s]
    if rows:
        headers = ["Model"] + METRICS
        body = [[r["model"] + ("" if r["complete"] else " (incomplete)")] + [r[m] for m in METRICS] for r in rows]
        caption = (f"Held-out test performance, {tier} tier ({'min--max over seeds ' + seeds if tier == 'headline' else 'seed ' + seeds}). "
                   "F1, Sensitivity, Accuracy and MCC use engine.py's Youden threshold fit on the test labels, as in the "
                   "submitted Table 3; validation-tuned operating points are reported separately.")
        with open(os.path.join(out_dir, f"table_metrics_{tier}.md"), "w") as f:
            f.write(f"# Held-out test metrics, {tier} tier\n\n{caption.replace('--', '–')}\n\n{_md_table(headers, body)}\n")
        with open(os.path.join(out_dir, f"table_metrics_{tier}.tex"), "w") as f:
            f.write(_tex_table(headers, body, caption, f"tab:metrics_{tier}"))
        written += [f"table_metrics_{tier}.md", f"table_metrics_{tier}.tex"]
        sections = [f"# Per-class held-out test metrics, {tier} tier\n\n{'Min–max over seeds ' + seeds if tier == 'headline' else 'Seed ' + seeds}. {THRESHOLD_NOTE}."]
        for metric in METRICS:
            prow = []
            for e in ordered:
                tested = [r for r in e["runs"] if r["result"] and r["result"]["per_class"].get(metric)]
                if tested:
                    prow.append([MODELS[e["model_key"]][0]] + [value_range([f"{r['result']['per_class'][metric][j]:.4f}" for r in tested])
                                                               for j in range(len(CLASSES))])
            if prow:
                sections.append(f"## {metric}\n\n" + _md_table(["Model"] + CLASSES, prow))
        with open(os.path.join(out_dir, f"table_per_class_{tier}.md"), "w") as f:
            f.write("\n\n".join(sections) + "\n")
        written.append(f"table_per_class_{tier}.md")
    tl_rows = []
    for e in ordered:
        for r in e["runs"]:
            if r["table"] is not None:
                t = training_length(r["table"])
                tl_rows.append([MODELS[e["model_key"]][0], r["seed"], t["epochs_logged"], t["best_val_auc_epoch"],
                                f"{t['best_val_auc']:.4f}", t["min_val_loss_epoch"], f"{t['min_val_loss']:.4f}",
                                f"{t['final_train_loss']:.4f}", f"{t['final_val_loss']:.4f}"])
    if tl_rows:
        h = ["Model", "Seed", "Epochs", "Best-AUC epoch", "Best val AUC", "Lowest val-loss epoch", "Lowest val loss",
             "Final train loss", "Final val loss"]
        cap = (f"Training length, {tier} tier: the epoch whose validation macro-AUC was best (the checkpoint kept and "
               "tested) and the epoch of lowest validation loss, out of 30.")
        with open(os.path.join(out_dir, f"table_training_length_{tier}.md"), "w") as f:
            f.write(f"# Training length, {tier} tier\n\n{cap}\n\n{_md_table(h, tl_rows)}\n")
        with open(os.path.join(out_dir, f"table_training_length_{tier}.tex"), "w") as f:
            f.write(_tex_table(h, tl_rows, cap, f"tab:training_length_{tier}"))
        written += [f"table_training_length_{tier}.md", f"table_training_length_{tier}.tex"]
    return written


def write_index(out_dir, exps):
    have = {tier: [MODELS[k][0] for k in MODELS if k in exps[tier]] for tier in SEEDS}
    lines = [
        "# Paper tables and figures",
        "",
        "Generated by `code/analysis/paper_outputs.py` from `results/<model>/` (engine.py's own results files and epoch "
        "tables only; nothing is recomputed). Re-run after copying in new results:",
        "`python code/analysis/paper_outputs.py --results results --out results/paper`.",
        "",
        f"Models included: headline -- {', '.join(have['headline']) or 'none'}; screening -- {', '.join(have['screening']) or 'none'}.",
        "",
        f"**Threshold note:** {THRESHOLD_NOTE}. Validation-tuned thresholds and calibration (Figure 1's method) come from "
        "`AllModels_Gaps_Colab.ipynb` and go in `results/gap1/`.",
        "",
        "| File | What it shows | Paper question |",
        "|---|---|---|",
        "| `table_metrics_<tier>.md/.tex` | All seven Table 1 metrics per model (AUC, F1, Sensitivity, Accuracy, MCC, AP, ECE), min–max over seeds | all |",
        "| `table_per_class_<tier>.md` | The same metrics for each of the 5 competition classes | all |",
        "| `table_training_length_<tier>.md/.tex` | Best-AUC epoch, lowest-val-loss epoch, final losses per seed | (a) training length |",
        "| `metric_ranges_<tier>.pdf/.png` | The seven metrics as min–max ranges with one marker per seed | all |",
        "| `best_epochs_<tier>.pdf/.png` | Epoch of best validation AUC and of lowest validation loss per model | (a) |",
        "| `arkplus_vs_swinl_headline.pdf/.png` | Train loss, validation loss and validation AUC per epoch, Ark+ vs Swin-L | (a) pretraining vs control |",
        "| `all_models_val_auc_headline.pdf/.png` | Validation AUC per epoch, one panel per model | (a) |",
        "| `all_models_loss_headline.pdf/.png` | Train and validation loss per epoch, one panel per model | (a) overfitting |",
        "| `curves/<experiment>_curves.pdf/.png` | Per model and tier: train vs validation loss, validation AUC, best epoch marked | (a) |",
        "| `metrics_per_seed_<tier>.csv`, `metrics_per_class_<tier>.csv`, `table3_format_<tier>.csv` | The numbers behind the tables, machine-readable | all |",
        "",
    ]
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("\n".join(lines))
    return ["README.md"]


# ---- entry points ------------------------------------------------------------------------------
def save_experiment_outputs(model_key, tier, model_path, output_path, out_dir):
    """Colab, after training: tables + curves for one experiment, written to out_dir."""
    e = load_experiment(model_key, tier, model_path, output_path)
    if not e["runs"]:
        print(f"{e['exp']}: nothing to save yet (no epoch table or test results found).")
        return None
    os.makedirs(out_dir, exist_ok=True)
    written = []
    pd.DataFrame(per_run_rows(e)).to_csv(os.path.join(out_dir, e["exp"] + "_metrics_per_seed.csv"), index=False)
    written.append(e["exp"] + "_metrics_per_seed.csv")
    pc = per_class_rows(e)
    if pc:
        pd.DataFrame(pc).to_csv(os.path.join(out_dir, e["exp"] + "_metrics_per_class.csv"), index=False)
        written.append(e["exp"] + "_metrics_per_class.csv")
    s = summary_row(e)
    if s:
        pd.DataFrame([s]).to_csv(os.path.join(out_dir, e["exp"] + "_summary_range.csv"), index=False)
        written.append(e["exp"] + "_summary_range.csv")
    written += [os.path.basename(p) for p in plot_experiment_curves(e, os.path.join(out_dir, e["exp"] + "_curves"))]
    print(f"{e['exp']}: saved to {out_dir}:\n  " + "\n  ".join(written))
    if s:
        print("  summary (min–max over seeds; " + THRESHOLD_NOTE + "):")
        print("  " + "  ".join(f"{k} {s[k]}" for k in METRICS))
    return s


def build_paper_outputs(results_root, out_dir):
    """Mac, from results/<model_key>/: cross-model tables and figures."""
    exps = {tier: {} for tier in SEEDS}
    for key in MODELS:
        base = os.path.join(results_root, key)
        if not os.path.isdir(base):
            continue
        for tier in SEEDS:
            e = load_experiment(key, tier, base, base)
            if e["runs"]:
                exps[tier][key] = e
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for tier in SEEDS:
        ordered = [exps[tier][k] for k in MODELS if k in exps[tier]]
        if not ordered:
            continue
        rows = [s for s in (summary_row(e) for e in ordered) if s]
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"table3_format_{tier}.csv"), index=False)
            written.append(f"table3_format_{tier}.csv")
        pd.DataFrame([r for e in ordered for r in per_run_rows(e)]).to_csv(
            os.path.join(out_dir, f"metrics_per_seed_{tier}.csv"), index=False)
        pd.DataFrame([r for e in ordered for r in per_class_rows(e)]).to_csv(
            os.path.join(out_dir, f"metrics_per_class_{tier}.csv"), index=False)
        written += [f"metrics_per_seed_{tier}.csv", f"metrics_per_class_{tier}.csv"]
        for e in ordered:
            written += [os.path.basename(p) for p in plot_experiment_curves(e, os.path.join(out_dir, "curves", e["exp"] + "_curves"))]
        written += write_paper_tables(exps[tier], tier, out_dir)
        written += [os.path.basename(p) for p in plot_best_epochs(exps[tier], tier, os.path.join(out_dir, f"best_epochs_{tier}"))]
        written += [os.path.basename(p) for p in plot_metric_ranges(exps[tier], tier, os.path.join(out_dir, f"metric_ranges_{tier}"))]
    pair = [exps["headline"][k] for k in ("arkplus", "swinl") if k in exps["headline"]]
    missing = "" if len(pair) == 2 else "  (Swin-L not run yet)"
    written += [os.path.basename(p) for p in plot_pair(pair, os.path.join(out_dir, "arkplus_vs_swinl_headline"),
                                                       "Ark+ vs Swin-L (same architecture, no pretraining), headline tier" + missing)]
    if exps["headline"]:
        written += [os.path.basename(p) for p in plot_small_multiples(exps["headline"], os.path.join(out_dir, "all_models_val_auc_headline"), "auc")]
        written += [os.path.basename(p) for p in plot_small_multiples(exps["headline"], os.path.join(out_dir, "all_models_loss_headline"), "loss")]
    written += write_index(out_dir, exps)
    print(f"Wrote to {out_dir}:\n  " + "\n  ".join(written))
    return exps


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", required=True, help="results/ folder with one sub-folder per model key")
    ap.add_argument("--out", required=True, help="where to write tables and figures (e.g. results/paper)")
    a = ap.parse_args()
    build_paper_outputs(a.results, a.out)
