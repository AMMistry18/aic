"""Audit and reanalyse committed perception records, without rerunning inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "full_frame_test": "sc_plug_pose_test_baseline.json",
    "crop6_test": "sc_plug_pose_test_pad6.json",
    "crop6_holdout": "sc_plug_pose_val_croprefine.json",
}
METRICS = ("position_error_mm", "lateral_error_mm", "axis_error_deg", "rotation_error_deg")


def identity(metadata):
    # Reports contain absolute TACC paths. Only their portable basename is used.
    name = str(metadata).replace("\\", "/").rsplit("/", 1)[-1]
    match = re.fullmatch(r"(trial_\d+)_v\d+_s\d+\.json", name)
    if not match:
        raise ValueError(f"Unrecognized trial metadata: {metadata}")
    return name, match.group(1)


def load_report(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for row in report["per_group"]:
        name, trial = identity(row["metadata"])
        if name in rows:
            raise ValueError(f"Duplicate group {name}")
        rows[name] = {**row, "group_id": name, "trial_id": trial}
    if len(rows) != report["groups_total"]:
        raise ValueError("Group denominator does not match per_group records")
    poses = [r for r in rows.values() if r["pose_estimated"]]
    if len(poses) != report["groups_with_pose"]:
        raise ValueError("Pose denominator mismatch")
    for metric in METRICS:
        vals = np.asarray([r[metric] for r in poses], dtype=float)
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"Nonfinite {metric}")
        for key, value in (("median", np.median(vals)), ("p95", np.percentile(vals, 95))):
            if not np.isclose(value, report[metric][key], atol=1e-9):
                raise ValueError(f"Stored {metric}.{key} disagrees with raw groups")
    return report, rows


def paired_cluster_bootstrap(before, after, replicates=10000, seed=731):
    if before.keys() != after.keys():
        raise ValueError("Paired comparison requires exactly matched groups")
    keys = sorted(before)
    if not all(before[k]["pose_estimated"] and after[k]["pose_estimated"] for k in keys):
        raise ValueError("Missing poses: report detection failure separately before comparing error")
    trials = sorted({before[k]["trial_id"] for k in keys})
    clusters = [[k for k in keys if before[k]["trial_id"] == t] for t in trials]
    rng = np.random.default_rng(seed)
    stats = []
    for selected in rng.integers(len(trials), size=(replicates, len(trials))):
        sampled = [k for i in selected for k in clusters[i]]
        a = np.median([before[k]["position_error_mm"] for k in sampled])
        b = np.median([after[k]["position_error_mm"] for k in sampled])
        stats.append([b - a, 100 * (1 - b / a)])
    a = np.median([before[k]["position_error_mm"] for k in keys])
    b = np.median([after[k]["position_error_mm"] for k in keys])
    return {
        "groups": len(keys), "independent_trial_clusters": len(trials),
        "bootstrap_seed": seed, "bootstrap_replicates": replicates,
        "difference_of_medians_mm": float(b - a),
        "difference_ci95_mm": np.percentile(stats, [2.5, 97.5], axis=0)[:, 0].tolist(),
        "relative_median_reduction_percent": float(100 * (1 - b / a)),
        "relative_reduction_ci95_percent": np.percentile(stats, [2.5, 97.5], axis=0)[:, 1].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/evidence/perception_reanalysis")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("Use at least 100 bootstrap replicates")
    args.out.mkdir(parents=True, exist_ok=True)
    reports, rows, provenance = {}, {}, {}
    for condition, filename in FILES.items():
        path = ROOT / "docs/reports" / filename
        reports[condition], rows[condition] = load_report(path)
        provenance[condition] = {"source": str(path.relative_to(ROOT)),
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    test_trials = {r["trial_id"] for r in rows["crop6_test"].values()}
    heldout_trials = {r["trial_id"] for r in rows["crop6_holdout"].values()}
    if test_trials & heldout_trials:
        raise ValueError("Test and holdout trial IDs overlap")
    stats = paired_cluster_bootstrap(rows["full_frame_test"], rows["crop6_test"], args.bootstrap)
    summary = {"evidence_type": "historical_record_reanalysis_not_new_inference",
               "provenance": provenance, "paired_test_comparison": stats,
               "test_holdout_trial_ids_disjoint": True,
               "training_split_leakage_verified": False, "conditions": {}}
    exported = []
    table = ["| Condition | Groups / trial clusters | Position median (mm) | Lateral P95 (mm) | Axis P95 (deg) | Miss rate |",
             "|---|---:|---:|---:|---:|---:|"]
    for condition, report in reports.items():
        clusters = len({r["trial_id"] for r in rows[condition].values()})
        summary["conditions"][condition] = {
            "groups": len(rows[condition]), "trial_clusters": clusters,
            **{k: report[k] for k in (*METRICS, "group_miss_rate", "inference_seconds_per_group")},
        }
        table.append(f"| {condition} | {len(rows[condition])} / {clusters} | "
                     f"{report['position_error_mm']['median']:.4f} | "
                     f"{report['lateral_error_mm']['p95']:.4f} | "
                     f"{report['axis_error_deg']['p95']:.3f} | {report['group_miss_rate']:.3f} |")
        exported.extend({"condition": condition, "group_id": r["group_id"],
                         "trial_id": r["trial_id"], "pose_estimated": r["pose_estimated"],
                         **{k: r.get(k) for k in METRICS}} for r in rows[condition].values())
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.out / "groups.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    (args.out / "table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), layout="constrained")
    for condition, label, color in (("full_frame_test", "Full frame (tuning)", "#666666"),
                                     ("crop6_test", "Crop 6 (tuning)", "#0072B2"),
                                     ("crop6_holdout", "Crop 6 (held out)", "#D55E00")):
        for ax, metric in zip(axes, ("position_error_mm", "lateral_error_mm")):
            values = np.sort([r[metric] for r in rows[condition].values() if r["pose_estimated"]])
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post",
                    label=label, color=color, linewidth=1.5)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.2)
    axes[0].set(xlabel="3D tip-position error (mm)", ylabel="Fraction of image groups")
    axes[1].set(xlabel="Lateral tip error (mm)")
    axes[1].legend(fontsize=7, loc="lower right")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(args.out / f"perception_ecdf.{ext}", dpi=300)
    plt.close(fig)
    print(json.dumps(stats, indent=2))
    print(f"Audited historical records; outputs: {args.out}")


if __name__ == "__main__":
    main()
