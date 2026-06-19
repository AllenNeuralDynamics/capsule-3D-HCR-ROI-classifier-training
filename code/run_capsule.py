"""Reproducible-run entry for the 3-D HCR ROI-quality classifier TRAINING capsule.

Retrains the classifier on the merged human label assets using ONLY the feature
values embedded in each label record (self-contained; **no feature extraction, no
attached HCR/features assets, NO base model**). Writes the model + meta + LOSO
metrics and logs the run to MLflow.

Input: attach one or more label data assets (each a per-session *.jsonl set). They are
auto-discovered under the data root and merged newest-wins — no path argument; attach as
many as you like. Each label carries segmentation_asset, code_commit, and the embedded
feature values. The capsule has NO subject argument: it trains on every subject present
in the labels (the subject list is read from the labels themselves).

The feature schema is DERIVED from the labels themselves (every self-contained label
embeds its features by name); a strict same-set check rejects a mixed label set. So
the capsule needs no vendored/base model — the retrained model + its schema are written
to --output_dir. The run is logged to Code Ocean's NATIVE MLflow (enable "Track this
Capsule" in Capsule Settings → MLflow tab; MLFLOW_TRACKING_URI is auto-injected — never
set it here); promote a run by registering its model from the MLflow UI. Warnings are
emitted for conflicting label values and re-labeled ROIs (newest-wins applied).
"""
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


def _discover_label_dir() -> str:
    """Find and merge ALL attached label assets under the data root — the only mode.

    Scans the data root for per-session label files (`roi_qc_actions*.jsonl`, else any
    `*.jsonl`) across every attached label-asset mount and returns one directory the loader
    can read. When they span multiple asset folders they are symlinked into a single staging
    dir so newest-wins merges across them. Robust to arbitrary CodeOcean mount names; attach
    as many label assets as you like and they are all merged. Raises if none are found.
    """
    data_root = Path(os.environ.get("MFISH_DATA_ROOT", "/root/capsule/data"))
    found = sorted(data_root.rglob("roi_qc_actions*.jsonl")) or sorted(data_root.rglob("*.jsonl"))
    if not found:
        raise FileNotFoundError(
            f"no label assets found under {data_root} — attach at least one label data asset "
            f"(per-session roi_qc_actions*.jsonl) to the capsule.")
    parents = {f.parent for f in found}
    if len(parents) == 1:
        d = str(next(iter(parents)))
        print(f"[labels] discovered {len(found)} label file(s) in {d}", flush=True)
        return d
    stage = Path(tempfile.mkdtemp(prefix="label_assets_"))
    for f in found:
        link = stage / f"{f.parent.name}__{f.name}"
        try:
            link.symlink_to(f)
        except OSError:
            shutil.copyfile(f, link)
    print(f"[labels] merged {len(found)} label file(s) from {len(parents)} asset(s) → {stage}", flush=True)
    return str(stage)


def _scan_labels(label_dir: str, trainable: set[str]) -> tuple[list[Path], list[str], int]:
    """Single pass over the discovered label files. Returns
    (files, sorted subjects with >=1 trainable label, total label events).

    The capsule has NO subject argument — it trains on every subject present in the
    labels. The subject list is read FROM the labels here (distinct sids that carry a
    trainable label, i.e. one of `trainable`; subjects with only `unsure`/`_undone_`
    events are excluded so they don't create empty LOSO folds)."""
    d = Path(label_dir)
    files = sorted(d.glob("*.jsonl")) if d.is_dir() else [d]
    sids: set[str] = set()
    n_events = 0
    for p in files:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                n_events += 1
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if rec.get("label") in trainable:
                    sids.add(str(rec.get("sid")))
    return files, sorted(sids), n_events


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Retrain HCR ROI-quality classifier from embedded label features; log to MLflow.")
    ap.add_argument("--output_dir", default="/root/capsule/results")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    # No base model: point the package's model dir at our OUTPUT so the import-time
    # schema load finds nothing (→ empty) and train_embedded derives the schema from
    # the labels, then writes the new model + meta here.
    os.environ["MFISH_MODELS_DIR"] = str(out)

    from roi_classifier import model

    label_dir = _discover_label_dir()
    # Train on EVERY subject present in the labels — no subject argument. The subject list
    # and counts are read from the labels themselves.
    label_files, subjects, n_total_labels = _scan_labels(label_dir, set(model.CLASS_NAMES))
    if not subjects:
        raise SystemExit("[error] no trainable labels found in the attached label assets.")
    print(f"[labels] {n_total_labels} label event(s) across {len(label_files)} file(s); "
          f"training on {len(subjects)} subject(s): {', '.join(subjects)}", flush=True)

    meta = model.train_embedded(subjects=subjects, label_log_path=Path(label_dir), out_dir=out)
    b, f = meta["binary"], meta["four_class"]
    print(f"[train] {len(meta['feature_columns'])} feats | binary LOSO AUC={b['loso_mean_auc']:.4f} "
          f"| 4-class f1_macro={f['loso_mean_f1_macro']:.4f}", flush=True)

    # ---- MLflow tracking (Code Ocean NATIVE integration) ----
    # Enable "Track this Capsule" in Capsule Settings → MLflow tab. MLFLOW_TRACKING_URI is
    # auto-injected by Code Ocean, which also auto-tags the run (codeocean.computationID /
    # capsuleID / userID / resultsFolder) and assigns the capsule's experiment — so we must
    # NOT set the tracking URI or the experiment here. We log explicitly rather than via
    # autolog: the custom LOSO + production loop doesn't map onto a single estimator.fit().
    # The try/except keeps local / headless runs (no MLflow server, or a viewer of a
    # non-Release capsule) working — the model + meta are in /results regardless.
    try:
        import mlflow
        n_subjects = len(subjects)
        params = {"n_features": len(meta["feature_columns"]),
                  "n_subjects": n_subjects,
                  "subjects": ",".join(subjects),
                  "training": "embedded_features_no_base_model"}
        # str() the hyper-params: some (e.g. lgb 'metric') are lists, which not all
        # MLflow versions accept as param values.
        params.update({f"binary_{k}": str(v) for k, v in b["params"].items()})
        params.update({f"fourclass_{k}": str(v) for k, v in f["params"].items()})
        # run_name is just the human-readable label in the MLflow Experiments list. Both
        # counts are read from the labels, so they track each retraining iteration.
        with mlflow.start_run(run_name=f"roi_quality_{n_subjects}subj_{n_total_labels}labels"):
            mlflow.log_params(params)
            mlflow.log_metrics({
                "binary_loso_auc": float(b["loso_mean_auc"]),
                "binary_loso_ap": float(b.get("loso_mean_ap", float("nan"))),
                "binary_loso_brier": float(b.get("loso_mean_brier", float("nan"))),
                "fourclass_loso_f1_macro": float(f["loso_mean_f1_macro"]),
                "fourclass_loso_acc": float(f.get("loso_mean_acc", float("nan"))),
                "binary_n_train": float(b["n_train_total"]),
                "fourclass_n_train": float(f["n_train_total"]),
            })
            # Label provenance for MLflow-UI readability (Code Ocean records the asset IDs
            # in the run lineage automatically; these tags are a convenience, not a need).
            mlflow.set_tags({"n_total_labels": n_total_labels,
                             "n_label_batches": len(label_files),
                             "model_kind": "lightgbm_roi_quality"})
            # Log the model files as artifacts under 'model/' so the run is registrable from
            # the MLflow UI (the same files are also in /results for a result data asset).
            for fn in ("roi_quality_binary.txt", "roi_quality_4class.txt", "roi_quality_meta.json"):
                p = out / fn
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path="model")
            print(f"[mlflow] logged run: {n_total_labels} label event(s) across "
                  f"{len(label_files)} batch(es); model artifacts under 'model/'.", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[mlflow] skipped/failed ({e}); model + meta are in {out}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
