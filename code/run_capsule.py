"""Reproducible-run entry for the 3-D HCR ROI-quality classifier TRAINING capsule.

Retrains the classifier on the merged human label assets using ONLY the feature
values embedded in each label record (self-contained; **no feature extraction, no
attached HCR/features assets**). Writes the model + meta + LOSO metrics and logs the
run to MLflow.

Input (attach as a data asset):
  --label_assets : directory of per-session *.jsonl label assets, merged newest-wins;
                   each label carries segmentation_asset, code_commit, and the embedded
                   feature values.

Warnings are emitted for code_commit mismatch, conflicting label values, and re-labeled
ROIs. The base model dir provides the feature-column list at import; the retrained model
is written to --output_dir.
"""
import argparse
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Retrain HCR ROI-quality classifier from embedded label features; log to MLflow.")
    ap.add_argument("--label_assets", default="/root/capsule/data/labels",
                    help="Directory of per-session *.jsonl label assets (merged newest-wins; embedded features).")
    ap.add_argument("--base_models_dir", default="/mfish-roi-classifier/models",
                    help="Existing model dir; provides the feature-column list at import.")
    ap.add_argument("--output_dir", default="/root/capsule/results")
    ap.add_argument("--subjects", default="",
                    help="Comma/space-separated subject ids; empty = all 6 benchmark subjects.")
    ap.add_argument("--experiment_name", default=os.environ.get("CO_CAPSULE_ID", "roi_quality"))
    ap.add_argument("--mlflow_uri", default=os.environ.get("MLFLOW_TRACKING_URI", ""))
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    os.environ["MFISH_MODELS_DIR"] = args.base_models_dir   # feature-column list at import

    from roi_classifier import model
    from roi_classifier.benchmark_data_loader import BENCHMARK_SUBJECTS
    subjects = args.subjects.replace(",", " ").split() or list(BENCHMARK_SUBJECTS)

    meta = model.train_embedded(subjects=subjects, label_log_path=Path(args.label_assets), out_dir=out)
    b, f = meta["binary"], meta["four_class"]
    print(f"[train] {len(meta['feature_columns'])} feats | binary LOSO AUC={b['loso_mean_auc']:.4f} "
          f"| 4-class f1_macro={f['loso_mean_f1_macro']:.4f}", flush=True)

    # ---- MLflow logging (finalize the tracking backend for your CodeOcean MLflow setup) ----
    try:
        import mlflow
        if args.mlflow_uri:
            mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(f"capsule_{args.experiment_name}")
        with mlflow.start_run():
            mlflow.log_params({"n_features": len(meta["feature_columns"]),
                               "subjects": ",".join(subjects),
                               "label_assets": args.label_assets,
                               "training": "embedded_features"})
            mlflow.log_metrics({"binary_loso_auc": float(b["loso_mean_auc"]),
                                "binary_loso_ap": float(b.get("loso_mean_ap", float("nan"))),
                                "fourclass_loso_f1_macro": float(f["loso_mean_f1_macro"]),
                                "fourclass_loso_acc": float(f.get("loso_mean_acc", float("nan")))})
            for fn in ("roi_quality_binary.txt", "roi_quality_4class.txt", "roi_quality_meta.json"):
                p = out / fn
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path="model")
            print("[mlflow] logged run + model artifacts.", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[mlflow] skipped/failed ({e}); model + meta are in {out}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
