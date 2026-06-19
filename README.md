# 3D HCR ROI-quality classifier training (capsule)

Reproducible-run CodeOcean capsule that **retrains** the HCR ROI-quality classifier and
logs the run to **MLflow**. Wraps `roi-classifier train` from
[`mfish-roi-classifier`](https://github.com/AllenNeuralDynamics/mfish-roi-classifier).
Scaffold mimics `mlflow_template`.

## Reproducible Run
`code/run` → `code/run_capsule.py`:
1. `roi-classifier train` on the merged label assets + the features asset
   (LOSO CV + production fit) → writes `roi_quality_{binary,4class}.txt` +
   `roi_quality_meta.json` + per-subject OOF to `/root/capsule/results`.
2. Logs params (n_features, subjects, label/features asset paths), metrics
   (binary LOSO AUC/AP, 4-class F1m/acc), and the model artifacts to MLflow.

### Parameters (`.codeocean/app-panel.json`)
| param | meaning |
|---|---|
| `label_assets` | dir of per-session `*.jsonl` label assets (merged **newest-wins**) |
| `features_dir` | dir with `{sid}_features_all.parquet` (the **features asset**, produced by the classifier capsule — **not** re-extracted here) |
| `subjects` | comma/space-separated subject ids; empty = all 6 benchmark subjects |
| `experiment_name` | MLflow experiment suffix |

### MLflow
`run_capsule.py` logs via the standard MLflow API. **Finalize the tracking backend** for your
CodeOcean MLflow setup (set `--mlflow_uri` / `MLFLOW_TRACKING_URI`, e.g. an S3/db URI like the
`mlflow_template`). Without a URI it logs locally; if MLflow is unavailable the model + meta
still land in `/results`.

### Promotion
Promote a logged run to "Production" in the MLflow registry, then vendor that version back into
`mfish-roi-classifier/models/` (release gate) so the classifier + labeling capsules pick it up.

Part of the 3-capsule workflow: classifier (extract+infer) → labeling → **this (training)**.
