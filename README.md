# 3D HCR ROI-quality classifier training (capsule)

Reproducible-run CodeOcean capsule that **retrains** the HCR ROI-quality classifier and
logs the run to **MLflow**. Uses `model.train_embedded` from
[`mfish-roi-classifier`](https://github.com/jkim0731/mfish-roi-classifier).
Scaffold mimics `mlflow_template`.

## Self-contained: labels asset only
Training reads the feature values **embedded in each label record** — **no feature
extraction, no attached HCR/features assets**. So the only input is the **labels asset**
(per-session `*.jsonl`, merged newest-wins; each label carries `segmentation_asset`,
`code_commit`, and the embedded features).

## Reproducible Run
`code/run` → `code/run_capsule.py`:
1. `model.train_embedded` on the merged label assets (LOSO CV + production fit) →
   `roi_quality_{binary,4class}.txt` + `roi_quality_meta.json` + per-subject OOF in `/results`.
2. Logs params/metrics/model artifacts to MLflow.

It **warns** (does not fail) on: `code_commit` mismatch (embedded features from a different
extractor version), conflicting label values for the same ROI, and re-labeled ROIs.

### Parameters (`.codeocean/app-panel.json`)
| param | meaning |
|---|---|
| `label_assets` | dir of per-session `*.jsonl` label assets (merged **newest-wins**, embedded features) |
| `subjects` | comma/space-separated subject ids; empty = all 6 benchmark subjects |
| `experiment_name` | MLflow experiment suffix |

### MLflow
Set the tracking backend for your CodeOcean MLflow (`--mlflow_uri` / `MLFLOW_TRACKING_URI`,
S3/db like the `mlflow_template`). Without a URI it logs locally; if MLflow is unavailable
the model + meta still land in `/results`.

### Promotion
Promote a logged run to "Production" in the MLflow registry, then vendor that version back
into `mfish-roi-classifier/models/` so the classifier + labeling capsules pick it up.

Part of the 3-capsule workflow: classifier (extract+infer) → labeling → **this (training)**.
