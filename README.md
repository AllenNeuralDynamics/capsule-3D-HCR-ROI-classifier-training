# 3D HCR ROI-quality classifier training (capsule)

Reproducible-run CodeOcean capsule that **retrains** the HCR ROI-quality classifier and
logs the run to Code Ocean's **native MLflow**. Uses `model.train_embedded` from
[`mfish-roi-classifier`](https://github.com/jkim0731/mfish-roi-classifier).

## Self-contained: labels asset only, no base model
Training reads the feature values **embedded in each label record** — **no feature
extraction, no attached HCR/features assets, no base model**. So the only input is the
**labels asset** (per-session `*.jsonl`, merged newest-wins; each label carries
`segmentation_asset`, `code_commit`, and the embedded features). The **feature schema is
derived from the labels themselves** (the embedded feature names), so the capsule does
not borrow a schema from any vendored model — it *produces* the authoritative model.

## Reproducible Run
`code/run` → `code/run_capsule.py`:
1. `model.train_embedded` on the merged label assets (LOSO CV + production fit) →
   `roi_quality_{binary,4class}.txt` + `roi_quality_meta.json` + per-subject OOF in `/results`.
2. Logs params/metrics/model artifacts to MLflow.

The **headline metric is the pooled out-of-fold (micro) AUC / f1_macro** — every held-out
subject's predictions go into one pool, so it stays well-defined even when individual
subjects have a single class or few labels (robust to uneven per-subject distributions).
Per-subject LOSO means are kept as **diagnostics**, and degenerate folds (single-class eval
→ undefined AUC, or `< 20` eval labels) are flagged in the log and meta
(`loso_nan_auc_subjects`, `loso_low_n_subjects`, `loso_valid_auc_folds`).

It **fails fast** if the labels' embedded feature sets disagree (a strict same-set check —
they must all carry the identical feature names; a mismatch means mixed extractor versions,
so re-extract + re-label). It **warns** (does not fail) on conflicting label values for the
same ROI and re-labeled ROIs (newest-wins applied). `code_commit` is kept per label as
provenance only — it is not compared (the repo can change without touching extraction).

Label assets are **auto-discovered and merged automatically** — attach one or more label
data assets (any mount names, e.g. `/data/labels_v1`, `/data/labels_v2`, …) and the run
finds every `roi_qc_actions*.jsonl` across them and merges newest-wins. There is no label
path parameter; this is the only mode.

### Parameters (`.codeocean/app-panel.json`)
| param | meaning |
|---|---|
| `subjects` | comma/space-separated subject ids; empty = all 6 benchmark subjects |

### MLflow (native Code Ocean integration)
Tracking is **native**: enable it once via **Capsule Settings → MLflow tab → "Track this
Capsule"**. `MLFLOW_TRACKING_URI` is **auto-injected** — the code never sets it (nor the
experiment); Code Ocean also auto-tags each run with `codeocean.computationID` /
`capsuleID` / etc. The run logs hyperparameters, LOSO metrics, label-provenance tags
(`n_total_labels`, `n_label_batches`), and the model files as artifacts under `model/`.
If no tracking server is available (local run, or a viewer of a non-Release capsule)
logging is skipped and the model + meta still land in `/results`.

> Requires `mlflow` installed in the environment (already in `.codeocean/environment.json`,
> compatible with Code Ocean's MLflow server) and the capsule released if collaborators
> need to log runs.

### Promotion
After a satisfactory run, click **"View in MLflow"** → select the run → **Register Model**.
It becomes a versioned Registered Model in the Code Ocean **Models** dashboard (with full
code-commit + data-asset provenance). Then attach that model version to the inference
capsule (Capsule 1) as its `/data/model` asset — or vendor it into
`mfish-roi-classifier/models/`.

Part of the 3-capsule workflow: classifier (extract+infer) → labeling → **this (training)**.
