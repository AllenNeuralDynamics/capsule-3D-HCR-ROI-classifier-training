"""MLflow pyfunc wrapper for the HCR ROI-quality classifier.

Bundles the trained 4-class LightGBM booster and the feature/class schema into a single
registrable, loadable MLflow model. The binary keep-score is DERIVED from the 4-class
marginal P(good)+P(bad_ok) — there is no separate binary model. Logging with
`mlflow.pyfunc.log_model` (rather than `log_artifact`) is what makes it appear as a model
under the run's `mlflow/` folder and registrable in the Code Ocean Models dashboard.

The wrapper is self-contained — it reads the booster + meta from the logged artifacts and
reproduces `roi_classifier.model.predict`'s contract, so loading the registered model needs
only lightgbm / pandas / numpy, not the training package.
"""
import json
from pathlib import Path

import lightgbm as lgb
import mlflow.pyfunc
import pandas as pd


class ROIQualityModel(mlflow.pyfunc.PythonModel):
    """binary keep-score + 4-class probabilities for HCR ROIs.

    `predict(model_input)` takes a DataFrame containing at least the model's feature
    columns (an optional `hcr_id` column is carried through) and returns a DataFrame with
    `binary_score` (P[good or bad_ok]) and one `p_<class>` column per 4-class label.
    """

    def load_context(self, context):
        self._mc = lgb.Booster(model_file=context.artifacts["four_class_model"])
        meta = json.loads(Path(context.artifacts["meta"]).read_text())
        self._features = list(meta["feature_columns"])
        self._classes = list(meta["class_names"])

    def predict(self, context, model_input, params=None):
        df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        X = df[self._features].copy()
        for c in X.columns:
            if pd.api.types.is_bool_dtype(X[c]):
                X[c] = X[c].astype("float32")
        proba = self._mc.predict(X, num_iteration=self._mc.best_iteration)
        out = pd.DataFrame()
        if "hcr_id" in df.columns:
            out["hcr_id"] = df["hcr_id"].to_numpy()
        for i, c in enumerate(self._classes):
            out[f"p_{c}"] = proba[:, i]
        # binary keep-score = 4-class marginal P(good) + P(bad_ok)
        out["binary_score"] = out["p_good"] + out["p_bad_ok"]
        return out
