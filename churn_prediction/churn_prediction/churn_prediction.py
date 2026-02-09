from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "churn_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importances.csv"
SCHEMA_PATH = ARTIFACTS_DIR / "schema.json"

RANDOM_STATE = 42


def setup_logging(log_path: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("churn_prediction")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logging()


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.lower()
    )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_column_names(df)
    df = df.drop_duplicates()
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "tenure" in df.columns:
        df["tenure_years"] = df["tenure"] / 12
    if "total_charges" in df.columns and "monthly_charges" in df.columns:
        df["charges_ratio"] = df["total_charges"] / df["monthly_charges"].replace(0, pd.NA)
    return df


def split_features_target(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")
    X = df.drop(columns=[target_column])
    y = df[target_column].copy()
    if y.dtype == "object":
        normalized = y.str.strip().str.lower()
        if set(normalized.unique()) <= {"yes", "no"}:
            y = normalized.map({"yes": 1, "no": 0})
        elif set(normalized.unique()) <= {"true", "false"}:
            y = normalized.map({"true": 1, "false": 0})
    return X, y


def make_train_test_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
) -> DatasetSplit:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    return DatasetSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


@dataclass
class TrainingArtifacts:
    model: Pipeline
    metrics: Dict[str, float]
    feature_importances: pd.DataFrame
    schema: Dict[str, Any]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(X)
    model = build_model()
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    feature_names: List[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
            encoder = transformer.named_steps["onehot"]
            if hasattr(encoder, "get_feature_names_out"):
                feature_names.extend(encoder.get_feature_names_out(columns).tolist())
            else:
                feature_names.extend(encoder.get_feature_names(columns))
        else:
            feature_names.extend(columns)
    return feature_names


def compute_feature_importances(pipeline: Pipeline) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(preprocessor)
    importances = model.feature_importances_
    data = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False)
    return data


def build_schema(X: pd.DataFrame) -> Dict[str, Any]:
    return {
        "feature_columns": X.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in X.dtypes.items()},
    }


def save_artifacts(artifacts: TrainingArtifacts) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(artifacts.metrics, indent=2))
    artifacts.feature_importances.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    SCHEMA_PATH.write_text(json.dumps(artifacts.schema, indent=2))


def load_model() -> Pipeline:
    return joblib.load(MODEL_PATH)


def load_metrics() -> Dict[str, float]:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


def load_feature_importances() -> pd.DataFrame:
    if FEATURE_IMPORTANCE_PATH.exists():
        return pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return pd.DataFrame(columns=["feature", "importance"])


def load_schema() -> Dict[str, Any]:
    if SCHEMA_PATH.exists():
        return json.loads(SCHEMA_PATH.read_text())
    return {"feature_columns": [], "dtypes": {}}


class PredictionRequest(BaseModel):
    data: Dict[str, Any]

    @field_validator("data")
    @classmethod
    def validate_data(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if not value:
            raise ValueError("Input data must include feature values.")
        return value


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int


app = FastAPI(title="Churn Prediction API")


@app.on_event("startup")
def load_assets() -> None:
    if not MODEL_PATH.exists():
        logger.warning("Model artifacts not found. Train the model before serving.")


def validate_payload(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    required_columns = schema.get("feature_columns", [])
    dtypes = schema.get("dtypes", {})
    if not required_columns:
        raise HTTPException(status_code=500, detail="Model schema not available.")

    missing = [col for col in required_columns if col not in data]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required fields: {', '.join(missing)}",
        )

    cleaned: Dict[str, Any] = {}
    for col in required_columns:
        value = data.get(col)
        dtype = dtypes.get(col, "")
        if dtype.startswith("int") or dtype.startswith("float"):
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid numeric value for '{col}'",
                ) from exc
        cleaned[col] = value
    return cleaned


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        schema = load_schema()
        cleaned = validate_payload(payload.data, schema)
        model = load_model()
        df = pd.DataFrame([cleaned])
        prob = model.predict_proba(df)[:, 1][0]
        pred = int(prob >= 0.5)
        return PredictionResponse(churn_probability=float(prob), churn_prediction=pred)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    metrics = load_metrics()
    importances = load_feature_importances().head(5)
    rows = "".join(
        f"<tr><td>{row['feature']}</td><td>{row['importance']:.4f}</td></tr>"
        for _, row in importances.iterrows()
    )
    metrics_html = "".join(
        f"<li><strong>{key.capitalize()}</strong>: {value:.4f}</li>"
        for key, value in metrics.items()
    )

    return f"""
    <html>
      <head>
        <title>Churn Model Dashboard</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 40px; }}
          table {{ border-collapse: collapse; width: 60%; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; }}
          th {{ background-color: #f2f2f2; }}
        </style>
      </head>
      <body>
        <h1>Churn Model Dashboard</h1>
        <h2>Model Metrics</h2>
        <ul>
          {metrics_html}
        </ul>
        <h2>Top 5 Feature Importances</h2>
        <table>
          <tr><th>Feature</th><th>Importance</th></tr>
          {rows}
        </table>
      </body>
    </html>
    """


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Churn prediction training CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train churn model")
    train_parser.add_argument("--csv", required=True, help="Path to CSV dataset")
    train_parser.add_argument(
        "--target",
        default="churn",
        help="Target column name (default: churn)",
    )
    return parser.parse_args()


def train(csv_path: str, target_column: str) -> None:
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    logger.info("Loading data from %s", csv_file)
    df = load_data(str(csv_file))
    df = clean_data(df)
    df = feature_engineering(df)

    X, y = split_features_target(df, target_column)
    splits = make_train_test_split(X, y)

    logger.info("Training model")
    pipeline = build_pipeline(splits.X_train)
    pipeline.fit(splits.X_train, splits.y_train)

    logger.info("Evaluating model")
    predictions = pipeline.predict(splits.X_test)
    metrics = evaluate_model(splits.y_test, predictions)

    logger.info("Computing feature importances")
    importances = compute_feature_importances(pipeline)

    schema = build_schema(X)

    artifacts = TrainingArtifacts(
        model=pipeline,
        metrics=metrics,
        feature_importances=importances,
        schema=schema,
    )

    save_artifacts(artifacts)
    logger.info("Artifacts saved")

    metrics_df = pd.DataFrame([metrics])
    logger.info("Model metrics:\n%s", metrics_df.to_string(index=False))


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args.csv, args.target)


if __name__ == "__main__":
    main()
