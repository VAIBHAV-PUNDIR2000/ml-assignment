"""Train and evaluate multiple classification models on Adult Income data.

This script:
1. Downloads the Adult dataset from OpenML (public UCI-origin dataset).
2. Cleans and preprocesses the data.
3. Trains six classification models with a consistent preprocessing pipeline.
4. Evaluates each model on a held-out test set.
5. Saves trained models and evaluation artifacts for the Streamlit app.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from xgboost import XGBClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.20
LOGGER = logging.getLogger("train_models")


def ensure_dirs(base_dir: Path) -> Tuple[Path, Path]:
    """Create and return data and model directories."""
    data_dir = base_dir / "data"
    model_dir = base_dir / "model"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, model_dir


def load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Load Adult Income dataset from OpenML as pandas objects.

    This project also ships a cached CSV in `data/`. We prefer local data when
    present to make training reproducible even without OpenML access.
    """
    project_root = Path(__file__).resolve().parent
    cached_path = project_root / "data" / "adult_income_raw.csv"
    if cached_path.exists():
        cached_df = pd.read_csv(cached_path)
        if "income" not in cached_df.columns:
            raise ValueError(f"Expected 'income' column in cached dataset: {cached_path}")
        features = cached_df.drop(columns=["income"])
        target = cached_df["income"]
        return features, target

    try:
        dataset = fetch_openml(data_id=1590, as_frame=True)
    except Exception:  # pragma: no cover - network/API dependent
        # Fallback: name-based lookup (best effort).
        dataset = fetch_openml(name="adult", as_frame=True)

    features = dataset.data.copy()
    target = dataset.target.copy()
    return features, target


def clean_dataset(features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Normalize missing values and target labels."""
    cleaned = features.copy()
    cleaned.columns = [column.strip() for column in cleaned.columns]

    categorical_columns = cleaned.select_dtypes(include=["object", "string", "category"]).columns
    for column in categorical_columns:
        cleaned[column] = cleaned[column].apply(lambda value: value.strip() if isinstance(value, str) else value)
        cleaned[column] = cleaned[column].replace({"?": np.nan, "": np.nan})
    cleaned = cleaned.replace({pd.NA: np.nan})

    y = target.astype(str).str.strip()
    y = y.replace({">50K.": ">50K", "<=50K.": "<=50K"})
    return cleaned, y


def build_onehot_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build one-hot preprocessing for linear and distance-based models."""
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def build_ordinal_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build ordinal preprocessing for tree and probabilistic models."""
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def get_preprocessor_for_model(
    model_name: str,
    onehot_preprocessor: ColumnTransformer,
    ordinal_preprocessor: ColumnTransformer,
) -> ColumnTransformer:
    """Select preprocessing per model based on empirical accuracy."""
    if model_name in {"Random Forest", "Naive Bayes"}:
        return ordinal_preprocessor
    return onehot_preprocessor


def build_models() -> Dict[str, object]:
    """Return all models required by the assignment."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=4000,
            C=1.0,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="entropy",
            max_depth=10,
            min_samples_leaf=15,
            min_samples_split=50,
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=31,
            weights="uniform",
            metric="euclidean",
        ),
        "Naive Bayes": GaussianNB(var_smoothing=1e-5),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=700,
            learning_rate=0.02,
            max_depth=6,
            min_child_weight=3,
            gamma=0.1,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def get_param_candidates() -> Dict[str, list[Dict[str, object]]]:
    """Return small per-model search spaces for fast accuracy tuning."""
    return {
        "Logistic Regression": [
            {"C": 1.0, "max_iter": 4000},
        ],
        "Decision Tree": [
            {
                "criterion": "entropy",
                "max_depth": 10,
                "min_samples_leaf": 15,
                "min_samples_split": 50,
                "class_weight": None,
            },
        ],
        "KNN": [
            {"n_neighbors": 31, "weights": "uniform", "metric": "euclidean"},
        ],
        "Naive Bayes": [
            {"var_smoothing": 1e-5},
        ],
        "Random Forest": [
            {
                "n_estimators": 200,
                "max_features": "sqrt",
                "max_depth": None,
                "min_samples_leaf": 2,
                "class_weight": None,
                "n_jobs": -1,
            },
        ],
        "XGBoost": [
            {
                "n_estimators": 700,
                "learning_rate": 0.02,
                "max_depth": 6,
                "min_child_weight": 3,
                "gamma": 0.1,
                "subsample": 0.85,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 2.0,
                "n_jobs": -1,
            },
        ],
    }


def fit_pipeline(
    pipeline: Pipeline,
    model_name: str,
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    positive_label: str,
) -> None:
    """Fit pipeline with model-specific target handling."""
    if model_name == "XGBoost":
        y_fit_binary = (y_fit == positive_label).astype(int)
        pipeline.fit(x_fit, y_fit_binary)
    else:
        pipeline.fit(x_fit, y_fit)


def convert_predictions(raw_predictions: np.ndarray, negative_label: str, positive_label: str) -> np.ndarray:
    """Convert numeric predictions to label space if required."""
    if np.issubdtype(np.array(raw_predictions).dtype, np.number):
        return np.where(np.array(raw_predictions) == 1, positive_label, negative_label)
    return np.array(raw_predictions, dtype=str)


def tune_estimator_for_accuracy(
    model_name: str,
    base_estimator: object,
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    positive_label: str,
) -> Tuple[object, float]:
    """Tune one estimator on a validation split for best accuracy."""
    x_subtrain, x_val, y_subtrain, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    negative_label = sorted(y_train.unique().tolist())[0]
    candidates = get_param_candidates().get(model_name, [{}])
    best_accuracy = -1.0
    best_estimator = clone(base_estimator)

    for params in candidates:
        estimator = clone(base_estimator)
        estimator.set_params(**params)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )
        fit_pipeline(pipeline, model_name, x_subtrain, y_subtrain, positive_label)
        validation_predictions = convert_predictions(pipeline.predict(x_val), negative_label, positive_label)
        validation_accuracy = accuracy_score(y_val, validation_predictions)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_estimator = estimator

    return best_estimator, best_accuracy


def evaluate_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    positive_label: str,
) -> Dict[str, float]:
    """Compute all requested classification metrics."""
    negative_label = sorted(y_test.unique().tolist())[0]
    raw_predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = convert_predictions(raw_predictions, negative_label, positive_label)

    y_true_binary = (y_test == positive_label).astype(int)

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_true_binary, probabilities),
        "Precision": precision_score(y_test, predictions, pos_label=positive_label),
        "Recall": recall_score(y_test, predictions, pos_label=positive_label),
        "F1 Score": f1_score(y_test, predictions, pos_label=positive_label),
        "MCC": matthews_corrcoef(y_test, predictions),
    }


def build_ui_hints(x_train: pd.DataFrame, max_categories: int = 25) -> Dict[str, Any]:
    """Create lightweight UI hints for the Streamlit app.

    This is intentionally small (top categories only) to keep metadata readable.
    """
    categorical_columns = x_train.select_dtypes(exclude=["number"]).columns.tolist()
    categorical_options: Dict[str, list[str]] = {}
    for col in categorical_columns:
        top_values = (
            x_train[col]
            .astype(str)
            .fillna("")
            .value_counts(dropna=False)
            .head(max_categories)
            .index.tolist()
        )
        cleaned = [v for v in top_values if v != ""]
        if len(cleaned) >= 2:
            categorical_options[col] = cleaned

    return {"categorical_options": categorical_options}


def main() -> None:
    """Run the full training and persistence workflow."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    LOGGER.info("Starting training pipeline.")
    project_root = Path(__file__).resolve().parent
    data_dir, model_dir = ensure_dirs(project_root)
    LOGGER.info("Using data directory: %s", data_dir)
    LOGGER.info("Using model directory: %s", model_dir)

    x_raw, y_raw = load_dataset()
    LOGGER.info("Dataset loaded from OpenML with shape: %s", x_raw.shape)
    x_clean, y_clean = clean_dataset(x_raw, y_raw)
    LOGGER.info("Cleaned dataset shape: %s", x_clean.shape)

    raw_df = x_clean.copy()
    raw_df["income"] = y_clean
    raw_df.to_csv(data_dir / "adult_income_raw.csv", index=False)
    LOGGER.info("Saved raw data snapshot to %s", data_dir / "adult_income_raw.csv")

    x_train, x_test, y_train, y_test = train_test_split(
        x_clean,
        y_clean,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_clean,
    )
    LOGGER.info(
        "Train/Test split complete. Train shape: %s, Test shape: %s",
        x_train.shape,
        x_test.shape,
    )

    onehot_preprocessor = build_onehot_preprocessor(x_clean)
    ordinal_preprocessor = build_ordinal_preprocessor(x_clean)
    models = build_models()
    LOGGER.info("Initialized %d models for training.", len(models))

    positive_label = ">50K"
    metrics_rows = []
    trained_model_names = []
    tuning_results: Dict[str, float] = {}

    for model_name, estimator in models.items():
        LOGGER.info("----- Processing model: %s -----", model_name)
        model_preprocessor = get_preprocessor_for_model(
            model_name=model_name,
            onehot_preprocessor=onehot_preprocessor,
            ordinal_preprocessor=ordinal_preprocessor,
        )
        tuned_estimator, val_accuracy = tune_estimator_for_accuracy(
            model_name=model_name,
            base_estimator=estimator,
            preprocessor=model_preprocessor,
            x_train=x_train,
            y_train=y_train,
            positive_label=positive_label,
        )
        tuning_results[model_name] = float(val_accuracy)
        LOGGER.info("Validation accuracy after tuning for %s: %.6f", model_name, val_accuracy)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", model_preprocessor),
                ("classifier", tuned_estimator),
            ]
        )
        fit_pipeline(pipeline, model_name, x_train, y_train, positive_label)
        LOGGER.info("Fitted final pipeline for %s.", model_name)

        metrics = evaluate_model(
            model=pipeline,
            x_test=x_test,
            y_test=y_test,
            positive_label=positive_label,
        )
        metrics["Model"] = model_name
        metrics_rows.append(metrics)
        trained_model_names.append(model_name)
        LOGGER.info(
            "Test metrics for %s | Accuracy: %.6f | AUC: %.6f | F1: %.6f",
            model_name,
            metrics["Accuracy"],
            metrics["AUC"],
            metrics["F1 Score"],
        )

        model_filename = model_name.lower().replace(" ", "_") + "_pipeline.joblib"
        joblib.dump(pipeline, model_dir / model_filename, compress=3)
        LOGGER.info("Saved model artifact: %s", model_dir / model_filename)

    results_df = pd.DataFrame(metrics_rows)
    results_df = results_df[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]]
    results_df = results_df.sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
    results_df.to_csv(model_dir / "model_comparison.csv", index=False)
    LOGGER.info("Saved model comparison to %s", model_dir / "model_comparison.csv")

    test_reference = x_test.copy()
    test_reference["income"] = y_test.values
    test_reference.to_csv(data_dir / "test_reference.csv", index=False)
    LOGGER.info("Saved test reference split to %s", data_dir / "test_reference.csv")

    default_values = {}
    for column in x_clean.columns:
        if pd.api.types.is_numeric_dtype(x_clean[column]):
            default_values[column] = float(x_train[column].median())
        else:
            mode_value = x_train[column].mode(dropna=True)
            default_values[column] = str(mode_value.iloc[0]) if not mode_value.empty else ""

    metadata = {
        "dataset_name": "Adult Income (OpenML)",
        "source": "https://www.openml.org/d/1590",
        "target_column": "income",
        "positive_label": positive_label,
        "feature_columns": x_clean.columns.tolist(),
        "trained_models": trained_model_names,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "class_labels": sorted(y_clean.unique().tolist()),
        "default_values": default_values,
        "validation_accuracy_after_tuning": tuning_results,
        "ui_hints": build_ui_hints(x_train=x_train),
    }

    with (model_dir / "metadata.json").open("w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2)
    LOGGER.info("Saved metadata to %s", model_dir / "metadata.json")

    cleaned_df = x_clean.copy()
    cleaned_df["income"] = y_clean
    cleaned_df.to_csv(data_dir / "adult_income_clean.csv", index=False)
    LOGGER.info("Saved cleaned dataset to %s", data_dir / "adult_income_clean.csv")

    LOGGER.info("Training complete. Artifacts saved in 'model/' and 'data/'.")


if __name__ == "__main__":
    main()
