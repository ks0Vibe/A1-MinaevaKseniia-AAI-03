from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"

MODEL_OUTPUT_PATH = MODELS_DIR / "adult_income_pipeline.joblib"
METRICS_OUTPUT_PATH = REPORTS_DIR / "metrics.json"

TARGET_COLUMN = "income"

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load processed train and test datasets produced
    by Stage 1.
    """

    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_DATA_PATH}\n"
            "Run src/preprocess.py first."
        )

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Testing data not found: {TEST_DATA_PATH}\n"
            "Run src/preprocess.py first."
        )

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    return train_df, test_df

def validate_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Check that all expected columns exist and that
    processed datasets contain no missing values.
    """

    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])

    for name, df in [
        ("train", train_df),
        ("test", test_df),
    ]:
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"{name} dataset is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if df[list(required_columns)].isna().any().any():
            missing_counts = (
                df[list(required_columns)]
                .isna()
                .sum()
            )

            raise ValueError(
                f"{name} dataset still contains missing values:\n"
                f"{missing_counts[missing_counts > 0]}"
            )

        invalid_targets = set(df[TARGET_COLUMN].unique()) - {0, 1}

        if invalid_targets:
            raise ValueError(
                f"{name} dataset contains invalid target values: "
                f"{invalid_targets}"
            )

def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """
    Separate features from the target.
    """

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    return X_train, y_train, X_test, y_test


def build_pipeline() -> Pipeline:

    numeric_transformer = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_COLUMNS,
            ),
        ],
    )

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    return pipeline

def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Train the complete sklearn pipeline.
    """

    print("\nTraining model...")

    pipeline.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate the model on the reserved test dataset.
    """

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    metrics = {
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities,
            )
        ),
    }

    return metrics

def save_model(
    pipeline: Pipeline,
) -> None:
    """
    Save the complete preprocessing + model pipeline.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        MODEL_OUTPUT_PATH,
    )

    print(
        f"\nModel saved to: "
        f"{MODEL_OUTPUT_PATH}"
    )


def save_metrics(
    metrics: dict,
) -> None:
    """
    Save test metrics as JSON.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        METRICS_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(
        f"Metrics saved to: "
        f"{METRICS_OUTPUT_PATH}"
    )


def print_dataset_info(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:

    print("\nDATASET")
    print("-" * 40)

    print(
        f"Train samples: {len(X_train)}"
    )
    print(
        f"Test samples:  {len(X_test)}"
    )

    print(
        f"Number of input features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print("\nTrain target distribution:")

    print(
        y_train
        .value_counts(normalize=True)
        .sort_index()
    )

    print("\nTest target distribution:")

    print(
        y_test
        .value_counts(normalize=True)
        .sort_index()
    )


def print_metrics(
    metrics: dict,
) -> None:

    print("\nTEST METRICS")
    print("-" * 40)

    for name, value in metrics.items():
        print(
            f"{name:10s}: "
            f"{value:.4f}"
        )


def main() -> None:
    print(
        "Start pipeline..."
    )

    train_df, test_df = load_data()

    validate_data(
        train_df,
        test_df,
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
    ) = prepare_features(
        train_df,
        test_df,
    )

    print_dataset_info(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    pipeline = build_pipeline()

    pipeline = train_model(
        pipeline,
        X_train,
        y_train,
    )

    metrics = evaluate_model(
        pipeline,
        X_test,
        y_test,
    )

    print_metrics(metrics)

    save_model(pipeline)

    save_metrics(metrics)

    print(
        "\npipeline "
        "completed successfully."
    )


if __name__ == "__main__":
    main()