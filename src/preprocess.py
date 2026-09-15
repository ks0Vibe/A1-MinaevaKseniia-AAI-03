from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "adult"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ADULT_DATA_PATH = RAW_DIR / "adult.data"
ADULT_TEST_PATH = RAW_DIR / "adult.test"

TRAIN_OUTPUT_PATH = PROCESSED_DIR / "train.csv"
TEST_OUTPUT_PATH = PROCESSED_DIR / "test.csv"



COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

TARGET_COLUMN = "income"

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

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


def load_raw_data() -> pd.DataFrame:
    """
    Load both original UCI Adult files and combine them
    into one raw DataFrame.
    """

    adult_data = pd.read_csv(
        ADULT_DATA_PATH,
        names=COLUMNS,
        skipinitialspace=True,
        na_values="?",
    )

    adult_test = pd.read_csv(
        ADULT_TEST_PATH,
        names=COLUMNS,
        skipinitialspace=True,
        na_values="?",
        skiprows=1,
    )

    df = pd.concat(
        [adult_data, adult_test],
        ignore_index=True,
    )

    return df

def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
  
    df = df.copy()

    object_columns = df.select_dtypes(include="object").columns

    for column in object_columns:
        df[column] = df[column].str.strip()

    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.rstrip(".")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)

    removed = before - len(df)

    print(f"Removed duplicate rows: {removed}")

    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    for column in CATEGORICAL_COLUMNS:
        if df[column].isna().any():
            mode = df[column].mode()[0]
            df[column] = df[column].fillna(mode)

    for column in NUMERIC_COLUMNS:
        if df[column].isna().any():
            median = df[column].median()
            df[column] = df[column].fillna(median)

    return df

def remove_age_outliers(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    q1 = df["age"].quantile(0.25)
    q3 = df["age"].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    before = len(df)

    df = df[
        (df["age"] >= lower_bound)
        & (df["age"] <= upper_bound)
    ].reset_index(drop=True)

    removed = before - len(df)

    print(
        f"Age outlier range: "
        f"[{lower_bound:.2f}, {upper_bound:.2f}]"
    )

    print(f"Removed age outliers: {removed}")

    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    mapping = {
        "<=50K": 0,
        ">50K": 1,
    }

    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(mapping)

    if df[TARGET_COLUMN].isna().any():
        invalid_values = df.loc[
            df[TARGET_COLUMN].isna(),
            TARGET_COLUMN,
        ]

        raise ValueError(
            "Unknown target labels were found: "
            f"{invalid_values.unique()}"
        )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df

def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_processed_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        TRAIN_OUTPUT_PATH,
        index=False,
    )

    test_df.to_csv(
        TEST_OUTPUT_PATH,
        index=False,
    )



def print_dataset_info(
    name: str,
    df: pd.DataFrame,
) -> None:
    """
    Print basic information useful for pipeline logs.
    """

    print(f"\n{name}")
    print("-" * 40)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nTarget distribution:")
    print(
        df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
    )


def main() -> None:
    print("Starting data preprocessing...")

    df = load_raw_data()

    print(f"\nRaw dataset shape: {df.shape}")

    df = normalize_strings(df)

    df = remove_duplicates(df)

    print("\nMissing values before imputation:")
    print(df.isna().sum())

    df = impute_missing_values(df)

    print("\nMissing values after imputation:")
    print(df.isna().sum())

    df = remove_age_outliers(df)

    df = encode_target(df)

    train_df, test_df = split_data(df)

    save_processed_data(
        train_df,
        test_df,
    )

    # 9. Diagnostics
    print_dataset_info(
        "TRAIN DATASET",
        train_df,
    )

    print_dataset_info(
        "TEST DATASET",
        test_df,
    )

    print("\nPreprocessing completed successfully.")

    print(f"Train saved to: {TRAIN_OUTPUT_PATH}")
    print(f"Test saved to: {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()