"""Data preprocessing utilities for the bank customer churn dataset."""

import os

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# --- Column Definitions ---
# Columns to drop (identifiers, not useful for prediction)
DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]

# Target column
TARGET_COLUMN = "Exited"

# Dataset locations. The raw dataset is preferred; the sample dataset keeps
# training and CI usable from a clean clone when the full CSV is not committed.
DEFAULT_RAW_DATA_PATH = os.path.join("data", "raw", "churn.csv")
SAMPLE_DATA_PATH = os.path.join("data", "sample", "sample_churn.csv")
PROCESSED_SAMPLE_PATH = os.path.join(
    "data", "processed", "processed_churn_sample.csv"
)

# Feature columns by type
CATEGORICAL_FEATURES = ["Geography", "Gender"]
NUMERICAL_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
VALID_CATEGORIES = {
    "Geography": ["France", "Germany", "Spain"],
    "Gender": ["Female", "Male"],
}


def resolve_data_path(path: str | None = None) -> str:
    """
    Resolve the dataset path, falling back to the committed sample dataset.

    Args:
        path: Optional explicit CSV path.

    Returns:
        Existing CSV path.

    Raises:
        FileNotFoundError: If neither the requested/default dataset nor the
            sample dataset exists.
    """
    candidate = path or DEFAULT_RAW_DATA_PATH
    if os.path.exists(candidate):
        return candidate

    can_fallback = path is None or candidate == DEFAULT_RAW_DATA_PATH
    if can_fallback and os.path.exists(SAMPLE_DATA_PATH):
        print(
            "[WARN] Full raw dataset not found. "
            f"Using sample dataset instead: {SAMPLE_DATA_PATH}"
        )
        return SAMPLE_DATA_PATH

    raise FileNotFoundError(
        f"Dataset not found at '{candidate}'. "
        f"Expected a CSV there or at '{SAMPLE_DATA_PATH}'."
    )


def load_data(path: str | None = None) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        A pandas DataFrame containing the raw dataset.
    """
    resolved_path = resolve_data_path(path)
    df = pd.read_csv(resolved_path)
    print(
        f"[INFO] Dataset loaded from {resolved_path}: "
        f"{df.shape[0]} rows, {df.shape[1]} columns"
    )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by:
    - Dropping identifier columns (RowNumber, CustomerId, Surname)
    - Handling missing values
    - Ensuring correct data types

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame ready for feature/target split.
    """
    df = df.copy()

    # Drop identifier columns
    cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"[INFO] Dropped columns: {cols_to_drop}")

    # Handle missing values - fill numerical with median, categorical with mode
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[INFO] Filled missing values in '{col}' with median: {median_val}")

    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"[INFO] Filled missing values in '{col}' with mode: {mode_val}")

    # Ensure target column is integer (binary 0/1)
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    print(f"[INFO] Cleaned dataset shape: {df.shape}")
    print(f"[INFO] Missing values remaining: {df.isnull().sum().sum()}")

    return df


def split_features_target(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> tuple:
    """
    Split the DataFrame into features (X) and target (y).

    Args:
        df: Cleaned DataFrame.
        target_column: Name of the target column.

    Returns:
        Tuple of (X, y) where X is the feature DataFrame and y is the target Series.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    print(f"[INFO] Features shape: {X.shape}")
    print(f"[INFO] Target distribution:\n{y.value_counts().to_string()}")

    return X, y


def build_preprocessor(
    categorical_features: list = None,
    numerical_features: list = None,
) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer preprocessing pipeline.

    Numerical features: SimpleImputer (median) -> StandardScaler
    Categorical features: SimpleImputer (most_frequent) -> OneHotEncoder

    Args:
        categorical_features: List of categorical column names.
        numerical_features: List of numerical column names.

    Returns:
        A fitted ColumnTransformer object.
    """
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES
    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES

    # Numerical pipeline: impute missing -> scale
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: impute missing -> one-hot encode
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",  # Drop any columns not specified
    )

    print(f"[INFO] Preprocessor built with:")
    print(f"       Numerical features: {numerical_features}")
    print(f"       Categorical features: {categorical_features}")

    return preprocessor


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Split features and target into training and test sets.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Proportion of data for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[INFO] Train set: {X_train.shape[0]} samples")
    print(f"[INFO] Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def get_processed_feature_names(
    preprocessor: ColumnTransformer,
    n_features: int,
) -> list[str]:
    """
    Return readable feature names from a fitted ColumnTransformer.

    Args:
        preprocessor: Fitted preprocessing pipeline.
        n_features: Number of transformed feature columns.

    Returns:
        List of feature names for the processed matrix.
    """
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except (AttributeError, ValueError):
        return [f"feature_{idx}" for idx in range(n_features)]


def save_processed_sample(
    X_processed,
    y,
    preprocessor: ColumnTransformer,
    output_path: str = PROCESSED_SAMPLE_PATH,
    max_rows: int = 100,
) -> str:
    """
    Save a small processed dataset sample for inspection and demos.

    The generated CSV is intentionally ignored by git because it can always be
    recreated from the raw/sample dataset and the preprocessing pipeline.

    Args:
        X_processed: Transformed feature matrix.
        y: Target values aligned with the processed matrix.
        preprocessor: Fitted preprocessing pipeline.
        output_path: Destination CSV path.
        max_rows: Maximum number of rows to write.

    Returns:
        The destination CSV path.
    """
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    processed_array = np.asarray(X_processed)
    row_count = min(max_rows, processed_array.shape[0])
    feature_names = get_processed_feature_names(preprocessor, processed_array.shape[1])

    processed_df = pd.DataFrame(
        processed_array[:row_count],
        columns=feature_names,
    )
    target_values = y.reset_index(drop=True).iloc[:row_count].to_numpy()
    processed_df[TARGET_COLUMN] = target_values

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    print(f"[INFO] Processed sample saved to: {output_path} ({row_count} rows)")

    return output_path


if __name__ == "__main__":
    # Quick test of the preprocessing pipeline
    df = load_data(DEFAULT_RAW_DATA_PATH)
    df = clean_data(df)
    X, y = split_features_target(df)
    preprocessor = build_preprocessor()
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    save_processed_sample(X_train_processed, y_train, preprocessor)

    print(f"\n[INFO] Processed train shape: {X_train_processed.shape}")
    print(f"[INFO] Processed test shape: {X_test_processed.shape}")
    print("\n[SUCCESS] Preprocessing pipeline works correctly!")
