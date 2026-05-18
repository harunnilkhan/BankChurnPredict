"""
Tests for the preprocessing pipeline.
"""

import pytest
import pandas as pd

from ml.preprocess import (
    load_data,
    clean_data,
    split_features_target,
    build_preprocessor,
    create_train_test_split,
    save_processed_sample,
    resolve_data_path,
    TARGET_COLUMN,
    DROP_COLUMNS,
)


DATA_PATH = resolve_data_path()


@pytest.fixture
def raw_df():
    """Load raw dataset for testing."""
    return load_data(DATA_PATH)


@pytest.fixture
def clean_df(raw_df):
    """Return cleaned dataset."""
    return clean_data(raw_df)


def test_load_data(raw_df):
    """Test that data loads correctly."""
    assert isinstance(raw_df, pd.DataFrame)
    assert raw_df.shape[0] > 0
    assert raw_df.shape[1] > 0


def test_clean_data_drops_columns(raw_df):
    """Test that identifier columns are dropped during cleaning."""
    cleaned = clean_data(raw_df)
    for col in DROP_COLUMNS:
        assert col not in cleaned.columns, f"Column {col} should be dropped"


def test_clean_data_no_missing_values(clean_df):
    """Test that cleaned data has no missing values."""
    assert clean_df.isnull().sum().sum() == 0


def test_clean_data_target_is_binary(clean_df):
    """Test that target column is binary (0 or 1)."""
    assert set(clean_df[TARGET_COLUMN].unique()).issubset({0, 1})


def test_split_features_target(clean_df):
    """Test feature/target split."""
    X, y = split_features_target(clean_df)
    assert TARGET_COLUMN not in X.columns
    assert len(y) == len(X)
    assert y.name == TARGET_COLUMN


def test_split_features_target_invalid_column(clean_df):
    """Test that invalid target column raises ValueError."""
    with pytest.raises(ValueError):
        split_features_target(clean_df, target_column="nonexistent")


def test_build_preprocessor():
    """Test that preprocessor can be built."""
    preprocessor = build_preprocessor()
    assert preprocessor is not None


def test_preprocessor_transforms_data(clean_df):
    """Test that preprocessor can fit and transform data."""
    X, y = split_features_target(clean_df)
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    assert X_transformed is not None
    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] > 0  # Should have features after encoding


def test_save_processed_sample(clean_df, tmp_path):
    """Test that a processed dataset sample can be generated."""
    X, y = split_features_target(clean_df)
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    output_path = tmp_path / "processed_churn_sample.csv"

    result_path = save_processed_sample(
        X_transformed,
        y,
        preprocessor,
        output_path=str(output_path),
        max_rows=5,
    )
    processed_sample = pd.read_csv(result_path)

    assert output_path.exists()
    assert processed_sample.shape[0] == 5
    assert TARGET_COLUMN in processed_sample.columns


def test_train_test_split(clean_df):
    """Test train/test split with stratification."""
    X, y = split_features_target(clean_df)
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)

    # Check stratification approximately preserves class distribution
    original_ratio = y.mean()
    train_ratio = y_train.mean()
    assert abs(original_ratio - train_ratio) < 0.05
