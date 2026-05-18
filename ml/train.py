"""
Model training orchestrator for BankChurnPredict.

This module coordinates the entire training workflow:
1. Load and clean data
2. Build preprocessing pipeline
3. Train multiple candidate models
4. Evaluate and compare models
5. Select the best model
6. Save artifacts (model, preprocessor, metadata)

Usage:
    python -m ml.train
"""

from ml.preprocess import (
    load_data,
    clean_data,
    split_features_target,
    build_preprocessor,
    create_train_test_split,
    save_processed_sample,
    resolve_data_path,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from ml.evaluate import evaluate_model, compare_models, print_classification_report
from ml.model_registry import get_model_candidates, save_artifacts, build_metadata


def train_pipeline(data_path: str = None, output_dir: str = "models") -> dict:
    """
    Execute the full training pipeline.

    Args:
        data_path: Path to the raw dataset CSV.
        output_dir: Directory to save model artifacts.

    Returns:
        Dictionary containing training results and metadata.
    """
    data_path = resolve_data_path(data_path)

    print("=" * 70)
    print("  BankChurnPredict - Training Pipeline")
    print("=" * 70)

    # --- Step 1: Load and Clean Data ---
    print("\n[STEP 1] Loading and cleaning data...")
    df = load_data(data_path)
    df = clean_data(df)

    # --- Step 2: Split Features and Target ---
    print("\n[STEP 2] Splitting features and target...")
    X, y = split_features_target(df, TARGET_COLUMN)

    # --- Step 3: Train/Test Split ---
    print("\n[STEP 3] Creating train/test split...")
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    # --- Step 4: Build and Fit Preprocessor ---
    print("\n[STEP 4] Building preprocessing pipeline...")
    preprocessor = build_preprocessor(CATEGORICAL_FEATURES, NUMERICAL_FEATURES)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print(f"[INFO] Processed feature shape: {X_train_processed.shape}")
    processed_sample_path = save_processed_sample(
        X_train_processed,
        y_train,
        preprocessor,
    )

    # --- Step 5: Train Candidate Models ---
    print("\n[STEP 5] Training candidate models...")
    candidates = get_model_candidates()
    trained_models = {}
    evaluation_results = {}

    for name, model in candidates.items():
        print(f"\n--- Training: {name} ---")
        model.fit(X_train_processed, y_train)
        trained_models[name] = model

        # Evaluate
        metrics = evaluate_model(model, X_test_processed, y_test)
        evaluation_results[name] = metrics
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1-Score: {metrics['f1_score']:.4f}")
        print(f"    ROC-AUC: {metrics['roc_auc']:.4f}" if metrics["roc_auc"] else "    ROC-AUC: N/A")

    # --- Step 6: Compare and Select Best Model ---
    print("\n[STEP 6] Comparing models...")
    best_model_name = compare_models(evaluation_results)
    best_model = trained_models[best_model_name]
    best_metrics = evaluation_results[best_model_name]

    # Print detailed report for best model
    print(f"\n[STEP 6b] Detailed report for {best_model_name}:")
    print_classification_report(best_model, X_test_processed, y_test)

    # --- Step 7: Save Artifacts ---
    print("\n[STEP 7] Saving model artifacts...")
    metadata = build_metadata(best_model_name, best_metrics)
    save_artifacts(best_model, preprocessor, metadata, output_dir)

    print("\n" + "=" * 70)
    print("  Training Complete!")
    print(f"  Best model: {best_model_name}")
    print(f"  ROC-AUC: {best_metrics.get('roc_auc', 'N/A')}")
    print(f"  Artifacts saved to: {output_dir}/")
    print("=" * 70)

    return {
        "best_model_name": best_model_name,
        "metrics": best_metrics,
        "metadata": metadata,
        "processed_sample_path": processed_sample_path,
    }


if __name__ == "__main__":
    train_pipeline()
