"""
Text Model Improvements for Day 1
- Stronger TF-IDF (preprocessing, n-grams, sublinear tf)
- CV tuning for Ridge and SGDRegressor (elastic net)
- Artifact saving (models + submission)
Use from ml_challenge_day1.ipynb to upgrade text-only performance.
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional

# Support both package and script execution
try:
    from .utils import (
        smape,
        smape_scorer,
        vectorize_text,
        combine_struct_and_text,
        tune_ridge_on_tfidf,
        tune_sgd_elasticnet_on_tfidf,
        save_text_features_cache,
        load_text_features_cache,
        summarize_predictions,
    )
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from utils import (
        smape,
        smape_scorer,
        vectorize_text,
        combine_struct_and_text,
        tune_ridge_on_tfidf,
        tune_sgd_elasticnet_on_tfidf,
        save_text_features_cache,
        load_text_features_cache,
        summarize_predictions,
    )


def run_text_tfidf_and_models(
    train_text: pd.Series,
    test_text: pd.Series,
    X_train_struct: pd.DataFrame,
    X_test_struct: pd.DataFrame,
    y_train: pd.Series,
    cache_dir: str = "models/cache_text_v2",
    tfidf_params: Optional[Dict[str, Any]] = None,
    ridge_alphas: Optional[np.ndarray] = None,
    sgd_l1_ratios: Optional[list] = None,
    cv_splits: int = 5,
    save_model_dir: str = "models",
    save_submission_path: str = "dataset/day1_submission_v2.csv",
) -> Dict[str, Any]:
    """
    Run improved TF-IDF vectorization and compare Ridge & SGDRegressor (elastic net).
    Returns dict with cv results, chosen model, predictions, and artifact paths.
    """
    os.makedirs(save_model_dir, exist_ok=True)

    # Default TF-IDF params
    if tfidf_params is None:
        tfidf_params = {
            "max_features": 20000,
            "sublinear_tf": True,
            "ngram_range": (1, 2),
            "min_df": 3,
            "max_df": 0.98,
            "stop_words": "english",
        }

    # Try loading cached text features first
    cached = load_text_features_cache(cache_dir)
    if cached is not None:
        tfidf, X_train_text, X_test_text = cached
        used_cache = True
    else:
        tfidf, X_train_text, X_test_text = vectorize_text(train_text, test_text, **tfidf_params)
        save_text_features_cache(cache_dir, tfidf, X_train_text, X_test_text)
        used_cache = False

    # Combine structured + text
    X_train_combined = combine_struct_and_text(X_train_struct, X_train_text)
    X_test_combined = combine_struct_and_text(X_test_struct, X_test_text)

    # Tune Ridge
    ridge_model, best_alpha, ridge_cv = tune_ridge_on_tfidf(
        X_train_combined, y_train.values, alphas=ridge_alphas, cv_splits=cv_splits
    )

    # Tune SGD (elastic net)
    sgd_model, sgd_params, sgd_cv = tune_sgd_elasticnet_on_tfidf(
        X_train_combined, y_train.values, l1_ratio_list=sgd_l1_ratios, cv_splits=cv_splits
    )

    # Choose best by CV SMAPE (lower is better)
    models_info = [
        {"name": "Ridge", "model": ridge_model, "cv_smape": ridge_cv, "params": {"alpha": best_alpha}},
        {"name": "SGDRegressor(elasticnet)", "model": sgd_model, "cv_smape": sgd_cv, "params": sgd_params},
    ]
    models_info.sort(key=lambda m: m["cv_smape"])  # ascending by SMAPE
    best = models_info[0]

    # Fit best on full training (already fitted, but refit to be safe)
    best_model = best["model"]
    best_model.fit(X_train_combined, y_train.values)

    # Predict on test
    y_pred = best_model.predict(X_test_combined)
    y_pred = np.clip(y_pred, 0.01, None)

    # Save artifacts
    tfidf_path = os.path.join(save_model_dir, "day1_tfidf_v2.pkl")
    model_path = os.path.join(save_model_dir, "day1_best_model_v2.pkl")
    joblib.dump(tfidf, tfidf_path)
    joblib.dump(best_model, model_path)

    # Save submission
    # Caller should handle sample_id externally; here we only return predictions
    pred_stats = summarize_predictions(y_pred)

    return {
        "used_cache": used_cache,
        "tfidf_params": tfidf_params,
        "ridge_cv_smape": ridge_cv,
        "ridge_best_alpha": best_alpha,
        "sgd_cv_smape": sgd_cv,
        "sgd_best_params": sgd_params,
        "best_model_name": best["name"],
        "best_model_params": best["params"],
        "tfidf_path": tfidf_path,
        "model_path": model_path,
        "predictions": y_pred,
        "prediction_stats": pred_stats,
    }


# Helper to pick a reasonable text column
def _select_text_column(df: pd.DataFrame) -> pd.Series:
    if 'catalog_content' in df.columns:
        return df['catalog_content'].fillna('')
    # Fallback: concatenate all object (string) columns
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        return df[obj_cols].astype(str).fillna('').agg(' '.join, axis=1)
    # If none, return empty strings
    return pd.Series([''] * len(df), index=df.index)

# Helper to build simple structured features (numeric columns)
def _build_struct_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_cols_train = train_df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols_test = test_df.select_dtypes(include=[np.number]).columns.tolist()
    # Common numeric columns present in both, exclude target if present
    common_num = [c for c in num_cols_train if c in num_cols_test and c.lower() not in ['price']]
    X_train_struct = train_df[common_num].copy()
    X_test_struct = test_df[common_num].copy()
    # If empty, return empty DataFrames with correct index
    if len(common_num) == 0:
        X_train_struct = pd.DataFrame(index=train_df.index)
        X_test_struct = pd.DataFrame(index=test_df.index)
    return X_train_struct, X_test_struct

# Robust CSV reader with fallbacks
def robust_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, engine='python')
        except Exception:
            try:
                return pd.read_csv(path, engine='python', encoding='utf-8', on_bad_lines='skip')
            except Exception:
                return pd.read_csv(path, engine='python', encoding='latin-1', on_bad_lines='skip')

# Save submission CSV
def save_submission_v2(sample_ids: pd.Series, predictions: np.ndarray, out_path: str) -> None:
    preds = np.clip(predictions, 0.01, None)
    submission = pd.DataFrame({'sample_id': sample_ids, 'price': preds})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    submission.to_csv(out_path, index=False)

if __name__ == "__main__":
    # Paths
    train_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'train.csv')
    test_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'test.csv')
    submission_out = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'day1_submission_v2.csv')
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

    print("📊 Loading train/test CSVs for v2 pipeline...")
    train_df = robust_read_csv(train_path)
    test_df = robust_read_csv(test_path)

    # Validate target
    if 'price' not in train_df.columns:
        raise RuntimeError("Target column 'price' not found in train.csv")
    if 'sample_id' not in test_df.columns:
        raise RuntimeError("Column 'sample_id' not found in test.csv")

    # Select text column(s)
    train_text = _select_text_column(train_df)
    test_text = _select_text_column(test_df)

    # Build simple structured features (optional, may be empty)
    X_train_struct, X_test_struct = _build_struct_features(train_df, test_df)

    # Target
    y_train = train_df['price']

    print("🚀 Running improved TF-IDF + model selection (Ridge vs SGD ElasticNet)...")
    results = run_text_tfidf_and_models(
        train_text=train_text,
        test_text=test_text,
        X_train_struct=X_train_struct,
        X_test_struct=X_test_struct,
        y_train=y_train,
        cache_dir=os.path.join(models_dir, 'cache_text_v2'),
        save_model_dir=models_dir,
        save_submission_path=submission_out,
        cv_splits=5,
    )

    # Save submission
    print("💾 Saving day1_submission_v2.csv...")
    save_submission_v2(test_df['sample_id'], results['predictions'], submission_out)

    # Report
    print("✅ v2 artifacts saved:")
    print(f"   TF-IDF: {results['tfidf_path']}")
    print(f"   Best model: {results['model_path']} ({results['best_model_name']})")
    print("📈 CV SMAPE:")
    print(f"   Ridge: {results['ridge_cv_smape']:.4f}% (alpha={results['ridge_best_alpha']})")
    print(f"   SGD ElasticNet: {results['sgd_cv_smape']:.4f}% (params={results['sgd_best_params']})")
    print("📊 Prediction stats:")
    print(results['prediction_stats'])
    print(f"📝 Submission saved to: {submission_out}")