import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import gc
import warnings
from typing import Tuple, List, Optional, Dict, Any

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Ensure local imports work when running as a script
sys.path.append(os.path.dirname(__file__))

from multimodal_fusion_run import (
    DATA_DIR,
    MODELS_DIR,
    robust_read_csv,
    smape,
    _select_text_column,
    svd_reduce,
    load_visual_mapping,
    build_visual_matrix,
)

# Try LightGBM, fallback gracefully if not installed
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
    print("⚠️ LightGBM not available")

# Optional: XGBoost and CatBoost (guarded imports)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("⚠️ XGBoost not available")

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except Exception:
    HAS_CAT = False
    print("⚠️ CatBoost not available")

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.base import clone
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        SMAPE score as percentage
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Handle edge cases
    mask = (y_true != 0) | (y_pred != 0)
    if not mask.any():
        return 0.0
        
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denominator = np.maximum(denominator, 1e-8)  # Avoid division by zero
    
    smape_val = np.mean(np.abs(y_true - y_pred) / denominator) * 100
    return float(smape_val)


def build_text_features(train_text: pd.Series, test_text: pd.Series) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, Dict]:
    """
    Build comprehensive text features using TF-IDF vectorization.
    
    Args:
        train_text: Training text data
        test_text: Test text data
        
    Returns:
        Tuple of (train_features, test_features, metadata)
    """
    print("  Building word-level TF-IDF features...")
    
    # Word-level TF-IDF with optimized parameters for SMAPE
    try:
        word_vec = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.8,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
            dtype=np.float32,
            stop_words='english'
        )
        Xw_train = word_vec.fit_transform(train_text.fillna(""))
        Xw_test = word_vec.transform(test_text.fillna(""))
        print(f"    Word features: {Xw_train.shape[1]} dimensions")
        gc.collect()
    except Exception as e:
        print(f"  Word TF-IDF failed ({e}), using fallback config...")
        word_vec = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 1),
            min_df=10,
            max_df=0.9,
            sublinear_tf=True,
            lowercase=True,
            dtype=np.float32
        )
        Xw_train = word_vec.fit_transform(train_text.fillna(""))
        Xw_test = word_vec.transform(test_text.fillna(""))
        gc.collect()

    print("  Building character-level TF-IDF features...")
    
    # Character-level TF-IDF
    try:
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=8,
            max_df=0.8,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
            max_features=5000,
            dtype=np.float32,
        )
        Xc_train = char_vec.fit_transform(train_text.fillna(""))
        Xc_test = char_vec.transform(test_text.fillna(""))
        print(f"    Char features: {Xc_train.shape[1]} dimensions")
        gc.collect()
    except Exception as e:
        print(f"  Char TF-IDF failed ({e}), skipping char features...")
        char_vec = None
        Xc_train = sparse.csr_matrix((Xw_train.shape[0], 0), dtype=np.float32)
        Xc_test = sparse.csr_matrix((Xw_test.shape[0], 0), dtype=np.float32)

    # Combine features
    X_train_text = sparse.hstack([Xw_train, Xc_train]).tocsr()
    X_test_text = sparse.hstack([Xw_test, Xc_test]).tocsr()
    
    # Clean up intermediate matrices
    del Xw_train, Xw_test, Xc_train, Xc_test
    gc.collect()

    meta = {"word_vec": word_vec, "char_vec": char_vec}
    return X_train_text, X_test_text, meta


def build_text_length_features(train_text: pd.Series, test_text: pd.Series) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Build simple text length and complexity features.
    
    Args:
        train_text: Training text data
        test_text: Test text data
        
    Returns:
        Tuple of (train_features, test_features, scaler)
    """
    print("  Building text length features...")
    
    def extract_text_stats(text_series):
        text_filled = text_series.fillna("")
        
        # Basic length features
        char_count = text_filled.str.len().values
        word_count = text_filled.str.split().str.len().fillna(0).values
        
        # Complexity features
        unique_char_ratio = text_filled.apply(
            lambda s: len(set(s)) / max(1, len(s)) if s else 0.0
        ).values
        
        # Punctuation and digit ratios
        punct_ratio = text_filled.apply(
            lambda s: sum(1 for c in s if not c.isalnum()) / max(1, len(s)) if s else 0.0
        ).values
        
        digit_ratio = text_filled.apply(
            lambda s: sum(1 for c in s if c.isdigit()) / max(1, len(s)) if s else 0.0
        ).values
        
        return np.column_stack([
            char_count, word_count, unique_char_ratio, punct_ratio, digit_ratio
        ]).astype(np.float32)
    
    train_features = extract_text_stats(train_text)
    test_features = extract_text_stats(test_text)
    
    # Scale features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    print(f"    Text length features: {train_features_scaled.shape[1]} dimensions")
    
    return train_features_scaled.astype(np.float32), test_features_scaled.astype(np.float32), scaler


def build_structured_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[StandardScaler]]:
    """
    Extract and process structured numeric features.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        
    Returns:
        Tuple of (train_features, test_features, feature_names, scaler)
    """
    print("  Building structured numeric features...")
    
    # Find common numeric columns (excluding target)
    train_numeric = train_df.select_dtypes(include=[np.number]).columns.tolist()
    test_numeric = test_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target and ID columns
    exclude_cols = ['price', 'sample_id']
    common_cols = [c for c in train_numeric if c in test_numeric and c not in exclude_cols]
    
    if not common_cols:
        print("    No common structured features found")
        return (
            np.zeros((len(train_df), 0), dtype=np.float32),
            np.zeros((len(test_df), 0), dtype=np.float32),
            [],
            None
        )
    
    # Extract features
    train_features = train_df[common_cols].fillna(0.0).values.astype(np.float32)
    test_features = test_df[common_cols].fillna(0.0).values.astype(np.float32)
    
    # Scale features using RobustScaler (less sensitive to outliers)
    scaler = RobustScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    print(f"    Structured features: {len(common_cols)} dimensions")
    
    return train_features_scaled.astype(np.float32), test_features_scaled.astype(np.float32), common_cols, scaler


def load_visual_transformers() -> Tuple[Optional[StandardScaler], Optional[PCA]]:
    """
    Load or create visual feature transformers.
    
    Returns:
        Tuple of (scaler, pca)
    """
    scaler_path = os.path.join(MODELS_DIR, "visual_scaler.pkl")
    pca_path = os.path.join(MODELS_DIR, "visual_pca.pkl")
    
    scaler = None
    pca = None
    
    # Try loading existing transformers
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            print(f"  Failed to load visual_scaler.pkl ({e})")
    
    if os.path.exists(pca_path):
        try:
            with open(pca_path, "rb") as f:
                pca = pickle.load(f)
        except Exception as e:
            print(f"  Failed to load visual_pca.pkl ({e})")
    
    # If transformers don't exist, create them from CSV
    if scaler is None or pca is None:
        csv_path = os.path.join(DATA_DIR, "visual_features_train.csv")
        if os.path.exists(csv_path):
            print("  Fitting visual transformers from CSV...")
            df = robust_read_csv(csv_path)
            
            # Identify feature columns
            feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "sample_id" in feat_cols:
                feat_cols.remove("sample_id")
            
            if feat_cols:
                Xv = df[feat_cols].values.astype(np.float32)
                
                if scaler is None:
                    scaler = StandardScaler()
                    scaler.fit(Xv)
                    try:
                        with open(scaler_path, "wb") as f:
                            pickle.dump(scaler, f)
                    except Exception:
                        pass
                
                if pca is None:
                    n_components = min(64, Xv.shape[1] // 2, 100)
                    pca = PCA(n_components=n_components, random_state=42)
                    pca.fit(scaler.transform(Xv))
                    try:
                        with open(pca_path, "wb") as f:
                            pickle.dump(pca, f)
                    except Exception:
                        pass
        else:
            print("  visual_features_train.csv not found")
    
    return scaler, pca


def generate_oof_predictions(model, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> Tuple[np.ndarray, float]:
    """
    Generate out-of-fold predictions using cross-validation.
    
    Args:
        model: Sklearn-compatible model
        X: Feature matrix
        y: Target values
        n_folds: Number of CV folds
        
    Returns:
        Tuple of (oof_predictions, cv_score)
    """
    # Sanitize inputs
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    
    # Remove non-finite samples
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if mask.sum() < len(y):
        print(f"    Dropping {len(y) - mask.sum()} samples with non-finite values")
    
    X_clean = X[mask]
    y_clean = y[mask]
    
    # Cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y_clean), dtype=np.float64)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_clean)):
        X_train_fold, X_val_fold = X_clean[train_idx], X_clean[val_idx]
        y_train_fold, y_val_fold = y_clean[train_idx], y_clean[val_idx]
        
        # Clone and fit model
        model_fold = clone(model)
        model_fold.fit(X_train_fold, y_train_fold)
        
        # Predict
        val_preds = model_fold.predict(X_val_fold)
        oof_preds[val_idx] = val_preds
    
    # Calculate CV score
    cv_score = calculate_smape(y_clean, oof_preds)
    
    return oof_preds, cv_score


def get_base_models() -> List[Tuple[str, Any]]:
    """
    Get list of base models for the ensemble.
    
    Returns:
        List of (name, model) tuples
    """
    models = []
    
    # Always available models
    models.extend([
        ("Ridge", Ridge(alpha=1.0, random_state=42)),
        ("ElasticNet", ElasticNet(alpha=0.5, l1_ratio=0.7, random_state=42, max_iter=2000)),
        ("RandomForest", RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42, 
            n_jobs=-1
        )),
    ])
    
    # Optional models based on availability
    if HAS_LGBM:
        models.append(("LightGBM", LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )))
    
    if HAS_XGB:
        models.append(("XGBoost", XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            random_state=42,
            n_jobs=1,
            verbosity=0
        )))
    
    if HAS_CAT:
        models.append(("CatBoost", CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            loss_function="MAE",
            random_state=42,
            verbose=False
        )))
    
    return models


def get_stacker_models() -> List[Tuple[str, Any]]:
    """
    Get list of stacker models.
    
    Returns:
        List of (name, model) tuples
    """
    stackers = [
        ("RidgeStacker", Ridge(alpha=1.0, random_state=42)),
        ("ElasticNetStacker", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))
    ]
    
    if HAS_LGBM:
        stackers.append(("LightGBMStacker", LGBMRegressor(
            n_estimators=500,
            learning_rate=0.1,
            num_leaves=15,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )))
    
    return stackers


def main():
    """
    Main function to run the stacked ensemble pipeline.
    """
    print("🚀 Starting Stacked Ensemble Pipeline")
    print("=" * 50)
    
    # Setup paths
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    submission_path = os.path.join(DATA_DIR, "day2_submission_stacked.csv")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Load data
    print("📊 Loading datasets...")
    train_df = robust_read_csv(train_path)
    test_df = robust_read_csv(test_path)
    
    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape: {test_df.shape}")
    
    # Validate required columns
    if "price" not in train_df.columns:
        raise ValueError("Target column 'price' not found in train.csv")
    if "sample_id" not in test_df.columns:
        raise ValueError("Column 'sample_id' not found in test.csv")
    
    # Extract target
    y_train = train_df["price"].values.astype(np.float32)
    print(f"  Target stats: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
    
    # Feature Engineering
    print("\n🔧 Feature Engineering")
    print("-" * 30)
    
    # Text features
    print("📝 Processing text features...")
    train_text = _select_text_column(train_df)
    test_text = _select_text_column(test_df)
    
    X_train_text, X_test_text, text_meta = build_text_features(train_text, test_text)
    
    # Apply SVD reduction to text features
    print("  Applying SVD reduction...")
    try:
        X_train_text_svd, X_test_text_svd, svd_model = svd_reduce(
            X_train_text, X_test_text, n_components=150
        )
        print(f"    SVD reduced to {X_train_text_svd.shape[1]} dimensions")
    except Exception as e:
        print(f"  SVD failed ({e}), using smaller reduction...")
        X_train_text_svd, X_test_text_svd, svd_model = svd_reduce(
            X_train_text, X_test_text, n_components=100
        )
    
    gc.collect()
    
    # Text length features
    X_train_tlen, X_test_tlen, tlen_scaler = build_text_length_features(train_text, test_text)
    
    # Structured features
    X_train_struct, X_test_struct, struct_cols, struct_scaler = build_structured_features(train_df, test_df)
    
    # Visual features
    print("🖼️ Processing visual features...")
    vis_mapping = load_visual_mapping()
    vis_scaler, vis_pca = load_visual_transformers()
    
    X_train_visual, _ = build_visual_matrix(train_df["sample_id"], vis_mapping, vis_scaler, vis_pca)
    X_test_visual, _ = build_visual_matrix(test_df["sample_id"], vis_mapping, vis_scaler, vis_pca)
    
    print(f"  Visual features: {X_train_visual.shape[1]} dimensions")
    
    # Combine all features
    print("\n🔗 Combining features...")
    feature_matrices = [X_train_text_svd, X_train_tlen, X_train_struct, X_train_visual]
    test_matrices = [X_test_text_svd, X_test_tlen, X_test_struct, X_test_visual]
    
    X_train_combined = np.hstack(feature_matrices).astype(np.float32)
    X_test_combined = np.hstack(test_matrices).astype(np.float32)
    
    print(f"  Combined features shape: {X_train_combined.shape}")
    
    # Sanitize features
    X_train_combined = np.nan_to_num(X_train_combined, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_combined = np.nan_to_num(X_test_combined, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Remove samples with non-finite targets or features
    mask_y = np.isfinite(y_train)
    mask_X = np.isfinite(X_train_combined).all(axis=1)
    mask = mask_y & mask_X
    
    if mask.sum() < len(y_train):
        print(f"  Removing {len(y_train) - mask.sum()} samples with non-finite values")
        X_train_combined = X_train_combined[mask]
        y_train = y_train[mask]
    
    # Final feature scaling
    print("  Applying final feature scaling...")
    final_scaler = StandardScaler()
    X_train_final = final_scaler.fit_transform(X_train_combined).astype(np.float32)
    X_test_final = final_scaler.transform(X_test_combined).astype(np.float32)
    
    # Clip extreme values for numerical stability
    X_train_final = np.clip(X_train_final, -10.0, 10.0)
    X_test_final = np.clip(X_test_final, -10.0, 10.0)
    
    print(f"  Final feature matrix: {X_train_final.shape}")
    
    # Base Model Training
    print("\n🤖 Training Base Models")
    print("-" * 30)
    
    base_models = get_base_models()
    base_oof_predictions = []
    base_cv_scores = {}
    base_names = []
    
    for name, model in base_models:
        print(f"  Training {name}...")
        try:
            oof_preds, cv_score = generate_oof_predictions(model, X_train_final, y_train)
            base_oof_predictions.append(oof_preds)
            base_cv_scores[name] = cv_score
            base_names.append(name)
            print(f"    {name} CV SMAPE: {cv_score:.4f}%")
            gc.collect()
        except Exception as e:
            print(f"    ⚠️ {name} failed: {e}")
    
    if not base_oof_predictions:
        raise RuntimeError("No base models could be trained successfully")
    
    # Create base prediction matrix
    base_oof_matrix = np.column_stack(base_oof_predictions).astype(np.float32)
    base_oof_matrix = np.nan_to_num(base_oof_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  Base OOF matrix shape: {base_oof_matrix.shape}")
    
    # Stacker Model Selection
    print("\n🎯 Training Stacker Models")
    print("-" * 30)
    
    stacker_models = get_stacker_models()
    best_stacker = None
    best_stacker_name = None
    best_stacker_cv = float('inf')
    stacker_results = {}
    
    for name, model in stacker_models:
        print(f"  Evaluating {name}...")
        try:
            _, cv_score = generate_oof_predictions(model, base_oof_matrix, y_train)
            stacker_results[name] = cv_score
            print(f"    {name} CV SMAPE: {cv_score:.4f}%")
            
            if cv_score < best_stacker_cv:
                best_stacker_cv = cv_score
                best_stacker_name = name
                best_stacker = model
                
        except Exception as e:
            print(f"    ⚠️ {name} failed: {e}")
    
    if best_stacker is None:
        raise RuntimeError("No stacker model could be trained successfully")
    
    print(f"  🏆 Best stacker: {best_stacker_name} (CV SMAPE: {best_stacker_cv:.4f}%)")
    
    # Final Training and Prediction
    print("\n🎯 Final Training and Prediction")
    print("-" * 30)
    
    print("  Training base models on full dataset...")
    base_test_predictions = []
    base_train_predictions = []
    
    for i, (name, model) in enumerate([(n, m) for n, m in base_models if n in base_names]):
        print(f"    Training {name} on full data...")
        try:
            model.fit(X_train_final, y_train)
            train_pred = model.predict(X_train_final)
            test_pred = model.predict(X_test_final)
            
            base_train_predictions.append(train_pred)
            base_test_predictions.append(test_pred)
        except Exception as e:
            print(f"    ⚠️ {name} failed: {e}")
    
    # Create stacker input matrices
    X_train_stacker = np.column_stack(base_train_predictions).astype(np.float32)
    X_test_stacker = np.column_stack(base_test_predictions).astype(np.float32)
    
    X_train_stacker = np.nan_to_num(X_train_stacker, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_stacker = np.nan_to_num(X_test_stacker, nan=0.0, posinf=0.0, neginf=0.0)
    
    print("  Training final stacker...")
    best_stacker.fit(X_train_stacker, y_train)
    
    # Generate final predictions
    final_predictions = best_stacker.predict(X_test_stacker)
    final_predictions = np.clip(final_predictions, 0.01, None)  # Ensure positive prices
    
    # Save Results
    print("\n💾 Saving Results")
    print("-" * 30)
    
    # Create submission
    submission_df = pd.DataFrame({
        "sample_id": test_df["sample_id"],
        "price": final_predictions
    })
    submission_df.to_csv(submission_path, index=False)
    print(f"  Submission saved: {submission_path}")
    
    # Save model artifacts
    stacker_model_path = os.path.join(MODELS_DIR, "day2_stacker_model.pkl")
    with open(stacker_model_path, "wb") as f:
        pickle.dump(best_stacker, f)
    print(f"  Stacker model saved: {stacker_model_path}")
    
    # Save metadata
    metadata = {
        "base_models": base_names,
        "base_cv_scores": base_cv_scores,
        "stacker_model": best_stacker_name,
        "stacker_cv_score": best_stacker_cv,
        "stacker_results": stacker_results,
        "n_train_samples": int(len(train_df)),
        "n_test_samples": int(len(test_df)),
        "n_features": int(X_train_final.shape[1]),
        "feature_breakdown": {
            "text_svd": int(X_train_text_svd.shape[1]),
            "text_length": int(X_train_tlen.shape[1]),
            "structured": int(X_train_struct.shape[1]),
            "visual": int(X_train_visual.shape[1])
        }
    }
    
    metadata_path = os.path.join(MODELS_DIR, "stacked_ensemble_info_day2.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {metadata_path}")
    
    # Summary
    print("\n✅ Pipeline Complete!")
    print("=" * 50)
    print(f"🎯 Best Stacker: {best_stacker_name}")
    print(f"📊 CV SMAPE: {best_stacker_cv:.4f}%")
    print(f"📁 Submission: {submission_path}")
    print(f"🔧 Total Features: {X_train_final.shape[1]}")
    print(f"🤖 Base Models: {len(base_names)}")
    
    return metadata


if __name__ == "__main__":
    main()