import os
import json
import pickle
import string
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone

# Try LightGBM, fallback gracefully if not installed
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# -------------------- Utilities --------------------

def robust_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, engine="python")
        except Exception:
            try:
                return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", engine="python")
            except Exception:
                return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip", engine="python")


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = 1.0
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)


def _select_text_column(df: pd.DataFrame) -> pd.Series:
    # Prefer common text-like columns if present
    for col in ["title", "name", "description", "text", "product_description"]:
        if col in df.columns:
            s = df[col].astype(str)
            if s.str.len().mean() > 0.0:
                return s
    # Fallback: concatenate object columns
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not obj_cols:
        return pd.Series([""] * len(df), index=df.index)
    return df[obj_cols].fillna("").astype(str).agg(" ".join, axis=1)


def build_text_features(train_text: pd.Series, test_text: pd.Series) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, dict]:
    # Word-level TF-IDF (1-1 ngrams), aggressively limited features for memory safety
    try:
        word_vec = TfidfVectorizer(
            max_features=30_000,
            ngram_range=(1, 1),
            min_df=5,
            max_df=0.90,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
            dtype=np.float32,
        )
        Xw_train = word_vec.fit_transform(train_text)
        Xw_test = word_vec.transform(test_text)
    except Exception as e:
        print("⚠️ Word TF-IDF failed (", str(e), ") — retrying with smaller config...")
        word_vec = TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 1),
            min_df=7,
            max_df=0.95,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
            dtype=np.float32,
        )
        Xw_train = word_vec.fit_transform(train_text)
        Xw_test = word_vec.transform(test_text)

    # Char-level TF-IDF (3-char ngrams within words), with memory-safe fallbacks
    try:
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 3),
            min_df=7,
            max_df=0.90,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
            max_features=8_000,
            dtype=np.float32,
        )
        Xc_train = char_vec.fit_transform(train_text)
        Xc_test = char_vec.transform(test_text)
    except Exception as e:
        print("⚠️ Char TF-IDF failed (", str(e), ") — retrying with smaller config...")
        try:
            char_vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 3),
                min_df=10,
                max_df=0.95,
                sublinear_tf=True,
                lowercase=True,
                strip_accents="unicode",
                max_features=5_000,
                dtype=np.float32,
            )
            Xc_train = char_vec.fit_transform(train_text)
            Xc_test = char_vec.transform(test_text)
        except Exception as e2:
            print("⚠️ Char TF-IDF failed again (", str(e2), ") — proceeding without char features.")
            char_vec = None
            Xc_train = sparse.csr_matrix((Xw_train.shape[0], 0), dtype=np.float32)
            Xc_test = sparse.csr_matrix((Xw_test.shape[0], 0), dtype=np.float32)

    # Combine sparse matrices; keep as CSR to remain memory efficient
    X_train_text = sparse.hstack([Xw_train, Xc_train]).tocsr()
    X_test_text = sparse.hstack([Xw_test, Xc_test]).tocsr()

    meta = {
        "word_vec": word_vec,
        "char_vec": char_vec,
    }
    return X_train_text, X_test_text, meta


def load_visual_mapping() -> Optional[dict]:
    pkl_path = os.path.join(MODELS_DIR, "visual_feature_mapping.pkl")
    # Try pickle first
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print("⚠️ Failed to load visual_feature_mapping.pkl (", str(e), ") — falling back to CSV...")
    # Fallback: build mapping from CSV
    csv_path = os.path.join(DATA_DIR, "visual_features_train.csv")
    if os.path.exists(csv_path):
        df = robust_read_csv(csv_path)
        if "sample_id" in df.columns:
            # Prefer numeric feature columns, exclude sample_id if numeric
            feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "sample_id" in feat_cols:
                feat_cols.remove("sample_id")
            # If numeric detection fails, try columns starting with 'f_'
            if not feat_cols:
                feat_cols = [c for c in df.columns if c.startswith("f_")]
            vis_map = {}
            for _, row in df.iterrows():
                sid = row["sample_id"]
                try:
                    vec = row[feat_cols].astype(float).to_numpy()
                except Exception:
                    vec = row[feat_cols].to_numpy(dtype=float)
                vis_map[str(sid)] = vec
            return vis_map
        else:
            print("⚠️ visual_features_train.csv missing 'sample_id' column — cannot build visual map.")
    print("⚠️ No visual mapping available — proceeding without visual features.")
    return None


def load_visual_transformers():
    scaler_path = os.path.join(MODELS_DIR, "visual_scaler.pkl")
    pca_path = os.path.join(MODELS_DIR, "visual_pca.pkl")
    scaler = None
    pca = None
    # Try loading pickled scaler
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            print("⚠️ Failed to load visual_scaler.pkl (", str(e), ") — will attempt to refit from CSV.")
    # Try loading pickled PCA
    if os.path.exists(pca_path):
        try:
            with open(pca_path, "rb") as f:
                pca = pickle.load(f)
        except Exception as e:
            print("⚠️ Failed to load visual_pca.pkl (", str(e), ") — will attempt to refit from CSV.")
    # If either is missing/corrupted, try to refit from CSV
    if scaler is None or pca is None:
        csv_path = os.path.join(DATA_DIR, "visual_features_train.csv")
        if os.path.exists(csv_path):
            print("🔧 Fitting visual transformers from CSV...")
            df = robust_read_csv(csv_path)
            # Identify feature columns (prefer numeric), exclude sample_id if numeric
            feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "sample_id" in feat_cols:
                feat_cols.remove("sample_id")
            if not feat_cols:
                feat_cols = [c for c in df.columns if c.startswith("f_")]
            if feat_cols:
                Xv = df[feat_cols].to_numpy(dtype=np.float32)
                # Fit scaler
                if scaler is None:
                    scaler = StandardScaler(copy=True)
                    scaler.fit(Xv)
                    try:
                        with open(scaler_path, "wb") as f:
                            pickle.dump(scaler, f)
                    except Exception:
                        pass
                # Fit PCA
                if pca is None:
                    n_feats = Xv.shape[1]
                    n_comp = int(min(64, max(2, n_feats)))
                    pca = PCA(n_components=n_comp, random_state=42)
                    pca.fit(scaler.transform(Xv))
                    try:
                        with open(pca_path, "wb") as f:
                            pickle.dump(pca, f)
                    except Exception:
                        pass
            else:
                print("⚠️ Could not identify visual feature columns to fit transformers.")
        else:
            print("⚠️ visual_features_train.csv not found — proceeding without visual transformers.")
    return scaler, pca


def build_visual_matrix(sample_ids: pd.Series, vis_map: Optional[dict], scaler=None, pca=None) -> Tuple[np.ndarray, np.ndarray]:
    # Returns (X_visual, has_visual_indicator)
    if vis_map is None:
        return np.zeros((len(sample_ids), 1), dtype=np.float32), np.zeros(len(sample_ids), dtype=np.float32)
    visuals = []
    has_flag = []
    for sid in sample_ids:
        key = str(sid)
        v = vis_map.get(key)
        if v is None:
            visuals.append(None)
            has_flag.append(0.0)
        else:
            visuals.append(np.asarray(v, dtype=np.float32))
            has_flag.append(1.0)
    # Determine visual dim
    dim = None
    for v in visuals:
        if v is not None:
            dim = v.shape[0]
            break
    if dim is None:
        return np.zeros((len(sample_ids), 1), dtype=np.float32), np.zeros(len(sample_ids), dtype=np.float32)
    # Fill missing with zeros
    mat = np.zeros((len(sample_ids), dim), dtype=np.float32)
    for i, v in enumerate(visuals):
        if v is not None:
            mat[i, :dim] = v[:dim]
    # Apply scaler/pca if available
    if scaler is not None:
        mat = scaler.transform(mat)
    if pca is not None:
        mat = pca.transform(mat)
    has_arr = np.asarray(has_flag, dtype=np.float32).reshape(-1, 1)
    # Append indicator as an extra column
    mat = np.hstack([mat, has_arr])
    return mat, has_arr.ravel()


def svd_reduce(X_train: sparse.csr_matrix, X_test: sparse.csr_matrix, n_components: int = 200) -> Tuple[np.ndarray, np.ndarray, TruncatedSVD]:
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Xtr = svd.fit_transform(X_train).astype(np.float32)
    Xte = svd.transform(X_test).astype(np.float32)
    return Xtr, Xte, svd


def kfold_smape(model, X, y, folds: int = 5) -> Tuple[float, list]:
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X):
        Xtr = X[train_idx]
        Xval = X[val_idx]
        ytr = y[train_idx]
        yval = y[val_idx]
        model_fit = clone(model)
        model_fit.fit(Xtr, ytr)
        pred = model_fit.predict(Xval)
        scores.append(smape(yval, pred))
    return float(np.mean(scores)), scores

# Helper: generate out-of-fold predictions for a given model
def generate_oof_predictions(model, X: np.ndarray, y: np.ndarray, folds: int = 5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, val_idx in kf.split(X):
        m = clone(model)
        m.fit(X[train_idx], y[train_idx])
        oof[val_idx] = m.predict(X[val_idx])
    cv = smape(y, oof)
    return oof, cv


def _select_text_column(df: pd.DataFrame) -> pd.Series:
    # Prefer common text-like columns if present
    for col in ["title", "name", "description", "text", "product_description"]:
        if col in df.columns:
            s = df[col].astype(str)
            if s.str.len().mean() > 0.0:
                return s
    # Fallback: concatenate object columns
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not obj_cols:
        return pd.Series([""] * len(df), index=df.index)
    return df[obj_cols].fillna("").astype(str).agg(" ".join, axis=1)


def build_text_features(train_text: pd.Series, test_text: pd.Series) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, dict]:
    # Word-level TF-IDF (strictly 1-gram), capped features for memory safety
    try:
        word_vec = TfidfVectorizer(
            max_features=30_000,
            ngram_range=(1, 1),
            min_df=5,
            max_df=0.90,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
            dtype=np.float32,
        )
        Xw_train = word_vec.fit_transform(train_text)
        Xw_test = word_vec.transform(test_text)
    except Exception as e:
        print("⚠️ Word TF-IDF failed (", str(e), ") — retrying with smaller config...")
        word_vec = TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 1),
            min_df=7,
            max_df=0.95,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
            dtype=np.float32,
        )
        Xw_train = word_vec.fit_transform(train_text)
        Xw_test = word_vec.transform(test_text)

    # Char-level TF-IDF (wb, 3-gram), with memory-safe fallbacks
    try:
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 3),
            min_df=7,
            max_df=0.90,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
            max_features=8_000,
            dtype=np.float32,
        )
        Xc_train = char_vec.fit_transform(train_text)
        Xc_test = char_vec.transform(test_text)
    except Exception as e:
        print("⚠️ Char TF-IDF failed (", str(e), ") — retrying with smaller config...")
        try:
            char_vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 3),
                min_df=10,
                max_df=0.95,
                sublinear_tf=True,
                lowercase=True,
                strip_accents="unicode",
                max_features=5_000,
                dtype=np.float32,
            )
            Xc_train = char_vec.fit_transform(train_text)
            Xc_test = char_vec.transform(test_text)
        except Exception as e2:
            print("⚠️ Char TF-IDF failed again (", str(e2), ") — proceeding without char features.")
            char_vec = None
            Xc_train = sparse.csr_matrix((Xw_train.shape[0], 0), dtype=np.float32)
            Xc_test = sparse.csr_matrix((Xw_test.shape[0], 0), dtype=np.float32)

    X_train_text = sparse.hstack([Xw_train, Xc_train]).tocsr()
    X_test_text = sparse.hstack([Xw_test, Xc_test]).tocsr()

    meta = {
        "word_vec": word_vec,
        "char_vec": char_vec,
    }
    return X_train_text, X_test_text, meta


def load_visual_mapping() -> Optional[dict]:
    pkl_path = os.path.join(MODELS_DIR, "visual_feature_mapping.pkl")
    # Try pickle first
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print("⚠️ Failed to load visual_feature_mapping.pkl (", str(e), ") — falling back to CSV...")
    # Fallback: build mapping from CSV
    csv_path = os.path.join(DATA_DIR, "visual_features_train.csv")
    if os.path.exists(csv_path):
        df = robust_read_csv(csv_path)
        if "sample_id" in df.columns:
            # Prefer numeric feature columns, exclude sample_id if numeric
            feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "sample_id" in feat_cols:
                feat_cols.remove("sample_id")
            # If numeric detection fails, try columns starting with 'f_'
            if not feat_cols:
                feat_cols = [c for c in df.columns if c.startswith("f_")]
            vis_map = {}
            for _, row in df.iterrows():
                sid = row["sample_id"]
                try:
                    vec = row[feat_cols].astype(float).to_numpy()
                except Exception:
                    vec = row[feat_cols].to_numpy(dtype=float)
                vis_map[str(sid)] = vec
            return vis_map
        else:
            print("⚠️ visual_features_train.csv missing 'sample_id' column — cannot build visual map.")
    print("⚠️ No visual mapping available — proceeding without visual features.")
    return None


def load_visual_transformers():
    scaler_path = os.path.join(MODELS_DIR, "visual_scaler.pkl")
    pca_path = os.path.join(MODELS_DIR, "visual_pca.pkl")
    scaler = None
    pca = None
    # Try loading pickled scaler
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            print("⚠️ Failed to load visual_scaler.pkl (", str(e), ") — will attempt to refit from CSV.")
    # Try loading pickled PCA
    if os.path.exists(pca_path):
        try:
            with open(pca_path, "rb") as f:
                pca = pickle.load(f)
        except Exception as e:
            print("⚠️ Failed to load visual_pca.pkl (", str(e), ") — will attempt to refit from CSV.")
    # If either is missing/corrupted, try to refit from CSV
    if scaler is None or pca is None:
        csv_path = os.path.join(DATA_DIR, "visual_features_train.csv")
        if os.path.exists(csv_path):
            print("🔧 Fitting visual transformers from CSV...")
            df = robust_read_csv(csv_path)
            # Identify feature columns (prefer numeric), exclude sample_id if numeric
            feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "sample_id" in feat_cols:
                feat_cols.remove("sample_id")
            if not feat_cols:
                feat_cols = [c for c in df.columns if c.startswith("f_")]
            if feat_cols:
                Xv = df[feat_cols].to_numpy(dtype=np.float32)
                # Fit scaler
                if scaler is None:
                    scaler = StandardScaler(copy=True)
                    scaler.fit(Xv)
                    try:
                        with open(scaler_path, "wb") as f:
                            pickle.dump(scaler, f)
                    except Exception:
                        pass
                # Fit PCA
                if pca is None:
                    n_feats = Xv.shape[1]
                    n_comp = int(min(64, max(2, n_feats)))
                    pca = PCA(n_components=n_comp, random_state=42)
                    pca.fit(scaler.transform(Xv))
                    try:
                        with open(pca_path, "wb") as f:
                            pickle.dump(pca, f)
                    except Exception:
                        pass
            else:
                print("⚠️ Could not identify visual feature columns to fit transformers.")
        else:
            print("⚠️ visual_features_train.csv not found — proceeding without visual transformers.")
    return scaler, pca


def build_visual_matrix(sample_ids: pd.Series, vis_map: Optional[dict], scaler=None, pca=None) -> Tuple[np.ndarray, np.ndarray]:
    # Returns (X_visual, has_visual_indicator)
    if vis_map is None:
        return np.zeros((len(sample_ids), 1), dtype=np.float32), np.zeros(len(sample_ids), dtype=np.float32)
    visuals = []
    has_flag = []
    for sid in sample_ids:
        key = str(sid)
        v = vis_map.get(key)
        if v is None:
            visuals.append(None)
            has_flag.append(0.0)
        else:
            visuals.append(np.asarray(v, dtype=np.float32))
            has_flag.append(1.0)
    # Determine visual dim
    dim = None
    for v in visuals:
        if v is not None:
            dim = v.shape[0]
            break
    if dim is None:
        return np.zeros((len(sample_ids), 1), dtype=np.float32), np.zeros(len(sample_ids), dtype=np.float32)
    # Fill missing with zeros
    mat = np.zeros((len(sample_ids), dim), dtype=np.float32)
    for i, v in enumerate(visuals):
        if v is not None:
            mat[i, :dim] = v[:dim]
    # Apply scaler/pca if available
    if scaler is not None:
        mat = scaler.transform(mat)
    if pca is not None:
        mat = pca.transform(mat)
    has_arr = np.asarray(has_flag, dtype=np.float32).reshape(-1, 1)
    # Append indicator as an extra column
    mat = np.hstack([mat, has_arr])
    return mat, has_arr.ravel()


def svd_reduce(X_train: sparse.csr_matrix, X_test: sparse.csr_matrix, n_components: int = 200) -> Tuple[np.ndarray, np.ndarray, TruncatedSVD]:
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Xtr = svd.fit_transform(X_train).astype(np.float32)
    Xte = svd.transform(X_test).astype(np.float32)
    return Xtr, Xte, svd


def kfold_smape(model, X, y, folds: int = 5) -> Tuple[float, list]:
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X):
        Xtr = X[train_idx]
        Xval = X[val_idx]
        ytr = y[train_idx]
        yval = y[val_idx]
        model_fit = clone(model)
        model_fit.fit(Xtr, ytr)
        pred = model_fit.predict(Xval)
        scores.append(smape(yval, pred))
    return float(np.mean(scores)), scores


def _select_text_column(df: pd.DataFrame) -> pd.Series:
    # Prefer common text-like columns if present
    for col in ["title", "name", "description", "text", "product_description"]:
        if col in df.columns:
            s = df[col].astype(str)
            if s.str.len().mean() > 0.0:
                return s
    # Fallback: concatenate object columns
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not obj_cols:
        return pd.Series([""] * len(df), index=df.index)
    return df[obj_cols].fillna("").astype(str).agg(" ".join, axis=1)


def build_text_features(train_text: pd.Series, test_text: pd.Series) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, dict]:
    # Word-level TF-IDF (1-2 ngrams), memory-friendly dtype
    word_vec = TfidfVectorizer(
        max_features=60_000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    Xw_train = word_vec.fit_transform(train_text)
    Xw_test = word_vec.transform(test_text)

    # Char-level TF-IDF (3-4 char ngrams within words), with memory-safe fallbacks
    try:
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=5,
            max_df=0.90,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
            max_features=20_000,
            dtype=np.float32,
        )
        Xc_train = char_vec.fit_transform(train_text)
        Xc_test = char_vec.transform(test_text)
    except Exception as e:
        print("⚠️ Char TF-IDF failed (", str(e), ") — retrying with smaller config...")
        try:
            char_vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 4),
                min_df=5,
                max_df=0.90,
                sublinear_tf=True,
                lowercase=True,
                strip_accents="unicode",
                max_features=10_000,
                dtype=np.float32,
            )
            Xc_train = char_vec.fit_transform(train_text)
            Xc_test = char_vec.transform(test_text)
        except Exception as e2:
            print("⚠️ Char TF-IDF failed again (", str(e2), ") — proceeding without char features.")
            char_vec = None
            Xc_train = sparse.csr_matrix((Xw_train.shape[0], 0), dtype=np.float32)
            Xc_test = sparse.csr_matrix((Xw_test.shape[0], 0), dtype=np.float32)

    X_train_text = sparse.hstack([Xw_train, Xc_train]).tocsr()
    X_test_text = sparse.hstack([Xw_test, Xc_test]).tocsr()

    meta = {
        "word_vec": word_vec,
        "char_vec": char_vec,
    }
    return X_train_text, X_test_text, meta


def load_visual_mapping() -> Optional[dict]:
    pkl_path = os.path.join(MODELS_DIR, "visual_feature_mapping.pkl")
    # Try pickle first
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print("⚠️ Failed to load visual_feature_mapping.pkl (", str(e), ") — falling back to CSV...")
    # Fallback: build mapping from CSV
    csv_path = os.path.join(DATA_DIR, "visual_features_train.csv")
    if os.path.exists(csv_path):
        df = robust_read_csv(csv_path)
        if "sample_id" in df.columns:
            # Prefer numeric feature columns, exclude sample_id if numeric
            feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "sample_id" in feat_cols:
                feat_cols.remove("sample_id")
            # If numeric detection fails, try columns starting with 'f_'
            if not feat_cols:
                feat_cols = [c for c in df.columns if c.startswith("f_")]
            vis_map = {}
            for _, row in df.iterrows():
                sid = row["sample_id"]
                try:
                    vec = row[feat_cols].astype(float).to_numpy()
                except Exception:
                    vec = row[feat_cols].to_numpy(dtype=float)
                vis_map[str(sid)] = vec
            return vis_map
        else:
            print("⚠️ visual_features_train.csv missing 'sample_id' column — cannot build visual map.")
    print("⚠️ No visual mapping available — proceeding without visual features.")
    return None


def load_visual_transformers():
    scaler_path = os.path.join(MODELS_DIR, "visual_scaler.pkl")
    pca_path = os.path.join(MODELS_DIR, "visual_pca.pkl")
    scaler = None
    pca = None
    # Try loading pickled scaler
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            print("⚠️ Failed to load visual_scaler.pkl (", str(e), ") — will attempt to refit from CSV.")
    # Try loading pickled PCA
    if os.path.exists(pca_path):
        try:
            with open(pca_path, "rb") as f:
                pca = pickle.load(f)
        except Exception as e:
            print("⚠️ Failed to load visual_pca.pkl (", str(e), ") — will attempt to refit from CSV.")
    # If either is missing/corrupted, try to refit from CSV
    if scaler is None or pca is None:
        csv_path = os.path.join(DATA_DIR, "visual_features_train.csv")
        if os.path.exists(csv_path):
            print("🔧 Fitting visual transformers from CSV...")
            df = robust_read_csv(csv_path)
            # Identify feature columns (prefer numeric), exclude sample_id if numeric
            feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "sample_id" in feat_cols:
                feat_cols.remove("sample_id")
            if not feat_cols:
                feat_cols = [c for c in df.columns if c.startswith("f_")]
            if feat_cols:
                Xv = df[feat_cols].to_numpy(dtype=np.float32)
                # Fit scaler
                if scaler is None:
                    scaler = StandardScaler(copy=True)
                    scaler.fit(Xv)
                    try:
                        with open(scaler_path, "wb") as f:
                            pickle.dump(scaler, f)
                    except Exception:
                        pass
                # Fit PCA
                if pca is None:
                    n_feats = Xv.shape[1]
                    n_comp = int(min(64, max(2, n_feats)))
                    pca = PCA(n_components=n_comp, random_state=42)
                    pca.fit(scaler.transform(Xv))
                    try:
                        with open(pca_path, "wb") as f:
                            pickle.dump(pca, f)
                    except Exception:
                        pass
            else:
                print("⚠️ Could not identify visual feature columns to fit transformers.")
        else:
            print("⚠️ visual_features_train.csv not found — proceeding without visual transformers.")
    return scaler, pca


def build_visual_matrix(sample_ids: pd.Series, vis_map: Optional[dict], scaler=None, pca=None) -> Tuple[np.ndarray, np.ndarray]:
    # Returns (X_visual, has_visual_indicator)
    if vis_map is None:
        return np.zeros((len(sample_ids), 1), dtype=np.float32), np.zeros(len(sample_ids), dtype=np.float32)
    visuals = []
    has_flag = []
    for sid in sample_ids:
        key = str(sid)
        v = vis_map.get(key)
        if v is None:
            visuals.append(None)
            has_flag.append(0.0)
        else:
            visuals.append(np.asarray(v, dtype=np.float32))
            has_flag.append(1.0)
    # Determine visual dim
    dim = None
    for v in visuals:
        if v is not None:
            dim = v.shape[0]
            break
    if dim is None:
        return np.zeros((len(sample_ids), 1), dtype=np.float32), np.zeros(len(sample_ids), dtype=np.float32)
    # Fill missing with zeros
    mat = np.zeros((len(sample_ids), dim), dtype=np.float32)
    for i, v in enumerate(visuals):
        if v is not None:
            mat[i, :dim] = v[:dim]
    # Apply scaler/pca if available
    if scaler is not None:
        mat = scaler.transform(mat)
    if pca is not None:
        mat = pca.transform(mat)
    has_arr = np.asarray(has_flag, dtype=np.float32).reshape(-1, 1)
    # Append indicator as an extra column
    mat = np.hstack([mat, has_arr])
    return mat, has_arr.ravel()


def svd_reduce(X_train: sparse.csr_matrix, X_test: sparse.csr_matrix, n_components: int = 200) -> Tuple[np.ndarray, np.ndarray, TruncatedSVD]:
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Xtr = svd.fit_transform(X_train).astype(np.float32)
    Xte = svd.transform(X_test).astype(np.float32)
    return Xtr, Xte, svd


def kfold_smape(model, X, y, folds: int = 5) -> Tuple[float, list]:
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X):
        Xtr = X[train_idx]
        Xval = X[val_idx]
        ytr = y[train_idx]
        yval = y[val_idx]
        model_fit = clone(model)
        model_fit.fit(Xtr, ytr)
        pred = model_fit.predict(Xval)
        scores.append(smape(yval, pred))
    return float(np.mean(scores)), scores


def _select_text_column(df: pd.DataFrame) -> pd.Series:
    # Prefer common text-like columns if present
    for col in ["title", "name", "description", "text", "product_description"]:
        if col in df.columns:
            s = df[col].astype(str)
            if s.str.len().mean() > 0.0:
                return s
    # Fallback: concatenate object columns
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not obj_cols:
        return pd.Series([""] * len(df), index=df.index)
    return df[obj_cols].fillna("").astype(str).agg(" ".join, axis=1)


def build_text_features(train_text: pd.Series, test_text: pd.Series) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, dict]:
    # Word-level TF-IDF (1-2 ngrams), memory-friendly dtype
    word_vec = TfidfVectorizer(
        max_features=60_000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    Xw_train = word_vec.fit_transform(train_text)
    Xw_test = word_vec.transform(test_text)

    # Char-level TF-IDF (3-4 char ngrams within words), with memory-safe fallbacks
    try:
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=5,
            max_df=0.90,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
            max_features=20_000,
            dtype=np.float32,
        )
        Xc_train = char_vec.fit_transform(train_text)
        Xc_test = char_vec.transform(test_text)
    except Exception as e:
        print("⚠️ Char TF-IDF failed (", str(e), ") — retrying with smaller config...")
        try:
            char_vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 4),
                min_df=5,
                max_df=0.90,
                sublinear_tf=True,
                lowercase=True,
                strip_accents="unicode",
                max_features=10_000,
                dtype=np.float32,
            )
            Xc_train = char_vec.fit_transform(train_text)
            Xc_test = char_vec.transform(test_text)
        except Exception as e2:
            print("⚠️ Char TF-IDF failed again (", str(e2), ") — proceeding without char features.")
            char_vec = None
            Xc_train = sparse.csr_matrix((Xw_train.shape[0], 0), dtype=np.float32)
            Xc_test = sparse.csr_matrix((Xw_test.shape[0], 0), dtype=np.float32)

    X_train_text = sparse.hstack([Xw_train, Xc_train]).tocsr()
    X_test_text = sparse.hstack([Xw_test, Xc_test]).tocsr()

    meta = {
        "word_vec": word_vec,
        "char_vec": char_vec,
    }
    return X_train_text, X_test_text, meta


def load_visual_mapping() -> Optional[dict]:
    pkl_path = os.path.join(MODELS_DIR, "visual_feature_mapping.pkl")
    # Try pickle first
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print("⚠️ Failed to load visual_feature_mapping.pkl (", str(e), ") — falling back to CSV...")
    # Fallback: build mapping from CSV
    csv_path = os.path.join(DATA_DIR, "visual_features_train.csv")
    if os.path.exists(csv_path):
        df = robust_read_csv(csv_path)
        if "sample_id" in df.columns:
            # Prefer numeric feature columns, exclude sample_id if numeric
            feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "sample_id" in feat_cols:
                feat_cols.remove("sample_id")
            # If numeric detection fails, try columns starting with 'f_'
            if not feat_cols:
                feat_cols = [c for c in df.columns if c.startswith("f_")]
            vis_map = {}
            for _, row in df.iterrows():
                sid = row["sample_id"]
                try:
                    vec = row[feat_cols].astype(float).to_numpy()
                except Exception:
                    vec = row[feat_cols].to_numpy(dtype=float)
                vis_map[str(sid)] = vec
            return vis_map
        else:
            print("⚠️ visual_features_train.csv missing 'sample_id' column — cannot build visual map.")
    print("⚠️ No visual mapping available — proceeding without visual features.")
    return None


def load_visual_transformers():
    scaler_path = os.path.join(MODELS_DIR, "visual_scaler.pkl")
    pca_path = os.path.join(MODELS_DIR, "visual_pca.pkl")
    scaler = None
    pca = None
    # Try loading pickled scaler
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            print("⚠️ Failed to load visual_scaler.pkl (", str(e), ") — will attempt to refit from CSV.")
    # Try loading pickled PCA
    if os.path.exists(pca_path):
        try:
            with open(pca_path, "rb") as f:
                pca = pickle.load(f)
        except Exception as e:
            print("⚠️ Failed to load visual_pca.pkl (", str(e), ") — will attempt to refit from CSV.")
    # If either is missing/corrupted, try to refit from CSV
    if scaler is None or pca is None:
        csv_path = os.path.join(DATA_DIR, "visual_features_train.csv")
        if os.path.exists(csv_path):
            print("🔧 Fitting visual transformers from CSV...")
            df = robust_read_csv(csv_path)
            # Identify feature columns (prefer numeric), exclude sample_id if numeric
            feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "sample_id" in feat_cols:
                feat_cols.remove("sample_id")
            if not feat_cols:
                feat_cols = [c for c in df.columns if c.startswith("f_")]
            if feat_cols:
                Xv = df[feat_cols].to_numpy(dtype=np.float32)
                # Fit scaler
                if scaler is None:
                    scaler = StandardScaler(copy=True)
                    scaler.fit(Xv)
                    try:
                        with open(scaler_path, "wb") as f:
                            pickle.dump(scaler, f)
                    except Exception:
                        pass
                # Fit PCA
                if pca is None:
                    n_feats = Xv.shape[1]
                    n_comp = int(min(64, max(2, n_feats)))
                    pca = PCA(n_components=n_comp, random_state=42)
                    pca.fit(scaler.transform(Xv))
                    try:
                        with open(pca_path, "wb") as f:
                            pickle.dump(pca, f)
                    except Exception:
                        pass
            else:
                print("⚠️ Could not identify visual feature columns to fit transformers.")
        else:
            print("⚠️ visual_features_train.csv not found — proceeding without visual transformers.")
    return scaler, pca


def build_visual_matrix(sample_ids: pd.Series, vis_map: Optional[dict], scaler=None, pca=None) -> Tuple[np.ndarray, np.ndarray]:
    # Returns (X_visual, has_visual_indicator)
    if vis_map is None:
        return np.zeros((len(sample_ids), 1), dtype=np.float32), np.zeros(len(sample_ids), dtype=np.float32)
    visuals = []
    has_flag = []
    for sid in sample_ids:
        key = str(sid)
        v = vis_map.get(key)
        if v is None:
            visuals.append(None)
            has_flag.append(0.0)
        else:
            visuals.append(np.asarray(v, dtype=np.float32))
            has_flag.append(1.0)
    # Determine visual dim
    dim = None
    for v in visuals:
        if v is not None:
            dim = v.shape[0]
            break
    if dim is None:
        return np.zeros((len(sample_ids), 1), dtype=np.float32), np.zeros(len(sample_ids), dtype=np.float32)
    # Fill missing with zeros
    mat = np.zeros((len(sample_ids), dim), dtype=np.float32)
    for i, v in enumerate(visuals):
        if v is not None:
            mat[i, :dim] = v[:dim]
    # Apply scaler/pca if available
    if scaler is not None:
        mat = scaler.transform(mat)
    if pca is not None:
        mat = pca.transform(mat)
    has_arr = np.asarray(has_flag, dtype=np.float32).reshape(-1, 1)
    # Append indicator as an extra column
    mat = np.hstack([mat, has_arr])
    return mat, has_arr.ravel()


def svd_reduce(X_train: sparse.csr_matrix, X_test: sparse.csr_matrix, n_components: int = 200) -> Tuple[np.ndarray, np.ndarray, TruncatedSVD]:
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Xtr = svd.fit_transform(X_train).astype(np.float32)
    Xte = svd.transform(X_test).astype(np.float32)
    return Xtr, Xte, svd


def kfold_smape(model, X, y, folds: int = 5) -> Tuple[float, list]:
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X):
        Xtr = X[train_idx]
        Xval = X[val_idx]
        ytr = y[train_idx]
        yval = y[val_idx]
        model_fit = clone(model)
        model_fit.fit(Xtr, ytr)
        pred = model_fit.predict(Xval)
        scores.append(smape(yval, pred))
    return float(np.mean(scores)), scores


def _select_text_column(df: pd.DataFrame) -> pd.Series:
    # Prefer common text-like columns if present
    for col in ["title", "name", "description", "text", "product_description"]:
        if col in df.columns:
            s = df[col].astype(str)
            if s.str.len().mean() > 0.0:
                return s
    # Fallback: concatenate object columns
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not obj_cols:
        return pd.Series([""] * len(df), index=df.index)
    return df[obj_cols].fillna("").astype(str).agg(" ".join, axis=1)


def build_text_features(train_text: pd.Series, test_text: pd.Series) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, dict]:
    # Word-level TF-IDF (1-2 ngrams), memory-friendly dtype
    word_vec = TfidfVectorizer(
        max_features=60_000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    Xw_train = word_vec.fit_transform(train_text)
    Xw_test = word_vec.transform(test_text)

    # Char-level TF-IDF (3-4 char ngrams within words), with memory-safe fallbacks
    try:
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=5,
            max_df=0.90,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
            max_features=20_000,
            dtype=np.float32,
        )
        Xc_train = char_vec.fit_transform(train_text)
        Xc_test = char_vec.transform(test_text)
    except Exception as e:
        print("⚠️ Char TF-IDF failed (", str(e), ") — retrying with smaller config...")
        try:
            char_vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 4),
                min_df=5,
                max_df=0.90,
                sublinear_tf=True,
                lowercase=True,
                strip_accents="unicode",
                max_features=10_000,
                dtype=np.float32,
            )
            Xc_train = char_vec.fit_transform(train_text)
            Xc_test = char_vec.transform(test_text)
        except Exception as e2:
            print("⚠️ Char TF-IDF failed again (", str(e2), ") — proceeding without char features.")
            char_vec = None
            Xc_train = sparse.csr_matrix((Xw_train.shape[0], 0), dtype=np.float32)
            Xc_test = sparse.csr_matrix((Xw_test.shape[0], 0), dtype=np.float32)

    X_train_text = sparse.hstack([Xw_train, Xc_train]).tocsr()
    X_test_text = sparse.hstack([Xw_test, Xc_test]).tocsr()

    meta = {
        "word_vec": word_vec,
        "char_vec": char_vec,
    }
    return X_train_text, X_test_text, meta


def load_visual_mapping() -> Optional[dict]:
    pkl_path = os.path.join(MODELS_DIR, "visual_feature_mapping.pkl")
    # Try pickle first
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print("⚠️ Failed to load visual_feature_mapping.pkl (", str(e), ") — falling back to CSV...")
    # Fallback: build mapping from CSV
    csv_path = os.path.join(DATA_DIR, "visual_features_train.csv")
    if os.path.exists(csv_path):
        df = robust_read_csv(csv_path)
        if "sample_id" in df.columns:
            # Prefer numeric feature columns, exclude sample_id if numeric
            feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "sample_id" in feat_cols:
                feat_cols.remove("sample_id")
            # If numeric detection fails, try columns starting with 'f_'
            if not feat_cols:
                feat_cols = [c for c in df.columns if c.startswith("f_")]
            vis_map = {}
            for _, row in df.iterrows():
                sid = row["sample_id"]
                try:
                    vec = row[feat_cols].astype(float).to_numpy()
                except Exception:
                    vec = row[feat_cols].to_numpy(dtype=float)
                vis_map[str(sid)] = vec
            return vis_map
        else:
            print("⚠️ visual_features_train.csv missing 'sample_id' column — cannot build visual map.")
    print("⚠️ No visual mapping available — proceeding without visual features.")
    return None


def load_visual_transformers():
    scaler_path = os.path.join(MODELS_DIR, "visual_scaler.pkl")
    pca_path = os.path.join(MODELS_DIR, "visual_pca.pkl")
    scaler = None
    pca = None
    # Try loading pickled scaler
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            print("⚠️ Failed to load visual_scaler.pkl (", str(e), ") — will attempt to refit from CSV.")
    # Try loading pickled PCA
    if os.path.exists(pca_path):
        try:
            with open(pca_path, "rb") as f:
                pca = pickle.load(f)
        except Exception as e:
            print("⚠️ Failed to load visual_pca.pkl (", str(e), ") — will attempt to refit from CSV.")
    # If either is missing/corrupted, try to refit from CSV
    if scaler is None or pca is None:
        csv_path = os.path.join(DATA_DIR, "visual_features_train.csv")
        if os.path.exists(csv_path):
            print("🔧 Fitting visual transformers from CSV...")
            df = robust_read_csv(csv_path)
            # Identify feature columns (prefer numeric), exclude sample_id if numeric
            feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "sample_id" in feat_cols:
                feat_cols.remove("sample_id")
            if not feat_cols:
                feat_cols = [c for c in df.columns if c.startswith("f_")]
            if feat_cols:
                Xv = df[feat_cols].to_numpy(dtype=np.float32)
                # Fit scaler
                if scaler is None:
                    scaler = StandardScaler(copy=True)
                    scaler.fit(Xv)
                    try:
                        with open(scaler_path, "wb") as f:
                            pickle.dump(scaler, f)
                    except Exception:
                        pass
                # Fit PCA
                if pca is None:
                    n_feats = Xv.shape[1]
                    n_comp = int(min(64, max(2, n_feats)))
                    pca = PCA(n_components=n_comp, random_state=42)
                    pca.fit(scaler.transform(Xv))
                    try:
                        with open(pca_path, "wb") as f:
                            pickle.dump(pca, f)
                    except Exception:
                        pass
            else:
                print("⚠️ Could not identify visual feature columns to fit transformers.")
        else:
            print("⚠️ visual_features_train.csv not found — proceeding without visual transformers.")
    return scaler, pca


def build_visual_matrix(sample_ids: pd.Series, vis_map: Optional[dict], scaler=None, pca=None) -> Tuple[np.ndarray, np.ndarray]:
    # Returns (X_visual, has_visual_indicator)
    if vis_map is None:
        return np.zeros((len(sample_ids), 1), dtype=np.float32), np.zeros(len(sample_ids), dtype=np.float32)
    visuals = []
    has_flag = []
    for sid in sample_ids:
        key = str(sid)
        v = vis_map.get(key)
        if v is None:
            visuals.append(None)
            has_flag.append(0.0)
        else:
            visuals.append(np.asarray(v, dtype=np.float32))
            has_flag.append(1.0)
    # Determine visual dim
    dim = None
    for v in visuals:
        if v is not None:
            dim = v.shape[0]
            break
    if dim is None:
        return np.zeros((len(sample_ids), 1), dtype=np.float32), np.zeros(len(sample_ids), dtype=np.float32)
    # Fill missing with zeros
    mat = np.zeros((len(sample_ids), dim), dtype=np.float32)
    for i, v in enumerate(visuals):
        if v is not None:
            mat[i, :dim] = v[:dim]
    # Apply scaler/pca if available
    if scaler is not None:
        mat = scaler.transform(mat)
    if pca is not None:
        mat = pca.transform(mat)
    has_arr = np.asarray(has_flag, dtype=np.float32).reshape(-1, 1)
    # Append indicator as an extra column
    mat = np.hstack([mat, has_arr])
    return mat, has_arr.ravel()


def svd_reduce(X_train: sparse.csr_matrix, X_test: sparse.csr_matrix, n_components: int = 200) -> Tuple[np.ndarray, np.ndarray, TruncatedSVD]:
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Xtr = svd.fit_transform(X_train).astype(np.float32)
    Xte = svd.transform(X_test).astype(np.float32)
    return Xtr, Xte, svd


def kfold_smape(model, X, y, folds: int = 5) -> Tuple[float, list]:
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X):
        Xtr = X[train_idx]
        Xval = X[val_idx]
        ytr = y[train_idx]
        yval = y[val_idx]
        model_fit = clone(model)
        model_fit.fit(Xtr, ytr)
        pred = model_fit.predict(Xval)
        scores.append(smape(yval, pred))
    return float(np.mean(scores)), scores


def _select_text_column(df: pd.DataFrame) -> pd.Series:
    # Prefer common text-like columns if present
    for col in ["title", "name", "description", "text", "product_description"]:
        if col in df.columns:
            s = df[col].astype(str)
            if s.str.len().mean() > 0.0:
                return s
    # Fallback: concatenate object columns
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not obj_cols:
        return pd.Series([""] * len(df), index=df.index)
    return df[obj_cols].fillna("").astype(str).agg(" ".join, axis=1)


def build_text_features(train_text: pd.Series, test_text: pd.Series) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, dict]:
    # Word-level TF-IDF (1-2 ngrams), memory-friendly dtype
    word_vec = TfidfVectorizer(
        max_features=60_000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    Xw_train = word_vec.fit_transform(train_text)
    Xw_test = word_vec.transform(test_text)

    # Char-level TF-IDF (3-4 char ngrams within words), with memory-safe fallbacks
    try:
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=5,
            max_df=0.90,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
            max_features=20_000,
            dtype=np.float32,
        )
        Xc_train = char_vec.fit_transform(train_text)
        Xc_test = char_vec.transform(test_text)
    except Exception as e:
        print("⚠️ Char TF-IDF failed (", str(e), ") — retrying with smaller config...")
        try:
            char_vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 4),
                min_df=5,
                max_df=0.90,
                sublinear_tf=True,
                lowercase=True,
                strip_accents="unicode",
                max_features=10_000,
                dtype=np.float32,
            )
            Xc_train = char_vec.fit_transform(train_text)
            Xc_test = char_vec.transform(test_text)
        except Exception as e2:
            print("⚠️ Char TF-IDF failed again (", str(e2), ") — proceeding without char features.")
            char_vec = None
            Xc_train = sparse.csr_matrix((Xw_train.shape[0], 0), dtype=np.float32)
            Xc_test = sparse.csr_matrix((Xw_test.shape[0], 0), dtype=np.float32)

    X_train_text = sparse.hstack([Xw_train, Xc_train]).tocsr()
    X_test_text = sparse.hstack([Xw_test, Xc_test]).tocsr()

    meta = {
        "word_vec": word_vec,
        "char_vec": char_vec,
    }
    return X_train_text, X_test_text, meta


def load_visual_mapping() -> Optional[dict]:
    pkl_path = os.path.join(MODELS_DIR, "visual_feature_mapping.pkl")
    # Try pickle first
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print("⚠️ Failed to load visual_feature_mapping.pkl (", str(e), ") — falling back to CSV...")
    # Fallback: build mapping from CSV
    csv_path = os.path.join(DATA_DIR, "visual_features_train.csv")
    if os.path.exists(csv_path):
        df = robust_read_csv(csv_path)
        if "sample_id" in df.columns:
            # Prefer numeric feature columns, exclude sample_id if numeric
            feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "sample_id" in feat_cols:
                feat_cols.remove("sample_id")
            # If numeric detection fails, try columns starting with 'f_'
            if not feat_cols:
                feat_cols = [c for c in df.columns if c.startswith("f_")]
            vis_map = {}
            for _, row in df.iterrows():
                sid = row["sample_id"]
                try:
                    vec = row[feat_cols].astype(float).to_numpy()
                except Exception:
                    vec = row[feat_cols].to_numpy(dtype=float)
                vis_map[str(sid)] = vec
            return vis_map
        else:
            print("⚠️ visual_features_train.csv missing 'sample_id' column — cannot build visual map.")
    print("⚠️ No visual mapping available — proceeding without visual features.")
    return None


def load_visual_transformers():
    scaler_path = os.path.join(MODELS_DIR, "visual_scaler.pkl")
    pca_path = os.path.join(MODELS_DIR, "visual_pca.pkl")
    scaler = None
    pca = None
    # Try loading pickled scaler
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            print("⚠️ Failed to load visual_scaler.pkl (", str(e), ") — will attempt to refit from CSV.")
    # Try loading pickled PCA
    if os.path.exists(pca_path):
        try:
            with open(pca_path, "rb") as f:
                pca = pickle.load(f)
        except Exception as e:
            print("⚠️ Failed to load visual_pca.pkl (", str(e), ") — will attempt to refit from CSV.")
    # If either is missing/corrupted, try to refit from CSV
    if scaler is None or pca is None:
        csv_path = os.path.join(DATA_DIR, "visual_features_train.csv")
        if os.path.exists(csv_path):
            print("🔧 Fitting visual transformers from CSV...")
            df = robust_read_csv(csv_path)
            # Identify feature columns (prefer numeric), exclude sample_id if numeric
            feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "sample_id" in feat_cols:
                feat_cols.remove("sample_id")
            if not feat_cols:
                feat_cols = [c for c in df.columns if c.startswith("f_")]
