# Module-level imports
import re
import os
import pandas as pd
import multiprocessing
from time import time as timer
from tqdm import tqdm
import numpy as np
from pathlib import Path
from functools import partial
import requests
import urllib
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- ML utilities for Day 1 improvements ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import make_scorer
from scipy.sparse import hstack

# Optional persistence
import joblib

# Add robust retry support
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def download_image(image_link, savefolder):
    import time
    if(isinstance(image_link, str)):
        filename = Path(image_link).name
        image_save_path = os.path.join(savefolder, filename)
        if(not os.path.exists(image_save_path)):
            headers = {'User-Agent': 'Mozilla/5.0'}
            # Robust retry: retry on 429/5xx and connection/read timeouts
            retry = Retry(
                total=5,
                connect=3,
                read=3,
                backoff_factor=1.5,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"]
            )
            session = requests.Session()
            adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            try:
                with session.get(image_link, stream=True, timeout=(5, 30), headers=headers) as r:
                    r.raise_for_status()
                    with open(image_save_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
            except Exception as ex:
                print('Warning: Not able to download - {}\n{}'.format(image_link, ex))
                try:
                    fail_log = os.path.join(savefolder, 'failed_downloads.txt')
                    with open(fail_log, 'a', encoding='utf-8') as lf:
                        lf.write(f'{image_link}\n')
                except Exception:
                    pass
            finally:
                session.close()
        else:
            return
    return

def download_images(image_links, download_folder, max_workers: int | None = None):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    results = []
    download_image_partial = partial(download_image, savefolder=download_folder)
    # Use threads for I/O-bound downloads and keep worker count under Windows handle limits
    default_workers = min(32, (os.cpu_count() or 2) * 4)
    worker_count = max_workers if isinstance(max_workers, int) and max_workers > 0 else default_workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        for result in tqdm(executor.map(download_image_partial, image_links), total=len(image_links)):
            results.append(result)


# =====================
# Day 1 Text Improvements
# =====================

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (lower is better)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred))
    # Avoid division by zero
    denom = np.where(denom == 0, 1e-8, denom)
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)


def smape_scorer():
    """sklearn scorer that MINIMIZES SMAPE (returns negative score)."""
    return make_scorer(smape, greater_is_better=False)


def preprocess_text_series(s: pd.Series) -> pd.Series:
    """Lightweight text normalization for TF-IDF.
    - Lowercase
    - Normalize common unit tokens (lbs->lb, ounces->oz, kilograms->kg, grams->g, liters->l, milliliters->ml)
    - Remove extra whitespace
    """
    s = s.fillna("")
    s = s.str.lower()
    # Unit normalization
    replacements = {
        r"\blbs\b": "lb",
        r"\bounces\b": "oz",
        r"\bokgs\b": "kg",
        r"\bkilograms\b": "kg",
        r"\bgrams\b": "g",
        r"\bliters\b": "l",
        r"\bmilliliters\b": "ml",
    }
    for pat, rep in replacements.items():
        s = s.str.replace(pat, rep, regex=True)
    # Collapse whitespace
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def vectorize_text(train_text: pd.Series, test_text: pd.Series, **tfidf_kwargs):
    """Fit TF-IDF on train text and transform both train & test."""
    train_text = preprocess_text_series(train_text)
    test_text = preprocess_text_series(test_text)
    tfidf = TfidfVectorizer(**tfidf_kwargs)
    X_train_text = tfidf.fit_transform(train_text)
    X_test_text = tfidf.transform(test_text)
    return tfidf, X_train_text, X_test_text


def combine_struct_and_text(X_struct_df: pd.DataFrame, X_text_sparse):
    """Horizontally stack structured numeric features and sparse text features."""
    X_struct = np.asarray(X_struct_df.values)
    return hstack([X_struct, X_text_sparse])


def get_kfold(cv_splits: int = 5, random_state: int = 42):
    return KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)


def cv_smape(model, X, y, cv_splits: int = 5):
    kf = get_kfold(cv_splits)
    scores = cross_val_score(model, X, y, cv=kf, scoring=smape_scorer(), n_jobs=-1)
    # cross_val_score returns NEGATIVE values because greater_is_better=False
    smape_scores = -scores
    return float(np.mean(smape_scores)), smape_scores


def tune_ridge_on_tfidf(X, y, alphas=None, cv_splits: int = 5):
    """Tune Ridge alpha via CV on SMAPE, then fit final model."""
    if alphas is None:
        alphas = np.logspace(-3, 3, 8)  # 1e-3 ... 1e3
    best_alpha = None
    best_cv = np.inf
    for a in alphas:
        model = Ridge(alpha=a, solver='lsqr')
        mean_cv, _ = cv_smape(model, X, y, cv_splits=cv_splits)
        if mean_cv < best_cv:
            best_cv = mean_cv
            best_alpha = a
    final_model = Ridge(alpha=best_alpha, solver='lsqr')
    final_model.fit(X, y)
    return final_model, best_alpha, best_cv


def tune_sgd_elasticnet_on_tfidf(X, y, l1_ratio_list=None, cv_splits: int = 5):
    """Tune SGDRegressor (elasticnet penalty) l1_ratio via CV on SMAPE."""
    if l1_ratio_list is None:
        l1_ratio_list = [0.05, 0.15, 0.3, 0.5]
    best_lr = None
    best_cv = np.inf
    best_params = None
    for lr in l1_ratio_list:
        model = SGDRegressor(
            penalty='elasticnet', l1_ratio=lr, random_state=42, max_iter=2000,
            early_stopping=True, n_iter_no_change=10, learning_rate='optimal'
        )
        mean_cv, _ = cv_smape(model, X, y, cv_splits=cv_splits)
        if mean_cv < best_cv:
            best_cv = mean_cv
            best_lr = lr
    final_model = SGDRegressor(
        penalty='elasticnet', l1_ratio=best_lr, random_state=42, max_iter=2000,
        early_stopping=False, learning_rate='optimal'
    )
    final_model.fit(X, y)
    best_params = {'l1_ratio': best_lr}
    return final_model, best_params, best_cv


def save_text_features_cache(out_dir: str, tfidf, X_train_text, X_test_text):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(tfidf, os.path.join(out_dir, 'tfidf.pkl'))
    joblib.dump(X_train_text, os.path.join(out_dir, 'X_train_text.pkl'))
    joblib.dump(X_test_text, os.path.join(out_dir, 'X_test_text.pkl'))


def load_text_features_cache(out_dir: str):
    tfidf_path = os.path.join(out_dir, 'tfidf.pkl')
    Xtr_path = os.path.join(out_dir, 'X_train_text.pkl')
    Xte_path = os.path.join(out_dir, 'X_test_text.pkl')
    if all(os.path.exists(p) for p in [tfidf_path, Xtr_path, Xte_path]):
        tfidf = joblib.load(tfidf_path)
        X_train_text = joblib.load(Xtr_path)
        X_test_text = joblib.load(Xte_path)
        return tfidf, X_train_text, X_test_text
    return None


def summarize_predictions(y_pred: np.ndarray):
    return {
        'min': float(np.min(y_pred)),
        'max': float(np.max(y_pred)),
        'mean': float(np.mean(y_pred)),
        'std': float(np.std(y_pred)),
    }