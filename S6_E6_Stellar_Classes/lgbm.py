import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score


# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIR     = "data/"
N_SPLITS     = 5
RANDOM_STATE = 31415
CAT_COLS     = ["spectral_type", "galaxy_population"]
ID_COL       = "id"
TARGET_COL   = "class"
NUM_COLS     = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
CLASSES      = ["GALAXY", "QSO", "STAR"]

tree_depth = 7

LGB_PARAMS = dict(
    n_estimators      = 2_000,
    learning_rate     = 0.025,
    num_leaves        = 2 ** tree_depth - 1,
    max_depth         = -1,
    min_child_samples = 5,
    subsample         = 0.75,
    colsample_bytree  = 0.75,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    min_gain_to_split = 0.01,
    objective         = "multiclass",
    metric            = "multi_logloss",
    class_weight      = "balanced",
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
    verbose           = -1,
)

PCA_CONFIGS = {
    "raw_bands": {
        "cols":         ["u", "g", "r", "i", "z"],
        "n_components": 3,
    },
    "colors": {
        "cols":         ["color_ug", "color_gr", "color_ri",
                         "color_rz", "color_ui", "color_uz",
                         "color_gi", "color_gz", "color_iz", 
                         "curv_g"  , "curv_r"  , "curv_i"  ,
                         "gr_over_ri", "ug_over_gr", "locus_dist", "color_total"],
        "n_components": 2,
    },
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(f"{data_dir}train.csv")
    test  = pd.read_csv(f"{data_dir}test.csv")

    # cast to float32 to reduce memory ~50%, eliminates float64 noise
    float_cols = train.select_dtypes(include='float64').columns
    train[float_cols] = train[float_cols].astype(np.float32)
    test[float_cols]  = test[float_cols].astype(np.float32)

    print(f"train: {train.shape} | test: {test.shape}")
    return train, test


# ── Feature engineering ────────────────────────────────────────────────────────

class FeatureEngineering:
    def __init__(self, target, cat_cols, num_cols):
        self.target   = target
        self.cat_cols = cat_cols
        self.num_cols = num_cols

    # ── Photometric colors ─────────────────────────────────────────────────────

    def color_indices(self, df, **_):
        df['color_ug'] = df['u'] - df['g']
        df['color_gr'] = df['g'] - df['r']
        df['color_ri'] = df['r'] - df['i']
        df['color_rz'] = df['r'] - df['z']
        df['color_ui'] = df['u'] - df['i']
        df['color_uz'] = df['u'] - df['z']
        df['color_gi'] = df['g'] - df['i']
        df['color_gz'] = df['g'] - df['z']
        df['color_iz'] = df['i'] - df['z']
        return df

    def color_derived(self, df, **_):
        df['curv_g']      = df['color_ug'] - df['color_gr']
        df['curv_r']      = df['color_gr'] - df['color_ri']
        df['curv_i']      = df['color_ri'] - df['color_rz']
        df['gr_over_ri']  = df['color_gr'] / (df['color_ri'] + 1e-6)
        df['ug_over_gr']  = df['color_ug'] / (df['color_gr'] + 1e-6)
        df['locus_dist']  = df['color_gr'] - (0.6 * df['color_ri'] + 0.1)
        df['color_total'] = df['color_ug'] + df['color_gr'] + df['color_ri'] + df['color_rz']

        # stellar and QSO locus distances in g-r / r-i colour space
        df['stellar_locus_dist'] = np.sqrt((df['color_gr'] - 0.52)**2 + (df['color_ri'] - 0.25)**2)
        df['qso_locus_dist']     = np.sqrt((df['color_gr'] - 0.24)**2 + (df['color_ri'] - 0.15)**2)

        return df

    # ── Flux space ─────────────────────────────────────────────────────────────

    def flux_features(self, df, **_):
        bands = ['u', 'g', 'r', 'i', 'z']
        for b in bands:
            df[f'flux_{b}'] = 10 ** (-0.4 * df[b])
        df['flux_total']    = sum(df[f'flux_{b}'] for b in bands)
        df['flux_ratio_ug'] = df['flux_u'] / (df['flux_g'] + 1e-9)
        df['flux_ratio_ri'] = df['flux_r'] / (df['flux_i'] + 1e-9)
        df['mag_mean']      = df[bands].mean(axis=1)
        df['mag_std']       = df[bands].std(axis=1)
        df['mag_range']     = df[bands].max(axis=1) - df[bands].min(axis=1)
        return df

    # ── Redshift ───────────────────────────────────────────────────────────────

    def redshift_features(self, df, **_):
        z                  = np.asarray(df['redshift'], dtype=float)
        d_L                = (299792.458 / 70) * z * (1 + 0.775 * z)
        mu                 = np.where(z > 1e-4, 5 * np.log10(np.maximum(d_L * 1e6, 1e-10)) - 5, 0.0)
        df['redshift_log'] = np.log1p(z)
        for b in ['u', 'g', 'r', 'i', 'z']:
            df[f'abs_mag_{b}'] = np.asarray(df[b], dtype=float) - mu
        for c in ['color_ug', 'color_gr', 'color_ri', 'curv_r']:
            df[f'{c}_x_z'] = df[c] * df['redshift_log']
        df = df.drop(columns=['redshift'])
        return df

    # ── Sky coordinates ────────────────────────────────────────────────────────

    def coord_features(self, df, **_):
        df['coord_dist']       = np.sqrt(df['alpha'] ** 2 + df['delta'] ** 2)
        df['sin_alpha']        = np.sin(np.radians(df['alpha']))
        df['cos_alpha']        = np.cos(np.radians(df['alpha']))
        df['sin_delta']        = np.sin(np.radians(df['delta']))
        df['cos_delta']        = np.cos(np.radians(df['delta']))
        df['coord_x']          = np.cos(np.radians(df['delta'])) * np.cos(np.radians(df['alpha']))
        df['coord_y']          = np.cos(np.radians(df['delta'])) * np.sin(np.radians(df['alpha']))
        df['coord_z']          = np.sin(np.radians(df['delta']))
        df['alpha_bin']        = pd.cut(df['alpha'], bins=10, labels=False)
        df['delta_bin']        = pd.cut(df['delta'], bins=10, labels=False)
        df['gal_lat']          = np.degrees(np.arcsin(
            np.sin(np.radians(df['delta'])) * np.cos(np.radians(62.9)) -
            np.cos(np.radians(df['delta'])) * np.sin(np.radians(62.9)) *
            np.sin(np.radians(df['alpha'] - 282.25))
        ))
        df['near_plane']       = (df['gal_lat'].abs() < 15).astype(int)
        df['coord_x_redshift'] = df['coord_dist'] * df['redshift_log']
        return df

    # ── Target encoding ────────────────────────────────────────────────────────

    def target_encoding(self, df, df_train=None, **_):
        if df_train is None:
            return df
        _tmp_col = "__target_enc__"
        df_train = df_train.copy()
        df_train[_tmp_col] = pd.factorize(df_train[self.target])[0]
        for col in self.cat_cols:
            means  = df_train.groupby(col)[_tmp_col].mean()
            counts = df_train.groupby(col)[self.target].count()
            df[f'{col}_target_mean']  = df[col].map(means).fillna(means.mean())
            df[f'{col}_target_count'] = df[col].map(counts).fillna(0)
        return df

    # ── Dispatcher ─────────────────────────────────────────────────────────────

    def transform(self, df, df_train=None, methods=None):
        df = df.copy()
        if methods is None:
            methods = [
                'color_indices',
                'color_derived',
                'flux_features',
                'redshift_features',    # needs raw color cols → after color_indices
                'coord_features',       # needs redshift_log  → after redshift_features
                'target_encoding',      # needs df_train      → always last
            ]
        for method_name in methods:
            if not hasattr(self, method_name):
                print(f"Warning: '{method_name}' not found, skipping.")
                continue
            df = getattr(self, method_name)(df, df_train=df_train)

        return df.replace([np.inf, -np.inf], np.nan)


# ── PCA embeddings ─────────────────────────────────────────────────────────────

def add_pca_features(
    train, test, cols, prefix,
    n_components=3, scale=True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([train[cols], test[cols]], axis=0).values.astype(float)

    if scale:
        combined = StandardScaler().fit_transform(combined)

    embedding = PCA(
        n_components = n_components,
        random_state = RANDOM_STATE,
    ).fit_transform(combined)

    n_train = len(train)
    for i in range(n_components):
        col        = f"{prefix}_pca_{i}"
        train[col] = embedding[:n_train, i]
        test[col]  = embedding[n_train:, i]

    print(f"[PCA] '{prefix}': {len(cols)} cols → {n_components}D")
    return train, test


def apply_all_pca(
    train:   pd.DataFrame,
    test:    pd.DataFrame,
    configs: dict = PCA_CONFIGS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for prefix, cfg in configs.items():
        train, test = add_pca_features(
            train, test,
            cols         = cfg["cols"],
            prefix       = prefix,
            n_components = cfg.get("n_components", 3),
        )
    return train, test


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(
    train:    pd.DataFrame,
    test:     pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, LabelEncoder]:
    """Encode target, cast categoricals, align columns."""
    le = LabelEncoder()
    y  = le.fit_transform(train[TARGET_COL].astype(str))

    for c in cat_cols:
        train[c] = train[c].astype("category")
        test[c]  = test[c].astype("category")

    X      = train.drop(columns=[TARGET_COL])
    X_test = test[X.columns]

    print(f"X_train: {X.shape} | X_test: {X_test.shape}")
    print(f"Classes: {list(le.classes_)}")
    return X, X_test, y, le


# ── Cross-validation ───────────────────────────────────────────────────────────

def run_cv(
    X:        pd.DataFrame,
    y:        np.ndarray,
    X_test:   pd.DataFrame,
    params:   dict,
    n_splits: int = N_SPLITS,
    seed:     int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Sequential stratified K-fold CV."""
    n_class    = len(np.unique(y))
    oof_preds  = np.zeros((len(X), n_class))
    test_preds = np.zeros((len(X_test), n_class))
    scores:    list[float] = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx],  y[va_idx]

        model = LGBMClassifier(**params, num_class=n_class)
        model.fit(
            X_tr, y_tr,
            eval_set  = [(X_va, y_va)],
            callbacks = [lgb.early_stopping(200), lgb.log_evaluation(50)],
        )

        oof   = np.asarray(model.predict_proba(X_va))
        score = balanced_accuracy_score(y_va, oof.argmax(axis=1))

        oof_preds[va_idx] = oof
        test_preds       += np.asarray(model.predict_proba(X_test)) / n_splits
        scores.append(score)

        print(f"  Fold {fold + 1}/{n_splits}: balanced_accuracy = {score:.5f}"
              f"  (best iter: {model.best_iteration_})")

    print(f"\nMean OOF balanced accuracy: "
          f"{np.mean(scores):.5f} ± {np.std(scores):.5f}")

    return oof_preds, test_preds, scores


# ── Output ─────────────────────────────────────────────────────────────────────

def make_submission(
    test_preds: np.ndarray,
    test_ids:   pd.Series,
    le:         LabelEncoder,
    path:       str = "submission.csv",
) -> pd.DataFrame:
    predicted_labels = le.inverse_transform(test_preds.argmax(axis=1))
    submission = pd.DataFrame({ID_COL: test_ids.values, TARGET_COL: predicted_labels})
    submission.to_csv(path, index=False)
    print(f"Submission saved → {path}")
    return submission


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # 1. load
    train, test = load_data(DATA_DIR)
    test_ids    = test[ID_COL]

    # 2. feature engineering
    fe    = FeatureEngineering(target=TARGET_COL, cat_cols=CAT_COLS, num_cols=NUM_COLS)
    train = fe.transform(train, df_train=train)
    test  = fe.transform(test,  df_train=train)  # always pass train as reference

    # 3. PCA embeddings
    train, test = apply_all_pca(train, test)

    # 4. preprocess
    X, X_test, y, le = preprocess(train, test, CAT_COLS)

    # 5. cross-validate
    print("\nRunning cross-validation...")
    oof_preds, test_preds, scores = run_cv(X, y, X_test, LGB_PARAMS)

    # 6. submission
    make_submission(test_preds, test_ids, le)


if __name__ == "__main__":
    main()