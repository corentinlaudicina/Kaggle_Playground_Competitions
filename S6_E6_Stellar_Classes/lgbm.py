import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import shap

os.makedirs("plots/", exist_ok=True)

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
    n_estimators      = 100,
    learning_rate     = 0.025,
    num_leaves        = 2 ** tree_depth - 1,
    max_depth         = -1,
    min_child_samples = 20,
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

EARLY_STOPPING = 80

# ── SHAP config ────────────────────────────────────────────────────────────────
SHAP_SUBSAMPLE = 3_000

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

        df['stellar_locus_dist'] = np.sqrt((df['color_gr'] - 0.52)**2 + (df['color_ri'] - 0.25)**2)
        df['qso_locus_dist']     = np.sqrt((df['color_gr'] - 0.24)**2 + (df['color_ri'] - 0.15)**2)

        return df

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

    def transform(self, df, df_train=None, methods=None):
        df = df.copy()
        if methods is None:
            methods = [
                'color_indices',
                'color_derived',
                'flux_features',
                'redshift_features',
                'coord_features',
                'target_encoding',
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
    scaler = StandardScaler() if scale else None
    train_vals = train[cols].values.astype(float)
    test_vals  = test[cols].values.astype(float)

    if scaler is not None:
        train_vals = scaler.fit_transform(train_vals)
        test_vals  = scaler.transform(test_vals)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    train_emb = pca.fit_transform(train_vals)
    test_emb  = pca.transform(test_vals)

    for i in range(n_components):
        col        = f"{prefix}_pca_{i}"
        train[col] = train_emb[:, i]
        test[col]  = test_emb[:, i]

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


# ── Feature pruning ───────────────────────────────────────────────────────────

def prune_features(
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: np.ndarray,
    cat_cols: list[str],
    threshold_frac: float = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train a quick model and drop features with near-zero gain importance."""
    print("Pruning near-zero-importance features…")
    n_class = len(np.unique(y))
    model = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=127,
        max_depth=-1, subsample=0.8, colsample_bytree=0.8,
        objective="multiclass", metric="multi_logloss",
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1, num_class=n_class,
    )
    model.fit(X, y)

    gain = model.booster_.feature_importance(importance_type="gain")
    gain_series = pd.Series(gain, index=X.columns)
    threshold = gain_series.max() * threshold_frac

    keep = gain_series[gain_series >= threshold].index.tolist()
    for c in cat_cols:
        if c in X.columns and c not in keep:
            keep.append(c)

    dropped = X.shape[1] - len(keep)
    print(f"  Keeping {len(keep)}/{X.shape[1]} features (dropped {dropped} near-zero)")
    return X[keep], X_test[keep]


# ── Cross-validation ───────────────────────────────────────────────────────────

def run_cv(
    X:        pd.DataFrame,
    y:        np.ndarray,
    X_test:   pd.DataFrame,
    params:   dict,
    n_splits: int = N_SPLITS,
    seed:     int = RANDOM_STATE,
    early_stopping_rounds: int = EARLY_STOPPING,
) -> tuple[np.ndarray, np.ndarray, list[float], LGBMClassifier]:
    """Sequential stratified K-fold CV. Returns (oof_preds, test_preds, scores, last_model)."""
    n_class    = len(np.unique(y))
    oof_preds  = np.zeros((len(X), n_class))
    test_preds = np.zeros((len(X_test), n_class))
    scores:    list[float] = []
    last_model = None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx],  y[va_idx]

        print(f"CV fold {fold + 1}/{n_splits} — training ({len(tr_idx):,} rows)…")

        model = LGBMClassifier(**params, num_class=n_class)
        model.fit(
            X_tr, y_tr,
            eval_set  = [(X_va, y_va)],
            callbacks = [
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        oof   = np.asarray(model.predict_proba(X_va))
        score = balanced_accuracy_score(y_va, oof.argmax(axis=1))

        oof_preds[va_idx] = oof
        test_preds       += np.asarray(model.predict_proba(X_test)) / n_splits
        scores.append(score)
        last_model = model

        print(f"  Fold {fold + 1}/{n_splits}: balanced_accuracy = {score:.5f}  best_iter={model.best_iteration_}")

    print(f"\nMean OOF balanced accuracy: "
          f"{np.mean(scores):.5f} ± {np.std(scores):.5f}")

    return oof_preds, test_preds, scores, last_model


# ── Feature importance plots ───────────────────────────────────────────────────

def _bar_chart(importances: pd.Series, title: str, path: str,
                    annotate_missing: set | None = None):
    """Horizontal bar chart with magma colormap on a dark background."""
    top = importances.nlargest(30).sort_values()
    colors_arr = cm.magma(np.linspace(0.2, 0.9, len(top)))

    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#ffffff")
    # ax.set_facecolor("#ffffff")

    bars = ax.barh(top.index, top.values, color=colors_arr, edgecolor="none")

    for bar, feat, val in zip(bars, top.index, top.values):
        label = f" {val:.1f}"
        marker = " ★" if (annotate_missing and feat in annotate_missing) else ""
        ax.text(bar.get_width() + top.values.max() * 0.005, bar.get_y() + bar.get_height() / 2,
                label + marker, va="center", ha="left", fontsize=8)

    ax.set_xlabel("Importance")
    ax.set_title(title, pad=12, fontsize=13)
    # ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.setp(ax.get_yticklabels(), fontsize=8)
    plt.setp(ax.get_xticklabels(), fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_importance_gain(model: LGBMClassifier, feature_names: list[str]):
    booster    = model.booster_
    gain_raw   = booster.feature_importance(importance_type="gain")
    gain_series = pd.Series(gain_raw, index=feature_names)
    _bar_chart(gain_series, "Top 30 Features — GAIN importance", "plots/importance_gain.png")
    return gain_series


def plot_importance_split(model: LGBMClassifier, feature_names: list[str],
                          gain_series: pd.Series):
    booster      = model.booster_
    split_raw    = booster.feature_importance(importance_type="split")
    split_series = pd.Series(split_raw, index=feature_names)

    top30_gain  = set(gain_series.nlargest(30).index)
    top30_split = set(split_series.nlargest(30).index)
    only_split  = top30_split - top30_gain

    _bar_chart(split_series, "Top 30 Features — SPLIT importance  (★ = not in top-30 gain)",
                    "plots/importance_split.png", annotate_missing=only_split)
    return split_series


def plot_importance_shap(model: LGBMClassifier, X: pd.DataFrame):
    rng  = np.random.default_rng(RANDOM_STATE)
    idx  = rng.choice(len(X), size=min(SHAP_SUBSAMPLE, len(X)), replace=False)
    X_sub = X.iloc[idx]

    print(f"SHAP: building TreeExplainer on {len(X_sub):,}-row subsample…")
    explainer   = shap.TreeExplainer(model.booster_)
    shap_values = explainer.shap_values(X_sub)
    # Normalise multiclass SHAP output to list[(n_samples, n_features)]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        n_cls = len(CLASSES)
        if shap_values.shape[0] == n_cls:
            # (n_classes, n_samples, n_features)
            shap_values = [shap_values[i] for i in range(n_cls)]
        else:
            # (n_samples, n_features, n_classes)
            shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

    class_names = CLASSES
    fig, axes = plt.subplots(1, 3, figsize=(24, 20))

    for cls_idx, (ax, cls_name) in enumerate(zip(axes, class_names)):
        plt.sca(ax)
        shap.summary_plot(
            shap_values[cls_idx],
            X_sub,
            plot_type  = "dot",
            show       = False,
            max_display = 20,
            color_bar   = (cls_idx == 2),
        )
        ax.set_title(f"SHAP — {cls_name}", fontsize=12, pad=8)

    fig.suptitle("SHAP Beeswarm  (top 20 features per class)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig("plots/importance_shap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved → plots/importance_shap.png")

    return shap_values, X_sub


def print_importance_summary(gain_series: pd.Series, split_series: pd.Series,
                             shap_values, X_sub: pd.DataFrame):
    top30_gain  = set(gain_series.nlargest(30).index)
    top30_split = set(split_series.nlargest(30).index)

    print("\n" + "═" * 60)
    print("FEATURE IMPORTANCE SUMMARY")
    print("═" * 60)

    print("\nTop 5 by GAIN:")
    for feat, val in gain_series.nlargest(5).items():
        print(f"  {feat:<35s}  {val:>12.1f}")

    only_split = top30_split - top30_gain
    if only_split:
        print(f"\nHigh split / low gain (potential noise — {len(only_split)} features):")
        for feat in sorted(only_split):
            print(f"  {feat:<35s}  split={split_series[feat]:.0f}  gain={gain_series[feat]:.1f}")

    threshold  = gain_series.max() * 0.001
    near_zero  = gain_series[gain_series < threshold].index.tolist()
    near_zero2 = split_series[split_series < split_series.max() * 0.001].index.tolist()
    both_zero  = [f for f in near_zero if f in near_zero2]
    print(f"\nNear-zero importance in both metrics ({len(both_zero)} features):")
    for feat in both_zero[:20]:
        print(f"  {feat}")

    print("\nStrongest directional SHAP effect per class:")
    feat_names = X_sub.columns.tolist()
    for cls_idx, cls_name in enumerate(CLASSES):
        sv   = shap_values[cls_idx]
        mean_abs = np.abs(sv).mean(axis=0)
        top_feat = feat_names[np.argmax(mean_abs)]
        top_val  = mean_abs.max()
        direction = "+" if sv[:, np.argmax(mean_abs)].mean() > 0 else "-"
        print(f"  {cls_name:<8s}: {top_feat:<35s}  mean|SHAP|={top_val:.4f}  direction={direction}")

    print("═" * 60)


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
    print("Loading data…")
    train, test = load_data(DATA_DIR)
    test_ids    = test[ID_COL]

    # 2. feature engineering
    print("Feature engineering…")
    fe    = FeatureEngineering(target=TARGET_COL, cat_cols=CAT_COLS, num_cols=NUM_COLS)
    train = fe.transform(train, df_train=train)
    test  = fe.transform(test,  df_train=train)

    # 3. PCA embeddings
    print("PCA embeddings…")
    train, test = apply_all_pca(train, test)

    # 4. preprocess
    print("Preprocessing…")
    X, X_test, y, le = preprocess(train, test, CAT_COLS)

    # 5. prune near-zero-importance features
    X, X_test = prune_features(X, X_test, y, CAT_COLS)

    # 6. cross-validation
    print("\n" + "═" * 60)
    print(f"FINAL {N_SPLITS}-FOLD CV")
    print("═" * 60)

    oof_preds, test_preds, scores, last_model = run_cv(X, y, X_test, LGB_PARAMS)

    make_submission(test_preds, test_ids, le, path="submission.csv")

    # 7. feature importance plots
    print("\n" + "═" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("═" * 60)

    feature_names = X.columns.tolist()

    print("Plotting gain importance…")
    gain_series  = plot_importance_gain(last_model, feature_names)
    print("Plotting split importance…")
    split_series = plot_importance_split(last_model, feature_names, gain_series)
    print("Running SHAP (this is the slow step)…")
    shap_values, X_sub = plot_importance_shap(last_model, X)

    print_importance_summary(gain_series, split_series, shap_values, X_sub)

    near_zero_count = int((gain_series < gain_series.max() * 0.001).sum())

    print("\n" + "═" * 60)
    print("FINAL SUMMARY")
    print("═" * 60)
    print(f"  Final {N_SPLITS}-fold score: {np.mean(scores):.5f} ± {np.std(scores):.5f}")
    print(f"  Number of features used: {X.shape[1]}")
    print(f"  Number of near-zero importance features: {near_zero_count}")
    print("═" * 60)


if __name__ == "__main__":
    main()
