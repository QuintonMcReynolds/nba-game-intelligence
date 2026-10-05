"""Win-probability models: logistic regression and gradient boosting (scikit-learn) and a
neural network (TensorFlow/Keras), plus the metrics they are judged on."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

EPS = 1e-6


def fit_logistic(df: pd.DataFrame):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))
    return m.fit(df[FEATURES], df["home_win"])


def fit_gbm(df: pd.DataFrame, seed: int = 0, constrained: bool = True):
    """Gradient boosting. Unconstrained, it memorizes games: the pregame Elo gap is nearly
    unique per game, so small leaves learn "this game's home team won" (train log loss 0.39,
    validation 0.49). Monotone constraints (more lead / better team can only raise win
    probability) and large leaves stop that."""
    if not constrained:
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=500,
            l2_regularization=1.0, early_stopping=False, random_state=seed)
        return m.fit(df[FEATURES], df["home_win"])
    mono = {"margin": 1, "elo_diff": 1, "margin_per_root_t": 1, "elo_x_time": 1,
            "ball_x_late": 1}  # by name; HistGradientBoosting maps them to FEATURES
    m = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=5000,
        l2_regularization=1.0, monotonic_cst=mono, early_stopping=False, random_state=seed)
    return m.fit(df[FEATURES], df["home_win"])


def fit_keras(train: pd.DataFrame, valid: pd.DataFrame, seed: int = 0, epochs: int = 6):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    norm = tf.keras.layers.Normalization()
    norm.adapt(train[FEATURES].to_numpy("float32"))
    model = tf.keras.Sequential([
        tf.keras.Input((len(FEATURES),)), norm,
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    model.fit(train[FEATURES].to_numpy("float32"), train["home_win"].to_numpy("float32"),
              validation_data=(valid[FEATURES].to_numpy("float32"),
                               valid["home_win"].to_numpy("float32")),
              epochs=epochs, batch_size=4096, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=1,
                                                          restore_best_weights=True)])
    return model


def predict(model, df: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(df[FEATURES])[:, 1]
    return model.predict(df[FEATURES].to_numpy("float32"), batch_size=65536, verbose=0).ravel()


def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ece(y, p, bins: int = 20) -> float:
    """Expected calibration error: |mean prediction - win rate|, weighted over bins."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = np.bincount(idx, minlength=bins)
    gap = np.abs(np.bincount(idx, p, bins) - np.bincount(idx, y, bins))
    return float(gap.sum() / tot.sum())


def time_bucket(df: pd.DataFrame) -> pd.Series:
    t, ot = df["sec_left"], df["overtime"] == 1
    return pd.Series(np.select(
        [ot, t > 2160, t > 1440, t > 720, t > 300],
        ["overtime", "1st quarter", "2nd quarter", "3rd quarter", "4th: >5 min left"],
        "4th: last 5 min"), index=df.index)


def game_bootstrap(loss_a: np.ndarray, loss_b: np.ndarray, games: np.ndarray,
                   reps: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI for mean(loss_a - loss_b), resampling whole games."""
    d = pd.DataFrame({"g": games, "d": loss_a - loss_b}).groupby("g")["d"].agg(["sum", "size"])
    s, n = d["sum"].to_numpy(), d["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (reps, len(s)))
    stats = s[idx].sum(1) / n[idx].sum(1)
    return tuple(np.percentile(stats, [2.5, 97.5]))
