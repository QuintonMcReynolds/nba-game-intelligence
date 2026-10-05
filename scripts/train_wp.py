"""Train and compare win-probability models; save test-season predictions for WPA and recaps."""

import json

import numpy as np
import pandas as pd

from nbagi import ARTIFACTS, load_config, wp
from nbagi.features import CURATED, elo_table, wp_frame


def main():
    cfg = load_config()
    w = cfg["win_probability"]
    elo = elo_table(cfg)
    elo.to_parquet(CURATED / "elo.parquet")
    train = wp_frame(*w["train_seasons"], elo, game_frac=0.4)
    if len(train) > w["train_sample"]:
        train = train.sample(w["train_sample"], random_state=cfg["seed"])
    valid = wp_frame(*w["valid_seasons"], elo)
    test = wp_frame(*w["test_seasons"], elo)
    print(f"train {len(train):,} events, valid {len(valid):,}, test {len(test):,}")

    models = {"logistic": wp.fit_logistic(train),
              "gbm_unconstrained": wp.fit_gbm(train, cfg["seed"], constrained=False),
              "gbm_monotone": wp.fit_gbm(train, cfg["seed"])}
    models["keras_mlp"] = wp.fit_keras(train, valid, cfg["seed"])

    # pregame-only baseline: the Elo expectation, constant through the game
    base = {"pregame_elo": lambda d: 1 / (1 + 10 ** (-d["elo_diff"] / 400))}
    out = {"n_train": len(train), "n_valid": len(valid), "n_test": len(test), "models": {}}
    preds = {}
    for name, m in {**models, **base}.items():
        f = m if callable(m) and name in base else (lambda d, m=m: wp.predict(m, d))
        pv, pt = np.asarray(f(valid)), np.asarray(f(test))
        preds[name] = pt
        out["models"][name] = {
            "valid_log_loss": float(wp.log_loss(valid["home_win"], pv).mean()),
            "test_log_loss": float(wp.log_loss(test["home_win"], pt).mean()),
            "test_brier": float(((pt - test["home_win"]) ** 2).mean()),
            "test_ece": wp.ece(test["home_win"].to_numpy(), pt),
        }
        if name in models and name != "keras_mlp":
            ptr = np.asarray(f(train))
            out["models"][name]["train_log_loss"] = float(
                wp.log_loss(train["home_win"], ptr).mean())
    best = min(models, key=lambda k: out["models"][k]["valid_log_loss"])  # chosen on validation
    out["selected_on_validation"] = best

    y = test["home_win"].to_numpy()
    losses = {k: wp.log_loss(y, p) for k, p in preds.items()}
    out["vs_selected_ci95"] = {
        k: wp.game_bootstrap(losses[k], losses[best], test["game_id"].to_numpy())
        for k in preds if k != best}
    bucket = wp.time_bucket(test)
    out["by_time_left"] = {
        b: {k: float(losses[k][bucket == b].mean()) for k in preds}
        for b in ["1st quarter", "2nd quarter", "3rd quarter", "4th: >5 min left",
                  "4th: last 5 min", "overtime"]}

    # calibration table for the selected model (figure + Tableau)
    p = preds[best]
    bins = np.minimum((p * 20).astype(int), 19)
    cal = pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin").agg(
        predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()
    cal.to_csv(ARTIFACTS / "wp_calibration.csv", index=False)

    keep = test[["season", "game_id", "action_number", "period", "sec_left", "margin",
                 "home_has_ball", "elo_diff", "home_win"]].copy()
    for k, v in preds.items():
        keep[f"wp_{k}"] = v.astype("float32")
    keep["wp"] = keep[f"wp_{best}"]
    keep.to_parquet(CURATED / "wp_test.parquet", index=False)

    (ARTIFACTS / "wp_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out["models"], indent=2))
    print("selected:", best, "| CI of loss difference vs selected:", out["vs_selected_ci95"])
    print(pd.DataFrame(out["by_time_left"]).T.round(4))


if __name__ == "__main__":
    main()
