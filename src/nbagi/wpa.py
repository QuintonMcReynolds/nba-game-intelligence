"""Win probability added and a shrunken clutch leaderboard."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from . import ROOT
from .features import CURATED


def wpa_events(cfg: dict) -> pd.DataFrame:
    c = cfg["clutch"]
    sql = (ROOT / "sql" / "wpa_events.sql").read_text().format(
        curated=CURATED, clutch_sec=c["minutes_left"] * 60, clutch_margin=c["max_margin"])
    df = duckdb.sql(sql).df()
    # credit the acting player: home-team players get +WPA(home), away players the negative
    home = df["is_home_team"].fillna(0).astype(int).to_numpy() == 1
    df["wpa_player"] = np.where(home, df["wpa_home"], -df["wpa_home"])
    return df


def shrink(total: np.ndarray, n: np.ndarray, per_event_var: np.ndarray) -> tuple:
    """Empirical-Bayes normal shrinkage of each player's WPA per event toward 0.

    Prior: true rate ~ N(0, tau2); observed rate ~ N(true, s2/n). tau2 by method of moments.
    Returns (shrunk rate, shrinkage factor B)."""
    rate = total / n
    se2 = per_event_var / n
    tau2 = max(np.average(rate**2 - se2, weights=n), 1e-12)
    b = tau2 / (tau2 + se2)
    return b * rate, b


def clutch_leaderboard(ev: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = ev[ev["late"] & ev["close"] & ev["person_id"].notna()]
    g = c.groupby(["season", "person_id"]).agg(
        player=("player_name_i", "first"), team=("team", "last"),
        clutch_events=("wpa_player", "size"), clutch_wpa=("wpa_player", "sum"),
        var=("wpa_player", "var")).reset_index()
    g = g[g["clutch_events"] >= cfg["clutch"]["min_events"]].copy()
    pooled_var = np.average(g["var"], weights=g["clutch_events"])
    rate, b = shrink(g["clutch_wpa"].to_numpy(), g["clutch_events"].to_numpy(),
                     np.full(len(g), pooled_var))
    g["shrinkage"] = b
    g["clutch_wpa_shrunk"] = rate * g["clutch_events"]
    return g.drop(columns="var").sort_values(["season", "clutch_wpa_shrunk"],
                                             ascending=[True, False])
