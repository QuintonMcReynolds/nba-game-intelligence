"""Pregame Elo ratings (FiveThirtyEight-style) from game results, in game order."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pregame_elo(games: pd.DataFrame, k: float = 20, home_advantage: float = 100,
                carryover: float = 0.75) -> pd.DataFrame:
    """Returns game_id, home_elo, away_elo, elo_diff (home - away + home advantage)."""
    g = games.sort_values(["season", "game_id"])
    rating: dict[int, float] = {}
    season_now = None
    rows = []
    for r in g.itertuples(index=False):
        if r.season != season_now:  # regress a quarter of the way to the mean each season
            rating = {t: 1500 + carryover * (v - 1500) for t, v in rating.items()}
            season_now = r.season
        h, a = rating.get(r.home_team_id, 1500.0), rating.get(r.away_team_id, 1500.0)
        diff = h - a + home_advantage
        rows.append((r.game_id, h, a, diff))
        exp_home = 1 / (1 + 10 ** (-diff / 400))
        margin = abs(r.home_final - r.away_final)
        # margin-of-victory multiplier, damped for favourites
        elo_gap = diff if r.home_win else -diff
        mult = np.log(margin + 1) * 2.2 / (elo_gap * 0.001 + 2.2)
        delta = k * mult * (r.home_win - exp_home)
        rating[r.home_team_id], rating[r.away_team_id] = h + delta, a - delta
    return pd.DataFrame(rows, columns=["game_id", "home_elo", "away_elo", "elo_diff"])
