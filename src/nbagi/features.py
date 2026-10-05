"""Model-ready tables pulled from the curated lake with DuckDB SQL."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from . import LAKE, ROOT
from .elo import pregame_elo

CURATED = LAKE / "curated"
# No overtime flag: overtime is modeled as what it is, a game with <= 5 minutes left, so
# it borrows the thousands of late-regulation situations instead of a few hundred OT games
# (an OT flag put tied overtimes at a 0.30 home win probability).
FEATURES = ["margin", "sec_left", "home_has_ball", "ball_unknown", "elo_diff",
            "margin_per_root_t", "elo_x_time", "ball_x_late"]


def games() -> pd.DataFrame:
    return duckdb.sql(f"SELECT * FROM read_parquet('{CURATED}/games/*.parquet')").df()


def elo_table(cfg: dict) -> pd.DataFrame:
    return pregame_elo(games(), **cfg["win_probability"]["elo"])


def wp_frame(lo: int, hi: int, elo: pd.DataFrame, game_frac: float = 1.0) -> pd.DataFrame:
    """Events for seasons lo..hi. game_frac < 1 keeps a deterministic sample of whole games."""
    sql = (ROOT / "sql" / "wp_features.sql").read_text().format(
        curated=CURATED, lo=lo, hi=hi)
    if game_frac < 1:
        sql += f"  AND hash(e.game_id) % 1000 < {int(game_frac * 1000)}\n"
    con = duckdb.connect()
    con.register("elo", elo)
    df = con.execute(sql).df()
    return add_features(df)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df["sec_left_game"].clip(lower=0).astype(float)
    df["sec_left"] = t
    df["overtime"] = df["overtime"].astype(int)
    df["ball_unknown"] = df["ball_unknown"].astype(int)
    df["margin_per_root_t"] = df["margin"] / np.sqrt(t + 1)
    # the pregame edge matters in proportion to the time still to play
    df["elo_x_time"] = df["elo_diff"] * t / 2880
    # possession is worth more as time runs down, and nothing once the clock reads 0:00
    df["ball_x_late"] = ((2 * df["home_has_ball"] - 1) * (1 - df["ball_unknown"])
                         * (t > 0) / np.sqrt(t + 1))
    return df
