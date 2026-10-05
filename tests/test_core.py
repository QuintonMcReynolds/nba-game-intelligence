import numpy as np
import pandas as pd

from nbagi.elo import pregame_elo
from nbagi.features import add_features
from nbagi.wp import ece, game_bootstrap, log_loss
from nbagi.wpa import shrink


def test_elo_zero_sum_and_home_edge():
    g = pd.DataFrame({"season": [2024] * 3, "game_id": ["1", "2", "3"],
                      "home_team_id": [1, 2, 1], "away_team_id": [2, 1, 2],
                      "home_final": [110, 100, 120], "away_final": [100, 105, 90],
                      "home_win": [1, 0, 1]})
    e = pregame_elo(g)
    assert e.loc[0, "elo_diff"] == 100  # equal ratings + home advantage
    # team 1 won twice: by game 3 it is rated above team 2
    assert e.loc[2, "home_elo"] > e.loc[2, "away_elo"]


def test_features_monotone_in_time():
    df = pd.DataFrame({"sec_left_game": [2880, 60], "margin": [5, 5], "overtime": [False, False],
                       "home_has_ball": [1, 1], "ball_unknown": [False, False],
                       "elo_diff": [100, 100]})
    f = add_features(df)
    assert f.loc[1, "margin_per_root_t"] > f.loc[0, "margin_per_root_t"]
    assert f.loc[1, "elo_x_time"] < f.loc[0, "elo_x_time"]


def test_metrics():
    y = np.array([1, 0, 1, 0])
    assert np.isclose(log_loss(y, np.full(4, 0.5)).mean(), np.log(2))
    assert ece(y, np.array([1.0, 0.0, 1.0, 0.0])) == 0
    lo, hi = game_bootstrap(np.ones(8), np.zeros(8), np.repeat(["a", "b", "c", "d"], 2))
    assert lo == hi == 1


def test_shrinkage_pulls_small_samples_harder():
    rate, b = shrink(np.array([2.0, 2.0]), np.array([10, 1000]), np.array([1.0, 1.0]))
    assert b[0] < b[1] and abs(rate[0]) < abs(2.0 / 10)
