# Tableau dashboards

`make tableau` writes tidy extracts to `tableau/data/` (one grain per file). Open Tableau
Public (free) or Desktop → *Text file* → pick a CSV, then build the sheets below.

| File | Grain | Use |
|---|---|---|
| `wp_curves_2025.csv` | game × minute, 2025-26 | Win-probability chart for any game (filter on `matchup`) |
| `top_plays_2025.csv` | play | Biggest swings of the season (`wpa_player`, `description`) |
| `clutch_leaderboard.csv` | player × season | Clutch WPA, raw and shrunk, with sample size |
| `team_seasons.csv` | team × season | Wins, point differential, 2004-05 to 2025-26 |
| `era_trends.csv` | season | 3-point rate, pace, eFG%, home win rate |
| `wp_calibration.csv` | probability bin | Predicted vs observed win rate (model check) |
| `recap_scores.csv` | generated recap | Claims, false claims and arm of the LLM experiment |

## Dashboard 1: Game Flow

- **Sheet "WP curve":** Columns `game_minute` (continuous), Rows `home_win_prob`. Filter
  `matchup` (single value, dropdown). Add a reference line at 0.5. Tooltip: `home_margin`.
- **Sheet "Swings":** from `top_plays_2025.csv`. Filter by the same `game_id` with an action
  filter. Show `description`, `player` and `wpa_player` as a table sorted by
  |`wpa_player`|.

## Dashboard 2: Clutch

- **Bars:** `player` by SUM(`clutch_wpa_shrunk`), filtered to `season` = 2025, top 20.
- **Scatter:** `clutch_events` vs `clutch_wpa`, colored by `shrinkage`. It shows why
  low-volume players are pulled toward zero.

## Dashboard 3: How the league changed

- **Lines:** `season` vs `three_point_rate`, `pace_est`, `home_win_rate`, as small multiples.
  Don't use a dual axis: the measures have different scales.
- **Highlight table:** `team_seasons.csv`, `team_name` × `season`, colored by `wins`.

These CSVs are generated and checked by the pipeline. The workbooks themselves are built in
Tableau and are not in this repo.
