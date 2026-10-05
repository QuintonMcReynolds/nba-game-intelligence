"""Tableau-ready extracts (tidy CSVs, one grain per file) for the dashboards in tableau/."""

import shutil

import duckdb

from nbagi import ARTIFACTS, ROOT
from nbagi.features import CURATED
from nbagi.recaps import TEAMS

OUT = ROOT / "tableau" / "data"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.register("team_names", __import__("pandas").DataFrame(
        {"team": list(TEAMS), "team_name": list(TEAMS.values())}))
    # win-probability curve of every 2025-26 game, one row per game-minute
    con.execute(f"""
        COPY (
          SELECT w.game_id, g.away_team || ' @ ' || g.home_team AS matchup,
                 g.home_team, g.away_team, g.home_final, g.away_final,
                 floor((2880 - least(w.sec_left, 2880)) / 60)::INT
                     + CASE WHEN w.period > 4 THEN 48 + 5 * (w.period - 5) ELSE 0 END
                     AS game_minute,
                 last(w.wp ORDER BY w.action_number) AS home_win_prob,
                 last(w.margin ORDER BY w.action_number) AS home_margin
          FROM read_parquet('{CURATED}/wp_test.parquet') w
          JOIN read_parquet('{CURATED}/games/*.parquet') g USING (game_id)
          WHERE w.season = 2025
          GROUP BY ALL ORDER BY w.game_id, game_minute
        ) TO '{OUT}/wp_curves_2025.csv' (HEADER)""")
    # team-season results
    con.execute(f"""
        COPY (
          WITH t AS (
            SELECT season, home_team AS team, home_win AS win, home_final - away_final AS diff
            FROM read_parquet('{CURATED}/games/*.parquet')
            UNION ALL
            SELECT season, away_team, 1 - home_win, away_final - home_final
            FROM read_parquet('{CURATED}/games/*.parquet'))
          SELECT season, team, coalesce(n.team_name, team) AS team_name, count(*) AS games,
                 sum(win) AS wins, avg(diff) AS point_diff_per_game
          FROM t LEFT JOIN team_names n USING (team) GROUP BY ALL ORDER BY season, wins DESC
        ) TO '{OUT}/team_seasons.csv' (HEADER)""")
    # biggest single plays of 2025-26 by win probability added
    con.execute(f"""
        COPY (
          SELECT game_id, player_name_i AS player, team, period, round(sec_left) AS sec_left,
                 description, wpa_player, wp AS home_wp_after
          FROM read_parquet('{CURATED}/wpa_events.parquet')
          WHERE season = 2025 AND person_id IS NOT NULL
          ORDER BY abs(wpa_player) DESC LIMIT 2000
        ) TO '{OUT}/top_plays_2025.csv' (HEADER)""")
    for name in ["era_trends.csv", "clutch_leaderboard.csv", "wp_calibration.csv",
                 "recap_scores.csv"]:
        if (ARTIFACTS / name).exists():
            shutil.copy(ARTIFACTS / name, OUT / name)
    for p in sorted(OUT.glob("*.csv")):
        print(f"  {p.name}: {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
