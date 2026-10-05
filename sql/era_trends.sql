-- How the game changed, 2004-05 to 2025-26, straight from play-by-play.
WITH shots AS (
    SELECT season,
        count(*) FILTER (WHERE event IN ('Made Shot', 'Missed Shot')) AS fga,
        count(*) FILTER (WHERE event IN ('Made Shot', 'Missed Shot') AND is_three) AS fg3a,
        count(*) FILTER (WHERE event = 'Made Shot') AS fgm,
        count(*) FILTER (WHERE event = 'Made Shot' AND is_three) AS fg3m,
        count(*) FILTER (WHERE event = 'Free Throw') AS fta,
        count(*) FILTER (WHERE event = 'Turnover') AS tov,
        count(*) FILTER (WHERE event = 'Rebound' AND person_id IS NOT NULL) AS reb,
        count(DISTINCT game_id) AS games
    FROM read_parquet('{curated}/events/*/*.parquet', hive_partitioning = true)
    GROUP BY season
),
g AS (
    SELECT season, avg(home_final + away_final) AS points_per_game,
        avg(home_win) AS home_win_rate,
        avg((periods > 4)::INT) AS overtime_rate
    FROM read_parquet('{curated}/games/*.parquet') GROUP BY season
)
SELECT season, games,
    points_per_game,
    fga / games / 2 AS fga_per_team_game,
    fg3a / fga AS three_point_rate,
    fg3m / fg3a AS three_point_pct,
    (fgm + 0.5 * fg3m) / fga AS efg_pct,
    fta / fga AS free_throw_rate,
    -- possessions per team-game (standard estimate)
    (fga + 0.44 * fta + tov - 0.5 * (fga - fgm) * 0.27) / games / 2 AS pace_est,
    home_win_rate, overtime_rate
FROM shots JOIN g USING (season)
ORDER BY season
