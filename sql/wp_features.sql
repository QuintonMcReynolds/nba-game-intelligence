-- One row per event: the game state a win-probability model sees, plus the outcome.
-- `elo` is the pregame Elo table registered by Python; paths point at the Spark output.
SELECT
    e.season,
    e.game_id,
    e.action_number,
    e.period,
    e.period > 4 AS overtime,
    e.sec_left_game,
    e.home_score - e.away_score AS margin,
    coalesce(e.home_has_ball, 0) AS home_has_ball,
    e.home_has_ball IS NULL AS ball_unknown,
    elo.elo_diff,
    g.home_win
FROM read_parquet('{curated}/events/*/*.parquet', hive_partitioning = true) e
JOIN read_parquet('{curated}/games/*.parquet') g USING (game_id)
JOIN elo USING (game_id)
WHERE e.season BETWEEN {lo} AND {hi}
