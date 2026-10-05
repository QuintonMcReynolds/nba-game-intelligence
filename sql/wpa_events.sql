-- Win probability added (WPA) per event, home perspective, with the player credited.
-- The first event of a game starts from the pregame-state prediction; the last
-- event is followed by the final result (1 or 0).
WITH p AS (
    SELECT w.*, e.person_id, e.player_name_i, e.team, e.is_home_team, e.event,
           e.description, e.home_score, e.away_score,
           lag(w.wp) OVER g AS wp_before,
           lag(w.margin) OVER g AS margin_before,
           lead(w.wp) OVER g IS NULL AS last_event
    FROM read_parquet('{curated}/wp_test.parquet') w
    JOIN read_parquet('{curated}/events/*/*.parquet', hive_partitioning = true) e
      USING (game_id, action_number)
    WINDOW g AS (PARTITION BY w.game_id ORDER BY w.action_number)
)
SELECT *,
    CASE WHEN last_event THEN home_win ELSE wp END
        - coalesce(wp_before, wp) AS wpa_home,
    (period >= 4 AND sec_left <= {clutch_sec}) OR period > 4 AS late,
    abs(coalesce(margin_before, 0)) <= {clutch_margin} AS close
FROM p
