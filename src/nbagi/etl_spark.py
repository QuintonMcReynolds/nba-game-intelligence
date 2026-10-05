"""PySpark ETL: raw play-by-play -> curated events, games, player box scores, quality report.

Runs locally (`python scripts/etl.py`) or unchanged as an AWS Glue job with S3 paths
(`cloud/template.yaml`). Raw quirks handled here:

* scores are only trustworthy on scoring rows (others are blank, 0-0 before 2017-18, or junk
  on period markers) -> taken from made shots / free throws and forward-filled
* `shotValue` only exists from 2024-25 -> points come from score deltas
* blocks and steals have a blank actionType -> identified from the description text
* assists exist only as text, "(James 2 AST)" -> parsed and matched to the game roster
* actionType has trailing spaces in some seasons
* actionNumber repeats for linked events (turnover + steal) -> actionId is the key
* possession is not recorded -> inferred from made shots, rebounds and turnovers
"""

from __future__ import annotations

import argparse

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

GAME_SECONDS = 4 * 720


def spark_session(local_dir: str | None = None) -> SparkSession:
    b = (SparkSession.builder.appName("nba-pbp-etl")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.ui.enabled", "false")
         .config("spark.sql.session.timeZone", "UTC"))
    if local_dir:  # local runs: bounded memory and a scratch dir inside the repo
        b = (b.master("local[4]").config("spark.driver.memory", "2g")
             .config("spark.local.dir", local_dir))
    return b.getOrCreate()


def clean(raw: DataFrame) -> DataFrame:
    """Typed, trimmed events with game clock, filled scores and parsed text fields."""
    game = Window.partitionBy("game_id").orderBy("action_number")
    upto = game.rowsBetween(Window.unboundedPreceding, 0)
    clock = F.regexp_extract("clock", r"PT(\d+)M([\d.]+)S", 1).try_cast("int") * 60 + \
        F.regexp_extract("clock", r"PT(\d+)M([\d.]+)S", 2).try_cast("double")
    desc = F.trim(F.col("description"))
    action = F.trim(F.col("actionType"))
    df = (raw.select(
        F.col("season").try_cast("int"),
        F.lpad("gameId", 10, "0").alias("game_id"),
        # actionId is the unique, chronological key; actionNumber is shared by linked events
        # (a turnover and its steal, a miss and its block) and must not be used as a key
        F.col("actionId").try_cast("int").alias("action_number"),
        F.col("period").try_cast("int"),
        clock.alias("sec_left_period"),
        F.when(F.col("teamId").try_cast("long") > 0, F.col("teamId").try_cast("long"))
        .alias("team_id"),
        F.nullif(F.col("teamTricode"), F.lit("")).alias("team"),
        F.when(F.col("personId").try_cast("long") > 0, F.col("personId").try_cast("long"))
        .alias("person_id"),
        F.nullif(F.col("playerName"), F.lit("")).alias("player_name"),
        F.nullif(F.col("playerNameI"), F.lit("")).alias("player_name_i"),
        F.nullif(F.col("location"), F.lit("")).alias("location"),
        desc.alias("description"),
        F.when(action == "", F.lit(None)).otherwise(action).alias("action_type"),
        F.trim("subType").alias("sub_type"),
        F.nullif(F.col("scoreHome"), F.lit("")).try_cast("int").alias("score_home_raw"),
        F.nullif(F.col("scoreAway"), F.lit("")).try_cast("int").alias("score_away_raw"),
        F.nullif(F.col("shotDistance"), F.lit("")).try_cast("int").alias("shot_distance"),
    ).dropDuplicates(["game_id", "action_number"]))

    # only scoring events update the score. Before 2017-18 non-scoring rows carry "0","0", and
    # period-marker rows ("End of 3rd Period") carry unrelated numbers in every season.
    scoring = F.col("action_type").isin("Made Shot", "Free Throw")
    zero = ~scoring | ((F.coalesce("score_home_raw", F.lit(0))
                        + F.coalesce("score_away_raw", F.lit(0))) == 0)
    df = (df
          .withColumn("score_home_raw", F.when(~zero, F.col("score_home_raw")))
          .withColumn("score_away_raw", F.when(~zero, F.col("score_away_raw")))
          .withColumn("home_score", F.coalesce(
              F.last("score_home_raw", ignorenulls=True).over(upto), F.lit(0)))
          .withColumn("away_score", F.coalesce(
              F.last("score_away_raw", ignorenulls=True).over(upto), F.lit(0)))
          .withColumn("prev_total", F.lag(F.col("home_score") + F.col("away_score"), 1, 0)
                      .over(game))
          .withColumn("points", F.col("home_score") + F.col("away_score") - F.col("prev_total"))
          .drop("score_home_raw", "score_away_raw", "prev_total"))

    is_ft = F.col("action_type") == "Free Throw"
    df = (df
          .withColumn("event", F.coalesce(
              F.col("action_type"),
              F.when(F.col("description").contains("STEAL"), "Steal")
               .when(F.col("description").contains("BLOCK"), "Block"),
              F.lit("Other")))
          .withColumn("is_three", F.col("description").rlike(r"\b3PT\b"))
          .withColumn("ft_made", is_ft & ~F.col("description").startswith("MISS"))
          .withColumn("ft_last", is_ft & F.col("sub_type").rlike(r"(\d) of \1$"))
          .withColumn("assist_name", F.nullif(F.regexp_extract(
              "description", r"\(([A-Za-z.' \-]+?) \d+ AST\)", 1), F.lit("")))
          .withColumn("offensive_rebound", (F.col("event") == "Rebound")
                      & F.col("description").rlike(r"Off:\d+ Def:\d+")
                      & F.col("description").rlike(r"REBOUND \(Off:(\d+) Def:\d+\)")
                      & (F.regexp_extract("description", r"Off:(\d+) Def:(\d+)", 1).try_cast("int")
                         > F.lag(F.regexp_extract("description", r"Off:(\d+)", 1).try_cast("int"),
                                 1).over(Window.partitionBy("game_id", "person_id")
                                         .orderBy("action_number")))))
    # game clock in seconds left in regulation; overtime periods count down from 300
    df = df.withColumn("sec_left_game", F.when(
        F.col("period") <= 4,
        (4 - F.col("period")) * 720 + F.col("sec_left_period")).otherwise(
        F.col("sec_left_period")))
    return df


def build_games(ev: DataFrame) -> DataFrame:
    teams = (ev.where(F.col("team_id").isNotNull() & F.col("location").isin("h", "v"))
             .groupBy("game_id", "location")
             .agg(F.first("team_id").alias("tid"), F.first("team").alias("tri")))
    home = teams.where("location = 'h'").select("game_id", F.col("tid").alias("home_team_id"),
                                                F.col("tri").alias("home_team"))
    away = teams.where("location = 'v'").select("game_id", F.col("tid").alias("away_team_id"),
                                                F.col("tri").alias("away_team"))
    fin = ev.groupBy("game_id", "season").agg(
        F.max("home_score").alias("home_final"), F.max("away_score").alias("away_final"),
        F.max("period").alias("periods"), F.count("*").alias("n_events"))
    return (fin.join(home, "game_id").join(away, "game_id")
            .withColumn("home_win", (F.col("home_final") > F.col("away_final")).cast("int")))


def add_possession(ev: DataFrame, games: DataFrame) -> DataFrame:
    """Home team has the ball (1/0), inferred event by event and carried forward."""
    g = games.select("game_id", "home_team_id", "away_team_id")
    ev = ev.join(g, "game_id")
    other_is_home = F.col("team_id") == F.col("away_team_id")
    own_is_home = F.col("team_id") == F.col("home_team_id")
    after = (F.when((F.col("event") == "Made Shot"), other_is_home)
             .when(F.col("ft_last") & F.col("ft_made"), other_is_home)
             .when(F.col("event") == "Rebound", own_is_home)
             .when(F.col("event") == "Turnover", other_is_home))
    w = Window.partitionBy("game_id").orderBy("action_number").rowsBetween(
        Window.unboundedPreceding, 0)
    return (ev.withColumn("poss_after", F.when(F.col("team_id").isNotNull(), after))
            .withColumn("home_has_ball", F.last("poss_after", ignorenulls=True).over(w)
                        .cast("int"))
            .withColumn("is_home_team", own_is_home.cast("int"))
            .drop("poss_after"))


def build_player_games(ev: DataFrame) -> DataFrame:
    p = ev.where(F.col("person_id").isNotNull() & F.col("team_id").isNotNull())
    fga = F.col("event").isin("Made Shot", "Missed Shot")
    stats = p.groupBy("season", "game_id", "team_id", "team", "person_id").agg(
        F.first("player_name", ignorenulls=True).alias("player_name"),
        F.first("player_name_i", ignorenulls=True).alias("player_name_i"),
        F.sum(F.when(F.col("event").isin("Made Shot", "Free Throw"), F.col("points"))
              .otherwise(0)).alias("pts"),
        F.sum((F.col("event") == "Made Shot").cast("int")).alias("fgm"),
        F.sum(fga.cast("int")).alias("fga"),
        F.sum(((F.col("event") == "Made Shot") & F.col("is_three")).cast("int")).alias("fg3m"),
        F.sum((fga & F.col("is_three")).cast("int")).alias("fg3a"),
        F.sum(F.col("ft_made").cast("int")).alias("ftm"),
        F.sum((F.col("event") == "Free Throw").cast("int")).alias("fta"),
        F.sum((F.col("event") == "Rebound").cast("int")).alias("reb"),
        F.sum((F.col("event") == "Steal").cast("int")).alias("stl"),
        F.sum((F.col("event") == "Block").cast("int")).alias("blk"),
        F.sum((F.col("event") == "Turnover").cast("int")).alias("tov"),
    )
    # assists: match the parsed name to a teammate in the same game (surname or "A. Name")
    roster = stats.select("game_id", "team_id", "person_id", "player_name", "player_name_i")
    a = p.where(F.col("assist_name").isNotNull()).select(
        "game_id", "team_id", "action_number", "assist_name")
    m = a.join(roster.withColumnRenamed("person_id", "assister"), ["game_id", "team_id"]).where(
        (F.col("assist_name") == F.col("player_name"))
        | (F.col("assist_name") == F.col("player_name_i")))
    uniq = m.groupBy("game_id", "action_number").agg(
        F.count("*").alias("k"), F.first("assister").alias("assister"))
    ast = (uniq.where("k = 1").groupBy("game_id", "assister").agg(F.count("*").alias("ast"))
           .withColumnRenamed("assister", "person_id"))
    return stats.join(ast, ["game_id", "person_id"], "left").fillna({"ast": 0})


def quality(ev: DataFrame, games: DataFrame, pg: DataFrame, raw_n: int) -> dict:
    team_pts = pg.groupBy("game_id", "team_id").agg(F.sum("pts").alias("pts"))
    g = games.alias("g")
    hp = team_pts.alias("h")
    chk = (g.join(hp, (F.col("g.game_id") == F.col("h.game_id"))
                  & (F.col("g.home_team_id") == F.col("h.team_id")))
           .where(F.col("h.pts") != F.col("g.home_final")).count())
    neg = ev.where(F.col("points") < 0).count()
    assists_text = ev.where(F.col("assist_name").isNotNull()).count()
    assists_matched = pg.agg(F.sum("ast")).first()[0]
    per_season = {r["season"]: r["n"] for r in
                  games.groupBy("season").agg(F.count("*").alias("n")).collect()}
    return {
        "raw_rows": raw_n,
        "events": ev.count(),
        "games": games.count(),
        "games_per_season": dict(sorted(per_season.items())),
        "games_home_points_mismatch": chk,
        "events_negative_score_delta": neg,
        "assists_in_text": assists_text,
        "assists_matched_to_player": int(assists_matched),
        "ties_at_final": games.where("home_final = away_final").count(),
    }


def run(spark: SparkSession, src: str, dst: str, seasons: list[int] | None = None) -> dict:
    """Process one season at a time (bounded shuffle; same shape as an incremental daily job),
    appending to season-partitioned outputs, then compute the quality report over everything."""
    raw_all = spark.read.option("basePath", f"{src}/raw_pbp").parquet(f"{src}/raw_pbp")
    if seasons is None:
        seasons = sorted(r[0] for r in raw_all.select("season").distinct().collect())
    for i, season in enumerate(seasons):
        mode = "overwrite" if i == 0 else "append"
        raw = raw_all.where(F.col("season") == season)
        ev = clean(raw).cache()
        games = build_games(ev)
        games.write.mode(mode).parquet(f"{dst}/games")
        ev = add_possession(ev, spark.read.parquet(f"{dst}/games")
                            .where(F.col("season") == season))
        ev.write.mode(mode).partitionBy("season").parquet(f"{dst}/events")
        ev.unpersist()
        sev = spark.read.parquet(f"{dst}/events").where(F.col("season") == season)
        build_player_games(sev).write.mode(mode).partitionBy("season").parquet(
            f"{dst}/player_games")
        print(f"  season {season} done", flush=True)
    ev = spark.read.parquet(f"{dst}/events")
    games = spark.read.parquet(f"{dst}/games")
    pg = spark.read.parquet(f"{dst}/player_games")
    return quality(ev, games, pg, raw_all.count())


if __name__ == "__main__":  # entry point for AWS Glue / spark-submit
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args, _ = ap.parse_known_args()
    print(run(spark_session(), args.input, args.output))
