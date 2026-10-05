"""WPA for every test-season event, the clutch leaderboard, and SQL era trends."""

import json

import duckdb
import numpy as np

from nbagi import ARTIFACTS, ROOT, load_config
from nbagi.features import CURATED
from nbagi.wpa import clutch_leaderboard, wpa_events


def main():
    cfg = load_config()
    ev = wpa_events(cfg)
    # sanity: WPA sums to (result - first prediction) in every game
    ev = ev.sort_values(["game_id", "action_number"])
    per_game = ev.groupby("game_id").agg(s=("wpa_home", "sum"), y=("home_win", "first"),
                                         p0=("wp", "first"))
    resid = np.abs(per_game["s"] - (per_game["y"] - per_game["p0"])).max()
    ev[["season", "game_id", "action_number", "person_id", "player_name_i", "team", "event",
        "description", "period", "sec_left", "margin", "wp", "wpa_home", "wpa_player",
        "late", "close"]].to_parquet(CURATED / "wpa_events.parquet", index=False)

    lb = clutch_leaderboard(ev, cfg)
    lb.to_csv(ARTIFACTS / "clutch_leaderboard.csv", index=False)
    last = lb[lb["season"] == lb["season"].max()]
    raw_top = last.sort_values("clutch_wpa", ascending=False).head(10)
    shr_top = last.head(10)

    trends = duckdb.sql((ROOT / "sql" / "era_trends.sql").read_text().format(
        curated=CURATED)).df()
    trends.to_csv(ARTIFACTS / "era_trends.csv", index=False)

    out = {"max_wpa_identity_error": float(resid),
           "clutch_events": int((ev["late"] & ev["close"]).sum()),
           "players_ranked": len(lb),
           "top10_overlap_raw_vs_shrunk": len(set(raw_top["person_id"])
                                              & set(shr_top["person_id"])),
           "median_shrinkage": float(lb["shrinkage"].median())}
    (ARTIFACTS / "clutch.json").write_text(json.dumps(out, indent=2))
    print(out)
    print(shr_top[["player", "team", "clutch_events", "clutch_wpa", "clutch_wpa_shrunk",
                   "shrinkage"]].round(3).to_string(index=False))
    print(trends.iloc[[0, 5, 10, 15, 21]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
