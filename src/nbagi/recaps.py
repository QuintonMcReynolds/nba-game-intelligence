"""Game facts from the lake, two prompt designs, local LLM generation, and a template control.

Arm A ("raw"): the final score and the last plays of the play-by-play log, the way a
            quick prototype would prompt a model.
Arm B ("grounded"): a structured fact sheet built by the pipeline (box score leaders,
            largest leads, the win-probability swings that decided the game).
"""

from __future__ import annotations

import duckdb
import pandas as pd

from .features import CURATED

TEAMS = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

SYSTEM = ("You are an NBA beat writer. Write a short, accurate game recap. Never invent "
          "names, numbers or events that are not in the information provided.")
TASK = ("Write a 4-sentence recap of this game: who won and the final score, the top "
        "performers with their stats, and the stretch that decided it.")


def clock(sec_left_period: float) -> str:
    m, s = divmod(int(round(sec_left_period)), 60)
    return f"{m}:{s:02d}"


def quarter(period: int) -> str:
    return {1: "1st quarter", 2: "2nd quarter", 3: "3rd quarter", 4: "4th quarter"}.get(
        period, f"overtime {period - 4}" if period > 5 else "overtime")


def season_facts(season: int) -> dict[str, dict]:
    """Everything a recap may say about each game, keyed by game_id."""
    con = duckdb.connect()
    games = con.execute(f"""
        SELECT * FROM read_parquet('{CURATED}/games/*.parquet') WHERE season = {season}
    """).df()
    pg = con.execute(f"""
        SELECT * FROM read_parquet('{CURATED}/player_games/*/*.parquet',
                                   hive_partitioning = true) WHERE season = {season}
    """).df()
    ev = con.execute(f"""
        SELECT game_id, action_number, period, sec_left_period, home_score, away_score,
               team, description, event
        FROM read_parquet('{CURATED}/events/*/*.parquet', hive_partitioning = true)
        WHERE season = {season}
    """).df()
    wpa = con.execute(f"""
        SELECT game_id, action_number, wp, wpa_home
        FROM read_parquet('{CURATED}/wpa_events.parquet') WHERE season = {season}
    """).df()
    ev = ev.merge(wpa, on=["game_id", "action_number"], how="left").sort_values(
        ["game_id", "action_number"])
    pts = pg.groupby(["game_id", "team_id"])["pts"].sum()

    out = {}
    for g in games.itertuples(index=False):
        if g.home_team not in TEAMS or g.away_team not in TEAMS:
            continue
        # skip games whose play-by-play doesn't reconcile with the final score
        if pts.get((g.game_id, g.home_team_id)) != g.home_final or \
                pts.get((g.game_id, g.away_team_id)) != g.away_final:
            continue
        e = ev[ev["game_id"] == g.game_id]
        p = pg[pg["game_id"] == g.game_id]
        margin = e["home_score"] - e["away_score"]
        players = []
        for r in p.itertuples(index=False):
            players.append({"name": r.player_name_i, "surname": r.player_name, "team": r.team,
                            "pts": int(r.pts), "reb": int(r.reb), "ast": int(r.ast),
                            "stl": int(r.stl), "blk": int(r.blk), "fg3m": int(r.fg3m)})
        swings = e.dropna(subset=["wpa_home"]).reindex(
            e["wpa_home"].abs().sort_values(ascending=False).index).head(3)
        home_won = g.home_final > g.away_final
        out[g.game_id] = {
            "game_id": g.game_id,
            "home": g.home_team, "away": g.away_team,
            "home_name": TEAMS[g.home_team], "away_name": TEAMS[g.away_team],
            "home_final": int(g.home_final), "away_final": int(g.away_final),
            "winner": g.home_team if home_won else g.away_team,
            "overtime": int(g.periods) > 4,
            "home_max_lead": int(max(margin.max(), 0)),
            "away_max_lead": int(max(-margin.min(), 0)),
            "players": players,
            "swings": [{"when": f"{quarter(int(s.period))}, {clock(s.sec_left_period)} left",
                        "play": s.description, "score": f"{int(s.away_score)}-{int(s.home_score)}",
                        "wp_home_after": round(float(s.wp), 2)}
                       for s in swings.itertuples(index=False)],
            "last_plays": [f"Q{int(r.period)} {clock(r.sec_left_period)} "
                           f"[{r.team if isinstance(r.team, str) else '-'}] {r.description} "
                           f"({int(r.away_score)}-{int(r.home_score)})"
                           for r in e.tail(40).itertuples(index=False)
                           if r.event not in ("Substitution", "period", "Timeout")],
        }
    return out


def leaders(f: dict, team: str, k: int = 3) -> list[dict]:
    ps = [p for p in f["players"] if p["team"] == team]
    return sorted(ps, key=lambda p: (-p["pts"], -p["reb"]))[:k]


def header(f: dict) -> str:
    return (f"{f['away_name']} ({f['away']}) at {f['home_name']} ({f['home']}). "
            f"Final: {f['away_name']} {f['away_final']}, {f['home_name']} {f['home_final']}"
            + (" (overtime)." if f["overtime"] else "."))


def prompt_raw(f: dict, n_lines: int) -> str:
    log = "\n".join(f["last_plays"][-n_lines:])
    return f"{header(f)}\n\nPlay-by-play, final minutes (score shown away-home):\n{log}\n\n{TASK}"


def prompt_grounded(f: dict) -> str:
    lines = [header(f), "", "Top performers:"]
    for t in (f["away"], f["home"]):
        for p in leaders(f, t):
            lines.append(f"- {p['name']} ({TEAMS[t]}): {p['pts']} points, {p['reb']} rebounds, "
                         f"{p['ast']} assists")
    lines += ["", f"Largest lead: {f['home_name']} {f['home_max_lead']} points, "
                  f"{f['away_name']} {f['away_max_lead']} points.", "",
              "Plays that swung the win probability most (score away-home):"]
    for s in f["swings"]:
        lines.append(f"- {s['when']}: {s['play']} ({s['score']})")
    return "\n".join(lines + ["", TASK])


def template_recap(f: dict) -> str:
    """Deterministic control: every claim is true by construction (validates the checker)."""
    w, l_ = (f["home"], f["away"]) if f["winner"] == f["home"] else (f["away"], f["home"])
    ws = f["home_final"] if w == f["home"] else f["away_final"]
    ls = f["away_final"] if w == f["home"] else f["home_final"]
    a, b = leaders(f, w, 1)[0], leaders(f, l_, 1)[0]
    lead = f["home_max_lead"] if w == f["home"] else f["away_max_lead"]
    return (f"The {TEAMS[w]} beat the {TEAMS[l_]} {ws}-{ls}. "
            f"{a['name']} led the {TEAMS[w]} with {a['pts']} points, {a['reb']} rebounds and "
            f"{a['ast']} assists. {b['name']} scored {b['pts']} points for the {TEAMS[l_]}. "
            f"The {TEAMS[w]} led by as many as {lead} points.")


class Generator:
    """Greedy decoding with a small local instruction model (PyTorch, Apple GPU if present)."""

    def __init__(self, model_name: str, max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16 if self.device == "mps" else torch.float32
        ).to(self.device).eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, repetition_penalty=1.05)
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()


def facts_frame(facts: dict) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in f.items() if k not in ("players", "swings",
                                                                  "last_plays")}
                         for f in facts.values()])
