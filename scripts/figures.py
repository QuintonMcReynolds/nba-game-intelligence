"""README figures."""

import json
import sys

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nbagi import ARTIFACTS, DOCS  # noqa: E402
from nbagi.features import CURATED  # noqa: E402
from nbagi.recaps import TEAMS  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "lines.linewidth": 2,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(DOCS / name, dpi=160)
    plt.close(fig)


def season_label(s):
    return f"{s}-{str(s + 1)[2:]}"


def era():
    t = pd.read_csv(ARTIFACTS / "era_trends.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, col, title, fmt in [
        (axes[0], "three_point_rate", "Share of shots that are 3s", "{:.0%}"),
        (axes[1], "pace_est", "Possessions per team-game", "{:.0f}"),
        (axes[2], "home_win_rate", "Home win rate", "{:.0%}"),
    ]:
        ax.plot(t["season"], t[col], color=BLUE, marker="o", ms=3)
        for i in (0, len(t) - 1):
            ax.annotate(fmt.format(t[col].iloc[i]), (t["season"].iloc[i], t[col].iloc[i]),
                        xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9,
                        color=INK2)
        ax.set_title(title)
        ax.set_xticks([2004, 2012, 2020, 2025], [season_label(s) for s in (2004, 2012, 2020,
                                                                          2025)])
    axes[2].axhline(0.5, color=MUTED, lw=1)
    fig.suptitle("22 seasons of play-by-play (12.8M events), aggregated with Spark SQL / DuckDB",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    save(fig, "era_trends.png")


def calibration():
    c = pd.read_csv(ARTIFACTS / "wp_calibration.csv")
    m = json.loads((ARTIFACTS / "wp_metrics.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.3]})
    a1.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    a1.plot(c["predicted"], c["observed"], color=BLUE, marker="o", ms=5)
    a1.set_xlabel("Predicted home win probability")
    a1.set_ylabel("Observed home win rate")
    a1.set_title(f"Calibration, {m['selected_on_validation']} (2.4M test events)")
    a1.set_aspect("equal")
    bt = pd.DataFrame(m["by_time_left"]).T
    cols = [("logistic", BLUE, "logistic (scikit-learn)"),
            ("keras_mlp", AQUA, "neural net (TensorFlow)"),
            ("gbm_unconstrained", ORANGE, "boosted trees, unconstrained"),
            ("pregame_elo", MUTED, "pregame Elo only")]
    x = np.arange(len(bt))
    for k, (col, color, lab) in enumerate(cols):
        a2.bar(x + (k - 1.5) * 0.2, bt[col], width=0.18, color=color, label=lab)
    a2.set_xticks(x, [s.replace("4th: ", "4th\n") for s in bt.index], fontsize=8.5)
    a2.set_ylabel("Log loss (lower is better)")
    a2.set_title("Error by game situation, 2022-23 to 2025-26")
    a2.legend(fontsize=8, loc="upper right")
    a2.grid(axis="x", visible=False)
    save(fig, "wp_models.png")


def game_curve():
    """The 2025-26 game with the biggest comeback by win probability."""
    con = duckdb.connect()
    g = con.execute(f"""
        WITH w AS (
          SELECT game_id, home_win,
                 min(CASE WHEN home_win = 1 THEN wp ELSE 1 - wp END) AS winner_min_wp
          FROM read_parquet('{CURATED}/wp_test.parquet') WHERE season = 2025 AND period <= 4
          GROUP BY ALL)
        SELECT w.*, g.home_team, g.away_team, g.home_final, g.away_final
        FROM w JOIN read_parquet('{CURATED}/games/*.parquet') g USING (game_id)
        WHERE g.periods = 4 ORDER BY winner_min_wp LIMIT 1""").df().iloc[0]
    ev = con.execute(f"""
        SELECT action_number, period, sec_left, wp, wpa_home, description
        FROM read_parquet('{CURATED}/wpa_events.parquet') WHERE game_id = '{g.game_id}'
        ORDER BY action_number""").df()
    ev["minute"] = (2880 - ev["sec_left"]) / 60
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0.5, color=MUTED, lw=1)
    for q in (12, 24, 36):
        ax.axvline(q, color=GRID, lw=1)
    ax.plot(ev["minute"], ev["wp"], color=BLUE, lw=1.8)
    top = ev.reindex(ev["wpa_home"].abs().sort_values(ascending=False).index).head(3)
    for r in top.itertuples():
        ax.plot(r.minute, r.wp, "o", color=ORANGE, ms=7, mec=SURFACE, mew=1.5, zorder=4)
        label = r.description[:42] + ("…" if len(r.description) > 42 else "")
        ax.annotate(f"{label}\n{r.wpa_home:+.0%}", (r.minute, r.wp), xytext=(-8, 18 if
                    r.wp < 0.5 else -30), textcoords="offset points", ha="right", fontsize=8.5,
                    color=INK2)
    ax.set_xlim(0, 48)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 12, 24, 36, 48], ["Q1", "Q2", "Q3", "Q4", "end"])
    ax.set_ylabel(f"{TEAMS.get(g.home_team, g.home_team)} (home)\nwin probability")
    winner = g.home_team if g.home_win else g.away_team
    ax.set_title(f"Biggest comeback of 2025-26: {TEAMS.get(g.away_team)} {g.away_final}, "
                 f"{TEAMS.get(g.home_team)} {g.home_final} — {winner} won from "
                 f"{g.winner_min_wp:.1%}")
    save(fig, "wp_comeback.png")
    return g


def clutch():
    lb = pd.read_csv(ARTIFACTS / "clutch_leaderboard.csv")
    s = lb[lb["season"] == lb["season"].max()].head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([f"{p} ({t})" for p, t in zip(s["player"], s["team"])], s["clutch_wpa_shrunk"],
            color=BLUE, height=0.65)
    for i, (v, n) in enumerate(zip(s["clutch_wpa_shrunk"], s["clutch_events"])):
        ax.text(v + 0.1, i, f"{v:.1f}  ({n} plays)", va="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("Clutch wins added (last 5 min / OT, within 5 points), shrunk toward 0")
    ax.set_title("2025-26 clutch leaders by win probability added")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, s["clutch_wpa_shrunk"].max() * 1.25)
    save(fig, "clutch_2025.png")


def recap_ab():
    r = json.loads((ARTIFACTS / "recap_experiment.json").read_text())
    m = r["main"]
    k = pd.read_csv(ARTIFACTS / "recap_errors_by_kind.csv").set_index("arm")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.5]})
    arms = ["raw", "grounded"]
    rates = [m["any_error_raw"], m["any_error_grounded"]]
    ns = [m["games_raw"], m["games_grounded"]]
    ci = [1.96 * np.sqrt(p * (1 - p) / n) for p, n in zip(rates, ns)]
    a1.bar([0, 1], rates, yerr=ci, color=[ORANGE, BLUE], width=0.55, capsize=4,
           error_kw={"ecolor": INK2, "lw": 1})
    for i, v in enumerate(rates):
        a1.text(i, v + ci[i] + 0.02, f"{v:.0%}", ha="center", fontsize=10)
    a1.set_xticks([0, 1], ["A: raw play-by-play", "B: grounded facts"])
    a1.set_ylim(0, 1.12)
    a1.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    a1.set_ylabel("Recaps with ≥1 false claim")
    p_txt = "< 0.001" if m["p_value"] < 0.001 else f"= {m['p_value']:.3f}"
    a1.set_title(f"A/B test, {sum(ns)} games (p {p_txt})")
    a1.grid(axis="x", visible=False)
    kinds = ["err_winner", "err_score", "err_stat", "err_player", "err_lead", "err_margin"]
    names = ["wrong\nwinner", "wrong\nscore", "wrong\nstat", "invented\nplayer", "wrong\nlead",
             "wrong\nmargin"]
    x = np.arange(len(kinds))
    for j, (arm, c) in enumerate(zip(arms, [ORANGE, BLUE])):
        a2.bar(x + (j - 0.5) * 0.38, k.loc[arm, kinds], width=0.36, color=c,
               label={"raw": "A: raw", "grounded": "B: grounded"}[arm])
    a2.set_xticks(x, names, fontsize=9)
    a2.set_ylabel("False claims per recap")
    a2.set_title("What goes wrong")
    a2.legend()
    a2.grid(axis="x", visible=False)
    save(fig, "recap_ab.png")


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    which = sys.argv[1:] or ["era", "calibration", "game", "clutch", "recap"]
    if "era" in which:
        era()
    if "calibration" in which:
        calibration()
    if "game" in which:
        print(game_curve().to_dict())
    if "clutch" in which:
        clutch()
    if "recap" in which:
        recap_ab()
