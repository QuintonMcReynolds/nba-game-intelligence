from nbagi.factcheck import check, player_mentions

FACTS = {
    "home": "BOS", "away": "NYK", "home_final": 112, "away_final": 104, "winner": "BOS",
    "home_max_lead": 15, "away_max_lead": 6,
    "players": [
        {"name": "J. Tatum", "surname": "Tatum", "team": "BOS", "pts": 31, "reb": 9, "ast": 5,
         "stl": 1, "blk": 0, "fg3m": 4},
        {"name": "J. Brunson", "surname": "Brunson", "team": "NYK", "pts": 28, "reb": 3,
         "ast": 7, "stl": 0, "blk": 0, "fg3m": 2},
        {"name": "J. Jackson Jr.", "surname": "Jackson Jr.", "team": "NYK", "pts": 12, "reb": 4,
         "ast": 1, "stl": 0, "blk": 0, "fg3m": 0},
        {"name": "G. Jackson", "surname": "Jackson", "team": "NYK", "pts": 6, "reb": 2,
         "ast": 0, "stl": 0, "blk": 0, "fg3m": 0},
    ],
}


def kinds(text, ok=None):
    return [(c.kind, c.ok) for c in check(text, FACTS).claims if ok is None or c.ok == ok]


def test_true_recap_has_no_errors():
    t = ("The Boston Celtics beat the New York Knicks 112-104. J. Tatum had 31 points and "
         "9 rebounds. Brunson scored 28 points with 7 assists. Boston led by 15.")
    assert kinds(t, ok=False) == []
    assert len(kinds(t)) >= 6


def test_wrong_winner_score_and_stat():
    t = "The Knicks defeated the Celtics 110-104. Tatum scored 35 points."
    bad = kinds(t, ok=False)
    assert ("winner", False) in bad and ("score", False) in bad and ("stat", False) in bad


def test_invented_player_and_lead():
    t = "Kevin Garnett added 20 points as Boston built a 30-point lead."
    bad = kinds(t, ok=False)
    assert ("player", False) in bad and ("lead", False) in bad


def test_suffix_and_shared_surnames():
    t = "J. Jackson Jr. scored 12 points and G. Jackson had 6 points."
    assert kinds(t, ok=False) == []


def test_team_name_is_not_a_player():
    facts = dict(FACTS, players=FACTS["players"] + [
        {"name": "P. Washington", "surname": "Washington", "team": "NYK", "pts": 3, "reb": 1,
         "ast": 0, "stl": 0, "blk": 0, "fg3m": 0}])
    r = check("The Boston Celtics beat New York 112-104 with 31 points from Tatum.", facts)
    assert all(c.ok for c in r.claims)


def test_longest_name_wins():
    m = player_mentions("j. jackson jr. scored", FACTS["players"])
    assert len(m) == 1 and m[0][1][0]["name"] == "J. Jackson Jr."


def test_winner_after_long_clause_and_passive():
    t = ("The New York Knicks, led by J. Brunson with 28 points and 7 assists, defeated the "
         "Boston Celtics 104-112.")
    assert ("winner", False) in kinds(t)
    assert ("winner", True) in kinds("The Knicks were defeated by the Celtics.")
    assert ("winner", True) in kinds("It was the Celtics' third straight win.")


def test_victory_noun_needs_adjacent_team():
    t = "Boston beat the Knicks to secure a 112-104 victory."
    assert [k for k in kinds(t) if k[0] == "winner"] == [("winner", True)]


def test_team_attribution():
    assert ("team", False) in kinds("J. Brunson was the standout player for the Celtics.")
    assert ("team", True) in kinds("The Knicks' top performer was J. Brunson.")


def test_possessive_is_not_an_invented_player():
    assert kinds("J. Brunson's 28 points kept New York close.", ok=False) == []


def test_lead_phrase_is_not_a_stat_line():
    assert [k for k in kinds("Tatum and the Celtics led by 6 points at the half.")
            if k[0] == "stat"] == []
