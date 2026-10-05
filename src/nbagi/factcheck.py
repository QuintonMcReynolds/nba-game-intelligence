"""Rule-based NLP fact checker for game recaps.

Extracts checkable claims from free text and verifies each against the game's facts:

* score      "112-104", "112 to 104"          -> must be the final score
* winner     "<team> ... beat/defeated", "<team> fell to", "<team>'s win" -> must match
* team       "<player> ... for the <team>", "<team>'s top scorer was <player>" -> roster
* stat       "<player> ... 31 points / 9 rebounds / 7 assists"  -> must match the box score
* player     any capitalized name that matches no one in the game but looks like a player
             reference ("X scored") -> invented player
* lead       "led by 15", "a 15-point lead/deficit" -> no team ever led by that much

Claims it can't parse are ignored, so the error rate is a lower bound. The template
control (true by construction) measures false alarms.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .recaps import TEAMS

STAT_WORDS = {
    "points": "pts", "point": "pts", "pts": "pts",
    "rebounds": "reb", "rebound": "reb", "boards": "reb",
    "assists": "ast", "assist": "ast", "dimes": "ast",
    "steals": "stl", "steal": "stl", "blocks": "blk", "block": "blk",
    "three-pointers": "fg3m", "3-pointers": "fg3m", "threes": "fg3m", "3s": "fg3m",
}
WIN_VERBS = r"(?:beat|beats|defeated|topped|edged|outlasted|held off|routed|downed|" \
            r"knocked off|cruised past|rallied past|rallied to beat|pulled away from|" \
            r"overcame|won)"
LOSE_VERBS = r"(?:lost to|fell to|were defeated|was defeated|were beaten|was beaten|lost|fell|" \
             r"dropped)"
WIN_NOUNS = r"(?:win|victory)"     # count only when the team sits right before the noun,
LOSE_NOUNS = r"(?:loss|defeat)"    # e.g. "the Bucks' first win"
# a number + "point(s)" followed by one of these describes the game, not a player's stat line
MARGIN_WORDS = r"(?:lead|deficit|comeback|hole|margin|run|down|advantage|cushion|swing)"
RESULT_WORDS = r"(?:win|victory|loss|defeat|blowout|rout)"
NOT_NAMES = {"the", "this", "a", "an", "in", "with", "after", "despite", "while", "their",
             "his", "game", "key", "both", "top", "players", "player", "fourth", "third",
             "second", "first", "quarter", "nba", "monday", "tuesday", "wednesday", "thursday",
             "friday", "saturday", "sunday", "january", "february", "march", "april",
             "october", "november", "december", "overtime", "however", "meanwhile",
             # play-by-play vocabulary the model copies from the log
             "offensive", "defensive", "foul", "turnover", "rebound", "jump", "shot", "free",
             "throw", "layup", "dunk", "pullup", "driving", "running", "step", "back",
             "floating", "reverse", "cutting", "tip", "hook", "fadeaway", "alley", "oop",
             "personal", "take", "shooting", "bad", "pass", "lost", "ball", "steal", "block",
             "miss", "timeout", "violation", "charge", "finals", "conference", "western",
             "eastern", "playoffs"}
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
             "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
             "nineteen": 19, "twenty": 20}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower()


@dataclass
class Claim:
    kind: str
    text: str
    ok: bool
    detail: str = ""


@dataclass
class Report:
    claims: list[Claim] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.claims)

    @property
    def errors(self) -> int:
        return sum(not c.ok for c in self.claims)

    def has(self, kind: str, ok: bool = True) -> bool:
        return any(c.kind == kind and c.ok == ok for c in self.claims)


def team_aliases(code: str) -> list[str]:
    full = TEAMS[code]
    parts = full.split()
    nick = " ".join(parts[2:]) if full.startswith(("Los Angeles", "Golden State", "New York",
                                                   "New Orleans", "Oklahoma City",
                                                   "San Antonio")) else parts[-1]
    if code == "POR":
        nick = "Trail Blazers"
    city = full[: -len(nick)].strip()
    al = {full, nick, code}
    if city and city not in ("Los Angeles", "LA", "New York"):  # ambiguous cities
        al.add(city)
    if code == "POR":
        al.add("Blazers")
    if code == "LAC":
        al.add("Los Angeles Clippers")
    if code == "PHI":
        al.add("Sixers")
    return sorted(al, key=len, reverse=True)


def find_team(text: str, f: dict) -> list[tuple[int, int, str]]:
    """Non-overlapping mentions of either team: [(start, end, code)], longest alias wins."""
    hits = []
    for code in (f["home"], f["away"]):
        for a in team_aliases(code):
            for m in re.finditer(rf"\b{re.escape(a)}\b", text):
                hits.append((m.start(), m.end(), code))
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    out, covered = [], -1
    for h in hits:
        if h[0] >= covered:
            out.append(h)
            covered = h[1]
    return out


def winner_claims(s: str, f: dict) -> list[tuple[str, str]]:
    """[(claimed winner code, evidence)] from one sentence."""
    teams = find_team(s, f)
    other = {f["home"]: f["away"], f["away"]: f["home"]}
    out = []
    events = [(m.start(), m.group(0), "lose") for m in re.finditer(rf"\b{LOSE_VERBS}\b", s)]
    lose_spans = [(m.start(), m.end()) for m in re.finditer(rf"\b{LOSE_VERBS}\b", s)]
    for m in re.finditer(rf"\b{WIN_VERBS}\b", s):
        if not any(a <= m.start() < b for a, b in lose_spans):  # "were defeated" is a loss
            events.append((m.start(), m.group(0), "win"))
    for pos, word, kind in events:
        before = [t for t in teams if t[1] <= pos]
        if before:
            t = before[-1]
            claimed = t[2] if kind == "win" else other[t[2]]
            out.append((claimed, s[t[0]:pos + len(word)]))
    for m in re.finditer(rf"\b{WIN_NOUNS}|{LOSE_NOUNS}\b", s):
        # only "the Bucks' third straight win" or "a Bucks win": possessive or adjacent
        before = [t for t in teams if t[1] <= m.start() and (
            re.fullmatch(r"(?:'s?|’s?)\s+(?:[a-z-]+\s+){0,3}", s[t[1]:m.start()])
            or re.fullmatch(r"\s", s[t[1]:m.start()]))]
        if before:
            t = before[-1]
            won = re.match(WIN_NOUNS, m.group(0)) is not None
            out.append((t[2] if won else other[t[2]], s[t[0]:m.end()]))
    return out


def team_claims(s: str, ns_masked: str, f: dict, roster: list[dict]) -> list[Claim]:
    """Player-team attributions: "X ... for the Celtics", "X led the Celtics",
    "the Celtics' top performer was X"."""
    out = []
    mentions = player_mentions(ns_masked, roster)
    for start, end, code in find_team(s, f):
        pre = s[max(0, start - 12):start].lower()
        if re.search(r"\b(?:for|led) the\s*$", pre):
            prev = [m for m in mentions if m[0] < start]
            if prev:
                ps = prev[-1][1]
                out.append(Claim("team", f"{ps[0]['name']} -> {code}",
                                 any(p["team"] == code for p in ps), f"plays for "
                                 f"{sorted({p['team'] for p in ps})}"))
        post = s[end:end + 40]
        mm = re.match(r"(?:'s?|’s?) (?:top (?:performer|scorer)|leading scorer) was", post)
        if mm:
            nxt = [m for m in mentions if m[0] >= end + mm.end()]
            if nxt and nxt[0][0] - (end + mm.end()) <= 3:
                ps = nxt[0][1]
                out.append(Claim("team", f"{ps[0]['name']} -> {code}",
                                 any(p["team"] == code for p in ps), f"plays for "
                                 f"{sorted({p['team'] for p in ps})}"))
    return out


def player_mentions(text: str, roster: list[dict]) -> list[tuple[int, list[dict]]]:
    """[(position, candidate players)] for each player reference in normalized text.

    "J. Jackson Jr." resolves to exactly that player; a bare surname maps to everyone in the
    game with it. Overlapping matches keep the longest (so "Jackson Jr." beats "Jackson")."""
    keys: dict[str, list[dict]] = {}
    for p in roster:
        keys.setdefault(norm(p["name"]), []).append(p)        # "j. jackson jr."
        keys.setdefault(norm(p["surname"]), []).append(p)     # "jackson jr."
    spans = []
    for key, ps in keys.items():
        for m in re.finditer(rf"(?<![\w.]){re.escape(key)}(?!\w)", text):
            spans.append((m.start(), m.end(), ps))
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    out, covered_to = [], -1
    for start, end, ps in spans:
        if start < covered_to:  # inside a longer match already taken
            continue
        # a bare surname directly after an initial belongs to that initial's match
        out.append((start, ps))
        covered_to = end
    return out


def sentences(text: str) -> list[str]:
    # don't break after an initial ("J. Brunson") or a suffix ("Jackson Jr. scored")
    parts = re.split(r"(?<=[.!?])(?<! [A-Z]\.)(?<!^[A-Z]\.)(?<!Jr\.)(?<!Sr\.)\s+(?=[A-Z\"'])",
                     text)
    return [s.strip() for s in parts if s.strip()]


def check(text: str, f: dict) -> Report:
    rep = Report()
    final = {f["home_final"], f["away_final"]}
    roster = f["players"]
    by_surname: dict[str, list[dict]] = {}
    for p in roster:
        by_surname.setdefault(norm(p["surname"]), []).append(p)
    max_lead = max(f["home_max_lead"], f["away_max_lead"])
    surnames = set(by_surname)

    for s in sentences(text):
        ns = norm(s)
        # --- scores: two numbers 70-200 joined by a dash or "to"
        for m in re.finditer(r"\b(\d{2,3})\s*(?:-|–|—|to)\s*(\d{2,3})\b", s):
            a, b = int(m.group(1)), int(m.group(2))
            if 60 <= a <= 200 and 60 <= b <= 200:
                rep.claims.append(Claim("score", m.group(0), {a, b} == final and a != b or
                                        (a == b and len(final) == 1),
                                        f"final {sorted(final, reverse=True)}"))
        # --- winner / loser
        for claimed, evidence in winner_claims(s, f):
            rep.claims.append(Claim("winner", evidence[-60:], claimed == f["winner"],
                                    f"winner {f['winner']}"))
        # --- player stats: number + stat word, attributed to the nearest preceding player
        # mask team names first so "Washington Wizards" is not read as P. Washington
        masked = ns
        for code in (f["home"], f["away"]):
            for a in team_aliases(code):
                masked = re.sub(rf"\b{re.escape(norm(a))}\b", lambda m: "#" * len(m.group(0)),
                                masked)
        mentions = player_mentions(masked, roster)
        rep.claims.extend(team_claims(s, masked, f, roster))
        num = r"(\d{1,2}|" + "|".join(NUM_WORDS) + r")"
        for m in re.finditer(rf"\b{num}[- ](?:{'|'.join(map(re.escape, STAT_WORDS))})\b", ns):
            raw_n, word = m.group(0).replace("-", " ").split(" ", 1)
            n = int(raw_n) if raw_n.isdigit() else NUM_WORDS[raw_n]
            if n == 3 and word.strip() == "point":  # "3-point shot", not a 3-point stat line
                continue
            after = ns[m.end():m.end() + 14]
            if re.search(r"\b(?:led|lead|up|down|trailed|ahead|behind) by\s*$", ns[:m.start()]):
                continue  # "led by 6 points" is a lead, checked below, not a stat line
            if STAT_WORDS[word.strip()] == "pts" and re.match(rf"\s*{MARGIN_WORDS}", after):
                rep.claims.append(Claim("lead", f"{m.group(0)} {after.split()[0]}", n <= max_lead,
                                        f"largest lead {max_lead}"))
                continue
            if STAT_WORDS[word.strip()] == "pts" and re.match(rf"\s*{RESULT_WORDS}", after):
                rep.claims.append(Claim("margin", f"{m.group(0)} {after.split()[0]}",
                                        n == abs(f["home_final"] - f["away_final"]),
                                        f"final margin {abs(f['home_final'] - f['away_final'])}"))
                continue
            prev = [x for x in mentions if x[0] < m.start()]
            if not prev:
                continue
            ps = prev[-1][1]
            stat = STAT_WORDS[word.strip()]
            ok = any(p[stat] == n for p in ps)
            rep.claims.append(Claim("stat", f"{ps[0]['name']}: {m.group(0)}", ok,
                                    f"actual {[p[stat] for p in ps]}"))
        # --- invented players: "J. Name" or "First Last" whose surname nobody in the game has
        team_words = {norm(w) for c in (f["home"], f["away"]) for a in team_aliases(c)
                      for w in a.split()}
        for m in re.finditer(r"\b(?:([A-Z])\. ?|([A-Z][a-zA-Z]+) )([A-Z][a-zA-Z'\-]+)"
                             r"(?: (?:Jr|Sr)\.|\b)", s):
            first, sur = m.group(2), re.sub(r"(?:'s?|’s?)$", "", norm(m.group(3)))
            if first and (norm(first) in NOT_NAMES or norm(first) in team_words):
                continue
            if sur in ("jr", "sr", "ii", "iii", "iv") or sur in team_words or sur in NOT_NAMES \
                    or sur in surnames:
                continue
            if any(sur == k.split()[0] for k in surnames):  # "Jackson" vs "jackson jr."
                continue
            rep.claims.append(Claim("player", m.group(0).strip(), False, "not in this game"))
        # --- leads
        for m in re.finditer(r"(?:led by|lead of|up by|as many as)\s+(\d{1,2})|"
                             r"(\d{1,2})-point (?:lead|deficit|hole|advantage)", ns):
            n = int(m.group(1) or m.group(2))
            rep.claims.append(Claim("lead", m.group(0), n <= max_lead,
                                    f"largest lead {max_lead}"))
    # one invented-player claim per distinct name, however often it is repeated
    seen, claims = set(), []
    for c in rep.claims:
        key = (c.kind, norm(c.text).rstrip("'s")) if c.kind == "player" else None
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        claims.append(c)
    rep.claims = claims
    return rep
