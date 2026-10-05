"""Score every recap with the fact checker, validate the checker, run the pilot power
analysis and the pre-registered A/B test."""

import json
import re

import numpy as np
import pandas as pd
from scipy import stats

from nbagi import ARTIFACTS, DATA, load_config
from nbagi.factcheck import check
from nbagi.recaps import season_facts, template_recap

ALPHA = 0.05


def score(d: dict, f: dict) -> dict:
    r = check(d["recap"], f)
    kinds = {k: sum(c.kind == k for c in r.claims) for k in
             ("score", "winner", "stat", "player", "lead", "margin")}
    errs = {f"err_{k}": sum(c.kind == k and not c.ok for c in r.claims) for k in kinds}
    return {**{k: d[k] for k in ("phase", "game_id", "arm", "seconds")},
            "claims": r.n, "errors": r.errors, "any_error": int(r.errors > 0),
            "winner_right": int(r.has("winner", True) and not r.has("winner", False)),
            "score_right": int(r.has("score", True) and not r.has("score", False)),
            "words": len(d["recap"].split()), **{f"n_{k}": v for k, v in kinds.items()}, **errs}


def mutate(text: str, rng) -> str:
    """Plant one false number in a true recap (for checker recall)."""
    nums = list(re.finditer(r"\b(\d{1,3})\b", text))
    m = nums[rng.integers(len(nums))]
    new = str(int(m.group(1)) + int(rng.choice([-3, -2, 2, 3, 5])))
    return text[:m.start()] + new + text[m.end():]


def two_prop_power(p1: float, p2: float, n: int) -> float:
    """Power of a two-sided two-proportion z-test with n units per arm."""
    pbar = (p1 + p2) / 2
    se0 = np.sqrt(2 * pbar * (1 - pbar) / n)
    se1 = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    z = stats.norm.ppf(1 - ALPHA / 2)
    return float(stats.norm.sf((z * se0 - abs(p1 - p2)) / se1))


def main():
    cfg = load_config()
    facts = season_facts(cfg["recaps"]["season"])
    gens = [json.loads(x) for x in (DATA / "recaps" / "generations.jsonl").open()]
    df = pd.DataFrame([score(d, facts[d["game_id"]]) for d in gens])
    df.to_csv(ARTIFACTS / "recap_scores.csv", index=False)
    out = {}

    # 1. checker validation: false alarms on true text, recall on planted errors
    rng = np.random.default_rng(cfg["seed"])
    tmpl = [(template_recap(f), f) for f in facts.values()]
    fa = sum(check(t, f).errors for t, f in tmpl)
    n_claims = sum(check(t, f).n for t, f in tmpl)
    caught = [check(mutate(t, rng), f).errors > 0 for t, f in tmpl]
    out["checker"] = {"template_claims": n_claims, "false_alarms": fa,
                      "planted_errors": len(caught), "planted_caught": float(np.mean(caught))}

    # 2. pilot (paired): effect size -> power for the main test
    pilot = df[df["phase"] == "pilot"].pivot(index="game_id", columns="arm", values="any_error")
    p_raw, p_gr = pilot["raw"].mean(), pilot["grounded"].mean()
    n_main = cfg["recaps"]["experiment_games"] // 2
    out["pilot"] = {"games": len(pilot), "error_rate_raw": p_raw, "error_rate_grounded": p_gr,
                    "power_at_main_n": two_prop_power(p_raw, p_gr, n_main),
                    "mcnemar_discordant": [int(((pilot.raw == 1) & (pilot.grounded == 0)).sum()),
                                           int(((pilot.raw == 0) & (pilot.grounded == 1)).sum())]}

    # 3. main experiment: unit = game, one arm per game
    m = df[df["phase"] == "main"]
    a, b = m[m["arm"] == "raw"], m[m["arm"] == "grounded"]
    srm_p = stats.binomtest(len(a), len(m), 0.5).pvalue
    x1, n1, x2, n2 = a["any_error"].sum(), len(a), b["any_error"].sum(), len(b)
    p1, p2 = x1 / n1, x2 / n2
    pp = (x1 + x2) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    # claim-level precision, bootstrapped by game (claims within a recap are correlated)
    def precision(d):
        return 1 - d["errors"].sum() / max(d["claims"].sum(), 1)
    boot = []
    for _ in range(2000):
        ra = a.sample(len(a), replace=True, random_state=rng.integers(1e9))
        rb = b.sample(len(b), replace=True, random_state=rng.integers(1e9))
        boot.append(precision(rb) - precision(ra))
    out["main"] = {
        "games_raw": n1, "games_grounded": n2, "srm_p": srm_p,
        "any_error_raw": p1, "any_error_grounded": p2, "diff": p2 - p1,
        "diff_ci95": [p2 - p1 - 1.96 * se, p2 - p1 + 1.96 * se],
        "z": z, "p_value": float(2 * stats.norm.sf(abs(z))),
        "errors_per_recap": [a["errors"].mean(), b["errors"].mean()],
        "claims_per_recap": [a["claims"].mean(), b["claims"].mean()],
        "claim_precision": [precision(a), precision(b)],
        "precision_diff_ci95": list(np.percentile(boot, [2.5, 97.5])),
        "winner_right": [a["winner_right"].mean(), b["winner_right"].mean()],
        "score_right": [a["score_right"].mean(), b["score_right"].mean()],
        "invented_player_rate": [(a["err_player"] > 0).mean(), (b["err_player"] > 0).mean()],
        "seconds": [a["seconds"].mean(), b["seconds"].mean()],
    }
    by_kind = m.groupby("arm")[[c for c in m.columns if c.startswith("err_")]].mean()
    by_kind.to_csv(ARTIFACTS / "recap_errors_by_kind.csv")
    (ARTIFACTS / "recap_experiment.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))
    print(by_kind.round(3))


if __name__ == "__main__":
    main()
