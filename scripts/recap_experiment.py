"""Generate recaps for the pilot and the randomized experiment (resumable).

Design (fixed before any generation, see config.yaml -> recaps):
  * population: 2025-26 regular-season games whose play-by-play reconciles with the score
  * pilot: 40 games, both arms on every game (paired) -> effect size for the power analysis
  * main: 300 different games, each randomized to ONE arm (1:1), unit = game
  * primary metric: recap contains at least one false checkable claim
  * secondary: false claims per recap, claim-level precision, winner/score stated correctly
"""

import json
import time

import numpy as np

from nbagi import DATA, load_config
from nbagi.recaps import Generator, prompt_grounded, prompt_raw, season_facts

OUT = DATA / "recaps" / "generations.jsonl"


def design(cfg: dict, game_ids: list[str]) -> list[dict]:
    r = cfg["recaps"]
    rng = np.random.default_rng(cfg["seed"])
    ids = list(rng.permutation(sorted(game_ids)))
    pilot = ids[: r["pilot_games"]]
    main = ids[r["pilot_games"]: r["pilot_games"] + r["experiment_games"]]
    arms = rng.permutation(np.repeat(["raw", "grounded"], len(main) // 2))
    rows = [{"phase": "pilot", "game_id": g, "arm": a} for g in pilot for a in ("raw", "grounded")]
    rows += [{"phase": "main", "game_id": g, "arm": a} for g, a in zip(main, arms)]
    return rows


def main():
    cfg = load_config()
    r = cfg["recaps"]
    facts = season_facts(r["season"])
    plan = design(cfg, list(facts))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {(d["phase"], d["game_id"], d["arm"]) for d in map(json.loads, OUT.open())}
    todo = [p for p in plan if (p["phase"], p["game_id"], p["arm"]) not in done]
    print(f"{len(plan)} generations planned, {len(todo)} to go", flush=True)
    gen = Generator(r["model"], r["max_new_tokens"])
    t0 = time.time()
    with OUT.open("a") as f:
        for i, p in enumerate(todo, 1):
            fa = facts[p["game_id"]]
            if p["arm"] == "raw":
                prompt = prompt_raw(fa, r["raw_pbp_lines"])
            else:
                prompt = prompt_grounded(fa)
            t = time.time()
            text = gen(prompt)
            f.write(json.dumps({**p, "prompt": prompt, "recap": text,
                                "seconds": round(time.time() - t, 2)}) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}  {(time.time() - t0) / i:.1f}s each", flush=True)


if __name__ == "__main__":
    main()
