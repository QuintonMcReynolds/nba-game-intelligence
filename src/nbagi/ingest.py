"""Download each season's play-by-play, convert to Parquet, delete the raw file.

Output: data/lake/raw_pbp/season=YYYY/pbp.parquet (Hive partitioning, read by Spark).
"""

from __future__ import annotations

import lzma
import shutil
import ssl
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import certifi
import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.parquet as pq

from . import LAKE

# every column as string: the CSVs mix blanks and numbers, the ETL casts explicitly
SCHEMA_COLS = [
    "actionNumber", "clock", "period", "teamId", "teamTricode", "personId", "playerName",
    "playerNameI", "xLegacy", "yLegacy", "shotDistance", "shotResult", "isFieldGoal",
    "scoreHome", "scoreAway", "pointsTotal", "location", "description", "actionType",
    "subType", "videoAvailable", "shotValue", "actionId", "gameId",
]


def ingest_season(season: int, url_template: str, force: bool = False) -> Path:
    out = LAKE / "raw_pbp" / f"season={season}" / "pbp.parquet"
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "s.tar.xz"
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url_template.format(season=season), context=ctx) as r:
            archive.write_bytes(r.read())
        with lzma.open(archive) as xz, tarfile.open(fileobj=xz) as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith(".csv"))
            tar.extract(member, tmp, filter="data")
        archive.unlink()
        csv = Path(tmp) / member.name
        opts = pc.ConvertOptions(column_types={c: pa.string() for c in SCHEMA_COLS},
                                 strings_can_be_null=False)
        table = pc.read_csv(csv, convert_options=opts)
        missing = [c for c in SCHEMA_COLS if c not in table.column_names]
        for c in missing:  # older seasons lack a few columns (e.g. shotValue)
            table = table.append_column(c, pa.nulls(table.num_rows, pa.string()))
        if missing:
            print(f"  {season}: missing columns filled with nulls: {missing}")
        pq.write_table(table.select(SCHEMA_COLS), out, compression="zstd")
        csv.unlink()
    return out


def ingest(cfg: dict) -> list[Path]:
    d = cfg["data"]
    paths = []
    for season in range(d["first_season"], d["last_season"] + 1):
        p = ingest_season(season, d["source"])
        print(f"  {season}: {p.stat().st_size / 1e6:.1f} MB")
        paths.append(p)
    shutil.rmtree(LAKE / "_tmp", ignore_errors=True)
    return paths
