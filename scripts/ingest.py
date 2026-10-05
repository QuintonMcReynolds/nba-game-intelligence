"""Download 22 seasons of NBA play-by-play into the Parquet lake."""

from nbagi import load_config
from nbagi.ingest import ingest

if __name__ == "__main__":
    ingest(load_config())
