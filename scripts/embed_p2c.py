from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hatelens.embedding import embed_p2c


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate P2C Generator embeddings.")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--source", choices=["official", "generated"], default="official")
    args = parser.parse_args()
    embed_p2c(args.dataset, ROOT, source=args.source)


if __name__ == "__main__":
    main()
