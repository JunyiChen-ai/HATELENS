from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hatelens.modeling import run_reproduction


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train fixed seeds and reproduce HATELENS main performance.")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    results = run_reproduction(args.dataset, ROOT, dry=args.dry_run)
    if args.out and results:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
