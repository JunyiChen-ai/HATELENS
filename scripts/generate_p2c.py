from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hatelens.p2c_generator import main as p2c_main


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Optionally regenerate P2C Generator outputs.")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--max-concurrent", type=int, default=10)
    args = parser.parse_args()
    p2c_main(["--dataset", args.dataset, "--root", str(ROOT), "--max-concurrent", str(args.max_concurrent)])


if __name__ == "__main__":
    main()
