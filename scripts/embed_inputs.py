from __future__ import annotations

import sys
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hatelens import embedding


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate text, frame, and audio embeddings.")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--only", choices=["text", "frames", "audio"], default=None)
    args = parser.parse_args()
    if args.only is None:
        for stage in ("text", "frames", "audio"):
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--dataset", args.dataset, "--only", stage],
                check=True,
            )
        return
    if args.only in (None, "text"):
        embedding.embed_text(args.dataset, ROOT)
    if args.only in (None, "frames"):
        embedding.embed_frames(args.dataset, ROOT)
    if args.only in (None, "audio"):
        embedding.embed_audio(args.dataset, ROOT)


if __name__ == "__main__":
    main()
