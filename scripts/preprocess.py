from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hatelens.preprocess import preprocess_dataset


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames, quad frames, and audio.")
    parser.add_argument("--raw-video-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=32)
    args = parser.parse_args()
    preprocess_dataset(Path(args.raw_video_dir), Path(args.dataset_dir), args.num_frames)


if __name__ == "__main__":
    main()
