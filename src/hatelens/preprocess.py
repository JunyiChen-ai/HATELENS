from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import cv2
from PIL import Image


VIDEO_SUFFIXES = {".mp4", ".webm", ".avi", ".mkv", ".mov"}


def extract_frames(video_path: Path, out_dir: Path, num_frames: int = 32) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise RuntimeError(f"Could not read frame count for {video_path}")
    indices = [round(i * (total - 1) / max(1, num_frames - 1)) for i in range(num_frames)]
    wanted = set(indices)
    frame_index = 0
    saved = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in wanted:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(out_dir / f"frame_{saved + 1:03d}.jpg")
            saved += 1
        frame_index += 1
    capture.release()


def build_quads(frame_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    for start in range(0, len(frames), 4):
        chunk = frames[start : start + 4]
        if len(chunk) < 4:
            break
        images = [Image.open(path).convert("RGB").resize((224, 224)) for path in chunk]
        canvas = Image.new("RGB", (448, 448))
        for idx, image in enumerate(images):
            canvas.paste(image, ((idx % 2) * 224, (idx // 2) * 224))
        canvas.save(out_dir / f"quad_{start // 4 + 1:03d}.jpg")


def extract_audio(video_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def preprocess_dataset(raw_video_dir: Path, dataset_dir: Path, num_frames: int = 32) -> None:
    videos = sorted([p for p in raw_video_dir.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES])
    for video in videos:
        video_id = video.stem
        print(f"[preprocess] {video_id}")
        frame_dir = dataset_dir / "frames" / video_id
        quad_dir = dataset_dir / "quad" / video_id
        extract_frames(video, frame_dir, num_frames=num_frames)
        build_quads(frame_dir, quad_dir)
        extract_audio(video, dataset_dir / "audios" / f"{video_id}.wav")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare frames, quad frames, and audio files.")
    parser.add_argument("--raw-video-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    preprocess_dataset(Path(args.raw_video_dir), Path(args.dataset_dir), args.num_frames)


if __name__ == "__main__":
    main()
