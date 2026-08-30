from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from PIL import Image
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    BertModel,
    BertTokenizer,
    ViTFeatureExtractor,
    ViTModel,
)

from .config import selected_configs


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_MODEL = "google-bert/bert-base-uncased"
VISION_MODEL = "google/vit-base-patch16-224"
AUDIO_MODEL = "microsoft/wavlm-base-plus"
P2C_FIELDS = ["what", "target", "where", "why", "how"]


def load_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(TEXT_MODEL)
        self.text_model = BertModel.from_pretrained(TEXT_MODEL)
        self.max_length = 128

    def forward(self, text: str) -> torch.Tensor:
        text_encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_features = self.text_model(**text_encoding).last_hidden_state
            text_features = text_features[:, 0, :].squeeze(0)
        return text_features


class FrameFeatureExtractor(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        frame = frame.unsqueeze(0)
        with torch.no_grad():
            outputs = self.vit(pixel_values=frame)
            pool_output = outputs.pooler_output
        return pool_output


def annotation_path(cfg) -> Path:
    return cfg.raw_dataset_dir / cfg.annotation_file


def materialize_artifact(artifact_path: Path | None, out_path: Path, tag: str) -> bool:
    if not artifact_path or not artifact_path.exists():
        return False
    archive = np.load(artifact_path)
    features = {
        str(video_id): torch.from_numpy(feature.copy())
        for video_id, feature in zip(archive["video_ids"], archive["features"])
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, out_path)
    print(f"[embed-inputs] {tag} from {artifact_path}")
    return True


def encode_texts(texts_by_id: dict[str, str], out_path: Path, max_length: int = 128) -> None:
    tokenizer = BertTokenizer.from_pretrained(TEXT_MODEL)
    model = BertModel.from_pretrained(TEXT_MODEL).to(DEVICE).eval()
    features = {}
    for video_id, text in texts_by_id.items():
        if not text or text.strip() == "":
            text = "No content available."
        encoded = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
        with torch.no_grad():
            features[video_id] = model(**encoded).last_hidden_state[:, 0, :].squeeze(0).cpu()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, out_path)


def p2c_source_path(cfg, source: str) -> Path:
    if source == "official":
        return cfg.p2c_output
    if source == "generated":
        return cfg.generated_p2c_output
    raise ValueError(f"Unknown P2C source: {source}")


def embed_p2c(dataset: str, root: Path, source: str = "official") -> None:
    for cfg in selected_configs(dataset, root=root):
        rows = load_json(p2c_source_path(cfg, source))
        by_field = {field: {} for field in P2C_FIELDS}
        for row in rows:
            response = row.get("p2c_response", {})
            video_id = row["Video_ID"]
            for field in P2C_FIELDS:
                by_field[field][video_id] = response.get(field, "")
        for field, texts in by_field.items():
            print(f"[embed-p2c] {cfg.name} ans_{field}")
            encode_texts(texts, cfg.embedding_dir / f"p2c_ans_{field}_features.pth")


def embed_text(dataset: str, root: Path) -> None:
    for cfg in selected_configs(dataset, root=root):
        out_path = cfg.embedding_dir / "text_features.pth"
        if materialize_artifact(cfg.text_embedding_artifact, out_path, f"{cfg.name} text"):
            continue
        rows = load_json(annotation_path(cfg))
        model = TextModel()
        features = {}
        print(f"[embed-inputs] {cfg.name} text")
        for row in rows:
            video_id = row.get("Video_ID")
            text = f"{row.get('Title') or ''} {row.get('Transcript') or ''}".strip() or " "
            print(f"Processing {video_id}...")
            features[video_id] = model(text).to(DEVICE)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, out_path)


def embed_frames(dataset: str, root: Path, frame_interval: int = 2) -> None:
    feature_extractor = None
    vit_model = None
    for cfg in selected_configs(dataset, root=root):
        out_path = cfg.embedding_dir / "frame_features.pth"
        if materialize_artifact(cfg.frame_embedding_artifact, out_path, f"{cfg.name} frames"):
            continue
        if feature_extractor is None or vit_model is None:
            feature_extractor = ViTFeatureExtractor.from_pretrained(VISION_MODEL)
            vit_model = ViTModel.from_pretrained(VISION_MODEL).to(DEVICE)
        rows = load_json(annotation_path(cfg))
        frame_extractor = FrameFeatureExtractor(vit_model).to(DEVICE)
        features = {}
        for row in rows:
            video_id = row["Video_ID"]
            frame_dir = Path(row.get("Frames_path", ""))
            if not frame_dir or not frame_dir.exists():
                frame_dir = cfg.raw_dataset_dir / "frames" / video_id
            if not frame_dir.exists():
                print(f"Warning: Frames path does not exist for Video_ID {video_id}")
                continue
            print(f"Processing {video_id}...")
            frame_files = sorted(os.listdir(frame_dir))
            selected = frame_files[::frame_interval]
            frame_features = []
            for frame_file in selected:
                if not frame_file.lower().endswith(("png", "jpg", "jpeg")):
                    continue
                frame_path = frame_dir / frame_file
                image = Image.open(frame_path).convert("RGB")
                inputs = feature_extractor(images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].squeeze(0).to(DEVICE)
                feature = frame_extractor(pixel_values)
                torch.cuda.empty_cache()
                frame_features.append(feature)
            if frame_features:
                features[video_id] = torch.cat(frame_features, dim=0).mean(dim=0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, out_path)


def embed_audio(dataset: str, root: Path) -> None:
    feature_extractor = None
    model = None
    for cfg in selected_configs(dataset, root=root):
        out_path = cfg.embedding_dir / "wavlm_audio_features.pth"
        if materialize_artifact(cfg.audio_embedding_artifact, out_path, f"{cfg.name} audio"):
            continue
        if feature_extractor is None or model is None:
            feature_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
            model = AutoModel.from_pretrained(AUDIO_MODEL).to(DEVICE).eval()
        rows = load_json(annotation_path(cfg))
        features = {}
        for index, row in enumerate(rows):
            video_id = row["Video_ID"]
            audio_path = cfg.raw_dataset_dir / "audios" / f"{video_id}.wav"
            if not audio_path.exists():
                print(f"  {video_id}: no audio")
                continue
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                if sample_rate != 16000:
                    waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                waveform = waveform.squeeze(0)
                max_samples = 16000 * 30
                if waveform.shape[0] > max_samples:
                    waveform = waveform[:max_samples]
                inputs = feature_extractor(waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
                with torch.no_grad():
                    out = model(**inputs)
                    features[video_id] = out.last_hidden_state.mean(dim=1).squeeze(0).cpu()
            except Exception as exc:
                print(f"  {video_id}: error - {exc}")
            if (index + 1) % 100 == 0:
                print(f"Processed {index + 1}/{len(rows)}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, out_path)


def normalize_features(path: Path) -> dict[str, torch.Tensor]:
    features = torch.load(path, map_location="cpu")
    return {key: F.normalize(value.float(), dim=0) if value.ndim == 1 else value.float() for key, value in features.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate HATELENS embeddings.")
    parser.add_argument("stage", choices=["p2c", "inputs"])
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--root", default=".")
    parser.add_argument("--only", choices=["text", "frames", "audio"], default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    if args.stage == "p2c":
        embed_p2c(args.dataset, root)
        return
    if args.only in (None, "text"):
        embed_text(args.dataset, root)
    if args.only in (None, "frames"):
        embed_frames(args.dataset, root)
    if args.only in (None, "audio"):
        embed_audio(args.dataset, root)


if __name__ == "__main__":
    main()
