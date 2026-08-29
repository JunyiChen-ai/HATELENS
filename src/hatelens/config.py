from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PACKAGE_ROOT / "configs"
DATASET_ALIASES = {
    "hatemm": "hatemm",
    "mhclip-english": "mhclip_english",
    "mhclip_en": "mhclip_english",
    "mhclip_english": "mhclip_english",
    "mhclip-chinese": "mhclip_chinese",
    "mhclip_zh": "mhclip_chinese",
    "mhclip_chinese": "mhclip_chinese",
    "implihatevid": "implihatevid",
    "implicit-hate-vid": "implihatevid",
}
DATASET_ORDER = ["hatemm", "mhclip_english", "mhclip_chinese", "implihatevid"]


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    name: str
    p2c_output: Path
    generated_p2c_output: Path
    split_dir: Path
    raw_dataset_dir: Path
    annotation_file: str
    embedding_dir: Path
    label_map: dict[str, int]
    seed: int
    retrieval: dict[str, Any]
    p2c_prompt_profile: str
    text_embedding_artifact: Path | None
    frame_embedding_artifact: Path | None
    audio_embedding_artifact: Path | None


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(dataset: str, root: Path | str = PACKAGE_ROOT) -> DatasetConfig:
    root = Path(root).resolve()
    key = DATASET_ALIASES.get(dataset.lower(), dataset.lower())
    path = CONFIG_DIR / f"{key}.json"
    if not path.exists():
        valid = ", ".join(DATASET_ORDER)
        raise ValueError(f"Unknown dataset '{dataset}'. Valid choices: {valid}, all")
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return DatasetConfig(
        key=key,
        name=raw["name"],
        p2c_output=resolve_path(root, raw["p2c_output"]),
        generated_p2c_output=resolve_path(root, raw["generated_p2c_output"]),
        split_dir=resolve_path(root, raw["split_dir"]),
        raw_dataset_dir=resolve_path(root, raw["raw_dataset_dir"]),
        annotation_file=raw["annotation_file"],
        embedding_dir=resolve_path(root, raw["embedding_dir"]),
        label_map=raw["label_map"],
        seed=int(raw["seed"]),
        retrieval=raw["retrieval"],
        p2c_prompt_profile=raw["p2c_prompt_profile"],
        text_embedding_artifact=(
            resolve_path(root, raw["text_embedding_artifact"])
            if raw.get("text_embedding_artifact")
            else None
        ),
        frame_embedding_artifact=(
            resolve_path(root, raw["frame_embedding_artifact"])
            if raw.get("frame_embedding_artifact")
            else None
        ),
        audio_embedding_artifact=(
            resolve_path(root, raw["audio_embedding_artifact"])
            if raw.get("audio_embedding_artifact")
            else None
        ),
    )


def selected_configs(dataset: str, root: Path | str = PACKAGE_ROOT) -> list[DatasetConfig]:
    if dataset == "all":
        return [load_config(key, root=root) for key in DATASET_ORDER]
    return [load_config(dataset, root=root)]
