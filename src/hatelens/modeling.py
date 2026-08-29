from __future__ import annotations

import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from .config import DatasetConfig, selected_configs


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODALITIES = ["text", "audio", "frame", "ans_what", "ans_target", "ans_where", "ans_why", "ans_how"]


class FeatureDataset(Dataset):
    def __init__(self, video_ids: list[str], features: dict, label_map: dict[str, int], modalities: list[str]):
        self.video_ids = video_ids
        self.features = features
        self.label_map = label_map
        self.modalities = modalities

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        video_id = self.video_ids[index]
        item = {key: self.features[key][video_id] for key in self.modalities}
        item["label"] = torch.tensor(self.label_map[self.features["labels"][video_id]["Label"]], dtype=torch.long)
        return item


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


class Fusion(nn.Module):
    def __init__(self, modalities: list[str], hidden: int = 192, heads: int = 4, classes: int = 2):
        super().__init__()
        self.modalities = modalities
        self.modality_dropout = 0.15
        self.projections = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(768, hidden), nn.GELU(), nn.Dropout(0.15), nn.LayerNorm(hidden))
                for _ in modalities
            ]
        )
        self.routes = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)) for _ in range(heads)]
        )
        combined = heads * hidden + hidden
        self.pre_classifier = nn.Sequential(
            nn.LayerNorm(combined),
            nn.Linear(combined, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.075),
        )
        self.head = nn.Linear(64, classes)

    def forward(self, batch: dict[str, torch.Tensor], training: bool = False, return_penult: bool = False):
        representations = []
        for projection, key in zip(self.projections, self.modalities):
            hidden = projection(batch[key])
            if training and self.modality_dropout > 0:
                mask = (torch.rand(hidden.size(0), 1, device=hidden.device) > self.modality_dropout).float()
                hidden = hidden * mask
            representations.append(hidden)
        stacked = torch.stack(representations, dim=1)
        routed = [
            (stacked * torch.softmax(route(stacked).squeeze(-1), dim=1).unsqueeze(-1)).sum(dim=1)
            for route in self.routes
        ]
        fused = torch.cat(routed + [stacked.mean(dim=1)], dim=-1)
        penult = self.pre_classifier(fused)
        logits = self.head(penult)
        return (logits, penult) if return_penult else logits


def load_split_ids(split_dir: Path) -> dict[str, list[str]]:
    splits = {}
    for name in ["train", "valid", "test"]:
        with (split_dir / f"{name}.csv").open() as f:
            splits[name] = [row[0] for row in csv.reader(f) if row]
    return splits


def load_features(cfg: DatasetConfig) -> dict:
    features = {
        "text": torch.load(cfg.embedding_dir / "text_features.pth", map_location="cpu"),
        "audio": torch.load(cfg.embedding_dir / "wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(cfg.embedding_dir / "frame_features.pth", map_location="cpu"),
    }
    for field in ["what", "target", "where", "why", "how"]:
        features[f"ans_{field}"] = torch.load(cfg.embedding_dir / f"p2c_ans_{field}_features.pth", map_location="cpu")
    with cfg.p2c_output.open(encoding="utf-8") as f:
        features["labels"] = {row["Video_ID"]: row for row in json.load(f)}
    return features


def scheduler(optimizer, warmup_steps: int, total_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0, 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps))))

    return LambdaLR(optimizer, fn)


def get_penult_and_logits(model: Fusion, loader: DataLoader):
    model.eval()
    penults, logits, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            logit, penult = model(batch, return_penult=True)
            penults.append(penult.cpu())
            logits.append(logit.cpu())
            labels.extend(batch["label"].cpu().numpy())
    return torch.cat(penults), torch.cat(logits).numpy(), np.array(labels)


def zca_whiten(train_z, val_z, test_z):
    mean = train_z.mean(dim=0, keepdim=True)
    centered = train_z - mean
    cov = (centered.t() @ centered) / (centered.size(0) - 1)
    eye = torch.eye(cov.size(0))
    u, s, v = torch.svd(cov + 1e-5 * eye)
    transform = u @ torch.diag(1.0 / torch.sqrt(s)) @ v.t()
    return (
        F.normalize((train_z - mean) @ transform, dim=1),
        F.normalize((val_z - mean) @ transform, dim=1),
        F.normalize((test_z - mean) @ transform, dim=1),
    )


def spca_whiten(train_z, val_z, test_z, rank: int):
    mean = train_z.mean(dim=0, keepdim=True)
    centered = (train_z - mean).numpy()
    cov = torch.tensor(LedoitWolf().fit(centered).covariance_, dtype=torch.float32)
    u, s, _ = torch.svd(cov)
    u = u[:, :rank]
    s = s[:rank]
    transform = u @ torch.diag(1.0 / torch.sqrt(s + 1e-6))
    return (
        F.normalize((train_z - mean) @ transform, dim=1),
        F.normalize((val_z - mean) @ transform, dim=1),
        F.normalize((test_z - mean) @ transform, dim=1),
    )


def cosine_knn(query, bank, bank_labels, k: int, temperature: float):
    query = F.normalize(query, dim=1)
    bank = F.normalize(bank, dim=1)
    sims, idx = torch.mm(query, bank.t()).topk(k, dim=1)
    labels = bank_labels[idx]
    weights = F.softmax(sims / temperature, dim=1)
    out = torch.zeros(query.size(0), 2)
    for cls in range(2):
        out[:, cls] = (weights * (labels == cls).float()).sum(dim=1)
    return out.numpy()


def csls_knn(query, bank, bank_labels, k: int, temperature: float, hub_k: int = 10):
    query = F.normalize(query, dim=1)
    bank = F.normalize(bank, dim=1)
    sim = torch.mm(query, bank.t())
    bank_hub = sim.topk(min(hub_k, sim.size(0)), dim=0).values.mean(dim=0)
    csls = 2 * sim - bank_hub.unsqueeze(0)
    sims, idx = csls.topk(k, dim=1)
    labels = bank_labels[idx]
    weights = F.softmax(sims / temperature, dim=1)
    out = torch.zeros(query.size(0), 2)
    for cls in range(2):
        out[:, cls] = (weights * (labels == cls).float()).sum(dim=1)
    return out.numpy()


def apply_retrieval(cfg: DatasetConfig, train_z, val_z, test_z, train_labels, test_logits):
    retrieval = cfg.retrieval
    whiten = retrieval["whiten"]
    if whiten == "none":
        train_w, test_w = train_z, test_z
    elif whiten == "zca":
        train_w, _, test_w = zca_whiten(train_z, val_z, test_z)
    elif whiten.startswith("spca_r"):
        rank = int(whiten.split("r", 1)[1])
        train_w, _, test_w = spca_whiten(train_z, val_z, test_z, rank)
    else:
        raise ValueError(f"Unknown whitening method: {whiten}")
    bank_labels = torch.tensor(train_labels)
    if retrieval["knn_type"] == "cosine":
        knn = cosine_knn(test_w, train_w, bank_labels, retrieval["k"], retrieval["temp"])
    elif retrieval["knn_type"] == "csls":
        knn = csls_knn(test_w, train_w, bank_labels, retrieval["k"], retrieval["temp"])
    else:
        raise ValueError(f"Unknown kNN type: {retrieval['knn_type']}")
    return (1 - retrieval["alpha"]) * test_logits + retrieval["alpha"] * knn


def metrics(labels, predictions) -> dict:
    return {
        "acc": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, average="macro")),
        "p": float(precision_score(labels, predictions, average="macro")),
        "r": float(recall_score(labels, predictions, average="macro")),
        "cm": confusion_matrix(labels, predictions).tolist(),
    }


def reproduce_dataset(cfg: DatasetConfig) -> dict:
    features = load_features(cfg)
    splits = load_split_ids(cfg.split_dir)
    common = set.intersection(*[set(features[key].keys()) for key in MODALITIES]) & set(features["labels"].keys())
    current = {split: [video_id for video_id in ids if video_id in common] for split, ids in splits.items()}

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    train = FeatureDataset(current["train"], features, cfg.label_map, MODALITIES)
    valid = FeatureDataset(current["valid"], features, cfg.label_map, MODALITIES)
    test = FeatureDataset(current["test"], features, cfg.label_map, MODALITIES)
    train_loader = DataLoader(train, 32, True, collate_fn=collate)
    valid_loader = DataLoader(valid, 64, False, collate_fn=collate)
    test_loader = DataLoader(test, 64, False, collate_fn=collate)
    train_eval_loader = DataLoader(train, 64, False, collate_fn=collate)

    model = Fusion(MODALITIES).to(DEVICE)
    ema = copy.deepcopy(model)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5], dtype=torch.float).to(DEVICE), label_smoothing=0.03)
    epochs = 45
    lr_scheduler = scheduler(optimizer, 5 * len(train_loader), epochs * len(train_loader))
    best_val, best_state = -1.0, None

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            optimizer.zero_grad()
            criterion(model(batch, training=True), batch["label"]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            with torch.no_grad():
                for param, ema_param in zip(model.parameters(), ema.parameters()):
                    ema_param.data.mul_(0.999).add_(param.data, alpha=0.001)
        ema.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                batch = {key: value.to(DEVICE) for key, value in batch.items()}
                preds.extend(ema(batch).argmax(1).cpu().numpy())
                labels.extend(batch["label"].cpu().numpy())
        val_acc = accuracy_score(labels, preds)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {key: value.clone() for key, value in ema.state_dict().items()}

    ema.load_state_dict(best_state)
    train_z, _, train_labels = get_penult_and_logits(ema, train_eval_loader)
    valid_z, _, _ = get_penult_and_logits(ema, valid_loader)
    test_z, test_logits, test_labels = get_penult_and_logits(ema, test_loader)
    blended = apply_retrieval(cfg, train_z, valid_z, test_z, train_labels, test_logits)
    thresh = cfg.retrieval["thresh"]
    if thresh is None:
        predictions = np.argmax(blended, axis=1)
    else:
        predictions = ((blended[:, 1] - blended[:, 0]) > thresh).astype(int)
    result = metrics(test_labels, predictions)
    result["seed"] = cfg.seed
    result["retrieval"] = cfg.retrieval
    result["split_sizes"] = {split: len(ids) for split, ids in current.items()}
    return result


def dry_run(configs: list[DatasetConfig]) -> None:
    for cfg in configs:
        print(f"{cfg.name}: seed={cfg.seed}, retrieval={cfg.retrieval}")


def run_reproduction(dataset: str, root: Path, dry: bool = False) -> dict[str, dict]:
    configs = selected_configs(dataset, root=root)
    if dry:
        dry_run(configs)
        return {}
    results = {}
    for cfg in configs:
        print(f"[main-performance] {cfg.name}")
        result = reproduce_dataset(cfg)
        print(
            f"[result] {cfg.name}: ACC={result['acc']:.4f} F1={result['f1']:.4f} "
            f"P={result['p']:.4f} R={result['r']:.4f}"
        )
        results[cfg.name] = result
    return results
