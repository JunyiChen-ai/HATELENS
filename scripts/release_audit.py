from __future__ import annotations

import re
import sys
from pathlib import Path


BLOCKED_DIR_NAMES = {
    "__pycache__",
    "logs",
    "seed_search",
    "baselines",
    "archive",
    "paper",
    "HATELENS",
    "figures",
}
BLOCKED_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".ckpt",
    ".env",
    ".jpeg",
    ".jpg",
    ".log",
    ".mkv",
    ".mov",
    ".mp4",
    ".out",
    ".pdf",
    ".png",
    ".pth",
    ".tex",
    ".wav",
    ".webm",
}
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}
BLOCKED_BYTE_PATTERNS = {
    ("/" + "home" + "/").encode(),
    ("/" + "Users" + "/").encode(),
    ("jun" + "yi").encode(),
    ("EMNLP" + "2026").encode(),
    ("HV" + "Guard").encode(),
}
EMAIL_RE = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def is_text_file(path: Path) -> bool:
    return path.name == ".gitignore" or path.suffix in TEXT_SUFFIXES


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    violations = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if ".git" in rel.parts:
            continue
        if path.is_symlink():
            violations.append(f"{rel} (symlink)")
            continue
        if rel.parts[:2] == ("data", "raw") and path.is_file() and path.name not in {".gitkeep", "README.md"}:
            violations.append(str(rel))
            continue
        if any(part in BLOCKED_DIR_NAMES for part in rel.parts):
            violations.append(str(rel))
            continue
        if path.is_file() and path.suffix in BLOCKED_SUFFIXES:
            violations.append(str(rel))
            continue
        if path.is_file() and is_text_file(path):
            content = path.read_bytes()
            leaks_local_path = any(pattern in content for pattern in BLOCKED_BYTE_PATTERNS)
            leaks_email = rel.parts[:2] != ("artifacts", "p2c_outputs") and EMAIL_RE.search(content)
            if leaks_local_path or leaks_email:
                violations.append(f"{rel} (identity or local-path string)")
    if violations:
        print("Release audit failed; blocked files found:")
        for item in violations[:200]:
            print(f"  {item}")
        if len(violations) > 200:
            print(f"  ... {len(violations) - 200} more")
        sys.exit(1)
    print("Release audit passed.")


if __name__ == "__main__":
    main()
