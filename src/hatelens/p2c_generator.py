from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from tqdm import tqdm

from .config import load_config, selected_configs
from .original_prompts import (
    IMPLICIT_SYSTEM_PROMPT,
    IMPLICIT_USER_PROMPT_TEMPLATE,
    STANDARD_SYSTEM_PROMPT,
    STANDARD_USER_PROMPT_TEMPLATE,
)


load_dotenv()
MODEL_NAME = "gpt-5.4-nano"
PROMPT_PROFILES = {
    "standard_harmful": (STANDARD_SYSTEM_PROMPT, STANDARD_USER_PROMPT_TEMPLATE),
    "implicit_video_harmful": (IMPLICIT_SYSTEM_PROMPT, IMPLICIT_USER_PROMPT_TEMPLATE),
}


def encode_image(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_image_content(frames: list[Path]) -> list[dict]:
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(frame)}"}}
        for frame in frames
    ]


def _strip_tag_value(value: str) -> str:
    cleaned = re.sub(r"^[\s:：\-–—]+", "", value.strip())
    cleaned = re.sub(r"(?:</[^>]+>\s*)+$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _extract_block(text: str, tag: str, fallback_starts: tuple[str, ...] = ()) -> str | None:
    start_match = re.search(rf"<{tag}>", text, re.IGNORECASE)
    if not start_match:
        return None
    start = start_match.end()

    end_candidates = []
    close_match = re.search(rf"</{tag}>", text[start:], re.IGNORECASE)
    if close_match:
        end_candidates.append(start + close_match.start())

    for fallback in fallback_starts:
        fallback_match = re.search(rf"<{fallback}>", text[start:], re.IGNORECASE)
        if fallback_match:
            end_candidates.append(start + fallback_match.start())

    end = min(end_candidates) if end_candidates else len(text)
    return text[start:end].strip()


def _extract_tag_value(
    text: str,
    tag: str,
    next_tags: tuple[str, ...] = (),
    block_end_tags: tuple[str, ...] = (),
) -> str | None:
    start_match = re.search(rf"(?:<{tag}>|</{tag}>)", text, re.IGNORECASE)
    if not start_match:
        return None
    start = start_match.end()

    end_candidates = []
    close_match = re.search(rf"</{tag}>", text[start:], re.IGNORECASE)
    if close_match:
        end_candidates.append(start + close_match.start())

    for next_tag in next_tags:
        for boundary in (rf"<{next_tag}>", rf"</{next_tag}>"):
            next_match = re.search(boundary, text[start:], re.IGNORECASE)
            if next_match:
                end_candidates.append(start + next_match.start())

    for end_tag in block_end_tags:
        end_match = re.search(rf"</{end_tag}>", text[start:], re.IGNORECASE)
        if end_match:
            end_candidates.append(start + end_match.start())

    end = min(end_candidates) if end_candidates else len(text)
    value = _strip_tag_value(text[start:end])
    return value or None


def _normalize_which_label(value: str | None) -> str | None:
    if not value:
        return None
    lowered = value.lower()
    if "normal" in lowered:
        return "Normal"
    if "harmful" in lowered:
        return "Harmful"
    return _strip_tag_value(value)


def parse_p2c_response(text: str) -> dict[str, str]:
    result = {"raw": text}

    think_text = _extract_block(text, "think", fallback_starts=("answer",))
    if think_text:
        result["think"] = think_text
        step_tags = tuple(f"step{i}" for i in range(1, 5))
        for idx, step_tag in enumerate(step_tags):
            value = _extract_tag_value(
                think_text,
                step_tag,
                next_tags=step_tags[idx + 1:],
                block_end_tags=("think", "answer"),
            )
            if value:
                result[step_tag] = value

    answer_text = _extract_block(text, "answer")
    search_text = answer_text if answer_text else text
    if answer_text:
        result["answer"] = answer_text

    answer_tags = ("which", "what", "target", "where", "why", "how")
    for idx, tag in enumerate(answer_tags):
        value = _extract_tag_value(
            search_text,
            tag,
            next_tags=answer_tags[idx + 1:],
            block_end_tags=("answer",),
        )
        if value:
            result[tag] = value

    normalized = _normalize_which_label(result.get("which"))
    if normalized:
        result["which"] = normalized

    return {key: value for key, value in result.items() if value is not None}


def is_valid(item: dict) -> bool:
    response = item.get("p2c_response", {})
    return bool(response.get("step1") and response.get("which"))


def load_generation_rows(input_path: Path, output_path: Path) -> list[dict]:
    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not output_path.exists():
        return data
    with output_path.open(encoding="utf-8") as f:
        saved = json.load(f)
    saved_by_id = {row.get("Video_ID"): row for row in saved if row.get("Video_ID")}
    for item in data:
        saved_item = saved_by_id.get(item.get("Video_ID"))
        if saved_item and "p2c_response" in saved_item:
            item["p2c_response"] = saved_item["p2c_response"]
    return data


async def request_with_retries(client: AsyncOpenAI, messages: list[dict], max_tokens: int) -> str:
    for attempt in range(5):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except (RateLimitError, APIConnectionError):
            await asyncio.sleep(2**attempt)
        except Exception:
            if attempt < 4:
                await asyncio.sleep(2)
    return ""


async def generate_for_config(dataset: str, max_concurrent: int, root: Path) -> None:
    cfg = load_config(dataset, root=root)
    input_path = cfg.raw_dataset_dir / cfg.annotation_file
    output_path = cfg.generated_p2c_output
    quad_root = cfg.raw_dataset_dir / "quad"
    system_prompt, user_prompt_template = PROMPT_PROFILES[cfg.p2c_prompt_profile]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_generation_rows(input_path, output_path)
    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    semaphore = asyncio.Semaphore(max_concurrent)
    lock = asyncio.Lock()
    pbar = tqdm(total=len(data), initial=sum(is_valid(row) for row in data), desc=cfg.name)

    async def worker(item: dict) -> None:
        try:
            if is_valid(item):
                return
            video_id = item["Video_ID"]
            frame_dir = quad_root / video_id
            frames = sorted([p for p in frame_dir.glob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
            if not frames:
                item["p2c_response"] = {"error": "missing quad frames"}
                return
            prompt = user_prompt_template.format(
                title=item.get("Title", ""),
                transcript=item.get("Transcript", ""),
            )
            async with semaphore:
                raw = await request_with_retries(
                    client,
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": build_image_content(frames) + [{"type": "text", "text": prompt}]},
                    ],
                    max_tokens=2048,
                )
            item["p2c_response"] = parse_p2c_response(raw) if raw else {"error": "empty response", "raw": ""}
            async with lock:
                output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            pbar.update(1)

    await asyncio.gather(*(worker(item) for item in data))
    pbar.close()
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the optional P2C Generator stage.")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--root", default=".")
    parser.add_argument("--max-concurrent", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    datasets = selected_configs(args.dataset, root=root)
    for cfg in datasets:
        asyncio.run(generate_for_config(cfg.key, args.max_concurrent, root))


if __name__ == "__main__":
    main()
