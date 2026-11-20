"""Shared helpers for the dataset builder, scorers, and eval."""

from __future__ import annotations

import json
import mimetypes
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from slopbot.detector import ImageInput
from slopbot.slack_inputs import prepare_image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "evals" / "data"
DATASET_NAME = "slop-detection-v1"
LABELS = ("ai", "human")


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON ({e})") from e


def load_images(input: dict[str, Any]) -> list[ImageInput]:
    """Load images referenced by an eval case (`image_path` or `image_paths`, repo-relative)."""
    paths = list(input.get("image_paths") or [])
    if input.get("image_path"):
        paths.append(input["image_path"])
    images = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        images.append(prepare_image(path.read_bytes(), media_type, path.name))
    return images
