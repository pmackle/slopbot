"""Braintrust eval for Slopbot. Runs the production detect() + render() on the labeled dataset.

  uv run braintrust eval evals/eval_slopbot.py                 # uses the Braintrust dataset
  SLOPBOT_EVAL_FILE=evals/data/dataset.jsonl \\
    uv run braintrust eval --no-send-logs evals/eval_slopbot.py  # local file, nothing uploaded

Knobs (env): SLOPBOT_MODEL, SLOPBOT_EFFORT, SLOPBOT_TRIALS, SLOPBOT_EVAL_CONCURRENCY,
SLOPBOT_JUDGE_MODEL, SLOPBOT_JUDGE_EFFORT, SLOPBOT_SKIP_JUDGES=1 (code scorers only, cheaper).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from braintrust import Eval, init_dataset  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from evals.common import DATASET_NAME, ROOT, load_images, read_jsonl  # noqa: E402
from evals.scorers import ALL_SCORERS, CODE_SCORERS  # noqa: E402
from slopbot.config import get_settings  # noqa: E402
from slopbot.detector import detect  # noqa: E402
from slopbot.render import render_text  # noqa: E402

load_dotenv(ROOT / ".env")
settings = get_settings()


def data() -> Any:
    local = os.getenv("SLOPBOT_EVAL_FILE")
    if local:
        return [
            {
                "input": r["input"],
                "expected": r["expected"],
                "metadata": r.get("metadata", {}),
                "tags": r.get("tags", []),
            }
            for r in read_jsonl(local)
        ]
    return init_dataset(project=settings.braintrust_project, name=DATASET_NAME)


def task(input: dict[str, Any], hooks: Any) -> dict[str, Any]:
    # Errors propagate so Braintrust records them as errors, not as wrong answers.
    result = detect(input.get("text"), load_images(input), surface="eval", settings=settings)
    hooks.metadata["served_model"] = result.served_model
    hooks.metadata["fell_back"] = result.fell_back
    return {**result.verdict.model_dump(), "rendered": render_text(result.verdict)}


skip_judges = os.getenv("SLOPBOT_SKIP_JUDGES", "").lower() in ("1", "true", "yes")

Eval(
    settings.braintrust_project,
    data=data,
    task=task,
    scores=CODE_SCORERS if skip_judges else ALL_SCORERS,
    experiment_name=f"{settings.model}-effort-{settings.effort}",
    metadata={
        "model": settings.model,
        "effort": settings.effort,
        "fallbacks": settings.fallbacks,
        "judge_model": None if skip_judges else os.getenv("SLOPBOT_JUDGE_MODEL", "claude-opus-5-5"),
        "dataset": os.getenv("SLOPBOT_EVAL_FILE") or DATASET_NAME,
    },
    trial_count=int(os.getenv("SLOPBOT_TRIALS", "1")),
    max_concurrency=int(os.getenv("SLOPBOT_EVAL_CONCURRENCY", "8")),
)
