"""Braintrust eval for Slopbot. Runs the production detect() + render() on the labeled dataset.

  uv run braintrust eval evals/eval_slopbot.py                 # one experiment, SLOPBOT_MODEL
  SLOPBOT_EVAL_MODELS=candidates uv run python evals/eval_slopbot.py   # one per candidate model,
                                                               # run one after another
  SLOPBOT_EVAL_FILE=evals/data/dataset.jsonl \\
    uv run braintrust eval --no-send-logs evals/eval_slopbot.py  # local file, nothing uploaded

Knobs (env): SLOPBOT_MODEL, SLOPBOT_EVAL_MODELS (comma list, or "candidates" for the list in
evals/models.py), SLOPBOT_JUDGE_MODEL, SLOPBOT_JUDGE_EFFORT, SLOPBOT_EFFORT, SLOPBOT_TRIALS,
SLOPBOT_EVAL_CONCURRENCY, SLOPBOT_SKIP_JUDGES=1 (code scorers only).

The judge model stays fixed across a comparison so scores are comparable. Every model in use
is checked against the gateway's /model/info before anything runs (and before anything costs).
"""

from __future__ import annotations

import atexit
import os
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from braintrust import Eval, init_dataset  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from evals.common import DATASET_NAME, ROOT, load_images, read_jsonl  # noqa: E402
from evals.models import resolve_models  # noqa: E402
from evals.scorers import ALL_SCORERS, CODE_SCORERS  # noqa: E402
from slopbot.config import Settings, get_settings  # noqa: E402
from slopbot.detector import detect  # noqa: E402
from slopbot.gateway import SPEND, verify_models  # noqa: E402
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
    return init_dataset(**settings.braintrust_target, name=DATASET_NAME)


def make_task(run_settings: Settings) -> Callable[[dict[str, Any], Any], dict[str, Any]]:
    def task(input: dict[str, Any], hooks: Any) -> dict[str, Any]:
        # Errors propagate so Braintrust records them as errors, not as wrong answers.
        result = detect(
            input.get("text"), load_images(input), surface="eval", settings=run_settings
        )
        hooks.metadata["served_model"] = result.served_model
        hooks.metadata["fell_back"] = result.fell_back
        hooks.metadata["cost_usd"] = result.cost_usd
        return {**result.verdict.model_dump(), "rendered": render_text(result.verdict)}

    return task


skip_judges = os.getenv("SLOPBOT_SKIP_JUDGES", "").lower() in ("1", "true", "yes")
models = resolve_models(os.getenv("SLOPBOT_EVAL_MODELS"), settings.model)

# Fail fast (before spending anything) if any model isn't on the gateway.
models_in_use = {f"detector[{m}]": m for m in models}
if not skip_judges:
    models_in_use["SLOPBOT_JUDGE_MODEL"] = settings.judge_model
verify_models(settings, models_in_use)


def report_spend() -> None:
    """What this run actually cost, per model, as reported by the gateway."""
    spend = SPEND.snapshot()
    if not spend:
        return
    print("\nGateway spend this run (detector + judge calls):")
    for name, cost in sorted(spend.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<24} ${cost:7.3f}  ({SPEND.calls_by_model[name]} calls)")
    print(f"  {'total':<24} ${SPEND.total:7.3f}")


atexit.register(report_spend)

for model in models:
    run_settings = replace(settings, model=model)
    Eval(
        settings.braintrust_project,
        project_id=settings.braintrust_project_id,  # wins over the name when set
        data=data,
        task=make_task(run_settings),
        scores=CODE_SCORERS if skip_judges else ALL_SCORERS,
        experiment_name=f"{model}-effort-{settings.effort}",
        metadata={
            "model": model,
            "effort": settings.effort,
            "fallbacks": settings.fallbacks,
            "judge_model": None if skip_judges else settings.judge_model,
            "judge_effort": None if skip_judges else os.getenv("SLOPBOT_JUDGE_EFFORT", "medium"),
            "via_gateway": settings.uses_gateway,
            "dataset": os.getenv("SLOPBOT_EVAL_FILE") or DATASET_NAME,
        },
        trial_count=int(os.getenv("SLOPBOT_TRIALS", "1")),
        max_concurrency=int(os.getenv("SLOPBOT_EVAL_CONCURRENCY", "8")),
    )
