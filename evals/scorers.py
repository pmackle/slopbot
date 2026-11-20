"""Braintrust scorers for Slopbot. One property per scorer so failures are diagnosable.

Code scorers (free, deterministic):
  verdict_correct    exact label match (mixed/unsure count as wrong)
  committed          1 if Slopbot gave a definite ai/human verdict
  ai_recall          on AI rows only: did it say "ai"?
  human_specificity  on human rows only: did it say "human"? (false accusations hurt)
  calibration        1 - Brier score of ai_probability against the label
  concise            1 at <=60 words, linear down to 0 at 150 words

LLM judges (Claude, structured output, rationale logged):
  grounded_reasons   do the reasons point at things actually present in the content?
  tone               on-voice (deadpan, understated, funny) and aimed at the writing?
"""

from __future__ import annotations

import os
from typing import Any

from braintrust import Score
from pydantic import BaseModel, Field

from evals.common import load_images
from slopbot.detector import FALLBACK_BETA, build_messages, get_client

CONCISE_FULL_MARKS = 60
CONCISE_ZERO = 150


def _label(expected: Any) -> str | None:
    if isinstance(expected, dict):
        return expected.get("label")
    return expected if isinstance(expected, str) else None


def verdict_correct(input: Any, output: dict, expected: Any = None, **_: Any) -> list[Score]:
    label = _label(expected)
    verdict = output.get("verdict")
    return [
        Score(name="verdict_correct", score=float(verdict == label)),
        Score(name="committed", score=float(verdict in ("ai", "human"))),
    ]


def ai_recall(input: Any, output: dict, expected: Any = None, **_: Any) -> Score | None:
    if _label(expected) != "ai":
        return None  # not applicable; Braintrust skips None
    return Score(name="ai_recall", score=float(output.get("verdict") == "ai"))


def human_specificity(input: Any, output: dict, expected: Any = None, **_: Any) -> Score | None:
    if _label(expected) != "human":
        return None
    return Score(name="human_specificity", score=float(output.get("verdict") == "human"))


def calibration(input: Any, output: dict, expected: Any = None, **_: Any) -> Score | None:
    label = _label(expected)
    if label not in ("ai", "human"):
        return None
    p = max(0, min(100, int(output.get("ai_probability", 50)))) / 100
    y = 1.0 if label == "ai" else 0.0
    return Score(name="calibration", score=1 - (p - y) ** 2)


def concise(input: Any, output: dict, expected: Any = None, **_: Any) -> Score:
    n = len((output.get("rendered") or "").split())
    if n <= CONCISE_FULL_MARKS:
        value = 1.0
    elif n >= CONCISE_ZERO:
        value = 0.0
    else:
        value = 1 - (n - CONCISE_FULL_MARKS) / (CONCISE_ZERO - CONCISE_FULL_MARKS)
    return Score(name="concise", score=value, metadata={"words": n})


# --- LLM judges -----------------------------------------------------------------------------


class JudgeResult(BaseModel):
    rationale: str = Field(description="Two or three sentences explaining the score.")
    score: int = Field(description="Integer from 1 (worst) to 5 (best).")


GROUNDED_SYSTEM = """\
You grade the evidence an AI-text detector gave for its verdict. You'll see the content it \
judged and the reasons it gave. Do not grade whether the verdict is right; grade whether the \
reasons are grounded and specific.

5: every reason points at something verifiably present in the content (quoted phrases really \
appear; described visual artifacts are really visible) and is specific to this content.
4: all reasons are real, but one is somewhat generic.
3: reasons are mostly real but generic enough to apply to almost anything.
2: at least one reason misquotes or describes something that isn't there.
1: reasons are boilerplate or largely invented.
"""

TONE_SYSTEM = """\
You grade the voice of a Slack bot reply. The bot, Slopbot, judges whether content is \
AI-generated. Its intended voice is TARS from Interstellar: calm, deadpan, understated, and \
matter-of-fact, with light self-aware robot humor. The humor comes from flat delivery, not from \
jokes, and it pokes at the writing, never at the person who wrote or posted it.

5: deadpan and genuinely funny; sounds like a dry, unflappable robot crewmate; aimed at the writing.
4: on-voice and pleasant, but the line is weak or forgettable.
3: flat with no personality, or try-hard (exclamation points, emoji, puns, announcing the joke).
2: harsh, sneering, or long-winded.
1: insults the person (intelligence, appearance, identity) or is hostile.
"""


def _judge(system: str, content: list[dict[str, Any]]) -> JudgeResult:
    client = get_client()
    response = client.beta.messages.parse(
        model=os.getenv("SLOPBOT_JUDGE_MODEL", "claude-opus-5-5"),
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": content}],
        output_format=JudgeResult,
        output_config={"effort": os.getenv("SLOPBOT_JUDGE_EFFORT", "medium")},
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
    if response.stop_reason != "end_turn" or response.parsed_output is None:
        # Raise so Braintrust records a grader error instead of a fake zero.
        raise RuntimeError(f"judge returned no score (stop_reason={response.stop_reason})")
    return response.parsed_output


def _normalize(result: JudgeResult, name: str) -> Score:
    clamped = max(1, min(5, result.score))
    return Score(
        name=name,
        score=(clamped - 1) / 4,
        metadata={"raw": clamped, "rationale": result.rationale},
    )


def grounded_reasons(input: dict, output: dict, expected: Any = None, **_: Any) -> Score:
    reasons = "\n".join(f"- {r}" for r in output.get("reasons") or []) or "(none given)"
    content = build_messages(input.get("text"), load_images(input))[0]["content"]
    content.append({"type": "text", "text": f"<detector_reasons>\n{reasons}\n</detector_reasons>"})
    return _normalize(_judge(GROUNDED_SYSTEM, content), "grounded_reasons")


def tone(input: Any, output: dict, expected: Any = None, **_: Any) -> Score:
    reply = output.get("rendered") or ""
    text = f"<slopbot_reply>\n{reply}\n</slopbot_reply>"
    return _normalize(_judge(TONE_SYSTEM, [{"type": "text", "text": text}]), "tone")


CODE_SCORERS = [verdict_correct, ai_recall, human_specificity, calibration, concise]
JUDGE_SCORERS = [grounded_reasons, tone]
ALL_SCORERS = CODE_SCORERS + JUDGE_SCORERS
