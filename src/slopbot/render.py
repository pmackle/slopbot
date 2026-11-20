"""Turn a SlopVerdict into a short Slack reply. Formatting lives here, not in the model."""

from __future__ import annotations

from typing import Any

from .detector import SlopVerdict

MAX_REASONS = 3
MAX_REASON_CHARS = 160
MAX_QUIP_CHARS = 200

FEEDBACK_UP = "slopbot_feedback_up"
FEEDBACK_DOWN = "slopbot_feedback_down"

_EMOJI = {
    "ai": ":robot_face:",
    "human": ":bust_in_silhouette:",
    "mixed": ":performing_arts:",
    "unsure": ":shrug:",
}


def escape(text: str) -> str:
    """Escape Slack control characters so quoted content can't ping @channel or inject links."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def clamp_probability(value: int) -> int:
    return max(0, min(100, int(value)))


def headline(verdict: SlopVerdict) -> str:
    emoji = _EMOJI.get(verdict.verdict, ":shrug:")
    if verdict.verdict == "unsure":
        score = "Inconclusive."
    elif verdict.verdict == "mixed":
        score = f"{clamp_probability(verdict.ai_probability)}% slop, partially."
    else:
        score = f"{clamp_probability(verdict.ai_probability)}% slop."
    quip = escape(_clip(verdict.quip, MAX_QUIP_CHARS))
    return f"{emoji} *{score}* {quip}".rstrip()


def reason_lines(verdict: SlopVerdict) -> list[str]:
    reasons = [r for r in (verdict.reasons or []) if r and r.strip()]
    return [f"• {escape(_clip(r, MAX_REASON_CHARS))}" for r in reasons[:MAX_REASONS]]


def render_text(verdict: SlopVerdict) -> str:
    """The full reply as Slack mrkdwn. Also used as notification fallback text and for evals."""
    return "\n".join([headline(verdict), *reason_lines(verdict)])


def render_blocks(verdict: SlopVerdict, span_id: str | None = None) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": render_text(verdict)}}
    ]
    if span_id:
        blocks.append(
            {
                "type": "actions",
                "block_id": "slopbot_feedback",
                "elements": [
                    {
                        "type": "button",
                        "action_id": FEEDBACK_UP,
                        "text": {"type": "plain_text", "text": ":+1: Nailed it", "emoji": True},
                        "value": span_id,
                    },
                    {
                        "type": "button",
                        "action_id": FEEDBACK_DOWN,
                        "text": {"type": "plain_text", "text": ":-1: Wrong", "emoji": True},
                        "value": span_id,
                    },
                ],
            }
        )
    return blocks
