"""Models to compare as Slopbot's detector, and models that write the dataset's fresh AI examples.

Names are as the LiteLLM gateway exposes them (`uv run slopbot models`). Picked for family
diversity: LLM detectors tend to spot text from their own family best, so a fair comparison
needs detectors and generators from several labs.
"""

# Detector candidates: `SLOPBOT_EVAL_MODELS=candidates` runs one experiment per model.
CANDIDATES: dict[str, str] = {
    "claude-opus-5-5": "Current default; Anthropic's frontier model.",
    "claude-sonnet-5-5": "Anthropic mid tier, about half Opus's price.",
    "claude-haiku-5-5": "Anthropic's small model, ~40x cheaper than Opus. The budget option.",
    "gpt-6.2-sol": "OpenAI's everyday model. Same family as the ChatGPT-written examples.",
    "gemini-4-carbon": "Google's strongest Gemini 4 tier.",
    "kimi-k3.1": "Open-weight (Moonshot), with vision.",
}

# Generators for the dataset's fresh AI rows (`SLOPBOT_GEN_MODELS=mixed`), one per lab, so no
# detector family gets an easy ride spotting its own writing.
MIXED_GENERATORS: list[str] = [
    "claude-opus-5-5",
    "gpt-6.2-sol",
    "gemini-4-carbon",
    "kimi-k3.1",
]

PRESETS: dict[str, list[str]] = {
    "candidates": list(CANDIDATES),
    "mixed": MIXED_GENERATORS,
}


def resolve_models(raw: str | None, default: str) -> list[str]:
    """Parse a model-list env var: empty -> [default]; a preset name; or a comma list."""
    raw = (raw or "").strip()
    if not raw:
        return [default]
    if raw in PRESETS:
        return list(PRESETS[raw])
    return list(dict.fromkeys(m.strip() for m in raw.split(",") if m.strip()))
