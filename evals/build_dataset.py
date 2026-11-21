"""Build Slopbot's labeled eval set and upload it to Braintrust.

Sources (balanced 50/50 human vs AI on the same questions, so topic can't give the answer away):
  human  HC3 `reddit_eli5` human answers: real Reddit comments written before ChatGPT existed.
  ai     HC3 ChatGPT answers (Dec 2022) to the same questions, plus fresh answers generated
         by Claude via SLOPBOT_GEN_MODEL (half in default assistant style, half asked to pass
         as a Redditor).
  yours  evals/data/custom.jsonl (gitignored): real messages you've labeled yourself.

HC3: Guo et al. 2023, "How Close is ChatGPT to Human Experts?"
     https://huggingface.co/datasets/Hello-SimpleAI/HC3 (CC BY-SA 4.0). Downloaded, not committed.

HC3's human answers were tokenized ("ca n't", "word ,") while its ChatGPT answers weren't, which
would let a detector win on spacing alone. We detokenize the human text, flatten line breaks on
both HC3 sides, and drop any row where artifacts survive.

Usage:
  uv run python evals/build_dataset.py                 # build + upload
  uv run python evals/build_dataset.py --no-generate   # skip Claude generation (no API cost)
  uv run python evals/build_dataset.py --no-upload     # local JSONL only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.common import DATA_DIR, DATASET_NAME, LABELS, ROOT, read_jsonl  # noqa: E402

HC3_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/reddit_eli5.jsonl"
SEED = 7
SCAN_ROWS = 4000
MIN_WORDS, MAX_WORDS = 40, 250

_DETOK = [
    (re.compile(r" n't\b"), "n't"),
    (re.compile(r" '(s|re|ve|ll|d|m)\b", re.IGNORECASE), r"'\1"),
    (re.compile(r" ([,.;:!?%)\]}])"), r"\1"),
    (re.compile(r"([(\[{$]) "), r"\1"),
    (re.compile(r'" ([^"]*?) "'), r'"\1"'),
    (re.compile(r"\s+"), " "),
]
_ARTIFACTS = re.compile(r" [,.;:!?](\s|$)| n't\b|URL_\d|`` |''|\[ deleted \]|&gt;|&amp;")


def detokenize(text: str) -> str:
    for pattern, repl in _DETOK:
        text = pattern.sub(repl, text)
    return text.strip()


def flatten(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> int:
    return len(text.split())


def usable(text: str) -> bool:
    return MIN_WORDS <= words(text) <= MAX_WORDS and not _ARTIFACTS.search(text)


def stream_hc3(limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with httpx.stream("GET", HC3_URL, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def pick_pairs(rows: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, str]]:
    pairs = []
    for row in rows:
        chatgpt = next(
            (c for c in map(flatten, row.get("chatgpt_answers") or []) if usable(c)), None
        )
        humans = [h for h in map(detokenize, row.get("human_answers") or []) if usable(h)]
        if not (chatgpt and humans):
            continue
        # Pick the human answer closest in length to the AI one, so length can't leak the label.
        human = min(humans, key=lambda h: abs(words(h) - words(chatgpt)))
        if abs(words(human) - words(chatgpt)) > 0.2 * words(chatgpt):
            continue
        pairs.append({"question": detokenize(row["question"]), "human": human, "chatgpt": chatgpt})
    if len(pairs) < n:
        raise SystemExit(f"Only {len(pairs)} usable HC3 rows found; scan more with --scan.")
    return rng.sample(pairs, n)


GEN_PROMPTS = {
    "default": "{question}\n\nKeep your answer under 200 words.",
    "disguised": (
        "Write a comment replying to this r/explainlikeimfive post, the way a regular Redditor "
        "would. Under 200 words. Reply with only the comment text.\n\nPost: {question}"
    ),
}


def generate_answer(client: Any, settings: Any, question: str, style: str) -> str | None:
    from slopbot.detector import fallback_kwargs

    response = client.beta.messages.create(
        model=settings.gen_model,
        max_tokens=16000,
        output_config={"effort": "medium"},
        **fallback_kwargs(settings),
        messages=[{"role": "user", "content": GEN_PROMPTS[style].format(question=question)}],
    )
    if response.stop_reason != "end_turn":
        return None
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text or None


def row(text: str | None, label: str, source: str, label_source: str, **extra: Any) -> dict:
    input_: dict[str, Any] = {"text": text} if text else {}
    for key in ("image_path", "image_paths"):
        if extra.get(key):
            input_[key] = extra.pop(key)
    digest = hashlib.sha1(f"{source}|{json.dumps(input_, sort_keys=True)}".encode()).hexdigest()
    return {
        "id": f"{label}-{digest[:16]}",
        "input": input_,
        "expected": {"label": label},
        "metadata": {"source": source, "label_source": label_source, **extra},
        "tags": [label, source.split("/")[0]],
    }


def custom_rows(path: Path) -> list[dict]:
    out = []
    for r in read_jsonl(path):
        label = r.get("label")
        if label not in LABELS:
            raise SystemExit(f"{path}: label must be one of {LABELS}, got {label!r}")
        if not (r.get("text") or r.get("image_path") or r.get("image_paths")):
            raise SystemExit(f"{path}: each row needs text and/or image_path")
        out.append(
            row(
                r.get("text"),
                label,
                f"custom/{r.get('source', 'unspecified')}",
                r.get("label_source", "labeled by repo owner"),
                image_path=r.get("image_path"),
                image_paths=r.get("image_paths"),
            )
        )
    return out


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--questions", type=int, default=40, help="HC3 questions to sample")
    parser.add_argument("--scan", type=int, default=SCAN_ROWS, help="HC3 rows to download")
    parser.add_argument("--no-generate", action="store_true", help="skip Claude-generated AI rows")
    parser.add_argument("--no-upload", action="store_true", help="don't upload to Braintrust")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "dataset.jsonl")
    args = parser.parse_args()

    rng = random.Random(SEED)
    print(f"Downloading up to {args.scan} HC3 rows...")
    pairs = pick_pairs(stream_hc3(args.scan), args.questions, rng)

    from slopbot.config import get_settings
    from slopbot.gateway import make_anthropic_client, verify_models

    settings = get_settings()
    client = None
    if not args.no_generate:
        verify_models(settings, {"SLOPBOT_GEN_MODEL": settings.gen_model})
        client = make_anthropic_client(settings)

    rows: list[dict] = []
    for i, pair in enumerate(pairs):
        q = pair["question"]
        rows.append(
            row(
                pair["human"],
                "human",
                "hc3/reddit_eli5/human",
                "Reddit comment written before ChatGPT (HC3)",
                question=q,
            )
        )
        use_hc3 = client is None or i % 2 == 0
        generated, style = None, None
        if not use_hc3:
            style = "default" if (i // 2) % 2 == 0 else "disguised"
            print(f"  generating ({style}) {i + 1}/{len(pairs)}")
            generated = generate_answer(client, settings, q, style)
        if generated:
            rows.append(
                row(
                    generated,
                    "ai",
                    f"generated/{settings.gen_model}/{style}",
                    f"generated by {settings.gen_model}",
                    question=q,
                )
            )
        else:
            rows.append(
                row(
                    pair["chatgpt"],
                    "ai",
                    "hc3/reddit_eli5/chatgpt",
                    "ChatGPT answer, Dec 2022 (HC3)",
                    question=q,
                )
            )

    custom_path = DATA_DIR / "custom.jsonl"
    if custom_path.exists():
        extra = custom_rows(custom_path)
        print(f"Adding {len(extra)} rows from {custom_path.relative_to(ROOT)}")
        rows.extend(extra)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    labels = Counter(r["expected"]["label"] for r in rows)
    sources = Counter(r["metadata"]["source"] for r in rows)
    print(f"Wrote {len(rows)} rows to {args.out}  labels={dict(labels)}")
    for source, count in sorted(sources.items()):
        print(f"  {count:3d}  {source}")
    majority = max(labels.values()) / len(rows)
    print(f"Majority-class baseline accuracy: {majority:.0%}")

    if args.no_upload:
        return 0
    if not os.getenv("BRAINTRUST_API_KEY"):
        print("BRAINTRUST_API_KEY not set; skipping upload (local file written).")
        return 0

    import braintrust

    target = settings.braintrust_target
    dataset = braintrust.init_dataset(**target, name=DATASET_NAME)
    for r in rows:
        dataset.insert(
            id=r["id"],
            input=r["input"],
            expected=r["expected"],
            metadata=r["metadata"],
            tags=r["tags"],
        )
    dataset.flush()
    print(f"Uploaded to Braintrust: {target} dataset={DATASET_NAME!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
