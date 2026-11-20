"""`slopbot serve` runs the Slack bot; `slopbot check` judges text or images from your terminal."""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import get_settings


def _serve() -> int:
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    from .slack_app import create_app

    settings = get_settings()
    missing = [
        name
        for name, value in (
            ("SLACK_BOT_TOKEN", settings.slack_bot_token),
            ("SLACK_APP_TOKEN", settings.slack_app_token),
        )
        if not value
    ]
    if missing:
        print(f"Missing {', '.join(missing)}. See README > Slack setup.", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = create_app(settings)
    tracing = "on" if settings.braintrust_enabled else "off"
    logging.getLogger("slopbot").info(
        "Slopbot online (model=%s, effort=%s, braintrust=%s)",
        settings.model,
        settings.effort,
        tracing,
    )
    SocketModeHandler(app, settings.slack_app_token).start()
    return 0


def _check(text: str | None, image_paths: list[str], as_json: bool) -> int:
    import anthropic
    import braintrust

    from .detector import SlopbotError, detect
    from .render import render_text
    from .slack_inputs import prepare_image

    settings = get_settings()
    if settings.braintrust_enabled:
        braintrust.init_logger(project=settings.braintrust_project)
    if text == "-":
        text = sys.stdin.read()

    images = []
    for raw in image_paths:
        path = Path(raw)
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        images.append(prepare_image(path.read_bytes(), media_type, path.name))

    if not (text or "").strip() and not images:
        print("Give me some text (or '-' for stdin) or --image PATH.", file=sys.stderr)
        return 2

    try:
        result = detect(text, images, surface="cli", settings=settings)
    except SlopbotError as e:
        print(f"Slopbot couldn't decide: {e}", file=sys.stderr)
        return 1
    except anthropic.APIError as e:
        print(f"Claude API error: {e}", file=sys.stderr)
        return 1
    except TypeError as e:
        if "authentication" not in str(e):
            raise
        print(
            "No Anthropic credentials found. Set ANTHROPIC_API_KEY in .env "
            "(or run `ant auth login`).",
            file=sys.stderr,
        )
        return 1
    if as_json:
        print(
            json.dumps(
                {
                    **result.verdict.model_dump(),
                    "served_model": result.served_model,
                    "usage": result.usage,
                },
                indent=2,
            )
        )
    else:
        print(render_text(result.verdict))
    if settings.braintrust_enabled:
        braintrust.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="slopbot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="Run the Slack bot (Socket Mode).")
    check = sub.add_parser("check", help="Judge text and/or images without Slack.")
    check.add_argument("text", nargs="?", help="Text to judge, or '-' to read stdin.")
    check.add_argument("--image", action="append", default=[], help="Image file (repeatable).")
    check.add_argument("--json", action="store_true", help="Print the raw verdict as JSON.")
    args = parser.parse_args(argv)

    if args.command == "serve":
        return _serve()
    return _check(args.text, args.image, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
