"""Slopbot from the terminal.

slopbot serve     run the Slack bot (Socket Mode)
slopbot check     judge text or images without Slack
slopbot models    show the gateway's models and whether the configured ones are there
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import ConfigError, Settings, get_settings
from .gateway import GatewayError, fetch_gateway_models, iap_source, verify_models


def _configured_models(settings: Settings) -> dict[str, str]:
    return {
        "SLOPBOT_MODEL": settings.model,
        "SLOPBOT_JUDGE_MODEL": settings.judge_model,
        "SLOPBOT_GEN_MODEL": settings.gen_model,
    }


def _models() -> int:
    settings = get_settings()
    if not settings.uses_gateway:
        print("ANTHROPIC_BASE_URL isn't set, so Slopbot talks to Anthropic directly.")
        for env, name in _configured_models(settings).items():
            print(f"  {env:<20} {name}")
        return 0

    print(f"Gateway: {settings.base_url}")
    iap_status = f"on, signing as {settings.iap_service_account}" if settings.iap else "off"
    print(f"Cloud IAP: {iap_status}")
    available = fetch_gateway_models(settings)
    print(f"\nThis key can use {len(available)} model(s):")
    for name in available:
        print(f"  {name}")
    print("\nConfigured:")
    ok = True
    for env, name in _configured_models(settings).items():
        found = name in available
        ok &= found
        print(f"  {'✓' if found else '✗'} {env:<20} {name}")
    return 0 if ok else 1


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
    verify_models(settings, {"SLOPBOT_MODEL": settings.model})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = create_app(settings)
    logging.getLogger("slopbot").info(
        "Slopbot online (model=%s, effort=%s, gateway=%s, iap=%s, braintrust=%s)",
        settings.model,
        settings.effort,
        settings.base_url or "off",
        "on" if settings.iap else "off",
        "on" if settings.braintrust_enabled else "off",
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
        braintrust.init_logger(**settings.braintrust_target)
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

    # Sign the IAP token up front so a setup problem shows its real message instead of
    # surfacing later as a generic connection error.
    iap = iap_source(settings)
    if iap is not None:
        iap.token()

    try:
        result = detect(text, images, surface="cli", settings=settings)
    except SlopbotError as e:
        print(f"Slopbot couldn't decide: {e}", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError as e:
        cause = f" ({e.__cause__})" if e.__cause__ else ""
        print(f"Couldn't reach the model API: {e}{cause}", file=sys.stderr)
        return 1
    except anthropic.APIError as e:
        print(f"Claude API error: {e}", file=sys.stderr)
        return 1
    except TypeError as e:
        if "authentication" not in str(e):
            raise
        print(
            "No credentials found. Set ANTHROPIC_API_KEY in .env (your Anthropic key, or your "
            "LiteLLM virtual key when ANTHROPIC_BASE_URL points at a gateway).",
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
                    "cost_usd": result.cost_usd,
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
    parser = argparse.ArgumentParser(
        prog="slopbot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="Run the Slack bot (Socket Mode).")
    check = sub.add_parser("check", help="Judge text and/or images without Slack.")
    check.add_argument("text", nargs="?", help="Text to judge, or '-' to read stdin.")
    check.add_argument("--image", action="append", default=[], help="Image file (repeatable).")
    check.add_argument("--json", action="store_true", help="Print the raw verdict as JSON.")
    sub.add_parser("models", help="List the gateway's models and check the configured ones.")
    args = parser.parse_args(argv)

    try:
        if args.command == "serve":
            return _serve()
        if args.command == "models":
            return _models()
        return _check(args.text, args.image, args.json)
    except (ConfigError, GatewayError) as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
