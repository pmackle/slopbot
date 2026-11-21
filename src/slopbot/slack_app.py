"""Slack wiring: @mentions, DMs, the "Slop check" message shortcut, and feedback buttons."""

from __future__ import annotations

import logging
from typing import Any

import braintrust
from slack_bolt import App
from slack_sdk.errors import SlackApiError

from .config import Settings
from .detector import DetectionRefused, SlopbotError, detect
from .render import FEEDBACK_DOWN, FEEDBACK_UP, render_blocks, render_text
from .slack_inputs import SlackContent, extract_content, find_target_message

log = logging.getLogger("slopbot")

WORKING_EMOJI = "hourglass_flowing_sand"

USAGE = (
    "Mention me in a thread to judge the parent message, mention me right after a message to "
    "judge that one, DM me the suspect text, or use the *Slop check* message shortcut "
    "(the ⋯ menu on any message)."
)
EMPTY = "There's nothing here to judge. I checked twice."
REFUSED = "I've been instructed not to judge this one. I'm choosing to respect that."
BROKEN = "Something on my end broke. It wasn't you. Try again in a minute."
NOT_IN_CHANNEL = (
    "I'm not in this channel, so only you can see this. Invite me (`/invite @Slopbot`) "
    "if you'd like witnesses."
)


def _skipped_note(content: SlackContent) -> str:
    if not content.skipped:
        return ""
    names = ", ".join(content.skipped[:3])
    more = "" if len(content.skipped) <= 3 else f" and {len(content.skipped) - 3} more"
    return f"\n_(Ignored {names}{more}: I only judge text and images.)_"


def judge(content: SlackContent, surface: str) -> tuple[str, list[dict[str, Any]] | None]:
    """Return (fallback_text, blocks) for a reply. Never raises."""
    if content.is_empty:
        return EMPTY + _skipped_note(content), None
    try:
        result = detect(content.text, content.images, surface=surface)
    except DetectionRefused:
        return REFUSED, None
    except SlopbotError as e:
        log.warning("detection failed: %s", e)
        return BROKEN, None
    except Exception:
        log.exception("detection crashed")
        return BROKEN, None
    text = render_text(result.verdict) + _skipped_note(content)
    blocks = render_blocks(result.verdict, result.span_id)
    blocks[0]["text"]["text"] = text
    return text, blocks


def _react(client: Any, channel: str, ts: str, add: bool) -> None:
    try:
        if add:
            client.reactions_add(channel=channel, timestamp=ts, name=WORKING_EMOJI)
        else:
            client.reactions_remove(channel=channel, timestamp=ts, name=WORKING_EMOJI)
    except SlackApiError:
        pass  # cosmetic only


def _judge_and_reply(
    client: Any,
    token: str,
    *,
    message: dict[str, Any],
    channel: str,
    thread_ts: str,
    react_ts: str,
    surface: str,
) -> None:
    _react(client, channel, react_ts, add=True)
    try:
        content = extract_content(message, token)
        text, blocks = judge(content, surface)
        client.chat_postMessage(
            channel=channel, thread_ts=thread_ts, text=text, blocks=blocks, unfurl_links=False
        )
    finally:
        _react(client, channel, react_ts, add=False)


def create_app(settings: Settings, **app_kwargs: Any) -> App:
    if not settings.slack_bot_token:
        raise RuntimeError("SLACK_BOT_TOKEN is not set")
    token = settings.slack_bot_token
    app = App(token=token, **app_kwargs)
    # Production tracing: every detect() call becomes a trace in this Braintrust project.
    bt_logger = (
        braintrust.init_logger(**settings.braintrust_target)
        if settings.braintrust_enabled
        else None
    )

    @app.event("app_mention")
    def on_mention(event: dict[str, Any], client: Any) -> None:
        channel = event["channel"]
        reply_ts = event.get("thread_ts") or event["ts"]
        target, where = find_target_message(event, client)
        if target is None:
            client.chat_postMessage(channel=channel, thread_ts=reply_ts, text=USAGE)
            return
        _judge_and_reply(
            client,
            token,
            message=target,
            channel=channel,
            thread_ts=reply_ts,
            react_ts=event["ts"],
            surface=f"mention:{where}",
        )

    @app.event("message")
    def on_message(event: dict[str, Any], client: Any) -> None:
        # Only DMs; channel messages reach us via app_mention instead.
        if event.get("channel_type") != "im" or event.get("bot_id"):
            return
        if event.get("subtype") not in (None, "file_share"):
            return
        _judge_and_reply(
            client,
            token,
            message=event,
            channel=event["channel"],
            thread_ts=event.get("thread_ts") or event["ts"],
            react_ts=event["ts"],
            surface="dm",
        )

    @app.shortcut("slop_check")
    def on_shortcut(ack: Any, shortcut: dict[str, Any], client: Any, respond: Any) -> None:
        ack()
        message = shortcut["message"]
        channel = shortcut["channel"]["id"]
        content = extract_content(message, token)
        text, blocks = judge(content, "shortcut")
        try:
            client.chat_postMessage(
                channel=channel,
                thread_ts=message.get("thread_ts") or message["ts"],
                text=text,
                blocks=blocks,
                unfurl_links=False,
            )
        except SlackApiError as e:
            if e.response.get("error") not in ("not_in_channel", "channel_not_found"):
                raise
            respond(
                response_type="ephemeral",
                replace_original=False,
                text=f"{text}\n\n_{NOT_IN_CHANNEL}_",
            )

    def _feedback(ack: Any, action: dict[str, Any], respond: Any, score: float) -> None:
        ack()
        span_id = action.get("value")
        if bt_logger is not None and span_id:
            bt_logger.log_feedback(id=span_id, scores={"slack_thumbs": score}, source="external")
        reply = (
            "Noted. I'll try to stay humble about it."
            if score
            else "Noted. I'll do better. Statistically."
        )
        respond(response_type="ephemeral", replace_original=False, text=reply)

    @app.action(FEEDBACK_UP)
    def on_feedback_up(ack: Any, action: dict[str, Any], respond: Any) -> None:
        _feedback(ack, action, respond, 1.0)

    @app.action(FEEDBACK_DOWN)
    def on_feedback_down(ack: Any, action: dict[str, Any], respond: Any) -> None:
        _feedback(ack, action, respond, 0.0)

    return app
