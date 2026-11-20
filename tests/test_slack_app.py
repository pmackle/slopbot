import pytest

from slopbot import slack_app
from slopbot.detector import DetectionRefused, DetectionResult, SlopbotError
from slopbot.slack_inputs import SlackContent
from tests.conftest import make_verdict


def test_empty_content_short_circuits(monkeypatch):
    monkeypatch.setattr(slack_app, "detect", lambda *a, **k: pytest.fail("should not call"))
    text, blocks = slack_app.judge(SlackContent(text="  ", skipped=["deck.pdf"]), "dm")
    assert text.startswith(slack_app.EMPTY) and "deck.pdf" in text
    assert blocks is None


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (DetectionRefused("cyber"), slack_app.REFUSED),
        (SlopbotError("bad"), slack_app.BROKEN),
        (RuntimeError("boom"), slack_app.BROKEN),
    ],
)
def test_errors_become_friendly_replies(monkeypatch, error, expected):
    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr(slack_app, "detect", boom)
    assert slack_app.judge(SlackContent(text="some text"), "dm") == (expected, None)


def test_success_renders_blocks_with_feedback(monkeypatch):
    result = DetectionResult(
        verdict=make_verdict(), served_model="claude-opus-5-5", fell_back=False, span_id="abc"
    )
    seen = {}

    def fake_detect(text, images, *, surface):
        seen.update(text=text, surface=surface)
        return result

    monkeypatch.setattr(slack_app, "detect", fake_detect)
    content = SlackContent(text="hello there", skipped=["clip.mp4"])
    text, blocks = slack_app.judge(content, "shortcut")
    assert seen == {"text": "hello there", "surface": "shortcut"}
    assert "*92% slop.*" in text and "clip.mp4" in text
    assert blocks[0]["text"]["text"] == text
    assert blocks[1]["elements"][0]["value"] == "abc"


def test_create_app_registers_listeners():
    from slopbot.config import Settings

    app = slack_app.create_app(
        Settings(slack_bot_token="xoxb-test"), token_verification_enabled=False
    )
    names = sorted(listener.ack_function.__name__ for listener in app._listeners)
    assert names == [
        "on_feedback_down",
        "on_feedback_up",
        "on_mention",
        "on_message",
        "on_shortcut",
    ]
