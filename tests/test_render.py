from slopbot.render import (
    FEEDBACK_DOWN,
    FEEDBACK_UP,
    MAX_REASONS,
    escape,
    headline,
    render_blocks,
    render_text,
)
from tests.conftest import make_verdict


def test_typical_reply_is_short(verdict):
    text = render_text(verdict)
    assert text.startswith(":robot_face: *92% slop.* More em dashes")
    assert len(text.split()) <= 60
    assert text.count("\n• ") == 2


def test_reasons_are_capped_and_blank_ones_dropped():
    v = make_verdict(reasons=["one", "  ", "two", "three", "four", "five"])
    lines = render_text(v).split("\n")[1:]
    assert lines == ["• one", "• two", "• three"]
    assert len(lines) == MAX_REASONS


def test_long_reason_is_clipped():
    v = make_verdict(reasons=["x " * 300])
    reason = render_text(v).split("\n")[1]
    assert len(reason) <= 165 and reason.endswith("…")


def test_slack_control_characters_are_escaped():
    v = make_verdict(quip="<!channel> & friends", reasons=["Contains <https://evil.example|link>"])
    text = render_text(v)
    assert "<!channel>" not in text and "&lt;!channel&gt; &amp; friends" in text
    assert "<https://" not in text
    assert escape("a<b>&c") == "a&lt;b&gt;&amp;c"


def test_headlines_per_verdict():
    assert headline(make_verdict(verdict="human", ai_probability=8)).startswith(
        ":bust_in_silhouette: *8% slop.*"
    )
    assert "*Inconclusive.*" in headline(make_verdict(verdict="unsure", ai_probability=50))
    assert "partially" in headline(make_verdict(verdict="mixed", ai_probability=60))
    assert "*100% slop.*" in headline(make_verdict(ai_probability=140))


def test_feedback_buttons_only_with_span_id(verdict):
    assert len(render_blocks(verdict)) == 1
    blocks = render_blocks(verdict, span_id="span-123")
    assert len(blocks) == 2
    buttons = blocks[1]["elements"]
    assert [b["action_id"] for b in buttons] == [FEEDBACK_UP, FEEDBACK_DOWN]
    assert all(b["value"] == "span-123" for b in buttons)
