import io

from PIL import Image

from slopbot.slack_inputs import (
    MAX_EDGE,
    MAX_IMAGES,
    extract_content,
    find_target_message,
    message_text,
    prepare_image,
    strip_mentions,
)
from tests.conftest import png_bytes


def test_strip_mentions():
    assert strip_mentions("<@U123ABC> is this AI?") == "is this AI?"
    assert strip_mentions("hey <@U1|slopbot>  look\nhere") == "hey look\nhere"
    assert strip_mentions("<@U123>") == ""


def test_message_text_includes_shared_attachments():
    msg = {"text": "look at this", "attachments": [{"text": "forwarded slop"}, {"fallback": "fb"}]}
    assert message_text(msg) == "look at this\n\nforwarded slop\n\nfb"


def test_small_image_keeps_original_bytes():
    data = png_bytes((100, 80))
    image = prepare_image(data, "image/png", "a.png")
    assert image.data == data and image.media_type == "image/png"


def test_large_image_is_downscaled_to_jpeg():
    out = io.BytesIO()
    Image.new("RGB", (4000, 3000), (10, 20, 30)).save(out, format="JPEG")
    image = prepare_image(out.getvalue(), "image/jpeg")
    assert image.media_type == "image/jpeg"
    with Image.open(io.BytesIO(image.data)) as img:
        assert max(img.size) == MAX_EDGE


def test_large_transparent_image_stays_png():
    image = prepare_image(png_bytes((3000, 2000), mode="RGBA"), "image/png")
    assert image.media_type == "image/png"
    with Image.open(io.BytesIO(image.data)) as img:
        assert max(img.size) <= MAX_EDGE


def test_unsupported_format_is_converted():
    out = io.BytesIO()
    Image.new("RGB", (50, 50)).save(out, format="BMP")
    image = prepare_image(out.getvalue(), "image/bmp")
    assert image.media_type == "image/jpeg"


def test_extract_content_handles_files():
    files = [{"name": f"img{i}.png", "mimetype": "image/png"} for i in range(MAX_IMAGES + 1)]
    files += [
        {"name": "notes.pdf", "mimetype": "application/pdf"},
        {"name": "broken.png", "mimetype": "image/png", "broken": True},
    ]
    calls = []

    def downloader(file, token):
        calls.append((file["name"], token))
        if file.get("broken"):
            raise PermissionError("nope")
        return png_bytes()

    content = extract_content({"text": "<@U1> check", "files": files}, "xoxb-test", downloader)
    assert content.text == "check"
    assert len(content.images) == MAX_IMAGES
    assert set(content.skipped) == {f"img{MAX_IMAGES}.png", "notes.pdf", "broken.png"}
    assert all(token == "xoxb-test" for _, token in calls)
    assert not content.is_empty


class FakeSlack:
    def __init__(self, replies=None, history=None):
        self.replies, self.history, self.calls = replies or [], history or [], []

    def conversations_replies(self, **kwargs):
        self.calls.append(("replies", kwargs))
        return {"messages": self.replies}

    def conversations_history(self, **kwargs):
        self.calls.append(("history", kwargs))
        return {"messages": self.history}


def test_mention_in_thread_judges_parent():
    parent = {"ts": "1.0", "text": "a suspicious post"}
    slack = FakeSlack(replies=[parent])
    event = {"channel": "C1", "ts": "2.0", "thread_ts": "1.0", "text": "<@U1> is this AI?"}
    assert find_target_message(event, slack) == (parent, "thread_parent")
    assert slack.calls[0][1]["ts"] == "1.0"


def test_mention_with_text_at_top_level_judges_itself():
    event = {"channel": "C1", "ts": "2.0", "text": "<@U1> In today's fast-paced world..."}
    assert find_target_message(event, FakeSlack()) == (event, "self")


def test_mention_with_files_judges_itself_even_in_thread():
    event = {"channel": "C1", "ts": "2.0", "thread_ts": "1.0", "text": "<@U1>", "files": [{}]}
    assert find_target_message(event, FakeSlack()) == (event, "self")


def test_bare_mention_judges_previous_message():
    previous = {"ts": "1.5", "text": "earlier message"}
    slack = FakeSlack(history=[previous])
    event = {"channel": "C1", "ts": "2.0", "text": "<@U1>"}
    assert find_target_message(event, slack) == (previous, "previous")
    assert slack.calls[0][1]["latest"] == "2.0"


def test_bare_mention_in_empty_channel():
    event = {"channel": "C1", "ts": "2.0", "text": "<@U1>"}
    assert find_target_message(event, FakeSlack()) == (None, "none")
