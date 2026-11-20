"""Pull judgeable text and images out of Slack messages."""

from __future__ import annotations

import io
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from PIL import Image, ImageOps

from .detector import SUPPORTED_IMAGE_TYPES, ImageInput

MAX_IMAGES = 4
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
# Claude downsamples anything larger than this on the long edge, so sending more is wasted bytes.
MAX_EDGE = 1568
# Keep each base64 image comfortably under the API's per-image size limit.
MAX_RAW_IMAGE_BYTES = 3_500_000

_MENTION_RE = re.compile(r"<@[A-Z0-9]+(?:\|[^>]*)?>")

Downloader = Callable[[dict[str, Any], str], bytes]


@dataclass
class SlackContent:
    text: str
    images: list[ImageInput] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip() and not self.images


def strip_mentions(text: str) -> str:
    without = _MENTION_RE.sub(" ", text or "")
    return re.sub(r"[ \t]{2,}", " ", without).strip()


def message_text(message: dict[str, Any]) -> str:
    """Message text plus text from shared/forwarded message attachments."""
    parts = [message.get("text") or ""]
    for attachment in message.get("attachments") or []:
        parts.append(attachment.get("text") or attachment.get("fallback") or "")
    return "\n\n".join(p for p in parts if p.strip())


def download_file(file: dict[str, Any], token: str) -> bytes:
    url = file.get("url_private_download") or file.get("url_private")
    if not url:
        raise ValueError("file has no private URL")
    with httpx.stream(
        "GET",
        url,
        headers={"Authorization": f"Bearer {token}"},
        follow_redirects=True,
        timeout=30,
    ) as response:
        response.raise_for_status()
        if "text/html" in response.headers.get("content-type", ""):
            # Slack serves a login page instead of the file when the token lacks files:read.
            raise PermissionError("Slack returned HTML instead of the file (missing files:read?)")
        data = bytearray()
        for chunk in response.iter_bytes():
            data.extend(chunk)
            if len(data) > MAX_DOWNLOAD_BYTES:
                raise ValueError("file too large")
        return bytes(data)


def prepare_image(data: bytes, media_type: str, name: str = "image") -> ImageInput:
    """Keep the original bytes when possible (artifacts matter); otherwise downscale/re-encode."""
    with Image.open(io.BytesIO(data)) as img:
        width, height = img.size
        if (
            media_type in SUPPORTED_IMAGE_TYPES
            and max(width, height) <= MAX_EDGE
            and len(data) <= MAX_RAW_IMAGE_BYTES
        ):
            return ImageInput(media_type=media_type, data=data, name=name)

        img.seek(0)  # first frame for animated images
        frame = ImageOps.exif_transpose(img)
        frame.thumbnail((MAX_EDGE, MAX_EDGE))
        has_alpha = frame.mode in ("RGBA", "LA") or "transparency" in frame.info
        out = io.BytesIO()
        if has_alpha:
            frame.convert("RGBA").save(out, format="PNG", optimize=True)
            return ImageInput(media_type="image/png", data=out.getvalue(), name=name)
        frame.convert("RGB").save(out, format="JPEG", quality=92)
        return ImageInput(media_type="image/jpeg", data=out.getvalue(), name=name)


def extract_content(
    message: dict[str, Any],
    token: str,
    downloader: Downloader = download_file,
) -> SlackContent:
    content = SlackContent(text=strip_mentions(message_text(message)))
    for file in message.get("files") or []:
        name = file.get("name") or file.get("title") or "file"
        mimetype = (file.get("mimetype") or "").lower()
        if not mimetype.startswith("image/"):
            content.skipped.append(name)
            continue
        if len(content.images) >= MAX_IMAGES:
            content.skipped.append(name)
            continue
        try:
            data = downloader(file, token)
            content.images.append(prepare_image(data, mimetype, name))
        except Exception:  # noqa: BLE001 - one bad file shouldn't sink the whole verdict
            content.skipped.append(name)
    return content


def find_target_message(event: dict[str, Any], client: Any) -> tuple[dict[str, Any] | None, str]:
    """Decide which message an @mention is asking about.

    Returns (message, where) with `where` in {"self", "thread_parent", "previous", "none"}:
    - the mention carries files, or has real text at top level: judge the mention itself
    - the mention is a thread reply: judge the thread's parent message
    - a bare mention at top level: judge the message posted just before it
    """
    has_files = bool(event.get("files"))
    own_text = strip_mentions(message_text(event))
    thread_ts = event.get("thread_ts")
    in_thread = bool(thread_ts) and thread_ts != event.get("ts")

    if has_files:
        return event, "self"
    if in_thread:
        replies = client.conversations_replies(
            channel=event["channel"], ts=thread_ts, limit=1, inclusive=True
        )
        messages = replies.get("messages") or []
        return (messages[0], "thread_parent") if messages else (None, "none")
    if own_text:
        return event, "self"
    history = client.conversations_history(
        channel=event["channel"], latest=event["ts"], limit=1, inclusive=False
    )
    messages = history.get("messages") or []
    return (messages[0], "previous") if messages else (None, "none")
