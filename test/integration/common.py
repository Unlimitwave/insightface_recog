"""Shared HTTP helpers for integration tests."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import requests

TEST_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENROLL_ROOT = TEST_ROOT / "enroll_images"
DEFAULT_ENROLL_ADD_ROOT = TEST_ROOT / "enroll_add"
DEFAULT_RESULTS_DIR = TEST_ROOT / "results"

# Prefer persons whose stock photos usually pass size/liveness gates.
PREFERRED_ENROLL_DIRS: tuple[str, ...] = ("ym", "wjr", "whd", "zjy", "tjc")


def headers(api_key: str | None) -> dict[str, str]:
    h: dict[str, str] = {}
    if api_key:
        h["X-API-Key"] = api_key
    return h


def list_images(directory: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts)


def list_face_images(
    enroll_root: Path | None = None,
    *,
    prefer_dirs: tuple[str, ...] = PREFERRED_ENROLL_DIRS,
) -> list[Path]:
    """All face images under enroll_root, preferred person dirs first."""
    root = enroll_root or DEFAULT_ENROLL_ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"Enroll root not found: {root}")

    by_dir: dict[str, list[Path]] = {}
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        images = list_images(subdir)
        if images:
            by_dir[subdir.name] = images

    ordered: list[Path] = []
    seen: set[Path] = set()
    for name in prefer_dirs:
        for path in by_dir.get(name, []):
            if path not in seen:
                ordered.append(path)
                seen.add(path)
    for name in sorted(by_dir):
        for path in by_dir[name]:
            if path not in seen:
                ordered.append(path)
                seen.add(path)
    return ordered


def pick_first_face_image(enroll_root: Path | None = None) -> Path:
    """Return a preferred face image from enroll_images subdirectories."""
    images = list_face_images(enroll_root)
    if not images:
        root = enroll_root or DEFAULT_ENROLL_ROOT
        raise FileNotFoundError(f"No face images found under {root}")
    return images[0]


def make_no_face_jpeg() -> bytes:
    """Solid gray JPEG without a face (for FACE_NOT_DETECTED tests)."""
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (640, 480), color=(128, 128, 128)).save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c"
            b"\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c"
            b"\x1c $.\' \",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08"
            b"\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01"
            b"\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03"
            b"\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03"
            b"\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05"
            b"\x12!1A\x06\x13Qa\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0"
            b"$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghij"
            b"stuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98"
            b"\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7"
            b"\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6"
            b"\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3"
            b"\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb"
            b"\xd5\xdb \xff\xd9"
        )


def post_json(
    session: requests.Session,
    method: str,
    url: str,
    hdrs: dict[str, str],
    timeout: float,
    **kwargs: Any,
) -> tuple[int, dict[str, Any]]:
    r = session.request(method, url, headers=hdrs, timeout=timeout, **kwargs)
    try:
        body = r.json() if r.content else {}
    except Exception:
        body = {"raw": r.text}
    return r.status_code, body


def post_multipart(
    session: requests.Session,
    url: str,
    files: dict[str, tuple] | list[tuple],
    params: dict[str, str],
    hdrs: dict[str, str],
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    r = session.post(url, files=files, params=params, headers=hdrs, timeout=timeout)
    try:
        body = r.json()
    except Exception:
        body = {"raw": r.text}
    return r.status_code, body


def server_requires_auth(
    session: requests.Session,
    base_url: str,
    timeout: float,
) -> bool:
    """True when the service rejects requests without a valid API key."""
    status, _ = post_json(session, "GET", f"{base_url}/v1/persons", {}, timeout)
    if status == 401:
        return True
    status, _ = post_json(
        session,
        "GET",
        f"{base_url}/v1/persons",
        {"X-API-Key": "__invalid_probe_key__"},
        timeout,
    )
    return status == 401


def allows_skip_liveness(
    session: requests.Session,
    base_url: str,
    hdrs: dict[str, str],
    timeout: float,
) -> bool:
    """False when ENVIRONMENT=production (skip_liveness → 403 INVALID_REQUEST)."""
    status, body = post_multipart(
        session,
        f"{base_url.rstrip('/')}/v1/identify",
        {"image": ("probe.jpg", make_no_face_jpeg(), "image/jpeg")},
        {"skip_liveness": "true"},
        hdrs,
        timeout,
    )
    err = body.get("error", {})
    return not (status == 403 and err.get("code") == "INVALID_REQUEST")


def enroll_params(skip_liveness: bool) -> dict[str, str]:
    return {"skip_liveness": "true"} if skip_liveness else {}


def try_enroll_one_face(
    session: requests.Session,
    base_url: str,
    person_id: str,
    image_path: Path,
    hdrs: dict[str, str],
    timeout: float,
    *,
    skip_liveness: bool,
) -> tuple[bool, int, dict[str, Any]]:
    """POST one image; return (ok, status, body). ok when at least one face enrolled."""
    status, body = post_multipart(
        session,
        f"{base_url.rstrip('/')}/v1/persons/{person_id}/faces",
        [("images", (image_path.name, image_path.read_bytes(), "image/jpeg"))],
        enroll_params(skip_liveness),
        hdrs,
        timeout,
    )
    ok = status == 200 and len(body.get("enrolled", [])) > 0
    return ok, status, body


def seed_gallery_person(
    session: requests.Session,
    base_url: str,
    person_id: str,
    hdrs: dict[str, str],
    timeout: float,
    *,
    skip_liveness: bool,
    enroll_root: Path | None = None,
    display_name: str = "seed",
) -> tuple[bool, str, dict[str, Any]]:
    """Create person and enroll first successful image. Cleans up person on total failure."""
    create_status, create_body = post_json(
        session,
        "POST",
        f"{base_url.rstrip('/')}/v1/persons",
        hdrs,
        timeout,
        json={"person_id": person_id, "display_name": display_name},
    )
    if create_status not in (201, 400):
        return False, f"create_person status={create_status}", create_body

    last_detail = "no images"
    last_body: dict[str, Any] = {}
    for path in list_face_images(enroll_root):
        ok, status, body = try_enroll_one_face(
            session,
            base_url,
            person_id,
            path,
            hdrs,
            timeout,
            skip_liveness=skip_liveness,
        )
        last_body = body
        if ok:
            return True, f"enrolled from {path.parent.name}/{path.name}", body
        err = body.get("error", {})
        last_detail = f"{path.name} status={status} code={err.get('code')} msg={err.get('message')}"

    session.delete(
        f"{base_url.rstrip('/')}/v1/persons/{person_id}",
        headers=hdrs,
        timeout=timeout,
    )
    return False, last_detail, last_body


def reset_gallery(
    session: requests.Session,
    base_url: str,
    hdrs: dict[str, str],
    timeout: float,
) -> int:
    status, body = post_json(
        session, "GET", f"{base_url}/v1/persons", hdrs, timeout, params={"limit": 500}
    )
    if status != 200:
        return 0
    deleted = 0
    for item in body.get("items", []):
        r = session.delete(
            f"{base_url}/v1/persons/{item['person_id']}",
            headers=hdrs,
            timeout=timeout,
        )
        if r.status_code in (204, 404):
            deleted += 1
    return deleted
