#!/usr/bin/env python3
"""
P0 安全与负向路径集成测试（deploy/ Face API）

覆盖:
  - 鉴权 401 (无 Key / 错误 Key / 正确 Key)
  - 生产环境 skip_liveness → 403
  - 空底库 1:N → 503 GALLERY_EMPTY
  - 404/400 负向路径 (PERSON_NOT_FOUND, FACE_NOT_FOUND, NO_ENROLLED_FACES, FACE_NOT_DETECTED)

用法:
  pip install -r test/requirements.txt
  API_KEY=your-secret ./test/scripts/run_security_test.sh

生产验收 (COMPOSE_MODE=prod):
  REQUIRE_PRODUCTION=true API_KEY=your-secret ./test/scripts/run_security_test.sh
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests

from common import (
    DEFAULT_ENROLL_ROOT,
    DEFAULT_RESULTS_DIR,
    allows_skip_liveness,
    headers,
    make_no_face_jpeg,
    pick_first_face_image,
    post_json,
    post_multipart,
    reset_gallery,
    seed_gallery_person,
    server_requires_auth,
)


@dataclass
class StepResult:
    name: str
    passed: bool
    skipped: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityTestReport:
    base_url: str
    device: str
    auth_enabled: bool
    steps: list[StepResult]
    passed: int
    failed: int
    skipped: int
    notes: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Face API P0 security & negative-path tests")
    p.add_argument("--base-url", default="http://localhost:8123")
    p.add_argument("--api-key", default=None, help="Valid API key (required when server auth is on)")
    p.add_argument(
        "--require-production",
        action="store_true",
        help="Fail if skip_liveness is not blocked with 403 (production policy)",
    )
    p.add_argument("--enroll-root", type=Path, default=DEFAULT_ENROLL_ROOT)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "security_smoke_report.json")
    return p.parse_args()


def run_step(
    name: str,
    fn: Callable[[], tuple[bool, bool, str, dict[str, Any]]],
) -> StepResult:
    """Returns (passed, skipped, detail, data)."""
    try:
        passed, skipped, detail, data = fn()
        return StepResult(name=name, passed=passed, skipped=skipped, detail=detail, data=data)
    except Exception as exc:
        return StepResult(name=name, passed=False, skipped=False, detail=f"exception: {exc}")


def run_tests(args: argparse.Namespace) -> SecurityTestReport:
    base_url = args.base_url.rstrip("/")
    hdrs = headers(args.api_key)
    session = requests.Session()
    steps: list[StepResult] = []
    notes: list[str] = []
    run_tag = uuid.uuid4().hex[:8]
    test_person_id = f"sec_{run_tag}"
    fake_person_id = f"nonexistent_{run_tag}"
    fake_face_id = f"00000000-0000-0000-0000-{run_tag}000000"

    _, health = post_json(session, "GET", f"{base_url}/v1/health", hdrs, args.timeout)
    device = health.get("device", "unknown")

    auth_enabled = server_requires_auth(session, base_url, args.timeout)
    notes.append(f"server_requires_auth={auth_enabled}")

    skip_allowed = allows_skip_liveness(session, base_url, hdrs, args.timeout)
    notes.append(f"allows_skip_liveness={skip_allowed}")

    face_image: Path | None = None
    try:
        face_image = pick_first_face_image(args.enroll_root)
    except FileNotFoundError as exc:
        notes.append(f"warning: {exc}")

    # --- Auth tests ---
    def step_auth_no_key() -> tuple[bool, bool, str, dict[str, Any]]:
        if not auth_enabled:
            return True, True, "skipped: server has no API_KEY configured", {}
        status, body = post_json(session, "GET", f"{base_url}/v1/persons", {}, args.timeout)
        ok = status == 401 and body.get("error", {}).get("code") == "UNAUTHORIZED"
        return ok, False, f"status={status} code={body.get('error', {}).get('code')}", body

    steps.append(run_step("auth_no_key", step_auth_no_key))

    def step_auth_wrong_key() -> tuple[bool, bool, str, dict[str, Any]]:
        if not auth_enabled:
            return True, True, "skipped: server has no API_KEY configured", {}
        status, body = post_json(
            session,
            "GET",
            f"{base_url}/v1/persons",
            {"X-API-Key": "__wrong_key__"},
            args.timeout,
        )
        ok = status == 401 and body.get("error", {}).get("code") == "UNAUTHORIZED"
        return ok, False, f"status={status} code={body.get('error', {}).get('code')}", body

    steps.append(run_step("auth_wrong_key", step_auth_wrong_key))

    def step_auth_valid_key() -> tuple[bool, bool, str, dict[str, Any]]:
        if not auth_enabled:
            return True, True, "skipped: server has no API_KEY configured", {}
        if not args.api_key:
            return False, False, "failed: --api-key required when server auth is enabled", {}
        status, body = post_json(session, "GET", f"{base_url}/v1/persons", hdrs, args.timeout)
        ok = status == 200
        return ok, False, f"status={status} total={body.get('total')}", body

    steps.append(run_step("auth_valid_key", step_auth_valid_key))

    # --- Production skip_liveness ---
    def step_prod_skip_liveness_identify() -> tuple[bool, bool, str, dict[str, Any]]:
        if face_image is None:
            return False, False, "no face image available for probe", {}
        files = {"image": (face_image.name, face_image.read_bytes(), "image/jpeg")}
        status, body = post_multipart(
            session,
            f"{base_url}/v1/identify",
            files,
            {"skip_liveness": "true"},
            hdrs,
            args.timeout,
        )
        err = body.get("error", {})
        if status == 403 and err.get("code") == "INVALID_REQUEST":
            return True, False, f"production blocks skip_liveness status={status}", body
        if status in (200, 400, 503):
            if args.require_production:
                return (
                    False,
                    False,
                    f"expected 403 in production, got status={status} (development mode?)",
                    body,
                )
            return True, True, f"skipped: development allows skip_liveness status={status}", body
        return False, False, f"unexpected status={status} code={err.get('code')}", body

    steps.append(run_step("prod_skip_liveness_identify", step_prod_skip_liveness_identify))

    def step_prod_skip_liveness_enroll() -> tuple[bool, bool, str, dict[str, Any]]:
        if face_image is None:
            return False, False, "no face image available", {}
        # Create temp person for enroll probe
        create_status, _ = post_json(
            session,
            "POST",
            f"{base_url}/v1/persons",
            hdrs,
            args.timeout,
            json={"person_id": test_person_id, "display_name": "sec_test"},
        )
        if create_status not in (201, 400):
            return False, False, f"setup create_person failed status={create_status}", {}

        files = [("images", (face_image.name, face_image.read_bytes(), "image/jpeg"))]
        status, body = post_multipart(
            session,
            f"{base_url}/v1/persons/{test_person_id}/faces",
            files,
            {"skip_liveness": "true"},
            hdrs,
            args.timeout,
        )
        err = body.get("error", {})
        # cleanup person regardless
        session.delete(f"{base_url}/v1/persons/{test_person_id}", headers=hdrs, timeout=args.timeout)

        if status == 403 and err.get("code") == "INVALID_REQUEST":
            return True, False, f"production blocks skip_liveness on enroll status={status}", body
        if status in (200, 400):
            if args.require_production:
                return (
                    False,
                    False,
                    f"expected 403 in production, got status={status}",
                    body,
                )
            return True, True, f"skipped: development allows skip_liveness on enroll status={status}", body
        return False, False, f"unexpected status={status} code={err.get('code')}", body

    steps.append(run_step("prod_skip_liveness_enroll", step_prod_skip_liveness_enroll))

    # --- GALLERY_EMPTY ---
    def step_gallery_empty_identify() -> tuple[bool, bool, str, dict[str, Any]]:
        deleted = reset_gallery(session, base_url, hdrs, args.timeout)
        no_face = make_no_face_jpeg()
        files = {"image": ("no_gallery.jpg", no_face, "image/jpeg")}
        status, body = post_multipart(
            session,
            f"{base_url}/v1/identify",
            files,
            {},
            hdrs,
            args.timeout,
        )
        err = body.get("error", {})
        ok = status == 503 and err.get("code") == "GALLERY_EMPTY"
        return ok, False, f"deleted={deleted} status={status} code={err.get('code')}", body

    steps.append(run_step("gallery_empty_identify", step_gallery_empty_identify))

    # --- Negative paths ---
    def step_person_not_found_get() -> tuple[bool, bool, str, dict[str, Any]]:
        status, body = post_json(
            session, "GET", f"{base_url}/v1/persons/{fake_person_id}", hdrs, args.timeout
        )
        err = body.get("error", {})
        ok = status == 404 and err.get("code") == "PERSON_NOT_FOUND"
        return ok, False, f"status={status} code={err.get('code')}", body

    steps.append(run_step("person_not_found_get", step_person_not_found_get))

    def step_person_not_found_delete() -> tuple[bool, bool, str, dict[str, Any]]:
        r = session.delete(
            f"{base_url}/v1/persons/{fake_person_id}",
            headers=hdrs,
            timeout=args.timeout,
        )
        try:
            body = r.json()
        except Exception:
            body = {}
        err = body.get("error", {})
        ok = r.status_code == 404 and err.get("code") == "PERSON_NOT_FOUND"
        return ok, False, f"status={r.status_code} code={err.get('code')}", body

    steps.append(run_step("person_not_found_delete", step_person_not_found_delete))

    def step_face_not_found_delete() -> tuple[bool, bool, str, dict[str, Any]]:
        pid = f"sec_face_{run_tag}"
        post_json(
            session,
            "POST",
            f"{base_url}/v1/persons",
            hdrs,
            args.timeout,
            json={"person_id": pid, "display_name": "face_nf_test"},
        )
        r = session.delete(
            f"{base_url}/v1/persons/{pid}/faces/{fake_face_id}",
            headers=hdrs,
            timeout=args.timeout,
        )
        try:
            body = r.json()
        except Exception:
            body = {}
        err = body.get("error", {})
        session.delete(f"{base_url}/v1/persons/{pid}", headers=hdrs, timeout=args.timeout)
        ok = r.status_code == 404 and err.get("code") == "FACE_NOT_FOUND"
        return ok, False, f"status={r.status_code} code={err.get('code')}", body

    steps.append(run_step("face_not_found_delete", step_face_not_found_delete))

    def step_verify_no_enrolled_faces() -> tuple[bool, bool, str, dict[str, Any]]:
        pid = f"sec_noface_{run_tag}"
        post_json(
            session,
            "POST",
            f"{base_url}/v1/persons",
            hdrs,
            args.timeout,
            json={"person_id": pid, "display_name": "no_faces_test"},
        )
        probe = make_no_face_jpeg() if face_image is None else face_image.read_bytes()
        probe_name = "gray.jpg" if face_image is None else face_image.name
        status, body = post_multipart(
            session,
            f"{base_url}/v1/verify",
            {"image": (probe_name, probe, "image/jpeg")},
            {"person_id": pid},
            hdrs,
            args.timeout,
        )
        err = body.get("error", {})
        session.delete(f"{base_url}/v1/persons/{pid}", headers=hdrs, timeout=args.timeout)
        ok = status == 422 and err.get("code") == "NO_ENROLLED_FACES"
        return ok, False, f"status={status} code={err.get('code')}", body

    steps.append(run_step("verify_no_enrolled_faces", step_verify_no_enrolled_faces))

    def step_face_not_detected_identify() -> tuple[bool, bool, str, dict[str, Any]]:
        # Seed gallery without skip_liveness when production forbids it.
        pid = f"sec_seed_{run_tag}"
        seeded, seed_detail, seed_body = seed_gallery_person(
            session,
            base_url,
            pid,
            hdrs,
            args.timeout,
            skip_liveness=skip_allowed,
            enroll_root=args.enroll_root,
            display_name="seed",
        )
        if not seeded:
            return False, False, f"could not seed gallery ({seed_detail})", seed_body

        no_face = make_no_face_jpeg()
        status, body = post_multipart(
            session,
            f"{base_url}/v1/identify",
            {"image": ("no_face.jpg", no_face, "image/jpeg")},
            {},
            hdrs,
            args.timeout,
        )
        err = body.get("error", {})
        session.delete(f"{base_url}/v1/persons/{pid}", headers=hdrs, timeout=args.timeout)
        ok = status == 400 and err.get("code") == "FACE_NOT_DETECTED"
        return (
            ok,
            False,
            f"status={status} code={err.get('code')} seed={seed_detail}",
            body,
        )

    steps.append(run_step("face_not_detected_identify", step_face_not_detected_identify))

    passed = sum(1 for s in steps if s.passed and not s.skipped)
    skipped = sum(1 for s in steps if s.skipped)
    failed = sum(1 for s in steps if not s.passed and not s.skipped)

    return SecurityTestReport(
        base_url=base_url,
        device=device,
        auth_enabled=auth_enabled,
        steps=steps,
        passed=passed,
        failed=failed,
        skipped=skipped,
        notes=notes,
    )


def print_report(report: SecurityTestReport) -> None:
    print("\n" + "=" * 60)
    print("Face Access Control — Security Smoke Test")
    print("=" * 60)
    print(f"Base URL    : {report.base_url}")
    print(f"Device      : {report.device}")
    print(f"Auth enabled: {report.auth_enabled}")
    total = len(report.steps)
    print(f"Result      : {report.passed} passed, {report.failed} failed, {report.skipped} skipped / {total}")

    for step in report.steps:
        if step.skipped:
            mark = "SKIP"
        else:
            mark = "PASS" if step.passed else "FAIL"
        print(f"\n[{mark}] {step.name}")
        print(f"       {step.detail}")

    if report.notes:
        print("\nNotes:")
        for n in report.notes:
            print(f"  - {n}")
    print("\n" + "=" * 60)


def main() -> None:
    args = parse_args()
    report = run_tests(args)
    print_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(report)
        for step in payload["steps"]:
            step.pop("data", None)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"JSON report: {args.output}")

    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()
