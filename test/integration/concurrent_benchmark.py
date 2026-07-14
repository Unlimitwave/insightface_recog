#!/usr/bin/env python3
"""
Concurrent / QPS benchmark for the face access-control API (deploy/).

Sends parallel identify or verify requests and reports throughput + latency percentiles.

Usage:
  python test/integration/concurrent_benchmark.py \\
    --endpoint identify \\
    --probe-image test/enroll_images/wjr/photo.jpg \\
    --workers 4 --requests 100 \\
    --output test/results/concurrent_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import local
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests

from common import (
    DEFAULT_RESULTS_DIR,
    headers,
    list_face_images,
    pick_first_face_image,
    try_enroll_one_face,
)
from latency_benchmark import LatencyStats, check_health, create_person, delete_person


_thread_local = local()


def _session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


@dataclass
class ConcurrentReport:
    base_url: str
    endpoint: str
    device: str
    person_id: str | None
    workers: int
    total_requests: int
    successful: int
    failed: int
    qps: float
    duration_sec: float
    latency_wall_ms: LatencyStats
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Face API concurrent / QPS benchmark")
    p.add_argument("--base-url", default="http://localhost:8123")
    p.add_argument("--api-key", default=None)
    p.add_argument("--endpoint", choices=["identify", "verify"], default="identify")
    p.add_argument("--probe-image", type=Path, help="Probe face image")
    p.add_argument("--enroll-image", type=Path, help="Enrollment image (verify mode setup)")
    p.add_argument("--person-id", default=None, help="Person ID for verify (auto-created if unset)")
    p.add_argument("--workers", type=int, default=4, help="Concurrent worker threads")
    p.add_argument("--requests", type=int, default=100, help="Total requests to send")
    p.add_argument("--skip-liveness", action="store_true")
    p.add_argument("--skip-setup", action="store_true", help="Skip enroll setup (gallery must exist)")
    p.add_argument("--skip-cleanup", action="store_true")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "concurrent_report.json")
    return p.parse_args()


def _one_identify(
    base_url: str,
    probe_bytes: bytes,
    probe_name: str,
    skip_liveness: bool,
    hdrs: dict[str, str],
    timeout: float,
) -> tuple[float, bool, str]:
    params = {"skip_liveness": "true"} if skip_liveness else {}
    files = {"image": (probe_name, probe_bytes, "image/jpeg")}
    t0 = time.perf_counter()
    try:
        r = _session().post(
            f"{base_url}/v1/identify",
            files=files,
            params=params,
            headers=hdrs,
            timeout=timeout,
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if r.status_code == 200:
            return wall_ms, True, ""
        body = r.json() if r.content else {}
        msg = body.get("error", {}).get("message", r.text[:200])
        return wall_ms, False, f"HTTP {r.status_code}: {msg}"
    except Exception as exc:
        wall_ms = (time.perf_counter() - t0) * 1000.0
        return wall_ms, False, str(exc)


def _one_verify(
    base_url: str,
    person_id: str,
    probe_bytes: bytes,
    probe_name: str,
    skip_liveness: bool,
    hdrs: dict[str, str],
    timeout: float,
) -> tuple[float, bool, str]:
    params = {"person_id": person_id}
    if skip_liveness:
        params["skip_liveness"] = "true"
    files = {"image": (probe_name, probe_bytes, "image/jpeg")}
    t0 = time.perf_counter()
    try:
        r = _session().post(
            f"{base_url}/v1/verify",
            files=files,
            params=params,
            headers=hdrs,
            timeout=timeout,
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if r.status_code == 200:
            return wall_ms, True, ""
        body = r.json() if r.content else {}
        msg = body.get("error", {}).get("message", r.text[:200])
        return wall_ms, False, f"HTTP {r.status_code}: {msg}"
    except Exception as exc:
        wall_ms = (time.perf_counter() - t0) * 1000.0
        return wall_ms, False, str(exc)


def run_benchmark(args: argparse.Namespace) -> ConcurrentReport:
    probe_image = args.probe_image
    if probe_image is None:
        probe_image = pick_first_face_image()
    if not probe_image.is_file():
        print(f"ERROR: probe image not found: {probe_image}", file=sys.stderr)
        sys.exit(2)

    enroll_image = args.enroll_image or probe_image
    base_url = args.base_url.rstrip("/")
    hdrs = headers(args.api_key)
    person_id = args.person_id or f"conc_{uuid.uuid4().hex[:12]}"
    notes: list[str] = []
    setup_session = requests.Session()

    health = check_health(setup_session, base_url, args.timeout)
    device = health.get("device", "unknown")

    if not args.skip_setup:
        create_person(setup_session, base_url, person_id, hdrs, args.timeout)
        candidates = [enroll_image]
        for path in list_face_images():
            if path.resolve() != enroll_image.resolve():
                candidates.append(path)
        enrolled_from: Path | None = None
        last_detail = ""
        for path in candidates:
            ok, status, body = try_enroll_one_face(
                setup_session,
                base_url,
                person_id,
                path,
                hdrs,
                args.timeout,
                skip_liveness=args.skip_liveness,
            )
            if ok:
                enrolled_from = path
                break
            err = body.get("error", {})
            last_detail = f"{path.name} status={status} code={err.get('code')}"
        if enrolled_from is None:
            delete_person(setup_session, base_url, person_id, hdrs, args.timeout)
            print(f"ERROR: could not enroll any face for setup ({last_detail})", file=sys.stderr)
            sys.exit(2)
        if enrolled_from != enroll_image:
            notes.append(f"setup: enroll fallback to {enrolled_from}")
            probe_image = enrolled_from
        notes.append(f"setup: enrolled person_id={person_id} from {enrolled_from.name}")
    else:
        notes.append("skipped setup (--skip-setup)")
        if args.endpoint == "verify" and not args.person_id:
            print("ERROR: verify mode with --skip-setup requires --person-id", file=sys.stderr)
            sys.exit(2)

    probe_bytes = probe_image.read_bytes()
    probe_name = probe_image.name
    wall_samples: list[float] = []
    errors: list[str] = []
    successful = 0
    failed = 0

    def task(_: int) -> tuple[float, bool, str]:
        if args.endpoint == "identify":
            return _one_identify(
                base_url, probe_bytes, probe_name, args.skip_liveness, hdrs, args.timeout
            )
        return _one_verify(
            base_url, person_id, probe_bytes, probe_name, args.skip_liveness, hdrs, args.timeout
        )

    print(
        f"Concurrent {args.endpoint}: workers={args.workers} "
        f"requests={args.requests} device={device} ..."
    )
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(task, i) for i in range(args.requests)]
        done = 0
        for fut in as_completed(futures):
            wall_ms, ok, err = fut.result()
            wall_samples.append(wall_ms)
            if ok:
                successful += 1
            else:
                failed += 1
                if len(errors) < 10:
                    errors.append(err)
            done += 1
            if done % max(args.requests // 5, 1) == 0:
                print(f"  progress {done}/{args.requests}")
    duration_sec = time.perf_counter() - t_start
    qps = args.requests / duration_sec if duration_sec > 0 else 0.0

    if not args.skip_cleanup and not args.skip_setup:
        delete_person(setup_session, base_url, person_id, hdrs, args.timeout)
        notes.append(f"cleaned up {person_id}")

    return ConcurrentReport(
        base_url=base_url,
        endpoint=args.endpoint,
        device=device,
        person_id=person_id if not args.skip_setup else args.person_id,
        workers=args.workers,
        total_requests=args.requests,
        successful=successful,
        failed=failed,
        qps=round(qps, 2),
        duration_sec=round(duration_sec, 3),
        latency_wall_ms=LatencyStats.from_samples(wall_samples),
        errors=errors,
        notes=notes,
    )


def print_report(report: ConcurrentReport) -> None:
    lat = report.latency_wall_ms
    print("\n" + "=" * 60)
    print("Face Access Control — Concurrent Benchmark Report")
    print("=" * 60)
    print(f"Base URL    : {report.base_url}")
    print(f"Endpoint    : {report.endpoint}")
    print(f"Device      : {report.device}")
    print(f"Workers     : {report.workers}")
    print(f"Requests    : {report.total_requests}")
    print(f"Successful  : {report.successful}")
    print(f"Failed      : {report.failed}")
    print(f"Duration    : {report.duration_sec} s")
    print(f"QPS         : {report.qps}")
    if lat.count:
        print(f"Latency p50 : {lat.p50_ms:.2f} ms")
        print(f"Latency p95 : {lat.p95_ms:.2f} ms")
        print(f"Latency p99 : {lat.p99_ms:.2f} ms")
        print(f"Latency max : {lat.max_ms:.2f} ms")
    if report.errors:
        print("\nSample errors:")
        for e in report.errors:
            print(f"  - {e}")
    if report.notes:
        print("\nNotes:")
        for n in report.notes:
            print(f"  - {n}")
    print("\n" + "=" * 60)


def report_to_json(report: ConcurrentReport) -> dict[str, Any]:
    data = asdict(report)
    data["latency_wall_ms"].pop("samples_ms", None)
    return data


def main() -> None:
    args = parse_args()
    report = run_benchmark(args)
    print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report_to_json(report), indent=2, ensure_ascii=False) + "\n")
        print(f"\nJSON report written to {args.output}")
    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()
