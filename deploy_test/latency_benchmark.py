#!/usr/bin/env python3
"""
Standard end-to-end latency benchmark for the face access-control API (deploy/).

Measures:
  - Service health / inference device (cuda:0 vs cpu)
  - Enrollment (register sample) wall-clock latency
  - Identification (1:N probe) wall-clock + server-reported inference/search latency

Usage:
  pip install -r deploy_test/requirements.txt

  python deploy_test/latency_benchmark.py \\
    --base-url http://localhost:8000 \\
    --enroll-image /path/to/face_enroll.jpg \\
    --probe-image /path/to/face_probe.jpg

  # Same image for enroll and probe (latency only, match not guaranteed):
  python deploy_test/latency_benchmark.py --image ./sample.jpg
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests


@dataclass
class LatencyStats:
    count: int
    min_ms: float
    max_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    samples_ms: list[float] = field(repr=False, default_factory=list)

    @classmethod
    def from_samples(cls, samples: list[float]) -> LatencyStats:
        if not samples:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [])
        ordered = sorted(samples)
        n = len(ordered)

        def pct(p: float) -> float:
            idx = min(int(round(p / 100.0 * (n - 1))), n - 1)
            return ordered[idx]

        return cls(
            count=n,
            min_ms=ordered[0],
            max_ms=ordered[-1],
            mean_ms=statistics.mean(ordered),
            p50_ms=pct(50),
            p95_ms=pct(95),
            p99_ms=pct(99),
            samples_ms=ordered,
        )


@dataclass
class BenchmarkReport:
    base_url: str
    person_id: str
    device: str
    liveness_enabled: bool
    liveness_models_loaded: bool
    gallery_size_before: int
    gallery_size_after: int
    warmup_runs: int
    identify_runs: int
    enroll_wall_ms: float
    enroll_per_face_server_ms: list[float]
    identify_wall: LatencyStats
    identify_server_inference: LatencyStats
    identify_server_search: LatencyStats
    identify_server_total: LatencyStats
    identify_matched_count: int
    notes: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Face API latency benchmark (enroll + 1:N identify)")
    p.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    p.add_argument("--api-key", default=None, help="X-API-Key if service requires auth")
    p.add_argument("--enroll-image", type=Path, help="Face image for enrollment")
    p.add_argument("--probe-image", type=Path, help="Face image for identification probe")
    p.add_argument("--image", type=Path, help="Use one image for both enroll and probe")
    p.add_argument("--warmup", type=int, default=2, help="Warmup identify requests before timing")
    p.add_argument("--runs", type=int, default=20, help="Timed identify requests")
    p.add_argument("--person-id", default=None, help="Test person_id (auto-generated if unset)")
    p.add_argument("--skip-cleanup", action="store_true", help="Keep test person in gallery")
    p.add_argument("--skip-liveness", action="store_true", help="Pass skip_liveness=true (debug only)")
    p.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds")
    p.add_argument("--output", type=Path, default=None, help="Write JSON report to file")
    p.add_argument("--enroll-count", type=int, default=1, help="Number of enroll images (same file repeated)")
    return p.parse_args()


def headers(api_key: str | None) -> dict[str, str]:
    h: dict[str, str] = {}
    if api_key:
        h["X-API-Key"] = api_key
    return h


def check_health(session: requests.Session, base_url: str, timeout: float) -> dict[str, Any]:
    r = session.get(f"{base_url}/v1/health", timeout=timeout)
    r.raise_for_status()
    return r.json()


def check_ready(session: requests.Session, base_url: str, timeout: float) -> dict[str, Any]:
    r = session.get(f"{base_url}/v1/ready", timeout=timeout)
    r.raise_for_status()
    return r.json()


def create_person(
    session: requests.Session,
    base_url: str,
    person_id: str,
    hdrs: dict[str, str],
    timeout: float,
) -> None:
    r = session.post(
        f"{base_url}/v1/persons",
        json={"person_id": person_id, "display_name": f"bench_{person_id}"},
        headers=hdrs,
        timeout=timeout,
    )
    if r.status_code == 201:
        return
    # idempotent re-run: person may already exist
    if r.status_code == 400:
        body = r.json()
        err = body.get("error", {})
        if "already exists" in err.get("message", ""):
            return
    r.raise_for_status()


def enroll_faces(
    session: requests.Session,
    base_url: str,
    person_id: str,
    image_path: Path,
    enroll_count: int,
    skip_liveness: bool,
    hdrs: dict[str, str],
    timeout: float,
) -> tuple[float, list[float]]:
    files = [("images", (image_path.name, image_path.read_bytes(), "image/jpeg"))] * enroll_count
    params = {}
    if skip_liveness:
        params["skip_liveness"] = "true"

    t0 = time.perf_counter()
    r = session.post(
        f"{base_url}/v1/persons/{person_id}/faces",
        files=files,
        params=params,
        headers=hdrs,
        timeout=timeout,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    body = r.json()
    # Server does not yet expose per-enroll latency; report wall-clock only
    enrolled = body.get("enrolled", [])
    per_face = [wall_ms / max(len(enrolled), 1)] * len(enrolled)
    return wall_ms, per_face


def identify_once(
    session: requests.Session,
    base_url: str,
    image_path: Path,
    skip_liveness: bool,
    hdrs: dict[str, str],
    timeout: float,
) -> tuple[float, dict[str, Any]]:
    params = {}
    if skip_liveness:
        params["skip_liveness"] = "true"
    files = {"image": (image_path.name, image_path.read_bytes(), "image/jpeg")}

    t0 = time.perf_counter()
    r = session.post(
        f"{base_url}/v1/identify",
        files=files,
        params=params,
        headers=hdrs,
        timeout=timeout,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    return wall_ms, r.json()


def delete_person(
    session: requests.Session,
    base_url: str,
    person_id: str,
    hdrs: dict[str, str],
    timeout: float,
) -> None:
    r = session.delete(f"{base_url}/v1/persons/{person_id}", headers=hdrs, timeout=timeout)
    if r.status_code in (204, 404):
        return
    r.raise_for_status()


def run_benchmark(args: argparse.Namespace) -> BenchmarkReport:
    enroll_image = args.enroll_image or args.image
    probe_image = args.probe_image or args.image
    if not enroll_image or not probe_image:
        print("ERROR: provide --enroll-image and --probe-image, or --image", file=sys.stderr)
        sys.exit(2)
    if not enroll_image.is_file():
        print(f"ERROR: enroll image not found: {enroll_image}", file=sys.stderr)
        sys.exit(2)
    if not probe_image.is_file():
        print(f"ERROR: probe image not found: {probe_image}", file=sys.stderr)
        sys.exit(2)

    base_url = args.base_url.rstrip("/")
    person_id = args.person_id or f"bench_{uuid.uuid4().hex[:12]}"
    hdrs = headers(args.api_key)
    notes: list[str] = []

    session = requests.Session()

    print(f"Checking service at {base_url} ...")
    ready = check_ready(session, base_url, args.timeout)
    if not ready.get("ready"):
        print(f"WARNING: service not ready: {ready}", file=sys.stderr)
        notes.append(f"ready=false: {ready.get('checks')}")

    health = check_health(session, base_url, args.timeout)
    device = health.get("device", "unknown")
    gallery_before = int(health.get("gallery_size", 0))

    print(f"Device reported by API: {device}")
    print(f"Liveness: enabled={health.get('liveness_enabled')} models_loaded={health.get('liveness_models_loaded')}")

    if health.get("liveness_enabled") and not health.get("liveness_models_loaded"):
        notes.append("liveness enabled but models not loaded; enroll/identify may return 503")

    create_person(session, base_url, person_id, hdrs, args.timeout)
    print(f"Enrolling {args.enroll_count} face(s) for person_id={person_id} ...")
    enroll_wall_ms, enroll_per_face = enroll_faces(
        session,
        base_url,
        person_id,
        enroll_image,
        args.enroll_count,
        args.skip_liveness,
        hdrs,
        args.timeout,
    )

    if args.warmup > 0:
        print(f"Warmup: {args.warmup} identify request(s) ...")
        for _ in range(args.warmup):
            identify_once(session, base_url, probe_image, args.skip_liveness, hdrs, args.timeout)

    wall_samples: list[float] = []
    infer_samples: list[float] = []
    search_samples: list[float] = []
    total_server_samples: list[float] = []
    matched = 0

    print(f"Timed identify: {args.runs} run(s) ...")
    for i in range(args.runs):
        wall_ms, body = identify_once(
            session, base_url, probe_image, args.skip_liveness, hdrs, args.timeout
        )
        wall_samples.append(wall_ms)
        lat = body.get("latency_ms") or {}
        infer_samples.append(float(lat.get("inference", 0.0)))
        search_samples.append(float(lat.get("search", 0.0)))
        total_server_samples.append(float(lat.get("total", 0.0)))
        if body.get("matched"):
            matched += 1
        if (i + 1) % max(args.runs // 5, 1) == 0:
            print(f"  progress {i + 1}/{args.runs}")

    health_after = check_health(session, base_url, args.timeout)

    if not args.skip_cleanup:
        delete_person(session, base_url, person_id, hdrs, args.timeout)
        notes.append(f"cleaned up test person {person_id}")
    else:
        notes.append(f"skipped cleanup; person_id={person_id}")

    return BenchmarkReport(
        base_url=base_url,
        person_id=person_id,
        device=device,
        liveness_enabled=bool(health.get("liveness_enabled")),
        liveness_models_loaded=bool(health.get("liveness_models_loaded")),
        gallery_size_before=gallery_before,
        gallery_size_after=int(health_after.get("gallery_size", 0)),
        warmup_runs=args.warmup,
        identify_runs=args.runs,
        enroll_wall_ms=enroll_wall_ms,
        enroll_per_face_server_ms=enroll_per_face,
        identify_wall=LatencyStats.from_samples(wall_samples),
        identify_server_inference=LatencyStats.from_samples(infer_samples),
        identify_server_search=LatencyStats.from_samples(search_samples),
        identify_server_total=LatencyStats.from_samples(total_server_samples),
        identify_matched_count=matched,
        notes=notes,
    )


def print_report(report: BenchmarkReport) -> None:
    def fmt_stats(title: str, stats: LatencyStats, unit: str = "ms") -> None:
        print(f"\n{title}")
        if stats.count == 0:
            print("  (no samples)")
            return
        print(f"  count : {stats.count}")
        print(f"  min   : {stats.min_ms:.2f} {unit}")
        print(f"  p50   : {stats.p50_ms:.2f} {unit}")
        print(f"  mean  : {stats.mean_ms:.2f} {unit}")
        print(f"  p95   : {stats.p95_ms:.2f} {unit}")
        print(f"  p99   : {stats.p99_ms:.2f} {unit}")
        print(f"  max   : {stats.max_ms:.2f} {unit}")

    print("\n" + "=" * 60)
    print("Face Access Control — Latency Benchmark Report")
    print("=" * 60)
    print(f"Base URL     : {report.base_url}")
    print(f"Device       : {report.device}  (cuda:0 = GPU, cpu = CPU fallback)")
    print(f"Liveness     : enabled={report.liveness_enabled} models_loaded={report.liveness_models_loaded}")
    print(f"Gallery size : {report.gallery_size_before} -> {report.gallery_size_after}")
    print(f"Test person  : {report.person_id}")

    print("\n--- Enrollment (register sample) ---")
    print(f"  end-to-end wall time (total) : {report.enroll_wall_ms:.2f} ms")
    if report.enroll_per_face_server_ms:
        avg = statistics.mean(report.enroll_per_face_server_ms)
        print(f"  approx per-face (wall/count) : {avg:.2f} ms")
    print(f"  inference device             : {report.device}")

    print("\n--- Identification (1:N probe) ---")
    print(f"  warmup runs : {report.warmup_runs}")
    print(f"  timed runs  : {report.identify_runs}")
    print(f"  matched     : {report.identify_matched_count}/{report.identify_runs}")
    print(f"  inference device : {report.device}")

    fmt_stats("  Client end-to-end (HTTP wall clock)", report.identify_wall)
    fmt_stats("  Server inference only (latency_ms.inference)", report.identify_server_inference)
    fmt_stats("  Server FAISS search (latency_ms.search)", report.identify_server_search)
    fmt_stats("  Server total (latency_ms.total)", report.identify_server_total)

    if report.notes:
        print("\nNotes:")
        for n in report.notes:
            print(f"  - {n}")

    print("\n" + "=" * 60)


def report_to_json(report: BenchmarkReport) -> dict[str, Any]:
    data = asdict(report)
    for key in (
        "identify_wall",
        "identify_server_inference",
        "identify_server_search",
        "identify_server_total",
    ):
        stats = data[key]
        stats.pop("samples_ms", None)
    return data


def main() -> None:
    args = parse_args()
    report = run_benchmark(args)
    print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report_to_json(report), indent=2, ensure_ascii=False) + "\n")
        print(f"\nJSON report written to {args.output}")


if __name__ == "__main__":
    main()
