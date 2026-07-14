#!/usr/bin/env python3
"""
End-to-end latency benchmark for the face access-control API (deploy/).

Measures:
  - Enrollment wall-clock latency (single image, same-image batch, or multi-image batch/sequential)
  - 1:N identify: client wall + server latency_ms (inference/search/total)
  - 1:1 verify: client wall + server latency_ms (inference/verify/total)

Usage:
  pip install -r test/requirements.txt

  # Single image (default)
  python test/integration/latency_benchmark.py \\
    --enroll-image test/enroll_images/wjr/photo.jpg \\
    --probe-image test/enroll_images/wjr/probe.jpg \\
    --mode both \\
    --output test/results/latency_report.json

  # Multiple distinct images — one HTTP request (batch)
  python test/integration/latency_benchmark.py \\
    --enroll-dir test/enroll_images/wjr \\
    --enroll-strategy batch \\
    --mode enroll \\
    --output test/results/enroll_batch_report.json

  # Multiple distinct images — one request per image (sequential)
  python test/integration/latency_benchmark.py \\
    --enroll-images img1.jpg img2.jpg img3.jpg \\
    --enroll-strategy sequential \\
    --mode enroll
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
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests

from common import DEFAULT_RESULTS_DIR, headers, list_face_images, list_images, pick_first_face_image


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
class EndpointLatency:
    wall: LatencyStats
    server_inference: LatencyStats
    server_secondary: LatencyStats  # search (identify) or verify (verify)
    server_total: LatencyStats
    success_count: int
    runs: int


@dataclass
class EnrollmentLatency:
    strategy: str  # batch | sequential
    image_count: int
    enrolled_count: int
    wall_ms_total: float
    wall_ms_per_request: list[float]
    wall_ms_per_face: list[float]
    image_paths: list[str]


@dataclass
class BenchmarkReport:
    base_url: str
    mode: str
    person_id: str
    device: str
    liveness_enabled: bool
    liveness_models_loaded: bool
    gallery_size_before: int
    gallery_size_after: int
    warmup_runs: int
    timed_runs: int
    enroll: EnrollmentLatency | None
    enroll_wall_ms: float
    enroll_per_face_server_ms: list[float]
    identify: EndpointLatency | None
    verify: EndpointLatency | None
    notes: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Face API latency benchmark (enroll + identify + verify)")
    p.add_argument("--base-url", default="http://localhost:8123")
    p.add_argument("--api-key", default=None)
    p.add_argument("--enroll-image", type=Path, help="Face image for enrollment")
    p.add_argument("--probe-image", type=Path, help="Probe image for identify/verify")
    p.add_argument("--image", type=Path, help="Use one image for enroll and probe")
    p.add_argument(
        "--mode",
        choices=["identify", "verify", "both", "enroll"],
        default="both",
        help="Which endpoints to benchmark after enrollment (enroll = enrollment latency only)",
    )
    p.add_argument(
        "--enroll-images",
        type=Path,
        nargs="+",
        help="Distinct face images for enrollment (batch or sequential)",
    )
    p.add_argument(
        "--enroll-dir",
        type=Path,
        help="Directory of face images for enrollment (uses all images in dir)",
    )
    p.add_argument(
        "--enroll-strategy",
        choices=["batch", "sequential"],
        default="batch",
        help="batch = one POST with all images; sequential = one POST per image",
    )
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--person-id", default=None)
    p.add_argument("--skip-cleanup", action="store_true")
    p.add_argument("--skip-liveness", action="store_true")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "latency_report.json")
    p.add_argument(
        "--enroll-count",
        type=int,
        default=1,
        help="With a single --enroll-image, duplicate that file N times in one request",
    )
    return p.parse_args()


def resolve_enroll_images(args: argparse.Namespace) -> list[Path]:
    """Resolve enrollment image list from CLI flags."""
    if args.enroll_images:
        paths = [p.resolve() for p in args.enroll_images]
    elif args.enroll_dir:
        paths = list_images(args.enroll_dir.resolve())
        if not paths:
            print(f"ERROR: no images found under {args.enroll_dir}", file=sys.stderr)
            sys.exit(2)
    else:
        single = args.enroll_image or args.image
        if not single:
            return []
        if args.enroll_count < 1:
            print("ERROR: --enroll-count must be >= 1", file=sys.stderr)
            sys.exit(2)
        paths = [single.resolve()] * args.enroll_count
        return paths

    missing = [p for p in paths if not p.is_file()]
    if missing:
        for p in missing:
            print(f"ERROR: enroll image not found: {p}", file=sys.stderr)
        sys.exit(2)
    return paths


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
    if r.status_code == 400:
        body = r.json()
        if "already exists" in body.get("error", {}).get("message", ""):
            return
    r.raise_for_status()


def enroll_faces_once(
    session: requests.Session,
    base_url: str,
    person_id: str,
    image_paths: list[Path],
    skip_liveness: bool,
    hdrs: dict[str, str],
    timeout: float,
) -> tuple[float, int]:
    files = [
        ("images", (path.name, path.read_bytes(), "image/jpeg"))
        for path in image_paths
    ]
    params = {"skip_liveness": "true"} if skip_liveness else {}
    t0 = time.perf_counter()
    r = session.post(
        f"{base_url}/v1/persons/{person_id}/faces",
        files=files,
        params=params,
        headers=hdrs,
        timeout=timeout,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    if r.status_code >= 400:
        try:
            err = r.json().get("error", {})
            msg = err.get("message") or r.text
            code = err.get("code")
        except Exception:
            msg, code = r.text, None
        raise RuntimeError(f"enroll failed status={r.status_code} code={code} msg={msg}")
    body = r.json()
    enrolled_count = len(body.get("enrolled", []))
    return wall_ms, enrolled_count


def benchmark_enrollment(
    session: requests.Session,
    base_url: str,
    person_id: str,
    image_paths: list[Path],
    strategy: str,
    skip_liveness: bool,
    hdrs: dict[str, str],
    timeout: float,
) -> EnrollmentLatency:
    per_request: list[float] = []
    per_face: list[float] = []
    enrolled_total = 0

    if strategy == "batch":
        wall_ms, enrolled_count = enroll_faces_once(
            session, base_url, person_id, image_paths, skip_liveness, hdrs, timeout
        )
        per_request.append(wall_ms)
        enrolled_total = enrolled_count
        if enrolled_count:
            per_face.extend([wall_ms / enrolled_count] * enrolled_count)
    else:
        for path in image_paths:
            wall_ms, enrolled_count = enroll_faces_once(
                session, base_url, person_id, [path], skip_liveness, hdrs, timeout
            )
            per_request.append(wall_ms)
            enrolled_total += enrolled_count
            if enrolled_count:
                per_face.extend([wall_ms / enrolled_count] * enrolled_count)

    return EnrollmentLatency(
        strategy=strategy,
        image_count=len(image_paths),
        enrolled_count=enrolled_total,
        wall_ms_total=sum(per_request),
        wall_ms_per_request=per_request,
        wall_ms_per_face=per_face,
        image_paths=[str(p) for p in image_paths],
    )


def identify_once(
    session: requests.Session,
    base_url: str,
    image_path: Path,
    skip_liveness: bool,
    hdrs: dict[str, str],
    timeout: float,
) -> tuple[float, dict[str, Any]]:
    params = {"skip_liveness": "true"} if skip_liveness else {}
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


def verify_once(
    session: requests.Session,
    base_url: str,
    person_id: str,
    image_path: Path,
    skip_liveness: bool,
    hdrs: dict[str, str],
    timeout: float,
) -> tuple[float, dict[str, Any]]:
    params = {"person_id": person_id}
    if skip_liveness:
        params["skip_liveness"] = "true"
    files = {"image": (image_path.name, image_path.read_bytes(), "image/jpeg")}
    t0 = time.perf_counter()
    r = session.post(
        f"{base_url}/v1/verify",
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


def benchmark_endpoint(
    session: requests.Session,
    base_url: str,
    person_id: str,
    probe_image: Path,
    endpoint: Literal["identify", "verify"],
    skip_liveness: bool,
    hdrs: dict[str, str],
    timeout: float,
    warmup: int,
    runs: int,
) -> EndpointLatency:
    secondary_key = "search" if endpoint == "identify" else "verify"
    success_key = "matched" if endpoint == "identify" else "verified"

    for _ in range(warmup):
        if endpoint == "identify":
            identify_once(session, base_url, probe_image, skip_liveness, hdrs, timeout)
        else:
            verify_once(session, base_url, person_id, probe_image, skip_liveness, hdrs, timeout)

    wall_samples: list[float] = []
    infer_samples: list[float] = []
    secondary_samples: list[float] = []
    total_samples: list[float] = []
    success = 0

    for i in range(runs):
        if endpoint == "identify":
            wall_ms, body = identify_once(
                session, base_url, probe_image, skip_liveness, hdrs, timeout
            )
        else:
            wall_ms, body = verify_once(
                session, base_url, person_id, probe_image, skip_liveness, hdrs, timeout
            )
        wall_samples.append(wall_ms)
        lat = body.get("latency_ms") or {}
        infer_samples.append(float(lat.get("inference", 0.0)))
        secondary_samples.append(float(lat.get(secondary_key, 0.0)))
        total_samples.append(float(lat.get("total", 0.0)))
        if body.get(success_key):
            success += 1
        if (i + 1) % max(runs // 5, 1) == 0:
            print(f"  {endpoint} progress {i + 1}/{runs}")

    return EndpointLatency(
        wall=LatencyStats.from_samples(wall_samples),
        server_inference=LatencyStats.from_samples(infer_samples),
        server_secondary=LatencyStats.from_samples(secondary_samples),
        server_total=LatencyStats.from_samples(total_samples),
        success_count=success,
        runs=runs,
    )


def run_benchmark(args: argparse.Namespace) -> BenchmarkReport:
    enroll_paths = resolve_enroll_images(args)
    probe_image = args.probe_image or args.image
    needs_probe = args.mode in ("identify", "verify", "both")

    if not enroll_paths:
        try:
            default_img = pick_first_face_image()
            enroll_paths = [default_img.resolve()] * max(args.enroll_count, 1)
        except FileNotFoundError:
            print(
                "ERROR: provide --enroll-image, --enroll-images, --enroll-dir, or --image",
                file=sys.stderr,
            )
            sys.exit(2)

    if needs_probe:
        if not probe_image:
            try:
                probe_image = pick_first_face_image()
            except FileNotFoundError:
                print("ERROR: provide --probe-image or --image for identify/verify", file=sys.stderr)
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
        notes.append(f"ready=false: {ready.get('checks')}")

    health = check_health(session, base_url, args.timeout)
    device = health.get("device", "unknown")
    gallery_before = int(health.get("gallery_size", 0))
    print(f"Device: {device}")

    create_person(session, base_url, person_id, hdrs, args.timeout)
    print(
        f"Enrolling {len(enroll_paths)} image(s) "
        f"(strategy={args.enroll_strategy}) for person_id={person_id} ..."
    )
    try:
        enroll_result = benchmark_enrollment(
            session,
            base_url,
            person_id,
            enroll_paths,
            args.enroll_strategy,
            args.skip_liveness,
            hdrs,
            args.timeout,
        )
    except RuntimeError as exc:
        # Single-image path: try other preferred photos when liveness/quality fails.
        if len(enroll_paths) != 1:
            raise
        notes.append(f"primary enroll failed: {exc}")
        enroll_result = None
        for alt in list_face_images():
            if alt.resolve() == enroll_paths[0].resolve():
                continue
            print(f"  retry enroll with {alt} ...")
            try:
                enroll_result = benchmark_enrollment(
                    session,
                    base_url,
                    person_id,
                    [alt],
                    args.enroll_strategy,
                    args.skip_liveness,
                    hdrs,
                    args.timeout,
                )
            except RuntimeError as alt_exc:
                notes.append(f"retry {alt.name} failed: {alt_exc}")
                continue
            if enroll_result.enrolled_count > 0:
                enroll_paths = [alt]
                if needs_probe and (args.probe_image is None):
                    # Align probe with enrolled face when probe was not explicitly set.
                    probe_image = alt
                notes.append(f"enroll fallback to {alt}")
                break
        if enroll_result is None or enroll_result.enrolled_count == 0:
            delete_person(session, base_url, person_id, hdrs, args.timeout)
            raise RuntimeError("could not enroll any face for latency benchmark") from exc

    if enroll_result.enrolled_count < enroll_result.image_count:
        notes.append(
            f"enrolled {enroll_result.enrolled_count}/{enroll_result.image_count} "
            "(some images may fail quality/liveness checks)"
        )
    if enroll_result.enrolled_count == 0:
        delete_person(session, base_url, person_id, hdrs, args.timeout)
        raise RuntimeError("enrollment produced 0 faces; cannot benchmark identify/verify")

    identify_result: EndpointLatency | None = None
    verify_result: EndpointLatency | None = None

    if args.mode in ("identify", "both"):
        print(f"Benchmark identify: warmup={args.warmup} runs={args.runs} ...")
        identify_result = benchmark_endpoint(
            session,
            base_url,
            person_id,
            probe_image,
            "identify",
            args.skip_liveness,
            hdrs,
            args.timeout,
            args.warmup,
            args.runs,
        )

    if args.mode in ("verify", "both"):
        print(f"Benchmark verify: warmup={args.warmup} runs={args.runs} ...")
        verify_result = benchmark_endpoint(
            session,
            base_url,
            person_id,
            probe_image,
            "verify",
            args.skip_liveness,
            hdrs,
            args.timeout,
            args.warmup,
            args.runs,
        )

    health_after = check_health(session, base_url, args.timeout)

    if not args.skip_cleanup:
        delete_person(session, base_url, person_id, hdrs, args.timeout)
        notes.append(f"cleaned up test person {person_id}")
    else:
        notes.append(f"skipped cleanup; person_id={person_id}")

    return BenchmarkReport(
        base_url=base_url,
        mode=args.mode,
        person_id=person_id,
        device=device,
        liveness_enabled=bool(health.get("liveness_enabled")),
        liveness_models_loaded=bool(health.get("liveness_models_loaded")),
        gallery_size_before=gallery_before,
        gallery_size_after=int(health_after.get("gallery_size", 0)),
        warmup_runs=args.warmup,
        timed_runs=args.runs,
        enroll=enroll_result,
        enroll_wall_ms=enroll_result.wall_ms_total,
        enroll_per_face_server_ms=enroll_result.wall_ms_per_face,
        identify=identify_result,
        verify=verify_result,
        notes=notes,
    )


def fmt_stats(title: str, stats: LatencyStats) -> None:
    print(f"\n{title}")
    if stats.count == 0:
        print("  (no samples)")
        return
    print(f"  count : {stats.count}")
    print(f"  min   : {stats.min_ms:.2f} ms")
    print(f"  p50   : {stats.p50_ms:.2f} ms")
    print(f"  mean  : {stats.mean_ms:.2f} ms")
    print(f"  p95   : {stats.p95_ms:.2f} ms")
    print(f"  p99   : {stats.p99_ms:.2f} ms")
    print(f"  max   : {stats.max_ms:.2f} ms")


def print_endpoint(name: str, ep: EndpointLatency, secondary_label: str, success_label: str) -> None:
    print(f"\n--- {name} ---")
    print(f"  timed runs  : {ep.runs}")
    print(f"  {success_label}: {ep.success_count}/{ep.runs}")
    fmt_stats("  Client end-to-end (HTTP wall clock)", ep.wall)
    fmt_stats("  Server inference (latency_ms.inference)", ep.server_inference)
    fmt_stats(f"  Server {secondary_label}", ep.server_secondary)
    fmt_stats("  Server total (latency_ms.total)", ep.server_total)


def print_report(report: BenchmarkReport) -> None:
    print("\n" + "=" * 60)
    print("Face Access Control — Latency Benchmark Report")
    print("=" * 60)
    print(f"Base URL     : {report.base_url}")
    print(f"Mode         : {report.mode}")
    print(f"Device       : {report.device}")
    print(f"Test person  : {report.person_id}")
    print(f"Gallery size : {report.gallery_size_before} -> {report.gallery_size_after}")

    print("\n--- Enrollment ---")
    if report.enroll:
        e = report.enroll
        print(f"  strategy          : {e.strategy}")
        print(f"  images submitted  : {e.image_count}")
        print(f"  faces enrolled    : {e.enrolled_count}")
        print(f"  wall time (total) : {e.wall_ms_total:.2f} ms")
        if e.wall_ms_per_face:
            print(f"  approx per-face   : {statistics.mean(e.wall_ms_per_face):.2f} ms")
        if e.strategy == "sequential" and len(e.wall_ms_per_request) > 1:
            req_stats = LatencyStats.from_samples(e.wall_ms_per_request)
            print(f"  per-request p50   : {req_stats.p50_ms:.2f} ms")
            print(f"  per-request mean  : {req_stats.mean_ms:.2f} ms")
    else:
        print(f"  wall time (total) : {report.enroll_wall_ms:.2f} ms")
        if report.enroll_per_face_server_ms:
            print(f"  approx per-face   : {statistics.mean(report.enroll_per_face_server_ms):.2f} ms")

    if report.identify:
        print_endpoint("Identification (1:N)", report.identify, "search", "matched")
    if report.verify:
        print_endpoint("Verification (1:1)", report.verify, "verify", "verified")

    if report.notes:
        print("\nNotes:")
        for n in report.notes:
            print(f"  - {n}")
    print("\n" + "=" * 60)


def endpoint_to_json(ep: EndpointLatency | None) -> dict[str, Any] | None:
    if ep is None:
        return None
    data = asdict(ep)
    for key in ("wall", "server_inference", "server_secondary", "server_total"):
        data[key].pop("samples_ms", None)
    return data


def enrollment_to_json(enroll: EnrollmentLatency | None) -> dict[str, Any] | None:
    if enroll is None:
        return None
    return asdict(enroll)


def report_to_json(report: BenchmarkReport) -> dict[str, Any]:
    return {
        "base_url": report.base_url,
        "mode": report.mode,
        "person_id": report.person_id,
        "device": report.device,
        "liveness_enabled": report.liveness_enabled,
        "liveness_models_loaded": report.liveness_models_loaded,
        "gallery_size_before": report.gallery_size_before,
        "gallery_size_after": report.gallery_size_after,
        "warmup_runs": report.warmup_runs,
        "timed_runs": report.timed_runs,
        "enroll": enrollment_to_json(report.enroll),
        "enroll_wall_ms": report.enroll_wall_ms,
        "enroll_per_face_server_ms": report.enroll_per_face_server_ms,
        "identify": endpoint_to_json(report.identify),
        "verify": endpoint_to_json(report.verify),
        "notes": report.notes,
    }


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
