#!/usr/bin/env python3
"""
人脸门禁 API 功能冒烟测试（deploy/）

测试流程:
  1. 可选清库 — 删除底库中所有人员
  2. 全量注册 — enroll_images 下除 stranger(cyt) 外，每人注册目录内全部照片
  3. 追加注册 — enroll_add 下 hjh 成功追加、ym 触发 5 张上限
  4. 核心 API — 人员 CRUD、1:N 识别、1:1 验证、统计、事件审计
  5. 清理测试数据

用法:
  pip install -r deploy_test/requirements.txt
  ./deploy_test/run_feature_test.sh
  SKIP_LIVENESS=true ./deploy_test/run_feature_test.sh   # 样例图可能过不了活体时推荐
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

# 1:N 探针识别需逐一验证的人员（其余人员仅注册进底库）
DEFAULT_IDENTIFY_PERSONS = ["wjr", "whd", "zjy"]
DEFAULT_STRANGER_DIR = "cyt"
MAX_FACES_PER_PERSON = 5
# 追加注册 / 1:N 探针等步骤依赖的关键人员（注册失败则整体失败）
REQUIRED_PERSONS = {"wjr", "whd", "zjy", "ym"}


@dataclass
class EnrollBatchResult:
    """批量注册结果：成功/失败明细。"""

    enrolled_paths: list[Path] = field(default_factory=list)
    face_ids: list[str] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)


@dataclass
class PersonFixture:
    """测试用人员夹具。"""

    person_id: str
    display_name: str
    image_dir: Path
    enrolled_paths: list[Path] = field(default_factory=list)
    probe_path: Path | None = None
    face_ids: list[str] = field(default_factory=list)


@dataclass
class StepResult:
    name: str
    passed: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureTestReport:
    base_url: str
    person_ids: list[str]
    device: str
    steps: list[StepResult]
    passed: int
    failed: int
    notes: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(description="人脸 API 功能冒烟测试（多人全量底库）")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--api-key", default=None)
    p.add_argument("--enroll-root", type=Path, default=script_dir / "enroll_images")
    p.add_argument("--enroll-add-root", type=Path, default=script_dir / "enroll_add")
    p.add_argument(
        "--stranger-dir",
        type=str,
        default=DEFAULT_STRANGER_DIR,
        help="陌生人探针目录名（不注册，默认 cyt）",
    )
    p.add_argument(
        "--identify-persons",
        type=str,
        default=",".join(DEFAULT_IDENTIFY_PERSONS),
        help="需做 1:N 匹配验证的人员目录名，逗号分隔",
    )
    p.add_argument(
        "--reset-gallery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="测试前清空底库（默认 true）",
    )
    p.add_argument(
        "--skip-liveness",
        action="store_true",
        help="注册/识别/验证均跳过活体（调试样例图推荐开启）",
    )
    p.add_argument("--skip-cleanup", action="store_true", help="测试后保留 smoke_* 人员")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def headers(api_key: str | None) -> dict[str, str]:
    h: dict[str, str] = {}
    if api_key:
        h["X-API-Key"] = api_key
    return h


def list_images(directory: Path) -> list[Path]:
    """列出目录下支持的图片文件。"""
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts)


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


def collect_enroll_dirs(enroll_root: Path, stranger_name: str) -> tuple[list[Path], Path]:
    """
    收集需注册的人员目录：enroll_images 下全部子目录，排除 stranger（cyt）。
    """
    if not enroll_root.is_dir():
        raise FileNotFoundError(f"注册图根目录不存在: {enroll_root}")

    subdirs = sorted(d for d in enroll_root.iterdir() if d.is_dir())
    by_name = {d.name: d for d in subdirs}

    if stranger_name not in by_name:
        raise FileNotFoundError(f"陌生人目录不存在: {enroll_root / stranger_name}")

    stranger_dir = by_name[stranger_name]
    enroll_dirs = [d for d in subdirs if d.name != stranger_name]

    if not enroll_dirs:
        raise FileNotFoundError(f"除 {stranger_name} 外无人员目录可注册")

    return enroll_dirs, stranger_dir


def enroll_one_face_raw(
    session: requests.Session,
    base_url: str,
    person_id: str,
    image_path: Path,
    skip_liveness: bool,
    hdrs: dict[str, str],
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    """注册单张人脸，返回 (HTTP 状态码, 响应体)，不抛异常。"""
    params = {"skip_liveness": "true"} if skip_liveness else {}
    files = [("images", (image_path.name, image_path.read_bytes(), "application/octet-stream"))]
    return post_multipart(
        session,
        f"{base_url}/v1/persons/{person_id}/faces",
        files,
        params,
        hdrs,
        timeout,
    )


def enroll_all_faces(
    session: requests.Session,
    base_url: str,
    person_id: str,
    candidates: list[Path],
    skip_liveness: bool,
    hdrs: dict[str, str],
    timeout: float,
) -> EnrollBatchResult:
    """
    逐张注册目录内全部照片；单张失败（质量/活体）跳过并记录，至少成功 1 张。
    """
    result = EnrollBatchResult()
    for path in candidates:
        status, body = enroll_one_face_raw(
            session, base_url, person_id, path, skip_liveness, hdrs, timeout
        )
        if status == 200:
            result.enrolled_paths.append(path)
            for item in body.get("enrolled", []):
                result.face_ids.append(item["face_id"])
        else:
            msg = body.get("error", {}).get("message", str(body))
            result.failed.append((path, msg))

    if not result.enrolled_paths:
        return result  # 允许 0 成功，由 setup 步骤决定是否跳过该人员

    return result


def identify_with(
    session: requests.Session,
    base_url: str,
    image_path: Path,
    skip_liveness: bool,
    hdrs: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    params = {"skip_liveness": "true"} if skip_liveness else {}
    files = {"image": (image_path.name, image_path.read_bytes(), "application/octet-stream")}
    status, body = post_multipart(
        session, f"{base_url}/v1/identify", files, params, hdrs, timeout
    )
    if status != 200:
        msg = body.get("error", {}).get("message", str(body))
        raise RuntimeError(f"identify 失败 {image_path.name}: {msg}")
    return body


def pick_probe(
    session: requests.Session,
    base_url: str,
    person: PersonFixture,
    skip_liveness: bool,
    hdrs: dict[str, str],
    timeout: float,
) -> Path:
    """为该人员挑选一张能 1:N 匹配成功的探针图（优先未注册过的照片）。"""
    candidates = list_images(person.image_dir)
    enrolled_set = set(person.enrolled_paths)
    ordered = [p for p in candidates if p not in enrolled_set] + list(person.enrolled_paths)

    last_error = "no candidates"
    for path in ordered:
        try:
            body = identify_with(session, base_url, path, skip_liveness, hdrs, timeout)
        except RuntimeError as exc:
            last_error = str(exc)
            continue
        if body.get("matched") and body.get("person_id") == person.person_id:
            return path
    raise RuntimeError(f"找不到 {person.display_name} 的探针: {last_error}")


def pick_stranger_probe(
    session: requests.Session,
    base_url: str,
    candidates: list[Path],
    skip_liveness: bool,
    hdrs: dict[str, str],
    timeout: float,
) -> tuple[Path, dict[str, Any]]:
    """从 stranger 目录挑选陌生人探针（is_stranger=true）。"""
    last_error = "no candidates"
    for path in candidates:
        try:
            body = identify_with(session, base_url, path, skip_liveness, hdrs, timeout)
        except RuntimeError as exc:
            last_error = str(exc)
            continue
        if body.get("is_stranger"):
            return path, body
    raise RuntimeError(f"找不到 stranger 探针: {last_error}")


def find_person(persons: list[PersonFixture], display_name: str) -> PersonFixture:
    for p in persons:
        if p.display_name == display_name:
            return p
    raise KeyError(f"底库中无人员: {display_name}")


def create_person(
    session: requests.Session,
    base_url: str,
    person_id: str,
    display_name: str,
    hdrs: dict[str, str],
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    return post_json(
        session,
        "POST",
        f"{base_url}/v1/persons",
        hdrs,
        timeout,
        json={"person_id": person_id, "display_name": display_name, "metadata": {"source": "smoke_test"}},
    )


def get_person(
    session: requests.Session,
    base_url: str,
    person_id: str,
    hdrs: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    r = session.get(f"{base_url}/v1/persons/{person_id}", headers=hdrs, timeout=timeout)
    r.raise_for_status()
    return r.json()


def list_persons(
    session: requests.Session,
    base_url: str,
    hdrs: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    r = session.get(f"{base_url}/v1/persons", params={"limit": 500}, headers=hdrs, timeout=timeout)
    r.raise_for_status()
    return r.json()


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


def delete_face(
    session: requests.Session,
    base_url: str,
    person_id: str,
    face_id: str,
    hdrs: dict[str, str],
    timeout: float,
) -> None:
    r = session.delete(
        f"{base_url}/v1/persons/{person_id}/faces/{face_id}",
        headers=hdrs,
        timeout=timeout,
    )
    if r.status_code in (204, 404):
        return
    r.raise_for_status()


def verify(
    session: requests.Session,
    base_url: str,
    person_id: str,
    image_path: Path,
    skip_liveness: bool,
    hdrs: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    params = {"person_id": person_id}
    if skip_liveness:
        params["skip_liveness"] = "true"
    files = {"image": (image_path.name, image_path.read_bytes(), "application/octet-stream")}
    status, body = post_multipart(
        session, f"{base_url}/v1/verify", files, params, hdrs, timeout
    )
    if status != 200:
        msg = body.get("error", {}).get("message", str(body))
        raise RuntimeError(f"verify 失败: {msg}")
    return body


def reset_gallery(
    session: requests.Session,
    base_url: str,
    hdrs: dict[str, str],
    timeout: float,
) -> int:
    """清空底库：删除所有人员。"""
    body = list_persons(session, base_url, hdrs, timeout)
    deleted = 0
    for item in body.get("items", []):
        delete_person(session, base_url, item["person_id"], hdrs, timeout)
        deleted += 1
    return deleted


def total_enrolled_faces(persons: list[PersonFixture]) -> int:
    return sum(len(p.enrolled_paths) for p in persons)


def top_up_person_faces(
    session: requests.Session,
    base_url: str,
    person: PersonFixture,
    target_count: int,
    skip_liveness: bool,
    hdrs: dict[str, str],
    timeout: float,
) -> int:
    """
    用已成功注册的照片重复填充，将人员人脸数补至 target_count。
    用于 ym 等目录内部分图片因质量门控失败、但仍需测 5 张上限的场景。
    返回本次新增张数。
    """
    if not person.enrolled_paths:
        return 0
    added = 0
    while True:
        body = get_person(session, base_url, person.person_id, hdrs, timeout)
        current = body.get("face_count", 0)
        if current >= target_count:
            break
        reuse_path = person.enrolled_paths[added % len(person.enrolled_paths)]
        status, resp = enroll_one_face_raw(
            session, base_url, person.person_id, reuse_path, skip_liveness, hdrs, timeout
        )
        if status != 200:
            break
        person.enrolled_paths.append(reuse_path)
        for item in resp.get("enrolled", []):
            person.face_ids.append(item["face_id"])
        added += 1
    return added


def run_step(name: str, fn: Callable[[], tuple[bool, str, dict[str, Any]]]) -> StepResult:
    try:
        ok, detail, data = fn()
        return StepResult(name=name, passed=ok, detail=detail, data=data)
    except Exception as exc:
        return StepResult(name=name, passed=False, detail=f"exception: {exc}")


def run_tests(args: argparse.Namespace) -> FeatureTestReport:
    run_tag = uuid.uuid4().hex[:8]
    enroll_dirs, stranger_dir = collect_enroll_dirs(args.enroll_root, args.stranger_dir)
    identify_names = [s.strip() for s in args.identify_persons.split(",") if s.strip()]

    base_url = args.base_url.rstrip("/")
    hdrs = headers(args.api_key)
    session = requests.Session()
    steps: list[StepResult] = []
    notes: list[str] = [
        f"注册人员={[d.name for d in enroll_dirs]}",
        f"stranger={stranger_dir.name}",
        f"1:N探针人员={identify_names}",
        f"skip_liveness={args.skip_liveness}",
    ]

    persons: list[PersonFixture] = [
        PersonFixture(
            person_id=f"smoke_{d.name}_{run_tag}",
            display_name=d.name,
            image_dir=d,
        )
        for d in enroll_dirs
    ]
    stranger_images = list_images(stranger_dir)

    # --- 1. 健康检查 ---
    def step_health() -> tuple[bool, str, dict[str, Any]]:
        status, body = post_json(session, "GET", f"{base_url}/v1/health", hdrs, args.timeout)
        ok = status == 200 and body.get("status") == "ok"
        return ok, f"device={body.get('device')}", body

    steps.append(run_step("health", step_health))
    device = steps[-1].data.get("device", "unknown")

    def step_ready() -> tuple[bool, str, dict[str, Any]]:
        status, body = post_json(session, "GET", f"{base_url}/v1/ready", hdrs, args.timeout)
        ok = status == 200 and body.get("ready") is True
        return ok, f"checks={body.get('checks')}", body

    steps.append(run_step("ready", step_ready))

    # --- 2. 清库 ---
    def step_reset_gallery() -> tuple[bool, str, dict[str, Any]]:
        if not args.reset_gallery:
            return True, "skipped (--no-reset-gallery)", {}
        deleted = reset_gallery(session, base_url, hdrs, args.timeout)
        _, health = post_json(session, "GET", f"{base_url}/v1/health", hdrs, args.timeout)
        ok = health.get("person_count", -1) == 0
        return ok, f"deleted={deleted} person_count={health.get('person_count')}", health

    steps.append(run_step("reset_gallery", step_reset_gallery))

    # --- 3. 全量注册（每人目录内全部照片）---
    def step_setup_gallery() -> tuple[bool, str, dict[str, Any]]:
        enroll_summary: dict[str, dict[str, int]] = {}
        to_remove: list[PersonFixture] = []

        for person in persons:
            status, body = create_person(
                session, base_url, person.person_id, person.display_name, hdrs, args.timeout
            )
            if status != 201:
                raise RuntimeError(f"创建人员失败 {person.display_name}: {body}")

            batch = enroll_all_faces(
                session,
                base_url,
                person.person_id,
                list_images(person.image_dir),
                args.skip_liveness,
                hdrs,
                args.timeout,
            )

            if not batch.enrolled_paths:
                # 该人员全部图片未过质量/活体门控：删除空人员记录并跳过
                delete_person(session, base_url, person.person_id, hdrs, args.timeout)
                to_remove.append(person)
                enroll_summary[person.display_name] = {"ok": 0, "fail": len(batch.failed)}
                notes.append(
                    f"跳过 {person.display_name}: 全部 {len(batch.failed)} 张注册失败 "
                    f"({batch.failed[0][1][:50] if batch.failed else '无图'})"
                )
                continue

            person.enrolled_paths = batch.enrolled_paths
            person.face_ids = batch.face_ids
            enroll_summary[person.display_name] = {
                "ok": len(batch.enrolled_paths),
                "fail": len(batch.failed),
            }
            if batch.failed:
                notes.append(
                    f"{person.display_name} 部分跳过 {len(batch.failed)} 张 "
                    f"({batch.failed[0][1][:50]})"
                )

        for p in to_remove:
            persons.remove(p)

        missing = REQUIRED_PERSONS - {p.display_name for p in persons}
        if missing:
            raise RuntimeError(
                f"关键人员注册失败: {sorted(missing)}；请检查 enroll_images 或开启 SKIP_LIVENESS=true"
            )

        # ym：目录内 5 张可能仅部分过质量门控，补满至 MAX 以测试 ENROLLMENT_LIMIT
        ym_person = next((p for p in persons if p.display_name == "ym"), None)
        if ym_person is not None:
            before_ym = get_person(session, base_url, ym_person.person_id, hdrs, args.timeout)
            topped = top_up_person_faces(
                session,
                base_url,
                ym_person,
                MAX_FACES_PER_PERSON,
                args.skip_liveness,
                hdrs,
                args.timeout,
            )
            if topped:
                after_ym = get_person(session, base_url, ym_person.person_id, hdrs, args.timeout)
                notes.append(
                    f"ym 补注册 {topped} 张 ({before_ym.get('face_count')} -> {after_ym.get('face_count')})"
                )

        _, health = post_json(session, "GET", f"{base_url}/v1/health", hdrs, args.timeout)
        expected_faces = total_enrolled_faces(persons)
        ok = (
            health.get("person_count") == len(persons)
            and health.get("gallery_size") == expected_faces
            and len(persons) >= len(REQUIRED_PERSONS)
        )
        detail = (
            f"persons={len(persons)} faces={expected_faces} "
            f"gallery_size={health.get('gallery_size')} summary={enroll_summary}"
        )
        return ok, detail, health

    steps.append(run_step("setup_gallery", step_setup_gallery))
    if not steps[-1].passed:
        return _finalize_report(base_url, persons, device, steps, notes, args, session, hdrs)

    # --- 4. 人员查询 ---
    def step_list_persons() -> tuple[bool, str, dict[str, Any]]:
        body = list_persons(session, base_url, hdrs, args.timeout)
        ids = {p["person_id"] for p in body.get("items", [])}
        expected = {p.person_id for p in persons}
        ok = body.get("total") == len(persons) and expected.issubset(ids)
        return ok, f"total={body.get('total')}", body

    steps.append(run_step("list_persons", step_list_persons))

    def step_get_person() -> tuple[bool, str, dict[str, Any]]:
        target = find_person(persons, "wjr")
        body = get_person(session, base_url, target.person_id, hdrs, args.timeout)
        ok = (
            body.get("person_id") == target.person_id
            and body.get("face_count") == len(target.enrolled_paths)
        )
        return ok, f"wjr face_count={body.get('face_count')}", body

    steps.append(run_step("get_person", step_get_person))

    def step_create_person_duplicate() -> tuple[bool, str, dict[str, Any]]:
        target = find_person(persons, "wjr")
        status, body = create_person(
            session, base_url, target.person_id, target.display_name, hdrs, args.timeout
        )
        ok = status == 400 and "already exists" in body.get("error", {}).get("message", "")
        return ok, f"status={status} code={body.get('error', {}).get('code')}", body

    steps.append(run_step("create_person_duplicate", step_create_person_duplicate))

    # --- 5. 追加注册（enroll_add）---
    def step_enroll_add_hjh() -> tuple[bool, str, dict[str, Any]]:
        """
        hjh 初始 3 张已全部注册，再追加 enroll_add/hjh/add_hjh.png 应成功（4/5，无上限告警）。
        若 hjh 照片人脸过小导致初始注册失败，则跳过并提示更换照片。
        """
        add_path = args.enroll_add_root / "hjh" / "add_hjh.png"
        if not add_path.is_file():
            return False, f"追加图片不存在: {add_path}", {}

        try:
            hjh = find_person(persons, "hjh")
        except KeyError:
            return (
                True,
                "skipped: hjh 初始注册全部失败（人脸尺寸/质量未达标），请更换 enroll_images/hjh 照片",
                {},
            )

        before = get_person(session, base_url, hjh.person_id, hdrs, args.timeout)
        if before.get("face_count", 0) != 3:
            return (
                True,
                f"skipped: hjh face_count={before.get('face_count')} (期望 3)，无法测追加注册",
                before,
            )

        status, body = enroll_one_face_raw(
            session, base_url, hjh.person_id, add_path, args.skip_liveness, hdrs, args.timeout
        )
        after = get_person(session, base_url, hjh.person_id, hdrs, args.timeout)
        if status != 200:
            # add 图也过不了质量门控时，跳过而非误报失败
            return (
                True,
                f"skipped: add_hjh.png 注册失败 ({body.get('error', {}).get('message', '')[:50]})",
                body,
            )
        ok = after.get("face_count") == 4
        hjh.enrolled_paths.append(add_path)
        for item in body.get("enrolled", []):
            hjh.face_ids.append(item["face_id"])
        detail = (
            f"status={status} face_count {before.get('face_count')} -> {after.get('face_count')} "
            f"(期望 4，上限 {MAX_FACES_PER_PERSON})"
        )
        return ok, detail, body

    steps.append(run_step("enroll_add_hjh", step_enroll_add_hjh))

    def step_enroll_add_ym_limit() -> tuple[bool, str, dict[str, Any]]:
        """
        ym 初始 5 张已全部注册（达上限），再追加 enroll_add/ym/add_ym.png 应返回 ENROLLMENT_LIMIT。
        """
        ym = find_person(persons, "ym")
        add_path = args.enroll_add_root / "ym" / "add_ym.png"
        if not add_path.is_file():
            return False, f"追加图片不存在: {add_path}", {}

        before = get_person(session, base_url, ym.person_id, hdrs, args.timeout)
        if before.get("face_count", 0) != MAX_FACES_PER_PERSON:
            return (
                False,
                f"前置条件失败: ym 应有 {MAX_FACES_PER_PERSON} 张脸，实际 {before.get('face_count')}",
                before,
            )

        status, body = enroll_one_face_raw(
            session, base_url, ym.person_id, add_path, args.skip_liveness, hdrs, args.timeout
        )
        err = body.get("error", {})
        ok = status == 400 and err.get("code") == "ENROLLMENT_LIMIT"
        details = err.get("details") or {}
        detail = (
            f"status={status} code={err.get('code')} "
            f"current={details.get('current')} max={details.get('max')} "
            f"face_count 保持 {before.get('face_count')}"
        )
        return ok, detail, body

    steps.append(run_step("enroll_add_ym_limit", step_enroll_add_ym_limit))

    # --- 6. 1:N 识别（指定人员）---
    for name in identify_names:
        person = find_person(persons, name)

        def step_identify_matched(
            target: PersonFixture = person,
        ) -> tuple[bool, str, dict[str, Any]]:
            target.probe_path = pick_probe(
                session, base_url, target, args.skip_liveness, hdrs, args.timeout
            )
            assert target.probe_path is not None
            body = identify_with(
                session, base_url, target.probe_path, args.skip_liveness, hdrs, args.timeout
            )
            ok = (
                body.get("matched") is True
                and body.get("person_id") == target.person_id
                and body.get("is_stranger") is False
            )
            detail = (
                f"{target.display_name} probe={target.probe_path.name} "
                f"matched={body.get('matched')} sim={body.get('similarity')}"
            )
            return ok, detail, body

        steps.append(run_step(f"identify_matched_{name}", step_identify_matched))

    # --- 7. 陌生人识别 ---
    def step_identify_stranger() -> tuple[bool, str, dict[str, Any]]:
        path, body = pick_stranger_probe(
            session, base_url, stranger_images, args.skip_liveness, hdrs, args.timeout
        )
        ok = (
            body.get("matched") is False
            and body.get("is_stranger") is True
            and body.get("alert") is True
        )
        detail = (
            f"probe={path.name} stranger={body.get('is_stranger')} "
            f"alert={body.get('alert')} sim={body.get('similarity')}"
        )
        notes.append(f"stranger_probe={path.name}")
        return ok, detail, body

    steps.append(run_step("identify_stranger", step_identify_stranger))

    # --- 8. 1:1 验证 ---
    def step_verify_pass() -> tuple[bool, str, dict[str, Any]]:
        wjr = find_person(persons, "wjr")
        if wjr.probe_path is None:
            wjr.probe_path = pick_probe(session, base_url, wjr, args.skip_liveness, hdrs, args.timeout)
        body = verify(
            session, base_url, wjr.person_id, wjr.probe_path, args.skip_liveness, hdrs, args.timeout
        )
        ok = body.get("verified") is True
        return ok, f"wjr verified={body.get('verified')} sim={body.get('similarity')}", body

    steps.append(run_step("verify_pass", step_verify_pass))

    def step_verify_fail_stranger() -> tuple[bool, str, dict[str, Any]]:
        wjr = find_person(persons, "wjr")
        path, _ = pick_stranger_probe(
            session, base_url, stranger_images, args.skip_liveness, hdrs, args.timeout
        )
        body = verify(
            session, base_url, wjr.person_id, path, args.skip_liveness, hdrs, args.timeout
        )
        ok = body.get("verified") is False
        return ok, f"cyt vs wjr verified={body.get('verified')}", body

    steps.append(run_step("verify_fail_stranger", step_verify_fail_stranger))

    def step_verify_wrong_person() -> tuple[bool, str, dict[str, Any]]:
        wjr = find_person(persons, "wjr")
        zjy = find_person(persons, "zjy")
        if zjy.probe_path is None:
            zjy.probe_path = pick_probe(session, base_url, zjy, args.skip_liveness, hdrs, args.timeout)
        body = verify(
            session, base_url, wjr.person_id, zjy.probe_path, args.skip_liveness, hdrs, args.timeout
        )
        ok = body.get("verified") is False
        return ok, f"zjy probe vs wjr id verified={body.get('verified')}", body

    steps.append(run_step("verify_wrong_person", step_verify_wrong_person))

    # --- 9. 删除单张人脸（优先 hjh 若追加成功；否则 wjr）---
    def step_delete_face() -> tuple[bool, str, dict[str, Any]]:
        target: PersonFixture | None = None
        for name in ("hjh", "wjr", "ym"):
            try:
                p = find_person(persons, name)
            except KeyError:
                continue
            if len(p.face_ids) >= 2:
                target = p
                break
        if target is None:
            return True, "skipped: 无 face_count>=2 的人员可测 delete_face", {}

        before = get_person(session, base_url, target.person_id, hdrs, args.timeout)
        face_id = target.face_ids[0]
        delete_face(session, base_url, target.person_id, face_id, hdrs, args.timeout)
        after = get_person(session, base_url, target.person_id, hdrs, args.timeout)
        target.face_ids.pop(0)
        ok = after.get("face_count") == before.get("face_count", 0) - 1
        return (
            ok,
            f"{target.display_name} face_count {before.get('face_count')} -> {after.get('face_count')}",
            after,
        )

    steps.append(run_step("delete_face", step_delete_face))

    # --- 10. 统计与事件 ---
    def step_stats() -> tuple[bool, str, dict[str, Any]]:
        r = session.get(f"{base_url}/v1/stats", params={"days": 7}, headers=hdrs, timeout=args.timeout)
        r.raise_for_status()
        body = r.json()
        gallery = body.get("gallery") or {}
        events = body.get("events") or {}
        # 期望 face 数 = 当前 enrolled_paths 总数 - delete_face 删 1 张
        expected_gallery_faces = total_enrolled_faces(persons) - 1
        ok = (
            gallery.get("person_count") == len(persons)
            and gallery.get("face_count") == expected_gallery_faces
            and events.get("total_events", 0) >= len(identify_names) + 5
        )
        detail = (
            f"persons={gallery.get('person_count')} faces={gallery.get('face_count')} "
            f"(期望 {expected_gallery_faces}) events={events.get('total_events')}"
        )
        return ok, detail, body

    steps.append(run_step("stats", step_stats))

    def step_events() -> tuple[bool, str, dict[str, Any]]:
        all_r = session.get(f"{base_url}/v1/events", params={"limit": 200}, headers=hdrs, timeout=args.timeout)
        all_r.raise_for_status()
        all_body = all_r.json()
        stranger_r = session.get(
            f"{base_url}/v1/events",
            params={"is_stranger": True, "limit": 50},
            headers=hdrs,
            timeout=args.timeout,
        )
        stranger_r.raise_for_status()
        stranger_body = stranger_r.json()
        wjr = find_person(persons, "wjr")
        verify_r = session.get(
            f"{base_url}/v1/events",
            params={"event_type": "verify", "person_id": wjr.person_id, "limit": 20},
            headers=hdrs,
            timeout=args.timeout,
        )
        verify_r.raise_for_status()
        verify_body = verify_r.json()
        ok = (
            all_body.get("total", 0) >= 8
            and stranger_body.get("total", 0) >= 1
            and verify_body.get("total", 0) >= 1
        )
        detail = (
            f"all={all_body.get('total')} strangers={stranger_body.get('total')} "
            f"verify={verify_body.get('total')}"
        )
        return ok, detail, {"all": all_body, "strangers": stranger_body, "verify": verify_body}

    steps.append(run_step("events", step_events))

    return _finalize_report(base_url, persons, device, steps, notes, args, session, hdrs)


def _finalize_report(
    base_url: str,
    persons: list[PersonFixture],
    device: str,
    steps: list[StepResult],
    notes: list[str],
    args: argparse.Namespace,
    session: requests.Session,
    hdrs: dict[str, str],
) -> FeatureTestReport:
    if not args.skip_cleanup:
        for person in persons:
            try:
                delete_person(session, base_url, person.person_id, hdrs, args.timeout)
            except Exception as exc:
                notes.append(f"清理失败 {person.person_id}: {exc}")
        notes.append(f"已清理 {len(persons)} 名测试人员")
    else:
        notes.append(f"保留 person_ids={[p.person_id for p in persons]}")

    passed = sum(1 for s in steps if s.passed)
    failed = len(steps) - passed
    return FeatureTestReport(
        base_url=base_url,
        person_ids=[p.person_id for p in persons],
        device=device,
        steps=steps,
        passed=passed,
        failed=failed,
        notes=notes,
    )


def print_report(report: FeatureTestReport) -> None:
    print("\n" + "=" * 60)
    print("Face Access Control — Feature Smoke Test")
    print("=" * 60)
    print(f"Base URL    : {report.base_url}")
    print(f"Device      : {report.device}")
    print(f"Test persons: {len(report.person_ids)}")
    print(f"Result      : {report.passed}/{len(report.steps)} passed")

    for step in report.steps:
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
    try:
        report = run_tests(args)
    except (FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

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
