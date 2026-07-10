#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

MEGAFACE_ROOT="${MEGAFACE_ROOT:-/home/cmsr/桌面/东风/数据集/megaface}"
ALGO="${ALGO:-buffalo_l}"
GALLERY_SIZE="${GALLERY_SIZE:-1000000}"
ENROLL_COUNTS="${ENROLL_COUNTS:-1 3 5 10}"
PYTHON="${PYTHON:-python3}"
OUTPUT="${OUTPUT:-${ROOT}/results/${ALGO}/enroll_ablation.json}"

cd "${ROOT}"
"${PYTHON}" -u enroll_count_eval.py \
  --megaface-root "${MEGAFACE_ROOT}" \
  --algo "${ALGO}" \
  --gallery-size "${GALLERY_SIZE}" \
  --enroll-counts ${ENROLL_COUNTS} \
  --output "${OUTPUT}"
