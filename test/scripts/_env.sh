# Shared env helpers for test/scripts (source after ensure_test_deps).
# Expects: TEST_PYTHON, TEST_ROOT

: "${BASE_URL:=http://localhost:8123}"

# Prefer person dirs whose sample photos usually pass size + liveness.
_PREFERRED_DIRS=(ym wjr whd zjy tjc)

pick_preferred_face_image() {
  local root="${1:-${TEST_ROOT}/enroll_images}"
  local name d f
  for name in "${_PREFERRED_DIRS[@]}"; do
    d="${root}/${name}"
    [[ -d "${d}" ]] || continue
    for f in "${d}"/*.{jpg,jpeg,png,webp}; do
      [[ -f "${f}" ]] || continue
      echo "${f}"
      return 0
    done
  done
  for d in "${root}"/*/; do
    [[ -d "${d}" ]] || continue
    for f in "${d}"*.{jpg,jpeg,png,webp}; do
      [[ -f "${f}" ]] || continue
      echo "${f}"
      return 0
    done
  done
  return 1
}

# Exit 0 if server allows skip_liveness; 1 if production blocks it.
probe_allows_skip_liveness() {
  BASE_URL="${BASE_URL}" API_KEY="${API_KEY:-}" TEST_ROOT="${TEST_ROOT}" \
    "${TEST_PYTHON}" - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ["TEST_ROOT"]) / "integration"))
import requests
from common import allows_skip_liveness, headers

base = os.environ.get("BASE_URL", "http://localhost:8123").rstrip("/")
key = os.environ.get("API_KEY") or None
ok = allows_skip_liveness(requests.Session(), base, headers(key), 30.0)
sys.exit(0 if ok else 1)
PY
}

# If SKIP_LIVENESS / REQUIRE_PRODUCTION unset, infer from live server policy.
resolve_runtime_env() {
  export TEST_ROOT BASE_URL

  if [[ "${SKIP_ENV_RESOLVE:-}" == "1" ]]; then
    echo "Env inherited: SKIP_LIVENESS=${SKIP_LIVENESS:-?} REQUIRE_PRODUCTION=${REQUIRE_PRODUCTION:-?} API_KEY=${API_KEY:+set}"
    return 0
  fi

  local skip_allowed=0
  if probe_allows_skip_liveness; then
    skip_allowed=1
  fi

  if [[ -z "${SKIP_LIVENESS+x}" ]]; then
    if [[ "${skip_allowed}" -eq 1 ]]; then
      export SKIP_LIVENESS=true
      echo "Auto: development → SKIP_LIVENESS=true"
    else
      export SKIP_LIVENESS=false
      echo "Auto: production blocks skip_liveness → SKIP_LIVENESS=false"
    fi
  else
    echo "SKIP_LIVENESS=${SKIP_LIVENESS} (user-set)"
  fi

  if [[ -z "${REQUIRE_PRODUCTION+x}" ]]; then
    if [[ "${skip_allowed}" -eq 0 ]]; then
      export REQUIRE_PRODUCTION=true
      echo "Auto: REQUIRE_PRODUCTION=true"
    else
      export REQUIRE_PRODUCTION=false
    fi
  else
    echo "REQUIRE_PRODUCTION=${REQUIRE_PRODUCTION} (user-set)"
  fi

  if [[ -n "${API_KEY:-}" ]]; then
    echo "API_KEY=set"
  else
    echo "API_KEY=unset (dev/no-auth only)"
  fi
}
