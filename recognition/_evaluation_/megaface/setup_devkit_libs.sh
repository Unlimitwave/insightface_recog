#!/usr/bin/env bash
# Fetch runtime libs required by MegaFace devkit binaries (Identification, FuseResults).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LIBDIR="${ROOT}/third_party/lib"
mkdir -p "${LIBDIR}"

fetch_deb_libs() {
  local url="$1"
  local glob="$2"
  local tmp deb
  tmp="$(mktemp -d)"
  deb="${tmp}/pkg.deb"
  wget -q -O "${deb}" "${url}"
  dpkg-deb -x "${deb}" "${tmp}/extract"
  cp -a ${tmp}/extract/${glob} "${LIBDIR}/"
  rm -rf "${tmp}"
}

if [[ ! -f "${LIBDIR}/libtbb.so.2" ]]; then
  echo "Fetching libtbb.so.2 ..."
  fetch_deb_libs \
    "http://archive.ubuntu.com/ubuntu/pool/universe/libt/libtbb2/libtbb2_2020.3-1ubuntu3_amd64.deb" \
    "usr/lib/x86_64-linux-gnu/libtbb.so.2*"
fi

if [[ ! -e "${LIBDIR}/libopencv_core.so.2.4" ]]; then
  echo "Fetching libopencv_core.so.2.4 ..."
  fetch_deb_libs \
    "http://archive.ubuntu.com/ubuntu/pool/universe/o/opencv/libopencv-core2.4v5_2.4.9.1+dfsg-1.5ubuntu1.1_amd64.deb" \
    "usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4*"
fi

export LD_LIBRARY_PATH="${LIBDIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
