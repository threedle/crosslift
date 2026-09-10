#!/usr/bin/env bash
# Fetch and build the QuadWild-BiMDF binaries used by the "quadwild" extraction
# backend. Optional: every other backend works without it.
set -euo pipefail

REPO="https://github.com/cgg-bern/quadwild-bimdf.git"
REF="cbda68e5deddf9d0e24c24382852f37a6eb2a630"  # v0.0.2-7

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
prefix="$root_dir/third_party/quadwild"
jobs=$(nproc 2>/dev/null || echo 4)
ref="$REF"

usage() {
  cat <<EOF
Usage: scripts/install_quadwild.sh [options]

  --prefix DIR   where to clone and build (default: third_party/quadwild)
  --ref COMMIT   quadwild-bimdf commit to build (default: ${REF:0:12})
  --jobs N, -j N build parallelism (default: $jobs)
  -h, --help     this message

Export QUADWILD_DIR=<prefix> to use an install outside the default location.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) prefix=$2; shift 2 ;;
    --ref) ref=$2; shift 2 ;;
    --jobs|-j) jobs=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for tool in git cmake; do
  command -v "$tool" >/dev/null || { echo "error: $tool is required" >&2; exit 1; }
done

# quadwild-bimdf is C++20; GCC 10 is the first release that compiles it.
if command -v g++ >/dev/null; then
  gcc_major=$(g++ -dumpversion | cut -d. -f1)
  if [ "$gcc_major" -lt 10 ]; then
    echo "error: g++ $gcc_major is too old for C++20; need 10 or newer" >&2
    exit 1
  fi
fi

echo "==> QuadWild source: $prefix"
if [ -d "$prefix/.git" ]; then
  git -C "$prefix" fetch --depth 1 origin "$ref"
  git -C "$prefix" checkout --detach FETCH_HEAD
else
  mkdir -p "$(dirname "$prefix")"
  git clone --filter=blob:none "$REPO" "$prefix"
  git -C "$prefix" checkout --detach "$ref"
fi

echo "==> Submodules"
if ! git -C "$prefix" submodule update --init --recursive --depth 1; then
  # Fallback: retry with git's proxy bypassed for the RWTH GitLab on port 9000.
  echo "==> Retrying submodules without the git proxy for graphics.rwth-aachen.de:9000"
  git -c "http.https://www.graphics.rwth-aachen.de:9000/.proxy=" \
      -c "http.https://gitlab.vci.rwth-aachen.de:9000/.proxy=" \
      -C "$prefix" submodule update --init --recursive --depth 1 --force
fi

# CoMISo, lemon and lpsolve declare pre-3.5 minimums and policies that CMake 4 rejects.
policy_flag=()
if [ "$(cmake --version | head -1 | cut -d' ' -f3 | cut -d. -f1)" -ge 4 ]; then
  policy_flag=(-D CMAKE_POLICY_VERSION_MINIMUM=3.5)
  sed -i 's/CMAKE_POLICY(SET CMP0048 OLD)/CMAKE_POLICY(SET CMP0048 NEW)/' \
    "$prefix/libs/lemon/CMakeLists.txt"
fi

generator=()
command -v ninja >/dev/null && generator=(-G Ninja)

echo "==> Configuring"
cmake -S "$prefix" -B "$prefix/build" \
  "${generator[@]}" "${policy_flag[@]}" \
  -D CMAKE_BUILD_TYPE=Release \
  -D QUADRETOPOLOGY_WITH_GUROBI=OFF \
  -D SATSUMA_ENABLE_BLOSSOM5=OFF

echo "==> Building (-j $jobs)"
cmake --build "$prefix/build" -j "$jobs" --target quadwild quad_from_patches

bin_dir="$prefix/build/Build/bin"
for binary in quadwild quad_from_patches; do
  [ -x "$bin_dir/$binary" ] || { echo "error: $bin_dir/$binary missing" >&2; exit 1; }
done

echo
echo "QuadWild built: $bin_dir"
if [ "$prefix" != "$root_dir/third_party/quadwild" ]; then
  echo "Add to your environment:"
  echo "  export QUADWILD_DIR=$prefix"
fi
