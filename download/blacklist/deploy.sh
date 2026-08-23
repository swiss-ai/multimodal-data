#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
BUILD_DIR=""
ALLOW_HTTP_FALLBACK=false

usage() {
  echo "usage: $0 [--allow-plain-http-fallback] TARGET_DIR" >&2
}

cleanup() {
  if [[ -n "$BUILD_DIR" && -d "$BUILD_DIR" ]]; then
    rm -rf -- "$BUILD_DIR"
  fi
}

trap cleanup EXIT

if [[ "${1:-}" == "--allow-plain-http-fallback" ]]; then
  ALLOW_HTTP_FALLBACK=true
  shift
fi
if [[ $# -ne 1 || -z "$1" || "$1" == "/" ]]; then
  usage
  exit 2
fi

TARGET_PARENT=$(cd -- "$(dirname -- "$1")" && pwd -P)
TARGET_DIR="$TARGET_PARENT/$(basename -- "$1")"
if [[ "$TARGET_DIR" == "$SCRIPT_DIR" ]]; then
  echo "source checkout and deployment target must differ" >&2
  exit 1
fi
if [[ -L "$TARGET_DIR" ]]; then
  echo "refusing to deploy through a symbolic link: $TARGET_DIR" >&2
  exit 1
fi

BUILD_DIR=$(mktemp -d "${TMPDIR:-/tmp}/blacklist-build.XXXXXX")
build_args=(build --output-dir "$BUILD_DIR")
if [[ "$ALLOW_HTTP_FALLBACK" == true ]]; then
  build_args+=(--allow-plain-http-fallback)
fi
python3 "$SCRIPT_DIR/main.py" "${build_args[@]}"

for required_file in blocked_ip.txt blocked_domains.txt blocked_urls.txt manifest.json; do
  if [[ ! -s "$BUILD_DIR/$required_file" ]]; then
    echo "missing or empty build output: $required_file" >&2
    exit 1
  fi
done

install -m 0755 "$SCRIPT_DIR/proxy.sh" "$BUILD_DIR/proxy.sh"
mkdir -p -- "$TARGET_DIR"
rsync -a --delete --delete-excluded \
  --include='/proxy.sh' \
  --include='/blocked_ip.txt' \
  --include='/blocked_domains.txt' \
  --include='/blocked_urls.txt' \
  --include='/blocked_url_paths.txt' \
  --include='/manifest.json' \
  --exclude='*' \
  "$BUILD_DIR/" "$TARGET_DIR/"

echo "deployed blacklist runtime to $TARGET_DIR"
