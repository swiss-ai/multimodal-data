#!/usr/bin/env bash
# Wipe the legacy raw/commonvoice24/ tree (TSVs, leftover clips/, *.tar.gz,
# *_clips.tar.zst, nested cv-corpus-24.0-2025-12-05/). The v25 parquet
# pipeline never re-creates raw/, so this is a one-shot decommission of
# the v24 staging area.
#
# Defaults to DRY-RUN. Pass --force to actually delete.
#
# Usage:
#   ./wipe_cv_raw.sh             # show what would be deleted
#   ./wipe_cv_raw.sh --force     # actually delete

set -euo pipefail

RAW_ROOT="${RAW_ROOT:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/commonvoice24}"

DRY_RUN=1
for arg in "$@"; do
    case "$arg" in
        --force) DRY_RUN=0 ;;
        --raw-root=*) RAW_ROOT="${arg#*=}" ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

if [[ ! -d "${RAW_ROOT}" ]]; then
    echo "RAW_ROOT does not exist: ${RAW_ROOT}"
    exit 0
fi

echo "Target:  ${RAW_ROOT}"
echo "Mode:    $([[ "${DRY_RUN}" -eq 1 ]] && echo "DRY-RUN" || echo "DELETE")"
echo "Top-level entries (count: $(ls "${RAW_ROOT}" | wc -l)):"
ls -la "${RAW_ROOT}" | head -20
echo "..."
echo "(Skipping size/inode scan — Lustre stat-walk on millions of mp3s is slow."
echo " Run \`du -sh ${RAW_ROOT}\` separately if you want the full size.)"
echo

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "DRY-RUN — re-run with --force to actually wipe."
    echo "Equivalent command: rm -rf '${RAW_ROOT}' && mkdir -p '${RAW_ROOT}'"
    exit 0
fi

echo "Confirm deletion of ${RAW_ROOT} ? [y/N]"
read -r ans
if [[ "${ans}" != "y" && "${ans}" != "Y" ]]; then
    echo "Aborted."
    exit 1
fi

echo "Deleting ${RAW_ROOT} ..."
time rm -rf "${RAW_ROOT}"
mkdir -p "${RAW_ROOT}"
echo "Done. ${RAW_ROOT} is now empty."
