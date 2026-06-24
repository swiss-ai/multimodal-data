#!/bin/bash
# Compatibility shim. Source the repo-level runtime bootstrap instead.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer}"
source "${REPO_DIR}/scripts/utils/source_lhotse_runtime.sh"
