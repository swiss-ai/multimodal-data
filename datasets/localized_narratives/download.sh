#!/usr/bin/env bash
# Download Localized Narratives audio + text transcriptions from Google Cloud Storage.
#
# Layout under $DEST:
#   annotations/<split>_*.jsonl              full annotations (caption + mouse trace + audio offsets)
#   voice-recordings/<split>/<id>.ogg        per-utterance audio
#   urls.txt, audio_paths.txt                intermediate, kept for resumability
#
# Resume: re-run skips files already present (aria2c --auto-file-renaming=false).
#
# Usage:  bash download.sh [SPLIT_REGEX]
#   SPLIT_REGEX (optional) e.g. coco_val, open_images, ade20k. Default = all.

set -euo pipefail

DEST="${DEST:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/localized_narratives}"
GCS="https://storage.googleapis.com/localized-narratives"
SPLIT_REGEX="${1:-.*}"

ANN_DIR="$DEST/annotations"
AUDIO_DIR="$DEST/voice-recordings"
mkdir -p "$ANN_DIR" "$AUDIO_DIR"

# All annotation file basenames (https://google.github.io/localized-narratives/)
ANN_FILES=(
  open_images_train_v6_localized_narratives-00000-of-00010.jsonl
  open_images_train_v6_localized_narratives-00001-of-00010.jsonl
  open_images_train_v6_localized_narratives-00002-of-00010.jsonl
  open_images_train_v6_localized_narratives-00003-of-00010.jsonl
  open_images_train_v6_localized_narratives-00004-of-00010.jsonl
  open_images_train_v6_localized_narratives-00005-of-00010.jsonl
  open_images_train_v6_localized_narratives-00006-of-00010.jsonl
  open_images_train_v6_localized_narratives-00007-of-00010.jsonl
  open_images_train_v6_localized_narratives-00008-of-00010.jsonl
  open_images_train_v6_localized_narratives-00009-of-00010.jsonl
  open_images_validation_localized_narratives.jsonl
  open_images_test_localized_narratives.jsonl
  coco_train_localized_narratives-00000-of-00004.jsonl
  coco_train_localized_narratives-00001-of-00004.jsonl
  coco_train_localized_narratives-00002-of-00004.jsonl
  coco_train_localized_narratives-00003-of-00004.jsonl
  coco_val_localized_narratives.jsonl
  flickr30k_train_localized_narratives.jsonl
  flickr30k_val_localized_narratives.jsonl
  flickr30k_test_localized_narratives.jsonl
  ade20k_train_localized_narratives.jsonl
  ade20k_validation_localized_narratives.jsonl
)

echo "[1/3] downloading annotation JSONLs to $ANN_DIR"
for f in "${ANN_FILES[@]}"; do
  if [[ ! "$f" =~ $SPLIT_REGEX ]]; then continue; fi
  if [[ -s "$ANN_DIR/$f" ]]; then
    echo "  skip (exists): $f"
    continue
  fi
  echo "  fetch: $f"
  wget -q --show-progress -P "$ANN_DIR" "$GCS/annotations/$f"
done

echo "[2/3] extracting voice_recording paths from JSONLs"
PATHS_FILE="$DEST/audio_paths.txt"
URLS_FILE="$DEST/urls.txt"
: > "$PATHS_FILE"
shopt -s nullglob
for f in "$ANN_DIR"/*.jsonl; do
  bn=$(basename "$f")
  if [[ ! "$bn" =~ $SPLIT_REGEX ]]; then continue; fi
  python3 - "$f" >> "$PATHS_FILE" <<'PY'
import json, sys
with open(sys.argv[1]) as fh:
    for line in fh:
        try:
            rec = json.loads(line)
            vr = rec.get("voice_recording")
            if vr:
                print(vr)
        except Exception:
            pass
PY
done
sort -u "$PATHS_FILE" -o "$PATHS_FILE"
N=$(wc -l < "$PATHS_FILE")
echo "  $N unique audio files"

# aria2c input format: URL on one line, "  out=<relative path>" on the next
# (preserves per-split subdirectory structure under $AUDIO_DIR)
awk -v g="$GCS/voice-recordings" '{print g"/"$0; print "  out="$0}' "$PATHS_FILE" > "$URLS_FILE"

echo "[3/3] downloading audio with aria2c (64 parallel, 16 conn/file, resume on)"
cd "$AUDIO_DIR"
aria2c \
  -i "$URLS_FILE" \
  -j 64 -x 16 -s 16 \
  --auto-file-renaming=false \
  --allow-overwrite=false \
  --continue=true \
  --console-log-level=warn \
  --summary-interval=60 \
  --dir="$AUDIO_DIR"

echo "done. files under: $AUDIO_DIR"
du -sh "$AUDIO_DIR" || true
