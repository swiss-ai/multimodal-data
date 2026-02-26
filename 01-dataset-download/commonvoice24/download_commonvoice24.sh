#!/bin/bash
# Download & extract Common Voice 24.0 from Mozilla Data Collective using aria2c
#
# Usage:
#   # Download specific languages
#   ./download_commonvoice24.sh it es en
#
#   # Download all EU languages
#   ./download_commonvoice24.sh all
#
#   # Custom destination
#   DEST_DIR=/my/path ./download_commonvoice24.sh it

set -euo pipefail

# Activate conda env with aria2c

DEST_DIR=${DEST_DIR:-/iopsstor/scratch/cscs/xyixuan/audio-datasets/raw/commonvoice24}
TOKEN_FILE="/iopsstor/scratch/cscs/${USER}/.mozilla_dc_token"
CONNECTIONS=${CONNECTIONS:-16}
API_BASE="https://datacollective.mozillafoundation.org/api"

declare -A DATASETS=(
  [sq]="cmj8u3ptn00ppnxxbr4rq2gpq"       # Albanian
  [hy-AM]="cmj8u3p8l00bpnxxbsku96l4i"     # Armenian
  [eu]="cmj8u3p2v007tnxxbk5ng5qvh"        # Basque
  [be]="cmj8u3oug0029nxxboll1mh9e"        # Belarusian
  [bg]="cmj8u3ov5002pnxxbn93qhttp"        # Bulgarian
  [ca]="cmj8u3oxm004dnxxb1ye0bcy9"        # Catalan
  [cs]="cmj8u3oyh004xnxxbd9uih97g"        # Czech
  [da]="cmj8u3ozj005hnxxbpiiwy7ph"        # Danish
  [nl]="cmj8u3pjj00j5nxxbnk8ajqgc"        # Dutch
  [en]="cmj8u3p1w0075nxxbe8bedl00"        # English
  [eo]="cmj8u3p210079nxxble126pq1"        # Esperanto
  [et]="cmj8u3p2i007lnxxbhnvfmx0m"        # Estonian
  [fi]="cmj8u3p3k0089nxxb4u6hcqd5"        # Finnish
  [fr]="cmj8u3p3x008hnxxb07gm9ke3"        # French
  [gl]="cmj8u3p5r009tnxxb88cqm12d"        # Galician
  [ka]="cmj8u3paf00cxnxxbkieg36mi"        # Georgian
  [de]="cmj8u3p05005xnxxbqah56m27"        # German
  [el]="cmj8u3p1q0071nxxbjywkh3f5"        # Greek
  [hu]="cmj8u3p8900bhnxxb50f37mkm"        # Hungarian
  [is]="cmj8u3p9l00cdnxxbuuascykx"        # Icelandic
  [ga-IE]="cmj8u3p4j008xnxxbub9hvwrd"     # Irish
  [it]="cmj8u3p9q00chnxxb5fj12aw8"        # Italian
  [ja]="cmj8u3p9x00clnxxbsv97o45e"        # Japanese
  [lv]="cmj8u3pec00flnxxbntvfb4as"        # Latvian
  [lt]="cmj8u3pdo00f5nxxb9uuewruj"        # Lithuanian
  [mk]="cmj8u3pg800gxnxxb36xwpf1l"        # Macedonian
  [mt]="cmj8u3phj00htnxxb83nuaddd"        # Maltese
  [nb-NO]="cmj8u3piw00ipnxxbo85o6mkz"     # Norwegian Bokmal
  [nn-NO]="cmj8u3pkd00jpnxxbsgaqp9gi"     # Norwegian Nynorsk
  [pl]="cmj8u3pmr00l9nxxb5spuq4bc"        # Polish
  [pt]="cmj8u3pnh00lpnxxb4djcexb1"        # Portuguese
  [ro]="cmj8u3pr700o1nxxbx35tiwrn"        # Romanian
  [ru]="cmj8u3prj00o9nxxbg5pbn88l"        # Russian
  [sr]="cmj8u3pts00ptnxxbjh4x5z2f"        # Serbian
  [sk]="cmj8u3pt400pdnxxb7y6dtbza"        # Slovak
  [sl]="cmj8u3pti00plnxxbhsgn2nek"        # Slovenian
  [es]="cmj8u3p26007dnxxbwyo07lb8"        # Spanish
  [sv-SE]="cmj8u3pud00q9nxxbcmq6uz24"     # Swedish
  [tr]="cmj8u3px500s1nxxb4qh79iqr"        # Turkish
  [uk]="cmj8u3pys00t5nxxb56wugqgq"        # Ukrainian
  [cy]="cmj8u3oz9005dnxxbiuyru29h"        # Welsh
  [zh-CN]="cmj8u3q2n00vhnxxbzrjcugwc"     # Chinese (China)
)

if [ ! -f "$TOKEN_FILE" ]; then
  echo "ERROR: Token not found at $TOKEN_FILE"
  echo "Create with: echo YOUR_TOKEN > $TOKEN_FILE && chmod 600 $TOKEN_FILE"
  exit 1
fi

TOKEN=$(cat "$TOKEN_FILE")
mkdir -p "$DEST_DIR"

download_and_extract() {
  local locale="$1"
  local dataset_id="${DATASETS[$locale]}"
  local tarfile="${DEST_DIR}/commonvoice24_${locale}.tar.gz"

  # Skip if already extracted
  if [ -d "${DEST_DIR}/${locale}" ] && [ -f "${DEST_DIR}/${locale}/train.tsv" ]; then
    echo "[${locale}] Already extracted, skipping"
    return 0
  fi

  # Download if needed
  if [ ! -f "$tarfile" ]; then
    echo "[${locale}] Getting download URL..."
    local response
    response=$(curl -s -X POST "${API_BASE}/datasets/${dataset_id}/download" \
      -H "Authorization: Bearer ${TOKEN}" \
      -H "Content-Type: application/json")

    local url
    url=$(echo "$response" | jq -r '.downloadUrl')

    if [ "$url" = "null" ] || [ -z "$url" ]; then
      echo "[${locale}] ERROR: Failed to get download URL"
      echo "  Response: $response"
      return 1
    fi

    echo "[${locale}] Downloading with aria2c (${CONNECTIONS} connections)..."
    aria2c -x "$CONNECTIONS" -s "$CONNECTIONS" \
      --continue=true \
      --max-tries=5 \
      --retry-wait=10 \
      --summary-interval=30 \
      -d "$DEST_DIR" \
      -o "commonvoice24_${locale}.tar.gz" \
      "$url"
  else
    echo "[${locale}] Archive exists, skipping download"
  fi

  # Extract
  echo "[${locale}] Extracting..."
  tar -xzf "$tarfile" -C "$DEST_DIR/"

  # Common Voice extracts to cv-corpus-24.0-2025-12-05/{locale}/
  # Move to flat structure: {DEST_DIR}/{locale}/
  local extracted="${DEST_DIR}/cv-corpus-24.0-2025-12-05/${locale}"
  if [ -d "$extracted" ]; then
    mv "$extracted" "${DEST_DIR}/${locale}"
    rmdir "${DEST_DIR}/cv-corpus-24.0-2025-12-05" 2>/dev/null || true
  fi

  # Clean up tar.gz
  rm -f "$tarfile"
  echo "[${locale}] Done: $(du -sh "${DEST_DIR}/${locale}" | cut -f1)"
}

# Parse arguments
if [ $# -eq 0 ]; then
  echo "Usage: $0 <lang_code...> | all"
  echo ""
  echo "Available languages:"
  for k in $(echo "${!DATASETS[@]}" | tr ' ' '\n' | sort); do
    echo "  $k"
  done
  exit 0
fi

LANGS=("$@")
if [ "${LANGS[0]}" = "all" ]; then
  LANGS=($(echo "${!DATASETS[@]}" | tr ' ' '\n' | sort))
fi

echo "Downloading & extracting ${#LANGS[@]} language(s) to ${DEST_DIR}"
echo "Connections per file: ${CONNECTIONS}"
echo ""

FAILED=()
for locale in "${LANGS[@]}"; do
  if [ -z "${DATASETS[$locale]+x}" ]; then
    echo "[${locale}] Unknown language code, skipping"
    FAILED+=("$locale")
    continue
  fi
  if ! download_and_extract "$locale"; then
    FAILED+=("$locale")
  fi
  echo ""
done

echo "=========================================="
echo "Done. Failed: ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "Failed languages: ${FAILED[*]}"
fi
