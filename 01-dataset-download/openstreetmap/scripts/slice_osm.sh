#!/usr/bin/env bash
#SBATCH --account=infra01
#SBATCH --job-name=sliceOSM
#SBATCH --output=/capstor/scratch/cscs/%u/logs/osm-%x-%A.out
#SBATCH --error=/capstor/scratch/cscs/%u/logs/osm-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1

set -euo pipefail
# set -x

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --west)
        west_bound="$2"
        shift 2
        ;;
    --south)
        south_bound="$2"
        shift 2
        ;;
    --east)
        east_bound="$2"
        shift 2
        ;;
    --north)
        north_bound="$2"
        shift 2
        ;;
    --lon-step)
        lon_step="$2"
        shift 2
        ;;
    --lat-step)
        lat_step="$2"
        shift 2
        ;;
    --planet-path)
        planet_path="$2"
        shift 2
        ;;
    --slice-dir)
        slice_dir="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done

# Validate required arguments
if [[ -z "${west_bound:-}" ||
    -z "${south_bound:-}" ||
    -z "${east_bound:-}" ||
    -z "${north_bound:-}" ||
    -z "${lon_step:-}" ||
    -z "${lat_step:-}" ||
    -z "${planet_path:-}" ||
    -z "${slice_dir:-}" ]]; then
    echo "ERROR: Missing required arguments. See README for details."
    exit 1
fi

# Check if osmium is installed
if ! command -v osmium &>/dev/null; then
    echo "$(date) | ERROR: osmium is not installed. Please install osmium-tools to proceed."
    exit 1
fi

# Create slice directory if it doesn't exist
mkdir -p "${slice_dir}"

function slice_osm() {
    local planet_path="$1"
    local slice_path="$2"
    local west="$3"
    local south="$4"
    local east="$5"
    local north="$6"

    if [ ! -f "${planet_path}" ]; then
        echo "$(date) | ERROR: Planet file not found: ${planet_path}"
        exit 1
    fi

    if [ -f "${slice_path}" ]; then
        echo "$(date) | WARNING: Slice file already exists: ${slice_path}, skipping extraction."
        return
    fi

    echo "$(date) | Extracting slice: ${slice_path}"
    osmium extract -b ${west},${south},${east},${north} -o "${slice_path}" "${planet_path}"
}

echo "$(date) | Starting slice extraction from planet file: ${planet_path}"

# Loop over the bounding box and extract slices
for west in $(seq ${west_bound} ${lon_step} $(echo "${east_bound} - ${lon_step}" | bc)); do
    east=$(echo "$west + $lon_step" | bc)
    for south in $(seq ${south_bound} ${lat_step} $(echo "${north_bound} - ${lat_step}" | bc)); do
        north=$(echo "$south + $lat_step" | bc)
        slice_path="${slice_dir}/slice_${west}_${south}_${east}_${north}.pbf"
        slice_osm "${planet_path}" "${slice_path}" "${west}" "${south}" "${east}" "${north}"
    done
done

echo "$(date) | All slices extracted."
