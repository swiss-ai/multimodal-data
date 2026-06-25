#!/usr/bin/env bash
#SBATCH --account=infra01
#SBATCH --job-name=runOSM
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
    --run-split)
        run_split="$2"
        shift 2
        ;;
    --total-splits)
        total_splits="$2"
        shift 2
        ;;
    --volume-dir)
        volume_dir="$2"
        shift 2
        ;;
    --save-dir)
        save_dir="$2"
        shift 2
        ;;
    --zoom-level)
        zoom_level="$2"
        shift 2
        ;;
    --sample-ratio)
        sample_ratio="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done

# Validate required arguments
if [[ -z "${run_split:-}" ||
    -z "${total_splits:-}" ||
    -z "${volume_dir:-}" ||
    -z "${save_dir:-}" ||
    -z "${zoom_level:-}" ||
    -z "${sample_ratio:-}" ]]; then
    echo "ERROR: Missing required arguments. See README for details."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "${save_dir}"

function start_tile_server() {
    local export_volume_path="$1"
    echo "$(date) | Starting tile server with volume: ${export_volume_path}"
    podman volume create osm-data
    podman volume import osm-data ${export_volume_path}
    podman run -p 8888:80 -v osm-data:/data/database/ --name osm-server --detach --rm overv/openstreetmap-tile-server run
    echo "$(date) | Tile server is running and accessible on port 8888."
}

function stop_tile_server() {
    echo "$(date) | Stopping tile server."
    podman stop osm-server
    podman volume rm osm-data
    echo "$(date) | Tile server stopped and volume removed."
}

function download_tiles() {
    local zoom="$1"
    local sample_ratio="$6"
    local save_dir="$7"
    local url="$8"

    echo "$(date) | Downloading $(echo "${sample_ratio} * 100" | bc)% of tiles at zoom level: ${zoom}"
    uv run run.py \
        --zoom ${zoom} \
        --bbox ${west} ${south} ${east} ${north} \
        --sample_ratio "${sample_ratio}" \
        --save_dir "${save_dir}" \
        --url "${url}"
    echo "$(date) | Download completed for zoom level: ${zoom}."
}

# Sort and split the list of volumes
volume_list=($(ls ${volume_dir}/osm-data_*.tar))
IFS=$'\n' volume_list=($(sort <<<"${volume_list[*]}"))
unset IFS
total_volumes=${#volume_list[@]}
volumes_per_split=$(((total_volumes + total_splits - 1) / total_splits))
start_index=$((run_split * volumes_per_split))
end_index=$((start_index + volumes_per_split))
if [ ${end_index} -gt ${total_volumes} ]; then
    end_index=${total_volumes}
fi
volume_list=("${volume_list[@]:start_index:end_index-start_index}")

echo "$(date) | Running tile downloads for split $((run_split + 1))/${total_splits}: indices ${start_index} to ${end_index} (total volumes: ${total_volumes})"

# Process each volume in the assigned split
for export_volume_path in "${volume_list[@]}"; do
    bbox=$(basename "${export_volume_path}" .tar | sed 's/osm-data_//')
    IFS='_' read -r west south east north <<<"${bbox}"
    echo "$(date) | Processing volume: ${export_volume_path} with bbox: west=${west}, south=${south}, east=${east}, north=${north}"

    start_tile_server "${export_volume_path}"
    sleep 60 # Wait for the server to be fully up

    tile_server_url="http://localhost:8888/tile/"
    mkdir -p "${save_dir}"
    download_tiles "${zoom_level}" "${west}" "${south}" "${east}" "${north}" "${sample_ratio}" "${save_dir}" "${tile_server_url}"
    stop_tile_server
done

echo "$(date) | All splits completed."
