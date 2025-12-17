#!/usr/bin/env bash
#SBATCH --account=infra01
#SBATCH --job-name=impOSM
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
    --import-split)
        import_split="$2"
        shift 2
        ;;
    --total-splits)
        total_splits="$2"
        shift 2
        ;;
    --slice-dir)
        slice_dir="$2"
        shift 2
        ;;
    --volume-dir)
        volume_dir="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done

# Validate required arguments
if [[ -z "${import_split:-}" ||
    -z "${total_splits:-}" ||
    -z "${slice_dir:-}" ||
    -z "${volume_dir:-}" ]]; then
    echo "ERROR: Missing required arguments. See README for details."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "${volume_dir}"

function import_osm() {
    local pbf_path="$1"
    local export_path="$2"
    local volume_name="osm-data-$(date +%s)"

    if [ ! -f "${pbf_path}" ]; then
        echo "$(date) | ERROR: PBF file not found: ${pbf_path}"
        exit 1
    fi

    if [ -f "${export_path}" ]; then
        echo "$(date) | WARNING: Export file already exists: ${export_path}, skipping import."
        return
    fi

    # echo "$(date) | Resetting podman environment"
    # podman system reset -f

    echo "$(date) | Importing OSM PBF file: ${pbf_path}"
    chmod 644 "${pbf_path}"
    podman volume create ${volume_name}
    # Add -e "FLAT_NODES=enabled" for tuning performance on large imports
    podman create --name osm-import -v ${volume_name}:/data/database/ overv/openstreetmap-tile-server import
    podman cp ${pbf_path} osm-import:/data/region.osm.pbf # Necessary to avoid permission issues
    podman start --attach osm-import
    podman rm osm-import

    echo "$(date) | Exporting volume: ${export_path}"
    podman volume export ${volume_name} --output "${export_path}"
    podman volume rm ${volume_name}
    echo "$(date) | Export completed."
}

# Sort and split the list of slices
slice_list=($(ls ${slice_dir}/slice_*.pbf))
IFS=$'\n' slice_list=($(sort <<<"${slice_list[*]}"))
unset IFS
total_slices=${#slice_list[@]}
slices_per_split=$(((total_slices + total_splits - 1) / total_splits))
start_index=$((import_split * slices_per_split))
end_index=$((start_index + slices_per_split))
if [ ${end_index} -gt ${total_slices} ]; then
    end_index=${total_slices}
fi
slice_list=("${slice_list[@]:start_index:end_index-start_index}")

echo "$(date) | Importing slices for split ${import_split}/${total_splits}: indices ${start_index} to ${end_index} (total slices: ${total_slices})"

# Import each slice in the assigned split
for pbf in "${slice_list[@]}"; do
    bbox=$(basename "${pbf}" .pbf | sed 's/slice_//')
    export="${volume_dir}/osm-data_${bbox}.tar"
    import_osm "${pbf}" "${export}"
done

echo "$(date) | All imports completed."
