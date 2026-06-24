#!/usr/bin/env bash
#SBATCH --account=infra01
#SBATCH --job-name=dlOSM
#SBATCH --output=/capstor/scratch/cscs/%u/logs/osm-%x-%A.out
#SBATCH --error=/capstor/scratch/cscs/%u/logs/osm-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1

set -euo pipefail
# set -x

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --download-dir)
        download="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done

# Validate required arguments
if [[ -z "${download:-}" ]]; then
    echo "ERROR: Missing required arguments. See README for details."
    exit 1
fi

# Create download directory if it doesn't exist
mkdir -p ${download}

echo "$(date) | Downloading OSM planet file to ${download}"
podman run --rm -it -v ${download}:/download openmaptiles/openmaptiles-tools download-osm planet -- -d /download
echo "$(date) | Download completed."
