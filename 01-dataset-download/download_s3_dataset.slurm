#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=awscli
#SBATCH --job-name=downloadS3bucket
#SBATCH --output=/iopsstor/scratch/cscs/%u/multimodal-data/01-dataset-download/logs/s3-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/multimodal-data/01-dataset-download/logs/s3-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1

################################################################################
# S3 Data Downloader
################################################################################
# Downloads a directory from public S3 bucket to local storage.
#
# USAGE:
#   sbatch download_s3_dataset.slurm
#
# PARAMETERS (environment variables):
#   S3_URI       - S3 URI of the dataset to download
#   LOCAL_PATH   - Absolute path to local directory to store dataset
#   DRY_RUN      - If set to "true", only print commands without executing
#                  (default: false)
#
################################################################################

# Pre-execution checks

if ! command -v aws &> /dev/null; then
  echo "Error: AWS CLI is not installed or not in PATH."
  exit 1
fi

if [ -z "$S3_URI" ]; then
  echo "Error: S3_URI environment variable is not set."
  exit 1
fi

if [ -z "$LOCAL_PATH" ]; then
  echo "Error: LOCAL_PATH environment variable is not set."
  exit 1
fi

if [[ ! "$S3_URI" =~ ^s3://.+ ]]; then
  echo "Error: s3_uri '$S3_URI' must start with 's3://'"
  exit 1
fi

if [[ ! "$LOCAL_PATH" =~ ^/.* ]]; then
  echo "Error: local_path '$LOCAL_PATH' must be an absolute path"
  exit 1
fi

DRY_RUN="${DRY_RUN:-false}"
DRY_RUN_FLAG=""
if [ "$DRY_RUN" = "true" ]; then
  DRY_RUN_FLAG="--dryrun"
fi

echo "=========================================="
echo "S3 Bucket Download Job:"
echo "=========================================="
echo "S3 URI:        $S3_URI"
echo "Local Path:    $LOCAL_PATH"
echo "Dry Run:       $DRY_RUN"
echo ""

# Execute S3 download

echo "=========================================="
echo "Starting S3 download at: $(date)"
echo "=========================================="
echo ""

BASH_XTRACEFD=1
set -x

aws s3 sync "$S3_URI" "$LOCAL_PATH" \
  $DRY_RUN_FLAG \
  --no-sign-request \
  --cli-read-timeout 300 \
  --cli-connect-timeout 300

set +x

EXIT_CODE=$?

# Job summary

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="
