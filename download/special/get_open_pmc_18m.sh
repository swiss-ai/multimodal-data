#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --job-name=ingest
#SBATCH --output=download.log
#SBATCH --time=10:00:00

cd /path/to/data/medical/raw
hf download vector-institute/open-pmc-18m --local-dir /path/to/data/medical/raw/open-pmc-18m --repo-type dataset
