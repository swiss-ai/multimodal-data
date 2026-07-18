# Preprocessing (predecessor of `medical/`)

Frozen snapshot of the streaming adapter framework that became
[`medical/`](../medical/README.md). Same architecture (`adapters/`, `filters/`,
`pipeline/`, `writers/`, `main.py`, `run.slurm`) and same config format, kept
for history and pending consolidation into `medical/`.

For pipeline architecture, configuration, data schema, resume logic, and how to
add adapters, filters, and writers, see [`medical/README.md`](../medical/README.md).
Everything documented there applies here unchanged.

## Differences from `medical/`

This snapshot carries a smaller, older subset:

- **Adapters** (`adapters/`): `meditron`, `medmax`, `medtrinity`,
  `medtrinity_demo`, `mmc4`, `pmc_oa`. All are also present, and maintained, in
  `medical/adapters/`.
- **Writers** (`writers/`): HuggingFace sharded parquet only. `medical/` adds a
  webdataset writer.
- **Filters**: none bundled here. `medical/` adds resolution, deduplication, and
  downsample filters.

Prefer `medical/` for new work. This directory is not maintained.
