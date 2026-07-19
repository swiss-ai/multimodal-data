# Datasets

Per-dataset preprocessing scripts, one directory per dataset. Each directory is
self-contained: a dataset's download script (`download.slurm`, see
[`../download/README.md`](../download/README.md) for the convention) and any
conversion or captioning steps specific to it.

## Inventory

The full inventory (license, modality, stage, processing scripts, upstream
source) is in [`DATASETS.md`](DATASETS.md). It is generated, do not edit it by
hand:

- [`inventory.yaml`](inventory.yaml) is the source of truth.
- [`build_index.py`](build_index.py) renders it to `DATASETS.md`.

To update the table, edit `inventory.yaml` and rerun:

```bash
python build_index.py
```
