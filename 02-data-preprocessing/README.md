# Multimodal data pipeline

A streaming pipeline for loading, filtering, and writing large multimodal datasets.

## Usage

**Execution**

```sh
sbatch run.slurm config.json  # HPC (slurm)
python3 main.py config.json   # Local
```

**Output structure**

Target format is sharded parquets compatible with hf datasets.
```
output_dir/
├── data/
│   ├── train-00000-of-01811.parquet
│   ├── train-00001-of-01811.parquet
│   └── ...
├── dataset_info.json
└── state.json
```

**Loading**
```python
from datasets import load_dataset

data_dir = "/capstor/store/cscs/swissai/infra01/medical/apertus_image_only_v1"
dataset = load_dataset("parquet", data_dir=data_dir, split="train")
```

## Architecture

```
[Source] -> Adapters -> Filters -> Writer -> [Disk]
```

| Component | Role | Parallelism |
|-----------|------|-------------|
| **Adapters** | stream raw data          | internal (optional) |
| **Filters**  | accept/reject samples    | multiprocessing     |
| **Writer**   | serialize & write shards | multithreading      |

**Execution steps**

1. Adapter yields sample batch.
2. Batch is distributed to worker processes.
3. Workers apply filters sequentially.
4. Accepted samples are sent to writer.
5. Writer serializes (threaded) and writes to disk.

## Configuration

```json
{
  "adapters": [
    { "id": "Meditron", ... },
    { "id": "MedMax", ... },
    { "id": "MedTrinityFull", ... }
  ],
  "filters": [
    { "id": "ImageResolution",    "min_width": 64, "min_height": 64 },
    { "id": "ImageDeduplication", "db_path": "/dedup/path.db", "algorithm": "phash" }
  ],
  "writer": {
    "output_dir": "/output/path",
    "target_shard_bytes": 500000000,
    "num_workers": 100
  },
  "pipeline": {
    "log_file": "pipeline.log",
    "manifest_db": "manifest.db",
    "checkpoint_db": "checkpoint.db",
    "num_workers": 100,
    "batch_size": 1000
  }
}
```

`adapters`: List of source configurations. Each must specify an `id`. \
`filters`: List of processing filters. Each must specify an `id`. \
`writer`: Writer configuration. \
`pipeline`: Global pipeline settings.

## Data schema

See `pipeline/schema.py`.

**Metadata**

```python
@dataclass
class SampleMetadata:
    dataset_id: str      # Global ID
    sample_id: int       # Local ID
    data: dict[str, Any] # Payload
```

**Sample types**

| Class | Fields |
|-------|--------|
| `ImageSample`     | `image: PIL.Image` |
| `TextSample`      | `text: str` |
| `ImageTextSample` | `image: PIL.Image`, `text: str` |

## Development

### 1. Adapters

Located in `adapters/`. Must inherit `BaseDataset`.

**Interface**

```python
class BaseDataset(ABC):
    @property
    @abstractmethod
    def id(self) -> str: ...

    @abstractmethod
    def stream(
        self,
        logger,
        skip: int | None = None,
        batch_size: int = 1
    ) -> Iterator[list[Sample]]: ...
```

**Resume**: Must respect `skip` parameter (index of last processed sample). \
**Parallelism**: Optional. Can use multithreading for i/o-bound tasks, or multiprocessing for cpu-bound. \
**Registration**: Add to `adapters/__init__.py`.

### 2. Filters

Located in `filters/`. Must inherit `BaseFilter`.

**Interface**

```python
class BaseFilter(ABC):
    @abstractmethod
    def process_batch(self, samples: list[Sample]) -> list[bool]: ...
```

**Concurrency**: Filters run in parallel processes (must be thread/process-safe). \
**Statefulness**: Can be stateful (e.g., maintain deduplication DB), but must handle concurrent access. \
**Serialization**: Class instance must be picklable. \
**Registration**: Add to `filters/__init__.py`.

### 3. Writers

Located in `writers/`. Must inherit `BaseWriter`.

**Interface**

```python
class BaseWriter(ABC):
    @abstractmethod
    def open(self): ...

    @abstractmethod
    def write_batch(self, samples: list[Sample]): ...

    @abstractmethod
    def close(self): ...
```

## Resume logic

The pipeline is fault-tolerant.

1. **Checkpointing**: `checkpoint_db` records `last_sample_id` per dataset.
2. **State**: `state.json` records current shard index and global counters.
3. **Restart**: \
    Pipeline queries DB for last ID. \
    Passes `skip` count to Adapter. \
    Adapter fast-forwards to correct index. \
    Writer appends to next available shard.

New adapters added to config are detected automatically. Completed datasets are skipped.
