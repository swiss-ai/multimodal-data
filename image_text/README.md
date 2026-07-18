# Image-Text Pairing

Pair loose images with their captions or annotations and write the result as
webdataset shards. One script per source layout.

| Script | Source | Notes |
|--------|--------|-------|
| `rsteller.py` | RSTeller JSON annotations plus JPG images | Caption pairing |
| `geochat.py` | GeoChat JSON plus multi-part ZIP | Conversation format |
| `maptrace.py` | MapTrace parquet (`image_bytes` plus caption) | Includes dedup |
| `geospatial.py` | Geospatial captions dataset | Dataset selectable via `--help` |
| `main.py`, `main2.py` | Generic JSON/ZIP sources | Shared pairing entry points |

Each script is standalone with its own `--help`. Most have paths configured at
the top of the file.
