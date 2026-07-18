# Deduplication

Perceptual-hash deduplication for webdataset shards. A generic three-stage
pipeline, plus per-dataset variants that adapt it to unusual source layouts.

## Three-stage pipeline

| Stage | Script | Purpose |
|-------|--------|---------|
| 1 | `hash_webdataset.py` | Compute a perceptual hash for every image, write to parquet |
| 2 | `classify_hashes.py` | Check hashes against a RocksDB of seen hashes, emit a reject list |
| 3 | `rewrite_clean.py` | Drop rejected samples, write clean shards |

```bash
python dedup/hash_webdataset.py --input "data/*.tar" --output-dir hashes/
python dedup/classify_hashes.py --hash-dir hashes/ --db-path dedup.db --reject-list rejects.txt
python dedup/rewrite_clean.py   --input "data/*.tar" --output clean/ --reject-list rejects.txt
```

## Dataset-specific variants

| Script | Dataset | Notes |
|--------|---------|-------|
| `dedup_wds.py` | Generic WDS | Master script with configurable stages |
| `dedup_mint_html.py`, `dedup_mint_html_v2.py` | MINT-1T-HTML | Large-scale run with sharded DB |
| `dedup_mint_arxiv.py` | MINT-1T-ArXiv | Streams via HuggingFace `datasets` |
| `dedup_mint_pdf.py` | MINT-1T-PDF | Multi-page TIFF splitting |
| `dedup_bigdocs.py` | BigDocs-7.5M | Parquet-based dedup |
| `dedup_swisstopo.py` | Swisstopo Maps | WDS with URL keys |

Each script is standalone with its own `--help`.
