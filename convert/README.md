# Format and Text Conversion

Utilities that rewrite the contents of webdataset/tar shards in place: image
format conversion and text cleaning.

| Script | Purpose |
|--------|---------|
| `png_to_jpeg.py` | Convert PNG images in tar shards to JPEG (quality and optional resize) |
| `tiff_to_jpeg.py` | Convert TIFF images in tar shards to JPEG |
| `clean_sentences.py` | Flatten and clean captions (BLIP3-Grounding-50M) |
| `clean_text_tar.py` | Clean and normalize the `.txt` members inside tar shards |

Each script is standalone with its own `--help`.
