import json
import os

import duckdb

PARQUET_GLOB = os.environ.get("PARQUET_GLOB", "/path/to/data/medical-datasets/raw/dailymed_spl/interleaved/*.parquet")
SAMPLE_ID = os.environ.get("SAMPLE_ID", "animal/20260318_dcd2ac5e-a42b-4fcc-ab0a-0cef02de1560")
OUT_DIR = os.environ.get("OUT_DIR", "/tmp/toolbox/duplic/data")

os.makedirs(OUT_DIR, exist_ok=True)

con = duckdb.connect()
row = con.execute(
    f"SELECT segments_json, images_bytes FROM read_parquet('{PARQUET_GLOB}') WHERE id=? LIMIT 1",
    [SAMPLE_ID],
).fetchone()

if row is None:
    print("Sample not found")
    exit(1)

segments_json, images_bytes = row
segments = json.loads(segments_json)

image_idx = 0
segment_idx = 0
manifest = []

for seg in segments:
    if seg["type"] == "image":
        if image_idx < len(images_bytes):
            img_bytes = bytes(images_bytes[image_idx])
            # Detect format from magic bytes
            if img_bytes[:2] == b"\xff\xd8":
                ext = "jpg"
            elif img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
                ext = "png"
            elif img_bytes[:4] == b"GIF8":
                ext = "gif"
            else:
                ext = "bin"
            fname = f"segment_{segment_idx:03d}_image_{image_idx:03d}.{ext}"
            fpath = os.path.join(OUT_DIR, fname)
            with open(fpath, "wb") as f:
                f.write(img_bytes)
            manifest.append(
                {
                    "segment": segment_idx,
                    "type": "image",
                    "file": fname,
                    "size": len(img_bytes),
                }
            )
            print(f"Saved image: {fname} ({len(img_bytes)} bytes)")
            image_idx += 1
        segment_idx += 1
    elif seg["type"] == "text":
        fname = f"segment_{segment_idx:03d}_text.txt"
        fpath = os.path.join(OUT_DIR, fname)
        text = seg.get("text", seg.get("content", str(seg)))
        with open(fpath, "w") as f:
            f.write(text)
        manifest.append({"segment": segment_idx, "type": "text", "file": fname, "chars": len(text)})
        print(f"Saved text:  {fname} ({len(text)} chars)")
        segment_idx += 1
    else:
        print(f"Unknown segment type: {seg['type']}")
        segment_idx += 1

# Save full segments JSON for inspection
with open(os.path.join(OUT_DIR, "segments.json"), "w") as f:
    json.dump(segments, f, indent=2)

# Save manifest
with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nDone. {len(manifest)} segments exported to {OUT_DIR}/")
print("segments.json and manifest.json also saved.")
