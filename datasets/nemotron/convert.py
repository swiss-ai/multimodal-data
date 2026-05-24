import pyarrow as pa

schema = pa.schema(
    [
        ("id", pa.string()),
        ("messages", pa.string()),
        ("images", pa.map_(pa.string(), pa.binary())),
    ]
)
