import itertools
import json
import os

import webdataset as wds

dir = "/path/to/data/vision-datasets/swisstopo"
output_dir = os.path.join(dir, "paired")
input_dir = os.path.join(dir, "sorted")
tar_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".tar")])
tar_files = [os.path.join(input_dir, f) for f in tar_files]

sink = wds.ShardWriter(os.path.join(output_dir, "%06d.tar"), maxcount=20000)
ds = wds.WebDataset(tar_files, shardshuffle=False)
for sat_item, map_item in itertools.batched(ds, 2):
    map_json = json.loads(map_item["json"].decode("utf-8"))
    sat_json = json.loads(sat_item["json"].decode("utf-8"))
    assert map_json["pair_id"] == sat_json["pair_id"]

    pair_id = map_json["pair_id"]
    map_img = map_item["png"]
    sat_img = sat_item["png"]

    sink.write(
        {
            "__key__": pair_id,
            "map.png": map_img,
            "sat.png": sat_img,
            # "map.json": map_item["json"],
            # "sat.json": sat_item["json"],
        }
    )

sink.close()
