import os
import pprint

import requests

# 1. Paste your info here
dataset_id = "UCSC-VLAA/MedTrinity-25M"
token = os.environ.get("HF_TOKEN", "")
if not token:
    raise SystemExit("Set HF_TOKEN environment variable")
config = "25M_full"
split = "train"

# 2. Request the pre-computed stats
url = f"https://datasets-server.huggingface.co/statistics?dataset={dataset_id}&config={config}&split={split}"
headers = {"Authorization": f"Bearer {token}"}

response = requests.get(url, headers=headers)
data = response.json()
pprint.pp(data)
