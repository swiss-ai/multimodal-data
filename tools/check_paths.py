import os
from contextlib import redirect_stdout

with open("paths.txt") as f:
    paths = f.read().splitlines()

with open("check_output.txt", "w") as f:
    with redirect_stdout(f):
        for path in paths:
            if os.path.exists(path):
                print(f"OK: {path} exists.")
            else:
                print(f"BAD: {path} does not exist.")
