import os
import sys
from contextlib import redirect_stdout

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "paths.txt"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "check_output.txt"

    with open(input_file) as f:
        paths = f.read().splitlines()

    with open(output_file, "w") as f:
        with redirect_stdout(f):
            for path in paths:
                if os.path.exists(path):
                    print(f"OK: {path} exists.")
                else:
                    print(f"BAD: {path} does not exist.")
