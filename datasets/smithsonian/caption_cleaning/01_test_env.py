#!/usr/bin/env python3
"""
01_test_env.py

Sanity-check that vLLM + Gemma 4 31B-it is working correctly for text-only
caption cleaning. Runs 3 example captions and prints the cleaned output.

Expected: dimensions stripped, metadata prefixes removed, narrative kept verbatim.
Exit 0 on success.
"""

import sys
from pathlib import Path

from vllm import LLM, SamplingParams

MODEL_PATH = "/tmp/models/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"

TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.80
MAX_TOKENS = 1024
TEMPERATURE = 0.1
TOP_P = 0.9

# sys.path so we can import prompts.py from the same dir
sys.path.insert(0, str(Path(__file__).parent))
from prompts import PROMPT_BY_SUBSET  # noqa: E402

EXAMPLES = [
    # tier1/nmaahc — rich prose after object title, no dims
    (
        "tier1/nmaahc",
        (
            "Shell casing from Normandy Beaches, D-Day 1944 "
            "One brass shell casing (.4a) stored in a cardboard tube (.4b). "
            "The casing has an imprint on the headstamp: [P / unidentified mark / "
            'unidentified mark / 20]. The tube has the inscription: "Shell casing '
            'from D-DAY JUNE 6, 1944/From: M.Sgt. Wallace B. Jackson".'
        ),
    ),
    # tier1/nmafa — material + dims prefix before useful prose
    (
        "tier1/nmafa",
        (
            "Distemper and gesso on wood H x W x D: 30.2 x 15.9 x 2.9 cm "
            "(11 7/8 x 6 1/4 x 1 1/8 in.) Triptych of distemper and gesso on wood "
            "with three registers on each panel and raised border painted white with "
            "red inscription. Triptych features various saints and religious "
            "iconography, including a central female figure."
        ),
    ),
    # tier2/design/chndm — metadata prefix then descriptive prose + dims mixed in
    (
        "tier2/design/chndm",
        (
            "Ceiling paper. Wallcoverings | ceiling paper. "
            "Created by Liberty Wall Paper Company, Schuylerville, New York, 1905–1915. "
            "Medium: Machine-printed paper, liquid mica | 84 x 49 cm (33 1/16 x 19 5/16 in.). "
            "Place of origin: Schuylerville, New York, USA. "
            "From Cooper Hewitt, Smithsonian Design Museum. "
            "Rococo-style ceiling paper with a large central floral medallion surrounded by "
            "scrolling acanthus leaves and stylized botanical ornaments printed in ochre, "
            "green, and cream on an ivory ground with a subtle mica sheen."
        ),
    ),
]


def build_messages(subset: str, caption: str) -> list[dict]:
    prompt_template = PROMPT_BY_SUBSET.get(subset, PROMPT_BY_SUBSET["default"])
    prompt = prompt_template.format(caption=caption)
    return [{"role": "user", "content": prompt}]


def main() -> None:
    print(f"Loading model from {MODEL_PATH} ...")
    llm = LLM(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        dtype="bfloat16",
        max_model_len=4096,
        trust_remote_code=True,
    )
    print("Model loaded.\n")

    sampling = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS)

    conversations = [build_messages(subset, caption) for subset, caption in EXAMPLES]
    outputs = llm.chat(conversations, sampling_params=sampling, use_tqdm=False)

    all_ok = True
    for i, ((subset, raw), output) in enumerate(zip(EXAMPLES, outputs), start=1):
        cleaned = output.outputs[0].text.strip()
        print(f"{'=' * 70}")
        print(f"Example {i} [{subset}]")
        print(f"  RAW ({len(raw)}c): {raw[:200]!r}{'...' if len(raw) > 200 else ''}")
        print(f"  CLEANED ({len(cleaned)}c): {cleaned[:400]!r}{'...' if len(cleaned) > 400 else ''}")
        if len(cleaned) < 20:
            print("  WARNING: cleaned output is suspiciously short")
            all_ok = False

    print(f"\n{'=' * 70}")
    if all_ok:
        print("All 3 examples cleaned successfully. Environment OK.")
        sys.exit(0)
    else:
        print("One or more examples produced suspiciously short output. Check the model.")
        sys.exit(1)


if __name__ == "__main__":
    main()
