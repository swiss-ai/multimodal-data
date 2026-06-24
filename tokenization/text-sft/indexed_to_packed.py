#!/usr/bin/env python3
"""
Convert Megatron-LM IndexedDataset (bin/idx) to Megatron-Bridge packed npy format.

Supports both multimodal SFT and text-only SFT datasets.
Uses parallel processing for efficient conversion on HPC clusters.

Usage:
    python indexed_to_packed.py \
        --input /path/to/dataset.bin \
        --output /path/to/output_dir \
        --tokenizer /path/to/tokenizer \
        --pack_size 4096 \
        --num_workers 32
"""

import argparse
import collections
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from megatron.bridge.data.datasets.packing_utils import (
    create_packing_strategy,
    fill_packing_strategy,
)
from megatron.core.datasets.indexed_dataset import IndexedDataset
from transformers import AutoTokenizer

from chat_preprocess import _build_mask_llama3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# IndexedDataset Utilities
# =============================================================================


def get_index_prefix(path: str) -> str:
    """Get prefix path for IndexedDataset (remove .bin/.idx extension)."""
    for ext in ['.bin', '.idx']:
        if path.endswith(ext):
            return path[:-len(ext)]
    return path


def load_dataset(path: str) -> IndexedDataset:
    """Load IndexedDataset with mmap for random access."""
    return IndexedDataset(get_index_prefix(path), multimodal=False, mmap=True)


# =============================================================================
# Parallel Worker
# =============================================================================


def process_chunk(args: Tuple) -> Tuple[Dict, Dict]:
    """
    Process a chunk of samples in parallel.

    Each worker:
    1. Opens its own mmap handle to the dataset
    2. Builds masks for samples in its range
    3. Returns sequences dict and statistics
    """
    (input_path, start_idx, end_idx, pack_size,
     assistant_header, eot_id, drop_long) = args

    dataset = load_dataset(input_path)
    train_flags = np.array([], dtype=np.bool_)

    sequences = collections.defaultdict(list)
    stats = {"processed": 0, "dropped": 0, "tokens": 0, "trained": 0}

    for i in range(start_idx, end_idx):
        tokens = np.array(dataset[i], dtype=np.int64)
        seq_len = len(tokens) - 1  # Megatron-Bridge convention

        if seq_len > pack_size:
            if drop_long:
                stats["dropped"] += 1
                continue

        mask = _build_mask_llama3(tokens, train_flags, assistant_header, eot_id)

        sequences[seq_len].append({
            "input_ids": tokens.tolist(),
            "loss_mask": mask.tolist(),
        })

        stats["processed"] += 1
        stats["tokens"] += len(tokens)
        stats["trained"] += int(mask.sum())

    return dict(sequences), stats


def merge_sequences(results: List[Tuple[Dict, Dict]]) -> Tuple[Dict, Dict]:
    """Merge sequences and statistics from all workers."""
    merged = collections.defaultdict(list)
    total_stats = {"processed": 0, "dropped": 0, "tokens": 0, "trained": 0}

    for sequences, stats in results:
        for seq_len, samples in sequences.items():
            merged[seq_len].extend(samples)
        for key in total_stats:
            total_stats[key] += stats[key]

    return dict(merged), total_stats


# =============================================================================
# Main Converter
# =============================================================================


def convert(
    input_path: str,
    output_dir: str,
    tokenizer_path: str,
    pack_size: int,
    num_workers: int = 32,
    drop_long: bool = True,
    packing_algorithm: str = "first_fit_decreasing",
) -> Dict:
    """
    Convert IndexedDataset to Megatron-Bridge packed npy format.

    Args:
        input_path: Path to input .bin file
        output_dir: Output directory
        tokenizer_path: Path to tokenizer
        pack_size: Maximum sequence length per pack
        num_workers: Number of parallel workers
        drop_long: Drop samples exceeding pack_size
        packing_algorithm: Packing algorithm from Megatron-Bridge

    Returns:
        Metadata dictionary
    """
    start_time = time.time()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Pack size: {pack_size}")
    logger.info(f"Workers: {num_workers}")

    # Load tokenizer and get mask tokens
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    assert 'sft_assistant_begin_sequence' in tokenizer.init_kwargs, \
        "Tokenizer missing sft_assistant_begin_sequence"
    assert 'sft_eot_token' in tokenizer.init_kwargs, \
        "Tokenizer missing sft_eot_token"

    assistant_header = np.array(
        tokenizer.init_kwargs['sft_assistant_begin_sequence'], dtype=np.int64
    )
    eot_id = tokenizer.init_kwargs['sft_eot_token'][0]
    logger.info(f"Mask tokens: header={assistant_header.tolist()}, eot={eot_id}")

    # Get dataset size
    dataset = load_dataset(input_path)
    num_samples = len(dataset)
    del dataset
    logger.info(f"Total samples: {num_samples:,}")

    # -------------------------------------------------------------------------
    # Parallel sequence building
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"BUILDING SEQUENCES ({num_workers} workers)")
    logger.info("=" * 60)

    # Create chunks for workers
    chunk_size = (num_samples + num_workers - 1) // num_workers
    chunks = [
        (input_path, i * chunk_size, min((i + 1) * chunk_size, num_samples),
         pack_size, assistant_header, eot_id, drop_long)
        for i in range(num_workers)
    ]

    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_chunk, chunks))

    # Merge results
    sequences, stats = merge_sequences(results)

    logger.info(f"Processed: {stats['processed']:,} samples")
    logger.info(f"Dropped: {stats['dropped']:,} samples")
    logger.info(f"Tokens: {stats['tokens']:,} total, {stats['trained']:,} trained")

    if stats['processed'] == 0:
        logger.error("No samples remaining after filtering!")
        return {"error": "No samples remaining"}

    # -------------------------------------------------------------------------
    # Packing
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("PACKING")
    logger.info("=" * 60)

    # Create histogram from sequences
    histogram = [len(sequences.get(i, [])) for i in range(pack_size + 1)]

    assignments, packing_metadata = create_packing_strategy(
        histogram, pack_size, packing_algorithm
    )
    logger.info(f"Packing metadata: {packing_metadata}")

    output_data = fill_packing_strategy(
        assignments, sequences, pack_size, tokenizer.eos_token_id
    )
    logger.info(f"Created {len(output_data):,} packs")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("SAVING")
    logger.info("=" * 60)

    input_name = Path(get_index_prefix(input_path)).name
    output_file = output_path / f"{input_name}.npy"
    np.save(output_file, output_data, allow_pickle=True)
    logger.info(f"Saved: {output_file}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Samples: {num_samples:,} input -> {stats['processed']:,} processed")
    logger.info(f"Packs: {len(output_data):,}")
    logger.info(f"Time: {elapsed:.1f}s ({stats['processed']/elapsed:.0f} samples/s)")

    # Save metadata
    metadata = {
        "input_path": str(input_path),
        "tokenizer_path": tokenizer_path,
        "pack_size": pack_size,
        "num_workers": num_workers,
        "packing_algorithm": packing_algorithm,
        "packing_metadata": packing_metadata,
        "statistics": {
            "input_samples": num_samples,
            "processed_samples": stats["processed"],
            "dropped_samples": stats["dropped"],
            "total_tokens": stats["tokens"],
            "trained_tokens": stats["trained"],
            "num_packs": len(output_data),
        },
        "processing_time_seconds": elapsed,
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Convert IndexedDataset to Megatron-Bridge packed format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Input .bin file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer path")
    parser.add_argument("--pack_size", type=int, default=4096, help="Pack size")
    parser.add_argument("--num_workers", type=int, default=32, help="Parallel workers")
    parser.add_argument("--no_drop_long", action="store_true", help="Keep long samples")
    parser.add_argument(
        "--packing_algorithm",
        default="first_fit_decreasing",
        choices=["first_fit_decreasing", "first_fit_shuffle"],
    )

    args = parser.parse_args()

    result = convert(
        input_path=args.input,
        output_dir=args.output,
        tokenizer_path=args.tokenizer,
        pack_size=args.pack_size,
        num_workers=args.num_workers,
        drop_long=not args.no_drop_long,
        packing_algorithm=args.packing_algorithm,
    )

    sys.exit(1 if "error" in result else 0)


if __name__ == "__main__":
    main()
