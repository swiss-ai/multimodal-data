#!/usr/bin/env python3
"""
SFT Tokenization Script

Uses chat_preprocess.py with Numba-JIT mask builders and Megatron-Bridge packing.
Supports local JSONL files and HuggingFace Hub datasets.

Input formats:
    - messages: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    - conversations: [{"from": "User", "value": "..."}, {"from": "Assistant", "value": "..."}]
    - texts: [{"user": "...", "assistant": "..."}]  (FineVision format)

Usage:
    # From local JSONL file
    python tokenize_sft.py \
        --input data.jsonl \
        --output ./output/tokenized \
        --tokenizer /path/to/tokenizer \
        --style llama3

    # From HuggingFace Hub dataset
    python tokenize_sft.py \
        --input HuggingFaceM4/FineVision \
        --hf_dataset \
        --hf_subset text_openhermes_2_5 \
        --hf_split train \
        --output ./output/tokenized \
        --tokenizer /path/to/tokenizer \
        --style llama3

    # With packing
    python tokenize_sft.py \
        --input data.jsonl \
        --output ./output/tokenized \
        --tokenizer /path/to/tokenizer \
        --style llama3 \
        --pack --pack_size 4096
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="SFT Tokenization with Numba-JIT mask builders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input/Output
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file or HuggingFace dataset name")
    parser.add_argument("--output", type=str, required=True,
                        help="Output prefix (without extension)")

    # HuggingFace dataset options
    parser.add_argument("--hf_dataset", action="store_true",
                        help="Load from HuggingFace Hub dataset")
    parser.add_argument("--hf_subset", type=str, default=None,
                        help="HuggingFace dataset subset/config name")
    parser.add_argument("--hf_split", type=str, default="train",
                        help="HuggingFace dataset split")

    # Tokenizer
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="HuggingFace tokenizer path (must have sft_assistant_begin_sequence)")
    parser.add_argument("--style", type=str, required=True, choices=["apertus", "llama3"],
                        help="Tokenizer style for mask building")

    # Sequence settings
    parser.add_argument("--seq_length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--add_bos", action="store_true",
                        help="Add BOS token at start")
    parser.add_argument("--add_eos", action="store_true", default=True,
                        help="Add EOS token at end")

    # Packing
    parser.add_argument("--pack", action="store_true",
                        help="Enable sequence packing")
    parser.add_argument("--pack_size", type=int, default=None,
                        help="Pack size (defaults to seq_length)")
    parser.add_argument("--packing_algorithm", type=str, default="first_fit_shuffle",
                        choices=["first_fit_shuffle", "first_fit_decreasing"],
                        help="Packing algorithm")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup pack size
    if args.pack_size is None:
        args.pack_size = args.seq_length

    from transformers import AutoTokenizer
    from chat_preprocess import SFTChatDataset, prepare_packed_sft_data

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    logger.info(f"  Vocab size: {len(tokenizer)}")
    logger.info(f"  BOS ID: {tokenizer.bos_token_id}")
    logger.info(f"  EOS ID: {tokenizer.eos_token_id}")

    # Check for SFT tokens
    sft_begin = tokenizer.init_kwargs.get('sft_assistant_begin_sequence')
    sft_eot = tokenizer.init_kwargs.get('sft_eot_token')
    if not sft_begin or not sft_eot:
        logger.error("Tokenizer missing sft_assistant_begin_sequence or sft_eot_token!")
        logger.error("Use omni_tokenizer/create_instruct.py to create the tokenizer.")
        sys.exit(1)
    logger.info(f"  SFT assistant begin: {sft_begin}")
    logger.info(f"  SFT EOT: {sft_eot}")

    # Setup output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    if args.pack:
        # Use packing
        logger.info(f"Preparing packed sequences...")
        logger.info(f"  Input: {args.input}")
        logger.info(f"  Pack size: {args.pack_size}")
        logger.info(f"  Algorithm: {args.packing_algorithm}")
        logger.info(f"  Style: {args.style}")

        output_npy = Path(str(output_path) + "_packed.npy")
        output_metadata = Path(str(output_path) + "_packed_metadata.json")

        output_data, metadata = prepare_packed_sft_data(
            input_path=args.input,
            output_path=output_npy,
            output_metadata_path=output_metadata,
            packed_sequence_size=args.pack_size,
            tokenizer=tokenizer,
            max_seq_length=args.seq_length,
            style=args.style,
            seed=args.seed,
            packing_algorithm=args.packing_algorithm,
            add_bos=args.add_bos,
            add_eos=args.add_eos,
            hf_dataset=args.hf_dataset,
            hf_dataset_subset=args.hf_subset,
            hf_dataset_split=args.hf_split,
        )

        logger.info(f"Output saved to:")
        logger.info(f"  Data: {output_npy}")
        logger.info(f"  Metadata: {output_metadata}")

        logger.info(f"Packing statistics:")
        logger.info(f"  Packing factor: {metadata.get('packing_factor', 'N/A')}")
        logger.info(f"  Packing efficiency: {metadata.get('packing_efficiency', 'N/A')}%")
        logger.info(f"  Max samples per bin: {metadata.get('max_samples_per_bin', 'N/A')}")
        logger.info(f"  Total packed sequences: {len(output_data)}")

    else:
        # Unpacked tokenization
        logger.info(f"Tokenizing dataset (unpacked)...")
        logger.info(f"  Input: {args.input}")
        logger.info(f"  Seq length: {args.seq_length}")
        logger.info(f"  Style: {args.style}")

        dataset = SFTChatDataset(
            file_path=args.input,
            tokenizer=tokenizer,
            max_seq_length=args.seq_length,
            style=args.style,
            add_bos=args.add_bos,
            add_eos=args.add_eos,
            seed=args.seed,
            hf_dataset=args.hf_dataset,
            hf_dataset_subset=args.hf_subset,
            hf_dataset_split=args.hf_split,
        )

        logger.info(f"  Total samples: {len(dataset)}")

        # Convert to numpy array
        logger.info("Converting to numpy array...")
        tokenized_data = []
        for i in range(len(dataset)):
            sample = dataset[i]
            input_ids = sample["input_ids"]
            if hasattr(input_ids, 'tolist'):
                input_ids = input_ids.tolist()
            loss_mask = sample.get("loss_mask", [True] * len(input_ids))
            if hasattr(loss_mask, 'tolist'):
                loss_mask = loss_mask.tolist()

            tokenized_data.append({
                "input_ids": input_ids,
                "loss_mask": loss_mask,
            })

        output_npy = Path(str(output_path) + ".npy")
        np.save(output_npy, tokenized_data)

        logger.info(f"Output saved to: {output_npy}")

        # Print statistics
        token_counts = [len(s["input_ids"]) for s in tokenized_data]
        loss_counts = [sum(s["loss_mask"]) for s in tokenized_data]
        logger.info(f"Tokenization statistics:")
        logger.info(f"  Total samples: {len(tokenized_data)}")
        logger.info(f"  Total tokens: {sum(token_counts):,}")
        logger.info(f"  Avg tokens/sample: {np.mean(token_counts):.1f}")
        logger.info(f"  Min tokens: {min(token_counts)}")
        logger.info(f"  Max tokens: {max(token_counts)}")
        logger.info(f"  Avg trained tokens/sample: {np.mean(loss_counts):.1f}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
